import * as THREE from 'three/webgpu';
import { Fn, abs, atan, attribute, cos, exp, float, floor, length, max, min, mix, positionGeometry, sin, smoothstep, step, texture, uniform, varying, vec2, vec4 } from 'three/tsl';
import { TEMPORAL } from '../post/Antialiasing.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import type { FrameGraph, ResourceHandle } from '../post/graph/FrameGraph.ts';
import { onSceneDispose } from '../SceneDispose.ts';

// The displacement field: how far the things that part the meadow push its grass over.
//
// The displacement field: a two-channel map of how far the things that part the meadow push it
// over, on the ground over the whole island, cleared every frame and drawn again from the sprites
// the frame holds: after the 2D displacement buffer about the camera, written
// by particles of a kind of their own, that bends and flattens the grass where the player passes
// in Bill Rockenbeck's *Blowing from the West: Simulating Wind in Ghost of Tsushima* (GDC 2021).
// The blades read it (Grass.ts): the direction turns their facing, its size adds to their lean.
//
// Rules:
// - It covers a square of the ground centred on the world's origin, as wide as the scene asks when
//   it makes the field (`createDisplacementField`: the island gives its land's whole lattice, 192 m
//   on a side), at DISPLACEMENT_SIZE texels, fixed on the world:
//   the island camera sees the whole island from up to 190 m, where a trail (metres long, its core
//   some ten pixels wide there) still shows: a window that followed the camera's point of interest
//   would leave the islanders beyond it without trails, and show them appear and vanish at its edge
//   as the camera panned. A texel is 0.375 m, about a trail's half width at half height: the
//   cubic B-spline read draws it smoothly at the closest view (22 m). Half float, two channels:
//   1 MB. A point (x, z) reads it at uv = ( x, z ) / coverage + 0.5, v down the texture (row 0 at
//   -z's edge: the sprites' clip y is 1 - 2 v). The land grows within the lattice, so the field
//   never moves or resizes with it. Its content is a function of the frame's sprites, with no
//   history of its own, so a paused frame is bit-identical and a scene jump leaves nothing of the
//   scene before (the temporal modes' copy of the previous frame is a history, which the jump's cut
//   drops; `fgview=displacement` shows the field).
// - A sprite is an oriented quad (its centre, its half axis along, its half width across) carrying a
//   vector map V in the quad's frame (along, across), analytic (no texture), of three shapes:
//   - a strip, a stretch of a walker's way, its axis toward the walker (`trailPush`): at d m from
//     its axis, p( d ) along ( cos alpha, s sin alpha ), the push out from the axis by alpha( d ) =
//     TRAIL.angle + TRAIL.spread d off straight ahead (a chevron: the middle pressed nearly ahead,
//     the sides the more aside the farther out), s the side's sign eased in over TRAIL.middle m, so
//     the middle's push is ahead alone and continuous across it; p( d ) a narrow Gaussian (TRAIL
//     `core`, `weight`) over a faint skirt that reaches TRAIL.skirt[ 1 ] m, 1 on the axis;
//   - a half disc ahead of its centre, closing a way's newest end: at r m from its centre the same
//     section p( r ) and spread, its push turned toward straight ahead in front, min( phi, alpha( r ) )
//     off it, phi the point's bearing off the axis, so where it meets the strip (phi a right angle,
//     r = d) the two agree and nothing steps at the feet;
//   - a disc pushing out radially (`standPush`), rising over STAND.rise m from its centre, gone by
//     STAND.fall[ 1 ] m, so the vector has no turn at the centre.
//   No component of V is squared or normalised on its own: a map gathered toward its middle per
//   component (V |V| taken component-wise) turns its directions toward the axes, a 17 degree
//   chevron to about 5 and a disc's radial push off its diagonals. V is turned into the world by the
//   quad's axes, times the sprite's strength, added (one blend, a fixed order: the order the owner
//   writes them in).
// - A sprite's strength is a share of a parted blade's lean: I = A( a ) share tan( angle ) / ( k
//   h_ref ), where the reference blade (RESPONSE.height m) leans over by `angle` at full push under
//   the blades' response (tan lean += k h |v|, k = RESPONSE.lean per m), `share` is the sprite's
//   (the way's 1, a standing walker's disc less) and A( a ) is the sprites' source's curve at the
//   sprite's age a, the age interpolated along a strip from one end's to the other's; the disc does
//   not age (A 1). The curve multiplies the lean, not the angle: a long blade lies flat through
//   the curve's first lobe, then springs up and is pressed down again. The field only draws the
//   sprites: what writes them gives the curve when it registers (`DisplacementField.source`: the
//   islanders' trail, Trample.ts), and the sprites' material is built then.
// - On the frame graph it is a pass of its own (`declare`), first in the frame, its size fixed (the
//   dynamic resolution's steps never resize it, so its pipelines are never rebuilt); its readers
//   declare it as a read. The sprites draw only while there are any (the mesh hidden otherwise),
//   registered with the pass's warm-up as a latent, so the first walker builds no pipeline. Under
//   the temporal modes the graph copies it at the frame's end into a persistent texture
//   (`copyToPersistent`: many materials read it, and a ping-pong pair through `previous()` would
//   rebind their texture every frame). Its readers take the previous frame's copy for their previous
//   position (`DisplacementUniforms.previous`; `previousValid` 0 after a cut of the histories, when
//   they take the current one instead).
// - Its readers sample it with a cubic B-spline (`displacementAt`), four bilinear taps of the field
//   (Sigg and Hadwiger, *Fast Third-Order Texture Filtering*, GPU Gems 2 ch. 20), so a blade that
//   leans over a texel's edge turns smoothly: bilinear, its slope would step at every texel.
//   The taps' places and weights are one `Fn( ... ).setLayout( ... )` shared by every reader, the
//   taps sampled at level 0 (the vertex stage) in the reader's graph; a point outside the field
//   reads 0. Plants that join the field read it through the same function.
// - `debugFlags.trample = false` at load builds no field (`displacementField` null): nothing
//   declared on the graph, no sprite drawn, the blades read nothing (`displacementAt` 0).

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;

/** The field's texels on a side. The ground it covers is the field's own (`coverage`). */
export const DISPLACEMENT_SIZE = 512;

/** The blades' response the sprites' strengths are given for: a blade of `height` m leans by
 * tan += `lean` height |v| under the field's value v (`lean` per m: neutral 1). */
export const RESPONSE = { height: 0.28, lean: 1 } as const;

/** Whether the field exists (`debugFlags.trample`, read at load). */
export const DISPLACEMENT_ON = debugFlags.trample !== false;

/** The sprites' shapes (`DisplacementField.sprite`). */
export const SPRITE = { strip: 0, ahead: 1, disc: 2 } as const;

/** The way's cross-section and chevron (`trailPush`): p( d ) = `weight` e^( -d^2 / 2 `core`^2 ) +
 * ( 1 - `weight` ) ( 1 - smoothstep( skirt[ 0 ], skirt[ 1 ], d ) ) at d m from the middle (0.45 at
 * 0.3 m, 0.18 at 0.5 m, 0.1 at 0.8 m, 0.04 at 1.1 m, 0 from 1.45 m: a core some 0.55 m wide at half
 * its height, which the field's B-spline read widens, in a skirt that fades the trail's edge), the
 * push `angle` (rad) off straight ahead in the middle and `spread` (rad per m) more each metre out,
 * its side eased in over `middle` m. */
const TRAIL = { core: 0.215, weight: 0.88, skirt: [ 0.575, 1.45 ], angle: 17 * Math.PI / 180, spread: 40 * Math.PI / 180, middle: 0.1 } as const;

/** The standing disc's profile (`standPush`): rising over `rise` m from its centre, falling over
 * `fall` m, so its push is at its most 0.3 to 0.6 m out and gone a metre out. */
const STAND = { rise: 0.3, fall: [ 0.6, 1 ] } as const;

// the sprites a frame holds at most: TRAMPLE.walkers of ( samples - 1 ) strips, a half disc and a
// disc each
const CAPACITY = 2560;

/** The field's uniforms and the base nodes its readers sample it through (Pipeline.ts sets the
 * graph's textures): the field of the frame being rendered and, under the temporal modes, of the
 * previous frame, and whether that one is valid (0 after a cut). */
export function createDisplacementUniforms() {

	const blank = () => new THREE.DataTexture( new Uint16Array( 4 ), 1, 1, THREE.RGBAFormat, THREE.HalfFloatType );
	return {
		field: texture( blank() ),
		previous: texture( blank() ),
		previousValid: uniform( 0 ),
	};

}

export type DisplacementUniforms = ReturnType<typeof createDisplacementUniforms>;

/** The field's uniforms (shared.ts holds them as `shared.displacement`). */
export const displacementUniforms = createDisplacementUniforms();

// The cubic B-spline's four bilinear taps about the texel coordinate x (texel k's centre at x = k,
// the four texels i - 1 .. i + 2 about it, each pair taken as one bilinear tap between its two):
// the taps' uv ( u0, v0, u1, v1 ) and their weights ( g0x, g0y, g1x, g1y ). Pure arithmetic on its
// inputs, so its one shader function serves every material and pass that reads the field; the taps
// themselves are sampled in each material's own graph (`displacementAt`), as a layout function
// that reads a texture or uniform is emitted with one build's names for them and fails in another's
// (the meadow's shadow pass: `struct member nodeUniform0 not found`).
const bsplineTaps = Fn( ( [ x ]: [ V2 ] ) => {

	const i = floor( x ).toVar();
	const f = x.sub( i ).toVar();
	const f2 = f.mul( f ), f3 = f2.mul( f );
	const w0 = f.oneMinus().mul( f.oneMinus() ).mul( f.oneMinus() ).div( 6 );
	const w1 = f3.mul( 3 ).sub( f2.mul( 6 ) ).add( 4 ).div( 6 );
	const w3 = f3.div( 6 );
	const g0 = w0.add( w1 ).toVar();
	const g1 = float( 1 ).sub( g0 );
	const t0 = i.sub( 1 ).add( w1.div( g0 ) ).add( 0.5 ).div( DISPLACEMENT_SIZE );
	const t1 = i.add( 1 ).add( w3.div( max( g1, 1e-6 ) ) ).add( 0.5 ).div( DISPLACEMENT_SIZE );
	return vec4( t0, t1 );

} ).setLayout( { name: 'displacementTaps', type: 'vec4', inputs: [ { name: 'x', type: 'vec2' } ] } );

const bsplineWeights = Fn( ( [ x ]: [ V2 ] ) => {

	const f = x.sub( floor( x ) ).toVar();
	const f2 = f.mul( f ), f3 = f2.mul( f );
	const g0 = f.oneMinus().mul( f.oneMinus() ).mul( f.oneMinus() ).add( f3.mul( 3 ).sub( f2.mul( 6 ) ).add( 4 ) ).div( 6 );
	return vec4( g0, float( 1 ).sub( g0 ) );

} ).setLayout( { name: 'displacementWeights', type: 'vec4', inputs: [ { name: 'x', type: 'vec2' } ] } );

/** `field` (the field or the previous frame's) at the point `p` (world x, z): the cubic B-spline of its
 * texels there, four bilinear taps at level 0, 0 outside it. */
function readField( field: DisplacementUniforms[ 'field' ], coverage: number, p: V2 ): V2 {

	const uv = p.div( coverage ).add( 0.5 ).toVar();
	const x = uv.mul( DISPLACEMENT_SIZE ).sub( 0.5 ).toVar();
	const t = bsplineTaps( x ).toVar(), g = bsplineWeights( x ).toVar();
	const tap = ( s: F, r: F ) => field.sample( vec2( s, r ) ).level( float( 0 ) ).xy;
	const sum = tap( t.x, t.y ).mul( g.x ).add( tap( t.z, t.y ).mul( g.z ) ).mul( g.y )
		.add( tap( t.x, t.w ).mul( g.x ).add( tap( t.z, t.w ).mul( g.z ) ).mul( g.w ) );
	const inside = step( max( abs( uv.x.sub( 0.5 ) ), abs( uv.y.sub( 0.5 ) ) ), float( 0.5 ) );
	return sum.mul( inside );

}

/** The field's value (world x, z) at the point `p` (world x, z), in the vertex stage: of the frame
 * being rendered, or with `previous` of the previous frame under the temporal modes (the current one
 * after a cut, or outside them). 0 without a field. */
export function displacementAt( p: V2, previous = false ): V2 {

	const u = displacementUniforms;
	if ( ! DISPLACEMENT_ON ) return vec2( 0 );
	if ( ! displacementField ) throw new Error( 'Displacement: the field is read before createDisplacementField made it' );
	const coverage = displacementField.coverage;
	if ( ! previous || ! TEMPORAL ) return readField( u.field, coverage, p );
	return mix( readField( u.field, coverage, p ), readField( u.previous, coverage, p ), step( 0.5, u.previousValid ) );

}

/** The field, its sprites and its pass. */
export class DisplacementField {

	readonly uniforms: DisplacementUniforms;
	/** The side of the square of ground the field covers (m), centred on the world's origin: a plain
	 * number, compiled into the sprites' and the readers' shaders. */
	readonly coverage: number;
	/** The sprites' scene, drawn by the field's pass with `camera` (whose matrices the sprites
	 * ignore: they place themselves in the field). */
	readonly scene = new THREE.Scene();
	readonly camera = new THREE.OrthographicCamera();
	readonly mesh: THREE.Mesh;
	// per sprite: ( centre x, z, half axis along x, z ), ( half width across, shape, age at the
	// axis's start, age at its end ), its strength at A 1
	private readonly a: THREE.InstancedBufferAttribute;
	private readonly b: THREE.InstancedBufferAttribute;
	private readonly c: THREE.InstancedBufferAttribute;
	private readonly geometry: THREE.InstancedBufferGeometry;
	private count = 0;
	private sourced = false;
	private handle: ResourceHandle | null = null;
	private previousHandle: ResourceHandle | null = null;
	private graph: FrameGraph<unknown> | null = null;

	constructor( uniforms: DisplacementUniforms, coverage: number ) {

		this.uniforms = uniforms;
		this.coverage = coverage;
		this.scene.name = 'displacement sprites';
		const geometry = this.geometry = new THREE.InstancedBufferGeometry();
		geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( [ - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 1, 0 ], 3 ) );
		geometry.setIndex( new THREE.BufferAttribute( new Uint32Array( [ 0, 1, 2, 0, 2, 3 ] ), 1 ) );
		const make = ( size: number ) => new THREE.InstancedBufferAttribute( new Float32Array( CAPACITY * size ), size );
		this.a = make( 4 );
		this.b = make( 4 );
		this.c = make( 1 );
		geometry.setAttribute( 'spriteA', this.a );
		geometry.setAttribute( 'spriteB', this.b );
		geometry.setAttribute( 'spriteC', this.c );
		geometry.instanceCount = 0;
		this.mesh = new THREE.Mesh( geometry );
		this.mesh.frustumCulled = false;
		this.mesh.visible = false;
		this.scene.add( this.mesh );

	}

	/** Registers what writes the sprites, once, before the first frame: `curve` is the share of its
	 * strength a strip or a half disc keeps at `age` seconds (the disc does not age). Builds the
	 * sprites' material. */
	source( curve: ( age: F ) => F ): void {

		if ( this.sourced ) throw new Error( 'Displacement: the sprites have a source already' );
		this.sourced = true;
		this.mesh.material = spriteMaterial( this.coverage, curve );

	}

	/** Starts the frame's sprites (none). */
	begin(): void {

		this.count = 0;

	}

	/** Adds a sprite: its centre (world x, z), its half axis along (world x, z: its direction and
	 * half length, m; for the half disc and the disc the radius, pointing ahead), its half width
	 * across (m), its shape (SPRITE), the ages at the axis's start and end (s), the angle a full
	 * push leans the reference blade by (rad) and the sprite's share of a full push. */
	sprite( x: number, z: number, ax: number, az: number, across: number, shape: number, age0: number, age1: number, angle: number, share: number ): void {

		if ( this.count >= CAPACITY ) return;
		const k = this.count ++, i = k * 4;
		this.a.array.set( [ x, z, ax, az ], i );
		this.b.array.set( [ across, shape, age0, age1 ], i );
		this.c.array[ k ] = share * Math.tan( angle ) / ( RESPONSE.lean * RESPONSE.height );

	}

	/** Ends the frame's sprites: uploads those written, and draws them while there are any. */
	end(): void {

		const n = this.count;
		this.geometry.instanceCount = n;
		this.mesh.visible = n > 0;
		if ( n === 0 ) return;
		for ( const attribute of [ this.a, this.b, this.c ] ) {

			attribute.clearUpdateRanges();
			attribute.addUpdateRange( 0, n * attribute.itemSize );
			attribute.needsUpdate = true;

		}

	}

	/** Declares the field and its pass on `graph`, first in the frame; returns the field as the pass
	 * leaves it, for its readers' reads, and under the temporal modes the previous frame's. */
	declare<F>( graph: FrameGraph<F> ): { field: ResourceHandle; previous: ResourceHandle | null } {

		const g = graph;
		this.graph = g as unknown as FrameGraph<unknown>;
		const size = { of: 'fixed' as const, width: DISPLACEMENT_SIZE, height: DISPLACEMENT_SIZE };
		const desc = { size, format: THREE.RGFormat, type: THREE.HalfFloatType, filter: 'linear' as const };
		let field = g.color( 'displacement', desc );
		// the previous frame's, copied at the frame's end: every material and kernel that draws the
		// meadow reads it, so it is one persistent texture rather than a ping-pong pair
		const previous = TEMPORAL ? g.color( 'displacementPrev', desc, true ) : null;
		const u = this.uniforms;
		const bind = () => {

			u.field.value = g.texture( this.handle! );
			if ( previous ) {

				u.previous.value = g.texture( previous );
				u.previousValid.value = g.historyValid( previous ) ? 1 : 0;

			}

		};
		g.addPass( 'displacement', 'scene', ( b ) => {

			field = b.write( field, { clear: true } );
			return {
				label: 'displacement',
				scene: this.scene,
				camera: this.camera,
				latent: () => this.sourced ? [ this.mesh ] : [],
				run: () => {

					const r = g.renderer;
					const colour = r.getClearColor( _colour ), alpha = r.getClearAlpha();
					r.setClearColor( 0x000000, 0 );
					bind();
					r.render( this.scene, this.camera );
					r.setClearColor( colour, alpha );

				},
			};

		} );
		if ( previous ) g.copyToPersistent( field, previous );
		this.handle = field;
		this.previousHandle = previous;
		bind();
		return { field, previous };

	}

	/** A cut of the temporal modes' histories: the previous frame's field is not read. */
	cut(): void {

		if ( this.graph && this.previousHandle ) this.graph.resetHistory( this.previousHandle );

	}

}

const _colour = new THREE.Color();

/** The way's push at ( x, y ) m in a strip's or half disc's frame (x along its axis, toward the
 * walker; y across), its distance d from the axis or the half disc's centre and the push's turn off
 * straight ahead (TRAIL): p( d ) ( cos alpha, s sin alpha ). */
const trailPush = Fn( ( [ y, d, turn ]: [ F, F, F ] ) => {

	const p = mix( float( 1 ).sub( smoothstep( TRAIL.skirt[ 0 ], TRAIL.skirt[ 1 ], d ) ), exp( d.mul( d ).mul( - 0.5 / TRAIL.core ** 2 ) ), TRAIL.weight );
	const side = y.sign().mul( smoothstep( 0, TRAIL.middle, abs( y ) ) );
	return vec2( cos( turn ), side.mul( sin( turn ) ) ).mul( p );

} ).setLayout( { name: 'displacementTrail', type: 'vec2', inputs: [ { name: 'y', type: 'float' }, { name: 'd', type: 'float' }, { name: 'turn', type: 'float' } ] } );

/** The standing disc's push at `local` (m from its centre): radial, of STAND's profile. */
const standPush = Fn( ( [ local ]: [ V2 ] ) => {

	const r = length( local );
	const p = smoothstep( 0, STAND.rise, r ).mul( float( 1 ).sub( smoothstep( STAND.fall[ 0 ], STAND.fall[ 1 ], r ) ) );
	return local.div( max( r, 1e-6 ) ).mul( p );

} ).setLayout( { name: 'displacementStand', type: 'vec2', inputs: [ { name: 'local', type: 'vec2' } ] } );

function spriteMaterial( coverage: number, curve: ( age: F ) => F ): THREE.NodeMaterial {

	const a = attribute<'vec4'>( 'spriteA', 'vec4' ), b = attribute<'vec4'>( 'spriteB', 'vec4' ), c = attribute<'float'>( 'spriteC', 'float' );
	const st = positionGeometry.xy;
	const half = length( a.zw ).toVar();
	const dir = a.zw.div( max( half, 1e-6 ) ).toVar();
	const across = vec2( dir.y.negate(), dir.x );
	const world = a.xy.add( dir.mul( st.x.mul( half ) ) ).add( across.mul( st.y.mul( b.x ) ) );
	const uv = world.div( coverage ).add( 0.5 );
	const material = new THREE.NodeMaterial();
	material.name = 'displacement sprite';
	// (v runs down the target: row 0 at clip y = +1)
	material.vertexNode = vec4( uv.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uv.y.mul( 2 ) ), 0, 1 );
	// the point in the quad's frame (m: along, across), the share along its axis, the axis, and the
	// sprite's shape, ages and strength
	const local = varying( st.mul( vec2( half, b.x ) ) as V2 );
	const along = varying( st.x.mul( 0.5 ).add( 0.5 ) as F );
	const vDir = varying( dir as V2 );
	const vb = varying( b ), gain = varying( c );
	const shape = vb.y;
	const y = abs( local.y ).toVar(), r = length( local ).toVar();
	// the strip, the half disc ahead (nothing behind its centre) and the disc, blended by 0/1
	// weights rather than selected (TSL may compile a select of large operands into branches)
	const ahead = step( 0.5, shape ).mul( step( shape, 1.5 ) ), disc = step( 1.5, shape );
	const strip = trailPush( local.y, y, y.mul( TRAIL.spread ).add( TRAIL.angle ) );
	const bearing = atan( y, max( local.x, 1e-6 ) );
	const front = trailPush( local.y, r, min( bearing, r.mul( TRAIL.spread ).add( TRAIL.angle ) ) ).mul( step( 0, local.x ) );
	const V = mix( mix( strip, front, ahead ), standPush( local ), disc );
	const age = mix( vb.z, vb.w, along );
	const strength = gain.mul( mix( curve( age ), float( 1 ), disc ) );
	material.fragmentNode = vec4( vDir.mul( V.x ).add( vec2( vDir.y.negate(), vDir.x ).mul( V.y ) ).mul( strength ), 0, 0 );
	material.blending = THREE.CustomBlending;
	material.blendEquation = THREE.AddEquation;
	material.blendSrc = THREE.OneFactor;
	material.blendDst = THREE.OneFactor;
	material.blendSrcAlpha = THREE.OneFactor;
	material.blendDstAlpha = THREE.OneFactor;
	material.depthTest = false;
	material.depthWrite = false;
	material.side = THREE.DoubleSide;
	return material;

}

/** The field once `createDisplacementField` has made it; null before, and with
 * `debugFlags.trample = false` at load. */
export let displacementField: DisplacementField | null = null;

/** Makes the page's one field over `coverage` m of ground on a side (centred on the world's origin),
 * before anything reads it (`displacementAt`) or declares it on the frame graph; null with
 * `debugFlags.trample = false` at load. The frame's passes declare it only if it exists when they
 * are built (PipelinePasses.ts), so made after them the frame silently has no displacement pass:
 * the island's trail stage makes it before the render pipeline's `createSeaAndPipeline`. */
export function createDisplacementField( coverage: number ): DisplacementField | null {

	if ( displacementField ) throw new Error( 'Displacement: the field is made once' );
	if ( DISPLACEMENT_ON ) displacementField = new DisplacementField( displacementUniforms, coverage );
	return displacementField;

}
onSceneDispose( 'displacement', () => { displacementField = null; } );
