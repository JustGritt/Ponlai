import * as THREE from 'three/webgpu';
import { Fn, attribute, cameraPosition, cos, float, instanceIndex, length, mix, normalGeometry, normalLocal, positionGeometry, select, sin, sqrt, uniform, varying, vec2, vec3, vec4 } from 'three/tsl';
import { AttributeType, freeAttribute, uploadSteps } from '../util/Upload.ts';
import { stippleThreshold } from './Coverage.ts';
import { LEVEL_SWITCH, levelFade, switchScale } from './Lod.ts';
import { VegetationCull, poolBuffers, poolSteps, type CullBatchInput, type CullBuffers, type CullPayload, type CullPool } from './VegetationCull.ts';

// Plants drawn as instances of a few prototypes, at two levels of detail, culled on the GPU.
//
// Plants drawn as instances of a few prototypes, at two levels of detail, culled on the GPU: the
// path any authored plant (a mesh of stem, branches and leaf cards, made once) takes. Today the
// bamboo (Bamboo.ts) and the yunpeng (Yunpeng.ts); the meadow keeps its own blades generated per
// view (Grass.ts).
//
// - Prototypes: a kind builds a few meshes of one upright plant at the origin (the bamboo: four
//   heights by two girths), each at a near level (the full mesh) and a far one (fewer sides, rings
//   and leaf cards). Every plant is an instance of one: its foot, its turn about the vertical, its
//   height scale (the stem's tube taking a girth scale of its own across, everything off the tube
//   a limb scale), its lean and bow, its wind's load and four values of the kind's own (`look`).
//   The vertex stage places a prototype's point (`PlantVertex.place`): scaled, turned, and moved
//   with the stem's axis at the share of its height it hangs from (the `plantPart` attribute: that
//   share, and 1 on the stem's tube), so a branch and its leaves ride their node rigidly while the
//   tube follows the axis ring by ring. The axis leans and bows over by `stemAxis`'s curve, so a
//   few prototypes give plants that each stand their own way.
// - The draws: one per prototype, level and part (the bamboo's culms and its leaves are two
//   vegetation kinds, two roots here), each a mesh over its prototype's geometry (its own geometry
//   object sharing the prototype's buffers) drawn indirectly from the cull (VegetationCull.ts): one
//   pass per view counts, scans and scatters the visible instances of every draw into its list, in
//   the plants' order, so a paused frame is bit-identical. Each draw holds every plant of its
//   prototype; the cull keeps those inside the view and inside the draw's band of distance (its
//   `fold`), so the indirect arguments count only what draws.
// - The switch (Lod.ts LEVEL_SWITCH): at `start`, where the plant's stem is a few render pixels wide
//   (its `width` times the camera's `switchScale`), the near level gives way to the far one over
//   the band beyond it. Inside the band both draw, split pixel by pixel by the stipple (Coverage.ts
//   `stippleThreshold`) at the far level's share v = saturate( a d^2 + b ) (Lod.ts `levelFade`): the
//   near level keeps a pixel whose threshold is at or above v, the far level one below it (`fade`'s
//   sign tells them apart), so each pixel shows one level, never both and never neither. A part
//   whose material is opaque splits its draws into a whole one outside the band and a fading one
//   inside it (`levelDraws` with `split`), so most of its draws keep an opaque program; leaves fade
//   in their own cut-out. The stipple goes through the material's alpha test (`useStipple`): a
//   dropped pixel's threshold lies above any alpha, a kept pixel keeps its own test, so at 4x a
//   kept leaf pixel keeps its edge's alpha to coverage and under the temporal modes its coverage
//   band. The depth prepass copies the alpha test (VegetationPrepass.ts), so the leaves' shaded
//   draw stays an equal-depth draw.
// - Cameras: the view's perspective cameras draw through the cull. Any other (the sun's map, the
//   rain's collision bake: orthographic) draws every plant at its near level, whole, from arguments
//   of its own the CPU writes with each pool (`whole`): the draws whose band starts at 0 take all
//   their instances, the rest cast no shadow and leave the bake. So the shadows and the rain's map
//   stay functions of the plants alone, not of the view (the stepped shadow redraws and a grown
//   island's bake repeat a fresh load's). Every draw is indirect with an instance count of 1 on the
//   mesh, so the mesh keeps one render object and program whatever the camera.
// - A per-object uniform reads the drawn object's draw (`userData.plantDraw`) and camera, so a depth
//   copy or a bake proxy over the mesh's geometry draws as its mesh does once it runs the mesh's
//   `onBeforeRender` and carries the field (`followPlantDraw`).
// - Growth: a new land plants anew and the pool is rebuilt in steps (`prepare`: the records a few
//   dozen at a step, the buffers made and uploaded ahead), then swapped in by pointing the cull's
//   storage nodes at the new buffers (`swap`): the prototypes, materials and pipelines stay, so no
//   swap builds a pipeline (island/world/Growth.ts). The draws' boxes (the sun's caster box and the sea's
//   reflection box read them through `Box3.setFromObject`) follow the plants.

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

/** One plant: an instance of a prototype. */
export interface PlantInstance {
	prototype: number;
	/** its foot (m, world) */
	x: number; y: number; z: number;
	/** its turn about the vertical (rad): the prototype's +x to the way it leans */
	yaw: number;
	/** its stem's height (m), the prototype's times `scale` */
	height: number;
	/** the prototype's height scale, the tube's scale across (`girth`) and everything else's (`limb`) */
	scale: number; girth: number; limb: number;
	/** its lean (rad) and the bow at its top (m): `stemAxis` */
	lean: number; bow: number;
	/** its wind's load: its main bend's load in fair's mean wind (Deform.ts mainBend's lambda_0) */
	load: number;
	/** its stem's width (m), which sets where it changes level */
	width: number;
	/** how far its parts reach out from its stem's axis (m), for its bounding sphere */
	reach: number;
	/** four values of the kind's own */
	look: readonly [ number, number, number, number ];
}

/** One draw: a prototype's part at one level, over the distances of its band. */
export interface PlantDraw {
	/** the root it draws in (`PlantInstances.roots`) */
	part: number;
	prototype: number;
	level: 0 | 1;
	geometry: THREE.BufferGeometry;
	/** the shares of the switch's start distance it draws from and to */
	band: readonly [ number, number ];
	/** 1 the near level fading out over the switch's band, -1 the far one fading in, 0 whole */
	fade: - 1 | 0 | 1;
}

/** The draws of one part of a prototype: its near and far geometry over the switch (see the
 * header). With `split` (an opaque part) whole draws outside the band and fading ones inside it;
 * without, one draw a level, fading across the band. */
export function levelDraws( part: number, prototype: number, near: THREE.BufferGeometry, far: THREE.BufferGeometry, split: boolean ): PlantDraw[] {

	const end = 1 + LEVEL_SWITCH.band;
	const draw = ( level: 0 | 1, geometry: THREE.BufferGeometry, band: readonly [ number, number ], fade: - 1 | 0 | 1 ): PlantDraw => ( { part, prototype, level, geometry, band, fade } );
	return split ? [
		draw( 0, near, [ 0, 1 ], 0 ), draw( 0, near, [ 1, end ], 1 ),
		draw( 1, far, [ 1, end ], - 1 ), draw( 1, far, [ end, Infinity ], 0 ),
	] : [ draw( 0, near, [ 0, end ], 1 ), draw( 1, far, [ 1, Infinity ], - 1 ) ];

}

/** A stem's axis at s (0 foot, 1 top, of its height h), upright at the origin, leaning out along +x
 * by `lean` (rad) and bowing over by `bow` (m) at the top: ( out, up ) offsets from the foot. The
 * vertex stage bends the prototypes' upright axes by the same curve (`plantPlace`). */
export function stemAxis( s: number, h: number, lean: number, bow: number ): { out: number; up: number } {

	return { out: s * h * Math.sin( lean ) + bow * Math.pow( s, 2.5 ), up: s * h * Math.cos( lean ) - bow * 0.35 * Math.pow( s, 4 ) };

}

/** A prototype's point ( x, y, z ) hanging from share `s` of the stem (on its tube with `tube`)
 * placed at rest for `plant`, in the world: the vertex stage's `plantPlace` on the CPU. */
export function plantPoint( plant: PlantInstance, x: number, y: number, z: number, s: number, tube: boolean, out = new THREE.Vector3() ): THREE.Vector3 {

	const hang = s * plant.height;
	// on the tube: across by the girth, along by the height; off it, about the point it hangs from
	const qx = tube ? x * plant.girth : x * plant.limb;
	const qy = tube ? y * plant.scale : hang + ( y - hang / plant.scale ) * plant.limb;
	const qz = tube ? z * plant.girth : z * plant.limb;
	const c = Math.cos( plant.yaw ), n = Math.sin( plant.yaw );
	const axis = stemAxis( s, plant.height, plant.lean, plant.bow );
	// the axis's lean and bow at s, less the upright axis's height there
	return out.set( plant.x + qx * c - qz * n + c * axis.out, plant.y + qy + axis.up - hang, plant.z + qx * n + qz * c + n * axis.out );

}

/** The per-vertex nodes a plant material builds on (`PlantInstances.vertex`): every node of the
 * vertex stage unless named a varying. */
export interface PlantVertex {
	/** a prototype point placed at rest in the world: by the vertex's own `plantPart` */
	place( point: V3 ): V3;
	/** a prototype direction turned with the plant */
	turn( direction: V3 ): V3;
	/** the plant's foot, in the vertex stage and as a varying */
	foot: V3; footVarying: V3;
	/** its wind's load (Deform.ts mainBend's lambda_0) */
	load: F;
	/** the kind's own values, in the vertex stage and as a varying */
	look: V4; lookVarying: V4;
}

/** A prototype's vertex placed for a plant material's vertex stage (`placeVertex`). */
export interface PlacedVertex {
	/** its rest position (world) */
	position: V3;
	/** its normal, turned with the plant */
	normal: V3;
	/** a limb attribute (TreeMesh.ts BRANCH_ATTRIBUTES: a point and its yield) placed as its vertex */
	limb( name: string ): V4;
	/** the wind's `leafSway` (TreeSurfaces.ts CrownWind): x the prototype's
	 * ( 1 - stiffness ) eta^2 */
	sway: V4;
	/** the plant's wind load (Deform.ts mainBend's `load`) */
	load: F;
	/** the plant's foot, the wind's root; and as a varying, for the fragment's per-plant values */
	root: V3; rootVarying: V3;
	/** assigns the turned normal: the material's `VegetationMotion.beforePosition` */
	before(): void;
}

/** The vertex of a prototype mesh, its `leafSway` (x the prototype's eta^2) and the plant's wind
 * load, placed by `v`. */
export function placeVertex( v: PlantVertex ): PlacedVertex {

	const sway = attribute<'vec4'>( 'leafSway', 'vec4' );
	const normal = v.turn( normalGeometry );
	return {
		position: v.place( positionGeometry ),
		normal,
		limb: ( name ) => {

			const limb = attribute<'vec4'>( name, 'vec4' );
			return vec4( v.place( limb.xyz ), limb.w );

		},
		sway,
		load: v.load,
		root: v.foot, rootVarying: v.footVarying,
		before: () => {

			normalLocal.assign( normal );

		},
	};

}

/** The share of the stem's height each vertex hangs from and whether it lies on the stem's tube
 * (1) or off it (0): every prototype carries it. */
export const PLANT_PART = 'plantPart';

// Places a prototype's point p hanging from share s of the stem (on the tube when `tube` is 1) by
// the plant's place ( x, y, z, yaw ), form ( scale, girth, limb, height ) and bend ( lean, bow, ... ).
const plantPlace = /*@__PURE__*/ Fn( ( [ p, s, tube, place, form, bend ]: [ V3, F, F, V4, V4, V4 ] ) => {

	const h = form.w;
	const hang = s.mul( h );
	// on the tube: across by the girth, along by the height; off it, about the point it hangs from
	const onTube = vec3( p.x.mul( form.y ), p.y.mul( form.x ), p.z.mul( form.y ) );
	const offTube = vec3( 0, hang, 0 ).add( p.sub( vec3( 0, hang.div( form.x ), 0 ) ).mul( form.z ) );
	const q = mix( offTube, onTube, tube ).toVar();
	const c = cos( place.w ), n = sin( place.w );
	const turned = vec3( q.x.mul( c ).sub( q.z.mul( n ) ), q.y, q.x.mul( n ).add( q.z.mul( c ) ) );
	// the axis's lean and bow at s (`stemAxis`), less the upright axis's height there
	const s2 = s.mul( s );
	const out = hang.mul( sin( bend.x ) ).add( bend.y.mul( s2 ).mul( sqrt( s ) ) );
	const up = hang.mul( cos( bend.x ).sub( 1 ) ).sub( bend.y.mul( 0.35 ).mul( s2 ).mul( s2 ) );
	return place.xyz.add( turned ).add( vec3( c.mul( out ), up, n.mul( out ) ) );

} ).setLayout( { name: 'plantPlace', type: 'vec3', inputs: [
	{ name: 'p', type: 'vec3' }, { name: 's', type: 'float' }, { name: 'tube', type: 'float' },
	{ name: 'place', type: 'vec4' }, { name: 'form', type: 'vec4' }, { name: 'bend', type: 'vec4' },
] } );

type PlantRecord = { plant: PlantInstance; draw: PlantDraw };

/** A pool made ahead for `swap`: its arrays, its buffers uploaded with `renderer`, its plants' box. */
export interface PlantPool { pool: CullPool; buffers: CullBuffers; box: THREE.Box3; renderer: THREE.WebGPURenderer | null }

// sway room round a plant's sphere and box (m), past what the wind bends it within the sphere
const SWAY = 0.6;

/** A kind's plants (see the header): its draws' meshes under `roots` (one root a part), drawn from
 * one cull. `material` makes each draw's material once the instances' nodes exist (`vertex`). */
export class PlantInstances {

	readonly roots: THREE.Group[];
	readonly meshes: THREE.Mesh[] = [];
	readonly cull: VegetationCull<PlantRecord>;
	private readonly draws: PlantDraw[];
	// metres of distance per metre of stem width at the switch, for the cull's kernels (the view
	// camera's, set before each update)
	private readonly scale = uniform( 0 );
	private nodes: PlantVertex | null = null;
	// the far level's share in the switch's band and the draw's sign, a varying (`vertex`)
	private fade: THREE.Node<'vec2'> | null = null;
	// every draw's arguments for the cameras that draw every plant at its near level (see the
	// header): written from the pool on the CPU, so the draws keep one instance count and one
	// render object whatever the camera (a count above 1 would key a mesh's program by the object)
	private readonly whole: THREE.IndirectStorageBufferAttribute;

	constructor( options: { name: string; parts: string[]; draws: PlantDraw[]; plants: PlantInstance[]; material: ( draw: PlantDraw, plants: PlantInstances ) => THREE.Material } ) {

		this.draws = options.draws;
		this.roots = options.parts.map( ( name ) => Object.assign( new THREE.Group(), { name } ) );
		const scale = this.scale;
		const payload: CullPayload<PlantRecord> = {
			buffers: [ 'place', 'form', 'bend', 'look', 'level' ].map( ( name ) => ( { name, floats: 4, itemSize: 4 } ) ),
			origin: 'place',
			write: ( r, i, arrays ) => {

				const p = r.plant, d = r.draw;
				arrays.place.set( [ p.x, p.y, p.z, p.yaw ], i * 4 );
				arrays.form.set( [ p.scale, p.girth, p.limb, p.height ], i * 4 );
				arrays.bend.set( [ p.lean, p.bow, p.load, 0 ], i * 4 );
				arrays.look.set( p.look, i * 4 );
				arrays.level.set( [ d.band[ 0 ], Math.min( d.band[ 1 ], 1e9 ), d.fade, p.width ], i * 4 );

			},
			sphere: ( r, _geometry, spheres, i ) => {

				const p = r.plant, top = stemAxis( 1, p.height, p.lean, p.bow );
				// (the main bend turns a point about the foot, keeping its distance: past the chord's
				// half as the plant bows to the wind, so a share of its height besides)
				spheres.set( [ p.x + Math.cos( p.yaw ) * top.out / 2, p.y + top.up / 2, p.z + Math.sin( p.yaw ) * top.out / 2, Math.hypot( top.out, top.up ) / 2 + 0.3 * p.height + p.reach + SWAY ], i * 4 );

			},
			// a draw keeps its plants inside its band of the switch's start distance
			fold: { far: scale, folded: ( slot, origin, camera, node ) => {

				const level = node( 'level' ).element( slot ).toVar();
				const d = length( origin.sub( camera ) );
				const start = level.w.mul( scale );
				return d.lessThan( level.x.mul( start ) ).or( d.greaterThanEqual( level.y.mul( start ) ) );

			} },
		};
		this.cull = new VegetationCull( { name: options.name, payload, lod: { count: () => float( 1 ), on: uniform( 1 ), bands: uniform( 1, 'uint' ) } }, this.batches( options.plants ) );
		this.whole = new THREE.IndirectStorageBufferAttribute( new Uint32Array( this.draws.length * 5 ), 1 );
		this.draws.forEach( ( draw, k ) => {

			const range = this.cull.ranges[ k ];
			range.indexCount.value = draw.geometry.index!.count;
			const geometry = new THREE.BufferGeometry();
			geometry.name = draw.geometry.name;
			geometry.setIndex( draw.geometry.index );
			for ( const [ name, attribute ] of Object.entries( draw.geometry.attributes ) ) geometry.setAttribute( name, attribute );
			geometry.boundingBox = new THREE.Box3();
			geometry.boundingSphere = new THREE.Sphere();
			const mesh = new THREE.Mesh( geometry, options.material( draw, this ) );
			mesh.name = `${ geometry.name } ${ [ 'near', 'far' ][ draw.level ] }${ draw.fade !== 0 ? ' fading' : '' }`;
			// (the cull decides what draws: three's test would read the prototype's box at the origin)
			mesh.frustumCulled = false;
			mesh.castShadow = true;
			mesh.receiveShadow = true;
			mesh.userData.plantDraw = k;
			// the rain's bake and the sun's map draw the near level, whole (see the header)
			if ( draw.band[ 0 ] !== 0 ) {

				mesh.castShadow = false;
				mesh.userData.rainBake = false;

			}
			mesh.onBeforeRender = ( renderer, _scene, camera ) => {

				if ( ( camera as THREE.PerspectiveCamera ).isPerspectiveCamera ) {

					this.scale.value = switchScale( camera.projectionMatrix.elements[ 5 ] );
					this.cull.update( renderer as unknown as THREE.Renderer, camera );
					geometry.setIndirect( this.cull.indirect, range.indirectOffsets[ 0 ] );

				} else geometry.setIndirect( this.whole, k * 20 );

			};
			// (the arguments belong to this draw alone)
			mesh.onAfterRender = () => {

				( geometry as unknown as { indirect: unknown } ).indirect = null;

			};
			this.meshes.push( mesh );
			this.roots[ draw.part ].add( mesh );

		} );
		this.follow( this.box( options.plants ) );

	}

	/** The nodes every material of the kind places its vertices by (built once, shared). */
	vertex(): PlantVertex {

		if ( this.nodes ) return this.nodes;
		const cull = this.cull;
		// the drawn object's draw: its first slot in the pool and whether it draws directly
		const start = uniform( 0, 'uint' ).onObjectUpdate( ( { object } ) => object ? cull.ranges[ object.userData.plantDraw as number ]?.start ?? 0 : undefined );
		const direct = uniform( 0 ).onObjectUpdate( ( { camera } ) => camera ? ( ( camera as THREE.PerspectiveCamera ).isPerspectiveCamera ? 0 : 1 ) : undefined );
		const scale = uniform( 0 ).onObjectUpdate( ( { camera } ) => camera ? switchScale( camera.projectionMatrix.elements[ 5 ] ) : undefined );
		const at = start.add( instanceIndex );
		const slot = select( direct.greaterThan( 0.5 ), at, cull.node( 'visible', 'uint', true ).element( at ) ).toVar( 'plantSlot' );
		const read = ( name: string ) => cull.node( name, 'vec4', true ).element( slot ).toVar( `plant_${ name }` ) as unknown as V4;
		const place = read( 'place' ), form = read( 'form' ), bend = read( 'bend' ), look = read( 'look' ), level = read( 'level' );
		const part = attribute<'vec2'>( PLANT_PART, 'vec2' );
		const foot = place.xyz;
		// the far level's share in the switch's band, none drawn directly; the draw's sign
		const v = levelFade( length( foot.sub( cameraPosition ) ), level.w.mul( scale ) ).mul( float( 1 ).sub( direct ) );
		this.fade = varying( vec2( v, level.z ), 'vPlantFade' );
		this.nodes = {
			place: ( point ) => plantPlace( point, part.x, part.y, place, form, bend ),
			turn: ( d ) => {

				const c = cos( place.w ), n = sin( place.w );
				return vec3( d.x.mul( c ).sub( d.z.mul( n ) ), d.y, d.x.mul( n ).add( d.z.mul( c ) ) );

			},
			foot, footVarying: varying( foot, 'vPlantFoot' ),
			load: bend.z,
			look, lookVarying: varying( look, 'vPlantLook' ),
		};
		return this.nodes;

	}

	/** Cuts `material` (a fading draw's: `PlantDraw.fade` not 0) by the level stipple (see the
	 * header): call once its alpha test and coverage are set. */
	useStipple( material: THREE.NodeMaterial ): void {

		this.vertex();
		const fade = this.fade!;
		const d = stippleThreshold().sub( fade.x );
		// near (+1) keeps t >= v, far (-1) keeps t < v
		const keep = select( fade.y.greaterThan( 0 ), select( d.greaterThanEqual( 0 ), float( 1 ), float( 0 ) ), select( d.lessThan( 0 ), float( 1 ), float( 0 ) ) );
		// A dropped pixel's threshold lies above any alpha: three discards it, or at 4x gives it no
		// coverage, while a kept pixel keeps its own test (its edge's coverage, the temporal modes'
		// band). Added, not selected: the temporal band's threshold takes a derivative (Coverage.ts).
		const base = material.alphaTestNode ? material.alphaTestNode as F : float( material.alphaTest > 0 ? material.alphaTest : 0.5 );
		material.alphaTestNode = base.add( float( 1 ).sub( keep ).mul( 2 ) );

	}

	// each draw's records: every plant of its prototype, in the plants' order
	private batches( plants: PlantInstance[] ): CullBatchInput<PlantRecord>[] {

		return this.draws.map( ( draw ) => ( {
			records: plants.filter( ( p ) => p.prototype === draw.prototype ).map( ( plant ) => ( { plant, draw } ) ),
			geometry: draw.geometry, blades: 1, floor: 1, caps: [ 1 ], bladeIndices: draw.geometry.index!.count,
		} ) );

	}

	// the plants' box: the bounds of every plant's sphere (the cull's, `sphere` above)
	private box( plants: PlantInstance[] ): THREE.Box3 {

		const box = new THREE.Box3(), sphere = new THREE.Sphere();
		for ( const p of plants ) {

			const top = stemAxis( 1, p.height, p.lean, p.bow );
			sphere.center.set( p.x + Math.cos( p.yaw ) * top.out / 2, p.y + top.up / 2, p.z + Math.sin( p.yaw ) * top.out / 2 );
			sphere.radius = Math.hypot( top.out, top.up ) / 2 + 0.3 * p.height + p.reach + SWAY;
			box.union( sphere.getBoundingBox( new THREE.Box3() ) );

		}
		return box;

	}

	// the draws' counts and boxes from the pool shown
	private follow( box: THREE.Box3 ): void {

		const args = this.whole.array as Uint32Array;
		this.meshes.forEach( ( mesh, k ) => {

			const draw = this.draws[ k ], range = this.cull.ranges[ k ];
			// (first instance 0: an indirect draw honours another only with a feature; the vertex stage
			// adds the draw's start)
			args.set( [ draw.geometry.index!.count, draw.band[ 0 ] === 0 ? range.count : 0, 0, 0, 0 ], k * 5 );
			mesh.geometry.boundingBox!.copy( box );
			box.getBoundingSphere( mesh.geometry.boundingSphere! );

		} );
		this.whole.needsUpdate = true;

	}

	/** The pool of `plants` (the same prototypes and draws), its records `per` at a step and its
	 * buffers uploaded with `renderer`: what `swap` takes. */
	* prepare( plants: PlantInstance[], renderer: THREE.WebGPURenderer | null, per = 64 ): Generator<void, PlantPool> {

		const pool = yield* poolSteps( this.cull.payload, this.batches( plants ), per );
		const buffers = poolBuffers( this.cull.payload, pool );
		if ( renderer ) for ( const buffer of Object.values( buffers ) ) yield* uploadSteps( renderer, buffer, AttributeType.STORAGE );
		return { pool, buffers, box: this.box( plants ), renderer };

	}

	/** Swaps `prepared` in (see the header). Returns what frees the buffers it replaced, once
	 * nothing draws with them. */
	swap( prepared: PlantPool ): () => void {

		const old = this.cull.setPool( prepared.pool, prepared.buffers );
		this.follow( prepared.box );
		// (the app's one renderer uploaded the old pool too, ahead or at its first draw)
		const renderer = prepared.renderer;
		return () => {

			if ( renderer ) for ( const buffer of Object.values( old ) ) freeAttribute( renderer, buffer );

		};

	}

}

/** A copy of a plant draw (a depth copy, a bake proxy) over its mesh's geometry draws as the mesh
 * does: call it from the copy's `onBeforeRender`, which runs the mesh's first. */
export function followPlantDraw( copy: THREE.Object3D, mesh: THREE.Object3D ): void {

	if ( mesh.userData.plantDraw !== undefined ) copy.userData.plantDraw = mesh.userData.plantDraw;

}
