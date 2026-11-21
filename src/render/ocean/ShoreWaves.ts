/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { Fn, If, texture, uniform, uv, float, int, vec2, vec3, vec4, dot, length, sqrt, max, min, mix, smoothstep, exp, select, atan, fract, pow, mx_noise_float } from 'three/tsl';
import type { OceanFFT } from './OceanFFT.ts';
import type { Shared } from '../shared.ts';
import type { ShoreField } from '../RenderTypes.ts';
import { shoreBoundaryPixels, shoreShelterPixels, type FieldPixels } from './ShoreWaveField.ts';
import { waveDamping } from './WaveDamping.ts';
import type { ShoreFieldJob, ShoreFieldReply } from './shoreFieldWorker.ts';
import { TEMPORAL } from '../post/Antialiasing.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;

// The white water at the rock: the shore's wave fields as textures and the foam they drive.

// The white water at the rock (see the note at the class): `omega` the one frequency (rad/s) its
// phase takes the incident wave at (a 3.1 s wave's: the waves that reach the rock run about 3.3 s
// in the kept weathers), `crest` the share of a wave's cycle from the
// start of its surge to its crest; `energy` the incident amplitude (m) from which a wave striking
// square on pushes from nothing to all it can; `waves` how many waves' foam lies on the water at
// once (the latest and those before it, a period apart); `life` how long a wave's foam lasts (s),
// for a small push and a full one; `reach` how far (m) it spreads from the wall as the wave
// drains, small and full: the sea's everyday waves, a decimetre or two, leave a narrow band hugging
// the rock, not the wide fields of a surf day; `drift` how fast (m/s) the backwash carries it off
// after, at most `farthest` (m); `tongues` the scale (per metre) and pace (per second) of the noise
// that carries it farther out in some stretches than in others, `patches` those of the noise it
// breaks into as it ages. `lace` the faint lace under it all: its cover at most (the water draws a
// cover this low as lace alone), how far out it lies (m) where the rock is least and most exposed,
// the scale and pace of the noise it breathes with.
const FOAM = { omega: 2, crest: 0.15, energy: [ 0.006, 0.08 ], waves: 6, life: [ 2.5, 6 ], reach: [ 0.15, 0.5 ], drift: 0.12, farthest: 3,
	tongues: [ 0.4, 0.03 ], patches: [ 0.8, 0.06 ], lace: { cover: 0.2, reach: [ 0.3, 1.1 ], breathe: [ 0.15, 0.04 ] } };
const PERIOD = 2 * Math.PI / FOAM.omega;
// Each cascade's waves at the rock, for their amplitude: the frequency (rad/s) of the waves that
// carry most of its energy (deep water, omega^2 = g k), about 36, 15 and 4 m long; and how far
// upwave (m) the amplitude is read a second time, a quarter of the waves that reach the rock.
const CASCADE_OMEGA = [ 1.3, 2, 4 ];
const QUARTER = 4;

// The reflected surface's texel (m): the first island's 96 m field over 512 texels,
// kept as the shore field grows with the land, so the contact strip along the rock stays as sharp.
const TEXEL = 96 / 512;

// A field's texels as a texture (filtered, no mips).
function fieldTexture( field: FieldPixels, name: string ): THREE.DataTexture {

	const texture = new THREE.DataTexture( field.pixels, field.nx, field.nz, THREE.RGBAFormat, THREE.FloatType );
	texture.name = name;
	texture.minFilter = texture.magFilter = THREE.LinearFilter;
	texture.generateMipmaps = false; texture.flipY = false; texture.needsUpdate = true;
	return texture;

}

function shelterTexture( field: FieldPixels & { info: Record<string, unknown> } ): THREE.DataTexture {

	const texture = fieldTexture( field, 'shoreWaveShelter' );
	texture.userData.shoreWaveShelter = field.info;
	return texture;

}

/** A grown island's wave fields (ShoreWaves.fieldsFor), for `setFields`. */
export interface ShoreFields { shoreField: ShoreField; boundary: THREE.DataTexture; shelter: THREE.DataTexture; direction: THREE.Vector2 }

// Tidewater's FFT supplies every incident wave. This rocky boundary is a Ponlai extension:
// a weak mirrored response and whitewater use that wave's height and orbital velocity,
// without adding the independent, single-period beach wave over the open sea.
//
// The white water (A): where a wave runs into the rock it surges up the wall and turns white, then
// drains back, and the foam it leaves spreads seaward, thins into lace (the water's foam pattern,
// Water.ts) and fades. The foam follows the waves the water shows and nothing else: the incident
// wave just off the wall, the long and medium cascades as the water weights them, is taken as one
// oscillation of height h = a cos( phase ) and rise dh/dt = - a omega sin( phase ), so that its
// phase says how long ago the last crest struck (FOAM.omega: one frequency for the sea's mix of
// periods). How hard is the waves' envelope there: each cascade's amplitude at its own waves'
// frequency, averaged with the same read a quarter of a wave upwave, so it changes with the sets
// and not within each wave (read once, with the one frequency, it swung by a third within each wave
// and all the foam swelled and shrank with it). The sea's waves are short-crested and uneven, so where
// one runs high the rock turns white and where it runs low it does not, as the eye sees them
// arrive; both change smoothly with the clock (the phase jumps only where a is near nothing, where
// there is no foam to jump). A wave turns white only as it strikes: a wall in deep water does not
// break the wave before it. Noises over the coast and a count of the waves picked bursts and sets
// before, and a site's own threshold where it foamed; the count stepped whenever the phase did,
// and the foam jumped with it, and the fixed thresholds kept some rocks white and others bare: both
// read as fake. How hard a wave pushes on the wall is its amplitude, taken into its band, times
// the share of it that strikes the wall square on (the wall's facing against the swell's heading);
// the push sets how white the fresh foam is, how far it spreads and how long it stays, which
// together fall off about as the square of the facing, as a wave's push on a wall does (measured:
// at 45 degrees half the head's, at 60 a fifth, sliding along the flank almost none), so no white
// line chases round the island. A wave's foam does not go with it: it lasts several seconds,
// thinning to lace, so the foam of the waves before still lies on the water when the next strikes,
// and each new one brightens it and pushes it out rather than starting it again. So the foam is
// the sum of the latest wave's and of those before it, a period apart, each faded, thinned and
// drifted out by its age, at the push the latest has (the sea changes it only slowly). The backwash
// does not carry it out evenly: it runs out in tongues in some stretches and hugs the rock in
// others, and it breaks into patches as it ages, so it does not lie in a ring at one distance from
// the rock. Under it all lies a faint lace, as much as the rock's exposure brings (its facing, out
// of the island's wave shadow), breathing with a slow noise over the coast and the clock. The lee's waves are
// small (the water's weights, WaveDamping.ts) and its rock is not exposed, so it keeps no foam.
// A paused frame holds the same foam however it was reached.
//
// A grown island (Growth.ts) brings a new shore field: its wave fields are built in a worker
// (`fieldsFor`: shoreFieldWorker.ts, ShoreWaveField's plain-data functions, on a copy of the
// field the caller makes and hands over: the app keeps its own; their textures the caller
// uploads) and set in place
// (`setFields`: the base nodes' values, the surface targets resized to the field), so no material
// changes and no pipeline is built.
//
// The surface field is baked once per change of the scene clock. Under the temporal modes
// (Antialiasing.ts) it alternates between two targets, so the one baked before stays readable as
// the field a step of the clock ago (`previousSurfaceAt`): the water's motion vectors take its
// vertices' previous heights from it (Water.ts). `texture` and `previous` swap their targets at
// each bake; whoever samples them follows.
export class ShoreWaves {

	readonly direction = uniform( new THREE.Vector2() );
	readonly reflection = uniform( 0.25 );
	readonly boundary: THREE.TextureNode;
	readonly shelter: THREE.TextureNode;
	// RGBA: local reflected height, its x/z slopes, the white water's cover (0..1).
	readonly texture: THREE.TextureNode;
	// The surface field as baked before the current one (temporal modes; otherwise the current one).
	readonly previous: THREE.TextureNode;
	readonly surfaceAt: ( p: V2, time: F ) => THREE.Node<'vec4'>;
	/** The surface field at the clock's previous bake (see the note above). */
	readonly previousSurfaceAt: ( p: V2 ) => THREE.Node<'vec4'>;
	fieldRevision = 0;
	private shoreField: ShoreField;
	private revision = - 1;
	private renderer: THREE.WebGPURenderer;
	// the surface field's targets: [ current, previous ] (one under MSAA)
	private surfaces: THREE.RenderTarget[];
	private quad: THREE.QuadMesh;
	private lastTime: number | undefined;
	private lastReflection = - 1;

	// `shoreField`: the island's shore distance field (the island's world data `shore`), which `shore`
	// holds as a texture; `floor`: the seabed's height over the shore field's bounds
	// (WorldFields.floor), for the waves' shallow-water damping (as the water's, OceanFFT.attenuation)
	constructor( renderer: THREE.WebGPURenderer, shoreField: ShoreField, shared: Shared, shore: THREE.Texture | THREE.TextureNode, fft: OceanFFT,
		floor: THREE.Texture | THREE.TextureNode ) {

		this.renderer = renderer;
		this.shoreField = shoreField;
		this.setDirection( fft );
		this.boundary = texture( fieldTexture( shoreBoundaryPixels( shoreField ), 'shoreWaveBoundary' ) );
		this.shelter = texture( shelterTexture( shoreShelterPixels( shoreField, this.direction.value ) ) );
		this.revision = fft.spectrumRevision;
		const toUV = ( p: V2 ) => p.sub( shared.shoreXform.xy ).mul( shared.shoreXform.zw );
		// The white water at a distance sd from the wall, from the incident wave's height and rise there
		// ( h, dh/dt ), the wall's confidence and facing against the swell (1 square on, 0 along it),
		// the contact point and the scene's clock (see the note above).
		const whiteWater = Fn( ( [ wave, a, sd, confidence, facing, exposure, at, here, time ]: [ V2, F, F, F, F, F, V2, V2, F ] ) => {

			// 0 as the water starts to surge up the wall, the crest at FOAM.crest, then the drain
			const cycle = fract( atan( wave.y.div( FOAM.omega ).negate(), wave.x ).div( 2 * Math.PI ).add( 1 + FOAM.crest ) ).toVar();
			// this wave's push on the wall: its height, times the share striking it square on
			const push = smoothstep( FOAM.energy[ 0 ], FOAM.energy[ 1 ], a ).mul( max( facing, 0 ) ).toVar();
			// the backwash carries the foam farther out in some stretches than in others, and it breaks
			// into patches as it ages
			const tongue = mx_noise_float( vec3( at.mul( FOAM.tongues[ 0 ] ).add( 7.7 ), time.mul( FOAM.tongues[ 1 ] ) ) ).mul( 1.2 ).add( 1 ).clamp( 0.15, 2.2 ).toVar();
			const patches = smoothstep( - 0.25, 0.3, mx_noise_float( vec3( here.mul( FOAM.patches[ 0 ] ), time.mul( FOAM.patches[ 1 ] ) ) ) ).toVar();
			const life = mix( float( FOAM.life[ 0 ] ), float( FOAM.life[ 1 ] ), push ).toVar();
			const spreads = mix( float( FOAM.reach[ 0 ] ), float( FOAM.reach[ 1 ] ), push ).toVar();
			// The foam of this wave and of those before it, each struck a period before the next: the
			// latest turns white as its crest strikes, and each fades, thins and drifts out with its age
			// (the push of those before is taken as this one's, which the sea changes only slowly). A
			// wave's term at the end of its cycle is the next one's at the start of the next, so the sum
			// runs on through each new surge; the oldest fades out before it leaves the sum.
			const foam = float( 0 ).toVar();
			for ( let k = 0; k < FOAM.waves; k ++ ) {

				const age = max( cycle.sub( FOAM.crest ).add( k ).mul( PERIOD ), 0 );
				const appear = k === 0 ? smoothstep( 0, FOAM.crest * 0.8, cycle ) : float( 1 );
				const reach = spreads.mul( float( 1 ).sub( exp( age.div( - 0.8 ) ) ) ).add( age.mul( FOAM.drift ).mul( tongue ) ).add( 0.06 ).min( FOAM.farthest );
				// (its edge thins over two texels at least: a sharper edge steps along the texels)
				const spread = float( 1 ).sub( smoothstep( reach.mul( 0.3 ), reach.add( 2 * TEXEL ), sd ) ).mul( mix( float( 1 ), float( 0.45 ), min( sd.div( reach ), 1 ) ) );
				foam.addAssign( appear.mul( exp( age.negate().div( life ) ) ).div( age.mul( 0.25 ).add( 1 ) ).mul( spread )
					.mul( mix( float( 1 ), patches, smoothstep( 0.5, 4, age ) ) )
					.mul( float( 1 ).sub( smoothstep( ( FOAM.waves - 1.5 ) * PERIOD, ( FOAM.waves - 0.2 ) * PERIOD, age ) ) ) );

			}
			const fresh = float( 1 ).sub( exp( foam.mul( - 1.3 ) ) ).mul( pow( push, 0.7 ) ).mul( smoothstep( 0, 0.15, push ) );
			// the faint lace under it all, as much as the rock's exposure brings, breathing slowly along
			// the coast
			const breathe = mx_noise_float( vec3( at.mul( FOAM.lace.breathe[ 0 ] ), time.mul( FOAM.lace.breathe[ 1 ] ) ) ).add( 0.55 ).clamp( 0, 1 );
			const lace = exposure.mul( breathe ).toVar();
			const laceReach = mix( float( FOAM.lace.reach[ 0 ] ), float( FOAM.lace.reach[ 1 ] ), lace ).mul( tongue );
			const left = lace.mul( FOAM.lace.cover ).mul( patches.mul( 0.7 ).add( 0.3 ) ).mul( float( 1 ).sub( smoothstep( laceReach.mul( 0.2 ), laceReach.add( 2 * TEXEL ), sd ) ) )
				.mul( mix( float( 1 ), float( 0.4 ), min( sd.div( laceReach ), 1 ) ) );
			return max( fresh, left ).mul( confidence ).mul( smoothstep( 0, 0.02, sd ) );

		} ).setLayout( { name: 'rockWhiteWater', type: 'float', inputs: [ { name: 'wave', type: 'vec2' }, { name: 'a', type: 'float' }, { name: 'sd', type: 'float' },
			{ name: 'confidence', type: 'float' }, { name: 'facing', type: 'float' }, { name: 'exposure', type: 'float' }, { name: 'at', type: 'vec2' },
			{ name: 'here', type: 'vec2' }, { name: 'time', type: 'float' } ] } );
		const damping = { fft, shared, shore, seabedHeight: floor, shoreWaves: this };
		// Sampling stays in each material graph; reusable layout functions take explicit values,
		// so their uniforms/samplers cannot be cached from a different render pass.
		// The incident wave at p: ( height, its rise dh/dt, its squared amplitude ) of the long and
		// medium cascades, weighted as the water weights them; the amplitude each cascade's own, at its
		// waves' frequency. The finest cascade's capillary detail must not spawn separate white
		// dots.
		const incidence = ( p: V2 ) => Fn( () => {

			const w = waveDamping( damping, p ).weights.toVar();
			const ws = [ w.x, w.y, w.z ];
			const surface = vec3( 0 ).toVar();
			for ( let i = 0; i < 3; i ++ ) {

				const at = p.div( fft.sizes[ i ] );
				const height = texture( fft.displacement, at ).depth( int( i ) ).level( float( 0 ) ).y.mul( ws[ i ] ).toVar();
				const rise = texture( fft.velocity, at ).depth( int( i ) ).level( float( 0 ) ).z.mul( ws[ i ] ).toVar();
				surface.addAssign( vec3( height, rise, height.mul( height ).add( rise.div( CASCADE_OMEGA[ i ] ).pow( 2 ) ) ) );

			}
			return surface;

		} )();
		// `at` passes the shore distance and boundary already read at p's texel, so the centre
		// sample does not read them a second time.
		const reflectionHeight = ( p: V2, at?: { q: V2; sd: F; boundary: THREE.Node<'vec4'> } ) => Fn( () => {

			const q = at ? at.q : toUV( p );
			const sd = at ? at.sd : texture( shore, q ).level( float( 0 ) ).r;
			const height = float( 0 ).toVar();
			If( sd.greaterThan( - 0.8 ).and( sd.lessThan( 3.5 ) ), () => {

				const boundary = at ? at.boundary : this.boundary.sample( q ).level( float( 0 ) );
				const normal = boundary.ba.div( max( length( boundary.ba ), 1e-5 ) );
				const waterP = p.add( normal.mul( max( sd.negate(), 0 ) ) );
				const waterSD = max( sd, 0 );
				const mirror = waterP.sub( normal.mul( waterSD.mul( 2 ) ) );
				// (the long and medium cascades as the water weights them at the mirrored point)
				const w = waveDamping( damping, mirror ).weights.toVar();
				const heightAt = ( c: number ) => texture( fft.displacement, mirror.div( fft.sizes[ c ] ) ).depth( int( c ) ).level( float( 0 ) ).y;
				const reflected = heightAt( 0 ).mul( w.x ).add( heightAt( 1 ).mul( w.y ) ).add( heightAt( 2 ).mul( w.z ).mul( 0.7 ) );
				// Extend height into solid rock for filtering/derivatives only, with no land flow or foam.
				// A ghost sample mirrored across a locally planar wall shares the incoming phase.
				// It is a weak boundary approximation, not a reflection/diffraction fluid solver.
				height.assign( reflected
					.mul( boundary.g ).mul( this.reflection ).mul( exp( sd.mul( - 0.8 ) ) )
					.mul( float( 1 ).sub( smoothstep( 2, 3.5, sd ) ) ) );

			} );
			return height;

		} )();
		// Every channel is zero away from the rock: the reflection fades out by 3.5 m from it (its
		// slopes read 0.12 m to either side) and the white water by FOAM.farthest. Only that strip
		// evaluates the waves, so the pass costs by the coast's length rather than the field's area,
		// which grows with the land.
		const e = 0.12;
		const surface = ( p: V2 ) => Fn( () => {

			const q = toUV( p ).toVar(), edge = min( q.div( shared.shoreXform.zw ), vec2( 1 ).sub( q ).div( shared.shoreXform.zw ) );
			const sd = texture( shore, q ).level( float( 0 ) ).r.toVar();
			const out = vec4( 0 ).toVar();
			If( sd.greaterThan( - 1 ).and( sd.lessThan( 3.75 ) ), () => {

				const boundary = this.boundary.sample( q ).level( float( 0 ) ).toVar();
				const normal = boundary.ba.div( max( length( boundary.ba ), 1e-5 ) );
				const impact = float( 0 ).toVar();
				If( sd.greaterThan( 0 ).and( sd.lessThan( FOAM.farthest + 2 * TEXEL ) ), () => {

					const contactP = p.sub( normal.mul( sd ) ).add( normal.mul( 0.16 ) );
					// (the wall's normal points into the water; the swell runs along `direction`)
					const facing = dot( normal, this.direction ).negate().toVar();
					// the rock's exposure: square on to the swell and out of the island's wave shadow (read
					// a little off the rock, where the shadow field is not blurred into the land)
					const open = this.shelter.sample( toUV( contactP.add( normal.mul( 1.5 ) ) ) ).level( float( 0 ) ).r;
					const exposure = smoothstep( 0.35, 0.9, open ).mul( max( facing, 0 ) );
					// The amplitude read at the rock alone rises and falls within each wave wherever the waves'
					// period is not the one assumed, and all the foam swelled and shrank with every wave (a
					// third either way, measured). Read again a quarter of a wave upwave, where the wave is a
					// quarter of a cycle on, the two swings cancel: the mean is the waves' envelope, which
					// changes with the sets.
					const wave = incidence( contactP ).toVar();
					const a = sqrt( wave.z.add( incidence( contactP.sub( this.direction.mul( QUARTER ) ) ).z ).mul( 0.5 ) );
					impact.assign( whiteWater( wave.xy, a, sd, boundary.g, facing, exposure, contactP, p, shared.time ) );

				} );
				const inWater = sd.greaterThan( 0 ).and( min( edge.x, edge.y ).greaterThan( 1 ) );
				const dx = reflectionHeight( p.add( vec2( e, 0 ) ) ).sub( reflectionHeight( p.sub( vec2( e, 0 ) ) ) ).div( 2 * e );
				const dz = reflectionHeight( p.add( vec2( 0, e ) ) ).sub( reflectionHeight( p.sub( vec2( 0, e ) ) ) ).div( 2 * e );
				out.assign( vec4( reflectionHeight( p, { q, sd, boundary } ), dx, dz, select( inWater, impact, float( 0 ) ) ) );

			} );
			return out;

		} )();
		const [ width, height ] = surfaceSize( shoreField );
		const field = ( name: string ) => {

			const target = new THREE.RenderTarget( width, height, { type: THREE.HalfFloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false, generateMipmaps: false } );
			target.texture.name = name;
			return target;

		};
		this.surfaces = TEMPORAL ? [ field( 'rockWaveSurface' ), field( 'rockWaveSurfacePrevious' ) ] : [ field( 'rockWaveSurface' ) ];
		this.texture = texture( this.surfaces[ 0 ].texture );
		this.previous = texture( this.surfaces[ this.surfaces.length - 1 ].texture );
		const material = new THREE.NodeMaterial();
		material.fragmentNode = Fn( () => surface( uv().div( shared.shoreXform.zw ).add( shared.shoreXform.xy ) ) )();
		this.quad = new THREE.QuadMesh( material );
		this.surfaceAt = ( p, _time ) => this.texture.sample( toUV( p ) ).level( float( 0 ) );
		this.previousSurfaceAt = ( p ) => this.previous.sample( toUV( p ) ).level( float( 0 ) );

	}

	update( fft: OceanFFT, time?: number ): void {

		if ( this.revision !== fft.spectrumRevision ) {

			const previous = this.direction.value.clone();
			this.setDirection( fft );
			if ( ! previous.equals( this.direction.value ) ) {

				const old = this.shelter.value;
				this.shelter.value = shelterTexture( shoreShelterPixels( this.shoreField, this.direction.value ) );
				old.dispose();

			}
			this.revision = fft.spectrumRevision;
			this.fieldRevision ++;
			this.lastTime = undefined;

		}
		// (a paused page whose wind changes rebuilds the FFT's spectrum, above, and bakes again)
		if ( time === undefined || ( time === this.lastTime && this.lastReflection === this.reflection.value ) ) return;
		const previousTarget = this.renderer.getRenderTarget();
		// (temporal modes: the surface is baked into the target that held the field before last)
		if ( this.surfaces.length > 1 ) {

			this.surfaces.reverse();
			this.texture.value = this.surfaces[ 0 ].texture;
			this.previous.value = this.surfaces[ 1 ].texture;

		}
		this.renderer.setRenderTarget( this.surfaces[ 0 ] );
		this.quad.render( this.renderer );
		this.renderer.setRenderTarget( previousTarget );
		this.lastTime = time;
		this.lastReflection = this.reflection.value;

	}

	/** The wave fields of a grown island's shore field, built in a worker from `shore` (a copy of
	 * `shoreField.data`, transferred to it); their textures are not uploaded yet. */
	fieldsFor( shoreField: ShoreField, shore: Float32Array ): Promise<ShoreFields> {

		const worker = new Worker( new URL( './shoreFieldWorker.ts', import.meta.url ), { type: 'module' } );
		const direction = this.direction.value.clone();
		return new Promise<ShoreFields>( ( resolve, reject ) => {

			worker.onmessage = ( e: MessageEvent<ShoreFieldReply> ) => {

				worker.terminate();
				resolve( { shoreField, boundary: fieldTexture( e.data.boundary, 'shoreWaveBoundary' ), shelter: shelterTexture( e.data.shelter ), direction } );

			};
			worker.onerror = ( e ) => { worker.terminate(); reject( e ); };
			worker.postMessage( { shore: { ...shoreField, data: shore }, direction: { x: direction.x, y: direction.y } } satisfies ShoreFieldJob, { transfer: [ shore.buffer ] } );

		} );

	}

	/** Takes a grown island's wave fields (`fieldsFor`): their textures on the nodes, the surface
	 * targets sized to its shore field, the surface baked again on the next update. Returns the
	 * textures replaced, to dispose of once no frame draws with them. (A swell that turned since
	 * `fieldsFor` has its shelter field made here, as a turn makes it on the main thread.) */
	setFields( fields: ShoreFields ): THREE.Texture[] {

		const old = [ this.boundary.value, this.shelter.value ];
		this.shoreField = fields.shoreField;
		this.boundary.value = fields.boundary;
		if ( fields.direction.equals( this.direction.value ) ) this.shelter.value = fields.shelter;
		else {

			fields.shelter.dispose();
			this.shelter.value = shelterTexture( shoreShelterPixels( fields.shoreField, this.direction.value ) );

		}
		const [ width, height ] = surfaceSize( fields.shoreField );
		for ( const target of this.surfaces ) if ( target.width !== width || target.height !== height ) target.setSize( width, height );
		this.fieldRevision ++;
		this.lastTime = undefined;
		return old;

	}

	private setDirection( fft: OceanFFT ): void {

		const angle = fft.swellDirection * Math.PI / 180;
		this.direction.value.set( - Math.cos( angle ), - Math.sin( angle ) );

	}

}

// The reflected surface's targets for a shore field: TEXEL over its extent.
function surfaceSize( shore: ShoreField ): [ number, number ] {

	return [ Math.ceil( shore.nx * shore.cell / TEXEL ), Math.ceil( shore.nz * shore.cell / TEXEL ) ];

}
