/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, int, ivec2, vec2, vec3, vec4, uint, instanceIndex, instancedArray, storage, texture,
	clamp, max, mix, exp, smoothstep, select, floor, fract, abs, dot, mx_noise_float,
} from 'three/tsl';
import type { OceanFFT } from './OceanFFT.ts';
import type { ShoreWaves } from './ShoreWaves.ts';
import { LEE } from './WaveDamping.ts';
import type { Shared } from '../shared.ts';
import { createFoamPattern } from './FoamPattern.ts';
import { Spray, type SprayLighting } from './Spray.ts';
import type { SceneLatents } from '../post/ExtraPass.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type IV2 = THREE.Node<'ivec2'>;

// The storm sea's memory: persistent whitecap foam and the spray torn off breaking crests.
//
// The sea is calm in everyday weather: no whitecaps on the open water (the foam the waves leave
// at the rock is ShoreWaves'). Only a storm's gale builds a rough sea with whitecaps on folding crests, foam
// that trails behind them and fades, and spray.
// Storm foam and spray are lit by the scene, never self-lit.
//
// Whitecaps are born where the sea's horizontal Jacobian nears a fold (Tessendorf's choppy-wave
// criterion), but real foam outlives the crest that made it: it trails behind the breaker and
// fades over seconds. After tidewater's OceanFFT, foam therefore integrates in time:
//   foam = foam * exp( -decay dt ) + generation * add * dt, at least the fresh generation.
// Tidewater keeps that per FFT cascade; here a single cascade folds rarely, and a short tile
// would repeat its whitecaps in step, so the foam lives on one world-space grid that sees the
// summed cascades with the same shore shelter as the water's own displacement. The grid sits in
// the water's undisplaced (Lagrangian) coordinates, so foam rides the orbiting water rather than
// the travelling wave form, and a crest leaves its foam behind as it moves on.
//
// The grid is a torus: cell n of the world lives in texel n mod RES, so the window can follow the
// camera in whole cells without copying; cells that wrap into a new place start empty. The water
// reads it as a storage buffer (foam, fresh breaking) with its own bilinear filter: its fragment
// stage already binds as many sampled textures as WebGPU guarantees. It fades out toward the
// window's edge, and where a pixel covers several cells (the pixel's own fold takes over there).
//
// Determinism: a paused frame must be bit-identical, so the state is a function of scene
// time and weather. A jump in time, or a change of the weather while the clock stands still,
// rebuilds the state by replaying the last seconds at fixed steps (the FFT is a closed-form
// function of time); live frames integrate their own time step. The window follows the camera with
// slack, so it depends on where the storm was first seen from: a scene jump forgets the state
// (`reset`), and the next update places the window and replays as on a new page.

const RES = 512;
const CELL = 0.625; // m
const SPAN = RES * CELL;
const MOVE = 64; // window moves in steps of this many cells
const WARM_SECONDS = 6;
const WARM_STEP = 0.1;
const MAX_LIVE_STEP = 0.25;

export class StormSea {

	readonly params = {
		// Generation ramps up where the summed Jacobian falls from foamBias by foamWidth. At the
		// field's cell size the storm sea's Jacobian is narrow (mean ~1, deviation ~0.09, it never
		// folds outright), so the bias sits about two deviations down: only the steepest crests
		// break, a few percent of the sea, instead of the compressed half of every wave.
		foamBias: uniform( 0.86 ),
		foamWidth: uniform( 0.07 ),
		foamDecay: uniform( 0.5 ), // 1/s: a trail thins over a few seconds
		foamAdd: uniform( 1.5 ),
		groupDepth: uniform( 0.025 ), // how far wave groups move the threshold
		groupSpeed: uniform( 2.5 ), // m/s downwind
	};
	/** Scales the whitecap coverage the water draws from the field. */
	readonly coverage = uniform( 1 );
	/** Diffuse albedo of the foam under the scene's light: at most as bright as a white surface
	 * lit by the same sky, so it reads grey-blue under a storm sky and never glows. Aged lace is
	 * a thinner film than that (its opacity, in the water shader). */
	readonly albedo = uniform( 1 );
	/** Repeats of the foam pattern per metre, and its size in texels. */
	readonly patternScale = 0.09;
	readonly patternSize = 1024;
	/** Storm strength (0 calm): gates foam generation and spray. */
	readonly gate = uniform( 0 );
	/** World xz of the window's centre; the water fades the field out toward its edge. */
	readonly center = uniform( new THREE.Vector2() );
	readonly span = SPAN;
	readonly pattern: THREE.StorageTexture;
	readonly spray: Spray;

	private renderer: THREE.WebGPURenderer;
	private fft: OceanFFT;
	private state;
	private stateRead;
	private dt = uniform( 0 );
	private clock = uniform( 0 );
	private lo = uniform( new THREE.Vector2() ); // first world cell of the window
	private previousLo = uniform( new THREE.Vector2() );
	private stepKernel: THREE.ComputeNode;
	private clearKernel: THREE.ComputeNode;
	private time: number | undefined;
	private signature = '';
	private active = false;
	private readonly desired = new THREE.Vector2();
	private readonly forward = new THREE.Vector3();

	constructor( renderer: THREE.WebGPURenderer, fft: OceanFFT, shared: Shared, shore: THREE.Texture | THREE.TextureNode, shoreWaves: ShoreWaves,
		light: Omit<SprayLighting, 'wind' | 'waterLevel'> ) {

		this.renderer = renderer;
		this.fft = fft;
		this.pattern = createFoamPattern( renderer, this.patternSize );
		this.state = instancedArray( RES * RES, 'vec2' ).setName( 'stormFoamState' ); // foam, fresh breaking
		this.stateRead = storage( this.state.value, 'vec2', RES * RES ).toReadOnly().setName( 'stormFoamRead' );

		const sizes = fft.sizes;
		const cascade = ( field: THREE.Texture, c: number, at: V2 ) => texture( field, at.div( sizes[ c ] ) ).depth( int( c ) );
		// The water's own shore shelter of each cascade (Water.ts vertex stage: the two long ones
		// alike), with the fine cascade at full weight: breaking is physical, not a level of detail.
		const cascadeWeights = ( xz: V2 ): THREE.Node<'vec4'> => {

			const q = xz.sub( shared.shoreXform.xy ).mul( shared.shoreXform.zw );
			const sd = texture( shore, q ).level( float( 0 ) ).r;
			const shelter = shoreWaves.shelter.sample( q ).level( float( 0 ) ).r;
			const coastDamp = mix( float( 0.45 ), float( 1 ), smoothstep( 0, 7, sd ) );
			const long = mix( float( LEE.long ), float( 1 ), shelter );
			return vec4( long, long, mix( float( LEE.medium ), float( 1 ), shelter ), 1 ).mul( coastDamp );

		};
		const weightOf = ( w: THREE.Node<'vec4'>, c: number ) => [ w.x, w.y, w.z, w.w ][ c ];
		// A cell spans many texels of the shorter cascades: read them through their mip chains,
		// filtered to the cell, instead of aliasing their fine chop into false folds.
		const lod = ( c: number ) => Math.max( 0, Math.log2( CELL / ( sizes[ c ] / 256 ) ) );

		const cellOf = ( texel: IV2, lo: THREE.Node<'vec2'> ): V2 => {

			// the world cell n with n = texel ( mod RES ) inside the window starting at lo
			const d = vec2( texel ).sub( lo );
			return lo.add( d.sub( floor( d.div( RES ) ).mul( RES ) ) );

		};

		this.clearKernel = Fn( () => {

			this.state.element( instanceIndex ).assign( vec2( 0 ) );

		} )().compute( RES * RES, [ 64 ] ).setName( 'StormSea.Clear' );

		const P = this.params;
		this.stepKernel = Fn( () => {

			const texel = ivec2( int( instanceIndex.mod( uint( RES ) ) ), int( instanceIndex.div( uint( RES ) ) ) ).toVar();
			const n = cellOf( texel, this.lo ).toVar();
			const moved = abs( n.sub( cellOf( texel, this.previousLo ) ) ).dot( vec2( 1 ) ).greaterThan( 0.5 );
			const xz = n.add( 0.5 ).mul( CELL ).toVar();
			const w = cascadeWeights( xz ).toVar();
			// (the horizontal Jacobian without its cross term, as tidewater's water surface takes it)
			const jxx = float( 1 ).toVar(), jzz = float( 1 ).toVar();
			for ( let c = 0; c < sizes.length; c ++ ) {

				const d = cascade( fft.derivatives, c, xz ).level( float( lod( c ) ) );
				const weight = weightOf( w, c );
				jxx.addAssign( d.z.mul( weight ) );
				jzz.addAssign( d.w.mul( weight ) );

			}
			const J = jxx.mul( jzz ).toVar();
			// Waves break in groups, not on every crest of a tile: a soft noise of group size,
			// travelling downwind at about a group velocity, switches breaking on in some groups and
			// off in others, and nudges the threshold a little within them. It also hides the shorter
			// cascade's tile, which would otherwise break in step. The noise only gates crests the
			// Jacobian already picked out: if it moved the threshold by more than the Jacobian's own
			// spread, the foam would take the noise's round blobs instead of the crests' shape.
			const wind = shared.wind.direction;
			const along = dot( xz, wind ).sub( this.clock.mul( P.groupSpeed ) ), across = dot( xz, vec2( wind.y.negate(), wind.x ) );
			const groups = mx_noise_float( vec3( along.div( 42 ), across.div( 26 ), this.clock.mul( 0.03 ) ) ).toVar();
			const bias = P.foamBias.add( groups.mul( P.groupDepth ) );
			const generation = clamp( bias.sub( J ).div( P.foamWidth ), 0, 1 ).mul( smoothstep( - 0.25, 0.3, groups ) )
				.mul( this.gate.min( 1 ) ).toVar();
			// A crest's foam is dense while it breaks and then decays; it never stays thicker than
			// the fresh mat (1).
			const previous = select( moved, float( 0 ), this.state.element( instanceIndex ).x );
			const foam = previous.mul( exp( P.foamDecay.mul( this.dt ).negate() ) ).add( generation.mul( P.foamAdd ).mul( this.dt ) );
			const next = clamp( max( foam, generation ), 0, 1 ).toVar();
			this.state.element( instanceIndex ).assign( vec2( next, generation ) );

		} )().compute( RES * RES, [ 64 ] ).setName( 'StormSea.Foam' );

		const surfaceOffset = ( xz: V2 ): V3 => {

			const w = cascadeWeights( xz ).toVar();
			let offset: V3 = vec3( 0 );
			for ( let c = 0; c < sizes.length; c ++ ) {

				offset = offset.add( cascade( fft.displacement, c, xz ).level( float( 0 ) ).xyz.mul( weightOf( w, c ) ) );

			}
			return offset;

		};
		const surfaceVelocity = ( xz: V2 ): V3 => {

			const w = cascadeWeights( xz ).toVar();
			let velocity: V3 = vec3( 0 );
			for ( let c = 0; c < sizes.length; c ++ ) {

				const v = cascade( fft.velocity, c, xz ).level( float( 0 ) ); // ( Ux, Uz, Vy )
				velocity = velocity.add( vec3( v.x, v.z, v.y ).mul( weightOf( w, c ) ) );

			}
			return velocity;

		};
		this.spray = new Spray( renderer, {
			breakingAt: ( xz ) => {

				const n = floor( xz.div( CELL ) );
				const t = n.sub( floor( n.div( RES ) ).mul( RES ) );
				const d = n.sub( this.lo );
				const inside = d.x.greaterThanEqual( 0 ).and( d.y.greaterThanEqual( 0 ) ).and( d.x.lessThan( RES ) ).and( d.y.lessThan( RES ) );
				return select( inside, this.state.element( int( t.y ).mul( RES ).add( int( t.x ) ) ).y, float( 0 ) );

			},
			surfaceOffset, surfaceVelocity,
		}, { ...light, wind: shared.wind.velocity, waterLevel: shared.waterLevel } );
		renderer.compute( this.clearKernel );

	}

	/** Foam for the water shader at an undisplaced xz: ( persistent, fresh, weight ), bilinear.
	 * The weight fades the field out at the window's edge and where the pixel's footprint (m)
	 * spans several cells, which the unfiltered buffer would alias. */
	foamAt( xz: V2, footprint: F ): V3 {

		const g = xz.div( CELL ).sub( 0.5 ).toVar();
		const n0 = floor( g ).toVar();
		const f = fract( g ).toVar();
		const at = ( dx: number, dz: number ) => {

			const n = n0.add( vec2( dx, dz ) );
			const t = n.sub( floor( n.div( RES ) ).mul( RES ) );
			return this.stateRead.element( int( t.y ).mul( RES ).add( int( t.x ) ) );

		};
		const s = mix( mix( at( 0, 0 ), at( 1, 0 ), f.x ), mix( at( 0, 1 ), at( 1, 1 ), f.x ), f.y );
		const edge = abs( xz.sub( this.center ) ).div( SPAN );
		const inside = float( 1 ).sub( smoothstep( 0.38, 0.47, max( edge.x, edge.y ) ) )
			.mul( float( 1 ).sub( smoothstep( 1, 2.5, footprint.div( CELL ) ) ) );
		return vec3( s.x, s.y, inside );

	}

	/** Forgets the storm's state, as a new page has none (a scene jump, assemble/SceneJump.ts `establishScene`):
	 * the window is placed afresh and the history replayed on the next update. The window follows the
	 * camera with slack, so without this a storm seen from the scene before kept that scene's window. */
	reset(): void {

		this.active = false;
		this.time = undefined;
		this.signature = '';

	}

	/** After the FFT has been evaluated at `time` this frame. */
	update( time: number, storm: number, camera: THREE.Camera ): void {

		this.gate.value = storm;
		this.spray.params.gate.value = storm;
		if ( storm <= 1e-3 ) {

			// Calm: nothing to integrate. A later storm starts from a clean sea.
			this.active = false;
			this.spray.mesh.visible = false;
			this.time = time;
			return;

		}
		this.spray.mesh.visible = true;
		this.spray.eye.value.copy( camera.position );

		// Keep the window ahead of the camera, moving it in whole steps (with some slack) so the
		// state rarely has to drop cells.
		camera.getWorldDirection( this.forward );
		const ahead = Math.hypot( this.forward.x, this.forward.z ) > 1e-3 ? SPAN * 0.25 / Math.hypot( this.forward.x, this.forward.z ) : 0;
		this.desired.set( camera.position.x + this.forward.x * ahead, camera.position.z + this.forward.z * ahead );
		const step = MOVE * CELL;
		const lo = this.lo.value;
		const centre = this.center.value;
		if ( ! this.active || Math.max( Math.abs( this.desired.x - centre.x ), Math.abs( this.desired.y - centre.y ) ) > step ) {

			lo.set( Math.round( this.desired.x / step ) * step / CELL - RES / 2, Math.round( this.desired.y / step ) * step / CELL - RES / 2 );

		}
		this.center.value.set( ( lo.x + RES / 2 ) * CELL, ( lo.y + RES / 2 ) * CELL );

		const signature = `${ storm.toFixed( 4 ) },${ this.fft.spectrumRevision },${ lo.x },${ lo.y }`;
		const delta = this.time === undefined ? Infinity : time - this.time;
		const onset = ! this.active;
		// A jump in time, or anything new while the clock stands still, replays the history. A
		// storm that sets in on a running clock builds its foam as it goes.
		const rebuild = delta < 0 || delta > MAX_LIVE_STEP || ( delta === 0 && ( onset || signature !== this.signature ) );
		this.signature = signature;
		this.time = time;
		this.active = true;
		if ( rebuild ) {

			this.replay( time );
			return;

		}
		if ( onset ) this.clear();
		if ( delta === 0 ) return;
		this.integrate( delta, time );
		this.previousLo.value.copy( lo );

	}

	private integrate( dt: number, time: number ): void {

		this.dt.value = dt;
		this.clock.value = time;
		this.renderer.compute( this.stepKernel );
		this.spray.step( dt, Math.round( time * 1000 ) );

	}

	// Replay the last seconds at fixed steps, ending at `time` (the FFT is left evaluated there).
	private clear(): void {

		this.renderer.compute( this.clearKernel );
		this.spray.clear();
		this.previousLo.value.copy( this.lo.value );

	}

	private replay( time: number ): void {

		this.clear();
		const steps = Math.round( WARM_SECONDS / WARM_STEP );
		for ( let s = steps; s >= 0; s -- ) {

			const t = time - s * WARM_STEP;
			if ( t < 0 ) continue;
			this.fft.update( 0, t );
			this.integrate( s === steps ? 0 : WARM_STEP, t );

		}

	}

	/** Registers the storm's draws and kernels with the frame graph's water pass (MSAA draws the
	 * spray in the water scene): the spray's mesh, the foam's step and clear and the spray's
	 * kernels, so the graph's warm-up compiles them before a storm's onset, without running them
	 * (the calm sea's state stays as it is). Call it with the spray's mesh in the water scene. */
	registerLatent( graph: SceneLatents ): Promise<void> {

		return graph.latent( 'water', [ this.spray.mesh ], [ this.stepKernel, this.clearKernel, ...this.spray.kernels ] );

	}

	dispose(): void {

		this.stepKernel.dispose();
		this.clearKernel.dispose();
		this.pattern.dispose();
		this.spray.dispose();

	}

}
