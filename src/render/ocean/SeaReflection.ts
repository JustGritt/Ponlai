/*!
 * The reflection denoiser adapted from the AMD FidelityFX SDK 1.1.4, Denoiser
 * (https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, ffx_denoiser_reflections_reproject.h,
 * ffx_denoiser_reflections_prefilter.h, ffx_denoiser_reflections_resolve_temporal.h,
 * ffx_denoiser_reflections_common.h, ffx_denoiser_reflections_config.h), and the stochastic
 * reflection's sampling from its Stochastic Screen-Space Reflections (ffx_sssr_intersect.h).
 * SPDX-FileCopyrightText: Copyright (C) 2024 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * The history's clip to a box (ClipAABB) as FidelityFX takes it from Playdead's INSIDE
 * (https://github.com/playdeadgames/temporal, TemporalReprojection.shader).
 * SPDX-FileCopyrightText: Copyright (c) 2015 Playdead
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, float, ivec2, vec2, vec3, vec4, mat4, uniform, texture, textureLoad, screenCoordinate, outputStruct, mrt, output,
	floor, fract, mix, min, max, abs, sqrt, exp, exp2, log, log2, trunc, clamp, step, pow, dot, length, normalize, select,
} from 'three/tsl';
import type { FrameGraph, ResourceHandle } from '../post/graph/FrameGraph.ts';
import { blueNoiseTexture, BLUE_NOISE } from '../post/BlueNoise.ts';
import { PRE_EXPOSURE_BIAS, preExposure } from '../post/PreExposure.ts';
import { TemporalRest, restEpoch } from '../post/TemporalRest.ts';
import type { DepthPyramid } from '../post/DepthPyramid.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { shared } from '../shared.ts';
import { FARTHEST, sampleBeckmannVisible, screenReflection, type ReflectionCamera } from './ScreenReflection.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;
type Columns = { element( i: number ): V4 };
// what the reflection keeps of the graph between frames: its resources and their histories
type Histories = Pick<FrameGraph<never>, 'color' | 'previous' | 'texture' | 'copyToPersistent' | 'resetHistory' | 'historyValid'>;

// The sea's reflection of what stands above it: one ray per pixel, denoised over space and time.
//
// AMD FidelityFX's Stochastic Screen-Space Reflections and its reflection denoiser, rewritten as
// passes of the frame graph between the opaque copy and the water (Pipeline.ts, Frame graph):
// - seaReflectionPrepass: the water's surface alone (Water.ts `prepassMesh`), single-sampled, into
//   `seaSurface`: its normal's x and z, the roughness of the slopes the pixel does not resolve
//   (Beckmann's alpha, -1 where the pixel traces nothing: a reflectance under the quality's
//   `ssrMinFresnel`, the reflection switched off) and the view distance (0: no sea, or an opaque
//   surface nearer). FidelityFX reads these from a G-buffer; the sea is drawn after the opaque
//   scene's copy it reflects, so it is drawn once more ahead of it. Copied at the frame's end into
//   `seaSurfacePrev`, the reprojection's history of the surface.
// - seaReflectionTrace: one reflected direction per pixel, a normal drawn from Beckmann's visible
//   normals (ScreenReflection.ts `sampleBeckmannVisible`) for two of the pixel's three blue-noise
//   numbers, and the free path it runs hidden among the plants for the third (`setPlants`),
//   traced over the depth pyramid (ScreenReflection.ts) into `seaRay`: the hit's colour times its
//   confidence, and the confidence (premultiplied: a miss is 0 and the water keeps its sky there),
//   and `seaRayLength`, the distance along the ray to where the trace ended. A ray drawn down
//   into the sea meets the next wave's face and leaves it upward again: it is traced mirrored
//   about the horizontal (its y made positive), so the water under a cliff keeps the cliff in
//   those directions instead of the sky's model, which would light it. FidelityFX's
//   Sobol-scrambled table (about 330 000 numbers) is replaced by the blue-noise tile read at three
//   offsets, moved each frame by the R3 sequence (Roberts, "The Unreasonable Effectiveness of
//   Quasirandom Sequences", 2018).
// - seaReflectionMoments: the 9 x 9 Gaussian moments (`exp( -3 i^2 / 25 )` per axis) FidelityFX
//   takes in shared memory, here separable: a pass sums the rows, and the passes that read the
//   moments sum the rows' sums down the column at their own pixels (`columns`), which saves a pass
//   whose two half-float targets were stored whole for the pixels of the sea alone to read.
// - seaReflectionReproject: the history reprojected, FidelityFX's `PickReprojection`: the hit's
//   (the reflected point seen through the surface, reprojected by the previous frame's view) where
//   the surface's normal and roughness match there, else the surface's own if its history lies
//   near the neighbourhood's mean, else none; then the disocclusion factor from the normals and
//   the depths, a 3 x 3 search and a 2 x 2 bilinear rescue at edges. Writes the reprojected
//   history, the variance, the samples (`seaSamples`, a history of its own) and the radiance the
//   average takes.
// - seaReflectionAverage: the radiance averaged over 8 x 8 tiles, weighted toward the dark
//   (`max( exp( -0.3 L ), 0.01 )`), which FidelityFX gathers in shared memory in the same pass;
//   here in two passes, the weighted sums of 2 x 2 pixels at half the size, then of 4 x 4 of those
//   per tile (a tile's 64 loads in one fragment left the pass waiting on its loads).
// - seaReflectionPrefilter: FidelityFX's 15 taps (Halton 2, 3 within 3 pixels), weighted by the
//   normals, the depths, the radiance against the tile's average and the variance.
// - seaReflectionMoments again, of the prefiltered radiance.
// - seaReflectionResolve: the temporal resolve, the reprojected history clipped to the
//   neighbourhood's box (`ClipAABB`) and blended by the samples, into `seaResolved` and its variance
//   (both histories), which the water reads (`result`).
// The constants are FidelityFX's (ffx_denoiser_reflections_config.h; the sample's temporal
// stability 0.7 and 32 samples at most), but for two, which its scenes in metres did not meet:
// - The prefilter's depth weight: FidelityFX's `exp( -|d - d'| d 4 )` falls to nothing between
//   neighbours on a sea seen far off at a grazing angle, whose depth steps by metres a pixel; here
//   the relative difference, `exp( -4 |d - d'| / d )`, which a plane keeps near 1 over the taps.
// - Every pixel of the sea that reflects is glossy: the sample's roughness threshold (0.2, a
//   speed-up for rough materials falling back to a probe) would leave most of a breezy sea to the
//   sky. The roughness compared and the samples' count take sqrt( alpha ).
// FidelityFX takes motion vectors for the surface's history; under MSAA there are none, and the
// surface's point is reprojected by the previous frame's view-projection (the hit's is FidelityFX's
// own). The signal is the opaque colour (pre-exposed, PreExposure.ts) times 2^PRE_EXPOSURE_BIAS,
// about the exposed image's scale, which FidelityFX's absolute constants (a luminance of 0.5 as the
// variance's floor, the average's weight) assume; it is clamped to SIGNAL_CEILING, whose square
// stays in half float. When the pre-exposure moves, the history's colour is rescaled by the ratio and
// its confidence (alpha) is kept: rescaled with the colour, a rise of two stops took the confidence
// to 4, the water's weight of its own sky, 1 - confidence, went negative, and the sea where the
// reflection is confident came out at half float's largest value for a few frames. The meter read
// that frame as a bright view and asked for the lower pre-exposure back, so on a night view the
// pre-exposure alternated and a paused page never settled.
//
// Rest (the test mode's paused frame is reproducible, however it was reached: within a tolerance where the
// reflection reaches, exactly elsewhere): the temporal rest (TemporalRest.ts) keyed by the camera,
// the clock, the sizes, the scene's epoch, the pre-exposure, the app's signature and the box the
// reflection traces in. At its first frame of the sequence the histories are cleared, and the
// passes run its `sequence` frames with the rest's own noise indices: what they hold after the last
// one is a function of the scene and of the opaque frames they read. It is then held: the passes
// stop and the water reads the last resolve. `pending` holds the page's settled signal until then.
// A cut (SceneHistory) and a scene jump clear the histories and the rest. Under MSAA a few opaque
// frames of a paused page can differ in their last bits from load to load (the browser's or the
// driver's doing), and the held result keeps what those frames held; so the pixels it reaches
// (`surfaceHistory`, debug/HeldRegion.ts) are compared within tools/kit/tolerance.py's `HELD`.

// the target the signal is stored at: the opaque colour, pre-exposed, times this (see the header)
const SIGNAL = 2 ** PRE_EXPOSURE_BIAS;
const SIGNAL_CEILING = 64;
// ffx_denoiser_reflections_config.h
const GAUSSIAN_K = 3;
const NEIGHBOURHOOD = 4;
const RADIANCE_WEIGHT_BIAS = 0.6;
const RADIANCE_WEIGHT_VARIANCE_K = 0.1;
const AVG_RADIANCE_LUMINANCE_WEIGHT = 0.3;
const PREFILTER_VARIANCE_WEIGHT = 4.4;
const PREFILTER_VARIANCE_BIAS = 0.1;
const PREFILTER_NORMAL_SIGMA = 512;
const PREFILTER_DEPTH_SIGMA = 4;
const REPROJECT_SURFACE_DISCARD_VARIANCE_WEIGHT = 1.5;
const DISOCCLUSION_NORMAL_WEIGHT = 1.4;
const DISOCCLUSION_DEPTH_WEIGHT = 1;
const DISOCCLUSION_THRESHOLD = 0.9;
const REPROJECTION_NORMAL_SIMILARITY_THRESHOLD = 0.9999;
// the sample's settings (samples/sssr)
const TEMPORAL_STABILITY = 0.7;
const MAX_SAMPLES = 32;
// FidelityFX's prefilter taps: the first 15 of Halton( 2, 3 ) stretched to [ -3, 3 ], the centre left out
const PREFILTER_TAPS = [ [ 0, 1 ], [ - 2, 1 ], [ 2, - 3 ], [ - 3, 0 ], [ 1, 2 ], [ - 1, - 2 ], [ 3, 0 ], [ - 3, 3 ], [ 0, - 3 ], [ - 1, - 1 ], [ 2, 1 ], [ - 2, - 2 ], [ 1, 0 ], [ 0, 2 ], [ 3, - 1 ] ] as const;
// the R3 sequence's steps (the inverse powers of the root of x^4 = x + 1)
const R3 = [ 0.8191725133961645, 0.6710436067037893, 0.5497004779019703 ] as const;

const RENDER = { of: 'render' } as const;
const TILES = { of: 'render', divisor: 8 } as const;

const luminance = ( c: V4 ) => max( dot( c.rgb, vec3( 0.299, 0.587, 0.114 ) ), 0.001 );
// FidelityFX's temporal variance of two radiances
const temporalVariance = ( a: V4, b: V4 ) => {

	const la = luminance( a ), lb = luminance( b );
	const d = abs( la.sub( lb ) ).div( max( max( la, lb ), 0.5 ) );
	return d.mul( d );

};
// ClipAABB: `p` pulled toward the box's centre onto its surface, if outside it
const clipToBox = ( lo: V4, hi: V4, p: V4 ) => {

	const centre = lo.add( hi ).mul( 0.5 ).toVar();
	const extent = hi.sub( lo ).mul( 0.5 ).add( 0.001 ).toVar();
	const v = p.sub( centre ).toVar();
	const u = abs( v.div( extent ) ).toVar();
	const most = max( max( u.x, u.y ), max( u.z, u.w ) ).toVar();
	return select( most.greaterThan( 1 ), centre.add( v.div( most ) ), p );

};
// FidelityFX's radiance weight
const radianceWeight = ( a: V4, b: V4, variance: F ) =>
	max( exp( float( RADIANCE_WEIGHT_BIAS ).add( variance.mul( RADIANCE_WEIGHT_VARIANCE_K ) ).negate().mul( length( a.rgb.sub( b.rgb ) ) ) ), 0.01 );
// the disocclusion factor of a history's normal and depth (view distances)
const disocclusion = ( n: V3, hn: V3, d: F, hd: F ) =>
	exp( abs( float( 1 ).sub( max( dot( n, hn ), 0 ) ) ).negate().mul( DISOCCLUSION_NORMAL_WEIGHT ) )
		.mul( exp( abs( hd.sub( d ) ).div( d ).negate().mul( DISOCCLUSION_DEPTH_WEIGHT ) ) );
// a surface texel's normal (its x and z stored; the sea's normals face up)
const normalOf = ( s: V4 ) => vec3( s.x, sqrt( max( float( 1 ).sub( s.x.mul( s.x ) ).sub( s.y.mul( s.y ) ), 0 ) ), s.y );
// `x` rounded to half float toward zero (normal and subnormal halves; no overflow)
const halfTowardZero = ( x: V4 ) => {

	const a = max( abs( x ), 1e-30 ).toVar();
	const e0 = floor( log2( a ) ).toVar();
	const p = exp2( e0 ).toVar();
	// (log2 may land one off next to a power of two)
	const e = clamp( e0.sub( float( 1 ).sub( step( p, a ) ) ).add( step( p.mul( 2 ), a ) ), - 14, 15 ).toVar();
	return trunc( x.mul( exp2( float( 10 ).sub( e ) ) ) ).mul( exp2( e.sub( 10 ) ) );

};
// a 9-tap weight of the moments' Gaussian
const momentWeight = ( i: number ) => Math.exp( - GAUSSIAN_K * i * i / ( ( NEIGHBOURHOOD + 1 ) ** 2 ) );

/** What the reflection takes from the pipeline. */
export interface SeaReflectionInputs {
	linearDepth: ResourceHandle; // the opaque pass's linear depth (sample 0's view z in x)
	opaqueColor: ResourceHandle; // its resolved colour, pre-exposed
	pyramid: DepthPyramid;
	iterations: number; // the trace's most steps (WaterQuality `ssrIterations`)
}

/** How the passes meet the pipeline's frame: whether it draws the sea, its camera, and the water's
 * per-frame setup (Water.ts `prepare`), run before the prepass draws. */
export interface SeaReflectionHooks<Frame> {
	active( f: Frame ): boolean;
	camera( f: Frame ): THREE.Camera;
	before( f: Frame ): void;
}

/** What the page tells it each frame (the haze's frame: VolumetricHaze.ts `HazeFrame`). */
export interface SeaReflectionFrame { time: number; signature: readonly number[]; stable: boolean }

export class SeaReflection {

	/** The water's prepass mesh goes here (Water.ts). */
	readonly prepassScene = new THREE.Scene();
	/** The reflection's box (world space), which the water sets each frame (Water.ts `prepare`). */
	readonly boxMin = uniform( new THREE.Vector3() );
	readonly boxMax = uniform( new THREE.Vector3() );
	/** The plants' volume (world space) and its mean free path (m, 0 with none): `setPlants`. */
	readonly plantsMin = uniform( new THREE.Vector3() );
	readonly plantsMax = uniform( new THREE.Vector3() );
	readonly freePath = uniform( 0 );
	/** The resolved reflection, premultiplied by its confidence, at the signal's scale: what the water
	 * reads at its pixel (`shown`, then `sceneUnits`). Bound to the history written last. */
	readonly result: THREE.TextureNode;
	/** The rest (see the header); `pending` while it has not converged. */
	readonly rest = new TemporalRest();
	/** Scale from the result to the scene's pre-exposed colour. */
	readonly shown = uniform( 1 / SIGNAL );
	/** 1 while the reflection runs, 0 with `flags.seaReflection = false` (the water then keeps its
	 * sky): what the water multiplies the result by. */
	readonly weight = uniform( 1 );

	private readonly graph: Histories;
	private readonly camera: { [ K in keyof ReflectionCamera ]: THREE.UniformNode<'mat4', THREE.Matrix4> };
	private readonly previousViewProjection = uniform( new THREE.Matrix4() );
	private readonly noise = uniform( new THREE.Vector3() );
	private readonly historyValid = uniform( 0 );
	// the history's colour rescaled to this frame's pre-exposure, its confidence (alpha) kept (the header)
	private readonly historyScale = uniform( new THREE.Vector4( 1, 1, 1, 1 ) );
	private readonly key: number[] = [];
	private readonly viewProjection = new THREE.Matrix4();
	private lastViewProjection: THREE.Matrix4 | null = null;
	private historyStops = 0;
	private frame = 0;
	private due = true;
	// the resources
	private readonly surface: ResourceHandle;
	private readonly surfacePrev: ResourceHandle;
	private readonly resolved: ResourceHandle;
	private readonly resolvedPrev: ResourceHandle;
	private readonly varianceHistory: ResourceHandle;
	private readonly varianceHistoryPrev: ResourceHandle;
	private readonly samples: ResourceHandle;
	private readonly samplesPrev: ResourceHandle;
	// texture nodes bound to a ping-pong's parity before each pass that reads them
	private readonly bound: { node: THREE.TextureNode; handle: ResourceHandle }[] = [];

	constructor( graph: Histories ) {

		this.graph = graph;
		const g = graph;
		this.camera = {
			view: uniform( new THREE.Matrix4() ), world: uniform( new THREE.Matrix4() ),
			projection: uniform( new THREE.Matrix4() ), projectionInverse: uniform( new THREE.Matrix4() ),
		};
		const half = ( channels: 'rgba' | 'r', filter: 'linear' | 'nearest' = 'nearest', size: typeof RENDER | typeof TILES = RENDER ) =>
			( { size, format: channels === 'rgba' ? THREE.RGBAFormat : THREE.RedFormat, type: THREE.HalfFloatType, filter } );
		this.surface = g.color( 'seaSurface', half( 'rgba', 'linear' ) );
		this.surfacePrev = g.color( 'seaSurfacePrev', half( 'rgba', 'linear' ), true );
		g.copyToPersistent( this.surface, this.surfacePrev );
		this.resolved = g.color( 'seaResolved', half( 'rgba', 'linear' ) );
		this.resolvedPrev = g.previous( this.resolved );
		this.varianceHistory = g.color( 'seaVarianceHistory', half( 'r', 'linear' ) );
		this.varianceHistoryPrev = g.previous( this.varianceHistory );
		this.samples = g.color( 'seaSamples', half( 'r', 'linear' ) );
		this.samplesPrev = g.previous( this.samples );
		this.result = texture( g.texture( this.resolved, 0 ) );
		this.rest.settle = 8;
		this.rest.sequence = 32;

	}

	/** Declares the passes, after the opaque copy and the depth pyramid and before the water;
	 * returns the resolved reflection's handle, which the water pass reads (`readOptional`). */
	passes<Frame>( graph: FrameGraph<Frame>, inputs: SeaReflectionInputs, jittered: boolean, hooks: SeaReflectionHooks<Frame> ): ResourceHandle {

		const g = graph;
		const renderer = g.renderer;
		const when = ( f: Frame ) => hooks.active( f ) && this.due && this.on;
		const size = g.extent( RENDER );
		const tiles = g.extent( TILES );
		const half = ( channels: 'rgba' | 'r', filter: 'linear' | 'nearest' = 'nearest', of: typeof RENDER | typeof TILES = RENDER ) =>
			( { size: of, format: channels === 'rgba' ? THREE.RGBAFormat : THREE.RedFormat, type: THREE.HalfFloatType, filter } );
		let surface = this.surface, resolved = this.resolved, varianceHistory = this.varianceHistory, samples = this.samples;
		let ray = g.color( 'seaRay', half( 'rgba' ) ), rayLength = g.color( 'seaRayLength', half( 'r' ) );
		let mean = g.color( 'seaMomentsMean', half( 'rgba' ) ), meanSq = g.color( 'seaMomentsSquare', half( 'rgba' ) );
		let reprojected = g.color( 'seaReprojected', half( 'rgba' ) ), variance = g.color( 'seaVariance', half( 'r' ) );
		let mixed = g.color( 'seaMixed', half( 'rgba' ) );
		let average = g.color( 'seaAverage', half( 'rgba', 'linear', TILES ) );
		let filtered = g.color( 'seaFiltered', half( 'rgba' ) ), filteredVariance = g.color( 'seaFilteredVariance', half( 'r' ) );
		let surfaceDepth = g.depth( 'seaSurfaceDepth', { size: RENDER } );
		const T = ( h: ResourceHandle ) => g.texture( h );
		// a texture node that follows a ping-pong resource's parity (bound in `bind`)
		const follow = ( h: ResourceHandle ) => {

			const node = texture( g.texture( h, 0 ) );
			this.bound.push( { node, handle: h } );
			return node;

		};
		const quad = ( name: string, fragment: V4 ) => {

			const material = new THREE.NodeMaterial();
			material.name = name;
			material.fragmentNode = fragment;
			const q = new THREE.QuadMesh( material );
			q.name = name;
			return q;

		};
		const run = ( q: THREE.QuadMesh ) => () => {

			this.bind();
			q.render( renderer );

		};
		const pixel = () => floor( screenCoordinate.xy );
		const inside = ( p: V2 ) => p.x.greaterThanEqual( 0 ).and( p.y.greaterThanEqual( 0 ) ).and( p.x.lessThan( size.x ) ).and( p.y.lessThan( size.y ) );
		// a texel, 0 off the render (FidelityFX's loads off the texture)
		const load = ( t: THREE.Texture | THREE.TextureNode, p: V2 ) => select( inside( p ), textureLoad( t as THREE.Texture, ivec2( p ) ) as unknown as V4, vec4( 0 ) );
		const sample = ( t: THREE.Texture | THREE.TextureNode, uv: V2 ) => texture( t as THREE.Texture, uv ).level( float( 0 ) ) as unknown as V4;
		const C = this.camera;
		// the view-space point of the sea at pixel centre `q` at view distance `d`
		const viewPoint = ( q: V2, d: F ) => {

			const r = C.projectionInverse.mul( vec4( q.x.div( size.x ).mul( 2 ).sub( 1 ), float( 1 ).sub( q.y.div( size.y ).mul( 2 ) ), 1, 1 ) ).xyz.toVar();
			return r.mul( d.negate().div( r.z ) );

		};
		// a world point on the previous frame's screen (uv, y down)
		const previousUV = ( w: V3 ) => {

			const h = this.previousViewProjection.mul( vec4( w, 1 ) ).toVar();
			const ndc = h.xy.div( h.w );
			return vec2( ndc.x.mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( ndc.y.mul( 0.5 ) ) );

		};
		const traced = ( s: V4 ) => s.w.greaterThan( 0 ).and( s.z.greaterThanEqual( 0 ) );

		// ---- the prepass: the sea's surface
		g.addPass( 'seaReflectionPrepass', 'scene', ( b ) => {

			b.read( inputs.linearDepth );
			surface = b.write( surface, { clear: true } );
			surfaceDepth = b.depth( surfaceDepth, { write: true, clear: true } );
			return {
				label: 'sea reflection', jittered, when, whenLabel: 'sea reflection due',
				mrt: mrt( { seaSurface: output } ).setClearColor( 'seaSurface', new THREE.Color( 0, 0, 0 ), 0 ),
				before: ( f ) => hooks.before( f ), scene: () => this.prepassScene, camera: ( f ) => hooks.camera( f ),
			};

		} );

		// ---- the trace
		const surfaceNow = texture( T( this.surface ) );
		const trace = screenReflection( {
			depth: texture( T( inputs.linearDepth ) ), color: texture( T( inputs.opaqueColor ) ), pyramid: inputs.pyramid, iterations: inputs.iterations,
			box: { min: this.boxMin, max: this.boxMax, cull: debugFlags.ssrCull !== false, clip: debugFlags.ssrClip !== false },
			plants: { min: this.plantsMin, max: this.plantsMax },
			waterLevel: shared.waterLevel, gale: shared.wind.gale, camera: C, size,
		} );
		const blue = blueNoiseTexture();
		const traceColumns = Fn( () => {

			const p = pixel().toVar();
			const s = textureLoad( surfaceNow, ivec2( p ) ).toVar() as unknown as V4;
			const colour = vec4( 0 ).toVar();
			const travelled = float( FARTHEST ).toVar();
			If( traced( s ), () => {

				const posV = viewPoint( p.add( 0.5 ), s.w ).toVar();
				// the pixel's three numbers: the tile at three offsets, moved by the frame's R3 step
				const tile = ( dx: number, dy: number ) => ( textureLoad( blue, ivec2( p.add( vec2( dx, dy ) ) ).mod( ivec2( BLUE_NOISE ) ) ) as unknown as V4 ).x;
				const xi = fract( vec3( tile( 0, 0 ), tile( 23, 41 ), tile( 47, 13 ) ).add( this.noise ) ).toVar();
				// how far the ray runs hidden in the plants before it meets one: a free path drawn
				// from the exponential distribution of mean `freePath`
				const reach = this.freePath.mul( log( float( 1 ).sub( min( xi.z, 0.9999 ) ) ).negate() ).toVar();
				const V = normalize( C.world.mul( vec4( posV.negate(), 0 ) ).xyz ).toVar();
				const N = normalOf( s ).toVar();
				const H = N.toVar();
				If( s.z.greaterThanEqual( 1e-4 ), () => {

					const u = xi.xy;
					// Duff et al., "Building an Orthonormal Basis, Revisited", JCGT 6:1, 2017 (z the normal)
					const n = vec3( N.x, N.z, N.y ).toVar();
					const sgn = select( n.z.greaterThanEqual( 0 ), float( 1 ), float( - 1 ) );
					const a = float( - 1 ).div( sgn.add( n.z ) ).toVar();
					const bb = n.x.mul( n.y ).mul( a ).toVar();
					const t1 = vec3( float( 1 ).add( sgn.mul( n.x ).mul( n.x ).mul( a ) ), sgn.mul( bb ), sgn.mul( n.x ).negate() ).toVar();
					const t2 = vec3( bb, sgn.add( n.y.mul( n.y ).mul( a ) ), n.y.negate() ).toVar();
					// (back from the basis's z-up order to the world's y-up)
					const T1 = vec3( t1.x, t1.z, t1.y ), T2 = vec3( t2.x, t2.z, t2.y );
					const wi = vec3( dot( V, T1 ), dot( V, T2 ), dot( V, N ) );
					const m = sampleBeckmannVisible( wi, s.z, u ).toVar();
					H.assign( normalize( T1.mul( m.x ).add( T2.mul( m.y ) ).add( N.mul( m.z ) ) ) );

				} );
				const Rs = H.mul( dot( V, H ).mul( 2 ) ).sub( V ).toVar();
				// (down into the sea: off the next wave and up again, the header)
				const R = vec3( Rs.x, abs( Rs.y ), Rs.z ).toVar();
				If( R.y.greaterThan( 1e-4 ), () => {

					const hit = trace( posV, normalize( C.view.mul( vec4( R, 0 ) ).xyz ), reach );
					const c = hit.colour.toVar();
					colour.assign( vec4( min( c.rgb.mul( SIGNAL ), vec3( SIGNAL_CEILING ) ).mul( c.a ), c.a ) );
					travelled.assign( hit.length );

				} );

			} );
			return mat4( colour, vec4( travelled, 0, 0, 0 ), vec4( 0 ), vec4( 0 ) );

		} )() as unknown as Columns;
		const traceQuad = quad( 'SeaReflection.Trace', outputStruct( traceColumns.element( 0 ), traceColumns.element( 1 ) ) as unknown as V4 );
		g.addPass( 'seaReflectionTrace', 'quad', ( b ) => {

			b.readOptional( surface );
			b.read( inputs.linearDepth );
			b.read( inputs.opaqueColor );
			ray = b.write( ray );
			rayLength = b.write( rayLength );
			return { label: 'sea reflection', jittered, when, whenLabel: 'sea reflection due', quad: traceQuad, mrt: mrt( {} ) };

		} );

		// ---- the moments of a radiance over its 9 x 9 neighbourhood: the rows' sums in a pass, the
		// columns' where they are read (`columns`)
		const rows = ( source: ResourceHandle, stage: string ): [ ResourceHandle, ResourceHandle ] => {

			const sums = Fn( () => {

				const p = pixel().toVar();
				const m1 = vec4( 0 ).toVar(), m2 = vec4( 0 ).toVar();
				let total = 0;
				for ( let i = - NEIGHBOURHOOD; i <= NEIGHBOURHOOD; i ++ ) {

					const w = momentWeight( i );
					total += w;
					const c = load( T( source ), p.add( vec2( i, 0 ) ) ).toVar();
					m1.addAssign( c.mul( w ) );
					m2.addAssign( c.mul( c ).mul( w ) );

				}
				return mat4( m1.div( total ), m2.div( total ), vec4( 0 ), vec4( 0 ) );

			} )() as unknown as Columns;
			const rowsQuad = quad( `SeaReflection.Moments${ stage }`, outputStruct( sums.element( 0 ), sums.element( 1 ) ) as unknown as V4 );
			g.addPass( `seaReflectionMoments${ stage }`, 'quad', ( b ) => {

				b.readOptional( source );
				mean = b.write( mean );
				meanSq = b.write( meanSq );
				return { label: 'sea reflection', jittered, when, whenLabel: 'sea reflection due', quad: rowsQuad, mrt: mrt( {} ) };

			} );
			return [ mean, meanSq ];

		};
		// the mean and the variance at `p`, the rows' sums summed down the column: each rounded to
		// half float toward zero, as the half-float targets of a pass of their own stored them on the
		// GPU measured (the reflection is then the one that pass gave, within the held tolerance), so a
		// pass's whole-target store is saved for the few loads of the pixels that read them
		const columns = ( rowMean: ResourceHandle, rowSquare: ResourceHandle, p: V2 ): [ V4, V4 ] => {

			const m1 = vec4( 0 ).toVar(), m2 = vec4( 0 ).toVar();
			let total = 0;
			for ( let i = - NEIGHBOURHOOD; i <= NEIGHBOURHOOD; i ++ ) {

				const w = momentWeight( i );
				total += w;
				const q = p.add( vec2( 0, i ) ).toVar();
				m1.addAssign( load( T( rowMean ), q ).mul( w ) );
				m2.addAssign( load( T( rowSquare ), q ).mul( w ) );

			}
			const m = m1.div( total ).toVar(), sq = m2.div( total ).toVar();
			return [ halfTowardZero( m ), halfTowardZero( abs( sq.sub( m.mul( m ) ) ) ) ];

		};
		const [ rayRows, raySquares ] = rows( ray, 'Ray' );

		// ---- the reprojection
		const surfaceBefore = texture( T( this.surfacePrev ) );
		const radianceBefore = follow( this.resolvedPrev );
		const varianceBefore = follow( this.varianceHistoryPrev );
		const samplesBefore = follow( this.samplesPrev );
		const reprojectColumns = Fn( () => {

			const p = pixel().toVar();
			const s = load( surfaceNow, p ).toVar();
			const radiance = load( T( ray ), p ).toVar();
			const outReprojected = vec4( 0 ).toVar();
			const outVariance = float( 1 ).toVar();
			const outSamples = float( 1 ).toVar();
			If( traced( s ), () => {

				const roughness = sqrt( s.z ).toVar();
				const N = normalOf( s ).toVar();
				const d = s.w.toVar();
				const posV = viewPoint( p.add( 0.5 ), d ).toVar();
				const posW = C.world.mul( vec4( posV, 1 ) ).xyz.toVar();
				const rayLen = load( T( rayLength ), p ).x;
				const hitW = C.world.mul( vec4( normalize( posV ).mul( length( posV ).add( rayLen ) ), 1 ) ).xyz;
				const surfaceUV = previousUV( posW ).toVar(), hitUV = previousUV( hitW ).toVar();
				const [ nm, nv ] = columns( rayRows, raySquares, p );
				const neighbourMean = nm.toVar(), neighbourVariance = nv.toVar();
				const surfaceS = sample( surfaceBefore, surfaceUV ).toVar(), hitS = sample( surfaceBefore, hitUV ).toVar();
				const surfaceHistory = sample( radianceBefore, surfaceUV ).mul( this.historyScale ).toVar();
				const hitHistory = sample( radianceBefore, hitUV ).mul( this.historyScale ).toVar();
				const hitN = normalize( normalOf( hitS ) ).toVar(), surfaceN = normalize( normalOf( surfaceS ) ).toVar();
				const hitSimilarity = dot( hitN, N ).toVar(), surfaceSimilarity = dot( surfaceN, N ).toVar();
				const hitRoughness = sqrt( max( hitS.z, 0 ) ), surfaceRoughness = sqrt( max( surfaceS.z, 0 ) );
				const factor = float( 0 ).toVar();
				const historyUV = vec2( - 1 ).toVar();
				const reprojection = vec4( 0 ).toVar();
				const historyN = vec3( 0, 1, 0 ).toVar();
				const historyD = float( 0 ).toVar();
				const picked = float( 1 ).toVar();
				If( hitSimilarity.greaterThan( REPROJECTION_NORMAL_SIMILARITY_THRESHOLD ).and( hitSimilarity.add( 1e-3 ).greaterThan( surfaceSimilarity ) )
					.and( abs( hitRoughness.sub( roughness ) ).lessThan( abs( surfaceRoughness.sub( roughness ) ).add( 1e-3 ) ) ), () => {

					historyN.assign( hitN );
					historyD.assign( hitS.w );
					historyUV.assign( hitUV );
					reprojection.assign( hitHistory );

				} ).ElseIf( dot( surfaceHistory.sub( neighbourMean ), surfaceHistory.sub( neighbourMean ) ).lessThan( length( neighbourVariance ).mul( REPROJECT_SURFACE_DISCARD_VARIANCE_WEIGHT ) ), () => {

					historyN.assign( surfaceN );
					historyD.assign( surfaceS.w );
					historyUV.assign( surfaceUV );
					reprojection.assign( surfaceHistory );

				} ).Else( () => {

					picked.assign( 0 );
					// (FidelityFX returns here with a factor of 0, and the uv it was given)
					historyUV.assign( surfaceUV );

				} );
				If( picked.greaterThan( 0.5 ), () => {

					factor.assign( disocclusion( N, historyN, d, historyD ) );
					If( factor.lessThan( DISOCCLUSION_THRESHOLD ), () => {

						// the closest history in the 3 x 3 around
						const centre = historyUV.toVar();
						for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

							const q = centre.add( vec2( x, y ).div( size ) ).toVar();
							const h = sample( surfaceBefore, q ).toVar();
							const w = disocclusion( N, normalOf( h ), d, h.w ).toVar();
							If( w.greaterThan( factor ), () => {

								factor.assign( w );
								historyUV.assign( q );

							} );

						}
						reprojection.assign( sample( radianceBefore, historyUV ).mul( this.historyScale ) );

					} );
					If( factor.lessThan( DISOCCLUSION_THRESHOLD ), () => {

						// the rare slow path, at edges: the 2 x 2 bilinear taps kept by their own factors
						const f = fract( historyUV.mul( size ).add( 0.5 ) ).toVar();
						const base = floor( historyUV.mul( size ).sub( 0.5 ) ).toVar();
						const r = vec4( 0 ).toVar(), dd = float( 0 ).toVar(), nn = vec3( 0 ).toVar(), ws = float( 0 ).toVar();
						for ( const [ ox, oy ] of [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ] ) {

							const q = base.add( vec2( ox, oy ) ).toVar();
							const h = load( surfaceBefore, q ).toVar();
							const hn = normalOf( h ).toVar();
							const keep = select( disocclusion( N, hn, d, h.w ).greaterThan( DISOCCLUSION_THRESHOLD / 2 ), float( 1 ), float( 0 ) );
							const w = keep.mul( ox ? f.x : float( 1 ).sub( f.x ) ).mul( oy ? f.y : float( 1 ).sub( f.y ) ).toVar();
							r.addAssign( load( radianceBefore, q ).mul( this.historyScale ).mul( w ) );
							dd.addAssign( h.w.mul( w ) );
							nn.addAssign( hn.mul( w ) );
							ws.addAssign( w );

						}
						const norm = max( ws, 1e-3 ).toVar();
						reprojection.assign( r.div( norm ) );
						factor.assign( disocclusion( N, nn.div( norm ), d, dd.div( norm ) ) );

					} );
					factor.assign( select( factor.lessThan( DISOCCLUSION_THRESHOLD ), float( 0 ), factor ) );

				} );
				const onScreen = historyUV.x.greaterThan( 0 ).and( historyUV.y.greaterThan( 0 ) ).and( historyUV.x.lessThan( 1 ) ).and( historyUV.y.lessThan( 1 ) );
				If( onScreen.and( this.historyValid.greaterThan( 0.5 ) ), () => {

					const previousVariance = sample( varianceBefore, historyUV ).x;
					const most = max( 8, float( MAX_SAMPLES ).mul( float( 1 ).sub( exp( roughness.mul( - 100 ) ) ) ) );
					const count = min( most, sample( samplesBefore, historyUV ).x.mul( factor ).add( 1 ) ).toVar();
					If( factor.greaterThanEqual( DISOCCLUSION_THRESHOLD ), () => {

						outReprojected.assign( reprojection );
						outVariance.assign( mix( temporalVariance( radiance, reprojection ), previousVariance, float( 1 ).div( count ) ) );
						outSamples.assign( count );
						// (the average takes the history mixed in)
						radiance.assign( mix( radiance, reprojection, 0.3 ) );

					} );

				} );

			} );
			return mat4( outReprojected, vec4( outVariance, 0, 0, 0 ), vec4( outSamples, 0, 0, 0 ), radiance );

		} )() as unknown as Columns;
		const reprojectQuad = quad( 'SeaReflection.Reproject', outputStruct( reprojectColumns.element( 0 ), reprojectColumns.element( 1 ), reprojectColumns.element( 2 ), reprojectColumns.element( 3 ) ) as unknown as V4 );
		g.addPass( 'seaReflectionReproject', 'quad', ( b ) => {

			b.readOptional( surface );
			b.readOptional( ray );
			b.readOptional( rayLength );
			b.readOptional( rayRows );
			b.readOptional( raySquares );
			b.read( this.surfacePrev );
			b.read( this.resolvedPrev );
			b.read( this.varianceHistoryPrev );
			b.read( this.samplesPrev );
			reprojected = b.write( reprojected );
			variance = b.write( variance );
			samples = b.write( samples );
			mixed = b.write( mixed );
			return {
				label: 'sea reflection', jittered, when, whenLabel: 'sea reflection due', quad: reprojectQuad, mrt: mrt( {} ),
				run: () => {

					this.historyValid.value = g.historyValid( this.resolved ) ? 1 : 0;
					run( reprojectQuad )();

				},
			};

		} );

		// ---- the 8 x 8 average, weighted toward the dark, in two stages: the weighted sums of 2 x 2
		// pixels at half the size (in float), then of 4 x 4 of those
		const HALF = { of: 'render', divisor: 2 } as const;
		const halfSize = g.extent( HALF );
		let sums = g.color( 'seaAverageSum', { size: HALF, format: THREE.RGBAFormat, type: THREE.FloatType, filter: 'nearest' } );
		let weights = g.color( 'seaAverageWeight', { size: HALF, format: THREE.RedFormat, type: THREE.FloatType, filter: 'nearest' } );
		const sumColumns = Fn( () => {

			const base = pixel().mul( 2 ).toVar();
			const sum = vec4( 0 ).toVar(), weight = float( 0 ).toVar();
			for ( let i = 0; i < 4; i ++ ) {

				const q = base.add( vec2( i % 2, Math.floor( i / 2 ) ) ).toVar();
				const c = load( T( mixed ), q ).toVar();
				const w = max( exp( luminance( c ).mul( - AVG_RADIANCE_LUMINANCE_WEIGHT ) ), 0.01 ).mul( select( inside( q ), float( 1 ), float( 0 ) ) ).toVar();
				sum.addAssign( c.mul( w ) );
				weight.addAssign( w );

			}
			return mat4( sum, vec4( weight, 0, 0, 0 ), vec4( 0 ), vec4( 0 ) );

		} )() as unknown as Columns;
		const sumQuad = quad( 'SeaReflection.AverageSum', outputStruct( sumColumns.element( 0 ), sumColumns.element( 1 ) ) as unknown as V4 );
		g.addPass( 'seaReflectionAverageSum', 'quad', ( b ) => {

			b.readOptional( mixed );
			sums = b.write( sums );
			weights = b.write( weights );
			return { label: 'sea reflection', jittered, when, whenLabel: 'sea reflection due', quad: sumQuad, mrt: mrt( {} ) };

		} );
		const averageQuad = quad( 'SeaReflection.Average', Fn( () => {

			const base = pixel().mul( 4 ).toVar();
			const sum = vec4( 0 ).toVar(), weight = float( 0 ).toVar();
			for ( let i = 0; i < 16; i ++ ) {

				const q = base.add( vec2( i % 4, Math.floor( i / 4 ) ) ).toVar();
				const keep = q.x.lessThan( halfSize.x ).and( q.y.lessThan( halfSize.y ) );
				sum.addAssign( select( keep, textureLoad( T( sums ), ivec2( q ) ) as unknown as V4, vec4( 0 ) ) );
				weight.addAssign( select( keep, ( textureLoad( T( weights ), ivec2( q ) ) as unknown as V4 ).x, float( 0 ) ) );

			}
			return sum.div( max( weight, 1e-3 ) );

		} )() as unknown as V4 );
		g.addPass( 'seaReflectionAverage', 'quad', ( b ) => {

			b.readOptional( sums );
			b.readOptional( weights );
			average = b.write( average );
			return { label: 'sea reflection', jittered, when, whenLabel: 'sea reflection due', quad: averageQuad };

		} );
		const averageAt = ( p: V2 ) => sample( T( average ), p.add( 0.5 ).div( tiles.mul( 8 ) ) );

		// ---- the prefilter
		const prefilterColumns = Fn( () => {

			const p = pixel().toVar();
			const s = load( surfaceNow, p ).toVar();
			const centre = load( T( ray ), p ).toVar();
			const centreVariance = load( T( variance ), p ).x.toVar();
			const outRadiance = centre.toVar(), outVariance = centreVariance.toVar();
			If( centreVariance.greaterThan( 0 ).and( traced( s ) ).and( s.z.greaterThanEqual( 1e-4 ) ), () => {

				const avg = averageAt( p ).toVar();
				const N = normalOf( s ).toVar();
				const d = s.w.toVar();
				const w0 = radianceWeight( avg, centre, centreVariance ).toVar();
				const W = w0.toVar();
				const radiance = centre.mul( w0 ).toVar();
				const v = centreVariance.mul( w0 ).mul( w0 ).toVar();
				const varianceWeight = max( PREFILTER_VARIANCE_BIAS, float( 1 ).sub( exp( centreVariance.mul( - PREFILTER_VARIANCE_WEIGHT ) ) ) ).toVar();
				for ( const [ ox, oy ] of PREFILTER_TAPS ) {

					const q = p.add( vec2( ox, oy ) ).toVar();
					const ns = load( surfaceNow, q ).toVar();
					const n = load( T( ray ), q ).toVar();
					const nv = load( T( variance ), q ).x;
					const w = select( ns.w.greaterThan( 0 ), float( 1 ), float( 0 ) )
						.mul( pow( max( dot( N, normalOf( ns ) ), 0 ), PREFILTER_NORMAL_SIGMA ) )
						.mul( exp( abs( d.sub( ns.w ) ).div( d ).mul( - PREFILTER_DEPTH_SIGMA ) ) )
						.mul( radianceWeight( avg, n, centreVariance ) )
						.mul( varianceWeight ).toVar();
					W.addAssign( w );
					radiance.addAssign( n.mul( w ) );
					v.addAssign( nv.mul( w ).mul( w ) );

				}
				outRadiance.assign( radiance.div( W ) );
				outVariance.assign( v.div( W.mul( W ) ) );

			} );
			return mat4( outRadiance, vec4( outVariance, 0, 0, 0 ), vec4( 0 ), vec4( 0 ) );

		} )() as unknown as Columns;
		const prefilterQuad = quad( 'SeaReflection.Prefilter', outputStruct( prefilterColumns.element( 0 ), prefilterColumns.element( 1 ) ) as unknown as V4 );
		g.addPass( 'seaReflectionPrefilter', 'quad', ( b ) => {

			b.readOptional( surface );
			b.readOptional( ray );
			b.readOptional( variance );
			b.readOptional( average );
			filtered = b.write( filtered );
			filteredVariance = b.write( filteredVariance );
			return { label: 'sea reflection', jittered, when, whenLabel: 'sea reflection due', quad: prefilterQuad, mrt: mrt( {} ) };

		} );
		const [ filteredRows, filteredSquares ] = rows( filtered, 'Filtered' );

		// ---- the temporal resolve
		const samplesNow = follow( this.samples );
		const resolveColumns = Fn( () => {

			const p = pixel().toVar();
			const s = load( surfaceNow, p ).toVar();
			const signal = load( T( filtered ), p ).toVar();
			const signalVariance = load( T( filteredVariance ), p ).x.toVar();
			const outRadiance = signal.toVar(), outVariance = signalVariance.toVar();
			If( signal.r.add( signal.g ).add( signal.b ).add( signal.a ).lessThan( 1e-4 ), () => {

				outRadiance.assign( vec4( 0 ) );
				outVariance.assign( 0 );

			} ).ElseIf( traced( s ), () => {

				const count = load( samplesNow, p ).x.toVar();
				const avg = averageAt( p ).toVar();
				const old = load( T( reprojected ), p ).toVar();
				const [ fm, fs ] = columns( filteredRows, filteredSquares, p );
				const m = fm.toVar();
				const spread = fs.toVar();
				const deviation = sqrt( spread ).add( length( m.sub( avg ) ) ).mul( TEMPORAL_STABILITY * 1.4 ).toVar();
				const centre = mix( m, avg, 0.2 ).toVar();
				const oldClipped = clipToBox( centre.sub( deviation ), centre.add( deviation ), old ).toVar();
				const weight = float( 1 ).sub( float( 1 ).div( max( count, 1 ) ) ).toVar();
				const next = mix( signal, avg, float( 1 ).div( max( count.add( 1 ), 1 ) ) ).toVar();
				next.assign( clipToBox( avg.sub( deviation ), avg.add( deviation ), next ) );
				next.assign( mix( next, oldClipped, weight ) );
				const nextVariance = mix( temporalVariance( next, oldClipped ), signalVariance, weight ).toVar();
				// (NaN and infinity to 0, as FidelityFX does)
				const finite = ( x: F ) => x.equal( x ).and( abs( x ).lessThan( 65504 ) );
				const ok = finite( next.r ).and( finite( next.g ) ).and( finite( next.b ) ).and( finite( next.a ) ).and( finite( nextVariance ) );
				outRadiance.assign( select( ok, next, vec4( 0 ) ) );
				outVariance.assign( select( ok, nextVariance, float( 0 ) ) );

			} );
			return mat4( outRadiance, vec4( outVariance, 0, 0, 0 ), vec4( 0 ), vec4( 0 ) );

		} )() as unknown as Columns;
		const resolveQuad = quad( 'SeaReflection.Resolve', outputStruct( resolveColumns.element( 0 ), resolveColumns.element( 1 ) ) as unknown as V4 );
		g.addPass( 'seaReflectionResolve', 'quad', ( b ) => {

			b.readOptional( surface );
			b.readOptional( filtered );
			b.readOptional( filteredVariance );
			b.readOptional( filteredRows );
			b.readOptional( filteredSquares );
			b.readOptional( samples );
			b.readOptional( average );
			b.readOptional( reprojected );
			resolved = b.write( resolved );
			varianceHistory = b.write( varianceHistory );
			return {
				label: 'sea reflection', jittered, when, whenLabel: 'sea reflection due', quad: resolveQuad, mrt: mrt( {} ),
				run: () => {

					run( resolveQuad )();
					this.historyStops = preExposure.stops;

				},
			};

		} );
		return resolved;

	}

	/** Binds the texture nodes that follow a ping-pong's parity, and the result to the history
	 * written last; before each pass that reads them and before the water's. */
	bind(): void {

		const g = this.graph;
		for ( const { node, handle } of this.bound ) node.value = g.texture( handle );
		this.result.value = g.texture( this.resolved );
		this.shown.value = 2 ** - PRE_EXPOSURE_BIAS * 2 ** ( preExposure.stops - this.historyStops );
		this.weight.value = this.on ? 1 : 0;

	}

	/** The reflection runs (off with `flags.seaReflection = false`, read every frame, for A/B timing). */
	get on(): boolean {

		return debugFlags.seaReflection !== false;

	}

	/** Once per frame, before the graph runs: the camera (its view, and its projection without the
	 * temporal modes' jitter) and the app's frame. Sets the rest and the frame's noise. */
	update( camera: THREE.Camera, projection: THREE.Matrix4, width: number, height: number, frame: SeaReflectionFrame ): void {

		const C = this.camera;
		C.view.value.copy( camera.matrixWorldInverse );
		C.world.value.copy( camera.matrixWorld );
		this.setProjection( projection );
		this.viewProjection.multiplyMatrices( projection, camera.matrixWorldInverse );
		this.previousViewProjection.value.copy( this.lastViewProjection ?? this.viewProjection );
		this.lastViewProjection = ( this.lastViewProjection ?? new THREE.Matrix4() ).copy( this.viewProjection );
		const key = this.key;
		key.length = 0;
		key.push( ...camera.matrixWorld.elements, ...projection.elements, frame.time, width, height, restEpoch.scene, preExposure.stops,
			...frame.signature, ...this.boxMin.value.toArray(), ...this.boxMax.value.toArray(),
			...this.plantsMin.value.toArray(), ...this.plantsMax.value.toArray(), this.freePath.value );
		this.rest.observe( key, frame.stable );
		const index = this.rest.index;
		this.due = ! this.rest.converged || index !== null;
		if ( index === 0 ) this.clearHistory();
		const k = index ?? ( this.frame = ( this.frame + 1 ) % 1024 );
		this.noise.value.set( ( k * R3[ 0 ] ) % 1, ( k * R3[ 1 ] ) % 1, ( k * R3[ 2 ] ) % 1 );
		const ratio = 2 ** ( preExposure.stops - this.historyStops );
		this.historyScale.value.set( ratio, ratio, ratio, 1 );

	}

	/** The projection the sea was drawn with: the temporal modes' jittered one inside their
	 * jittered span (Pipeline.ts `jitterBegin`). */
	setProjection( projection: THREE.Matrix4 ): void {

		this.camera.projection.value.copy( projection );
		this.camera.projectionInverse.value.copy( projection ).invert();

	}

	/** The surface the reflection last traced from (the prepass's `seaSurface`, copied at the end of
	 * each frame that drew it): a texel traces where its view distance (w) is over 0 and its roughness
	 * (z) is not negative, and only there is the result not 0 (debug/HeldRegion.ts reads it). */
	get surfaceHistory(): ResourceHandle {

		return this.surfacePrev;

	}

	/** The rest has not converged (the page's settled signal waits for it). */
	get pending(): boolean {

		return ! this.rest.converged;

	}

	/** The plants' volume (world space), which a ray hidden behind a surface inside it crosses as a
	 * medium of mean free path `freePath` (m: ScreenReflection.ts, The plants), or none (an empty box). */
	setPlants( box: THREE.Box3, freePath: number ): void {

		const none = box.isEmpty();
		this.plantsMin.value.copy( none ? new THREE.Vector3() : box.min );
		this.plantsMax.value.copy( none ? new THREE.Vector3() : box.max );
		this.freePath.value = none ? 0 : freePath;

	}

	/** Drops the histories (a cut of the temporal modes' history; the rest's first frame). */
	clearHistory(): void {

		const g = this.graph;
		g.resetHistory( this.resolved );
		g.resetHistory( this.varianceHistory );
		g.resetHistory( this.samples );
		this.lastViewProjection = null;

	}

	/** Drops the histories and the rest (a scene jump: `establishScene`). */
	invalidate(): void {

		this.clearHistory();
		this.rest.restart();

	}

}
