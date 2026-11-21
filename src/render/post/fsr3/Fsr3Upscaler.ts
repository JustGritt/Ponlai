import * as THREE from 'three/webgpu';
import { texture, textureLoad, uniform, float, int, ivec2, max, select } from 'three/tsl';
import { debugFlags } from '../../debug/DebugFlags.ts';
import type { TemporalSettings } from '../../Quality.ts';
import { jitter } from '../Jitter.ts';
import { motionState, onTaaCut } from '../Motion.ts';
import { TemporalRest, restEpoch, restKey } from '../TemporalRest.ts';
import { sceneEvents } from '../SceneEvents.ts';
import type { TemporalFrame } from '../TemporalAA.ts';
import type { FrameGraph, ResourceHandle, SizeClass } from '../graph/FrameGraph.ts';
import type { PassBuilder } from '../graph/GraphCompiler.ts';
import { Fsr3Bindings, type Fsr3Inputs } from './Bindings.ts';
import { createPassKernels, type Fsr3PassKernels } from './Passes.ts';
import * as H from './Host.ts';
import * as R from './Resources.ts';
import { fsr3MipBiasScale } from './MipBias.ts';
import { preExposure } from '../PreExposure.ts';
import type { N } from './Core.ts';

// The app's FSR 3.1 upscaler, on the TSL port of AMD's FidelityFX FSR 3.1.5 in this folder.
//
// The app's FSR 3.1 upscaler (Antialiasing.ts: aa=fsr3), on the TSL port of AMD's FidelityFX FSR
// 3.1.5 in this folder (Host.ts: the host's maths and frame plan; Bindings.ts: the resources and
// constants; Passes.ts: the kernels). It stands where AMD's integration guide puts it: after the
// scene and its composite of haze and shafts (`hdr`, scene radiance at the render's size, FSR's
// HDR input), before the blooms, the depth of field, the look's glow and the grade, which run on its
// output at the canvas's size (Pipeline.ts).
//
// Inputs, as FSR takes them:
// - colour: `hdr`, with the HDR flag; exposure: the factor the display chain applies to it,
//   2^( EV - P ), EV loaded from the exposure state's texel the meter writes (Pipeline's
//   `exposureState`, declared as a read of the passes; so FSR's luma heuristics see what the grade
//   sees) over the pre-exposure 2^P hdr carries (PreExposure.ts), which is FSR's pre-exposure: FSR
//   3.1 multiplies its input by the exposure and its history by the ratio of this frame's
//   pre-exposure to the last (`deltaPreExposure`), so the history follows a new P exactly. The
//   reactive generator compares the scene's colour with the opaque image in scene units (both
//   divided by 2^P), as it did before the pre-exposure. While the rest's sequence runs after a new
//   P, the held copy (stored at the old P) is not shown: the sequence's own frames are;
// - depth: the nearest surface's view z (`surfaceDepth`, the sea included) as reversed infinite
//   device depth `near / -z` (0 for the sky, z <= -9e4), flags DEPTH_INVERTED | DEPTH_INFINITE, the
//   camera's near and far and its live vertical field of view (from the projection);
// - motion: the velocity's rg (NDC now less NDC previous, unjittered), so `motionVectorScale` is
//   ( -0.5 renderWidth, 0.5 renderHeight ) in upstream's pixel units (FSR's previous uv = uv +
//   motion; screen uv's y runs down) and there is no jitter cancellation;
// - reactive: the larger of the velocity's alpha (glints, the overlay's rain) and the rain veil's
//   weight (`veilReactive`), and, with the generator, its mask; no transparency mask;
// - jitter: Jitter.ts runs FSR's sequence under fsr3, FSR's `jitterOffset` = -px;
// - frameTimeDelta (ms): the scene clock's step while it runs, the pacer's nominal interval while
//   paused (it only drives the luma adaptation of the frame info), never the wall clock;
// - viewSpaceToMetersFactor 1; sharpening on (RCAS, `fsr3.sharpness` of the tier's settings).
//
// FSR's internal resources (accumulation, luma history, locks, the SPD mips, the dilated buffers,
// the reconstructed depth buffer, the frame info) are owned here as upstream's context owns them,
// not declared on the frame graph: P1's bindings swap the ping-pong sides by FSR's own resource
// frame index, which stops while the rest holds, nothing else reads them, and the attachment ops
// never touch storage. Render-max = upscale-max = the canvas's size, so a render-scale step changes
// only the live sizes in the constants (no texture, no pipeline); a canvas resize resizes every
// texture and buffer (a `graph.onResize` hook) and starts a new context (a reset).
//
// The graph sees the boundary: one compute pass per kernel in upstream's order (the clears, prepare
// inputs, the luma pyramid and its tail, the shading-change pyramid, shading change, prepare
// reactivity, luma instability, accumulate with sharpening, RCAS, and with `?fsrDebug` the debug
// view), each declaring the graph resources its kernel binds as reads, `neverCull` where it writes
// nothing the graph knows; RCAS writes FSR's output (an imported canvas-size rgba16float texture)
// through `storage`. Each pass dispatches the plan's job for its kernel with upstream's live
// workgroup counts (`renderer.compute( kernel, [ x, y, 1 ] )`) and lists every kernel it may
// dispatch for the warm-up. The shading-change pyramid's tail is not dispatched: it stores only
// levels above 5, which nothing reads (Passes.ts). Every pass runs only while `dispatching`.
//
// Reset (FSR's `reset`): a cut (`onTaaCut`: every scene event that cuts, the camera's jumps), a
// canvas resize, and the rest's start; never a render-scale step, which FSR follows itself from
// its previous frame's render size.
//
// The rest (TemporalRest.ts), as the temporal modes have it, keyed the same way (`restKey` with the
// settings' version and the page's signature): after `restSettle` frames at rest, FSR starts from a
// new context (its first execution's clears and a reset, so the rest begins from the same state
// whatever ran before) and runs `fsr3.restSequence` frames from jitter index 0 (FSR's phase
// sequence repeated: Jitter.ts), while the post chain shows the last running frame's output, copied
// into a held texture by the clears pass at the rest's first frame. From the sequence's last frame
// on, FSR's output is the rest image: FSR does not dispatch while the rest holds, and the jitter
// stays at the sequence's last index, so the post chain after FSR, whose depth reads take this
// frame's jitter, draws the same frame (`jitter.restIndex` = the sequence's length). Leaving the
// rest continues from FSR's state.
//
// The reactive-mask generator (`fsr3.autoReactive`, `debugFlags.fsrAutoReactive`; a load-time setting: its
// kernel is built only then) runs before the upscaler on the scene colour (before the composite)
// against the opaque copy, its mask taken with the reactive input by the larger value. The mip bias
// (MipBias.ts) follows the live render and canvas widths every frame.

const CANVAS: SizeClass = { of: 'canvas' };
const RENDER: SizeClass = { of: 'render' };
const HALF: SizeClass = { of: 'render', scale: 0.5, min: 1 };

/** The graph resources FSR reads (Pipeline.ts). */
export interface Fsr3GraphInputs {
	hdr: ResourceHandle; surfaceDepth: ResourceHandle; velocity: ResourceHandle; veilReactive: ResourceHandle | null;
	// the scene colour before the composite and the opaque copy (the generator's)
	sceneColor: ResourceHandle; opaqueColor: ResourceHandle;
	// the exposure state the exposure node loads (Pipeline's, Metering.ts), read by every kernel pass
	exposure?: ResourceHandle;
}

export interface Fsr3Options {
	settings: TemporalSettings;
	// the display chain's exposure of `hdr` (2^EV from Pipeline's exposure state): builds a new
	// texture read at each call, as every accessor must (Bindings.ts)
	exposure: () => THREE.Node<'float'>;
	// FSR's debug view (DRAW_DEBUG_VIEW over its output: `?fsrDebug`, debug builds)
	debugView: boolean;
}

type Size = { width: number; height: number };

// the three type of a port format
function formatOf( f: R.Fsr3Format ): { format: THREE.AnyPixelFormat; type: THREE.TextureDataType } {

	switch ( f ) {

		case 'r8unorm': return { format: THREE.RedFormat, type: THREE.UnsignedByteType };
		case 'r16float': return { format: THREE.RedFormat, type: THREE.HalfFloatType };
		case 'rg16float': return { format: THREE.RGFormat, type: THREE.HalfFloatType };
		case 'rgba8unorm': return { format: THREE.RGBAFormat, type: THREE.UnsignedByteType };
		case 'rgba16float': return { format: THREE.RGBAFormat, type: THREE.HalfFloatType };
		case 'r32float': return { format: THREE.RedFormat, type: THREE.FloatType };
		case 'r32uint': return { format: THREE.RedIntegerFormat, type: THREE.UnsignedIntType };
		default: throw new Error( `FSR3: ${ f } is a buffer` );

	}

}

function storageTexture( name: string, f: R.Fsr3Format, linear: boolean, mips = 1 ): THREE.StorageTexture {

	const t = new THREE.StorageTexture( 1, 1 );
	const { format, type } = formatOf( f );
	t.format = format;
	t.type = type;
	t.name = name;
	t.generateMipmaps = false;
	( t as unknown as { mipmapsAutoUpdate: boolean } ).mipmapsAutoUpdate = false;
	t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
	t.magFilter = linear ? THREE.LinearFilter : THREE.NearestFilter;
	t.minFilter = linear ? ( mips > 1 ? THREE.LinearMipmapNearestFilter : THREE.LinearFilter ) : THREE.NearestFilter;
	// (three allocates as many levels as `mipmaps` lists and generates none for a storage texture)
	if ( mips > 1 ) t.mipmaps = Array.from( { length: mips }, () => ( {} ) ) as never;
	return t;

}

const groups = ( n: number, d: number ) => Math.trunc( ( n + d - 1 ) / d );
// the vertical field of view of a projection, in radians
const fovOf = ( p: THREE.Matrix4 ) => 2 * Math.atan( 1 / p.elements[ 5 ] );

export class Fsr3Upscaler {

	/** The app's upscaler, once built. */
	static current: Fsr3Upscaler | null = null;
	readonly settings: TemporalSettings;
	/** The rest convergence. */
	readonly rest = new TemporalRest();
	/** FSR's output (RCAS, at the canvas's size, scene radiance) and the rest's held copy of the last
	 * running frame's. */
	readonly output: THREE.StorageTexture;
	readonly held: THREE.StorageTexture;
	/** Whether FSR dispatches this frame (not while the rest holds its image). */
	dispatching = true;
	/** Resets FSR asked for since the page loaded (cuts, resizes, rests), and the last one's reason. */
	resets = 0;
	lastReset: string | null = null;
	private readonly renderer: THREE.WebGPURenderer;
	private readonly options: Fsr3Options;
	private readonly flags = H.FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE | H.FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED | H.FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;
	private context: H.Fsr3UpscalerContext;
	// (until the canvas's size is known: large enough for SPD_MIPS's six levels)
	private maxSize: Size = { width: 64, height: 64 };
	private readonly textures = new Map<number, THREE.Texture>();
	private readonly bindings: Fsr3Bindings;
	private readonly kernels: Fsr3PassKernels;
	// the generator's: its own bindings (its colour input is the scene's, not hdr) on the same resources
	private readonly generator: { bindings: Fsr3Bindings; kernel: THREE.ComputeNode } | null = null;
	private reconstructedDepth: THREE.StorageBufferAttribute;
	private readonly frameInfo: THREE.StorageBufferAttribute;
	private plan: H.Fsr3FramePlan | null = null;
	private generate: { dispatchX: number; dispatchY: number } | null = null;
	// the texture nodes the post chain reads FSR's output through (their value: the output or the held copy)
	private readonly sources: THREE.TextureNode[] = [];
	private showHeld = false;
	// the pre-exposure (PreExposure.ts `stops`) FSR's output was last written at, and the held copy's
	private outputStops = 0;
	private heldStops = 0;
	private resetReason: string | null = 'start';
	private readonly key: number[] = [];
	private settingsKey = '';
	private settingsVersion = 0;
	private readonly u = { near: uniform( 1 ) };
	// the clear kernels any frame may schedule (the first execution's list holds every one)
	private readonly clearJobs: H.Fsr3ClearJob[];
	private readonly frameStep = { step: 0, interval: 1000 / 60 };

	constructor( renderer: THREE.WebGPURenderer, graph: FrameGraph<unknown>, inputs: Fsr3GraphInputs, options: Fsr3Options ) {

		this.renderer = renderer;
		this.options = options;
		this.settings = options.settings;
		const autoReactive = options.settings.fsr3.autoReactive || debugFlags.fsrAutoReactive === true;
		this.context = this.newContext();
		// the internal resources, at a placeholder size until the canvas's is known (`resize`), in
		// upstream's narrow formats where the device stores them (Resources.ts `formatFor`)
		const tier1 = renderer.hasFeature( 'texture-formats-tier1' );
		for ( const d of R.FSR3_INTERNAL_RESOURCES ) {

			if ( d.format === 'buffer-u32' || d.format === 'buffer-vec4f' ) continue;
			if ( d.id === R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LANCZOS_LUT ) continue; // (the reference Lanczos, no LUT)
			this.textures.set( d.id, storageTexture( d.name, R.formatFor( d, tier1 ), d.linear, d.mips ) );

		}
		this.output = storageTexture( 'FSR3UPSCALER_UpscaledOutput', 'rgba16float', true );
		this.held = storageTexture( 'FSR3UPSCALER_Held', 'rgba16float', true );
		this.textures.set( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT, this.output );
		if ( autoReactive ) this.textures.set( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_AUTOREACTIVE, storageTexture( 'FSR3UPSCALER_AutoReactive', 'rgba8unorm', true ) );
		this.reconstructedDepth = new THREE.StorageBufferAttribute( new Uint32Array( this.maxSize.width * this.maxSize.height ), 1 );
		this.frameInfo = new THREE.StorageBufferAttribute( new Float32Array( 4 ), 4 );
		this.sizeTextures( this.maxSize );

		// the frame's inputs
		const g = graph;
		const tHdr = texture( g.texture( inputs.hdr ) ), tDepth = texture( g.texture( inputs.surfaceDepth ) ), tVelocity = texture( g.texture( inputs.velocity ) );
		const tVeil = inputs.veilReactive ? texture( g.texture( inputs.veilReactive ) ) : null;
		const tAuto = autoReactive ? texture( this.textures.get( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_AUTOREACTIVE )! ) : null;
		// (a load with no uv matrix: three r186 gives a load of a graph texture a mat3 uniform per load
		// site, the identity, multiplied into every load's texel coordinates, as Bindings.ts `loadAt`)
		const load = ( t: THREE.TextureNode, p: N ) => ( textureLoad( t, ivec2( p ), int( 0 ) ) as unknown as N ).setUpdateMatrix( false ) as N;
		const reactive = ( p: N ): N => {

			let r = load( tVelocity, p ).a;
			if ( tVeil ) r = max( r, load( tVeil, p ).r );
			if ( tAuto ) r = max( r, load( tAuto, p ).r );
			return r;

		};
		const near = this.u.near;
		const fsrInputs: Fsr3Inputs = {
			LoadInputColor: ( p ) => load( tHdr, p ).rgb,
			// reversed infinite device depth from the nearest surface's view z (the sky: 0, the far plane)
			LoadInputDepth: ( p ) => {

				const z = load( tDepth, p ).r;
				return select( z.greaterThan( - 9e4 ), near.div( max( z.negate(), 1e-6 ) ), float( 0 ) );

			},
			LoadInputMotionVector: ( p ) => load( tVelocity, p ).xy,
			LoadInputExposure: options.exposure,
			LoadReactiveMask: reactive,
			GetReactiveMaskResourceDimensions: () => ivec2( this.bindings.RenderSize() ),
		};
		const description = () => ( {
			flags: this.flags, textures: this.textures,
			buffers: { reconstructedPrevNearestDepth: this.reconstructedDepth, frameInfo: this.frameInfo },
		} );
		this.bindings = new Fsr3Bindings( { ...description(), inputs: fsrInputs } );
		this.kernels = createPassKernels( this.bindings, this.maxSize, this.maxSize );
		if ( autoReactive ) {

			const tScene = texture( g.texture( inputs.sceneColor ) ), tOpaque = texture( g.texture( inputs.opaqueColor ) );
			// (in scene units: its threshold and tone map are absolute, PreExposure.ts)
			const bindings = new Fsr3Bindings( { ...description(), inputs: { ...fsrInputs, LoadInputColor: ( p ) => load( tScene, p ).rgb.mul( preExposure.inverse ), LoadOpaqueOnly: ( p ) => load( tOpaque, p ).rgb.mul( preExposure.inverse ) } } );
			( this as unknown as { generator: Fsr3Upscaler[ 'generator' ] } ).generator = { bindings, kernel: createPassKernels( bindings, this.maxSize, this.maxSize ).pass( H.FFX_FSR3UPSCALER_PASS_GENERATE_REACTIVE ) };

		}
		// every clear a frame may schedule: a first execution schedules them all
		this.clearJobs = this.newContext().dispatch( this.description( { width: 1, height: 1 }, { width: 1, height: 1 } ) ).clears;
		// the canvas's size: every texture and buffer, and a new context
		graph.onResize( CANVAS, ( w, h ) => this.resize( w, h ) );
		onTaaCut( ( reason ) => this.requestReset( reason ) );
		Fsr3Upscaler.current = this;

	}

	// ---- the graph

	/** A texture node the post chain reads FSR's output through (at `uvNode`): the held copy while
	 * the rest's sequence runs, FSR's output otherwise. */
	source( uvNode?: THREE.Node<'vec2'> ): THREE.TextureNode {

		const node = texture( this.output, uvNode );
		this.sources.push( node );
		return node;

	}

	/** Declares FSR's passes after the composite; returns the versions of its output and of the
	 * held copy the post chain reads (optionally: FSR does not dispatch while the rest holds). */
	declare( graph: FrameGraph<unknown>, inputs: Fsr3GraphInputs ): { output: ResourceHandle; held: ResourceHandle } {

		const g = graph;
		let output = g.import( 'fsr3Output', this.output, CANVAS );
		let held = g.import( 'fsr3Held', this.held, CANVAS );
		const when = () => this.dispatching;
		const k = this.kernels;
		const r = this.renderer;
		const dispatch = ( passId: number ) => {

			for ( const job of this.plan!.jobs ) if ( job.pass === passId ) r.compute( k.pass( passId ), [ job.dispatchX, job.dispatchY, 1 ] );

		};
		const generator = this.generator;
		if ( generator ) g.addPass( 'fsr3Generate', 'compute', ( b ) => {

			b.readOptional( inputs.sceneColor );
			b.read( inputs.opaqueColor );
			return {
				label: 'FSR 3', size: RENDER, neverCull: true, when, whenLabel: 'fsr3 dispatching', compute: [ generator.kernel ],
				run: () => r.compute( generator.kernel, [ this.generate!.dispatchX, this.generate!.dispatchY, 1 ] ),
			};

		} );
		// the plan's clears, and at the rest's first frame the copy of the last running output
		const clearKernels = [ ...new Set( this.clearJobs.map( ( job ) => k.clear( job ) ) ) ];
		g.addPass( 'fsr3Clears', 'compute', ( b ) => {

			held = b.storage( held );
			return {
				label: 'FSR 3', size: CANVAS, neverCull: true, when, whenLabel: 'fsr3 dispatching', compute: clearKernels,
				run: () => {

					if ( this.rest.index === 0 ) r.copyTextureToTexture( this.output, this.held );
					for ( const job of this.plan!.clears ) r.compute( k.clear( job ), this.clearDispatch( job ) );

				},
			};

		} );
		const pass = ( name: string, passId: number, size: SizeClass, reads: ( b: PassBuilder ) => void, extra: THREE.ComputeNode[] = [], after?: () => void ) => {

			g.addPass( name, 'compute', ( b ) => {

				if ( inputs.exposure ) b.read( inputs.exposure );
				reads( b );
				return {
					label: 'FSR 3', size, neverCull: true, when, whenLabel: 'fsr3 dispatching', compute: [ k.pass( passId ), ...extra ],
					run: () => {

						dispatch( passId );
						after?.();

					},
				};

			} );

		};
		pass( 'fsr3PrepareInputs', H.FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS, RENDER, ( b ) => {

			b.read( inputs.hdr );
			b.read( inputs.surfaceDepth );
			b.readOptional( inputs.velocity );

		} );
		const tail = k.lumaPyramidTail();
		pass( 'fsr3LumaPyramid', H.FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID, RENDER, () => {}, [ tail ], () => {

			if ( this.plan!.spdTail ) r.compute( tail, [ 1, 1, 1 ] );

		} );
		// (its tail stores only levels above 5, which nothing reads: not dispatched)
		pass( 'fsr3ShadingChangePyramid', H.FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID, RENDER, () => {} );
		pass( 'fsr3ShadingChange', H.FFX_FSR3UPSCALER_PASS_SHADING_CHANGE, HALF, () => {} );
		pass( 'fsr3PrepareReactivity', H.FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY, RENDER, ( b ) => {

			b.readOptional( inputs.velocity );
			if ( inputs.veilReactive ) b.read( inputs.veilReactive );

		} );
		pass( 'fsr3LumaInstability', H.FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY, RENDER, () => {} );
		pass( 'fsr3Accumulate', H.FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN, CANVAS, ( b ) => {

			b.read( inputs.hdr );
			b.readOptional( inputs.velocity );

		} );
		g.addPass( 'fsr3Rcas', 'compute', ( b ) => {

			output = b.storage( output );
			return { label: 'FSR 3', size: CANVAS, when, whenLabel: 'fsr3 dispatching', compute: [ k.pass( H.FFX_FSR3UPSCALER_PASS_RCAS ) ], run: () => dispatch( H.FFX_FSR3UPSCALER_PASS_RCAS ) };

		} );
		if ( this.options.debugView ) g.addPass( 'fsr3DebugView', 'compute', ( b ) => {

			output = b.storage( output );
			return { label: 'FSR 3', size: CANVAS, when, whenLabel: 'fsr3 dispatching', compute: [ k.pass( H.FFX_FSR3UPSCALER_PASS_DEBUG_VIEW ) ], run: () => dispatch( H.FFX_FSR3UPSCALER_PASS_DEBUG_VIEW ) };

		} );
		return { output, held };

	}

	// the workgroup counts of a clear job at the context's sizes (the kernels' own fixed counts were
	// taken at the size they were built with)
	private clearDispatch( job: H.Fsr3ClearJob ): [ number, number, number ] {

		const m = this.maxSize;
		switch ( job.target ) {

			case R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FRAME_INFO: return [ 1, 1, 1 ];
			case R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH: return [ groups( m.width, 8 ), groups( m.height, 8 ), 1 ];
			case R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS: return [ groups( Math.trunc( m.width / 2 ), 8 ), groups( Math.trunc( m.height / 2 ), 8 ), 1 ];
			default: {

				const s = R.getResourceSize( R.FSR3_INTERNAL_RESOURCES.find( ( d ) => d.id === job.target )!, m, m );
				return [ groups( s.width, 8 ), groups( s.height, 8 ), 1 ];

			}

		}

	}

	// ---- sizes, context, resets

	private newContext(): H.Fsr3UpscalerContext {

		return new H.Fsr3UpscalerContext( { flags: this.flags, maxRenderSize: { ...this.maxSize }, maxUpscaleSize: { ...this.maxSize } } );

	}

	private resize( width: number, height: number ): void {

		const size = { width: Math.max( width, 1 ), height: Math.max( height, 1 ) };
		this.maxSize = size;
		this.sizeTextures( size );
		// (a storage buffer cannot grow in place: a new one, bound by the same nodes)
		this.reconstructedDepth = new THREE.StorageBufferAttribute( new Uint32Array( size.width * size.height ), 1 );
		this.bindings.setReconstructedDepthBuffer( this.reconstructedDepth );
		this.generator?.bindings.setReconstructedDepthBuffer( this.reconstructedDepth );
		this.context = this.newContext();
		this.requestReset( 'resize' );

	}

	// every texture at its class's size for a render-max = upscale-max size
	private sizeTextures( size: Size ): void {

		for ( const d of R.FSR3_INTERNAL_RESOURCES ) {

			const t = this.textures.get( d.id ) as THREE.StorageTexture | undefined;
			if ( ! t ) continue;
			const s = R.getResourceSize( d, size, size );
			t.setSize( Math.max( s.width, 1 ), Math.max( s.height, 1 ), 1 );

		}
		for ( const t of [ this.output, this.held, this.textures.get( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_AUTOREACTIVE ) as THREE.StorageTexture | undefined ] ) t?.setSize( size.width, size.height, 1 );

	}

	private requestReset( reason: string ): void {

		this.resetReason ??= reason;

	}

	// ---- the frame

	/** Before the frame's jitter is chosen (the camera unjittered): the rest test, which sets the
	 * jitter's rest index, and whether FSR dispatches. */
	beginFrame( camera: THREE.Camera, renderWidth: number, renderHeight: number, outputWidth: number, outputHeight: number, time: number, frame: TemporalFrame = {} ): void {

		const S = this.settings;
		const k = JSON.stringify( S.fsr3 ) + S.restSettle;
		if ( k !== this.settingsKey ) {

			this.settingsKey = k;
			this.settingsVersion ++;

		}
		const key = restKey( this.key, camera, renderWidth, renderHeight, outputWidth, outputHeight, time, [ this.settingsVersion, ...( frame.signature ?? [] ) ] );
		const rest = this.rest;
		rest.settle = Math.max( 0, Math.round( S.restSettle ) );
		rest.sequence = Math.max( 0, Math.round( S.fsr3.restSequence ) );
		rest.observe( key, frame.stable !== false );
		// (while converged the jitter stays at the sequence's last index: see the header)
		jitter.restIndex = rest.index !== null ? rest.index + 1 : rest.converged ? rest.sequence : null;
		jitter.restLength = rest.sequence;
		this.dispatching = ! rest.converged || rest.completing;
		this.showHeld = rest.index !== null && ! rest.completing;
		this.frameStep.step = Math.max( frame.step ?? 0, 0 );
		this.frameStep.interval = frame.interval ?? 1000 / 60;

	}

	/** After the jitter and the motion's frame (cuts applied): this frame's plan, its constants in the
	 * bindings, the post chain's source and the mip bias. */
	prepareFrame( camera: THREE.PerspectiveCamera, renderWidth: number, renderHeight: number, outputWidth: number, outputHeight: number ): void {

		// (the held copy is taken at the rest's first frame from the last output, at that output's P)
		if ( this.dispatching && this.rest.index === 0 ) this.heldStops = this.outputStops;
		const value = this.showHeld && this.heldStops === preExposure.stops ? this.held : this.output;
		for ( const node of this.sources ) node.value = value;
		fsr3MipBiasScale.value = debugFlags.fsrMipBias === false ? 1 : renderWidth / Math.max( outputWidth, 1 );
		if ( ! this.dispatching ) return;
		// the rest starts from a new context, whatever ran before (see the header)
		if ( this.rest.index === 0 ) {

			this.context = this.newContext();
			this.requestReset( 'rest' );

		}
		const reset = this.resetReason !== null;
		if ( reset ) {

			this.resets ++;
			this.lastReset = this.resetReason;
			this.resetReason = null;

		}
		this.u.near.value = camera.near;
		const desc = this.description( { width: renderWidth, height: renderHeight }, { width: outputWidth, height: outputHeight } );
		desc.reset = reset;
		desc.cameraNear = camera.near;
		desc.cameraFar = camera.far;
		desc.cameraFovAngleVertical = fovOf( jitter.uniforms.unjitteredProjection.value );
		const plan = this.plan = this.context.dispatch( desc );
		this.outputStops = preExposure.stops;
		this.bindings.setFrame( plan );
		const generator = this.generator;
		if ( generator ) {

			generator.bindings.setFrame( plan );
			const gen = this.context.generateReactiveMask( {
				renderSize: { width: renderWidth, height: renderHeight }, scale: 1, cutoffThreshold: 0.2, binaryValue: 0.9,
				flags: R.FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_TONEMAP | R.FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_THRESHOLD | R.FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_USE_COMPONENTS_MAX,
			} );
			generator.bindings.setGenerateReactive( gen.constants );
			this.generate = gen;

		}

	}

	// a dispatch description at these sizes (the camera's fields filled by the caller)
	private description( render: Size, upscale: Size ): H.FfxFsr3UpscalerDispatchDescription {

		const step = this.frameStep;
		// (the conventions' checks: `flags.fsrJitterSign` and `flags.fsrMotionScale` scale what FSR is
		// told the jitter and the motion are, to show its image is best at 1)
		const js = typeof debugFlags.fsrJitterSign === 'number' ? debugFlags.fsrJitterSign : 1;
		const ms = typeof debugFlags.fsrMotionScale === 'number' ? debugFlags.fsrMotionScale : 1;
		return {
			hasExposure: true,
			jitterOffset: { x: - js * jitter.px.x, y: - js * jitter.px.y },
			motionVectorScale: { x: - 0.5 * ms * render.width, y: 0.5 * ms * render.height },
			renderSize: { ...render }, upscaleSize: { ...upscale },
			enableSharpening: true, sharpness: this.settings.fsr3.sharpness,
			frameTimeDelta: step.step > 0 ? step.step * 1000 : step.interval,
			preExposure: preExposure.scale.value, reset: false, cameraNear: 1, cameraFar: 60000, cameraFovAngleVertical: Math.PI / 4,
			viewSpaceToMetersFactor: 1,
			flags: this.options.debugView ? H.FFX_FSR3UPSCALER_DISPATCH_DRAW_DEBUG_VIEW : 0,
		};

	}

	/** Whether the image shown is the rest's. */
	get converged(): boolean {

		return this.rest.converged;

	}

	/** Restarts the rest (the next frame is the first at rest). */
	restartRest(): void {

		this.rest.restart();

	}

	/** The graph's warm-up compiles every kernel its passes list: nothing more. */
	warm(): Promise<void> {

		return Promise.resolve();

	}

	/** The diagnostics' `taa` block (TemporalAA's shape, plus `fsr3`). */
	diagnostics(): Record<string, unknown> {

		const c = this.plan?.constants;
		return {
			mode: 'fsr3',
			converged: this.rest.converged,
			restFrames: this.rest.frames,
			restTarget: this.rest.target,
			restIndex: this.rest.index,
			restEpoch: restEpoch.value,
			restReason: restEpoch.reason,
			lastEvent: sceneEvents.last,
			jitterIndex: jitter.k + 1,
			jitterLength: jitter.restIndex !== null ? this.rest.sequence : jitter.length,
			historyResets: this.resets,
			cuts: motionState.historyResets,
			inputSize: c ? [ ...c.renderSize ] : null,
			outputSize: c ? [ ...c.upscaleSize ] : null,
			sharpness: this.settings.fsr3.sharpness,
			fsr3: {
				dispatching: this.dispatching, held: this.showHeld, lastReset: this.lastReset,
				frameIndex: c?.frameIndex ?? null, jitterPhaseCount: c?.jitterPhaseCount ?? null, jitter: c ? [ ...c.jitterOffset ] : null,
				deltaTime: c?.deltaTime ?? null, motionVectorScale: c ? [ ...c.motionVectorScale ] : null, deviceToViewDepth: c ? [ ...c.deviceToViewDepth ] : null,
				tanHalfFOV: c?.tanHalfFOV ?? null, maxSize: [ this.maxSize.width, this.maxSize.height ], mipBias: Math.log2( fsr3MipBiasScale.value ),
				spdMips: this.plan?.spd.mips ?? null, jobs: this.plan?.jobs.map( ( j ) => `${ j.pass }:${ j.dispatchX }x${ j.dispatchY }` ) ?? null,
				clears: this.plan?.clears.map( ( j ) => j.label ) ?? null, generator: this.generator !== null,
			},
		};

	}

}
