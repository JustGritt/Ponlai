// A running scene's frame: the canvas's size, the render scale, the tick, the loop and its controls.
//
// A running scene's frame (StartScene.ts): the canvas's size and the render scale (fixed, dynamic
// or stepped), the frame pacer, the tick, the loop that paces it, the loading screen's lift, and
// the controls a tool holds, resumes and steps it by (`loop`, `step`). The tick is the order of a
// frame. It calls the render pipeline's frame stages (RenderStages.ts `createFrameStages`) in their
// order and the caller's content (`SceneFrameContent`) in the slots between them: the timer and the
// frame's deltas, the dynamic render scale, the pacer's queue and the diagnostics are its own. The
// order is part of the image: a stage reads what the ones before it wrote in the same frame, so
// content goes in the slot its statements belong to. The frame writes no uniform and no cadence of
// the render pipeline's here; it reads `render.paused` (the scene's pause) and the rest.
// - The render scale: high's is dynamic (post/DynamicResolution.ts) on a page that asks for it
//   (`dynamicScale`): it starts at its baseline (the nominal: 2K's pixel count, native at 2K output
//   and 1440p at 4K output), moves between the display's ratio and a floor (about 1080p's pixel
//   count) by the GPU's frame time, and renders at the baseline while the scene is paused, so paused
//   frames and captures render at the scale its trade-offs are judged at (pinned, `dpr` or full
//   resolution, at that ratio). Without it the render is at the tier's nominal scale.
// - The pacer (post/FramePacer.ts) holds the tier's frame rate on a faster display. `frameRate=auto`
//   is that default; `frameRate=<fps>` holds a fixed rate (0 renders on every refresh: the
//   profiler's uncapped runs, where the animation frames' intervals measure the work, not a display);
//   `frameFallback=0` keeps 'auto' from falling back (A/B runs). A settings menu sets the rate
//   through the pacer (`choices()`, `setRate()`, `mode`, `fps`).
// - A hidden page draws nothing: browsers stop a hidden tab's animation frames, but an app's
//   embedded browser can keep running them (a page left in a hidden pane rendered at 60 fps and took
//   half the GPU from the page on screen). The loop stops while the page is hidden and starts again
//   when it shows, unless it was stopped or started otherwise meanwhile (the profiler's two pages).
// Deterministic stepping for the clip tools (tools/kit/parity/capture-clip.mjs; debug builds only,
// `step`): stops the live loop and drives the tick itself, `frames` times, `dt` seconds apart, with
// nothing taken from the wall clock. Each frame gets the timestamp t0 + i * dt * 1000 and a delta of
// exactly `dt` (three's Timer would take it from the timestamps less its own start time, which rounds
// differently from load to load), does the node-frame bookkeeping three's animation loop does,
// bypasses the frame pacer and renders at `scale` (a render pixel ratio, as `dpr`; one value, or one
// per frame with the last repeated; the tier's nominal by default) with the dynamic controller out of
// the loop. `camera( i )` runs before frame i's tick and `onFrame( i )` is awaited after it (a
// screenshot then sees that frame: nothing renders until it returns). `clock` pauses or runs the
// scene clock for the clip; left out, it keeps the app's. The loop stays stopped afterwards:
// `loop.start()` brings the live one back.
// What would still follow the wall clock or the page's history is pinned too, so a clip is the
// same however long its frames, its `onFrame` or the wait before it took:
// - a whole sky bake (a seek, a preset, a weather jump) publishes after a GPU readback, so the next
//   frame waits for it (the live loop shows the old sky for a few frames first);
// - the timer that completes a periodic bake spread over frames once the loop stalls is cancelled
//   after each frame, so the bake takes its share of every frame as in a running loop;
// - the clouds pick their trace slot and map refresh phase from the frames rendered since the load
//   (Clouds.ts), and the periodic sky bake and the refracted sun's map count wall-clock seconds
//   that grow while a paused page waits: all three restart with each call, as after a seek;
// - a periodic bake reads its sky irradiance back from the GPU (`Sky.skyIrradiance`, which sets
//   a lightning strike's fill light), landing whenever the GPU answers:
//   stepped, the next frame waits for the value, so it lands on the frame after the bake however
//   long the frames took.
// - `converge( i )` (the temporal modes' ground truth, capture-clip.mjs --aa gt): after frame i's tick
//   the scene is held still (the clock paused, nothing advancing with time: a delta of 0, the sky's
//   transitions and wall-clock cadences included) and rendered again until the temporal rest has
//   converged on it (TemporalRest.ts), so `onFrame( i )` sees frame i's scene averaged over the rest's
//   jitters. The clock and the cadences continue afterwards as if the hold had not happened.
import * as THREE from 'three/webgpu';
import { createFrameStages, type FrameStages, type RenderParts } from '../RenderStages.ts';
import { settings as params } from '../Settings.ts';
import { DynamicResolution, GpuFrameTimer, PassScaling, TAAU_MAX_UPSCALE, renderScaleKey } from '../post/DynamicResolution.ts';
import { FramePacer, PAUSED_QUEUE } from '../post/FramePacer.ts';
import { dynamicRange, outputPixelRatio, renderPixelRatio } from '../post/RenderScale.ts';
import { AA_MODE } from '../post/Antialiasing.ts';
import { renderEvent } from '../post/SceneEvents.ts';
import { TEST_MODE } from '../debug/TestMode.ts';
import { gpuDevice } from '../three/Backend.ts';
import { frameBookkeeping, threeAnimation } from '../three/FrameLoop.ts';
import type { SceneHost } from './SceneHost.ts';

export interface StepOptions {
	frames: number;
	dt?: number;
	scale?: number | number[];
	camera?: ( i: number ) => void;
	onFrame?: ( i: number ) => Promise<void> | void;
	clock?: 'paused' | 'running';
	converge?: ( i: number ) => boolean;
}

export interface StepResult { frames: number; t0: number; simTime: number; frameId: number }

/** The caller's part of each frame, in the tick's slots (in the order they run). */
export interface SceneFrameContent {
	/** first, once the frame is known to draw: before the deltas are taken */
	start?( time: number ): void;
	/** before the scene clock advances */
	beforeClock?(): void;
	/** the day clock's step for the raw delta (the raw delta by default) */
	dayStep?( rawDt: number ): number;
	/** before the day clock advances */
	beforeDay?(): void;
	/** the hour the sky shows, when something other than the day clock sets it */
	hour?: () => number;
	/** after the day clock: the camera, the GUI's mirrors */
	afterDay?( time: number, dt: number, rawDt: number ): void;
	/** the local lights' share of the night (0 by default) */
	localLights?(): number;
	/** after the night's light, with the night's ambient */
	afterNight?( nightAmbient: number ): void;
	/** after the weather and the sea */
	afterWeather?( dt: number ): void;
	/** after the relight, before the draw */
	afterRelight?( dt: number, rawDt: number ): void;
	/** why the frame is not settled yet, besides the render pipeline's own reasons */
	pending?( reasons: string[] ): void;
}

/** What the frame's setup asks of the caller before the first resize. */
export interface SceneFrameSetup {
	/** called with each GPU frame time as it is read back (a load that measures its GPU) */
	gpuSample?: ( frame: number, ms: number ) => void;
	/** false: the dynamic scale starts at its nominal rather than at the scale it remembered */
	restore?: boolean;
}

export interface SceneFrameOptions {
	/** whether high's render scale follows the GPU's frame time (else the tier's nominal) */
	dynamicScale: boolean;
	/** the switches the frame's stages read (RenderStages.ts `FrameHost`) */
	weatherEffects: { rain: boolean; wetness: boolean };
	environmentLook: { exposure: boolean };
	/** after the pacer and the timers are made, before the first resize */
	setup?( frame: SceneFrame ): SceneFrameSetup | void;
	/** wraps the canvas's resize (`apply`) on a change of the window's size */
	resize?( apply: () => void ): void;
	/** the caller's part of each frame, made once the frame's stages exist */
	content?( frame: SceneFrame ): SceneFrameContent;
	/** the diagnostics, refreshed after each frame (`now`: a whole refresh) */
	diagnostics( now?: boolean ): void;
}

/** The canvas's size and the render scale: made before the frame, as an app's GUI may resize. */
export interface SceneSizing {
	/** sizes the canvas and the render to the window */
	resize(): void;
	/** whether the canvas has been sized yet */
	readonly sized: boolean;
	applyRenderScale( force?: boolean ): void;
	describeRenderScale(): void;
	refreshStoreKey(): void;
	nominalScale: number;
	renderScale: number;
	/** the render scale `loop.step` holds for the frame it draws (null otherwise) */
	steppedScale: number | null;
	readonly canvasSize: THREE.Vector2;
}

export interface SceneFrame {
	readonly sizing: SceneSizing;
	readonly pacer: FramePacer;
	readonly dynamicResolution: DynamicResolution | null;
	readonly dynamicRule: { dynamic: { max: number; baseline: number; floor: number; floorShare: number; step: number } } | null;
	readonly gpuTimer: GpuFrameTimer | null;
	readonly passScaling: PassScaling | null;
	readonly gpuQueue: { onSubmittedWorkDone(): Promise<void> };
	readonly clock: THREE.Timer;
	readonly stages: FrameStages;
	/** the frames drawn since the load */
	readonly frame: number;
	/** the last 120 frames' intervals (ms) */
	readonly frameTimes: number[];
	readonly tick: ( time: number ) => void;
	/** the tick as the loop runs it (the profiler's wrap) */
	readonly loopBody: ( time: number ) => void;
	loop: { running: boolean; stop(): Promise<void>; start(): void };
	step( options: StepOptions ): Promise<StepResult>;
	/** stops the loop and the frame's listeners */
	dispose(): void;
}

/** The canvas's size and the render scale's operations. */
export function createSceneSizing( render: RenderParts, host: SceneHost, frame: () => SceneFrame | null ): SceneSizing {

	const { profiler, qualityTier, quality, renderer, DPR_CAP, resolution, camera, pipeline } = render;
	let storeKeyGeneration = NaN;
	let scaleMemory = false;
	const sizing: SceneSizing & { sized: boolean; scaleMemory: boolean } = {
		sized: false,
		nominalScale: 1,
		renderScale: 1,
		steppedScale: null,
		canvasSize: new THREE.Vector2(),
		get scaleMemory() { return scaleMemory; },
		set scaleMemory( on: boolean ) { scaleMemory = on; },
		resize,
		applyRenderScale,
		describeRenderScale,
		refreshStoreKey,
	};
	function resize(): void {

		sizing.sized = true;
		const { width: w, height: h } = host.size();
		const f = frame();
		sizing.nominalScale = renderPixelRatio( w, h, quality.render, resolution.full, DPR_CAP );
		// the ground truth renders at the canvas's ratio, every pixel of it
		if ( AA_MODE === 'gt' ) sizing.nominalScale = outputPixelRatio( sizing.nominalScale );
		// the dynamic range for this view: from the display's ratio (a pinned one's) down to the floor,
		// its baseline among them
		if ( f?.dynamicResolution && f.dynamicRule ) f.dynamicResolution.setRange( dynamicRange( w, h, resolution.full ? sizing.nominalScale : host.pixelRatio(), f.dynamicRule, DPR_CAP ) );
		renderer.setPixelRatio( outputPixelRatio( sizing.nominalScale ) );
		f?.dynamicResolution?.setCanvasRatio( renderer.getPixelRatio() );
		renderer.setSize( w, h );
		camera.aspect = w / h;
		camera.updateProjectionMatrix();
		applyRenderScale( true );
		refreshStoreKey();

	}

	// The key the dynamic scale remembers its scales under (the frame rate's in it): set on a resize and
	// whenever the pacer's rate changes (`generation`).
	function refreshStoreKey(): void {

		const f = frame();
		if ( ! f ) return;
		storeKeyGeneration = f.pacer.generation;
		if ( scaleMemory ) f.dynamicResolution?.setStoreKey( renderScaleKey( renderer, qualityTier, AA_MODE, f.pacer.rate, sizing.canvasSize.x, sizing.canvasSize.y, f.gpuTimer !== null || profiler?.timestamps === true ? 'gpu' : 'frames' ) );

	}
	( sizing as unknown as { storeKeyStale(): boolean } ).storeKeyStale = () => frame()?.pacer.generation !== storeKeyGeneration;

	/** The scale this frame renders at: the dynamic one, unless pinned or paused. */
	function currentRenderScale(): number {

		const d = frame()?.dynamicResolution;
		if ( sizing.steppedScale !== null ) return sizing.steppedScale;
		if ( ! d || resolution.full || render.paused ) return sizing.nominalScale;
		return d.scale;

	}

	// Sizes the frame graph to the render scale when it changed (a resize sets the canvas first).
	function applyRenderScale( force = false ): void {

		const ratio = currentRenderScale();
		if ( ! force && ratio === sizing.renderScale ) return;
		sizing.renderScale = ratio;
		if ( profiler ) profiler.renderScale = ratio;
		const { canvasSize } = sizing;
		renderer.getDrawingBufferSize( canvasSize );
		const { width: w, height: h } = host.size();
		// at the display's ratio the render is the canvas, to the pixel, and nothing is upscaled
		const rw = ratio === renderer.getPixelRatio() ? canvasSize.x : Math.floor( w * ratio );
		const rh = ratio === renderer.getPixelRatio() ? canvasSize.y : Math.floor( h * ratio );
		pipeline.setSize( rw, rh, canvasSize.x, canvasSize.y );
		frame()?.passScaling?.setSize( rw, rh );
		resolution.render = `${ rw }x${ rh } → ${ canvasSize.x }x${ canvasSize.y }` + ( pipeline.upscaling || ( pipeline.fsr3 && rw !== canvasSize.x ) ? ` (${ Math.round( rw / canvasSize.x * 100 ) } %, ${ pipeline.fsr3 ? 'FSR 3' : 'FSR' })` : '' );
		describeRenderScale();

	}

	// The GUI's lines: the scale and how it is set, and (dynamic) the GPU's recent frame time.
	function describeRenderScale(): void {

		const d = frame()?.dynamicResolution ?? null;
		const how = ! d ? 'fixed' : resolution.full ? 'pinned' : render.paused ? 'paused' : `${ d.floor }–${ d.max }`;
		resolution.scale = `${ sizing.renderScale.toFixed( 2 ) } (${ how })`;
		resolution.gpu = d && d.observed.length ? `${ [ ...d.observed ].sort( ( a, b ) => a - b )[ d.observed.length >> 1 ].toFixed( 1 ) } ms` : '–';

	}

	return sizing;

}

/** The dynamic resolution, the pacer and the tick (the loop starts with `startSceneLoop`). */
export function createSceneFrame( render: RenderParts, host: SceneHost, sizing: SceneSizing, options: SceneFrameOptions, publish: ( frame: SceneFrame ) => void ): SceneFrame {

	const { profiler, DEBUG, quality, gpuTiming, renderer, DPR_CAP, resolution, sky, dayCycle, pipeline } = render;
	const internal = sizing as SceneSizing & { scaleMemory: boolean; storeKeyStale(): boolean };
	const removers: Array<() => void> = [];

	// (the ground truth renders at the canvas's resolution: no dynamic scale)
	const dynamicRule = options.dynamicScale && 'dynamic' in quality.render && AA_MODE !== 'gt' ? quality.render : null;
	// Under TAAU and FSR 3 the floor rises so the canvas is at most TAAU_MAX_UPSCALE times the render per axis
	// (DynamicResolution.ts).
	const start = host.size();
	const dynamicResolution = dynamicRule ? new DynamicResolution( { ...dynamicRange( start.width, start.height, host.pixelRatio(), dynamicRule, DPR_CAP ), step: dynamicRule.dynamic.step, maxUpscale: AA_MODE === 'taau' || AA_MODE === 'fsr3' ? TAAU_MAX_UPSCALE : Infinity, store: host.store } ) : null;
	const frameRateParam = params.get( 'frameRate' );
	// (the queue whose completed work bounds a paused page's frames in flight)
	const gpuQueue = gpuDevice<{ queue: { onSubmittedWorkDone(): Promise<void> } }>( renderer ).queue;
	const pacer = new FramePacer( quality.frameRate, {
		rate: frameRateParam === null || frameRateParam === 'auto' ? 'auto' : Number( frameRateParam ),
		allowFallback: params.get( 'frameFallback' ) !== '0',
	} );
	// the render scale each frame was drawn at and its periodic work (by three's frame id), for the
	// GPU times read back later
	const frameScales = new Map<number, { scale: number; heavy: number }>();
	let gpuTimer: GpuFrameTimer | null = null;
	let gpuSample: ( ( frame: number, ms: number ) => void ) | null = null;
	// which passes follow the render's pixel count: the GPU time's split into a fixed part and one
	// that follows the scale
	const passScaling = dynamicResolution && gpuTiming ? PassScaling.install( renderer ) : null;
	if ( passScaling ) passScaling.graph = pipeline.graph;
	if ( dynamicResolution ) {

		const sample = ( frame: number, ms: number, pixel: number ) => {

			gpuSample?.( frame, ms );
			const drawn = frameScales.get( frame );
			if ( drawn !== undefined ) dynamicResolution.sample( ms, drawn.scale, { pixel, heavy: drawn.heavy } );

		};
		// the frame profiler owns the timestamp queries while it times passes
		if ( profiler?.timestamps ) {

			if ( passScaling ) profiler.pixelPass = ( uid ) => passScaling.take( uid );
			profiler.onRecord = ( record ) => { if ( record.missing === 0 && record.passes > 0 ) sample( record.id, record.gpu, record.pixel ); };

		} else if ( passScaling ) {

			gpuTimer = GpuFrameTimer.create( renderer, passScaling );
			if ( gpuTimer ) gpuTimer.onFrame = sample;

		}
		if ( DEBUG ) dynamicResolution.onWarning = ( message ) => console.warn( message );
		// the dynamic scale decides when the pacer falls back to its slower rate and returns
		dynamicResolution.pacer = pacer;

	}

	const clock = new THREE.Timer();
	const frameTimes: number[] = [];
	let frameCount = 0;
	let content: SceneFrameContent = {};
	let stages: FrameStages | null = null;
	// set while the loop stands still because the page is hidden (see the header)
	let hiddenStop = false;
	// whether the last rendered frame steered the scale (false again at a loop start)
	let steeredLast = false;
	let animation: { start(): void; stop(): void; setContext( context: unknown ): void } | null = null;
	const frame: SceneFrame = {
		sizing,
		pacer,
		dynamicResolution,
		dynamicRule,
		get gpuTimer() { return gpuTimer; },
		passScaling,
		gpuQueue,
		clock,
		get stages() { return stages!; },
		get frame() { return frameCount; },
		frameTimes,
		tick,
		loopBody: profiler ? profiler.wrapLoop( tick ) : tick,
		loop: {
			running: false,
			async stop(): Promise<void> {

				hiddenStop = false;
				if ( ! frame.loop.running ) return;
				frame.loop.running = false;
				animation?.stop();
				// eslint-disable-next-line ponlai/scene-clock -- the timeout of the wait for the sky's bakes, not a scene state
				const until = performance.now() + 2000;
				// eslint-disable-next-line ponlai/scene-clock -- the timeout of the wait for the sky's bakes, not a scene state
				while ( sky.busy && performance.now() < until ) await new Promise( ( resolve ) => setTimeout( resolve, 10 ) );
				await gpuQueue.onSubmittedWorkDone();

			},
			start(): void {

				hiddenStop = false;
				if ( frame.loop.running ) return;
				frame.loop.running = true;
				steeredLast = false;
				renderEvent( 'loop start' );
				clock.reset();
				animation?.start();

			},
		},
		step,
		dispose() {

			animation?.stop();
			renderer.setAnimationLoop( null );
			frame.loop.running = false;
			for ( const remove of removers ) remove();
			removers.length = 0;

		},
	};
	publish( frame );
	const setup = options.setup?.( frame ) ?? {};
	gpuSample = setup.gpuSample ?? null;
	// High remembers the scale it held (DynamicResolution.ts), except in an automated browser: a
	// profile or capture would start where the last run on the same origin ended, so its start-up and
	// first decision would depend on run history. `drsStore=on` turns it on there (a returning load's
	// test), `off` turns it off anywhere (neither restored nor saved: an A/B between modes or builds
	// starts both sides from the nominal scale).
	const drsStoreParam = params.get( 'drsStore' );
	internal.scaleMemory = drsStoreParam === 'on' || ( drsStoreParam !== 'off' && ! host.automated );

	// a resize of the canvas cuts the history; resize() itself also serves render-scale changes, which do not
	removers.push( host.onResize( () => {

		renderEvent( 'resize' );
		if ( options.resize ) options.resize( sizing.resize );
		else sizing.resize();

	} ) );
	// (a caller that sized the canvas before the haze's bake keeps that size)
	if ( ! sizing.sized ) sizing.resize();
	else sizing.refreshStoreKey();
	// High starts from the scale it last held on this display, tier and frame rate (remembered in the
	// browser, unless `scaleMemory` is off), and settles from there once its first frames are in; a
	// load that measures the GPU for its tier measures at the nominal scale.
	if ( dynamicResolution && setup.restore !== false ) {

		dynamicResolution.restore();
		sizing.applyRenderScale();

	}

	if ( host.timerDocument ) clock.connect( host.timerDocument );
	// The frame's stages (RenderStages.ts `createFrameStages`), which the tick calls in turn. The frame
	// gives them what only it knows: the caller's switches, the pacer's interval, the caller's pending
	// reasons and whether the loading screen still covers the canvas.
	let revealed = false;
	stages = createFrameStages( render, {
		weatherEffects: options.weatherEffects,
		environmentLook: options.environmentLook,
		interval: () => pacer.target > 0 ? 1000 / pacer.target : 1000 / 60,
		pending( reasons ) {

			content.pending?.( reasons );

		},
		shown() {

			// the loading screen covers the canvas until its fade has ended
			if ( ! revealed ) revealed = host.loadingGone();
			return revealed;

		},
	} );
	content = options.content?.( frame ) ?? {};
	const frames = stages;
	const { diagnostics } = options;

	function tick( time: number ): void {

		// the frame's own vblank time, not the moment the tick starts: the main thread reaches the tick a
		// few milliseconds into the frame, by a varying amount, which made the scene's steps uneven
		clock.update( time );
		// a jump's explicit bake in flight: no frame, and the clock holds
		if ( frames.held() ) return;
		content.start?.( time );
		// The clock never runs backwards. Timer.reset() restarts it at performance.now(): `loop.start()`
		// does so, and so does the timer itself when the page shows again (`connect( document )`). The
		// next animation frame's timestamp (its vblank) can be earlier than that moment, which made its
		// delta negative (DayCycle.advance rejects it); that frame counts as the restart's first, with a
		// delta of 0, whichever reset came before it.
		const rawDt = Math.max( clock.getDelta(), 0 );
		frameTimes.push( rawDt * 1000 );
		if ( frameTimes.length > 120 ) frameTimes.shift();
		const dt = Math.min( rawDt, 1 / 20 );
		// Lighting cadence follows wall time even below 20 FPS. Bound a resumed/stalled frame;
		// the scheduler still submits at most one coalesced bake instead of catching up a queue.
		const environmentDt = Math.min( rawDt, 1 );
		content.beforeClock?.();
		frames.advanceClock( dt );
		const clockDt = content.dayStep ? content.dayStep( rawDt ) : rawDt;
		content.beforeDay?.();
		frames.advanceDay( clockDt, content.hour ?? null );
		content.afterDay?.( time, dt, rawDt );

		const nightAmbient = frames.trackSky( environmentDt );
		frames.lightNight( content.localLights ? content.localLights() : 0, nightAmbient );
		content.afterNight?.( nightAmbient );
		frames.advanceWeatherAndSea( dt, environmentDt );
		content.afterWeather?.( dt );
		frames.relight();
		content.afterRelight?.( dt, rawDt );

		frames.draw( rawDt, dt );
		steerRenderScale( rawDt );
		frames.submit();
		// a paused page in the test mode keeps at most PAUSED_QUEUE frames queued on the GPU
		// (FramePacer.ts, "A paused page's queue")
		const bounded = render.paused && TEST_MODE;
		pacer.queueLimit = bounded ? PAUSED_QUEUE : 0;
		if ( bounded ) pacer.submitted( gpuQueue.onSubmittedWorkDone() );
		frameCount ++;
		diagnostics();

	}

	// After the frame's passes: read the GPU times back and move the dynamic scale (it applies from the
	// next frame on). Paused or pinned, the window keeps no samples and the scale holds.
	function steerRenderScale( rawDt: number ): void {

		if ( ! dynamicResolution ) return;
		if ( internal.storeKeyStale() ) sizing.refreshStoreKey();
		// the GPU's budget follows the paced frame interval (a 60 Hz frame unpaced)
		dynamicResolution.setFrameInterval( pacer.rate > 0 ? pacer.interval : 1000 / 60 );
		const steering = sizing.steppedScale === null && ! render.paused && ! resolution.full && sizing.renderScale === dynamicResolution.scale;
		frameScales.set( renderer.info.frame, { scale: steering ? sizing.renderScale : NaN, heavy: frames.periodicWork() } );
		if ( frameScales.size > 240 ) frameScales.delete( frameScales.keys().next().value! );
		gpuTimer?.resolve();
		if ( steering ) {

			// an interval since a frame that did not steer (or a loop start) spans the break
			if ( steeredLast ) dynamicResolution.sampleInterval( rawDt * 1000 );
			else dynamicResolution.resume();
			dynamicResolution.update();

		}
		steeredLast = steering;
		sizing.applyRenderScale();
		if ( frameCount % 30 === 0 ) sizing.describeRenderScale();

	}

	/** Starts the loop, through `wrap` (a load that measures its GPU) when given. */
	( frame as unknown as { startLoop( wrap?: ( body: ( time: number ) => void ) => ( time: number ) => void ): void } ).startLoop = ( wrap ) => {

		const animationLoop = wrap ? wrap( frame.loopBody ) : frame.loopBody;
		dynamicResolution?.begin();
		// three's loop asks the pacer for its animation frames, so the frames it skips never reach it
		animation = threeAnimation( renderer );
		animation.setContext( pacer.context );
		renderer.setAnimationLoop( animationLoop );
		frame.loop.running = true;
		renderEvent( 'loop start' );
		removers.push( host.onVisibility( ( hidden ) => {

			if ( hidden ) {

				if ( ! frame.loop.running ) return;
				void frame.loop.stop();
				hiddenStop = true;

			} else {

				renderEvent( 'visible' );
				if ( ! hiddenStop ) return;
				frame.loop.start();
				// its first frames (cloud history, a whole bake) do not steer the scale
				dynamicResolution?.begin();

			}

		} ) );

	};

	async function step( stepOptions: StepOptions ): Promise<StepResult> {

		const { frames: count, dt = 1 / 60, scale, camera: moveCamera, onFrame, clock: clockMode, converge } = stepOptions;
		if ( ! Number.isInteger( count ) || count < 0 ) throw new RangeError( 'frames must be a whole number, 0 or more' );
		if ( ! Number.isFinite( dt ) || dt <= 0 ) throw new RangeError( 'dt must be finite and positive' );
		const scales = scale === undefined ? [] : Array.isArray( scale ) ? scale : [ scale ];
		if ( scales.some( ( s ) => ! Number.isFinite( s ) || s <= 0 ) ) throw new RangeError( 'scale must be positive' );
		await frame.loop.stop();
		if ( clockMode !== undefined ) {

			render.paused = clockMode === 'paused';
			dayCycle.paused = render.paused;

		}
		const bookkeeping = frameBookkeeping( renderer );
		const timer = clock as unknown as { update( time?: number ): unknown; getDelta(): number };
		timer.update = () => timer;
		timer.getDelta = () => dt;
		const skyBake = sky as unknown as { slices: unknown[]; sliceTimer: ReturnType<typeof setTimeout> | null; pendingUpdate: Promise<void> | null };
		// the sky irradiance readbacks in flight, if any, have landed (each resolves once the GPU
		// answers; a map is not expected to take seconds, so a hang is reported rather than waited out)
		const settleSkyReadback = async (): Promise<void> => {

			// eslint-disable-next-line ponlai/scene-clock -- the timeout of the wait for the irradiance readback, not a scene state
			const begun = performance.now();
			while ( sky.readbacks > 0 ) {

				// eslint-disable-next-line ponlai/scene-clock -- the timeout of the wait for the irradiance readback, not a scene state
				if ( performance.now() - begun > 10000 ) throw new Error( 'loop.step: the sky irradiance readback did not land within 10 s' );
				await new Promise( ( resolve ) => setTimeout( resolve, 0 ) );

			}

		};
		await settleSkyReadback();
		frames.restartCadences();
		// eslint-disable-next-line ponlai/scene-clock -- a stepped clip's base timestamp: its frames' deltas are exactly dt (see the header)
		const t0 = Math.round( performance.now() );
		try {

			// one tick of the loop at `time`, with the bookkeeping three's animation loop does around it
			const tickAt = async ( time: number ): Promise<void> => {

				// an explicit bake in flight holds the loop (`bakeExplicitly`), so its tick would draw
				// nothing: start a retune's (`retuneSky`), wait for the bakes queued, and refuse a scene
				// jump still on its way to one
				render.retuneSky();
				await render.sunUpdates.catch( () => {} );
				if ( render.jumpsInFlight > 0 ) throw new Error( 'loop.step: a scene jump is in flight; await establishScene before stepping' );
				// three's Animation does this before each frame of its loop
				bookkeeping.begin();
				// (the frame's size follows `steppedScale` from this tick's start: applyRenderScale)
				sizing.applyRenderScale();
				frame.loopBody( time );
				bookkeeping.end();
				if ( skyBake.sliceTimer !== null ) {

					clearTimeout( skyBake.sliceTimer );
					skyBake.sliceTimer = null;

				}
				await settleSkyReadback();

			};
			for ( let i = 0; i < count; i ++ ) {

				if ( sky.busy && skyBake.slices.length === 0 ) await skyBake.pendingUpdate?.catch( () => {} );

				// never above the canvas's ratio: the render would outgrow the canvas it is shown on
				const ratio = scales.length ? scales[ Math.min( i, scales.length - 1 ) ] : sizing.nominalScale;
				sizing.steppedScale = Math.min( ratio, renderer.getPixelRatio() );
				moveCamera?.( i );
				await tickAt( t0 + i * dt * 1000 );
				const temporal = pipeline.resolver;
				if ( temporal && converge?.( i ) ) {

					// hold the frame's scene still until the rest has converged on it
					const release = frames.holdStill();
					timer.getDelta = () => 0;
					try {

						for ( let k = 0; k < temporal.rest.target + 4 && ! temporal.converged; k ++ ) {

							if ( sky.busy && skyBake.slices.length === 0 ) await skyBake.pendingUpdate?.catch( () => {} );
							await tickAt( t0 + i * dt * 1000 );

						}

					} finally {

						timer.getDelta = () => dt;
						release();

					}
					if ( ! temporal.converged ) console.warn( `loop.step: frame ${ i } did not converge (the rest's ${ temporal.rest.frames } of ${ temporal.rest.target } frames)` );

				}
				await onFrame?.( i );

			}

		} finally {

			delete ( timer as Partial<typeof timer> ).update;
			delete ( timer as Partial<typeof timer> ).getDelta;
			sizing.steppedScale = null;
			sizing.applyRenderScale();
			diagnostics( true );

		}
		return { frames: count, t0, simTime: render.simTime, frameId: renderer.info.frame };

	}

	return frame;

}

/** Starts the frame's loop, through `wrap` when given (a load that measures its GPU). */
export function startSceneLoop( frame: SceneFrame, wrap?: ( body: ( time: number ) => void ) => ( time: number ) => void ): void {

	( frame as unknown as { startLoop( wrap?: ( body: ( time: number ) => void ) => ( time: number ) => void ): void } ).startLoop( wrap );

}
