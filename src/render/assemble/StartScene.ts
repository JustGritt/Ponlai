// The render pipeline's assembly: builds a scene from its stages and the caller's content, and runs it.
//
// The render pipeline's assembly: `startScene( host, options, content )` builds a scene and starts
// its frame loop. It calls the render pipeline's stages (RenderStages.ts) in the one order they run
// in, and the caller's content (`SceneContent`) in the slots between them; it makes the frame
// (SceneFrame.ts), the scene jump (SceneJump.ts) and, on debug pages, the diagnostics and handles
// the tools read (SceneHandles.ts). The island's page (app/main.ts) and the render pipeline's test
// scenes (app/scenes/ScenePage.ts) are both assembled by it.
// - The order is part of the image: the frame graph's build order and the order materials first
//   compile in decide a few bytes of the frame (docs/reference/three-behaviour.md, Build order). The
//   stages run in this order and no other, and a slot runs where the caller's statements stood
//   among them; new content goes in the slot its statements belong to.
// - It reads the installed settings (Settings.ts; the page installs them before the render pipeline
//   loads) and their tier, the look the caller picked, and the scene's defaults (the hour and day
//   length where the address names none).
// - It reads no browser global (ARCHITECTURE.md, rule 13): the window, the loading screen and the
//   page's globals come through the host (SceneHost.ts; in a browser, render/browser.ts's
//   `browserHost`).
// - One page holds one render pipeline at a time (the modules' state: `shared`, the settings, the
//   local lights' table): a second `startScene` follows the first's `dispose()`, which puts the
//   modules back as the page loaded them (SceneDispose.ts), so the second draws as a fresh load
//   does.
// - The tools' contract (docs/bindings/writing-tools-and-checks.md, Build) is kept by every page it
//   assembles: on debug pages the handles and the diagnostics globals with `settled`, the loading
//   screen lifted once the scene shows, `establishScene` and `loop.step`.
import * as THREE from 'three/webgpu';
import { awaitWarmUp, bakeWithHaze, checkMotion, createAmbient, createBounceAndShore, createEnvironment, createKey, createLight, createRenderer, createScenes, createSeaAndPipeline, createSky, createWater, primeSharedBuilds, publishSkyLight, sceneOptions, setUpRenderer, startWaveMaps, type RenderParts, type SceneDefaults, type SceneOptions } from '../RenderStages.ts';
import { settings as params, type SettingsQuery } from '../Settings.ts';
import { loadTier, qualityProfile } from '../Quality.ts';
import type { Look } from '../look/Look.ts';
import { createShared, installShared, type Shared } from '../shared.ts';
import { releaseScene, startingScene } from '../SceneDispose.ts';
import { setAssetBase } from '../FetchAsset.ts';
import type { ExtraPass } from '../post/ExtraPass.ts';
import type { PipelineCount } from '../debug/PipelineCounter.ts';
import { createSceneFrame, createSceneSizing, startSceneLoop, type SceneFrame, type SceneFrameContent, type SceneFrameSetup, type SceneSizing } from './SceneFrame.ts';
import { createSceneJump, type SceneJump, type SceneJumpContent } from './SceneJump.ts';
import { DEFAULT_GLOBALS, createSceneDiagnostics, exposeSceneHandles, type SceneGlobals, type SceneHandlesContent } from './SceneHandles.ts';
import type { SceneHost } from './SceneHost.ts';

export interface StartOptions {
	/** the look's tables the scene is drawn with (look/Look.ts: the render pipeline's NEUTRAL_LOOK, or an app's own) */
	look: Look;
	/** the scene a query names where it names no hour or day length; `parkedWeather` defaults to the debug page's */
	defaults: { hour: number; dayLength: number; parkedWeather?: boolean };
	/** the address the assets are served from (FetchAsset.ts; the build's base by default) */
	assetBase?: string;
	/** whether high's render scale follows the GPU's frame time (SceneFrame.ts; true by default) */
	dynamicScale?: boolean;
	/** the globals' names on debug pages (SceneHandles.ts `DEFAULT_GLOBALS`) */
	globals?: Partial<SceneGlobals>;
}

/** What the caller's hooks are given: the scene as far as it is built. */
export interface SceneContext {
	readonly render: RenderParts;
	readonly shared: Shared;
	readonly host: SceneHost;
	readonly look: Look;
	/** a debug page: the dev server, `?debug` or `?profile` */
	readonly DEBUG: boolean;
	/** the scene a query names, with the app's defaults: the load's and a scene jump's */
	sceneOptions( query: SettingsQuery ): SceneOptions;
	/** the canvas's size and the render scale (from `afterWater` on) */
	readonly sizing: SceneSizing;
	/** the frame (from `frame` on) */
	readonly frame: SceneFrame;
	/** the pipelines created so far (debug pages) */
	readonly pipelineCount: PipelineCount | null;
}

/** The ground the scene stands on, for the ground's bounce and the shore (RenderStages.ts `createBounceAndShore`). */
export type SceneGround = Parameters<typeof createBounceAndShore>[ 1 ];
/** The ambient light's input (RenderStages.ts `createAmbient`). */
export type SceneAmbient = Parameters<typeof createAmbient>[ 1 ];
/** The water's input (RenderStages.ts `createWater`). */
export type SceneWater = Parameters<typeof createWater>[ 1 ];

type Step = Promise<void> | void;

/** The caller's content, by slot; every slot runs once, in this order, between the render pipeline's stages. */
export interface SceneContent {
	/** after the scenes and the camera exist */
	afterScenes?( scene: SceneContext ): Step;
	/** after the sky: the camera's first pose */
	afterSky?( scene: SceneContext ): Step;
	/** after the light */
	afterLight?( scene: SceneContext ): Step;
	/** after the key (the sun's or the moon's light and its shadow) */
	afterKey?( scene: SceneContext ): Step;
	/** after the environment's first bake */
	afterEnvironment?( scene: SceneContext ): Step;
	/** the rain's ceiling (m) and the passes the caller adds to the frame graph */
	passes( scene: SceneContext ): { rainCeiling: number; extraPasses: readonly ExtraPass[] };
	/** the ground, once the sea and the frame graph exist */
	ground( scene: SceneContext ): Promise<SceneGround> | SceneGround;
	/** after the ground's bounce and the shore: the world's objects */
	afterBounce?( scene: SceneContext ): Step;
	/** the ambient light's input, once the wave maps run */
	ambient( scene: SceneContext ): Promise<SceneAmbient> | SceneAmbient;
	/** after the ambient light: the plants, the figures */
	afterAmbient?( scene: SceneContext ): Step;
	/** the water's input */
	water( scene: SceneContext ): SceneWater;
	/** after the water: what reads the canvas's size (`scene.sizing`), the GUI */
	afterWater?( scene: SceneContext ): Step;
	/** after the haze's bake */
	afterHaze?( scene: SceneContext ): Step;
	/** the frame, asked for after the haze's bake: its switches, its setup, the resize, the caller's
	 * part of each frame */
	frame?( scene: SceneContext ): SceneFrameSpec;
	/** after the frame's draws were precompiled (the warm-up) */
	afterWarmUp?( scene: SceneContext ): Step;
	/** after the shared builds are primed, before the loop starts */
	afterShared?( scene: SceneContext ): Step;
	/** wraps the loop's body (a load that measures its GPU) */
	wrapLoop?( body: ( time: number ) => void ): ( time: number ) => void;
	/** whether the loading screen still waits, `until` the time (ms, performance.now()) it lifts anyway;
	 * by default until the sun's map has drawn and the ambient light's first relight is in */
	revealWait?( scene: SceneContext, until: number ): boolean;
	/** once the loading screen lifts */
	onRevealed?( scene: SceneContext ): void;
	/** the caller's part of the scene jump, asked for once the loop runs */
	jump?( scene: SceneContext ): SceneJumpContent;
	/** the caller's diagnostics and handles, asked for after the haze's bake */
	handles?( scene: SceneContext ): SceneHandlesContent;
}

/** The caller's frame (SceneFrame.ts): what the frame's stages read and the caller's part of it. */
export interface SceneFrameSpec {
	/** the switches for the rain and the wet surfaces (both on by default) */
	weatherEffects?: { rain: boolean; wetness: boolean };
	/** whether the exposure follows the scene (on by default) */
	environmentLook?: { exposure: boolean };
	/** after the pacer and the timers are made, before the first resize */
	setup?( frame: SceneFrame ): SceneFrameSetup | void;
	/** wraps the canvas's resize (`apply`) on a change of the window's size */
	resize?( apply: () => void ): void;
	/** the caller's part of each frame, made once the frame's stages exist */
	content?( frame: SceneFrame ): SceneFrameContent;
}

export interface RunningScene extends SceneContext {
	readonly jump: SceneJump;
	/** stops the loop and the scene's listeners and releases the render pipeline */
	dispose(): Promise<void>;
}

/** Builds the scene and starts its frame loop. */
export async function startScene( host: SceneHost, options: StartOptions, content: SceneContent ): Promise<RunningScene> {

	const globals: SceneGlobals = { ...DEFAULT_GLOBALS, ...options.globals };
	if ( options.assetBase !== undefined ) setAssetBase( options.assetBase );
	const { look } = options;
	startingScene();
	// the scene's uniforms, which every material reads: made from the caller's look and installed
	// before anything builds
	const shared = createShared( look );
	installShared( shared );
	const render = {} as RenderParts;

	// GPU time per pass, tick time and startup marks for tools/render/profile-frame.mjs (?profile, or
	// ?profile=frames without the GPU timestamps). The profiler loads only then; without it the
	// renderer, its pipelines and the image are unchanged.
	const profiler = params.has( 'profile' ) ? ( await import( '../debug/FrameProfiler.ts' ) ).createFrameProfiler( THREE.InspectorBase, { timestamps: params.get( 'profile' ) !== 'frames' } ) : null;
	if ( profiler ) host.publish( globals.profile, profiler.api );
	// scene handles and frame diagnostics on the page, for the capture tool and the console: dev
	// server, ?debug or ?profile only
	const DEBUG = !! import.meta.env?.DEV || params.has( 'debug' ) || profiler !== null;
	// The quality tier (Quality.ts): its load-time settings are read as the parts they shape are
	// built, its runtime ones applied by applySkyQuality. The app chose it with the settings, since
	// the anti-aliasing mode and other load-time settings are read as the modules load; an app that
	// installed none builds the `quality` option's, else high (`loadTier`, as those modules read it).
	const qualityTier = loadTier();
	const quality = qualityProfile( qualityTier );
	await createRenderer( render, { profiler, mark: ( name ) => profiler?.mark( name ), DEBUG, qualityTier, quality, setStatus: ( s ) => host.status( s ) } );
	setUpRenderer( render );
	// every render and compute pipeline the render pipeline creates, counted for the diagnostics
	// (debug only)
	const pipelineCount = DEBUG ? ( await import( '../debug/PipelineCounter.ts' ) ).installPipelineCounter( render.renderer ) : null;

	let sizing: SceneSizing | null = null;
	let frame: SceneFrame | null = null;
	const defaults: SceneDefaults = { hour: options.defaults.hour, dayLength: options.defaults.dayLength, parkedWeather: options.defaults.parkedWeather ?? DEBUG };
	const scene: SceneContext = {
		render, shared, host, look, DEBUG, pipelineCount,
		sceneOptions: ( query ) => sceneOptions( query, defaults ),
		get sizing() { if ( ! sizing ) throw new Error( 'startScene: the canvas is sized from afterWater on' ); return sizing; },
		get frame() { if ( ! frame ) throw new Error( 'startScene: the frame exists from its content on' ); return frame; },
	};

	// where the canvas goes and the size it starts at; `dpr`: the app's pixel ratio, pinned
	// (captures), over the display's and the tier's
	const size = host.size();
	createScenes( render, { viewport: { host: host.canvasHost, width: size.width, height: size.height }, dpr: params.has( 'dpr' ) ? Number( params.get( 'dpr' ) ) : null } );
	await content.afterScenes?.( scene );
	createSky( render, { scene: scene.sceneOptions( params ), seaFog: params.has( 'seaFog' ), dayFog: params.has( 'dayFog' ), look } );
	await content.afterSky?.( scene );
	createLight( render, { look } );
	await content.afterLight?.( scene );
	createKey( render );
	await content.afterKey?.( scene );
	await createEnvironment( render );
	publishSkyLight( render );
	await content.afterEnvironment?.( scene );
	const { rainCeiling, extraPasses } = content.passes( scene );
	createSeaAndPipeline( render, { rainCeiling, extraPasses, look } );
	// the clip tools' stepped clip, which the exposure follows (post/Pipeline.ts)
	render.pipeline.clipping = () => host.hasGlobal( globals.clip );
	createBounceAndShore( render, await content.ground( scene ) );
	await content.afterBounce?.( scene );
	startWaveMaps( render );
	await createAmbient( render, await content.ambient( scene ) );
	await content.afterAmbient?.( scene );
	createWater( render, content.water( scene ) );
	sizing = createSceneSizing( render, host, () => frame );
	await content.afterWater?.( scene );
	await bakeWithHaze( render );
	await content.afterHaze?.( scene );

	const spec = content.frame?.( scene ) ?? {};
	const handles = content.handles?.( scene ) ?? {};
	const diagnostics = createSceneDiagnostics( render, shared, host, globals, () => frame!, pipelineCount, handles );
	createSceneFrame( render, host, sizing, {
		dynamicScale: options.dynamicScale ?? true,
		weatherEffects: spec.weatherEffects ?? { rain: true, wetness: true },
		environmentLook: spec.environmentLook ?? { exposure: true },
		setup: spec.setup,
		resize: spec.resize,
		content: spec.content,
		diagnostics,
	}, ( made ) => { frame = made; } );
	const running = frame as unknown as SceneFrame;
	await awaitWarmUp( render );
	checkMotion( render );
	await content.afterWarmUp?.( scene );
	primeSharedBuilds( render );
	await content.afterShared?.( scene );

	render.mark( 'loop' );
	startSceneLoop( running, content.wrapLoop );
	// The loading screen lifts once the scene shows what it will show (by default: the sun's map has
	// drawn once its casters' pipelines exist, a few frames in, and the ambient light's first relight
	// is in, once the sky has been projected), at most REVEAL_WAIT ms on; a paused page lifts it
	// without its fade, so a capture never sees through it.
	const REVEAL_WAIT = 3000;
	// eslint-disable-next-line ponlai/scene-clock -- the loading screen's longest wait: when it lifts, not what is drawn
	const revealAt = performance.now() + REVEAL_WAIT;
	const waits = content.revealWait ?? ( ( _: SceneContext, until: number ) => ( render.sunShadow.draws === 0 || ! render.ambientRelight.ready ) &&
		// eslint-disable-next-line ponlai/scene-clock -- the loading screen's longest wait: when it lifts, not what is drawn
		performance.now() < until );
	let revealTimer: ReturnType<typeof setTimeout> | null = null;
	const reveal = () => {

		revealTimer = null;
		if ( waits( scene, revealAt ) ) {

			revealTimer = setTimeout( reveal, 50 );
			return;

		}
		// A paused page lifts the loading screen at once, without its fade: the frame holds still, and
		// the captures and reviews of paused frames capture it as soon as it is ready, which for a
		// temporal rest is about 40 frames after a view's eval, well inside the second the fade takes.
		// A rest captured so showed up to 2 % of the loading screen's colour over the whole image (a
		// converged rest is the canvas alone: TemporalRest.ts).
		host.liftLoading( render.paused );
		content.onRevealed?.( scene );

	};
	reveal();

	const jump = createSceneJump( render, defaults, running.pacer, content.jump?.( scene ) ?? {} );
	exposeSceneHandles( render, shared, host, globals, running, jump, pipelineCount, handles );

	return Object.assign( Object.create( null ) as object, {
		render, shared, host, look, DEBUG, pipelineCount, jump,
		sceneOptions: scene.sceneOptions,
		sizing,
		frame: running,
		async dispose(): Promise<void> {

			if ( revealTimer !== null ) clearTimeout( revealTimer );
			await running.loop.stop();
			running.dispose();
			render.renderer.dispose();
			render.renderer.domElement.remove();
			releaseScene();

		},
	} ) as RunningScene;

}
