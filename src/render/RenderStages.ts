// The render pipeline's stages: the pipeline built a stage at a time, in the entry's order.
//
// The render pipeline's stages: the app's render pipeline built a stage at a time, in the order
// the entry (main.ts) calls them. Each stage is a plain function that puts what it builds on the
// one RenderParts object, which the later stages, the island's stages (IslandStages.ts) and the
// app read. The stages come in three groups: those that need no land (the render pipeline, the
// scenes, the sky, the sun's light, the environment, the sea's waves and the frame's passes), those
// the island's data builds (the ground's bounce and the shore's waves, the ambient light's probes,
// the water) and those that close the build (the haze's bake, the warm-up, the shared builds).
// - The order of the statements inside a stage and of the stages is part of the rendered image: the order
//   materials first build in decides their shader code (docs/reference/three-behaviour.md, Build order). A new
//   statement goes where its build belongs, never at the end by habit.
// - A stage that ends in a wait returns its promise, which the entry awaits before the next stage:
//   a stage holds no await of its own, so the frames and microtasks between two stages stay the
//   entry's to order.
// - What the island gives a stage comes as an argument typed by the render pipeline's own types:
//   this file names none of the island's things.
// - The registered plant kinds are placed by a stage too (`placeVegetation`: the scene, the
//   wind's motion flag, the two depth prepasses), which closes their registry; the sun's caster box
//   and the rain's collision bake read the same registry (`vegetationCasters`,
//   `bakeRainCollision`), each called where the scene's order has it.
// The frame's stages (`createFrameStages`) are the render pipeline's work in a frame, cut where the
// scene's own work goes between: whether the frame is held, the scene clock, the day and the
// weather, the sky and its light, the local lights, the weather's effects with the sky's bake, the
// sun's maps and the sea, the relight, the draw and the submit. The frame's tick calls them in that
// order, once each (assemble/SceneFrame.ts), with an app's content between them (the island's:
// IslandStages.ts `createIslandFrame`).
// - The order of the statements inside a frame stage, and of the stages, is part of the rendered
//   image and of what a stepped clip reproduces: a statement reads what the ones before it wrote in the same
//   frame (the sky's state, the wind, the scene clock). New per-frame work goes into the stage
//   whose statements it follows, or into the scene's own between two stages.
// - What only the app knows comes through `FrameHost`: its switches, the pacer's interval, its
//   own reasons a still frame is not final, and whether the canvas shows.
// - The cadences that count wall-clock seconds (the sky's periodic bake, the refracted sun's map)
//   and the hold of a still scene are the stages' own state; a tool that steps the frames restarts
//   or holds them through `restartCadences` and `holdStill`.
import { settings as params, numberSetting, type SettingsQuery } from './Settings.ts';
import * as THREE from 'three/webgpu';
import { Sky } from './sky/Sky.ts';
import { createExposure, type ExposureTables } from './look/ExposureLook.ts';
import { createDayWindows, type DayWindowTable } from './look/DayLook.ts';
import { createAmbientTables, createKeyLift } from './look/AmbientLook.ts';
import { createHazeTables, type HazeTables } from './look/HazeLook.ts';
import { createSkyTables } from './look/SkyLook.ts';
import { createDisplayTables } from './look/DisplayLook.ts';
import type { Look } from './look/Look.ts';
import { DAY_PRESETS, type DayPreset, DayCycle, sunRising, duskForce, nightAmbientShare } from './sky/DayCycle.ts';
import { isParkedWeather, WEATHER_PRESETS, type WeatherName, isWindLevel, type WindLevel, Weather, type TunedField, type WeatherState, cloudedShare, type WindChoice } from './sky/Weather.ts';
import type { QualityProfile, QualityTier } from './Quality.ts';
import { Wind, weatherWind } from './weather/Wind.ts';
import { Rain, setRainShading } from './weather/Rain.ts';
import { OceanFFT } from './ocean/OceanFFT.ts';
import { seaState } from './ocean/OceanSpectrum.ts';
import { ShoreWaves } from './ocean/ShoreWaves.ts';
import { WaveMaps } from './ocean/WaveMaps.ts';
import { SeaDetail } from './ocean/SeaDetail.ts';
import { showUnderwater } from './ocean/Refraction.ts';
import { keyChroma, Water } from './ocean/Water.ts';
import { StormSea } from './ocean/StormSea.ts';
import { Pipeline } from './post/Pipeline.ts';
import { postOptionsFromURL } from './post/PostOptions.ts';
import { hazeKeyAt, type HazeKey } from './weather/HazeKeys.ts';
import { OUTPUT_PIXEL_RATIO_CAP, outputPixelRatio } from './post/RenderScale.ts';
import { SubmitBatch, type SubmitDevice } from './post/SubmitBatch.ts';
import { installCacheKeyPatches, installBindingPatches, primeSharedNodes } from './three/ThreePatches.ts';
import { checkRenderer } from './three/Internals.ts';
import { isWebGPU, gpuDevice } from './three/Backend.ts';
import { shared } from './shared.ts';
import { installOpenShadow } from './lighting/SolarShadow.ts';
import { createLocalLights, type LocalLights, type LocalLightsTables } from './lighting/LocalLights.ts';
import { SunShadow } from './lighting/SunShadow.ts';
import { RefractedSun } from './lighting/RefractedSun.ts';
import { installGroundBounce, GroundBounce, type BounceGround } from './lighting/GroundBounce.ts';
import { updateKeyLift, liftedKeyDirection, liftedKeyHeight } from './lighting/KeyLift.ts';
import { keyGlint } from './lighting/SurfaceBRDF.ts';
import { AmbientVolume, type AmbientView } from './lighting/AmbientVolume.ts';
import { installAmbientVolume, installEnvironmentHarmonics } from './lighting/AmbientSample.ts';
import { lowSunSkyLift, AmbientSky, AmbientRelight } from './lighting/AmbientRelight.ts';
import { type TuneOptions, tune, tuneApply } from './debug/Tune.ts';
import { markSkyRetune, skyRetune } from './sky/SkyRetune.ts';
import { TEMPORAL, AFTER_RESOLVE } from './post/Antialiasing.ts';
import type { ExtraPass, SceneLatents } from './post/ExtraPass.ts';
import type { AmbientLattice } from './lighting/ProbeFormat.ts';
import type { SubmergedSubset } from './lighting/CasterFormat.ts';
import type { Bounds, ShoreField } from './RenderTypes.ts';
import type { FrameProfiler } from './debug/FrameProfiler.ts';
import { motionUniforms, MOTION_CLASS, motionCheck } from './post/Motion.ts';
import { SceneHistory } from './post/SceneHistory.ts';
import { jitter } from './post/Jitter.ts';
import { preExposure, PRE_EXPOSURE_PENDING } from './post/PreExposure.ts';
import { renderEvent } from './post/SceneEvents.ts';
import { TemporalAA } from './post/TemporalAA.ts';
import { watchPipelines } from './post/Settled.ts';
import { MOON_GLINT_RADIUS, SCENE_RADIANCE_SCALE } from './lighting/Radiometry.ts';
import { SUN_GLINT_DISC } from './lighting/SunDisc.ts';
import { debugViewFromURL, installDebugView } from './post/graph/DebugView.ts';
import { createAmbientGrade, type AmbientGrade } from './lighting/AmbientGrade.ts';
import { updateLeafLight } from './vegetation/LeafLight.ts';
import { prepassScene, installVegetationPrepass } from './vegetation/VegetationPrepass.ts';
import { opaquePrepassScene, installOpaquePrepass } from './post/OpaquePrepass.ts';
import { PeriodicWork } from './post/DynamicResolution.ts';
import type { TemporalFrame } from './post/TemporalAA.ts';
import type { SettleFrame } from './post/Settled.ts';
import type { HazeFrame } from './weather/VolumetricHaze.ts';
import { installWindMotionFlag } from './vegetation/WindMotion.ts';
import { GTAO } from './post/GTAO.ts';
import { SS_SHADOW } from './post/MeadowShadows.ts';
import type { Vegetation } from './vegetation/Registry.ts';

/** A scene as the query names it: what a load and a scene jump both establish (`sceneOptions`). */
export interface SceneOptions {
	weather: WeatherName;
	wind: { level: WindLevel | undefined; heading: number | undefined };
	/** the weather evolves on its own (the query names neither the weather nor its wind) */
	autoWeather: boolean;
	day: { hour: number; dayLength: number; paused: boolean; moonPhase: number };
	preset: DayPreset | null;
	/** the scene clock held */
	paused: boolean;
	/** the scene clock (s) */
	time: number;
	/** the clouds' coverage, over the weather's (0..1), or null */
	clouds: number | null;
}

/** What a query leaves to the app: the day clock's hour and length (s) where it names none, and
 * whether a parked weather may load (Weather.ts `PARKED_WEATHER`: debug pages only). */
export interface SceneDefaults { hour: number; dayLength: number; parkedWeather: boolean }

/**
 * The scene a query's options set: the weather (held when one is asked for, else evolving on its
 * own), the day clock and its preset, the scene clock and the cloud override. The load reads its
 * own query with it, and a scene jump (`establishScene`) the query it is given, with the same
 * defaults, so the two agree. The moon (a full one by default) and the stars stand fixed in the
 * sky (DayCycle.ts MOON, STARS); the phase shapes only the drawn moon. A parked weather
 * (Weather.ts `PARKED_WEATHER`) is refused unless the defaults allow it and falls back to fair. The
 * wind (Weather.ts) takes `wind` (a level) and `windDir` (a heading, degrees in WeatherState's
 * convention), each over the sky's own wind; naming the weather or either of them holds the
 * weather, its wind included.
 */
export function sceneOptions( options: SettingsQuery, defaults: SceneDefaults ): SceneOptions {

	let weather = options.get( 'weather' ) ?? 'fair';
	const preset = options.get( 'preset' );
	if ( ! defaults.parkedWeather && isParkedWeather( weather ) ) {

		console.warn( `weather=${ weather } is parked (debug builds only, ?debug): showing fair` );
		weather = 'fair';

	}
	return {
		weather: Object.hasOwn( WEATHER_PRESETS, weather ) ? weather as WeatherName : 'fair',
		wind: {
			level: isWindLevel( options.get( 'wind' ) ) ? options.get( 'wind' ) as WindLevel : undefined,
			heading: options.has( 'windDir' ) && Number.isFinite( Number( options.get( 'windDir' ) ) ) ? Number( options.get( 'windDir' ) ) : undefined,
		},
		autoWeather: ! options.has( 'weather' ) && ! options.has( 'wind' ) && ! options.has( 'windDir' ),
		day: { hour: numberSetting( 'hour', defaults.hour, options ),
			dayLength: Math.max( 30, numberSetting( 'dayLength', defaults.dayLength, options ) ),
			paused: options.has( 'paused' ) || options.get( 'clock' ) === 'paused', moonPhase: numberSetting( 'moon', 0.5, options ) },
		preset: preset && Object.hasOwn( DAY_PRESETS, preset ) ? preset as DayPreset : null,
		paused: options.has( 'paused' ),
		time: Math.max( 0, numberSetting( 't', 0, options ) ),
		// Explicit cloud tuning is a review override; selecting a weather preset restores authored coverage.
		clouds: options.has( 'clouds' ) ? THREE.MathUtils.clamp( numberSetting( 'clouds', 0.32, options ), 0, 1 ) : null,
	};

}

/** What the app hands the render pipeline as it creates it (`createRenderer`), which the later stages
 * and the app read on RenderParts. */
export interface RenderHost {
	/** the frame profiler (?profile), loaded only then */
	profiler: FrameProfiler | null;
	/** marks a step of the load for the profiler (nothing without one) */
	mark: ( name: string ) => void;
	/** a debug page: the dev server, ?debug or ?profile */
	DEBUG: boolean;
	qualityTier: QualityTier;
	quality: Readonly<QualityProfile>;
	/** shows a line on the loading screen */
	setStatus: ( s: string ) => void;
}

export interface RenderParts extends RenderHost {
	gpuTiming: boolean;
	renderer: THREE.WebGPURenderer;
	submitBatch: SubmitBatch;
	DPR_CAP: number;
	resolution: { full: boolean; render: string; scale: string; gpu: string };
	scene: THREE.Scene;
	waterScene: THREE.Scene;
	rainScene: THREE.Scene;
	afterScene: THREE.Scene;
	camera: THREE.PerspectiveCamera;
	sky: Sky;
	dayCycle: DayCycle;
	weather: Weather;
	seaFogView: { on: boolean };
	dayFogView: { on: boolean };
	manualSun: boolean;
	paused: boolean;
	simTime: number;
	wind: Wind;
	/** the rain, made with the frame's passes (`createSeaAndPipeline`) */
	rain?: Rain;
	skyElapsed: number;
	/** the wall-clock seconds since the refracted sun's map last redrew (`createFrameStages`) */
	shadowElapsed: number;
	lastSkyTime: number;
	lastSkyHour: number;
	lastWeatherSignature: string;
	cloudOverride: number | null;
	skyQuality: { value: QualityTier };
	applySkyQuality: ( q?: Readonly<QualityProfile> ) => void;
	runtimeQuality: Readonly<QualityProfile>;
	lighting: { skyFill: number };
	applyEnvironmentFill: ( fill: number ) => void;
	sun: THREE.DirectionalLight;
	sunShadow: SunShadow;
	localLights: LocalLights;
	/** the ambient grade's controller, made from the look's tables (`createLight`) */
	ambientGrade: AmbientGrade;
	history: SceneHistory;
	invalidateSunShadow: () => void;
	shadowDirection: THREE.Vector3;
	sunUpdates: Promise<void>;
	prepareHaze: ( state: Readonly<WeatherState> ) => void;
	prepareSkyState: () => void;
	publishLighting: () => void;
	jumpsInFlight: number;
	bakeExplicitly: () => Promise<void>;
	retuneSky: () => boolean;
	refreshEnvironment: () => Promise<void>;
	applySun: () => Promise<void>;
	setHour: ( hour: number ) => Promise<void>;
	setPreset: ( preset: DayPreset ) => Promise<void>;
	setWeather: ( name: WeatherName, seconds?: number ) => Promise<void>;
	setWind: ( choice: WindChoice, seconds?: number ) => Promise<void>;
	applySkyFill: () => void;
	/** the sea's surface, made once the island's data exists (`createWater`) */
	water?: Water;
	fft: OceanFFT;
	waveMaps: WaveMaps;
	seaDetail: SeaDetail;
	fgview: string | null;
	pipeline: Pipeline;
	latents: SceneLatents;
	groundBounce: GroundBounce;
	shoreWaves: ShoreWaves;
	ambient: AmbientVolume;
	ambientRelight: AmbientRelight;
	ambientView: { mode: AmbientView };
	refractedSun: RefractedSun;
	stormSea: StormSea;
	hazeKey: HazeKey;
	motionPasses: () => { pass: string; scene: THREE.Scene; keeps: boolean }[];
}

/** The scene's fields the sea and the ground's bounce read, over the bounds of shared.shoreXform:
 * the shore's sea-level distance (m, negative over the land) and the sea floor's height (m). */
export interface SceneFields { shore: THREE.TextureNode; floor: THREE.TextureNode }

/** What lies under the sea for the refracted sun: the group it draws, and the triangles of each
 * mesh that lie under the water (the rest are left out). */
export interface SubmergedWorld { group: THREE.Group; subsets?: ReadonlyMap<THREE.Object3D, SubmergedSubset> }

/** What the render pipeline's stages hand one another and nothing else reads: on the same object, off the
 * RenderParts the island and the app see. */
interface StageParts {
	/** the scene the app loaded */
	loaded: SceneOptions;
	runtime: { refractedSun: RefractedSun | null };
	/** the haze's tables, made from the look (`createSky`), which `prepareHaze` reads */
	hazeTables: HazeTables;
	hazeReady: boolean;
	bakeEnvironment: () => Promise<void>;
	ambientSky: AmbientSky;
	overlayLatents: SceneLatents;
	/** the pipeline precompilation before the first frame, which `awaitWarmUp` waits for */
	warming: Promise<unknown>;
	hazeShadowCompiled: Promise<void>;
}

/** The render pipeline's parts as its own stages see them. */
function staged( render: RenderParts ): RenderParts & StageParts {

	return render as RenderParts & StageParts;

}

/** The render pipeline itself, with what the app hands it. Returns its init, which the entry awaits. */
export function createRenderer( render: RenderParts, host: RenderHost ) {

	const { profiler, mark, DEBUG, qualityTier, quality, setStatus } = host;
	Object.assign( render, { profiler, mark, DEBUG, qualityTier, quality, setStatus } );

	// ------------------------------------------------------------------ renderer

	// A dynamic render scale steers by the GPU's frame time, read from timestamp queries
	// (`gpuTimer=off` leaves them out: it then steers by the frame intervals, as where they are missing).
	const gpuTiming = ( profiler?.timestamps ?? false ) || ( 'dynamic' in quality.render && params.get( 'gpuTimer' ) !== 'off' );
	render.gpuTiming = gpuTiming;
	// eslint-disable-next-line ponlai/timestamp-owner -- the renderer is created here: the option only turns the queries on, and DynamicResolution.ts reads them
	const renderer = new THREE.WebGPURenderer( { antialias: false, powerPreference: 'high-performance', trackTimestamp: gpuTiming } );
	render.renderer = renderer;
	if ( profiler ) renderer.inspector = profiler;
	return renderer.init();

}

/** What must hold before any material builds: WebGPU, three's patches, the submit batching. */
export function setUpRenderer( render: RenderParts ) {

	const { mark, DEBUG, setStatus, renderer } = render;
	mark( 'renderer' );
	if ( ! isWebGPU( renderer ) ) {

		setStatus( 'This scene needs WebGPU (Chrome / Edge 113+, Safari 26+).' );
		throw new Error( 'WebGPU is not available' );

	}
	// the three.js internals of the renderer the app uses beyond its public interface: a three.js
	// without one stops the app here, naming it (three/Internals.ts)
	checkRenderer( renderer );
	// three r186 keys its node builds and pipelines by the first attachment, not the whole layout (an
	// MRT picks its outputs by the textures' names): patched before any material builds, so a material
	// drawn into two layouts gets a build each (three/ThreePatches.ts)
	installCacheKeyPatches( renderer );
	// a new render object's texture bindings start at their node's current texture, so a texture the
	// growth replaced and disposed is not uploaded again (three/ThreePatches.ts)
	installBindingPatches( renderer );
	// three submits every pass on its own; the frame reaches the GPU in a few submits instead
	// (SubmitBatch.ts: a submit per pass kept Chrome's GPU process from presenting the page on time)
	render.submitBatch = new SubmitBatch( gpuDevice<SubmitDevice>( renderer ), { debug: DEBUG } );
	// the settled signal counts the async pipeline compiles in flight (post/Settled.ts; debug only)
	if ( DEBUG ) watchPipelines( gpuDevice( renderer ) );

}

/** The canvas at its starting size, the scenes the frame draws and the camera. */
export function createScenes( render: RenderParts, { viewport, dpr }: { viewport: { host: HTMLElement; width: number; height: number }; dpr: number | null } ) {

	const { renderer } = render;
	// the table of what a cut or a scene jump clears, which every owner of a history joins
	render.history = new SceneHistory();
	// The scene renders at a pixel ratio set by the quality tier (see RenderScale.ts), at most the
	// display's and DPR_CAP (the canvas's cap: native 4K on a 2x display at high's maximum, or `dpr`,
	// the app's explicit ratio); the canvas matches the display and the frame graph upscales to it.
	// Explicit DPR captures and the full-resolution control lift the tier's pixel budget and pin a
	// dynamic scale at its nominal.
	render.DPR_CAP = dpr ?? OUTPUT_PIXEL_RATIO_CAP;
	render.resolution = { full: dpr !== null, render: '', scale: '', gpu: '–' };
	renderer.setPixelRatio( outputPixelRatio( 1 ) );
	renderer.setSize( viewport.width, viewport.height );
	renderer.shadowMap.enabled = true;
	renderer.shadowMap.type = THREE.PCFShadowMap;
	renderer.info.autoReset = false;
	viewport.host.appendChild( renderer.domElement );

	render.scene = new THREE.Scene();
	render.waterScene = new THREE.Scene();
	render.rainScene = new THREE.Scene();
	// under taa and taau, what draws after the resolve: the rain's streaks and the storm's spray
	// (post/AfterResolve.ts)
	render.afterScene = new THREE.Scene();

	// The far plane reaches past the horizon so the sea's skirt meets the sky (the resolved sky
	// depth, -1e5, and the passes' sky threshold, -9e4, stay beyond it).
	render.camera = new THREE.PerspectiveCamera( 27, viewport.width / viewport.height, 1, 60000 );

}

/**
 * The scene the app loads (`sceneOptions`), the sky, the day and the weather, the wind, the rain
 * and the sky's quality settings. `seaFog` and `dayFog` start the deep sea fog and the day's fog layer
 * on. `look`: the look's tables the exposure, the day's windows, the low key's lift and the sky's
 * tables are made from (look/ExposureLook.ts `createExposure`, look/DayLook.ts `createDayWindows`,
 * look/AmbientLook.ts `createKeyLift`, look/SkyLook.ts `createSkyTables`), which the sky carries to
 * the day, the grade, the frame's passes, the night's readers, the key's lifted direction and its
 * own state, bakes and the haze's key (`sky.exposure`, `sky.dayWindows`, `sky.keyLift`,
 * `sky.look`), and the haze's (look/HazeLook.ts `createHazeTables`), which the haze's key reads
 * by the hour and the weather (`createEnvironment`'s `prepareHaze`).
 */
export function createSky( render: RenderParts, { scene: loaded, seaFog, dayFog, look }: { scene: SceneOptions; seaFog: boolean; dayFog: boolean; look: ExposureTables & { day: DayWindowTable } & Pick<Look, 'ambient' | 'haze' | 'sky'> } ) {

	const { quality, renderer } = render;

	// ------------------------------------------------------------------ sky, light, water optics

	const exposure = createExposure( look );
	const dayWindows = createDayWindows( look.day );
	const keyLift = createKeyLift( look.ambient.keyLift );
	const sky = new Sky( renderer, { exposure, dayWindows, keyLift, look: createSkyTables( look.sky ), elevation: 32, azimuth: - 8 } );
	render.sky = sky;
	sky.setEnvironmentSource( quality.environmentSource );
	staged( render ).loaded = loaded;
	const dayCycle = new DayCycle( exposure, dayWindows, loaded.day );
	render.dayCycle = dayCycle;
	const weather = new Weather( loaded.weather, loaded.wind );
	render.weather = weather;
	// the haze's tables, which the haze's key eases by the hour and the weather (`prepareHaze`)
	staged( render ).hazeTables = createHazeTables( look.haze );
	// The deep sea fog (look/HazeLook.ts `seaFog`): a switch beside the weather, off unless
	// `seaFog` or the GUI turns it on, whatever the sky.
	render.seaFogView = { on: seaFog };
	// The day's fog layer under a clear or fair sky (weather/HazeKeys.ts `fogLayer`): a switch, off
	// unless `dayFog` or the GUI turns it on.
	render.dayFogView = { on: dayFog };
	weather.auto = loaded.autoWeather;
	// The day sky's level and exponent per clouded sky (Weather.ts `daySky`, `daySkyExponent`),
	// tunable on debug pages
	// without a reload: the presets the weather moves between take them, and the state that stands on
	// that sky follows (`setPresetValue`); the sky rebakes. A `?tune=` value is the start, applied
	// here as a load would.
	const PRESET_TUNABLES: ReadonlyArray<readonly [ TunedField, TuneOptions ]> = [
		[ 'daySky', { min: 0.005, max: 2, scale: 'log' } ],
		[ 'daySkyExponent', { min: 0, max: 3 } ],
	];
	for ( const name of [ 'cloudy', 'overcast' ] as const ) for ( const [ field, range ] of PRESET_TUNABLES ) {

		const value = tune( `sky.${ field }@${ name }`, WEATHER_PRESETS[ name ][ field ], { ...range, onChange: ( next ) => {

			weather.setPresetValue( name, field, next );
			markSkyRetune();

		} } );
		weather.setPresetValue( name, field, value() );

	}
	if ( loaded.preset ) dayCycle.setPreset( loaded.preset );
	render.manualSun = false;
	render.paused = loaded.paused;
	render.simTime = loaded.time;
	const wind = new Wind( shared.wind, { ...weatherWind( weather.state ), time: render.simTime } );
	render.wind = wind;
	// Across a cut of the temporal modes' history (Motion.ts) the vegetation's wind has no motion either:
	// its previous state becomes the current one.
	render.history.onCut( 'wind', () => wind.cut() );

}

/** The sky's runtime quality, the environment's fill, the sun's light and its shadow map. `look`:
 * the look's tables the local lights and the ambient grade take (lighting/LocalLights.ts
 * `createLocalLights`; look/AmbientLook.ts `createAmbientTables`, lighting/AmbientGrade.ts
 * `createAmbientGrade`, whose tables the ambient light's relight reads too: `createAmbient`). */
export function createLight( render: RenderParts, { look }: { look: LocalLightsTables & Pick<Look, 'ambient'> } ) {

	const { qualityTier, quality, scene, sky, dayCycle } = render;
	const { loaded } = staged( render );
	render.applySkyQuality = applySkyQuality;
	render.applyEnvironmentFill = applyEnvironmentFill;
	// the local lights' registry, which the island's kinds of light join (LocalLights.ts)
	render.localLights = createLocalLights( look );
	// the ambient grade's controller, which the frame writes into the lighting models' uniforms
	// (lighting/AmbientGrade.ts)
	render.ambientGrade = createAmbientGrade( createAmbientTables( look.ambient ) );

	render.skyElapsed = 0;
	render.lastSkyTime = render.simTime;
	render.lastSkyHour = dayCycle.hour;
	render.lastWeatherSignature = '';
	render.cloudOverride = loaded.clouds;
	render.skyQuality = { value: qualityTier };
	// The tier's runtime settings (a debug override may replace some of them: `s.applyQuality( { ... } )`).
	const runtime = { refractedSun: null as RefractedSun | null };
	staged( render ).runtime = runtime;
	function applySkyQuality( q: Readonly<QualityProfile> = quality ): void {

		sky.setCloudView( q.cloudView );
		sky.clouds.setDome( q.cloudDome.width, q.cloudDome.height, q.cloudDome.strideX, q.cloudDome.strideY );
		sky.clouds.setMarch( q.cloudMarch );
		sky.clouds.setShadowRefresh( q.cloudShadowRows );
		sky.setEnvironmentCadence( q.environmentEvery );
		render.rain?.setQuality( q.rainDrops );
		if ( runtime.refractedSun ) runtime.refractedSun.angleGate = THREE.MathUtils.degToRad( q.refractedSun.angleGate );
		render.runtimeQuality = q;
		renderEvent( 'apply quality' );

	}
	render.runtimeQuality = quality;
	applySkyQuality();
	scene.add( sky.dome, sky.lightning.group );
	render.lighting = { skyFill: 1 };
	// The environment's intensity: the fill times what takes the displayed environment's texels to
	// scene units (its atlases store the sky's radiance at a power of two, under the debug press:
	// Sky.ts `environmentRead`), so a fill of 1 lights as the sky stands. A bake that publishes after
	// the frame set it sets it again, before the frame renders (`onEnvironmentStore`).
	let environmentFill = 1;
	function applyEnvironmentFill( fill: number ): void {

		environmentFill = fill;
		scene.environmentIntensity = fill * sky.environmentRead;

	}
	applyEnvironmentFill( 1 );
	sky.onEnvironmentStore = () => applyEnvironmentFill( environmentFill );

	const sun = new THREE.DirectionalLight( 0xffffff, 1 );
	render.sun = sun;
	sun.castShadow = true;
	// (the water's refraction source draws the world under the sea lit by it: ocean/Refraction.ts)
	showUnderwater( sun );
	sun.shadow.mapSize.set( 2048, 2048 );
	// The map is fitted to the island's casters once the world exists, and redrawn in small steps of
	// the key light with a cross-fade (SunShadow). Its bias, 1 cm along the light ray, follows each
	// fit's depth range; receiver-plane correction in the shared filter keeps contacts sharp without
	// a wide blur. The normal offset is SunShadow's (in texels, growing with view depth): three's
	// own stays 0.
	sun.shadow.normalBias = 0;
	const sunShadow = new SunShadow( sun, { bias: 0.01 } );
	render.sunShadow = sunShadow;
	sunShadow.install();
	// Cloud shadowing shares the density field in the visible/reflected sky. Bake its long solar
	// path once at low resolution, then sample the same transmission on every receiver, including
	// those beyond the island's shadow map (the sea floor seen through the water).
	installOpenShadow( sun, ( position ) => sky.keyDirectVisibility( position ), sky.castShadow );

}

/** The sun in the scene, and the direction the sun's map and the figures' are drawn along. */
export function createKey( render: RenderParts ) {

	const { scene, sun, sunShadow } = render;
	render.invalidateSunShadow = invalidateSunShadow;
	// Vegetation casts its rest pose; wind only changes the visible mesh. The key light's steps
	// redraw the map; a jump of the light or an edit to the casters redraws it from scratch.
	function invalidateSunShadow(): void { sunShadow.invalidate(); }
	scene.add( sun, sun.target );
	// the direction the sun's map and the figures' are drawn along: the key's, lifted under a low key
	// (lighting/KeyLift.ts)
	render.shadowDirection = new THREE.Vector3( 0, 1, 0 );

}

/**
 * The sky's bakes and the lighting they publish, and the operations that change the hour, the
 * weather and the wind. Returns the first refresh, which the entry awaits.
 */
export function createEnvironment( render: RenderParts ) {

	const { setStatus, renderer, scene, sky, dayCycle, weather, seaFogView, dayFogView, lighting, applyEnvironmentFill, sun, invalidateSunShadow, shadowDirection } = render;
	const stage = staged( render );
	render.prepareHaze = prepareHaze;
	render.prepareSkyState = prepareSkyState;
	render.publishLighting = publishLighting;
	stage.bakeEnvironment = bakeEnvironment;
	render.bakeExplicitly = bakeExplicitly;
	render.retuneSky = retuneSky;
	render.refreshEnvironment = refreshEnvironment;
	render.applySun = applySun;
	render.applySkyFill = applySkyFill;

	function fresnel( cosI: number, eta: number ): number {

		const g = Math.sqrt( eta * eta - 1 + cosI * cosI );
		const a = ( g - cosI ) / ( g + cosI ), b = ( cosI * ( g + cosI ) - 1 ) / ( cosI * ( g - cosI ) + 1 );
		return 0.5 * a * a * ( b * b + 1 );

	}

	render.sunUpdates = Promise.resolve();
	// The haze by the hour and the weather (weather/HazeKeys.ts): its amount over the weather's, the
	// morning mist beside the weather's sea fog, its spectral extinction and its light, set on the
	// medium the frame and the environment's bake read (the bake lays it over its source: Sky.ts
	// `fogEnvironment`), so a bake sees the haze of the state it bakes.
	// (the first bake runs before the pipeline holds the haze: `hazeReady` below)
	stage.hazeReady = false;
	function prepareHaze( state: Readonly<WeatherState> ): void {

		if ( ! stage.hazeReady ) return;
		const rising = sunRising( sky.sunDir );
		// the parked fog weather's share of the sky, by the front's smoothstep as `skyMix` blends: exact
		// at rest and, for the sun's Rayleigh beam, on a front between the fog and the overcast sky, for
		// the fog layer's droplets on one between the fog and the cloudy or overcast sky
		// (weather/HazeKeys.ts `hazeKeyAt`)
		const progress = weather.progress, blend = progress * progress * ( 3 - 2 * progress );
		const fogWeather = ( weather.origin === 'fog' ? 1 - blend : 0 ) + ( weather.target === 'fog' ? blend : 0 );
		hazeKeyAt( stage.hazeTables, sky.dayWindows, sky.look, sky.elevation, rising, weather.skyMix, sky.weather.windSpeed, sky.weather.rain, dayFogView.on ? 1 : 0, seaFogView.on ? 1 : 0, fogWeather, render.hazeKey );
		render.pipeline.air.haze.value = sky.weather.haze * render.hazeKey.amount;
		render.pipeline.air.fog.value = Math.max( sky.weather.fog, render.hazeKey.mist );
		render.pipeline.air.setLook( render.hazeKey );
		// the air's sunlit haze takes the key the land does, through the clouds' shadow (its own read),
		// and its sky light the weather's sky as the sky takes it; the weather's day sky comes in with
		// the sun's light the sky lends the air (sky/SkyTrack.ts `skyGain`)
		render.pipeline.air.weatherLight.direct = state.directVisibility;
		render.pipeline.air.weatherLight.ambient = state.ambientGain;
		// the floor of the key under a closed deck (Sky.ts `deckKeyFloor`), which the seen panorama's
		// bake reads as the frame's air does: set with the medium, so a bake takes its own scene's
		render.pipeline.air.keyFloor.value = sky.deckKeyFloor;
		render.pipeline.air.prepareMedium();

	}

	function prepareSkyState(): void {

		const celestial = render.manualSun ? null : dayCycle.snapshot();
		const state = weather.snapshot();
		if ( render.cloudOverride !== null ) state.coverage = render.cloudOverride;
		sky.setWeather( state );
		keyGlint.spread.value = state.glintSpread ?? 0;
		sky.track( celestial, state, render.cloudOverride, weather.skyMix );
		prepareHaze( state );
		render.lastWeatherSignature = `${ state.coverage },${ state.density },${ state.directVisibility },${ state.ambientGain },${ state.daySky },${ state.daySkyExponent },${ state.haze }`;

	}

	function publishLighting(): void {

		// three's sun: the direct key, the sky's, which the clouds' shadow cuts per point over the land's
		// floor, the share of the key a cloud passes on (Sky.ts `keyDirectVisibility`, `look.keyThroughCloud`); the
		// land's key (shared.landKey) and the sea's (shared.sunColor): both the sky's key, the sea's cut
		// by the clouds' own transmittance over the land's floor (Sky.ts `seaKeyVisibility`)
		sun.color.copy( sky.keyColor );
		shared.landKey.value.set( sun.color.r, sun.color.g, sun.color.b );
		// a low sun's lift of the sky's light on the water: the sea's bounce onto the land, the water's
		// sky light and the sea's reflection (lighting/AmbientRelight.ts `lowSunSkyLift`)
		shared.skyLift.value = lowSunSkyLift( sky.elevation, cloudedShare( weather.skyMix ) );
		sun.intensity = 1;
		sun.position.copy( sky.keyDir ).multiplyScalar( 120 ).add( sun.target.position );
		shared.sunDir.value.copy( sky.keyDir );
		updateKeyLift( sky.keyLift, sky.keyDir, sky.keyIsMoon );
		keyGlint.radius.value = sky.keyIsMoon ? MOON_GLINT_RADIUS : SUN_GLINT_DISC;
		liftedKeyDirection( sky.keyLift, sky.keyDir, sky.keyIsMoon, shadowDirection );
		shared.sunColor.value.set( sun.color.r, sun.color.g, sun.color.b );
		keyChroma.value = sky.keyIsMoon ? 0 : 1;
		const cosI = Math.max( sky.keyDir.y, 0 );
		shared.sunIntoWater.value = sky.keyDir.y > 0 ? 1 - fresnel( cosI, 1.333 ) : 0;
		const eta = 1 / 1.333;
		shared.muSun.value = Math.sqrt( 1 - eta * eta * ( 1 - cosI * cosI ) );
		applySkyFill();

	}

	async function bakeEnvironment(): Promise<void> {

		// Explicit preset changes settle any in-flight bake first. The running clock uses a single
		// coalesced request below, never a growing async queue or a suspended animation loop.
		if ( sky.busy ) await sky.update();
		// Every explicit bake bakes as a fresh load's first bake does, the clouds at the scene clock (Sky.reset):
		// a paused page keeps its products, so the load, a jump, a seek, a new hour or weather and a
		// retune each make the ones a fresh load of the same scene makes, whatever the frames before
		// left in the sky. It takes every value as it stands (`retuneSky`).
		sky.reset( shared.wind.direction.value, render.simTime );
		skyRetune.dirty = false;
		if ( stage.hazeReady ) render.pipeline.air.noteBake();
		prepareSkyState();
		await sky.update();
		scene.environment = sky.envMap!.texture;
		( scene as unknown as { environmentNode: THREE.Node<'vec3'> | null } ).environmentNode = sky.environmentNode;
		publishLighting();
		invalidateSunShadow();
		render.lastSkyHour = dayCycle.hour;
		render.lastSkyTime = render.simTime;
		render.skyElapsed = 0;

	}

	// An explicit bake (a jump of the hour, the weather or the whole scene, a seek, a retune of what the
	// bake depends on) shows at once, its sky's light read back before it publishes (sky/SkyBake.ts `bake`).
	// The state it bakes changes before it is called and the readback takes a frame or two, so the loop
	// draws no frame from that change until the bake has published (`tick`): the frame on screen holds
	// the old state, and the first frame drawn after it shows the new weather or hour in its own light.
	// (a count: the requests queue on `sunUpdates`, and a scene jump holds from its first change.) A
	// held tick still advances three's frame count (`renderer.info.frame`, which probe.mjs `--frames`
	// counts as drawn frames), and the first frame drawn after a hold records the whole hold as one
	// interval in the frame statistics.
	render.jumpsInFlight = 0;
	function bakeExplicitly(): Promise<void> {

		render.jumpsInFlight ++;
		render.sunUpdates = render.sunUpdates.catch( ( error: unknown ) => console.error( error ) ).then( () => bakeEnvironment() )
			.finally( () => { render.jumpsInFlight --; } );
		return render.sunUpdates;

	}

	// A value the environment's bakes depend on has changed since they ran (sky/SkyRetune.ts: a
	// tunable's mark, or a haze setting the bake reads: VolumetricHaze.ts `bakeStale`): bake again
	// explicitly, as the load does, and raise the `retune` event. Run by the `tune` handle's `set` and
	// before each frame (`tick`, `loop.step`), so the GUI and scripts take the same path; true when it
	// started a bake.
	function retuneSky(): boolean {

		if ( stage.hazeReady && render.pipeline.air.bakeStale ) markSkyRetune();
		if ( ! skyRetune.dirty ) return false;
		skyRetune.dirty = false;
		renderEvent( 'retune' );
		void bakeExplicitly();
		return true;

	}

	function refreshEnvironment(): Promise<void> {

		render.manualSun = false;
		return bakeExplicitly();

	}

	function applySun(): Promise<void> {

		render.manualSun = true;
		return bakeExplicitly();

	}

	render.setHour = async ( hour: number ) => { dayCycle.setHour( hour ); renderEvent( 'set hour' ); await refreshEnvironment(); };
	render.setPreset = async ( preset: DayPreset ) => { dayCycle.setPreset( preset ); renderEvent( 'set preset' ); await refreshEnvironment(); };
	render.setWeather = async ( name: WeatherName, seconds = 20 ) => {

		render.cloudOverride = null;
		weather.auto = false;
		weather.setWeather( name, seconds );
		renderEvent( seconds === 0 ? 'weather jump' : 'weather front' );
		if ( seconds === 0 ) await refreshEnvironment();

	};
	// The wind alone (its level, its heading or both), the sky keeping its course; as setWeather, it
	// holds the weather.
	render.setWind = async ( choice: WindChoice, seconds = 20 ) => {

		weather.auto = false;
		weather.setWind( choice, seconds );
		renderEvent( seconds === 0 ? 'weather jump' : 'weather front' );
		if ( seconds === 0 ) await refreshEnvironment();

	};
	// (a whole scene at once, as a query loads it: `establishScene`, below the frame loop)

	function applySkyFill(): void {

		// One artistic fill gain drives PBR ambient, water scattering and sky reflections alike.
		applyEnvironmentFill( lighting.skyFill );
		shared.skyFill.value = lighting.skyFill;
		shared.skyIrradiance.value.set( sky.skyIrradiance.r, sky.skyIrradiance.g, sky.skyIrradiance.b ).multiplyScalar( lighting.skyFill );
		if ( render.water ) render.water.uniforms.reflection.value = lighting.skyFill;

	}



	// the sky's harmonics for the ambient light's relight (lighting/AmbientRelight.ts), projected from
	// every bake's environment source from the first on
	const ambientSky = new AmbientSky( renderer );
	stage.ambientSky = ambientSky;
	sky.onEnvironmentSource = ( source, atlases, read ) => ambientSky.project( source, atlases, read );

	setStatus( 'Lighting the sky…' );
	return refreshEnvironment();

}

/** The sky's light as the nodes every material reads. */
export function publishSkyLight( render: RenderParts ) {

	const { mark, sky } = render;
	mark( 'sky' );
	// (the sky's light on what it lights takes a low sun's lift, shared.ts `skyLift`; the sky drawn and
	// its meter never do)
	shared.skyIrradianceSource = sky.irradianceNode.mul( shared.skyFill ).mul( shared.skyLift );
	shared.skyFillIrradiance = sky.landIrradianceNode.mul( shared.skyFill );
	shared.skyMeterLog2 = sky.meterLog2Node;

}

/** The sea's waves and detail, then the frame's passes with the passes the entry adds. `look`: the
 * look's tables the sea takes, its detail's strengths and its distant swell (look/Look.ts
 * `SeaLook`), and the display's, which the frame's passes take (look/DisplayLook.ts). */
export function createSeaAndPipeline( render: RenderParts, { rainCeiling, extraPasses, look }: { rainCeiling: number; extraPasses: readonly ExtraPass[]; look: Pick<Look, 'sea' | 'display'> } ) {

	const { profiler, DEBUG, quality, renderer, scene, waterScene, rainScene, afterScene, camera, sky, wind, sun, sunShadow } = render;
	// tidewater's two wave systems at the prevailing heading (Wind.ts `spectrumWind`) and tidewater's
	// sea state for the wind (OceanSpectrum.ts `seaState`: its fetch, choppiness and swell), rebuilt
	// whenever the weather's wind speed or heading changes (`setWind`, below in the frame loop: the
	// wind sea turns with the heading, the swell keeps the prevailing one).
	const sea = seaState( wind.spectrumWind.windSpeed );
	const fft = new OceanFFT( renderer, {
		...wind.spectrumWind, fetch: sea.fetch, choppiness: sea.choppiness, distantSwell: { ...look.sea.swell, scale: sea.swell },
		anisotropy: quality.water.fftAnisotropy, subPixelShare: look.sea.subPixelSlope,
	} );
	// (a debug page's `?tune=` value applies here, before the first update builds the spectrum)
	tuneApply( 'sea.slopes.subPixelShare', look.sea.subPixelSlope, { min: 0, max: 0.9 }, ( share ) => fft.setSubPixelShare( share ) );
	render.fft = fft;
	const waveMaps = new WaveMaps( renderer, fft );
	render.waveMaps = waveMaps;
	// the sea's gusts, slicks and windrows (the water's roughness)
	render.seaDetail = new SeaDetail( shared, look.sea.detail );
	// the veil's shells and the sea's ripple layers are built into the rain's and the water's materials
	setRainShading( quality.rainShading );
	render.rain = new Rain( { count: render.runtimeQuality.rainDrops, ceiling: rainCeiling } );
	// Drops scatter the same linear environment and key light as the island, without an emissive
	// floor. Using the GPU integral keeps their light in step with the displayed sky transition.
	// A strike's fill lights the ground they show in full; the sky they show and reflect takes it
	// damped (like the storm spray's), since the displayed deck brightens far less than the fill.
	render.rain.setLightNode( shared.skyIrradianceSource, shared.sunDir, shared.sunColor, shared.skyIrradianceSource.div( shared.skyFill.max( 1 ).pow( 0.35 ) ) );
	// `fgview=<resource>[.<output>]` (debug builds): the frame graph's debug view shows one of its
	// resources on the canvas (packages/render/post/graph/DebugView.ts): only then it appends its pass to the graph
	// before the pipeline compiles it.
	const fgview = DEBUG ? debugViewFromURL() : null;
	render.fgview = fgview;
	if ( fgview ) installDebugView( fgview, {
		// (the targets hold scene radiance pre-exposed: PreExposure.ts)
		exposure: () => pipeline.settings.exposure / SCENE_RADIANCE_SCALE * preExposure.inverse.value,
		// (the resolve exists while the pipeline compiles its graph, before `pipeline` is assigned)
		temporal: () => TemporalAA.current,
		motion: {
			previousViewProjection: motionUniforms.previousViewProjection, projectionInverse: jitter.uniforms.unjitteredProjectionInverse,
			jitterNdc: jitter.uniforms.ndc, opaqueBelow: ( MOTION_CLASS.opaque + MOTION_CLASS.vegetation ) / 2,
		},
	} );
	const pipeline = new Pipeline( renderer, sky, createDisplayTables( look.display ), sunShadow, render.rain, postOptionsFromURL( params, quality.post ), quality.taa, extraPasses, quality.water.ssrIterations );
	render.pipeline = pipeline;
	// across a cut the sea's reflection reprojects nothing from the frames before
	render.history.onCut( 'sea reflection', () => pipeline.seaReflection.clearHistory() );
	// the pipeline's scene passes by their typed names, for what registers a draw for later
	render.latents = pipeline.graph;
	// The profiler labels each pass the graph runs by the pass's label (its groups), unless a
	// target or object it knows labels it otherwise (the passes inside the display chain).
	if ( profiler ) {

		const hooks = pipeline.graph.hooks, begin = hooks.onPassBegin, end = hooks.onPassEnd;
		hooks.onPassBegin = ( pass, f ) => {

			begin?.( pass, f );
			profiler.graphPass( pass.label );

		};
		hooks.onPassEnd = ( pass, f ) => {

			profiler.graphPass( null );
			end?.( pass, f );

		};

	}
	render.rain.bindOpaqueDepth( pipeline.depthRT.texture, pipeline.surfaceRT.texture );
	rainScene.add( render.rain.group );
	if ( render.rain.afterGroup ) afterScene.add( render.rain.afterGroup );
	// Under the ground truth the rain and the storm's spray draw in the overlay pass (rainScene,
	// Pipeline.ts), which keeps the motion vectors of the surfaces under them; under taa and taau they
	// draw after the resolve (afterScene, the `rainAfter` pass). Their owners register their latent
	// draws for those passes.
	const overlayLatents: SceneLatents = TEMPORAL
		? { latent: ( pass, objects, kernels ) => pipeline.graph.latent( pass === 'rainAfter' || ( pass === 'water' && AFTER_RESOLVE ) ? 'rainAfter' : 'overlay', objects, kernels ) }
		: pipeline.graph;
	staged( render ).overlayLatents = overlayLatents;

	// ------------------------------------------------------------------ first-use pipelines

	// A weather change, the first bolt and the sky's first periodic bake used to build their
	// pipelines in the frame they first drew (rain's drops and splashes, the bolt's two passes, the
	// sky's blends and filters, the fog's passes: a dropped frame each). They are precompiled
	// instead, through the async pipeline calls, so the GPU process builds them off the main thread
	// while the world is generated and built; the loop starts once they are ready. The frame graph's
	// warm-up compiles every pass the frame may run, those that run only at times included (fog, rain,
	// the view distance, either output), each under its own binding (its target and MRT are part of
	// its pipelines' keys), with the objects their owners register as drawn only at times; the sky
	// and the caustics' splats (drawn only while the sun lights them) compile their own renders. The
	// storm's foam and spray need the world's shore fields and register with it.
	const warming = Promise.all( [
		sky.warm( renderer ),
		waveMaps.warm(),
		// the drops and splashes, and the bolts, registered with the passes they draw in before the
		// graph's warm-up starts, which compiles them with every pass (graph/FrameGraph.ts `warm`)
		render.rain.registerLatent( overlayLatents ),
		sky.lightning.registerLatent( pipeline.graph ),
		pipeline.graph.warm( { scene, waterScene, rainScene, afterScene: AFTER_RESOLVE ? afterScene : undefined, camera, water: null, sun, time: render.simTime } ).then( () => pipeline.resolver?.warm() ),
	] );
	staged( render ).warming = warming;
	warming.catch( () => undefined ); // reported where it is awaited, before the loop

}

/** What the island's fields build first: the ground's bounce and the shore's waves. */
export function createBounceAndShore( render: RenderParts, { fields, bounceGround, shoreData }: { fields: SceneFields; bounceGround: BounceGround; shoreData: ShoreField } ) {

	const { renderer, sky, sunShadow, fft, waveMaps } = render;
	// the sunlight the ground bounces up (lighting/GroundBounce.ts), which every lit material adds: so
	// installed before any of them builds; its map shares the wave maps' target
	render.groundBounce = installGroundBounce( new GroundBounce( { renderer, shared, sunShadow, fields, ground: bounceGround, target: waveMaps.target, layer: waveMaps.bounceLayer, cloud: ( position ) => sky.keyDirectVisibility( position ) } ) );
	render.shoreWaves = new ShoreWaves( renderer, shoreData, shared, fields.shore, fft, fields.floor );

}

/** Starts the wave maps' bake in the frame's passes (post/PipelinePasses.ts), once the world's
 * meshes that read them are built. */
export function startWaveMaps( render: RenderParts ) {

	const { pipeline, waveMaps } = render;
	pipeline.waveMaps = waveMaps;

}

/**
 * The ambient light's probes and their relight, the refracted sun and the storm sea. Returns the
 * refracted sun's first map, which the entry awaits.
 */
export function createAmbient( render: RenderParts, { ambientData, world }: { ambientData: AmbientLattice; world: SubmergedWorld } ) {

	const { DEBUG, renderer, scene, sky, weather, waveMaps, pipeline } = render;
	const { runtime, ambientSky } = staged( render );
	// The ambient light's probes on the GPU (lighting/AmbientVolume.ts), relit every frame into their
	// irradiance (lighting/AmbientRelight.ts), which every lit island material reads for its ambient
	// diffuse (lighting/AmbientSample.ts: installed before any of them builds), and their debug view
	// (`ambientProbes=sky|bounce|irradiance` on a debug page, the GUI's Ground folder), built on first
	// show.
	const ambient = new AmbientVolume( renderer, ambientData );
	render.ambient = ambient;
	installAmbientVolume( ambient );
	const ambientRelight = new AmbientRelight( { renderer, volume: ambient, shared, sky: ambientSky, weights: () => sky.environmentWeights(), cloud: ( position ) => sky.keyDirectVisibility( position ), seaCloud: ( position ) => sky.cloudVisibility( position ), keyHeight: () => liftedKeyHeight( sky.keyLift, sky.keyDir, sky.keyIsMoon ), dayWindows: sky.dayWindows, grade: render.ambientGrade.tables, sunElevation: () => sky.elevation, rising: () => sky.rising, clouded: () => cloudedShare( weather.skyMix ), overcast: () => weather.skyMix[ 3 ] ?? 0, moon: () => ( { elevation: sky.moonElevation, skyShare: sky.moonSkyShare } ) } );
	render.ambientRelight = ambientRelight;
	// the environment's own harmonics, which the specular's relight divides by (AmbientSample.ts)
	installEnvironmentHarmonics( ambientRelight );
	// the haze's shadow kernels read the probes' sky visibility, so they compile now (Pipeline.ts)
	staged( render ).hazeShadowCompiled = pipeline.compileHazeShadow();
	const ambientView = { mode: ( DEBUG && params.get( 'ambientProbes' ) || 'off' ) as AmbientView };
	render.ambientView = ambientView;
	ambient.show( ambientView.mode, scene );
	const refractedSun = new RefractedSun( renderer, shared );
	render.refractedSun = refractedSun;
	runtime.refractedSun = refractedSun;
	refractedSun.angleGate = THREE.MathUtils.degToRad( render.runtimeQuality.refractedSun.angleGate );
	// the stones', floor's and island parts' triangles under the water come with the chunks (World.ts)
	refractedSun.setCasters( world.group, world.subsets );
	refractedSun.bindTerrain( world.group, waveMaps );
	return refractedSun.prepare();

}

/**
 * Places the registered plant kinds in the frame, in registry order: their roots in the scene, the
 * wind's motion flag on the scene's materials, the depth prepass of the kinds that take one, and
 * the opaque scene's prepass over `ground` and the kinds that take none. It closes the registry:
 * a kind registered afterwards would be in none of them (Registry.ts `close`). The sun's caster
 * box and the rain's collision bake read the same registry (`vegetationCasters`,
 * `bakeRainCollision`), which the scene calls where its own order has them.
 */
export function placeVegetation<L>( render: RenderParts, vegetation: Vegetation<L>, { ground }: { ground: THREE.Object3D[] } ) {

	const { scene } = render;
	// (a registry without kinds, a scene with no plants, has no roots to add)
	const roots = vegetation.roots();
	if ( roots.length ) scene.add( ...roots );
	installWindMotionFlag( scene );
	installVegetationPrepass( scene, vegetation.roots( ( kind ) => kind.prepass ) );
	// the opaque scene's depth for the ambient occlusion (post/GTAO.ts, the high tier's `ao` setting)
	// and the screen-space shadows over the whole scene (post/MeadowShadows.ts `SS_SHADOW`): the ground, and the
	// plants without a prepass of their own
	if ( GTAO || SS_SHADOW ) installOpaquePrepass( [ ...ground, ...vegetation.roots( ( kind ) => ! kind.prepass ) ] );
	vegetation.close( 'placeVegetation has placed the kinds in the scene and its prepasses' );

}

/** The box of everything that casts into the sun's map, into `box`: the meshes that cast a shadow
 * under `ground` and under the roots of the kinds registered as casters. */
export function vegetationCasters<L>( vegetation: Vegetation<L>, { ground, box }: { ground: THREE.Object3D[]; box: THREE.Box3 } ): THREE.Box3 {

	box.makeEmpty();
	for ( const root of [ ...ground, ...vegetation.roots( ( kind ) => kind.sunCaster ) ] ) {

		root.updateWorldMatrix( true, true );
		root.traverse( ( object ) => { if ( ( object as THREE.Mesh ).isMesh && object.castShadow ) box.expandByObject( object ); } );

	}
	return box;

}

/** Bakes the rain's collision map over `area`: drops stop on the first surface they meet, `ground`
 * and the kinds registered for the bake. */
export function bakeRainCollision<L>( render: RenderParts, vegetation: Vegetation<L>, { ground, area }: { ground: THREE.Object3D[]; area: Bounds } ) {

	const { renderer } = render;
	render.rain!.collision.setArea( area );
	render.rain!.bakeCollision( renderer, [ ...ground, ...vegetation.roots( ( kind ) => kind.rainBake ) ] );

}

/** The water's surface, over the land's box and what the game holds above it. */
export function createWater( render: RenderParts, { fields, landBox, seaShape, heldOpaque, plants }: { fields: SceneFields; landBox: Bounds; seaShape: { readonly openSea: { readonly from: number; readonly to: number }; readonly reflectorTop: number }; heldOpaque: THREE.Box3[]; plants?: { box: THREE.Box3; freePath: number } } ) {

	const { quality, renderer, waterScene, rainScene, afterScene, sky, sunShadow, applySkyFill, fft, waveMaps, seaDetail, pipeline, shoreWaves } = render;

	// Storm whitecap foam with a memory, and the spray torn off breaking crests (idle in calm weather).
	const stormSea = new StormSea( renderer, fft, shared, fields.shore, shoreWaves, {
		sunDir: shared.sunDir, sunColor: shared.sunColor,
		sky: shared.skyIrradianceSource.div( shared.skyFill.max( 1 ).sqrt() ),
		// (after the resolve the spray tests the nearest surface, the sea's included: it has no depth buffer)
		sunVisibility: ( p ) => sky.seaKeyVisibility( p ), opaqueDepth: AFTER_RESOLVE ? pipeline.surfaceRT.texture : pipeline.depthRT.texture,
	} );
	render.stormSea = stormSea;
	render.water = new Water( {
		fft, sky, shared, shore: fields.shore, seabedHeight: fields.floor, shoreWaves,
		land: landBox, ...seaShape, sunShadow, storm: stormSea, detail: seaDetail,
		sceneColor: pipeline.opaqueRT.texture, linearDepth: pipeline.depthRT.texture, reflection: pipeline.seaReflection, underwater: pipeline.textures.underwater, quality: quality.water,
		refraction: { color: pipeline.textures.refraction, depth: pipeline.textures.refractionDepth, guard: pipeline.refraction.guard },
	} );
	pipeline.seaLevel.value = shared.waterLevel.value;
	// the boxes held so far (the plants', a game's props shown before the sea was made)
	for ( const box of heldOpaque ) render.water.hold( box );
	// the plants' volume, which a reflected ray crosses as a medium (Water.ts `setPlants`)
	if ( plants ) render.water.setPlants( plants.box, plants.freePath );
	// the light under the water follows the long waves the sea's surface shows (WaveMaps.ts)
	waveMaps.attachWaves( { fft, shared, shore: fields.shore, seabedHeight: fields.floor, shoreWaves } );
	applySkyFill();
	waterScene.add( render.water.mesh );
	( AFTER_RESOLVE ? afterScene : TEMPORAL ? rainScene : waterScene ).add( stormSea.spray.mesh );

}

/**
 * The haze's key for the scene, then the environment's bake with it. Returns the bake, which the
 * entry awaits.
 */
export function bakeWithHaze( render: RenderParts ) {

	const stage = staged( render );
	const { bakeEnvironment } = stage;
	// the hour's and the weather's key, written each frame (weather/HazeKeys.ts)
	render.hazeKey = { deck: 0, overcastDeck: 0, amount: 1, sigmaRed: 1, sigmaGreen: 1, sigmaBlue: 1, desaturate: 0, gain: 1, mist: 0, sunRayleigh: 0, rayleighWeight: 1, mieWeight: 1,
		top: 10, width: 50, aerosolShare: 0.035, aerosolHeight: 1400, aerosolTop: Infinity, skyDepth: 0, groundLight: 1 };
	// the haze exists now: bake the environment again, so it holds the haze every later bake lays over
	// it (Sky.ts `fogEnvironment`) and a fresh load equals a jump
	stage.hazeReady = true;
	return bakeEnvironment();

}

/** Returns the warm-up of every pipeline the frame draws with, which the entry awaits. */
export function awaitWarmUp( render: RenderParts ) {

	const { stormSea } = render;
	const { overlayLatents, warming, hazeShadowCompiled } = staged( render );

	// The storm's foam and spray need the world's shore fields, so they register (and compile) last;
	// the loop then waits for every first-use pipeline (see "first-use pipelines"), the haze's shadow
	// kernels' included.
	return Promise.all( [ warming, hazeShadowCompiled, stormSea.registerLatent( overlayLatents ) ] );

}

/** Checks that every scene material writes its velocity (the temporal modes, debug only). */
export function checkMotion( render: RenderParts ) {

	const { DEBUG, scene, waterScene, rainScene } = render;

	// The temporal modes' debug check (Motion.ts): the materials whose velocity is missing or partial,
	// for the movers' motion vectors still to come.
	const motionPasses = () => [ { pass: 'opaque', scene, keeps: false }, { pass: 'water', scene: waterScene, keeps: false }, { pass: 'overlay', scene: rainScene, keeps: true } ];
	render.motionPasses = motionPasses;
	if ( DEBUG && TEMPORAL ) console.info( 'motion check', motionCheck( motionPasses() ) );

}

/** Builds the nodes every lit receiver shares, ahead of the first frame. */
export function primeSharedBuilds( render: RenderParts ) {

	const { renderer, scene, camera } = render;
	// the nodes r186 makes inside the first lit build, made by a throwaway one, so the frame's receivers
	// build the same code whichever comes first (three/ThreePatches.ts)
	primeSharedNodes( renderer, scene, camera );

}

/** What the app gives the frame's stages (`createFrameStages`). Each is read in the frame, at the
 * stage that names it. */
export interface FrameHost {
	/** the app's switches for the rain and the wet surfaces (`advanceWeatherAndSea`) */
	weatherEffects: { readonly rain: boolean; readonly wetness: boolean };
	/** whether the exposure follows the sky's grade (`advanceWeatherAndSea`) */
	environmentLook: { readonly exposure: boolean };
	/** the paced frame's interval (ms): the temporal resolve's nominal frame time (`draw`) */
	interval(): number;
	/** adds the app's own reasons the frame will still change without a change of its key, for the
	 * settled signal (`draw`; debug pages) */
	pending( reasons: string[] ): void;
	/** whether the canvas shows, with nothing of the page's over it (`draw`; debug pages) */
	shown(): boolean;
}

/**
 * A frame's stages, in the order an app calls them, every frame. The scene's own per-frame work
 * goes between them where its order has it (the island's: IslandStages.ts `createIslandFrame`);
 * the order inside a stage is the render pipeline's.
 */
export interface FrameStages {
	/** True while the frame must not be drawn and no clock may move: an explicit bake of the sky is
	 * in flight (`bakeExplicitly`), or a retune just started one (`retuneSky`). */
	held(): boolean;
	/** The scene clock by `dt` (s), unless it is paused. */
	advanceClock( dt: number ): void;
	/** The day clock by `clockDt` (s) and the weather with it; `hour`, when given, then sets the
	 * hour the sky shows (a host whose own clock keeps the hour). */
	advanceDay( clockDt: number, hour: ( () => number ) | null ): void;
	/** The sky at the day's and the weather's state, the lighting it publishes and the ambient
	 * grade; `environmentDt` (s) is the wall clock's step, which the sky's cross-fades follow.
	 * Returns the night ambient's share. */
	trackSky( environmentDt: number ): number;
	/** The local lights at `lights` (their strength, 0..1) and what the leaves pass on of the key. */
	lightNight( lights: number, nightAmbient: number ): void;
	/** The haze, the wind, the lightning, the clouds, the rain, the exposure's grade, the sky's
	 * periodic bake, the sun's maps and the sea. `dt` is the scene clock's step (s) before the pause. */
	advanceWeatherAndSea( dt: number, environmentDt: number ): void;
	/** The ground's bounce and the ambient light's relight. */
	relight(): void;
	/** Draws the frame. `rawDt` is the wall clock's step (s), `dt` the scene clock's before the pause. */
	draw( rawDt: number, dt: number ): void;
	/** The periodic work in the frame just drawn (PeriodicWork bits), for a dynamic render scale:
	 * asked once a frame, after `draw`. */
	periodicWork(): number;
	/** Hands the frame's command buffers to the GPU. */
	submit(): void;
	/** Restarts what counts frames or wall-clock seconds since the load, as after a seek: the
	 * clouds' trace slot, the sky's periodic bake and the refracted sun's map. */
	restartCadences(): void;
	/** Holds the scene still for the frames that follow (both clocks paused, a sky transition
	 * allowed to rest). Returns the release, which puts the clocks and the cadences back as they were. */
	holdStill(): () => void;
}

/**
 * The frame's stages on the built renderer: what advances the scene clock, the day, the weather,
 * the wind and everything that follows them, then draws. An app's tick calls them in the order
 * FrameStages lists them, once each a frame.
 */
export function createFrameStages( render: RenderParts, host: FrameHost ): FrameStages {

	const { DEBUG, renderer, submitBatch, scene, waterScene, rainScene, afterScene, camera, sky, dayCycle, weather, wind, lighting, applyEnvironmentFill, sun, sunShadow, shadowDirection, prepareHaze, prepareSkyState, publishLighting, retuneSky, fft, waveMaps, pipeline, groundBounce, shoreWaves, ambientRelight, ambientGrade, refractedSun, stormSea } = render;
	const S = pipeline.settings;
	render.shadowElapsed = 0;
	// The temporal resolve's per-frame inputs (TemporalAA.ts): what besides the camera, the clock and the
	// sizes keys a paused scene's rest (a forced strike: a review sets one while paused, and the rest
	// starts over for it), whether the frame may rest (not while a sky bake is in flight or a sky
	// transition runs, unless the frame is held still: see `holding`), and its delta.
	let strikeKey = 'null', strikeVersion = 0;
	// set while the scene is held still for the rest to converge (nothing advances, a transition included)
	let holding = false;
	function strikeSignature(): number {

		const forced = JSON.stringify( ( sky.lightning as unknown as { forced: unknown } ).forced ?? null );
		if ( forced !== strikeKey ) {

			strikeKey = forced;
			strikeVersion ++;

		}
		return strikeVersion;

	}
	const restStable = () => ! sky.busy && ( ! sky.transitioning || holding );
	function temporalFrame( dt: number, step: number ): TemporalFrame | undefined {

		if ( ! pipeline.resolver ) return undefined;
		// (FSR 3's frame time: the scene clock's step, the pacer's nominal interval while paused)
		// The temporal rest starts once the haze's has ended, so it averages frames of the finished haze.
		return { signature: [ strikeSignature() ], stable: restStable() && ! pipeline.air.pending, dt, step, interval: host.interval() };

	}
	// The haze's rest (weather/VolumetricHaze.ts): the temporal rest's signature and the sun maps' redraw
	// counts (a redraw changes its shadow without changing the key), and whether the frame may rest.
	function hazeFrame(): HazeFrame {

		return { time: render.simTime, signature: [ strikeSignature(), sunShadow.draws, refractedSun.draws ], stable: restStable() };

	}
	// The settled signal's inputs (post/Settled.ts; debug builds): the rest's, the sun maps' redraw
	// counts, and what will still change the frame without changing the key.
	function settleFrame( bakeDue: boolean ): SettleFrame | undefined {

		if ( ! DEBUG ) return undefined;
		const pending: string[] = [];
		if ( bakeDue ) pending.push( 'sky bake due' );
		if ( sky.readbacks > 0 ) pending.push( 'sky irradiance' );
		if ( refractedSun.pending ) pending.push( 'refracted sun' );
		host.pending( pending );
		if ( sunShadow.pending ) pending.push( 'sun shadow' );
		if ( pipeline.air.pending ) pending.push( 'haze rest' );
		if ( pipeline.seaReflection.pending ) pending.push( 'sea reflection rest' );
		if ( pipeline.exposurePending ) pending.push( 'exposure' );
		if ( preExposure.pending ) pending.push( PRE_EXPOSURE_PENDING );
		if ( ! ambientRelight.ready ) pending.push( 'ambient relight' );
		return { signature: [ strikeSignature(), sunShadow.draws, refractedSun.draws ], stable: restStable(), pending, shown: host.shown() };

	}
	// whether a periodic bake of the sky waits for its interval (`advanceWeatherAndSea` to `draw`)
	let bakeDue = false;
	// The periodic work in a frame (PeriodicWork bits): its frames stay out of the dynamic scale's
	// quantiles. A periodic sky bake runs its slices in the ticks while the sky is busy, from the one
	// that starts it to the one that publishes it.
	let skyWasBusy = false;
	const periodicCounts = { sunShadow: 0, refractedSun: 0 };

	return {

		// a jump's explicit bake in flight: no frame, and the clock holds (`bakeExplicitly`); a retune
		// starts one (`retuneSky`)
		held: () => render.jumpsInFlight > 0 || retuneSky(),

		advanceClock( dt ) {

			if ( ! render.paused ) render.simTime += dt;

		},

		advanceDay( clockDt, hour ) {

			dayCycle.advance( clockDt );
			// the sky shows the host's hour, whatever else moved the day clock
			if ( hour ) dayCycle.setHour( hour() );
			weather.update( render.paused ? 0 : clockDt );

		},

		trackSky( environmentDt ) {

			renderer.info.reset();
			// Sun, moon, key light and grade follow the clock every frame; only the baked reflection,
			// environment and cloud products cross-fade between bakes.
			sky.track( render.manualSun ? null : dayCycle.snapshot(), weather.state, render.cloudOverride, weather.skyMix );
			sky.advance( environmentDt );
			publishLighting();
			// the ambient grade by the sun and the weather's blend of the four skies (lighting/AmbientGrade.ts)
			ambientGrade.update( sky.elevation, weather.skyMix, sky.rising, sky.moonElevation, sky.moonKeyShare );
			shared.clouded.value = cloudedShare( weather.skyMix );
			// the leaves' night key and the sward's go by the night ambient's share (sky/DayCycle.ts
			// `nightAmbientShare`), the local lights by a share of the scene's own (`lightNight`); a debug
			// page may force each reader's (DayCycle.ts `duskForce`)
			return nightAmbientShare( sky.dayWindows, sky.elevation, sky.rising, shared.clouded.value );

		},

		lightNight( lights, nightAmbient ) {

			// (lighting/LocalLights.ts: its pass sorts the sources in view)
			render.localLights.update( duskForce( 'localLights', lights ) );
			// what the crowns' leaves pass on of the key (vegetation/LeafLight.ts)
			updateLeafLight( sky.elevation, duskForce( 'nightAmbient', nightAmbient ), shared.clouded.value );

		},

		advanceWeatherAndSea( dt, environmentDt ) {

			prepareHaze( weather.state );
			pipeline.air.keyDir.value.copy( sky.keyDir );
			pipeline.air.keyColor.value.copy( sky.keyColor ).multiplyScalar( 1 - THREE.MathUtils.smoothstep( sky.clouds.coverage.value, 0.7, 0.97 ) );
			pipeline.shafts.setSun( sky.sunDir, sky.sunColor );
			pipeline.shafts.setMoon( sky.moonDir, sky.keyColor, sky.keyIsMoon );
			wind.update( render.paused ? 0 : dt, weather.state.windSpeed, weather.state.windDirection, weather.state.gustiness );
			// Lightning follows the scene clock, so a paused frame holds its strike; it lights the clouds'
			// march below, so it goes first.
			// (the scene's render size: the canvas is larger while the frame is upscaled)
			sky.lightning.update( render.simTime, weather.state.storm, camera, pipeline.height, sky.clouds );
			sky.updateClouds( camera, pipeline.width, pipeline.height, shared.wind.direction.value, render.simTime );
			shared.rain.value = host.weatherEffects.rain ? sky.weather.rain : 0;
			shared.storm.value = weather.state.storm;
			// A strike lights the island for an instant through the sky's fill: the lit cloud deck adds its
			// radiance to the sky's, so a flash that barely shows by day floods a dark night.
			const skyLuma = sky.skyIrradiance.r * 0.2126 + sky.skyIrradiance.g * 0.7152 + sky.skyIrradiance.b * 0.0722;
			const strikeFill = lighting.skyFill * Math.min( 1 + sky.lightning.fill / Math.max( skyLuma, 1e-5 ), 60 );
			applyEnvironmentFill( strikeFill );
			shared.skyFill.value = strikeFill;
			shared.skyIrradiance.value.set( sky.skyIrradiance.r, sky.skyIrradiance.g, sky.skyIrradiance.b ).multiplyScalar( strikeFill );
			// the app's own wetness, in the game too (soaking and drying on the day clock)
			shared.wetness.value = host.weatherEffects.wetness ? weather.state.wetness : 0;
			fft.setWind( wind.speed, wind.heading );
			render.rain!.uniforms.rain.value = shared.rain.value;
			render.rain!.uniforms.wind.value.copy( shared.wind.velocity.value );
			render.rain!.uniforms.waterLevel.value = shared.waterLevel.value;
			render.rain!.update( renderer, camera, render.simTime, shared.wind.displacement.value, pipeline.height, AFTER_RESOLVE ? pipeline.outputHeight : pipeline.height );
			if ( ! render.manualSun && host.environmentLook.exposure ) {

				S.exposure = sky.grade.exposure;
				// the night's grade (Pipeline NIGHT)
				S.night = sky.grade.night;
				// the exposure's own night weight (its keys: look/ExposureLook.ts)
				S.exposureNight = sky.grade.exposureNight;

			}
			// the weather's stops of the exposure and the clouded day's keys (Pipeline `skyMix`)
			const skyMix = weather.skyMix;
			for ( let i = 0; i < skyMix.length; i ++ ) S.skyMix[ i ] = skyMix[ i ];
			render.skyElapsed += environmentDt;
			const skyInterval = render.runtimeQuality.skyUpdateInterval;
			const ws = weather.state;
			const weatherSignature = `${ render.cloudOverride ?? ws.coverage },${ ws.density },${ ws.directVisibility },${ ws.ambientGain },${ ws.daySky },${ ws.daySkyExponent },${ ws.haze }`;
			if ( ! sky.busy && ! sky.transitioning && render.skyElapsed >= skyInterval &&
				( render.simTime !== render.lastSkyTime || weatherSignature !== render.lastWeatherSignature || ( ! render.manualSun && dayCycle.hour !== render.lastSkyHour ) ) ) {

				render.skyElapsed = 0;
				render.lastSkyTime = render.simTime;
				render.lastSkyHour = dayCycle.hour;
				prepareSkyState();
				void sky.update( skyInterval ).catch( ( error: unknown ) => console.error( error ) );

			}
			// a periodic bake waits for its interval (the settled signal waits for it too)
			bakeDue = ! sky.busy && ( render.simTime !== render.lastSkyTime || weatherSignature !== render.lastWeatherSignature || ( ! render.manualSun && dayCycle.hour !== render.lastSkyHour ) );
			// The sun's map steps with the key light and cross-fades between steps, so its shadows move
			// smoothly without re-rasterising sub-texel casters every frame; it is drawn along the lifted
			// key. The underwater map keeps the quality cadence.
			sunShadow.update( renderer, scene, camera, shadowDirection );
			render.shadowElapsed += environmentDt;
			if ( render.shadowElapsed >= render.runtimeQuality.refractedSun.interval && refractedSun.update() ) render.shadowElapsed = 0;
			shared.time.value = render.simTime;
			fft.update( render.paused ? 0 : dt, render.simTime );
			stormSea.update( render.simTime, weather.state.storm, camera );
			shoreWaves.update( fft, render.simTime );
			// Whitewater is disabled in this scene; leave its history idle.
			// (the wave maps and the caustics are drawn by their pass on the frame graph: WaveMaps.ts; the
			// caustics only while the sun is the key and gives light, and when the sea's time or the key moved)
			waveMaps.prepare( camera, sky.keyDir, ! sky.keyIsMoon && Math.max( shared.sunColor.value.x, shared.sunColor.value.y, shared.sunColor.value.z ) > 0, render.simTime, fft.spectrumRevision );

		},

		relight() {

			groundBounce.update( sky.seaRadiance );
			ambientRelight.update();

		},

		draw( rawDt, dt ) {

			// (the ground truth draws the storm's spray with the rain, in the overlay pass; taa and taau draw
			// it with the streaks after the resolve)
			const overlay = ! AFTER_RESOLVE && ( render.rain!.mesh.visible || ( TEMPORAL && stormSea.spray.mesh.visible ) );
			const after = AFTER_RESOLVE && ( render.rain!.mesh.visible || stormSea.spray.mesh.visible );
			pipeline.render( { scene, prepassScene, opaquePrepassScene, waterScene, rainScene: overlay ? rainScene : undefined, overlay, afterScene: after ? afterScene : undefined, camera, water: render.water!, sun, keyDirection: shadowDirection, time: render.simTime, temporal: temporalFrame( rawDt, render.paused ? 0 : dt ), settle: settleFrame( bakeDue ), haze: hazeFrame() } );

		},

		periodicWork() {

			let heavy = 0;
			if ( sky.busy || skyWasBusy ) heavy |= PeriodicWork.skyBake;
			skyWasBusy = sky.busy;
			if ( sunShadow.draws !== periodicCounts.sunShadow ) heavy |= PeriodicWork.sunShadow;
			if ( refractedSun.draws !== periodicCounts.refractedSun ) heavy |= PeriodicWork.refractedSun;
			periodicCounts.sunShadow = sunShadow.draws;
			periodicCounts.refractedSun = refractedSun.draws;
			return heavy;

		},

		submit() {

			submitBatch.flush();

		},

		restartCadences() {

			( sky.clouds as unknown as { frame: number } ).frame = 0;
			render.skyElapsed = 0;
			render.shadowElapsed = 0;

		},

		holdStill() {

			const was = { paused: render.paused, day: dayCycle.paused, skyElapsed: render.skyElapsed, shadowElapsed: render.shadowElapsed };
			render.paused = true;
			dayCycle.paused = true;
			holding = true;
			return () => {

				holding = false;
				render.paused = was.paused;
				dayCycle.paused = was.day;
				render.skyElapsed = was.skyElapsed;
				render.shadowElapsed = was.shadowElapsed;

			};

		},

	};

}

/** Registers what a scene jump clears of the render pipeline's, ahead of the island's: the wind's
 * integrated state, the storm's strikes and its sea's foam window (post/SceneHistory.ts; the
 * order registered is the order cleared). */
export function registerSceneHistories( render: RenderParts ) {

	const { history, wind, weather, sky, stormSea } = render;
	history.onJump( 'wind', () => wind.reset( { ...weatherWind( weather.state ), time: render.simTime } ) );
	history.onJump( 'lightning', () => sky.lightning.reset() );
	history.onJump( 'storm sea', () => stormSea.reset() );

}

/** Registers what a scene jump clears of the render pipeline's, after the island's: the fog's bank, the
 * ground's bounce, the underwater map's cadence, and after the jump's bake the underwater map, at its constructed size and
 * drawn at once for the jump's light (published by the bake), as the load's `prepare` draws it
 * before the first frame: left to the loop's cadence (wall time), a capture ready within a few
 * frames captured the cleared map, the sea without its refracted sun. */
export function registerLightHistories( render: RenderParts ) {

	const { history, pipeline, groundBounce, refractedSun } = render;
	history.onJump( 'haze', () => pipeline.air.invalidate() );
	history.onJump( 'sea reflection', () => pipeline.seaReflection.invalidate() );
	history.onJump( 'ground bounce', () => groundBounce.invalidate() );
	history.onJump( 'refracted sun', () => {

		refractedSun.reset();
		refractedSun.update();

	}, 'after bake' );
	history.onJump( 'shadow cadence', () => { render.shadowElapsed = 0; } );

}
