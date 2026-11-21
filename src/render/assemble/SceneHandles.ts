// A running scene's diagnostics and debug handles for the tools and the console, on debug pages only.
//
// A running scene's diagnostics and debug handles (StartScene.ts), the app's contract with the
// capture, probe and profiling tools (docs/bindings/writing-tools-and-checks.md, Build): the
// diagnostics global (`__THREE_GAME_DIAGNOSTICS__` by default), wholly refreshed every 30 frames
// and, every frame, in the fields the capture tools wait on (`settled`, `taa`, `preExposure`), and
// the handles global (`__PONLAI__` by default): the scene's parts, the scene jump, the loop and its
// step, the tunables, the radiance reader. Both exist only on debug pages (the dev server, `?debug`,
// `?profile`). An app adds its own fields to both (`SceneHandlesContent`).
import * as THREE from 'three/webgpu';
import type { RenderParts } from '../RenderStages.ts';
import type { QualityProfile } from '../Quality.ts';
import type { Shared } from '../shared.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { tuneApi } from '../debug/Tune.ts';
import type { PipelineCount } from '../debug/PipelineCounter.ts';
import { liftedKeyDirection } from '../lighting/KeyLift.ts';
import { AA_MODE, AA_SOURCE } from '../post/Antialiasing.ts';
import type { DynamicResolution } from '../post/DynamicResolution.ts';
import { gtaoUniforms } from '../post/GTAO.ts';
import type { GradeChanges } from '../post/GradeLut.ts';
import { jitter } from '../post/Jitter.ts';
import { motionCheck, motionState, taaCut } from '../post/Motion.ts';
import { preExposure } from '../post/PreExposure.ts';
import { renderEvent } from '../post/SceneEvents.ts';
import { restEpoch } from '../post/TemporalRest.ts';
import type { SceneFrame } from './SceneFrame.ts';
import type { SceneHost } from './SceneHost.ts';
import type { SceneJump } from './SceneJump.ts';
import { SCENE_JUMP_VERSION } from './SceneJump.ts';

/** The names of the globals a debug page puts its handles and diagnostics on. */
export interface SceneGlobals {
	/** the scene's handles */
	handles: string;
	/** the diagnostics snapshot the tools wait on */
	diagnostics: string;
	/** the frame profiler's API (`?profile`) */
	profile: string;
	/** set by the clip tools while they step a clip: the exposure snaps every frame (post/Pipeline.ts) */
	clip: string;
}

export const DEFAULT_GLOBALS: SceneGlobals = Object.freeze( {
	handles: '__PONLAI__',
	diagnostics: '__THREE_GAME_DIAGNOSTICS__',
	profile: '__PONLAI_PROFILE__',
	clip: '__PONLAI_CLIP__',
} );

export interface SceneHandlesContent {
	/** the app's fields of the whole snapshot (every 30 frames) */
	diagnostics?(): Record<string, unknown>;
	/** the app's fields refreshed every frame */
	everyFrame?(): Record<string, unknown>;
	/** the app's handles, added to the scene's (`jump`: the scene's jump, for handles that move it) */
	handles?( jump: SceneJump ): Record<string, unknown>;
}

// The dynamic scale's clock as a debug hook (`dynamicClock`): `suspend()` holds the controller's
// clock where it is, `resume()` lets it run on from there (the time between is skipped).
export function dynamicClock( d: DynamicResolution ): { suspend(): void; resume(): void; readonly suspended: boolean } {

	let skipped = 0, since = NaN;
	return {
		suspend() {

			if ( ! Number.isNaN( since ) ) return;
			// eslint-disable-next-line ponlai/scene-clock -- the dynamic scale's controller clock (a debug hold), not a scene state
			since = performance.now() / 1000;
			const held = since - skipped;
			d.clock = () => held;

		},
		resume() {

			if ( Number.isNaN( since ) ) return;
			// eslint-disable-next-line ponlai/scene-clock -- the dynamic scale's controller clock (a debug hold), not a scene state
			skipped += performance.now() / 1000 - since;
			since = NaN;
			// eslint-disable-next-line ponlai/scene-clock -- the dynamic scale's controller clock (a debug hold), not a scene state
			d.clock = () => performance.now() / 1000 - skipped;

		},
		get suspended() { return ! Number.isNaN( since ); },
	};

}

/** The diagnostics the tools wait on, refreshed after each frame (`now`: wholly). */
export function createSceneDiagnostics( render: RenderParts, shared: Shared, host: SceneHost, globals: SceneGlobals, frame: () => SceneFrame, pipelineCount: PipelineCount | null, content: SceneHandlesContent ): ( now?: boolean ) => void {

	const { DEBUG, qualityTier, renderer, submitBatch, resolution, sky, dayCycle, weather, wind, pipeline } = render;
	let current: Record<string, unknown> | null = null;

	// The frame graph's state for the diagnostics: whether it is compiled and sized, its variant, its
	// passes (declared, culled, and run or skipped in the last frame), its graph-owned textures and
	// their memory, and its dev counters.
	function graphDiagnostics(): Record<string, unknown> {

		const g = pipeline.graph;
		const d = g.describe();
		const run = d.passes.filter( ( p ) => p.runs ).length;
		return {
			ready: d.ready,
			variant: d.variant,
			passes: d.passes.length,
			culled: d.culled,
			targets: d.resources.filter( ( r ) => ! r.imported ).map( ( r ) => ( { name: r.name, w: r.actual.width, h: r.actual.height, format: r.format, type: r.type, samples: r.samples } ) ),
			bytes: d.bytes,
			rebuilds: d.rebuilds,
			passesRun: run,
			passesSkipped: d.passes.length - d.culled.length - run,
			opsApplied: g.opsApplied,
			opsExpected: g.opsExpected,
			opsMismatchFrames: g.opsMismatchFrames,
			sizeErrors: g.sizeErrors,
		};

	}

	return function diagnostics( now = false ): void {

		if ( ! DEBUG ) return;
		const f = frame();
		if ( ! now && f.frame % 30 !== 1 ) {

			// the temporal resolve's state, the settled signal and the pre-exposure on every frame: the
			// capture tools wait for them
			if ( current && pipeline.resolver ) current.taa = pipeline.resolver.diagnostics();
			if ( current ) Object.assign( current, { settled: pipeline.settled.value, settle: pipeline.settled.diagnostics(), preExposure: preExposure.diagnostics() }, content.everyFrame?.() );
			return;

		}
		const info = renderer.info;
		const { frameTimes, dynamicResolution, pacer, sizing } = f;
		current = {
			frame: f.frame,
			frameMs: + ( frameTimes.reduce( ( a, b ) => a + b, 0 ) / frameTimes.length ).toFixed( 2 ),
			frameMsP95: + ( [ ...frameTimes ].sort( ( a, b ) => a - b )[ Math.floor( frameTimes.length * 0.95 ) ] ?? NaN ).toFixed( 2 ),
			canvas: [ renderer.domElement.width, renderer.domElement.height ],
			render: [ pipeline.width, pipeline.height ],
			upscaled: pipeline.upscaling,
			calls: info.render.drawCalls,
			triangles: info.render.triangles,
			computeCalls: info.compute.frameCalls,
			geometries: info.memory.geometries,
			textures: info.memory.textures,
			dpr: + ( pipeline.width / host.size().width ).toFixed( 4 ), // the render's pixel ratio
			renderScale: sizing.renderScale,
			dynamicResolution: dynamicResolution ? { scale: dynamicResolution.scale, source: dynamicResolution.source, changes: dynamicResolution.changes.length } : null,
			pacing: { target: pacer.target, rate: pacer.rate, mode: pacer.mode, refreshMs: + pacer.period.toFixed( 3 ), divisor: pacer.divisor, base: pacer.base, fallback: pacer.fallen, falls: pacer.falls, returns: pacer.returns, rendered: pacer.rendered, skipped: pacer.skipped, held: pacer.held, queued: pacer.queued, queueLimit: pacer.queueLimit },
			submits: { ...submitBatch.stats, why: { ...submitBatch.stats.why } },
			canvasDpr: renderer.getPixelRatio(),
			backend: 'webgpu',
			hour: + dayCycle.hour.toFixed( 4 ),
			clockPaused: dayCycle.paused,
			scalePinned: resolution.full,
			skyBakeMs: + sky.lastBakeMs.toFixed( 2 ),
			skyBusy: sky.busy,
			skyTransitioning: sky.transitioning,
			quality: qualityTier,
			weather: weather.target,
			// the ambient grade's values (lighting/AmbientGrade.ts) and whether the GUI overrides them
			ambient: { ...render.ambientGrade.current, override: render.ambientGrade.override },
			weatherProgress: + weather.progress.toFixed( 3 ),
			rain: + sky.weather.rain.toFixed( 3 ),
			wetness: + shared.wetness.value.toFixed( 3 ),
			windSpeed: + wind.speed.toFixed( 3 ),
			windHeading: + wind.heading.toFixed( 2 ),
			windTarget: weather.windTarget,
			...content.diagnostics?.(),
			// the scene's anti-aliasing, the pipelines created so far and the frame graph's state and
			// dev counters, for the clip and first-use tools
			aa: AA_MODE,
			// where the mode came from: the URL (`aa`, `msaa`) or the quality tier's `aa` setting
			aaSource: AA_SOURCE,
			pipelines: pipelineCount?.total ?? 0,
			graph: graphDiagnostics(),
			// the temporal resolve: its mode, the rest (converged, frames, target), the jitter, the history's
			// resets, the input and output sizes and whether reactive inputs are read (TemporalAA.ts)
			taa: pipeline.resolver?.diagnostics() ?? null,
			// whether the paused page will draw the same frame from now on, and why not (post/Settled.ts)
			settled: pipeline.settled.value,
			settle: pipeline.settled.diagnostics(),
			// the pre-exposure's stops, the P the last reading asks for, whether it is pending, the
			// exposure EV last read back, its changes and its readings, the view and its quiet frames
			// (post/PreExposure.ts)
			preExposure: preExposure.diagnostics(),
			...content.everyFrame?.(),
		};
		host.publish( globals.diagnostics, current );

	};

}

/** The scene's handles on the handles global (debug pages). */
export function exposeSceneHandles( render: RenderParts, shared: Shared, host: SceneHost, globals: SceneGlobals, frame: SceneFrame, jump: SceneJump, pipelineCount: PipelineCount | null, content: SceneHandlesContent ): void {

	const { DEBUG, quality, renderer, submitBatch, scene, camera, sky, dayCycle, weather, wind, applySkyQuality, sun, sunShadow, invalidateSunShadow, retuneSky, refreshEnvironment, applySun, setHour, setPreset, setWeather, setWind, fft, waveMaps, seaDetail, pipeline, groundBounce, shoreWaves, ambient, ambientRelight, refractedSun, stormSea, hazeKey, motionPasses } = render;
	const { dynamicResolution, pacer, loop, step } = frame;
	if ( ! DEBUG ) return;
	host.publish( globals.handles, {
		renderer, scene, camera, sky, sun, fft, shoreWaves, stormSea, waveMaps, seaDetail, water: render.water, pipeline, shared, refractedSun, groundBounce,
		// the direction the surfaces take the key's light from, the low key's lift included (lighting/KeyLift.ts)
		keyShading: () => liftedKeyDirection( sky.keyLift, sky.keyDir, sky.keyIsMoon, new THREE.Vector3() ),
		// the ambient grade (lighting/AmbientGrade.ts): `current`, `table`, `override` and `manual`
		ambient: render.ambientGrade,
		// the local lights (lighting/LocalLights.ts): `sources`, `gain`, `enabled`, `info()` (a promise: the
		// sources in view, the fullest cluster, the clusters that overflowed)
		localLights: render.localLights,
		// the screen-space ambient occlusion's parameters (post/GTAO.ts `gtaoUniforms`: cutoff, recede, base, maxTexels, fullDistance)
		gtao: gtaoUniforms,
		// the ambient light's probes (lighting/AmbientVolume.ts): `info()` and `hash()`
		ambientVolume: ambient,
		// their relight (lighting/AmbientRelight.ts): `readback()` and `hash()` of its output
		ambientRelight,
		// the look's parameters from a script (the GUI's raise their own events): the first table's grade
		// (GradeLut.ts), the leaves' albedo (shared.leafGain)
		setGrade: ( changes?: GradeChanges ) => { pipeline.setGrade( changes ); renderEvent( 'retune' ); },
		setLeafGain: ( gain: number ) => { shared.leafGain.value = gain; renderEvent( 'retune' ); },
		meadowCanopy: shared.meadowCanopy,
		dayCycle, weather, wind, rain: render.rain, setWeather, setWind,
		// the scene jump (SceneJump.ts) and the options it may change
		establishScene: jump.establishScene, sceneJump: SCENE_JUMP_VERSION, sceneOptions: jump.sceneOptions, setTime: jump.setTime,
		applySun, refreshEnvironment, setHour, setPreset, sunShadow, invalidateSunShadow,
		clockTime: () => render.simTime, nominalScale: () => frame.sizing.nominalScale,
		// the loop's hold, resume and step (SceneFrame.ts)
		loop: Object.assign( loop, { step } ), pipelines: pipelineCount, flags: debugFlags,
		// the tunable parameters (debug/Tune.ts): set( { name: value } ), get, list, reset, query; a value
		// the environment's bakes depend on rebakes them at once (`retuneSky`, sky/SkyRetune.ts), as the
		// load bakes
		tune: tuneApi( () => {

			if ( ! retuneSky() ) renderEvent( 'retune' );

		} ),
		// the scene's linear radiance over screen boxes (debug/Radiance.ts), loaded on first call
		radiance: async ( options?: import( '../debug/Radiance.ts' ).RadianceOptions ) => ( await import( '../debug/Radiance.ts' ) ).readRadiance( renderer, pipeline, options, camera ),
		// where a paused frame is compared within the held tolerance (debug/HeldRegion.ts), loaded on first call
		heldRegion: async ( cell?: number ) => ( await import( '../debug/HeldRegion.ts' ) ).readHeldRegion( renderer, pipeline, cell ),
		// the haze's key as the last frame set it (weather/HazeKeys.ts `hazeKeyAt`, `prepareHaze`): the
		// medium's amount and shape, the light's weights, the deck's shares; read, never written
		hazeKey,
		quality, dynamicResolution, pacer, submitBatch, jitter, motion: motionState, taaCut, motionCheck: () => motionCheck( motionPasses() ), applyQuality: ( overrides: Partial<QualityProfile> ) => applySkyQuality( { ...quality, ...overrides } ),
		// the temporal resolve (TemporalAA.ts, TemporalRest.ts): its live settings and parameters (`params`), the rest, the
		// epoch, and `restart()`, which starts the history and the rest over (a cut and an epoch bump)
		// (under fsr3 the same handles from FSR 3's upscaler, which has no such parameters: the capture tools wait on
		// its rest through them)
		taa: pipeline.resolver ? {
			resolve: pipeline.resolver, settings: pipeline.resolver.settings, params: pipeline.temporal?.params ?? null, rest: pipeline.resolver.rest, epoch: restEpoch,
			get converged() { return pipeline.resolver!.converged; },
			restart: () => renderEvent( 'restart' ),
		} : null,
		// FSR 3 (post/fsr3/Fsr3Upscaler.ts, aa=fsr3): its rest, live settings and diagnostics()
		fsr3: pipeline.fsr3,
		// the profiler's A/B (profile-frame --compare): while it keeps this page stopped for the other
		// page's slices, the dynamic scale's clock stands still (DynamicResolution.ts `clock`), so its
		// holds and bars do not run out while the page draws nothing
		dynamicClock: dynamicResolution ? dynamicClock( dynamicResolution ) : null,
		...content.handles?.( jump ),
	} );

}
