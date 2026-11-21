// The quality tiers: every setting the frame's cost depends on, per tier, in one table.
//
// `?quality` picks the tier at load; without it the app loads high (QualityDetect.ts). The GUI's
// quality control reloads the page with it.
//
// High is the reference image and the only active tier: every trade-off of image quality against
// time is judged on it, at its performance target: 4K output (1920x1080 at a device pixel ratio of 2) at 60 fps with
// MSAA 4x (`aa`), with a GPU budget of about 14 ms a frame in the normal scenes (island views, the
// meadow close up), leaving about 15 % of the 16.7 ms frame for pacing and the OS. 1440p's pixel
// count, upscaled by FSR 1 (scale 4/3), is the guaranteed level: a load starts there and a paused
// page renders there; the dynamic render scale (`render`, post/DynamicResolution.ts) climbs toward
// native 4K with headroom and drops to about 1080p's pixel count when busy. Native 4K takes about
// half as much GPU time again as the baseline, about 45 fps in the normal scenes, so 60 fps there
// is the scale's reach and not the performance target. A 2K display (2560x1440 at 1x) renders the same pixels
// natively and drops to about 1080p's. The meadow is built for the island view's
// distance (main.ts starts it at 45 m): its blades stay a few pixels wide there (`grass`), so MSAA
// resolves their edges within a frame, with no history to ghost or soften. Measured at 4K output:
// at native 4K, MSAA held the frame's flicker and sharpness within about 5 % of the many-sample
// reference from 40 to 55 m (one sample a pixel: a fifth to a third more flicker and grain), and
// 1440p with MSAA, upscaled to 4K, looked nearly the same at 1:1, a little softer, for about seven
// tenths of native 4K's GPU time; `taau` at 1440p softened the meadow to about four fifths of the
// reference's sharpness.
//
// Parked tiers (`PARKED_QUALITY`: medium, low) slowed every line of work down, so until they are
// revived as their own milestone they are out of the default choice (no GPU timing picks them), the
// GUI's list and the default checks (the tools' default sets name high only). Their rows stay in the
// table and keep compiling, and `?quality=medium|low` (or a tool's `--quality` / `--tiers`) still
// loads them; new work need not support them. Their last tuning: medium runs `taau` at a fixed
// budget (msaa missed 60 fps there where taau held it), its TAAU sharpening softer than high's; low runs `msaa` at a smaller budget. A revival measures them again against high.
//
// Each tier picks its anti-aliasing (the `aa` setting, post/Antialiasing.ts; `?aa` or the GUI choose
// any mode over it). Without multisampling or a temporal resolve geometric edges alias by a pixel's
// step and the meadow's widened blades are kept whole or dropped by their hash (Grass.ts), so no
// tier runs `off`. No setting alters the weather,
// clock or lighting state: a tier changes the cost, never the mood.
//
// Two kinds of setting:
// - load-time: compiled into materials or the frame graph (shader loops, pass structure, map
//   sizes), read once while they are built, so they change only with a reload;
// - runtime: setters the app applies through `applySkyQuality` (RenderStages.ts) at load; debug
//   code can override them there (`s.applyQuality( { ... } )`) to time one at a time.
// Load-time settings reach their modules as `water` (Water.ts; its `fftAnisotropy` goes to OceanFFT's
// output targets), `rainShading` (`setRainShading`), `post` (`PostOptions` for
// `new Pipeline`), `environmentSource` (`Sky.setEnvironmentSource`, before the first bake),
// `frameRate` (FramePacer), `ao` (post/GTAO.ts, read there as it loads), `aa` (the anti-aliasing
// mode, post/Antialiasing.ts, which also chooses the tier, since the vegetation's modules read the
// mode as they load), `taa` (the temporal resolve's settings, TemporalAA.ts) and `render`
// (RenderScale.ts). Runtime ones: the cloud view
// (`Sky.setCloudView`), the clouds' `setDome`, `setMarch` and `setShadowRefresh`,
// `Sky.setEnvironmentCadence`, the sky bake interval, the refracted sun's interval and
// `angleGate`, and the rain's drop count. Every default in the modules themselves is high's.
// `?tier.<setting>=<value>` overrides one setting for an A/B.
//
// The parked medium and low traded a little of high's image quality for time, each setting judged
// in captures:
// one screen-space reflection ray (its 14 steps grow by 1.51, so it reaches as far as high's), fewer lawn blades
// with the canopy light per vertex, a coarser haze grid and fewer depth-of-field taps, a cheaper
// cloud march and panorama refresh, slower sky bakes with the environment's PMREM on every second
// one from a 256-wide source, a coarser refracted-sun gate and fewer rain drops; low also draws
// fewer veil shells and ripple layers. Low keeps the full storm foam: the merged foam turned the crests'
// streaks into scattered lace.
//
// This module imports types only, so the CPU-only tools can read the table under plain node.
import type { WaterQuality } from './ocean/Water.ts';
import type { CloudMarch, CloudViewScale } from './sky/CloudModel.ts';
import type { PostOptions } from './post/PostOptions.ts';
import type { AAMode } from './post/Antialiasing.ts';
import { settings } from './Settings.ts';

export const QUALITY_TIERS = [ 'low', 'medium', 'high' ] as const;
export type QualityTier = typeof QUALITY_TIERS[ number ];
export const DEFAULT_QUALITY: QualityTier = 'high';
/** Tiers out of the default choice, the GUI and the default checks; still loaded by name. */
export const PARKED_QUALITY: ReadonlyArray<QualityTier> = [ 'medium', 'low' ];
export const isParkedQuality = ( tier: string ): boolean => ( PARKED_QUALITY as readonly string[] ).includes( tier );
/** The active tiers (today high alone). */
export const ACTIVE_QUALITY: ReadonlyArray<QualityTier> = QUALITY_TIERS.filter( ( tier ) => ! isParkedQuality( tier ) );

/** How the scene's render scale (its pixel ratio relative to CSS pixels) is chosen. */
export type RenderScaleRule =
	// a fixed budget of rendered pixels (the scale follows the window's size)
	| { budget: number }
	// dynamic resolution: from `max` (capped by the display's own ratio) down to the ratio that renders
	// `floor` pixels in `step`s, steered by the GPU's frame time (post/DynamicResolution.ts). A load,
	// a paused page and a pinned one render at the ratio that renders `baseline` pixels, capped by
	// the maximum. Pixel counts rather than shares of the display's ratio, so a frame costs about
	// the same on any display; the floor is at most `floorShare` of the maximum per axis, which keeps
	// a small display's floor below its maximum (post/RenderScale.ts `dynamicRange`).
	| { dynamic: { max: number; baseline: number; floor: number; floorShare: number; step: number } };

/** The temporal resolve's settings (taa and taau; read once as the resolve is built). */
export interface TemporalSettings {
	// the history's effective frame count (its blend weight 1 / n) for a still pixel and for one
	// moving `speedFull` render pixels a frame or more (eased between). 24 still:
	// with the movers' motion cap (TemporalAA.ts) a moving pixel keeps at most 10 whatever this is,
	// so the longer still history no longer lengthens the islanders' trail (the islander walk's
	// ghosting at 24 with the movers no higher than at 12 without), and the second stage
	// (`taaSmooth`), which averages the history's image again over as many frames, needs the 24 to
	// take a still camera's sunlit meadow to within twice MSAA's high-pass flicker (2.4 times at
	// 12). The debug flags that turn the movers or the second stage off leave it at 24; lite (no
	// movers, no second stage) keeps 12.
	framesStill: number;
	// the sea's still frame count (its class share and `waterFar` apply to it as to `framesStill`):
	// held apart so a longer still history elsewhere does not lengthen it (a longer one, 24, raised the
	// ghosting and softened the horizon in sea pans)
	framesStillWater: number;
	framesMoving: number;
	speedFull: number;
	// the neighbourhood clip box's width in standard deviations (variance clipping after Salvi),
	// still and moving
	gammaStill: number;
	gammaMove: number;
	// per surface class (Motion.ts MOTION_CLASS: opaque, vegetation, water, character, sky): the share
	// of the history's frames it keeps, and what it adds to the clip box's width
	classFrames: number[];
	classGamma: number[];
	// the view distance (m) over which the sea's share of the history grows from its class's to a whole
	// one (its waves' phase motion, which the motion vectors leave out, shrinks under a pixel far off)
	waterFar: number[];
	// the sea's glints' reactive weight (Water.ts): from 0 at `from` to `max` at `to` times the
	// luminance of a white diffuse surface under the same light
	glintReactive: { from: number; to: number; max: number };
	// the reconstruction filter's Gaussian width (input pixels), running and while the scene rests
	sigma: number;
	sigmaRest: number;
	// the output's sharpening: taa's, and taau's at an upscale of 1 and of 2 per axis (eased between).
	// High's taau at its baseline's 1.5 sat 9 % sharper than the ground truth at rest (overshoot beyond
	// its range about 4x fsr3's) and flickered more in motion; swept on the Lanczos history fetch
	// (TemporalHistory.ts) from 0.75 to 2 stops at 1.5, 1.25 (1.5 / 1.0) was the softest that kept the
	// clips' ghosting (0.137 against 0.136): at rest FLIP against the ground truth 0.049 -> 0.045, the
	// sharpness 1.09 -> 1.03, the overshoot halved; in motion the shimmer (TI) and high-pass flicker
	// 3 % and 5 % lower, at fsr3's, FLIP unchanged, the sharpness 0.95 -> 0.90 of the ground truth's.
	// Softer still raised the pans' ghosting (0.088 -> 0.095 with none).
	sharpness: { taa: number; taau: [ number, number ] };
	// more sharpening where a surface moves on screen (the history's resampling softens moving detail):
	// RCAS stops taken off the sharpening at `speed` output pixels a frame and over, per surface class
	// (as classFrames; all 0: none, and the canvas pass reads no motion; read at load)
	sharpenMoving: { stops: number[]; speed: number };
	// the sky (far depth) resolved from the current frame without a history
	skyPassThrough: boolean;
	// accumulate in linear light (the display curve undone) rather than display values
	linear: boolean;
	// a resting scene (paused clock, still camera): frames rendered before the jitter sequence
	// starts, and the jitter sequence's length, after which the image holds
	restSettle: number;
	restSequence: number;
	// the cheaper resolve (fewer taps, no reactive mask) for the lowest tier
	lite: boolean;
	// the ground truth's jitter count (`aa=gt`, debug builds: the rest's sequence at the canvas's
	// resolution, box-filtered): 64 for a still image, 32 (`tier.taa.gtJitters=32`) per frame of a clip
	gtJitters: number;
	// FSR 3 (`aa=fsr3`, post/fsr3/Fsr3Upscaler.ts): its RCAS sharpness (0 the least, RCAS at two
	// stops of attenuation, 1 the most), the rest's jitter sequence (frames from FSR's reset at the rest's start,
	// repeating FSR's phase sequence), and whether the reactive-mask generator runs. The sequence is
	// the shortest multiple of the paused page's phase count (18: 4K output rendered at 1440p) after
	// which a rest of twice its length differs from it by more than one level in under 0.1 % of the
	// pixels on three close views: 126 (7 phases; 108 left 0.16 % on the meadow, 126 0.075 %).
	// FSR blends each frame into a running history rather than averaging the sequence, so its rest
	// needs far more frames than taa's 32 (measured at 4K output; a 1600x900 page has phase 18 too).
	// The sharpness is 0, not AMD's sample's 0.8: against the ground truth's rest on twelve crops of
	// three views (4K output), 0.8 sharpened past it (its mean gradient 1.11-1.29 times the ground
	// truth's on the meadow) and raised the error by 15 % (1 by 35 %); 0 lands within 5 % of the
	// ground truth's gradient and is as close as RCAS turned off, and it lowered the cliff edge's
	// flicker against the sky by about a fifth.
	fsr3: { sharpness: number; restSequence: number; autoReactive: boolean };
}

// Each setting names who reads it. The table is closed: it is the render pipeline's, a new setting
// is a new field here, and a registered definition (a plant kind, a light source, a pass of the frame)
// names the field that governs its cost, or that it has none (Extensions.ts `cost`).
export interface QualityProfile {

	// ---- load
	// the scene's anti-aliasing (post/Antialiasing.ts; `?aa` overrides it) and the temporal modes'
	// resolve
	// read by: post/Antialiasing.ts as it loads (`AA_MODE`, and from it `TEMPORAL`, `SCENE_SAMPLES`)
	aa: AAMode;
	// read by: RenderStages.ts createSeaAndPipeline, which hands it to the Pipeline (TemporalAA.ts,
	// TemporalRest.ts, fsr3/Fsr3Upscaler.ts)
	taa: TemporalSettings;
	// read by: RenderStages.ts setUpRenderer (whether the GPU is timed), the app's frame sizing
	// (post/RenderScale.ts `renderPixelRatio`, post/DynamicResolution.ts) and the tier's detection
	render: RenderScaleRule;
	// read by: RenderStages.ts createSeaAndPipeline (`fftAnisotropy`, the ocean FFT's targets) and
	// createWater (ocean/Water.ts: the reflection's rays, the storm foam)
	water: WaterQuality;
	// the frame graph's structural options (the haze's froxel grid); the rest keep their defaults
	// read by: RenderStages.ts createSeaAndPipeline (`postOptionsFromURL`, post/Pipeline.ts)
	post: Pick<PostOptions, 'hazeGrid'>;
	// the screen-space ambient occlusion and the opaque depth prepass it reads (post/GTAO.ts; built
	// only where the meadow's screen-space shadows run, at 4x outside the temporal modes; `flags=gtao`
	// overrides it; the screen-space shadows over the whole scene build that prepass too, post/MeadowShadows.ts)
	// read by: post/GTAO.ts as it loads (`GTAO`)
	ao: boolean;
	// the rain veil's shells and the sea's ripple layers
	// read by: RenderStages.ts createSeaAndPipeline (`setRainShading`: weather/Rain.ts, ocean/Water.ts)
	rainShading: { veilShells: number; rippleLayers: number };
	// the environment's PMREM source width (fixes the atlases' size before the first bake)
	// read by: RenderStages.ts createEnvironment (`Sky.setEnvironmentSource`)
	environmentSource: number;
	// the frame rate the loop holds on a display faster than that (post/FramePacer.ts: 60 renders on
	// every second refresh at 120 Hz), or 0 to render on every refresh
	// read by: the app's frame loop (post/FramePacer.ts)
	frameRate: number;

	// ---- runtime: all read by RenderStages.ts `applySkyQuality`, which applies them to the parts
	// named below, and kept as the app's runtime quality (`runtimeQuality`), which the frame reads
	// applied to: `Sky.setCloudView`
	cloudView: CloudViewScale;
	// the cloud panorama's size and its refresh pattern (one texel of every strideX x strideY per frame)
	// applied to: sky/Clouds.ts `setDome`
	cloudDome: { width: number; height: number; strideX: number; strideY: number };
	// applied to: sky/Clouds.ts `setMarch`
	cloudMarch: CloudMarch;
	// the share of the cloud shadow map's rows refreshed per frame
	// applied to: sky/Clouds.ts `setShadowRefresh`
	cloudShadowRows: number;
	// seconds of wall time between periodic sky bakes, and the environment's PMREM on every nth one
	// read by: the app's frame (the bakes' cadence); applied to: `Sky.setEnvironmentCadence`
	skyUpdateInterval: number;
	environmentEvery: number;
	// the refracted sun's map: the least interval between redraws (s) and the key light's turn (deg)
	// that redraws it
	// read by: the app's frame (the interval); applied to: lighting/RefractedSun.ts `angleGate`
	refractedSun: { interval: number; angleGate: number };
	// applied to: weather/Rain.ts `setQuality` (the drops drawn)
	rainDrops: number;

}

const TAA: TemporalSettings = {
	framesStill: 24, framesStillWater: 12, framesMoving: 5, speedFull: 16, gammaStill: 1.25, gammaMove: 0.75,
	classFrames: [ 1, 1, 0.4, 0.8, 0 ], classGamma: [ 0, 0.25, 0, - 0.25, 0 ], waterFar: [ 150, 600 ], glintReactive: { from: 2, to: 8, max: 1 },
	sigma: 0.42, sigmaRest: 0.30,
	sharpness: { taa: 1.0, taau: [ 1.5, 1.0 ] }, sharpenMoving: { stops: [ 0, 0, 1, 0, 0 ], speed: 4 }, skyPassThrough: true, linear: true, restSettle: 8, restSequence: 32, lite: false,
	gtJitters: 64,
	fsr3: { sharpness: 0, restSequence: 126, autoReactive: false },
};

export const QUALITY: Readonly<Record<QualityTier, Readonly<QualityProfile>>> = {
	high: {
		aa: 'msaa',
		taa: { ...TAA },
		// 1440p's pixels at the baseline (the standard upscales them, a 2K display is native) and 1080p's
		// at the floor; a display of 1080p or fewer pixels drops at most a quarter per axis
		render: { dynamic: { max: 2, baseline: 2560 * 1440, floor: 1920 * 1080, floorShare: 0.75, step: 0.05 } },
		water: { ssrIterations: 48, ssrMinFresnel: 0.05, stormFoam: 'full', fftAnisotropy: 4 },
		post: { hazeGrid: { width: 160, height: 90, depth: 64 } },
		ao: true,
		rainShading: { veilShells: 3, rippleLayers: 2 },
		environmentSource: 512,
		frameRate: 60,
		cloudView: { display: 0.45 },
		cloudDome: { width: 1024, height: 256, strideX: 4, strideY: 8 },
		cloudMarch: { viewSteps: 256, panoramaSteps: 128, lightTaps: 6, rainSteps: 16 },
		cloudShadowRows: 1 / 8,
		skyUpdateInterval: 0.5,
		environmentEvery: 1,
		refractedSun: { interval: 0.08, angleGate: 0.03 },
		rainDrops: 49152,
	},
	medium: {
		aa: 'taau',
		// TAAU's sharpening softer than high's (higher is softer)
		taa: { ...TAA, sharpness: { taa: TAA.sharpness.taa, taau: [ 2.0, 1.5 ] } },
		render: { budget: 1600 * 900 },
		water: { ssrIterations: 32, ssrMinFresnel: 0.05, stormFoam: 'full', fftAnisotropy: 4 },
		post: { hazeGrid: { width: 128, height: 72, depth: 64 } },
		ao: false,
		rainShading: { veilShells: 3, rippleLayers: 2 },
		environmentSource: 256,
		frameRate: 60,
		cloudView: { render: 0.5 },
		cloudDome: { width: 1024, height: 256, strideX: 8, strideY: 8 },
		cloudMarch: { viewSteps: 192, panoramaSteps: 128, lightTaps: 5, rainSteps: 16 },
		cloudShadowRows: 1 / 8,
		skyUpdateInterval: 1.2,
		environmentEvery: 2,
		refractedSun: { interval: 0.12, angleGate: 0.05 },
		rainDrops: 32768,
	},
	low: {
		aa: 'msaa',
		taa: { ...TAA, lite: true, framesStill: 12 },
		render: { budget: 1280 * 720 },
		// (the merged storm foam turned the crests' streaks into scattered lace: kept as on high)
		water: { ssrIterations: 32, ssrMinFresnel: 0.05, stormFoam: 'full', fftAnisotropy: 4 },
		post: { hazeGrid: { width: 96, height: 54, depth: 64 } },
		ao: false,
		rainShading: { veilShells: 2, rippleLayers: 1 },
		environmentSource: 256,
		// light enough to render on most refreshes of a fast display: held to 60 Hz there, its
		// frames finished right at the compositor's deadline and were shown one or three refreshes
		// apart in turn (a quarter of them), where unpaced they keep the refresh's own steps
		frameRate: 0,
		cloudView: { render: 0.4 },
		cloudDome: { width: 768, height: 192, strideX: 8, strideY: 8 },
		cloudMarch: { viewSteps: 128, panoramaSteps: 96, lightTaps: 3, rainSteps: 10 },
		cloudShadowRows: 1 / 8,
		skyUpdateInterval: 2,
		environmentEvery: 2,
		refractedSun: { interval: 0.2, angleGate: 0.1 },
		rainDrops: 16384,
	},
};

export function isQualityTier( value: unknown ): value is QualityTier {

	return typeof value === 'string' && ( QUALITY_TIERS as readonly string[] ).includes( value );

}

/** The tier this page builds: the one its entry chose (PageSettings.ts), else the `quality` option's
 * (a tool's installed settings), else high. */
export function loadTier(): QualityTier {

	const tier = settings.tier ?? settings.get( 'quality' );
	return isQualityTier( tier ) ? tier : DEFAULT_QUALITY;

}

/** The tier's profile with `?tier.<setting path>=<value>` options over it, for A/B captures of one
 * setting (`tier.post.hazeGrid.width=128`, `tier.water.ssrIterations=64`; `overrides` the app's).
 * A path must name an existing setting; numbers, booleans and comma lists are parsed. */
export function qualityProfile( tier: QualityTier, overrides: ReadonlyArray<readonly [ string, string ]> = settings.prefixed( 'tier.' ) ): QualityProfile {

	// a JSON copy, not structuredClone: that keeps the table's shared objects shared (the tiers'
	// `taa` copies share TAA's nested arrays), so an override of one setting would change its siblings too
	const profile = JSON.parse( JSON.stringify( QUALITY[ tier ] ) ) as QualityProfile;
	for ( const [ key, raw ] of overrides ) {

		const path = key.slice( 5 ).split( '.' );
		let node = profile as unknown as Record<string, unknown>;
		for ( const name of path.slice( 0, - 1 ) ) node = node?.[ name ] as Record<string, unknown>;
		const last = path[ path.length - 1 ];
		if ( ! node || typeof node !== 'object' || ! ( last in node ) ) throw new RangeError( `Unknown quality setting: ${ key }` );
		const parse = ( v: string ): unknown => v === 'true' ? true : v === 'false' ? false : v !== '' && Number.isFinite( Number( v ) ) ? Number( v ) : v;
		node[ last ] = Array.isArray( node[ last ] ) ? raw.split( ',' ).map( parse ) : parse( raw );

	}
	return profile;

}
