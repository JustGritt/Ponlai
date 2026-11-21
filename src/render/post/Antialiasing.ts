// The scene's anti-aliasing mode, chosen once at load, that materials and the graph are built for.
//
// The scene's anti-aliasing, chosen once at load like a game's AA setting: what the frame graph,
// the vegetation and the post chain are built for. The choice is compiled into materials (alpha to
// coverage or a dither / alpha test) and into the frame graph (the scene's sample count, its
// passes), so it changes only with a reload (the GUI's control reloads with `aa=`).
//
// Modes:
// - `msaa`: the scene multisampled 4x, the vegetation's cut-outs and thin blades by alpha to
//   coverage, upscaled by FSR 1 when the render is below the canvas (the high tier's at 4K output, 1440p;
//   at 2K output it renders natively: Quality.ts);
// - `off`: one sample, leaves alpha-tested and grass blades dithered (`msaa=1` before);
// - `taa`, `taau`: temporal anti-aliasing at the render's size, and temporal upsampling to the
//   canvas: one sample, jittered, with motion vectors (Jitter.ts, Motion.ts), resolved against a
//   history (TemporalAA.ts). A tier's setting may ask for them (the parked medium tier runs taau,
//   Quality.ts), and `aa=` or the GUI choose them on any tier. Their scene materials build one
//   shader per MRT, through three's cache keys patched for it (three/ThreePatches.ts, which
//   throws on a three it is not written for);
// - `fsr3`: AMD's FSR 3.1 upscaler (post/fsr3/, Fsr3Upscaler.ts): one sample, jittered with FSR's
//   own sequence, with motion vectors, like the temporal modes, but resolved by FSR at AMD's
//   placement: on the composite's HDR image, before the bloom and the grade, which run on FSR's
//   output at the canvas's size (Pipeline.ts). It has no TemporalAA (`RESOLVE` false), draws the
//   rain before the composite as the ground truth does (`AFTER_RESOLVE` false) and needs the
//   composite pass (`post=inline-composite` is refused with a warning: msaa);
// - `gt` (debug builds only): the ground truth the temporal modes are measured against, the
//   temporal graph rendered at the canvas's resolution whose image at rest averages many jittered
//   frames with a box filter (TemporalAA.ts, TemporalRest.ts): 64 by default, `tier.taa.gtJitters`.
//
// WebGPU multisamples with 1 or 4 samples only (three turns any other count into 1), so there is
// no 2x: `msaa=1` and `msaa=2` are read as `off`, `msaa=4` as `msaa`.
//
// The mode comes from the URL's `aa`, else its `msaa` alias, else `msaa` on a paused page (the test
// mode), else the quality tier's `aa` setting (`QualityProfile.aa`, with `tier.aa=` over it), else
// `msaa`: the URL wins over the test mode and the tier, and
// `AA_SOURCE` says which of them chose (main.ts reports it as `aaSource`). `gt` is taken from the
// URL in a debug build only, never from a tier. The tier is the one the app's entry module chose
// (PageSettings.ts, Quality.ts `loadTier`), installed before any module loads, so modules that read
// the mode when they load (vegetation/Coverage.ts, Grass.ts) take the app's. Code that differs by
// mode reads this module's exports (`TEMPORAL`, `SCENE_SAMPLES`, `ALPHA_TO_COVERAGE`), never the
// tier: `aa=` picks any mode on any tier.
import { loadTier, qualityProfile, type QualityProfile, type TemporalSettings } from '../Quality.ts';
import { settings, type SettingsQuery } from '../Settings.ts';

export const AA_MODES = [ 'msaa', 'taa', 'taau', 'fsr3', 'off', 'gt' ] as const;
export type AAMode = typeof AA_MODES[ number ];
/** Where the app's mode came from: the URL (`aa` or its `msaa` alias), the tier's setting, or neither. */
export type AASource = 'url' | 'test' | 'tier' | 'default';
export type SceneSampleCount = 1 | 4;

export interface AAPlan {
	mode: AAMode;
	// the opaque scene's (and the water's and rain's) sample count
	sceneSamples: SceneSampleCount;
	// whether the frame is resolved against a history (jitter, motion vectors, the resolve)
	temporal: boolean;
	// how a render below the canvas reaches it
	upscale: 'fsr1' | 'taau' | 'fsr3' | 'none';
}

export function isAAMode( value: unknown ): value is AAMode {

	return typeof value === 'string' && ( AA_MODES as readonly string[] ).includes( value );

}

/** The mode requested by an app's options, and where it came from: `aa`, else the `msaa` alias,
 * else the tier's `aa` setting, else `msaa`. An unknown value warns and falls through to the next
 * source; so does `gt` outside a debug build (`debug`: the dev server or `?debug`), and `gt` from a
 * tier (the ground truth is a measuring tool, never a tier's default). */
export function requestedAAMode( params: SettingsQuery, profile: Pick<QualityProfile, 'aa'> | null, debug = false ): { mode: AAMode; source: AASource } {

	const aa = params.get( 'aa' );
	if ( aa !== null ) {

		if ( aa === 'gt' && ! debug ) console.warn( 'aa=gt (the ground truth) exists in debug builds only (?debug)' );
		else if ( isAAMode( aa ) ) return { mode: aa, source: 'url' };
		else console.warn( `Unknown aa=${ aa }: expected one of ${ AA_MODES.join( ', ' ) }` );

	}
	const msaa = params.get( 'msaa' );
	if ( msaa !== null ) {

		if ( msaa === '1' || msaa === '2' ) return { mode: 'off', source: 'url' };
		if ( msaa === '4' ) return { mode: 'msaa', source: 'url' };
		console.warn( `Unknown msaa=${ msaa }: expected 1 or 4` );

	}
	// The test mode (ARCHITECTURE.md, rule 4) is a paused page under MSAA, whatever the tier runs:
	// every capture, parity and verify tool loads its pages paused, and their byte-for-byte checks hold under MSAA.
	if ( params.has( 'paused' ) || params.get( 'clock' ) === 'paused' ) return { mode: 'msaa', source: 'test' };
	const tier = profile?.aa;
	if ( tier !== undefined ) {

		if ( tier === 'gt' ) console.warn( 'The quality tier\'s aa setting asks for gt: the ground truth is chosen by the URL only (aa=gt, debug builds); running msaa' );
		else if ( isAAMode( tier ) ) return { mode: tier, source: 'tier' };
		else console.warn( `Unknown quality aa setting ${ String( tier ) }: expected one of ${ AA_MODES.join( ', ' ) }` );

	}
	return { mode: 'msaa', source: 'default' };

}

/** What a mode builds. */
export function aaPlan( requested: AAMode ): AAPlan {

	let mode: AAMode = requested;
	if ( mode === 'fsr3' && settings.get( 'post' )?.split( ',' ).some( ( s ) => s === 'legacy' || s === 'inline-composite' ) ) {

		console.warn( 'aa=fsr3 takes the composite pass\'s HDR image, which post=inline-composite (and post=legacy) leave out: running msaa' );
		mode = 'msaa';

	}
	const temporal = mode === 'taa' || mode === 'taau' || mode === 'fsr3' || mode === 'gt';
	return {
		mode,
		sceneSamples: mode === 'msaa' ? 4 : 1,
		temporal,
		// (the ground truth renders at the canvas's resolution: nothing to upscale)
		upscale: mode === 'taau' ? 'taau' : mode === 'fsr3' ? 'fsr3' : mode === 'gt' ? 'none' : 'fsr1',
	};

}

// The mode the app's tier (Quality.ts `loadTier`) and options ask for.
const params = settings;
const REQUEST = requestedAAMode( params, qualityProfile( loadTier() ), import.meta.env.DEV || params.has( 'debug' ) || params.has( 'profile' ) );
/** The mode the app asked for (before a refused one falls back). */
export const AA_REQUESTED: AAMode = REQUEST.mode;
/** Where the requested mode came from (the URL wins over the tier). */
export const AA_SOURCE: AASource = REQUEST.source;
export const AA_PLAN: Readonly<AAPlan> = Object.freeze( aaPlan( AA_REQUESTED ) );
/** The mode the app runs. */
export const AA_MODE: AAMode = AA_PLAN.mode;
export const TEMPORAL: boolean = AA_PLAN.temporal;
/** Whether FSR 3 resolves the frame (post/fsr3/Fsr3Upscaler.ts): temporal, with no TemporalAA. */
export const FSR3: boolean = AA_MODE === 'fsr3';
/** Whether the temporal modes resolve their frames through TemporalAA.ts (not under fsr3, which
 * FSR 3 resolves); `taaResolve=off` (debug builds) shows the jittered frame as it is. */
export const RESOLVE: boolean = TEMPORAL && ! FSR3 && ! ( ( import.meta.env.DEV || params.has( 'debug' ) ) && params.get( 'taaResolve' ) === 'off' );
/** Whether the rain's streaks and the storm's spray draw after the resolve, at the canvas's
 * resolution and unjittered (post/AfterResolve.ts): under taa and taau. The ground truth keeps them
 * before its resolve, so its box filter over the jitters is the supersampled image they are measured
 * against. */
export const AFTER_RESOLVE: boolean = RESOLVE && AA_MODE !== 'gt';
export const SCENE_SAMPLES: SceneSampleCount = AA_PLAN.sceneSamples;
// Whether the vegetation's cut-outs and thin blades use alpha to coverage: with 4 samples only
// (one sample has no coverage to give, so the pipeline would ignore alpha and draw them solid).
export const ALPHA_TO_COVERAGE: boolean = SCENE_SAMPLES === 4;

/** The temporal resolve's sharpening for a mode, in RCAS's stops (0 the strongest, 2 none): TAA's
 * own, TAAU's eased from its 1:1 value to its 2:1 value as the upscale `ratio` (canvas over render,
 * per axis) goes from 1 to 2. The ground truth is not sharpened (2: none). */
export function temporalSharpness( settings: TemporalSettings, mode: AAMode, ratio: number ): number {

	const s = settings.sharpness;
	if ( mode === 'gt' ) return 2;
	if ( mode !== 'taau' ) return s.taa;
	const t = Math.min( Math.max( ratio - 1, 0 ), 1 );
	return s.taau[ 0 ] + ( s.taau[ 1 ] - s.taau[ 0 ] ) * t;

}
