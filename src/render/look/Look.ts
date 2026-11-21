import type { DayKey, DayKeys, KeyValues } from '../post/DayKeys.ts';
import type { GradeChanges } from '../post/GradeLut.ts';
import type { ToneCurve } from '../post/ToneCurve.ts';
import type { SwellParams } from '../ocean/OceanSpectrum.ts';

// The look the render pipeline draws with: the types of the numbers chosen for a rendered image.
//
// The look the render pipeline draws with: the numbers chosen for a rendered image, one set per
// system, handed in by whoever owns the image. The render pipeline holds the types (here), the code that
// reads the numbers and registers their tunables (the modules beside this one: AmbientLook.ts,
// ExposureLook.ts, DisplayLook.ts, HazeLook.ts, SkyLook.ts) and a neutral set (NeutralLook.ts). The
// island's numbers are the island's (packages/island/look/). An app picks one set (the island's
// in PageSettings.ts, the test scenes' in ScenesBoot.ts) and hands it to the render pipeline's
// stages; a tool hands the set it wants to what it creates.
//
// No module holds a look: every system takes its tables as an argument where it is created, so two
// made from two sets in one process hold their own numbers (tools/render/verify-look-params.mjs):
// the exposure, the day's windows, the low key's lift and the sky's tables, where the render
// pipeline's sky stage creates them (ExposureLook.ts `createExposure`, DayLook.ts
// `createDayWindows`, AmbientLook.ts `createKeyLift`, SkyLook.ts `createSkyTables`, which the sky's
// state for a frame, its bakes and the haze's key read); the ambient grade's table and its gains
// under overcast's deck, where the light stage creates the grade's controller (AmbientLook.ts
// `createAmbientTables`, lighting/AmbientGrade.ts `createAmbientGrade`), which the ambient light's
// relight reads too; the haze's keys, rows and per-sky scales, where the sky stage creates them
// (HazeLook.ts `createHazeTables`), which weather/HazeKeys.ts `hazeKeyAt` reads by the hour and the
// weather; the ambient occlusion's floor and the meadow canopy's media, which every material's
// shader reads, in the shared uniforms (shared.ts `createShared`); the local lights' gain
// (lighting/LocalLights.ts `createLocalLights`, from the stage `createLight`); the sea's detail and
// distant swell (ocean/SeaDetail.ts, ocean/OceanFFT.ts, from the stage `createSeaAndPipeline`); the
// display's tables, which the frame's passes read each frame and build the display chain and the
// bloom from (DisplayLook.ts `createDisplayTables`, from the same stage); and the wind on the
// plants, in the shared uniforms' wind (weather/Wind.ts `windResponse`, from `createShared`). On a
// debug page two of a kind still share their tunables by name (debug/Tune.ts), and the shared
// uniforms, the grade's uniforms and the light table stay one per page.
//
// Not everything chosen rather than measured passes through the look yet. Three sets stay the
// render pipeline's: the weathers' presets with the wind's levels (sky/Weather.ts, which the
// headless game reads too), the sea's chlorophyll (ocean/SeaOptics.ts, which the chunk workers
// read, so a change to it rebuilds the world's cache) and the quality table (Quality.ts). A few
// numbers that sit between a method's constants and the look stay in their modules too: the ambient
// occlusion kernel's reach (post/GTAO.ts), the meadow canopy's depths (lighting/MeadowCanopy.ts),
// the march of the meadow's screen-space shadows (post/MeadowShadows.ts), the gust field's scale and evolution
// (weather/Wind.ts) and each plant part's answer to the wind (vegetation/Deform.ts).
//
// Headless: types and one list of names, no three.js and nothing of the app.

/** A span of the sun's elevation (degrees): [ whole, none ] or [ from, to ], as its reader says. */
export type Span = readonly [ number, number ];
export type Rgb = { r: number; g: number; b: number };
type Triple = readonly [ number, number, number ];

// ---- the sea (RenderStages.ts `createSeaAndPipeline`, read by ocean/SeaDetail.ts and ocean/OceanFFT.ts)

export interface SeaLook {
	/** The strengths of the sea's large-scale detail: the gusts' contrast, the slicks, the windrows. */
	detail: { gust: number; slick: number; streak: number };
	/** The distant swell under the weather's own sea. */
	swell: Readonly<SwellParams>;
	/** The share of Cox and Munk's mean square slope held below every pixel, as waves finer than any
	 * read of the cascades (0 to below 1; ocean/OceanFFT.ts `slopeBudget`). */
	subPixelSlope: number;
}

// ---- the day (DayLook.ts `createDayWindows` and ExposureLook.ts `createExposure`, from the stage
// `createSky`)

/** A dusk share's windows: the clear evening's, the clear morning's and the clouded sky's. */
export interface DuskWindow { evening: Span; morning: Span; clouded: Span }
export interface DayLook {
	/** The highest elevation at which the key may turn from the sun toward the moon. */
	handoffTop: number;
	/** The elevation at which the moon's key starts to rise, in full two degrees lower. */
	moonKeyTop: number;
	/** Where each of the dusk's shares comes in (DayLook.ts `DayWindows.dusk` says which is which). */
	duskWindows: {
		display: Pick<DuskWindow, 'evening' | 'morning'>;
		exposure: DuskWindow;
		stars: Pick<DuskWindow, 'evening'>;
		sky: DuskWindow;
		ambient: DuskWindow;
		nightGlow: DuskWindow;
		cloudMoon: DuskWindow;
	};
	/** The daylight's window in which a clouded sky's display and stars take the night. */
	cloudedDaylight: Span;
	/** How deep under all of it (stops, over 0) each night layer comes in at its window's day end. */
	duskDepth: { sky: number; ambient: number; stars: number; haze: number; cloudMoon: number };
	/** Over how many degrees at the window's night end the night layers ease in. */
	duskEase: number;
	/** The exposure by the sun's elevation: [ elevation, exposure ], ascending. */
	exposure: readonly ( readonly [ number, number ] )[];
	/** The stops a clear night's metered exposure lies above the twilight's. */
	nightMeterLift: number;
	/** The night's own gain over the night sky's exposure. */
	nightExposureGain: number;
	/** The exposure the night sky's own light keeps its displayed brightness at. */
	nightSkyExposure: number;
}

// ---- the environment light (AmbientLook.ts)

export interface AmbientSettings {
	skySpecular: number;
	skyBounce: number;
	sunBounce: number;
	directionality: number;
	aoStrength: number;
}
/** The ambient grade's columns, the skies of the weather's blend. */
export type AmbientColumn = 'clear' | 'fair' | 'cloudy' | 'overcast';
/** A row of the ambient grade: per sky, its changes over the parameters at no grade. */
export type AmbientRowLook = { at: number } & Partial<Record<AmbientColumn, Partial<AmbientSettings>>>;
export interface AmbientLook {
	/** The grade by the sun's elevation (degrees, descending), setting side and rising side. */
	grade: readonly AmbientRowLook[];
	gradeRising: readonly AmbientRowLook[];
	/** A gain on the grade's `sunBounce` at every hour. */
	sunBounceGain: number;
	/** The weather's gain on the land's ambient light under overcast's deck: the sun's part, the moon's. */
	overcastLand: number;
	overcastMoonLand: number;
	/** The low key's lift: the shading elevation at the key's (ascending), and the corner's half width. */
	keyLift: { rows: readonly { at: number; shading: number }[]; join: number };
}

// ---- the exposure (ExposureLook.ts)

export interface ExposureKey extends Record<string, number> {
	// κ, stops: where a white diffuser under the frame's typical light lands in the exposed image
	key: number;
	// κ_H, stops: the highest the frame's bright band (Y's 90th to 97th percentile) lands
	highlight: number;
	// β: the sky's share of the meter (its own exposure's, against the scene's)
	sky: number;
	// τ, seconds: the adaptation's time constant while the exposure rises (the view grew darker)
	adaptUp: number;
	// τ, seconds: the adaptation's time constant while the exposure falls (the view grew brighter)
	adaptDown: number;
}
/** A key some of whose values are another key's: `shares` names, per value, the key (by `name`)
 * it reads that value from, so the two hold one number and one tunable. */
export type SharingKey<T extends KeyValues> = DayKey<T> & { readonly shares?: Partial<Record<keyof T, string>> };
export interface ExposureLook {
	keys: {
		day: readonly SharingKey<ExposureKey>[];
		rising: readonly SharingKey<ExposureKey>[];
		cloudy: readonly SharingKey<ExposureKey>[];
		night: ExposureKey;
	};
	/** Overcast's key on top of the clouded one (stops), toward a high and toward a low sun. */
	overcastKey: number;
	overcastKeyLow: number;
	/** Where overcast's key passes from its low sun's value to its high sun's (degrees). */
	overcastKeySpan: Span;
	/** The share of a clear sky's light the overcast deck lets onto the land by night (0..1]. */
	deckLandThrough: number;
	/** The share of that darkness an overcast night keeps on screen. */
	overcastNight: number;
	/** Where overcast holds the clouded keys from (degrees). */
	overcastHeld: number;
}

// ---- the display (DisplayLook.ts)

export type BloomKey = {
	threshold: number; knee: number; share: number; cap: number;
	skyThreshold: number; skyKnee: number; skyShare: number; skyCap: number;
	w0: number; w1: number; w2: number; w3: number; w4: number; w5: number; w6: number; w7: number;
};
export type DesaturationKey = { amount: number; cool: number };
export interface DisplayLook {
	shadowContrast: DayKeys<{ pivot: number; slope: number }>;
	desaturation: DayKeys<DesaturationKey>;
	coolShare: { from: number; to: number };
	highlightGrey: { amount: number; from: number; to: number };
	/** The bloom's and the vignette's strengths. */
	strengths: { bloom: number; vignette: number };
	vignette: { inner: number; outer: number; depth: number };
	whiteBalance: DayKeys<Rgb> & { readonly night: Rgb };
	whiteOvercast: Rgb;
	toneCurve: Readonly<ToneCurve>;
	/** The base grade, as its changes over no grade (post/GradeLut.ts NEUTRAL_GRADE). */
	baseGrade: GradeChanges;
	/** The bloom's keys, and the key its uniforms start from. */
	bloom: DayKeys<BloomKey>;
	bloomStart: BloomKey;
	purkinje: { retina: number; strength: number };
}

// ---- the haze (HazeLook.ts `createHazeTables`, from the stage `createSky`, read by
// weather/HazeKeys.ts)

export interface HazeKey {

	// the deck's share (Weather.ts `deckLight` by day, the clouded share by night):
	// under it the air is lit level, not as a clear sky's (VolumetricHaze.ts)
	deck: number;
	// overcast's share of that deck (a parked sky counting as overcast): its sky darker away from the
	// sun and toward the horizon (VolumetricHaze.ts)
	overcastDeck: number;
	amount: number;
	sigmaRed: number;
	sigmaGreen: number;
	sigmaBlue: number;
	desaturate: number;
	// the weather's gain over both shares of the light (1 in every key)
	gain: number;
	mist: number;
	sunRayleigh: number;
	rayleighWeight: number;
	mieWeight: number;
	top: number;
	width: number;
	aerosolShare: number;
	aerosolHeight: number;
	// the aerosol layer's top (m; Infinity for none): a low deck's base
	aerosolTop: number;
	skyDepth: number;
	// the share of the air's in-scattered light laid over the land it fogs: the sea and the sky keep
	// all of it, its transmittance is the same everywhere (weather/VolumetricHaze.ts `apply`)
	groundLight: number;
	// the fog layer, optional so a key written elsewhere without them holds none:
	// the share of the ground layer that is the fog layer (its density from `visibility`, its top
	// `fogTop` and the width of its edge `fogWidth`), the share of the ground layer's light that is the droplets' (VolumetricHaze.ts), how
	// far the extinction turns neutral, the droplets' multiple scattering (Wrenninge's a), the
	// visibility at the sea (km) and the banks' threshold at the layer's top (FogNoise.ts `high`)
	fogLayer?: number;
	droplet?: number;
	neutral?: number;
	scatter?: number;
	fogTop?: number;
	fogWidth?: number;
	visibility?: number;
	fogCoverHigh?: number;

}
export type FullKey = Required<HazeKey>;
/** The hours a haze look gives a key for: four with a medium's shape of their own (day, golden,
 * night, mist), the morning, and the twilights before dawn and after sunset. */
export const HAZE_KEY_NAMES = [ 'day', 'golden', 'night', 'mist', 'morning', 'earlyDawn', 'beforeDawn', 'afterSunset', 'afterSunsetDeep', 'afterSunsetClouded' ] as const;
export type HazeKeyName = typeof HAZE_KEY_NAMES[ number ];
/** Rows by the sun's elevation (degrees, ascending), each naming its key. */
export type HazeRows = readonly ( readonly [ number, HazeKeyName ] )[];
export type FogFields = { fogLayer: number; neutral: number; scatter: number; fogTop: number; fogWidth: number; visibility: number };
export type HazeScale = { sigmaRed: number; sigmaGreen: number; sigmaBlue: number; aerosolScale: number; aerosolHeight: number; topScale: number; widthScale: number };
export interface HazeLook {
	keys: Record<HazeKeyName, FullKey>;
	/** The rising sun's rows, the setting sun's, and the setting sun's under cloud (two rows at least). */
	rising: HazeRows;
	setting: HazeRows;
	settingClouded: HazeRows;
	/** The fog layer's fields where a key holds none. */
	noFog: FogFields;
	/** The deep sea fog, a switch beside the weather. */
	seaFog: { neutral: number; scatter: number; fogTop: number; fogWidth: number; visibility: number; coverHigh: number };
	/** Per sky of the weather's blend (clear, fair, cloudy, overcast): the share of the light's
	 * colour kept, the gain, the scales on the hour's medium, whether a deck ends the aerosol layer
	 * at its base, and the sun's share in the haze's Rayleigh light. */
	keep: readonly number[];
	gain: readonly number[];
	scales: readonly HazeScale[];
	decked: readonly boolean[];
	sunRayleigh: readonly number[];
}

// ---- the sky (SkyLook.ts)

/** A row of the light the sky lends the land: the colour it is drawn toward (linear, any level:
 * the reader takes it at unit luminance), how far, and the gain. */
export interface LandSkyRow { at: number; tint?: Triple; amount?: number; gain?: number }
/** A kind of sky as a row of the clouds' parameters; no `base` or `top`: the cloud model's shell. */
export type CloudSkyLook = { base?: number; top?: number; type: number; spread: number; farFade: number };
export type StreakKeys = { afterglow: Rgb; golden: Rgb; low: Rgb; predawn: Rgb; sunrise: Rgb; morning: Rgb; high: Rgb; night: Rgb };
export interface SkyLook {
	/** The colour a clouded sky's light to the land is drawn toward under a high sun, and how far. */
	cloudLightTint: readonly [ number, number, number, number ];
	/** The light the sky lends the land by the sun's elevation (degrees, descending): the setting
	 * side, the setting side under cloud and the rising side. */
	landSky: { setting: readonly LandSkyRow[]; clouded: readonly LandSkyRow[]; rising: readonly LandSkyRow[] };
	/** The clear-sky fill light (an A/B, off by default): the gain's extra share by elevation, its tint and amount. */
	clearSkyFill: readonly { at: number; share: number }[];
	clearSkyFillTint: readonly [ number, number, number, number ];
	/** The low sun's shadow tint (an A/B, off by default) and its colour. */
	lowSun: { full: number; top: number; amount: number };
	lowSunTint: Triple;
	/** How far a deck draws the key toward white at the same luminance. */
	keyDeckWhite: number;
	/** The sky's light against the sun's, and the clear day's gain over it. */
	skyLight: DayKeys<Rgb>;
	clearSky: DayKeys<{ gain: number }>;
	/** The clouds' row per sky of the blend (clear, fair, cloudy, overcast). */
	cloudSkies: readonly CloudSkyLook[];
	/** Per sky of the blend, each store's range under the moon: [ brightest, first percentile ], stops over the moonlight on the clouds. */
	storeRange: Record<'dome' | 'layer' | 'panorama', readonly Span[]>;
	/** The low sun's key: a gain by elevation (degrees, descending). */
	lowKey: readonly { at: number; gain: number }[];
	/** The distant sea's knee, times the sun's illuminance. */
	seaKnee: number;
	/** The key's hue: the caps of the depth that sets it (the setting sun's, the rising sun's), its
	 * white (red over green, blue over green) and the onsets of the air's change in each ratio. */
	keyHue: { depth: number; dawnDepth: number; white: readonly [ number, number ]; reddens: number; yellows: number };
	/** The thin cloud layer's colours on its two shares. */
	streakDirect: StreakKeys;
	streakAmbient: StreakKeys;
	/** The moon's key's colour (any level: its readers take it at unit luminance), and its red and
	 * blue over green as written. */
	moonKey: { colour: Triple; red: number; blue: number };
	/** The moonlit sky's colour and the night's ambient's, at unit luminance; the night's ambient's share of the moon's key. */
	moonSkyHue: Triple;
	nightAmbientHue: Triple;
	nightAmbientShare: number;
	/** The island's moonlight's red and blue over green. */
	moonHue: { red: number; blue: number };
	/** The share of a cloud layer's diffuse transmission the land keeps as a light along the key. */
	keyThroughCloud: number;
}

// ---- the ambient occlusion (shared.ts `createShared`, read by post/GTAO.ts `floorOcclusion`)

export interface OcclusionLook {
	/** The least ambient occlusion a surface keeps where its own and the screen's multiply. */
	floor: number;
}

// ---- the local lights (lighting/LocalLights.ts `createLocalLights`)

export interface LocalLightsLook {
	/** The gain on every source's own intensity. */
	gain: number;
}

// ---- the meadow's canopy (shared.ts `createShared`, read by lighting/MeadowCanopy.ts)

/** A canopy's medium: its extinction per metre along the key's ray, and the share of the key that
 * passes between its blades whatever the path. */
export type CanopyMedium = { density: number; open: number };
export interface MeadowCanopyLook {
	/** The level canopy over every point of the meadow. */
	canopy: CanopyMedium;
	/** The crowns that stand over their neighbours and over the ground. */
	cast: CanopyMedium;
}

// ---- the wind on the plants (shared.ts `createShared`, read by weather/Wind.ts `windResponse`)

export interface WindLook {
	/** The gusts' share of the wind's strength on the plants: S ( 1 + gustGain q ). */
	gustGain: number;
	/** The largest offset (m, each axis) a blade's gust sample takes from its root. */
	gustJitter: number;
	/** The plants' speed (m/s) by the mean wind 10 m up (m/s): the points of a line, ascending, two
	 * at least. */
	plantWind: readonly ( readonly [ number, number ] )[];
	/** The plants' speed (m/s) at which a limb takes a load of 1. */
	plantSpeed: number;
	/** How much quicker than its kind's rate a plant may sway (a share of the rate). */
	plantRateSpread: number;
}

// ---- the current look

export interface Look {
	sea: SeaLook;
	day: DayLook;
	ambient: AmbientLook;
	exposure: ExposureLook;
	display: DisplayLook;
	haze: HazeLook;
	sky: SkyLook;
	occlusion: OcclusionLook;
	localLights: LocalLightsLook;
	meadowCanopy: MeadowCanopyLook;
	wind: WindLook;
}
