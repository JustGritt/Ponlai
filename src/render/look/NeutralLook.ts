import type { BloomKey, DuskWindow, FullKey, Look, Rgb, Span, StreakKeys } from './Look.ts';

// The neutral look: every gain 1 and every grade off, for the test scenes' app.
//
// The neutral look: what the test scenes' app draws with (app/ScenesBoot.ts), and what
// `renderLook=neutral` loads on a debug page of the island's. It is no picture of its own: every gain is 1,
// every offset, strength and cast 0, no grade, no vignette, no bloom, nothing done for the night,
// and one key held all day in every table that is keyed by the sun. Where a number has no such
// identity it takes the plainest value its reader accepts, and the comment beside it says why.
// The tables keep the shapes the readers take (a keyed table holds one key, a table by elevation
// one row, a table per sky four equal rows). Numbers only: no three.js, nothing of the app.

const WHITE = (): Rgb => ( { r: 1, g: 1, b: 1 } );

// One window for every dusk share and every side of the day: from the end of civil twilight to
// the end of nautical twilight (-6 to -12 degrees), the convention for where the night comes in.
const DUSK: Span = [ - 12, - 6 ];
const WINDOW: DuskWindow = { evening: DUSK, morning: DUSK, clouded: DUSK };

// The bloom's key with nothing let in: no share and no weight on any level. The thresholds and
// knees only have to be positive (the knee divides).
const NO_BLOOM = (): BloomKey => ( {
	threshold: 1, knee: 0.5, share: 0, cap: 1,
	skyThreshold: 1, skyKnee: 0.5, skyShare: 0, skyCap: 1,
	w0: 0, w1: 0, w2: 0, w3: 0, w4: 0, w5: 0, w6: 0, w7: 0,
} );

// Clean air: no haze at all (amount 0), so nothing else in the key shows. The spectral weights
// are grey, the light's colour is kept, both shares' weights are 1; the medium's shape and the fog
// layer's fields only have to be valid (widths and the visibility divide), in round numbers.
const NO_FOG = { fogLayer: 0, neutral: 0, scatter: 0, fogTop: 10, fogWidth: 10, visibility: 1 };
const CLEAN_AIR = (): FullKey => ( {
	deck: 0, overcastDeck: 0, amount: 0, sigmaRed: 1, sigmaGreen: 1, sigmaBlue: 1, desaturate: 0, gain: 1, mist: 0, sunRayleigh: 1, rayleighWeight: 1, mieWeight: 1,
	top: 10, width: 10, aerosolShare: 0, aerosolHeight: 1000, aerosolTop: Infinity, skyDepth: 0, groundLight: 1,
	...NO_FOG, droplet: 0, fogCoverHigh: 0,
} );
const UNIT_SCALE = { sigmaRed: 1, sigmaGreen: 1, sigmaBlue: 1, aerosolScale: 1, aerosolHeight: 1, topScale: 1, widthScale: 1 };

const WHITE_STREAKS = (): StreakKeys => ( {
	afterglow: WHITE(), golden: WHITE(), low: WHITE(), predawn: WHITE(), sunrise: WHITE(), morning: WHITE(), high: WHITE(), night: WHITE(),
} );

export const NEUTRAL_LOOK: Look = {

	sea: {
		// no large-scale detail
		detail: { gust: 0, slick: 0, streak: 0 },
		// no distant swell (scale 0); the rest only has to be a valid spectrum: a fully developed
		// sea's shape (peak enhancement 1) under a light wind, along the local sea's heading
		swell: { scale: 0, windSpeed: 1, fetch: 100e3, directionOffset: 0, spread: 1, swell: 0, peakEnhancement: 1, shortWavesFade: 0 },
		// the cascades hold all of Cox and Munk's slope they can
		subPixelSlope: 0,
	},

	day: {
		// The key turns from the sun to the moon just under the horizon, the moon's key rising two
		// degrees lower: no identity, so whole degrees under the setting disk's fade.
		handoffTop: - 1,
		moonKeyTop: - 3,
		duskWindows: {
			display: WINDOW, exposure: WINDOW, stars: WINDOW, sky: WINDOW, ambient: WINDOW, nightGlow: WINDOW, cloudMoon: WINDOW,
		},
		// the clouded sky's night over the whole of the daylight's fall
		cloudedDaylight: [ 0, 1 ],
		// A depth of 0 stops would come in linearly, but the ease by stops divides by 1 - 2^-depth:
		// one stop is the shallowest whole depth.
		duskDepth: { sky: 1, ambient: 1, stars: 1, haze: 1, cloudMoon: 1 },
		duskEase: 0,
		// one exposure all day: 1
		exposure: [ [ 0, 1 ] ],
		nightMeterLift: 0,
		nightExposureGain: 1,
		// the one exposure, so the night sky's own light is neither dimmed nor lifted against it
		nightSkyExposure: 1,
	},

	ambient: {
		// one row with no change over the parameters at no grade
		grade: [ { at: 0 } ],
		gradeRising: [ { at: 0 } ],
		sunBounceGain: 1,
		// no gain under a deck, on the sun's part or the moon's (lighting/AmbientGrade.ts takes a moon's
		// part of 1 as none, with no ratio of the two)
		overcastLand: 1,
		overcastMoonLand: 1,
		// no lift: one row at the nadir, from which the shading holds its own elevation
		keyLift: { rows: [ { at: - 90, shading: - 90 } ], join: 0 },
	},

	exposure: {
		// One key all day and all night. Its κ is 0 stops (no offset). κ_H only guards a frame's
		// brightest band and has no identity: 16 stops never binds. The sky takes no share of the
		// meter of its own. The adaptation's time constant has no identity either (0 divides on a
		// paused frame): a tenth of a second.
		keys: ( () => {

			const values = { key: 0, highlight: 16, sky: 0, adaptUp: 0.1, adaptDown: 0.1 };
			const day = [ { at: 0, name: 'day', values } ];
			return { day, rising: day, cloudy: day, night: values };

		} )(),
		overcastKey: 0,
		overcastKeyLow: 0,
		// (any span: both ends hold 0)
		overcastKeySpan: [ 0, 1 ],
		deckLandThrough: 1,
		overcastNight: 0,
		// overcast holds nothing: from the nadir up every elevation is its own
		overcastHeld: - 90,
	},

	display: {
		shadowContrast: { day: [ { at: 0, name: 'day', values: { pivot: 0, slope: 0 } } ], night: { pivot: 0, slope: 0 } },
		desaturation: { day: [ { at: 0, name: 'day', values: { amount: 0, cool: 0 } } ], night: { amount: 0, cool: 0 } },
		// (no desaturation takes it; the edges only have to differ)
		coolShare: { from: 0, to: 1 },
		highlightGrey: { amount: 0, from: 0, to: 1 },
		strengths: { bloom: 0, vignette: 0 },
		vignette: { inner: 0, outer: 1, depth: 0 },
		whiteBalance: { day: [ { at: 0, name: 'day', values: WHITE() } ], night: WHITE() },
		whiteOvercast: WHITE(),
		// The tone curve cannot be the identity (post/ToneCurve.ts fixes its value at 1 and its slope
		// into white): the flattest set it solves is a slope of 1 at black, mid grey left where it is
		// (0.18) and a white of 32, five stops over the exposed 1. With that toe and mid grey a white
		// of 4, 8 or 16 has no solution without a pole.
		toneCurve: { toe: 1, mid: 0.18, white: 32 },
		baseGrade: {},
		bloom: { day: [ { at: 0, name: 'day', values: NO_BLOOM() } ], night: NO_BLOOM() },
		bloomStart: NO_BLOOM(),
		// (the retinal illuminance only has to be positive: the strength is 0)
		purkinje: { retina: 1, strength: 0 },
	},

	haze: {
		keys: {
			day: CLEAN_AIR(), golden: CLEAN_AIR(), night: CLEAN_AIR(), mist: CLEAN_AIR(), morning: CLEAN_AIR(),
			earlyDawn: CLEAN_AIR(), beforeDawn: CLEAN_AIR(), afterSunset: CLEAN_AIR(), afterSunsetDeep: CLEAN_AIR(), afterSunsetClouded: CLEAN_AIR(),
		},
		// one key all day (the rows' reader takes two rows at least)
		rising: [ [ - 90, 'day' ], [ 90, 'day' ] ],
		setting: [ [ - 90, 'day' ], [ 90, 'day' ] ],
		settingClouded: [ [ - 90, 'day' ], [ 90, 'day' ] ],
		noFog: NO_FOG,
		// The sea fog is a switch, off unless asked for: a plain grey fog a kilometre deep in sight,
		// in the same round numbers.
		seaFog: { neutral: 1, scatter: 0, fogTop: 10, fogWidth: 10, visibility: 1, coverHigh: 1 },
		keep: [ 1, 1, 1, 1 ],
		gain: [ 1, 1, 1, 1 ],
		scales: [ UNIT_SCALE, UNIT_SCALE, UNIT_SCALE, UNIT_SCALE ],
		decked: [ false, false, false, false ],
		// (the keys hold the sun's Rayleigh light whole; no sky adds to it)
		sunRayleigh: [ 0, 0, 0, 0 ],
	},

	sky: {
		cloudLightTint: [ 1, 1, 1, 0 ],
		landSky: { setting: [ { at: 0 } ], clouded: [ { at: 0 } ], rising: [ { at: 0 } ] },
		clearSkyFill: [ { at: 0, share: 0 } ],
		clearSkyFillTint: [ 1, 1, 1, 0 ],
		// (the elevations only have to differ: the amount is 0)
		lowSun: { full: 0, top: 1, amount: 0 },
		lowSunTint: [ 1, 1, 1 ],
		keyDeckWhite: 0,
		// the atmosphere's own light, at every hour
		skyLight: { day: [ { at: 0, name: 'day', values: WHITE() } ], night: WHITE() },
		clearSky: { day: [ { at: 0, name: 'day', values: { gain: 1 } } ], night: { gain: 1 } },
		// four equal skies: cumulus in the cloud model's own shell, the same everywhere, not thinned
		// with distance
		cloudSkies: [ { type: 1, spread: 0, farFade: 0 }, { type: 1, spread: 0, farFade: 0 }, { type: 1, spread: 0, farFade: 0 }, { type: 1, spread: 0, farFade: 0 } ],
		// No estimate of the clouds' range under the moon: each store centred on the moonlight itself.
		storeRange: {
			dome: [ [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ] ],
			layer: [ [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ] ],
			panorama: [ [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ] ],
		},
		lowKey: [ { at: 0, gain: 1 } ],
		// no knee: the sea's body follows the light up to the sun's whole illuminance
		seaKnee: 1,
		// The key takes the air's own colour: no cap on the depth that sets its hue, a white of 1,
		// and the air's change in each ratio whole from the first of its depth (an onset of 0 would
		// divide 0 by 0 where the air takes nothing).
		keyHue: { depth: Infinity, dawnDepth: Infinity, white: [ 1, 1 ], reddens: 1e-9, yellows: 1e-9 },
		streakDirect: WHITE_STREAKS(),
		streakAmbient: WHITE_STREAKS(),
		// a white moon (its readers take the colour at unit luminance, so the level is free)
		moonKey: { colour: [ 1, 1, 1 ], red: 1, blue: 1 },
		moonSkyHue: [ 1, 1, 1 ],
		nightAmbientHue: [ 1, 1, 1 ],
		// no ambient of the night's own
		nightAmbientShare: 0,
		moonHue: { red: 1, blue: 1 },
		// the land keeps a cloud layer's diffuse transmission as it is
		keyThroughCloud: 1,
	},

	occlusion: {
		// no floor: the occlusion is the surface's own times the screen's, down to 0
		floor: 0,
	},

	localLights: {
		// every source at its own intensity
		gain: 1,
	},

	meadowCanopy: {
		// No medium: the key crosses the canopy and the crowns whole (an extinction of 0), so the
		// meadow's own shadowing drops out and the share that passes between the blades has nothing to add.
		canopy: { density: 0, open: 0 },
		cast: { density: 0, open: 0 },
	},

	wind: {
		// no gusts on the plants: the wind's strength is the plants' speed itself
		gustGain: 0,
		// every blade reads the gust at its own root
		gustJitter: 0,
		// the plants take the wind 10 m up as it is
		plantWind: [ [ 0, 0 ], [ 1, 1 ] ],
		// A limb's full load has no identity. It takes the relation the speed stands in to the rest
		// of this table: the strongest level's wind (sky/Weather.ts, 11 m/s) as the plants take it,
		// at its full gust, which with the two rows above is 11.
		plantSpeed: 11,
		// every plant sways at its kind's rate
		plantRateSpread: 0,
	},

};
