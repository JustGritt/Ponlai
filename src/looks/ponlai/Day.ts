import type { DayLook } from '../../render/setup.ts';

// The island's day look: the key's turn to the moon, dusk windows, night layers and exposure.
//
// The island's day look (packages/island/look/: the island's numbers for the render pipeline's look
// tables, look/Look.ts; look/DayLook.ts `createDayWindows` makes the day's windows from these, and
// its comments say what each table is): where the key turns from the sun to the moon, the dusk's
// windows, how deep under all of it each night layer comes in and the ease at the window's night
// end, and the exposure by the sun's elevation with the night sky's own exposure. sky/DayCycle.ts's
// header says how the day runs.

// The exposure by the sun's elevation (degrees): [ elevation, exposure ], interpolated in log
// exposure and held beyond the ends. With the sun up, tidewater's auto exposure brings its frames to
// one metered level (its meter's centre-weighted log average: about 0.15 of its curve's input by day,
// falling through the low sun to about 0.03 at the horizon), and these bring the island views
// (toward and away from the sun, clear weather, the log mean of the two) to the same level at the
// same sun heights. After sunset tidewater's frame falls to near black, lit by its village; the
// twilight here keeps the brightness the island showed under the earlier grade (25 from -6 to -7
// degrees). Below -7 degrees the keys follow the metered exposure down to the night: each is 25
// times 2 to the stops the clear evening's island view meters at that height above its meter under
// the night these keys were first set for (log interpolated between them: within about half a stop of the
// metered curve), the full NIGHT_METER_LIFT by -12 degrees (the lowest row, the exposure's night).
const TWILIGHT_EXPOSURE = 25;
const nightKey = ( elevation: number, stops: number ) => [ elevation, TWILIGHT_EXPOSURE * 2 ** stops ] as const;
/** The stops a clear night's metered exposure lies above the twilight's exposure (25): the
 * moon's light at its magnitude against the sun's, metered under the night key
 * (look/ExposureLook.ts), measured on the clear midnight's island view with the lamps off. The
 * yunpeng's glow and every local light are scaled by 2 to minus this, so a lamp keeps its display
 * at night whatever the hour. It and the night keys below are measured, not derived: whenever the
 * night's light or its metering changes (the moon's level or its share through the dusk, the night
 * key, the meter), measure them again with tools/island/dusk-grid.mjs on the clear evening's night points
 * and midnight, `--night-split`, against a run of the page at the commit before the moon was lit
 * (`--compare`; the commit titled "perf(sky): rescale the cloud dome to a new power of two instead
 * of marching it again", the parent of "feat(sky): light the night by the moon as the sun lights
 * the day"): its compare.txt lists the lift at each point, the night keys' stops, and at midnight
 * the lift less this constant, which should stay within a few hundredths of a stop. */
const NIGHT_METER_LIFT = 12.44;

export const DAY: DayLook = {
	// Highest elevation (degrees) at which the key may turn from the sun toward the moon: below the
	// end of the sun's fade as its disk sinks into the sea (Radiometry.ts SUN_SETTING_RADIUS,
	// Atmosphere.sunlightAt).
	handoffTop: - 0.7,
	// Elevation (degrees) at which the moon's key starts to rise, in full two degrees lower: the
	// handoff has just turned the key to the moon, so it turns through zero light (`handoffTop` to here).
	moonKeyTop: - 2.5,
	// Where each of the dusk's shares comes in (look/DayLook.ts `dusk`). Every share is 0 at
	// +2.5 degrees and over.
	duskWindows: {
		display: { evening: [ - 11, - 7 ], morning: [ - 12, - 7 ] },
		exposure: { evening: [ - 11, - 7 ], morning: [ - 12, - 7 ], clouded: [ - 5, 0 ] },
		stars: { evening: [ - 11, - 7 ] },
		sky: { evening: [ - 11, - 7 ], morning: [ - 13, - 3 ], clouded: [ - 6, 2 ] },
		ambient: { evening: [ - 11, - 7 ], morning: [ - 12, - 7 ], clouded: [ - 5, 1 ] },
		nightGlow: { evening: [ - 11, - 7 ], morning: [ - 12, - 7 ], clouded: [ - 5, 1 ] },
		cloudMoon: { evening: [ - 11, - 7 ], morning: [ - 7, - 3 ], clouded: [ - 7, - 3 ] },
	},
	// the daylight's window in which a clouded sky's display and stars take the night (the
	// snapshot's `daylight`, which falls from 8 to -7 degrees): all of it under the first, none over
	// the second
	cloudedDaylight: [ 0.2, 0.45 ],
	// How deep under all of it (stops) each night layer comes in at the evening's window's day end
	// (-7 degrees), rising by even stops a degree toward all of it at -11 (`duskIn`, which eases the
	// last `duskEase` degrees; the header). Each starts under the twilight's own light it adds to: the
	// depth is that layer's full light over the twilight's at -7 (measured, in the comments) and about
	// five stops more. (The moon's key and its sky take none: they lie under the twilight's light
	// already, the header.)
	duskDepth: {
		// the night key in the sky's light (`skyNight`): 1.35 over the land's
		sky: 6.25,
		// the night ambient (`nightAmbientShare`) and the yunpeng's glow and light (`nightGlowShare`),
		// which share this depth: 3.1 over the land's
		ambient: 8,
		// the stars (`starNight`): 2.3 over the sky's
		stars: 7.25,
		// the haze's night key (weather/HazeKeys.ts): 1.8 over the sky's
		haze: 6.75,
		// the clouds' own moonlight (`cloudMoonShare`): 3.9 over the sky's
		cloudMoon: 9,
	},
	// Over how many degrees at the night's end of the evening's window (`duskWindows`' evening windows)
	// its night layers ease into all of their light (`duskIn`). By even stops to the end,
	// each came in fastest at -11 degrees, an eighth of its depth in the last half degree, and stopped
	// there: the clear evening's exposure over the horizon fell 1.5 stops a half degree into -11 and
	// not at all after it, and in that last step the sky over the horizon turned up by 0.005 to 0.007
	// in Oklab L and the sea under it by 0.016. Eased over three degrees the exposure's last steps
	// there are 1.8, 1.3 and 0.5 stops (the ones before them steeper, 2.1 at most), the sky turns by
	// about 0.001, the sea's rise to the night's level is spread over three steps (0.008 at most), and
	// the window's day end, where the depths are set, is as it was (every frame above -8 degrees
	// unchanged); over two the last step was still 0.7 stops and the sea rose 0.0125 in one, over four
	// the day end moved.
	duskEase: 3,
	exposure: [
		nightKey( - 12, NIGHT_METER_LIFT ), nightKey( - 11, 12.37 ), nightKey( - 10, 10.38 ), nightKey( - 9, 6.27 ), nightKey( - 8, 2.26 ),
		[ - 7, TWILIGHT_EXPOSURE ], [ - 6, TWILIGHT_EXPOSURE ], [ - 2.65, 19 ], [ 0, 11 ], [ 2, 10 ], [ 6.6, 5.5 ], [ 13.2, 3.4 ], [ 16.5, 2.85 ],
		[ 19.75, 2.4 ], [ 26.2, 1.82 ], [ 32.5, 1.48 ], [ 50, 1.19 ],
	],
	nightMeterLift: NIGHT_METER_LIFT,
	// The night sky's own light (stars, the moon and its in-scatter, clouds) was tuned at nightSkyExposure, with the
	// night's own gain over it (nightExposureGain) and the blue hour's adaptation, and keeps that
	// displayed brightness whatever the frame's exposure (see Sky.track).
	nightExposureGain: 1.25,
	nightSkyExposure: 4.4,
};
