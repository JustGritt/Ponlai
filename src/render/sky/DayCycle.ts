import { MathUtils, Matrix3, Quaternion, Vector3 } from 'three/webgpu';
import { TUNE_ON, tune, tuneObject } from '../debug/Tune.ts';
import { markSkyRetune } from './SkyRetune.ts';
import type { DayWindows, Span } from '../look/DayLook.ts';
import type { Exposure } from '../look/ExposureLook.ts';

// The day cycle: the sun, the moon and the stars over an equinox day, and the key light.
//
// The day cycle: an equinox day at a fixed latitude, its stars a fixed real sky; the moon is
// placed, not orbited. One dominant key light (the sun by day, the moon by
// night) drives terrain, characters, water and every shadow map; a sun/moon handoff
// passes through zero direct light: the sun's key fades as its drawn disk sinks below the horizon,
// and only then turns toward the moon (the windows' `handoffTop`; `keyDir` turns along the great circle from the
// sun to the moon). The sun is drawn at twice its true
// size and the moon larger (NightSky.ts `MOON_ANGULAR_RADIUS`); shadows and glints use the true sun. A low sun's key stays
// orange red: the optical depth that sets its hue is capped (and a high sun's key is a warm white
// of its own, taking the air's reddening only as the sun sinks), while its brightness still follows
// the full extinction (look/SkyLook.ts `keyHue`). The sky, sun, moon, stars, key light and grade follow
// the clock every frame. `reset` sets the clock as the constructor does (a scene jump).
//
// The base exposure follows the sun's elevation (the exposure's `at`, look/ExposureLook.ts: the
// cycle is given the exposure it reads), set once under an earlier
// display chain: it only rises from afternoon into night. It does not set the frame's exposure,
// which the display chain meters from the light (post/Exposure.ts): it is the exposure the night
// sky's own lights, the stars and the moon's disk, are tuned against (`nightSkyLevel`),
// and nothing they light reads it. So it follows the metered exposure: under the twilight it holds
// from -6 to -7 degrees, and below it climbs with the meter as the night falls to the moon's light,
// the exposure's `nightMeterLift` stops by about -12 degrees, so the stars and the disk keep their
// display.
// The night ambient rises as the sun's skylight fades after sunset, and the moon's key comes in
// once the handoff has turned it (the windows' `moonKeyTop`). Each night share comes in over a window placed
// so that what it adds in half a degree is no more than what the twilight loses in it. The aim:
// every light the dusk is judged by (the land's, the sea's, the sky's, the haze's) falls as the
// sun sinks, rising by no more than 2 % of itself in any half degree (tools/island/dusk-grid.mjs, its
// step-wise check, from +2 down to -8 degrees) and never climbing more than 2 % over the level it
// held at any higher sun down to midnight (its overall check); and the exposure, in stops of
// gain, never falls as the sun sinks to -8 degrees (its exposure check). Known exceptions to the
// overall check, as the moon's light takes over toward midnight: the sea's radiance under a clear
// or fair evening climbs back about twofold, the far meadow's under cloudy and overcast skies
// about 10 to 27 %, and the land's light and its illuminance 2 to 4 %; and the step-wise check
// stops at -8 degrees.
// So the night is the twilight's end, not a lift after a dip; the night sky's own light (stars, the moon's disk and its in-scatter in the
// sky, clouds) scales back to keep its tuned look in deep night (Sky.track). The night is
// art-directed, not a simulation of the eye's night vision (docs/design/night.md).
// A clear twilight is not yet the night, on either side of it. After a clear sunset the glow
// holds low in the west under a blue sky for a long while, falling slowly to about -9 degrees,
// and long before a clear sunrise the air is blue and the sea a clear blue under it, lit by the
// sun's own light on the high air. So under a clear or fair sky the display's night weight
// (`displayNight`), the sky's night key (`skyNight`), the night's ambient (`nightAmbientShare`)
// and the night glow (`nightGlowShare`: glowing plants and their local lights, which would glow at
// their brightest under a day's exposure) come in only from -7 to -11 degrees after sunset, and the rising sun takes them away
// early, from about -12 degrees. The moon's key and its sky take no window: at the moon's
// magnitude (SkyTrack.ts MOON_OVER_SUN) the key's whole light on the land is 2.9 stops under the
// twilight's own at -7 degrees after sunset and 2.4 under it before sunrise, and the moonlit
// sky's light on it 4.5 stops under, so each adds to a twilight that is falling and is in whole
// as soon as the moon holds the key (the snapshot's `moonKeyGain`). Held out from -7 to -11 by
// stops, as a moon some ten stops brighter had been, the moon gave the land under a thousandth
// of its light at -8 degrees, and the twilight's haze in front of them, twice the
// land's light after sunset and eight times it before sunrise, filled the frame with its glow. The evening's window ends where the atmosphere's own twilight does: the sun's light on
// the high air above about 10 degrees falls by three orders of magnitude from -7 to -9 (its table
// lights the multiple scattering only where the sun reaches), and a night that came in only from
// -11 to -14 left the upper sky black at -9 to -11 under a day's exposure. Through the evening's
// window each of the other night layers comes in by stops (`duskIn`: the night ambient, the sky's
// night key, the stars and the haze's and the clouds' own moonlight), from some stops under the
// twilight's own light it adds to (the windows' `depth`) to all of
// it at -11, easing into it over the last three degrees (the windows' `ease`); the display's night weight,
// which keys the exposure to that light, keeps an even fade. Faded evenly in light from -7, the
// night's layers raised the land's light 2.5 stops in the first half degree: the land rose by 0.05
// in Oklab L and the exposure fell 2.2 stops from -7 to -7.5 degrees, where the steps before it rose
// 0.9. By stops, the clear evening's exposure turns at -7.5, falls 1.6 to 2.1 stops each half degree
// from -8 to -10 and lands at -11 by 1.4 and 0.5, no step on the land rises by more than 0.01 in L,
// and the sky's falls to the night's without turning. The clouded share keeps
// the daylight's timing for the display (its night whole by about -2.5 degrees); the exposure's
// night comes in from the horizon to -5 and the sky's by -6 (the windows' `dusk`), and a
// side's flip at noon or midnight moves nothing: every weight here is 0 by day and 1 by night on
// both sides.
//
// The moon is fixed in the sky (MOON, `moonDirection`): 20 degrees up on the noon meridian unless
// set, where it stands all night, every night. The default island view's sea carries its path, and
// its light comes in at a slant, so slopes have a lit and a shaded side and the moon's shadows run
// long; it neither rises nor sets, so the night's composition holds while the game is played. It is the
// night's one cool key: lighting/KeyLift.ts shades from its direction for as long as it is the
// key, so the night keeps one shading and one level; the moon is drawn there, the sky's and the
// haze's moonlight come from there, and the key's direction turns there over the handoff, along
// the great circle from the sun, and holds there all night. It shows as the sun's light leaves:
// its visibility (and so the key's moonlight) rises over the sun's first five degrees under the
// horizon.
// The phase (a full moon unless set) shapes only the drawn moon: its disk is lit from the phase's
// angle round from the moon's own direction toward the west (`moonSunDir`: from behind it at a new
// moon, from the west at the first quarter, from the front at the full), and nothing else follows
// it (the key, the night sky's moonlit wash, the clouds' moonlight), so advancing it never darkens a
// night.
// The numbers chosen for the look (where the key turns to the moon, the dusk's windows and depths,
// the ease) are the day's windows the cycle is given (look/DayLook.ts `createDayWindows`), which
// every share's function here takes as its first argument; the exposure keys and the night sky's
// exposure are the exposure's it is given (look/ExposureLook.ts). This file keeps the sun's path,
// the moon's and the stars' places, the shares' forms that read those tables, and the clock.

export type DayPreset = keyof typeof DAY_PRESETS;

export interface DayCycleOptions {
	hour?: number;
	dayLength?: number;
	paused?: boolean;
	moonPhase?: number;
	advanceMoonPhase?: boolean;
	lunarPeriodDays?: number;
}

export interface DayCycleSnapshot {
	hour: number;
	day: number;
	sunDir: Vector3;
	moonDir: Vector3;
	// the direction the drawn disk is lit from at the phase
	moonSunDir: Vector3;
	// the key's direction: the sun's, turned toward the moon's along the great circle between them
	// over the handoff
	keyDir: Vector3;
	// how far the key has turned from the sun toward the moon, 0 to 1
	handoff: number;
	elevation: number;
	azimuth: number;
	moonElevation: number;
	moonAzimuth: number;
	moonPhase: number;
	moonIllumination: number;
	moonVisibility: number;
	moonKeyGain: number;
	// World to equatorial (J2000) frame: x toward right ascension 0h, z toward the north
	// celestial pole. The same at every hour and day (STARS).
	starFrame: Matrix3;
	daylight: number;
	// The stars and the Milky Way: they come out late, as the sky darkens.
	nightVisibility: number;
	exposure: number;
	// Further exposure, as a fraction, under a fully closed deck through the twilight, none from -6
	// degrees down (see Sky.track).
	overcastAdaptation: number;
	// Scale on the night sky's own light (stars, the moon's disk and its in-scatter in the sky,
	// clouds): it was tuned at a night exposure of the exposure's `nightSky` and keeps that look,
	// brighter by its `nightGain`, whatever the frame's exposure.
	nightSkyLevel: number;
}

const TAU = 2 * Math.PI;
const DEG = 180 / Math.PI;
/** The sun's elevation at noon (degrees): its path is tilted this far from the horizon. */
export const NOON_ELEVATION = 52;
const INCLINATION = NOON_ELEVATION / DEG;
const SIN_INCLINATION = Math.sin( INCLINATION );
const COS_INCLINATION = Math.cos( INCLINATION );
// The noon meridian's azimuth in the world (radians, about 66.8 degrees), fixed whatever the
// path's tilt: the default island view looks 23 degrees east of it.
const NOON_AZIMUTH = 1.166480189932717;
const SIN_NOON_AZIMUTH = Math.sin( NOON_AZIMUTH );
const COS_NOON_AZIMUTH = Math.cos( NOON_AZIMUTH );
// direction(a) = cos(a) N + sin(a) WEST, N the noon sun ( COS_INCLINATION COS_NOON_AZIMUTH,
// SIN_INCLINATION, COS_INCLINATION SIN_NOON_AZIMUTH ): a rotation by a about N x WEST.
// The sun stands at the celestial equator (an equinox, every day): at noon it marks where the
// equator crosses the meridian and at 18:00 the west point. Its path is tilted NOON_ELEVATION
// degrees from the horizon, so its elevated pole stands 90 - NOON_ELEVATION degrees up; seen from
// above, the sun turns anticlockwise, so that pole is the south celestial pole. A 52 degree noon is
// a mid-latitude equinox's daylight: high enough to read as a bright day, low enough that slopes
// keep a lit and a shaded side from the island view, which looks 50 degrees down. Near the horizon
// the sun climbs or sinks 11.8 degrees an hour, so the low sun lingers: 3.4 hours a day under 20
// degrees and 1.7 under 10, about a tenth longer than under a 62 degree noon. A noon that high is a
// subtropical light, at odds with the island's temperate bamboo and meadow.
const WEST = new Vector3( SIN_NOON_AZIMUTH, 0, - COS_NOON_AZIMUTH );
/** The sun's elevation (degrees) at `hour` on its path. */
export function sunElevationAt( hour: number ): number {

	return Math.asin( SIN_INCLINATION * Math.cos( ( hour - 12 ) / 24 * TAU ) ) * DEG;

}
// the hour angle (hours from noon, 0 to 12) at which the sun stands at `elevation` (degrees)
function hoursFromNoon( elevation: number ): number {

	finite( elevation, 'elevation' );
	if ( Math.abs( elevation ) > NOON_ELEVATION ) throw new RangeError( `elevation must lie within ${ NOON_ELEVATION } degrees of the horizon` );
	const c = Math.max( - 1, Math.min( 1, Math.sin( elevation / DEG ) / SIN_INCLINATION ) );
	return Math.acos( c ) / TAU * 24;

}
/** The hour (0 to 12) at which the morning sun stands at `elevation` (degrees): the inverse of
 * `sunElevationAt` before noon. */
export function morningHour( elevation: number ): number {

	return 12 - hoursFromNoon( elevation );

}
/** The hour (12 to 24) at which the evening sun stands at `elevation` (degrees): the inverse of
 * `sunElevationAt` after noon. */
export function eveningHour( elevation: number ): number {

	return 12 + hoursFromNoon( elevation );

}
/** The day's presets (hours): each the hour its sun stands at its elevation, the morning's or the
 * evening's, so a preset keeps its light whatever the path's tilt (dawn and dusk 8.6 degrees under
 * the horizon, sunrise and sunset 2 up, the golden hour 26.2 up in the afternoon); noon and
 * midnight are 12 and 0. The clock starts at the golden hour. */
export const DAY_PRESETS = {
	'dawn': morningHour( - 8.6 ),
	'sunrise': morningHour( 2 ),
	'noon': 12,
	'golden-hour': eveningHour( 26.2 ),
	'sunset': eveningHour( 2 ),
	'dusk': eveningHour( - 8.6 ),
	'midnight': 0,
} as const;
/** Whether the sun at `dir` (unit, on its path) is rising: east of the meridian, before noon (the
 * haze's keys tell the morning from the evening by it: weather/HazeKeys.ts). */
export const sunRising = ( dir: Readonly<Vector3> ): boolean => dir.dot( WEST ) < 0;
/** A share that comes in as the sun sinks through `window` ( [ whole, none ], degrees): a
 * smoothstep, exactly 0 and 1 at the ends. */
export function rampIn( elevation: number, [ whole, none ]: Span ): number {

	return 1 - smoothstep( whole, none, elevation );

}
// a clouded sky's night by the daylight (the windows' `cloudedDaylight`)
function fromDaylight( windows: DayWindows, daylight: number ): number {

	return 1 - smoothstep( windows.cloudedDaylight[ 0 ], windows.cloudedDaylight[ 1 ], daylight );

}
// how far the sun has come through a twilight's window (degrees: 0 at `from`, 1 at `to`)
function across( elevation: number, [ from, to ]: readonly [ number, number ] ): number {

	return MathUtils.clamp( ( elevation - from ) / ( to - from ), 0, 1 );

}
// a night layer's share at `t` through its twilight's window (`across`: all of it at 0, none at 1),
// a clear sky's rising by even stops from `least` of all of it at 1, under a clouded share `cloud`
// all of it
function inStops( t: number, least: number, cloud = 0 ): number {

	const left = least ** t;
	return cloud + ( 1 - cloud ) * ( left - least ) / ( 1 - least );

}
/** A night layer's share through the evening's `window` (degrees: all of it under the first, none
 * over the second), coming in from `depth` stops under all of it (the windows' `depth`), under a
 * clouded share `cloud` all of it: exactly none and all at the window's ends, as the smoothstep it
 * replaced, and over the night end's last degrees (the windows' `ease`) its rate falls to none (a
 * cubic in the window's place there, whose slope is the even stops' where it starts). */
export function duskIn( windows: DayWindows, elevation: number, window: readonly [ number, number ], depth: number, cloud = 0 ): number {

	const t = across( elevation, window );
	const ease = Math.min( windows.ease.degrees / ( window[ 1 ] - window[ 0 ] ), 1 );
	const x = t < ease ? t / ease : 1;
	return inStops( t < ease ? ease * x * x * ( 2 - x ) : t, 2 ** - depth, cloud );

}
/** The display's night weight (Pipeline NIGHT: the tone, the grade, the bloom; the flying fish), at
 * `daylight` (the snapshot's) and the sun's `elevation` (degrees), `rising` or setting, under a
 * clouded share `cloud` (Weather.ts `cloudedShare`). Under cloud none through the afterglow, all of
 * it once the daylight has gone; a clear sky's comes in from -7 to -11 degrees after sunset and
 * leaves from -12 to -7 before sunrise (the header), evenly (a smoothstep) on both sides: it keys
 * the exposure to the light, and by stops it lagged the night's light through the evening's
 * window, the land rising by 0.06 in Oklab L. */
export function displayNight( windows: DayWindows, daylight: number, elevation: number, rising: boolean, cloud: number ): number {

	const night = fromDaylight( windows, daylight );
	const dawn = rampIn( elevation, rising ? windows.dusk.display.morning : windows.dusk.display.evening );
	return dawn + ( night - dawn ) * cloud;

}
/** The exposure's night weight (look/ExposureLook.ts `keysAt`: the night key's share, and the
 * overcast night's offset by it) at the sun's `elevation` (degrees), `rising` or setting, under a
 * clouded share `cloud` (Weather.ts `cloudedShare`), over its own windows (`windows.dusk.exposure`):
 * the meter's handover to the night key is a question of the light it reads, the display's of how
 * the night looks, so each is set apart. A clear sky's comes in as the display's does; a clouded
 * one's from the horizon to -5 degrees, a smoothstep by elevation, as the deck's twilight goes
 * (its own window, `windows.dusk.exposure.clouded`). Taken by the daylight, as the display's is, it
 * came in from the horizon to about -3 degrees, and the handover from the deck's dusk keys
 * (look/ExposureLook.ts `cloudySunset` -2.5, `cloudyDusk` -2.9) to the night key (-2.57) came too
 * early for the metered light: the exposure fell while the land under the deck grew darker. */
export function exposureNight( windows: DayWindows, elevation: number, rising: boolean, cloud: number ): number {

	const night = rampIn( elevation, windows.dusk.exposure.clouded );
	const dawn = rampIn( elevation, rising ? windows.dusk.exposure.morning : windows.dusk.exposure.evening );
	return dawn + ( night - dawn ) * cloud;

}
/** The stars' share (SkyTrack.ts `starVisibility`): the display's night (`displayNight`), but the
 * evening's clear share by stops (`windows.depth.stars`), as the stars are light the meter reads. */
export function starNight( windows: DayWindows, daylight: number, elevation: number, rising: boolean, cloud: number ): number {

	if ( rising ) return displayNight( windows, daylight, elevation, rising, cloud );
	const night = fromDaylight( windows, daylight );
	const dusk = duskIn( windows, elevation, windows.dusk.stars.evening, windows.depth.stars );
	return dusk + ( night - dusk ) * cloud;

}
/** The night key's share in the sky's light (look/SkyLook.ts `skyLight`, its one reader), at the sun's
 * `elevation` (degrees), `rising` or setting, under a clouded share `cloud`: under cloud none from
 * 2 degrees up, all of it by -6; a clear sky's comes in from -7 to -11 degrees after sunset by
 * stops (`windows.depth.sky`) and leaves from -13 to -3 before sunrise (the header). A debug page can
 * force it (`duskForce`). */
export function skyNight( windows: DayWindows, elevation: number, rising: boolean, cloud: number ): number {

	const night = rampIn( elevation, windows.dusk.sky.clouded );
	const dawn = rising ? rampIn( elevation, windows.dusk.sky.morning ) : duskIn( windows, elevation, windows.dusk.sky.evening, windows.depth.sky );
	return duskForce( 'skyNight', dawn + ( night - dawn ) * cloud );

}
/** The stars' sky: a real sky, the one seen from `latitude` degrees south at the local sidereal
 * time `sidereal` (hours), fixed. It neither turns through the night nor drifts over the year, so
 * the night's composition holds while the game is played, and its constellations stand upright as
 * seen from that latitude. 38 degrees south at 14h raises the Milky Way out of the sea in the
 * horizon view, split by the Great Rift, up to its core 43 degrees high in the east below Scorpius,
 * with Crux and Alpha Centauri high in the south, and sets the moon (MOON) on a dark part of the
 * sky, 81 degrees from the core, with no star brighter than the second magnitude within 8 degrees
 * of it. 38 degrees south is the latitude we set the island at: the stars' sky and the sun's path
 * (NOON_ELEVATION, 90 less it) share one pole, 38 degrees up on the side away from noon. Tunable on
 * debug pages (debug/Tune.ts): `sky.stars.latitude`, `sky.stars.sidereal`. */
export const STARS = tuneObject( { latitude: 38, sidereal: 14 }, { latitude: 'sky.stars.latitude', sidereal: 'sky.stars.sidereal' }, {
	latitude: { min: 0, max: 90, onChange: placeStars },
	sidereal: { min: 0, max: 24, onChange: placeStars },
} );
/** Where the moon stands (the header): `elevation` degrees up, at the noon meridian's azimuth
 * turned `turn` degrees (0 the noon side, 180 the pole's side, positive toward the west). At least
 * 15 degrees up: from there down the low key's floor (lighting/KeyLift.ts) would hold the shading
 * up to the end of the handoff, a small turn at no light. Tunable on debug pages (debug/Tune.ts):
 * `sky.moon.elevation`, `sky.moon.turn`. */
export const MOON = tuneObject( { elevation: 20, turn: 0 }, { elevation: 'sky.moon.elevation', turn: 'sky.moon.turn' }, {
	elevation: { min: 15, max: 40, onChange: markSkyRetune },
	turn: { min: - 180, max: 180, onChange: markSkyRetune },
} );
/** The moon's share of the key at a sun elevation (degrees), before its visibility (the snapshot's
 * `moonKeyGain`): none above the windows' `moonKeyTop`, whole two degrees lower. */
export function moonKeyShare( windows: DayWindows, elevation: number ): number {

	return rampIn( elevation, windows.dusk.moonKey );

}

/** The night's ambient share at a sun elevation (degrees): the sun's scattered skylight falls about
 * tenfold between the horizon and -4 degrees, and the night's ambient rises across the same band so
 * the two together never dip. */
export function nightAmbientAt( windows: DayWindows, elevation: number ): number {

	return rampIn( elevation, windows.dusk.ambient.clouded );

}
/** The night ambient's share under a clouded share `cloud` (the night sky's ambient in the
 * clouds and the sea, the leaves' night key, and the night that
 * takes the sward's tint under cloud away, materials/SwardTint.ts): under cloud
 * `nightAmbientAt`'s, a clear sky's in from -7 to -11 degrees after sunset by stops
 * (`windows.depth.ambient`) and out from -12 to -7 before sunrise with the display's night
 * (`displayNight`). Through a clear twilight the frame
 * keeps a day's exposure, five stops over the night's at -7 degrees: the clumps lit by -5 then
 * glowed brightest in the twilight and dimmed as the night came in. */
export function nightAmbientShare( windows: DayWindows, elevation: number, rising: boolean, cloud: number ): number {

	const clear = rising ? rampIn( elevation, windows.dusk.ambient.morning ) : duskIn( windows, elevation, windows.dusk.ambient.evening, windows.depth.ambient );
	return clear + ( nightAmbientAt( windows, elevation ) - clear ) * cloud;

}
/** The night glow's share of the night (what glows by night and its local lights, set from it by
 * the scene), at the sun's `elevation` (degrees), `rising` or setting, under a clouded share `cloud`: the night ambient's
 * own form (`nightAmbientShare`) over windows of its own (`windows.dusk.nightGlow`), so the lamps keep
 * their coming in whatever the ambient's windows become. */
export function nightGlowShare( windows: DayWindows, elevation: number, rising: boolean, cloud: number ): number {

	const own = windows.dusk.nightGlow;
	const clear = rising ? rampIn( elevation, own.morning ) : duskIn( windows, elevation, own.evening, windows.depth.ambient );
	return clear + ( rampIn( elevation, own.clouded ) - clear ) * cloud;

}
/** The clouds' own moonlight's share (Sky.ts, the clouds' `moonShare`), at the sun's `elevation`
 * (degrees), `rising` or setting, under a clouded share `cloud`: a clear evening's with the night
 * sky's own light, by stops (`windows.depth.cloudMoon`), the morning's and a clouded sky's over their
 * own windows (`windows.dusk.cloudMoon`). */
export function cloudMoonShare( windows: DayWindows, elevation: number, rising: boolean, cloud: number ): number {

	const own = windows.dusk.cloudMoon;
	const clouded = rampIn( elevation, own.clouded );
	return rising ? rampIn( elevation, own.morning ) : MathUtils.lerp( duskIn( windows, elevation, own.evening, windows.depth.cloudMoon ), clouded, cloud );

}

/** The dusk's shares a script can force on a debug page, to take the light of the twilight and of
 * the night apart (tools/island/dusk-grid.mjs `--split`, `--night-split`), in the order that tool puts them
 * back one at a time. The night's: `moonKey`, the moon's share of the key (SkyTrack.ts `moonGain`);
 * `moonSky`, of the moonlit sky (Sky.ts `moonSkyShare`, which the probes' sky gain mixes its moon's
 * part by: lighting/AmbientRelight.ts); `skyNight`, the sky's twilight night key (`skyNight`);
 * `nightAmbient`, the night's ambient in the sky, the clouds and the sea, the leaves'
 * night key and the sward's (`nightAmbientShare`); `landNight`, the moon's share of the land's gain
 * under overcast's deck (lighting/AmbientRelight.ts `underDeck`); `hazeNight`, the night key's
 * weight in the haze's (weather/HazeKeys.ts `keyAlong`); `cloudMoon`, the clouds' moonlight (Sky.ts
 * `moonShare`); `deck`, the deck's light in the haze (HazeKeys.ts `deck`); `exposureNight`, the
 * night weight the exposure's keys take (`exposureNight`), and `displayNight`, the one the tone,
 * the grade and the bloom take (`displayNight`; both read in post/Pipeline.ts `updateDisplay`); `nightGlow`, the night glow's share
 * (what glows by night and its lights), and `localLights`, the local lights' strength (RenderStages.ts `lightNight`); `stars`, the stars' share in the
 * drawn sky (SkyTrack.ts `starVisibility`: `nightVisibility` times `starNight`; no bake holds them,
 * so the land takes none of their light). The twilight's, each a weight on
 * the sun's light (1 when not forced): `sun`, the sun's part of the key (Sky.ts `keyColor`);
 * `hazeSun`, the haze's sunlight; `hazeSky`, the haze's skylight from the sun's sky
 * (weather/VolumetricHaze.ts). */
export const DUSK_FORCES = [ 'moonKey', 'moonSky', 'skyNight', 'nightAmbient', 'landNight', 'hazeNight', 'cloudMoon', 'deck', 'exposureNight', 'displayNight', 'nightGlow', 'localLights', 'stars', 'sun', 'hazeSun', 'hazeSky' ] as const;
export type DuskForce = typeof DUSK_FORCES[ number ];
// `sky.dusk.force.<name>` (debug/Tune.ts): -1, the default, leaves the share as the code makes it,
// 0 to 1 forces it to that value; the sky is baked again on a change
const duskForces = Object.fromEntries( DUSK_FORCES.map( ( name ) => [ name, tune( `sky.dusk.force.${ name }`, - 1, { min: - 1, max: 1, onChange: markSkyRetune } ) ] ) ) as Record<DuskForce, () => number>;

/** The value the share `name` is forced to on a debug page (DUSK_FORCES), or null when it is not. */
export function duskForced( name: DuskForce ): number | null {

	const value = duskForces[ name ]();
	return value < 0 ? null : value;

}

/** The values each share took as the code made it, forced or not, since a script last cleared the
 * map: [ least, greatest ] over every call (a share some frames or readers take at other values
 * shows the range). Kept on debug pages only (`TUNE_ON`, where `__PONLAI__` exists to read it):
 * elsewhere it stays empty and a call costs nothing. tools/island/dusk-grid.mjs reads it to report the
 * shares and to skip forcing a share to the value it already has. `hazeNight` has no value of its
 * own here (`duskForced`). */
export const duskShares = new Map<DuskForce, [ number, number ]>();

/** The share `name` (DUSK_FORCES): `value` itself unless it is forced. */
export function duskForce( name: DuskForce, value: number ): number {

	if ( ! TUNE_ON ) return value;
	const range = duskShares.get( name );
	if ( range === undefined ) duskShares.set( name, [ value, value ] );
	else {

		range[ 0 ] = Math.min( range[ 0 ], value );
		range[ 1 ] = Math.max( range[ 1 ], value );

	}
	return duskForced( name ) ?? value;

}

function finite( value: number, name: string ): number {

	if ( ! Number.isFinite( value ) ) throw new RangeError( `${ name } must be finite` );
	return value;

}

function positive( value: number, name: string ): number {

	finite( value, name );
	if ( value <= 0 ) throw new RangeError( `${ name } must be greater than zero` );
	return value;

}

function wrap( value: number, period: number ): number {

	return ( value % period + period ) % period;

}

function smoothstep( low: number, high: number, value: number ): number {

	const t = Math.max( 0, Math.min( 1, ( value - low ) / ( high - low ) ) );
	return t * t * ( 3 - 2 * t );

}

// World to equatorial frame of STARS' sky: a star at right ascension a and declination d lies at
// hour angle H = (sidereal - a), cos(H) meridian + sin(H) WEST, raised by d toward the north pole;
// the meridian point is where the equator crosses the noon meridian, 90 - latitude degrees up.
function equatorialFrame(): Matrix3 {

	const elevation = ( 90 - STARS.latitude ) / DEG;
	const meridian = new Vector3( Math.cos( elevation ) * COS_NOON_AZIMUTH, Math.sin( elevation ), Math.cos( elevation ) * SIN_NOON_AZIMUTH );
	const pole = WEST.clone().cross( meridian ).normalize();
	const sidereal = STARS.sidereal / 24 * TAU;
	const c = Math.cos( sidereal ), s = Math.sin( sidereal );
	const x = meridian.clone().multiplyScalar( c ).addScaledVector( WEST, s );
	const y = meridian.clone().multiplyScalar( s ).addScaledVector( WEST, - c );
	return new Matrix3().set( x.x, x.y, x.z, y.x, y.y, y.z, pole.x, pole.y, pole.z );

}

// the stars' frame, worked out at load and again when STARS changes
let starFrame = equatorialFrame();
function placeStars(): void {

	starFrame = equatorialFrame();
	markSkyRetune();

}

function direction( angle: number ): Vector3 {

	const c = Math.cos( angle ), s = Math.sin( angle );
	return new Vector3(
		COS_INCLINATION * c * COS_NOON_AZIMUTH + s * SIN_NOON_AZIMUTH,
		SIN_INCLINATION * c,
		COS_INCLINATION * c * SIN_NOON_AZIMUTH - s * COS_NOON_AZIMUTH,
	).normalize();

}

/** The moon's fixed direction (unit, world: MOON's elevation and azimuth), into `target`. The low
 * key's lift (lighting/KeyLift.ts) reads it at every publish of the lighting. */
export function moonDirection( target: Vector3 ): Vector3 {

	const azimuth = NOON_AZIMUTH - MOON.turn / DEG, e = MathUtils.degToRad( MOON.elevation );
	return target.set( Math.cos( e ) * Math.cos( azimuth ), Math.sin( e ), Math.cos( e ) * Math.sin( azimuth ) );

}
const _noTurn = new Quaternion(), _turn = new Quaternion(), _keyTurn = new Quaternion();

/** A deterministic art-directed celestial clock. Hours and moon phase are independent of the
 * wave/vegetation animation clock, so lighting presets do not jump existing scene animation. */
export class DayCycle {

	paused: boolean;
	advanceMoonPhase: boolean;
	private currentHour: number = DAY_PRESETS[ 'golden-hour' ];
	private secondsPerDay = 2880;
	private phase = 0.5;
	private lunarDays = 29.53;
	private completedDays = 0;
	/** The exposure the cycle reads its base exposure and the night sky's from. */
	readonly exposure: Exposure;
	/** The day's windows the cycle's handoff and the night's shares read (look/DayLook.ts). */
	readonly windows: DayWindows;

	/** `exposure`: the base exposure by the sun's elevation and the night sky's (look/ExposureLook.ts);
	 * `windows`: the day's windows (look/DayLook.ts `createDayWindows`). */
	constructor( exposure: Exposure, windows: DayWindows, options: DayCycleOptions = {} ) {

		this.exposure = exposure;
		this.windows = windows;
		this.paused = false;
		this.advanceMoonPhase = false;
		this.reset( options );

	}

	/** The clock as the constructor sets it from `options` (a scene jump: no days completed). */
	reset( options: DayCycleOptions = {} ): void {

		this.hour = options.hour ?? DAY_PRESETS[ 'golden-hour' ];
		this.dayLength = options.dayLength ?? 2880;
		this.moonPhase = options.moonPhase ?? 0.5;
		this.lunarPeriodDays = options.lunarPeriodDays ?? 29.53;
		this.paused = options.paused ?? false;
		this.advanceMoonPhase = options.advanceMoonPhase ?? false;
		this.completedDays = 0;

	}

	get hour(): number { return this.currentHour; }
	set hour( value: number ) { this.currentHour = wrap( finite( value, 'hour' ), 24 ); }
	get dayLength(): number { return this.secondsPerDay; }
	set dayLength( value: number ) { this.secondsPerDay = positive( value, 'dayLength' ); }
	get moonPhase(): number { return this.phase; }
	set moonPhase( value: number ) { this.phase = wrap( finite( value, 'moonPhase' ), 1 ); }
	get lunarPeriodDays(): number { return this.lunarDays; }
	set lunarPeriodDays( value: number ) { this.lunarDays = positive( value, 'lunarPeriodDays' ); }
	get day(): number { return this.completedDays; }

	setHour( hour: number ): void { this.hour = hour; }

	setPreset( preset: DayPreset ): void {

		if ( ! Object.hasOwn( DAY_PRESETS, preset ) ) throw new RangeError( `Unknown day preset: ${ preset }` );
		this.hour = DAY_PRESETS[ preset ];

	}

	advance( dt: number ): void {

		finite( dt, 'dt' );
		if ( dt < 0 ) throw new RangeError( 'dt must not be negative' );
		if ( this.paused || dt === 0 ) return;
		const days = dt / this.secondsPerDay;
		const hours = finite( this.currentHour + days * 24, 'advanced hour' );
		const phaseStep = this.advanceMoonPhase ? finite( days / this.lunarDays, 'advanced moon phase' ) : 0;
		this.completedDays += Math.floor( hours / 24 );
		this.currentHour = wrap( hours, 24 );
		this.phase = wrap( this.phase + phaseStep, 1 );

	}

	snapshot(): DayCycleSnapshot {

		const hourAngle = ( this.currentHour - 12 ) / 24 * TAU;
		const sunDir = direction( hourAngle );
		// The moon stands where MOON puts it at every hour and phase (the header). Its disk is lit
		// from the phase's angle round from its own direction toward the west: WEST made square to
		// the moon's direction.
		const moonDir = moonDirection( new Vector3() );
		const moonWest = WEST.clone().addScaledVector( moonDir, - WEST.dot( moonDir ) ).normalize();
		const moonSunDir = moonDir.clone().multiplyScalar( Math.cos( TAU * this.phase ) ).addScaledVector( moonWest, Math.sin( TAU * this.phase ) );
		const elevation = Math.asin( Math.max( - 1, Math.min( 1, sunDir.y ) ) ) * DEG;
		const daylight = smoothstep( - 7, 8, elevation );
		// The night sky was tuned under a low-sun adaptation that faded out over -5 to -14 degrees.
		const skyAdaptation = 1 + 0.6 * smoothstep( - 14, - 5, elevation ) * ( 1 - smoothstep( 4, 18, elevation ) );
		const exposure = this.exposure.at( elevation );
		// A closed deck's further adaptation (Sky.track): from the end of the civil twilight up it
		// rises to five, as the deck's own light leaves with the sun's; the night takes none, the
		// moon lighting the deck as the sun does.
		const dusk = smoothstep( - 6, - 1, elevation );
		const moonIllumination = ( 1 - Math.cos( TAU * this.phase ) ) * 0.5;
		// over the sun's first five degrees under the horizon (the header)
		const moonVisibility = smoothstep( 0, 5, - elevation );
		// Turn along the great circle from the sun to the moon only after the sun is blocked and
		// before the moon's key light fades in. Through the handoff the sun stands near the east or
		// west point; at the default turn the moon stands on the meridian, about a right angle away,
		// and at no turn opposed (the moon stands above the horizon).
		const handoff = rampIn( elevation, this.windows.dusk.handoff );
		const keyDir = handoff === 0 ? sunDir.clone() : handoff === 1 ? moonDir.clone() :
			sunDir.clone().applyQuaternion( _keyTurn.slerpQuaternions( _noTurn, _turn.setFromUnitVectors( sunDir, moonDir ), handoff ) );
		return {
			hour: this.currentHour,
			day: this.completedDays,
			sunDir,
			moonDir,
			moonSunDir,
			keyDir,
			handoff,
			elevation,
			azimuth: Math.atan2( sunDir.z, sunDir.x ) * DEG,
			moonElevation: MOON.elevation,
			moonAzimuth: Math.atan2( moonDir.z, moonDir.x ) * DEG,
			moonPhase: this.phase,
			moonIllumination,
			moonVisibility,
			// The moon's key comes in as soon as the handoff has turned it (from the windows' `moonKeyTop`), not
			// four degrees later: between the sun's last light and the night's ambient the meadow
			// fell to about half its night brightness. It does not follow the phase.
			moonKeyGain: moonKeyShare( this.windows, elevation ) * moonVisibility,
			starFrame: starFrame.clone(),
			daylight,
			nightVisibility: 1 - smoothstep( - 15, - 3, elevation ),
			// (the night sky stays dark through its own scaled light: see the exposure's `nightSky`)
			exposure,
			overcastAdaptation: 5 * dusk * ( 1 - daylight ),
			nightSkyLevel: Math.min( 1, this.exposure.nightSky * this.exposure.nightGain * skyAdaptation / exposure ),
		};

	}

}
