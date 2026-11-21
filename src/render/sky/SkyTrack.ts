import * as THREE from 'three/webgpu';
import { SUN_ILLUMINANCE } from './Atmosphere.ts';
import { SCENE_RADIANCE_SCALE, SUN_ANGULAR_RADIUS } from '../lighting/Radiometry.ts';
import { SHADOW_SLOPE_FLOOR, forwardLevel } from './CloudModel.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { shared } from '../shared.ts';
import { storeStops } from '../post/PreExposure.ts';
import { MOON, cloudMoonShare, displayNight, duskForce, exposureNight, nightAmbientShare, skyNight, starNight, sunRising, type DayCycleSnapshot } from './DayCycle.ts';
import { cloudedHighSun, cloudedShare, overcastDeckLight, overcastFloorSun, daySkyAt, type WeatherState } from './Weather.ts';
import { keyed } from '../post/DayKeys.ts';
import { tune } from '../debug/Tune.ts';
import { type CloudSky, type SkyTables, byElevation, landSky, keyHue, luma, luminance709, moonHue, storeNight } from '../look/SkyLook.ts';
import type { Sky } from './Sky.ts';
import type { DayWindows } from '../look/DayLook.ts';

// The sky's state for a frame: the sun, the moon, the key, the stars, the grade and the clouds.
//
// The sky's state for a frame (Sky.ts `track`, whose header says what it holds and why): the sun,
// the moon, the key, the stars, the grade and the displayed weather, and the clouds' light and shell
// (`trackClouds`), each a function of the sky. Everything that moves with the clock is continuous
// here; the bakes (SkyBake.ts) only capture this state. A name in backticks with no file is this
// file's or Sky.ts's; the sky's tables are the sky's (Sky.ts `look`, look/SkyLook.ts).

/** How far the low sun's look has come in at `elevation` (degrees) under a sky of direct
 * visibility `direct`: 0 above the tables' `lowSun.top` and below the horizon, 1 from their
 * `lowSun.full` down to 3 degrees (the island's blue-violet shadows). */
export function lowSun( look: SkyTables, elevation: number, direct: number ): number {

	// (off unless a number is set, with the clear-sky fill light)
	const scale = typeof debugFlags.lowSun === 'number' ? debugFlags.lowSun : debugFlags.lowSun === true ? 1 : 0;
	return THREE.MathUtils.smoothstep( elevation, 0, 3 ) * ( 1 - THREE.MathUtils.smoothstep( elevation, look.lowSun.full, look.lowSun.top ) )
		* THREE.MathUtils.smoothstep( direct, 0.3, 0.95 ) * scale;

}

// The tables' `clearSkyFill` share at `elevation` under a sky of direct visibility `direct`.
function clearSkyFill( look: SkyTables, elevation: number, direct: number ): number {

	const rows = look.clearSkyFill;
	let share = rows[ rows.length - 1 ].share;
	if ( elevation >= rows[ 0 ].at ) share = rows[ 0 ].share;
	else for ( let i = 1; i < rows.length; i ++ ) if ( elevation >= rows[ i ].at ) {

		const t = ( elevation - rows[ i ].at ) / ( rows[ i - 1 ].at - rows[ i ].at );
		share = rows[ i ].share + ( rows[ i - 1 ].share - rows[ i ].share ) * t;
		break;

	}
	// (off unless a number is set: the lifted shadows took the light's direction out of the meadow)
	const scale = typeof debugFlags.clearSkyFill === 'number' ? debugFlags.clearSkyFill : debugFlags.clearSkyFill === true ? 1 : 0;
	return share * THREE.MathUtils.smoothstep( direct, 0.3, 0.95 ) * scale;

}

const _skyLight = { r: 0, g: 0, b: 0 };
const _clearSky = { gain: 1 };
// The tables' `clearSky` gain at `elevation` under a sky of clouded share `cloud`: exactly 1 with the
// sun at or under the horizon and under a cloudy or overcast sky (`flags=clearSky:false` leaves it 1,
// for A/Bs)
function clearSky( look: SkyTables, elevation: number, rising: boolean, cloud: number ): number {

	if ( debugFlags.clearSky === false ) return 1;
	const day = keyed( look.clearSky, elevation, 0, _clearSky, 0, rising ).gain;
	return 1 + ( day - 1 ) * ( 1 - cloud );

}
// The sky's gain on the sun's light it lends: the tables' `skyLight` keys, the weather's day sky `daySky`
// (Weather.ts `daySkyAt`) on the day keys' share alone, so it is exactly the keys' own light where
// the night key has the sky (DayCycle.ts `skyNight` at 1) or the day sky is 1 (every weather but
// overcast), and their `clearSky` over both. (`flags=skyGain:false` leaves the sky as the atmosphere
// gives it, but for the day sky, `lowKey:false` the key as its transmittance gives it, for A/Bs)
const skyGain = ( look: SkyTables, windows: DayWindows, elevation: number, rising: boolean, cloud: number, daySky: number, target: THREE.Color ) => {

	const night = skyNight( windows, elevation, rising, cloud ), off = debugFlags.skyGain === false;
	if ( off ) target.setScalar( 1 );
	else {

		const { r, g, b } = keyed( look.skyLight, elevation, night, _skyLight, cloud, rising );
		target.setRGB( r, g, b );

	}
	if ( daySky !== 1 && night < 1 ) {

		const share = ( daySky - 1 ) * ( 1 - night );
		if ( off ) target.addScalar( share );
		else {

			const { r, g, b } = keyed( look.skyLight, elevation, 0, _skyDay, cloud, rising );
			target.r += r * share;
			target.g += g * share;
			target.b += b * share;

		}

	}
	return off ? target : target.multiplyScalar( clearSky( look, elevation, rising, cloud ) );

};
const _skyGain = new THREE.Color(), _skyBeam = new THREE.Color(), _skyMoon = new THREE.Color(), _skyDay = { r: 0, g: 0, b: 0 };
const lowKey = ( look: SkyTables, elevation: number ) => debugFlags.lowKey === false ? 1 : byElevation( look.lowKey, elevation, 'gain' );

// The low sun's lift of the clouds' beam and fill (trackClouds), fading as the sky closes.
const lowSunOf = ( elevation: number, coverage: number ) => THREE.MathUtils.smoothstep( elevation, - 4, 1 )
	* ( 1 - THREE.MathUtils.smoothstep( elevation, 6, 20 ) ) * ( 1 - THREE.MathUtils.smoothstep( coverage, 0.35, 0.65 ) );

const _white = new THREE.Color();
const _zenith = new THREE.Vector3( 0, 1, 0 );
// The clouds' own moonlight (DayCycle.ts `cloudMoonShare`, its windows the day windows'
// `dusk.cloudMoon`) comes in after sunset and leaves before sunrise: in the evening over the window
// of the night sky's light (`dusk.sky`), by stops as they come in (`depth.cloudMoon`: faded evenly, the clouds' moonlight was the largest share of the step the sky's meter
// made from -7 to -7.5, 0.7 of 1.6 stops). From -7 to -3 the moonlit clouds came in
// while the sky behind them was still the sun's afterglow: at -5.3 degrees they gave about half the light
// of the sky 8 degrees up, which greyed the glow and the far sea under it. The morning keeps -7 to
// -3, inside its night sky's window (-13 to -3): there the moonlit clouds are part of the
// sky the dawn's exposure meters (on a clear morning the cirrus most of all), and without it the land
// and the low band over the horizon at -6.5 degrees read lighter. Under cloud the evening keeps -7
// to -3 too, as every night layer keeps the daylight's timing there (DayCycle.ts: the
// clouded sky's night is whole by -6): no afterglow lights the sky behind the deck, and taken in
// from -11 by stops, an overcast deck's moonlight was under a hundredth of this at -8 degrees.
const _streakKey = { r: 0, g: 0, b: 0 };
// The sun's elevations (degrees) over which the light the clouds march toward sets from the moon's
// key to the sun's height, then comes round to the sun's azimuth: where no moonlight reaches them
// (look/DayLook.ts, the day windows' `dusk.cloudMoon`) and before the sun's does (`sunlightAt` at 2 km,
// above about -2).
const CLOUD_TURN = [ - 3, - 2.5, - 2 ] as const;
// The sine of the sun's elevation over which the clouds' shadow comes in as it rises.
const CLOUD_SHADOW_SET = [ - 0.02, 0.02 ] as const;
// The sine of the key's elevation over which the rays of the clouds' shadow gather from the height
// of its disc to its centre as it climbs, from 8 to 11 degrees: from there up a cloud's own edge
// takes longer to pass than the disc does (near 0.8 degree at 10 degrees, 0.15 at 4.7). The rays
// are as many as keep the disc from stepping: four, 0.135 degree apart, stepped the light by a
// quarter of the sun; these are 0.045 degree apart.
// Twelve cost about one percent of the frame's GPU time at their dearest, an overcast sky near
// 11 degrees, and only from about 4.3 to 11 degrees: under that the rays are one ray.
const SHADOW_DISC = [ 0.139, 0.191 ] as const;
const SHADOW_DISC_RAYS = 12;
// The key's elevations (degrees) over which the channel of the clouds' shadow the air reads narrows
// from the whole disc to its two centre rays, a function of the elevation alone. At 6 degrees the
// whole disc leaves the haze no shaft. Under the fade's foot the channel must be the whole disc:
// the sky's dome meters the air's light, and as a gap in the deck left the sun near 4.7 degrees the
// two rays stepped the exposure toward the sun by half a stop per 0.02 degree, the whole disc by a
// seventh of one. Over the fade the band of haze steps less than one ray's does; in a front at 8 degrees
// the air steps up to a third more than under the whole disc, under half a level in 255.
const SHADOW_AIR_NARROW = [ 4.9, 5.6 ] as const;

// The moon's light above the air, a share of the sun's illuminance (luminance): the full moon's over
// the sun's by the two bodies' apparent magnitudes at their mean distances (the full moon's -12.7,
// the sun's -26.7), 10^( -0.4 ( m_moon - m_sun ) ), about 2.5e-6 or 18.6 stops under the sun, times
// the debug gain (`moonKeyGain`, 1). The sky, the haze and the clouds take it and lose it along their
// own paths through the air, as they do the sun's; the key takes it through the air toward the moon
// (Atmosphere `transmittanceToMoon`), as the sun's key does the sun's.
const MOON_MAGNITUDE = - 12.7, SUN_MAGNITUDE = - 26.7;
const MOON_OVER_SUN = Math.pow( 10, - 0.4 * ( MOON_MAGNITUDE - SUN_MAGNITUDE ) );
const moonAboveAir = () => MOON_OVER_SUN * moonKeyGain();
// a gain on the moon's light above the air (`moonAboveAir`), so on both of its paths: the key
// (keyColor: the land, the sea's glitter, the haze's scatter of the key) and the sky, the clouds
// and the haze. A debug parameter, not the look's: 1 is the physical full moon, and so it stays.
const moonKeyGain = tune( 'lighting.moon.gain', 1, { min: 0.05, max: 4, scale: 'log' } );
const _moonKey = new THREE.Color();
const _moonHue = new THREE.Color();
const _cloudMoon = new THREE.Color();

export function trackSky( sky: Sky, celestial: DayCycleSnapshot | null, weather: Readonly<WeatherState> | null, coverageOverride: number | null = null, skyMix: readonly number[] | null = null ): void {

	const look = sky.look;
	if ( celestial ) {

		sky.sunDir.copy( celestial.sunDir );
		sky.elevation = celestial.elevation;
		sky.azimuth = celestial.azimuth;

	} else {

		const el = THREE.MathUtils.degToRad( sky.elevation ), az = THREE.MathUtils.degToRad( sky.azimuth );
		sky.sunDir.set( Math.cos( el ) * Math.cos( az ), Math.sin( el ), Math.cos( el ) * Math.sin( az ) ).normalize();

	}
	const illuminance = sky.sunIlluminance * SCENE_RADIANCE_SCALE;
	sky.solarIlluminance.copy( sky.sunTint ).multiplyScalar( illuminance );
	// The atmosphere's slices are cut again only when the sun or the moon has moved.
	sky.atmosphere.sunDir.value.copy( sky.sunDir );
	// The moon stands where the clock puts it (DayCycle.ts MOON): drawn there, lighting the sky
	// and the haze from there, as it shades the island.
	if ( celestial ) sky.atmosphere.moonDir.value.copy( celestial.moonDir );
	else sky.atmosphere.moonDir.value.copy( sky.sunDir ).negate();
	// The moon lights the sky as the sun does: its light above the air (`moonAboveAir`) times the
	// sky's gain the sun's light takes at the moon's height (`skyGain` on the evening's keys, with
	// the weather's day sky there), at the moonlit sky's own hue (`moonSkyHue`) and the moonlit
	// sky's share (the snapshot's `moonKeyGain`, as the key's: at its magnitude the moon's light
	// lies under the twilight's, so it takes no window of its own, DayCycle.ts). The key takes the
	// moon's share of the key and its own gain (`moonKeyGain`).
	const cloud = skyMix ? cloudedShare( skyMix ) : 0;
	sky.rising = sunRising( sky.sunDir );
	const moonGain = celestial ? duskForce( 'moonKey', celestial.moonKeyGain ) : 0;
	const moonSkyShare = celestial ? duskForce( 'moonSky', celestial.moonKeyGain ) : 0;
	sky.moonElevation = celestial ? celestial.moonElevation : MOON.elevation;
	sky.moonKeyShare = moonGain;
	sky.moonSkyShare = moonSkyShare;
	sky.moonAbove = celestial ? illuminance * moonAboveAir() : 0;
	// the key: the light above the air through the air toward the moon (its luminance), at the
	// key's hue (`moonKey` at unit luminance)
	const toMoon = sky.atmosphere.transmittanceToMoon();
	sky.moonKeyLevel = sky.moonAbove * luminance709( toMoon.r, toMoon.g, toMoon.b );
	const moonKey = sky.moonKeyLevel * moonGain / look.moonKeyLuminance;
	sky.moonKeyColor.setRGB( look.moonKey.colour[ 0 ] * moonKey, look.moonKey.colour[ 1 ] * moonKey, look.moonKey.colour[ 2 ] * moonKey );
	// the weather's moonlit sky at the moon's height, on its own level (Weather.ts `moonSky`)
	sky.daySkyMoon = weather ? daySkyAt( weather.moonSky, weather.moonSkyExponent, sky.moonElevation ) : 1;
	const moonSky = moonSkyShare * sky.moonAbove * luma( skyGain( look, sky.dayWindows, sky.moonElevation, false, cloud, sky.daySkyMoon, _skyMoon ) );
	sky.atmosphere.moonIlluminance.value.setRGB( look.moonSkyHue[ 0 ] * moonSky, look.moonSkyHue[ 1 ] * moonSky, look.moonSkyHue[ 2 ] * moonSky );
	// the moon's beam the air scatters, as the sun's (`sunBeamIlluminance`): the sky's gain at a
	// day sky of 1; the moon's visible share is in its sky's share already
	const moonBeam = moonSkyShare * sky.moonAbove * luma( skyGain( look, sky.dayWindows, sky.moonElevation, false, cloud, 1, _skyMoon ) );
	sky.moonBeamIlluminance.setRGB( look.moonSkyHue[ 0 ] * moonBeam, look.moonSkyHue[ 1 ] * moonBeam, look.moonSkyHue[ 2 ] * moonBeam );
	// the weather's day sky at this sun (Weather.ts `daySkyAt`), on the sun's light the sky takes
	// (`skyGain`), the clouds' sunlight (`trackClouds`) and the distant sea's (SkyBake.ts `beginBake`)
	sky.daySky = weather ? daySkyAt( weather.daySky, weather.daySkyExponent, sky.elevation ) : 1;
	sky.atmosphere.sunIlluminance.value.copy( sky.solarIlluminance ).multiply( skyGain( look, sky.dayWindows, sky.elevation, sky.rising, cloud, sky.daySky, _skyGain ) );
	// (on overcast's share it fades with the setting disk's visible share, as the land's key does:
	// see `sunBeamIlluminance`)
	const beamFade = 1 + ( sky.atmosphere.sunVisibleShareToSun() - 1 ) * ( skyMix ? skyMix[ 3 ] ?? 0 : 0 );
	sky.sunBeamIlluminance.copy( sky.solarIlluminance ).multiply( skyGain( look, sky.dayWindows, sky.elevation, sky.rising, cloud, 1, _skyBeam ) ).multiplyScalar( beamFade );
	// the open sky's gain behind the clouds, on each body's share of the sky: the sun's at the
	// sun's height, the moon's at the moon's, as each body's sky light takes its gain (`skyGain`)
	const off = debugFlags.skyGain === false;
	sky.gapGain.sun.value = off ? 1 : clearSky( look, sky.elevation, sky.rising, 0 ) / clearSky( look, sky.elevation, sky.rising, cloud );
	sky.gapGain.moon.value = off ? 1 : clearSky( look, sky.moonElevation, false, 0 ) / clearSky( look, sky.moonElevation, false, cloud );
	sky.atmosphere.updateSync();
	// Cap the hue of the air's extinction first, then fade by the disk's visible share, so the
	// setting disk only dims the key and does not bleach it.
	const sunVisible = sky.atmosphere.sunVisibleShareToSun();
	if ( sunVisible > 0 ) sky.sunColor.copy( keyHue( look, sky.atmosphere.transmittanceToSun(), sky.sunTint, sky.rising ) ).multiplyScalar( sunVisible * illuminance );
	else sky.sunColor.setRGB( 0, 0, 0 );
	// The key turns from the sun to the moon over the handoff (DayCycle `keyDir`) and holds there
	// all night: everything that reads the key's direction (the water's glint, the refracted
	// light under the water, the clouds, the haze) then sees the moon where it is drawn. By day
	// (no handoff) the key is the sun's direction itself.
	sky.keyDir.copy( celestial?.keyDir ?? sky.sunDir );
	sky.keyIsMoon = moonGain > 0;
	sky.keyColor.copy( sky.sunColor ).multiplyScalar( lowKey( look, sky.elevation ) * duskForce( 'sun', 1 ) );
	const moonColor = sky.night.moonColor.value;
	if ( celestial ) {

		// The full moon's key, some 19 stops under a high sun's, in the island's own hue: only the
		// moon's share of the key takes it, so over the handoff
		// it grows with that share, and by day (no moon) the sun's key is untouched.
		sky.moonKeyAbove.copy( sky.moonKeyColor ).multiply( moonHue( look, _moonHue ) );
		sky.keyColor.add( sky.moonKeyAbove );
		sky.night.moonDir.value.copy( celestial.moonDir );
		sky.night.frame.value.copy( celestial.starFrame );
		// (the moon's light does not follow its phase: DayCycle's header)
		sky.night.moonWash.value = celestial.moonVisibility;
		// The disk sits in the upper tones at night exposure so its maria stay visible, as to the
		// eye; the moonlight on the island is the key light above, not this disk.
		moonColor.setRGB( 0.04, 0.041, 0.043 ).multiplyScalar( 0.32 * celestial.moonVisibility * sky.sunIlluminance / SUN_ILLUMINANCE );
		sky.grade.night = displayNight( sky.dayWindows, celestial.daylight, celestial.elevation, sky.rising, cloud );
		sky.grade.exposureNight = exposureNight( sky.dayWindows, celestial.elevation, sky.rising, cloud );
		const stars = starNight( sky.dayWindows, celestial.daylight, celestial.elevation, sky.rising, cloud );
		sky.night.starVisibility.value = duskForce( 'stars', celestial.nightVisibility * stars ) * ( 1 - 0.22 * celestial.moonVisibility );
		// the night's ambient in the clouds and the distant sea: a uniform sky's radiance that gives
		// level ground the tables' `nightAmbientShare` of the moon's key on it (the key's light at normal
		// incidence, the moon's sine), by its share (DayCycle.ts `nightAmbientShare`)
		const floor = sky.moonKeyLevel * Math.sin( THREE.MathUtils.degToRad( sky.moonElevation ) ) * look.nightAmbientShare / Math.PI
			* duskForce( 'nightAmbient', nightAmbientShare( sky.dayWindows, celestial.elevation, sky.rising, cloud ) );
		sky.displayNightAmbient.value.setRGB( look.nightAmbientHue[ 0 ] * floor, look.nightAmbientHue[ 1 ] * floor, look.nightAmbientHue[ 2 ] * floor );
		sky.grade.exposure = celestial.exposure;
		sky.grade.daylight = celestial.daylight;
		sky.landGain.value = landSky( look, celestial.elevation, sky.rising, sky.landTint.value, cloud );
		const direct = weather?.directVisibility ?? 1;
		const shade = clearSkyFill( look, celestial.elevation, direct ), low = lowSun( look, celestial.elevation, direct );
		const tint = sky.landTint.value;
		if ( shade > 0 ) {

			sky.landGain.value *= 1 + shade;
			tint.set( look.clearSkyFillTint[ 0 ], look.clearSkyFillTint[ 1 ], look.clearSkyFillTint[ 2 ], look.clearSkyFillTint[ 3 ] * shade );

		}
		if ( low > 0 ) tint.set( tint.x + ( look.lowSunTint[ 0 ] - tint.x ) * low, tint.y + ( look.lowSunTint[ 1 ] - tint.y ) * low,
			tint.z + ( look.lowSunTint[ 2 ] - tint.z ) * low, Math.max( tint.w, look.lowSun.amount * low ) );
		const clouded = skyMix ? cloudedHighSun( celestial.elevation, skyMix, 12 ) : 0;
		if ( clouded > 0 ) tint.set( look.cloudLightTint[ 0 ], look.cloudLightTint[ 1 ], look.cloudLightTint[ 2 ], look.cloudLightTint[ 3 ] * clouded );

	} else {

		sky.night.moonDir.value.copy( sky.sunDir ).negate();
		sky.moonKeyAbove.setRGB( 0, 0, 0 );
		moonColor.setRGB( 0, 0, 0 );
		sky.night.starVisibility.value = 0;
		sky.displayNightAmbient.value.setRGB( 0, 0, 0 );
		sky.grade.exposure = sky.exposure.day;
		sky.grade.daylight = 1;
		sky.grade.night = 0;
		sky.grade.exposureNight = 0;
		sky.landTint.value.set( 1, 1, 1, 0 );
		sky.landGain.value = 1;

	}
	sky.keyColorAboveClouds.copy( sky.keyColor );
	const coverage = coverageOverride ?? weather?.coverage ?? 0;
	const overcast = weather ? THREE.MathUtils.smoothstep( coverage, 0.65, 0.98 ) : 0;
	// the weather's sky light (its day sky is on the sun's light alone: `skyGain`, `trackClouds`)
	const ambientGain = weather?.ambientGain ?? 1;
	const directVisibility = weather?.directVisibility ?? 1;
	// The base exposure (the frame's is metered from the light: post/Exposure.ts; this one is
	// what the night sky's own light is tuned against) adapts to the dimmer diffuse daylight
	// under a closed deck, as tidewater's auto exposure would: it lifts a frame darker than its
	// metered level only partly (by the ratio's 0.8th power), 4.7 times under a closed deck; less so in
	// rain.
	sky.grade.exposure *= 1 + 3.7 * overcast * sky.grade.daylight * ( 1 - 0.7 * ( weather?.rain ?? 0 ) );
	// Through the blue hour a closed deck's adaptation goes on rising as the light goes (DayCycle
	// `overcastAdaptation`, none from -6 degrees down): the night under the deck takes no gain of
	// its own, being lit by the moon through the deck as the day is by the sun.
	sky.grade.exposure *= 1 + overcast * ( celestial?.overcastAdaptation ?? 0 );
	// The night sky's own lights, the stars and the moon's disk (nothing they light), keep the
	// brightness they were tuned at, under the exposure the night had then, but for the night's
	// own gain (the exposure's `nightGain`, look/ExposureLook.ts), which brightens them with the island; under a
	// deck the clouds drawn over them hide them.
	const nightSkyLevel = celestial ? Math.min( 1, celestial.nightSkyLevel ) : 1;
	sky.night.starVisibility.value *= nightSkyLevel;
	moonColor.multiplyScalar( nightSkyLevel );
	sky.sunColor.multiplyScalar( directVisibility );
	sky.keyColor.multiplyScalar( directVisibility );
	// Under a deck the sun's light is whiter than a clear day's at the same height, part of it
	// scattered in the cloud: the key keeps its luminance and moves `keyDeckWhite` of the way to
	// white under overcast, where the sun still shines through, and all the way under rain's
	// closed deck (a rainy evening's light is not a sunset's).
	const keyLuminance = sky.keyColor.r * 0.2126 + sky.keyColor.g * 0.7152 + sky.keyColor.b * 0.0722;
	const rain = weather?.rain ?? 0;
	sky.keyColor.lerp( _white.setScalar( keyLuminance ), overcast * ( look.keyDeckWhite + ( 1 - look.keyDeckWhite ) * rain ) );
	sky.solarIlluminance.multiplyScalar( directVisibility );
	moonColor.multiplyScalar( directVisibility );
	sky.night.starVisibility.value *= ambientGain;
	// the drawn moon's disk takes its phase's light (DayCycle `moonSunDir`)
	sky.night.sunDir.value.copy( celestial ? celestial.moonSunDir : sky.sunDir );
	sky.displaySunDir.value.copy( sky.sunDir );
	sky.displaySolarIlluminance.value.copy( sky.solarIlluminance );
	sky.displayAmbientGain.value = ambientGain;
	// the key's floor under cloud (`keyThroughCloud`), after the clouds' shell and density are set
	const overcastShare = skyMix ? skyMix[ 3 ] ?? 0 : 0;
	const keyLight = sky.keyColorAboveClouds.r * 0.2126 + sky.keyColorAboveClouds.g * 0.7152 + sky.keyColorAboveClouds.b * 0.0722;
	const moonLight = sky.moonKeyAbove.r * 0.2126 + sky.moonKeyAbove.g * 0.7152 + sky.moonKeyAbove.b * 0.0722;
	const sunShare = keyLight > 0 ? Math.max( keyLight - moonLight, 0 ) / keyLight : 0;
	sky.deckDay.value = overcastDeckLight( sky.elevation );
	const seenNight = 2 ** ( - ( weather?.seenNight ?? 0 ) * sky.grade.night );
	shared.seenNight.value = seenNight;
	sky.seenDeck.value = seenNight;
	sky.domeGapGain.moon.value = sky.gapGain.moon.value * seenNight;
	const floorSun = 1 - overcastShare * sunShare * ( 1 - overcastFloorSun( sky.elevation ) );
	sky.castShadow.value = weather?.castShadow ?? 1;
	trackClouds( sky, celestial, weather, coverage, overcast, illuminance, skyMix );
	const keyFloor = Math.min( look.keyThroughCloud() * sky.clouds.layerTransmission(), 1 ) * floorSun;
	sky.keyCloudFloor.value = keyFloor;
	sky.keyFloorUnderDeck = keyFloor * overcastShare;
	// (`flags=streaks:<n>` scales the cirrus's amount, 0 turns it off)
	sky.clouds.streaks.value = ( weather?.streaks ?? 0 ) * ( typeof debugFlags.streaks === 'number' ? debugFlags.streaks : 1 );
	// A rainbow needs falling rain and direct sun at once, the sun lower than 42 degrees behind
	// the viewer: most often as a shower clears, while it is still raining a few kilometres off.
	// The clouds draw it in their sunlit rain curtains.
	const showers = weather?.showers ?? 0, sunlit = weather?.directVisibility ?? 1;
	sky.clouds.showers.value = showers;
	// Drops fall at about 9 m/s and drift with the wind, slower near the sea than aloft; a gale
	// slants them steeply.
	sky.clouds.rainSlant.value = Math.min( ( weather?.windSpeed ?? 3.8 ) / 11, 1 );
	// A fog's cover is mist and stratus, not rain cells: its broken sky does not shower.
	sky.clouds.heapShowers.value = 1 - THREE.MathUtils.smoothstep( weather?.fog ?? 0, 0.3, 0.8 );
	sky.clouds.rainbow.value = THREE.MathUtils.smoothstep( showers, 0.04, 0.3 ) * THREE.MathUtils.smoothstep( sunlit, 0.3, 0.7 )
		* THREE.MathUtils.smoothstep( sky.sunDir.y, 0, 0.08 ) * ( 1 - THREE.MathUtils.smoothstep( sky.sunDir.y, 0.6, 0.72 ) );
	sky.weather.haze = weather?.haze ?? 0;
	sky.weather.rain = weather?.rain ?? 0;
	// (the morning mist is the haze's: weather/HazeKeys.ts)
	sky.weather.fog = weather?.fog ?? 0;
	sky.weather.windSpeed = weather?.windSpeed ?? 3.8;
	sky.weather.windDirection = weather?.windDirection ?? 40;

}

function trackClouds( sky: Sky, celestial: DayCycleSnapshot | null, weather: Readonly<WeatherState> | null, coverage: number, overcast: number, illuminance: number, skyMix: readonly number[] | null ): void {

	const look = sky.look, clouds = sky.clouds;
	clouds.coverage.value = coverage;
	clouds.density.value = weather?.density ?? 1;
	// The shadow's share of the deck by the weather's shares of the four skies, or the weather's
	// own without them: a weather change or a retune rebakes the whole map (`bake`), a front
	// under way refreshes it row by row.
	clouds.shadowDensity.value = skyMix ? look.shadowDensity.reduce( ( sum, k, s ) => sum + ( skyMix[ s ] ?? 0 ) * k.shadowDensity, 0 ) : weather?.shadowDensity ?? 1;
	// The shell by the same shares (the high one's without them): every reader follows its
	// uniforms, the march, the shadow, the rain, the haze's shafts and the lightning.
	const row = ( key: keyof CloudSky ) => skyMix ? look.cloudSkies.reduce( ( sum, k, s ) => sum + ( skyMix[ s ] ?? 0 ) * k[ key ], 0 ) : look.cloudSkies[ 0 ][ key ];
	const base = row( 'base' );
	clouds.setShell( base, row( 'top' ) - base, skyMix ? cloudedShare( skyMix ) : 0 );
	clouds.cloudType.value = row( 'type' );
	clouds.typeSpread.value = row( 'spread' );
	clouds.farFade.value = row( 'farFade' );
	clouds.rain.value = weather?.rain ?? 0;
	// Cloud tops a couple of kilometres up keep the sun a little after it has set below. The
	// rising sun's reaches the shell where it stands (its middle, 6.6 km under the clear and
	// fair skies' cumulus), so the clouds take the dawn's light from about -2.6 degrees, as the
	// Earth's shadow sinks below them; the
	// height goes back to the evening's from the sun's rise to 3 degrees, so the morning's
	// higher sun lights them as the afternoon's does, and with a clouded share (Weather.ts
	// `cloudedShare`).
	const cloud = skyMix ? cloudedShare( skyMix ) : 0;
	const rising = sky.rising && cloud < 1;
	const toEvening = Math.max( cloud, THREE.MathUtils.smoothstep( celestial?.elevation ?? 90, 0, 3 ) );
	const middle = ( clouds.shellBottom.value + clouds.shellThickness.value / 2 ) / 1000;
	const sunHeight = rising ? 2 + ( middle - 2 ) * ( 1 - toEvening ) : 2;
	const sunAbove = sky.atmosphere.sunlightAt( sky.sunDir, sunHeight ).multiply( sky.sunTint ).multiplyScalar( illuminance );
	// Under a closed deck the sun reaches the deck as it reaches the island, scattered through
	// the cloud: it keeps its luminance and loses its colour as the key does (track), so an
	// overcast evening's deck is not a sunset's, but for the glow about the sun's place:
	// overcast's deck's forward lobe keeps the colour of the sun above the tops (Clouds.ts
	// `lobeTint`): per channel the colour over the grey's, c / ( c + ( 1 - c ) overcast ), by
	// overcast's share of the sky (1 under the other skies, and without the sun: below). Its
	// colour is the sun's from overhead, not along the sun's own path: taken along it, a low
	// sun's reddening would come back whole and the glow about a low sun read orange, a sunset's
	// under the deck.
	const overcastShare = skyMix ? skyMix[ 3 ] ?? 0 : 0;
	// the closed layer's share (Clouds.ts `overcastShare`)
	clouds.overcastShare.value = overcastShare;
	// the closed layer passes the weather's day sky whole: its own mean transmission is divided
	// out, so `daySky` alone sets the light under it (Clouds.ts `columnLevel`, `layerTransmission`)
	clouds.columnLevel.value = THREE.MathUtils.lerp( 1, 1 / clouds.layerTransmission(), overcastShare );
	const sunY = luma( sunAbove );
	const tint = clouds.lobeTint.value.setRGB( 1, 1, 1 );
	const above = overcastShare > 0 ? sky.atmosphere.transmittanceAt( _zenith, sunHeight ).multiply( sky.sunTint ) : null;
	const aboveY = above ? luma( above ) : 0;
	if ( above && sunY > 0 && aboveY > 0 ) for ( const c of [ 'r', 'g', 'b' ] as const ) {

		const chroma = above[ c ] / aboveY;
		tint[ c ] = THREE.MathUtils.lerp( 1, chroma / ( chroma + ( 1 - chroma ) * overcast ), overcastShare );

	}
	sunAbove.lerp( _white.setScalar( sunY ), overcast );
	// the weather's day sky (Weather.ts `daySkyAt`): the sun's light the clouds take, which lights
	// a deck's base and the land's sky under it, is the sky's light from the sun (`skyGain`)
	sunAbove.multiplyScalar( sky.daySky );
	// The clouds belong to the seen sky: their moonlight comes in with the night sky's own light
	// (DayCycle.ts `cloudMoonShare`), not before it while the afterglow still lights the sky behind
	// them; under cloud with the clouded night's, by the clouded share.
	const moonShare = ! celestial ? 0 : duskForce( 'cloudMoon', cloudMoonShare( sky.dayWindows, celestial.elevation, sky.rising, cloud ) );
	const moonGain = celestial ? moonShare * celestial.moonVisibility : 0;
	const sunUp = ! celestial || celestial.elevation > - 2.5 || rising;
	// The march's depth shades the clouds' sky light as well as their direct light, and the same
	// direction casts their shadow, so it moves without a jump even while no direct light reaches
	// them: a jump there stepped the deck's brightness, half a stop on the meter under overcast.
	// It sets first, its shadow fading out on the way down, so its turn round to the sun, under
	// the horizon, sweeps no shadow across the island. Under a closed deck, on overcast's share,
	// it fades to the deck's mean transmittance instead (Clouds.ts `deckShadowShare`): faded to
	// lit, the haze under the deck took the sun the deck still hid, and the sky's light rose by
	// up to two thirds as the sun set.
	const elevation = celestial?.elevation ?? 90;
	const setting = THREE.MathUtils.smoothstep( elevation, CLOUD_TURN[ 0 ], CLOUD_TURN[ 1 ] ), round = THREE.MathUtils.smoothstep( elevation, CLOUD_TURN[ 1 ], CLOUD_TURN[ 2 ] );
	clouds.shadowShare.value = elevation < CLOUD_TURN[ 1 ] ? 1 - setting : THREE.MathUtils.smoothstep( sky.sunDir.y, CLOUD_SHADOW_SET[ 0 ], CLOUD_SHADOW_SET[ 1 ] );
	clouds.deckShadowShare.value = overcastShare;
	clouds.lightDir.value.copy( setting === 0 ? sky.keyDir : sky.sunDir );
	if ( celestial && setting > 0 && round < 1 ) {

		const from = celestial.moonDir;
		const y = THREE.MathUtils.lerp( from.y, sky.sunDir.y, setting ), start = Math.atan2( from.z, from.x );
		const turn = Math.atan2( sky.sunDir.z, sky.sunDir.x ) - start;
		const azimuth = start + Math.atan2( Math.sin( turn ), Math.cos( turn ) ) * round, across = Math.sqrt( 1 - y * y );
		clouds.lightDir.value.set( across * Math.cos( azimuth ), y, across * Math.sin( azimuth ) );

	}
	// the low deck's forward light on level ground at the key's height (Clouds.ts forwardLevel): the
	// narrow lobe's, the light overcast's share spreads by the diffusion law (a lobe of mean cosine
	// `overcastBroad`'s), and the two by that share, the deck's level
	clouds.lobe.value = clouds.forwardLobe.value;
	clouds.broadShare.value = overcastShare;
	// (by day: `deckDay`)
	clouds.deckFlat.value = ( 1 - clouds.deckGradation.value ) * overcastShare * sky.deckDay.value;
	clouds.weight.value = THREE.MathUtils.lerp( clouds.forwardWeight.value, clouds.overcastWeight.value, overcastShare );
	const keyY = clouds.lightDir.value.y;
	clouds.narrowLevel.value = forwardLevel( keyY, clouds.lobe.value );
	clouds.forwardLevel.value = overcastShare > 0
		? THREE.MathUtils.lerp( clouds.narrowLevel.value, forwardLevel( keyY, clouds.overcastBroad.value ), overcastShare )
		: clouds.narrowLevel.value;
	const disc = 1 - THREE.MathUtils.smoothstep( clouds.lightDir.value.y, SHADOW_DISC[ 0 ], SHADOW_DISC[ 1 ] );
	clouds.shadowDisc.value = disc * SUN_ANGULAR_RADIUS;
	// With the whole disc under the floor the rays are one ray.
	clouds.shadowRays.value = disc > 0 && clouds.lightDir.value.y + clouds.shadowDisc.value > SHADOW_SLOPE_FLOOR ? SHADOW_DISC_RAYS : 1;
	const keyElevation = THREE.MathUtils.radToDeg( Math.asin( clouds.lightDir.value.y ) );
	clouds.shadowAirNarrow.value = THREE.MathUtils.smoothstep( keyElevation, SHADOW_AIR_NARROW[ 0 ], SHADOW_AIR_NARROW[ 1 ] );
	clouds.lightColor.value.copy( sunUp ? sunAbove : new THREE.Color( 0, 0, 0 ) );
	// The moon lights the clouds as the sun does: its light above the air through the air down to
	// the same height (`sunlightAt` toward the moon, in its colour), the weather's day sky at the
	// moon's height and the moon's share of the clouds' light, at the island's moonlight's hue
	// (`moonKey` times `moonHue`, at unit luminance) and no other.
	const moonCloud = _cloudMoon.setRGB( 0, 0, 0 );
	if ( celestial ) moonCloud.setRGB( look.moonKey.colour[ 0 ], look.moonKey.colour[ 1 ], look.moonKey.colour[ 2 ] ).multiply( moonHue( look, _moonHue ) )
		.multiplyScalar( sky.moonAbove * moonGain * sky.daySkyMoon / look.moonKeyLuminance ).multiply( sky.atmosphere.sunlightAt( celestial.moonDir, sunHeight ) );
	const moonY = luma( moonCloud );
	// The luminance of that light, for the stores' powers of two (PreExposure.ts `storeStops`),
	// the sun's before the day sky's gain: under a closed deck what a bake holds brightest is the
	// deck's base, which takes the sun through the cloud, not the day sky's share of it. Under
	// the moon each store's range follows the weather's clouds (`storeNight`).
	clouds.storeLight = ( sunUp ? sunY : 0 ) + moonY;
	const moonPart = clouds.storeLight > 0 ? moonY / clouds.storeLight : 0;
	clouds.storeStops = storeStops( clouds.storeLight, moonPart, storeNight( look, 'dome', skyMix ) );
	sky.layerStops = storeStops( clouds.storeLight, moonPart, storeNight( look, 'layer', skyMix ) );
	sky.panoramaStops = storeStops( clouds.storeLight, moonPart, storeNight( look, 'panorama', skyMix ) );
	const light = clouds.lightColor.value;
	light.add( moonCloud );
	// (the low sun's lift of the beam, as `directGain`'s)
	const beam = 1 + 0.7 * lowSunOf( sky.elevation, coverage );
	// The cirrus's sunlight before the atmosphere, lifted alike: each of its points takes it through the air
	// toward the sun at the sun's elevation there, and loses it as the disc sinks under the
	// layer's own horizon there (Clouds.ts `streaksAt`), so no one moment ends the whole layer's.
	clouds.streakSunColour.value.copy( sky.sunTint ).multiplyScalar( illuminance * beam * sky.daySky );
	const { r: dr, g: dg, b: db } = keyed( look.streakDirect, elevation, moonShare, _streakKey, cloud, rising );
	clouds.keyDirect.value.setRGB( dr, dg, db );
	const { r: ar, g: ag, b: ab } = keyed( look.streakAmbient, elevation, moonShare, _streakKey, cloud, rising );
	clouds.keyAmbient.value.setRGB( ar, ag, ab );
	// (the moon's beam lifted as the sun's is, at the moon's height)
	clouds.moonLight.value = moonY * ( 1 + 0.7 * lowSunOf( sky.moonElevation, coverage ) );
	// The night ambient is the clouds' own (the clear sky draws none), taken as their sky light.
	clouds.nightAmbient.value.copy( sky.displayNightAmbient.value );
	// A low deck's column passes down the key's light above its tops: by day the sun's above the
	// cloud, by night the island's moonlight (`moonKeyAbove`), which keeps the island's level and
	// hue, not the clouds' (their moonlight belongs to the seen sky), and loses its colour under a
	// closed deck as the sun's does (above), and takes the weather's day sky at the moon's height
	// as the sun's takes it at the sun's (`daySkyMoon`; 1 but under overcast). The deck's base and
	// the land under it then take one moon, as by day they take one sun.
	const moonAbove = _moonKey.copy( sky.moonKeyAbove );
	const column = clouds.columnKey.value.copy( moonAbove.lerp( _white.setScalar( luma( moonAbove ) ), overcast ).multiplyScalar( sky.daySkyMoon ) );
	// A heap's part of the column's light is the sun's alone (Clouds.ts `columnSunShare`): it
	// goes by the sun's share of this key itself, so that part is the sun's addend and leaves
	// with it, whenever the moon's comes in. Lit by the moon too, the fair clouds read brighter
	// in late twilight than the night's level was set with, and the meter took the scene down
	// by a third of a stop at -11 to -12 degrees.
	const moonColumn = luma( column ), sunColumn = sunUp ? luma( sunAbove ) : 0;
	clouds.columnSunShare.value = sunColumn + moonColumn > 0 ? sunColumn / ( sunColumn + moonColumn ) : 1;
	if ( sunUp ) column.add( sunAbove );
	else tint.setRGB( 1, 1, 1 );
	// Low sun: lift the reddened beam, and the cool violet fill on the shaded sides with it. The
	// lift fades as the sky closes: a fair sky's scattered cumulus keep their golden light, a
	// cloudy sky's cover shows near neutral, and a closed deck, whose sunlight has lost its
	// colour, takes none.
	const lowSun = lowSunOf( sky.elevation, coverage );
	clouds.directGain.value = 1 + 0.7 * lowSun;
	clouds.ambientGain.value = 1 + 0.35 * lowSun;
	clouds.openSkyGain.sun.value = sky.gapGain.sun.value;
	clouds.openSkyGain.moon.value = sky.gapGain.moon.value;

}
