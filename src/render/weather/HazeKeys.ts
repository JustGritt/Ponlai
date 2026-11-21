import * as THREE from 'three/webgpu';
import type { DayWindows } from '../look/DayLook.ts';
import { duskForce, duskForced, duskIn, moonKeyShare } from '../sky/DayCycle.ts';
import { MIX_SKIES, WEATHER_PRESETS, cloudedShare, deckLight, overcastDeckLight } from '../sky/Weather.ts';
import { cloudBase, type SkyTables } from '../look/SkyLook.ts';
import { FOG_NOISE } from './FogNoise.ts';
import { blankKey, type FullKey, type HazeKey, type HazeKeyRows, type HazeTables } from '../look/HazeLook.ts';

// The haze's key at an hour and a weather, eased by the sun's elevation and blended by weather.

export type { HazeKey };

// The haze's key at an hour and a weather: the look's keys (look/HazeLook.ts `createHazeTables`,
// whose header says what each key holds; every function here takes the tables as its first
// argument) eased by the sun's elevation, the morning's and the evening's apart; the share of the
// air's light over the land (`groundLight`) leaves at dawn over the evening's night window
// mirrored, so a dawn and a dusk at one elevation lay one share; and blended by the weather's shares of the four skies, its wind, its rain and the fog's switches.
// A function of those alone, never of the path to them: a paused frame and a scene jump hold one
// value, and a change of hour or weather writes the haze's uniforms only.

// (`skyDepth` is eased from each key's own value: the keys without a shape of their own keep
// SHAPE's 0 rather than taking the day's)
const FIELDS = [ 'amount', 'sigmaRed', 'sigmaGreen', 'sigmaBlue', 'desaturate', 'mist', 'sunRayleigh', 'rayleighWeight', 'mieWeight', 'skyDepth', 'groundLight' ] as const;
const SHAPE_FIELDS = [ 'top', 'width', 'aerosolShare', 'aerosolHeight' ] as const;
const FOG_FIELDS = [ 'fogLayer', 'neutral', 'scatter', 'fogTop', 'fogWidth', 'visibility' ] as const;

// the key of `rows` at `elevation`, its fields, its shape and its fog layer, into `out`, the keys
// `tables`'s, the night's windows `windows`'s
function keyAlong( tables: HazeTables, windows: DayWindows, rows: HazeKeyRows, elevation: number, out: FullKey ): void {

	const { keys: { day: dayKey, night: nightKey }, shaped, fogOf } = tables;

	let i = 0;
	while ( i < rows.length - 2 && elevation > rows[ i + 1 ][ 0 ] ) i ++;
	const [ e0, a ] = rows[ i ], [ e1, b ] = rows[ i + 1 ];
	// (a clear evening's night key comes in by stops, as the night's light does: DayCycle.ts
	// `duskIn`, the day windows' `depth.haze`)
	// (a debug page may force the night key's weight, DayCycle.ts `duskForce`: its segment's ease,
	// and above it a blend toward it)
	const night = duskForced( 'hazeNight' );
	const t = night !== null && a === nightKey ? 1 - night : rows === tables.setting && a === nightKey ? 1 - duskIn( windows, elevation, [ e0, e1 ], windows.depth.haze ) : THREE.MathUtils.smoothstep( elevation, e0, e1 );
	for ( const f of FIELDS ) out[ f ] = a[ f ] + ( b[ f ] - a[ f ] ) * t;
	// (`groundLight` at dawn: the evening's night segment, mirrored, by `duskIn` (see the header); an
	// ease along the morning rows would take half the air's light off the land at -9 degrees, where
	// the twilight's air is bright and the rods' lift still small)
	if ( night === null && rows === tables.rising && a === nightKey ) {

		const s = tables.setting, k = s.findIndex( ( row ) => row[ 1 ] === nightKey );
		out.groundLight = b.groundLight + ( nightKey.groundLight - b.groundLight ) * duskIn( windows, elevation, [ s[ k ][ 0 ], s[ k + 1 ][ 0 ] ], windows.depth.haze );

	}
	const sa = shaped.has( a ) ? a : dayKey, sb = shaped.has( b ) ? b : dayKey;
	for ( const f of SHAPE_FIELDS ) out[ f ] = sa[ f ] + ( sb[ f ] - sa[ f ] ) * t;
	const fa = fogOf( a ), fb = fogOf( b );
	for ( const f of FOG_FIELDS ) out[ f ] = fa[ f ] + ( fb[ f ] - fa[ f ] ) * t;
	if ( night === null || a === nightKey ) return;
	for ( const f of FIELDS ) out[ f ] += ( nightKey[ f ] - out[ f ] ) * night;
	for ( const f of SHAPE_FIELDS ) out[ f ] += ( nightKey[ f ] - out[ f ] ) * night;
	for ( const f of FOG_FIELDS ) out[ f ] += ( nightKey[ f ] - out[ f ] ) * night;

}
const _cloudedKey = blankKey();
const _key = blankKey();

/** The haze's key from the haze's `tables` (look/HazeLook.ts `createHazeTables`) under the day's
 * `windows` (look/DayLook.ts `createDayWindows`: the night key's depth through a clear evening and
 * the moon's key under cloud), with the sky's tables `sky` (look/SkyLook.ts `createSkyTables`: the
 * clouds' base, where a deck ends the aerosol) at the sun's `elevation` (degrees),
 * `rising` or setting, under the weather's
 * blend of the four skies, its wind (m/s), its rain (0 - 1), the sea fog's share (0 - 1,
 * its switch) and the parked fog weather's share of the sky (0 - 1: Weather.ts `origin` and
 * `target` by the front's progress). The fog weather counts as overcast in `skyMix`, but its air
 * keeps the sun's Rayleigh light the closed deck takes from overcast's and the fog layer's droplets
 * the clouded share takes back (see the header): the fog is parked and stays as it stands. The
 * share is taken out by a product. For the droplets, which go by the cloudy and overcast columns
 * together, it is exact at rest and on a front between the fog and the cloudy or overcast sky,
 * not on one between the fog and a clear or fair sky. For the sun's Rayleigh beam, which goes by
 * overcast's column alone, it is exact at rest and on a front between the fog and the overcast
 * sky, not on one between the fog and a clear, fair or cloudy sky.
 * Writes into `out`. */
export function hazeKeyAt( tables: HazeTables, windows: DayWindows, sky: SkyTables, elevation: number, rising: boolean, skyMix: readonly number[], wind: number, rain: number, dayFog: number, seaFog: number, fogWeather: number, target: HazeKey ): HazeKey {

	const out = _key;
	keyAlong( tables, windows, rising ? tables.rising : tables.setting, elevation, out );
	// a clouded evening's afterglow (see the header)
	const clouded = cloudedShare( skyMix );
	if ( ! rising && clouded > 0 ) {

		keyAlong( tables, windows, tables.settingClouded, elevation, _cloudedKey );
		for ( const f of FIELDS ) out[ f ] += ( _cloudedKey[ f ] - out[ f ] ) * clouded;
		for ( const f of SHAPE_FIELDS ) out[ f ] += ( _cloudedKey[ f ] - out[ f ] ) * clouded;
		for ( const f of FOG_FIELDS ) out[ f ] += ( _cloudedKey[ f ] - out[ f ] ) * clouded;

	}
	// the day's fog layer by its switch: off, every key holds no fog (see the header)
	if ( dayFog < 1 ) for ( const f of FOG_FIELDS ) out[ f ] += ( tables.noFog[ f ] - out[ f ] ) * ( 1 - dayFog );
	let keep = 0, gain = 0, red = 0, green = 0, blue = 0, aerosol = 0, height = 0, top = 0, width = 0, haze = 0, sunRayleigh = 0, inverseTop = 0;
	for ( let s = 0; s < tables.keep.length; s ++ ) {

		const w = skyMix[ s ] ?? 0;
		keep += w * tables.keep[ s ];
		gain += w * tables.gain[ s ];
		red += w * tables.skyScale[ s ].sigmaRed;
		green += w * tables.skyScale[ s ].sigmaGreen;
		blue += w * tables.skyScale[ s ].sigmaBlue;
		aerosol += w * tables.skyScale[ s ].aerosolScale;
		height += w * tables.skyScale[ s ].aerosolHeight;
		top += w * tables.skyScale[ s ].topScale;
		width += w * tables.skyScale[ s ].widthScale;
		haze += w * tables.skyScale[ s ].haze / WEATHER_PRESETS[ MIX_SKIES[ s ] ].haze;
		sunRayleigh += w * tables.sunRayleigh[ s ];
		if ( tables.decked[ s ] ) inverseTop += w / cloudBase( sky, s );

	}
	out.desaturate = 1 - ( 1 - out.desaturate ) * keep;
	out.gain = gain;
	out.sigmaRed *= red;
	out.sigmaGreen *= green;
	out.sigmaBlue *= blue;
	out.aerosolShare *= aerosol;
	out.aerosolHeight *= height;
	// the fog layer under a clear or a fair sky, by its share, and the sea fog by its switch at
	// every hour, which takes the open sky's layer's place: the ground layer takes its top and its
	// edge's width; without it cloudy and overcast keep their own air (see the header)
	const sea = seaFog, open = out.fogLayer * ( 1 - clouded ) * ( 1 - sea );
	const fog = open + sea;
	out.neutral *= 1 - clouded;
	out.fogCoverHigh = FOG_NOISE.high();
	if ( sea > 0 ) {

		for ( const f of [ 'neutral', 'scatter', 'fogTop', 'fogWidth', 'visibility' ] as const ) out[ f ] = ( out[ f ] * open + tables.seaFog[ f ] * sea ) / fog;
		out.fogCoverHigh = ( out.fogCoverHigh * open + tables.seaFog.coverHigh * sea ) / fog;

	}
	// the droplets' share of the ground layer's light follows the fog layer's own share, the
	// clouded share taking it back with the layer (the sea fog's kept); the parked fog weather, which
	// counts as clouded, keeps the key's whole (exact at rest and on a front between the fog and the
	// cloudy or overcast sky: see the header). (Written so that with no cloud, or under the fog
	// weather alone, the key's share passes through unchanged to the bit.)
	const dropletLayer = out.fogLayer * ( 1 - clouded * ( 1 - fogWeather ) );
	out.droplet = dropletLayer + ( 1 - dropletLayer ) * sea;
	out.fogLayer = fog;
	out.top = out.top * top * ( 1 - fog ) + out.fogTop * fog;
	out.width = out.width * width * ( 1 - fog ) + out.fogWidth * fog;
	out.aerosolTop = inverseTop > 0 ? 1 / inverseTop : Infinity;
	out.amount *= haze;
	out.sunRayleigh = 1 - ( 1 - out.sunRayleigh ) * ( 1 - sunRayleigh );
	out.mist *= ( 1 - THREE.MathUtils.smoothstep( wind, 3.5, 5 ) ) * ( 1 - rain );
	// (by night the whole clouded share, once the moon's key has come in: see the header)
	out.deck = duskForce( 'deck', Math.max( deckLight( elevation, clouded, skyMix[ 3 ] ?? 0 ), clouded * moonKeyShare( windows, elevation ) ) );
	out.overcastDeck = ( skyMix[ 3 ] ?? 0 ) * overcastDeckLight( elevation );
	// no beam under overcast's closed deck: its share of the sun's Rayleigh light goes, the parked
	// fog's kept (exact at rest and on a front between the fog and the overcast sky: see the header)
	out.sunRayleigh *= 1 - out.overcastDeck * ( 1 - fogWeather );
	return Object.assign( target, out );

}
