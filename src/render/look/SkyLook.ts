import * as THREE from 'three/webgpu';
import { SHELL_BOTTOM, SHELL_THICKNESS } from '../sky/CloudModel.ts';
import { MIX_SKIES, WEATHER_PRESETS } from '../sky/Weather.ts';
import type { DayKey, DayKeys } from '../post/DayKeys.ts';
import { tune, tuneKeys, tuneObject } from '../debug/Tune.ts';
import { markSkyRetune } from '../sky/SkyRetune.ts';
import type { LandSkyRow, Rgb, SkyLook, Span, StreakKeys } from './Look.ts';

// The sky's tables built from a look's table, with the CPU code that reads only them.
//
// The sky's tables built from a look's table (look/Look.ts `SkyLook`), with the CPU code that reads
// only them (`createSkyTables`): the light the sky lends the land after sunset and under cloud
// (`landSky`, `cloudLightTint`), the sky's gains on the sun's light (`skyLight`, `clearSky`), the
// clouds' rows per sky of the blend (`cloudSkies`, `shadowDensity`), the stores' night ranges
// (`storeNight`), the low key (`lowKey`), the distant sea's knee (`seaKnee`), the key's hue
// (`keyHue`), the cirrus colours (`streakDirect`, `streakAmbient`), the moon's key
// (`moonKey`, `moonHue`) and the key under cloud (`keyThroughCloud`). Like the exposure
// (look/ExposureLook.ts), they are taken as an argument where the render pipeline's sky stage
// creates them (RenderStages.ts `createSky`, which hands them to the sky: sky/Sky.ts `look`), not
// as a module loads, so two made from two tables hold their own numbers
// (tools/render/verify-look-params.mjs) and this module holds none at module level: only the
// functions that read no look (`luma`, `luminance709`, `byElevation`). sky/SkyTrack.ts reads them
// each frame, sky/SkyBake.ts the sea's knee at each bake and weather/HazeKeys.ts the clouds' base
// each frame; every function here that reads them takes them as its first argument. A name in
// backticks with no file is a field of the tables or sky/Sky.ts's. The A/B switches that turn a
// table off (`flags=clearSkyFill`, `lowSun`, `clearSky`, `skyGain`, `lowKey`) stay with their readers
// in sky/SkyTrack.ts.
// - The tables register their tunables (debug/Tune.ts) as they are created, under the names they
//   always had (`sky.light.*`, `sky.clear.*`, `sky.weather.shadowDensity@*`, `sky.clouds.*`,
//   `sky.sea.knee`, `sky.key.*`, `sky.streaks.*`, `lighting.moon.red|blue`,
//   `lighting.keyThroughCloud`). The tune table is one per process: on a debug page a second set made
//   from another table reads the first one's tunables.

type Triple = readonly [ number, number, number ];
type Tint = readonly [ number, number, number, number ];

// a colour at unit luminance (Rec. 709)
const unit = ( r: number, g: number, b: number ) => {

	const l = r * 0.2126 + g * 0.7152 + b * 0.0722;
	return [ r / l, g / l, b / l ] as const;

};
// a colour at unit luminance with its amount
const unitTint = ( [ r, g, b, amount ]: Tint ): Tint => [ ...unit( r, g, b ), amount ] as const;

/** A row of the clouds' parameters per sky of the blend (`cloudSkies`). */
export type CloudSky = { base: number; top: number; type: number; spread: number; farFade: number };

/** The sky's tables, made from a look's (`createSkyTables`). */
export interface SkyTables {
	// By day a clouded sky lends the land its own light: from 12 to 40 degrees, as the sky's high
	// key climbs, the light lent to the land is drawn toward this tint (unit luminance) by its
	// amount (Weather.ts `cloudedHighSun`). The meter's level is the same, the tint keeping the
	// luminance.
	readonly cloudLightTint: Tint;
	// The light the sky lends the land takes a gain per sun elevation and a cast: the environment's
	// source panorama, which lights the terrain and plants, in its bake (its haze without the fog
	// layer: `fogEnvironment`), and the islanders' sky fill (`landIrradianceNode`, the irradiance
	// probe graded the same way). The seen sky, the panorama the sea reflects and the irradiance probe
	// itself, which lights the sea's body, keep the atmosphere's own. Each row is an elevation
	// (degrees, descending): the colour the light is drawn toward (linear, at the light's own
	// luminance), how far, and the gain; rows interpolate linearly and the end rows hold beyond them.
	// The setting sun's rows, the setting sun's under cloud and the rising sun's (`landSky`).
	readonly landSky: SkyLook[ 'landSky' ];
	// The clear-sky fill light, off by default (`flags=clearSkyFill:<k>` or `true` brings it back for A/Bs): the
	// sky light lent to the land (as `landSky`'s) takes a gain and a tint. The gain's extra share
	// per sun elevation (degrees, descending; linear between rows, the end rows held beyond).
	readonly clearSkyFill: readonly { at: number; share: number }[];
	// the sky light's tint (unit luminance) and how far it is drawn toward it at a share of 1
	readonly clearSkyFillTint: Tint;
	// The low sun's shadow tint, off by default with the clear-sky fill light (`flags=lowSun:<k>` or `true` for
	// A/Bs): from `full` degrees down to the horizon the sky light lent to the land is drawn toward
	// `lowSunTint` by up to `amount`, fading out above `top`.
	readonly lowSun: SkyLook[ 'lowSun' ];
	/** How far a deck draws the key toward white at the same luminance (sky/SkyTrack.ts `trackSky`). */
	readonly keyDeckWhite: number;
	readonly lowSunTint: Triple;
	// The sky's light against the sun's: one gain (an illuminance colour per channel, keyed by the
	// sun's elevation: DayKeys.ts) on the light the sun lends the sky, the atmosphere's in-scattered
	// light (Atmosphere.ts `skyRadiance`). Everything that reads the sky takes it: the seen sky, the
	// environment's bakes and the irradiance probe that light the island, the sea's reflection and the
	// clouds' and the haze's sky light, so the sky the eye sees stays the sky that lights. The clouds'
	// sunlight takes none. Under cloud the gain moves to the night key from 2 degrees to -6 under the
	// horizon; a clear sky's holds the day's keys through the twilight (DayCycle.ts `skyNight`). The
	// rising sun reads the `rising` keys, a clouded sky the `cloudy` ones by its clouded share
	// (Weather.ts `cloudedShare`).
	// Tunable on debug pages (debug/Tune.ts): `sky.light.<r|g|b>@<key name>`.
	readonly skyLight: DayKeys<Rgb>;
	// The clear day's sky against the key: under a clear or fair day's sun the sky's light takes a
	// gain of its own over `skyLight`, keyed by the sun's elevation, on every path `skyLight` reaches
	// (`skyGain`): the sky the eye sees, the sea's reflection and its body, the clouds' and the haze's
	// sky light, the land's and the exposure's sky meter. The gain leaves by the weather's clouded
	// share (Weather.ts `cloudedShare`); the open sky seen through a deck's gaps takes it back
	// (`gapSky`).
	// Tunable on debug pages: `sky.clear.gain@<key name>`.
	readonly clearSky: DayKeys<{ gain: number }>;
	// per sky of the blend, the share of the deck's optical depth its shadow is cast with (Weather.ts
	// `shadowDensity`, its preset's until tuned: the weather's, not the look's, made here beside the
	// clouds' rows), blended by the weather's shares of the four skies;
	// tunable on debug pages: `sky.weather.shadowDensity@<cloudy|overcast>`
	readonly shadowDensity: readonly { shadowDensity: number }[];
	// The clouds are one model, and a kind of sky is a row of its parameters: one row per sky of the
	// blend (in MIX_SKIES' order), blended each frame by the weather's shares of the four skies
	// (`trackClouds`), so a change of weather crosses them by those shares. The parked weathers count
	// as overcast in those shares (Weather.ts), so they take overcast's row. The coverage and the
	// density are the weather state's (Weather.ts WEATHER_PRESETS), interpolated with it.
	//
	// `base`, `top` (m): the shell the clouds stand in (Clouds.ts `setShell`); a look's row without
	// them stands in the cloud model's own shell (sky/CloudModel.ts).
	// `type`, `spread`: the cloud type (0 a stratus, 1 a cumulus: sky/CloudModel.ts STRATUS, CUMULUS) and how
	// far it varies from place to place, either way.
	// `farFade`: how far the clouds thin with distance (Clouds.ts `farFade`).
	// Tunable on debug pages: `sky.clouds.base|top|type|spread|farFade@<clear|fair|cloudy|overcast>`.
	readonly cloudSkies: readonly CloudSky[];
	// The range each store holds under the moon, per sky of the blend (MIX_SKIES' order), in stops over the
	// luminance of the moonlight on the clouds (`storeLight`): the brightest value and the first
	// percentile (the cloud dome's and the layer's over the texels at least a tenth covered). The rows
	// are estimates that centre the range, with several stops to spare on either side (NIGHT_TOP,
	// NIGHT_FLOOR). A weather between skies takes the rows by its shares of them, as its shell and
	// form do (`cloudSkies`).
	readonly storeRange: SkyLook[ 'storeRange' ];
	// The low sun's key: the key (the island's and the sea's sunlight) takes a gain by the sun's
	// elevation (degrees, descending; linear between rows, the last row held down to the horizon). The
	// shadows, lit by the sky, do not move; the disk and the sky's own sun are untouched.
	readonly lowKey: readonly { at: number; gain: number }[];
	// The distant sea's body (the environment's lower half under its reflection, `seaPanorama`): its
	// colour's share of the light that reaches the sea, the sky's irradiance with its clouds (the bake's
	// deck irradiance) and the sun's direct light on the level water, in luminance, up to a knee at
	// this share of the sun's illuminance (scene units, both at the weather's ambient gain), where it
	// is the colour times that illuminance. Tunable: `sky.sea.knee`.
	readonly seaKnee: () => number;
	// The key's hue (`keyHue`): the caps of the optical depth that sets it (`sky.key.hueDepth`, the
	// rising sun's `sky.key.dawnHueDepth`), the key's white (linear red/green and blue/green while the
	// sun is high) and the onsets of the air's change in each ratio (`sky.key.reddens`,
	// `sky.key.yellows`).
	readonly keyHue: { readonly depth: () => number; readonly dawnDepth: () => number; readonly white: readonly [ number, number ]; readonly reddens: () => number; readonly yellows: () => number };
	// The overall colour on each of the cirrus's two shares (Clouds.ts `keyDirect`, `keyAmbient`),
	// by the sun's elevation, the morning's apart (DayKeys.ts `rising`), the night's coming in with
	// the clouds' moonlight (`moonShare`). Tunable on debug pages (debug/Tune.ts):
	// `sky.streaks.<keyDirect|keyAmbient>.<r|g|b>@<afterglow|golden|low|predawn|sunrise|morning|high|night>`.
	readonly streakDirect: DayKeys<Rgb>;
	readonly streakAmbient: DayKeys<Rgb>;
	// The moon's key's colour as the look writes it (the island's moonlight takes the tuned hue over
	// it, `moonHue`), its level the moon's light above the air through the air toward the moon
	// (`moonAboveAir`), as the sun's key is the sun's through the air toward the sun; how bright the
	// night shows is the exposure's (post/Exposure.ts). The island's moonlight takes a hue of its own
	// (`moonHue`), and the clouds' direct moonlight the same; the moonlit sky, the clouds' sky light
	// and the haze take the moon's light above the air at their own hue (`moonSkyHue`). Its red and
	// blue over green as the look writes them (the quotients of its channels may differ by an ulp), so
	// a hue equal to them is exactly one.
	readonly moonKey: SkyLook[ 'moonKey' ];
	// the moon's key's Rec. 709 luminance, the level the hue keeps
	readonly moonKeyLuminance: number;
	// The moonlit sky's colour at unit luminance. The moonlit sky, the haze's moonlight and the
	// clouds' sky light keep it; their level is the sun's sky at the moon's height (`track`). The
	// sky's night key itself stays the sun's deep twilight key.
	readonly moonSkyHue: Triple;
	// The night's ambient's colour at unit luminance (`displayNightAmbient`).
	readonly nightAmbientHue: Triple;
	// The night's ambient in the clouds and the distant sea (`displayNightAmbient`), a radiance: a
	// uniform sky giving level ground this share of the moon's key on it (its light through the air at
	// normal incidence).
	readonly nightAmbientShare: number;
	// The island's moonlight's red and blue over green. They move the moonlit ground's colour without
	// its level (the hue keeps `moonKey`'s luminance) and without the moonlit sky's, clouds' and haze's,
	// which keep theirs. At `moonKey`'s own (its `red`, `blue`) the hue is exactly white.
	// The clouds take the same moonlight (`trackClouds`). Tunable: `lighting.moon.red|blue`.
	readonly moonHue: { readonly red: () => number; readonly blue: () => number };
	// The key under cloud, one rule for every sky. Where cloud stands along the key the land does not
	// lose the key whole: the land's key keeps mix( floor, 1, T ), T the clouds' transmittance along
	// the key (`keyDirectVisibility`), and the floor is the diffuse transmission of a layer of the
	// sky's shell and density (Clouds.ts `layerTransmission`) times this share, at most 1: how much of
	// what the layer passes down the land keeps as a light along the key, the one parameter for how deep
	// the clouds shade the land. The sea keeps the same floor (`seaKeyVisibility`; its glint only the
	// direct beam, `cloudDirectVisibility`), and under a closed sky the air does (`deckKeyFloor`, by
	// overcast's share). The floor rides on whichever light keys the scene, the
	// moon's by night as the sun's by day; under a closed sky the sun's part of it leaves over the
	// last 2.5 degrees before the horizon (Weather.ts `overcastFloorSun`).
	// Tunable: `lighting.keyThroughCloud`.
	readonly keyThroughCloud: () => number;
}

/** The sky's tables made from a look's table: registers their tunables (see the header). */
export function createSkyTables( look: SkyLook ): SkyTables {

	const skyLight = tuneKeys( 'sky.light', look.skyLight,
		{ r: { min: 0.001, max: 2, scale: 'log', onChange: markSkyRetune }, g: { min: 0.001, max: 2, scale: 'log', onChange: markSkyRetune }, b: { min: 0.001, max: 2, scale: 'log', onChange: markSkyRetune } } );
	const clearSky = tuneKeys( 'sky.clear', look.clearSky, { gain: { min: 0.25, max: 8, scale: 'log', onChange: markSkyRetune } } );
	const shadowDensity = MIX_SKIES.map( ( name ) => ( { shadowDensity: WEATHER_PRESETS[ name ].shadowDensity } ) );
	for ( const [ s, name ] of [ [ 2, 'cloudy' ], [ 3, 'overcast' ] ] as const ) {

		tuneObject( shadowDensity[ s ], { shadowDensity: `sky.weather.shadowDensity@${ name }` }, { shadowDensity: { min: 0.02, max: 1, scale: 'log', onChange: markSkyRetune } } );

	}
	const cloudSkies: CloudSky[] = look.cloudSkies.map( ( row ) => (
		{ base: row.base ?? SHELL_BOTTOM, top: row.top ?? SHELL_BOTTOM + SHELL_THICKNESS, type: row.type, spread: row.spread, farFade: row.farFade } ) );
	for ( const [ s, name ] of MIX_SKIES.entries() ) {

		tuneObject( cloudSkies[ s ], { base: `sky.clouds.base@${ name }`, top: `sky.clouds.top@${ name }`, type: `sky.clouds.type@${ name }`, spread: `sky.clouds.spread@${ name }`, farFade: `sky.clouds.farFade@${ name }` },
			{ base: { min: 300, max: 6000, onChange: markSkyRetune }, top: { min: 1500, max: 10000, onChange: markSkyRetune }, type: { min: 0, max: 1, onChange: markSkyRetune },
				spread: { min: 0, max: 1, onChange: markSkyRetune }, farFade: { min: 0, max: 1, onChange: markSkyRetune } } );

	}
	const seaKnee = tune( 'sky.sea.knee', look.seaKnee, { min: 0.05, max: 1, onChange: markSkyRetune } );
	const keyHue = {
		depth: tune( 'sky.key.hueDepth', look.keyHue.depth, { min: 1, max: 4, onChange: markSkyRetune } ),
		dawnDepth: tune( 'sky.key.dawnHueDepth', look.keyHue.dawnDepth, { min: 0.3, max: 4, onChange: markSkyRetune } ),
		white: look.keyHue.white,
		reddens: tune( 'sky.key.reddens', look.keyHue.reddens, { min: 0.3, max: 1.5, onChange: markSkyRetune } ),
		yellows: tune( 'sky.key.yellows', look.keyHue.yellows, { min: 0.2, max: 1.2, onChange: markSkyRetune } ),
	};
	const streakDirect = streakKeys( 'sky.streaks.keyDirect', look.streakDirect );
	const streakAmbient = streakKeys( 'sky.streaks.keyAmbient', look.streakAmbient );
	const moonHue = {
		red: tune( 'lighting.moon.red', look.moonHue.red, { min: 0.3, max: 1.2 } ),
		blue: tune( 'lighting.moon.blue', look.moonHue.blue, { min: 0.8, max: 4 } ),
	};
	const keyThroughCloud = tune( 'lighting.keyThroughCloud', look.keyThroughCloud, { min: 0, max: 4, onChange: markSkyRetune } );
	return {
		cloudLightTint: unitTint( look.cloudLightTint ),
		landSky: look.landSky,
		clearSkyFill: look.clearSkyFill,
		clearSkyFillTint: unitTint( look.clearSkyFillTint ),
		lowSun: look.lowSun,
		keyDeckWhite: look.keyDeckWhite,
		lowSunTint: unit( ...look.lowSunTint ),
		skyLight, clearSky, shadowDensity, cloudSkies,
		storeRange: look.storeRange,
		lowKey: look.lowKey,
		seaKnee, keyHue, streakDirect, streakAmbient,
		moonKey: look.moonKey,
		moonKeyLuminance: luminance709( look.moonKey.colour[ 0 ], look.moonKey.colour[ 1 ], look.moonKey.colour[ 2 ] ),
		moonSkyHue: look.moonSkyHue,
		nightAmbientHue: look.nightAmbientHue,
		nightAmbientShare: look.nightAmbientShare,
		moonHue, keyThroughCloud,
	};

}

// the cirrus keyframes (`streakDirect`, `streakAmbient`) from a look's, their tunables under `prefix`
const streakKeys = ( prefix: string, keys: StreakKeys ): DayKeys<Rgb> => {

	const range = { min: 0.01, max: 4, scale: 'log' as const, onChange: markSkyRetune };
	const high: DayKey<Rgb> = { at: 30, name: 'high', values: keys.high };
	return tuneKeys( prefix, {
		day: [ { at: - 3, name: 'afterglow', values: keys.afterglow }, { at: 2, name: 'golden', values: keys.golden }, { at: 8, name: 'low', values: keys.low }, high ],
		rising: [ { at: - 3, name: 'predawn', values: keys.predawn }, { at: 2, name: 'sunrise', values: keys.sunrise }, { at: 8, name: 'morning', values: keys.morning }, high ],
		night: keys.night,
	}, { r: range, g: range, b: range } );

};

// The tables' `landSky` at `elevation`, on the rising or the setting sun's side: the tint
// (normalised to unit luminance) and its amount into `tint`; returns the gain. A clouded share
// `cloud` (Weather.ts `cloudedShare`) draws the setting rows to the clouded evening's. The rising
// rows' tints hold under any sky, but their gains were set on a clear morning's exposure
// (look/ExposureLook.ts, `dawn`), so the clouded share takes the clouded evening's gain back, as
// the display's night weight does (DayCycle.ts `displayNight`).
export function landSky( tables: SkyTables, elevation: number, rising: boolean, tint: THREE.Vector4, cloud: number ): number {

	const at = ( row: LandSkyRow ) => {

		const [ r, g, b ] = row.tint ?? [ 1, 1, 1 ], l = r * 0.2126 + g * 0.7152 + b * 0.0722;
		return [ r / l, g / l, b / l, row.amount ?? 0, row.gain ?? 1 ];

	};
	const along = ( rows: readonly LandSkyRow[] ) => {

		let v = at( rows[ rows.length - 1 ] );
		if ( elevation >= rows[ 0 ].at ) v = at( rows[ 0 ] );
		else for ( let i = 1; i < rows.length; i ++ ) if ( elevation >= rows[ i ].at ) {

			const t = ( elevation - rows[ i ].at ) / ( rows[ i - 1 ].at - rows[ i ].at );
			const a = at( rows[ i ] ), b = at( rows[ i - 1 ] );
			v = a.map( ( x, c ) => x + ( b[ c ] - x ) * t );
			break;

		}
		return v;

	};
	let v = along( tables.landSky.setting );
	if ( cloud > 0 ) {

		const c = along( tables.landSky.clouded );
		v = v.map( ( x, i ) => x + ( c[ i ] - x ) * cloud );

	}
	if ( rising ) {

		const r = along( tables.landSky.rising );
		if ( cloud > 0 ) r[ 4 ] += ( v[ 4 ] - r[ 4 ] ) * cloud;
		v = r;

	}
	tint.set( v[ 0 ], v[ 1 ], v[ 2 ], v[ 3 ] );
	return v[ 4 ];

}

/** The cloud shell's base (m) under the sky `s` (Weather.ts MIX_SKIES), as tuned: the haze's
 * aerosol ends at a layer's base (weather/HazeKeys.ts `aerosolTop`). */
export const cloudBase = ( tables: SkyTables, s: number ): number => tables.cloudSkies[ s ].base;

// What a night's store must keep, stored (stops): its brightest under 2^13 (two stops under
// STORE_CEILING), and its first percentile, were the night pressed 12 stops darker (the debug
// press, PreExposure.ts `storePress`), over 2^-13, a stop over half float's 2^-14.
const NIGHT_TOP = 13, NIGHT_FLOOR = - 13 + 12;

/** Where the store `group`'s range sits best under the moon (stops over the light, PreExposure.ts
 * `storeStops`): its estimated brightest and first percentile (the tables' `storeRange`, by the
 * weather's shares of the four skies) centred between NIGHT_FLOOR and NIGHT_TOP, so both keep the
 * same margin. */
export function storeNight( tables: SkyTables, group: keyof SkyLook[ 'storeRange' ], skyMix: readonly number[] | null ): number {

	const rows: readonly Span[] = tables.storeRange[ group ];
	let top = 0, floor = 0;
	for ( let s = 0; s < rows.length; s ++ ) {

		const share = skyMix ? skyMix[ s ] ?? 0 : s === 0 ? 1 : 0;
		top += share * rows[ s ][ 0 ];
		floor += share * rows[ s ][ 1 ];

	}
	return ( NIGHT_TOP + NIGHT_FLOOR - top - floor ) / 2;

}

// a gain by the sun's elevation from rows of descending elevation: linear between rows, the end
// rows held beyond
export function byElevation<K extends string>( rows: readonly ( { at: number } & Record<K, number> )[], elevation: number, key: K ): number {

	if ( elevation >= rows[ 0 ].at ) return rows[ 0 ][ key ];
	for ( let i = 1; i < rows.length; i ++ ) if ( elevation >= rows[ i ].at ) {

		const t = ( elevation - rows[ i ].at ) / ( rows[ i - 1 ].at - rows[ i ].at );
		return rows[ i ][ key ] + ( rows[ i - 1 ][ key ] - rows[ i ][ key ] ) * t;

	}
	return rows[ rows.length - 1 ][ key ];

}

export const luma = ( c: THREE.Color ) => c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;

// The key's hue. The optical depth that sets it is capped softly (the tables' `keyHue.depth`, the
// rising sun's `dawnDepth`), while its luminance keeps the full extinction, so a low sun still
// dims as it sets. Above the cap the hue is chosen rather than the air's own course: a white
// while the sun is high (`keyHue.white`: red/green, blue/green), each of the two ratios taking the
// air's change of it in part, by the air's depth in green (`reddens` for red/green, `yellows` for
// blue/green: about none far under the onset, two thirds at it, all of it past twice it). The sky
// and clouds keep their own colours, and the key's luminance is the sun tint's.
// The key's colour per unit of the sun's illuminance, into `transmittance`: its hue the white with
// the share of the air's change in each ratio its depth gives (the depth capped), its luminance
// the air's full extinction of `tint`.
export function keyHue( tables: SkyTables, transmittance: THREE.Color, tint: THREE.Color, rising: boolean ): THREE.Color {

	const { depth: sunHueDepth, dawnDepth: dawnHueDepth, white, reddens, yellows } = tables.keyHue;
	const { r, g, b } = transmittance;
	if ( g <= 0 ) return transmittance.setRGB( 0, 0, 0 );
	const depth = - Math.log( g );
	const scale = 1 / Math.pow( 1 + Math.pow( depth / ( rising ? dawnHueDepth() : sunHueDepth() ), 6 ), 1 / 6 );
	const share = ( onset: number ) => scale * ( 1 - Math.exp( - Math.pow( depth / onset, 2.5 ) ) );
	const before = luma( transmittance.clone().multiply( tint ) );
	transmittance.setRGB( white[ 0 ] * Math.pow( r / g, share( reddens() ) ), 1, white[ 1 ] * Math.pow( b / g, share( yellows() ) ) );
	return transmittance.multiplyScalar( before / luma( transmittance ) );

}

export const luminance709 = ( r: number, g: number, b: number ) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** The factor per channel that turns the tables' `moonKey` into the tuned hue (`moonHue`) at the
 * moon key's luminance. */
export function moonHue( tables: SkyTables, target: THREE.Color ): THREE.Color {

	const { colour, red, blue } = tables.moonKey;
	const r = tables.moonHue.red() / red, b = tables.moonHue.blue() / blue;
	const level = tables.moonKeyLuminance / luminance709( colour[ 0 ] * r, colour[ 1 ], colour[ 2 ] * b );
	return target.setRGB( r * level, level, b * level );

}
