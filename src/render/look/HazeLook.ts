import { tuneObject, type TuneOptions } from '../debug/Tune.ts';
import { markSkyRetune } from '../sky/SkyRetune.ts';
import { MIX_SKIES, WEATHER_PRESETS } from '../sky/Weather.ts';
import { HAZE_KEY_NAMES, type FogFields, type FullKey, type HazeKey, type HazeKeyName, type HazeLook, type HazeRows, type HazeScale } from './Look.ts';

// The haze's tables built from a look's table: its keys by the hour and by the weather.
//
// The haze's tables built from a look's table (look/Look.ts `HazeLook`): its keys by the hour, its
// rows of them by the sun's elevation and what the weather adds per sky of its blend
// (`createHazeTables`). weather/HazeKeys.ts `hazeKeyAt` eases them by the hour and the weather,
// taking them as its first argument; the haze (weather/VolumetricHaze.ts) takes the result. Like
// the exposure (look/ExposureLook.ts), they are taken as an argument where the render pipeline's
// sky stage creates them (RenderStages.ts `createSky`, for its `prepareHaze`), so two made from two
// tables hold their own numbers (tools/render/verify-look-params.mjs) and this module holds none at
// module level: only the blank key, which no look changes. The aerosol layer's top under a deck is
// the sky's (look/SkyLook.ts `cloudBase`).
// - The tables register their tunables (debug/Tune.ts) as they are created, under the names they
//   always had (`haze.key.*`, `haze.fog.*`, `sky.weather.*@<cloudy|overcast>`). The tune table is
//   one per process: on a debug page a second set made from another table reads the first one's
//   tunables.
//
// The haze's keys: what the air does by the hour and the weather, set as a few quantities one sees
// (how far the light carries, the colour it loses, how bright and how coloured the air shows) at
// a few heights of the sun, the morning's and the evening's apart, and eased between them. A key
// is a function of the sun's elevation, whether it is rising, the weather's blend of the four skies
// (Weather.ts `skyMix`), its wind and its rain, never of the path to them: a paused frame and a
// scene jump hold one value, and a change of hour or weather writes the haze's uniforms only.
//
// Each key holds (look/Look.ts `HazeKey`):
//   - amount: the weather's haze over its preset's;
//   - the medium's shape (VolumetricHaze's header): `top`, the ground layer's top over the sea (m),
//     `width`, the width of its step (m), `aerosolShare`, the deep aerosol layer's density at sea
//     level over the ground layer's, and `aerosolHeight`, its scale height (m); `skyDepth`, a thin
//     haze that fills the air's whole height, which only the sky dome takes (green's optical depth
//     per kilometre of a sky ray's length);
//   - sigma: the medium's spectral extinction, each band's weight on the density;
//   - desaturate: how far each of the haze's two shares of light turns toward its luminance;
//   - mist: a morning mist in the sea fog's units, which thickens the haze's ground layer;
//   - the fog layer (VolumetricHaze's header, The fog layer): `fogLayer` its share, `fogTop` its
//     top over the sea, `fogWidth` the width of its edge, `visibility` the Koschmieder visibility
//     it gives at the sea (km), `neutral` how far its extinction turns grey and `scatter` its
//     droplets' multiple scattering (Wrenninge's a);
//   - sunRayleigh: the sun's share in the haze's Rayleigh light, beside the open sky's;
//   - rayleighWeight, mieWeight: the two shares' weights;
//   - groundLight: the share of the air's light laid over the land (VolumetricHaze.ts `apply`).
// A look gives a key for each of ten hours (look/Look.ts HAZE_KEY_NAMES) and rows of them by the
// sun's elevation; a look whose rising rows hold the night key holds it in its setting rows too,
// with a row after it (weather/HazeKeys.ts takes the dawn's `groundLight` from them). Four keys hold a medium's shape of their own (the day's, the golden hour's, the
// night's and the mist's: `shaped`); the others take the day's. Three hold a fog layer of their own
// (the day's, the golden hour's and the mist's); the morning takes the mist's, the others none
// (`fogOf`). The weather adds its own per sky of its blend (`keep`, `gain`, `skyScale`, `decked`,
// `sunRayleigh`).

export type { FullKey, HazeKey };

/** A key with every field 0, for a reader that fills all of them (weather/HazeKeys.ts). */
export const blankKey = (): FullKey => ( {
	deck: 0, overcastDeck: 0, amount: 0, sigmaRed: 0, sigmaGreen: 0, sigmaBlue: 0, desaturate: 0, gain: 0, mist: 0, sunRayleigh: 0, rayleighWeight: 0, mieWeight: 0,
	top: 0, width: 0, aerosolShare: 0, aerosolHeight: 0, aerosolTop: 0, skyDepth: 0, groundLight: 0,
	fogLayer: 0, neutral: 0, scatter: 0, fogTop: 0, fogWidth: 0, visibility: 0, droplet: 0, fogCoverHigh: 0,
} );

/** Rows by the sun's elevation (degrees, ascending), each holding its key, eased between
 * neighbours, the end rows held. */
export type HazeKeyRows = ReadonlyArray<readonly [ number, FullKey ]>;

/** The haze's tables a look's table makes (`createHazeTables`). */
export interface HazeTables {
	/** The look's keys, each a copy these tables own (its tunables are registered on it). */
	readonly keys: Readonly<Record<HazeKeyName, FullKey>>;
	/** The rising sun's rows, the setting sun's, and the setting sun's under cloud, which a clouded
	 * share draws the key to. */
	readonly rising: HazeKeyRows;
	readonly setting: HazeKeyRows;
	readonly settingClouded: HazeKeyRows;
	/** The keys with a shape of their own; the others take the day's. */
	readonly shaped: ReadonlySet<FullKey>;
	/** The key whose fog layer `k` takes: its own for the keys with one (the day's, the golden
	 * hour's, the mist's), the mist's for the morning's, none (the night's) for the others. */
	readonly fogOf: ( k: FullKey ) => FullKey;
	/** The fog layer's fields where a key holds none: the hours without a fog of their own (the
	 * night and the twilights) hold this. */
	readonly noFog: FogFields;
	/** The sea fog: a deep layer with a soft top, held at every hour while its switch is on. */
	readonly seaFog: HazeLook[ 'seaFog' ];
	/** Per sky of the blend (Weather.ts MIX_SKIES: clear, fair, cloudy, overcast): the share of the
	 * light's colour kept over a clear day's and the gain over a clear day's. */
	readonly keep: readonly number[];
	readonly gain: readonly number[];
	/** Per sky of the blend, whether a deck ends the aerosol layer at its base (look/SkyLook.ts
	 * `cloudBase`). */
	readonly decked: readonly boolean[];
	/** Per sky of the blend, the scale of each band's sigma, of the aerosol layer's share and of its
	 * scale height, of the ground layer's top and width, and the weather's haze (its preset's until
	 * tuned). */
	readonly skyScale: readonly ( HazeScale & { haze: number } )[];
	/** Per sky of the blend, the sun's share in the haze's Rayleigh light. */
	readonly sunRayleigh: readonly number[];
}

// the keys with a shape are tunable on debug pages (debug/Tune.ts):
// `haze.key.<amount|sigmaRed|sigmaBlue|desaturate|rayleighWeight|mieWeight|top|width|
// aerosolShare|aerosolHeight|skyDepth>@<day|golden|night|mist>` (green is sigma's unit), the
// morning's, the twilight's before dawn (deep and late) and after sunset (the afterglow and the
// deep twilight) their six up to `mieWeight` (`@morning`, `@earlyDawn`, `@beforeDawn`,
// `@afterSunset`, `@afterSunsetDeep`, under cloud `@afterSunsetClouded`); every key's
// `haze.key.groundLight@<key>`
const SHAPED_NAMES: readonly HazeKeyName[] = [ 'day', 'golden', 'night', 'mist' ];
const TUNED = [ 'amount', 'sigmaRed', 'sigmaBlue', 'desaturate', 'rayleighWeight', 'mieWeight', 'top', 'width', 'aerosolShare', 'aerosolHeight', 'skyDepth' ] as const;
const TUNE_RANGE: Record<typeof TUNED[ number ] | 'groundLight', TuneOptions> = {
	amount: { min: 0.1, max: 10, scale: 'log', onChange: markSkyRetune }, sigmaRed: { min: 0.05, max: 1.5, scale: 'log', onChange: markSkyRetune },
	sigmaBlue: { min: 0.3, max: 3, scale: 'log', onChange: markSkyRetune }, desaturate: { min: 0, max: 1, onChange: markSkyRetune },
	rayleighWeight: { min: 0.05, max: 40, scale: 'log', onChange: markSkyRetune }, mieWeight: { min: 0.05, max: 50, scale: 'log', onChange: markSkyRetune },
	top: { min: 0, max: 60, onChange: markSkyRetune }, width: { min: 5, max: 300, scale: 'log', onChange: markSkyRetune },
	aerosolShare: { min: 0.003, max: 3, scale: 'log', onChange: markSkyRetune }, aerosolHeight: { min: 300, max: 8000, scale: 'log', onChange: markSkyRetune },
	skyDepth: { min: 0, max: 0.5, onChange: markSkyRetune },
	// (the light over the land is the composite's alone: no bake or sky reads it)
	groundLight: { min: 0, max: 1 },
};

/** The haze's tables made from a look's table: registers their tunables (see the header). */
export function createHazeTables( look: HazeLook ): HazeTables {

	// the look's keys, each a copy these tables own (its tunables are registered on it)
	const keys = Object.fromEntries( HAZE_KEY_NAMES.map( ( name ) => [ name, { ...look.keys[ name ] } ] ) ) as Record<HazeKeyName, FullKey>;
	const { day, night, mist, morning } = keys;
	for ( const name of HAZE_KEY_NAMES ) {

		const fields = SHAPED_NAMES.includes( name ) ? TUNED : TUNED.slice( 0, 6 );
		tuneObject( keys[ name ], Object.fromEntries( fields.map( ( f ) => [ f, `haze.key.${ f }@${ name }` ] ) ), TUNE_RANGE );
		tuneObject( keys[ name ], { groundLight: `haze.key.groundLight@${ name }` }, TUNE_RANGE );

	}
	tuneObject( mist, { mist: 'haze.key.mist@mist' }, { mist: { min: 0, max: 2, onChange: markSkyRetune } } );
	// the fog layer's parameters on the keyframes that hold one:
	// `haze.fog.<top|width|visibility|neutral|scatter>@<day|golden|mist>`, the visibility's range the key's
	for ( const [ name, low, high ] of [ [ 'day', 0.2, 1.5 ], [ 'golden', 0.3, 2.5 ], [ 'mist', 0.2, 2 ] ] as const ) {

		const range: Record<'fogTop' | 'fogWidth' | 'visibility' | 'neutral' | 'scatter', TuneOptions> = {
			fogTop: { min: 12, max: 40, onChange: markSkyRetune }, fogWidth: { min: 2, max: 15, onChange: markSkyRetune },
			visibility: { min: low, max: high, scale: 'log', onChange: markSkyRetune },
			neutral: { min: 0, max: 1, onChange: markSkyRetune }, scatter: { min: 0, max: 0.4, onChange: markSkyRetune },
		};
		tuneObject( keys[ name ], { fogTop: `haze.fog.top@${ name }`, fogWidth: `haze.fog.width@${ name }`, visibility: `haze.fog.visibility@${ name }`,
			neutral: `haze.fog.neutral@${ name }`, scatter: `haze.fog.scatter@${ name }` }, range );

	}

	const rows = ( r: HazeRows ): HazeKeyRows => r.map( ( [ at, name ] ) => [ at, keys[ name ] ] as const );
	const fogged: ReadonlySet<FullKey> = new Set( [ day, keys.golden, mist ] );
	// the sea fog, tunable as `haze.fog.<field>@sea`
	const seaFog = { ...look.seaFog };
	tuneObject( seaFog, { fogTop: 'haze.fog.top@sea', fogWidth: 'haze.fog.width@sea', visibility: 'haze.fog.visibility@sea',
		neutral: 'haze.fog.neutral@sea', scatter: 'haze.fog.scatter@sea', coverHigh: 'haze.fog.coverHigh@sea' }, {
		fogTop: { min: 0, max: 150, onChange: markSkyRetune }, fogWidth: { min: 2, max: 100, onChange: markSkyRetune },
		coverHigh: { min: - 1, max: 2.5, onChange: markSkyRetune },
		visibility: { min: 0.08, max: 2, scale: 'log', onChange: markSkyRetune },
		neutral: { min: 0, max: 1, onChange: markSkyRetune }, scatter: { min: 0, max: 0.6, onChange: markSkyRetune },
	} );
	// the weather's haze per sky of the blend is its preset's until tuned
	const skyScale = MIX_SKIES.map( ( name, s ) => {

		const { haze } = WEATHER_PRESETS[ name ];
		return { ...look.scales[ s ], haze };

	} );
	for ( const [ s, name ] of [ [ 2, 'cloudy' ], [ 3, 'overcast' ] ] as const ) {

		tuneObject( skyScale[ s ], { haze: `sky.weather.haze@${ name }` }, { haze: { min: 2e-4, max: 0.02, scale: 'log', onChange: markSkyRetune } } );
		tuneObject( skyScale[ s ], { sigmaRed: `sky.weather.sigmaRed@${ name }`, sigmaBlue: `sky.weather.sigmaBlue@${ name }`, aerosolScale: `sky.weather.aerosolScale@${ name }`, aerosolHeight: `sky.weather.aerosolHeight@${ name }` },
			{ sigmaRed: { min: 0.2, max: 5, scale: 'log', onChange: markSkyRetune }, sigmaBlue: { min: 0.2, max: 5, scale: 'log', onChange: markSkyRetune },
				aerosolScale: { min: 0.1, max: 30, scale: 'log', onChange: markSkyRetune }, aerosolHeight: { min: 0.2, max: 10, scale: 'log', onChange: markSkyRetune } } );
		tuneObject( skyScale[ s ], { topScale: `sky.weather.topScale@${ name }`, widthScale: `sky.weather.widthScale@${ name }` },
			{ topScale: { min: 0.1, max: 5, scale: 'log', onChange: markSkyRetune }, widthScale: { min: 0.2, max: 5, scale: 'log', onChange: markSkyRetune } } );

	}
	return {
		keys,
		rising: rows( look.rising ),
		setting: rows( look.setting ),
		settingClouded: rows( look.settingClouded ),
		shaped: new Set( SHAPED_NAMES.map( ( name ) => keys[ name ] ) ),
		fogOf: ( k ) => fogged.has( k ) ? k : k === morning ? mist : night,
		noFog: look.noFog,
		seaFog,
		keep: look.keep,
		gain: look.gain,
		decked: look.decked,
		skyScale,
		sunRayleigh: look.sunRayleigh,
	};

}
