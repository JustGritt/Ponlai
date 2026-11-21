import * as THREE from 'three/webgpu';
import { keyed, type DayKey, type DayKeys } from '../post/DayKeys.ts';
import { tune, tuneKeys } from '../debug/Tune.ts';
import { cloudedShare } from '../sky/Weather.ts';
import type { DayLook, ExposureKey, ExposureLook, SharingKey } from './Look.ts';

// The exposure built from a look's tables: its keys by the sun's elevation and the weather.
//
// The exposure built from a look's tables (look/Look.ts `ExposureLook`, and the exposure by the
// sun's elevation in `DayLook`): the exposure takes its tables as an argument where the render
// pipeline's sky stage creates it (RenderStages.ts `createSky`), so two exposures made from two
// tables hold their own numbers (tools/render/verify-look-params.mjs) and this module holds none at
// module level. What it makes (`Exposure`): the exposure's keys by the sun's elevation, the rising
// sun, the clouded day and the night, overcast's keys on top of them, and `keysAt`, which reads
// them for a frame; post/Exposure.ts turns the keys into the target and the adaptation (its header
// says what each key does), and post/Pipeline.ts `updateDisplay` reads the keys once a frame.
// Beside them, the base exposure by the sun's elevation (`at`, `day`, `night`) and the night sky's
// own exposure (`nightSky`, `nightGain`), which sky/DayCycle.ts and sky/SkyTrack.ts read, and the
// night's metered lift (`nightMeterLift`), which scales the island's lamps. The sky (sky/Sky.ts
// `exposure`) carries it to them and to the pipeline.
// - It registers its tunables (debug/Tune.ts) as it is created, under the names it always had
//   (`post.exposure.<value>@<key>`, `post.exposure.overcastKey`, `overcastKeyLow`,
//   `overcastNight`). The tune table is one per process: on a debug page a second exposure made
//   from another table reads the first one's tunables under the same names.

export type { ExposureKey };

/** The exposure a look's tables make (`createExposure`). */
export interface Exposure {
	/** The exposure's keys (DayKeys.ts): by the sun's elevation, the rising sun's, a clouded day's
	 * and the night's. Tunable on debug pages (debug/Tune.ts): `post.exposure.<value>@<key>`. */
	readonly keys: DayKeys<ExposureKey>;
	/** The exposure's keys for a frame into `out` (`keysAt` below says how). */
	keysAt( elevation: number, night: number, skyMix: readonly number[], out: ExposureKey, rising?: boolean ): ExposureKey;
	/** The base exposure at the sun's elevation (degrees): the look's [ elevation, exposure ] rows,
	 * interpolated in log exposure and held beyond the ends. */
	at( elevation: number ): number;
	/** The base exposure by day (the highest elevation's) and by night (the lowest's). */
	readonly day: number;
	readonly night: number;
	/** The night sky's own light (stars, the moon and its in-scatter, clouds) keeps the displayed
	 * brightness it has at `nightSky`, with the night's own gain over it (`nightGain`), whatever the
	 * frame's exposure (see Sky.track). */
	readonly nightSky: number;
	readonly nightGain: number;
	/** The stops a clear night's metered exposure lies above the twilight's exposure. The night glow
	 * (what glows by night) and every local light are scaled by 2 to minus this, so a lamp keeps its display at night
	 * whatever the hour. */
	readonly nightMeterLift: number;
}

/** The look's tables the exposure is made from: the exposure's, and the day's exposure by the
 * sun's elevation with the night's (the day's windows are made apart: look/DayLook.ts `createDayWindows`). */
export interface ExposureTables {
	exposure: ExposureLook;
	day: Pick<DayLook, 'exposure' | 'nightMeterLift' | 'nightExposureGain' | 'nightSkyExposure'>;
}

// a scratch key for overcast's held keys (`keysAt`)
const _held = { key: 0, highlight: 0, sky: 0, adaptUp: 0, adaptDown: 0 };

/** The exposure made from `look`'s tables (see the header); registers its tunables. */
export function createExposure( look: ExposureTables ): Exposure {

	const LOOK = look.exposure;
	// A look's key may share a value with another key (look/Look.ts `SharingKey`): that value
	// becomes a getter of the named key's in the table below, so the two hold one number and one
	// tunable (debug/Tune.ts `tuneKeys` leaves a getter as it is). A key no value of which is shared
	// passes through as the look gave it, so keys the table's sides share stay one object.
	const named = ( name: string ): DayKey<ExposureKey> => {

		const lists = [ keys.day, keys.rising ?? [], Array.isArray( keys.cloudy ) ? keys.cloudy as readonly DayKey<ExposureKey>[] : [] ];
		for ( const list of lists ) for ( const key of list ) if ( key.name === name ) return key;
		throw new Error( `look/ExposureLook.ts: a key shares a value of '${ name }', which no key is named` );

	};
	const sharing = new Map<SharingKey<ExposureKey>, DayKey<ExposureKey>>();
	const shared = ( list: readonly SharingKey<ExposureKey>[] ): DayKey<ExposureKey>[] => list.map( ( key ) => {

		const { shares } = key;
		if ( ! shares ) return key;
		let out = sharing.get( key );
		if ( ! out ) {

			const values = {} as ExposureKey;
			for ( const k of Object.keys( key.values ) ) {

				const from = shares[ k ];
				if ( from === undefined ) values[ k ] = key.values[ k ];
				else Object.defineProperty( values, k, { enumerable: true, get: () => named( from ).values[ k ] } );

			}
			out = { at: key.at, name: key.name, values };
			sharing.set( key, out );

		}
		return out;

	} );

	const keys: DayKeys<ExposureKey> = tuneKeys( 'post.exposure', {
		day: shared( LOOK.keys.day ),
		rising: shared( LOOK.keys.rising ),
		cloudy: shared( LOOK.keys.cloudy ),
		night: LOOK.keys.night,
	}, { key: { min: - 6, max: 3 }, highlight: { min: - 3, max: 4 }, sky: { min: 0, max: 1 }, adaptUp: { min: 0.02, max: 4, scale: 'log' }, adaptDown: { min: 0.02, max: 4, scale: 'log' } } );

	// Overcast's key on top of the clouded one toward a high sun, stops, by overcast's share of the
	// sky (Weather.ts `skyMix`, a parked sky counting as overcast), whole by day and handing over to
	// the night by the exposure's own night weight, as the keys do. It enters here alone, never in
	// the keys' own blend. Toward a low sun it passes to overcastKeyLow (overcastKeySpan). Tunable:
	// `post.exposure.overcastKey`.
	const overcastKey = tune( 'post.exposure.overcastKey', LOOK.overcastKey, { min: - 4, max: 1 } );
	// Overcast's key toward a low sun, stops, in place of overcastKey. Tunable:
	// `post.exposure.overcastKeyLow`.
	const overcastKeyLow = tune( 'post.exposure.overcastKeyLow', LOOK.overcastKeyLow, { min: - 4, max: 1 } );
	// Where overcast's key passes from its low sun's value (overcastKeyLow, at and under the first)
	// to its high sun's (overcastKey, at and over the second), degrees: a smoothstep.
	const overcastKeySpan = LOOK.overcastKeySpan;
	// The share of a clear sky's light the overcast deck lets through onto the land by night: in
	// scene units an overcast night lies log2( 1 / share ) stops under a clear one.
	const deckLandThrough = LOOK.deckLandThrough;
	// Overcast night's share f of that darkness the display keeps: the night key goes down by f
	// times it, so the meter takes back only 1 - f of it and an overcast night reads darker than a
	// clear one on screen, not only in scene units (0: the meter takes it all back, 1: none). It
	// comes in by overcast's share and the exposure's night weight, a function of the clock and the
	// weather alone. Tunable: `post.exposure.overcastNight`.
	const overcastNight = tune( 'post.exposure.overcastNight', LOOK.overcastNight, { min: 0, max: 1 } );
	// Where overcast holds the clouded keys from, degrees: below it overcast keeps that key, its own
	// key toward a low sun (overcastKeyLow) taking the low sun's light, and hands over only to the
	// night's.
	const overcastHeld = LOOK.overcastHeld;

	// The base exposure by the sun's elevation (degrees): [ elevation, exposure ].
	const rows = look.day.exposure;
	const day = rows[ rows.length - 1 ][ 1 ];

	return {
		keys,
		/** The exposure's keys at the sun's `elevation` (degrees), the exposure's own night weight
		 * `night` (sky/DayCycle.ts `exposureNight`, set apart from the display's so the meter's
		 * handover to the night key can follow the light while the tone, the grade and the bloom keep
		 * theirs) and the weather's blend of the four skies `skyMix`, into `out`, the morning's where
		 * the sun is `rising`: the day's (or the rising sun's) and the clouded keys (DayKeys.ts
		 * `keyed`), overcast's share holding the clouded keys at overcastHeld below it, overcast's key
		 * (overcastKeyLow to overcastKey over overcastKeySpan) by overcast's share and the day's
		 * weight, and overcast's night (overcastNight) by overcast's share and the night weight. */
		keysAt( elevation: number, night: number, skyMix: readonly number[], out: ExposureKey, rising = false ): ExposureKey {

			const clouded = cloudedShare( skyMix ), overcast = skyMix[ 3 ] ?? 0;
			keyed( keys, elevation, night, out, overcast < 1 ? ( clouded - overcast ) / ( 1 - overcast ) : 0, rising );
			if ( overcast > 0 ) {

				keyed( keys, Math.max( elevation, overcastHeld ), night, _held, 1 );
				for ( const k in out ) out[ k ] += ( _held[ k as keyof typeof _held ] - out[ k ] ) * overcast;
				const high = THREE.MathUtils.smoothstep( elevation, overcastKeySpan[ 0 ], overcastKeySpan[ 1 ] );
				out.key += ( overcastKeyLow() + ( overcastKey() - overcastKeyLow() ) * high ) * overcast * ( 1 - night );
				out.key -= overcastNight() * Math.log2( 1 / deckLandThrough ) * overcast * night;

			}
			return out;

		},
		at( elevation: number ): number {

			if ( elevation <= rows[ 0 ][ 0 ] ) return rows[ 0 ][ 1 ];
			for ( let i = 1; i < rows.length; i ++ ) {

				const [ e1, x1 ] = rows[ i ];
				if ( elevation > e1 ) continue;
				const [ e0, x0 ] = rows[ i - 1 ];
				return x0 * Math.pow( x1 / x0, ( elevation - e0 ) / ( e1 - e0 ) );

			}
			return day;

		},
		day,
		night: rows[ 0 ][ 1 ],
		nightSky: look.day.nightSkyExposure,
		nightGain: look.day.nightExposureGain,
		nightMeterLift: look.day.nightMeterLift,
	};

}
