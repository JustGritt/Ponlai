import { tune } from '../debug/Tune.ts';
import type { AmbientColumn, AmbientSettings, AmbientLook, AmbientRowLook } from './Look.ts';

// The environment light's look built from a look's table: the ambient grade and the key's lift.
//
// The environment light's look built from a look's table (look/Look.ts `AmbientLook`): the ambient
// grade's table by the sun's elevation and the weather's skies, with the gains under overcast's
// deck, which lighting/AmbientGrade.ts writes into the lighting models' uniforms once a frame
// (`createAmbientTables`); and the low key's lift, the shading elevation under a low sun, which
// lighting/KeyLift.ts turns into the lifted direction (`createKeyLift`). Like the exposure
// (look/ExposureLook.ts), they are taken as an argument where the render pipeline's stages create
// them (RenderStages.ts: the tables in `createLight`, for the grade's controller and the ambient
// light's relight; the lift in `createSky`, which the sky carries beside the key's direction), so
// two made from two tables hold their own numbers (tools/render/verify-look-params.mjs) and this
// module holds none at module level: only the parameters' and the columns' names and the
// parameters at no grade, which no look changes. The land's floor under cloud is the sky's (look/SkyLook.ts
// `keyThroughCloud`). Each module's header says what its numbers do on screen.
// - The tables register their tunables (debug/Tune.ts) as they are created, under the names they
//   always had (`lighting.grade.sunBounce`, `lighting.relight.overcastLand`). The tune table is one
//   per process: on a debug page a second set made from another table reads the first one's
//   tunables.

export type { AmbientSettings };

export const AMBIENT_SETTINGS: ReadonlyArray<keyof AmbientSettings> = [ 'skySpecular', 'skyBounce', 'sunBounce', 'directionality', 'aoStrength' ];

/** The table's columns, the skies of the weather's blend (Weather.ts MIX_SKIES' order). */
export const AMBIENT_COLUMNS = [ 'clear', 'fair', 'cloudy', 'overcast' ] as const satisfies readonly AmbientColumn[];
type AmbientRow = { at: number } & Record<AmbientColumn, Readonly<AmbientSettings>>;

// The parameters at no grade: a look's rows spell their changes over it.
export const DEFAULT_AMBIENT_SETTINGS: Readonly<AmbientSettings> = { skySpecular: 1, skyBounce: 1, sunBounce: 1, directionality: 0.25, aoStrength: 1 };
const D = DEFAULT_AMBIENT_SETTINGS;
const whole = ( row: AmbientRowLook ): AmbientRow => ( { at: row.at, clear: { ...D, ...row.clear }, fair: { ...D, ...row.fair }, cloudy: { ...D, ...row.cloudy }, overcast: { ...D, ...row.overcast } } );

/** The ambient grade's table and its gains a look's table makes (`createAmbientTables`). */
export interface AmbientTables {
	/** The table at a sun elevation (degrees), `rising` or setting, and the weather's column
	 * weights, into `out`. */
	evaluate( elevation: number, weights: readonly number[], out: AmbientSettings, rising?: boolean ): AmbientSettings;
	/** A gain on the table's `sunBounce` at every hour and in every weather: scales the key's bounce
	 * throughout without reshaping the table: the shadows' level is the light the ground sends up.
	 * Tunable: `lighting.grade.sunBounce`. */
	readonly sunBounceGain: () => number;
	/** The sun's part of the weather's gain under overcast's deck, per unit, on the land's ambient
	 * light: on the sky's light per unit of the bake (after the unit that makes it a clear day's in
	 * the same place: AmbientRelight.ts OVERCAST_LAND_UNIT), its bounce with it, by overcast's share
	 * and its deck's light (AmbientGrade.ts `overcastLandWeight`), and on the key's bounce per unit
	 * of the table's (AmbientGrade.ts `update`), by overcast's share and the deck's light at the
	 * sun's height (the moon's at its own height takes `overcastMoonLand`). It is not the land's
	 * whole ambient light under the deck over a clear day's: that ratio also holds the bake's ratio
	 * of the two skies and the sun's bounce's share of the light. 1 leaves the land as the sky and
	 * the table give it. Tunable: `lighting.relight.overcastLand`. */
	readonly overcastLand: () => number;
	/** The moon's part of the weather's land gain under overcast's deck (`overcastLand` is the
	 * sun's): the moonlit land's sky light and the moon's bounce under the deck. */
	readonly overcastMoonLand: number;
}

/** The ambient grade's tables made from a look's table: registers their tunables (see the header). */
export function createAmbientTables( look: AmbientLook ): AmbientTables {

	// The grade by the sun's elevation (degrees, descending), and the rising sun's.
	const grade: readonly AmbientRow[] = look.grade.map( whole );
	const gradeRising: readonly AmbientRow[] = look.gradeRising.map( whole );
	const sunBounceGain = tune( 'lighting.grade.sunBounce', look.sunBounceGain, { min: 0.25, max: 4, scale: 'log' } );
	const overcastLand = tune( 'lighting.relight.overcastLand', look.overcastLand, { min: 1, max: 2 } );
	return {
		evaluate( elevation, weights, out, rising = false ) {

			const rows = rising ? gradeRising : grade;
			let a = rows[ rows.length - 1 ], b = a, t = 0;
			if ( elevation >= rows[ 0 ].at ) a = b = rows[ 0 ];
			else for ( let i = 1; i < rows.length; i ++ ) if ( elevation >= rows[ i ].at ) {

				a = rows[ i ]; b = rows[ i - 1 ];
				t = ( elevation - a.at ) / ( b.at - a.at );
				break;

			}
			for ( const key of AMBIENT_SETTINGS ) {

				let sum = 0, weight = 0;
				AMBIENT_COLUMNS.forEach( ( column, c ) => {

					const w = weights[ c ] ?? 0, x = a[ column ][ key ];
					sum += w * ( x + ( b[ column ][ key ] - x ) * t );
					weight += w;

				} );
				out[ key ] = weight > 0 ? sum / weight : D[ key ];

			}
			return out;

		},
		sunBounceGain,
		overcastLand,
		overcastMoonLand: look.overcastMoonLand,
	};

}

// ---- the low key's lift (lighting/KeyLift.ts)

/** The shading elevation (degrees) under a sun key at `elevation` (degrees), a look's lift
 * (`createKeyLift`). */
export type KeyLiftShading = ( elevation: number ) => number;

/** The low key's lift made from a look's table (look/Look.ts `AmbientLook.keyLift`): the shading
 * direction's elevation at the sun key's (its rows' `at`, ascending), held below the first row;
 * from the last row's on, the shading holds at the last row's `shading` until the key's own
 * elevation passes it, the corner rounded over `join` degrees either side. A moon key shades from
 * the moon's own direction (sky/DayCycle.ts MOON). A single row whose `at` and `shading` are equal,
 * with no corner (the neutral look's), is no lift: the key shades as itself at every elevation. */
export function createKeyLift( keyLift: AmbientLook[ 'keyLift' ] ): KeyLiftShading {

	const { rows, join } = keyLift;
	// The rows' slopes (shading per degree of the key): Fritsch and Carlson's monotone cubic
	// ("Monotone Piecewise Cubic Interpolation", SIAM J. Numer. Anal. 1980), flat at both ends, so
	// the lift meets the hold below and the floor above smoothly.
	const slope = rows.map( ( r, i ) => {

		if ( i === 0 || i === rows.length - 1 ) return 0;
		const h0 = r.at - rows[ i - 1 ].at, h1 = rows[ i + 1 ].at - r.at;
		const d0 = ( r.shading - rows[ i - 1 ].shading ) / h0, d1 = ( rows[ i + 1 ].shading - r.shading ) / h1;
		return d0 * d1 <= 0 ? 0 : 3 * ( h0 + h1 ) / ( ( 2 * h1 + h0 ) / d0 + ( h1 + 2 * h0 ) / d1 );

	} );
	const last = rows[ rows.length - 1 ];
	return ( elevation ) => {

		if ( elevation <= rows[ 0 ].at ) return rows[ 0 ].shading;
		if ( elevation >= last.at ) {

			// max( elevation, floor ), its corner a parabola meeting both lines with their slopes
			const floor = last.shading, w = join, x = elevation - floor;
			return x <= - w ? floor : x >= w ? elevation : floor + ( x + w ) * ( x + w ) / ( 4 * w );

		}
		let i = 1;
		while ( elevation >= rows[ i ].at ) i ++;
		const a = rows[ i - 1 ], b = rows[ i ], h = b.at - a.at, t = ( elevation - a.at ) / h, t2 = t * t, t3 = t2 * t;
		return ( 2 * t3 - 3 * t2 + 1 ) * a.shading + ( t3 - 2 * t2 + t ) * h * slope[ i - 1 ]
			+ ( 3 * t2 - 2 * t3 ) * b.shading + ( t3 - t2 ) * h * slope[ i ];

	};

}
