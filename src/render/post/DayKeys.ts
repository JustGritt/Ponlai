// A look's values through the day: keys by the sun's elevation, blended by cloud and night.
//
// A look's values through the day: a few keys by the sun's elevation (degrees), each key's values
// reached at its elevation and blended between neighbouring keys by a smoothstep (so a value rests
// at each key and moves between them), held beyond the first and the last; a clouded day's values,
// where a look has them, come in over them by the weather's cloud (Pipeline.ts `cloud`: 0 under a
// clear or fair sky, 1 under a cloudy or overcast one), and the night's, where a look has them,
// over both by the night's weight (Pipeline.ts `night`, from the daylight). A look's clouded values are one set where they
// hold at every sun, or keys by the elevation of their own, read as the day's are, where a flat set
// would leave the land as bright at sunset as in the afternoon. Each look sets its own keys: the
// shadows' contrast holds one value from the noon down to a 9 degree sun, the local tone moves
// between 40, 12 and 3 degrees, and the exposure between keys from 40 degrees down to the dusk
// (0.4 and -0.4 degrees, and a clouded day's from 12 down to -1). A look whose morning is not its evening has keys of its own
// for the rising sun (`rising`), which take the place of the clear day's while the sun climbs, and
// share the day's key objects from where the two sides of the day are one, so a key holds one value
// there and the change of side at noon moves nothing. Read on the CPU once a frame into uniforms, so a
// value is a function of the clock and the weather alone and a paused frame is bit-identical. A key may carry
// a `name` (`golden`), which names its values as tunables (debug/Tune.ts `tuneKeys`:
// `<prefix>.<value>@<name>`, else `@day<i>`, `@cloudy<i>`, `@rising<i>`).

export type KeyValues = Record<string, number>;
export interface DayKey<T extends KeyValues> {
	readonly at: number;
	readonly values: T;
	readonly name?: string;
}
export interface DayKeys<T extends KeyValues> {
	// ascending by elevation
	readonly day: readonly DayKey<T>[];
	// the clear day's while the sun rises, ascending by elevation (none: the day's)
	readonly rising?: readonly DayKey<T>[];
	// a cloudy or overcast day's: one set at every sun, or keys ascending by elevation (none: the
	// clear day's)
	readonly cloudy?: T | readonly DayKey<T>[];
	// the night's (none: the day's, as for a look whose night is lit as its day, by the moon)
	readonly night?: T;
}

const smooth = ( t: number ) => t * t * ( 3 - 2 * t );

/** Whether a look's clouded values are keys by the elevation. */
export function hasKeys<T extends KeyValues>( cloudy: T | readonly DayKey<T>[] ): cloudy is readonly DayKey<T>[] {

	return Array.isArray( cloudy );

}

// The value `k` of `keys` at the sun's `elevation`.
function along<T extends KeyValues>( keys: readonly DayKey<T>[], elevation: number, k: string ): number {

	let i = 0;
	while ( i < keys.length - 1 && elevation > keys[ i + 1 ].at ) i ++;
	const a = keys[ i ], b = keys[ Math.min( i + 1, keys.length - 1 ) ];
	const t = b.at > a.at ? smooth( Math.min( Math.max( ( elevation - a.at ) / ( b.at - a.at ), 0 ), 1 ) ) : 0;
	return a.values[ k ] + ( b.values[ k ] - a.values[ k ] ) * t;

}

/** `keys`' values at the sun's `elevation` (degrees), the night's weight `night` and the weather's
 * cloud `cloud`, into `out`, the morning's where the sun is `rising` and the look has one. */
export function keyed<T extends KeyValues>( keys: DayKeys<T>, elevation: number, night: number, out: T, cloud = 0, rising = false ): T {

	const { cloudy, night: atNight } = keys;
	const day = rising && keys.rising ? keys.rising : keys.day;
	const o = out as KeyValues;
	for ( const k in day[ 0 ].values ) {

		let v = along( day, elevation, k );
		if ( cloudy ) v += ( ( hasKeys( cloudy ) ? along( cloudy, elevation, k ) : cloudy[ k ] ) - v ) * cloud;
		o[ k ] = atNight ? v + ( atNight[ k ] - v ) * night : v;

	}
	return out;

}
