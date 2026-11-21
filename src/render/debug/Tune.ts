import { uniform } from 'three/tsl';
import { hasKeys, type DayKey, type DayKeys, type KeyValues } from '../post/DayKeys.ts';
import { settings } from '../Settings.ts';
import { onSceneDispose, onSceneStart } from '../SceneDispose.ts';

// Tunable parameters: named numbers a script or the console can set on a running debug page.
//
// Tunable parameters: named numbers a script or the console can set on a running page. Debug pages
// only (the dev server, ?debug, ?profile); elsewhere every function here returns the default as given and registers nothing, so
// a production page pays nothing.
//
//   const gain = tune( 'lighting.thin.gain', 0.8, { min: 0, max: 2 } );          // CPU: gain()
//   const g = tuneUniform( 'lighting.thin.gain', 0.8, { min: 0, max: 2 } );      // a float uniform
//   tuneUniformOf( G, 'lighting.thin.gain', 0.8, { min: 0, max: 2 } );           // one built earlier
//   tuneApply( 'lighting.vegetation.kappa', 0.95, { min: 0.5 }, ( k ) => { u.value = 1 - k * k; } );
//   export const KEYS = tuneKeys( 'post.exposure', { day: [ ... ], night: { ... } }, ranges );
//   settings.curve = tuneObject( { ...curve }, { toe: 'post.tone.blackSlope' }, ranges );
//   tuneCheck( [ 'post.tone.blackSlope', ... ], () => checkToneCurve( ... ) );    // null or why not
//
// - Names are `<line>.<module>.<quantity>` (`post.tone.blackSlope`). A DayKeys table (DayKeys.ts)
//   registers each value of each key as `<prefix>.<value>@<key>`: the day's keys by their `name`,
//   else `day<i>` in their order, and `cloudy` and `night` (`post.exposure.key@high`,
//   `post.exposure.key@night`); a look's clouded keys by the elevation go by their `name`, else
//   `cloudy<i>`; `keyed()` reads them through getters, so a set value is read on the next frame. A
//   key's value written as a getter (`get highlight() { return KEYS.day[ 0 ].values.highlight; }`)
//   shares another key's value: it registers no tunable of its own and follows the other's.
//   `tuneObject` turns properties of an object the code already reads (a settings object the GUI
//   edits too) into accessors of the registry.
// - Options: `min` / `max`, the range a script may move the value in (a hint: nothing is clamped
//   here, the script may give its own); `scale` 'linear' (the default) or 'log' (a script moves the
//   value in log space); `load`, the value is read once as the app builds (an array length, a
//   branch, a material choice), so a new value needs a reload (`?tune=`) to show; `onChange`, called
//   with the new value (a cache to invalidate).
// - A value set runs no new pipeline: a shader reads a uniform (`tuneUniform`), the CPU reads the
//   getter every frame. Load-time values are flagged so a script leaves them fixed or reloads.
// - A set of values the code cannot use (the tone curve's three, where its solve has a pole) is
//   told by a check registered for their names (`tuneCheck`): the code keeps its last good state
//   there, so a script must know the point is out of bounds rather than read an unchanged frame.
// - At the defaults nothing changes: a uniform holds the default the code had as a constant or a
//   uniform of its own, a getter returns it, so a paused frame stays byte for byte the same.
// - `?tune=<base64 JSON>` ({ name: value }, standard or URL-safe base64) sets values before
//   anything registers: each applies as its name registers, before the materials build, so
//   load-time values take effect too.
// - At run time (assemble/SceneHandles.ts `__PONLAI__.tune`, `tuneApi`): `set( { name: value } )` (returns the names
//   applied, those not registered (kept for when they do), the load-time ones that need a reload and
//   `invalid`, the failing checks over the values now held ({ names, why }); raises the app's
//   `retune` scene event, so `settled` waits for the new frame), `get( name )`,
//   `list()` (name, value, default, min, max, scale, load, kind: 'cpu' | 'uniform'), `reset()` (every
//   value back to its default) and `query( values )` (the `tune=` parameter for those values).
// - A disposed scene's part goes with it (SceneDispose.ts): the entries, nodes, listeners and
//   checks it registered, and the values set since it started; what the app registered as it
//   loaded stays, at the value it started with, and `?tune=` applies again to the next scene.

export interface TuneOptions {
	min?: number;
	max?: number;
	scale?: 'linear' | 'log';
	load?: boolean;
	onChange?: ( value: number ) => void;
}

export interface TuneInfo {
	name: string; value: number; default: number; min: number | null; max: number | null;
	scale: 'linear' | 'log'; load: boolean; kind: 'cpu' | 'uniform';
}

const makeUniform = ( value: number ) => uniform( value );
type FloatUniform = ReturnType<typeof makeUniform>;

interface Entry {
	info: TuneInfo;
	nodes: FloatUniform[];
	listeners: ( ( value: number ) => void )[];
}

const query = settings;
/** Whether this page registers tunables (debug pages only). */
export const TUNE_ON = !! ( import.meta.env?.DEV || query.has( 'debug' ) || query.has( 'profile' ) );

const entries = new Map<string, Entry>();
const checks: { names: string[]; check: () => string | null }[] = [];
// values set before their name registered (`?tune=`, or `set` early)
const pending = new Map<string, number>();

/** The `tune=` URL parameter's JSON from its base64 text (standard or URL-safe). */
export function decodeTuneParam( text: string ): Record<string, number> {

	const b64 = text.replace( /-/g, '+' ).replace( /_/g, '/' );
	const json = JSON.parse( atob( b64 + '='.repeat( ( 4 - b64.length % 4 ) % 4 ) ) ) as Record<string, unknown>;
	const out: Record<string, number> = {};
	for ( const [ name, value ] of Object.entries( json ) ) if ( typeof value === 'number' && Number.isFinite( value ) ) out[ name ] = value;
	return out;

}

if ( TUNE_ON && query.has( 'tune' ) ) {

	try {

		for ( const [ name, value ] of Object.entries( decodeTuneParam( query.get( 'tune' ) ?? '' ) ) ) pending.set( name, value );

	} catch ( error ) {

		console.warn( `tune: the tune= parameter is not base64 JSON (${ ( error as Error ).message })` );

	}

}

function register( name: string, value: number, options: TuneOptions, kind: 'cpu' | 'uniform' ): Entry {

	let entry = entries.get( name );
	if ( ! entry ) {

		const start = pending.get( name ) ?? value;
		pending.delete( name );
		entry = {
			info: { name, value: start, default: value, min: options.min ?? null, max: options.max ?? null, scale: options.scale ?? 'linear', load: options.load ?? false, kind },
			nodes: [], listeners: [],
		};
		entries.set( name, entry );

	} else if ( kind === 'uniform' ) entry.info.kind = 'uniform';
	if ( options.onChange ) entry.listeners.push( options.onChange );
	return entry;

}

function assign( entry: Entry, value: number ): void {

	entry.info.value = value;
	for ( const node of entry.nodes ) node.value = value;
	for ( const listener of entry.listeners ) listener( value );

}

/** A tunable number read on the CPU: returns its getter (see the header). */
export function tune( name: string, value: number, options: TuneOptions = {} ): () => number {

	if ( ! TUNE_ON ) return () => value;
	const entry = register( name, value, options, 'cpu' );
	return () => entry.info.value;

}

/** A tunable number the code works other values out of: `apply` runs with its value now and with
 * each new one (uniforms holding terms of it worked out on the CPU, as the constants they replace
 * were, so a shader at the default reads the numbers it read before). */
export function tuneApply( name: string, value: number, options: TuneOptions, apply: ( value: number ) => void ): void {

	apply( tune( name, value, { ...options, onChange: apply } )() );

}

/** A tunable number a shader reads: returns a float uniform holding it (see the header). */
export function tuneUniform( name: string, value: number, options: TuneOptions = {} ): FloatUniform {

	if ( ! TUNE_ON ) return makeUniform( value );
	const entry = register( name, value, options, 'uniform' );
	const node = makeUniform( entry.info.value );
	entry.nodes.push( node );
	return node;

}

/** A tunable number a shader reads through `node`, a float uniform built earlier (one whose place
 * among the nodes is set as its module loads, before the number is known): sets it to the number
 * and registers it as `tuneUniform` does; returns `node`. */
export function tuneUniformOf( node: FloatUniform, name: string, value: number, options: TuneOptions = {} ): FloatUniform {

	node.value = value;
	if ( ! TUNE_ON ) return node;
	const entry = register( name, value, options, 'uniform' );
	node.value = entry.info.value;
	entry.nodes.push( node );
	return node;

}

/** A DayKeys table whose values are tunables `<prefix>.<value>@<key>` (see the header; reads return
 * the tunable's value, writes set it), with per-value options; the table itself where nothing
 * registers. */
export function tuneKeys<T extends KeyValues>( prefix: string, keys: DayKeys<T>, options: Partial<Record<keyof T, TuneOptions>> = {} ): DayKeys<T> {

	if ( ! TUNE_ON ) return keys;
	// (a key the table's sides share is one key object, so one tunable: its values wrap once)
	const wrapped = new Map<T, T>();
	const wrap = ( values: T, key: string ): T => {

		const held = wrapped.get( values );
		if ( held ) return held;
		const out = {} as KeyValues;
		for ( const k of Object.keys( values ) ) {

			// a value given as a getter is another key's, shared: read through it, not a tunable
			const shared = Object.getOwnPropertyDescriptor( values, k )?.get;
			if ( shared ) {

				Object.defineProperty( out, k, { enumerable: true, get: shared } );
				continue;

			}
			// writes set the tunable too (a GUI slider on the table's values)
			const entry = register( `${ prefix }.${ k }@${ key }`, values[ k ], options[ k as keyof T ] ?? {}, 'cpu' );
			Object.defineProperty( out, k, { enumerable: true, get: () => entry.info.value, set: ( v: number ) => assign( entry, v ) } );

		}
		wrapped.set( values, out as T );
		return out as T;

	};
	const along = ( list: readonly DayKey<T>[], prefix: string ) => list.map( ( d, i ) => ( { ...d, values: wrap( d.values, d.name ?? `${ prefix }${ i }` ) } ) );
	return {
		day: along( keys.day, 'day' ),
		...( keys.rising ? { rising: along( keys.rising, 'rising' ) } : {} ),
		...( keys.cloudy ? { cloudy: hasKeys( keys.cloudy ) ? along( keys.cloudy, 'cloudy' ) : wrap( keys.cloudy, 'cloudy' ) } : {} ),
		...( keys.night ? { night: wrap( keys.night, 'night' ) } : {} ),
	};

}

/** Turns `target`'s properties named in `names` into accessors of the tunables so named (reads
 * return the tunable's value, writes set it); returns `target`. */
export function tuneObject<T extends object>( target: T, names: { [ K in keyof T ]?: T[ K ] extends number ? string : never }, options: { [ K in keyof T ]?: TuneOptions } = {} ): T {

	if ( ! TUNE_ON ) return target;
	for ( const [ k, name ] of Object.entries( names ) as [ keyof T & string, string ][] ) {

		const entry = register( name, target[ k ] as number, ( options as Partial<Record<string, TuneOptions>> )[ k ] ?? {}, 'cpu' );
		Object.defineProperty( target, k, { enumerable: true, configurable: true, get: () => entry.info.value, set: ( v: number ) => assign( entry, v ) } );

	}
	return target;

}

// what the app held as the scene started: each entry's value and the lengths of its lists, the
// checks' count and the values still pending
let atStart: { entries: Map<string, { value: number; nodes: number; listeners: number }>; checks: number; pending: Map<string, number> } | null = null;
onSceneStart( 'tune', () => {

	atStart = {
		entries: new Map( [ ...entries ].map( ( [ name, e ] ) => [ name, { value: e.info.value, nodes: e.nodes.length, listeners: e.listeners.length } ] ) ),
		checks: checks.length, pending: new Map( pending ),
	};

} );
onSceneDispose( 'tune', () => {

	if ( ! atStart ) return;
	for ( const [ name, entry ] of entries ) {

		const was = atStart.entries.get( name );
		if ( ! was ) {

			entries.delete( name );
			continue;

		}
		entry.nodes.length = was.nodes;
		entry.listeners.length = was.listeners;
		if ( entry.info.value !== was.value ) assign( entry, was.value );

	}
	checks.length = atStart.checks;
	pending.clear();
	for ( const [ name, value ] of atStart.pending ) pending.set( name, value );
	atStart = null;

} );

/** A check over tunables the code can only use in some combinations (see the header): `check`
 * returns why the values now held fail, or null. */
export function tuneCheck( names: string[], check: () => string | null ): void {

	if ( TUNE_ON ) checks.push( { names, check } );

}

/** The runtime handle (`__PONLAI__.tune`); `changed` runs after any value was set (main.ts raises
 * the `retune` scene event there). */
export function tuneApi( changed: () => void ) {

	return {
		set( values: Record<string, number> ): { applied: string[]; unknown: string[]; reload: string[]; invalid: { names: string[]; why: string }[] } {

			const applied: string[] = [], unknown: string[] = [], reload: string[] = [];
			for ( const [ name, value ] of Object.entries( values ) ) {

				if ( typeof value !== 'number' || ! Number.isFinite( value ) ) throw new Error( `tune.set: ${ name } = ${ value } is not a finite number` );
				const entry = entries.get( name );
				if ( ! entry ) {

					pending.set( name, value );
					unknown.push( name );
					continue;

				}
				if ( entry.info.value === value ) continue;
				assign( entry, value );
				applied.push( name );
				if ( entry.info.load ) reload.push( name );

			}
			if ( applied.length ) changed();
			const invalid: { names: string[]; why: string }[] = [];
			for ( const { names, check } of checks ) {

				const why = check();
				if ( why ) invalid.push( { names, why } );

			}
			return { applied, unknown, reload, invalid };

		},
		get: ( name: string ): number | null => entries.get( name )?.info.value ?? null,
		list: (): TuneInfo[] => [ ...entries.values() ].map( ( e ) => ( { ...e.info } ) ),
		reset(): string[] {

			const reset: string[] = [];
			for ( const entry of entries.values() ) if ( entry.info.value !== entry.info.default ) {

				assign( entry, entry.info.default );
				reset.push( entry.info.name );

			}
			if ( reset.length ) changed();
			return reset;

		},
		query: ( values: Record<string, number> ): string => `tune=${ btoa( JSON.stringify( values ) ).replace( /\+/g, '-' ).replace( /\//g, '_' ).replace( /=+$/, '' ) }`,
	};

}
