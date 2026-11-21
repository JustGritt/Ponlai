import type { AAMode } from './post/Antialiasing.ts';
import type { QualityTier } from './Quality.ts';
import type { DayPreset } from './sky/DayCycle.ts';
import type { WeatherName, WindLevel } from './sky/Weather.ts';

// The app's settings: the render pipeline's URL options, and the one reader of every layer's.
//
// The app's settings: the render pipeline's own options in the page's URL, each declared once
// here with its type, its range and its default, and the one reader every module takes the app's
// options from. A layer above the render pipeline declares its own options in a table of the same
// shape (`SettingsTable`) beside its code and reads them by `settingsFor( its table )`, the same
// reader typed for its names and the render pipeline's; the app merges the layers' tables
// (`mergeSettings`: a name two layers declare throws) for the reports below.
//
//   import { settings } from './Settings.ts';
//   const option = settings.get( 'windView' );          // the value as written, or null
//   if ( settings.has( 'paused' ) ) ...                  // named at all (a bare `?paused` too)
//
// - The app's entry module parses the URL once (`parseSettings`), reports in one form each name its
//   merged table does not declare and each value outside its range (`settingProblems`: a console
//   warning `?name=value: why`, the option keeping whatever its reader makes of it), chooses the
//   quality tier and installs both (`installSettings`) before any other module loads, so a module
//   that reads a setting as it loads takes the value it always did. No other module parses the
//   address. (This repository's entries: app/PageSettings.ts, app/ScenesBoot.ts.)
// - Where nobody installed them (node under the tools, Vite's server-side loads, a worker), every
//   option reads as absent: each reader takes its own default, as on a page without the option, and
//   nothing throws. A tool that builds what a page with options builds installs them first
//   (`installSettings( parseSettings( query ) )`: tools/render/wgsl-build.mjs `--q`).
// - They are installed once, and before any read: a second install throws, and so does an install
//   after a module read an option (it took the default, whatever the address says), naming the
//   first option read and where. The look (look/Look.ts) and the scene's uniforms (shared.ts) keep
//   the same rule.
// - The reader keeps the address's own values (the first of a name for `get`, all for `getAll`):
//   each module still reads its options as it did, so the values, the defaults the modules apply
//   and their console warnings are unchanged. `get` takes only a name its reader is typed for, so a
//   misspelt read, or a render pipeline's read of a layer's option above it, fails the type check;
//   `prefixed` reads the declared prefixes (`tier.<setting>`, Quality.ts).
// - RENDER_SETTINGS lists the options by the groups of docs/reference/url-options.md, whose table of
//   every layer's options tools/repo/verify-settings.mjs writes from the tables and checks against
//   them: a new option is declared in the table of the lowest layer that reads it, explained there
//   in its group, and the table rewritten (`node tools/repo/verify-settings.mjs --write`).
//   `default` is the reader's default as words for that table, never read by the code; a `choice`
//   lists the values its reader tells apart (others are reported), a `number` its range and any
//   words it takes besides numbers (`words`). `debug` options are read on debug pages only (the dev
//   server, `?debug`, `?profile`).
// - The installed settings never change on a page. A setting read when materials and the frame
//   graph are built is a load setting: a new value reloads the page (PageSettings.ts `reloadWith`).
//   What may change on a running page is a named number of the tune table (debug/Tune.ts) or a
//   handle on `window.__PONLAI__`.

export interface SettingSpec {
	/** `switch`: named or not, its value unread; `choice`: one of `values`; `number`, `integer`: a
	 * number in `range` (or one of `words`); `list`: a comma list; `text`: any text. */
	readonly type: 'switch' | 'choice' | 'number' | 'integer' | 'list' | 'text';
	/** What the app does without it, as words for the docs' table. */
	readonly default: string;
	readonly values?: readonly string[];
	readonly range?: readonly [ number, number ];
	readonly words?: readonly string[];
	readonly debug?: boolean;
	/** One line for the docs' table: what it sets. */
	readonly about: string;
}

/** A layer's options by name (`RENDER_SETTINGS`; the island's, the game's and the app's own). */
export type SettingsTable = Readonly<Record<string, SettingSpec>>;

const WEATHERS = [ 'clear', 'fair', 'cloudy', 'overcast', 'rain', 'storm', 'fog' ] as const satisfies readonly WeatherName[];
const WIND_LEVELS = [ 'light', 'gentle', 'fresh', 'strong' ] as const satisfies readonly WindLevel[];
const PRESETS = [ 'dawn', 'sunrise', 'noon', 'golden-hour', 'sunset', 'dusk', 'midnight' ] as const satisfies readonly DayPreset[];
const TIERS = [ 'low', 'medium', 'high' ] as const satisfies readonly QualityTier[];
const AA = [ 'msaa', 'taa', 'taau', 'fsr3', 'off', 'gt' ] as const satisfies readonly AAMode[];

/** The render pipeline's options. */
export const RENDER_SETTINGS = {
	// view and capture
	paused: { type: 'switch', default: 'running', about: 'pauses the scene and both clocks; with it the page is in the test mode' },
	t: { type: 'number', default: '0', range: [ 0, Infinity ], about: 'the scene time (s)' },
	dpr: { type: 'number', default: 'the display\'s ratio, capped', range: [ 0, Infinity ], about: 'pins the canvas\'s pixel ratio' },
	// sky and weather
	hour: { type: 'number', default: 'golden-hour (8 under `game`)', about: 'the hour of the day' },
	preset: { type: 'choice', default: 'golden-hour', values: PRESETS, about: 'the hour by its sun' },
	dayLength: { type: 'number', default: '2880 (1440 under `game`)', range: [ 30, Infinity ], about: 'seconds per day' },
	moon: { type: 'number', default: '0.5', about: 'the moon\'s phase (0.5 full)' },
	clock: { type: 'choice', default: 'running', values: [ 'paused' ], about: '`paused`: freezes the day clock only' },
	weather: { type: 'choice', default: 'auto, from fair', values: WEATHERS, about: 'holds this weather (the parked ones on debug pages only)' },
	wind: { type: 'choice', default: 'the sky\'s', values: WIND_LEVELS, about: 'the wind\'s level; holds the weather' },
	windDir: { type: 'number', default: 'the sky\'s (40)', about: 'the wind\'s heading (degrees); holds the weather' },
	clouds: { type: 'number', default: 'the weather\'s', range: [ 0, 1 ], about: 'overrides the cloud coverage' },
	dayFog: { type: 'switch', default: 'off', about: 'the day\'s fog layer under a clear or fair sky' },
	seaFog: { type: 'switch', default: 'off', about: 'the deep sea fog' },
	// quality, anti-aliasing and the frame
	quality: { type: 'choice', default: 'high', values: TIERS, about: 'the quality tier (medium and low parked)' },
	aa: { type: 'choice', default: 'the tier\'s (high: msaa)', values: AA, about: 'the anti-aliasing mode (`gt` on debug pages only)' },
	msaa: { type: 'choice', default: 'none', values: [ '1', '2', '4' ], about: 'the older alias of `aa`: `4` msaa, `1` and `2` off' },
	fsrDebug: { type: 'switch', default: 'off', debug: true, about: 'FSR 3\'s debug view over its output, under `aa=fsr3`' },
	taaResolve: { type: 'choice', default: 'on', values: [ 'off' ], debug: true, about: '`off`: the temporal modes without their resolve' },
	post: { type: 'list', default: 'none', values: [ 'legacy', 'inline-composite', 'three-fsr', 'store-scene' ], about: 'the frame graph without the first performance round\'s changes' },
	gpuTimer: { type: 'choice', default: 'on', values: [ 'off' ], about: '`off`: the dynamic resolution without its timestamp queries' },
	drsStore: { type: 'choice', default: 'on, off in an automated browser', values: [ 'on', 'off' ], about: 'whether the dynamic resolution starts from the scale it last held' },
	frameRate: { type: 'number', default: 'auto', range: [ 0, Infinity ], words: [ 'auto' ], about: 'the frame rate the loop holds (0: every refresh)' },
	frameFallback: { type: 'choice', default: 'on', values: [ '0' ], about: '`0`: `auto` keeps the tier\'s rate' },
	submits: { type: 'choice', default: 'batched', values: [ 'each' ], about: '`each`: three\'s own submit per pass' },
	submitBatch: { type: 'list', default: 'SubmitBatch.ts\'s', about: 'the submit batching\'s policy: `hold,wait,idle`' },
	fgOps: { type: 'choice', default: 'derived', values: [ 'derived', 'legacy', 'store', 'off' ], about: 'the frame graph\'s attachment ops' },
	fgPrime: { type: 'choice', default: 'on', values: [ '0' ], about: '`0`: the warm-up renders nothing ahead' },
	fgview: { type: 'text', default: 'off', debug: true, about: 'the frame graph\'s debug view of one resource' },
	// debug
	debug: { type: 'switch', default: 'off', about: 'a debug page: the handles on `window`, the debug options' },
	flags: { type: 'list', default: 'none', debug: true, about: 'sets debug flags as the page loads (`name:value,...`)' },
	tune: { type: 'text', default: 'none', debug: true, about: 'tunable parameters\' values before anything builds (base64 JSON)' },
	testMode: { type: 'choice', default: 'on with `paused`', values: [ '0' ], about: '`0`: a paused page outside the test mode' },
	profile: { type: 'choice', default: 'off', values: [ '', 'frames' ], about: 'loads the frame profiler (`frames`: without GPU timestamps)' },
	vegPrepass: { type: 'choice', default: 'on', values: [ '0', 'coverage' ], about: 'the vegetation\'s depth prepass: off, or with coverage kept' },
	ambientProbes: { type: 'choice', default: 'off', values: [ 'off', 'sky', 'bounce', 'irradiance' ], debug: true, about: 'shows the ambient light\'s probes' },
	windView: { type: 'choice', default: 'off', values: [ '', 'arrows', 'plants' ], debug: true, about: 'makes the vegetation\'s wind visible' },
} as const satisfies SettingsTable;

/** Options read by a prefix: `tier.<setting path>=<value>` (Quality.ts `qualityProfile`). */
export const SETTING_PREFIXES = {
	'tier.': { type: 'text', default: 'the tier\'s', about: 'overrides one setting of the tier (`tier.post.hazeGrid.width=128`)' },
} as const satisfies SettingsTable;

/** The render pipeline's option names. */
export type SettingName = keyof typeof RENDER_SETTINGS;
export type SettingPrefix = keyof typeof SETTING_PREFIXES;

/** What a module reads its options through: the app's installed settings (`settings`, or a
 * layer's `settingsFor` its table), or a query of its own (`parseSettings`, a scene jump's), for
 * the names `N` (the render pipeline's by default). URLSearchParams has the same shape. */
export interface SettingsQuery<N extends string = SettingName> {
	get( name: N ): string | null;
	has( name: N ): boolean;
	getAll( name: N ): string[];
}

/** A parsed query: its options as the address gives them, in order. */
export interface ParsedSettings<N extends string = SettingName> extends SettingsQuery<N> {
	readonly entries: ReadonlyArray<readonly [ string, string ]>;
	prefixed( prefix: SettingPrefix ): Array<[ string, string ]>;
}

/** Parses a query (`?a=1&b`, the leading `?` optional) as the address's options. */
export function parseSettings( search: string ): ParsedSettings<string> {

	const entries = Object.freeze( [ ...new URLSearchParams( search ) ].map( ( e ) => Object.freeze( e ) ) );
	const values = new Map<string, string[]>();
	for ( const [ name, value ] of entries ) {

		const list = values.get( name );
		if ( list ) list.push( value );
		else values.set( name, [ value ] );

	}
	return Object.freeze( {
		entries,
		get: ( name: string ) => values.get( name )?.[ 0 ] ?? null,
		has: ( name: string ) => values.has( name ),
		getAll: ( name: string ) => [ ...( values.get( name ) ?? [] ) ],
		prefixed: ( prefix: SettingPrefix ) => entries.filter( ( [ name ] ) => name.startsWith( prefix ) ).map( ( [ name, value ] ): [ string, string ] => [ name, value ] ),
	} );

}

type Merged<T extends readonly SettingsTable[]> = T extends readonly [ infer First, ...infer Rest extends readonly SettingsTable[] ] ? First & Merged<Rest> : unknown;

/** The app's options: the layers' tables in one (the app's entry module merges them; a name declared by
 * two throws). */
export function mergeSettings<const T extends readonly SettingsTable[]>( ...tables: T ): Merged<T> {

	const merged: Record<string, SettingSpec> = {};
	for ( const table of tables ) for ( const [ name, spec ] of Object.entries( table ) ) {

		if ( Object.hasOwn( merged, name ) ) throw new Error( `mergeSettings: the option ${ name } is declared twice` );
		merged[ name ] = spec;

	}
	return Object.freeze( merged ) as Merged<T>;

}

/** Each name `table` (the app's merged options) does not declare and each value outside its
 * declared range, as `?name=value: why`. */
export function settingProblems( query: ParsedSettings<string>, table: SettingsTable ): string[] {

	const out: string[] = [];
	for ( const [ name, value ] of query.entries ) {

		const at = `?${ value === '' ? name : `${ name }=${ value }` }`;
		if ( ! Object.hasOwn( table, name ) ) {

			if ( ! Object.keys( SETTING_PREFIXES ).some( ( prefix ) => name.startsWith( prefix ) ) ) out.push( `${ at }: no such option` );
			continue;

		}
		const spec = table[ name ];
		const values = spec.values ?? [];
		if ( spec.type === 'choice' && ! values.includes( value ) ) out.push( `${ at }: expected ${ values.map( ( v ) => v === '' ? 'nothing' : v ).join( ', ' ) }` );
		else if ( spec.type === 'list' && spec.values && value !== '' ) {

			const unknown = value.split( ',' ).filter( ( item ) => ! values.includes( item ) );
			if ( unknown.length ) out.push( `${ at }: ${ unknown.join( ', ' ) } not among ${ values.join( ', ' ) }` );

		} else if ( ( spec.type === 'number' || spec.type === 'integer' ) && ! spec.words?.includes( value ) ) {

			const n = value.trim() === '' ? NaN : Number( value );
			const [ lo, hi ] = spec.range ?? [ - Infinity, Infinity ];
			if ( ! Number.isFinite( n ) ) out.push( `${ at }: not a number${ spec.words ? ` or ${ spec.words.join( ', ' ) }` : '' }` );
			else if ( spec.type === 'integer' && ! Number.isInteger( n ) ) out.push( `${ at }: not a whole number` );
			else if ( n < lo || n > hi ) out.push( `${ at }: outside ${ lo } to ${ hi }` );

		}

	}
	return out;

}

const NONE = parseSettings( '' );
let installed: { query: ParsedSettings<string>; tier: string | null } | null = null;
// the first option read before the install (a module that loaded before the entry installed them),
// with the line of the stack that read it where the engine gives one
let readEarly: string | null = null;
const current = ( name: string ) => {

	if ( ! installed && readEarly === null ) {

		const at = new Error().stack?.split( '\n' ).slice( 1 ).find( ( line ) => ! line.includes( 'Settings.ts' ) )?.trim();
		readEarly = `?${ name }${ at ? ` (${ at })` : '' }`;

	}
	return installed?.query ?? NONE;

};

/** Installs the app's settings and the quality tier its entry chose (null: the `quality` option's,
 * else the default: Quality.ts `loadTier`), before the modules that read them load. Once only, and
 * before any read: it throws on a second install, and when a module read an option before (it took
 * the default, whatever the address says), naming the first. */
export function installSettings( query: ParsedSettings<string>, tier: string | null = null ): void {

	if ( installed ) throw new Error( 'installSettings: the settings are installed once, before the modules that read them' );
	if ( readEarly !== null ) throw new Error( `installSettings: the settings were read before they were installed, first ${ readEarly }: install them before any module that reads them loads` );
	installed = { query, tier };

}

/** A number option of `query` (the installed settings by default): its value, or `fallback` where
 * it is missing or not a finite number. */
export function numberSetting<N extends string = SettingName>( name: N, fallback: number, query: SettingsQuery<N> = settings as SettingsQuery<string> ): number {

	const value = Number( query.get( name ) ?? fallback );
	return Number.isFinite( value ) ? value : fallback;

}

/** The installed settings, read by the names `N`. */
export type InstalledSettings<N extends string = SettingName> = ParsedSettings<N> & { readonly installed: boolean; readonly tier: string | null };

/** The installed settings, read by the render pipeline's names (every option absent where none are
 * installed). */
export const settings: InstalledSettings = Object.freeze( {
	get entries() { return current( 'entries' ).entries; },
	get installed() { return installed !== null; },
	get tier() { current( 'quality' ); return installed?.tier ?? null; },
	get: ( name: string ) => current( name ).get( name ),
	has: ( name: string ) => current( name ).has( name ),
	getAll: ( name: string ) => current( name ).getAll( name ),
	prefixed: ( prefix: SettingPrefix ) => current( prefix ).prefixed( prefix ),
} );

/** The installed settings read by a layer above the render pipeline: the same reader as `settings`, by
 * the render pipeline's names and those of `table` (the layer's own, or the app's merged ones). */
export function settingsFor<T extends SettingsTable>( _table: T ): InstalledSettings<SettingName | Extract<keyof T, string>> {

	return settings as InstalledSettings<string>;

}
