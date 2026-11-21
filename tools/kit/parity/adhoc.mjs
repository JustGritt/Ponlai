// Ad-hoc views for capture-set.mjs (and its --quick mode): views given on the command line or in a
// JSON file instead of the views table (tables.mjs), captured through the same path (one page per set, the scene jump,
// readiness, --jobs, the reference store). They replace the scripts that looped tools/kit/capture.mjs
// over a list of their own, one cold page per view.
//
//   --view "name:query[::js]"   one view: its query, and optionally an eval (js run in the page as a
//                               table view's eval is, with what the table gives it: Ponlai's gives `s`,
//                               the page's handle, and the helpers isl(), free(), orbitAround()).
//                               `like=<view>` in the query starts from that table view (its query,
//                               eval and crops; the view's own options win).
//   --views-file f.json         [ { name, q, eval?, crops?, like? }, ... ] or { name: { q, ... } }
//   --sweep "key=a,b,c"         every selected view once per value (several sweeps: every
//                               combination), named `<view>--<key>-<value>` (`<key>-<value>` alone
//                               when only sweeps are given; a dot in a value becomes `_`)
//
// --views (names and the table's groups, mixed freely) still selects the table's views, alone or
// beside these; with no --views and some ad-hoc view or sweep, the set is just those (not the
// default group).
//
// The query: an ad-hoc view's own options win over the set's (`t=12`, --quality, --aa, --q), so it
// may set its own time or tier; the page reads the first of a repeated option, so the view's query
// goes after the set's with the keys it sets taken out. The table's views keep the set's query first,
// unchanged. Views whose load-time options differ (anything outside the page's scene options: the
// tier, aa, debug switches, `game`, `seed` ...) cannot share a page (establishScene refuses them),
// so capture-set captures them in groups, one group of pages per set of load-time options, in the
// order the views were given. The groups are made before any page loads, so the scene options are
// read from the module the configuration's `sceneOptions` names (the list the page hands its
// assembly, which the page mounts as `sceneOptions` on its handle). A project that names none
// counts every option as a load-time one: views that differ at all take pages of their own, slower
// but never refused. A list that names an option the page does not let a jump change still fails
// at the page, which refuses the jump.
//
// An eval of an ad-hoc view's own (`::js`, a views-file `eval`, the js of a `like=` view or of a
// sweep over one) is not covered by the scene jump, which undoes the scene's options but not GUI
// or console changes: the view after it gets a fresh page (`jumpClean`, capture-set.mjs's header).
//
// Names: letters, digits, `-` and `_` (a file name, and no dot: `<view>.repeat.png` and the
// directory scan read a dot as a suffix), distinct from the table's. The reference store keys a set's
// views by name, so an ad-hoc view's coverage carries its definition's hash (`def`): a stored view of
// the same name but another definition is captured again, never served (reference.mjs SET).
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { viewTable } from './tables.mjs';
import PROJECT from '../../project.mjs';
import { ROOT } from '../root.mjs';

const { VIEWS, viewNames, expandViews, ANCHOR_NAMES } = viewTable;
const NAME = /^[A-Za-z0-9_-]+$/;

// the page's scene options: what a scene jump may change (the configuration's `sceneOptions`; none
// in a project that names no module, so that every option groups views by page)
const SCENE_OPTIONS = ! PROJECT.sceneOptions ? new Set() : ( () => {

	const m = readFileSync( path.join( ROOT, PROJECT.sceneOptions ), 'utf8' ).match( /const SCENE_OPTIONS = \[([^\]]*)\]/ );
	if ( ! m ) throw new Error( `adhoc.mjs: SCENE_OPTIONS not found in ${ PROJECT.sceneOptions }` );
	return new Set( [ ...m[ 1 ].matchAll( /'([^']+)'/g ) ].map( ( x ) => x[ 1 ] ) );

} )();

// a query as [ key, raw pair ] in order (no decoding: the pairs are passed on as written)
const pairs = ( q ) => ( q ?? '' ).split( '&' ).filter( Boolean ).map( ( p ) => [ p.split( '=' )[ 0 ], p ] );
// the first pair of each key, in order
const firstWins = ( list ) => {

	const seen = new Set();
	return list.filter( ( [ k ] ) => ! seen.has( k ) && seen.add( k ) );

};
const hash = ( text ) => createHash( 'sha256' ).update( text ).digest( 'hex' ).slice( 0, 16 );
// the table's crop anchor names
const ANCHORS = new Set( ANCHOR_NAMES );
// the table's own evals, taken before any ad-hoc view is defined (see jumpClean)
const VIEWS_EVALS = new Set( Object.values( VIEWS ).map( ( v ) => v.eval ).filter( Boolean ) );

/** Whether the next scene jump undoes what the view's eval did, so its page can serve another view:
 * no eval, or exactly a table view's (the jump's contract covers those: tools/island/verify-scene-jump.mjs
 * checks the views reached from each other). Any other js (an ad-hoc `::js`, a views-file `eval`, a
 * `like=` view's js on top, a sweep over such a view) may change what no jump resets (the GUI's
 * settings, Tune, debug flags, a uniform): capture-set ends that view's page after it. */
export const jumpClean = ( name ) => ! VIEWS[ name ].eval || VIEWS_EVALS.has( VIEWS[ name ].eval );

/** Registers an ad-hoc view into VIEWS (so viewEval and every reader of VIEWS take it). */
function define( { name, q = '', eval: js, crops, like } ) {

	if ( ! NAME.test( name ?? '' ) ) throw new Error( `ad-hoc view name ${ JSON.stringify( name ) }: letters, digits, - and _ only` );
	if ( VIEWS[ name ] ) throw new Error( `ad-hoc view ${ name }: ${ VIEWS[ name ].adhoc ? 'given twice' : 'a view of the table has that name' }` );
	let own = pairs( q );
	const likeName = like ?? own.find( ( [ k ] ) => k === 'like' )?.[ 1 ].slice( 5 );
	own = own.filter( ( [ k ] ) => k !== 'like' );
	const base = likeName ? VIEWS[ likeName ] : null;
	if ( likeName && ! base ) throw new Error( `ad-hoc view ${ name }: like=${ likeName } is not a view` );
	const query = firstWins( [ ...own, ...pairs( base?.q ) ] ).map( ( [ , p ] ) => p ).join( '&' );
	const evalJs = [ base?.eval, js ].filter( Boolean ).join( '\n' ) || undefined;
	const anchors = crops ?? base?.crops ?? [];
	const bad = anchors.filter( ( a ) => ! ANCHORS.has( a ) );
	if ( bad.length ) throw new Error( `ad-hoc view ${ name }: unknown crop anchor ${ bad.join( ', ' ) } (the table's: ${ [ ...ANCHORS ].join( ', ' ) })` );
	VIEWS[ name ] = { q: query, eval: evalJs, crops: anchors, adhoc: true, like: likeName ?? undefined,
		def: hash( JSON.stringify( [ query, evalJs ?? '', anchors ] ) ) };

}

/** The views a command line asks for, ad-hoc ones registered into VIEWS: an array of names.
 * `o`: { views, view: [...], 'views-file', sweep: [...] }. */
export function selectViews( o ) {

	const given = [];
	if ( o.views ) {

		for ( const v of expandViews( o.views ) ) {

			if ( ! VIEWS[ v ] ) throw new Error( `unknown view ${ v }` );
			given.push( v );

		}

	}
	const defs = [];
	if ( o[ 'views-file' ] ) {

		const data = JSON.parse( readFileSync( o[ 'views-file' ], 'utf8' ) );
		defs.push( ...( Array.isArray( data ) ? data : Object.entries( data ).map( ( [ name, d ] ) => ( { name, ...d } ) ) ) );

	}
	for ( const text of o.view ?? [] ) {

		const colon = text.indexOf( ':' );
		if ( colon < 1 ) throw new Error( `--view ${ JSON.stringify( text ) }: expected "name:query[::js]"` );
		const rest = text.slice( colon + 1 ), cut = rest.indexOf( '::' );
		defs.push( { name: text.slice( 0, colon ), q: cut >= 0 ? rest.slice( 0, cut ) : rest, eval: cut >= 0 ? rest.slice( cut + 2 ) : undefined } );

	}
	for ( const d of defs ) { define( d ); given.push( d.name ); }
	const sweeps = ( o.sweep ?? [] ).map( ( text ) => {

		const m = text.match( /^([^=&]+)=(.+)$/ );
		if ( ! m ) throw new Error( `--sweep ${ JSON.stringify( text ) }: expected "key=a,b,c"` );
		return { key: m[ 1 ], values: m[ 2 ].split( ',' ).filter( ( v ) => v !== '' ) };

	} );
	if ( ! sweeps.length ) return given.length ? given : viewNames();
	// every combination of the sweeps' values, over the given views (or one bare scene)
	let combos = [ [] ];
	for ( const s of sweeps ) combos = combos.flatMap( ( c ) => s.values.map( ( v ) => [ ...c, [ s.key, v ] ] ) );
	const out = [];
	for ( const base of given.length ? given : [ null ] ) for ( const c of combos ) {

		const tag = c.map( ( [ k, v ] ) => `${ k }-${ v }`.replace( /[^A-Za-z0-9_-]/g, '_' ) ).join( '--' );
		const name = base ? `${ base }--${ tag }` : tag;
		const q = c.map( ( [ k, v ] ) => `${ k }=${ v }` ).join( '&' );
		if ( base && VIEWS[ base ].adhoc ) define( { name, q: firstWins( [ ...pairs( q ), ...pairs( VIEWS[ base ].q ) ] ).map( ( [ , p ] ) => p ).join( '&' ), eval: VIEWS[ base ].eval, crops: VIEWS[ base ].crops } );
		else define( { name, q, like: base ?? undefined } );
		out.push( name );

	}
	return out;

}

/** A view's full query on top of the set's `common` one (see the header). */
export function viewQuery( name, common ) {

	const v = VIEWS[ name ];
	if ( ! v.adhoc ) return `${ common }&${ v.q }`;
	const own = new Set( pairs( v.q ).map( ( [ k ] ) => k ) );
	return [ ...pairs( common ).filter( ( [ k ] ) => ! own.has( k ) ).map( ( [ , p ] ) => p ), v.q ].filter( Boolean ).join( '&' );

}

/** What a page must have been loaded with to reach the query by a scene jump. */
export function loadSignature( query ) {

	return firstWins( pairs( query ) ).filter( ( [ k ] ) => ! SCENE_OPTIONS.has( k ) ).map( ( [ , p ] ) => p ).sort().join( '&' );

}

/** The ad-hoc views' definitions among `names`: { name: hash } (empty for the table's views). */
export function viewDefs( names ) {

	return Object.fromEntries( names.filter( ( n ) => VIEWS[ n ].adhoc ).map( ( n ) => [ n, VIEWS[ n ].def ] ) );

}

/** The ad-hoc views among `names` as written to a set's views.json. */
export function describeViews( names, common ) {

	return Object.fromEntries( names.filter( ( n ) => VIEWS[ n ].adhoc ).map( ( n ) => {

		const v = VIEWS[ n ];
		return [ n, { q: v.q, eval: v.eval ?? null, crops: v.crops, like: v.like ?? null, query: viewQuery( n, common ), def: v.def } ];

	} ) );

}
