// Captures a set of paused views in a few warm pages, and checks it against a stored reference.
//
// It answers whether a change moved the paused frame of a fixed set of views: the parity views of
// tools/island/parity/views.mjs, or ad-hoc views given on the command line. Each view's frame is
// the one its own fresh load draws, but most views are reached in place by the exact scene jump,
// so a set costs a fraction of a page load per view. Use it instead of a script that loops
// tools/kit/capture.mjs over views, which loads a cold page per view. For one view, use
// tools/kit/capture.mjs; for frames in motion, tools/kit/parity/capture-clip.mjs. How to judge a
// set (byte-exact or within a tolerance) and when to recapture a reference is method:
// tools/kit/parity/README.md.
//
// Usage:
//   node tools/kit/parity/capture-set.mjs <baseUrl> <outDir> [options] [-- capture.mjs options]
//   node tools/kit/parity/capture-set.mjs <refUrl> --reference <root>/<sha>/<set> [options]
//   node tools/kit/parity/capture-set.mjs <testUrl> <testDir> --check <root>/<sha>/<set> [--ref-url <u>] [options]
//   node tools/kit/parity/capture-set.mjs <url> <outDir> --quick --views a,b [quick options]
//   node tools/kit/parity/capture-set.mjs <url> out --views island-fair-noon --sweep weather=fair,rain --sweep hour=7,17
//
// Options:
//   --views <group|a,b>      views.mjs's groups (parity, taa, all, normal, special) or names; default parity
//   --view "name:query[::js]"  an ad-hoc view, repeatable (tools/kit/parity/adhoc.mjs)
//   --views-file <f.json>    ad-hoc views from a file
//   --sweep "key=a,b,c"      each selected view once per value, repeatable (every combination)
//   --quality <tier>         default high
//   --aa <mode>              adds `aa=<mode>`; none by default (a build without the option loads unchanged)
//   --q "<extra&query>"      added to every view's query (such as msaa=1)
//   --size <W>x<H>[@<dpr>]   CSS size and device pixel ratio; default 1728x1117@2
//   --repeat                 capture each view twice and compare the two
//   --jobs <n>               pages at once; default one per 5 captures (repeats counted), at most 2
//   --retries <n>            fresh-page retries of a failed view or a failed page start; default 1
//   --timeout <ms>           capture.mjs's waits for the page; default 240000
//   --page-per-view          a fresh page for every view
//   --reference <dir>        make the set a stored reference of a commit (tools/kit/reference.mjs)
//   --check <dir>            capture the test side and diff it against that stored reference
//   --ref-url <url>          with --check: where to capture the reference when it is missing or stale
//   --diff "<options>"       with --check: diff-set.py's options instead of the default
//   --recapture-reference    capture the reference afresh (one suspected bad)
//   --runs <file.csv>        append a line to a run log; default the PARITY_RUNS variable, else none
//   --task, --procedure, --sha  the run log's fields
//   --quick                  the tuning mode (tools/kit/parity/quick.mjs, whose header lists its options)
//   -- <options>             passed to every capture.mjs page (tools/kit/capture.mjs)
//
// Every view is the frame of `<baseUrl>?debug&paused&t=12&quality=<tier>[&aa=<mode>][&<extra>]&<the
// view's query>` (the page reads the first of a repeated option), captured after the view's eval
// once capture.mjs finds it ready. The default size renders 2K's pixel count on high (2388x1543), which
// FSR 1 brings to 3456x2234. A temporal mode (taa, taau, fsr3, and the ground truth gt: 64 jitters
// at the canvas's resolution) is captured once its paused rest has converged.
//
// One page per set: only load-time options (tier, aa, size and ratio, the debug switches in --q)
// need a new page. Each page loads the first view it is given and reaches each later one by
// `__PONLAI__.establishScene( <the view's full query> )`, whose contract
// (packages/render/assemble/SceneJump.ts) is that the frame equals a fresh load's, byte for byte
// wherever the fresh load is deterministic. The jump resets the
// scene's options and camera, not what an eval changed besides (GUI settings, Tune, debug flags, a
// uniform). So after a view with an eval of its own (an ad-hoc view's js; adhoc.mjs `jumpClean`) the
// page ends and the next view loads fresh, and an `--eval` or `--eval-ready` after `--` takes a page
// per view, unless `--ready-reads` declares the `--eval-ready` read-only. A build without the jump's
// marker (`__PONLAI__.sceneJump`) gets a page per view, said once. Views are handed out one at a
// time from one queue. Ad-hoc views whose load-time options differ are captured in groups of pages.
//
// Ad-hoc views follow adhoc.mjs's rules: `like=<view>` in a query starts from a views.mjs view; an
// ad-hoc view's own options win over the set's (t, tier, aa, --q); names are letters, digits, `-`
// and `_`, none of views.mjs's; a sweep names its views `<view>--<key>-<value>` (`<key>-<value>`, a
// bare scene, with no views given). With `--views` too, both sets are captured; with ad-hoc views
// only, the parity group is not.
//
// --repeat runs the repeats after all the first captures, on other pages and in reverse order, so a
// view's repeat is reached from another scene than its first capture: the A/A also shows anything a
// jump left behind. The two are compared by decoded pixels (PNG bytes depend on the encoder). The
// frames do not depend on --jobs. A view that differs only by sparse single-frame speckle on the
// vegetation while other pages render is GPU noise: capture it again.
//
// --reference reuses the stored set when its manifest matches and covers the views, waits for a
// live process capturing it, or captures what is missing; the store's rules and layout are in
// tools/kit/reference.mjs. Repeats that differ are recorded in the manifest, not an error. --check
// does the same for the reference (captured only from --ref-url), captures the test side with the
// same options and runs `uv run tools/kit/diff-set.py <ref> <testDir> --views <the views>` with
// --diff's options, by default `--temporal` for taa, taau, fsr3 and gt, else `--aa-floor --held`
// with --repeat, else `--max-px1 0 --held` (--held: the region a held effect reaches, which each
// capture records in its JSON as `held`, is compared within tools/kit/tolerance.py's HELD).
//
// A failed view (a timeout, a page or console error, a crash) ends its page; it is retried on a
// fresh page and the views still to come go on there. A page silent for three times --timeout plus
// two minutes after its view holds the GPU slot is killed and treated the same way. A page that
// ends before it asks for a view is started again; more than --retries such starts in a row end
// the run.
//
// Output: `<outDir>/<view>.png`, `<view>.json` (diagnostics, crop anchors, `ready`, and `page`: the
// visit, load or jump, the view before, the GPU lock wait, the page's memory) and `<view>.log`
// (capture.mjs's output, its last attempt); `<view>.repeat.*` with --repeat; `views.json` with the
// ad-hoc views' definitions. Stdout: a line per capture, a `-- reproducibility` list (identical or
// DIFFERS), a summary line; with --check the diff table, also in `<testDir>/diff.txt`, the diff's
// crops in `<testDir>/diff/` and a PASS or FAIL line. The run log gets a line per run and one per
// reference (date, task, procedure, sha, wall minutes, whether the GPU was busy, captures, failures,
// the set).
//
// GPU lock and browser: each page is a headless capture.mjs worker in its own profile slot
// (tools/kit/capture-browser.mjs) and leases a shared slot of the GPU lock from view to view
// (capture.mjs `--lease`, after `--`), so --jobs above the free slots only queues
// (docs/reference/gpu-lock.md). The set runs with the application scheduling policy
// (tools/kit/qos.mjs).
//
// Cost: a page load, then a jump per view, a fraction of a load. A second page earns its load on
// sets of more than about 5 captures.
//
// Exit codes: 0 every capture done (and, with --repeat, every pair identical; with --check, the
// diff passed); 1 a capture failed, a repeat differs, more than --retries page starts failed in a
// row, or --reference could not make the reference; with --check, diff-set.py's code, or 1 when a
// test capture failed; 2 usage, an unknown option, a bad --size or view, or with --check a
// reference that could not be made or needs --ref-url, or a server of another commit.
// --quick returns quick.mjs's code, 2 on its usage errors.
import '../help.mjs';
// The set and its capture.mjs workers run with the application scheduling policy on macOS; a worker
// is started with it, since re-executing would close its IPC channel.
import { appPolicy } from '../qos.mjs'; // first: the set and its workers run with the application policy
import { spawn } from 'node:child_process';
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { viewTable } from './tables.mjs';
import { selectViews, viewQuery, loadSignature, viewDefs, describeViews, jumpClean } from './adhoc.mjs';
import { gpuUtilisation } from '../gpu-lock.mjs';
import { samePixels } from '../capture-browser.mjs';
import { ensureReference, setKey, resolveRefDir, checkServed } from '../reference.mjs';
import PROJECT from '../../project.mjs';
import { ROOT } from '../root.mjs';

const REPO = ROOT;
const { viewEval } = viewTable;
// --quick: setting variants on a few views, against the stored references (quick.mjs)
if ( process.argv.includes( '--quick' ) ) {

	const { main } = await import( './quick.mjs' );
	process.exit( await main( process.argv.slice( 2 ) ).catch( ( error ) => { console.error( error.message ?? error ); return 2; } ) );

}
const argv = process.argv.slice( 2 );
const dash = argv.indexOf( '--' );
const passThrough = dash >= 0 ? argv.slice( dash + 1 ) : [];
const args = dash >= 0 ? argv.slice( 0, dash ) : argv;
const flags = new Set( [ 'repeat', 'recapture-reference', 'page-per-view' ] );
const lists = new Set( [ 'view', 'sweep' ] );
const options = { view: [], sweep: [] }, positional = [];
for ( let i = 0; i < args.length; i ++ ) {

	if ( args[ i ].startsWith( '--' ) ) {

		const k = args[ i ].slice( 2 );
		if ( flags.has( k ) ) options[ k ] = true;
		else if ( lists.has( k ) ) options[ k ].push( args[ ++ i ] );
		else options[ k ] = args[ ++ i ];

	} else positional.push( args[ i ] );

}
const known = [ 'repeat', 'quality', 'aa', 'q', 'views', 'view', 'views-file', 'sweep', 'jobs', 'retries', 'timeout', 'size', 'runs', 'task', 'procedure', 'sha',
	'reference', 'check', 'ref-url', 'diff', 'recapture-reference', 'page-per-view' ];
const unknown = Object.keys( options ).filter( ( k ) => ! known.includes( k ) );
const mode = options.reference ? 'reference' : options.check ? 'check' : 'plain';
const positionals = mode === 'reference' ? 1 : 2;
if ( positional.length !== positionals || unknown.length || ( options.reference && options.check ) ) {

	console.error( unknown.length ? `unknown option --${ unknown[ 0 ] }` : 'usage: capture-set.mjs <baseUrl> <outDir> | <refUrl> --reference <dir> | <testUrl> <testDir> --check <dir> [options] (see the file header)' );
	process.exit( 2 );

}
const slash = ( u ) => ! u.endsWith( '/' ) && ! u.includes( '?' ) ? u + '/' : u;
const base = slash( positional[ 0 ] );
const outDir = mode === 'reference' ? null : path.resolve( positional[ 1 ] );
const sizeText = options.size ?? '1728x1117@2';
const size = sizeText.match( /^(\d+)x(\d+)(?:@([\d.]+))?$/ );
if ( ! size ) { console.error( `bad --size ${ options.size }` ); process.exit( 2 ); }
const [ , W, H, scale = '1' ] = size;
// views.mjs's views and ad-hoc ones (adhoc.mjs)
let views;
try { views = selectViews( options ); } catch ( error ) { console.error( error.message ); process.exit( 2 ); }
const quality = options.quality ?? 'high';
let common = `debug&paused&t=12&quality=${ quality }`;
if ( options.aa ) common += `&aa=${ options.aa }`;
if ( options.q ) common += `&${ options.q }`;
const captures = views.length * ( options.repeat ? 2 : 1 );
// Captures are GPU-bound: 2 pages at once give about 1.75 times one page's throughput, 3 about 2
// times with the slowest views taking twice as long as at 2, 4 hardly more; and every extra page
// costs a load (a jump costs about half of one), which a small set does not earn back. So the
// default is one page per 5 captures, at most 2, the GPU lock's shared slots: a third page waits for
// a slot unless the set runs under an exclusive hold (6 captures or more take 2, up to 5 one).
const jobs = Math.max( 1, Number( options.jobs ?? Math.min( 2, Math.ceil( captures / 5 ) ) ) );
const retries = Math.max( 0, Number( options.retries ?? 1 ) );
// capture.mjs's waits for the page: generous, since other agents' pages may share the GPU
const timeout = Number( options.timeout ?? 240000 );
const procedure = options.procedure ?? `capture-set${ options.repeat ? ' --repeat' : '' }`;

function run( cmd, argList, log ) {

	return new Promise( ( resolve ) => {

		const child = spawn( cmd, argList, { cwd: REPO, stdio: [ 'ignore', 'pipe', 'pipe' ] } );
		const chunks = [];
		child.stdout.on( 'data', ( d ) => chunks.push( d ) );
		child.stderr.on( 'data', ( d ) => chunks.push( d ) );
		child.on( 'close', async ( code ) => { if ( log ) await writeFile( log, Buffer.concat( chunks ) ); resolve( { code: code ?? 1, out: Buffer.concat( chunks ).toString() } ); } );

	} );

}

let jumpless = false;
/** Captures [ view, suffix ] pairs from `url` into `dir`; resolves to the pairs that succeeded.
 * One page per worker (capture.mjs --worker, see the header): a page serves first captures or
 * repeats, never both, so a view's repeat comes from another page load than its first capture. */
async function captureList( url, dir, list ) {

	await mkdir( dir, { recursive: true } );
	// views whose load-time options differ cannot share a page: a group of pages per set of them, in
	// the order given (views.mjs's views all share the set's, so they make one group; adhoc.mjs)
	const groups = new Map();
	for ( const pair of list ) {

		const sig = loadSignature( viewQuery( pair[ 0 ], common ) );
		if ( ! groups.has( sig ) ) groups.set( sig, [] );
		groups.get( sig ).push( pair );

	}
	if ( groups.size > 1 ) console.log( `${ groups.size } groups of views by load-time options: ${ [ ...groups.keys() ].join( ' | ' ) }` );
	const done = [];
	for ( const group of groups.values() ) done.push( ...await captureGroup( url, dir, group ) );
	return done;

}

/** captureList's views of one set of load-time options. */
async function captureGroup( url, dir, list ) {

	// every first capture before every repeat, the repeats in reverse order (see the header)
	const queue = [ ...list.filter( ( [ , s ] ) => ! s ), ...list.filter( ( [ , s ] ) => s ).reverse() ]
		.map( ( [ view, suffix ] ) => ( { view, suffix, key: `${ view }${ suffix }`, attempts: 0 } ) );
	const done = [];
	const fail = ( item, why ) => {

		item.attempts ++;
		if ( item.attempts <= retries ) {

			// first in line, so a fresh page loads it
			queue.unshift( item );
			console.log( `${ item.key } ${ why }: retrying on a fresh page` );

		} else console.log( `${ item.key } rc=1 FAILED (${ why })` );

	};
	// pages in a row that ended before asking for a view (see the header), and why the last one did
	let failedStarts = 0, startError = null;
	// a page: one capture.mjs worker, serving items of one kind (first or repeat) until none is left
	const runPage = ( repeat ) => new Promise( ( resolve ) => {

		// started with the application policy: an exec would close its IPC channel (tools/kit/qos.mjs)
		const child = spawn( ...appPolicy( process.execPath, [ 'tools/kit/capture.mjs', '--worker', '--url', url, '--w', W, '--h', H, '--scale', scale,
			'--timeout', String( timeout ), ...( options[ 'page-per-view' ] ? [ '--page-per-view' ] : [] ), ...passThrough ] ),
		{ cwd: REPO, stdio: [ 'ignore', 'pipe', 'pipe', 'ipc' ] } );
		let output = [], current = null, timer = null, killer = null, writing = Promise.resolve(), asked = false;
		const stderr = [];
		child.stdout.on( 'data', ( d ) => output.push( d ) );
		child.stderr.on( 'data', ( d ) => { output.push( d ); stderr.push( d ); } );
		child.on( 'error', ( error ) => output.push( Buffer.from( `capture.mjs: ${ error.message }\n` ) ) );
		// (the page's close waits for the logs: the set's results are read once every page is done)
		const flushLog = ( item, text ) => {

			const log = Buffer.concat( [ ...output, Buffer.from( text + '\n' ) ] );
			output = [];
			writing = writing.then( () => writeFile( path.join( dir, `${ item.key }.log` ), log ) );

		};
		child.on( 'message', ( m ) => {

			if ( m.type === 'next' ) {

				// (the page started: the run of failed starts is over)
				if ( ! asked ) { asked = true; failedStarts = 0; }
				const i = queue.findIndex( ( item ) => !! item.suffix === repeat );
				current = i >= 0 ? queue.splice( i, 1 )[ 0 ] : null;
				if ( ! current ) { child.send( { type: 'done' }, () => {} ); return; }
				const item = current;
				// (a page that died meanwhile never got it: back in line, no attempt counted)
				// (`last`: an eval the jump does not undo, so the page ends after this view: see the header)
				child.send( { type: 'view', view: { key: item.key, query: viewQuery( item.view, common ), evals: [ viewEval( item.view ) ], last: ! jumpClean( item.view ), out: path.join( dir, item.key ) } },
					( error ) => { if ( error && current === item ) { current = null; queue.unshift( item ); } } );

			} else if ( m.type === 'locked' ) {

				// a page that stopped answering (capture.mjs gives up on a view after 3 x timeout + 1 min),
				// counted from the view's GPU slot: the wait for the lock is not silence (exclusive timing
				// runs and normal-priority work may hold it far longer than any timeout)
				timer = setTimeout( () => {

					console.log( `${ current?.key } no answer from its page: killing it` );
					child.kill( 'SIGTERM' );
					killer = setTimeout( () => child.kill( 'SIGKILL' ), 15000 );

				}, timeout * 3 + 120000 );

			} else if ( m.type === 'page' ) {

				if ( ! m.jump && ! jumpless ) {

					jumpless = true;
					console.log( `this build has no exact scene jump (${ PROJECT.page.handle }.sceneJump ${ m.marker ?? 'missing' }): a page per view` );

				}

			} else if ( m.type === 'shot' ) {

				clearTimeout( timer );
				const item = current;
				current = null;
				flushLog( item, m.log );
				if ( m.ok ) {

					done.push( [ item.view, item.suffix ] );
					console.log( `${ item.key } rc=0 ${ m.mode }${ m.from ? ` from ${ m.from }` : '' } ${ m.seconds.toFixed( 1 ) } s${ m.worldSeconds >= 0.5 ? ` (+${ m.worldSeconds.toFixed( 1 ) } s loading its world)` : '' }${ m.lockSeconds >= 0.5 ? ` (+${ m.lockSeconds.toFixed( 1 ) } s waiting for the GPU lock)` : '' }` );

				} else fail( item, 'failed' );

			}

		} );
		child.on( 'close', async ( code, signal ) => {

			clearTimeout( timer );
			clearTimeout( killer );
			if ( current ) {

				// the page died with a view in hand: that view's attempt failed, the rest go on elsewhere
				const item = current;
				current = null;
				flushLog( item, `the page was lost (exit ${ code ?? signal })` );
				fail( item, `page lost (exit ${ code ?? signal })` );

			} else if ( ! asked ) {

				// a failed start: no view was handed out, so none is charged; the group stops after too many
				failedStarts ++;
				const tail = Buffer.concat( stderr ).toString().split( '\n' ).filter( ( l ) => l.trim() ).slice( - 15 ).join( '\n' );
				startError = `capture.mjs exited ${ code ?? signal } before asking for a view${ tail ? `; its stderr ends:\n${ tail }` : ' (no stderr)' }`;
				if ( failedStarts <= retries ) console.log( `a page failed to start (exit ${ code ?? signal }): starting another` );

			}
			await writing;
			resolve();

		} );

	} );
	// (never more workers than views, so a view's repeat cannot start beside its first capture)
	const distinct = new Set( list.map( ( [ v ] ) => v ) ).size;
	await Promise.all( Array.from( { length: Math.min( jobs, distinct ) }, async () => {

		while ( queue.length && failedStarts <= retries ) await runPage( !! queue[ 0 ].suffix );

	} ) );
	if ( failedStarts > retries ) {

		// (the other pages have closed; a worker also exits when this process goes)
		console.error( `capture-set: ${ failedStarts } pages in a row failed to start, giving up. ${ startError }` );
		process.exit( 1 );

	}
	return done;

}

/** Pixel comparison of each view's two captures (decoded PNGs): { view: true | false | null (no pair) }. */
async function repeatability( dir, list, quiet ) {

	const out = {};
	if ( ! quiet ) console.log( '-- reproducibility' );
	for ( const v of list ) {

		const a = path.join( dir, `${ v }.png` ), b = path.join( dir, `${ v }.repeat.png` );
		out[ v ] = existsSync( a ) && existsSync( b ) ? samePixels( await readFile( a ), await readFile( b ) ) : null;
		if ( ! quiet ) console.log( `${ v } ${ out[ v ] ? 'identical' : 'DIFFERS' }` );

	}
	return out;

}

async function logRun( started, busy, proc, sha, captured, failed, out ) {

	const runs = options.runs ?? process.env.PARITY_RUNS;
	if ( ! runs ) return;
	const minutes = ( Date.now() - started ) / 60000;
	const header = 'date,task,procedure,sha,wall_min,contended,captures,failed,out\n';
	const field = ( s ) => /[",\n]/.test( s ) ? `"${ s.replace( /"/g, '""' ) }"` : s;
	const line = [ new Date( started ).toISOString(), options.task ?? '', proc, sha ?? '', minutes.toFixed( 2 ), busy === null ? '?' : busy > 15 ? 'y' : 'n',
		String( captured ), String( failed ), out ].map( field ).join( ',' ) + '\n';
	await mkdir( path.dirname( path.resolve( runs ) ), { recursive: true } );
	await appendFile( runs, existsSync( runs ) ? line : header + line );

}

// ------------------------------------------------------------------ the reference (--reference, --check)

let refCaptured = 0, served = null;
const defs = viewDefs( views );
async function reference( url, { dryRun = false } = {} ) {

	const { dir } = resolveRefDir( options.reference ?? options.check );
	// (how the views are reached is part of the key: a check of the jump needs fresh loads)
	const { key, files } = await setKey( { quality, aa: options.aa, q: options.q, size: sizeText, capture: passThrough, pages: options[ 'page-per-view' ] ? 'view' : 'jump' } );
	return ensureReference( {
		dir, kind: 'set', key, files, want: { views, repeat: !! options.repeat, defs }, url, siblings: true, dryRun,
		// (a check's test side, captured meanwhile, takes the wrapper's slot over itself)
		yieldGpu: mode === 'reference',
		force: options[ 'recapture-reference' ], label: `capture-set ${ options.task ?? '' } ${ procedure }`.trim(),
		capture: async ( todo, into ) => {

			// (also when this process takes over a capture that another one left unfinished)
			served ??= checkServed( url, resolveRefDir( into ).sha );
			const done = await captureList( url, into, todo );
			refCaptured += todo.length;
			const firsts = new Set( done.filter( ( [ , s ] ) => ! s ).map( ( [ v ] ) => v ) );
			const repeats = new Set( done.filter( ( [ , s ] ) => s ).map( ( [ v ] ) => v ) );
			const same = await repeatability( into, [ ...repeats ], true );
			const coverage = { views: {} };
			// (an ad-hoc view's definition goes with it: reference.mjs SET)
			const def = ( v ) => defs[ v ] ? { def: defs[ v ] } : {};
			for ( const v of firsts ) coverage.views[ v ] = { captured: true, ...def( v ) };
			for ( const v of repeats ) if ( firsts.has( v ) || existsSync( path.join( into, `${ v }.png` ) ) ) coverage.views[ v ] = { ...coverage.views[ v ], captured: true, repeat: true, identical: same[ v ], ...def( v ) };
			// a build without the exact jump got a page per view whatever was asked: stored as such
			return { coverage, failed: todo.length - done.length, captured: done.length, ...( jumpless ? { manifest: { key: { ...key, pages: 'view' }, jumpless: true } } : {} ) };

		},
	} );

}

/** The ad-hoc views' definitions beside their captures (`views.json`), when there are any. */
async function writeViews( dir ) {

	const adhoc = describeViews( views, common );
	if ( ! Object.keys( adhoc ).length ) return;
	await mkdir( dir, { recursive: true } );
	await writeFile( path.join( dir, 'views.json' ), JSON.stringify( adhoc, null, '\t' ) + '\n' );

}

const busy = await gpuUtilisation();
const started = Date.now();

if ( mode === 'plain' ) {

	const list = [ ...views.map( ( v ) => [ v, '' ] ), ...( options.repeat ? views.map( ( v ) => [ v, '.repeat' ] ) : [] ) ];
	await writeViews( outDir );
	const done = await captureList( base, outDir, list );
	const failed = list.length - done.length;
	const same = options.repeat ? await repeatability( outDir, views ) : {};
	const differ = Object.values( same ).filter( ( s ) => ! s ).length;
	const minutes = ( Date.now() - started ) / 60000;
	console.log( `${ views.length } views${ options.repeat ? ' x2' : '' } in ${ minutes.toFixed( 1 ) } min${ failed ? `, ${ failed } failed` : '' }` );
	await logRun( started, busy, procedure, options.sha, list.length, failed, outDir );
	process.exitCode = failed || differ ? 1 : 0;

} else {

	// the URL to capture the reference from: the positional one for --reference, --ref-url for --check
	const refUrl = mode === 'reference' ? base : options[ 'ref-url' ] ? slash( options[ 'ref-url' ] ) : null;
	const { sha } = resolveRefDir( options.reference ?? options.check );
	const probe = await reference( refUrl, { dryRun: true } );
	if ( probe.status === 'unavailable' ) {

		console.error( `reference ${ probe.rel } of ${ sha.slice( 0, 12 ) } must be captured (${ probe.reason }): pass --ref-url <a preview of ${ sha.slice( 0, 12 ) }>` );
		process.exit( 2 );

	}
	if ( probe.status === 'capture' ) {

		try { served = checkServed( refUrl, sha ); } catch ( error ) { console.error( `reference: ${ error.message }` ); process.exit( 2 ); }

	}
	const refStarted = Date.now();
	const refRun = async () => {

		const r = await reference( refUrl );
		await logRun( refStarted, busy, `${ procedure } ref ${ r.status }`, sha.slice( 0, 7 ), refCaptured, r.missing?.length ?? 0, r.dir );
		return r;

	};
	const testRun = async () => {

		if ( mode !== 'check' ) return null;
		const list = [ ...views.map( ( v ) => [ v, '' ] ), ...( options.repeat ? views.map( ( v ) => [ v, '.repeat' ] ) : [] ) ];
		const t0 = Date.now();
		await writeViews( outDir );
		const done = await captureList( base, outDir, list );
		if ( options.repeat ) await repeatability( outDir, views );
		await logRun( t0, busy, procedure, options.sha, list.length, list.length - done.length, outDir );
		return { failed: list.length - done.length };

	};
	// a reference to capture first (the GPU does one set at a time better); one stored or being
	// captured elsewhere beside the test side
	let ref, test;
	if ( probe.status === 'capture' ) { ref = await refRun(); test = await testRun(); } else [ ref, test ] = await Promise.all( [ refRun(), testRun() ] );
	if ( ! [ 'reused', 'waited', 'captured', 'extended' ].includes( ref.status ) ) {

		console.error( `reference ${ ref.rel }: ${ ref.status }${ ref.reason ? ` (${ ref.reason })` : '' }${ ref.error ? `: ${ ref.error }` : '' }${ ref.missing?.length ? `; missing ${ ref.missing.map( ( p ) => p.join( '' ) ).join( ', ' ) }` : '' }` );
		process.exit( mode === 'check' ? 2 : 1 );

	}
	console.log( `reference ${ path.relative( path.dirname( resolveRefDir( ref.dir ).shaDir ), ref.dir ) }: ${ ref.status }${ ref.waitedMinutes ? ` after ${ ref.waitedMinutes.toFixed( 1 ) } min of waiting` : '' }` );
	if ( mode === 'reference' ) {

		console.log( `${ ( ( Date.now() - started ) / 60000 ).toFixed( 1 ) } min` );
		process.exitCode = 0;

	} else {

		const temporal = [ 'taa', 'taau', 'fsr3', 'gt' ].includes( options.aa );
		const diffArgs = options.diff ? options.diff.split( /\s+/ ).filter( Boolean ) : temporal ? [ '--temporal' ] : options.repeat ? [ '--aa-floor', '--held' ] : [ '--max-px1', '0', '--held' ];
		// (not `python -I`: diff-set.py imports tolerance.py from its own directory)
		const { code, out } = await run( 'uv', [ 'run', '--with', 'numpy', '--with', 'pillow', 'python', path.join( REPO, 'tools/kit/diff-set.py' ),
			ref.dir, outDir, '--views', views.join( ',' ), '--out', path.join( outDir, 'diff' ), ...diffArgs ], path.join( outDir, 'diff.txt' ) );
		process.stdout.write( out );
		console.log( `check ${ code === 0 ? 'PASS' : 'FAIL' } against ${ ref.dir } (diff-set ${ diffArgs.join( ' ' ) }), ${ ( ( Date.now() - started ) / 60000 ).toFixed( 1 ) } min${ test.failed ? `; ${ test.failed } test captures failed` : '' }` );
		process.exitCode = test.failed ? 1 : code;

	}

}
