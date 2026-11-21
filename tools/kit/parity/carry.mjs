// Decides whether a commit can take over an earlier commit's MSAA baseline and ground truth uncaptured.
//
// A commit B takes the MSAA baseline and the aa=gt ground truth of an earlier commit A instead of
// capturing them again, when nothing between them can have changed those captures. A baseline costs
// the gates' clips (64 stepped frames each) and the ground truth 64 jittered renders per rest view
// and 32 per clip frame: most of a reference round, spent again on every commit whose diff only
// touched tools, docs or code that draws nothing. tools/kit/parity/baseline.mjs `--carry-from A`
// applies it; `check` prints the verdict alone.
//
// Usage:
//   node tools/kit/parity/carry.mjs check --from <A> --to <B> [options]
//   node tools/kit/parity/baseline.mjs --url <preview of B> --sha <B> --carry-from <A> [--carry-url <preview of A>]
//
// Options:
//   check
//     --from <A>, --to <B>   the two commits (any form git resolves)
//     --url <url>            a preview of B, where the paused sets and spot checks are captured
//     --carry-url <url>      a preview of A, for what A's store lacks for the checks
//     --root <dir>           the store's root; default artifacts/parity (`output.parity`)
//     --tiers <a,b>          default high
//     --views, --clips, --only  baseline.mjs's selection of entries
//     --all                  evaluate every condition and carry nothing
//     --recheck              run the spot checks again instead of reusing an earlier verdict
//     --jobs <n>             pages at once for the paused sets
//
// What could change a stored capture, and how each is ruled out (cheapest first; baseline.mjs
// stops at the first that fails, `check --all` evaluates every one):
//   1. `sameBuild`: `git diff A B` touches nothing the build reads (the configuration's
//      `reference.buildInputs`): the two builds are the same, so everything carries with no
//      further check;
//   2. key: an entry carries only when A's manifest key equals the key B asks for (tool hash,
//      browser build, tier, aa, size, query, clip, view, sampling, anchors); the checks use the
//      gates' size (1728x1117@2) with no extra query only;
//   3. motion (wind, the islanders' walk, the sea, lightning, anything under a running clock, the
//      stepper) is invisible to a paused frame, so `motionWatch`, a watch list (the configuration's
//      `reference.watch.motion`) on the diff, plus `spot` for the clips: the first SPOT.frames
//      frames of SPOT.clips (the clips table's CARRY_SPOT) under MSAA captured on B with A's
//      anchors and compared pixel for pixel with A's stored baseline entries (a differing clip is
//      captured once more, since a whole-frame outlier under a shared GPU does not recur);
//   4. the paused scene, `paused-<tier>`: B's paused set (S1 on high: the views table's default group
//      plus every view an entry to carry is on; S2/<tier> on lower tiers: those views) matches A's
//      with tools/kit/diff-set.py (`--aa-floor` where both sides hold repeats, else pixel identity),
//      captured or extended through the store (A's from --carry-url when its store lacks it). Since
//      that vouches for A's set, not A's entries, a rest baseline entry with a whole frame carries
//      only when that frame matches A's own paused frame of its view within STORED_TOLERANCE
//      (`stored`, files only: a capture that went wrong at A would otherwise carry on);
//   5. the ground truth's own path (aa=gt only): `gtWatch`, a watch list (`reference.watch.gt`) on
//      the diff, plus `gtSpot`, one rest ground truth (the first of SPOT.gtViews A holds) captured on
//      B and compared with A's within the ground truth's noise (GT_TOLERANCE): a temporal-only
//      change outside the watch list leaves MSAA alone and shows only there.
// The baseline carries on 1-4 (rest entries need no spot check: one paused frame), the ground
// truth on 1-5. GT-HI, the lower tiers' clips and the test labels never carry. The metrics are
// not captures: baseline.mjs recomputes them on every run, so a change to tools/kit/clip-metrics.py
// or the thresholds invalidates nothing here. A browser build verified for a paused set
// (tools/kit/reference.mjs, `verifiedBrowsers`) counts as the set's own.
//
// What the rule cannot see: a change that alters motion only, outside the watch list, in a view or
// clip the spot check does not cover; a temporal-only change visible in the ground truth of views
// other than the one spot-checked; the lower tiers' ground truth (checked on high only); and a rest
// baseline entry without a whole frame that was captured wrong at A.
//
// A carried entry is a real directory at B's path holding clones of A's files (copy-on-write on
// APFS, a plain copy elsewhere; never hard links, which a later write would change on both sides)
// and its own manifest: A's key (so later requests reuse it) and `carriedFrom` (A's sha, its entry
// path, the origin when A's own entry was carried, the time, the verdict file).
//
// Output: `check` prints a line per entry and side, `carry ...` or `keep  ...: <the condition that
// failed>`, after the checks' own log lines. The checks' captures, diffs and the verdict stay under
// `<root>/<B>/carry/<A>/` (verdict.json); the spot checks' results are reused by later runs with
// the same inputs (the clip tools, the browser, A's entries' last capture).
//
// GPU lock and browser: the paused sets go through tools/kit/parity/capture-set.mjs and the spot
// checks through tools/kit/parity/capture-clip.mjs, each taking a shared slot of the GPU lock and
// headless pages (docs/reference/gpu-lock.md). With `sameBuild`, or a verdict reused, none.
//
// Cost: from a git diff (under a second) to a paused set and a few short clips captured.
//
// Exit codes: 0 the verdict printed; 2 usage (no `check`, --from or --to, an unknown argument) or an
// error.
import '../help.mjs';
import { execFileSync, spawn } from 'node:child_process';
import { constants, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BUILD_INPUTS, describeKeyChange, keyMatches, setKey, stable, toolVersion, browserBuild, checkServed } from '../reference.mjs';
import { decodePng, compareFrames } from '../capture-browser.mjs';
import { viewTable, clipTable } from './tables.mjs';
import PROJECT from '../../project.mjs';
import { ROOT, outputDir } from '../root.mjs';

const HERE = path.dirname( fileURLToPath( import.meta.url ) );
const REPO = ROOT, TOOLS = path.join( ROOT, 'tools' );
const { viewNames } = viewTable;

// the build inputs whose change can alter motion (`motion`) or the ground truth (`gt`) without
// touching a paused frame (an entry ending in '/' is a folder)
const WATCH = PROJECT.reference.watch;
// the spot checks' clips and ground-truth views (the clips table's), each clip's first 16 frames
export const SPOT = { ...clipTable.CARRY_SPOT, frames: 16 };
// the ground truth's noise between two captures (measured: crops 1-2 levels, max 4; whole frames
// max 7 on a few pixels): per image, the largest channel difference and the share of pixels over 2 levels
export const GT_TOLERANCE = { max: 8, over2Share: 0.0005 };
// a rest entry's whole frame against the paused set's frame of its view (capture-clip's readback
// against capture-set's screenshot: 1 level on most pixels when both are good)
export const STORED_TOLERANCE = { max: 2 };
// the paused sets are taken at the gates' size and query only (another size would be another set)
const SIZE = '1728x1117@2';
const pausedSet = ( tier ) => tier === 'high' ? 'S1' : path.join( 'S2', tier );

const readJson = ( file ) => {

	try { return JSON.parse( readFileSync( file, 'utf8' ) ); } catch { return null; }

};
const git = ( args ) => execFileSync( 'git', args, { cwd: REPO, encoding: 'utf8', timeout: 30000, stdio: [ 'ignore', 'pipe', 'ignore' ] } ).trim();
const fullSha = ( sha ) => git( [ 'rev-parse', '--verify', '--quiet', `${ sha }^{commit}` ] );

/** The paths of `list` that `names` touch (an entry ending in '/' is a directory). */
export const watchHits = ( names, list ) => names.filter( ( n ) => list.some( ( w ) => w.endsWith( '/' ) ? n.startsWith( w ) : n === w ) );

function run( cmd, args, log ) {

	return new Promise( ( resolve ) => {

		const child = spawn( cmd, args, { cwd: REPO, stdio: [ 'ignore', 'pipe', 'pipe' ], env: process.env } );
		const chunks = [];
		child.stdout.on( 'data', ( d ) => chunks.push( d ) );
		child.stderr.on( 'data', ( d ) => chunks.push( d ) );
		child.on( 'close', ( code ) => {

			const out = Buffer.concat( chunks ).toString();
			if ( log ) { mkdirSync( path.dirname( log ), { recursive: true } ); writeFileSync( log, out ); }
			resolve( { code: code ?? 1, out } );

		} );

	} );

}

// pixel comparison of two PNG files (equal bytes are equal pixels)
function comparePng( a, b ) {

	const x = readFileSync( a ), y = readFileSync( b );
	if ( x.equals( y ) ) return { same: true, max: 0, over2: 0, pixels: 0 };
	const p = decodePng( x ), q = decodePng( y );
	const c = compareFrames( p, q );
	return { same: c.max === 0, max: c.max, over2: c.over[ 2 ], pixels: p.width * p.height };

}

// the images of a capture-clip folder: { '<crop>/<i>.png': path }, frames below `frames` only
function clipImages( dir, frames = Infinity ) {

	const out = {};
	for ( const d of readdirSync( dir, { withFileTypes: true } ) ) {

		if ( ! d.isDirectory() || d.name === 'metrics' ) continue;
		for ( const f of readdirSync( path.join( dir, d.name ) ) ) {

			const m = f.match( /^(\d+)\.png$/ );
			if ( m && Number( m[ 1 ] ) < frames ) out[ `${ d.name }/${ f }` ] = path.join( dir, d.name, f );

		}

	}
	return out;

}

const short = ( sha ) => sha.slice( 0, 12 );

/**
 * Decides what B may take over from A for a baseline run's reference entries.
 *
 * options: {
 *   from, to,        the two commits (any form git resolves)
 *   storeRoot,       the reference store's root (<root>/<sha>/...)
 *   targets,         [ { side: 'baseline'|'gt'|..., e } ] the run's reference entries (baseline.mjs entries())
 *   url,             a preview of B (its paused sets and spot checks are captured there)
 *   fromUrl,         a preview of A, optional: captures what A's store lacks for the checks
 *   gtAa,            the run's ground truth mode ('gt' carries, 'gt-hi' never)
 *   size, q,         the run's size and extra query (carry-over checks the gates' own only)
 *   jobs, all, recheck, log
 * }
 * Resolves to { from, to, verdict, allows( side, e ) -> { ok, why }, carryEntry( side, e, key, dir ) }.
 */
export async function planCarry( options ) {

	const log = options.log ?? console.log;
	const from = fullSha( options.from ), to = fullSha( options.to );
	if ( ! from || ! to ) throw new Error( `carry-over: ${ ! from ? options.from : options.to } is not a commit of this repository` );
	const storeRoot = path.resolve( options.storeRoot );
	const dirA = path.join( storeRoot, from ), dirB = path.join( storeRoot, to );
	const work = path.join( dirB, 'carry', from );
	mkdirSync( work, { recursive: true } );
	const verdictFile = path.join( work, 'verdict.json' );
	const previous = options.recheck ? null : readJson( verdictFile );
	const clipTool = toolVersion( 'clip' ), browser = await browserBuild();
	const v = { format: 1, from, to, started: new Date().toISOString(), conditions: {} };
	const cond = v.conditions;
	const say = ( text ) => log( `carry ${ short( from ) } -> ${ short( to ) }: ${ text }` );
	const save = () => writeFileSync( verdictFile, JSON.stringify( v, null, 1 ) );

	// the entries that could carry at all: the baseline's and aa=gt's, A holding a manifest
	const sides = new Set( [ 'baseline', ...( options.gtAa === 'gt' ? [ 'gt' ] : [] ) ] );
	const entryDir = ( root, side, e ) => path.join( root, side, e.tier, e.kind, e.name );
	const candidates = options.targets.filter( ( t ) => sides.has( t.side ) && existsSync( path.join( entryDir( dirA, t.side, t.e ), 'complete.json' ) ) );
	const needGt = candidates.some( ( t ) => t.side === 'gt' );
	const needClips = candidates.some( ( t ) => t.e.kind === 'clip' );

	const fail = ( name, why, extra = {} ) => { cond[ name ] = { ok: false, why, ...extra }; say( `${ name }: NO (${ why })` ); save(); return false; };
	const pass = ( name, what, extra = {} ) => { cond[ name ] = { ok: true, what, ...extra }; say( `${ name }: ok (${ what })` ); save(); return true; };

	// --- the free conditions
	const names = git( [ 'diff', '--name-only', from, to ] ).split( '\n' ).filter( Boolean );
	const buildNames = git( [ 'diff', '--name-only', from, to, '--', ...BUILD_INPUTS ] ).split( '\n' ).filter( Boolean );
	v.diff = { files: names.length, buildInputs: buildNames };
	let blocked = false;
	if ( from === to ) blocked = ! fail( 'commits', 'the same commit' );
	else if ( ! candidates.length ) blocked = ! fail( 'entries', `${ short( from ) } stores none of this run's baseline or ground-truth entries` );
	else if ( options.size && options.size !== SIZE || options.q ) blocked = ! fail( 'options', `carry-over checks the gates' size ${ SIZE } with no extra query only` );
	if ( ! blocked && ! buildNames.length ) pass( 'sameBuild', `the diff touches nothing the build reads (${ names.length } files changed)` );
	const sameBuild = !! cond.sameBuild?.ok;
	if ( ! blocked && ! sameBuild ) {

		const motion = watchHits( buildNames, WATCH.motion ), gt = watchHits( buildNames, WATCH.gt );
		if ( motion.length ) fail( 'motionWatch', `the diff touches what moves: ${ motion.join( ', ' ) }`, { files: motion } );
		else pass( 'motionWatch', `nothing on the motion watch list among ${ buildNames.length } changed build inputs` );
		if ( needGt ) {

			if ( gt.length ) fail( 'gtWatch', `the diff touches the ground truth's path: ${ gt.join( ', ' ) }`, { files: gt } );
			else pass( 'gtWatch', 'nothing on the ground truth\'s watch list' );

		}

	}
	// a preview must serve its commit's build (capture-set checks its own; the clip captures here)
	const servedOk = new Map();
	const serves = ( url, sha ) => {

		if ( ! servedOk.has( url ) ) {

			try { checkServed( url, sha, ( t ) => say( t ) ); servedOk.set( url, null ); } catch ( error ) { servedOk.set( url, error.message ); }

		}
		return servedOk.get( url );

	};
	const go = ( ...names ) => options.all || names.every( ( n ) => cond[ n ]?.ok !== false );

	// --- the paused sets, one per tier
	const tiers = [ ...new Set( candidates.map( ( t ) => t.e.tier ) ) ];
	if ( ! blocked && ! sameBuild && go( 'motionWatch' ) ) for ( const tier of tiers ) {

		const views = [ ...new Set( [ ...( tier === 'high' ? viewNames() : [] ), ...candidates.filter( ( t ) => t.e.tier === tier ).map( ( t ) => t.e.view ) ] ) ].sort();
		await pausedCheck( tier, views );

	}

	async function pausedCheck( tier, views ) {

		const name = `paused-${ tier }`, set = pausedSet( tier );
		const a = path.join( dirA, set ), b = path.join( dirB, set );
		const { key, files } = await setKey( { quality: tier, aa: undefined, q: undefined, size: SIZE, capture: [] } );
		const capture = async ( url, dir, which ) => {

			const args = [ path.join( HERE, 'capture-set.mjs' ), url, '--reference', dir, '--views', views.join( ',' ), '--quality', tier, '--jobs', String( options.jobs ?? 2 ), '--task', 'carry' ];
			const r = await run( process.execPath, args, path.join( work, `${ set.replace( /\//g, '-' ) }-${ which }.log` ) );
			return r.code === 0;

		};
		// A's: stored with the same key and every view, else captured from A's preview when given
		const m = readJson( path.join( a, 'complete.json' ) );
		const lacks = keyMatches( m, key ) ? views.filter( ( view ) => ! m.coverage?.views?.[ view ]?.captured ) : views;
		if ( lacks.length ) {

			const why = ! m ? `${ short( from ) } has no stored ${ set }` : ! keyMatches( m, key ) ? `${ short( from ) }'s ${ set } has another key (${ describeKeyChange( m.key, m.toolFiles, key, files ) })` : `${ short( from ) }'s ${ set } lacks ${ lacks.join( ', ' ) }`;
			if ( ! options.fromUrl ) return fail( name, `${ why }: pass --carry-url <a preview of ${ short( from ) }>` );
			say( `${ why }: capturing it from ${ options.fromUrl }` );
			if ( ! await capture( options.fromUrl, a, 'from' ) ) return fail( name, `${ why }, and its capture failed (${ path.join( work, set + '-from.log' ) })` );

		}
		// B's: the store reuses, extends or captures it
		const mb = readJson( path.join( b, 'complete.json' ) );
		const bLacks = keyMatches( mb, key ) ? views.filter( ( view ) => ! mb.coverage?.views?.[ view ]?.captured ) : views;
		if ( bLacks.length ) {

			if ( ! options.url ) return fail( name, `${ short( to ) }'s ${ set } lacks ${ bLacks.length } views and no preview of ${ short( to ) } was given` );
			say( `capturing ${ short( to ) }'s ${ set } (${ bLacks.length } views) from ${ options.url }` );
			if ( ! await capture( options.url, b, 'to' ) ) return fail( name, `${ short( to ) }'s ${ set } capture failed (${ path.join( work, set.replace( /\//g, '-' ) + '-to.log' ) })` );

		}
		const ma = readJson( path.join( a, 'complete.json' ) ), mb2 = readJson( path.join( b, 'complete.json' ) );
		const repeats = views.every( ( view ) => ma.coverage.views[ view ]?.repeat && mb2.coverage.views[ view ]?.repeat );
		const mode = repeats ? [ '--aa-floor' ] : [ '--max-px1', '0' ];
		const out = path.join( work, `${ set.replace( /\//g, '-' ) }-diff` );
		const r = await run( 'uv', [ 'run', '--with', 'numpy', '--with', 'pillow', 'python', path.join( TOOLS, 'kit', 'diff-set.py' ), a, b, '--views', views.join( ',' ), '--out', out, ...mode ], path.join( out, 'diff.txt' ) );
		const differ = r.out.split( '\n' ).filter( ( l ) => /\bFAIL\b|MISSING/i.test( l ) ).map( ( l ) => l.trim().split( /\s+/ )[ 0 ] );
		const extra = { set, views, mode: mode.join( ' ' ), diff: path.relative( dirB, path.join( out, 'diff.txt' ) ) };
		return r.code === 0 ? pass( name, `${ set } matches on ${ views.length } views (diff-set ${ mode.join( ' ' ) })`, extra )
			: fail( name, `${ set } differs (diff-set ${ mode.join( ' ' ) }${ differ.length ? ': ' + differ.join( ', ' ) : '' })`, extra );

	}

	// --- A's stored rest captures against A's own paused set (no capture: files only). The paused
	// check vouches for A's set, not for A's entries: a rest capture that went wrong at A (one
	// was once stored 132 levels off A's own paused frame of that view) would carry on into every
	// later sha. A rest entry with a whole frame carries only when that frame is A's paused frame of
	// its view within STORED_TOLERANCE (two capture paths, so not byte-equal: 1 level on most pixels).
	// Entries without a whole frame (the TAA rest views, the lower tiers) cannot be checked so.
	const stored = v.stored = {};
	if ( ! blocked && ! sameBuild ) for ( const t of candidates ) {

		if ( t.side !== 'baseline' || t.e.kind !== 'rest' ) continue;
		const full = path.join( entryDir( dirA, t.side, t.e ), 'full', '0.png' ), frame = path.join( dirA, pausedSet( t.e.tier ), `${ t.e.view }.png` );
		if ( ! existsSync( full ) || ! existsSync( frame ) ) { stored[ t.e.key ] = { ok: true, checked: false, why: ! existsSync( full ) ? 'no whole frame' : `${ short( from ) }'s ${ pausedSet( t.e.tier ) } lacks ${ t.e.view }` }; continue; }
		const c = comparePng( full, frame );
		stored[ t.e.key ] = { ok: c.max <= STORED_TOLERANCE.max, checked: true, max: c.max, over2: c.over2 };
		if ( ! stored[ t.e.key ].ok ) say( `stored: ${ short( from ) }'s baseline ${ t.e.key } is off its own ${ pausedSet( t.e.tier ) } frame (max ${ c.max }, ${ c.over2 } px over 2 levels): recaptured, not carried` );

	}
	if ( Object.keys( stored ).length ) save();

	// --- the motion spot check (only for clips to carry; high only)
	// (A's entries by their last capture: a recaptured one is checked again)
	const lastRound = ( dir ) => readJson( path.join( dir, 'complete.json' ) )?.rounds?.at( - 1 )?.finished ?? null;
	const spotInputs = { clipTool: clipTool.tool, browser, frames: SPOT.frames, clips: SPOT.clips, refs: SPOT.clips.map( ( c ) => lastRound( path.join( dirA, 'baseline/high/clip', c ) ) ) };
	if ( ! blocked && ! sameBuild && needClips && go( 'motionWatch', 'paused-high' ) ) {

		const prev = previous?.conditions?.spot;
		if ( prev && stable( prev.inputs ) === stable( spotInputs ) ) { cond.spot = prev; say( `spot: ${ prev.ok ? 'ok' : 'NO' } (from an earlier run: ${ prev.what ?? prev.why })` ); save(); }
		else await spotCheck();

	}

	async function spotCheck() {

		const results = {};
		for ( const clip of SPOT.clips ) {

			// A's: its stored baseline entry (all frames, every frame), else a spot capture from A's preview
			const stored = path.join( dirA, 'baseline/high/clip', clip );
			const sm = readJson( path.join( stored, 'complete.json' ) );
			let ref = null, refWhy = '';
			if ( ! sm ) refWhy = `${ short( from ) } stores no baseline ${ clip }`;
			else if ( sm.key.tool !== clipTool.tool || stable( sm.key.browser ) !== stable( browser ) ) refWhy = `${ short( from ) }'s baseline ${ clip } has another key (${ describeKeyChange( sm.key, sm.toolFiles, { ...sm.key, tool: clipTool.tool, browser }, clipTool.files ) })`;
			else if ( sm.key.every !== 1 || sm.key.size !== SIZE || sm.key.q ) refWhy = `${ short( from ) }'s baseline ${ clip } is not every frame at the gates' size`;
			else ref = stored;
			if ( ! ref && options.fromUrl ) {

				const bad = serves( options.fromUrl, from );
				if ( bad ) { results[ clip ] = { ok: false, why: `${ refWhy }; ${ bad }` }; break; }
				ref = path.join( work, 'spot-from', clip );
				rmSync( ref, { recursive: true, force: true } );
				const r = await spotCapture( options.fromUrl, clip, ref, null, 'from' );
				if ( ! r ) { results[ clip ] = { ok: false, why: `${ refWhy }; its capture from ${ options.fromUrl } failed` }; break; }

			}
			if ( ! ref ) { results[ clip ] = { ok: false, why: `${ refWhy }: pass --carry-url <a preview of ${ short( from ) }>` }; break; }
			if ( ! options.url ) { results[ clip ] = { ok: false, why: `no preview of ${ short( to ) } to capture the spot check from` }; break; }
			const bad = serves( options.url, to );
			if ( bad ) { results[ clip ] = { ok: false, why: bad }; break; }
			let tries = 0, diff = null;
			while ( tries < 2 ) {

				const out = path.join( work, 'spot', tries ? `${ clip }.retry` : clip );
				rmSync( out, { recursive: true, force: true } );
				if ( ! await spotCapture( options.url, clip, out, path.join( ref, 'clip.json' ), tries ? 'to-retry' : 'to' ) ) { diff = { error: `the capture failed (${ path.join( work, 'spot', clip + '.log' ) })` }; break; }
				diff = compareClips( ref, out );
				tries ++;
				if ( diff.same ) break;

			}
			results[ clip ] = diff.same ? { ok: true, images: diff.images, tries } : { ok: false, why: diff.error ?? `${ diff.differ.length } of ${ diff.images } images differ (${ diff.differ.slice( 0, 4 ).map( ( d ) => `${ d.image } max ${ d.max }` ).join( ', ' ) })${ tries > 1 ? ' on both tries' : '' }`, differ: diff.differ, tries };
			if ( ! results[ clip ].ok && ! options.all ) break;

		}
		const bad = Object.entries( results ).filter( ( [ , r ] ) => ! r.ok );
		const extra = { inputs: spotInputs, clips: results };
		return bad.length || Object.keys( results ).length < SPOT.clips.length
			? fail( 'spot', bad.map( ( [ c, r ] ) => `${ c }: ${ r.why }` ).join( '; ' ) || 'not run', extra )
			: pass( 'spot', SPOT.clips.map( ( c ) => `${ c } ${ results[ c ].images } images identical${ results[ c ].tries > 1 ? ' on the retry' : '' }` ).join( ', ' ), extra );

	}

	async function spotCapture( url, clip, out, anchors, which ) {

		const args = [ path.join( HERE, 'capture-clip.mjs' ), `carry-${ clip }`, '--url', url, '--clip', clip, '--aa', 'msaa', '--quality', 'high',
			'--frames', String( SPOT.frames ), '--out', out, ...( anchors ? [ '--anchors-from', anchors ] : [] ) ];
		const r = await run( process.execPath, args, path.join( path.dirname( out ), `${ path.basename( out ) }-${ which }.log` ) );
		return r.code === 0 && existsSync( path.join( out, 'clip.json' ) );

	}

	// every image of A's first frames against B's, pixel for pixel
	function compareClips( a, b ) {

		const ia = clipImages( a, SPOT.frames ), ib = clipImages( b, SPOT.frames );
		const differ = [];
		for ( const [ image, file ] of Object.entries( ia ) ) {

			if ( ! ib[ image ] ) { differ.push( { image, max: 'missing' } ); continue; }
			const c = comparePng( file, ib[ image ] );
			if ( ! c.same ) differ.push( { image, max: c.max, over2: c.over2 } );

		}
		for ( const image of Object.keys( ib ) ) if ( ! ia[ image ] ) differ.push( { image, max: 'extra' } );
		return { same: differ.length === 0 && Object.keys( ia ).length > 0, images: Object.keys( ia ).length, differ };

	}

	// --- the ground truth's spot check (one rest view on high)
	const gtGo = () => go( 'motionWatch', 'gtWatch', ...tiers.map( ( t ) => `paused-${ t }` ) );
	if ( ! blocked && ! sameBuild && needGt && gtGo() ) {

		const view = SPOT.gtViews.find( ( w ) => existsSync( path.join( dirA, 'gt/high/rest', w, 'complete.json' ) ) && existsSync( path.join( dirA, 'baseline/high/rest', w, 'clip.json' ) ) );
		const gtInputs = { clipTool: clipTool.tool, browser, view, tolerance: GT_TOLERANCE, ref: view ? lastRound( path.join( dirA, 'gt/high/rest', view ) ) : null };
		const prev = previous?.conditions?.gtSpot;
		if ( ! view ) fail( 'gtSpot', `${ short( from ) } stores no rest ground truth of ${ SPOT.gtViews.join( ', ' ) } with its baseline` );
		else if ( prev && stable( prev.inputs ) === stable( gtInputs ) ) { cond.gtSpot = prev; say( `gtSpot: ${ prev.ok ? 'ok' : 'NO' } (from an earlier run: ${ prev.what ?? prev.why })` ); save(); }
		else await gtSpotCheck( view, gtInputs );

	}

	async function gtSpotCheck( view, inputs ) {

		const ref = path.join( dirA, 'gt/high/rest', view );
		const m = readJson( path.join( ref, 'complete.json' ) );
		if ( m.key.tool !== clipTool.tool || stable( m.key.browser ) !== stable( browser ) ) return fail( 'gtSpot', `${ short( from ) }'s ground truth of ${ view } has another key (${ describeKeyChange( m.key, m.toolFiles, { ...m.key, tool: clipTool.tool, browser }, clipTool.files ) })`, { inputs } );
		if ( ! options.url ) return fail( 'gtSpot', `no preview of ${ short( to ) } to capture the ground truth's spot check from`, { inputs } );
		const bad = serves( options.url, to );
		if ( bad ) return fail( 'gtSpot', bad, { inputs } );
		const out = path.join( work, 'gt-spot', view );
		rmSync( out, { recursive: true, force: true } );
		const args = [ path.join( HERE, 'capture-clip.mjs' ), `carry-gt-${ view }`, '--url', options.url, '--clip', 'rest', '--view', view, '--aa', 'gt', '--quality', 'high',
			'--anchors-from', path.join( dirA, 'baseline/high/rest', view, 'clip.json' ), '--out', out, ...( m.key.full ? [ '--full', '0' ] : [] ) ];
		const r = await run( process.execPath, args, `${ out }.log` );
		if ( r.code !== 0 || ! existsSync( path.join( out, 'clip.json' ) ) ) return fail( 'gtSpot', `the capture failed (${ out }.log)`, { inputs } );
		const ia = clipImages( ref ), ib = clipImages( out );
		const images = {}, over = [];
		for ( const [ image, file ] of Object.entries( ia ) ) {

			if ( ! ib[ image ] ) { over.push( `${ image } missing` ); continue; }
			const c = comparePng( file, ib[ image ] );
			images[ image ] = { max: c.max, over2: c.over2, pixels: c.pixels };
			if ( c.max > GT_TOLERANCE.max || c.over2 > GT_TOLERANCE.over2Share * c.pixels ) over.push( `${ image } max ${ c.max }, ${ c.over2 } px over 2 levels` );

		}
		const extra = { inputs, view, images };
		if ( ! Object.keys( ia ).length ) return fail( 'gtSpot', `${ short( from ) }'s ground truth of ${ view } holds no images`, extra );
		const worst = Object.values( images ).reduce( ( w, c ) => Math.max( w, c.max ), 0 );
		return over.length ? fail( 'gtSpot', `${ view } differs beyond the ground truth's noise: ${ over.slice( 0, 4 ).join( '; ' ) }`, extra )
			: pass( 'gtSpot', `${ view }: ${ Object.keys( ia ).length } images within the noise (max ${ worst }; bound max ${ GT_TOLERANCE.max }, ${ GT_TOLERANCE.over2Share * 100 } % over 2 levels)`, extra );

	}

	v.finished = new Date().toISOString();
	save();

	/** Whether an entry may carry, and if not why. */
	function allows( side, e ) {

		const no = ( why ) => ( { ok: false, why } );
		if ( ! sides.has( side ) ) return no( side === 'gt-hi' ? 'GT-HI is not carried' : `${ side } is not a reference` );
		for ( const n of [ 'commits', 'entries', 'options' ] ) if ( cond[ n ] ) return no( cond[ n ].why );
		if ( sameBuild ) return { ok: true };
		const need = [ 'motionWatch', `paused-${ e.tier }` ];
		if ( e.kind === 'clip' ) need.push( 'spot' );
		if ( side === 'gt' ) need.push( 'gtWatch', 'gtSpot' );
		for ( const n of need ) {

			if ( ! cond[ n ] ) return no( `${ n } not checked (an earlier condition failed)` );
			if ( ! cond[ n ].ok ) return no( `${ n }: ${ cond[ n ].why }` );

		}
		if ( e.kind === 'clip' && e.tier !== 'high' ) return no( 'the spot check covers high\'s clips only' );
		if ( ! cond[ `paused-${ e.tier }` ].views.includes( e.view ) ) return no( `${ e.view } is not in the paused set checked` );
		const s = side === 'baseline' ? stored[ e.key ] : null;
		if ( s && ! s.ok ) return no( `stored: ${ short( from ) }'s entry is off its own ${ pausedSet( e.tier ) } frame of ${ e.view } (max ${ s.max }, bound ${ STORED_TOLERANCE.max })` );
		return { ok: true };

	}

	/** Clones A's entry into B's directory `dir` (claimed by the caller) when A's manifest has the key
	 * B asks for; resolves to the manifest's additions, or { why } when it cannot carry. */
	function carryEntry( side, e, key, dir ) {

		const src = entryDir( dirA, side, e );
		const m = readJson( path.join( src, 'complete.json' ) );
		if ( ! m?.coverage?.captured ) return { why: `${ short( from ) } stores no ${ side } ${ e.key }` };
		if ( stable( m.key ) !== stable( key ) ) return { why: `${ short( from ) }'s entry has another key (${ describeKeyChange( m.key, m.toolFiles, key, clipTool.files ) })` };
		const skip = new Set( [ 'complete.json', 'incomplete.json', '.capturing', 'metrics', 'metrics.log' ] );
		for ( const f of readdirSync( src ) ) {

			if ( skip.has( f ) ) continue;
			cpSync( path.join( src, f ), path.join( dir, f ), { recursive: true, mode: constants.COPYFILE_FICLONE } );

		}
		return { coverage: m.coverage, manifest: { url: m.url, carriedFrom: {
			sha: from, path: m.path, origin: m.carriedFrom?.origin ?? from, at: new Date().toISOString(), verdict: path.relative( path.dirname( storeRoot ), verdictFile ),
		} } };

	}

	return { from, to, verdict: v, allows, carryEntry };

}

// ------------------------------------------------------------------ CLI: what would carry, and why

async function cli( argv ) {

	const o = {};
	const FLAGS = new Set( [ 'all', 'recheck' ] );
	const VALUES = new Set( [ 'from', 'to', 'url', 'carry-url', 'root', 'tiers', 'views', 'clips', 'only', 'jobs' ] );
	for ( let i = 0; i < argv.length; i ++ ) {

		const k = argv[ i ].startsWith( '--' ) ? argv[ i ].slice( 2 ) : null;
		if ( ! FLAGS.has( k ) && ! VALUES.has( k ) ) { console.error( `carry.mjs check: unknown argument ${ argv[ i ] } (see the file header)` ); process.exit( 2 ); }
		if ( VALUES.has( k ) && ( i + 1 >= argv.length || argv[ i + 1 ].startsWith( '--' ) ) ) { console.error( `carry.mjs check: ${ argv[ i ] } takes a value` ); process.exit( 2 ); }
		o[ k ] = FLAGS.has( k ) ? true : argv[ ++ i ];

	}
	if ( ! o.from || ! o.to ) { console.error( 'usage: carry.mjs check --from A --to B [--url U] [--carry-url UA] [--root DIR] [--tiers high] [--views a,b] [--clips a,b] [--only rest|clips] [--all] [--recheck]' ); process.exit( 2 ); }
	const { entries } = await import( './baseline.mjs' );
	const list = entries( { tiers: o.tiers ?? 'high', views: o.views, clips: o.clips, only: o.only } );
	const targets = list.flatMap( ( e ) => [ { side: 'baseline', e }, { side: 'gt', e } ] );
	const carry = await planCarry( { from: o.from, to: o.to, storeRoot: o.root ?? outputDir( REPO, 'parity' ), targets, url: o.url, fromUrl: o[ 'carry-url' ],
		gtAa: 'gt', jobs: o.jobs, all: o.all, recheck: o.recheck } );
	for ( const t of targets ) {

		const a = carry.allows( t.side, t.e );
		console.log( `${ a.ok ? 'carry' : 'keep ' } ${ t.side } ${ t.e.key }${ a.ok ? '' : ': ' + a.why }` );

	}

}

if ( process.argv[ 1 ] && path.resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url ) ) {

	const [ cmd, ...rest ] = process.argv.slice( 2 );
	if ( cmd !== 'check' ) { console.error( 'usage: carry.mjs check --from A --to B [options] (see the file header)' ); process.exit( 2 ); }
	cli( rest ).catch( ( e ) => { console.error( e.message ?? e ); process.exit( 2 ); } );

}
