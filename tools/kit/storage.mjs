// Reports and removes what the tools and agents leave on disk, by age and under a size cap.
//
// Captures, clip frames and references would otherwise fill the disk with frames read once: in a
// week the captures, clip frames and references of the anti-aliasing and island work reached
// hundreds of GB, almost all PNG frames read once by the task that made them. The reports keep the
// numbers, and the reference store recaptures a commit on demand; two days covers a task and the
// one that builds on it. Nobody runs the clean by hand: tools/kit/job.mjs starts `clean --auto` in
// the background on every command, and it runs when the last clean is 6 h old.
//
// Usage:
//   node tools/kit/storage.mjs [report] [--days 2] [--cap 50] [--floor 12]
//   node tools/kit/storage.mjs clean [--days 2] [--cap 50] [--floor 12] [--auto]
//
// Options:
//   report      prints what each kind holds and what `clean` would remove; deletes nothing (the default)
//   clean       removes it
//   --days n    the age after which bulk files, unused references and idle profiles go (default 2)
//   --cap n     the GB that bulk files and dead references may hold past the age rule (default 50)
//   --floor n   the cap never removes anything younger than n hours (default 12)
//   --auto      with clean: only if the last clean is 6 h old (job.mjs's background run)
//
// What it removes, in the main checkout (git's common directory, where every worktree's tools write
// their artifacts) and the temp folders. artifacts/ below is the configuration's output folder
// (tools/project.mjs `output`: `root`, and `captures`, `profiles` and `storage` in it):
// - bulk: images and videos of 256 KB or more (captures, clip frames, crops: .png .jpg .jpeg .webm
//   .mp4 .mov .exr .raw .rgba .bin) and browser profiles left in a task folder, under the folders of
//   artifacts/ the configuration lists (tools/project.mjs `storage.folders`), once older than
//   --days. Reports, JSON and small images stay, and so does every image a report or page links to
//   (a path in a .md or .html file under those folders, read relative to that file, the checkout
//   or artifacts/). Other artifacts folders are never read. artifacts/captures, where one-off
//   screenshots land, keeps its images 7 days and stays outside the cap.
// - references: a commit's folder in a reference store (a folder named by a full sha under those
//   folders, such as artifacts/parity; tools/kit/reference.mjs). Kept while its commit is live: the
//   HEAD of a worktree, the tip of a local branch committed to in the last 7 days, or the commit
//   such a branch shares with main (it compares against it). Otherwise removed once nothing in it
//   changed for --days. `.replaced/` and `.verify/` inside a live store go after a day.
// - profiles: the warm browser profiles (tools/kit/capture-browser.mjs; safe to delete). A browser
//   build's folder or a named profile idle for --days goes whole. A slot idle for 6 h whose caches
//   passed 512 MB loses them (IndexedDB, the HTTP and code caches): its next load is cold, its
//   captures the same. The shader caches stay. A slot is trimmed under its own lock (`<n>.lock`),
//   so a capture takes another slot meanwhile. The project's browser profiles in the temp folder
//   (`storage.temp.profiles`: profile-frame's) are trimmed the same way when no Chrome runs in them.
// - temp: what killed processes left in the temp folder over a day ago: Playwright's profiles and
//   artifacts, the project's run folders (`storage.temp.runs`: profile-frame's), Instruments
//   recordings (instruments*.ktrace, written by xctrace), pip's unpack and build folders.
// - sessions: Claude Code's temp folders for this checkout and its worktrees (agent outputs,
//   scratchpads), a session's once it has been idle for 3 days.
// The cap: past the age rule, bulk files and references that are not live are removed oldest first
// while they hold more than --cap GB, never one younger than --floor hours. A folder under
// artifacts with a `.keep` file is never touched: put one in a set a later task needs.
//
// Output: the rules in force, a table per kind (held, to remove, kept), the total, the count of
// images kept because a report links them and the last clean. `clean` then prints what it freed
// and writes artifacts/.storage/last-clean.json; the background clean appends its output to
// artifacts/.storage/clean.log (both in the main checkout). One clean runs at a time
// (artifacts/.storage/clean.lock); a second prints `another clean is running`.
//
// GPU lock and browser: neither. Cost: seconds (it walks artifacts/ and the temp folders).
//
// Exit codes: 0 done; 1 a clean that failed to remove some files; 2 an unknown command or option.
import './help.mjs';
import { closeSync, existsSync, lstatSync, mkdirSync, openSync, readdirSync, readFileSync, readlinkSync, rmSync, statSync, unlinkSync, writeFileSync, writeSync } from 'node:fs';
import { execFileSync, spawn } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import PROJECT from '../project.mjs';
import { ROOT, outputDir } from './root.mjs';

const HOUR = 3600e3;
const DAY = 24 * HOUR;
const GB = 1024 ** 3;
const DEFAULTS = { days: 2, cap: 50, floor: 12 };
// the folders under artifacts/ read for bulk files and reference stores, and the project's own
// leftovers in the temp folder (the configuration's `storage`)
const GENERATED = PROJECT.storage.folders;
const TEMP_RUNS = PROJECT.storage.temp.runs, TEMP_PROFILES = PROJECT.storage.temp.profiles;
const BULK = new Set( [ '.png', '.jpg', '.jpeg', '.webm', '.mp4', '.mov', '.exr', '.raw', '.rgba', '.bin' ] );
const BULK_MIN = 256 * 1024;
const LINK = /[\w./~@+%-]+\.(?:png|jpe?g|webm|mp4|mov|exr|raw|rgba|bin)\b/gi;
const CAPTURES_DAYS = 7;
const SHA = /^[0-9a-f]{40}$/;
const STORE_SCRATCH = new Set( [ '.replaced', '.verify' ] );
const LIVE_DAYS = 7;
const SLOT_IDLE = 6 * HOUR;
const CACHES = [ 'IndexedDB', 'Cache', 'Code Cache' ];
const CACHE_MAX = 512 * 1024 ** 2;
// a run folder of the project's tools is one of its prefixes, never one of its kept profiles
const TEMP = [ /^playwright[_-]/, { test: ( name ) => TEMP_RUNS.some( ( p ) => name.startsWith( p ) ) && ! TEMP_PROFILES.includes( name ) }, /^instruments.*\.ktrace$/, /^pip-/ ];
const SESSION_DAYS = 3;
const AUTO_EVERY = 6 * HOUR;
const LOCK_STALE = 2 * HOUR;

function mainCheckout() {

	try {

		const common = execFileSync( 'git', [ 'rev-parse', '--path-format=absolute', '--git-common-dir' ], { cwd: ROOT, encoding: 'utf8', timeout: 5000 } ).trim();
		if ( path.basename( common ) === '.git' ) return path.dirname( common );

	} catch { /* not a git checkout: this one */ }
	return ROOT;

}

const list = ( dir ) => {

	try { return readdirSync( dir, { withFileTypes: true } ); } catch { return []; }

};

const mtime = ( p ) => {

	try { return lstatSync( p ).mtimeMs; } catch { return 0; }

};

/** A tree's disk use and its newest modification (the folder's own included). */
function tree( dir ) {

	let bytes = 0, newest = mtime( dir ), files = 0;
	const stack = [ dir ];
	while ( stack.length ) {

		const d = stack.pop();
		for ( const e of list( d ) ) {

			const p = path.join( d, e.name );
			if ( e.isDirectory() ) {

				stack.push( p );
				continue;

			}
			let st;
			try { st = lstatSync( p ); } catch { continue; }
			bytes += st.blocks * 512;
			files ++;
			if ( st.mtimeMs > newest ) newest = st.mtimeMs;

		}

	}
	return { bytes, newest, files };

}

/** The newest modification within `depth` levels: a cheap idle test for a browser profile, which
 * rewrites its `Local State` and `Default/Preferences` on every launch. */
function recent( dir, depth ) {

	let newest = mtime( dir );
	if ( depth > 0 ) for ( const e of list( dir ) ) newest = Math.max( newest, recent( path.join( dir, e.name ), e.isDirectory() ? depth - 1 : 0 ) );
	return newest;

}

const alive = ( pid ) => {

	try { process.kill( pid, 0 ); return true; } catch ( error ) { return error.code === 'EPERM'; }

};

const git = ( main, args ) => execFileSync( 'git', args, { cwd: main, encoding: 'utf8', timeout: 20000 } ).trim();

/** The commits a check may still compare against (see the header). */
function liveCommits( main, now ) {

	const live = new Set();
	const heads = [];
	try {

		for ( const line of git( main, [ 'for-each-ref', '--format=%(objectname) %(committerdate:unix)', 'refs/heads' ] ).split( '\n' ) ) {

			const [ sha, time ] = line.split( ' ' );
			if ( sha && now - Number( time ) * 1000 < LIVE_DAYS * DAY ) heads.push( sha );

		}
		for ( const line of git( main, [ 'worktree', 'list', '--porcelain' ] ).split( '\n' ) ) if ( line.startsWith( 'HEAD ' ) ) {

			const sha = line.slice( 5 );
			if ( now - Number( git( main, [ 'show', '-s', '--format=%ct', sha ] ) ) * 1000 < LIVE_DAYS * DAY ) heads.push( sha );

		}
		for ( const sha of new Set( heads ) ) {

			live.add( sha );
			try { live.add( git( main, [ 'merge-base', 'main', sha ] ) ); } catch { /* unrelated history */ }

		}
		live.add( git( main, [ 'rev-parse', 'main' ] ) );

	} catch ( error ) {

		// without git's answer no store can be judged dead: keep them all
		throw new Error( `storage: cannot list the live commits (${ error.message.split( '\n' )[ 0 ] })` );

	}
	return live;

}

/** Every path a report or page names that ends in an image or video extension, resolved against
 * the file's folder, the checkout and artifacts/ (any of them may be meant). */
function linksOf( file, main, linked ) {

	let text;
	try {

		if ( statSync( file ).size > 8 * 1024 ** 2 ) return;
		text = readFileSync( file, 'utf8' );

	} catch { return; }
	for ( const [ token ] of text.matchAll( LINK ) ) {

		let name = token;
		try { name = decodeURIComponent( token ); } catch { /* as written */ }
		for ( const base of [ path.dirname( file ), main, outputDir( main ) ] ) linked.add( path.resolve( base, name ) );

	}

}

function scanArtifacts( main, live, now, o, items ) {

	const artifacts = outputDir( main );
	const linked = new Set();
	const bulk = [];
	for ( const root of GENERATED ) {

		const captures = root === 'captures';
		const stack = [ path.join( artifacts, root ) ];
		while ( stack.length ) {

			const dir = stack.pop();
			const entries = list( dir );
			if ( entries.some( ( e ) => e.name === '.keep' ) ) continue;
			if ( entries.some( ( e ) => e.name === 'Local State' ) ) {

				// a browser profile left in a task folder
				const t = tree( dir );
				bulk.push( { kind: 'bulk', path: dir, bytes: t.bytes, files: t.files, newest: t.newest, cap: ! captures, captures } );
				continue;

			}
			for ( const e of entries ) {

				const p = path.join( dir, e.name );
				if ( e.isDirectory() ) {

					if ( SHA.test( e.name ) ) scanStore( p, live.has( e.name ), now, items );
					else stack.push( p );
					continue;

				}
				if ( ! e.isFile() ) continue;
				const ext = path.extname( e.name ).toLowerCase();
				if ( ext === '.md' || ext === '.html' ) linksOf( p, main, linked );
				if ( ! BULK.has( ext ) ) continue;
				let st;
				try { st = lstatSync( p ); } catch { continue; }
				if ( st.size >= BULK_MIN ) bulk.push( { kind: 'bulk', path: p, bytes: st.blocks * 512, files: 1, newest: st.mtimeMs, cap: ! captures, captures } );

			}

		}

	}

	let protectedFiles = 0;
	for ( const item of bulk ) {

		if ( linked.has( item.path ) ) {

			protectedFiles ++;
			continue;

		}
		if ( now - item.newest > ( item.captures ? CAPTURES_DAYS : o.days ) * DAY ) item.action = 'delete';
		items.push( item );

	}
	return protectedFiles;

}

function scanStore( dir, live, now, items ) {

	const entries = list( dir );
	if ( entries.some( ( e ) => e.name === '.keep' ) ) return;
	const t = tree( dir );
	if ( live ) {

		items.push( { kind: 'references', path: dir, bytes: t.bytes, files: t.files, newest: t.newest, live: true } );
		for ( const e of entries ) if ( e.isDirectory() && STORE_SCRATCH.has( e.name ) ) {

			const s = tree( path.join( dir, e.name ) );
			if ( now - s.newest > DAY ) items.push( { kind: 'references', path: path.join( dir, e.name ), bytes: s.bytes, files: s.files, newest: s.newest, action: 'delete', inside: true } );

		}

	} else items.push( { kind: 'references', path: dir, bytes: t.bytes, files: t.files, newest: t.newest, cap: true, aged: true } );

}

/** Takes a capture slot's lock the way capture-browser.mjs does (mkdir, then owner.json): false
 * when the slot is in use or its lock is stale (capture-browser recycles those). */
function takeSlot( slot ) {

	const lock = `${ slot }.lock`;
	try { mkdirSync( lock ); } catch { return null; }
	writeFileSync( path.join( lock, 'owner.json' ), JSON.stringify( { pid: process.pid, label: 'storage.mjs trim' } ) );
	return lock;

}

function chromeRunning( profile ) {

	try {

		const target = readlinkSync( path.join( profile, 'SingletonLock' ) );
		return alive( Number( target.split( '-' ).pop() ) );

	} catch { return false; }

}

function trimItem( slot, now, items, locked ) {

	if ( ! existsSync( path.join( slot, 'Default' ) ) ) return;
	if ( now - recent( slot, 2 ) < SLOT_IDLE ) return;
	const parts = CACHES.map( ( c ) => path.join( slot, 'Default', c ) ).filter( existsSync );
	let bytes = 0, files = 0;
	for ( const p of parts ) {

		const t = tree( p );
		bytes += t.bytes;
		files += t.files;

	}
	if ( bytes > CACHE_MAX ) items.push( { kind: 'profiles', path: slot, bytes, files, newest: recent( slot, 2 ), action: 'trim', parts, locked } );

}

function scanProfiles( main, now, o, items ) {

	const profiles = process.env[ `${ PROJECT.machine.envPrefix }CAPTURE_PROFILES` ];
	const root = profiles ? path.resolve( profiles ) : outputDir( main, 'profiles' );
	for ( const e of list( root ) ) if ( e.isDirectory() ) {

		const dir = path.join( root, e.name );
		if ( now - recent( dir, 3 ) > o.days * DAY ) {

			const t = tree( dir );
			items.push( { kind: 'profiles', path: dir, bytes: t.bytes, files: t.files, newest: t.newest, action: 'delete' } );
			continue;

		}
		if ( existsSync( path.join( dir, 'Default' ) ) ) {

			if ( ! chromeRunning( dir ) ) trimItem( dir, now, items, false );
			continue;

		}
		for ( const s of list( dir ) ) if ( s.isDirectory() && /^\d+$/.test( s.name ) ) trimItem( path.join( dir, s.name ), now, items, true );

	}
	for ( const name of TEMP_PROFILES ) {

		const profile = path.join( os.tmpdir(), name );
		if ( ! chromeRunning( profile ) ) trimItem( profile, now, items, false );

	}

}

function scanTemp( now, items ) {

	const dir = os.tmpdir();
	for ( const e of list( dir ) ) if ( TEMP.some( ( re ) => re.test( e.name ) ) ) {

		const t = tree( path.join( dir, e.name ) );
		if ( now - t.newest > DAY ) items.push( { kind: 'temp', path: path.join( dir, e.name ), bytes: t.bytes, files: t.files, newest: t.newest, action: 'delete' } );

	}

}

function scanSessions( main, now, items ) {

	if ( typeof process.getuid !== 'function' ) return;
	const base = path.join( '/tmp', `claude-${ process.getuid() }` );
	const name = main.replace( /[^A-Za-z0-9]/g, '-' );
	for ( const p of list( base ) ) if ( p.isDirectory() && ( p.name === name || p.name.startsWith( `${ name }-` ) ) ) {

		for ( const s of list( path.join( base, p.name ) ) ) if ( s.isDirectory() ) {

			const dir = path.join( base, p.name, s.name );
			const t = tree( dir );
			if ( now - t.newest > SESSION_DAYS * DAY ) items.push( { kind: 'sessions', path: dir, bytes: t.bytes, files: t.files, newest: t.newest, action: 'delete' } );

		}

	}

}

/** Everything the rules would remove, and what they keep of the kinds they judge. */
export function plan( o = DEFAULTS, now = Date.now() ) {

	const main = mainCheckout();
	const live = liveCommits( main, now );
	const items = [];
	const linked = scanArtifacts( main, live, now, o, items );
	for ( const item of items ) if ( item.aged && now - item.newest > o.days * DAY ) item.action = 'delete';
	scanProfiles( main, now, o, items );
	scanTemp( now, items );
	scanSessions( main, now, items );

	// the cap, oldest first, never past the floor
	const kept = items.filter( ( i ) => i.cap && ! i.action ).sort( ( a, b ) => a.newest - b.newest );
	let held = kept.reduce( ( sum, i ) => sum + i.bytes, 0 );
	for ( const item of kept ) {

		if ( held <= o.cap * GB || now - item.newest < o.floor * HOUR ) break;
		item.action = 'delete';
		item.byCap = true;
		held -= item.bytes;

	}
	return { main, items, linked };

}

const size = ( bytes ) => bytes >= GB ? `${ ( bytes / GB ).toFixed( 1 ) } GB` : `${ ( bytes / 1024 ** 2 ).toFixed( 0 ) } MB`;

function report( { main, items, linked }, o ) {

	console.log( `rules: --days ${ o.days }, --cap ${ o.cap } GB, --floor ${ o.floor } h (tools/kit/storage.mjs header)` );
	console.log( 'kind         held       remove                     keep' );
	let held = 0, removed = 0;
	for ( const kind of [ 'bulk', 'references', 'profiles', 'temp', 'sessions' ] ) {

		const of = items.filter( ( i ) => i.kind === kind && ! i.inside );
		const out = items.filter( ( i ) => i.kind === kind && i.action );
		const h = of.reduce( ( s, i ) => s + i.bytes, 0 );
		const r = out.reduce( ( s, i ) => s + i.bytes, 0 );
		const n = out.reduce( ( s, i ) => s + i.files, 0 );
		const cap = out.filter( ( i ) => i.byCap ).reduce( ( s, i ) => s + i.bytes, 0 );
		held += h;
		removed += r;
		const keep = kind === 'profiles' || kind === 'temp' || kind === 'sessions' ? '' : size( Math.max( 0, h - r ) );
		console.log( `${ kind.padEnd( 12 ) } ${ ( kind === 'bulk' || kind === 'references' ? size( h ) : '' ).padEnd( 10 ) } ${ `${ size( r ) } (${ n } files${ cap ? `, ${ size( cap ) } by the cap` : '' })`.padEnd( 26 ) } ${ keep }` );

	}
	console.log( `total to remove: ${ size( removed ) }` );
	console.log( `kept because a report or page links them: ${ linked } images` );
	const groups = new Map();
	for ( const i of items ) if ( i.action ) {

		const rel = path.relative( main, i.path );
		const key = rel.startsWith( '..' ) ? `(${ i.kind })` : rel.split( path.sep ).slice( 0, 3 ).join( '/' );
		groups.set( key, ( groups.get( key ) ?? 0 ) + i.bytes );

	}
	const top = [ ...groups ].sort( ( a, b ) => b[ 1 ] - a[ 1 ] ).slice( 0, 10 );
	if ( top.length ) console.log( `largest: ${ top.map( ( [ k, b ] ) => `${ k } ${ size( b ) }` ).join( ', ' ) }` );

}

function apply( { items } ) {

	let freed = 0, failed = 0;
	const done = [];
	for ( const item of items ) if ( item.action ) {

		if ( done.some( ( d ) => item.path.startsWith( d + path.sep ) ) ) continue;
		try {

			if ( item.action === 'trim' ) {

				const lock = item.locked ? takeSlot( item.path ) : null;
				if ( item.locked && ! lock ) continue;
				try {

					for ( const p of item.parts ) rmSync( p, { recursive: true, force: true } );

				} finally {

					if ( lock ) rmSync( lock, { recursive: true, force: true } );

				}

			} else rmSync( item.path, { recursive: true, force: true } );
			freed += item.bytes;
			if ( item.action === 'delete' ) done.push( item.path );

		} catch ( error ) {

			failed ++;
			console.error( `storage: ${ item.path }: ${ error.message }` );

		}

	}
	return { freed, failed };

}

const stateDir = ( main ) => outputDir( main, 'storage' );

function lastClean( main ) {

	try { return JSON.parse( readFileSync( path.join( stateDir( main ), 'last-clean.json' ), 'utf8' ) ); } catch { return null; }

}

/** Takes the clean's lock (a file created exclusively; one older than 2 h was abandoned). */
function takeLock( main ) {

	const dir = stateDir( main );
	mkdirSync( dir, { recursive: true } );
	const lock = path.join( dir, 'clean.lock' );
	if ( existsSync( lock ) && Date.now() - mtime( lock ) > LOCK_STALE ) try { unlinkSync( lock ); } catch { /* taken meanwhile */ }
	try {

		const fd = openSync( lock, 'wx' );
		writeSync( fd, String( process.pid ) );
		closeSync( fd );
		return lock;

	} catch { return null; }

}

const due = ( main ) => ! ( Date.parse( lastClean( main )?.at ?? '' ) > Date.now() - AUTO_EVERY );

/** Starts `clean --auto` in the background when the last clean is 6 h old (job.mjs calls this on
 * every command; it costs one git call and one file read otherwise). Never throws. */
export function autoClean() {

	try {

		const main = mainCheckout();
		if ( ! due( main ) || ! takeLock( main ) ) return false;
		const log = openSync( path.join( stateDir( main ), 'clean.log' ), 'a' );
		spawn( process.execPath, [ fileURLToPath( import.meta.url ), 'clean', '--auto', '--locked' ], { detached: true, stdio: [ 'ignore', log, log ] } ).unref();
		closeSync( log );
		return true;

	} catch { return false; }

}

function parse( argv ) {

	const o = { ...DEFAULTS, command: 'report', auto: false, locked: false };
	const args = [ ...argv ];
	if ( args[ 0 ] && ! args[ 0 ].startsWith( '--' ) ) o.command = args.shift();
	if ( o.command !== 'report' && o.command !== 'clean' ) throw new Error( `storage: unknown command ${ o.command } (report|clean)` );
	while ( args.length ) {

		const a = args.shift();
		if ( a === '--auto' ) o.auto = true;
		else if ( a === '--locked' ) o.locked = true;
		else if ( a === '--days' || a === '--cap' || a === '--floor' ) {

			const v = Number( args.shift() );
			if ( ! ( v >= 0 ) ) throw new Error( `storage: ${ a } takes a number` );
			o[ a.slice( 2 ) ] = v;

		} else throw new Error( `storage: unknown option ${ a }` );

	}
	return o;

}

async function main( argv ) {

	const o = parse( argv );
	const checkout = mainCheckout();
	const last = lastClean( checkout );
	if ( o.command === 'report' ) {

		report( plan( o ), o );
		console.log( `last clean: ${ last ? `${ last.at }, ${ size( last.freed ) } freed` : 'never' }` );
		return 0;

	}
	let lock = o.locked ? path.join( stateDir( checkout ), 'clean.lock' ) : null;
	try {

		if ( o.auto && ! o.locked && ! due( checkout ) ) {

			console.log( `storage: last clean ${ last.at }; the next is due 6 h after it` );
			return 0;

		}
		if ( ! lock ) lock = takeLock( checkout );
		if ( ! lock ) {

			console.log( 'storage: another clean is running' );
			return 0;

		}
		const started = new Date().toISOString();
		const p = plan( o );
		report( p, o );
		const { freed, failed } = apply( p );
		writeFileSync( path.join( stateDir( checkout ), 'last-clean.json' ), JSON.stringify( { at: started, freed, failed } ) + '\n' );
		console.log( `[${ started }] storage: freed ${ size( freed ) }${ failed ? `, ${ failed } failed` : '' }` );
		return failed ? 1 : 0;

	} finally {

		if ( lock ) try { unlinkSync( lock ); } catch { /* gone */ }

	}

}

if ( process.argv[ 1 ] && path.resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url ) ) {

	main( process.argv.slice( 2 ) ).then( ( code ) => process.exit( code ), ( error ) => {

		console.error( error.message );
		process.exit( 2 );

	} );

}
