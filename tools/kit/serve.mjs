// Starts the dev or preview server of the checkout the cwd is in, or of another commit, on a free port.
//
// A worktree's server must not collide with another worktree's, and an agent must not pick a port
// by hand, look it up with lsof or start vite itself. This picks a free port (or names who holds
// the one asked for), starts vite detached through tools/kit/job.mjs, waits until its own server
// answers and prints the URL and how to stop it. `--sha` serves another commit instead, for a
// comparison against an older build.
//
// Usage:
//   node tools/kit/serve.mjs dev|preview [--port N] [--no-build] [--sha <rev>]
//   node tools/kit/serve.mjs clean
//
// Options:
//   dev          the dev server (vite)
//   preview      the production build (the configuration's `serve.build` first, then `vite preview`)
//   --port N     uses port N, or fails naming the process that listens on it (default: the first
//                free port in the configuration's range, 6400-6499)
//   --no-build   preview: serves the dist/ there is, without building
//   --sha rev    serves that commit from a detached worktree instead of this checkout
//   clean        removes every --sha worktree but those with a live job (no options)
//
// The project's values come from the configuration (tools/project.mjs `serve`): the build, vite's
// command per mode, the ports, where `--sha` worktrees go and the module that fills them.
//
// The served checkout is the one containing the cwd, not the one holding this file: an agent may
// call the tool by the main checkout's absolute path from its worktree, and must not serve (and
// rebuild) main's dist by mistake. A cwd outside this repository exits 1. The first line printed
// names the served checkout: its path, HEAD (short sha) and branch.
//
// The URL is printed only when the server answering is this run's job: the job still runs and its
// log holds vite's ready line for that port, so a port another session takes at the same moment
// never passes for its own. When another session's server takes the port between the check and
// vite's bind, vite exits "already in use" and the next free port is tried (with `--port N`, exit 1
// naming the holder).
//
// `--sha <rev>`: the commit gets a detached worktree at the main checkout's
// .claude/worktrees/sha-<12 hex> (never a branch, so nothing is left to delete), reused by every
// later call for the same commit. When the commit's package-lock.json equals the main checkout's,
// its node_modules is a real folder holding a link to each of the main checkout's entries except
// vite's caches (.vite, .vite-temp, .cache): a link to the whole folder would put the worktree's
// prebundled dependencies into the main checkout's cache, which its own dev server uses. With
// another lockfile it runs `npm ci` there instead. Then the configuration's `serve.prepare` module
// fills what git does not hold (tools/repo/serve-checkout.mjs: the assets of a commit from before
// the asset lock). `preview` builds once per
// worktree: a successful build writes the sha into dist/.serve-sha, and a call that finds it skips
// the build. The vite job runs in the worktree, with its log in the calling checkout's
// artifacts/jobs/, so `clean` cannot delete it.
//
// `clean` skips, and names, a sha-* worktree with a live job (job.mjs list --cwd), since a server
// another session started may still be in use: stop its job first. It removes the others (git
// worktree remove --force: node_modules and dist are untracked; the links are removed, not
// followed) and runs git worktree prune.
//
// Output: the served checkout's line, then `<url>  log <path>  (stop: node tools/kit/job.mjs stop
// <path>)`. The log is artifacts/jobs/serve-<mode>-<port>.log in the calling checkout (the
// configuration's `output.jobs`)
// (serve-<mode>-sha-<12 hex>-<port>.log for `--sha`). On a failure the log's tail is printed.
//
// GPU lock and browser: neither; it starts a server (a detached job that outlives this process:
// stop it with `job.mjs stop <log>` before handing back). Cost: seconds for dev; a build for
// preview; a worktree and maybe `npm ci` on the first `--sha` call of a commit.
//
// Exit codes: 0 serving (or `clean` done); 1 bad usage, a cwd outside the repository, a port
// taken, no free port, a revision that is not a commit, vite exiting for another reason or not
// answering within 60 s (its job is stopped); a failed build or `npm ci` exits with its code.
import './help.mjs';
import './qos.mjs';
import net from 'node:net';
import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, mkdirSync, symlinkSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import PROJECT from '../project.mjs';
import { ROOT, outputDir } from './root.mjs';

// This checkout's root and the job tool beside this file; the checkout it serves is `caller`, below.
const here = ROOT;
const jobTool = join( dirname( fileURLToPath( import.meta.url ) ), 'job.mjs' );
// the build, vite's commands, the ports, where `--sha` worktrees go and what fills them (the
// configuration's `serve`)
const SERVE = PROJECT.serve;
const [ FIRST_PORT, LAST_PORT ] = SERVE.ports;
const usage = 'usage: node tools/kit/serve.mjs dev|preview [--port N] [--no-build] [--sha <rev>] | clean';

function fail( msg ) {

	console.error( msg );
	process.exit( 1 );

}

const args = process.argv.slice( 2 );
let mode = null, port = null, build = true, rev = null;
for ( let i = 0; i < args.length; i ++ ) {

	const a = args[ i ];
	if ( a === 'dev' || a === 'preview' || a === 'clean' ) {

		if ( mode ) fail( usage );
		mode = a;

	} else if ( a === '--port' ) {

		port = Number( args[ ++ i ] );
		if ( ! Number.isInteger( port ) || port < 1 || port > 65535 ) fail( `bad --port\n${ usage }` );

	} else if ( a === '--sha' ) {

		rev = args[ ++ i ];
		if ( ! rev ) fail( `--sha needs a revision\n${ usage }` );

	} else if ( a === '--no-build' ) build = false;
	else fail( `unknown option ${ a }\n${ usage }` );

}

if ( ! mode ) fail( usage );
if ( mode === 'clean' && ( args.length > 1 ) ) fail( `clean takes no options\n${ usage }` );

function git( a, cwd ) {

	return execFileSync( 'git', a, { cwd, encoding: 'utf8', stdio: [ 'ignore', 'pipe', 'pipe' ] } ).trim();

}

const commonDir = ( cwd ) => git( [ 'rev-parse', '--path-format=absolute', '--git-common-dir' ], cwd );

// The checkout the cwd is in: it is served, built and logged into. It must be a checkout of this
// repository (the same common git directory as the tool's own), or the tool would serve a stranger.
let caller;
try {

	caller = git( [ 'rev-parse', '--show-toplevel' ], process.cwd() );
	if ( commonDir( caller ) !== commonDir( here ) ) caller = null;

} catch {

	caller = null;

}

if ( ! caller ) fail( `${ process.cwd() } is not in a checkout of ${ dirname( commonDir( here ) ) }; run serve.mjs from one` );

// The main checkout: sha-* worktrees live there whichever worktree this script runs from.
const mainCheckout = dirname( commonDir( caller ) );
const shaRoot = join( mainCheckout, SERVE.worktrees );

if ( mode === 'clean' ) {

	const trees = git( [ 'worktree', 'list', '--porcelain' ], caller ).split( '\n' )
		.filter( ( l ) => l.startsWith( 'worktree ' ) ).map( ( l ) => l.slice( 9 ) )
		.filter( ( p ) => dirname( p ) === shaRoot && /^sha-[0-9a-f]+$/.test( p.slice( shaRoot.length + 1 ) ) );
	for ( const t of trees ) {

		const live = spawnSync( 'node', [ jobTool, 'list', '--cwd', t, '--lines', '0' ], { encoding: 'utf8' } ).stdout;
		if ( ! live.startsWith( 'no running jobs' ) ) {

			console.log( `skipped ${ t }: a job runs there (node tools/kit/job.mjs list --cwd ${ t })` );
			continue;

		}

		git( [ 'worktree', 'remove', '--force', t ], caller );
		console.log( `removed ${ t }` );

	}

	git( [ 'worktree', 'prune' ], caller );
	if ( trees.length === 0 ) console.log( 'no sha-* worktrees' );
	process.exit( 0 );

}

let repo = caller, tag = '';
if ( rev !== null ) {

	let sha;
	try {

		sha = git( [ 'rev-parse', '--verify', '--quiet', `${ rev }^{commit}` ], caller );

	} catch {

		fail( `--sha ${ rev }: not a commit` );

	}

	tag = sha.slice( 0, 12 );
	repo = join( shaRoot, `sha-${ tag }` );
	console.log( `serving ${ repo }  HEAD ${ sha.slice( 0, 7 ) }  (detached)` );
	if ( existsSync( repo ) ) {

		const head = git( [ 'rev-parse', 'HEAD' ], repo );
		if ( head !== sha ) fail( `${ repo } is at ${ head }, not ${ sha }; run node tools/kit/serve.mjs clean` );
		console.log( `reusing ${ repo }` );

	} else {

		git( [ 'worktree', 'add', '--detach', repo, sha ] );
		console.log( `created ${ repo }` );

	}

	const modules = join( repo, 'node_modules' );
	if ( ! existsSync( modules ) ) {

		let lock = null;
		try {

			lock = git( [ 'show', `${ sha }:package-lock.json` ], caller );

		} catch {}
		const mainLock = join( mainCheckout, 'package-lock.json' );
		const same = lock !== null && existsSync( mainLock ) && existsSync( join( mainCheckout, 'node_modules' ) ) &&
			readFileSync( mainLock, 'utf8' ).trim() === lock;
		if ( same ) {

			mkdirSync( modules );
			const caches = new Set( [ '.vite', '.vite-temp', '.cache' ] );
			for ( const e of readdirSync( join( mainCheckout, 'node_modules' ) ) ) {

				if ( ! caches.has( e ) ) symlinkSync( join( mainCheckout, 'node_modules', e ), join( modules, e ) );

			}

			console.log( 'node_modules: linked from the main checkout (same package-lock.json)' );

		} else {

			console.log( 'node_modules: package-lock.json differs from the main checkout\'s, running npm ci' );
			const r = spawnSync( 'npm', [ 'ci' ], { cwd: repo, stdio: 'inherit' } );
			if ( r.status !== 0 ) process.exit( r.status ?? 1 );

		}

	}

	// what git does not hold, by the project's module (the configuration's `serve.prepare`)
	const { prepareCheckout } = await import( pathToFileURL( join( here, SERVE.prepare ) ).href );
	prepareCheckout( { repo, caller } );

} else {

	const head = git( [ 'rev-parse', '--short=7', 'HEAD' ], repo );
	const branch = git( [ 'branch', '--show-current' ], repo ) || 'detached';
	console.log( `serving ${ repo }  HEAD ${ head }  (${ branch })` );

}

const stamp = join( repo, 'dist', '.serve-sha' );
if ( mode === 'preview' && build && rev !== null && existsSync( stamp ) && readFileSync( stamp, 'utf8' ).trim() === git( [ 'rev-parse', 'HEAD' ], repo ) ) {

	console.log( `dist/ already built for ${ tag }, skipping the build` );
	build = false;

}

function isFree( p ) {

	return new Promise( ( resolve ) => {

		const s = net.createServer();
		s.once( 'error', () => resolve( false ) );
		s.listen( p, '127.0.0.1', () => s.close( () => resolve( true ) ) );

	} );

}

function holder( p ) {

	const lsof = ( a ) => {

		try {

			return execFileSync( 'lsof', a, { encoding: 'utf8' } );

		} catch ( e ) {

			return e.stdout || '';

		}

	};
	const pid = ( lsof( [ '-nP', `-iTCP:${ p }`, '-sTCP:LISTEN', '-Fpn' ] ).match( /^p(\d+)/m ) || [] )[ 1 ];
	if ( ! pid ) return 'an unknown process';
	const cwd = ( lsof( [ '-a', '-p', pid, '-d', 'cwd', '-Fn' ] ).match( /^n(.*)$/m ) || [] )[ 1 ];
	return `pid ${ pid } (cwd ${ cwd || 'unknown' })`;

}

// A port free here can still be taken before vite binds it: two sessions starting at the same moment
// both saw 6400 free, the second vite exited "Port 6400 is already in use", and the URL check was
// answered by the other worktree's server, so serve.mjs printed a URL serving another checkout.
// So success is judged from this job alone: its log shows vite's ready line for this port and the
// job is still running (vite binds --strictPort before it prints the line). A vite that exits
// first moves on to the next free port, or with `--port N` fails naming who holds N.
const fixed = port !== null;
if ( fixed && ! await isFree( port ) ) fail( `port ${ port } is taken by ${ holder( port ) }` );

async function nextFree( from ) {

	for ( let p = from; p <= LAST_PORT; p ++ ) if ( await isFree( p ) ) return p;
	return null;

}

if ( ! fixed ) {

	port = await nextFree( FIRST_PORT );
	if ( port === null ) fail( `no free port in ${ FIRST_PORT }-${ LAST_PORT }` );

}

if ( mode === 'preview' && build ) {

	const r = spawnSync( SERVE.build[ 0 ], SERVE.build.slice( 1 ), { cwd: repo, stdio: 'inherit' } );
	if ( r.status !== 0 ) process.exit( r.status ?? 1 );
	if ( rev !== null ) writeFileSync( stamp, `${ git( [ 'rev-parse', 'HEAD' ], repo ) }\n` );

}

const readLog = ( f ) => {

	try {

		return readFileSync( f, 'utf8' );

	} catch {

		return '';

	}

};
const tailOf = ( text ) => text.split( '\n' ).slice( - 20 ).join( '\n' );
const jobRunning = ( f ) => spawnSync( 'node', [ jobTool, 'status', f, '--lines', '0' ], { stdio: 'ignore' } ).status === 124;

// Starts vite on `p` and waits for it: 'ready', 'exited' (its job ended, the log says why) or
// 'timeout'.
async function serve( p, log ) {

	mkdirSync( dirname( log ), { recursive: true } );
	const vite = `${ SERVE.vite[ mode ] } --host 127.0.0.1 --port ${ p } --strictPort`;
	const started = spawnSync( 'node', [ jobTool, 'start', '--log', log, '--', vite ], { cwd: repo, stdio: [ 'ignore', 'ignore', 'inherit' ] } );
	if ( started.status !== 0 ) fail( `job.mjs start failed (exit ${ started.status })` );

	const ready = new RegExp( `Local:\\s+http://127\\.0\\.0\\.1:${ p }/` );
	const deadline = Date.now() + 60000;
	while ( Date.now() < deadline ) {

		const text = readLog( log );
		if ( /^\[done\] /m.test( text ) ) return 'exited';
		if ( ready.test( text ) && jobRunning( log ) ) {

			try {

				await fetch( `http://127.0.0.1:${ p }/`, { signal: AbortSignal.timeout( 5000 ) } );
				// Checked again after the answer: a job that died meanwhile was not the one answering.
				if ( jobRunning( log ) ) return 'ready';

			} catch {}

		}

		await new Promise( ( r ) => setTimeout( r, 500 ) );

	}

	return 'timeout';

}

for ( ;; ) {

	const log = join( outputDir( caller, 'jobs' ), `serve-${ mode }${ tag ? `-sha-${ tag }` : '' }-${ port }.log` );
	const url = `http://127.0.0.1:${ port }/`;
	const result = await serve( port, log );
	if ( result === 'ready' ) {

		console.log( `${ url }  log ${ log }  (stop: node tools/kit/job.mjs stop ${ log })` );
		break;

	}

	const text = readLog( log );
	if ( result === 'timeout' ) {

		spawnSync( 'node', [ jobTool, 'stop', log ], { stdio: 'ignore' } );
		fail( `${ url } did not answer within 60 s (job stopped); log ${ log }:\n${ tailOf( text ) }` );

	}

	if ( ! /already in use/.test( text ) ) fail( `vite exited before serving ${ url }; log ${ log }:\n${ tailOf( text ) }` );
	if ( fixed ) fail( `port ${ port } was taken before vite bound it, by ${ holder( port ) }` );
	console.log( `port ${ port } was taken before vite bound it (by ${ holder( port ) }), trying the next` );
	const next = await nextFree( port + 1 );
	if ( next === null ) fail( `no free port in ${ port + 1 }-${ LAST_PORT }` );
	port = next;

}
