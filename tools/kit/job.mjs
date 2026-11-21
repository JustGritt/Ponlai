// Runs long commands as jobs with their output in a log, waits on jobs or a condition, and stops jobs.
//
// It is the one waiting tool: every wait goes through it, never a sleep or polling loop of one's
// own. A command that may run longer than a few minutes (a capture set, a profile, a build of the
// references) runs as a job; whoever waits for it, or for anything else, waits here. Every job
// writes the same start and end lines, and `wait` knows a job has ended even when it was killed
// before it could say so, so no wait hangs on a line that never comes. Every command also starts
// the disk clean in the background when one is due (tools/kit/storage.mjs, every 6 h): every
// session waits through here, so the clean needs no schedule of its own.
//
// Usage:
//   node tools/kit/job.mjs run [--log <file>] -- <command ...>
//   node tools/kit/job.mjs start [--log <file>] -- <command ...>
//   node tools/kit/job.mjs start --wait [--lines 20] [--role agent|main] [--max s] [--log <file>] -- <command ...>
//   node tools/kit/job.mjs wait <file>... [--role agent|main] [--max s] [--lines 5] [--tail <file> [n]]
//   node tools/kit/job.mjs wait [--role agent|main] [--max s] [condition] [--tail <file> [n]]
//   node tools/kit/job.mjs status <file>... [--lines 5]
//   node tools/kit/job.mjs list [--cwd <dir> | --all] [--lines 5]
//   node tools/kit/job.mjs stop <file>... | --cwd <dir> | --all [--grace 10]
//   node tools/kit/job.mjs start --wait -- npm run build
//
// Options:
//   --log file           the job's log (default <repo top>/artifacts/jobs/<UTC stamp>-<command>.log;
//                        the folder is the configuration's `output.jobs`)
//   --wait               with start: waits on the job within the budget
//   --lines n            the log's last lines shown (start --wait 20; wait, status, list 5)
//   --role agent|main    the budget: agent 240 s (the default), main 3300 s
//   --max s              lowers the budget to s seconds (it never raises it)
//   --tail file [n]      with wait: prints file's last n lines (20) at the end
//   --file F --match RE  condition: until file F has a line matching RE
//   --exists F           condition: until F exists
//   --pid N              condition: until process N has exited
//   --cmd 'shell'        condition: until the shell command exits 0 (each run killed at the deadline)
//   --cwd dir            list or stop the jobs in dir's own checkout (nearest `.git` entry)
//   --all                list or stop every job on the machine
//   --grace s            stop: seconds between SIGTERM and SIGKILL (default 10)
//
// Jobs. `run` runs the command in the foreground; `start` runs it detached from the calling shell
// (it outlives the shell) and returns once the start line is written, so a `wait` right after it
// finds the job. A single argument after `--` runs through `sh -c` (pipes, redirections, `&&`);
// several run as they are. A single argument with an `&` inside an unquoted word (a query such as
// `near:like=a&dist=40`) is refused with exit 2, since sh would background half of it: quote that
// word. Without `--log` the log's path is printed. The log starts with `[job] start <time> pid
// <runner pid>: <command>`, then holds the command's stdout and stderr, and ends with `[done] exit
// <code> at <time> after <duration>` (a command killed by a signal exits 128 + its number, one that
// cannot start 127), with ` (stopped)` appended when the runner was signalled. The runner forwards
// SIGINT, SIGTERM and SIGHUP to the command and still writes the end line. The command runs with
// the application policy (tools/kit/qos.mjs).
//
// `start --wait` starts the job the same way, then waits on it as `wait` does: when it ends within
// the budget it prints the log's last `--lines` lines and exits with the command's code; otherwise
// it exits 124 and prints the `wait` command to call next. Nothing is killed at the budget, and the
// output goes to the log from the start, so no child holds the caller's pipe (a Bash tool call
// returns as soon as the budget ends): a command that may be short or long needs no guess.
//
// Budgets. Every wait ends at a wall-clock deadline counted from this process's start, whatever the
// load, because a wait written as "N times sleep S" stretches on a busy machine. `--role agent`
// waits at most 240 s: an agent's prompt cache lives five minutes between its requests, so no
// single wait may outlive it; an agent runs it in the foreground with the Bash tool's timeout above
// that, one wait per message, and calls it again until the jobs end. `--role main` waits at most
// 3300 s, under a one-hour prompt cache: a session with that cache can keep a plain
// `wait --role main` running in the background as a keep-alive while it waits on agents or jobs,
// and re-arm it each time it fires.
//
// `wait <file>...` returns as soon as every job has ended. A job has ended when its log's last line
// is the end line, or when its runner is gone without one (killed: reported as lost, exit 1). A file
// without a start line is not a job and fails at once (exit 2). Wait on several jobs with one call.
// `wait` without job logs waits on one condition instead (logs and a condition do not mix), and
// with neither it is a plain timer (a keep-alive). `status` is `wait` with no
// budget. `list` shows the running jobs started under the current repo top (default), under a
// directory, or all of them, with the `status` screen; `--cwd dir` never matches a worktree nested
// under dir.
//
// `stop` exists so that an agent hands back with nothing of its own running. It snapshots each
// job's process tree (a child that detached into its own group is still found, while its parent
// lives), sends SIGTERM to the runner, which passes it to the command so tools run their own
// cleanup (lock release, browser close), and after `--grace` seconds SIGKILLs every survivor of the
// snapshot and anything newly under it.
//
// Output: `wait` and `status` print one screen: per job its state and elapsed time, where it stands
// in the GPU lock (holding a slot or the lock alone, or queued: its place, behind whom; found
// through the job's process tree, so a capture several wrappers down counts) and its log's last
// lines (the text after a line's last carriage return, so progress bars show their latest state),
// then one line for the lock as a whole. A condition's wait prints `done after Xs`, `still waiting
// after Xs` or `timer ended after Xs`, each with the budget and role. `stop` prints what it
// stopped. Every `run` and `start` registers itself in `<tmp>/ponlai-jobs/<runner pid>.json` (the
// log, the runner's and the command's pids, cwd, command, start time; the folder's name is the
// configuration's `machine.jobs`); `<tmp>` is the GPU lock's temporary root, so a scratch
// `<prefix>GPU_LOCK_DIR` (`<prefix>`: the configuration's `machine.envPrefix`) moves the registry with the lock. The runner removes its entry after the end
// line; `list` and `stop` prune entries whose runner is gone.
//
// GPU lock and browser: neither; it reads the lock's state to show where a job stands. Cost: under
// a second, plus the wait.
//
// Exit codes: 0 done (every job succeeded, a condition held, the timer ended, the jobs stopped);
// the first failing job's code; 124 still waiting at the budget; 2 a file that is not a job's log,
// or a command refused for its unquoted `&`; 1 bad arguments, a lost job or a job `stop` could
// not end.
import './help.mjs';
import { closeSync, mkdirSync, openSync, readdirSync, readFileSync, readSync, statSync, unlinkSync, writeFileSync, writeSync, existsSync } from 'node:fs';
import { execFileSync, spawn } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { estimateWaits, etaText, LOCK_DIR, lockState } from './gpu-lock.mjs';
import { appPolicy } from './qos.mjs';
import { autoClean } from './storage.mjs';
import PROJECT from '../project.mjs';
import { outputDir } from './root.mjs';

const BUDGETS = { agent: 240, main: 3300 };
const STILL_RUNNING = 124;
const POLL_MS = 1000;
const CONDITION_POLL_MS = 2000;
const REGISTRY = path.join( path.dirname( LOCK_DIR ), PROJECT.machine.jobs );
const BEGIN = performance.timeOrigin;

const sleep = ( ms ) => new Promise( ( r ) => setTimeout( r, Math.max( 0, ms ) ) );

const duration = ( ms ) => {

	const s = Math.round( ms / 1000 );
	if ( s < 60 ) return `${ s } s`;
	if ( s < 3600 ) return `${ Math.floor( s / 60 ) }m ${ String( s % 60 ).padStart( 2, '0' ) }s`;
	return `${ Math.floor( s / 3600 ) }h ${ String( Math.floor( s / 60 ) % 60 ).padStart( 2, '0' ) }m`;

};

function alive( pid ) {

	try {

		process.kill( pid, 0 );
		return true;

	} catch ( error ) {

		return error.code === 'EPERM';

	}

}

function repoTop( dir ) {

	try {

		return execFileSync( 'git', [ 'rev-parse', '--show-toplevel' ], { cwd: dir, encoding: 'utf8', stdio: [ 'ignore', 'pipe', 'ignore' ] } ).trim();

	} catch {

		return dir;

	}

}

// <repo top>/artifacts/jobs/<UTC stamp>-<slug>.log (the folder: the configuration's `output.jobs`),
// numbered when the same command starts twice in a second
function defaultLog( cmd ) {

	const stamp = new Date().toISOString().replace( /[-:]/g, '' ).replace( /\.\d+Z$/, 'Z' );
	const slug = cmd.join( ' ' ).replace( /\S*\//g, '' ).toLowerCase().replace( /[^a-z0-9]+/g, '-' ).replace( /^-|-$/g, '' ).slice( 0, 48 ).replace( /-$/, '' ) || 'job';
	const dir = outputDir( repoTop( process.cwd() ), 'jobs' );
	let file = path.join( dir, `${ stamp }-${ slug }.log` );
	for ( let i = 2; existsSync( file ); i ++ ) file = path.join( dir, `${ stamp }-${ slug }-${ i }.log` );
	return file;

}

// --- registry --------------------------------------------------------------------------------

const entryFile = ( runner ) => path.join( REGISTRY, `${ runner }.json` );

function register( entry ) {

	mkdirSync( REGISTRY, { recursive: true } );
	writeFileSync( entryFile( entry.runner ), JSON.stringify( entry ) + '\n' );

}

function unregister( runner ) {

	try {

		unlinkSync( entryFile( runner ) );

	} catch { /* already pruned */ }

}

// the running jobs' entries; those whose runner is gone are removed
function registry() {

	let files = [];
	try {

		files = readdirSync( REGISTRY ).filter( ( f ) => f.endsWith( '.json' ) );

	} catch {

		return [];

	}
	const entries = [];
	for ( const f of files ) {

		let entry;
		try {

			entry = JSON.parse( readFileSync( path.join( REGISTRY, f ), 'utf8' ) );

		} catch {

			continue; // being written

		}
		if ( alive( entry.runner ) ) entries.push( entry );
		else unregister( entry.runner );

	}
	return entries.sort( ( a, b ) => a.start.localeCompare( b.start ) );

}

const under = ( dir, root ) => dir === root || dir.startsWith( root.endsWith( path.sep ) ? root : root + path.sep );

// The checkout a directory belongs to: the nearest ancestor (or itself) holding a `.git` entry,
// a directory or a worktree's file, so a nested worktree counts as another checkout.
function checkoutOf( dir ) {

	for ( let d = dir; ; d = path.dirname( d ) ) {

		if ( existsSync( path.join( d, '.git' ) ) ) return d;
		if ( path.dirname( d ) === d ) return null;

	}

}

// --- run -------------------------------------------------------------------------------------

async function run( log, cmd ) {

	log = path.resolve( log );
	mkdirSync( path.dirname( log ), { recursive: true } );
	const start = Date.now();
	writeFileSync( log, `[job] start ${ new Date( start ).toISOString() } pid ${ process.pid }: ${ cmd.join( ' ' ) }\n` );
	const fd = openSync( log, 'a' );
	const shell = cmd.length === 1;
	// the command runs at the application policy too: children do not inherit it (tools/kit/qos.mjs)
	const child = spawn( ...appPolicy( shell ? '/bin/sh' : cmd[ 0 ], shell ? [ '-c', cmd[ 0 ] ] : cmd.slice( 1 ) ), { stdio: [ 'ignore', fd, fd ] } );
	// a signal to the runner is how `stop` (or a Ctrl-C) ends a job: the end line says so
	let stopped = false;
	for ( const signal of [ 'SIGINT', 'SIGTERM', 'SIGHUP' ] ) process.on( signal, () => {

		stopped = true;
		child.kill( signal );

	} );
	if ( child.pid ) register( { log, runner: process.pid, pid: child.pid, cwd: process.cwd(), command: cmd.join( ' ' ), start: new Date( start ).toISOString() } );
	const code = await new Promise( ( resolve ) => {

		child.on( 'error', ( error ) => {

			writeSync( fd, `${ error.message }\n` );
			resolve( 127 );

		} );
		child.on( 'exit', ( exitCode, signal ) => resolve( exitCode ?? 128 + ( os.constants.signals[ signal ] ?? 0 ) ) );

	} );
	const end = Date.now();
	writeSync( fd, `[done] exit ${ code } at ${ new Date( end ).toISOString() } after ${ duration( end - start ) }${ stopped ? ' (stopped)' : '' }\n` );
	closeSync( fd );
	unregister( process.pid );
	return code;

}

async function startDetached( log, cmd, quiet ) {

	log = path.resolve( log );
	const runner = spawn( process.execPath, [ fileURLToPath( import.meta.url ), 'run', '--log', log, '--', ...cmd ], { detached: true, stdio: 'ignore' } );
	runner.unref();
	for ( let i = 0; i < 100 && readJob( log )?.pid !== runner.pid; i ++ ) await sleep( 50 );
	console.log( `[job] started ${ log } (runner pid ${ runner.pid })${ quiet ? '' : `; wait with: node tools/kit/job.mjs wait ${ log }` }` );
	return log;

}

// --- reading a job ---------------------------------------------------------------------------

function readRange( file, from, length ) {

	const fd = openSync( file, 'r' );
	try {

		const buffer = Buffer.alloc( length );
		return buffer.subarray( 0, readSync( fd, buffer, 0, length, from ) ).toString( 'utf8' );

	} finally {

		closeSync( fd );

	}

}

// The job a log describes: { pid, start, command, ended, code, lost, endText }, or null when the
// file is missing or does not start with a start line.
function readJob( log ) {

	let size;
	try {

		size = statSync( log ).size;

	} catch {

		return null;

	}
	const head = readRange( log, 0, Math.min( size, 4096 ) ).split( '\n' )[ 0 ];
	const m = /^\[job\] start (\S+) pid (\d+): (.*)$/.exec( head );
	if ( ! m ) return null;
	const job = { pid: Number( m[ 2 ] ), start: Date.parse( m[ 1 ] ), command: m[ 3 ], size };
	const tail = readRange( log, Math.max( 0, size - 65536 ), Math.min( size, 65536 ) ).split( '\n' );
	if ( tail.at( - 1 ) === '' ) tail.pop();
	job.tail = tail;
	const end = /^\[done\] exit (\d+) at (\S+) after (.*)$/.exec( tail.at( - 1 ) ?? '' );
	if ( end ) Object.assign( job, { ended: true, code: Number( end[ 1 ] ), endText: `exit ${ end[ 1 ] } after ${ end[ 3 ] }` } );
	else if ( ! alive( job.pid ) ) Object.assign( job, { ended: true, lost: true, code: 1, endText: 'lost: its runner is gone without an end line (killed)' } );
	return job;

}

// --- status ----------------------------------------------------------------------------------

// pid -> parent pid for every process, read once
function parents() {

	const parent = new Map();
	for ( const line of execFileSync( 'ps', [ '-A', '-o', 'pid=,ppid=' ], { encoding: 'utf8' } ).split( '\n' ) ) {

		const [ pid, ppid ] = line.trim().split( /\s+/ ).map( Number );
		if ( pid ) parent.set( pid, ppid );

	}
	return parent;

}

function processTree() {

	const parent = parents();
	return ( pid, root ) => {

		for ( let p = pid, hops = 0; p > 1 && hops < 64; p = parent.get( p ), hops ++ ) if ( p === root ) return true;
		return false;

	};

}

// every process below the roots (the roots themselves excluded)
function descendants( roots ) {

	const children = new Map();
	for ( const [ pid, ppid ] of parents() ) children.set( ppid, [ ...( children.get( ppid ) ?? [] ), pid ] );
	const found = new Set();
	const stack = [ ...roots ];
	while ( stack.length ) for ( const c of children.get( stack.pop() ) ?? [] ) if ( ! found.has( c ) ) {

		found.add( c );
		stack.push( c );

	}
	return found;

}

const names = ( list, max = 3 ) => list.slice( 0, max ).map( ( e ) => e.label ).join( ', ' ) + ( list.length > max ? `, +${ list.length - max } more` : '' );

function lockLine( job, lock, below ) {

	const held = lock.holders.filter( ( h ) => below( h.pid, job.pid ) );
	const waiting = lock.queue.map( ( t, i ) => ( { t, i } ) ).filter( ( { t } ) => below( t.pid, job.pid ) );
	const level = ( t ) => `${ t.level }${ t.source === 'table' ? ` from the table's "${ t.match }"` : t.source === 'default' ? '' : ` from ${ t.source }` }`;
	const parts = held.map( ( h ) => h.mode === 'exclusive' ? `holding the lock alone (${ h.label }, ${ level( h ) })` : `holding shared slot ${ h.slot ?? '?' } (${ h.label }, ${ level( h ) })` );
	// when a queued job should start, from gpu-lock's run of the queue forward (an estimate)
	const estimate = waiting.length ? estimateWaits( { state: lock } ) : null;
	for ( const { t, i } of waiting ) {

		const ahead = lock.queue.slice( 0, i );
		const others = lock.holders.filter( ( h ) => ! below( h.pid, job.pid ) );
		parts.push( `queued #${ i + 1 } of ${ lock.queue.length } (${ t.mode === 'exclusive' ? 'exclusive' : 'shared' }, ${ level( t ) }, ${ duration( Date.now() - Date.parse( t.start ) ) })` +
			( ahead.length ? `, behind ${ names( ahead ) }` : '' ) + ( others.length ? `; held by ${ names( others ) }` : '' ) +
			`; estimated start in ${ etaText( estimate.queue.find( ( q ) => q.pid === t.pid )?.eta ) }` );

	}
	return parts.length ? parts.join( '; ' ) : 'not in it';

}

function lockSummary( lock ) {

	const exclusive = lock.holders.find( ( h ) => h.mode === 'exclusive' );
	const held = exclusive ? `held alone by ${ exclusive.label } for ${ duration( Date.now() - Date.parse( exclusive.start ) ) }`
		: lock.holders.length ? `${ lock.holders.length }/${ lock.slots } shared slots held (${ names( lock.holders ) })` : 'free';
	return `gpu lock: ${ held }${ lock.queue.length ? `, ${ lock.queue.length } queued` : '' }`;

}

// the command's last lines: without the start line (in the tail when the log is short) and the end line
function lastLines( job, n ) {

	const body = job.tail.filter( ( line, i ) => ! ( i === 0 && job.size <= 65536 ) && ! ( i === job.tail.length - 1 && job.ended && ! job.lost ) );
	return body.map( ( line ) => line.replace( /\r+$/, '' ) ).map( ( line ) => line.slice( line.lastIndexOf( '\r' ) + 1 ) ).filter( ( line ) => line.trim() ).slice( - n )
		.map( ( line ) => line.length > 200 ? line.slice( 0, 199 ) + '…' : line );

}

function report( logs, jobs, lines ) {

	const lock = lockState();
	const below = processTree();
	for ( const [ i, log ] of logs.entries() ) {

		const job = jobs[ i ];
		const state = job.ended ? job.endText : `running ${ duration( Date.now() - job.start ) } (runner pid ${ job.pid })`;
		console.log( `${ log }: ${ state }` );
		if ( ! job.ended ) console.log( `  gpu lock: ${ lockLine( job, lock, below ) }` );
		for ( const line of lastLines( job, lines ) ) console.log( `  | ${ line }` );

	}
	console.log( lockSummary( lock ) );

}

// --- wait ------------------------------------------------------------------------------------

const left = ( budget ) => BEGIN + budget * 1000 - Date.now();
const secs = () => `${ Math.round( ( Date.now() - BEGIN ) / 1000 ) }s`;

// polls until every job has ended or the budget is spent; null when a file is not a job's log
async function pollJobs( logs, budget ) {

	for ( ;; ) {

		const jobs = logs.map( readJob );
		const missing = logs.filter( ( log, i ) => ! jobs[ i ] );
		if ( missing.length ) {

			console.error( `not a job log (no start line): ${ missing.join( ', ' ) }; start jobs with node tools/kit/job.mjs run|start -- <command>` );
			return null;

		}
		if ( jobs.every( ( job ) => job.ended ) || left( budget ) <= 0 ) return jobs;
		await sleep( Math.min( POLL_MS, left( budget ) ) );

	}

}

async function waitJobs( logs, budget, lines ) {

	const jobs = await pollJobs( logs, budget );
	if ( ! jobs ) return 2;
	const waited = duration( Date.now() - BEGIN );
	report( logs, jobs, lines );
	const running = jobs.filter( ( job ) => ! job.ended ).length;
	if ( running ) {

		console.log( `[wait] ${ running } of ${ jobs.length } still running after ${ waited } (budget ${ budget } s): wait again` );
		return STILL_RUNNING;

	}
	const failed = jobs.find( ( job ) => job.code !== 0 );
	console.log( `[wait] ${ jobs.length > 1 ? `all ${ jobs.length } jobs` : 'the job' } ended (waited ${ waited })${ failed ? `; failed: exit ${ failed.code }` : '' }` );
	return failed ? failed.code : 0;

}

// --cmd runs in its own process group, so the deadline (or this process being stopped) kills the
// command and everything it started
let probe = null;
const killProbe = () => {

	if ( probe ) {

		try {

			process.kill( - probe.pid, 'SIGKILL' );

		} catch { /* gone */ }
		probe = null;

	}

};

function runProbe( shell, budget ) {

	return new Promise( ( resolve ) => {

		probe = spawn( '/bin/sh', [ '-c', shell ], { detached: true, stdio: 'ignore' } );
		const timer = setTimeout( () => {

			killProbe();
			resolve( false );

		}, Math.max( 0, left( budget ) ) );
		probe.on( 'exit', ( code ) => {

			clearTimeout( timer );
			probe = null;
			resolve( code === 0 );

		} );
		probe.on( 'error', () => {

			clearTimeout( timer );
			probe = null;
			resolve( false );

		} );

	} );

}

async function waitCondition( options, budget, role ) {

	let met = null;
	if ( options.file !== undefined ) {

		const re = new RegExp( options.match, 'm' );
		met = () => {

			try {

				return re.test( readFileSync( options.file, 'utf8' ) );

			} catch {

				return false;

			}

		};

	} else if ( options.exists !== undefined ) met = () => existsSync( options.exists );
	else if ( options.pid !== undefined ) met = () => ! alive( Number( options.pid ) );
	else if ( options.cmd !== undefined ) met = () => left( budget ) > 0 && runProbe( options.cmd, budget );

	let done = false;
	if ( met ) while ( ! ( done = await met() ) && left( budget ) > 0 ) await sleep( Math.min( CONDITION_POLL_MS, left( budget ) ) );
	else await sleep( left( budget ) );

	const what = ! met ? 'timer ended' : done ? 'done' : 'still waiting';
	console.log( `${ what } after ${ secs() }, budget ${ budget }s (${ role })` );
	printTail( options );
	return ! met || done ? 0 : STILL_RUNNING;

}

function printTail( options ) {

	if ( options.tail === undefined ) return;
	try {

		console.log( readFileSync( options.tail, 'utf8' ).trimEnd().split( '\n' ).slice( - options.tailLines ).join( '\n' ) );

	} catch {

		console.log( `(no ${ options.tail })` );

	}

}

async function startAndWait( log, cmd, budget, lines ) {

	log = await startDetached( log, cmd, true );
	const [ job ] = await pollJobs( [ log ], budget ) ?? [];
	if ( ! job ) return 2;
	if ( ! job.ended ) {

		report( [ log ], [ job ], 5 );
		console.log( `still running: node tools/kit/job.mjs wait ${ log }` );
		return STILL_RUNNING;

	}
	for ( const line of lastLines( job, lines ) ) console.log( line );
	console.log( `[job] ${ log }: ${ job.endText }` );
	return job.code;

}

// --- list and stop ---------------------------------------------------------------------------

function scoped( options ) {

	const entries = registry();
	if ( options.all ) return entries;
	const root = options.cwd !== undefined ? path.resolve( options.cwd ) : repoTop( process.cwd() );
	const top = checkoutOf( root );
	return entries.filter( ( e ) => under( e.cwd, root ) && checkoutOf( e.cwd ) === top );

}

function list( options, lines ) {

	const entries = scoped( options );
	if ( ! entries.length ) {

		console.log( `no running jobs ${ options.all ? '' : `under ${ options.cwd !== undefined ? path.resolve( options.cwd ) : repoTop( process.cwd() ) }` }`.trim() );
		return 0;

	}
	const logs = entries.map( ( e ) => e.log );
	const jobs = logs.map( readJob );
	// an entry whose log was removed still shows, as its registry entry
	for ( const [ i, e ] of entries.entries() ) jobs[ i ] ??= { pid: e.runner, start: Date.parse( e.start ), tail: [], size: 0 };
	for ( const e of entries ) console.log( `[job] runner ${ e.runner }, command pid ${ e.pid }, in ${ e.cwd }: ${ e.command }` );
	report( logs, jobs, lines );
	return 0;

}

async function stop( files, options ) {

	const grace = Math.max( 0, Number( options.grace ?? 10 ) );
	const entries = registry();
	const targets = [];
	if ( files.length ) {

		for ( const file of files ) {

			const log = path.resolve( file );
			const job = readJob( log );
			if ( ! job ) {

				console.error( `not a job log (no start line): ${ file }` );
				return 2;

			}
			if ( job.ended ) console.log( `${ log }: already ended (${ job.endText })` );
			else targets.push( entries.find( ( e ) => e.log === log ) ?? { log, runner: job.pid } );

		}

	} else targets.push( ...scoped( options ) );
	if ( ! targets.length ) {

		if ( ! files.length ) console.log( 'no running jobs to stop' );
		return 0;

	}

	// the trees as they stand before the signal: a child that detached into a group of its own is
	// found now, while its parent still lives, and reparented to launchd once that parent dies
	const tree = new Map( targets.map( ( t ) => [ t, descendants( [ t.runner ] ) ] ) );
	for ( const t of targets ) {

		try {

			process.kill( t.runner, 'SIGTERM' );

		} catch { /* ended meanwhile */ }

	}
	const until = Date.now() + grace * 1000;
	const settled = () => targets.every( ( t ) => readJob( t.log )?.ended && [ ...tree.get( t ) ].every( ( pid ) => ! alive( pid ) ) );
	while ( ! settled() && Date.now() < until ) await sleep( 200 );

	let failed = 0;
	for ( const t of targets ) {

		const snapshot = [ ...tree.get( t ) ].filter( alive );
		const survivors = [ ...new Set( [ ...snapshot, ...descendants( [ ...snapshot, t.runner ] ) ] ) ];
		for ( const pid of survivors ) {

			try {

				process.kill( pid, 'SIGKILL' );

			} catch { /* gone */ }

		}
		// the runner writes its end line once its command is gone; one that cannot is killed too
		for ( let i = 0; i < 25 && ! readJob( t.log )?.ended; i ++ ) await sleep( 200 );
		if ( ! readJob( t.log )?.ended ) {

			try {

				process.kill( t.runner, 'SIGKILL' );

			} catch { /* gone */ }
			await sleep( 200 );

		}
		const job = readJob( t.log );
		if ( ! job?.ended ) failed ++;
		console.log( `stopped ${ t.log }: ${ job?.endText ?? 'log gone' }${ survivors.length ? `; killed ${ survivors.length } survivor(s) after ${ grace } s: ${ survivors.join( ' ' ) }` : '' }` );

	}
	registry(); // prunes the entries of runners that were killed
	return failed ? 1 : 0;

}

// --- command line ----------------------------------------------------------------------------

const USAGE = `usage: node tools/kit/job.mjs
  run|start [--log file] -- command...
  start --wait [--lines n] [--role agent|main] [--max s] [--log file] -- command...
  wait file... [--role agent|main] [--max s] [--lines n]
  wait [--role agent|main] [--max s] [--file F --match RE | --exists F | --pid N | --cmd 'shell'] [--tail F [n]]
  status file... [--lines n]
  list [--cwd dir | --all] [--lines n]
  stop file... | --cwd dir | --all [--grace s]`;

// the flags each command takes; true marks a flag without a value
const FLAGS = {
	run: { log: false },
	start: { log: false, wait: true, lines: false, role: false, max: false },
	wait: { role: false, max: false, lines: false, file: false, match: false, exists: false, pid: false, cmd: false, tail: false },
	status: { lines: false },
	list: { cwd: false, all: true, lines: false },
	stop: { cwd: false, all: true, grace: false },
};

class UsageError extends Error {}

function parse( command, args ) {

	const flags = FLAGS[ command ];
	if ( ! flags ) throw new UsageError( command ? `unknown command ${ command }` : 'no command' );
	const options = {}, files = [];
	for ( let i = 0; i < args.length; i ++ ) {

		const a = args[ i ];
		if ( ! a.startsWith( '--' ) ) {

			files.push( a );
			continue;

		}
		const name = a.slice( 2 );
		if ( ! ( name in flags ) ) throw new UsageError( `${ command } does not take ${ a }` );
		if ( flags[ name ] ) {

			options[ name ] = true;
			continue;

		}
		const v = args[ i + 1 ];
		if ( v === undefined || ( v.startsWith( '--' ) && v.slice( 2 ) in flags ) ) throw new UsageError( `${ a } needs a value` );
		options[ name ] = v;
		i ++;
		if ( name === 'tail' ) options.tailLines = /^\d+$/.test( args[ i + 1 ] ?? '' ) ? Number( args[ ++ i ] ) : 20;

	}
	return { options, files };

}

function budgetOf( options ) {

	const role = options.role ?? 'agent';
	if ( ! ( role in BUDGETS ) ) throw new UsageError( `--role is agent or main, not ${ role }` );
	let budget = BUDGETS[ role ];
	if ( options.max !== undefined ) {

		const max = Number( options.max );
		if ( ! ( max >= 0 ) ) throw new UsageError( `--max needs a number of seconds, not ${ options.max }` );
		budget = Math.min( budget, max );

	}
	return { role, budget };

}

const count = ( value, fallback, flag ) => {

	if ( value === undefined ) return fallback;
	const n = Number( value );
	if ( ! ( Number.isInteger( n ) && n >= 0 ) ) throw new UsageError( `${ flag } needs a whole number, not ${ value }` );
	return n;

};

// A single-string command runs through sh -c, where an unquoted & inside a word (a query such as
// near:like=a&dist=40) backgrounds the first half and drops the rest without a word. Returns
// the word holding such an &, or null; 2>&1, &&, `cmd &` and `a & b` pass.
function splitAmpersand( line ) {

	let quote = null;
	for ( let i = 0; i < line.length; i ++ ) {

		const c = line[ i ];
		if ( quote ) {

			if ( c === quote ) quote = null;
			else if ( c === '\\' && quote === '"' ) i ++;
			continue;

		}
		if ( c === '\\' ) { i ++; continue; }
		if ( c === "'" || c === '"' ) { quote = c; continue; }
		if ( c !== '&' || i === 0 || /[\s&><|]/.test( line[ i - 1 ] ) || ! /\w/.test( line[ i + 1 ] ?? '' ) ) continue;
		let a = i, b = i;
		while ( a > 0 && ! /\s/.test( line[ a - 1 ] ) ) a --;
		while ( b < line.length && ! /\s/.test( line[ b ] ) ) b ++;
		return line.slice( a, b );

	}
	return null;

}

async function main( argv ) {

	// every session's waits come through here: the disk clean rides on them (tools/kit/storage.mjs)
	autoClean();
	const [ command, ...rest ] = argv;
	const split = rest.indexOf( '--' );
	const args = split >= 0 ? rest.slice( 0, split ) : rest;
	const cmd = split >= 0 ? rest.slice( split + 1 ) : [];
	const { options, files } = parse( command, args );

	if ( command === 'run' || command === 'start' ) {

		if ( ! cmd.length || files.length ) throw new UsageError( `${ command } takes its command after --` );
		const split = cmd.length === 1 ? splitAmpersand( cmd[ 0 ] ) : null;
		if ( split ) {

			console.error( `job: the shell would split \`${ split }\` at its unquoted &: quote that word ('${ split }')` );
			return 2;

		}
		if ( ! options.wait && ( options.lines !== undefined || options.role !== undefined || options.max !== undefined ) ) throw new UsageError( '--lines, --role and --max go with start --wait' );
		const budget = options.wait ? budgetOf( options ).budget : 0;
		const lines = options.wait ? count( options.lines, 20, '--lines' ) : 0;
		const log = options.log ?? defaultLog( cmd );
		if ( options.log === undefined ) console.log( `[job] log ${ log }` );
		if ( command === 'run' ) return run( log, cmd );
		if ( options.wait ) return startAndWait( log, cmd, budget, lines );
		await startDetached( log, cmd, false );
		return 0;

	}
	if ( cmd.length ) throw new UsageError( `${ command } takes no command after --` );
	const lines = count( options.lines, 5, '--lines' );

	if ( command === 'wait' ) {

		const { role, budget } = budgetOf( options );
		const conditions = [ 'file', 'exists', 'pid', 'cmd' ].filter( ( k ) => options[ k ] !== undefined );
		if ( ( options.file === undefined ) !== ( options.match === undefined ) ) throw new UsageError( '--file and --match go together' );
		if ( conditions.length > 1 ) throw new UsageError( `one condition at a time, not ${ conditions.join( ' and ' ) }` );
		if ( conditions.length && files.length ) throw new UsageError( 'wait takes job logs or a condition, not both' );
		if ( options.match !== undefined ) {

			try {

				new RegExp( options.match, 'm' );

			} catch ( error ) {

				throw new UsageError( `--match: ${ error.message }` );

			}

		}
		if ( options.pid !== undefined && ! ( Number.isInteger( Number( options.pid ) ) && Number( options.pid ) > 0 ) ) throw new UsageError( `--pid needs a process id, not ${ options.pid }` );
		if ( files.length ) {

			const code = await waitJobs( files, budget, lines );
			printTail( options );
			return code;

		}
		return waitCondition( options, budget, role );

	}
	if ( command === 'status' ) {

		if ( ! files.length ) throw new UsageError( 'status takes job logs' );
		return waitJobs( files, 0, lines );

	}
	if ( command === 'list' ) {

		if ( files.length || ( options.all && options.cwd !== undefined ) ) throw new UsageError( 'list takes --cwd or --all, not both, and no files' );
		return list( options, lines );

	}
	// stop
	if ( [ files.length > 0, options.all === true, options.cwd !== undefined ].filter( Boolean ).length !== 1 ) throw new UsageError( 'stop takes job logs, --cwd or --all (one of them)' );
	return stop( files, options );

}

for ( const signal of [ 'SIGTERM', 'SIGINT', 'SIGHUP' ] ) if ( process.argv[ 2 ] !== 'run' ) process.on( signal, () => {

	killProbe();
	process.exit( 1 );

} );

main( process.argv.slice( 2 ) ).then( ( code ) => process.exit( code ), ( error ) => {

	console.error( error instanceof UsageError ? `${ error.message }\n${ USAGE }` : error.message );
	process.exit( 1 );

} );
