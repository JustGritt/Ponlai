// Runs a command under the machine-wide GPU lock and reports, schedules and explains the lock's queue.
//
// Everything that renders the scene in a browser takes this lock: timing runs hold it alone, paused
// captures and probes share its slots. The lock's rules (the modes, the shared slots, the queue,
// the levels and how a request's level is decided, ageing, leases, inheritance below a holder, older
// copies of the lock) are in docs/reference/gpu-lock.md; this header is the command line and the
// entry that loads the lock. Tools take the lock through this file's exports (`acquireGpuLock`,
// `withGpuLock`, `leaseGpuLock`, ...; their forms: tools/kit/gpu-lock-impl.mjs's header).
//
// Usage:
//   node tools/kit/gpu-lock.mjs run [--shared] [--label name] [--timeout seconds] -- <command ...>
//   node tools/kit/gpu-lock.mjs status
//   node tools/kit/gpu-lock.mjs eta [--shared|--exclusive] [--level l] [--label name] [--pid p] [--json]
//   node tools/kit/gpu-lock.mjs prio [--json]
//   node tools/kit/gpu-lock.mjs prio set <match> <level> [--for 90m] [--note text]
//   node tools/kit/gpu-lock.mjs prio clear <match> | prio bump <pid|#ticket> <level>
//   node tools/kit/gpu-lock.mjs gpu [--seconds 3]
//   node tools/kit/gpu-lock.mjs log [--since 2h] [--by class|label] [--top 5] [--json]
//
// Options:
//   run --shared        takes a shared slot instead of the lock alone (paused captures, probes)
//   run --label name    the request's label (default the command's first 80 characters)
//   run --timeout s     gives up after s seconds of waiting (default 0: waits for ever)
//   eta --shared        a shared request made now; --exclusive (the default) one alone
//   eta --level l       at that level (urgent, high, normal, low or 0-3; default this shell's)
//   eta --label name    the label of the request made now (the table may match it)
//   eta --pid p         the estimate of p's queued request instead of a new one
//   prio set --for d    the entry lapses after d (90s, 30m, 2h, 1d; default never)
//   prio set --note t   a note kept with the entry
//   gpu --seconds n     how long utilisation is sampled (default 3)
//   log --since d       only the last d (90s, 30m, 2h, 1d; default the whole log)
//   log --by k          groups by `class` (the level, the default) or `label`
//   log --top n         the longest waits listed (default 5)
//   --json              prints JSON only (eta, prio, log)
//
// Commands. `run` waits for the lock, then runs the command with <prefix>GPU_LOCK_HELD and
// <prefix>GPU_LOCK_MODE set (`<prefix>` is the configuration's `machine.envPrefix`, here and below), so the command and anything below it inherit the hold, and with the
// application policy (tools/kit/qos.mjs); it forwards SIGINT, SIGTERM and SIGHUP to the command.
// `status` names the holders (shared: their slots, each with its level and source), the queue in
// turn (each with its estimated start) and the exclusive requests in turn (`waiting`). `eta`
// estimates when a request made now, or a queued one, starts. `prio` shows the scheduling table
// and every holder's and ticket's level with its source; `prio set` adds or replaces an entry and
// lists what it applies to now (it warns on stderr when nothing matches: a typo, or a session that
// has not asked yet); `prio clear` removes one; `prio bump` raises or lowers one request, by pid
// or `#ticket`, and refuses an exclusive holder and an older copy's request. The table ranks other
// runs: a run never adds an entry for itself. `gpu` shows the GPU's utilisation and the
// processes using it (macOS). `log` summarises the lock's history.
//
// A request that waits prints a timestamped line on stderr every 30 s: its level and source, its
// place in the queue, the next in turn, who holds the GPU and why it waits, and its estimated start
// (`estimated start in ~12 min (range 7 min-25 min, estimate)`) when it starts waiting and whenever
// the figure moves by more than a minute or a quarter. The estimate runs the queue forward under
// the lock's rules 200 times, each hold drawn from what holds of that kind took (the log's recent
// releases, or a built-in table until the log has five of a kind); `startIn` is the median start
// and `range` its 10th to 90th percentile. Later requests of a higher level are not foreseen, so a
// request can start after its range. To choose between waiting and handing work back, compare
// `startIn` with the threshold; the range's upper end is cautious.
//
// Output: `status` prints `free (...)` (the lock's folder, the slot count, this shell's levels and
// the file that runs) or one JSON line: the owner, `alive`, `dir`, `impl` (the file that runs and
// why), `slots` (`limit`, `used`, `holders`), `waiting` and `queue`, each queued request with `eta: {
// startIn, range: [ lo, hi ], basis }` in seconds from now (`basis` counts where the durations came
// from). `eta` prints a line (or `{ now, request | pid, eta }`). `prio` prints the table, `holding:`
// and `queue, in turn:`. `gpu` prints `{ utilisation, clients }`. `log` prints per group the
// requests, waits and holds (median, p95, total; lease uses and yields, take-overs, timeouts,
// abandoned and stale holds apart) and the longest waits with the holds that overlapped them. The
// log itself, `ponlai-gpu.lock.log.jsonl` beside the lock (the lock's name is the configuration's
// `machine.gpuLock`; rotated to `.log.1.jsonl` past 4 MB), holds a JSON line per request, grant,
// release, timeout and abandoned request: time, pid, label, mode, `class` (the level), `source`,
// ticket, `ppid` and `cwd`, the wait or hold in ms, and why a hold ended when its holder did not
// end it (`handed`, `yielded`, `stale`, `lease`).
//
// GPU lock and browser: `run` takes the lock (alone, or a slot with --shared) for its command's
// whole run; the other commands take nothing and open no browser. Cost: under a second, except the
// wait for the lock and `gpu`'s sampling.
//
// Exit codes: `run` exits with the command's code (127 when it cannot start, 128 + the signal's
// number when a signal ends it); 0 for the other commands; 1 an error (a bad option or level, a
// `--timeout` that ran out, `prio clear` of no entry, a pid with no ticket); 2 an unknown command.
//
// The entry. Every worktree carries its own copy of the lock, and copies that disagree on the rules
// share one lock, so this file loads whichever implementation declares the higher LOCK_VERSION,
// this worktree's tools/kit/gpu-lock-impl.mjs or the main checkout's (found through `git rev-parse
// --git-common-dir`), and the main checkout's on a tie: a branch runs every later lock upgrade once
// main has it, without merging main.
// - The version is read from the file's text (`export const LOCK_VERSION = n;`), so the copy not
//   chosen is never loaded. A main checkout without gpu-lock-impl.mjs counts as version 0.
// - The main checkout's implementation is tools/kit/gpu-lock-impl.mjs, or tools/gpu-lock-impl.mjs
//   on a main checkout from before the tools were grouped (the first of the two that exists). On a
//   main checkout after then, tools/gpu-lock-impl.mjs is a stand-in that re-exports the kit's.
// - If the main checkout's implementation does not load (a conflict mid-merge, a syntax error, no
//   readable version), this worktree's runs instead, said once on stderr.
// - <prefix>GPU_LOCK_IMPL=local forces this worktree's copy (to test a change to the lock);
//   <prefix>GPU_LOCK_MAIN=<dir> stands in for the main checkout's root (for the entry's own tests).
// A change to the lock goes in gpu-lock-impl.mjs, bumps LOCK_VERSION there and in the stand-in, and
// keeps the exports and the CLI additive: whatever this entry loads provides every name it
// re-exports.
import './help.mjs';
import './qos.mjs'; // first, as the implementation does: it re-executes the process before the choice below
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, realpathSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import PROJECT from '../project.mjs';

const NAME = 'gpu-lock-impl.mjs';
const MAIN_ENV = `${ PROJECT.machine.envPrefix }GPU_LOCK_MAIN`, IMPL_ENV = `${ PROJECT.machine.envPrefix }GPU_LOCK_IMPL`;
const here = path.dirname( fileURLToPath( import.meta.url ) );
const local = path.join( here, NAME );
// where a main checkout keeps the implementation: in the kit, or at the top of tools/ before the
// tools were grouped
const MAIN_PLACES = [ [ 'tools', 'kit', NAME ], [ 'tools', NAME ] ];

// the main checkout's root: the directory holding the repository's common .git
function mainRoot() {

	if ( process.env[ MAIN_ENV ] ) return path.resolve( process.env[ MAIN_ENV ] );
	try {

		const common = execFileSync( 'git', [ 'rev-parse', '--path-format=absolute', '--git-common-dir' ], { cwd: here, encoding: 'utf8', stdio: [ 'ignore', 'pipe', 'ignore' ], timeout: 5000 } ).trim();
		return common ? path.dirname( common ) : null;

	} catch {

		return null;

	}

}

// a file's LOCK_VERSION: 0 when the file is absent (a lock from before the entry), null when it is
// there without a readable version
function versionOf( file ) {

	let text;
	try {

		text = readFileSync( file, 'utf8' );

	} catch {

		return 0;

	}
	const m = /^export const LOCK_VERSION = (\d+);/m.exec( text );
	return m ? Number( m[ 1 ] ) : null;

}

const same = ( a, b ) => {

	try {

		return realpathSync( a ) === realpathSync( b );

	} catch {

		return false;

	}

};

const localVersion = versionOf( local );
let file = local, why;
const root = process.env[ IMPL_ENV ] === 'local' ? null : mainRoot();
const remote = root && ( MAIN_PLACES.map( ( p ) => path.join( root, ...p ) ).find( ( f ) => existsSync( f ) ) ?? path.join( root, ...MAIN_PLACES[ 0 ] ) );
if ( process.env[ IMPL_ENV ] === 'local' ) why = `${ IMPL_ENV }=local`;
else if ( ! remote ) why = 'no main checkout found';
else if ( same( remote, local ) ) why = 'this is the main checkout';
else {

	const mainVersion = versionOf( remote );
	if ( mainVersion === null ) {

		why = `the main checkout's ${ remote } has no readable LOCK_VERSION`;
		console.error( `[gpu-lock] ${ why }: running this worktree's lock` );

	} else if ( mainVersion >= localVersion ) {

		file = remote;
		why = `main checkout's v${ mainVersion } ${ mainVersion === localVersion ? '=' : '>' } this worktree's v${ localVersion }`;

	} else why = `this worktree's v${ localVersion } > the main checkout's v${ mainVersion }`;

}

let impl;
try {

	impl = await import( pathToFileURL( file ).href );

} catch ( error ) {

	if ( file === local ) throw error;
	why = `the main checkout's ${ file } failed to load (${ error.message.split( '\n' )[ 0 ] })`;
	console.error( `[gpu-lock] ${ why }: running this worktree's lock` );
	file = local;
	impl = await import( pathToFileURL( local ).href );

}

/** The implementation that runs: its file, its version and why it was chosen. */
export const IMPL = { file, version: impl.LOCK_VERSION ?? 0, why };

export const {
	LOCK_VERSION, LOCK_DIR, TABLE, SLOTS, LEVELS, parseLevel, defaultLevel, readTable, levelOf, levelRank,
	lockOwner, queueOrder, startableShared, lockState, releaseGpuLock, acquireGpuLock, withGpuLock,
	leaseGpuLock, yieldAncestorSlot, gpuUtilisation, gpuClients, sampleGpuUtilisation, waitForIdleGpu,
	labelKind, lockDurations, estimateWaits, etaText, readLockLog, summariseLockLog,
} = impl;

if ( process.argv[ 1 ] && path.resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url ) ) {

	impl.main( process.argv.slice( 2 ), { impl: `${ file } v${ IMPL.version } (${ why })` } ).then( ( code ) => {

		impl.releaseGpuLock();
		process.exit( code );

	}, ( error ) => {

		console.error( error.message );
		impl.releaseGpuLock();
		process.exit( 1 );

	} );

}
