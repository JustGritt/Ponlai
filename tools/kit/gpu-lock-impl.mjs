// The GPU lock's implementation: the machine-wide queue every tool that renders the scene in a browser waits in.
//
// Tools import the entry, tools/kit/gpu-lock.mjs, which loads this file (its own worktree's copy or
// the main checkout's, whichever declares the higher LOCK_VERSION) and runs its CLI; the commands and
// their options are in that entry's header, and the rules a caller meets (modes, slots, levels, the
// queue, ageing, the scheduling table) in docs/reference/gpu-lock.md. The lock's design (its files,
// the rules between copies of this file, the log and the estimate) is the comment below the imports.
//
// Usage:
//   node tools/kit/gpu-lock-impl.mjs <command> ...     (the same commands as tools/kit/gpu-lock.mjs, this copy only)
//   <prefix>GPU_LOCK_IMPL=local node tools/kit/gpu-lock.mjs <command> ...   (the entry, forced to this worktree's copy)
//   (`<prefix>`: the configuration's `machine.envPrefix`, which names every variable below)
//
// Options: those of tools/kit/gpu-lock.mjs.
//
// The module's use:
//   import { withGpuLock, acquireGpuLock } from './gpu-lock.mjs';
//   await withGpuLock( 'profile island', async () => { ... } );           // held while fn runs
//   await acquireGpuLock( 'capture overview' );                           // held until this process exits
//   await acquireGpuLock( 'probe', { shared: true } );                    // shared with other probes
//   await leaseGpuLock( 'capture v', { shared: true, budget: 20000 } );   // kept from the last piece of work, or yielded to a waiter
//   await leaseGpuLock( 'profile', { keepPlace: true } );                 // exclusive: yields to a waiter ranked before it, asks again in its old place
//   lockState();                                                          // holders and queue in turn, with levels (job.mjs)
//
// Output, cost and exit codes: as tools/kit/gpu-lock.mjs's. GPU lock and browser: it is the lock;
// it opens no browser.
//
// A change to the lock goes here, bumps LOCK_VERSION here and in the stand-in
// (tools/gpu-lock-impl.mjs), and keeps the exports and the CLI additive, since another worktree's
// entry may load this file.
import './help.mjs';
import { appPolicy } from './qos.mjs'; // first: every tool that takes the lock runs with the application policy
import { appendFileSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, unlinkSync, writeFileSync, statSync } from 'node:fs';
import { execFile, execFileSync, spawn } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import PROJECT from '../project.mjs';

// The lock's design.
//
// A machine-wide lock for everything that renders the scene in a browser (captures, profiles,
// GPU probes). Frame times are only meaningful one browser at a time: two pages sharing the GPU
// each measure the other's work, and parallel sessions or agents cannot see each other's browsers.
// Pixels are not: a paused capture or a probe renders the same image however busy the GPU is.
// So the lock has two modes:
//   - exclusive (the default, every timing run): one holder, nothing else rendering;
//   - shared (deterministic captures and probes): up to <prefix>GPU_SLOTS holders together (default
//     2), none while an exclusive holder runs. Captures are GPU-bound: two pages at once give about
//     1.75 times one page's throughput, a third about 2 times with the slowest views taking twice
//     as long as at two, a fourth hardly more (measured on 21 paused views, the frames identical
//     at every count); probes that sweep many points gain about 1.35 times at two and 1.4 at
//     three, where a probe's readings now and then differed from one alone's in the last digit (a
//     paused page bounds its GPU queue: packages/render/post/FramePacer.ts). Two slots take most of that gain
//     with the readings exact, so the rest queue (below). A slot is held per capture (capture-clip.mjs around its clip),
//     not per set or page, so a long set and a one-view probe interleave; capture.mjs's worker leases
//     its slot over the views of a set (leaseGpuLock): it keeps it from view to view and gives it up
//     for a waiter ranked before it that it blocks once it has held it for its lease (20 s), so a waiter's extra delay is bounded by
//     the lease (and the view in flight), not by the set.
//     An exclusive lease may keep its place (`keepPlace`: profile-frame holds the lock one scene at
//     a time): it yields only to a waiter ranked before its first request, and queues again under
//     that request's ticket number, so it keeps its rank within its level. A later exclusive request
//     of the same level still goes after the whole series, as after one long hold, while a request
//     ranked before it (a capture at normal ahead of a default timing run, or a session raised by
//     `prio set`) gets in between its pieces. (Asked for again under a new number, profile-frame's
//     first per-scene run gave its place up to a timing run queued after it, and waited behind four.)
// Both modes wait in one queue (tickets), under one rule: requests rank by level, then by ticket
// number (first in, first out), shared and exclusive alike, and a request starts when nothing ranked
// before it still waits: an exclusive one at the head once every holder has left, a shared one when
// a slot is free. A shared request may also start beside a shared head while that leaves the head a
// slot (startableShared); nothing ranked after the head ever holds it up.
// Levels, highest first: urgent (the user's own runs), high, normal, low (0-3 name them too).
// A request's level is, first match wins: a bump on its ticket or its shared hold (`prio bump`, by
// pid or `#ticket`; the process keeps it for its later requests: a lease's next one; an exclusive
// hold is not bumped, its owner.json goes without the mutex); the scheduling table;
// <prefix>GPU_PRIORITY; the default of its mode, normal for shared requests and low for exclusive
// ones. So by default timing runs get no special treatment and simply rank below captures and
// probes. One exception bounds that wait: an exclusive request at its default level ranks normal
// (source `aged`) once AGE_AFTER_MS (30 min) has passed since its run's first request, keeping its
// ticket number, so it goes before the normal requests queued after it. Replayed against a logged day (a simulation),
// it left the day's total agent wait unchanged and ended its one timing run that waited over 30 min;
// with captures half as frequent again (an overloaded queue) it cut the timing runs' p90 wall clock
// by over a quarter for 30 % more agent wait. Shorter thresholds or aging per piece rather than per
// run cost the captures far more (two to three times the agent wait under overload). A run's first request
// (`placeStart` on its ticket) is kept across the pieces of a keepPlace lease, so a run ages once,
// not per piece. Only the default ages: a request at an explicit level (the table, an env level, a
// bump) keeps it, so a session set to low stays low, and high and urgent still rank above an aged
// request; shared requests never age. The level is decided at every look from the clock (levelOf),
// and waiters re-rank on every poll, so a promotion needs no event. <prefix>GPU_LOCK_AGE_S
// shortens the threshold for the lock's own tests only; sessions leave it unset. The scheduling table
// (`ponlai-gpu.lock.priorities.json` beside the lock) ranks sessions, never set by a run for itself:
// entries { match, level, until?, note? }, `match` a substring of one of a request's
// session names or of its label; the longest match wins, an entry past `until` is ignored (and
// dropped at the next write), and `prio set` lists what an entry applies to now (and warns when
// nothing). A request's session names are the names of the checkouts (a worktree, or the main
// checkout) holding the script it runs and its working directory, and that directory's own name: a
// tool started by its absolute path from another directory is still found by its worktree's name,
// and the main checkout's name matches only requests from the main checkout (a substring of the
// whole path would match every worktree below it). A level set for a session applies to all its
// requests, shared and exclusive alike. Sessions leave <prefix>GPU_PRIORITY unset (inherited by
// every child, it would raise or lower everything below the command it is set on). Tickets, holders
// and the owner record the inputs (`session`, `env`, `bump`), never the level: every requester
// computes the effective level when it ranks, from those inputs and the table file read then, so a
// table change reranks tickets already queued and all requesters agree. The table is written under
// the mutex. Apart from the aging above nothing ages: an earlier rule let shared requests pass a
// waiting exclusive one until a cap (30 min) and promoted low requests after 10 min; the levels and
// the one aging rule replace both.
// Before the levels, under the earlier rule, a single long shared hold (20+ min) with an exclusive
// request queued behind it stopped 22 captures while two slots stood free, and back-to-back timing
// runs that raced for the lock (a mkdir every 500 ms) let one session win every time: tickets
// ended both.
//
// The lock is a directory (mkdir is atomic) under the user's temporary folder holding owner.json
// (pid, label, start time, mode); a lock whose process is gone is stale and taken over. Shared
// holders list themselves in its `shared/` folder (one file per slot, with its index) and owner.json
// names one of them, handed on to another when it leaves, so a tool that knows only the exclusive
// protocol (an older copy of this file) still sees one live owner and waits. Requests wait in
// `ponlai-gpu.lock.queue/` (tickets numbered from `ponlai-gpu.lock.seq`; an exclusive one carries
// `mode: 'exclusive'`); the exclusive request first in turn also announces itself in
// `ponlai-gpu.lock.waiting/`, which holds back the shared requests of copies that do not rank
// exclusive tickets. Taking a ticket, joining, leaving, taking the lock alone and changing the table
// are serialised by the `ponlai-gpu.lock.mutex` directory. A holder, ticket or waiter whose process
// is gone (killed) is dropped at the next look, so its slot frees itself and the queue moves on.
// Older copies of this file run on the same lock (worktrees branched before the entry below), and
// levels hold only among copies that know them. The rules between copies, by kind of older copy:
// - Every older copy with tickets ranks them by class (`priority`): normal ones first in, first out,
//   then low ones until their `promoteAt` (<prefix>GPU_PRIORITY=low there). Main's copy before the
//   levels (shared first) also lets a shared ticket (`sharedFirst`) pass an exclusive one until that
//   one's cap (`capAt`, 30 min) and holds shared tickets back behind a capped one; a copy from
//   before shared first lets nothing pass an exclusive ticket ranked before it.
// - This copy ranks an older copy's ticket or holder at its mode's default, whatever its class:
//   shared normal, exclusive low (source `older copy`). Neither the table nor a bump moves it (`prio
//   bump` refuses): raised here only, an older exclusive request would wait for nothing, while the
//   older copy kept it waiting for this copy's shared requests ranked after it, and this copy's
//   leased holders would yield to it. (It once matched the table by its label; that deadlocked.)
// - This copy's tickets tell older copies the class nearest their level (forOlderCopies): below
//   normal `low`, never promoted, so they rank it after their own normal requests as this copy
//   does; normal or higher `normal`, an exclusive ticket capped from its start, a shared one going
//   first. (Said `normal`, a default exclusive ticket and a later shared request of a copy from
//   before shared first each waited for the other, for ever.)
// - Where the two orders still differ, neither waits for a request that waits for it, so the
//   cost is a delay: a raised request (high, urgent) of this copy ranks before an older copy's
//   earlier normal ticket here but after it there, and both start when they can (an exclusive one
//   still waits for the holders to drain); an older copy's exclusive request ranks low here, so it
//   waits while this copy's shared requests keep arriving, as this copy's own default exclusive
//   requests do (and the older copy's shared requests queued behind it wait with it), while this
//   copy's low requests and that exclusive request race for the lock.
// - One stall remains: an older copy's shared request at class low (<prefix>GPU_PRIORITY=low) ranks
//   normal here but after this copy's normal shared ones there, so with one slot free each may wait
//   for the other until another slot frees or its promoteAt (10 min) passes. Sessions leave the
//   variable unset, so it ends with those copies.
// - An older copy's exclusive request that takes no ticket (a copy from before the tickets) is known
//   only by a waiter file without a ticket number; it blocks every shared join while it waits (as it
//   always did), and an exclusive request that asked after it lets it go first (one that asked
//   before it races it for the lock). A copy that predates the slots joins without a ticket, a slot
//   or the limit, and is not counted: its `run --shared` wrappers are members for a whole set, so
//   counting them would starve the queue. A ticket from before the classes (no `priority`) is a
//   barrier (queueOrder).
//
// This file is the implementation; tools import the entry, tools/kit/gpu-lock.mjs, which loads whichever
// copy declares the higher LOCK_VERSION: its own worktree's or the main checkout's (the main
// checkout's on a tie). So a worktree runs the newest lock on the machine once main has it, without
// merging, and the older-copy rules above matter only for copies from before the entry. Bump
// LOCK_VERSION with every change to the lock's rules or files (a change other copies must follow);
// keep the exports and main()'s CLI additive, since another worktree's entry may load this file.
// This file was tools/gpu-lock-impl.mjs before the tools were grouped, and the entries of checkouts
// from before then still look there in the main checkout: tools/gpu-lock-impl.mjs stays as a stand-in
// that re-exports this file and declares the same LOCK_VERSION (version 5 is the move), so those
// copies run this lock too. Bump the stand-in's LOCK_VERSION with this one.
//
// The lock keeps a history, so how long anyone waited is measured rather than guessed: every request,
// grant, release, timeout and abandoned request is one JSON line appended to `ponlai-gpu.lock.log.jsonl`
// next to the lock (t, ev, pid, label, mode, class, ticket, waitMs / holdMs, reason). A line is
// written whole by one append (O_APPEND), so concurrent writers never interleave, and logging never
// fails a run. Past LOG_CAP the file is renamed to `.log.1.jsonl` (one older file is kept; two writers
// rotating at the same instant may drop that older file, which only shortens the history). A release
// says why a hold ended when its holder did not end it itself: `handed` (a child took the slot over:
// the child's grant carries `from`), `yielded` (yieldAncestorSlot) or `stale` (the holder was killed
// and whoever looked next reclaimed the slot; the log line is written by the one whose unlink
// succeeded, so it is written once); `lease` marks a leased hold given up for a waiter, and a leased
// hold's release counts the pieces of work it served (`uses`). Every event carries the request's
// effective level as `class` (urgent, high, normal, low; older copies log normal or low there) and
// where it came from as `source` (bump, table, env, default, aged; `older copy` for an older copy's
// request seen by this copy). A request also logs its parent pid (`ppid`) and working directory
// (`cwd`), so waits can be grouped by agent.
// Inherited holds (re-entry, a child under an exclusive holder) take nothing and log nothing.
// A waiting request prints a timestamped line every 30 s: its level and where it comes from, its
// place in the queue, the next in turn, who holds the GPU and why it waits (an exclusive request
// ahead, requests of a higher level), and its estimated start when it first waits and whenever
// the estimate moves by more than a minute or a quarter.
//
// The estimated start (estimateWaits; `status` gives every queued request an `eta`, `eta` one made
// now or an existing ticket's) lets a waiting agent choose between waiting and handing its work back.
// It is an estimate, a range rather than a time. It runs the queue forward under the grant rules
// above (the levels in effect now, the head and the shared tickets behind it starting while slots
// are free, an exclusive ticket at the head once nothing holds the lock, leased holders giving up
// past their budget), 200 times, each hold lasting a duration
// drawn from what holds of its kind took: the label's kind (labelKind: `capture`, `capture-clip`,
// `profile-frame compare`, ...) and mode, from the log's recent releases, or from a built-in table
// (FALLBACK, p25 / median / p90 per kind) while the log has few of a kind. A current holder runs on
// for a draw longer than it has run; one older than anything drawn, for a share of its age. The
// median start over the runs is the estimate, the 10th and 90th percentiles the range. Drawn rather
// than a run with medians and one with p90s: hold times are skewed, so a queue of medians ends
// early (a sum of many holds tends to their mean, not their median) and a queue of p90s far too
// late (every hold long at once), and both errors grow with the queue. Requests made later are
// not foreseen: one of a higher level (a capture ahead of a default exclusive request) still goes
// first, so a request can start after its range.
//
// `log` summarises the window: waits (grants that queued, not take-overs) and holds per class or
// label, median, p95 and total, and the longest waits with who held the GPU meanwhile.
// A command started through `run` inherits the lock (<prefix>GPU_LOCK_HELD, <prefix>GPU_LOCK_MODE), and
// so does any process below a holder in the process tree, through any wrappers, so a wrapped tool
// that takes the lock itself does not wait for its own ancestor (a timing tool under a shared parent
// says so: its numbers may share the GPU). Below an exclusive holder everything is re-entrant. Below
// a shared holder, the first descendant that asks for a shared slot takes its ancestor's slot over:
// the ancestor (a `run --shared` around capture-set or baseline.mjs, an inner `run` per capture) is an
// orchestrator from then on and holds nothing, and its other children queue like anyone else. So
// an orchestrator never holds a slot its children wait for (with 2 slots, two wrapped sets would
// otherwise hold both while their captures wait), and an exclusive waiter gets in between the
// captures of a set rather than after the whole set. A descendant about to wait for something else
// (yieldAncestorSlot: a reference another process is capturing) frees that slot the same way.
// <prefix>GPU_LOCK_DIR moves the lock.
//
// Importing this module raises the process to the application scheduling policy on macOS
// (tools/kit/qos.mjs: agent-started processes otherwise starve on a busy machine), so every tool that
// takes the lock does; `run` starts its command with that policy too.

// The user's temporary folder. On macOS it is asked of the system rather than read from TMPDIR,
// which a sandboxed session may point elsewhere, and its runs would then not see each other's lock.
function tempRoot() {

	if ( process.platform === 'darwin' ) {

		try {

			const dir = execFileSync( 'getconf', [ 'DARWIN_USER_TEMP_DIR' ], { encoding: 'utf8', timeout: 2000 } ).trim();
			if ( dir ) return dir;

		} catch { /* fall back */ }

	}
	return os.tmpdir();

}

/** This implementation's version: the entry (gpu-lock.mjs) loads the highest on the machine. */
export const LOCK_VERSION = 5;

// the environment variables' prefix (`machine.envPrefix`), as every copy of the project names them
const ENV = PROJECT.machine.envPrefix;

// the lock's name is the configuration's (`machine.gpuLock`), the same in every copy of the project
export const LOCK_DIR = path.join( process.env[ `${ ENV }GPU_LOCK_DIR` ] || tempRoot(), PROJECT.machine.gpuLock );
const OWNER = path.join( LOCK_DIR, 'owner.json' );
const MEMBERS = path.join( LOCK_DIR, 'shared' );
const WAITING = `${ LOCK_DIR }.waiting`;
const QUEUE = `${ LOCK_DIR }.queue`;
const SEQ = `${ LOCK_DIR }.seq`;
const MUTEX = `${ LOCK_DIR }.mutex`;
const LOG = `${ LOCK_DIR }.log.jsonl`;
const LOG_OLD = `${ LOCK_DIR }.log.1.jsonl`;
/** The scheduling table (see the design comment above): { entries: [ { match, level, until?, note? } ] }. */
export const TABLE = `${ LOCK_DIR }.priorities.json`;
const LOG_CAP = 4 << 20; // ~20k events, days of busy use
const POLL_MS = 500;
/** How many shared holders may render at once (each requester reads its own environment). */
export const SLOTS = Math.max( 1, Math.floor( Number( process.env[ `${ ENV }GPU_SLOTS` ] ) ) || 2 );
/** The levels, highest first (see the design comment above); 0-3 name them too. */
export const LEVELS = [ 'urgent', 'high', 'normal', 'low' ];
/** A level from its name or number, or null. */
export function parseLevel( value ) {

	const text = String( value ?? '' ).trim().toLowerCase();
	if ( /^[0-3]$/.test( text ) ) return LEVELS[ Number( text ) ];
	return LEVELS.includes( text ) ? text : null;

}
// <prefix>GPU_PRIORITY's level (null when unset or not a level); children inherit it. Sessions leave
// it unset: the scheduling table is how levels are set (see the design comment above).
const ENV_LEVEL = parseLevel( process.env[ `${ ENV }GPU_PRIORITY` ] );
/** The level a request has unless a bump, the table or its environment says otherwise: shared
 * requests normal, exclusive ones low (see the design comment above). */
export const defaultLevel = ( t ) => t.mode === 'exclusive' ? 'low' : 'normal';
// How long an exclusive request at its default level waits, from its run's first request, before it
// ranks normal (see the design comment above). <prefix>GPU_LOCK_AGE_S overrides it for the lock's own tests only.
const AGE_AFTER_MS = 30 * 60e3;
export const AGE_MS = Number( process.env[ `${ ENV }GPU_LOCK_AGE_S` ] ) > 0 ? Number( process.env[ `${ ENV }GPU_LOCK_AGE_S` ] ) * 1000 : AGE_AFTER_MS;
/** When a request would age (ms since the epoch), or Infinity: an exclusive ticket or holder of a copy
 * that records its run's first request (`placeStart`), AGE_MS after it. Whether it does depends on its
 * level's source: only one at its default level ages (levelOf). */
export const agesAt = ( t ) => t.mode === 'exclusive' && t.placeStart ? Date.parse( t.placeStart ) + AGE_MS : Infinity;

let depth = 0; // re-entrant within this process
let heldMode = null; // 'exclusive' | 'shared' while this process holds the lock itself
let installed = false;
let pending = null; // this process's request while it waits: { label, mode, ticket, at }
let holding = null; // this process's own hold while it lasts: { label, mode, ticket, at }
let leasing = 0; // the budget (ms) of the leaseGpuLock request under way, recorded on its hold for the estimate
let bumped = null; // a level `prio bump` gave this process (on its ticket or its hold); kept for its later requests

// One event on the log (see the design comment above). Never throws.
function logEvent( event ) {

	try {

		let size = 0;
		try { size = statSync( LOG ).size; } catch { /* the first event */ }
		if ( size > LOG_CAP ) {

			const aside = `${ LOG }.${ process.pid }.rot`;
			try { renameSync( LOG, aside ); renameSync( aside, LOG_OLD ); } catch { /* another writer rotated it */ }

		}
		appendFileSync( LOG, JSON.stringify( { t: new Date().toISOString(), ...event } ) + '\n' );

	} catch { /* the log is a record, never a reason to fail a run */ }

}
const since = ( iso ) => Math.max( 0, Date.now() - Date.parse( iso ) ) || 0;
// true for the one caller whose unlink removed the file (so a reclaim is logged once)
function unlinkOnce( file ) {

	try { unlinkSync( file ); return true; } catch { return false; }

}
const clock = ( date = new Date() ) => date.toTimeString().slice( 0, 8 );

function alive( pid ) {

	if ( ! ( pid > 0 ) ) return false;
	try {

		process.kill( pid, 0 );
		return true;

	} catch ( error ) {

		// EPERM: the process exists but belongs to someone else
		return error.code === 'EPERM';

	}

}

const readJson = ( file ) => {

	try { return JSON.parse( readFileSync( file, 'utf8' ) ); } catch { return null; }

};
// written whole or not at all, so a reader never sees half a file
function writeJson( file, value ) {

	const tmp = `${ file }.${ process.pid }.tmp`;
	writeFileSync( tmp, JSON.stringify( value ) );
	renameSync( tmp, file );

}

// ------------------------------------------------------------------ levels

/** The scheduling table's entries (none when it has no file). */
export function readTable() {

	const table = readJson( TABLE );
	return Array.isArray( table?.entries ) ? table.entries : [];

}

// The checkout (the main one or a worktree) holding a directory: the nearest one with a `.git`.
function checkoutOf( dir ) {

	for ( let d = path.resolve( dir ), up; ; d = up ) {

		try { statSync( path.join( d, '.git' ) ); return d; } catch { /* not here */ }
		up = path.dirname( d );
		if ( up === d ) return null;

	}

}
// This process's session names, which the table's entries match (see the design comment above): the names of the
// checkouts holding the script it runs and its working directory, and that directory's own name. A
// tool run by its absolute path from another directory is still found by its worktree's name, and
// the main checkout's name names only requests from the main checkout.
const SESSION = ( () => {

	const script = process.argv[ 1 ] ? checkoutOf( path.dirname( path.resolve( process.argv[ 1 ] ) ) ) : null;
	return [ ...new Set( [ script, checkoutOf( process.cwd() ), process.cwd() ].filter( Boolean ).map( ( d ) => path.basename( d ) ) ) ];

} )();

// The table's entry for a request, or null: the longest `match` found in one of its session names
// (`session`; a copy from before them recorded only its working directory, `cwd`, matched whole) or
// in its label, among the entries not past their `until`.
function tableEntry( t, table, now ) {

	const names = Array.isArray( t.session ) ? t.session.map( String ) : [ String( t.cwd ?? '' ) ];
	let best = null;
	for ( const e of table ) {

		if ( typeof e?.match !== 'string' || ! e.match || ! parseLevel( e.level ) ) continue;
		if ( e.until && ! ( now < Date.parse( e.until ) ) ) continue;
		if ( ! ( names.some( ( name ) => name.includes( e.match ) ) || String( t.label ?? '' ).includes( e.match ) ) ) continue;
		if ( ! best || e.match.length > best.match.length ) best = e;

	}
	return best;

}

/** A request's (ticket's, holder's) effective level from the inputs it records, { level, source,
 * match? }: a bump, else the table's entry, else its environment's level, else its mode's default
 * (defaultLevel), or normal (source `aged`) for an exclusive one AGE_MS after its run's first
 * request (agesAt), so the level depends on `now`. A ticket or holder of an older copy (no `levels`) always takes its mode's default
 * (source `older copy`): neither the table nor a bump moves it, since that copy ranks it by its own
 * rules and a level only this copy knows would make the two wait for each other (see the design comment above). */
export function levelOf( t, table = readTable(), now = Date.now() ) {

	if ( ! t.levels ) return { level: defaultLevel( t ), source: 'older copy' };
	const bump = parseLevel( t.bump );
	if ( bump ) return { level: bump, source: 'bump' };
	const entry = tableEntry( t, table, now );
	if ( entry ) return { level: parseLevel( entry.level ), source: 'table', match: entry.match };
	const env = parseLevel( t.env );
	if ( env ) return { level: env, source: 'env' };
	// a default exclusive request that has waited long enough ranks normal, decided at every look
	if ( now >= agesAt( t ) ) return { level: 'normal', source: 'aged' };
	return { level: defaultLevel( t ), source: 'default' };

}
// the record with its effective level (`level`, `source`, `match`) beside its inputs
const withLevel = ( t, table, now ) => {

	const { level, source, match } = levelOf( t, table, now );
	return { ...t, level, source, match };

};

/** A request's place among the levels, 0 (urgent) .. 3 (low). */
export const levelRank = ( t ) => LEVELS.indexOf( t.level ?? levelOf( t, [] ).level );

// What a request of this process records so that every requester can rank it (see the design comment above).
function requestInputs() {

	return { levels: true, cwd: process.cwd(), session: SESSION, ...( ENV_LEVEL ? { env: ENV_LEVEL } : {} ), ...( bumped ? { bump: bumped } : {} ) };

}

/** The lock's current owner, or null when it is free. A shared lock's owner carries
 * `mode: 'shared'` and `members` (every shared holder). */
export function lockOwner() {

	const owner = readJson( OWNER );
	if ( owner ) {

		if ( owner.mode === 'shared' ) owner.members = sharedMembers();
		return owner;

	}
	try {

		// the directory exists but owner.json is not written yet (or was lost)
		const stat = statSync( LOCK_DIR );
		return { pid: 0, label: '(unknown)', start: stat.mtime.toISOString(), unknown: true, ageMs: Date.now() - stat.mtimeMs };

	} catch {

		return null;

	}

}

function listJson( dir ) {

	try {

		return readdirSync( dir ).filter( ( f ) => f.endsWith( '.json' ) ).map( ( f ) => ( { file: path.join( dir, f ), ...readJson( path.join( dir, f ) ) } ) );

	} catch {

		return [];

	}

}

/** The shared holders that are still running (files of dead ones are removed). */
function sharedMembers() {

	return listJson( MEMBERS ).filter( ( m ) => {

		if ( alive( m.pid ) ) return true;
		if ( unlinkOnce( m.file ) ) logEvent( { ev: 'release', reason: 'stale', pid: m.pid, label: m.label, mode: 'shared', class: levelOf( m ).level, ticket: m.seq, holdMs: since( m.start ) } );
		return false;

	} ).map( ( { file, ...m } ) => m );

}

/** Exclusive holders-to-be announcing that they wait for the shared holders to drain (files of
 * dead ones are removed). */
function exclusiveWaiters() {

	return listJson( WAITING ).filter( ( w ) => {

		if ( w.pid !== process.pid && alive( w.pid ) ) return true;
		if ( w.pid !== process.pid ) rmSync( w.file, { force: true } );
		return false;

	} );

}
// Exclusive requests of an older copy of this file: they take no ticket, so they are known only by
// their waiter file, which carries no `seq` (this copy's waiter files name their ticket).
const untrackedWaiters = () => exclusiveWaiters().filter( ( w ) => w.seq === undefined );

/** The waiting requests (tickets), first in first (tickets of dead processes are removed), each
 * with its effective level from the table read now (levelOf). */
function queued( now = Date.now() ) {

	const table = readTable();
	return listJson( QUEUE ).filter( ( t ) => {

		if ( alive( t.pid ) ) return true;
		if ( unlinkOnce( t.file ) ) logEvent( { ev: 'abandon', reason: 'stale', pid: t.pid, label: t.label, mode: t.mode ?? 'shared', class: levelOf( t, table, now ).level, ticket: t.seq, waitMs: since( t.start ) } );
		return false;

	} ).sort( ( a, b ) => a.seq - b.seq ).map( ( { file, ...t } ) => withLevel( t, table, now ) );

}

// What a ticket tells the copies from before the levels (see the design comment above) so that they order it as
// this copy does, as far as they can. They know two classes (`priority`): normal tickets first in,
// first out, then low ones until their `promoteAt`. Those from shared first also let a shared request
// (`sharedFirst`) pass an exclusive one before its cap (`capAt`) and hold shared ones back after it.
// - A ticket below normal says `low`, never promoted: they then rank it after their own (normal)
//   requests, as this copy ranks their requests (at their mode's default) before it. Said `normal`,
//   a default exclusive ticket and a shared request of a copy from before shared first that queued
//   after it each waited for the other, for ever.
// - A low exclusive ticket's cap lies beyond any wait; a low shared one does not claim to go first,
//   or an older copy's exclusive request (low here, normal there) would wait for it before its cap
//   while it waits for that request.
// - One of normal level or higher says `normal`; an exclusive one is capped from its start (it holds
//   back the shared requests queued after it), a shared one goes first (`sharedFirst`), as this copy
//   ranks an older copy's exclusive request (low) after it.
// Every key is set (undefined drops it from the file), so a change of level rewrites them all.
const OLD_NEVER = '9999-12-31T00:00:00.000Z';
const forOlderCopies = ( t ) => {

	const low = levelRank( t ) > LEVELS.indexOf( 'normal' );
	const exclusive = t.mode === 'exclusive';
	return { priority: low ? 'low' : 'normal', promoteAt: low ? OLD_NEVER : undefined,
		capAt: exclusive ? ( low ? OLD_NEVER : t.start ) : undefined, sharedFirst: ! exclusive && ! low ? true : undefined };

};

// A place in the queue (the ticket); call inside the mutex. It records the inputs of its level
// (requestInputs), not the level, so that a table change reranks it. `place`: the number of an
// earlier ticket of this process to queue under again (leaseGpuLock's keepPlace); the seq file is
// left alone then, and the ticket's file is named by the pid, so the number is never taken twice.
// `placeStart`: when that earlier run first asked (kept with the place, so a run ages from its first
// request, not from each piece's); a first request's own start otherwise.
function enqueue( label, exclusive = false, place = null, placeStart = null ) {

	const seq = place ?? ( Number( readJson( SEQ ) ) || 0 ) + 1;
	if ( place === null ) writeJson( SEQ, seq );
	mkdirSync( QUEUE, { recursive: true } );
	const start = new Date().toISOString();
	const ticket = withLevel( { pid: process.pid, label, seq, start, placeStart: placeStart ?? start, priority: 'normal', ...( exclusive ? { mode: 'exclusive' } : {} ), ...requestInputs() }, readTable(), Date.now() );
	const { level, source, match, ...inputs } = ticket;
	const onFile = { ...inputs, ...forOlderCopies( ticket ) };
	writeJson( path.join( QUEUE, `${ process.pid }.json` ), onFile );
	return { ...onFile, level, source, match };

}

// This process's ticket as it stands now, with its effective level (a table change or a bump
// another process wrote on it applies while it waits; the bump is kept for this process's later
// requests); what older copies read of it follows the level (forOlderCopies). Call inside the mutex.
function refreshTicket( ticket ) {

	const file = path.join( QUEUE, `${ process.pid }.json` );
	const onFile = readJson( file );
	if ( ! onFile ) return ticket;
	if ( parseLevel( onFile.bump ) ) bumped = parseLevel( onFile.bump );
	const t = withLevel( onFile, readTable(), Date.now() );
	const old = forOlderCopies( t );
	if ( Object.entries( old ).some( ( [ k, v ] ) => onFile[ k ] !== v ) ) writeJson( file, { ...onFile, ...old } );
	return t;

}

// The waiting tickets in the order they are served: by effective level, then first in, first out,
// shared and exclusive alike. A ticket from before the classes (no priority) waits for every
// earlier ticket, so it is a barrier: everything taken before it ranks before it, everything after
// it ranks after it, and the levels order each stretch between barriers. Ranking a later ticket
// before an earlier one across a barrier would make the two wait for each other.
export function queueOrder( tickets ) {

	const barriers = tickets.filter( ( t ) => t.priority === undefined ).map( ( t ) => t.seq );
	const key = ( t ) => [
		barriers.filter( ( seq ) => seq < t.seq ).length,
		t.priority === undefined ? LEVELS.length : levelRank( t ),
		t.seq
	];
	return tickets.map( ( t ) => ( { t, k: key( t ) } ) )
		.sort( ( a, b ) => a.k[ 0 ] - b.k[ 0 ] || a.k[ 1 ] - b.k[ 1 ] || a.k[ 2 ] - b.k[ 2 ] ).map( ( { t } ) => t );

}
// the waiting tickets served before this one, in turn
function ticketsBefore( ticket ) {

	const order = queueOrder( [ ...queued().filter( ( t ) => t.pid !== ticket.pid ), ticket ] );
	return order.slice( 0, order.indexOf( ticket ) );

}
const ticketsAhead = ( ticket ) => ticketsBefore( ticket ).length;

// The shared tickets that may start now in `order` (the queue in turn), given `free` slots: the
// head and the shared tickets right behind it, up to the first exclusive one and no more than the
// free slots. A shared ticket behind a shared head starts beside it only while that leaves the head
// a slot; nothing passes an exclusive ticket, which waits for every holder to leave.
export function startableShared( order, free ) {

	const out = [];
	for ( const t of order ) {

		if ( t.mode === 'exclusive' || out.length >= free ) break;
		out.push( t );

	}
	return out;

}

/** The lock as other tools see it (job.mjs reports where a job stands; the estimate runs the queue
 * forward from it): the holders, each { pid, label, start, mode, slot?, lease?, level, source } (a
 * shared member without a slot runs inside an ancestor's), the waiting tickets in the order they are
 * served (each with its level and source), older copies' exclusive waiters (`untracked`) and the
 * slot count. */
export function lockState( now = Date.now() ) {

	const owner = lockOwner();
	const table = readTable();
	let holders = [];
	if ( owner?.mode === 'shared' ) holders = owner.members.map( ( m ) => ( { ...m, mode: 'shared' } ) );
	else if ( owner && ! owner.unknown ) holders = [ { ...owner, mode: 'exclusive' } ];
	return { holders: holders.map( ( h ) => withLevel( h, table, now ) ), queue: queueOrder( queued( now ), now ), untracked: untrackedWaiters().map( ( { file, ...w } ) => w ), slots: SLOTS };

}

// this process's parent's parent and so on (0 when it cannot be read)
function parentOf( pid ) {

	try {

		return Number( execFileSync( 'ps', [ '-o', 'ppid=', '-p', String( pid ) ], { encoding: 'utf8', timeout: 2000 } ).trim() ) || 0;

	} catch {

		return 0;

	}

}

// The holder above this process in the process tree, { pid, mode }, or null.
function holdingAncestor() {

	const owner = lockOwner();
	if ( ! owner ) return null;
	const mode = owner.mode === 'shared' ? 'shared' : 'exclusive';
	const holders = new Set( mode === 'shared' ? owner.members.map( ( m ) => m.pid ) : [ owner.pid ] );
	const held = Number( process.env[ `${ ENV }GPU_LOCK_HELD` ] );
	if ( held > 0 && holders.has( held ) && alive( held ) ) return { pid: held, mode };
	// A holder further up the process tree. The environment names only the nearest `run`, and a
	// wrapper in between need not be a member: a tool that runs each capture through its own `run`
	// (capture-growth.mjs) under an outer `run` names that inner `run`. Its child would then wait
	// for a slot of its own, which an exclusive waiter blocks while the holder waits for the child:
	// a deadlock.
	for ( let pid = process.ppid, hops = 0; pid > 1 && hops < 32; pid = parentOf( pid ), hops ++ ) {

		if ( holders.has( pid ) ) return { pid, mode };

	}
	return null;

}

// Takes the slot of a shared holder above this process over (see the design comment above); false when that
// holder has left or handed its slot on meanwhile. Call inside the mutex.
function takeOverSlot( ancestor, label, reason = 'handed' ) {

	const file = path.join( MEMBERS, `${ ancestor.pid }.json` );
	const held = readJson( file );
	if ( ! held || ! alive( ancestor.pid ) || readJson( OWNER )?.mode !== 'shared' ) return false;
	const inputs = requestInputs();
	writeJson( path.join( MEMBERS, `${ process.pid }.json` ), { pid: process.pid, label, start: new Date().toISOString(), slot: held.slot, priority: 'normal', from: ancestor.pid, ...inputs } );
	rmSync( file, { force: true } );
	logEvent( { ev: 'release', reason, pid: ancestor.pid, label: held.label, mode: 'shared', class: levelOf( held ).level, ticket: held.seq, holdMs: since( held.start ), to: process.pid } );
	repairShared();
	return true;

}

// The mutex serialises joining and leaving the shared lock (and repairing it). Held for a few
// file operations only; one whose holder is gone, or older than 10 s, is broken. Its holder
// removes it only while it is still its own (owner.json names it): a holder stalled past 10 s
// whose mutex another process broke and took must not remove that one's.
const MUTEX_OWNER = path.join( MUTEX, 'owner.json' );
// true when this process now holds the mutex; a stale one is broken on the way
function tryMutex() {

	try {

		mkdirSync( MUTEX );
		writeFileSync( MUTEX_OWNER, JSON.stringify( { pid: process.pid } ) );
		return true;

	} catch ( error ) {

		if ( error.code !== 'EEXIST' ) throw error;
		const holder = readJson( MUTEX_OWNER );
		let age = 0;
		try { age = Date.now() - statSync( MUTEX ).mtimeMs; } catch { return false; }
		if ( ( holder && ! alive( holder.pid ) ) || age > 10000 ) rmSync( MUTEX, { recursive: true, force: true } );
		return false;

	}

}
function leaveMutex() {

	if ( readJson( MUTEX_OWNER )?.pid === process.pid ) rmSync( MUTEX, { recursive: true, force: true } );

}
async function withMutex( fn ) {

	while ( ! tryMutex() ) await new Promise( ( resolve ) => setTimeout( resolve, 20 ) );
	try { return fn(); } finally { leaveMutex(); }

}
// The synchronous variant, for the exit handler (which cannot wait for an event loop). It tries for
// 2 s, breaking a stale mutex as withMutex does; a mutex still busy then (its holder alive, younger
// than 10 s) is left alone, and the release goes ahead without it rather than leave this process's
// hold behind (a release only removes this process's own files; repairShared then rechecks).
function withMutexSync( fn ) {

	const until = Date.now() + 2000;
	const nap = new Int32Array( new SharedArrayBuffer( 4 ) );
	let own = tryMutex();
	while ( ! own && Date.now() < until ) {

		Atomics.wait( nap, 0, 0, 5 );
		own = tryMutex();

	}
	try { return fn(); } finally { if ( own ) leaveMutex(); }

}

function removeStale( owner ) {

	// Move the stale lock aside first, then check it is still the one we judged stale: two
	// waiters may both find it stale, and the slower one must not delete the faster one's lock.
	const aside = `${ LOCK_DIR }.stale-${ process.pid }-${ Date.now() }`;
	try {

		renameSync( LOCK_DIR, aside );

	} catch {

		return;

	}
	const moved = readJson( path.join( aside, 'owner.json' ) );
	if ( moved && owner && moved.pid !== owner.pid && alive( moved.pid ) ) {

		// we moved a fresh lock: put it back if nobody took the place meanwhile
		try { renameSync( aside, LOCK_DIR ); return; } catch { /* taken: its owner keeps running unlocked */ }

	}
	if ( moved ) logEvent( { ev: 'release', reason: 'stale', pid: moved.pid, label: moved.label, mode: moved.mode ?? 'exclusive', class: levelOf( moved ).level, ticket: moved.seq, holdMs: since( moved.start ) } );
	rmSync( aside, { recursive: true, force: true } );

}

// A shared lock whose named owner is gone but whose other members still run: name one of them.
// (A tool knowing only the exclusive protocol would take such a lock for stale, so the owner is
// handed on whenever a member leaves.) Call inside the mutex.
function repairShared() {

	const owner = readJson( OWNER );
	if ( ! owner || owner.mode !== 'shared' ) return;
	const members = sharedMembers();
	if ( members.length === 0 ) {

		rmSync( LOCK_DIR, { recursive: true, force: true } );
		return;

	}
	if ( ! members.some( ( m ) => m.pid === owner.pid ) ) writeJson( OWNER, { pid: members[ 0 ].pid, label: `shared: ${ members[ 0 ].label }`, start: owner.start, mode: 'shared' } );

}

function tryAcquireExclusive( label, seq, placeStart ) {

	try {

		mkdirSync( LOCK_DIR );

	} catch ( error ) {

		if ( error.code !== 'EEXIST' ) throw error;
		const owner = lockOwner();
		if ( ! owner ) return false;
		if ( owner.mode === 'shared' ) return 'shared';
		const stale = owner.unknown ? owner.ageMs > 10000 : ! alive( owner.pid );
		if ( stale ) removeStale( owner.unknown ? null : owner );
		return false;

	}
	const inputs = requestInputs();
	writeFileSync( OWNER, JSON.stringify( { pid: process.pid, label, start: new Date().toISOString(), mode: 'exclusive', seq, ...( placeStart ? { placeStart } : {} ), priority: 'normal', ...( leasing ? { lease: leasing / 1000 } : {} ), ...inputs } ) );
	return true;

}

// Takes the lock alone when this ticket is first in turn (queueOrder) and no older copy's exclusive
// request that asked earlier waits; the holders drain first, and nothing ranked after it starts
// meanwhile. While first, its waiter file (which names the ticket) holds back the shared requests of
// older copies that do not rank exclusive tickets; it is withdrawn while anything ranks before this
// ticket (a bump or a table change), or an older copy's request ranked before it would wait for it
// while it waits for that request. Call inside the mutex.
function takeExclusive( label, ticket ) {

	const waitFile = path.join( WAITING, `${ process.pid }.json` );
	const mine = Date.parse( ticket.start );
	if ( ticketsAhead( ticket ) > 0 ) {

		rmSync( waitFile, { force: true } );
		return false;

	}
	let got = untrackedWaiters().some( ( w ) => Date.parse( w.start ) < mine ) ? false : tryAcquireExclusive( label, ticket.seq, ticket.placeStart );
	if ( got === 'shared' ) {

		// a shared lock whose named owner died while others hold it: hand it on
		repairShared();
		got = false;

	}
	if ( got ) return true;
	if ( ! readJson( waitFile ) ) {

		mkdirSync( WAITING, { recursive: true } );
		writeJson( waitFile, { pid: process.pid, label, start: ticket.start, seq: ticket.seq, priority: ticket.priority } );

	}
	return false;

}

// Joins the shared lock when a slot is free, no exclusive ticket ranks before this one and fewer
// shared tickets rank before it than there are free slots (startableShared: it starts beside a
// shared head only while that leaves the head a slot). Call inside the mutex (an older copy's
// exclusive request does not take it, so the lock may change under us: mkdir decides).
function joinShared( label, ticket ) {

	// an exclusive request through an older copy goes first (as it always did: it has no place)
	if ( untrackedWaiters().length ) return false;
	const before = ticketsBefore( ticket );
	if ( before.some( ( t ) => t.mode === 'exclusive' ) ) return false;
	const owner = readJson( OWNER );
	if ( owner ) {

		if ( owner.mode !== 'shared' ) {

			if ( ! alive( owner.pid ) ) removeStale( owner );
			return false;

		}
		repairShared(); // it may empty and go

	} else {

		// being written, or lost
		const info = lockOwner();
		if ( info ) {

			if ( info.unknown && info.ageMs > 10000 ) removeStale( null );
			return false;

		}

	}
	// holders that joined through an older copy of this file have no slot and are not counted
	const members = readJson( OWNER ) ? sharedMembers().filter( ( m ) => m.slot !== undefined ) : [];
	const free = SLOTS - members.length;
	if ( ! startableShared( [ ...before, ticket ], free ).includes( ticket ) ) return false;
	const taken = new Set( members.map( ( m ) => m.slot ) );
	let slot = 0;
	while ( taken.has( slot ) ) slot ++;
	const me = { pid: process.pid, label, start: new Date().toISOString(), slot, priority: ticket.priority, seq: ticket.seq, ...( leasing ? { lease: leasing / 1000 } : {} ), ...requestInputs() };
	try {

		mkdirSync( LOCK_DIR );

	} catch ( error ) {

		if ( error.code !== 'EEXIST' ) throw error;
		// an exclusive holder came first (or is writing its owner.json)
		if ( readJson( OWNER )?.mode !== 'shared' ) return false;

	}
	mkdirSync( MEMBERS, { recursive: true } );
	writeJson( path.join( MEMBERS, `${ process.pid }.json` ), me );
	if ( ! readJson( OWNER ) ) writeJson( OWNER, { pid: me.pid, label: `shared: ${ label }`, start: me.start, mode: 'shared' } );
	rmSync( path.join( QUEUE, `${ process.pid }.json` ), { force: true } );
	return true;

}

// false when this process held no slot any more (a child took it over)
function leaveShared() {

	return withMutexSync( () => {

		const left = unlinkOnce( path.join( MEMBERS, `${ process.pid }.json` ) );
		repairShared();
		return left;

	} );

}
const logRelease = ( hold, reason ) => logEvent( { ev: 'release', ...( reason ? { reason } : {} ), pid: process.pid, label: hold.label, mode: hold.mode, class: hold.level, source: hold.source, ticket: hold.ticket, holdMs: Date.now() - hold.at,
	...( hold.uses > 1 ? { uses: hold.uses } : {} ) } );

// This process's own hold record (its shared member file, or owner.json when it holds the lock
// alone), or null.
function ownRecord() {

	const record = heldMode === 'shared' ? readJson( path.join( MEMBERS, `${ process.pid }.json` ) ) : heldMode === 'exclusive' ? readJson( OWNER ) : null;
	return record?.pid === process.pid ? record : null;

}

/** Releases the lock if this process holds it. */
export function releaseGpuLock() {

	release();

}

// (`reason` only for the log: 'lease' when leaseGpuLock gives the hold up for a waiter)
function release( reason ) {

	depth = 0;
	// a bump given to the hold stays with this process's later requests (a lease's next one)
	const bump = parseLevel( ownRecord()?.bump );
	if ( bump ) bumped = bump;
	rmSync( path.join( WAITING, `${ process.pid }.json` ), { force: true } );
	rmSync( path.join( QUEUE, `${ process.pid }.json` ), { force: true } );
	if ( pending ) {

		logEvent( { ev: 'abandon', reason: 'exit', pid: process.pid, label: pending.label, mode: pending.mode, class: pending.level, source: pending.source, ticket: pending.ticket, waitMs: Date.now() - pending.at } );
		pending = null;

	}
	const hold = holding;
	holding = null;
	if ( heldMode === 'shared' ) {

		heldMode = null;
		if ( leaveShared() && hold ) logRelease( hold, reason );
		return;

	}
	heldMode = null;
	const owner = readJson( OWNER );
	if ( owner && owner.pid === process.pid && owner.mode !== 'shared' ) {

		rmSync( LOCK_DIR, { recursive: true, force: true } );
		if ( hold ) logRelease( hold, reason );

	}

}

function installRelease() {

	if ( installed ) return;
	installed = true;
	process.on( 'exit', releaseGpuLock );
	for ( const [ signal, code ] of [ [ 'SIGINT', 130 ], [ 'SIGTERM', 143 ], [ 'SIGHUP', 129 ] ] ) {

		process.on( signal, () => {

			releaseGpuLock();
			process.exit( code );

		} );

	}

}

/** Waits for the lock and takes it. It is held until releaseGpuLock() or this process exits.
 * `timeout` in seconds (0: wait for ever); `shared: true` for work whose result does not depend
 * on what else runs on the GPU (paused captures, pixel probes). Returns the wait in milliseconds. */
export async function acquireGpuLock( label = path.basename( process.argv[ 1 ] ?? 'node' ), { timeout = 0, quiet = false, shared = false, place = null, placeStart = null } = {} ) {

	const ancestor = depth > 0 ? null : holdingAncestor();
	if ( depth > 0 || ( ancestor && ! ( shared && ancestor.mode === 'shared' ) ) ) {

		const parentMode = depth > 0 ? heldMode : ancestor.mode;
		if ( ! shared && parentMode === 'shared' && ! quiet ) console.error( `[gpu-lock] "${ label }" wants the GPU to itself but runs under a shared lock: other probes may render meanwhile` );
		depth ++;
		return 0;

	}
	installRelease();
	const started = Date.now();
	// a shared request below a shared holder takes that holder's slot over (see the design comment above)
	if ( ancestor && await withMutex( () => takeOverSlot( ancestor, label ) ) ) {

		depth = 1;
		heldMode = 'shared';
		const { level, source } = levelOf( { ...requestInputs(), label } );
		holding = { label, mode: 'shared', at: Date.now(), level, source };
		logEvent( { ev: 'grant', pid: process.pid, label, mode: 'shared', class: level, source, waitMs: 0, from: ancestor.pid } );
		return 0;

	}
	let told = 0;
	let toldEta = null; // the estimate last printed, and the wait when it was
	const waitFile = path.join( WAITING, `${ process.pid }.json` );
	const ticket = path.join( QUEUE, `${ process.pid }.json` );
	let mine = await withMutex( () => enqueue( label, ! shared, place, place !== null ? placeStart : null ) );
	const mode = shared ? 'shared' : 'exclusive';
	pending = { label, mode, ticket: mine.seq, at: started, level: mine.level, source: mine.source };
	// the parent pid and the working directory group requests by agent in the log (an earlier reading of the log had to guess)
	logEvent( { ev: 'request', pid: process.pid, ppid: process.ppid, cwd: process.cwd(), label, mode, class: mine.level, source: mine.source, ticket: mine.seq, ...( place !== null ? { place: true } : {} ) } );
	try {

		for ( ;; ) {

			// the level is taken anew at every look: a bump or a table change applies while it waits
			if ( await withMutex( () => {

				mine = refreshTicket( mine );
				return shared ? joinShared( label, mine ) : takeExclusive( label, mine );

			} ) ) break;
			Object.assign( pending, { level: mine.level, source: mine.source } );
			const waited = Date.now() - started;
			if ( timeout > 0 && waited > timeout * 1000 ) {

				const owner = lockOwner();
				pending = null;
				logEvent( { ev: 'timeout', pid: process.pid, label, mode, class: mine.level, source: mine.source, ticket: mine.seq, waitMs: waited } );
				throw new Error( `GPU lock: still held by ${ owner?.label } (pid ${ owner?.pid }) after ${ timeout } s` );

			}
			if ( ! quiet && waited - told >= ( told === 0 ? 0 : 30000 ) ) {

				// one line: the time, the place in the queue, who holds the GPU and why this waits
				const owner = lockOwner();
				const before = ticketsBefore( mine );
				const total = queued().length;
				const first = untrackedWaiters()[ 0 ] ?? ( shared ? before.find( ( t ) => t.mode === 'exclusive' ) : null );
				const next = before.length ? `, next "${ before[ 0 ].label }"${ before[ 0 ].mode === 'exclusive' ? ' (exclusive)' : '' }` : '';
				let held = 'nobody (taking it)';
				if ( owner?.mode === 'shared' ) {

					const slotted = owner.members.filter( ( m ) => m.slot !== undefined );
					held = `${ slotted.length } of ${ SLOTS } shared slots (${ slotted.map( ( m ) => `"${ m.label }"` ).join( ', ' ) })`;

				} else if ( owner ) held = `"${ owner.label }" (pid ${ owner.pid }, exclusive since ${ clock( new Date( owner.start ) ) })`;
				const higher = before.filter( ( t ) => levelRank( t ) < levelRank( mine ) ).length;
				const why = first ? `; "${ first.label }" (pid ${ first.pid }${ first.level ? `, ${ first.level }` : '' }) wants the GPU to itself first`
					: higher ? `; ${ higher } request${ higher > 1 ? 's' : '' } of a higher level first` : '';
				// the estimated start, when it is new or has moved by more than a minute or a quarter
				const eta = estimateWaits( { logged: recentDurations() } ).queue.find( ( t ) => t.pid === process.pid )?.eta;
				const moved = eta && eta.startIn !== null && ( toldEta === null || Math.abs( eta.startIn - ( toldEta.startIn - ( waited - toldEta.at ) / 1000 ) ) > Math.max( 60, 0.25 * toldEta.startIn ) );
				if ( moved ) toldEta = { ...eta, at: waited };
				const estimate = moved ? `; estimated start in ${ etaText( eta ) }` : '';
				console.error( `[gpu-lock ${ clock() }] waiting (${ mode }, ${ mine.level } from ${ mine.source === 'table' ? `the table's "${ mine.match }"` : mine.source }) ${ Math.round( waited / 1000 ) } s: ${ before.length + 1 } of ${ Math.max( total, before.length + 1 ) } in the queue${ next }; held by ${ held }${ why }${ estimate }` );
				told = waited || 1;

			}
			await new Promise( ( resolve ) => setTimeout( resolve, POLL_MS ) );

		}

	} finally {

		rmSync( ticket, { force: true } );
		rmSync( waitFile, { force: true } );

	}
	depth = 1;
	heldMode = mode;
	pending = null;
	holding = { label, mode, ticket: mine.seq, placeStart: mine.placeStart, at: Date.now(), level: mine.level, source: mine.source };
	logEvent( { ev: 'grant', pid: process.pid, label, mode, class: mine.level, source: mine.source, ticket: mine.seq, waitMs: Date.now() - started,
	} );
	return Date.now() - started;

}

/** Runs `fn` while holding the lock (`options` as for acquireGpuLock). */
export async function withGpuLock( label, fn, options = {} ) {

	const outer = depth === 0;
	await acquireGpuLock( label, options );
	// held by this call itself, not inherited from this process or an ancestor
	const own = outer && heldMode !== null;
	try {

		return await fn();

	} finally {

		if ( own ) releaseGpuLock();
		else depth = Math.max( depth - 1, 0 );

	}

}

/** The waiting request that this process's own hold keeps waiting, the first in turn, or null: one
 * that ranks before the request this process would make if it gave the hold up now and asked again
 * at once (at its level now: a bump or a table change counts; under ticket number `place` when it
 * would keep its place, else last), and that this hold blocks. For an
 * exclusive holder that is any such request; for a shared holder an exclusive one (it waits for
 * every slot to drain) or, when every slot is taken, a shared one; an older copy's exclusive request
 * blocks either. Null while this process holds nothing itself (an inherited hold included). */
function gpuLockWaiter( place = null ) {

	if ( ! holding ) return null;
	const untracked = untrackedWaiters()[ 0 ];
	if ( untracked ) return { ...untracked, mode: 'exclusive' };
	const bump = parseLevel( ownRecord()?.bump );
	// kept in its place, the request ages from its run's first request (an aged run keeps its hold)
	const again = withLevel( { pid: process.pid, label: holding.label, seq: place ?? Infinity, ...( place !== null && holding.placeStart ? { placeStart: holding.placeStart } : {} ), priority: 'normal', ...( holding.mode === 'exclusive' ? { mode: 'exclusive' } : {} ), ...requestInputs(), ...( bump ? { bump } : {} ) }, readTable(), Date.now() );
	const order = queueOrder( [ ...queued().filter( ( t ) => t.pid !== process.pid ), again ] );
	const before = order.slice( 0, order.indexOf( again ) );
	if ( holding.mode === 'exclusive' ) return before[ 0 ] ?? null;
	const exclusive = before.find( ( t ) => t.mode === 'exclusive' );
	if ( exclusive || before.length === 0 ) return exclusive ?? null;
	const slotted = readJson( OWNER )?.mode === 'shared' ? sharedMembers().filter( ( m ) => m.slot !== undefined ).length : 0;
	return slotted >= SLOTS ? before[ 0 ] : null;

}

/** A hold kept from one piece of work to the next (capture.mjs's views): call it before each piece
 * instead of acquireGpuLock, and releaseGpuLock() after the last. This process keeps the hold it
 * already has (same mode) while nobody waits for it (gpuLockWaiter), or while it has held it for less
 * than `budget` ms; otherwise it gives the hold up (logged as `reason: 'lease'`) and queues again
 * like any request. A waiter's extra delay is therefore at most the budget plus the piece in flight
 * when it runs out, however long the series of pieces, and a long series never stalls the head of
 * the queue. Without a hold of its own (or in another mode) it simply acquires. The release that ends
 * a hold logs how many pieces it served (`uses`).
 * `keepPlace` (exclusive holds: profile-frame's scenes): the hold is ranked, and asked for again,
 * under the ticket number of the request that first took it, so it yields only to a waiter ranked
 * before that request (a higher level, or an earlier ticket of its own level) and keeps its rank
 * within its level: an exclusive request of the same level queued meanwhile goes after the whole
 * series, as it would after one long hold. Asked for again under a new number, it went behind every
 * request of its level queued since.
 * Returns { waitMs, kept }: the wait for the lock, and whether the hold was kept. */
export async function leaseGpuLock( label, { shared = false, budget = 0, keepPlace = false, ...options } = {} ) {

	if ( holding && heldMode === ( shared ? 'shared' : 'exclusive' ) ) {

		const place = keepPlace && ! shared ? holding.ticket ?? null : null;
		const waiter = gpuLockWaiter( place );
		const yieldNow = waiter && Date.now() - holding.at >= budget;
		if ( ! yieldNow ) {

			holding.uses = ( holding.uses ?? 1 ) + 1;
			return { waitMs: 0, kept: true };

		}
		options.place = place;
		options.placeStart = place !== null ? holding.placeStart ?? null : null;
		release( 'lease' );

	} else if ( holding ) release();
	leasing = budget;
	try {

		return { waitMs: await acquireGpuLock( label, { shared, ...options } ), kept: false };

	} finally {

		leasing = 0;

	}

}

/** Frees the shared slot that a `run --shared` above this process still holds, when this process
 * is about to wait for something other than the GPU (a reference another process is capturing:
 * tools/kit/reference.mjs). The slot is taken over and left at once, so the orchestrators above
 * hold nothing; the captures this process starts later queue like anyone else's. Returns whether
 * a slot was freed (false: no shared holder above, or its slot already taken over). */
export async function yieldAncestorSlot( label = path.basename( process.argv[ 1 ] ?? 'node' ) ) {

	if ( depth > 0 ) return false;
	const ancestor = holdingAncestor();
	if ( ancestor?.mode !== 'shared' ) return false;
	if ( ! await withMutex( () => takeOverSlot( ancestor, label, 'yielded' ) ) ) return false;
	heldMode = 'shared';
	depth = 1;
	releaseGpuLock();
	return true;

}

// ------------------------------------------------------------------ GPU utilisation (macOS)

/** The GPU's "Device Utilization %" from the IOAccelerator's statistics, or null where unknown. */
export function gpuUtilisation() {

	if ( process.platform !== 'darwin' ) return Promise.resolve( null );
	return new Promise( ( resolve ) => {

		execFile( 'ioreg', [ '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator' ], { timeout: 5000 }, ( error, stdout ) => {

			const values = [ ...( stdout ?? '' ).matchAll( /"Device Utilization %"=(\d+)/g ) ].map( ( m ) => Number( m[ 1 ] ) );
			resolve( error || values.length === 0 ? null : Math.max( ...values ) );

		} );

	} );

}

// GPU time per process: the Apple GPU driver keeps each client's accumulated GPU time (ns) in the
// I/O registry, so two snapshots name whatever else is rendering. ioreg lists a client's
// properties alphabetically, its "AppUsage" before its "IOUserClientCreator", so the time is
// attributed per client node (from one "+-o" line to the next): read line by line it went to the
// client listed before, and named the wrong process (another app's idle client) for a busy one.
function gpuTimes() {

	return new Promise( ( resolve ) => {

		execFile( 'ioreg', [ '-r', '-c', 'AGXDeviceUserClient', '-l', '-w', '0' ], { timeout: 5000, maxBuffer: 1 << 26 }, ( error, stdout ) => {

			if ( error ) return resolve( null );
			const times = new Map();
			let creator = null, ns = 0;
			const flush = () => {

				if ( creator ) times.set( creator, ( times.get( creator ) ?? 0 ) + ns );
				creator = null;
				ns = 0;

			};
			for ( const line of stdout.split( '\n' ) ) {

				if ( line.includes( '+-o ' ) ) { flush(); continue; }
				const c = line.match( /"IOUserClientCreator" = "pid (\d+), ([^"]*)"/ );
				if ( c ) { creator = `${ c[ 2 ] } (pid ${ c[ 1 ] })`; continue; }
				if ( line.includes( '"AppUsage"' ) ) ns += [ ...line.matchAll( /"accumulatedGPUTime"=(\d+)/g ) ].reduce( ( a, m ) => a + Number( m[ 1 ] ), 0 );

			}
			flush();
			resolve( times );

		} );

	} );

}

/** The processes that used the GPU over `seconds`, busiest first: [ { name, percent } ] (macOS
 * with an Apple GPU; null elsewhere), the `limit` busiest (profile-frame's load timeline takes
 * them all). */
export async function gpuClients( seconds = 2, limit = 6 ) {

	if ( process.platform !== 'darwin' ) return null;
	const a = await gpuTimes();
	const t0 = process.hrtime.bigint();
	await new Promise( ( resolve ) => setTimeout( resolve, seconds * 1000 ) );
	const b = await gpuTimes();
	const elapsed = Number( process.hrtime.bigint() - t0 );
	if ( ! a || ! b ) return null;
	return [ ...b ].map( ( [ name, ns ] ) => ( { name, percent: + ( ( ns - ( a.get( name ) ?? 0 ) ) / elapsed * 100 ).toFixed( 1 ) } ) )
		.filter( ( c ) => c.percent >= 0.5 ).sort( ( x, y ) => y.percent - x.percent ).slice( 0, limit );

}

/** Samples the utilisation for `seconds` (every 0.25 s): { mean, max, samples }. */
export async function sampleGpuUtilisation( seconds = 3 ) {

	const samples = [];
	const end = Date.now() + seconds * 1000;
	while ( Date.now() < end ) {

		const value = await gpuUtilisation();
		if ( value === null ) return null;
		samples.push( value );
		await new Promise( ( resolve ) => setTimeout( resolve, 250 ) );

	}
	const mean = samples.reduce( ( a, b ) => a + b, 0 ) / samples.length;
	return { mean: + mean.toFixed( 1 ), max: Math.max( ...samples ), samples: samples.length };

}

/** Checks that nothing else is rendering before a measurement (call it while holding the lock,
 * so none of our own runs are). Above `threshold` % it warns and waits up to `wait` seconds for
 * the GPU to settle. Returns { idle, contended, waitedMs, samples: [...] }. */
export async function waitForIdleGpu( { threshold = 15, seconds = 3, wait = 60, log = console.error } = {} ) {

	const started = Date.now();
	const history = [];
	for ( ;; ) {

		const sample = await sampleGpuUtilisation( seconds );
		if ( sample === null ) return { idle: null, contended: false, waitedMs: 0, samples: history, note: 'utilisation unavailable' };
		history.push( sample );
		if ( sample.mean <= threshold ) return { idle: sample, contended: false, waitedMs: Date.now() - started, samples: history };
		const waited = Date.now() - started;
		const clients = await gpuClients( 1 );
		const who = clients?.length ? `: ${ clients.map( ( c ) => `${ c.name } ${ c.percent } %` ).join( ', ' ) }` : '';
		if ( waited > wait * 1000 ) {

			log( `[gpu-lock] the GPU is still ${ sample.mean } % busy after ${ Math.round( waited / 1000 ) } s${ who }: contended` );
			return { idle: sample, contended: true, waitedMs: waited, samples: history, clients };

		}
		if ( history.length === 1 ) log( `[gpu-lock] the GPU is ${ sample.mean } % busy (max ${ sample.max } %) with no locked run${ who }; waiting up to ${ wait } s` );

	}

}

// ------------------------------------------------------------------ estimated wait

// How long holds of a kind of work take, in ms (p25, median, p90), for kinds the log has too few of
// (fewer than MIN_LOGGED releases). From 2026-10-11..13, before the log existed: the grant times and
// the next sightings of holders in the `gpu-lock status` outputs that sessions printed, and the run
// reports of the aa and island sessions (a view 5-7 s alone, 20 s leases, clips about a minute, a
// profile with --accept 7-8 min, a compare 11-17 min). A capture hold is one view, or several under
// capture.mjs's lease (its 20 s budget plus the view in flight); a capture-clip is one clip;
// profile-frame's exclusive holds are whole runs (--accept is labelled `profile` too and sits at its
// long end), its shared holds the preload before them; `node` is a script under `run`.
const FALLBACK = {
	'capture shared': [ 6e3, 12e3, 45e3 ],
	'capture exclusive': [ 10e3, 20e3, 60e3 ],
	'capture-clip shared': [ 15e3, 30e3, 120e3 ],
	'probe shared': [ 15e3, 45e3, 120e3 ],
	'verify shared': [ 60e3, 180e3, 450e3 ],
	'node shared': [ 30e3, 120e3, 420e3 ],
	'node exclusive': [ 120e3, 300e3, 900e3 ],
	'profile-frame shared': [ 10e3, 20e3, 60e3 ],
	'profile-frame profile exclusive': [ 180e3, 320e3, 600e3 ],
	'profile-frame compare exclusive': [ 180e3, 450e3, 950e3 ],
	'profile-frame toggle exclusive': [ 300e3, 500e3, 900e3 ],
	'profile-frame exclusive': [ 60e3, 200e3, 700e3 ],
	'shared': [ 10e3, 30e3, 150e3 ],
	'exclusive': [ 120e3, 300e3, 900e3 ],
};
const MIN_LOGGED = 5; // releases of a kind before the log's own holds replace the table
const LOG_RECENT = 100; // the most recent releases per kind that count
const PIECE = [ 3e3, 6e3, 20e3 ]; // one piece of leased work (a capture view), until the log has leased holds of the kind

/** A label's kind: the label without its run-specific tail, so that holds of the same work group
 * together. The first word names the tool (`capture island-fair-noon` -> `capture`, `capture-clip
 * baseline-...` -> `capture-clip`, `verify-scene-jump fresh` -> `verify-scene-jump`); a script run
 * through node or a shell is named by its file (`node tools/kit/parity/capture-set.mjs ...` -> `node
 * capture-set`); profile-frame keeps its mode (`profile-frame compare-2026-...` -> `profile-frame
 * compare`, a custom --label -> `profile-frame`). Timestamps, hashes and numbers are dropped. */
export function labelKind( label = '' ) {

	const clean = String( label ).replace( /^shared: /, '' ).replace( /\d{4}-\d\d-\d\dT[\d:.-]+Z?/g, '' ).trim();
	const [ first = '', ...rest ] = clean.split( /\s+/ );
	if ( /^(node|bash|sh|zsh)$/.test( first ) ) {

		const script = rest.find( ( w ) => ! w.startsWith( '-' ) );
		return script ? `node ${ path.basename( script ).replace( /\.(m?js|sh)$/, '' ) }` : first;

	}
	if ( first === 'profile-frame' ) {

		const mode = /^(profile|compare|toggle)-?$/.exec( rest[ 0 ] ?? '' );
		return mode ? `profile-frame ${ mode[ 1 ] }` : 'profile-frame';

	}
	return first.replace( /\b[0-9a-f]{7,40}\b/g, '' ).replace( /[-_.]*\d+$/, '' ) || '?';

}

/** Hold durations per kind and mode from the log's releases: Map( 'kind mode' -> { n, holds, pieces? } ),
 * the LOG_RECENT most recent holds of each (ms); `pieces` the pieces of leased work (holdMs / uses
 * over leased holds that served several), when there are enough. Holds a child took over or that
 * were yielded (an orchestrator's share before it handed on) say nothing about the work and are left
 * out. */
export function lockDurations( events = readLockLog() ) {

	const groups = new Map();
	for ( const e of events ) {

		if ( e.ev !== 'release' || ! Number.isFinite( e.holdMs ) || e.reason === 'handed' || e.reason === 'yielded' ) continue;
		const key = `${ labelKind( e.label ) } ${ e.mode === 'exclusive' ? 'exclusive' : 'shared' }`;
		if ( ! groups.has( key ) ) groups.set( key, { holds: [], pieces: [] } );
		groups.get( key ).holds.push( e.holdMs );
		if ( e.uses > 1 ) groups.get( key ).pieces.push( e.holdMs / e.uses );

	}
	const table = new Map();
	for ( const [ key, g ] of groups ) {

		const holds = g.holds.slice( - LOG_RECENT ), pieces = g.pieces.slice( - LOG_RECENT );
		table.set( key, { n: holds.length, holds, ...( pieces.length >= MIN_LOGGED ? { pieces } : {} ) } );

	}
	return table;

}

// A duration drawn from p25 / median / p90: a log-normal through the median, its spread below the
// median from p25 and above it from p90 (hold times are skewed: most short, a few long).
function drawQuantiles( [ p25, p50, p90 ], rand ) {

	const u = Math.min( Math.max( rand(), 1e-6 ), 1 - 1e-6 );
	// the standard normal quantile of u, near enough (the logistic distribution scaled to match it)
	const z = Math.log( u / ( 1 - u ) ) / 1.702;
	return p50 * Math.exp( z * ( z < 0 ? Math.log( p50 / p25 ) / 0.6745 : Math.log( p90 / p50 ) / 1.2816 ) );

}

// How long a hold of `label` in `mode` lasts, as something to draw from: the log's recent holds of
// the kind when it has MIN_LOGGED of them (drawn as they came), else the table's quantiles (the
// kind's, its tool's, or its mode's). { basis: 'log' | 'table' | 'default', hold( rand ), piece( rand ) }.
function durationOf( label, mode, logged ) {

	const kind = labelKind( label );
	const m = mode === 'exclusive' ? 'exclusive' : 'shared';
	const own = logged.get( `${ kind } ${ m }` );
	const piece = own?.pieces ? ( rand ) => own.pieces[ Math.floor( rand() * own.pieces.length ) ] : ( rand ) => drawQuantiles( PIECE, rand );
	if ( own && own.n >= MIN_LOGGED ) return { basis: 'log', hold: ( rand ) => own.holds[ Math.floor( rand() * own.holds.length ) ], piece };
	const tool = kind.split( ' ' )[ 0 ].replace( /^verify-.*/, 'verify' );
	const row = FALLBACK[ `${ kind } ${ m }` ] ?? FALLBACK[ `${ tool } ${ m }` ] ?? FALLBACK[ m ];
	return { basis: FALLBACK[ `${ kind } ${ m }` ] || FALLBACK[ `${ tool } ${ m }` ] ? 'table' : 'default', hold: ( rand ) => drawQuantiles( row, rand ), piece };

}

// One run of the queue forward in time under the lock's grant rules, every hold lasting a duration
// drawn for it (durationOf): Map( pid -> start time ). A current holder runs on for a duration drawn
// longer than it has already run; one older than anything drawn (work the table does not know, an
// own script, a hung page) is expected to run on for a share of its age, so a hold of a quarter of an
// hour is not taken to end within seconds at every look. At each moment the tickets in queueOrder
// start from the head: shared ones while slots are free (startableShared), an exclusive one at the
// head once nothing holds the lock; a leased holder gives its hold up at the first piece boundary
// past its budget once a waiter ranked before its next request is blocked by it (gpuLockWaiter's
// rule). Nothing that asks later is foreseen.
function simulateQueue( state, logged, rand, now, slots ) {

	// a shared member without a slot runs inside an ancestor's and frees nothing of its own
	const holders = state.holders.filter( ( h ) => h.mode !== 'shared' || h.slot !== undefined ).map( ( h ) => {

		const d = durationOf( h.label, h.mode, logged );
		const start = Date.parse( h.start ) || now;
		const elapsed = now - start;
		let left = 0;
		for ( let i = 0; i < 30 && left <= 0; i ++ ) left = d.hold( rand ) - elapsed;
		if ( left <= 0 ) left = Math.max( 5000, ( 0.05 + 0.45 * rand() ) * elapsed );
		return { pid: h.pid, mode: h.mode, level: h.level ?? levelOf( h, [] ).level, start, lease: h.lease, piece: d.piece, end: now + left };

	} );
	// an older copy's exclusive request goes before every ticket (it blocks every shared join)
	// copies: a request's level changes as it ages within a run, and every run starts from now
	const waiting = queueOrder( [ ...state.untracked.map( ( w, i ) => ( { ...w, mode: 'exclusive', priority: 'normal', level: LEVELS[ 0 ], seq: - 1 - i } ) ), ...state.queue.map( ( t ) => ( { ...t } ) ) ] );
	const starts = new Map();
	const grant = ( ticket, mode, t ) => {

		holders.push( { pid: ticket.pid, mode, level: ticket.level ?? levelOf( ticket, [] ).level, start: t, end: t + durationOf( ticket.label, mode, logged ).hold( rand ) } );
		starts.set( ticket.pid, t );
		waiting.splice( waiting.indexOf( ticket ), 1 );

	};
	let t = now;
	for ( let step = 0; waiting.length && step < 100000; step ++ ) {

		for ( let i = holders.length - 1; i >= 0; i -- ) if ( holders[ i ].end <= t ) holders.splice( i, 1 );
		// a default exclusive request ages while the queue runs forward (levelOf's rule, at time t)
		let aged = false;
		for ( const w of waiting ) if ( w.source === 'default' && t >= agesAt( w ) ) {

			Object.assign( w, { level: 'normal', source: 'aged' } );
			aged = true;

		}
		if ( aged ) waiting.splice( 0, waiting.length, ...queueOrder( waiting ) );
		const exclusiveHeld = holders.some( ( h ) => h.mode === 'exclusive' );
		for ( const ticket of startableShared( [ ...waiting ], exclusiveHeld ? 0 : slots - holders.length ) ) grant( ticket, 'shared', t );
		if ( waiting[ 0 ]?.mode === 'exclusive' && holders.length === 0 ) grant( waiting[ 0 ], 'exclusive', t );
		if ( ! waiting.length ) break;
		// leased holders yield to a blocked waiter ranked before their next request
		const full = ! exclusiveHeld && holders.length >= slots;
		for ( const h of holders ) {

			if ( ! h.lease || h.yields ) continue;
			const again = { pid: h.pid, seq: Infinity, priority: 'normal', level: h.level };
			const order = queueOrder( [ ...waiting, again ] );
			const before = order.slice( 0, order.indexOf( again ) );
			const waiter = h.mode === 'exclusive' ? before[ 0 ] : before.find( ( w ) => w.mode === 'exclusive' ) ?? ( full ? before[ 0 ] : null );
			if ( ! waiter ) continue;
			h.end = Math.min( h.end, Math.max( t, h.start + h.lease * 1000 ) + rand() * h.piece( rand ) );
			h.yields = true;

		}
		// the next moment anything changes: a hold ends
		const at = Math.min( ...holders.map( ( h ) => h.end ) );
		if ( ! Number.isFinite( at ) ) break; // nothing held and nothing granted: an older copy's own rules decide
		t = Math.max( at, t + 1 );

	}
	return starts;

}

const RUNS = 200; // simulated futures per estimate
// a small seeded generator (mulberry32), so the same queue gives the same estimate at every look
function seeded( seed ) {

	return () => {

		seed = ( seed + 0x6d2b79f5 ) | 0;
		let x = Math.imul( seed ^ ( seed >>> 15 ), 1 | seed );
		x = ( x + Math.imul( x ^ ( x >>> 7 ), 61 | x ) ) ^ x;
		return ( ( x ^ ( x >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

/** Estimated waits, in seconds from now, for every queued request (and, given `request` { mode:
 * 'shared' | 'exclusive', level, label }, for one made now; without a level it takes this process's
 * own, from the table and its environment): the queue simulated RUNS times, each hold's duration
 * drawn anew (simulateQueue), under the levels in effect now. Returns { now, queue: [ { pid, seq,
 * label, mode, level, source, eta } ], request?: eta }, eta = { startIn, range: [ lo, hi ], basis }:
 * the median start over the runs, the 10th and 90th percentiles, and where the durations of the holds
 * ahead came from (the log, the built-in table, a default per mode). Every number is an estimate:
 * holds vary, and requests made later (a higher-level one ahead of a lower one) are not foreseen.
 * `state`, `logged` and `now` may be given (replays, tests); they default to the lock's own. */
export function estimateWaits( { state = lockState(), logged = lockDurations(), now = Date.now(), slots = SLOTS, request = null } = {} ) {

	const queue = state.queue.map( ( t ) => ( { ...t } ) );
	let mine = null;
	if ( request ) {

		const exclusive = request.mode === 'exclusive';
		const label = request.label ?? '(this request)';
		const given = parseLevel( request.level ?? request.priority );
		const { level, source, match } = given ? { level: given, source: 'request' } : levelOf( { ...requestInputs(), label, ...( exclusive ? { mode: 'exclusive' } : {} ) }, readTable(), now );
		mine = { pid: - 1, label, seq: Math.max( 0, ...queue.map( ( t ) => t.seq ) ) + 1, start: new Date( now ).toISOString(), placeStart: new Date( now ).toISOString(), priority: 'normal', level, source, ...( match ? { match } : {} ), levels: true,
			...( exclusive ? { mode: 'exclusive' } : {} ) };
		queue.push( mine );

	}
	const full = { ...state, queue };
	const rand = seeded( 0x9e3779b9 );
	const starts = new Map( queue.map( ( t ) => [ t.pid, [] ] ) );
	for ( let run = 0; run < RUNS; run ++ ) for ( const [ pid, at ] of simulateQueue( full, logged, rand, now, slots ) ) starts.get( pid )?.push( at ); // an older copy's untracked request has no entry
	// where the durations came from: the holders' and every ticket's ahead
	const sources = state.holders.filter( ( h ) => h.mode !== 'shared' || h.slot !== undefined ).map( ( h ) => durationOf( h.label, h.mode, logged ).basis );
	const ahead = queueOrder( queue );
	const eta = ( ticket, index ) => {

		const basis = {};
		for ( const b of [ ...sources, ...ahead.slice( 0, index ).map( ( t ) => durationOf( t.label, t.mode, logged ).basis ) ] ) basis[ b ] = ( basis[ b ] ?? 0 ) + 1;
		const all = starts.get( ticket.pid ).sort( ( a, b ) => a - b );
		if ( ! all.length ) return { startIn: null, range: [ null, null ], basis };
		const s = ( q ) => Math.round( ( quantile( all, q ) - now ) / 1000 );
		return { startIn: s( 0.5 ), range: [ s( 0.1 ), s( 0.9 ) ], basis };

	};
	const out = ahead.map( ( t, i ) => ( { pid: t.pid, seq: t.seq, label: t.label, mode: t.mode ?? 'shared', level: t.level ?? levelOf( t, [] ).level, source: t.source ?? levelOf( t, [] ).source, priority: t.priority, eta: eta( t, i ) } ) );
	const result = { now: new Date( now ).toISOString(), queue: out.filter( ( t ) => t.pid !== - 1 ) };
	if ( mine ) result.request = { ...out.find( ( t ) => t.pid === - 1 ).eta, level: mine.level, source: mine.source };
	return result;

}

// the log's durations, read again every few minutes by a waiting process (the log may hold 20k events)
let durationsCache = null;
function recentDurations() {

	if ( ! durationsCache || Date.now() - durationsCache.at > 300000 ) durationsCache = { at: Date.now(), logged: lockDurations() };
	return durationsCache.logged;

}

// "~12 min (range 8-20 min, estimate)"
export function etaText( eta ) {

	if ( ! eta || eta.startIn === null ) return 'unknown (estimate)';
	const m = ( s ) => s < 90 ? `${ Math.max( 0, s ) } s` : `${ Math.round( s / 60 ) } min`;
	const [ lo, hi ] = eta.range;
	return `~${ m( eta.startIn ) } (range ${ m( lo ) }-${ hi === null ? '?' : m( hi ) }, estimate)`;

}

// ------------------------------------------------------------------ CLI

/** The log's events, oldest first (the rotated file, then the current one). */
export function readLockLog() {

	const events = [];
	for ( const file of [ LOG_OLD, LOG ] ) {

		let text = '';
		try { text = readFileSync( file, 'utf8' ); } catch { continue; }
		for ( const line of text.split( '\n' ) ) {

			if ( ! line ) continue;
			try { events.push( JSON.parse( line ) ); } catch { /* a line cut short by a full disk */ }

		}

	}
	return events;

}

const quantile = ( sorted, q ) => sorted.length ? sorted[ Math.min( sorted.length - 1, Math.ceil( q * sorted.length ) - 1 ) ] : null;
const stats = ( values ) => {

	const sorted = values.filter( Number.isFinite ).sort( ( a, b ) => a - b );
	return { n: sorted.length, median: quantile( sorted, 0.5 ), p95: quantile( sorted, 0.95 ), total: sorted.reduce( ( a, b ) => a + b, 0 ) };

};

/** Waits and holds per group ('class': mode and class, or 'label') and the `top` longest waits,
 * each with the holds that overlapped it (who held the GPU meanwhile, longest overlap first). */
export function summariseLockLog( events, { by = 'class', top = 5, now = Date.now() } = {} ) {

	const key = ( e ) => by === 'label' ? e.label ?? '?' : `${ e.mode ?? '?' } ${ e.class ?? '?' }`;
	const groups = new Map();
	const holds = []; // { pid, label, mode, start, end, open }
	const open = new Map(); // pid -> its grant without a release yet
	for ( const e of events ) {

		const k = key( e );
		if ( ! groups.has( k ) ) groups.set( k, { requests: 0, waits: [], holds: [], takeovers: 0, timeouts: 0, abandoned: 0, stale: 0, leases: 0, uses: 0 } );
		const g = groups.get( k );
		const t = Date.parse( e.t );
		if ( e.ev === 'request' ) g.requests ++;
		else if ( e.ev === 'grant' ) {

			if ( e.from ) g.takeovers ++;
			else g.waits.push( e.waitMs );
			open.set( e.pid, e );

		} else if ( e.ev === 'release' ) {

			g.holds.push( e.holdMs );
			if ( e.reason === 'stale' ) g.stale ++;
			if ( e.reason === 'lease' ) g.leases ++;
			g.uses += e.uses ?? 1;
			holds.push( { pid: e.pid, label: e.label, mode: e.mode, start: t - e.holdMs, end: t } );
			open.delete( e.pid );

		} else if ( e.ev === 'timeout' ) g.timeouts ++;
		else if ( e.ev === 'abandon' ) g.abandoned ++;

	}
	for ( const e of open.values() ) holds.push( { pid: e.pid, label: e.label, mode: e.mode, start: Date.parse( e.t ), end: now, open: true } );
	const longest = events.filter( ( e ) => e.ev === 'grant' && ! e.from && e.waitMs > 0 )
		.sort( ( a, b ) => b.waitMs - a.waitMs ).slice( 0, top ).map( ( e ) => {

			const end = Date.parse( e.t ), start = end - e.waitMs;
			const holders = holds.filter( ( h ) => h.pid !== e.pid && h.start < end && h.end > start )
				.map( ( h ) => ( { label: h.label, mode: h.mode, pid: h.pid, overlapMs: Math.min( end, h.end ) - Math.max( start, h.start ), ...( h.open ? { open: true } : {} ) } ) )
				.sort( ( a, b ) => b.overlapMs - a.overlapMs );
			return { t: e.t, label: e.label, mode: e.mode, class: e.class, waitMs: e.waitMs, holders };

		} );
	const rows = [ ...groups ].map( ( [ group, g ] ) => ( { group, requests: g.requests, wait: stats( g.waits ), hold: stats( g.holds ), takeovers: g.takeovers, timeouts: g.timeouts, abandoned: g.abandoned, stale: g.stale, leases: g.leases, uses: g.uses } ) )
		.sort( ( a, b ) => b.wait.total - a.wait.total || b.hold.total - a.hold.total );
	return { events: events.length, from: events[ 0 ]?.t ?? null, to: events.at( -1 )?.t ?? null, groups: rows, longest };

}

function printLockLog( summary, { by, sinceText } ) {

	const d = ( ms ) => ms === null ? '-' : ms < 1000 ? `${ Math.round( ms ) } ms` : ms < 120000 ? `${ ( ms / 1000 ).toFixed( 1 ) } s` : ms < 7200000 ? `${ ( ms / 60000 ).toFixed( 1 ) } min` : `${ ( ms / 3600000 ).toFixed( 1 ) } h`;
	const at = ( iso ) => new Date( iso ).toLocaleString( 'sv' ).slice( 5, 19 );
	console.log( `gpu-lock log${ sinceText ? ` (last ${ sinceText })` : '' }: ${ summary.events } events${ summary.from ? `, ${ at( summary.from ) } .. ${ at( summary.to ) }` : '' } (${ LOG })` );
	if ( ! summary.events ) return;
	const cut = ( text, n ) => text.length > n ? text.slice( 0, n - 1 ) + '~' : text;
	const width = Math.min( 40, Math.max( 5, ...summary.groups.map( ( r ) => r.group.length ) ) );
	const head = [ by === 'label' ? 'label' : 'class', 'req', 'waits', 'median', 'p95', 'total', 'holds', 'median', 'p95', 'total', 'other' ];
	const table = summary.groups.map( ( r ) => [ cut( r.group, width ), r.requests, r.wait.n, d( r.wait.median ), d( r.wait.p95 ), d( r.wait.total ), r.hold.n, d( r.hold.median ), d( r.hold.p95 ), d( r.hold.total ),
		[ r.uses > r.hold.n && `${ r.uses } uses`, r.leases && `${ r.leases } lease yields`, r.takeovers && `${ r.takeovers } taken over`, r.timeouts && `${ r.timeouts } timed out`, r.abandoned && `${ r.abandoned } abandoned`, r.stale && `${ r.stale } stale` ].filter( Boolean ).join( ', ' ) ].map( String ) );
	const widths = head.map( ( h, i ) => Math.max( h.length, ...table.map( ( row ) => row[ i ].length ) ) );
	for ( const row of [ head, ...table ] ) console.log( row.map( ( c, i ) => i === 0 || i === row.length - 1 ? c.padEnd( widths[ i ] ) : c.padStart( widths[ i ] ) ).join( '  ' ).trimEnd() );
	if ( summary.longest.length ) {

		console.log( '\nlongest waits (who held the GPU meanwhile, overlap):' );
		for ( const w of summary.longest ) {

			const who = w.holders.slice( 0, 3 ).map( ( h ) => `"${ cut( h.label ?? '?', 40 ) }" ${ h.mode === 'exclusive' ? 'excl ' : '' }${ d( h.overlapMs ) }${ h.open ? ' (no release logged)' : '' }` ).join( ', ' );
			console.log( `  ${ at( w.t ) }  ${ d( w.waitMs ).padStart( 8 ) }  ${ w.mode } ${ w.class }  "${ cut( w.label ?? '?', 40 ) }"  <- ${ who || 'nobody logged' }${ w.holders.length > 3 ? `, +${ w.holders.length - 3 } more` : '' }` );

		}

	}

}

// `prio`: the scheduling table and the levels in effect; `prio set|clear|bump` change them (see the
// header). Writes go under the mutex, like every change to the queue.
async function prioCommand( rest, opt, show ) {

	// the subcommand, if any, comes first (`prio --json` has none)
	const sub = rest[ 0 ]?.startsWith( '--' ) ? undefined : rest[ 0 ];
	const args = sub === undefined ? rest : rest.slice( 1 );
	const words = args.filter( ( a, i ) => ! a.startsWith( '--' ) && ! ( i > 0 && args[ i - 1 ].startsWith( '--' ) ) );
	const level = ( text ) => {

		const l = parseLevel( text );
		if ( ! l ) throw new Error( `a level is one of ${ LEVELS.join( ', ' ) } (or 0-3), not "${ text ?? '' }"` );
		return l;

	};
	// every write drops the entries past their `until` (an entry is replaced only by one of the same match)
	const writeTable = ( entries ) => {

		const now = Date.now();
		const kept = entries.filter( ( e ) => ! e.until || now < Date.parse( e.until ) );
		if ( kept.length ) writeJson( TABLE, { entries: kept } );
		else rmSync( TABLE, { force: true } );

	};
	// the holders and tickets whose level comes from the table's entry `match` now, and those of older
	// copies whose label it names (the table does not move them)
	const matched = ( match ) => {

		const state = lockState();
		const all = [ ...state.holders.map( ( h ) => ( { ...h, held: true } ) ), ...state.queue ];
		return { now: all.filter( ( t ) => t.source === 'table' && t.match === match ), older: all.filter( ( t ) => t.source === 'older copy' && String( t.label ?? '' ).includes( match ) ) };

	};
	const name = ( t ) => `${ t.held ? 'holder' : 'ticket' } pid ${ t.pid }${ t.seq !== undefined && ! t.held ? ` #${ t.seq }` : '' } "${ t.label }"${ t.mode === 'exclusive' ? ' (exclusive)' : '' }`;
	if ( sub === 'set' ) {

		const [ match, text ] = words;
		if ( ! match ) throw new Error( 'usage: prio set <match> <level> [--for 90m] [--note text]' );
		const entry = { match, level: level( text ) };
		const span = opt( 'for', null );
		if ( span ) {

			const m = /^(\d+(?:\.\d+)?)(s|m|min|h|d)$/.exec( span );
			if ( ! m ) throw new Error( '--for takes a duration such as 90m, 2h or 1d' );
			entry.until = new Date( Date.now() + Number( m[ 1 ] ) * { s: 1e3, m: 6e4, min: 6e4, h: 36e5, d: 864e5 }[ m[ 2 ] ] ).toISOString();

		}
		const note = opt( 'note', null );
		if ( note ) entry.note = note;
		entry.set = new Date().toISOString();
		await withMutex( () => writeTable( [ ...readTable().filter( ( e ) => e.match !== match ), entry ] ) );
		console.log( `set: "${ match }" -> ${ entry.level }${ entry.until ? ` until ${ entry.until }` : '' }` );
		// what it applies to now; a match nothing has may be a typo, or a session that has not asked yet
		const { now, older } = matched( match );
		for ( const t of now ) console.log( `  applies to ${ name( t ) }` );
		for ( const t of older ) console.log( `  not to ${ name( t ) }: an older copy of the lock, whose requests keep their mode's default` );
		const longer = readTable().some( ( e ) => e.match !== match && e.match.includes( match ) );
		if ( ! now.length ) console.error( `[gpu-lock] no holder or queued request matches "${ match }" now (a worktree's name or a label); it applies to later requests that do${ longer ? `, unless a longer entry matches them` : '' }` );
		return 0;

	}
	if ( sub === 'clear' ) {

		const [ match ] = words;
		if ( ! match ) throw new Error( 'usage: prio clear <match>' );
		const found = await withMutex( () => {

			const entries = readTable();
			const kept = entries.filter( ( e ) => e.match !== match );
			writeTable( kept );
			return entries.length - kept.length;

		} );
		console.log( found ? `cleared "${ match }"` : `no entry "${ match }"` );
		return found ? 0 : 1;

	}
	if ( sub === 'bump' ) {

		// `#n` is a ticket number; a bare number a pid (a waiting ticket's, else a shared holder's),
		// else a ticket number. Ticket numbers grow without bound into the range of pids, so a pid
		// that holds or waits is taken first.
		const [ id, text ] = words;
		const byTicket = /^#\d+$/.test( id ?? '' );
		const n = Number( byTicket ? id.slice( 1 ) : id );
		const to = level( text );
		if ( ! ( n > 0 ) ) throw new Error( 'usage: prio bump <pid|#ticket> <level>' );
		const done = await withMutex( () => {

			const tickets = listJson( QUEUE ).filter( ( t ) => alive( t.pid ) );
			const members = listJson( MEMBERS ).filter( ( m ) => alive( m.pid ) );
			const owner = readJson( OWNER );
			const found = byTicket ? tickets.find( ( t ) => t.seq === n )
				: tickets.find( ( t ) => t.pid === n ) ?? members.find( ( m ) => m.pid === n ) ?? ( owner?.pid === n && owner.mode !== 'shared' ? { file: OWNER, ...owner } : null ) ?? tickets.find( ( t ) => t.seq === n );
			if ( ! found ) return { error: `no ticket or holder with ${ byTicket ? 'ticket number' : 'pid or ticket number' } ${ n }` };
			const { file, ...record } = found;
			const what = file === OWNER ? 'the exclusive hold' : file.startsWith( QUEUE ) ? `ticket #${ record.seq }` : 'the shared hold';
			const who = `${ what } of pid ${ record.pid } ("${ record.label }")`;
			// an older copy ranks its requests by its own rules: a level only this copy knows would make
			// the two wait for each other (see the design comment above)
			if ( ! record.levels ) return { error: `${ who } comes from an older copy of the lock, which does not read levels; it keeps its mode's default here too, since raising it in this copy only would leave the two copies waiting for each other` };
			// an exclusive holder has the GPU to itself, and its owner.json is removed without the mutex
			// when it ends, so a write could race that; the bump would only matter for its next request
			if ( file === OWNER ) return { error: `${ who } already has the GPU to itself; to raise its next request, bump that request's ticket or set the table` };
			const bumpedRecord = { ...record, bump: to };
			// what older copies read of a ticket follows its new level (forOlderCopies)
			writeJson( file, file.startsWith( QUEUE ) ? { ...bumpedRecord, ...forOlderCopies( withLevel( bumpedRecord, readTable(), Date.now() ) ) } : bumpedRecord );
			return { who };

		} );
		if ( done.error ) throw new Error( done.error );
		console.log( `bumped ${ done.who } to ${ to }` );
		return 0;

	}
	if ( sub !== undefined ) throw new Error( 'usage: prio | prio set <match> <level> [--for 90m] [--note text] | prio clear <match> | prio bump <pid|#ticket> <level>' );
	// the table, then every holder and ticket with its effective level and its source
	const now = Date.now();
	const state = lockState( now );
	const entries = readTable();
	const at = ( iso ) => new Date( iso ).toLocaleString( 'sv' ).slice( 5, 16 );
	if ( rest.includes( '--json' ) ) {

		console.log( JSON.stringify( { table: entries, holders: state.holders.map( show ), queue: state.queue.map( show ) } ) );
		return 0;

	}
	console.log( `scheduling table (${ TABLE }):` );
	if ( ! entries.length ) console.log( '  (empty: shared requests normal, exclusive ones low)' );
	for ( const e of entries ) console.log( `  "${ e.match }" -> ${ parseLevel( e.level ) ?? `? (${ e.level })` }${ e.until ? `${ now < Date.parse( e.until ) ? ' until' : ' expired' } ${ at( e.until ) }` : '' }${ e.note ? `  (${ e.note })` : '' }` );
	const where = ( t ) => Array.isArray( t.session ) ? t.session.join( ', ' ) : t.cwd ? path.basename( t.cwd ) : '';
	const line = ( t ) => `${ t.level.padEnd( 6 ) } ${ t.source === 'table' ? `table "${ t.match }"` : t.source }`.padEnd( 34 ) + ` ${ t.mode === 'exclusive' ? 'exclusive' : 'shared   ' } pid ${ t.pid }${ t.seq !== undefined ? ` #${ t.seq }` : '' } "${ t.label }"${ where( t ) ? `  (${ where( t ) })` : '' }`;
	console.log( 'holding:' );
	if ( ! state.holders.length ) console.log( '  (nobody)' );
	for ( const h of state.holders ) console.log( `  ${ line( h ) }` );
	console.log( 'queue, in turn:' );
	if ( ! state.queue.length ) console.log( '  (empty)' );
	for ( const t of state.queue ) console.log( `  ${ line( t ) }` );
	return 0;

}

/** The CLI. `impl` describes the file that runs and why it was chosen (the entry's choice), for `status`. */
export async function main( argv, { impl = `${ fileURLToPath( import.meta.url ) } v${ LOCK_VERSION }` } = {} ) {

	const [ command, ...rest ] = argv;
	const opt = ( name, fallback ) => {

		const i = rest.indexOf( '--' + name );
		const end = rest.indexOf( '--' );
		return i >= 0 && ( end < 0 || i < end ) ? rest[ i + 1 ] : fallback;

	};
	// a request or holder as the CLI shows it: its level and where it comes from, without the
	// fields kept for older copies
	const show = ( { levels, sharedFirst, capAt, priority, promoteAt, match, file, ...t } ) => ( { ...t, ...( match ? { match } : {} ) } );
	if ( command === 'status' ) {

		const owner = lockOwner();
		const now = Date.now();
		const table = readTable();
		// the queue in turn, each with its level and its estimated start (estimateWaits)
		const etas = new Map( estimateWaits( { now } ).queue.map( ( t ) => [ t.pid, t.eta ] ) );
		const queue = queueOrder( queued( now ) ).map( ( t ) => etas.has( t.pid ) ? { ...show( t ), eta: etas.get( t.pid ) } : show( t ) );
		// the exclusive requests in turn: the queue's, with an older copy's (no ticket) placed before
		// the first of them that asked later (those yield to it; earlier ones race it for the lock)
		const waiting = queue.filter( ( t ) => t.mode === 'exclusive' ).map( ( { eta, ...t } ) => t );
		for ( const { file, ...w } of untrackedWaiters().sort( ( a, b ) => Date.parse( a.start ) - Date.parse( b.start ) ) ) {

			const i = waiting.findIndex( ( t ) => ! t.olderCopy && Date.parse( t.start ) > Date.parse( w.start ) );
			waiting.splice( i < 0 ? waiting.length : i, 0, { ...w, olderCopy: true } );

		}
		const extra = { ...( waiting.length ? { waiting } : {} ), ...( queue.length ? { queue } : {} ) };
		const mine = levelOf( { ...requestInputs(), label: '' }, table, now );
		if ( owner?.mode === 'shared' ) {

			const { members, ...rest } = owner;
			const levelled = members.map( ( m ) => show( withLevel( m, table, now ) ) );
			const holders = levelled.filter( ( m ) => m.slot !== undefined ), unslotted = levelled.filter( ( m ) => m.slot === undefined );
			console.log( JSON.stringify( { ...rest, alive: alive( owner.pid ), dir: LOCK_DIR, impl, slots: { limit: SLOTS, used: holders.length, holders, ...( unslotted.length ? { unslotted } : {} ) }, ...extra } ) );

		} else console.log( owner ? JSON.stringify( { ...( owner.unknown ? owner : show( withLevel( { ...owner, mode: 'exclusive' }, table, now ) ) ), alive: owner.pid ? alive( owner.pid ) : null, dir: LOCK_DIR, impl, ...extra } )
			: `free (${ LOCK_DIR }, ${ SLOTS } shared slots; this shell's shared requests ${ mine.source === 'default' ? 'normal' : mine.level }, exclusive ones ${ mine.source === 'default' ? `low, normal after ${ AGE_MS / 60e3 } min` : mine.level }, ${ mine.source === 'table' ? `from the table's "${ mine.match }"` : mine.source === 'default' ? 'the defaults' : `from ${ mine.source }` }; runs ${ impl })${ Object.keys( extra ).length ? ` ${ JSON.stringify( extra ) }` : '' }` );
		return 0;

	}
	if ( command === 'eta' ) {

		// a request made now (--shared / --exclusive, --level), or an existing ticket (--pid)
		const pid = Number( opt( 'pid', 0 ) );
		const flags = new Set( rest );
		const given = opt( 'level', opt( 'class', undefined ) );
		const request = pid ? null : { mode: flags.has( '--exclusive' ) || ! flags.has( '--shared' ) ? 'exclusive' : 'shared', label: opt( 'label', undefined ), ...( given !== undefined ? { level: parseLevel( given ) } : {} ) };
		if ( request && request.level === null ) throw new Error( `--level takes ${ LEVELS.join( ', ' ) } or 0-3` );
		const result = estimateWaits( { request } );
		const eta = pid ? result.queue.find( ( t ) => t.pid === pid )?.eta : result.request;
		if ( pid && ! eta ) throw new Error( `pid ${ pid } has no ticket in the queue (it holds the GPU, has not asked, or has gone)` );
		if ( flags.has( '--json' ) ) console.log( JSON.stringify( { now: result.now, ...( pid ? { pid } : { request } ), eta } ) );
		else {

			const basis = Object.entries( eta.basis ).map( ( [ k, n ] ) => `${ n } from the ${ k === 'log' ? 'log' : k === 'table' ? 'built-in table' : 'default per mode' }` ).join( ', ' );
			const what = pid ? `pid ${ pid } (${ result.queue.find( ( t ) => t.pid === pid ).level })` : `${ request.mode === 'exclusive' ? 'an' : 'a' } ${ request.mode } request now (${ eta.level } from ${ eta.source })`;
			console.log( `${ what }: estimated start in ${ etaText( eta ) }; ${ result.queue.length } queued, durations ${ basis || 'none needed' }` );

		}
		return 0;

	}
	if ( command === 'prio' ) return await prioCommand( rest, opt, show );
	if ( command === 'log' ) {

		const sinceText = opt( 'since', null );
		const by = opt( 'by', 'class' );
		if ( ! [ 'class', 'label' ].includes( by ) ) throw new Error( '--by takes class or label' );
		let events = readLockLog();
		if ( sinceText ) {

			const m = /^(\d+(?:\.\d+)?)(s|m|min|h|d)$/.exec( sinceText );
			if ( ! m ) throw new Error( '--since takes a duration such as 90s, 30m, 2h or 1d' );
			const from = Date.now() - Number( m[ 1 ] ) * { s: 1e3, m: 6e4, min: 6e4, h: 36e5, d: 864e5 }[ m[ 2 ] ];
			events = events.filter( ( e ) => Date.parse( e.t ) >= from );

		}
		const summary = summariseLockLog( events, { by, top: Number( opt( 'top', 5 ) ) } );
		if ( rest.includes( '--json' ) ) console.log( JSON.stringify( summary ) );
		else printLockLog( summary, { by, sinceText } );
		return 0;

	}
	if ( command === 'gpu' ) {

		const seconds = Number( opt( 'seconds', 3 ) );
		const [ utilisation, clients ] = await Promise.all( [ sampleGpuUtilisation( seconds ), gpuClients( seconds ) ] );
		console.log( JSON.stringify( { utilisation, clients } ) );
		return 0;

	}
	if ( command === 'run' ) {

		const split = rest.indexOf( '--' );
		const cmd = split >= 0 ? rest.slice( split + 1 ) : [];
		if ( cmd.length === 0 ) throw new Error( 'usage: node tools/kit/gpu-lock.mjs run [--shared] [--label name] [--timeout s] -- <command ...>' );
		const label = opt( 'label', cmd.join( ' ' ).slice( 0, 80 ) );
		const shared = ( split >= 0 ? rest.slice( 0, split ) : rest ).includes( '--shared' );
		const waited = await acquireGpuLock( label, { timeout: Number( opt( 'timeout', 0 ) ), shared } );
		if ( waited > 1000 ) console.error( `[gpu-lock ${ clock() }] acquired${ shared ? ' (shared)' : '' } after ${ ( waited / 1000 ).toFixed( 1 ) } s` );
		const child = spawn( ...appPolicy( cmd[ 0 ], cmd.slice( 1 ) ), { stdio: 'inherit', env: { ...process.env, [ `${ ENV }GPU_LOCK_HELD` ]: String( process.pid ), [ `${ ENV }GPU_LOCK_MODE` ]: shared ? 'shared' : 'exclusive' } } );
		for ( const signal of [ 'SIGINT', 'SIGTERM', 'SIGHUP' ] ) {

			process.removeAllListeners( signal );
			process.on( signal, () => child.kill( signal ) );

		}
		return await new Promise( ( resolve ) => {

			child.on( 'error', ( error ) => { console.error( error.message ); resolve( 127 ); } );
			child.on( 'exit', ( code, signal ) => resolve( code ?? ( signal ? 128 + ( os.constants.signals[ signal ] ?? 0 ) : 1 ) ) );

		} );

	}
	console.error( 'usage: node tools/kit/gpu-lock.mjs run [--shared] [--label name] [--timeout s] -- <command ...> | status | eta [--shared|--exclusive] [--level l] [--pid p] [--json] | prio [set <match> <level> [--for 90m] [--note text] | clear <match> | bump <pid|#ticket> <level>] | gpu [--seconds n] | log [--since 2h] [--by class|label] [--top n] [--json]' );
	return 2;

}

if ( process.argv[ 1 ] && path.resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url ) ) {

	main( process.argv.slice( 2 ) ).then( ( code ) => {

		releaseGpuLock();
		process.exit( code );

	}, ( error ) => {

		console.error( error.message );
		releaseGpuLock();
		process.exit( 1 );

	} );

}
