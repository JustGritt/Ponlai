// Loads the scene in a browser, runs JavaScript in it and prints the values as JSON.
//
// It answers a question about the running page (a diagnostic, a load timing, a uniform, the
// pipelines an action builds, a frame time) without a screenshot and without a Playwright script
// of one's own. Several queries can be probed in one call, each reached in the same page by the
// scene jump.
//
// Usage:
//   node tools/kit/probe.mjs [url] [--q query] [--query q]... [--init js|@file]...
//       [--eval js|@file] [--until js|@file] [--frames n] [--sleep ms] ...
//       [--w 1600 --h 900 --scale 1] [--timeout 60000] [--exclusive] [--fresh] [--load] [--console [prefix]]
//   node tools/kit/probe.mjs --q "debug" --eval "D.timings" --eval "P.camera.position"
//
// Options:
//   url             the server (default http://127.0.0.1:5190/, the dev server) or a page of it by its
//                   file (`<server>/scenes.html`, the test scenes); a production preview needs `debug`
//                   in the query for `window.__PONLAI__` and the diagnostics
//   --q query       the first query loaded (default `paused&t=12&dpr=1`)
//   --query q       a further query, reached by the scene jump; repeatable
//   --eval js       a step: runs in the page and records its value
//   --until js      a step: waits until the expression is truthy, up to --timeout
//   --frames n      a step: waits for n rendered frames
//   --sleep ms      a step: waits that long
//   --init js       runs before any of the page's scripts on every load (Playwright's addInitScript)
//   --w, --h        the viewport in CSS pixels (1600 by 900); --scale its device pixel ratio (1)
//   --timeout ms    each wait's limit, the navigation's included (60000)
//   --exclusive     takes the GPU lock alone instead of a shared slot, for frame times
//   --fresh         a new empty browser profile in the temp folder, deleted afterwards: a cold load
//   --load          loads every query in a page of its own instead of jumping to it
//   --console [p]   records every console line, or those starting with p
//
// Steps. The steps run in the order given, for every query. `js` is an expression (`D.timings`), a
// function (`async () => { ... }`, called) or statements with a `return`; `await` works in all
// three, and `@file` reads it from a file. In scope: `P` = `window.__PONLAI__`, `D` =
// `window.__THREE_GAME_DIAGNOSTICS__` as the step starts (most of its counters lag by up to 30
// frames: read live counters such as `P.pipelines.total` from `P`), `frames( n )`, `sleep( ms )`,
// `until( fn, timeoutMs )` and `pipelines` (`mark()`, `since( mark )`: which pipelines an action
// built, tools/render/pipeline-trace.mjs). The value is serialised in the page (cycles, functions,
// typed arrays, three's vectors). The forms, the scope and the serialiser are
// tools/kit/page-eval.mjs's, shared with tools/kit/capture.mjs's `--eval`. `--until` is checked
// every rendered frame (every 100 ms while the loop is stopped). Useful conditions: `D.settled` (the
// paused page will draw the same frame from now on), `D.taa?.converged`, `D.frame > 300`.
//
// Before the steps, every query waits until the page is up: `__PONLAI__` and the diagnostics exist,
// one frame is rendered and the loading screen's fade has finished (tools/kit/capture-browser.mjs
// `waitForScene`). The page is not stopped or settled further: add an `--until` for that.
//
// Queries. The first query is loaded. Each further one is reached in the same page by the exact
// scene jump (`await __PONLAI__.establishScene( query )`, the loop started, one frame rendered), as
// tools/kit/parity/capture-set.mjs reaches its views. A query whose load-time options differ
// (quality, aa, dpr, debug switches), or a build without the jump, gets a fresh load instead, and so
// does the query after one that failed. A jump draws a fresh load's image, not the world's
// generation path: to compare two loaded pages (a grown page against a fresh `grow=1` page) use
// `--load`.
//
// Common questions:
//   diagnostics and load timings      --eval "D.timings" --eval "D.worldCache"
//   a uniform or the camera           --eval "P.shared.wetness.value" --eval "P.camera.position"
//   the pipelines an action builds    --q "debug&paused" --until "D.settled" --eval "window.m = pipelines.mark()"
//                                     --eval "P.setWeather( 'storm', 0 )" --frames 120 --eval "pipelines.since( window.m )"
//   the game's state                  --q "debug&game" --eval "P.game.sim.state.villagers[ 0 ]"
//   frame time                        --exclusive --q "debug" --frames 300 --eval "D.frameMs"
//   linear radiance over boxes        --q "debug&paused&flags=lightOnly" --frames 60
//                                     --eval "await P.radiance( { boxes: { a: [ 0.4, 0.6, 0.1, 0.1 ] } } )"
// `pipelines.since` lists each pipeline's kind, label, material, object, target, frame and the
// first 6 app frames of its stack; it sees only pipelines built after the first `mark()`.
// `D.settled` never turns true on a running page. For an A/B frame time use
// tools/render/profile-frame.mjs. Debug handles and keys: docs/reference/debug-handles-and-keys.md.
//
// Output (stdout, JSON): the url, the browser, the profile slot, the lock and its wait, and per
// query its `mode` (load or jump), `readyMs` (until up), `ms` (the whole query), `steps` (`{ eval,
// value }`, `{ eval, error }`, `{ until, ms, frames }`, `{ frames }`, `{ sleep }`), the console's
// errors and warnings and the failed requests; `--console` adds the lines it records as `console`.
//
// GPU lock and browser: takes the GPU lock (docs/reference/gpu-lock.md) from before the load to the
// end of the last query, shared by default since a probe's values do not depend on what else
// renders, and opens the capture tools' headless browser in a persistent profile slot, so the world
// cache and the shader cache are warm from the second probe on.
//
// Cost: a page load (seconds warm; a cold `--fresh` load generates the island and compiles every
// shader), then the steps. Probes share the GPU as capture pages do: two probes sweeping a paused,
// grown island at 4K took about 1.4 times one probe's time, with readings identical to a probe
// alone; three probes gave little more throughput, and a reading could differ in the last digit.
//
// Exit codes: 0 every step ran; 1 an eval failed, an `--until` timed out or the page threw (the JSON
// is still printed); 2 a bad option.
//
// A long eval that waits on CPU work (worker generation, a grown page) stops the loop (`await
// P.loop.stop()`) and waits with `until` or `sleep`. While the loop stands, those helpers offer the
// shared slot back (tools/kit/page-eval.mjs): the probe gives it up when a request ranked before
// its next one waits (`leaseGpuLock` with no budget), queues again by rank and resolves the helper
// only once it holds the slot again, so the eval never renders without it. Each give-up is logged on
// stderr with its duration, and the step's `gaveUpMs` sums them. A default exclusive request ranks
// after a probe until it has waited 30 min. `--exclusive` keeps its hold.
import './help.mjs';
import { acquireGpuLock, leaseGpuLock, lockOwner, releaseGpuLock } from './gpu-lock.mjs';
import { launchCapture, waitForScene } from './capture-browser.mjs';
import { evalSource as source, runInPage } from './page-eval.mjs';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import PROJECT from '../project.mjs';

// the app's globals and its loading screen, as the project's configuration names them, for the
// page code below (passed in: a closure run in the page sees nothing of node's scope)
const PAGE = PROJECT.page;

const usage = 'usage: node tools/kit/probe.mjs [url] [--q query] [--query q]... [--init js|@file]... [--eval js|@file] [--until js] [--frames n] [--sleep ms] ... [--w --h --scale] [--timeout ms] [--exclusive] [--fresh] [--load] [--console [prefix]] (see the file header)';
const argv = process.argv.slice( 2 );
const options = { w: '1600', h: '900', scale: '1', timeout: '60000' };
const queries = [], inits = [], steps = [];
let url = PROJECT.url, consoleFilter = null;
for ( let i = 0; i < argv.length; i ++ ) {

	const a = argv[ i ];
	if ( ! a.startsWith( '--' ) ) { url = a; continue; }
	const k = a.slice( 2 );
	if ( k === 'exclusive' || k === 'fresh' || k === 'load' ) { options[ k ] = true; continue; }
	if ( k === 'console' ) { consoleFilter = argv[ i + 1 ] && ! argv[ i + 1 ].startsWith( '--' ) ? argv[ ++ i ] : ''; continue; }
	const v = argv[ ++ i ];
	if ( v === undefined ) { console.error( `--${ k } needs a value\n${ usage }` ); process.exit( 2 ); }
	if ( k === 'q' ) queries.unshift( v );
	else if ( k === 'query' ) queries.push( v );
	else if ( k === 'init' ) inits.push( source( v ) );
	else if ( k === 'eval' || k === 'until' ) steps.push( { [ k ]: v, js: source( v ) } );
	else if ( k === 'frames' || k === 'sleep' ) steps.push( { [ k ]: Number( v ) } );
	else if ( [ 'w', 'h', 'scale', 'timeout' ].includes( k ) ) options[ k ] = v;
	else { console.error( `unknown option --${ k }\n${ usage }` ); process.exit( 2 ); }

}
if ( queries.length === 0 ) queries.push( PROJECT.query );
// a server's root gets its slash; a page of it (scenes.html) is loaded as named
if ( ! url.endsWith( '/' ) && ! url.includes( '?' ) && ! /\.html$/.test( url ) ) url += '/';
const timeout = Number( options.timeout );

// the page's own frame counter (rendered frames)
const renderedFrame = ( page ) => page.evaluate( ( n ) => window[ n.handle ]?.renderer.info.frame ?? window[ n.diagnostics ]?.frame ?? 0, PAGE );

async function load( page, query ) {

	// the navigation takes --timeout too: Playwright's own 30 s runs out on a contended GPU
	await page.goto( `${ url }?${ query }`, { timeout } );
	await page.waitForFunction( ( n ) => window[ n.handle ] && window[ n.diagnostics ], PAGE, { timeout } ).catch( async ( error ) => {

		// where the load stood: the loading screen's status line
		const status = await page.evaluate( ( n ) => document.querySelector( `#${ n.loading } span` )?.textContent ?? null, PAGE ).catch( () => null );
		throw new Error( `page not up within ${ timeout } ms (loading screen: ${ JSON.stringify( status ) }; a production build needs debug in the query): ${ error.message.split( '\n' )[ 0 ] }` );

	} );

}

// the exact scene jump (as capture.mjs's worker makes it); false when the page refuses the query
async function jump( page, query ) {

	const jumped = await page.evaluate( async ( { q, handle } ) => {

		const s = window[ handle ];
		if ( s.sceneJump !== 1 ) return - 1;
		try { await s.establishScene( q ); } catch ( error ) {

			if ( /reload the page/.test( error.message ) ) return - 1;
			throw error;

		}
		s.loop.start();
		return s.renderer.info.frame;

	}, { q: query, handle: PAGE.handle } );
	if ( jumped < 0 ) return false;
	await page.waitForFunction( ( { f, handle } ) => window[ handle ].renderer.info.frame > f, { f: jumped, handle: PAGE.handle }, { timeout, polling: 'raf' } );
	return true;

}

const started = Date.now();
const fresh = options.fresh ? mkdtempSync( path.join( tmpdir(), `${ PROJECT.machine.tempPrefix }probe-` ) ) : null;
if ( fresh ) process.env[ `${ PROJECT.machine.envPrefix }CAPTURE_PROFILES` ] = fresh;
const label = `probe ${ queries[ 0 ] }`;
const lockMs = await acquireGpuLock( label, { shared: ! options.exclusive } );
const capture = await launchCapture( { viewport: { width: Number( options.w ), height: Number( options.h ) }, deviceScaleFactor: Number( options.scale ), url } );
const { page } = capture;

// The slot the page's waiting helpers offer back while its loop stands (see the header): only a
// shared slot this process holds itself (not an exclusive hold, nor one an ancestor holds alone).
const SLOT_BINDING = '__probeGiveUpSlot';
let gaveUpMs = 0;
const slotBinding = ! options.exclusive && lockOwner()?.members?.some( ( m ) => m.pid === process.pid ) ? SLOT_BINDING : null;
if ( slotBinding ) await page.exposeFunction( SLOT_BINDING, async () => {

	const from = Date.now();
	const { kept } = await leaseGpuLock( label, { shared: true, budget: 0 } );
	if ( kept ) return 0;
	const ms = Date.now() - from;
	gaveUpMs += ms;
	console.error( `[probe] gave the shared slot up at ${ new Date( from ).toTimeString().slice( 0, 8 ) } for a waiter ranked before it (loop stopped), held again after ${ ( ms / 1000 ).toFixed( 1 ) } s` );
	return ms;

} );
const gaveUp = () => slotBinding ? { gaveUpMs } : {};
for ( const js of inits ) await capture.context.addInitScript( js );
let record = null;
page.on( 'console', ( m ) => {

	if ( ! record ) return;
	const line = `${ m.type() }: ${ m.text() }`;
	if ( m.type() === 'error' || m.type() === 'warning' ) record.errors.push( line );
	if ( consoleFilter !== null && m.text().startsWith( consoleFilter ) ) record.console.push( consoleFilter ? m.text() : line );

} );
page.on( 'pageerror', ( e ) => record?.errors.push( 'pageerror: ' + e.message ) );
page.on( 'crash', () => record?.errors.push( 'pageerror: the page crashed' ) );
page.on( 'response', ( r ) => { if ( r.status() >= 400 ) record?.failures.push( `${ r.status() } ${ r.url() }` ); } );
page.on( 'requestfailed', ( r ) => record?.failures.push( `${ r.failure()?.errorText } ${ r.url() }` ) );

const out = { url, browser: capture.version, slot: fresh ? 'fresh' : capture.slot, lock: options.exclusive ? 'exclusive' : 'shared', lockMs, queries: [] };
let failed = false;
try {

	for ( let index = 0; index < queries.length; index ++ ) {

		const query = queries[ index ];
		record = { query, mode: null, readyMs: 0, ms: 0, steps: [], errors: [], failures: [], ...( consoleFilter !== null ? { console: [] } : {} ) };
		out.queries.push( record );
		const t0 = Date.now();
		try {

			// (a page whose last query failed is not trusted: a fresh load)
			record.mode = index > 0 && ! options.load && ! out.queries[ index - 1 ].broken && await jump( page, query ) ? 'jump' : 'load';
			if ( record.mode === 'load' ) await load( page, query );
			await waitForScene( page, { frames: 1, timeout } );
			record.readyMs = Date.now() - t0;
			for ( const step of steps ) {

				if ( step.eval !== undefined ) {

					gaveUpMs = 0;
					try { record.steps.push( { eval: step.eval, value: await runInPage( page, step.js, { timeout, slotBinding } ), ...gaveUp() } ); } catch ( error ) {

						record.steps.push( { eval: step.eval, error: error.message.split( '\n' )[ 0 ], ...gaveUp() } );
						failed = true;

					}

				} else if ( step.until !== undefined ) {

					gaveUpMs = 0;
					try { record.steps.push( { until: step.until, ...( await runInPage( page, step.js, { mode: 'until', timeout, slotBinding } ) ), ...gaveUp() } ); } catch ( error ) {

						record.steps.push( { until: step.until, error: error.message.split( '\n' )[ 0 ], ...gaveUp() } );
						failed = true;

					}

				} else if ( step.frames !== undefined ) {

					const from = await renderedFrame( page );
					await page.waitForFunction( ( { f, n } ) => ( window[ n.handle ]?.renderer.info.frame ?? window[ n.diagnostics ]?.frame ?? 0 ) >= f, { f: from + step.frames, n: PAGE }, { timeout, polling: 'raf' } );
					record.steps.push( { frames: step.frames } );

				} else {

					await page.waitForTimeout( step.sleep );
					record.steps.push( { sleep: step.sleep } );

				}

			}

		} catch ( error ) {

			record.errors.push( `probe: ${ error.message.split( '\n' )[ 0 ] }` );
			failed = true;

		}
		record.ms = Date.now() - t0;
		if ( record.errors.some( ( e ) => e.startsWith( 'pageerror:' ) || e.startsWith( 'probe:' ) ) ) {

			failed = true;
			Object.defineProperty( record, 'broken', { value: true } );

		}

	}

} finally {

	record = null;
	await capture.close().catch( () => {} );
	releaseGpuLock();
	if ( fresh ) rmSync( fresh, { recursive: true, force: true } );

}
out.ms = Date.now() - started;
console.log( JSON.stringify( out, null, 2 ) );
if ( failed ) process.exitCode = 1;
