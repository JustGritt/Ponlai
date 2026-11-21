// Takes one screenshot of the running dev server or a preview build, with the page's diagnostics.
//
// It answers what one view looks like now: a capture for the eye, or one to compare against another
// by decoded pixels (tools/kit/img.py). For many views, or a before and after set, use
// tools/kit/parity/capture-set.mjs, which drives this tool as a worker on one warm page; for values
// rather than pixels, tools/kit/probe.mjs. It needs Playwright's Chromium once (`npx playwright
// install chromium`), or `CHROME_PATH` set to a WebGPU-capable Chrome. The browser and its profile
// slots: tools/kit/capture-browser.mjs.
//
// Usage:
//   node tools/kit/capture.mjs [name] [--url base] [--q query] [--w 1600 --h 900 --scale 1]
//       [--eval js|@file]... [--eval-ready js|@file]... [--wait-for js] [--frames n --every k]
//       [--timeout 60000] [--stable-* ...] [--ready-frames n] [--wait ms] [--settle ms]
//       [--samples n] [--exclusive] [--verbose]
//   node tools/kit/capture.mjs --worker --url base [the options above] [--page-per-view] [--ready-reads] [--lease 20]
//   node tools/kit/capture.mjs overview --q "paused&t=12&dpr=1" --eval-ready "D.exposure"
//
// Options:
//   name                  the file name under artifacts/captures/ of the working directory (the
//                         configuration's `output.captures`; default `shot`)
//   --url base            the server (default http://127.0.0.1:5190/, the dev server)
//   --q query             the page's query (default `paused&t=12&dpr=1`; docs/reference/url-options.md)
//   --w, --h, --scale     the viewport in CSS pixels and its device pixel ratio (1600, 900, 1)
//   --eval js|@file       a step run once the page is up, before readiness; repeatable
//   --eval-ready js|@file a step run once the page is ready, on the frame the screenshot takes; repeatable
//   --wait-for js         waits, up to 60 s, until the expression is truthy, before readiness
//   --frames n --every k  n screenshots k rendered frames apart (1, 1)
//   --timeout ms          each wait's limit: the load, readiness, a screenshot (60000)
//   --stable-min n        rendered frames before the first comparison (1)
//   --stable-every n      frames between grabs (2)
//   --stable-repeats n    unchanged comparisons in a row that make the image stable (1)
//   --stable-patience n   frames after which --stable-tolerance counts too (40)
//   --stable-tolerance t  `<level>:<pixels>`: at most pixels differing by more than level (0:5000);
//                         `exact` turns it off
//   --stable-cap n        frames after which the page is captured anyway, marked capped (120)
//   --stable-trace        records each grab's PNG hash
//   --ready-frames n      waits for n rendered frames instead of stability (a running page: 60)
//   --wait, --settle ms   fixed delays before and after the floor's frames (0, 0)
//   --samples n           diagnostics read n times, a second apart (1)
//   --exclusive           takes the GPU lock alone even for a paused page
//   --verbose             records every console line, not only errors and warnings
//   --worker              capture-set's worker (below); --page-per-view, --ready-reads, --lease s
//
// Readiness. Every capture waits until the page is up (`__PONLAI__` and the diagnostics exist, the
// floor's frames rendered, the loading screen's fade finished: a screenshot during the fade has the
// screen blended over the frame). A paused page (MSAA or off) then waits for its `settled` signal
// with the loop running (docs/reference/test-mode.md; past --timeout it goes on and prints
// `unsettled`) and is captured once its image stops changing: grabbed every --stable-every frames,
// with the loop stopped for each grab, and captured once --stable-repeats comparisons in a row
// found it unchanged, with --stable-patience and --stable-cap as above. A paused page of a temporal mode
// (aa=taa|taau|fsr3|gt) is not compared: its rest image is started over once the page is up (what
// the evals changed need not be in its key), and it is captured once the rest has converged
// (`diag.taa.converged`) and `settled` holds (packages/render/post/TemporalRest.ts; the --stable-*
// options do not apply; with `tier.taa.restSequence=0` there is no rest and the page is compared
// like an MSAA one). A running page waits --ready-frames. The rule and its evidence:
// tools/kit/capture-browser.mjs, "readiness by stability". The frame loop is stopped (the GPU queue
// drained) for the screenshot, which is the compositor's image PNG-encoded for speed: the pixels of
// Playwright's screenshot, not its bytes, so compare captures by decoded pixels. Under --frames n
// the loop runs between the screenshots and stops for each.
//
// Evals. --eval and --eval-ready take probe's forms and scope (tools/kit/page-eval.mjs: an
// expression, a function or statements with a `return`, `await` in all three; `P`, `D`,
// `frames( n )`, `sleep( ms )`, `until( fn, timeoutMs )`), so an eval waits with `await until( ...
// )` instead of a polling loop. --eval runs in order once the page is up, before the floor's frames
// and readiness: on a fresh load that is before the sun's shadow map has drawn, so it sets the view
// up. --eval-ready runs once the page is ready, with the loop stopped on the frame the screenshot
// then takes: for readings of that frame (a render target read back, the diagnostics). It reads;
// one that changes the scene or renders changes the screenshot. Under --frames n it runs once,
// before the first. An eval that throws fails the capture, which is still taken and written.
//
// Worker (--worker): one page for many views, driven by capture-set over the IPC channel it opens.
// The worker asks for a view, captures it, reports and asks again until told it is done. The first
// view is a fresh load of `<url>?<query>`; each later one is reached in place by the exact scene
// jump (`await __PONLAI__.establishScene( <query> )`, the loop started, one frame rendered at the
// jump's camera), then the view's own eval (a throw fails the view) and everything above, as after a
// load: the frame then equals a fresh load's (app/main.ts). Only load-time options (quality, aa,
// size and dpr, the debug switches) need a new page, and every view of a set shares them.
// - Capability: the page must keep that contract (`__PONLAI__.sceneJump === SCENE_JUMP`); on an
//   older build, and with --page-per-view, the worker stops after its first view and capture-set
//   starts a fresh worker per view.
// - Evals the jump does not undo: the jump resets the scene's options and camera, not GUI or console
//   changes, so a view whose message says `last` ends its page after its capture. --eval takes a page
//   per view, and so does --eval-ready unless `--ready-reads` says those steps only read the frame
//   (tools/island/region-colours.mjs `capture --warm`).
// - The GPU slot is leased (`leaseGpuLock`): the worker keeps it from one view to the next and gives
//   it up before a view only when it has held it for --lease seconds (20) and a request ranked
//   before its next one waits for it; it then queues again. It frees the slot whenever it has no
//   next view. --lease 0 frees it after every view. The lease changes when views run, never what
//   they render.
// - A view that fails (a timeout, a page or console error, a crash, no answer within three times
//   --timeout plus a minute) ends the worker: the page is not trusted after it, and capture-set
//   retries the view on a fresh page. The worker says when the view holds its GPU slot (`locked`),
//   so capture-set's watchdog does not count a long wait for the lock as a hang.
//
// Output: artifacts/captures/<name>.png (and <name>.f1.png ... under --frames) and <name>.json: the
// url, viewport, browser, profile `slot`, `ready` (status exact|tolerance|capped|converged, frames
// waited, comparisons, the largest difference, the settled signal's block), the diagnostics
// samples and `diag`, the adapter, `evals` and `readyEvals` (`{ eval, value }` or `{ eval, error }`),
// console errors and failed requests, `meanFrameMs`, and `crops` when the page sets
// `window.__PONLAI_CROPS__` (tools/island/parity/views.mjs). stdout prints each eval, `unsettled` or
// `unstable` when so, the file with the diagnostics, and the errors. A worker writes to the paths
// capture-set names and adds `page`: the visit's index, `load` or `jump`, the view before, the world
// time, the lock wait (`lockKept` when kept), the time to the screenshot and the page's memory, and
// `held` when the page offers `heldRegion()` on its handle: the cells of the captured frame where a
// held history-dependent effect may differ between loads, which diff-set.py --held compares within
// a tolerance (tools/kit/tolerance.py).
//
// GPU lock and browser: a headless browser in a persistent profile slot. A paused capture takes the
// GPU lock shared (its pixels do not depend on what else renders); a running one, or --exclusive,
// takes it alone, since its frame times are reported too (docs/reference/gpu-lock.md). A load takes
// its slot only once the page has its world: until then an init script holds the page's GPU device
// back, and the tool waits without a slot for `window.__PONLAI_WORLD__.ready`, then takes the slot
// and opens the gate (a page without the marker takes it when it asks for its device).
//
// Cost: a page load (seconds warm; a cold one generates the island and compiles every shader); a
// paused capture takes about 5-7 s alone.
//
// Exit codes: 0 captured; 1 an eval threw, the page threw or logged an error, a request failed
// (the screenshot is still written), or a wait timed out; 2 --worker without capture-set's IPC channel.
import './help.mjs';
import { acquireGpuLock, leaseGpuLock, releaseGpuLock } from './gpu-lock.mjs';
import { launchCapture, waitForScene, waitForSettled, waitForStable, stableOptions, stopLoop, startLoop, pngCapturer, READY_FRAMES } from './capture-browser.mjs';
import { evalSource, runInPage } from './page-eval.mjs';
import { mkdir, writeFile } from 'node:fs/promises';
import PROJECT from '../project.mjs';
import { outputDir } from './root.mjs';

/** The scene jump contract's version this tool drives (`__PONLAI__.sceneJump`, packages/render/assemble/SceneJump.ts). */
const SCENE_JUMP = 1;
/** The worker's GPU slot lease in seconds (--lease; see the header). The worker used to free the
 * slot after every view, so a set behind another session's back-to-back timing runs waited one
 * whole timing run per view: nearly all of such a set's time went to waiting, not to its views.
 * Why 20 s: a paused view alone takes about a third of the lease, so such a set advances about four
 * views per timing run before it yields, and still two when three pages share the GPU and a view
 * takes nearly the whole lease; a waiting timing run gets the GPU at most one lease plus the view in
 * flight after its turn, small against the minutes it then holds it, and it already waited for a
 * view in flight (about as long as a lease). A longer lease gives the set little more (the timing
 * runs still alternate with it) and delays every run of a series by as much. Nobody waiting, the
 * slot is kept for the whole set. */
const LEASE = 20;

const args = process.argv.slice( 2 );
const opt = ( k, d ) => { const i = args.indexOf( '--' + k ); return i >= 0 ? args[ i + 1 ] : d; };
const worker = args.includes( '--worker' );
const name = ! worker && args[ 0 ] && ! args[ 0 ].startsWith( '--' ) ? args[ 0 ] : 'shot';
const W = Number( opt( 'w', 1600 ) ), H = Number( opt( 'h', 900 ) );
// the --eval steps, in order: `eval` as given (js or @file), `js` what runs (see the header)
const evalsOf = ( flag ) => args.flatMap( ( a, i ) => a === flag && args[ i + 1 ] ? [ { eval: args[ i + 1 ], js: evalSource( args[ i + 1 ] ) } ] : [] );
const argEvals = evalsOf( '--eval' );
// the --eval-ready steps: the same, run once the page is ready, before the screenshot (see the header)
const readyEvals = evalsOf( '--eval-ready' );
const wait = Number( opt( 'wait', 0 ) ), settle = Number( opt( 'settle', 0 ) );
const stable = stableOptions( ( k ) => k === 'stable-trace' ? ( args.includes( '--stable-trace' ) ? '' : undefined ) : opt( k ) );
const baseUrl = opt( 'url', PROJECT.url );
// the app's globals, as the project's configuration names them, for the page code below (passed
// in: a closure run in the page sees nothing of node's scope)
const PAGE = PROJECT.page;
const sampleCount = Number( opt( 'samples', 1 ) );
const waitFor = opt( 'wait-for', '' );
const timeout = Number( opt( 'timeout', 60000 ) );
// a load's wait for its world, without the slot (generous: a cold grown island under load)
const worldTimeout = timeout * 5;
const frameCount = Math.max( 1, Number( opt( 'frames', 1 ) ) );
const frameEvery = Math.max( 1, Number( opt( 'every', 1 ) ) );
const verbose = args.includes( '--verbose' );
const paused = ( query ) => new URLSearchParams( query ).has( 'paused' );
// the fixed frame count instead of stability: --ready-frames n, or a running scene (it never stands still)
const readyFramesFor = ( query ) => args.includes( '--ready-frames' ) ? Number( opt( 'ready-frames' ) ) : paused( query ) ? null : READY_FRAMES;
// tools/kit/gpu-lock.mjs: shared for a paused scene, exclusive otherwise
const sharedFor = ( query ) => paused( query ) && ! args.includes( '--exclusive' );

// a promise that rejects after `ms` (a page that stopped answering)
function within( promise, ms, what ) {

	let timer;
	return Promise.race( [
		promise.finally( () => clearTimeout( timer ) ),
		new Promise( ( resolve, reject ) => { timer = setTimeout( () => reject( new Error( `${ what }: no answer within ${ ms } ms` ) ), ms ); } ),
	] );

}

/** Opens the capture page and listens to it: `begin()` starts a view's error record. */
async function openPage( viewport ) {

	const capture = await launchCapture( { ...viewport, url: baseUrl } );
	const page = capture.page;
	let current = { errors: [], failures: [] };
	page.on( 'console', ( m ) => { if ( verbose || m.type() === 'error' || m.type() === 'warning' ) current.errors.push( m.type() + ': ' + m.text() ); } );
	page.on( 'pageerror', ( e ) => current.errors.push( 'pageerror: ' + e.message ) );
	page.on( 'crash', () => current.errors.push( 'pageerror: the page crashed' ) );
	page.on( 'response', ( r ) => { if ( r.status() >= 400 ) current.failures.push( `${ r.status() } ${ r.url() }` ); } );
	page.on( 'requestfailed', ( r ) => current.failures.push( `${ r.failure()?.errorText } ${ r.url() }` ) );
	// the GPU gate (loadToWorld, "The slot after the world"): a load gets its GPU device, and so
	// does any GPU work, only once the gate is opened (`asked`: the page has asked for it)
	await page.addInitScript( ( gpuGate ) => {

		if ( ! window.GPUAdapter || window !== window.top ) return;
		let open;
		const gate = new Promise( ( resolve ) => { open = resolve; } );
		const state = window[ gpuGate ] = { open: () => open(), asked: false };
		const proto = GPUAdapter.prototype, requestDevice = proto.requestDevice;
		proto.requestDevice = async function ( ...a ) {

			state.asked = true;
			await gate;
			return requestDevice.apply( this, a );

		};

	}, PAGE.gpuGate );
	return { capture, page, begin: () => ( current = { errors: [], failures: [] } ), record: () => current };

}

/** A fresh load of `url` up to its world, without the GPU (see the header, GPU lock and browser):
 * returns the time it took (ms), or null on a page that does not say when its world is in.
 * The slot after the world: up to now the page's GPU use is held back (the init script in openPage
 * gates `GPUAdapter.requestDevice`), so the page starts its world, then waits in the renderer's
 * start with no device, no render, no compile; the caller then takes the slot and opens the gate,
 * and the page sets up the sky, textures and pipelines, renders and is captured under it. The world's
 * marker is `window.__PONLAI_WORLD__.ready` (packages/island/world/WorldStart.ts, debug pages). The
 * tool used to take the slot before the load, so on a grown island a load held a shared slot through
 * its CPU world generation (7-35 s cold) with the GPU nearly idle. The production bundle starts the
 * world after the quality choice (a top-level await, which once requested the adapter), and the
 * gate stays on the device. The cost: the GPU set-up no longer overlaps the world's generation, so
 * a load takes that much longer from end to end. The world wait has its own deadline (five times
 * --timeout); a page without the marker (an older build) takes the slot once it asks for its
 * device, so its world is generated under the slot as before. The JSON's `page.worldMs` is the
 * load's time to its world. A jump has no load and is unchanged. */
async function loadToWorld( page, url ) {

	const started = Date.now();
	// (not the load event: the page's top-level awaits wait for the gate)
	await page.goto( url, { waitUntil: 'domcontentloaded', timeout: worldTimeout } );
	// the world in, or the device asked for (the renderer's start, after the world's module has run)
	// on a page without the marker
	await page.waitForFunction( ( n ) => window[ n.world ]?.ready || ( ! window[ n.world ] && window[ n.gpuGate ]?.asked ),
		PAGE, { timeout: worldTimeout, polling: 100 } );
	return await page.evaluate( ( n ) => window[ n.world ] !== undefined, PAGE ) ? Date.now() - started : null;

}

const failed = ( record ) => record.evalFailed || record.errors.some( ( e ) => e.startsWith( 'pageerror:' ) || e.startsWith( 'error:' ) ) || record.failures.length > 0;

// the renderer's frame counter (three's animation loop: rendered frames only)
const renderedFrame = ( page ) => page.evaluate( ( n ) => window[ n.handle ]?.renderer.info.frame ?? window[ n.diagnostics ]?.frame ?? 0, PAGE );

/** Brings the page to the view (`load`: the fresh load of `url` loadToWorld started, its GPU gate
 * opened; `jump`: the scene jump to `query`), runs the view's `evals` (js; a throw fails the view)
 * and the --eval `steps` (recorded in the JSON's `evals`; a throw is recorded and fails the capture,
 * which is still taken), waits until it is ready, runs the --eval-ready `readySteps` (recorded as
 * `readyEvals`, likewise) and captures it into `<out>.png` (and further
 * frames), `<out>.json`, with the view's error record (`session.begin()`, called before the load).
 * `pageInfo` (async) adds the JSON's `page`. Returns { out, ready, diag, record, evals, readyEvals }. */
async function captureView( session, { url, query, evals, steps, readySteps = [], out, mode, pageInfo = null } ) {

	const { capture, page } = session;
	const record = session.record();
	if ( mode === 'load' ) {

		await page.evaluate( ( n ) => window[ n.gpuGate ]?.open(), PAGE );
		await page.waitForFunction( ( n ) => window[ n.handle ] && window[ n.diagnostics ], PAGE, { timeout } );

	} else {

		// the scene jump (packages/render/assemble/SceneJump.ts `establishScene`), then a frame at the jump's camera before the
		// evals move it, as a fresh load's first frames are
		const jumped = await within( page.evaluate( async ( { q, handle } ) => {

			const s = window[ handle ];
			await s.establishScene( q );
			s.loop.start();
			return s.renderer.info.frame;

		}, { q: query, handle: PAGE.handle } ), timeout, 'establishScene' );
		await page.waitForFunction( ( { f, handle } ) => window[ handle ].renderer.info.frame > f, { f: jumped, handle: PAGE.handle }, { timeout, polling: 'raf' } );

	}
	// the --eval and --eval-ready steps, recorded (a throw fails the capture)
	const runSteps = async ( list ) => {

		const records = [];
		for ( const step of list ) {

			try { records.push( { eval: step.eval, value: await runInPage( page, step.js, { timeout } ) } ); } catch ( error ) {

				records.push( { eval: step.eval, error: error.message.split( '\n' )[ 0 ] } );
				record.evalFailed = true;

			}

		}
		return records;

	};
	// the view's evals, then the --eval steps, with probe's forms and scope (tools/kit/page-eval.mjs)
	for ( const js of evals ) await runInPage( page, js, { timeout } );
	const evalRecords = await runSteps( steps );
	const readyFrames = readyFramesFor( query );
	// up: the floor's frames after the evals, the loading screen gone (an edit can trigger a Vite reload)
	await waitForScene( page, { frames: readyFrames ?? stable.min, wait, settle, timeout } );
	// a paused temporal page: its rest, started over after the evals (its epoch: what the evals
	// changed need not be in its key), converged; the settled wait below follows, and a rest that
	// converged before the page was quiet is started over (packages/render/post/TemporalRest.ts)
	if ( paused( query ) ) {

		await page.evaluate( ( n ) => window[ n.handle ]?.taa?.epoch.bump( 'capture' ), PAGE );
		await page.waitForFunction( ( n ) => {

			const t = window[ n.diagnostics ]?.taa;
			return ! t || t.restTarget === 0 || ( t.converged && t.restEpoch === window[ n.handle ].taa.epoch.value );

		}, PAGE, { timeout, polling: 100 } );

	}
	if ( waitFor ) await page.waitForFunction( waitFor, null, { timeout: 60000, polling: 100 } );
	// a paused page's settled signal, with the loop running (tools/kit/capture-browser.mjs)
	const settled = readyFrames === null && paused( query ) ? await waitForSettled( page, { timeout } ) : null;
	// ready: the image stopped changing, or a temporal page's rest converged (or the fixed count
	// above); the loop is stopped
	const ready = readyFrames === null ? await waitForStable( page, { ...stable, timeout } ) : { rule: 'frames', frames: readyFrames };
	if ( settled ) ready.settled = settled;
	// the --eval-ready steps, on the frame the screenshot takes (the loop stopped)
	if ( readySteps.length ) await stopLoop( page );
	const readyRecords = await runSteps( readySteps );
	const samples = [];
	for ( let i = 0; i < sampleCount; i ++ ) {

		if ( i > 0 ) await page.waitForTimeout( 1000 );
		samples.push( await page.evaluate( ( n ) => ( { ...window[ n.diagnostics ] } ), PAGE ) );

	}
	const frames = [];
	// the screenshot: the compositor's image, PNG encoded for speed (tools/kit/capture-browser.mjs, "The screenshot")
	const capturer = await pngCapturer( page, { timeout } );
	for ( let i = 0; i < frameCount; i ++ ) {

		if ( i > 0 ) {

			await startLoop( page );
			await page.waitForFunction( ( { f, n } ) => ( window[ n.handle ]?.renderer.info.frame ?? window[ n.diagnostics ]?.frame ?? 0 ) >= f,
				{ f: frames[ i - 1 ].frame + frameEvery, n: PAGE }, { timeout, polling: 'raf' } );

		}
		// no frames queued behind the screenshot (tools/kit/capture-browser.mjs)
		await stopLoop( page );
		const frame = await renderedFrame( page );
		const file = i === 0 ? `${ out }.png` : `${ out }.f${ i }.png`;
		await writeFile( file, await capturer.capture() );
		frames.push( { file, frame } );

	}
	// (the device metrics back before the crop anchors read the page's ratio)
	await capturer.close();
	const diag = await page.evaluate( ( n ) => ( { ...window[ n.diagnostics ] } ), PAGE );
	const crops = await page.evaluate( ( n ) => window[ n.crops ] ?? null, PAGE );
	// a set's capture: where it is compared within the held tolerance (tools/kit/tolerance.py), the frame
	// just captured, read back
	const held = pageInfo ? await page.evaluate( ( n ) => window[ n.handle ]?.heldRegion?.() ?? null, PAGE ) : null;
	session.gpu ??= await page.evaluate( async () => {

		const adapter = await navigator.gpu.requestAdapter();
		return adapter ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description } : null;

	} );
	const visit = pageInfo ? await pageInfo() : null;
	const json = {
		url, viewport: { width: W, height: H }, browser: capture.version, slot: capture.slot, ready, samples, diag, gpu: session.gpu,
		...( evalRecords.length ? { evals: evalRecords } : {} ), ...( readyRecords.length ? { readyEvals: readyRecords } : {} ),
		errors: record.errors, failures: record.failures,
		meanFrameMs: samples.reduce( ( s, d ) => s + d.frameMs, 0 ) / samples.length,
		...( frameCount > 1 ? { frames } : {} ), ...( crops ? { crops } : {} ), ...( held ? { held } : {} ), ...( visit ? { page: visit } : {} ),
	};
	await writeFile( `${ out }.json`, JSON.stringify( json, null, 2 ) );
	return { out: `${ out }.png`, ready, diag, record, evals: evalRecords, readyEvals: readyRecords };

}

// the lines capture.mjs prints for a view
function report( { out, ready, diag, record, evals, readyEvals } ) {

	const lines = [];
	const line = ( e, kind ) => 'error' in e ? `${ kind } failed: ${ e.eval }: ${ e.error }` : `${ kind } ${ e.eval }: ${ JSON.stringify( e.value )?.slice( 0, 300 ) }`;
	for ( const e of evals ) lines.push( line( e, 'eval' ) );
	for ( const e of readyEvals ) lines.push( line( e, 'eval-ready' ) );
	if ( ready.settled?.settled === false ) lines.push( `unsettled: the settled signal did not hold within ${ ready.settled.wallMs } ms (${ JSON.stringify( ready.settled.settle?.pending ?? [] ) }), captured once stable` );
	if ( ready.status === 'capped' ) lines.push( `unstable: still changing after ${ ready.framesWaited } frames (max ${ ready.maxSeen } levels), captured at the cap` );
	lines.push( `${ out } ${ JSON.stringify( diag ) }` );
	if ( record.errors.length ) lines.push( record.errors.slice( 0, verbose ? 60 : 12 ).join( '\n' ) );
	if ( record.failures.length ) lines.push( record.failures.join( '\n' ) );
	return lines.join( '\n' );

}

// the page's memory after a capture (JS heap, DOM, renderer counts), for the worker's JSON
async function pageMemory( page ) {

	const cdp = await page.context().newCDPSession( page );
	try {

		await cdp.send( 'Performance.enable' );
		const { metrics } = await cdp.send( 'Performance.getMetrics' );
		await cdp.send( 'Performance.disable' );
		const m = Object.fromEntries( metrics.map( ( x ) => [ x.name, x.value ] ) );
		const info = await page.evaluate( ( handle ) => {

			const r = window[ handle ].renderer;
			return { geometries: r.info.memory.geometries, textures: r.info.memory.textures, frame: r.info.frame };

		}, PAGE.handle );
		return { jsHeapUsed: m.JSHeapUsedSize, jsHeapTotal: m.JSHeapTotalSize, nodes: m.Nodes, listeners: m.JSEventListeners, ...info };

	} finally {

		await cdp.detach().catch( () => {} );

	}

}

if ( ! worker ) {

	const query = opt( 'q', PROJECT.query );
	const url = `${ baseUrl }?${ query }`;
	const captures = outputDir( '', 'captures' ); // under the working directory
	await mkdir( captures, { recursive: true } );
	const session = await openPage( { viewport: { width: W, height: H }, deviceScaleFactor: Number( opt( 'scale', 1 ) ) } );
	try {

		// the world without the slot, then the rest under it (see the header)
		session.begin();
		await loadToWorld( session.page, url );
		await acquireGpuLock( `capture ${ name }`, { shared: sharedFor( query ) } );
		const shot = await captureView( session, { url, query, evals: [], steps: argEvals, readySteps: readyEvals, out: `${ captures }/${ name }`, mode: 'load' } );
		console.log( report( shot ) );
		if ( failed( shot.record ) ) process.exitCode = 1;

	} finally {

		await session.capture.close();
		releaseGpuLock();

	}

} else {

	// capture-set's worker (see the header): views arrive over IPC
	if ( ! process.send ) { console.error( 'capture.mjs --worker runs under capture-set.mjs (an IPC channel)' ); process.exit( 2 ); }
	const send = ( message ) => new Promise( ( resolve ) => process.send( message, () => resolve() ) );
	const next = () => new Promise( ( resolve ) => {

		process.once( 'message', resolve );
		process.send( { type: 'next' } );

	} );
	// (capture-set gone: no orphaned browser; Playwright closes it on exit)
	process.on( 'disconnect', () => process.exit( 1 ) );
	// (--eval and --eval-ready run on every view and no jump undoes them: a page per view, unless
	// --ready-reads says the --eval-ready steps only read; see the header)
	const onePerPage = args.includes( '--page-per-view' ) || argEvals.length > 0 || ( readyEvals.length > 0 && ! args.includes( '--ready-reads' ) );
	const viewDeadline = timeout * 3 + 60000;
	const lease = Number( opt( 'lease', LEASE ) ) * 1000;
	const session = await openPage( { viewport: { width: W, height: H }, deviceScaleFactor: Number( opt( 'scale', 1 ) ) } );
	let jump = false, previous = null;
	try {

		for ( let index = 0; ; index ++ ) {

			const message = await next();
			if ( message.type !== 'view' ) { releaseGpuLock(); break; }
			const { key, query, evals, out, last = false } = message.view;
			const mode = index === 0 ? 'load' : 'jump';
			const url = `${ baseUrl }?${ query }`;
			const started = Date.now();
			session.begin();
			// a load: its world first, without the slot (loadToWorld, "The slot after the world")
			let worldMs = 0;
			if ( mode === 'load' ) {

				try {

					worldMs = await loadToWorld( session.page, url ) ?? 0;

				} catch ( error ) {

					// a view that fails during the world's generation sends its failed `shot` with no
					// `locked` before it
					await send( { type: 'shot', key, mode, from: previous, seconds: ( Date.now() - started ) / 1000, lockSeconds: 0, ok: false, log: `${ key }: its world did not load: ${ error.message ?? error }` } );
					break;

				}

			}
			// the slot: kept from the view before while the lease allows (see the header), else queued for
			const { waitMs: lockMs, kept } = lease > 0 ? await leaseGpuLock( `capture ${ key }`, { shared: sharedFor( query ), budget: lease } )
				: { waitMs: await acquireGpuLock( `capture ${ key }`, { shared: sharedFor( query ) } ), kept: false };
			// capture-set's watchdog counts from here, not from the view's hand-out
			await send( { type: 'locked', key } );
			let result;
			try {

				const shot = await within( captureView( session, { url, query, evals, steps: argEvals, readySteps: readyEvals, out, mode,
					pageInfo: async () => ( { visit: index, mode, from: previous, pid: process.pid, ...( mode === 'load' ? { worldMs } : {} ), lockMs, ...( kept ? { lockKept: true } : {} ), ms: Date.now() - started - worldMs - lockMs, memory: await pageMemory( session.page ) } ) } ), viewDeadline, `view ${ key }` );
				result = { ok: ! failed( shot.record ), log: report( shot ) };

			} catch ( error ) {

				result = { ok: false, log: `${ key }: ${ error.message ?? error }` };

			} finally {

				if ( lease === 0 ) releaseGpuLock();

			}
			if ( index === 0 && result.ok ) {

				const marker = await session.page.evaluate( ( handle ) => window[ handle ]?.sceneJump ?? null, PAGE.handle ).catch( () => null );
				jump = marker === SCENE_JUMP;
				await send( { type: 'page', jump, marker } );

			}
			await send( { type: 'shot', key, mode, from: previous, seconds: ( Date.now() - started - worldMs - lockMs ) / 1000, lockSeconds: lockMs / 1000, worldSeconds: worldMs / 1000, ...result } );
			// a failed view leaves a page not to be trusted; without the jump, a page per view; after a
			// view whose evals the jump does not undo (`last`), a fresh page for the next
			if ( ! result.ok || ! jump || onePerPage || last ) { releaseGpuLock(); break; }
			previous = key;

		}

	} finally {

		await within( session.capture.close(), 30000, 'close' ).catch( () => {} );
		process.removeAllListeners( 'disconnect' );
		process.disconnect?.();

	}

}
