// The page-side evaluation that tools/kit/probe.mjs and tools/kit/capture.mjs share, so an `--eval` means
// the same in both: the same forms, the same scope, the same value in the JSON. capture's `--eval`
// used to be Playwright's own string evaluation: inline js only (agents passed `$(cat file)`), no
// `until`, so eval scripts carried polling loops of their own, and its value was dropped.
//
// Forms. `js` is an expression (`D.timings`), a function (`async () => { ... }`, called), or
// statements with a `return`; `await` works in all three. `@file` reads it from a file
// (`evalSource`).
// Scope. `P` = `window.__PONLAI__`, `D` = `window.__THREE_GAME_DIAGNOSTICS__` as the step starts
// (the page rebuilds that object every 30 frames: most of its counters lag by up to 30 frames,
// `D.frame` is the frame it was built on; `settled`, `settle` and `taa` are current every frame;
// read live counters from `P`, such as `P.renderer.info.frame`), `frames( n )` (resolves after n
// rendered frames), `sleep( ms )`, `until( fn, timeoutMs )` (resolves with fn's first truthy value,
// checked every rendered frame, or every 100 ms while the loop is stopped; rejects after timeoutMs,
// the tool's --timeout by default, the time spent giving the GPU slot up not counted), `pipelines` (which pipelines an action built, tools/
// pipeline-trace.mjs: `const m = pipelines.mark()` installs the tracer on first use, and
// `pipelines.since( m )` lists every pipeline created after the mark with its kind, label, material,
// object, target, frame and the app's stack; pipelines created before the first mark are not listed).
// Value. Serialised in the page: cycles become "[cycle]", functions "[function]", bigints strings,
// typed arrays plain arrays (the first 64 entries), three's vectors, colours and matrices their
// `toArray()`, undefined null.
// Giving the GPU slot up (`slotBinding`, probe.mjs's shared hold only): while the page's frame loop
// is stopped (`P.loop.running` false), `until` and `sleep` call that binding (page.exposeFunction)
// at most once a second; node gives its shared slot up there if a request ranked before its next
// one waits, and resolves only once it holds the slot again. So a long wait on CPU work (worker
// generation with the loop stopped) does not keep an exclusive run waiting, and the eval's next line
// never renders without the slot. `frames` never offers it (it waits for rendering). capture.mjs
// passes no binding: its captures are short and leased already.
// Rendered frames, not animation frames. `frames( n )` here and `renderedFrames( page, n )` below
// (for tools with a Playwright page of their own) count the page's own frames
// (`renderer.info.frame`, which three's loop bumps on the animation frames the page renders). An
// animation frame is not one: the frame pacer skips the refreshes between its frames, and a paused
// page in the test mode holds a due frame while two rendered frames are still queued on the GPU
// (packages/render/post/FramePacer.ts, "A paused page's queue"). A step that sets a state and waited a count of
// animation frames read the previous frame whenever the page did not tick after the state was set:
// the handoff check, beside another page, read the key and the exposure a frame late on 6 of its
// 324 samples, each a sample with no rendered frame after its clock was set. Rendered frames wait for
// the state the next tick applies; a still, converged image is waited for by `settled` (capture-
// browser's `waitForSettled`), never by a count of frames of either kind.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import PROJECT from '../project.mjs';
import { ROOT } from './root.mjs';

// the app's globals (the project's configuration names them) and the module that gives `pipelines`
// (a project without one: an eval that uses `pipelines` fails with that message)
const { handle: HANDLE, diagnostics: DIAGNOSTICS } = PROJECT.page;
const NO_TRACER = 'this project has no pipeline tracer (tools/project.mjs: pipelines is null)';
const installScript = PROJECT.pipelines ? ( await import( pathToFileURL( path.join( ROOT, PROJECT.pipelines ) ).href ) ).installScript
	: () => `( () => { throw new Error( ${ JSON.stringify( NO_TRACER ) } ); } )()`;

// (installed in the page on first use only: a probe that never asks pays nothing)
const TRACE = installScript();

/** An `--eval` argument's js: `@file` reads the file, anything else is the js itself. */
export const evalSource = ( text ) => text.startsWith( '@' ) ? readFileSync( text.slice( 1 ), 'utf8' ) : text;

/** The page's wait for `n` more rendered frames, as the source of a function `( n, timeout ) =>
 * Promise` (see the header, "Rendered frames"): it resolves once `renderer.info.frame` has risen by
 * `n`, rejects at once while the page's loop is stopped (`__PONLAI__.loop.running` false: no frame
 * will come) and after `timeout` ms. Every tool that waits for frames in the page runs this one:
 * `renderedFrames` from node, `installWaitFrames` as `window.__waitFrames` inside a page's own
 * evaluate, and a tool whose page code is a string splices it in (`( ${ WAIT_FRAMES } )( n, ms )`). */
export const WAIT_FRAMES = `( n, timeout = 60000 ) => new Promise( ( resolve, reject ) => {
	const s = window.${ HANDLE }, count = () => s.renderer.info.frame;
	const end = count() + n, deadline = performance.now() + timeout;
	const tick = () => {
		if ( count() >= end ) resolve();
		else if ( s.loop?.running === false ) reject( new Error( 'waitFrames: the loop is stopped, no frame ' + end + ' will come (at ' + count() + ')' ) );
		else if ( performance.now() > deadline ) reject( new Error( 'waitFrames: no frame ' + end + ' within ' + timeout + ' ms (at ' + count() + ', loop.running ' + s.loop?.running + ')' ) );
		else requestAnimationFrame( tick );
	};
	tick();
} )`;

/** Resolves once the page has rendered `n` more frames (WAIT_FRAMES): what a state set before the
 * call looks like on the page's next frames. */
export function renderedFrames( page, n = 1, { timeout = 60000 } = {} ) {

	return page.evaluate( `( ${ WAIT_FRAMES } )( ${ n }, ${ timeout } )` );

}

/** Gives the page `window.__waitFrames( n, timeout )` (WAIT_FRAMES), now and on every load after,
 * for waits inside a tool's own `page.evaluate` closures. */
export async function installWaitFrames( page ) {

	const js = `window.__waitFrames = ${ WAIT_FRAMES };`;
	await page.addInitScript( js );
	await page.evaluate( js );

}

/** Runs `js` in the page with the shared scope (see the header) and returns its value, serialised
 * in the page. `mode` 'until' polls it until truthy and returns `{ ms, frames }` instead. Rejects
 * when the js throws or an `until` times out (`timeout` ms). */
export function runInPage( page, js, { mode = 'eval', timeout = 60000, slotBinding = null } = {} ) {

	return page.evaluate( async ( { js, mode, timeout, trace, slotBinding, handle, diagnostics } ) => {

		const frameNow = () => window[ handle ]?.renderer.info.frame ?? window[ diagnostics ]?.frame ?? 0;
		const pause = ( ms ) => new Promise( ( resolve ) => setTimeout( resolve, ms ) );
		// a frame, or 100 ms when the loop stands (rAF still fires, but no frame is rendered)
		const tick = () => new Promise( ( resolve ) => { const t = setTimeout( resolve, 100 ); requestAnimationFrame( () => { clearTimeout( t ); resolve(); } ); } );
		// Offers the GPU slot to node while the loop stands (see the header): resolves once node holds
		// it again, with the time it was given up (0: kept). Once a second, since each offer reads the
		// lock's queue.
		const giveUp = slotBinding ? window[ slotBinding ] : null;
		let offered = - Infinity;
		const offer = async () => {

			if ( ! giveUp || window[ handle ]?.loop?.running !== false || performance.now() - offered < 1000 ) return 0;
			const ms = await giveUp();
			offered = performance.now();
			return ms;

		};
		const sleep = async ( ms ) => {

			if ( ! giveUp ) return pause( ms );
			const end = performance.now() + ms;
			for ( let left = ms; left > 0; left = end - performance.now() ) {

				await offer();
				await pause( Math.min( left, 100 ) );

			}

		};
		const until = async ( fn, ms = timeout ) => {

			let deadline = performance.now() + ms;
			for ( ;; ) {

				const v = await fn();
				if ( v ) return v;
				if ( performance.now() > deadline ) throw new Error( `until: not true within ${ ms } ms` );
				// a wait for the slot is the queue's time, not the page's
				deadline += await offer();
				await tick();

			}

		};
		const tracer = () => window.__pipelineTrace ?? ( 0, eval )( trace );
		const pipelines = { mark: () => tracer().mark(), since: ( mark ) => tracer().since( mark ) };
		const frames = ( n ) => { const target = frameNow() + n; return until( () => frameNow() >= target ); };
		// an expression, else statements; a function value is called (the test parses as an async
		// function, so an expression with `await` is still an expression)
		const AsyncFunction = ( async () => {} ).constructor;
		let body;
		try { new AsyncFunction( `return (${ js }\n)` ); body = `return (${ js }\n);`; } catch { body = js; }
		const fn = new AsyncFunction( 'P', 'D', 'frames', 'sleep', 'until', 'pipelines', body );
		const value = async () => {

			const v = await fn( window[ handle ], window[ diagnostics ], frames, sleep, until, pipelines );
			return typeof v === 'function' ? await v() : v;

		};
		const serialise = ( v ) => {

			const seen = new WeakSet();
			return JSON.stringify( v, function ( key, x ) {

				if ( typeof x === 'function' ) return '[function]';
				if ( typeof x === 'bigint' ) return String( x );
				if ( x === undefined && key === '' ) return null;
				if ( x && typeof x === 'object' ) {

					if ( ArrayBuffer.isView( x ) ) return Array.from( x.length > 64 ? x.subarray( 0, 64 ) : x );
					if ( ( x.isVector2 || x.isVector3 || x.isVector4 || x.isColor || x.isQuaternion || x.isMatrix3 || x.isMatrix4 || x.isEuler ) && x.toArray ) return x.toArray();
					if ( seen.has( x ) ) return '[cycle]';
					seen.add( x );

				}
				return x;

			} );

		};
		if ( mode === 'eval' ) return serialise( await value() );
		const from = frameNow(), start = performance.now();
		await until( value );
		return JSON.stringify( { ms: Math.round( performance.now() - start ), frames: frameNow() - from } );

	}, { js, mode, timeout, trace: TRACE, slotBinding, handle: HANDLE, diagnostics: DIAGNOSTICS } ).then( ( text ) => text === undefined ? undefined : JSON.parse( text ) );

}
