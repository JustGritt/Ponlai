// The browser and the readiness that tools/kit/capture.mjs and tools/kit/parity/capture-clip.mjs share.
//
// Profile slots. Each capture runs in a persistent browser profile instead of a fresh one, so the
// page's world cache (IndexedDB, packages/island/world/WorldCache.ts) and the browser's GPU shader cache hit
// from the second load on: a cache miss generates the island again (about 3 s alone, a minute
// under a busy CPU) and compiles every shader again. A profile serves one browser at a time, so
// the profiles are slots, `<root>/<browser build>/<n>`, each taken with its own lock: the
// directory `<n>.lock` (mkdir is atomic) holding owner.json (pid), stale once that process is gone.
// A capture takes the lowest free slot, so n jobs at once use slots 0..n-1 and keep them warm. The
// slots are independent of tools/kit/gpu-lock.mjs (taken after it, held only while the browser runs).
// - Affinity: a slot is warm for the pages it has served, not for every page (the world cache is per
//   origin, so per preview port), so the lowest free slot put a page on a cold slot whenever
//   another job held its warm one, and kept it there the next time too. A caller that names its
//   page (`url`) takes the slots that page used last first: `<root>/<browser build>/affinity.json`
//   maps the page's origin plus its build's id (a hash of the page's HTML without the query: a
//   production build's names its hashed bundles; the URL alone when it cannot be read) to the
//   slots it ran in, most recent first. The capture takes the first of them that is free, else the
//   lowest free slot, and moves the slot it took to the front. A list rather than one slot, so
//   that n jobs at once on one page find their n warm slots again. The file is rewritten whole
//   (by rename) on each launch; a lost race between two launches only drops one record, which
//   costs a cold slot, never a wrong capture. Callers without `url` take the lowest free slot.
// - The browser build is in the path because a profile does not move between browser versions:
//   Playwright's Chromium is named by its install folder (`chromium-1243`), a CHROME_PATH browser
//   by its `--version` output.
// - The root is `artifacts/.capture-profiles` of the main checkout (git's common directory; the
//   configuration's `output.profiles`), so every worktree shares the warm slots;
//   <prefix>CAPTURE_PROFILES (the configuration's `machine.envPrefix`, then the name) moves it. Safe to delete.
// - What the profile keeps cannot change a capture: the world cache is keyed on the generator's
//   sources, options and browser (a hit is byte-identical to a miss), and each preview port is its
//   own origin with its own entry; the shader cache is keyed on the shader. The HTTP cache
//   revalidates (Vite's dev and preview servers send `no-cache`). localStorage and sessionStorage
//   (the remembered quality tier, a saved game) are cleared before every page's scripts run.
//
// Readiness by stability. A paused page (MSAA or off; the temporal modes wait for their rest, below)
// is ready when its image stops changing, however many frames that takes: after the evals, `min`
// frames (1) and the loading screen gone (waitForScene), the capture stops the loop, grabs the
// image, and then repeatedly runs `every` frames (2), stops and grabs again (waitForStable). It
// captures once `repeats` comparisons (1) in a row found the image unchanged, on the frame grabbed
// last ("exact"). A paused page still differs from frame to frame on a few views (single frames
// that leave the settled image and come back to it, from a few pixels to 0.9 M; its source is not
// yet found), so after `patience` frames (40) a comparison within the tolerance also counts
// ("tolerance": at most 5000 pixels differ at all); a page still changing after `cap` frames (120)
// is captured anyway and marked "capped". The capture's JSON keeps the record (`ready`: status, frames
// waited, comparisons, the largest difference seen, per comparison its pixel counts).
// - Change is any pixel by default because no two frames carry the same noise: two equal grabs are
//   the settled image, the frame most loads agree on. The tolerance counts pixels that differ at all,
//   not levels: a settling flowers-close differed by 1 level over 5.1 M pixels. The smallest
//   settling step seen was 45 k pixels, the noise between consecutive frames 1 pixel to 0.9 M, so
//   5000 accepts much of the noise and no settling step.
// - The defaults (floor 1, every 2, one comparison) come from the offline replay over those traces
//   (floors 0-10 x every 1-4, one comparison): one equal pair already means the settled image, so
//   the floor only skips grabs that would differ anyway: no combination captured a load early,
//   and floor 1 with every 2 put 48/48 on the settled frame, all exact, mean 8.0 frames, about 2.1 s
//   of grabbing, at most 7.8 s (floor 0: 47 exact and 1 by tolerance, 2.5 s, at most 12.3 s, since
//   the earliest grabs of a settling page differ and each costs a decode and a cycle).
// - Measured: traces of the 24 parity views, two loads
//   each, every frame grabbed for 150 frames, showed every page on its settled image within 1-11
//   frames and changing afterwards only by the single-frame noise. On the GPU, two sets of the 24
//   views with --repeat (96 loads, then with floor 10 and two comparisons) were all captured exact, 4-32
//   frames after the floor (median 4, about 1.4 s of grabbing), every one on the frame most loads
//   of the earlier tools agreed on: no A/A difference, where the fixed waits left 15-17 % of loads
//   off it.
// - The grab is the compositor's image through CDP (pngCapturer, below: `Page.captureScreenshot`,
//   PNG optimised for speed, at the page's pixel ratio): about 0.27 s at 3456x2234, its pixels
//   equal to Playwright's screenshot's (which spends 1.3 s on its PNG), and taking it does not
//   change the next grab. Equal PNG bytes are equal pixels; only a changed frame is decoded (about
//   0.12 s). Reading the canvas in the page (drawImage) gives a blank image once the frame is
//   presented: WebGPU drops its drawing buffer.
// - The grab's CDP session drops the page's device metrics: a clipped `Page.captureScreenshot`
//   emulates the clip on its own session and then restores that session's emulation, which is
//   none, so Playwright's override (its deviceScaleFactor) is gone and `window.devicePixelRatio`
//   reads 1. The frames drawn stay the same (the page follows the ratio only on a resize, and
//   none fires), but anything reading the ratio afterwards is wrong, such as views.mjs's crop
//   anchors in capture-clip. So the capturer's close() (waitForStable ends with it) takes a
//   Playwright screenshot of one pixel (whose own clip emulation is restored to Playwright's
//   metrics) and waits until the page reads the ratio it had before the grabs: about 35 ms per
//   capture. Sending the same metrics again (`setViewportSize` with the same size) does nothing,
//   since Chrome sees no change on Playwright's session; a grab without a clip keeps the metrics
//   but is taken at CSS pixels.
// - Stopping the loop for each grab also waits for a sky bake in flight and drains the GPU queue,
//   so work landing late on a busy GPU is in the next grab rather than frames later.
// - A paused page of the temporal modes (taa, taau, fsr3, gt) is ready once its rest has converged
//   (packages/render/post/TemporalRest.ts: a fixed jitter sequence averaged with exact weights
//   after 8 settle frames, none while a bake or transition runs, then held), so it is not compared
//   at all: the loop is stopped on the held rest and captured ("converged"), the way engines test a temporal
//   image (a deterministic convergence, then a tolerance in the comparison: diff-set --temporal).
//   The stability options do not apply to it (the floor's frames pass inside the rest anyway).
//   Comparing grabs could not help: the loop's start restarts the rest, so each grab was a new rest
//   (about 40 frames) that only sampled the frame noise again (3 views x 2 loads: 0-52 pixels apart,
//   once 37 k; 3-13 s per capture). capture.mjs restarts the rest after the evals and waits for it.
// - "The loading screen gone" is its computed opacity at 0, not its `done` class: the class only
//   starts its fade (index.html: 0.8 s after a 0.2 s delay), and a capture during the fade carries
//   the screen over the whole frame. A temporal rest converges about 40 frames after the evals, so
//   rests were captured through the fade's last percent: up to 2.2 % of the screen's gradient blended
//   in, 0.3-1.9 levels darker over 3-7.7 M px, most on bright surfaces (taa A/A up to 268 against
//   0.13 for the earlier fixed 150 frames). The
//   wait holds on every build, older ones included, and costs at most the fade's remaining second.
// - A running (not paused) capture, or one given --ready-frames n, waits for that fixed count of
//   rendered frames instead (READY_FRAMES, 60, for a running one). `wait` and `settle` (ms) add
//   fixed delays before and after the floor.
// - Before the grabs a paused page also waits for its own settled signal (waitForSettled, which
//   capture.mjs and verify-scene-jump call; verify-growth-swap waits for it too):
//   packages/render/post/Settled.ts holds it false while a pipeline compiles asynchronously or a
//   sun shadow redraw is due, and until 8 quiet frames share the scene's key, so the grabs start on
//   the frame the page keeps. Stability alone met that only through the loading screen, which a
//   fresh load lifts once the sun's shadow map has drawn: frames drawn while a pipeline compiles
//   are identical to each other (three skips the draw), so two of them compare equal. The wait runs with the loop going, since each grab's restart of the loop starts
//   the signal's run over. Under the temporal modes the resolve's rest waits for the haze's, which
//   the capture's own epoch bump restarts. A refresh that lands under a held rest image still
//   leaves the signal `stale`: the wait starts the resolve's rest over then, by an epoch bump that
//   is not a scene bump, so the haze's rest is kept.
//
// The screenshot. The PNGs a capture writes (capture.mjs's frames, capture-clip's crops and full
// frames) are taken by the same capturer as the grab, not by Playwright's `page.screenshot`: the
// same pixels (the compositor's image at the page's ratio, the clip trimmed to the viewport as
// Playwright trims it), only Chrome's fast PNG encoder instead of its default one. At 3456x2234 the
// screenshot takes
// about 0.26 s instead of 1.3 s (1600x900: 0.07 s instead of 0.27 s), plus 35 ms for close(),
// and the file is 8-57 % larger (the 24 parity views'
// two encodings of one frame decode to the same pixels). PNG is lossless, so a file's bytes
// depend on the encoder but its pixels do not: every comparison in the tools decodes (samePixels,
// diff-set.py), and references written with either encoder stay comparable.
//
// Stopping the loop. Headless, Chrome itself back-pressures the page only once seconds of frames
// are queued (no vsync, no frame-rate limit), so a paused page in the test mode bounds its own queue
// to two frames (packages/render/post/FramePacer.ts, "A paused page's queue"): before that, two
// pages at once each queued hundreds of frames of work, and every read-back and screenshot waited
// behind it. A running page is not bounded. `stopLoop` stops the frame loop
// (`__PONLAI__.loop.stop()`: no more frames, then the queue drained) before the screenshot; the
// pixels are the same frame's, the screenshot waits for no queued frames (on an unbounded queue it
// waited for every frame queued), and the GPU is free for other captures meanwhile. Builds without
// `loop` are captured with the loop running.
//
// Importing this module raises the process to the application scheduling policy on macOS
// (tools/kit/qos.mjs): the PNG decode and frame compare of readiness run in this process while it holds
// a GPU slot, and took 5-10x as long at the legacy QoS agent-started processes get on a busy machine.
import './qos.mjs'; // first: the decode and compare below run with the application policy
import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { createHash } from 'node:crypto';
import PROJECT from '../project.mjs';
import { ROOT as REPO, outputDir } from './root.mjs';

// the app's globals and its loading screen, as the project's configuration names them, for the
// page code below (passed in: a closure run in the page sees nothing of node's scope)
const PAGE = PROJECT.page;

/** The rendered frames a running capture, or one given --ready-frames, waits for after the evals. */
export const READY_FRAMES = 60;

/** Readiness by stability's defaults (see the header): the floor before the first grab, frames
 * between grabs, comparisons in a row, the tolerance ({ level, pixels } or null), the frames before
 * the tolerance is accepted, the cap. */
export const STABLE = { min: 1, every: 2, repeats: 1, tolerance: { level: 0, pixels: 5000 }, patience: 40, cap: 120 };
export const LAUNCH_ARGS = [ '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu', '--disable-gpu-vsync', '--disable-frame-rate-limit' ];

function profileRoot() {

	const profiles = process.env[ `${ PROJECT.machine.envPrefix }CAPTURE_PROFILES` ];
	if ( profiles ) return path.resolve( profiles );
	let main = REPO;
	try {

		const common = execFileSync( 'git', [ 'rev-parse', '--path-format=absolute', '--git-common-dir' ], { cwd: REPO, encoding: 'utf8', timeout: 5000 } ).trim();
		if ( path.basename( common ) === '.git' ) main = path.dirname( common );

	} catch { /* not a git checkout: this one */ }
	return outputDir( main, 'profiles' );

}

// a folder name for the browser that will run: profiles do not move between versions
function browserBuild() {

	const exe = process.env.CHROME_PATH || chromium.executablePath();
	if ( ! process.env.CHROME_PATH ) {

		const install = exe.split( path.sep ).find( ( part ) => /^chromium[-_]\d+$/.test( part ) );
		if ( install ) return install;

	}
	let version = '';
	try { version = execFileSync( exe, [ '--version' ], { encoding: 'utf8', timeout: 10000 } ).trim(); } catch { /* named by its path alone */ }
	return ( version || exe ).replace( /[^A-Za-z0-9.]+/g, '-' ).replace( /^-|-$/g, '' ).slice( 0, 80 );

}

function alive( pid ) {

	if ( ! ( pid > 0 ) ) return false;
	try { process.kill( pid, 0 ); return true; } catch ( error ) { return error.code === 'EPERM'; }

}

const readJson = ( file ) => {

	try { return JSON.parse( readFileSync( file, 'utf8' ) ); } catch { return null; }

};

// takes slot n's lock, or returns false when another live process holds it
function tryTake( lock ) {

	try {

		mkdirSync( lock );

	} catch ( error ) {

		if ( error.code !== 'EEXIST' ) throw error;
		const owner = readJson( path.join( lock, 'owner.json' ) );
		// (a lock without its owner file yet is being taken; after 10 s it was abandoned)
		let age = 0;
		try { age = Date.now() - statSync( lock ).mtimeMs; } catch { return false; }
		if ( owner ? alive( owner.pid ) : age < 10000 ) return false;
		// stale: move it aside, then check it is still the one judged stale (two processes may both
		// find it stale, and the slower must not delete the faster one's fresh lock)
		const aside = `${ lock }.stale-${ process.pid }-${ Date.now() }`;
		try { renameSync( lock, aside ); } catch { return false; }
		const moved = readJson( path.join( aside, 'owner.json' ) );
		if ( moved && moved.pid !== owner?.pid && alive( moved.pid ) ) {

			try { renameSync( aside, lock ); } catch { /* its place was taken meanwhile */ }
			return false;

		}
		rmSync( aside, { recursive: true, force: true } );
		return tryTake( lock );

	}
	writeFileSync( path.join( lock, 'owner.json' ), JSON.stringify( { pid: process.pid, label: path.basename( process.argv[ 1 ] ?? 'node' ), start: new Date().toISOString() } ) );
	return true;

}

// the affinity key of a page: its origin and its build's id (see the header)
async function affinityKey( url ) {

	const page = new URL( url );
	page.search = '';
	page.hash = '';
	try {

		const response = await fetch( page, { signal: AbortSignal.timeout( 5000 ) } );
		if ( response.ok ) return `${ page.origin } ${ createHash( 'sha1' ).update( await response.text() ).digest( 'hex' ).slice( 0, 12 ) }`;

	} catch { /* unreadable: keyed by the URL */ }
	return page.href;

}

// takes a free slot in `dir`: the page's recent slots first, then the lowest; records the one taken
async function takeSlot( dir, url ) {

	const file = path.join( dir, 'affinity.json' );
	const key = url ? await affinityKey( url ) : null;
	const recent = key ? readJson( file )?.[ key ] ?? [] : [];
	let slot = recent.find( ( n ) => tryTake( path.join( dir, `${ n }.lock` ) ) );
	if ( slot === undefined ) {

		slot = 0;
		while ( ! tryTake( path.join( dir, `${ slot }.lock` ) ) ) slot ++;

	}
	if ( key ) {

		const map = readJson( file ) ?? {};
		map[ key ] = [ slot, ...( map[ key ] ?? [] ).filter( ( n ) => n !== slot ) ];
		const tmp = `${ file }.${ process.pid }`;
		writeFileSync( tmp, JSON.stringify( map, null, '\t' ) + '\n' );
		renameSync( tmp, file );

	}
	return slot;

}

let held = null;
function release() {

	if ( held ) rmSync( held, { recursive: true, force: true } );
	held = null;

}
let installed = false;
function installRelease() {

	if ( installed ) return;
	installed = true;
	process.on( 'exit', release );
	for ( const [ signal, code ] of [ [ 'SIGINT', 130 ], [ 'SIGTERM', 143 ], [ 'SIGHUP', 129 ] ] ) {

		process.on( signal, () => { release(); process.exit( code ); } );

	}

}

/** Launches the capture browser in a free profile slot (the slots `url`'s page used last first, else
 * the lowest; see the header), with one page of the given viewport and device scale factor, and
 * `args` as its switches (LAUNCH_ARGS; verify-first-use passes its own, vsync on). Returns { context, page, version, slot, profile, close }:
 * `version` as Playwright's browser.version() reports it (the parity tools compare it between
 * sets), `close()` closes the browser and frees the slot. */
export async function launchCapture( { viewport, deviceScaleFactor = 1, args = LAUNCH_ARGS, url = null } ) {

	const dir = path.join( profileRoot(), browserBuild() );
	mkdirSync( dir, { recursive: true } );
	installRelease();
	const slot = await takeSlot( dir, url );
	held = path.join( dir, `${ slot }.lock` );
	const profile = path.join( dir, String( slot ) );
	let context;
	try {

		// Playwright's full Chromium (`npx playwright install chromium`; the headless shell may lack
		// WebGPU), or another build through CHROME_PATH; no vsync, so frame times measure the GPU
		context = await chromium.launchPersistentContext( profile, {
			headless: true,
			...( process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : { channel: 'chromium' } ),
			args, viewport, deviceScaleFactor,
		} );

	} catch ( error ) {

		release();
		throw error;

	}
	await context.addInitScript( () => {

		if ( window !== window.top ) return;
		try { localStorage.clear(); sessionStorage.clear(); } catch { /* storage unavailable */ }

	} );
	const page = context.pages()[ 0 ] ?? await context.newPage();
	// (a persistent context has no Browser object to ask: the same string through the protocol)
	const cdp = await context.newCDPSession( page );
	const { product } = await cdp.send( 'Browser.getVersion' );
	await cdp.detach();
	const version = product.slice( product.indexOf( '/' ) + 1 );
	return {
		context, page, version, slot, profile,
		async close() {

			try { await context.close(); } finally { release(); }

		},
	};

}

/** Waits until the page is ready to capture after the evals: `frames` frames rendered since this
 * call and the loading screen gone (absent, or its computed opacity 0: its fade-out finished; see
 * the header), with the optional fixed delays `wait` (before) and `settle` (after), in ms. Call it
 * right after the evals. */
export async function waitForScene( page, { frames = READY_FRAMES, wait = 0, settle = 0, timeout = 60000 } = {} ) {

	const from = await page.evaluate( ( n ) => window[ n.handle ]?.renderer.info.frame ?? window[ n.diagnostics ]?.frame ?? 0, PAGE );
	if ( wait > 0 ) await page.waitForTimeout( wait );
	await page.waitForFunction( ( { f, n } ) => {

		if ( ( window[ n.handle ]?.renderer.info.frame ?? window[ n.diagnostics ]?.frame ?? 0 ) < f ) return false;
		const loading = document.getElementById( n.loading );
		return ! loading || getComputedStyle( loading ).opacity === '0';

	}, { f: from + frames, n: PAGE }, { timeout, polling: 'raf' } );
	if ( settle > 0 ) await page.waitForTimeout( settle );

}

/** Parses `--stable-tolerance`: `exact` (or `0`) for none, else `<level>:<pixels>`. */
export function parseTolerance( text ) {

	if ( text === undefined ) return STABLE.tolerance;
	if ( text === 'exact' || text === '0' ) return null;
	const m = /^(\d+):(\d+)$/.exec( text );
	if ( ! m ) throw new Error( `bad --stable-tolerance ${ text } (exact, or <level>:<pixels>)` );
	return { level: Number( m[ 1 ] ), pixels: Number( m[ 2 ] ) };

}

/** The stability options of capture.mjs and capture-clip.mjs, from their parsed options (`opt( k )`
 * returns the option's text or undefined). */
export function stableOptions( opt ) {

	const num = ( k, d ) => {

		const t = opt( k );
		if ( t === undefined ) return d;
		const v = Number( t );
		if ( ! Number.isInteger( v ) || v < 0 ) throw new Error( `bad --${ k } ${ t }` );
		return v;

	};
	return {
		min: num( 'stable-min', STABLE.min ), every: Math.max( 1, num( 'stable-every', STABLE.every ) ),
		repeats: Math.max( 1, num( 'stable-repeats', STABLE.repeats ) ),
		tolerance: parseTolerance( opt( 'stable-tolerance' ) ), patience: num( 'stable-patience', STABLE.patience ), cap: num( 'stable-cap', STABLE.cap ),
		trace: opt( 'stable-trace' ) !== undefined,
	};

}

// ------------------------------------------------------------------ PNG decoding (8-bit RGB or RGBA,
// non-interlaced: what Chrome's screenshots are)

/** Decodes a PNG into { width, height, data } with `data` RGB, 3 bytes per pixel. */
export function decodePng( buffer ) {

	let pos = 8, width = 0, height = 0, colorType = 0, depth = 0;
	const idat = [];
	while ( pos < buffer.length ) {

		const length = buffer.readUInt32BE( pos ), type = buffer.toString( 'latin1', pos + 4, pos + 8 );
		const data = buffer.subarray( pos + 8, pos + 8 + length );
		if ( type === 'IHDR' ) {

			width = data.readUInt32BE( 0 ); height = data.readUInt32BE( 4 ); depth = data[ 8 ]; colorType = data[ 9 ];
			if ( depth !== 8 || data[ 12 ] !== 0 || ( colorType !== 2 && colorType !== 6 ) ) throw new Error( `unsupported PNG (depth ${ depth }, colour type ${ colorType }, interlace ${ data[ 12 ] })` );

		} else if ( type === 'IDAT' ) idat.push( data );
		else if ( type === 'IEND' ) break;
		pos += 12 + length;

	}
	const channels = colorType === 6 ? 4 : 3, stride = width * channels;
	const raw = zlib.inflateSync( Buffer.concat( idat ) );
	const out = new Uint8Array( width * height * 3 );
	const prev = new Uint8Array( stride ), line = new Uint8Array( stride );
	for ( let y = 0; y < height; y ++ ) {

		const filter = raw[ y * ( stride + 1 ) ], src = raw.subarray( y * ( stride + 1 ) + 1, ( y + 1 ) * ( stride + 1 ) );
		for ( let x = 0; x < stride; x ++ ) {

			const a = x >= channels ? line[ x - channels ] : 0, b = prev[ x ], c = x >= channels ? prev[ x - channels ] : 0;
			let v = src[ x ];
			if ( filter === 1 ) v += a;
			else if ( filter === 2 ) v += b;
			else if ( filter === 3 ) v += ( a + b ) >> 1;
			else if ( filter === 4 ) { const p = a + b - c, pa = Math.abs( p - a ), pb = Math.abs( p - b ), pc = Math.abs( p - c ); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
			line[ x ] = v & 255;

		}
		for ( let x = 0, o = y * width * 3; x < width; x ++ ) for ( let k = 0; k < 3; k ++ ) out[ o ++ ] = line[ x * channels + k ];
		prev.set( line );

	}
	return { width, height, data: out };

}

// the steps of a comparison's histogram: pixels whose largest channel difference exceeds each
const STEPS = [ 0, 1, 2, 4, 8, 16, 32 ];

/** Compares two decoded frames: the largest channel difference, the pixels above each of STEPS'
 * levels, and the pixels above `level`. */
export function compareFrames( a, b, level = 0 ) {

	if ( a.width !== b.width || a.height !== b.height ) return { max: 255, over: STEPS.map( () => a.width * a.height ), overLevel: a.width * a.height };
	const x = a.data, y = b.data, over = new Array( STEPS.length ).fill( 0 );
	let max = 0, overLevel = 0;
	for ( let i = 0; i < x.length; i += 3 ) {

		const d = Math.max( Math.abs( x[ i ] - y[ i ] ), Math.abs( x[ i + 1 ] - y[ i + 1 ] ), Math.abs( x[ i + 2 ] - y[ i + 2 ] ) );
		if ( d === 0 ) continue;
		if ( d > max ) max = d;
		if ( d > level ) overLevel ++;
		for ( let j = 0; j < STEPS.length && d > STEPS[ j ]; j ++ ) over[ j ] ++;

	}
	return { max, over, overLevel };

}

/** A PNG capturer on a CDP session of its own (see "The screenshot" in the header): `capture( clip )`
 * returns the PNG of the viewport, or of `clip` ({ x, y, width, height } in CSS px, trimmed to the
 * viewport as Playwright trims it), at the page's pixel ratio, encoded by Chrome's fast PNG
 * encoder: the pixels of Playwright's `page.screenshot` with the same clip. `close()` detaches and
 * gives the page its device metrics back (the screenshots dropped them); call it before anything reads
 * `devicePixelRatio`. */
export async function pngCapturer( page, { timeout = 60000 } = {} ) {

	const cdp = await page.context().newCDPSession( page );
	// (a CDP session of its own has no device metrics override: its scale 1 is CSS pixels, so the
	// clip's scale is the page's ratio)
	const { width, height } = page.viewportSize();
	const scale = await page.evaluate( () => devicePixelRatio );
	return {
		scale,
		async capture( clip = null ) {

			let c = { x: 0, y: 0, width, height };
			if ( clip ) {

				const x = Math.max( 0, Math.min( clip.x, width ) ), y = Math.max( 0, Math.min( clip.y, height ) );
				const x2 = Math.max( 0, Math.min( clip.x + clip.width, width ) ), y2 = Math.max( 0, Math.min( clip.y + clip.height, height ) );
				c = { x, y, width: Math.floor( x2 - x + 1e-3 ), height: Math.floor( y2 - y + 1e-3 ) };

			}
			const { data } = await cdp.send( 'Page.captureScreenshot', { format: 'png', optimizeForSpeed: true, clip: { ...c, scale }, captureBeyondViewport: false } );
			return Buffer.from( data, 'base64' );

		},
		async close() {

			await cdp.detach().catch( () => {} );
			// a screenshot of one pixel through Playwright's own session applies its metrics again
			await page.screenshot( { clip: { x: 0, y: 0, width: 1, height: 1 }, timeout } );
			await page.waitForFunction( ( ratio ) => devicePixelRatio === ratio, scale, { timeout } );

		},
	};

}

/** Whether two PNGs hold the same pixels (equal bytes, or equal once decoded: the bytes depend on
 * the encoder, see "The screenshot" in the header). */
export function samePixels( a, b ) {

	if ( a.equals( b ) ) return true;
	const x = decodePng( a ), y = decodePng( b );
	return x.width === y.width && x.height === y.height && Buffer.compare( x.data, y.data ) === 0;

}

/** Waits, with the frame loop running, until the page's settled signal holds (packages/render/post/Settled.ts:
 * the paused page will draw the same frame from now on; see Readiness by stability above) and
 * returns the record a capture's JSON keeps as `ready.settled`: { settled, framesWaited, wallMs },
 * with the signal's `settle` block when it did not hold within `timeout` (ms; no error: the
 * stability wait follows and the screenshot is still taken), or { signal: false } on a page that has no
 * signal to wait for (an older build, or a temporal page without a rest, which never settles).
 * Call it before waitForStable: its grabs restart the loop, a scene event that starts the signal's
 * run over, so the signal cannot hold during them. */
export async function waitForSettled( page, { timeout = 60000 } = {} ) {

	return page.evaluate( async ( { timeout, n } ) => {

		const s = window[ n.handle ], diag = () => window[ n.diagnostics ];
		if ( typeof diag()?.settled !== 'boolean' || diag().taa?.restTarget === 0 ) return { signal: false };
		s.loop?.start();
		const from = s.renderer.info.frame, start = performance.now();
		const settled = await new Promise( ( resolve ) => {

			const tick = () => {

				if ( diag().settled === true ) resolve( true );
				else if ( performance.now() - start > timeout ) resolve( false );
				else {

					// a temporal rest that converged before the page was quiet starts over (Settled.ts `stale`),
					// not as a scene bump, which would restart the haze's rest as well
					if ( diag().settle?.stale ) s.taa?.epoch.bump( 'settle', false );
					requestAnimationFrame( tick );

				}

			};
			tick();

		} );
		return { settled, framesWaited: s.renderer.info.frame - from, wallMs: Math.round( performance.now() - start ), ...( settled ? {} : { settle: diag().settle } ) };

	}, { timeout, n: PAGE } );

}

/** Waits until the rendered image stops changing (see Readiness by stability above) and returns
 * the record a capture's JSON keeps; the loop is stopped on return, on the frame compared last.
 * A paused page of the temporal modes waits for its converged rest instead (waitForRest).
 * Call it once the page is up (waitForScene with the floor's frames). */
export async function waitForStable( page, { every = STABLE.every, repeats = STABLE.repeats, tolerance = STABLE.tolerance,
	patience = STABLE.patience, cap = STABLE.cap, trace = false, timeout = 60000 } = {} ) {

	// a paused page of the temporal modes shows its rest image once converged: no comparisons
	const temporal = await page.evaluate( ( n ) => {

		const t = window[ n.diagnostics ]?.taa;
		return !! ( t && t.restTarget > 0 && window[ n.handle ].taa );

	}, PAGE );
	if ( temporal ) return waitForRest( page, { timeout } );
	const capturer = await pngCapturer( page, { timeout } );
	const ms = { cycle: 0, grab: 0, decode: 0, compare: 0 };
	let decodes = 0;
	// the page's current image as the compositor shows it (the loop is stopped: the frame a screenshot
	// takes next); equal bytes are equal pixels, so only a frame that changed is decoded
	const grab = async () => {

		const t = performance.now();
		const png = await capturer.capture();
		ms.grab += performance.now() - t;
		return { png, pixels: null };

	};
	const pixels = ( shot ) => {

		if ( ! shot.pixels ) {

			const t = performance.now();
			shot.pixels = decodePng( shot.png );
			ms.decode += performance.now() - t;
			decodes ++;

		}
		return shot.pixels;

	};
	// the loop runs `every` frames, then stops with the GPU queue drained; returns the frame count
	const advance = ( every ) => page.evaluate( async ( { every, timeout, handle } ) => {

		const s = window[ handle ], frameNow = () => s.renderer.info.frame;
		const target = frameNow() + every, deadline = performance.now() + timeout;
		s.loop.start();
		await new Promise( ( resolve, reject ) => {

			const tick = () => {

				if ( frameNow() >= target ) resolve();
				else if ( performance.now() > deadline ) reject( new Error( `waitForStable: no frame ${ target } within ${ timeout } ms (at ${ frameNow() })` ) );
				else requestAnimationFrame( tick );

			};
			tick();

		} );
		await s.loop.stop();
		return frameNow();

	}, { every, timeout, handle: PAGE.handle } );
	try {

		const start = performance.now();
		const first = await page.evaluate( async ( handle ) => {

			const s = window[ handle ];
			await s.loop.stop();
			return s.renderer.info.frame;

		}, PAGE.handle );
		let previous = await grab();
		const history = [];
		let inExact = 0, inTolerance = 0, status = null, maxSeen = 0, waited = 0;
		while ( status === null ) {

			let t = performance.now();
			waited = ( await advance( every ) ) - first;
			ms.cycle += performance.now() - t;
			const current = await grab();
			let c = { max: 0, over: STEPS.map( () => 0 ), overLevel: 0 };
			if ( ! current.png.equals( previous.png ) ) {

				const a = pixels( previous ), b = pixels( current );
				t = performance.now();
				c = compareFrames( a, b, tolerance ? tolerance.level : 0 );
				ms.compare += performance.now() - t;

			}
			previous.pixels = null;
			previous = current;
			history.push( [ waited, c.max, ...c.over, ...( trace ? [ createHash( 'sha1' ).update( current.png ).digest( 'hex' ).slice( 0, 8 ) ] : [] ) ] );
			maxSeen = Math.max( maxSeen, c.max );
			const exact = c.max === 0, within = tolerance !== null && c.overLevel <= tolerance.pixels;
			inExact = exact ? inExact + 1 : 0;
			inTolerance = exact || within ? inTolerance + 1 : 0;
			if ( inExact >= repeats ) status = 'exact';
			else if ( inTolerance >= repeats && waited >= patience ) status = 'tolerance';
			else if ( waited >= cap ) status = 'capped';
			else if ( performance.now() - start > timeout ) throw new Error( `waitForStable: not stable within ${ timeout } ms (${ waited } frames)` );

		}
		const n = history.length;
		return {
			rule: 'stable', status, every, repeats, tolerance, patience, cap,
			framesWaited: waited, comparisons: n, maxSeen, wallMs: Math.round( performance.now() - start ),
			ms: { cycle: + ( ms.cycle / n ).toFixed( 1 ), grab: + ( ms.grab / ( n + 1 ) ).toFixed( 1 ), decode: decodes ? + ( ms.decode / decodes ).toFixed( 1 ) : 0, compare: + ( ms.compare / n ).toFixed( 1 ) },
			// per comparison: [ frames waited, largest channel difference, pixels differing by more than 0, 1, 2, 4, 8, 16, 32 levels
			// (, with `trace`, the grab's PNG hash) ]
			history,
		};

	} finally {

		// (the grabs dropped Playwright's device metrics: see the header)
		await capturer.close();

	}

}

/** A paused temporal page's readiness (see the header): its rest converged under the current epoch,
 * then the loop stopped (the GPU queue drained) with the rest's image held. Returns the record a
 * capture's JSON keeps: frames rendered while waiting (0 when the caller already waited), the
 * rest's epoch, restarts and completed rests, wall ms. */
export async function waitForRest( page, { timeout = 60000 } = {} ) {

	const start = performance.now();
	const record = await page.evaluate( async ( { timeout, n } ) => {

		const s = window[ n.handle ], diag = () => window[ n.diagnostics ].taa;
		const converged = () => diag().converged && diag().restEpoch === s.taa.epoch.value;
		const from = s.renderer.info.frame, deadline = performance.now() + timeout;
		await new Promise( ( resolve, reject ) => {

			const tick = () => {

				if ( converged() ) resolve();
				else if ( performance.now() > deadline ) reject( new Error( `waitForRest: no converged rest within ${ timeout } ms (rest ${ diag().restFrames } of ${ diag().restTarget } frames, epoch ${ diag().restEpoch })` ) );
				else requestAnimationFrame( tick );

			};
			tick();

		} );
		await s.loop.stop();
		return { framesWaited: s.renderer.info.frame - from, restEpoch: s.taa.epoch.value, restTarget: diag().restTarget,
			restCompleted: s.taa.rest.completed, restRestarts: s.taa.rest.restarts };

	}, { timeout, n: PAGE } );
	return { rule: 'rest', status: 'converged', temporal: true, ...record, wallMs: Math.round( performance.now() - start ) };

}

/** Stops the page's frame loop and waits for the GPU queue to drain (see the header); false when
 * the build has no `loop`. */
export function stopLoop( page ) {

	return page.evaluate( async ( handle ) => {

		const loop = window[ handle ]?.loop;
		if ( ! loop ) return false;
		await loop.stop();
		return true;

	}, PAGE.handle );

}

/** Starts the frame loop again after stopLoop. */
export function startLoop( page ) {

	return page.evaluate( ( handle ) => window[ handle ]?.loop?.start(), PAGE.handle );

}
