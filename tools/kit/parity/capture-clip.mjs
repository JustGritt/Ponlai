// Captures a named clip frame by frame with the page's stepper, as crops that follow their subjects.
//
// It answers how the anti-aliasing modes hold up in motion: a view of the views table set in motion
// by a named clip of the clips table (both through tools/kit/parity/tables.mjs; the clips are in
// tools/island/parity/clips.mjs), rendered with the page's deterministic stepper (the handle's
// `loop.step`, debug builds only: fixed dt, no wall clock, the frame pacer and the dynamic scale
// bypassed), with crops of named subjects captured as they move. tools/kit/clip-metrics.py
// measures the result; tools/kit/parity/baseline.mjs runs it for a whole plan. For paused views,
// use tools/kit/parity/capture-set.mjs.
//
// Usage:
//   node tools/kit/parity/capture-clip.mjs <name> --url <preview> --clip <clip> [options]
//   node tools/kit/parity/capture-clip.mjs pan --url http://localhost:5191/ --clip pan-slow --aa taau --full 0,30
//
// Options:
//   --aa <mode>             msaa, off, taa, taau, fsr3, gt or gt-hi; default msaa
//   --view <v>              the base view; default the clip's. Also an ad-hoc view, "name:query[::js]"
//   --views-file <f.json>   the base view from a file of one ad-hoc view (tools/kit/parity/adhoc.mjs)
//   --frames <n>            default the clip's
//   --every <k>             capture every k-th frame (frame 0 always); default 1
//   --dense <a-b,c-d>       runs of frames captured every frame under --every
//   --crops <a,b>           the anchors cropped; default the clip's, else the view's
//   --full <0,30|all>       frames also captured whole; `all`, every captured frame
//   --anchors-from <f>      take the anchors' world points from an earlier clip.json
//   --q "<extra&query>"     added to the page's query
//   --setup <js>            page code (`s` the handle, `dt` the step) run before frame 0
//   --dt <s>                the clock's step; default 1 / 60
//   --gt-jitters <n>        with --aa gt: renders averaged per frame; default 32, 64 for one frame
//   --quality <tier>        default high
//   --size <W>x<H>[@<dpr>]  default 1728x1117@2
//   --out <dir>             default artifacts/clips/<name> (the configuration's `output.clips`)
//   --timeout <ms>          the page's waits; default 240000
//   --stable-*, --ready-frames <n>, --wait <ms>, --settle <ms>  readiness, as tools/kit/capture.mjs
//   --runs <file.csv>       append a line to a run log; default the PARITY_RUNS variable, else none
//   --task <name>           the run log's task
//   --verbose               print up to 60 console errors instead of 12
//   --test-fault            test the rest's second capture (below)
//
// The page loads `<url>?debug&paused&testMode=0&t=12&quality=<tier>&aa=<mode>[&<extra>]&<view's
// query>[&<clip's query>]`. `aa=off` adds `msaa=1`; a build that does not read `aa` ignores it.
// `testMode=0` leaves the test mode's switch out, so what it turns off moves in the clips
// (docs/reference/test-mode.md). In a persistent profile slot the page runs the view's eval (the
// crop anchors' setup included), waits for the scene as capture.mjs does
// (tools/kit/capture-browser.mjs: the image stopped changing, or under a temporal mode the rest
// converged, or --ready-frames' fixed count), runs the clip's eval and --setup, reads the crop
// anchors once at the base pose and steps the clip. A clip's lead frames (clips.mjs) are stepped
// once before frame 0 and never captured. After
// each captured frame each anchor's world point is projected with that frame's camera into a box
// of 192 CSS px (384 device px at a ratio of 2, clamped into the frame); an anchor behind the camera
// is skipped. Pass a test run's clip.json to a ground-truth run with --anchors-from: the anchors
// pick their points by device pixels, so a run at another ratio could pick others.
//
// --setup takes a change the clip should watch unfold, such as `s.setWind( { heading: 220 }, 20 )`.
// --dt steps the scene's own motion further per frame (not the camera path's): a motion-vector test
// whose sub-pixel motion a reprojection's resampling would hide makes it a few pixels a frame.
//
// The exposure's adaptation is frozen on every side while a clip runs (the `page.clip` global set:
// the pipeline snaps the exposure to its target every frame, packages/render/post/Exposure.ts), so
// a temporal clip compared with its rest or with MSAA does not differ by the adaptation's lag. The
// per-pixel dither of a level stays: it is fixed per pixel.
//
// --aa gt is the temporal modes' ground truth (packages/render/post/Antialiasing.ts, debug builds):
// the temporal graph at the canvas's resolution, each captured frame held still after its tick
// until the rest has averaged --gt-jitters jittered renders of it with a box filter, unsharpened,
// cropped at the test's own size. It renders every tier, so each tier has its own ground truth.
// --aa gt-hi is the older ground truth, kept for cross-checks: MSAA on high at twice a nominal ratio,
// box-downsampled 2x2 (288 px crops at the default size); its rules are beside GT_RATIO below. Both
// render every frame at their own fixed ratio: a clip's scale schedule applies to the test and
// MSAA sides only (clip.json: `scaleSchedule`). Judge vegetation look against MSAA, not a ground
// truth: their blades are finer than MSAA's.
//
// A clip of one frame (`--clip rest`) is the image at rest. Under taa and taau it is held still
// until the rest has converged on it. Under msaa, off and gt-hi it is stepped again with the same
// state and captured whole until two captures agree within the readiness tolerance
// (--stable-tolerance), at most 6 captures, else the clip fails and nothing is written: a single
// stepped frame can be a discrete alternative image that the readiness check before it cannot see
// (the measurements are beside RECHECK_SHOTS below). clip.json records the captures (`recheck`).
// `--test-fault` draws frame 0 half a stop brighter, so the first capture must be rejected and the
// crops taken from a later one. Motion clips and temporal rests are not captured twice: a moving
// frame cannot be rendered again without replaying its state, and one odd render is a small share
// of a temporal rest.
//
// Output: under --out, `<crop>/<i>.png` per crop and frame, `full/<i>.png`, and clip.json (the url,
// the clip, the boxes, camera and render size per frame, the readiness record `ready`, the page's
// diagnostics before and after, the pipelines created while stepping, the scene time, the browser
// build, wall time, console errors, `recheck`). The PNGs are the compositor's image encoded for
// speed: compare them decoded. Stdout: a summary line, the clip's notes, console errors.
//
// GPU lock and browser: one headless page in a profile slot holds a shared slot of the GPU lock for
// the whole clip; stepped clips are deterministic (docs/reference/gpu-lock.md).
//
// Cost: a page load and readiness, then a stepped frame and its crops per captured frame. The
// rest's second capture adds one stepped frame and a whole-frame grab, under a second.
//
// Exit codes: 0 captured; 1 a page error, a console error, the rest's captures disagreeing or another
// failure; 2 usage, an unknown option, clip or view, gt-hi on a tier other than high, or a bad
// --size, --gt-jitters or --dense.
import '../help.mjs';
import { mkdir, readFile, writeFile, appendFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { acquireGpuLock, releaseGpuLock, gpuUtilisation } from '../gpu-lock.mjs';
import { viewTable, clipTable } from './tables.mjs';
import { selectViews, viewQuery } from './adhoc.mjs';
import { launchCapture, waitForScene, waitForStable, stableOptions, decodePng, compareFrames, pngCapturer } from '../capture-browser.mjs';
import PROJECT from '../../project.mjs';
import { ROOT, outputDir } from '../root.mjs';

const REPO = ROOT;
const { VIEWS, viewEval } = viewTable;
const { CLIPS, clipEval } = clipTable;
const PAGE = PROJECT.page;
const args = process.argv.slice( 2 );
const FLAGS = new Set( [ 'verbose', 'stable-trace', 'test-fault' ] );
const options = {}, positional = [];
for ( let i = 0; i < args.length; i ++ ) {

	if ( args[ i ].startsWith( '--' ) ) {

		const k = args[ i ].slice( 2 );
		options[ k ] = FLAGS.has( k ) ? true : args[ ++ i ];

	} else positional.push( args[ i ] );

}
const KNOWN = [ 'url', 'aa', 'clip', 'view', 'views-file', 'frames', 'dt', 'setup', 'every', 'dense', 'crops', 'full', 'q', 'quality', 'size', 'out', 'timeout', 'runs', 'task', 'verbose', 'anchors-from', 'gt-jitters', 'ready-frames', 'wait', 'settle',
	'stable-min', 'stable-every', 'stable-repeats', 'stable-tolerance', 'stable-patience', 'stable-cap', 'stable-trace', 'test-fault' ];
const unknown = Object.keys( options ).filter( ( k ) => ! KNOWN.includes( k ) );
const clip = CLIPS[ options.clip ];
if ( positional.length !== 1 || unknown.length || ! options.url || ! clip ) {

	console.error( unknown.length ? `unknown option --${ unknown[ 0 ] }` : ! clip && options.clip ? `unknown clip ${ options.clip }` : 'usage: capture-clip.mjs <name> --url U --clip <clip> [options] (see the file header)' );
	process.exit( 2 );

}
const name = positional[ 0 ];
const aa = options.aa ?? 'msaa';
const gt = aa === 'gt-hi';
// the temporal modes' ground truth: its jitters per frame (a clip of one frame is a rest GT)
const gtRest = aa === 'gt';
const quality = options.quality ?? 'high';
if ( gt && quality !== 'high' ) { console.error( 'gt-hi exists for high only' ); process.exit( 2 ); }
// an ad-hoc base view (a query and an eval of the caller's, such as a footage shot's camera) is
// registered into VIEWS; its own query options win over the clip's common ones (adhoc.mjs)
let viewName = options.view ?? clip.view;
if ( options[ 'views-file' ] || viewName.includes( ':' ) ) {

	let names;
	try { names = selectViews( { view: viewName.includes( ':' ) ? [ viewName ] : [], 'views-file': options[ 'views-file' ] } ); } catch ( error ) { console.error( error.message ); process.exit( 2 ); }
	if ( names.length !== 1 ) { console.error( `--view / --views-file: one base view, not ${ names.length }` ); process.exit( 2 ); }
	viewName = names[ 0 ];

}
const view = VIEWS[ viewName ];
if ( ! view ) { console.error( `unknown view ${ viewName }` ); process.exit( 2 ); }
const size = ( options.size ?? '1728x1117@2' ).match( /^(\d+)x(\d+)(?:@([\d.]+))?$/ );
if ( ! size ) { console.error( `bad --size ${ options.size }` ); process.exit( 2 ); }
const W = Number( size[ 1 ] ), H = Number( size[ 2 ] ), DSF = Number( size[ 3 ] ?? 1 );
// gt-hi: MSAA on high at twice a nominal ratio for the --size given (nominal = min( ratio, 1.5 ), so
// 3 at the default @2 and 2 at @1: high's paused scale until it became the 4/3 baseline, kept so the
// stored ground truths stay comparable), box-downsampled 2x2. The page runs at that device pixel
// ratio with the same `dpr` (the canvas then holds the render; checked against the canvas's size).
// If the canvas does not reach it, the tool falls back to twice the CSS size at the nominal ratio
// (the same render, the CSS boxes doubled) and records `gtFallback`. Every crop and frame is
// box-downsampled 2x2, so a gt-hi crop is 192 x nominal px (288 at the default size, where the
// test's crop is 384): clip-metrics.py resamples the test to the GT's size. Known bias: pixel-width
// vegetation rules (blades kept 2 px wide, the rim's 0.75 px, the coverage gain that fades in under
// 16 px) are in render pixels, so at 3x the blades are drawn a third narrower in scene terms than at
// 1.5 and fewer blades are widened: GT-HI's meadow is finer and slightly darker than any real 1.5x
// render. aa=gt's blades are finer than MSAA's too (one sample: at least 1 canvas px, MSAA's 2
// render px). The gates read aa=gt; gt-hi entries are kept under `<sha>/gt-hi/high/`.
// gt-hi's nominal render ratio at this size (the display's, up to 1.5), and its ratio
const NOMINAL_HIGH = Math.min( DSF, 1.5 ), GT_RATIO = 2 * NOMINAL_HIGH;
const frames = Number( options.frames ?? clip.frames );
const gtJitters = gtRest ? Number( options[ 'gt-jitters' ] ?? ( frames <= 1 ? 64 : 32 ) ) : null;
if ( gtRest && ! ( Number.isInteger( gtJitters ) && gtJitters > 0 ) ) { console.error( `bad --gt-jitters ${ options[ 'gt-jitters' ] }` ); process.exit( 2 ); }
const every = Math.max( 1, Number( options.every ?? 1 ) );
const fullAll = options.full === 'all';
const full = new Set( fullAll ? [] : ( options.full ?? '' ).split( ',' ).filter( Boolean ).map( Number ) );
const wholeFrame = ( i ) => fullAll || full.has( i );
// runs of frames captured whole-clip dense under --every (the ground truth's gtDense, clips.mjs)
const dense = ( options.dense ?? '' ).split( ',' ).filter( Boolean ).map( ( r ) => r.split( '-' ).map( Number ) );
if ( dense.some( ( r ) => r.length !== 2 || r.some( ( x ) => ! Number.isInteger( x ) ) ) ) { console.error( `bad --dense ${ options.dense }` ); process.exit( 2 ); }
// the clip's own crops (clips.mjs `crops`) over the view's
const crops = options.crops ? options.crops.split( ',' ).filter( Boolean ) : clip.crops ?? view.crops ?? [];
const timeout = Number( options.timeout ?? 240000 );
const outDir = path.resolve( options.out ?? path.join( outputDir( REPO, 'clips' ), name ) );
// the crop box in CSS px (384 device px at a ratio of 2, as the parity crops)
const BOX = 192;
// A paused one-frame clip's captures before it fails for want of two that agree (see the header).
// The readiness check vouches for the frames it compared, not for the frame the clip then steps:
// the clip's evals come after it, so the stepped frame is a new render, and a single frame can come
// out as a discrete alternative image (measured: 0.3-1 M px of scattered vegetation pixels off by up
// to 190 levels, mean about 0, the same image each time; about 1.5 % of the frames stepped in the
// first seconds after the readiness, and 2 of 24 rest clips stored it while other browsers
// rendered; a stored rest reference once was such a frame). A fifth to two fifths of a settled
// page's stepped frames differ from its settled frame by a few to about 2000 pixels of speckle,
// which the tolerance (by default at most 5000 pixels differ at all; `exact` asks for equal pixels)
// accepts; the alternative image (0.87 M px on island-fair-noon) is far outside it. The crops are
// those of an agreeing capture (the first one's, or cropped again on the later frame when the
// first was the odd one). Each further capture steps one frame at frame 0's pose and scale,
// without the lead frames (the clock is paused, so the scene is frame 0's). Cost: one more stepped
// frame and one whole-frame grab when the first two agree, about 0.3 s alone and 0.9 s on a shared
// GPU (plus the whole-frame screenshot itself when --full does not ask for it). A temporal rest
// averages 32-64 renders, so one odd render is a 1/32 share there, inside the temporal
// comparisons' tolerance.
const RECHECK_SHOTS = 6;

// ------------------------------------------------------------------ PNG (8-bit, non-interlaced)

const CRC = new Int32Array( 256 ).map( ( _, n ) => { let c = n; for ( let k = 0; k < 8; k ++ ) c = c & 1 ? 0xedb88320 ^ ( c >>> 1 ) : c >>> 1; return c; } );
const crc32 = ( buf ) => { let c = - 1; for ( const b of buf ) c = CRC[ ( c ^ b ) & 255 ] ^ ( c >>> 8 ); return ( c ^ - 1 ) >>> 0; };
function chunk( type, data ) {

	const head = Buffer.alloc( 8 ); head.writeUInt32BE( data.length, 0 ); head.write( type, 4, 'latin1' );
	const crc = Buffer.alloc( 4 ); crc.writeUInt32BE( crc32( Buffer.concat( [ head.subarray( 4 ), data ] ) ), 0 );
	return Buffer.concat( [ head, data, crc ] );

}
function encodePng( { width, height, data } ) {

	const ihdr = Buffer.alloc( 13 ); ihdr.writeUInt32BE( width, 0 ); ihdr.writeUInt32BE( height, 4 ); ihdr[ 8 ] = 8; ihdr[ 9 ] = 2;
	const raw = Buffer.alloc( ( width * 3 + 1 ) * height );
	for ( let y = 0; y < height; y ++ ) Buffer.from( data.buffer, data.byteOffset + y * width * 3, width * 3 ).copy( raw, y * ( width * 3 + 1 ) + 1 );
	return Buffer.concat( [ Buffer.from( [ 137, 80, 78, 71, 13, 10, 26, 10 ] ), chunk( 'IHDR', ihdr ), chunk( 'IDAT', zlib.deflateSync( raw, { level: 6 } ) ), chunk( 'IEND', Buffer.alloc( 0 ) ) ] );

}

// 2x2 box filter (rounded to nearest), for gt-hi
function downsample2( { width, height, data } ) {

	const w = width >> 1, h = height >> 1, out = new Uint8Array( w * h * 3 );
	for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) for ( let k = 0; k < 3; k ++ ) {

		const i = ( ( 2 * y ) * width + 2 * x ) * 3 + k;
		out[ ( y * w + x ) * 3 + k ] = ( data[ i ] + data[ i + 3 ] + data[ i + width * 3 ] + data[ i + width * 3 + 3 ] + 2 ) >> 2;

	}
	return { width: w, height: h, data: out };

}

// ------------------------------------------------------------------ the page

let query = `debug&paused&testMode=0&t=12&quality=${ quality }&aa=${ gt ? 'msaa' : aa }`;
if ( aa === 'off' ) query += '&msaa=1';
if ( gt ) query += `&dpr=${ GT_RATIO }`;
if ( gtRest ) query += `&tier.taa.gtJitters=${ gtJitters }`;
if ( options.q ) query += `&${ options.q }`;
query = viewQuery( viewName, query );
if ( clip.q ) query += `&${ clip.q }`;
let base = options.url;
if ( ! base.endsWith( '/' ) && ! base.includes( '?' ) ) base += '/';
const stable = stableOptions( ( k ) => options[ k ] === true ? '' : options[ k ] );
const url = `${ base }?${ query }`;

await mkdir( outDir, { recursive: true } );
const busy = await gpuUtilisation();
const started = Date.now();
await acquireGpuLock( `capture-clip ${ name }`, { shared: true } );
// the browser in its profile slot (one page; gt-hi's fallback relaunches it at another ratio)
let session = null, ready = null;
const errors = [];
let exitCode = 0;
try {

	// gt-hi: its ratio at the CSS size, or (if the canvas cannot hold it) twice the CSS size at the
	// nominal ratio
	let gtFallback = false;
	let page = await open( gt ? GT_RATIO : DSF, 1 );
	if ( gt ) {

		const canvas = await page.evaluate( ( n ) => [ window[ n ].renderer.domElement.width, window[ n ].renderer.domElement.height ], PAGE.handle );
		if ( canvas[ 0 ] !== Math.round( W * GT_RATIO ) ) {

			console.log( `gt-hi: the canvas is ${ canvas.join( 'x' ) }, not ${ W * GT_RATIO } wide; falling back to twice the CSS size at a ratio of ${ NOMINAL_HIGH }` );
			await session.close();
			session = null;
			gtFallback = true;
			page = await open( NOMINAL_HIGH, 2 );

		}

	}
	const cssScale = gtFallback ? 2 : 1, ratio = gt ? ( gtFallback ? NOMINAL_HIGH : GT_RATIO ) : DSF;

	async function open( deviceScaleFactor, cssScale ) {

		session = await launchCapture( { viewport: { width: W * cssScale, height: H * cssScale }, deviceScaleFactor, url } );
		const p = session.page;
		p.on( 'console', ( m ) => { if ( options.verbose || m.type() === 'error' || m.type() === 'warning' ) errors.push( m.type() + ': ' + m.text() ); } );
		p.on( 'pageerror', ( e ) => errors.push( 'pageerror: ' + e.message ) );
		await p.goto( url );
		await p.waitForFunction( ( g ) => window[ g.handle ] && window[ g.diagnostics ], PAGE, { timeout } );
		if ( ! ( await p.evaluate( ( n ) => typeof window[ n ].loop.step === 'function', PAGE.handle ) ) ) throw new Error( `this build has no ${ PAGE.handle }.loop.step` );
		await p.evaluate( viewEval( viewName, crops ) );
		// ready as capture.mjs is: the image stopped changing (a temporal page: its rest converged)
		// after the floor's frames and the loading screen gone (or --ready-frames' fixed count)
		const fixed = options[ 'ready-frames' ] !== undefined ? Number( options[ 'ready-frames' ] ) : null;
		await waitForScene( p, { frames: fixed ?? stable.min, wait: Number( options.wait ?? 0 ), settle: Number( options.settle ?? 0 ), timeout } );
		ready = fixed === null ? await waitForStable( p, { ...stable, timeout } ) : { rule: 'frames', frames: fixed };
		if ( ready.status === 'capped' ) console.log( `unstable: still changing after ${ ready.framesWaited } frames (max ${ ready.maxSeen } levels), stepped at the cap` );
		return p;

	}

	await page.evaluate( clipEval( options.clip, { frames, ...( options.dt ? { dt: Number( options.dt ) } : {} ), ...( options.setup ? { setup: options.setup } : {} ) } ) );
	// the GT keeps its own fixed ratio (no scale schedule)
	await page.evaluate( ( { gt, g } ) => {

		const s = window[ g.handle ];
		if ( gt && window[ g.clip ].scale ) window[ g.clip ].scale = null;
		// (a temporal page's rest starts over: what the evals changed need not be in its key)
		s.taa?.epoch.bump( 'clip' );

	}, { gt: gt || gtRest, g: PAGE } );
	if ( gtRest && ! ( await page.evaluate( ( g ) => window[ g.diagnostics ]?.aa === 'gt' && !! window[ g.handle ].taa, PAGE ) ) ) throw new Error( 'this build has no aa=gt (or it fell back to another mode)' );
	const scaleSchedule = clip.scale ? ( gt || gtRest ? 'ignored: the ground truth renders at its fixed ratio' : 'applied' ) : 'none';
	const diagBefore = await page.evaluate( ( n ) => ( { ...window[ n ] } ), PAGE.diagnostics );
	// the anchors' world points at the base pose, then their boxes per frame in CSS px
	const anchors = options[ 'anchors-from' ] ? Object.fromEntries( Object.entries( JSON.parse( await readFile( options[ 'anchors-from' ], 'utf8' ) ).anchors ).filter( ( [ n ] ) => crops.includes( n ) ) ) : await page.evaluate( ( { names, global } ) => {

		const all = window[ global ] ?? {};
		return Object.fromEntries( names.map( ( n ) => [ n, all[ n ] ?? { missing: 'no such anchor in this view' } ] ) );

	}, { names: crops, global: PAGE.crops } );
	const tracked = Object.entries( anchors ).filter( ( [ , a ] ) => a.world ).map( ( [ n, a ] ) => ( { name: n, world: a.world } ) );
	for ( const t of tracked ) await mkdir( path.join( outDir, t.name ), { recursive: true } );
	if ( full.size || fullAll ) await mkdir( path.join( outDir, 'full' ), { recursive: true } );

	const boxes = [];
	// the compositor's image, PNG encoded for speed (tools/kit/capture-browser.mjs, "The screenshot");
	// the device metrics come back once the clip is stepped (nothing reads the ratio while stepping)
	// The screenshots drop the page's device metrics until the clip has been stepped: the page reads its
	// ratio only on a resize, and none fires, so the frames are the same.
	const capturer = await pngCapturer( page, { timeout } );
	const captureRect = async ( clipRect ) => {

		const png = await capturer.capture( clipRect );
		return gt ? encodePng( downsample2( decodePng( png ) ) ) : png;

	};
	const captureCrops = async ( frameBoxes ) => {

		const out = {};
		for ( const [ n, b ] of Object.entries( frameBoxes ) ) if ( ! b.missing ) out[ n ] = await captureRect( { x: b.x, y: b.y, width: b.w, height: b.h } );
		return out;

	};
	// a paused frame without a temporal rest is captured whole until two captures agree (see the
	// header): its captures stay in memory until then, each { raw (the whole frame as grabbed), crops, frame }
	const recheck = frames === 1 && clip.clock === 'paused' && ! gtRest && ! ( await page.evaluate( ( n ) => !! window[ n ].taa, PAGE.handle ) );
	const fault = options[ 'test-fault' ] === true;
	if ( fault && ! recheck ) throw new Error( '--test-fault tests the rest\'s second capture: a paused one-frame clip under msaa, off or gt-hi' );
	const restShots = [];
	await page.exposeFunction( '__clipFrame', async ( i, frameBoxes, time, state ) => {

		boxes.push( { i, time, ...state, boxes: { ...frameBoxes } } );
		const crops = await captureCrops( frameBoxes );
		if ( recheck ) {

			restShots.push( { raw: await capturer.capture( null ), crops, frame: state.frame } );
			return;

		}
		for ( const [ n, png ] of Object.entries( crops ) ) await writeFile( path.join( outDir, n, `${ i }.png` ), png );
		if ( wholeFrame( i ) ) await writeFile( path.join( outDir, 'full', `${ i }.png` ), await captureRect( null ) );

	} );
	const stepped = await page.evaluate( async ( { tracked, every, dense, box, cssScale, fullFrames, gtRest, fault, g } ) => {

		const s = window[ g.handle ], clip = window[ g.clip ];
		const V = s.camera.position.constructor;
		const project = ( w ) => {

			const q = new V( w[ 0 ], w[ 1 ], w[ 2 ] ).project( s.camera );
			if ( q.z < - 1 || q.z > 1 ) return null;
			const r = s.renderer.domElement.getBoundingClientRect();
			return [ r.left + ( q.x + 1 ) / 2 * r.width, r.top + ( 1 - q.y ) / 2 * r.height ];

		};
		const b = box * cssScale;
		const t0 = performance.now();
		const before = s.pipelines?.total ?? null;
		const captured = ( i ) => i % every === 0 || fullFrames.includes( i ) || dense.some( ( [ a, b ] ) => i >= a && i <= b );
		// the clip's lead frames step before its frame 0, at frame 0's pose and scale, uncaptured
		// (clips.mjs); the frames are numbered from the clip's frame 0
		const lead = clip.lead ?? 0;
		// --test-fault: frame 0 (and any lead frames before it, never captured) drawn half a stop
		// brighter, the offset restored after frame 0's captures
		const offset = s.pipeline.settings.exposureOffset;
		if ( fault ) s.pipeline.settings.exposureOffset = offset + 0.5;
		const result = await s.loop.step( {
			frames: lead + clip.frames, dt: clip.dt, clock: clip.clock, camera: ( k ) => clip.camera( Math.max( k - lead, 0 ) ),
			...( clip.scale ? { scale: [ ...Array( lead ).fill( clip.scale[ 0 ] ), ...clip.scale ] } : {} ),
			// the ground truth: each captured frame held still until the rest has converged on it; a
			// clip of one frame is a rest image, so the other temporal modes wait for their rest too
			...( gtRest || ( s.taa && clip.frames <= 1 ) ? { converge: ( k ) => k >= lead && captured( k - lead ) } : {} ),
			onFrame: async ( k ) => {

				const i = k - lead;
				if ( i < 0 || ! captured( i ) ) return;
				s.camera.updateMatrixWorld();
				const out = {};
				for ( const t of tracked ) {

					const p = project( t.world );
					if ( ! p ) { out[ t.name ] = { missing: 'behind the camera' }; continue; }
					// whole CSS px, so the device box starts on a whole pixel at any ratio
					const x = Math.round( Math.min( Math.max( p[ 0 ] - b / 2, 0 ), innerWidth - b ) ), y = Math.round( Math.min( Math.max( p[ 1 ] - b / 2, 0 ), innerHeight - b ) );
					out[ t.name ] = { x, y, w: b, h: b, anchor: [ + p[ 0 ].toFixed( 2 ), + p[ 1 ].toFixed( 2 ) ] };

				}
				// what the frame was drawn with, to tell a scene or camera difference from a GPU one
				const hash = ( list ) => { let h = 2166136261; for ( const x of list ) { const b = new Float64Array( [ x ] ), u = new Uint32Array( b.buffer ); h = Math.imul( h ^ u[ 0 ], 16777619 ) >>> 0; h = Math.imul( h ^ u[ 1 ], 16777619 ) >>> 0; } return h.toString( 16 ); };
				const state = { render: [ s.pipeline.width, s.pipeline.height ], view: hash( [ ...s.camera.matrixWorld.elements, ...s.camera.projectionMatrix.elements ] ), frame: s.renderer.info.frame,
					...( s.taa ? { taa: { converged: s.taa.converged, restFrames: s.taa.rest.frames, restTarget: s.taa.rest.target } } : {} ) };
				await window.__clipFrame( i, out, s.clockTime(), state );
				if ( fault ) s.pipeline.settings.exposureOffset = offset;

			},
		} );
		return { ...result, pipelines: before === null ? null : s.pipelines.total - before, wallMs: performance.now() - t0, notes: window[ g.clipNotes ] ?? [] };

	}, { tracked, every, dense, box: BOX, cssScale, fullFrames: [ ...full ], gtRest, fault, g: PAGE } );
	// the rest's second capture: the frame stepped again with the same state and captured whole
	// until a capture agrees with an earlier one; then that pair's crops (and whole frame) are written
	let recheckRecord = null;
	if ( recheck ) {

		const t = performance.now();
		const tolerance = stable.tolerance;
		const history = [];
		let accepted = null;
		// [ capture, earlier capture, largest channel difference, pixels differing at all ]
		const agree = ( j, shot ) => {

			const a = restShots[ j ].raw, b = shot.raw;
			let c = { max: 0, over: [ 0 ], overLevel: 0 };
			if ( ! a.equals( b ) ) c = compareFrames( decodePng( a ), decodePng( b ), tolerance ? tolerance.level : 0 );
			history.push( [ restShots.length, j, c.max, c.over[ 0 ] ] );
			return c.max === 0 || ( tolerance !== null && c.overLevel <= tolerance.pixels );

		};
		await page.exposeFunction( '__clipAgain', async ( frame ) => {

			const shot = { raw: await capturer.capture( null ), crops: null, frame };
			for ( let j = 0; j < restShots.length && ! accepted; j ++ ) {

				if ( ! agree( j, shot ) ) continue;
				// the earlier capture's crops, or this frame's (it agrees with a capture taken without crops)
				accepted = restShots[ j ].crops ? restShots[ j ] : shot;
				accepted.crops ??= await captureCrops( boxes[ 0 ].boxes );

			}
			restShots.push( shot );

		} );
		while ( ! accepted && restShots.length < RECHECK_SHOTS ) {

			await page.evaluate( async ( g ) => {

				const s = window[ g.handle ], clip = window[ g.clip ];
				await s.loop.step( { frames: 1, dt: clip.dt, clock: clip.clock, camera: clip.camera, ...( clip.scale ? { scale: clip.scale } : {} ),
					onFrame: () => window.__clipAgain( s.renderer.info.frame ) } );

			}, PAGE );

		}
		recheckRecord = { status: accepted ? 'agreed' : 'failed', shots: restShots.length, accepted: accepted ? restShots.indexOf( accepted ) : null, tolerance,
			history, ms: Math.round( performance.now() - t ) };
		if ( ! accepted ) throw new Error( `rest: no two of ${ restShots.length } captures agree within the tolerance (${ JSON.stringify( history ) }); nothing written` );
		boxes[ 0 ].frame = accepted.frame;
		for ( const [ n, png ] of Object.entries( accepted.crops ) ) await writeFile( path.join( outDir, n, '0.png' ), png );
		if ( wholeFrame( 0 ) ) await writeFile( path.join( outDir, 'full', '0.png' ), gt ? encodePng( downsample2( decodePng( accepted.raw ) ) ) : accepted.raw );
		if ( accepted !== restShots[ 0 ] || restShots.length > 2 ) console.log( `rest: captured ${ restShots.length } times, kept capture ${ recheckRecord.accepted } (${ history.map( ( h ) => `${ h[ 0 ] } vs ${ h[ 1 ] }: ${ h[ 3 ] } px, max ${ h[ 2 ] }` ).join( '; ' ) })` );

	}
	await capturer.close();
	const diagAfter = await page.evaluate( ( n ) => ( { ...window[ n ] } ), PAGE.diagnostics );
	await writeFile( path.join( outDir, 'clip.json' ), JSON.stringify( {
		name, url, clip: options.clip, clipDef: { ...clip, at: undefined, setup: undefined }, ...( options.setup ? { setup: options.setup } : {} ), view: viewName, aa, quality, ...( gtRest ? { gtJitters } : {} ),
		viewport: { width: W * cssScale, height: H * cssScale, deviceScaleFactor: ratio }, gtFallback, downsample: gt ? 2 : 1,
		cropPx: gt ? BOX * cssScale * ratio / 2 : BOX * ratio, scaleSchedule, frames, every, dense, full: fullAll ? 'all' : [ ...full ],
		browser: session.version, slot: session.slot, ready, ...( recheckRecord ? { recheck: recheckRecord } : {} ), anchors, boxes, stepped, diagBefore, diagAfter, errors,
		wallMinutes: ( Date.now() - started ) / 60000, gpuBusyBefore: busy,
	}, null, 2 ) );
	console.log( `${ outDir }: ${ boxes.length } frames captured of ${ frames }, ${ tracked.length } crops, pipelines while stepping ${ stepped.pipelines }, ${ ( stepped.wallMs / 1000 ).toFixed( 1 ) } s` );
	if ( stepped.notes.length ) console.log( stepped.notes.join( '\n' ) );
	if ( errors.length ) console.log( errors.slice( 0, options.verbose ? 60 : 12 ).join( '\n' ) );
	if ( errors.some( ( e ) => e.startsWith( 'pageerror:' ) || e.startsWith( 'error:' ) ) ) exitCode = 1;

} catch ( error ) {

	console.error( error );
	if ( errors.length ) console.error( errors.slice( 0, 20 ).join( '\n' ) );
	exitCode = 1;

} finally {

	await session?.close();
	releaseGpuLock();

}

const minutes = ( Date.now() - started ) / 60000;
const runs = options.runs ?? process.env.PARITY_RUNS;
if ( runs ) {

	const header = 'date,task,procedure,sha,wall_min,contended,captures,failed,out\n';
	const field = ( s ) => /[",\n]/.test( s ) ? `"${ s.replace( /"/g, '""' ) }"` : s;
	const line = [ new Date( started ).toISOString(), options.task ?? '', `capture-clip ${ options.clip } ${ aa }`, '', minutes.toFixed( 2 ),
		busy === null ? '?' : busy > 15 ? 'y' : 'n', String( frames ), String( exitCode ), outDir ].map( field ).join( ',' ) + '\n';
	await mkdir( path.dirname( path.resolve( runs ) ), { recursive: true } );
	await appendFile( runs, existsSync( runs ) ? line : header + line );

}
process.exitCode = exitCode;
