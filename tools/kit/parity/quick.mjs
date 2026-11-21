// Tries setting variants on a few views in one page each, measured against a commit's stored references.
//
// This is capture-set.mjs's quick mode (`--quick`), a module run only through it, for tuning: it
// answers how a runtime setting moves a few crops against MSAA and the ground truth, in seconds per
// variant instead of a fresh load each. It is not a gate: the gates still run full sets
// (tools/kit/parity/capture-set.mjs --check, tools/kit/parity/baseline.mjs).
//
//   node tools/kit/parity/capture-set.mjs <url> <outDir> --quick --views a,b [--crops x,y] [--full]
//       [--view "name:query[::js]"]... [--views-file f.json] [--sweep "key=a,b,c"]...
//       [--variant "[label::]js"]... [--quality high] [--aa mode] [--q "extra&query"]
//       [--size 1728x1117@2] [--store <root>/<sha>] [--ref-url R] [--timeout 240000] [--runs file.csv]
//       [--task name]
//
// The views (--views, or ad-hoc views as capture-set takes them) are required. --crops names the
// crops (default the view's); --variant is repeatable, and a label before `::` names it in the
// table (default v1, v2, ...); --runs (default the PARITY_RUNS variable) appends a line to a run log.
//
// One page per view, loaded as capture-set loads it (`?debug&paused&t=12&quality=..[&aa=..][&q]&<view>`,
// in a profile slot, the view's eval with the crop anchors, so the crops compare with the store's
// rest entries). On that page:
//   1. the base: ready as capture.mjs is (below), then captured;
//   2. each --variant in order: its js runs in the page (awaited; `s` is the page's handle), the
//      loop starts again, the temporal rest restarts (an epoch bump: the variant need not be in the
//      rest's key), ready, captured.
// Variants apply on top of each other to the one page: give each the absolute value of every
// setting it tunes (three variants of one setting: `s.taa.settings.sigmaRest = 0.3`, `= 0.4`,
// `= 0.5`; a variant of another setting sets the first one back). Only runtime settings (`s.taa.settings.*`,
// uniforms, `s.applyQuality( { ... } )`) can vary in one page: load-time ones (the tier, aa, the
// size, the debug switches, anything read once as materials and the frame graph are built) need a
// load, that is another quick run with `--q`.
//
// Ready (the same sequence as capture.mjs, see tools/kit/capture-browser.mjs): the floor's frames and
// the loading screen gone (waitForScene), then on a temporal page (taa, taau, fsr3, gt) the rest started
// over and converged under the new epoch, then waitForStable: an MSAA page by stability (the image
// stopped changing), a temporal one on its held rest; the loop is left stopped. A variant's image
// therefore equals a fresh load with the setting in its URL within the rules sets are judged by:
// MSAA byte-identical (`diff-set.py --aa-floor`), taau within `--temporal` (at most 3 levels on a
// few pixels, as its own A/A). Keep this sequence in step with capture.mjs.
//
// Screenshots: each crop through the screenshot's `clip` (only its pixels are encoded), a box of
// 192 CSS px (384 device px at 2x) about the anchor's projected world point, rounded to whole CSS
// px, as capture-clip.mjs cuts them; the anchors are the store's MSAA rest entry's (its clip.json)
// when there is one, so the boxes match the references', else the page's own. `--full` also
// captures the whole frame. No repeat.
//
// References (--store <root>/<sha>, a commit's directory of the store, tools/kit/reference.mjs):
// the MSAA rest entries `baseline/<tier>/rest/<view>` and the ground truth's `gt/<tier>/rest/<view>`,
// as baseline.mjs keeps them. Before the screenshots this runs `baseline.mjs --capture-only --only
// rest --gt` for the views: entries that are stored are reused (or waited for while another process
// captures them); missing ones are captured from --ref-url (a preview of that commit) or, without
// it, reported, and those views are captured without metrics. Without --store there are no metrics.
// Ad-hoc views have no stored rest entries, so they are captured without metrics.
//
// Output: <outDir> is emptied first. Under <outDir>/<variant>/ (`base` first):
//   <view>/<crop>/0.png, <view>/clip.json   a capture-clip folder of the one-frame clip `rest`
//   <view>.png, <view>.json                 with --full: a capture set's files (diff-set.py reads them)
//   <view>/metrics/                         clip-metrics.py's output against the references
// and <outDir>/quick.json (timings, readiness, references, metrics) and quick.md (the table, also
// on stdout). Per variant, view and crop the table gives the mean NVIDIA FLIP against the GT (MAE
// in levels without flip-evaluator), its change from the base, MSAA's, the sharpness S and the
// brightness B against the GT, and dMSAA (mean luma difference to MSAA's crop, levels: about 0 for
// the base of an MSAA run).
//
// GPU lock and browser: views run one after another, each in a headless page holding one shared
// slot of the GPU lock (docs/reference/gpu-lock.md); the metrics then run in up to 4 clip-metrics.py
// processes at once.
//
// Cost: a variant costs its readiness and its crops, a few seconds, against several times that for
// a fresh load: 3 views with a base and 2 variants take about 0.4 of the fresh loads' time under
// MSAA and about 0.6 under taau, references reused and metrics included.
//
// Exit codes (main's return, which capture-set exits with): 0 every view captured; 1 a view failed. An
// unknown option, a missing view, a bad --size or variant labels that repeat throw, which
// capture-set turns into exit 2.
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, rm, writeFile, appendFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { acquireGpuLock, releaseGpuLock, gpuUtilisation } from '../gpu-lock.mjs';
import { launchCapture, waitForScene, waitForStable, stableOptions, startLoop } from '../capture-browser.mjs';
import { viewTable, clipTable } from './tables.mjs';
import { selectViews, viewQuery } from './adhoc.mjs';
import { resolveRefDir } from '../reference.mjs';
import PROJECT from '../../project.mjs';
import { ROOT } from '../root.mjs';

const { VIEWS, viewEval } = viewTable;
const { CLIPS } = clipTable;
const PAGE = PROJECT.page;

const HERE = path.dirname( fileURLToPath( import.meta.url ) );
const TOOLS = path.join( ROOT, 'tools' );
// the crop box in CSS px (capture-clip.mjs's BOX)
const BOX = 192;
const FLAGS = new Set( [ 'quick', 'full' ] );
const KNOWN = [ 'quick', 'full', 'views', 'view', 'views-file', 'sweep', 'crops', 'variant', 'quality', 'aa', 'q', 'size', 'store', 'ref-url', 'timeout', 'runs', 'task' ];

function parse( argv ) {

	const o = { variant: [], view: [], sweep: [] }, positional = [];
	for ( let i = 0; i < argv.length; i ++ ) {

		const a = argv[ i ];
		if ( ! a.startsWith( '--' ) ) { positional.push( a ); continue; }
		const k = a.slice( 2 );
		if ( ! KNOWN.includes( k ) ) throw new Error( `--quick: unknown option --${ k } (see tools/kit/parity/quick.mjs)` );
		if ( FLAGS.has( k ) ) o[ k ] = true;
		else if ( Array.isArray( o[ k ] ) ) o[ k ].push( argv[ ++ i ] );
		else o[ k ] = argv[ ++ i ];

	}
	if ( positional.length !== 2 ) throw new Error( 'usage: capture-set.mjs <url> <outDir> --quick --views a,b [options] (see tools/kit/parity/quick.mjs)' );
	return { o, positional };

}

function run( cmd, args, log ) {

	return new Promise( ( resolve ) => {

		const child = spawn( cmd, args, { stdio: [ 'ignore', 'pipe', 'pipe' ], env: process.env } );
		let out = '';
		child.stdout.on( 'data', ( d ) => { out += d; } );
		child.stderr.on( 'data', ( d ) => { out += d; } );
		child.on( 'close', async ( code ) => { if ( log ) await writeFile( log, out ); resolve( { code: code ?? 1, out } ); } );

	} );

}

/** The store's rest references for the views: { view: { msaa, gt } | { missing } }. */
async function references( o, views, quality, outDir ) {

	const { sha, shaDir } = resolveRefDir( path.join( o.store, 'baseline' ) );
	const args = [ path.join( HERE, 'baseline.mjs' ), '--sha', sha, '--root', path.dirname( shaDir ), '--tiers', quality, '--only', 'rest',
		'--views', views.join( ',' ), '--capture-only', '--gt', '--task', o.task ?? 'quick',
		...( o[ 'ref-url' ] ? [ '--url', o[ 'ref-url' ] ] : [] ), ...( o.size ? [ '--size', o.size ] : [] ), ...( o.q ? [ '--q', o.q ] : [] ) ];
	const t0 = Date.now();
	const r = await run( process.execPath, args, path.join( outDir, 'references.log' ) );
	const out = {};
	for ( const v of views ) {

		const ok = ( side ) => new RegExp( `^ok\\s+${ side } ${ quality }/rest/${ v } ` , 'm' ).test( r.out );
		const msaa = path.join( shaDir, 'baseline', quality, 'rest', v ), gt = path.join( shaDir, 'gt', quality, 'rest', v );
		if ( ok( 'baseline' ) && ok( 'gt' ) ) out[ v ] = { msaa, gt };
		else {

			const why = r.out.split( '\n' ).filter( ( l ) => l.includes( `${ quality }/rest/${ v }` ) ).map( ( l ) => l.trim() ).join( '; ' );
			out[ v ] = { missing: why || r.out.trim().split( '\n' ).pop() };

		}

	}
	return { views: out, seconds: ( Date.now() - t0 ) / 1000, sha };

}

// ready as capture.mjs is (the file header): the loop is stopped on return
async function ready( page, stable, timeout ) {

	await waitForScene( page, { frames: stable.min, timeout } );
	await page.evaluate( ( n ) => window[ n ]?.taa?.epoch.bump( 'capture' ), PAGE.handle );
	await page.waitForFunction( ( p ) => {

		const t = window[ p.diagnostics ]?.taa;
		return ! t || t.restTarget === 0 || ( t.converged && t.restEpoch === window[ p.handle ].taa.epoch.value );

	}, PAGE, { timeout, polling: 100 } );
	return waitForStable( page, { ...stable, timeout } );

}

export async function main( argv ) {

	const { o, positional } = parse( argv );
	if ( ! o.views && ! o.view.length && ! o[ 'views-file' ] && ! o.sweep.length ) throw new Error( '--quick needs --views a,b (or ad-hoc views: --view, --views-file, --sweep)' );
	const views = selectViews( o );
	const size = ( o.size ?? '1728x1117@2' ).match( /^(\d+)x(\d+)(?:@([\d.]+))?$/ );
	if ( ! size ) throw new Error( `bad --size ${ o.size }` );
	const W = Number( size[ 1 ] ), H = Number( size[ 2 ] ), DSF = Number( size[ 3 ] ?? 1 );
	const quality = o.quality ?? 'high';
	const timeout = Number( o.timeout ?? 240000 );
	let base = positional[ 0 ];
	if ( ! base.endsWith( '/' ) && ! base.includes( '?' ) ) base += '/';
	const outDir = path.resolve( positional[ 1 ] );
	let common = `debug&paused&t=12&quality=${ quality }`;
	if ( o.aa ) common += `&aa=${ o.aa }`;
	if ( o.q ) common += `&${ o.q }`;
	const variants = [ { label: 'base', js: null }, ...o.variant.map( ( text, i ) => {

		const cut = text.indexOf( '::' );
		return cut > 0 ? { label: text.slice( 0, cut ).trim(), js: text.slice( cut + 2 ) } : { label: `v${ i + 1 }`, js: text };

	} ) ];
	const labels = new Set( variants.map( ( v ) => v.label ) );
	if ( labels.size !== variants.length || [ ...labels ].some( ( l ) => ! /^[\w.+-]+$/.test( l ) ) ) throw new Error( 'variant labels must be distinct words (letters, digits, . + - _), and not "base"' );
	const stable = stableOptions( () => undefined );
	await rm( outDir, { recursive: true, force: true } );
	await mkdir( outDir, { recursive: true } );
	const started = Date.now();
	const busy = await gpuUtilisation();

	// the references first: their anchors frame the crops
	// (ad-hoc views have no stored rest entries: baseline.mjs keeps views.mjs's)
	const stored = views.filter( ( v ) => ! VIEWS[ v ].adhoc );
	const refs = o.store && stored.length ? await references( o, stored, quality, outDir ) : null;
	if ( refs ) for ( const v of views ) if ( VIEWS[ v ].adhoc ) refs.views[ v ] = { missing: 'an ad-hoc view' };
	if ( refs ) for ( const [ v, r ] of Object.entries( refs.views ) ) if ( r.missing ) console.log( `${ v }: no reference, captured without metrics (${ r.missing })` );
	if ( ! refs ) console.log( o.store ? 'ad-hoc views only (no stored references): screenshots only, no metrics' : 'no --store: screenshots only, no metrics' );

	const record = { url: base, common, quality, aa: o.aa ?? null, q: o.q ?? null, size: o.size ?? '1728x1117@2', variants, store: o.store ?? null, references: refs, views: {}, wall: {} };
	let failed = 0;
	const shotsStarted = Date.now();
	for ( const view of views ) {

		const crops = o.crops ? o.crops.split( ',' ).filter( Boolean ) : VIEWS[ view ].crops ?? [];
		const url = `${ base }?${ viewQuery( view, common ) }`;
		const anchorsFile = refs?.views[ view ]?.msaa ? path.join( refs.views[ view ].msaa, 'clip.json' ) : null;
		const viewRecord = record.views[ view ] = { url, variants: {} };
		const t0 = Date.now();
		await acquireGpuLock( `quick ${ view }`, { shared: true } );
		viewRecord.lockMs = Date.now() - t0;
		let session = null;
		const errors = [];
		try {

			session = await launchCapture( { viewport: { width: W, height: H }, deviceScaleFactor: DSF, url } );
			const page = session.page;
			page.on( 'console', ( m ) => { if ( m.type() === 'error' || m.type() === 'warning' ) errors.push( m.type() + ': ' + m.text() ); } );
			page.on( 'pageerror', ( e ) => errors.push( 'pageerror: ' + e.message ) );
			await page.goto( url );
			await page.waitForFunction( ( p ) => window[ p.handle ] && window[ p.diagnostics ], PAGE, { timeout } );
			await page.evaluate( viewEval( view, crops ) );
			viewRecord.loadMs = Date.now() - t0 - viewRecord.lockMs;
			// the anchors' world points: the reference's, else the page's own
			const anchors = anchorsFile && existsSync( anchorsFile )
				? Object.fromEntries( Object.entries( JSON.parse( readFileSync( anchorsFile, 'utf8' ) ).anchors ).filter( ( [ n ] ) => crops.includes( n ) ) )
				: await page.evaluate( ( { names, global } ) => {

					const all = window[ global ] ?? {};
					return Object.fromEntries( names.map( ( n ) => [ n, all[ n ] ?? { missing: 'no such anchor in this view' } ] ) );

				}, { names: crops, global: PAGE.crops } );
			for ( const variant of variants ) {

				const v0 = Date.now();
				if ( variant.js !== null ) {

					await page.evaluate( `(async () => { const s = window.${ PAGE.handle }; ${ variant.js }
})()` );
					await startLoop( page );

				}
				const readiness = await ready( page, stable, timeout );
				const readyMs = Date.now() - v0;
				const dir = path.join( outDir, variant.label, view );
				await mkdir( dir, { recursive: true } );
				// the boxes, as capture-clip.mjs projects them (whole CSS px, clamped into the frame)
				const state = await page.evaluate( ( { anchors, box, page } ) => {

					const s = window[ page.handle ];
					s.camera.updateMatrixWorld();
					const V = s.camera.position.constructor, r = s.renderer.domElement.getBoundingClientRect(), boxes = {};
					for ( const [ n, a ] of Object.entries( anchors ) ) {

						if ( ! a.world ) { boxes[ n ] = { missing: a.missing ?? 'no world point' }; continue; }
						const q = new V( a.world[ 0 ], a.world[ 1 ], a.world[ 2 ] ).project( s.camera );
						if ( q.z < - 1 || q.z > 1 ) { boxes[ n ] = { missing: 'behind the camera' }; continue; }
						const p = [ r.left + ( q.x + 1 ) / 2 * r.width, r.top + ( 1 - q.y ) / 2 * r.height ];
						const x = Math.round( Math.min( Math.max( p[ 0 ] - box / 2, 0 ), innerWidth - box ) ), y = Math.round( Math.min( Math.max( p[ 1 ] - box / 2, 0 ), innerHeight - box ) );
						boxes[ n ] = { x, y, w: box, h: box, anchor: [ + p[ 0 ].toFixed( 2 ), + p[ 1 ].toFixed( 2 ) ] };

					}
					return { boxes, frame: s.renderer.info.frame, render: [ s.pipeline.width, s.pipeline.height ], time: s.clockTime(), taa: s.taa ? { converged: s.taa.converged, restEpoch: window[ page.diagnostics ].taa?.restEpoch } : null };

				}, { anchors, box: BOX, page: PAGE } );
				const s0 = Date.now();
				for ( const [ n, b ] of Object.entries( state.boxes ) ) {

					if ( b.missing ) continue;
					await mkdir( path.join( dir, n ), { recursive: true } );
					await page.screenshot( { path: path.join( dir, n, '0.png' ), clip: { x: b.x, y: b.y, width: b.w, height: b.h }, timeout } );

				}
				if ( o.full ) {

					await page.screenshot( { path: path.join( outDir, variant.label, `${ view }.png` ), timeout } );
					const cropsDevice = await page.evaluate( ( n ) => window[ n ] ?? null, PAGE.crops );
					await writeFile( path.join( outDir, variant.label, `${ view }.json` ), JSON.stringify( { url, variant, viewport: { width: W, height: H }, browser: session.version, slot: session.slot, ready: readiness, ...( cropsDevice ? { crops: cropsDevice } : {} ) }, null, 2 ) );

				}
				const shotMs = Date.now() - s0;
				await writeFile( path.join( dir, 'clip.json' ), JSON.stringify( {
					name: `quick-${ variant.label }-${ view }`, url, quick: true, variant, clip: 'rest', clipDef: { ...CLIPS.rest }, view, aa: o.aa ?? null, quality,
					viewport: { width: W, height: H, deviceScaleFactor: DSF }, cropPx: BOX * DSF, frames: 1, every: 1, dense: [], full: [],
					browser: session.version, slot: session.slot, ready: readiness, anchors, anchorsFrom: anchorsFile,
					boxes: [ { i: 0, time: state.time, render: state.render, frame: state.frame, ...( state.taa ? { taa: state.taa } : {} ), boxes: state.boxes } ], errors: errors.slice(),
				}, null, 2 ) );
				viewRecord.variants[ variant.label ] = { readyMs, shotMs, ready: { status: readiness.status, framesWaited: readiness.framesWaited } };
				console.log( `${ view } ${ variant.label }: ${ readiness.status } after ${ readiness.framesWaited } frames, ready ${ ( readyMs / 1000 ).toFixed( 1 ) } s, screenshots ${ ( shotMs / 1000 ).toFixed( 1 ) } s` );

			}

		} catch ( error ) {

			failed ++;
			viewRecord.error = String( error?.message ?? error );
			console.error( `${ view } FAILED: ${ viewRecord.error }` );

		} finally {

			await session?.close();
			releaseGpuLock();

		}
		viewRecord.errors = errors;
		viewRecord.wallMs = Date.now() - t0;
		if ( errors.some( ( e ) => e.startsWith( 'pageerror:' ) || e.startsWith( 'error:' ) ) ) console.log( `${ view }: console errors\n  ${ errors.slice( 0, 8 ).join( '\n  ' ) }` );

	}
	record.wall.shotsSeconds = ( Date.now() - shotsStarted ) / 1000;

	// metrics: every variant's crops against the references, in a few processes at once
	const m0 = Date.now();
	const python = [ 'run', '--with', 'numpy', '--with', 'pillow', '--with', 'scipy', '--with', 'flip-evaluator', 'python', '-I', path.join( TOOLS, 'kit', 'clip-metrics.py' ) ];
	const jobs = [];
	for ( const view of views ) for ( const variant of variants ) {

		const r = refs?.views[ view ];
		const test = path.join( outDir, variant.label, view );
		if ( r?.msaa && existsSync( path.join( test, 'clip.json' ) ) ) jobs.push( { view, label: variant.label, test, ...r } );

	}
	const results = {};
	const queue = jobs.slice();
	await Promise.all( Array.from( { length: Math.min( 4, jobs.length ) }, async () => {

		while ( queue.length ) {

			const j = queue.shift();
			// one worker each: the frames are few, the processes run side by side
			const r = await run( 'uv', [ ...python, '--test', j.test, '--msaa', j.msaa, '--gt', j.gt, '--out', path.join( j.test, 'metrics' ), '--workers', '1' ], path.join( j.test, 'metrics.log' ) );
			const file = path.join( j.test, 'metrics', 'metrics.json' );
			( results[ j.view ] ??= {} )[ j.label ] = existsSync( file ) && r.code === 0 ? JSON.parse( readFileSync( file, 'utf8' ) ) : { error: r.out.trim().split( '\n' ).pop() };

		}

	} ) );
	record.wall.metricsSeconds = ( Date.now() - m0 ) / 1000;
	record.wall.totalSeconds = ( Date.now() - started ) / 1000;

	// the table: per variant, view and crop
	const fmt = ( v, d = 4 ) => typeof v === 'number' ? v.toFixed( d ) : '–';
	const lines = [ `# quick: ${ views.join( ', ' ) } (${ quality }${ o.aa ? ', aa=' + o.aa : '' }${ o.q ? ', ' + o.q : '' })`, '' ];
	for ( const variant of variants ) lines.push( `- ${ variant.label }: ${ variant.js === null ? 'as loaded' : '`' + variant.js.trim().replace( /\s+/g, ' ' ) + '`' }` );
	let metricName = null;
	for ( const variant of variants ) {

		const rows = [];
		for ( const view of views ) {

			const m = results[ view ]?.[ variant.label ], b = results[ view ]?.base;
			if ( ! m ) { rows.push( `| ${ view } | ${ refs?.views[ view ]?.missing ? 'no reference' : record.views[ view ].error ? 'failed' : 'no metrics' } | | | | | | |` ); continue; }
			if ( m.error ) { rows.push( `| ${ view } | metrics failed: ${ m.error } | | | | | | |` ); continue; }
			metricName ??= m.metric;
			for ( const [ crop, e ] of Object.entries( m.crops ) ) {

				if ( e.missing ) { rows.push( `| ${ view }/${ crop } | ${ e.missing } | | | | | | |` ); continue; }
				const key = m.metric, t = e.test[ key ], base0 = b?.crops?.[ crop ]?.test?.[ key ];
				rows.push( `| ${ view }/${ crop } | ${ fmt( t ) } | ${ variant.label === 'base' ? '' : fmt( typeof base0 === 'number' ? t - base0 : null ) } | ${ fmt( e.msaa[ key ] ) } | ${ fmt( e.test.S, 3 ) } | ${ fmt( e.test.B, 3 ) } | ${ fmt( e.test.dMSAA, 3 ) } |` );

			}

		}
		const wall = views.map( ( v ) => record.views[ v ].variants[ variant.label ] ).filter( Boolean );
		lines.push( '', `## ${ variant.label }${ wall.length ? ` (ready ${ ( wall.reduce( ( a, w ) => a + w.readyMs, 0 ) / 1000 ).toFixed( 1 ) } s, screenshots ${ ( wall.reduce( ( a, w ) => a + w.shotMs, 0 ) / 1000 ).toFixed( 1 ) } s)` : '' }`, '',
			`| view/crop | ${ metricName ?? 'FLIP' } vs GT | change from base | MSAA's | S | B | dMSAA |`, '|---|---|---|---|---|---|---|', ...rows );

	}
	record.wall.lockSeconds = views.reduce( ( a, v ) => a + ( record.views[ v ].lockMs ?? 0 ), 0 ) / 1000;
	lines.push( '', `${ views.length } views x ${ variants.length } variants in ${ record.wall.shotsSeconds.toFixed( 1 ) } s, ${ record.wall.lockSeconds.toFixed( 1 ) } s of it waiting for the GPU lock (references ${ refs ? refs.seconds.toFixed( 1 ) + ' s' : 'none' }, metrics ${ record.wall.metricsSeconds.toFixed( 1 ) } s, total ${ record.wall.totalSeconds.toFixed( 1 ) } s)` );
	record.metrics = results;
	await writeFile( path.join( outDir, 'quick.json' ), JSON.stringify( record, null, 1 ) );
	await writeFile( path.join( outDir, 'quick.md' ), lines.join( '\n' ) + '\n' );
	console.log( lines.join( '\n' ) );

	const runs = o.runs ?? process.env.PARITY_RUNS;
	if ( runs ) {

		const header = 'date,task,procedure,sha,wall_min,contended,captures,failed,out\n';
		const line = [ new Date( started ).toISOString(), o.task ?? '', `capture-set --quick ${ variants.length - 1 } variants`, refs?.sha?.slice( 0, 7 ) ?? '',
			( record.wall.totalSeconds / 60 ).toFixed( 2 ), busy === null ? '?' : busy > 15 ? 'y' : 'n', String( views.length * variants.length ), String( failed ), outDir ].join( ',' ) + '\n';
		await mkdir( path.dirname( path.resolve( runs ) ), { recursive: true } );
		await appendFile( runs, existsSync( runs ) ? line : header + line );

	}
	return failed ? 1 : 0;

}
