// The machine's load over a profile-frame run (`<out>/load.jsonl`): CPU per process group and GPU
// time per client, about once a second, so each measured slice can be judged by what ran beside it.
// A run's opening idle check sees the machine for a few seconds before it starts; this sees it for
// the whole run (an app that starts rendering mid-run, a build that takes the CPU between scenes).
// macOS only (top and the Apple GPU driver's per-client times); elsewhere startLoadTimeline returns
// null.
//
// Sources: one `top -l 0 -s 1 -stats pid,ppid,command,cpu` child streams for the whole run (a top
// per sample would cost a process launch a second, and top's first sample of a launch has no CPU
// figures), and gpuClients( 1 ) (gpu-lock-impl.mjs) runs in a loop, each call covering the next
// second. Each GPU sample writes one line, with the latest CPU sample beside it:
//   { t, cpu: { group: % }, gpu: { group: % } }
// t is the middle of the GPU sample's second in ms since the epoch: the clock of a page's frame
// times plus its `performance.timeOrigin`, so a slice's window (page time + origin) selects its
// samples. CPU % is top's (100 = one core); GPU % is the share of the GPU's time.
//
// Groups: `run` (the run's Chrome: its browser process and descendants, found through top's ppid
// column), `windowServer`, `app` (the desktop app: commands starting with `Claude`), `otherChrome`
// (any other Chrome or Chromium), `kernel` (kernel_task; CPU only, its GPU time counts as rest)
// and `rest`. The load outside the run is every group but `run` and `windowServer`: WindowServer
// composites the screen, the run's own window in full-screen `--display` and otherwise a desktop
// every run shares (counted, its share alone marked nearly every headless slice contended).
//
// The pure parts (grouping, top's parser, the slice marking) are exported for CPU-only tests.
import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { gpuClients } from './gpu-lock.mjs';

export const GROUPS = [ 'run', 'windowServer', 'app', 'otherChrome', 'kernel', 'rest' ];

export function groupOf( name, pid, run ) {

	if ( run.has( pid ) ) return 'run';
	if ( /^WindowServer\b/.test( name ) ) return 'windowServer';
	if ( /^Claude\b/.test( name ) ) return 'app';
	if ( /Chrom(e|ium)/.test( name ) ) return 'otherChrome';
	if ( /^kernel_task\b/.test( name ) ) return 'kernel';
	return 'rest';

}

// a process and its descendants, from a sample's [ { pid, ppid } ]
export function processTree( procs, root ) {

	const tree = new Set();
	if ( ! root ) return tree;
	const children = new Map();
	for ( const p of procs ) {

		if ( ! children.has( p.ppid ) ) children.set( p.ppid, [] );
		children.get( p.ppid ).push( p.pid );

	}
	tree.add( root );
	for ( const pid of tree ) for ( const child of children.get( pid ) ?? [] ) if ( child !== pid ) tree.add( child );
	return tree;

}

// top's logging output, fed in chunks: returns the samples completed by the chunk. A sample is
// complete once it listed as many processes as its "Processes: n total" line says (top writes a
// sample at once, then waits), or when the next one begins.
export function topParser() {

	let buffer = '', current = null;
	return ( chunk ) => {

		buffer += chunk;
		const lines = buffer.split( '\n' );
		buffer = lines.pop();
		const done = [];
		for ( const line of lines ) {

			const head = line.match( /^Processes:\s+(\d+) total/ );
			if ( head ) {

				if ( current ) done.push( current );
				current = { total: Number( head[ 1 ] ), procs: [] };
				continue;

			}
			const row = current && line.match( /^\s*(\d+)\s+(\d+)\s+(.+?)\s+([\d.]+)\s*$/ );
			if ( ! row ) continue;
			current.procs.push( { pid: Number( row[ 1 ] ), ppid: Number( row[ 2 ] ), name: row[ 3 ], cpu: Number( row[ 4 ] ) } );
			if ( current.procs.length >= current.total ) {

				done.push( current );
				current = null;

			}

		}
		return done;

	};

}

const r1 = ( x ) => Math.round( x * 10 ) / 10;
const sumGroups = ( list ) => {

	const out = Object.fromEntries( GROUPS.map( ( g ) => [ g, 0 ] ) );
	for ( const [ g, v ] of list ) out[ g ] += v;
	for ( const g of GROUPS ) out[ g ] = r1( out[ g ] );
	return out;

};

// Starts sampling; `runPid()` gives the run's Chrome (null before it opens; it changes when the
// browser is reopened). Returns { samples, until( t ), stop() }, or null off macOS.
export function startLoadTimeline( { file, runPid, log = console.error } ) {

	if ( process.platform !== 'darwin' ) {

		log( '[profile] the load timeline samples macOS only (top, the Apple GPU\'s client times): skipped' );
		return null;

	}
	const out = createWriteStream( file );
	const samples = [];
	let cpu = null, procs = [], stopped = false, first = true;
	const top = spawn( 'top', [ '-l', '0', '-s', '1', '-stats', 'pid,ppid,command,cpu' ], { stdio: [ 'ignore', 'pipe', 'ignore' ] } );
	const kill = () => top.kill();
	// a run that exits without stopping (its watchdog) must not leave top streaming
	process.on( 'exit', kill );
	const parse = topParser();
	top.stdout.setEncoding( 'utf8' );
	top.stdout.on( 'data', ( chunk ) => {

		for ( const sample of parse( chunk ) ) {

			procs = sample.procs;
			// the first sample of a launch has no CPU figures yet
			if ( first ) { first = false; continue; }
			const run = processTree( procs, runPid() );
			cpu = sumGroups( procs.map( ( p ) => [ groupOf( p.name, p.pid, run ), p.cpu ] ) );

		}

	} );
	top.on( 'error', ( error ) => log( `[profile] the load timeline's top failed: ${ error.message }` ) );
	const waiters = [];
	const gpuLoop = ( async () => {

		while ( ! stopped ) {

			const t0 = Date.now();
			const clients = await gpuClients( 1, Infinity );
			const t1 = Date.now();
			const run = processTree( procs, runPid() );
			const gpu = clients ? sumGroups( clients.map( ( c ) => {

				const m = c.name.match( /^(.*) \(pid (\d+)\)$/ );
				const g = m ? groupOf( m[ 1 ], Number( m[ 2 ] ), run ) : 'rest';
				return [ g === 'kernel' ? 'rest' : g, c.percent ];

			} ) ) : null;
			// gpuClients reads the registry, waits a second and reads it again: the second it
			// covers sits in the middle of the call
			const sample = { t: Math.round( ( t0 + t1 ) / 2 ), cpu, gpu };
			samples.push( sample );
			out.write( JSON.stringify( sample ) + '\n' );
			for ( const w of waiters.splice( 0 ) ) w();

		}

	} )();
	return {
		file, samples,
		// resolves once a sample covers time t (ms since the epoch), or after `wait` ms
		async until( t, wait = 3000 ) {

			const end = Date.now() + wait;
			while ( ! stopped && Date.now() < end && ! samples.some( ( s ) => s.t + 500 >= t ) ) {

				await new Promise( ( resolve ) => {

					waiters.push( resolve );
					setTimeout( resolve, Math.max( 0, end - Date.now() ) );

				} );

			}

		},
		async stop() {

			if ( stopped ) return;
			stopped = true;
			kill();
			process.off( 'exit', kill );
			await gpuLoop;
			await new Promise( ( resolve ) => out.end( resolve ) );

		},
	};

}

// the load outside the run in one sample's groups (null without that source)
export function outside( groups ) {

	if ( ! groups ) return null;
	return Object.entries( groups ).reduce( ( a, [ g, v ] ) => g === 'run' || g === 'windowServer' ? a : a + v, 0 );

}

const mean = ( v ) => v.length ? r1( v.reduce( ( a, b ) => a + b, 0 ) / v.length ) : null;

// A window's load: the mean outside the run over the samples whose second overlaps [ t0, t1 ]
// (ms since the epoch). Contended when the outside GPU mean exceeds `threshold`.
export function windowLoad( samples, t0, t1, { threshold } ) {

	const mine = samples.filter( ( s ) => s.t + 500 > t0 && s.t - 500 < t1 );
	const cpu = mean( mine.map( ( s ) => outside( s.cpu ) ).filter( ( x ) => x !== null ) );
	const gpu = mean( mine.map( ( s ) => outside( s.gpu ) ).filter( ( x ) => x !== null ) );
	return { cpu, gpu, samples: mine.length, contended: gpu !== null && gpu > threshold };

}

// A scene's slices (each with `at: [ t0, t1 ]`): each gets its `load`; the scene is contended when
// more than a quarter of its measured slices are. Returns the scene's { slices, contended (count),
// cpu, gpu (means over its slices), flag }.
export function markSlices( samples, slices, options ) {

	const loads = slices.filter( ( s ) => s.at ).map( ( s ) => s.load = windowLoad( samples, s.at[ 0 ], s.at[ 1 ], options ) );
	const contended = loads.filter( ( l ) => l.contended ).length;
	return {
		slices: loads.length, contended,
		cpu: mean( loads.map( ( l ) => l.cpu ).filter( ( x ) => x !== null ) ),
		gpu: mean( loads.map( ( l ) => l.gpu ).filter( ( x ) => x !== null ) ),
		flag: contended > loads.length / 4,
	};

}
