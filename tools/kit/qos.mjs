// Runs a command, or the importing process, with macOS's application scheduling policy.
//
// Processes an agent starts (its shell, node, Python, tsc, the tools) are not applications to macOS:
// their threads run at the legacy QoS whatever they ask for (a thread that requests user-initiated
// is still accounted as legacy), and on a busy machine the scheduler keeps that work on the
// efficiency cores while the performance cores stand idle. Measured while several agents captured
// and built at once (most of the CPU idle, the efficiency cores full), the same work took 3-14x as
// long as in a process started with the application policy (`taskpolicy -a`): a single-thread loop
// 3-14x, the capture tool's PNG decode for readiness 5-10x and its frame compare 4x, tsc 3-14x. On a
// quiet machine both run alike. The application policy is what the user's own apps get: it puts the
// tools on par with them, not above them. Elsewhere than macOS nothing changes.
//
// Usage:
//   node tools/kit/qos.mjs <command> [args ...]
//   node tools/kit/qos.mjs python3 my-script.py
//
// Options: none of its own; every argument belongs to the command it runs. Only a first argument
// of `--help` is read here.
//
// The policy is a property of a process, given when it starts, and its children do not inherit it,
// so each process gets it on its own:
// - Importing this module re-executes the current node process under `taskpolicy -a` in place
//   (process.execve: same pid, same stdio and arguments) unless it already runs so.
//   tools/kit/gpu-lock.mjs and tools/kit/capture-browser.mjs import it, so every tool that renders
//   or reads captures raises itself before it does anything; tools/kit/serve.mjs, tools/check.mjs
//   and other CPU-heavy tools import it themselves. A new CPU-heavy node tool imports it first.
// - `appPolicy( cmd, args )` gives the command line that starts a child with the policy, for a
//   child that cannot re-execute itself (one with an IPC channel, which an exec closes:
//   capture-set's capture.mjs workers) and for the commands `gpu-lock.mjs run` and `job.mjs`
//   start.
// - The Python tools that do heavy work (tools/kit/clip-metrics.py and its worker processes,
//   tools/kit/diff-set.py, tools/kit/img.py, tools/island/day-sweep-plot.py) re-execute themselves
//   the same way with os.execv; a new one copies those few lines of diff-set.py.
// - The CLI runs any command with the policy: `npm run build`, `typecheck` and `lint` start tsc,
//   vite and eslint through it, and a one-off script runs through it too.
// Chrome's helper processes (GPU, renderers, the world's workers) set their own QoS and are not held
// back, so the page's own work is unaffected; the node side of a capture (decode, compare, PNG
// writes, while it holds a GPU slot) and every CPU tool are.
//
// <prefix>QOS (`<prefix>` is the configuration's `machine.envPrefix`) holds the pid of the process that runs with the policy, so a process knows it already
// does and its children (another pid) still raise themselves. <prefix>QOS=off keeps the policy a
// process was given (for an A/B of this module).
//
// Output: the command's own; nothing of its own.
//
// GPU lock and browser: neither. Cost: one exec, a few milliseconds.
//
// Exit codes: the command's (the process becomes it); 2 no command given.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import PROJECT from '../project.mjs';

const QOS = `${ PROJECT.machine.envPrefix }QOS`;
const TASKPOLICY = '/usr/sbin/taskpolicy';
const enabled = process.platform === 'darwin' && process.env[ QOS ] !== 'off';

/** The command line that starts `cmd args` with the application policy: [ cmd, args ] unchanged
 * elsewhere. The child is told through <prefix>QOS (its own pid, set by the shell that becomes it)
 * that it already runs so. */
export function appPolicy( cmd, args = [] ) {

	if ( ! enabled ) return [ cmd, args ];
	return [ TASKPOLICY, [ '-a', '/bin/sh', '-c', `${ QOS }=$$ exec "$@"`, 'sh', cmd, ...args ] ];

}

const cli = process.argv[ 1 ] && path.resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url );

if ( cli ) {

	const [ cmd, ...args ] = process.argv.slice( 2 );
	// the arguments are the command's: only a first `--help` is this tool's (tools/kit/help.mjs)
	if ( cmd === '--help' ) await import( './help.mjs' );
	if ( ! cmd ) {

		console.error( 'usage: node tools/kit/qos.mjs <command> [args ...]' );
		process.exit( 2 );

	}
	const [ file, argv ] = appPolicy( cmd, args );
	// through the shell, which looks the command up in PATH (npm scripts put node_modules/.bin there)
	if ( file === TASKPOLICY ) process.execve( file, [ file, ...argv ] );
	else process.execve( '/bin/sh', [ 'sh', '-c', 'exec "$@"', 'sh', cmd, ...args ] );

} else if ( enabled && process.env[ QOS ] !== String( process.pid ) && ! process.channel ) {

	// (a child with an IPC channel was started raised by its parent, through appPolicy)
	process.execve( TASKPOLICY, [ TASKPOLICY, '-a', process.execPath, ...process.execArgv, ...process.argv.slice( 1 ) ],
		{ ...process.env, [ QOS ]: String( process.pid ) } );

}
