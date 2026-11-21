// Allows, forbids or reports headed browser runs, which take over the user's screen.
//
// A headed run (tools/render/profile-frame.mjs `--display`, any script that opens a visible Chrome
// window) takes over the screen of whoever uses the machine, so it runs only while the user allows
// it. When a run may be headed at all, and how many scenes it covers, is the policy in
// docs/reference/gpu-lock.md; this tool is the switch. Run `status` before a headed
// run; record a run that is not allowed now as pending instead.
//
// Usage:
//   node tools/kit/display-gate.mjs allow [hours]
//   node tools/kit/display-gate.mjs deny
//   node tools/kit/display-gate.mjs [status]
//
// Options:
//   allow [hours]   allows headed runs for that many hours (default 8)
//   deny            ends the permission now
//   status          says whether headed runs are allowed now (the default command)
//
// The permission is a file under the user's home, `.ponlai/display-allowed` (the folder is the
// configuration's `machine.home`), holding the time it ends (`GATE_FILE`). Scripts read it through
// this module: `requireDisplayAllowed( label )` throws unless headed runs are allowed now, and
// `profile-frame --display` refuses without it. `announceHeadedRun( { label, scenes, minutes } )`
// posts a desktop notification titled with the configuration's `name` (osascript: macOS only;
// elsewhere it fails and is logged) and then waits 10 s before the window opens, without asking for
// a reply. A notification that fails is logged on stderr and never stops the run.
//
// Output: one line on stdout, `headed runs allowed until <local time>` or `headed runs not
// allowed`. `allow` writes the file; `deny` removes it.
//
// GPU lock and browser: neither. Cost: under a second.
//
// Exit codes: 0 done, or allowed for `status`; 1 not allowed for `status`, an unknown command or
// hours that are not a positive number.
import './help.mjs';
import { execFile } from 'node:child_process';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import PROJECT from '../project.mjs';

export const GATE_FILE = path.join( os.homedir(), PROJECT.machine.home, 'display-allowed' );

/** When the permission for headed runs ends, or null if they are not allowed now. */
export function displayAllowedUntil() {

	try {

		const until = new Date( readFileSync( GATE_FILE, 'utf8' ).trim() );
		return Number.isFinite( until.getTime() ) && until.getTime() > Date.now() ? until : null;

	} catch {

		return null;

	}

}

/** Throws unless the user currently allows headed runs on the screen. */
export function requireDisplayAllowed( label = 'a headed run' ) {

	if ( displayAllowedUntil() ) return;
	throw new Error( `${ label } would open a window on the user's screen, and headed runs are not allowed now (${ GATE_FILE }). ` +
		'Use a headless run instead, or record this run as pending: it can be run once headed runs are ' +
		'allowed (node tools/kit/display-gate.mjs allow <hours>).' );

}

// A headed run takes over the screen: post a notification, then wait 10 s before the window opens,
// so whoever uses the machine sees it coming. A failed notification never stops a run.
export async function announceHeadedRun( { label, scenes, minutes } ) {

	const quote = ( t ) => t.replace( /\\/g, '\\\\' ).replace( /"/g, '\\"' );
	const text = quote( `${ label }: a headed run takes the screen in 10 s, ${ scenes } scenes, about ${ minutes } min` );
	await new Promise( ( resolve ) => execFile( 'osascript', [ '-e', `display notification "${ text }" with title "${ quote( PROJECT.name ) }"` ], ( error ) => {

		if ( error ) console.error( `[display-gate] notification failed: ${ error.message.split( '\n' )[ 0 ] }` );
		resolve();

	} ) );
	await new Promise( ( resolve ) => setTimeout( resolve, 10000 ) );

}

if ( process.argv[ 1 ] && path.resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url ) ) {

	const [ command = 'status', value ] = process.argv.slice( 2 );
	if ( command === 'allow' ) {

		const hours = Number( value ?? 8 );
		if ( ! ( hours > 0 ) ) throw new Error( 'allow takes a positive number of hours' );
		mkdirSync( path.dirname( GATE_FILE ), { recursive: true } );
		const until = new Date( Date.now() + hours * 3600e3 );
		writeFileSync( GATE_FILE, until.toISOString() + '\n' );
		console.log( `headed runs allowed until ${ until.toLocaleString() }` );

	} else if ( command === 'deny' ) {

		rmSync( GATE_FILE, { force: true } );
		console.log( 'headed runs not allowed' );

	} else if ( command === 'status' ) {

		const until = displayAllowedUntil();
		console.log( until ? `headed runs allowed until ${ until.toLocaleString() }` : 'headed runs not allowed' );
		process.exitCode = until ? 0 : 1;

	} else {

		throw new Error( `unknown command ${ command }: allow [hours], deny or status` );

	}

}
