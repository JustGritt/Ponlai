// Answers `--help` for the tool being run from that tool's own header comment.
//
// A tool's manual is its header comment, written once beside its code; this module prints it, so
// `--help` never drifts from the header. An entry tool imports it first, before any other import:
// ES modules run in the order they are imported, so the answer comes before anything the tool
// imports after it can take the GPU lock, re-execute the process, open a browser or start a server.
//   import './kit/help.mjs';      (the path from the tool's folder)
// A tool whose arguments are another command's (tools/kit/qos.mjs) imports it only when its first
// argument is `--help`.
//
// What it prints:
//   node <tool> --help            the header's first line, its Usage and Options sections
//   node <tool> --help --full     the whole header
//
// Only the arguments before the first `--` count: what follows belongs to a command the tool runs
// (`job.mjs start -- <command>`). Without `--help` among them, importing this module does nothing.
// With it, the module prints and exits 0; the tool's own code never runs.
//
// The header is the run of `//` lines at the top of the file, after a `/*! ... */` notice if one
// comes first. A section starts at a line that is not indented and opens with one to four words
// and a colon (`Usage:`, `Options:`, `Exit codes:`), and ends before the next blank line or the
// next section, so Usage and Options hold no blank line. The Python tools' helper,
// tools/kit/help.py, reads a module docstring by the same rules. tools/repo/verify-tool-manuals.mjs
// checks every entry tool against them.
import { readFileSync, writeSync } from 'node:fs';
import path from 'node:path';
import { ROOT } from './root.mjs';

const HEADING = /^[A-Z][A-Za-z0-9]*(?: [A-Za-z0-9]+){0,3}:(?: |$)/;

/** The header comment of a JavaScript file's text, as lines without their `//` marker. */
export function headerOf( text ) {

	const lines = text.replace( /^\/\*![\s\S]*?\*\/\s*/, '' ).split( '\n' );
	const header = [];
	for ( const line of lines ) {

		if ( ! line.startsWith( '//' ) ) break;
		header.push( line.replace( /^\/\/ ?/, '' ) );

	}
	return header;

}

/** The section that opens with `name:` in a header's lines: from its heading to the next blank line
 * or heading, or [] when the header has none. */
export function sectionOf( header, name ) {

	const start = header.findIndex( ( line ) => line.startsWith( `${ name }:` ) && HEADING.test( line ) );
	if ( start < 0 ) return [];
	let end = start + 1;
	while ( end < header.length && header[ end ].trim() !== '' && ! HEADING.test( header[ end ] ) ) end ++;
	return header.slice( start, end );

}

/** What `--help` prints for a header (`--help --full`: the whole header), for the tool at `file`. */
export function manualOf( header, file, full = false ) {

	if ( full ) return header.join( '\n' ) + '\n';
	const run = file.endsWith( '.py' ) ? 'uv run' : 'node';
	const parts = [ [ header[ 0 ] ?? '' ], sectionOf( header, 'Usage' ), sectionOf( header, 'Options' ),
		[ `The whole manual: ${ run } ${ path.relative( ROOT, file ) } --help --full` ] ];
	return parts.filter( ( p ) => p.length ).map( ( p ) => p.join( '\n' ) ).join( '\n\n' ) + '\n';

}

const end = process.argv.indexOf( '--', 2 );
const own = process.argv.slice( 2, end < 0 ? undefined : end );
if ( own.includes( '--help' ) && process.argv[ 1 ] ) {

	const file = path.resolve( process.argv[ 1 ] );
	writeSync( 1, manualOf( headerOf( readFileSync( file, 'utf8' ) ), file, own.includes( '--full' ) ) );
	process.exit( 0 );

}
