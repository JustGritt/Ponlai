// Where the repository and its tools are, found from this file's place (tools/kit/), so every tool
// reaches them the same way, whichever group's folder it is in; and where the tools write, from the
// project's configuration (tools/project.mjs `output`).
//   import { ROOT, TOOLS, outputDir } from './root.mjs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import PROJECT from '../project.mjs';

/** The tools' folder, `<repository>/tools`. */
export const TOOLS = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );

/** The repository's root: the checkout these tools are in. */
export const ROOT = path.resolve( TOOLS, '..' );

/** The tools' output folder under a checkout's top (`output.root`), or a kind's folder in it
 * (`output[ kind ]`: captures, clips, comparison, jobs, parity, profiles, storage). `top` may be
 * relative, as a tool that writes under the working directory gives it. */
export const outputDir = ( top, kind ) => kind === undefined ? path.join( top, PROJECT.output.root ) : path.join( top, PROJECT.output.root, PROJECT.output[ kind ] );
