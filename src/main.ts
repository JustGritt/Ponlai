import { installSettings, parseSettings, settingProblems, RENDER_SETTINGS } from './render/setup.ts';

// A fresh breeze by default, so the sea has waves worth floating on; `?wind=` or `?weather=` override it.
const params = new URLSearchParams( location.search );
if ( ! params.has( 'wind' ) && ! params.has( 'weather' ) ) params.set( 'wind', 'fresh' );
const query = parseSettings( `?${ params }` );
for ( const problem of settingProblems( query, RENDER_SETTINGS ) ) console.warn( `[settings] ${ problem }` );
installSettings( query );

const { startProject } = await import( './scene.ts' );
await startProject();
