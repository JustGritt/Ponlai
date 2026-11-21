import { installSettings, parseSettings, settingProblems, RENDER_SETTINGS } from './render/setup.ts';

const query = parseSettings( location.search );
for ( const problem of settingProblems( query, RENDER_SETTINGS ) ) console.warn( `[settings] ${ problem }` );
installSettings( query );

const { startProject } = await import( './scene.ts' );
await startProject();
