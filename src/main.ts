import { installSettings, parseSettings, settingProblems, RENDER_SETTINGS } from './render/setup.ts';
import { sceneFromAddress } from './page.ts';

// Two scenes, by `?scene=`: the lighthouse (the default) and the waterfall. The lighthouse's sea has a
// fresh breeze by default, so it has waves worth floating on; the waterfall's pool a light air.
// `?wind=` or `?weather=` override either.
const params = new URLSearchParams( location.search );
const scene = sceneFromAddress( params );
if ( ! params.has( 'wind' ) && ! params.has( 'weather' ) ) params.set( 'wind', scene === 'waterfall' ? 'light' : 'fresh' );
params.delete( 'scene' );
const query = parseSettings( `?${ params }` );
for ( const problem of settingProblems( query, RENDER_SETTINGS ) ) console.warn( `[settings] ${ problem }` );
installSettings( query );

if ( scene === 'waterfall' ) {

  const { startWaterfall } = await import( './falls/scene.ts' );
  await startWaterfall();

} else {

  const { startProject } = await import( './scene.ts' );
  await startProject();

}
