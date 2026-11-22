import { installSettings, parseSettings, settingProblems, RENDER_SETTINGS } from './render/setup.ts';
import { chooseGraphics } from './page.ts';

// The lighthouse. Its sea has a fresh breeze by default, so it has waves worth floating on; `?wind=`
// or `?weather=` override it.
const params = new URLSearchParams( location.search );

// A fresh visit picks its graphics on the start screen first; an address that already names them (a
// reload) and the tools' automated browser go straight in. The choice joins the address.
if ( ! params.has( 'quality' ) && ! navigator.webdriver ) {

  // the physics (most of the download) loads while the visitor chooses; floaters.ts initialises it
  import( '@dimforge/rapier3d-compat' ).catch( () => {} );
  const { tier, fps } = await chooseGraphics();
  params.set( 'quality', tier );
  if ( fps ) params.set( 'frameRate', String( fps ) );
  history.replaceState( null, '', `?${ params }` );

}
// the loading screen's lamp lights: the scene is loading
document.getElementById( 'loading' )!.classList.add( 'lit' );

if ( ! params.has( 'wind' ) && ! params.has( 'weather' ) ) params.set( 'wind', 'fresh' );
const query = parseSettings( `?${ params }` );
for ( const problem of settingProblems( query, RENDER_SETTINGS ) ) console.warn( `[settings] ${ problem }` );
installSettings( query );

const { startProject } = await import( './scene.ts' );
await startProject();
