// The page round the scenes: the HUD's text, its time-of-day and sound buttons, the link to the
// other scene, and the sound's mute kept between visits.
import type { DayPreset, RenderParts } from './render/index.ts';

/** The scenes the page can show, by their `?scene=` name. */
export const SCENES = { lighthouse: 'Lighthouse', waterfall: 'Waterfall' } as const;
export type SceneName = keyof typeof SCENES;

/** The scene the address names (the lighthouse by default). */
export function sceneFromAddress( params: URLSearchParams ): SceneName {

  const name = params.get( 'scene' );
  return name && name in SCENES ? name as SceneName : 'lighthouse';

}

/** Sets the page's title, the HUD's help line, and the link to the other scenes (the address keeps
 * its other options). */
export function setUpPage( scene: SceneName, help: string ): void {

  document.title = SCENES[ scene ];
  const helpLine = document.getElementById( 'help' );
  if ( helpLine ) helpLine.textContent = help;
  const links = document.getElementById( 'scenes' );
  if ( ! links ) return;
  links.replaceChildren();
  for ( const [ name, title ] of Object.entries( SCENES ) as [ SceneName, string ][] ) {

    if ( name === scene ) continue;
    const params = new URLSearchParams( location.search );
    if ( name === 'lighthouse' ) params.delete( 'scene' );
    else params.set( 'scene', name );
    const a = document.createElement( 'a' );
    const query = params.toString();
    a.href = query ? `?${ query }` : location.pathname;
    a.textContent = `${ title } →`;
    links.append( a );

  }

}

/** Whether the sound was turned off on an earlier visit. */
export function readMuted( key: string ): boolean {

  try {

    return localStorage.getItem( key ) === '1';

  } catch {

    return false;

  }

}

/** The sound's button: on or off, remembered under `key`. */
export function soundButton( sound: { muted: boolean }, key: string ): void {

  const button = document.getElementById( 'sound' );
  if ( ! button ) return;
  const show = () => {

    button.textContent = sound.muted ? 'Sound: off' : 'Sound: on';
    button.setAttribute( 'aria-pressed', String( ! sound.muted ) );

  };
  show();
  button.addEventListener( 'click', () => {

    sound.muted = ! sound.muted;
    show();
    try {

      localStorage.setItem( key, sound.muted ? '1' : '0' );

    } catch {

      // a private window may refuse; the choice then lasts this visit

    }

  } );

}

/** The time-of-day buttons. */
export function presetButtons( render: RenderParts ): void {

  for ( const button of document.querySelectorAll<HTMLButtonElement>( '[data-preset]' ) ) {

    button.addEventListener( 'click', () => void render.setPreset( button.dataset.preset as DayPreset ) );

  }

}

let shownStatus = '';
/** The HUD's status line (the count beside the buttons). */
export function showStatus( text: string ): void {

  if ( text === shownStatus ) return;
  shownStatus = text;
  const el = document.getElementById( 'count' );
  if ( el ) el.textContent = text;

}
