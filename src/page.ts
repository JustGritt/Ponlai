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
  // back to the start screen: the address without the graphics it chose
  const params = new URLSearchParams( location.search );
  params.delete( 'quality' );
  params.delete( 'frameRate' );
  const a = document.createElement( 'a' );
  a.href = params.size ? `?${ params }` : location.pathname;
  a.textContent = 'Graphics';
  links.append( a );

}

/** The graphics a visitor picks on the start screen: the quality tier, and the frame-rate cap (0:
 * the tier's own). */
export interface Graphics { tier: string; fps: number }

const GRAPHICS_KEY = 'ponlai.graphics';

/** Shows the start screen for `scene`, set to the last visit's choice or else this device's, with a
 * warning where WebGPU is missing; resolves with the choice once the visitor presses Start. */
export async function chooseGraphics( scene: SceneName ): Promise<Graphics> {

  const screen = document.getElementById( 'setup' )!;
  const form = screen.querySelector( 'form' )!;
  const note = document.getElementById( 'setup-note' )!;
  const start = form.querySelector( 'button' )!;
  const tier = form.elements.namedItem( 'tier' ) as RadioNodeList;
  const fps = form.elements.namedItem( 'fps' ) as RadioNodeList;
  document.getElementById( 'setup-title' )!.textContent = SCENES[ scene ];

  // (no WebGPU types in this TypeScript's DOM library)
  const gpu = ( navigator as { gpu?: { requestAdapter(): Promise<{ info?: { isFallbackAdapter?: boolean } } | null> } } ).gpu;
  const adapter = await gpu?.requestAdapter().catch( () => null ) ?? null;
  const phone = matchMedia( '(pointer: coarse)' ).matches;
  // Chrome only; 4 GB or less marks an older or budget phone
  const memory = ( navigator as { deviceMemory?: number } ).deviceMemory ?? 8;
  let pick: Graphics = phone ? { tier: 'low', fps: 30 } : { tier: 'high', fps: 0 };
  try {

    pick = { ...pick, ...JSON.parse( localStorage.getItem( GRAPHICS_KEY ) ?? '{}' ) };

  } catch {

    // a private window may refuse, or the stored choice be unreadable: this device's then

  }
  // a value no button has leaves its group unchecked, and `required` holds Start until one is picked
  tier.value = pick.tier;
  fps.value = String( pick.fps );

  if ( ! adapter ) {

    note.className = 'warn';
    note.textContent = 'This browser can\'t run WebGPU, so the scene won\'t load here. Try a recent Chrome or Edge, or Safari on iOS 26 or later.';
    start.disabled = true;

  } else if ( adapter.info?.isFallbackAdapter ) {

    note.className = 'warn';
    note.textContent = 'WebGPU is running in software on this device, so the scene will be very slow even on Low.';

  } else if ( phone ) {

    note.textContent = memory <= 4
      ? 'This looks like an older phone: Low at 30 fps is the best bet, and it may still stutter.'
      : 'On a phone, Low at 30 fps runs best and keeps it cool.';

  } else note.textContent = 'High needs a desktop graphics card; on a laptop, try Medium or Low.';

  screen.hidden = false;
  await new Promise( ( resolve ) => form.addEventListener( 'submit', ( e ) => {

    e.preventDefault();
    resolve( null );

  }, { once: true } ) );
  screen.hidden = true;
  const choice = { tier: tier.value, fps: Number( fps.value ) };
  try {

    localStorage.setItem( GRAPHICS_KEY, JSON.stringify( choice ) );

  } catch {

    // kept for this visit only

  }
  return choice;

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
