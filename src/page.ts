// The page round the scene: the start screen, the HUD's text, its time-of-day and sound controls, the
// link back to the start screen, and the sound's mute and volume kept between visits.
import type { DayPreset, RenderParts } from './render/index.ts';

/** Sets the HUD's help line and its link back to the start screen (the address keeps its other
 * options). */
export function setUpPage( help: string ): void {

  const helpLine = document.getElementById( 'help' );
  if ( helpLine ) helpLine.textContent = help;
  const links = document.getElementById( 'scenes' );
  if ( ! links ) return;
  links.replaceChildren();
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

/** Shows the start screen, set to the last visit's choice or else this device's, with a warning where
 * WebGPU is missing; resolves with the choice once the visitor presses Start. */
export async function chooseGraphics(): Promise<Graphics> {

  const screen = document.getElementById( 'setup' )!;
  const form = screen.querySelector( 'form' )!;
  const note = document.getElementById( 'setup-note' )!;
  const start = form.querySelector( 'button' )!;
  const tier = form.elements.namedItem( 'tier' ) as RadioNodeList;
  const fps = form.elements.namedItem( 'fps' ) as RadioNodeList;

  // (no WebGPU types in this TypeScript's DOM library)
  const gpu = ( navigator as { gpu?: { requestAdapter(): Promise<{ info?: { isFallbackAdapter?: boolean } } | null> } } ).gpu;
  const adapter = await gpu?.requestAdapter().catch( () => null ) ?? null;
  const phone = matchMedia( '(pointer: coarse)' ).matches;
  // Chrome only; 4 GB or less marks an older or budget phone
  const memory = ( navigator as { deviceMemory?: number } ).deviceMemory ?? 8;
  let pick: Graphics = phone ? { tier: 'low', fps: 30 } : { tier: 'high', fps: 0 };
  try {

    pick = { ...pick, ...JSON.parse( stored( GRAPHICS_KEY ) ?? '{}' ) };

  } catch {

    // an unreadable stored choice: this device's then

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
  store( GRAPHICS_KEY, JSON.stringify( choice ) );
  return choice;

}

// the volume of a first visit: a little under the full mix
const DEFAULT_VOLUME = 0.8;

// local storage, which a private window may refuse: a read then finds nothing and a write lasts
// this visit
const stored = ( key: string ) => {

  try {

    return localStorage.getItem( key );

  } catch {

    return null;

  }

};
const store = ( key: string, value: string ) => {

  try {

    localStorage.setItem( key, value );

  } catch {

    // kept for this visit only

  }

};

/** The sound's mute and volume from an earlier visit (`<prefix>.muted`, `<prefix>.volume`). */
export function readSound( prefix: string ): { muted: boolean; volume: number } {

  const volume = Number( stored( `${ prefix }.volume` ) ?? DEFAULT_VOLUME );
  return { muted: stored( `${ prefix }.muted` ) === '1', volume: Number.isFinite( volume ) ? Math.min( Math.max( volume, 0 ), 1 ) : DEFAULT_VOLUME };

}

/** The HUD's sound controls: the mute button and the volume slider, remembered under `prefix`.
 * Turning the volume up unmutes. */
export function soundControls( sound: { muted: boolean; volume: number }, prefix: string ): void {

  const button = document.getElementById( 'sound' );
  const slider = document.getElementById( 'volume' ) as HTMLInputElement | null;
  if ( ! button || ! slider ) return;
  const show = () => {

    button.textContent = sound.muted ? 'Sound: off' : 'Sound: on';
    button.setAttribute( 'aria-pressed', String( ! sound.muted ) );

  };
  const mute = ( muted: boolean ) => {

    sound.muted = muted;
    show();
    store( `${ prefix }.muted`, muted ? '1' : '0' );

  };
  slider.value = String( sound.volume );
  show();
  button.addEventListener( 'click', () => mute( ! sound.muted ) );
  slider.addEventListener( 'input', () => {

    sound.volume = Number( slider.value );
    store( `${ prefix }.volume`, slider.value );
    if ( sound.muted && sound.volume > 0 ) mute( false );

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
