// The page round the scene: the start screen, the HUD's text, its sound controls, its settings panel
// (the time of day and the graphics), and the sound's mute and volume kept between visits.
import type { DayPreset, RenderParts } from './render/index.ts';

/** Sets the HUD's help line and its Settings button, which shows and hides the settings panel. */
export function setUpPage( help: string ): void {

  const helpLine = document.getElementById( 'help' );
  if ( helpLine ) helpLine.textContent = help;
  const toggle = document.getElementById( 'settings-toggle' );
  const panel = document.getElementById( 'settings' );
  if ( ! toggle || ! panel ) return;
  toggle.addEventListener( 'click', () => {

    panel.hidden = ! panel.hidden;
    toggle.setAttribute( 'aria-expanded', String( ! panel.hidden ) );

  } );

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

/** The hour as a clock shows it ("19:05"). */
const clockTime =( hour: number ): string => {

  const minutes = Math.floor( hour * 60 ) % 1440;
  return `${ String( Math.floor( minutes / 60 ) ).padStart( 2, '0' ) }:${ String( minutes % 60 ).padStart( 2, '0' ) }`;

};

/** The settings panel's time of day: the slider and its clock, which follow the day as it passes, the
 * day clock's pause, and the presets. */
export function timeControls( render: RenderParts ): void {

  const slider = document.getElementById( 'hour' ) as HTMLInputElement | null;
  const clock = document.getElementById( 'clock' );
  const pause = document.getElementById( 'pause' );
  if ( ! slider || ! clock || ! pause ) return;
  const { dayCycle } = render;

  // each hour set bakes the sky again and the bakes queue, so a drag sets only its latest hour once
  // the bake before it is done
  let wanted: number | null = null, seeking = false;
  const seek = async () => {

    if ( seeking ) return;
    seeking = true;
    try {

      while ( wanted !== null ) {

        const hour = wanted;
        wanted = null;
        await render.setHour( hour );

      }

    } finally {

      seeking = false;

    }

  };
  slider.addEventListener( 'input', () => {

    wanted = Number( slider.value );
    clock.textContent = clockTime( wanted );
    void seek();

  } );
  // the slider follows the day, but not while it is held or its hour is still being set
  const follow = () => {

    if ( seeking || slider.matches( ':active' ) ) return;
    slider.value = String( dayCycle.hour );
    clock.textContent = clockTime( dayCycle.hour );

  };
  follow();
  setInterval( follow, 500 );

  const showPause = () => {

    pause.textContent = dayCycle.paused ? 'Resume' : 'Pause';

  };
  showPause();
  pause.addEventListener( 'click', () => {

    dayCycle.paused = ! dayCycle.paused;
    showPause();

  } );

  for ( const button of document.querySelectorAll<HTMLButtonElement>( '[data-preset]' ) ) {

    button.addEventListener( 'click', () => void render.setPreset( button.dataset.preset as DayPreset ).then( follow ) );

  }

}

/** The settings panel's graphics, kept in the address and as the start screen's next choice: the
 * frame-rate cap, set at once, and the quality tier, which the pipeline reads as it loads, so a new
 * one reloads the scene (straight in, at the same hour). */
export function graphicsControls( render: RenderParts, pacer: { readonly mode: string; readonly rate: number; setRate( rate: number | 'auto' ): void } ): void {

  const tiers = document.querySelectorAll<HTMLButtonElement>( '[data-tier]' );
  const rates = document.querySelectorAll<HTMLButtonElement>( '[data-fps]' );
  const params = new URLSearchParams( location.search );
  // the cap as the start screen names it: 0 is the tier's own rate
  let fps = pacer.mode === 'fixed' ? pacer.rate : 0;
  const show = () => {

    for ( const b of tiers ) b.setAttribute( 'aria-pressed', String( b.dataset.tier === render.qualityTier ) );
    for ( const b of rates ) b.setAttribute( 'aria-pressed', String( Number( b.dataset.fps ) === fps ) );

  };
  show();

  for ( const button of rates ) button.addEventListener( 'click', () => {

    fps = Number( button.dataset.fps );
    pacer.setRate( fps || 'auto' );
    if ( fps ) params.set( 'frameRate', String( fps ) );
    else params.delete( 'frameRate' );
    history.replaceState( null, '', `?${ params }` );
    store( GRAPHICS_KEY, JSON.stringify( { tier: render.qualityTier, fps } ) );
    show();

  } );

  for ( const button of tiers ) button.addEventListener( 'click', () => {

    const tier = button.dataset.tier!;
    if ( tier === render.qualityTier ) return;
    store( GRAPHICS_KEY, JSON.stringify( { tier, fps } ) );
    params.set( 'quality', tier );
    params.delete( 'preset' );
    params.set( 'hour', render.dayCycle.hour.toFixed( 3 ) );
    if ( render.dayCycle.paused ) params.set( 'clock', 'paused' );
    else params.delete( 'clock' );
    location.search = params.toString();

  } );

}

let shownStatus = '';
/** The HUD's status line (the count beside the buttons). */
export function showStatus( text: string ): void {

  if ( text === shownStatus ) return;
  shownStatus = text;
  const el = document.getElementById( 'count' );
  if ( el ) el.textContent = text;

}
