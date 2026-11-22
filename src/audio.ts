// What the scene's sound is built on: three's listener on the camera, a gentle compressor on the
// master, the clips loaded from public/audio, positional voices placed directly (not through the
// scene graph), the mute and the volume, and the quiet start. Browsers start audio only after a
// gesture: the context resumes on the first pointer or key press (the start screen's Start is one),
// and sleeps while the tab is hidden.
import * as THREE from 'three/webgpu';

// the share of the volume the sound creeps up to while the loading screen still hides the scene,
// and how fast (s, the time constant): a sea at full strength over a still-black page startles
const LOADING_SHARE = 0.25, LOADING_RISE = 4;
// the time constant (s) of the rise to the full volume once the scene shows
const REVEAL_RISE = 1.5;

export const random = ( a: number, b: number ) => a + Math.random() * ( b - a );
export const pick = <T>( list: readonly T[] ) => list[ Math.floor( Math.random() * list.length ) ];

export interface AudioRig {
  listener: THREE.AudioListener;
  context: AudioContext;
  /** whether the context is running (after a gesture, while the tab shows) */
  readonly running: boolean;
  muted: boolean;
  /** the visitor's volume, 0 to 1 (squared into the gain, so the slider reads as loudness) */
  volume: number;
  /** the clips have started: the sound rises from silence (slowly to a quiet share while loading) */
  begin(): void;
  /** the scene shows: the sound rises the rest of the way */
  reveal(): void;
  /** a clip from public/audio, by name */
  load( name: string ): Promise<AudioBuffer>;
  /** a positional voice: full at `ref` metres, falling off by `rolloff` (the inverse model) */
  positional( ref: number, rolloff: number ): THREE.PositionalAudio;
  /** places a panner at p now, or glides it there over `glide` seconds */
  place( audio: THREE.PositionalAudio, p: THREE.Vector3, glide?: number ): void;
  /** a loop playing from a random point: positional at `at` (with its `ref` and `rolloff`), or not */
  loop( buffer: AudioBuffer, volume: number, at?: { position: THREE.Vector3; ref: number; rolloff: number; rate?: number } ): THREE.Audio | THREE.PositionalAudio;
}

export function createAudioRig( camera: THREE.Camera, settings: { muted: boolean; volume: number } ): AudioRig {

  let { muted, volume } = settings;

  const listener = new THREE.AudioListener();
  camera.add( listener );
  const context = listener.context;
  // a gentle compressor on the master, so a splash close by does not clip over the rest
  const compressor = context.createDynamicsCompressor();
  compressor.threshold.value = - 14;
  compressor.knee.value = 12;
  compressor.ratio.value = 3;
  compressor.attack.value = 0.01;
  compressor.release.value = 0.3;
  listener.gain.disconnect();
  listener.gain.connect( compressor ).connect( context.destination );
  // the master: silent until the clips start, then the volume (or none, muted) times the page's share
  const master = listener.gain.gain;
  let share = LOADING_SHARE, begun = false;
  const glide = ( seconds: number ) => master.setTargetAtTime( begun && ! muted ? volume * volume * share : 0, context.currentTime, seconds );
  master.value = 0;

  // start on the first gesture; sleep while hidden
  let started = false;
  const resume = () => {

    started = true;
    if ( context.state !== 'running' && ! document.hidden ) void context.resume();

  };
  window.addEventListener( 'pointerdown', resume );
  window.addEventListener( 'keydown', resume );
  document.addEventListener( 'visibilitychange', () => {

    if ( document.hidden ) void context.suspend();
    else if ( started ) void context.resume();

  } );

  const place = ( audio: THREE.PositionalAudio, p: THREE.Vector3, glide = 0 ) => {

    const t = context.currentTime;
    const { positionX, positionY, positionZ } = audio.panner;
    for ( const [ param, value ] of [ [ positionX, p.x ], [ positionY, p.y ], [ positionZ, p.z ] ] as const ) {

      if ( glide > 0 ) {

        param.linearRampToValueAtTime( value, t + glide );

      } else {

        param.cancelScheduledValues( t );
        param.setValueAtTime( value, t );

      }

    }

  };

  const positional = ( ref: number, rolloff: number ) => {

    const audio = new THREE.PositionalAudio( listener );
    audio.setRefDistance( ref );
    audio.setRolloffFactor( rolloff );
    audio.setDistanceModel( 'inverse' );
    audio.setMaxDistance( 4000 );
    return audio;

  };

  const loader = new THREE.AudioLoader();
  const base = new URL( 'audio/', document.baseURI ).href;

  return {
    listener, context, place, positional,
    get running() {

      return context.state === 'running';

    },
    get muted() {

      return muted;

    },
    set muted( value: boolean ) {

      muted = value;
      glide( 0.15 );

    },
    get volume() {

      return volume;

    },
    set volume( value: number ) {

      volume = value;
      glide( 0.05 );

    },
    begin() {

      begun = true;
      glide( share < 1 ? LOADING_RISE : REVEAL_RISE );

    },
    reveal() {

      share = 1;
      glide( REVEAL_RISE );

    },
    load: ( name ) => loader.loadAsync( `${ base }${ name }.ogg` ),
    loop( buffer, volume, at ) {

      const audio = at ? positional( at.ref, at.rolloff ) : new THREE.Audio( listener );
      audio.setBuffer( buffer );
      audio.setLoop( true );
      audio.setVolume( volume );
      if ( at?.rate ) audio.setPlaybackRate( at.rate );
      audio.offset = Math.random() * buffer.duration;
      audio.play();
      if ( at ) place( audio as THREE.PositionalAudio, at.position );
      return audio;

    },
  };

}

/** A few positional voices for one kind of one-shot, reused in turn. */
export class Voices {

  private readonly voices: THREE.PositionalAudio[];
  private next = 0;

  constructor( private readonly rig: AudioRig, count: number, ref: number, rolloff: number ) {

    this.voices = Array.from( { length: count }, () => rig.positional( ref, rolloff ) );

  }

  play( buffer: AudioBuffer, at: THREE.Vector3, volume: number, rate: number ): THREE.PositionalAudio {

    const audio = this.voices[ this.next ++ % this.voices.length ];
    if ( audio.isPlaying ) audio.stop();
    audio.setBuffer( buffer );
    audio.setVolume( volume );
    audio.setPlaybackRate( rate );
    audio.play();
    this.rig.place( audio, at );
    return audio;

  }

}
