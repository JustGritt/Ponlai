// What both scenes' sound is built on: three's listener on the camera, a gentle compressor on the
// master, the clips loaded from public/audio, positional voices placed directly (not through the
// scene graph), and the mute. Browsers start audio only after a gesture: the context resumes on the
// first pointer or key press, and sleeps while the tab is hidden.
import * as THREE from 'three/webgpu';

export const random = ( a: number, b: number ) => a + Math.random() * ( b - a );
export const pick = <T>( list: readonly T[] ) => list[ Math.floor( Math.random() * list.length ) ];

export interface AudioRig {
  listener: THREE.AudioListener;
  context: AudioContext;
  /** whether the context is running (after a gesture, while the tab shows) */
  readonly running: boolean;
  muted: boolean;
  /** a clip from public/audio, by name */
  load( name: string ): Promise<AudioBuffer>;
  /** a positional voice: full at `ref` metres, falling off by `rolloff` (the inverse model) */
  positional( ref: number, rolloff: number ): THREE.PositionalAudio;
  /** places a panner at p now, or glides it there over `glide` seconds */
  place( audio: THREE.PositionalAudio, p: THREE.Vector3, glide?: number ): void;
  /** a loop playing from a random point: positional at `at` (with its `ref` and `rolloff`), or not */
  loop( buffer: AudioBuffer, volume: number, at?: { position: THREE.Vector3; ref: number; rolloff: number; rate?: number } ): THREE.Audio | THREE.PositionalAudio;
}

export function createAudioRig( camera: THREE.Camera, muted: boolean ): AudioRig {

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
  listener.gain.gain.value = muted ? 0 : 1;

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
      listener.gain.gain.setTargetAtTime( muted ? 0 : 1, context.currentTime, 0.15 );

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
