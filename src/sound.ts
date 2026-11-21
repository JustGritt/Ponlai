// The scene's sound: the open sea all round, surf on the rocks, the wind (more of it the higher the
// camera), gulls wheeling by day, the wreck's timbers creaking, and a splash where dropped wreckage
// meets the water.
//
// The clips are generated (public/audio; the README says how) and mixed here with Web Audio through
// three's listener on the camera. The surf, the gulls, the creaks and the splashes are positional,
// each through an HRTF panner that falls off with distance, so the surf swells as the camera nears a
// rock; the sea and the wind are not. The sources stand still apart from the gulls, so their panners
// are placed directly rather than through the scene graph. Browsers start audio only after a
// gesture: the context resumes on the first pointer or key press, and sleeps while the tab is hidden.
import * as THREE from 'three/webgpu';

const CLIPS = {
  ocean: 'ocean', surf: 'surf', wind: 'wind',
  gulls: [ 'gull-1', 'gull-2', 'gull-3' ], creaks: [ 'creak-1', 'creak-2' ], splashes: [ 'splash-1', 'splash-2' ],
};

// the mix: each layer's gain
const MIX = {
  ocean: 0.5,
  // the wind at the sea's level and high above it
  wind: { low: 0.07, high: 0.38 },
  surf: 0.85,
  gull: 0.75,
  creak: 0.7,
  splash: 1,
};

export interface SoundSources {
  /** where surf breaks on the rocks, and how loud each is (0 to 1) */
  surf: { position: THREE.Vector3; gain: number }[];
  /** where the gulls wheel: the circles' centres and radii, and how often each is chosen */
  roosts: { centre: THREE.Vector3; radius: number; weight: number }[];
  /** the wreck's timbers */
  wreck: THREE.Vector3;
}

export interface Soundscape {
  /** follows the camera and plays the gulls and creaks; `night` is the sky's night share, 0 to 1 */
  update( dt: number, night: number, seaLevel: number ): void;
  /** a splash at x, y, z where something struck the water at `speed` (m/s) */
  splash( x: number, y: number, z: number, speed: number ): void;
  muted: boolean;
}

const random = ( a: number, b: number ) => a + Math.random() * ( b - a );
const pick = <T>( list: readonly T[] ) => list[ Math.floor( Math.random() * list.length ) ];

export function createSoundscape( camera: THREE.Camera, sources: SoundSources, muted: boolean ): Soundscape {

  const listener = new THREE.AudioListener();
  camera.add( listener );
  const context = listener.context;
  // a gentle compressor on the master, so a splash close by does not clip over the surf
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

  /** places a panner at p now, or glides it there over `glide` seconds */
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

  // a few voices for each kind of one-shot, reused in turn
  class Voices {

    private readonly voices: THREE.PositionalAudio[];
    private next = 0;
    constructor( count: number, ref: number, rolloff: number ) {

      this.voices = Array.from( { length: count }, () => positional( ref, rolloff ) );

    }
    play( buffer: AudioBuffer, at: THREE.Vector3, volume: number, rate: number ): THREE.PositionalAudio {

      const audio = this.voices[ this.next ++ % this.voices.length ];
      if ( audio.isPlaying ) audio.stop();
      audio.setBuffer( buffer );
      audio.setVolume( volume );
      audio.setPlaybackRate( rate );
      audio.play();
      place( audio, at );
      return audio;

    }

  }

  let layers: {
    ocean: THREE.Audio; wind: THREE.Audio;
    gulls: AudioBuffer[]; creaks: AudioBuffer[]; splashes: AudioBuffer[];
    gullVoices: Voices; creakVoices: Voices; splashVoices: Voices;
  } | null = null;

  const loader = new THREE.AudioLoader();
  const base = new URL( 'audio/', document.baseURI ).href;
  const load = ( name: string ) => loader.loadAsync( `${ base }${ name }.ogg` );
  void Promise.all( [
    load( CLIPS.ocean ), load( CLIPS.surf ), load( CLIPS.wind ),
    Promise.all( CLIPS.gulls.map( load ) ), Promise.all( CLIPS.creaks.map( load ) ), Promise.all( CLIPS.splashes.map( load ) ),
  ] ).then( ( [ ocean, surf, wind, gulls, creaks, splashes ] ) => {

    const loop = ( buffer: AudioBuffer, volume: number ) => {

      const audio = new THREE.Audio( listener ).setBuffer( buffer );
      audio.setLoop( true );
      audio.setVolume( volume );
      audio.offset = Math.random() * buffer.duration;
      audio.play();
      return audio;

    };
    // the surf: one loop at each source, each at its own rate and point in the clip so none beat together
    for ( const source of sources.surf ) {

      const audio = positional( 10, 1.2 ).setBuffer( surf );
      audio.setLoop( true );
      audio.setVolume( MIX.surf * source.gain );
      audio.setPlaybackRate( random( 0.82, 1.08 ) );
      audio.offset = Math.random() * surf.duration;
      audio.play();
      place( audio, source.position );

    }
    layers = {
      ocean: loop( ocean, MIX.ocean ), wind: loop( wind, MIX.wind.low ),
      gulls, creaks, splashes,
      gullVoices: new Voices( 3, 14, 1 ), creakVoices: new Voices( 2, 7, 1.3 ), splashVoices: new Voices( 3, 8, 1 ),
    };

  } ).catch( ( error ) => console.warn( 'sound: the clips did not load', error ) );

  // the gulls in flight: each voice's circle, where it is on it and how fast it goes round
  const flights = new Map<THREE.PositionalAudio, { centre: THREE.Vector3; radius: number; angle: number; speed: number; height: number }>();
  const at = new THREE.Vector3();
  let gullIn = random( 2, 5 ), creakIn = random( 3, 8 );
  let windLevel = - 1, oceanLevel = - 1;

  const gull = () => {

    const total = sources.roosts.reduce( ( s, r ) => s + r.weight, 0 );
    let w = Math.random() * total;
    const roost = sources.roosts.find( ( r ) => ( w -= r.weight ) <= 0 ) ?? sources.roosts[ 0 ];
    const flight = { centre: roost.centre, radius: roost.radius * random( 0.6, 1.2 ), angle: random( 0, Math.PI * 2 ), speed: random( 0.25, 0.5 ) * pick( [ - 1, 1 ] ), height: random( - 3, 6 ) };
    at.set( Math.cos( flight.angle ) * flight.radius, flight.height, Math.sin( flight.angle ) * flight.radius ).add( flight.centre );
    const audio = layers!.gullVoices.play( pick( layers!.gulls ), at, MIX.gull * random( 0.55, 1 ), random( 0.9, 1.15 ) );
    flights.set( audio, flight );

  };

  return {
    get muted() {

      return muted;

    },
    set muted( value: boolean ) {

      muted = value;
      listener.gain.gain.setTargetAtTime( muted ? 0 : 1, context.currentTime, 0.15 );

    },
    update( dt, night, seaLevel ) {

      // the listener follows the camera
      camera.updateMatrixWorld();
      if ( ! layers || context.state !== 'running' ) return;
      // the wind rises and the sea's wash falls back as the camera climbs
      const high = THREE.MathUtils.smoothstep( camera.position.y - seaLevel, 4, 90 );
      const wind = THREE.MathUtils.lerp( MIX.wind.low, MIX.wind.high, high ), ocean = MIX.ocean * ( 1 - 0.35 * high );
      if ( Math.abs( wind - windLevel ) > 0.004 ) layers.wind.setVolume( windLevel = wind );
      if ( Math.abs( ocean - oceanLevel ) > 0.004 ) layers.ocean.setVolume( oceanLevel = ocean );
      // the gulls call by day, fewer at dusk, none at night; now and then one answers another
      const day = 1 - THREE.MathUtils.smoothstep( night, 0.15, 0.65 );
      gullIn -= dt * day;
      if ( day > 0.05 && gullIn <= 0 ) {

        gull();
        gullIn = Math.random() < 0.3 ? random( 0.5, 1.4 ) : random( 3, 11 );

      }
      for ( const [ audio, flight ] of flights ) {

        if ( ! audio.isPlaying ) {

          flights.delete( audio );
          continue;

        }
        flight.angle += flight.speed * dt;
        at.set( Math.cos( flight.angle ) * flight.radius, flight.height, Math.sin( flight.angle ) * flight.radius ).add( flight.centre );
        place( audio, at, Math.max( dt, 0.016 ) );

      }
      // the wreck groans as the swell works it
      creakIn -= dt;
      if ( creakIn <= 0 ) {

        at.copy( sources.wreck ).add( new THREE.Vector3( random( - 3, 3 ), random( - 1, 1 ), random( - 3, 3 ) ) );
        layers.creakVoices.play( pick( layers.creaks ), at, MIX.creak * random( 0.5, 1 ), random( 0.82, 1.05 ) );
        creakIn = random( 6, 16 );

      }

    },
    splash( x, y, z, speed ) {

      if ( ! layers || context.state !== 'running' ) return;
      const volume = MIX.splash * THREE.MathUtils.clamp( 0.35 + speed / 12, 0.35, 1 );
      layers.splashVoices.play( pick( layers.splashes ), at.set( x, y, z ), volume, random( 0.85, 1.15 ) );

    },
  };

}
