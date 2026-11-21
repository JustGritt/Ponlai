// The scene's sound: the open sea all round, surf on the rocks, the wind (more of it the higher the
// camera), the gulls calling by day, the fishing boat's engine, the wreck's timbers creaking, and a
// splash where dropped wreckage meets the water.
//
// The clips are generated (public/audio; the README says how) and mixed here with Web Audio through
// three's listener on the camera. The surf, the gulls, the engine, the creaks and the splashes are
// positional, each through an HRTF panner that falls off with distance, so the surf swells as the
// camera nears a rock; the sea and the wind are not. A gull's call comes from one of the gulls in
// the scene (gulls.ts) and follows it as it flies; the engine follows the boat and opens up with
// its throttle. The listener, the panners and the one-shot voices are audio.ts's.
import * as THREE from 'three/webgpu';
import { Voices, createAudioRig, pick, random } from './audio.ts';

const CLIPS = {
  ocean: 'ocean', surf: 'surf', wind: 'wind', engine: 'engine',
  gulls: [ 'gull-1', 'gull-2', 'gull-3' ], creaks: [ 'creak-1', 'creak-2' ], splashes: [ 'splash-1', 'splash-2' ],
};

// the mix: each layer's gain
const MIX = {
  ocean: 0.5,
  // the wind at the sea's level and high above it
  wind: { low: 0.07, high: 0.38 },
  surf: 0.85,
  gull: 0.75,
  // the engine at idle and at full throttle
  engine: { idle: 0.35, full: 0.6 },
  creak: 0.7,
  splash: 1,
};

export interface SoundSources {
  /** where surf breaks on the rocks, and how loud each is (0 to 1) */
  surf: { position: THREE.Vector3; gain: number }[];
  /** a gull to call from (its position, live), or none */
  gull: () => THREE.Vector3 | null;
  /** the boat: where it is (live) and its throttle, 0 to 1 */
  boat: { position: THREE.Vector3; readonly throttle: number };
  /** the wreck's timbers */
  wreck: THREE.Vector3;
}

export interface Soundscape {
  /** follows the camera and plays the gulls and creaks; `night` is the sky's night share, 0 to 1 */
  update( dt: number, night: number, seaLevel: number ): void;
  /** a splash at x, y, z where something struck the water at `speed` (m/s); `size` scales it
   * down for something small (about 0.2 for a fish's) */
  splash( x: number, y: number, z: number, speed: number, size?: number ): void;
  muted: boolean;
}

export function createSoundscape( camera: THREE.Camera, sources: SoundSources, muted: boolean ): Soundscape {

  const rig = createAudioRig( camera, muted );
  const { context, place, positional } = rig;

  let layers: {
    ocean: THREE.Audio; wind: THREE.Audio; engine: THREE.PositionalAudio;
    gulls: AudioBuffer[]; creaks: AudioBuffer[]; splashes: AudioBuffer[];
    gullVoices: Voices; creakVoices: Voices; splashVoices: Voices;
  } | null = null;

  const load = rig.load;
  void Promise.all( [
    load( CLIPS.ocean ), load( CLIPS.surf ), load( CLIPS.wind ), load( CLIPS.engine ),
    Promise.all( CLIPS.gulls.map( load ) ), Promise.all( CLIPS.creaks.map( load ) ), Promise.all( CLIPS.splashes.map( load ) ),
  ] ).then( ( [ ocean, surf, wind, engineClip, gulls, creaks, splashes ] ) => {

    const loop = ( buffer: AudioBuffer, volume: number ) => rig.loop( buffer, volume ) as THREE.Audio;
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
    // the engine: one loop on the boat
    const engine = positional( 6, 1.1 ).setBuffer( engineClip );
    engine.setLoop( true );
    engine.setVolume( MIX.engine.idle );
    engine.offset = Math.random() * engineClip.duration;
    engine.play();
    place( engine, sources.boat.position );
    layers = {
      ocean: loop( ocean, MIX.ocean ), wind: loop( wind, MIX.wind.low ), engine,
      gulls, creaks, splashes,
      gullVoices: new Voices( rig, 3, 14, 1 ), creakVoices: new Voices( rig, 2, 7, 1.3 ), splashVoices: new Voices( rig, 3, 8, 1 ),
    };

  } ).catch( ( error ) => console.warn( 'sound: the clips did not load', error ) );

  // the gulls calling: each voice and the gull it follows
  const flights = new Map<THREE.PositionalAudio, THREE.Vector3>();
  const at = new THREE.Vector3();
  let gullIn = random( 2, 5 ), creakIn = random( 3, 8 );
  let windLevel = - 1, oceanLevel = - 1, engineLevel = - 1;

  const gull = () => {

    const bird = sources.gull();
    if ( ! bird ) return;
    const audio = layers!.gullVoices.play( pick( layers!.gulls ), bird, MIX.gull * random( 0.55, 1 ), random( 0.9, 1.15 ) );
    flights.set( audio, bird );

  };

  return {
    get muted() {

      return rig.muted;

    },
    set muted( value: boolean ) {

      rig.muted = value;

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
      for ( const [ audio, bird ] of flights ) {

        if ( ! audio.isPlaying ) {

          flights.delete( audio );
          continue;

        }
        place( audio, bird, Math.max( dt, 0.016 ) );

      }
      // the engine follows the boat, louder and a touch faster as it opens up
      const throttle = sources.boat.throttle;
      place( layers.engine, sources.boat.position, Math.max( dt, 0.016 ) );
      const engine = THREE.MathUtils.lerp( MIX.engine.idle, MIX.engine.full, throttle );
      if ( Math.abs( engine - engineLevel ) > 0.01 ) {

        layers.engine.setVolume( engineLevel = engine );
        layers.engine.setPlaybackRate( 0.94 + 0.12 * throttle );

      }
      // the wreck groans as the swell works it
      creakIn -= dt;
      if ( creakIn <= 0 ) {

        at.copy( sources.wreck ).add( new THREE.Vector3( random( - 3, 3 ), random( - 1, 1 ), random( - 3, 3 ) ) );
        layers.creakVoices.play( pick( layers.creaks ), at, MIX.creak * random( 0.5, 1 ), random( 0.82, 1.05 ) );
        creakIn = random( 6, 16 );

      }

    },
    splash( x, y, z, speed, size = 1 ) {

      if ( ! layers || context.state !== 'running' ) return;
      const volume = MIX.splash * THREE.MathUtils.clamp( 0.35 + speed / 12, 0.35, 1 ) * size;
      // a small thing's splash is higher and shorter
      const rate = random( 0.85, 1.15 ) * ( 1 + ( 1 - size ) * 0.9 );
      layers.splashVoices.play( pick( layers.splashes ), at.set( x, y, z ), volume, rate );

    },
  };

}
