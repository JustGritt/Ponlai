// The waterfall's sound: the roar of both falls, the run babbling between them and at the pool's
// tail, the wind (more of it the higher the camera), and a splash wherever a salmon breaks the water.
//
// The falls' roar and the run's babble (public/audio/falls.ogg, river.ogg) are synthesized and
// periodic by construction, so they loop without a seam (the README says how); the wind and the
// splashes are the lighthouse's clips. Every water sound is positional: the main falls' roar comes
// from two points across its crest, so it spreads across the view, the upper falls' from its foot,
// slower and so deeper; the camera hears each swell as it nears it. The listener and the voices are
// audio.ts's.
import * as THREE from 'three/webgpu';
import { Voices, createAudioRig, pick, random } from '../audio.ts';

const CLIPS = { falls: 'falls', river: 'river', wind: 'wind', splashes: [ 'splash-1', 'splash-2' ] };

// the mix: each layer's gain
const MIX = {
  main: 0.75,
  upper: 0.7,
  run: 0.45,
  tail: 0.3,
  // the wind low in the gorge and high over it
  wind: { low: 0.04, high: 0.32 },
  splash: 0.85,
};

export interface FallsSoundSources {
  /** where the main falls' roar comes from (across its crest, at mid height) */
  main: THREE.Vector3[];
  /** the upper falls' foot */
  upper: THREE.Vector3;
  /** where the run babbles, and the pool's tail */
  run: THREE.Vector3[];
  tail: THREE.Vector3;
}

export interface FallsSound {
  /** follows the camera; `ground` is the pool's level */
  update( dt: number, ground: number ): void;
  /** a fish's splash at x, y, z, of `size` (1 a fish falling back in) */
  splash( x: number, y: number, z: number, size: number ): void;
  muted: boolean;
}

export function createFallsSound( camera: THREE.Camera, sources: FallsSoundSources, muted: boolean ): FallsSound {

  const rig = createAudioRig( camera, muted );
  let layers: { wind: THREE.Audio; splashes: AudioBuffer[]; splashVoices: Voices } | null = null;
  let windLevel = - 1;
  const at = new THREE.Vector3();

  void Promise.all( [ rig.load( CLIPS.falls ), rig.load( CLIPS.river ), rig.load( CLIPS.wind ), Promise.all( CLIPS.splashes.map( rig.load ) ) ] )
    .then( ( [ falls, river, wind, splashes ] ) => {

      // each loop from its own point in its clip, so none beat together
      for ( const position of sources.main ) rig.loop( falls, MIX.main / Math.sqrt( sources.main.length ), { position, ref: 7, rolloff: 1, rate: random( 0.96, 1.04 ) } );
      rig.loop( falls, MIX.upper, { position: sources.upper, ref: 8, rolloff: 1, rate: 0.8 } );
      for ( const position of sources.run ) rig.loop( river, MIX.run / Math.sqrt( sources.run.length ), { position, ref: 4, rolloff: 1.2, rate: random( 0.95, 1.05 ) } );
      rig.loop( river, MIX.tail, { position: sources.tail, ref: 4, rolloff: 1.2, rate: 1.15 } );
      layers = { wind: rig.loop( wind, MIX.wind.low ) as THREE.Audio, splashes, splashVoices: new Voices( rig, 4, 6, 1 ) };

    } ).catch( ( error ) => console.warn( 'sound: the clips did not load', error ) );

  return {
    get muted() {

      return rig.muted;

    },
    set muted( value: boolean ) {

      rig.muted = value;

    },
    update( _dt, ground ) {

      // the listener follows the camera
      camera.updateMatrixWorld();
      if ( ! layers || ! rig.running ) return;
      // the wind rises as the camera climbs out of the gorge
      const high = THREE.MathUtils.smoothstep( camera.position.y - ground, 6, 45 );
      const wind = THREE.MathUtils.lerp( MIX.wind.low, MIX.wind.high, high );
      if ( Math.abs( wind - windLevel ) > 0.004 ) layers.wind.setVolume( windLevel = wind );

    },
    splash( x, y, z, size ) {

      if ( ! layers || ! rig.running ) return;
      // a small splash is quieter, higher and shorter
      const volume = MIX.splash * ( 0.3 + 0.7 * size );
      const rate = random( 0.9, 1.1 ) * ( 1 + ( 1 - size ) * 0.6 );
      layers.splashVoices.play( pick( layers.splashes ), at.set( x, y, z ), volume, rate );

    },
  };

}
