// The river's water over the valley: the two falls, the run between them and the river above the
// upper falls, the foam on the pool, the mist at the falls' feet and the splashes. The shapes come
// from valley.ts, the drawing from water.ts.
//
// - The main falls pour over the whole crest, more in three tongues than between them (`mainFlow`),
//   2.7 m into the pool; the upper falls pour 7.8 m from their notch into the upper pool.
// - The run's current quickens from the upper pool's half a metre a second to nearly two at the
//   main crest, where the water goes glassy before it tips over; it foams under the upper falls,
//   in riffles over its shallows and in the wakes of the boulders that stand in it.
// - The pool's current spreads out from the main falls' feet and slows downstream, with an eddy
//   back up along each side; the foam boils at the feet and trails downstream on it.
import type * as THREE from 'three/webgpu';
import type { RenderParts } from '../render/index.ts';
import { CREST, UPPER, crestZ, runLevel, topLevel, upperZ, type Boulder, type HeightGrid, type Valley } from './valley.ts';
import { fallTime, fallsMist, fallsSheets, poolFoam, splashes, streamSurface, type FallsSpec, type Reach, type Splashes } from './water.ts';

const smooth = ( e0: number, e1: number, x: number ) => {

  const t = Math.min( Math.max( ( x - e0 ) / ( e1 - e0 ), 0 ), 1 );
  return t * t * ( 3 - 2 * t );

};

/** How much of the main falls' water goes over at x (0 to 1): three tongues, thinning to the ends. */
export function mainFlow( x: number ): number {

  const f = 0.62 + 0.3 * Math.sin( 1.15 * x + 0.6 ) + 0.18 * Math.sin( 2.7 * x + 2.1 );
  return Math.min( Math.max( f, 0.06 ), 1 ) * smooth( CREST.half, CREST.half - 1.3, Math.abs( x ) );

}

export const MAIN_FALLS: FallsSpec = {
  name: 'main falls', half: CREST.half, crest: crestZ, lip: CREST.height + 0.12, plunge: 0,
  speed: ( x ) => 1.45 + 0.7 * mainFlow( x ), flow: mainFlow, spread: 0.32,
};

export const UPPER_FALLS: FallsSpec = {
  name: 'upper falls', half: UPPER.half, crest: upperZ, lip: topLevel - 0.1, plunge: runLevel( - 21 ),
  speed: ( x ) => 1.9 + 0.5 * Math.cos( x * 1.3 ), spread: 0.5,
  flow: ( x ) => Math.min( 1, 0.78 + 0.22 * Math.sin( 2.1 * x + 1 ) ) * smooth( UPPER.half, UPPER.half - 0.6, Math.abs( x ) ),
};

/** Where the main falls' water at x meets the pool (z). */
export const mainPlunge = ( x: number ) => crestZ( x ) + MAIN_FALLS.speed( x ) * fallTime( MAIN_FALLS );
/** Where the upper falls' water meets the upper pool (z, at the middle). */
export const upperPlunge = upperZ( 0 ) + UPPER_FALLS.speed( 0 ) * fallTime( UPPER_FALLS );

/** The run, from the main crest up to the upper falls' face. */
function runReach( boulders: readonly Boulder[] ): Reach {

  const far = upperZ( 0 ) + 0.15;
  const inRun = boulders.filter( ( b ) => b.z < 0 && b.z > - 20 );
  return {
    across: 90, along: 130,
    at( u, s ) {

      const x = - 9 + 18 * u;
      const z = crestZ( x ) * ( 1 - s ) + far * s;
      const toCrest = z - crestZ( x );
      // the water drops to the lip over its last metre
      const y = runLevel( z ) - ( runLevel( z ) - MAIN_FALLS.lip ) * smooth( - 0.9, 0, toCrest );
      const speed = 0.4 + 1.3 * smooth( - 15, - 0.6, z ) + 0.3 * smooth( - 0.8, 0, toCrest );
      // foam: under the upper falls, in riffles, and in the boulders' wakes
      const below = z - upperPlunge;
      let foam = ( below < 0 ? 1 : Math.exp( - below / 2.4 ) ) * smooth( UPPER.half + 3, UPPER.half + 0.5, Math.abs( x ) );
      foam = Math.max( foam, 0.24 * smooth( - 14, - 4, z ) );
      for ( const b of inRun ) {

        const dz = z - b.z, dx = Math.abs( x - b.x );
        if ( dz > - b.r - 0.4 && dz < 5 && dx < b.r * ( 1.1 + Math.max( dz, 0 ) * 0.25 ) ) foam = Math.max( foam, 0.85 * Math.exp( - Math.max( dz, 0 ) / 2 ) );

      }
      // glassy where it gathers to go over
      foam *= smooth( 0, - 2.5, toCrest );
      return { x, y, z, vx: - 0.11 * Math.cos( z * 0.22 ) * speed, vz: speed, foam };

    },
  };

}

/** The river above the upper falls, following its channel up out of sight. */
function topReach(): Reach {

  return {
    across: 30, along: 110,
    at( u, s ) {

      const across = - 5 + 10 * u;
      const z0 = upperZ( across );
      const z = z0 * ( 1 - s ) + - 80 * s;
      const above = Math.max( 0, - z - 30 );
      const x = - 0.006 * above * above + across;
      const toCrest = z - upperZ( x );
      const y = topLevel - ( topLevel - UPPER_FALLS.lip ) * smooth( - 0.9, 0, toCrest );
      const speed = 0.9 + 0.9 * smooth( - 4, 0, toCrest );
      const foam = 0.12 * smooth( - 6, - 14, z ) * smooth( - 0.5, - 3, toCrest );
      return { x, y, z, vx: 0.012 * above * speed, vz: speed, foam };

    },
  };

}

/** The pool's current and its foam's density at x, z. */
function poolField( x: number, z: number ): { vx: number; vz: number; foam: number; boil: number } {

  const half = MAIN_FALLS.half;
  const cx = Math.max( - half, Math.min( half, x ) );
  const dz = z - mainPlunge( cx );
  const ax = Math.abs( x );
  const span = smooth( half + 1 + Math.max( dz, 0 ) * 0.35, half - 0.8, ax );
  const strength = 0.55 + 0.45 * MAIN_FALLS.flow( cx );
  const boil = dz < 0 ? 1 : Math.exp( - dz / 1.8 );
  const trail = 0.68 * Math.exp( - Math.max( dz, 0 ) / 7 );
  const foam = span * strength * Math.max( boil, trail );
  const solid = span * strength * ( dz < 0 ? 1 : Math.exp( - dz / 0.9 ) );
  let vz = dz < 0 ? - 0.25 : 0.28 + 1.15 * Math.exp( - dz / 3.2 ) * span;
  // the eddies: back up along each side toward the falls
  vz -= 0.55 * smooth( 5.8, 8.2, ax ) * Math.exp( - Math.max( dz, 0 ) / 6 );
  const vx = 0.16 * ( x / 8 ) * Math.exp( - Math.max( dz, 0 ) / 5 );
  return { vx, vz, foam, boil: solid };

}

/** The current at x, z (m/s) where a fish swims: the pool's, the run's or the top's. */
export function current( x: number, z: number ): { vx: number; vz: number } {

  if ( z > crestZ( x ) + 0.15 ) {

    const f = poolField( x, z );
    return { vx: f.vx, vz: f.vz };

  }
  if ( z > upperZ( x ) ) return { vx: 0, vz: 0.4 + 1.3 * smooth( - 15, - 0.6, z ) };
  return { vx: 0, vz: 0.9 };

}

export interface River {
  /** what draws in the water's pass */
  meshes: THREE.Mesh[];
  splashes: Splashes;
}

export function createRiver( render: RenderParts, valley: Valley, grid: HeightGrid, boulders: readonly Boulder[], sea: number ): River {

  const meshes = [
    streamSurface( render, grid, sea, [ runReach( boulders ), topReach() ] ),
    fallsSheets( render, grid, sea, MAIN_FALLS ),
    fallsSheets( render, grid, sea, UPPER_FALLS ),
    poolFoam( render, grid, sea, {
      area: { x0: - 11, z0: 0, x1: 11, z1: 30 }, step: 0.25,
      wet: ( x, z ) => valley.height( x, z ) < - 0.02, field: poolField,
    } ),
    fallsMist( render, sea, MAIN_FALLS, 170, 240 ),
    fallsMist( render, sea, UPPER_FALLS, 120, 150 ),
  ];
  const sprays = splashes( render );
  meshes.push( sprays.mesh );
  return { meshes, splashes: sprays };

}

