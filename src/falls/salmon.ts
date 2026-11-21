// The salmon: sockeye in their spawning colours, red with green heads, running up the river. They
// hold in the pool under the main falls and leap at it; a few make it over and go on up the run.
//
// The mesh and its material are Ponlai's fish (life/FishGeometry.ts, life/FishMaterial.ts) given a
// salmon's anatomy and a spawning sockeye's skin: deep in the body with a hump before the dorsal
// fin, an adipose fin, a broad tail barely forked; red, the head olive green, little silver left.
// One instanced mesh in the opaque pass, marked as a swimmer (render/ocean/Refraction.ts) and moved
// to where the eye sees it through the pool's surface (render/ocean/Water.ts swimmerSway); in the
// run above the falls the river's own surface (water.ts) shows it from the opaque image.
//
// What each fish does, on the CPU, per frame:
// - holds in the pool: a station in its deeper water, facing into the current (the fish's heading is
//   its swimming against the water, not its ground track), moving station now and then;
// - every so often (an exponential wait, longer by night) swims up under the falls, to a spot in
//   front of where the water lands, more often where more water falls, and leaps: up at 5.2 to 8.3
//   m/s, toward the crest at 1.3 to 2.6;
// - in the air it flies a ballistic arc. Where it meets the falling water, it carries on through
//   only if its arc would still clear the crest (by more where more water falls); otherwise the
//   water takes it, tumbling, back down into the pool. A fish over the crest lands in the run;
//   one that falls short drops back; each breaks the water with a splash;
// - back in the pool it drifts with the current, dazed, for a moment, then holds again;
// - over the crest it swims up the run to rest in the upper pool under the upper falls, which no
//   salmon can leap. After a while there it swims in under that falls' foam, where it is lost to
//   sight, and comes back into the river far downstream, out of sight, to run up to the pool again.
//
// History: the scene clock's step drives them, from a seeded stream (none while paused).
import * as THREE from 'three/webgpu';
import { markSwimmer, shared, swimmerSway, type RenderParts } from '../render/index.ts';
import { fishGeometry, wingRoot, type Anatomy } from '../life/FishGeometry.ts';
import { makeFishMaterial, type Skin } from '../life/FishMaterial.ts';
import { mulberry32 } from '../wreckage.ts';
import { CREST, crestZ, runLevel, type Boulder, type Valley } from './valley.ts';
import { MAIN_FALLS, current, mainFlow, mainPlunge, upperPlunge } from './river.ts';

const SALMON: Anatomy = {
  body: 0.82, sec: 2.3,
  top: [ [ 0, 0.008 ], [ 0.04, 0.035 ], [ 0.1, 0.07 ], [ 0.2, 0.1 ], [ 0.32, 0.112 ], [ 0.45, 0.1 ], [ 0.62, 0.066 ], [ 0.8, 0.036 ], [ 0.92, 0.028 ], [ 1, 0.03 ] ],
  bot: [ [ 0, 0.006 ], [ 0.04, 0.028 ], [ 0.1, 0.05 ], [ 0.2, 0.066 ], [ 0.35, 0.07 ], [ 0.5, 0.06 ], [ 0.7, 0.038 ], [ 0.85, 0.026 ], [ 1, 0.024 ] ],
  wid: [ [ 0, 0.005 ], [ 0.05, 0.028 ], [ 0.15, 0.05 ], [ 0.3, 0.056 ], [ 0.5, 0.048 ], [ 0.7, 0.03 ], [ 0.9, 0.016 ], [ 1, 0.013 ] ],
  dorsal: [
    { from: 0.36, to: 0.5, rays: 11, h: [ [ 0, 0.075 ], [ 1, 0.02 ] ], rake: [ 0.5, 0.9 ], notch: 0.01 },
    // the adipose fin
    { from: 0.79, to: 0.84, rays: 4, h: [ [ 0, 0.022 ], [ 1, 0.014 ] ], rake: [ 0.3, 0.5 ], notch: 0 },
  ],
  anal: [ { from: 0.66, to: 0.76, rays: 9, h: [ [ 0, 0.05 ], [ 1, 0.016 ] ], rake: [ 0.5, 0.9 ], notch: 0.01 } ],
  pectoral: { u: 0.19, y: - 0.035, rays: 12, lead: 0.9, last: 0.3, len: 0.1, tail: 0.5, dihedral: 0.2 },
  pelvic: { u: 0.5, y: - 0.055, rays: 8, lead: 0.7, last: 0.3, len: 0.075, tail: 0.6, dihedral: 0.05 },
  caudal: { len: 0.17, span: 0.2, fork: 0.12, lower: 1.0, rays: 15 },
};

const SKIN: Skin = {
  back: 0x7d1a16, marks: 0x7d1a16, flank: 0xa8261d, belly: 0xb8796a, head: 0x4a5b2c, snout: 0x6a7646,
  fin: 0x3d4527, finEdge: 0x1d2214, iris: 0xc8b070,
  headEnd: 0.21, snoutEnd: 0.035, headOver: 'all', silver: 0.12,
  eye: { u: 0.065, y: 0.022, r: 0.016 },
};

const COUNT = 34;
const LENGTH = [ 0.56, 0.74 ] as const;
// the mean wait between a holding fish's leaps (s), by day and by night; the speed out of the water
// (m/s): up and toward the crest
const LEAP = { day: 70, night: 260, up: [ 5.2, 8.3 ], along: [ 1.3, 2.6 ] } as const;
// how many rest in the upper pool before the longest there goes on
const UPPER_ROOM = 6;
const G = 9.81, TAU = Math.PI * 2;
const HOLD = 0, APPROACH = 1, RISE = 2, AIR = 3, CAUGHT = 4, RECOVER = 5, UPSTREAM = 6, RESTING = 7, LEAVING = 8, ENTERING = 9;
const SWIMMING = new Set( [ HOLD, APPROACH, RECOVER, UPSTREAM, RESTING, LEAVING, ENTERING ] );

export interface Salmon {
  mesh: THREE.Mesh;
  /** steps the fish by the scene clock's `dt`; `night` 0 by day to 1 by night */
  update( dt: number, night: number ): void;
  /** the leaps so far, and how many made it over */
  readonly leaps: number;
  readonly made: number;
  /** the fish in the air now (world), for the tools */
  airborne(): THREE.Vector3[];
  /** sends the fish nearest the falls up to leap now (for the tools) */
  leapNow(): void;
}

/** The river's bed and level on a fine grid over the fish's water (bilinear). */
class Riverbed {

  private readonly bed: Float32Array;
  private readonly level: Float32Array;
  private readonly nx: number;
  private readonly nz: number;

  constructor( valley: Valley, readonly x0: number, readonly z0: number, x1: number, z1: number, readonly cell: number ) {

    this.nx = Math.round( ( x1 - x0 ) / cell ) + 1;
    this.nz = Math.round( ( z1 - z0 ) / cell ) + 1;
    this.bed = new Float32Array( this.nx * this.nz );
    this.level = new Float32Array( this.nx * this.nz );
    for ( let j = 0; j < this.nz; j ++ ) for ( let i = 0; i < this.nx; i ++ ) {

      const x = x0 + i * cell, z = z0 + j * cell;
      this.bed[ j * this.nx + i ] = valley.height( x, z );
      this.level[ j * this.nx + i ] = valley.levelAt( x, z );

    }

  }

  private sample( data: Float32Array, x: number, z: number ): number {

    const fx = Math.min( Math.max( ( x - this.x0 ) / this.cell, 0 ), this.nx - 1.001 );
    const fz = Math.min( Math.max( ( z - this.z0 ) / this.cell, 0 ), this.nz - 1.001 );
    const i = Math.floor( fx ), j = Math.floor( fz ), tx = fx - i, tz = fz - j, n = this.nx, k = j * n + i;
    return ( data[ k ] * ( 1 - tx ) + data[ k + 1 ] * tx ) * ( 1 - tz ) + ( data[ k + n ] * ( 1 - tx ) + data[ k + n + 1 ] * tx ) * tz;

  }

  bedAt( x: number, z: number ): number { return this.sample( this.bed, x, z ); }
  levelAt( x: number, z: number ): number { return this.sample( this.level, x, z ); }
  depthAt( x: number, z: number ): number { return this.levelAt( x, z ) - this.bedAt( x, z ); }

  /** the deepest x across the river at z, within `reach` of x */
  deepest( x: number, z: number, reach = 10 ): number {

    let best = x, bestDepth = - Infinity;
    for ( let d = - reach; d <= reach; d += 0.5 ) {

      const depth = this.depthAt( x + d, z ) - Math.abs( d ) * 0.02;
      if ( depth > bestDepth ) {

        bestDepth = depth;
        best = x + d;

      }

    }
    return best;

  }

}

export function createSalmon( render: RenderParts, o: {
  sea: number;
  valley: Valley;
  boulders: readonly Boulder[];
  /** a fish broke the water at x, y, z (over the sea) on water at `level`; `size` 1 for a full splash */
  onSplash?: ( x: number, y: number, z: number, level: number, size: number ) => void;
} ): Salmon {

  const base = fishGeometry( SALMON, 0 );
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = base.index;
  for ( const name of [ 'position', 'normal', 'aData' ] ) geometry.setAttribute( name, base.getAttribute( name ) );
  const attr = () => new THREE.InstancedBufferAttribute( new Float32Array( COUNT * 4 ), 4 );
  const attrs = { iPose: attr(), iTurn: attr(), iMotion: attr(), iBefore: attr() };
  for ( const [ name, a ] of Object.entries( attrs ) ) geometry.setAttribute( name, a );
  geometry.instanceCount = COUNT;
  const roots = { pectoral: wingRoot( SALMON, SALMON.pectoral ), pelvic: wingRoot( SALMON, SALMON.pelvic ) };
  const material = makeFishMaterial( SALMON, SKIN, shared, render.waveMaps, roots, ( p ) => swimmerSway( render.fft, p, shared.waterLevel ) );
  material.name = 'salmon';
  markSwimmer( material );
  const mesh = new THREE.Mesh( geometry, material );
  mesh.name = 'salmon';
  mesh.frustumCulled = false;

  const rng = mulberry32( 1907 );
  const between = ( [ a, b ]: readonly [ number, number ] ) => a + rng() * ( b - a );
  const river = new Riverbed( o.valley, - 14, - 25, 14, 72, 0.25 );
  const stones = o.boulders.filter( ( b ) => b.z > - 25 && b.z < 72 );

  // per fish: position (y over the sea), velocity, heading, the previous position and phase,
  // the target it swims to, its timer, and the launch it leaps with
  const pos = new Float32Array( COUNT * 3 ), vel = new Float32Array( COUNT * 3 ), head = new Float32Array( COUNT * 3 );
  const prev = new Float32Array( COUNT * 3 ), target = new Float32Array( COUNT * 3 ), launch = new Float32Array( COUNT * 3 );
  const size = new Float32Array( COUNT ), roll = new Float32Array( COUNT ), spin = new Float32Array( COUNT ), bend = new Float32Array( COUNT );
  const phase = new Float32Array( COUNT ), prevPhase = new Float32Array( COUNT ), amp = new Float32Array( COUNT ), effort = new Float32Array( COUNT );
  const state = new Uint8Array( COUNT ), timer = new Float32Array( COUNT ), through = new Uint8Array( COUNT ), restedAt = new Float32Array( COUNT );
  let leaps = 0, made = 0, clock = 0;

  const wait = ( night: number ) => - Math.log( 1 - rng() * 0.999 ) * ( LEAP.day + ( LEAP.night - LEAP.day ) * night );

  /** a station in the pool's deeper water, clear of the falls' feet and the stones */
  function poolStation( i: number ): void {

    for ( let k = 0; k < 40; k ++ ) {

      const x = ( rng() - 0.5 ) * 15, z = 3 + rng() * 15;
      if ( z < mainPlunge( Math.max( - CREST.half, Math.min( CREST.half, x ) ) ) + 1.2 ) continue;
      const depth = river.depthAt( x, z );
      if ( depth < 0.9 || ! clearOfStones( x, z, 0.5 ) ) continue;
      target.set( [ x, - ( 0.35 + rng() * Math.min( 1.4, depth - 0.45 ) ), z ], i * 3 );
      return;

    }
    target.set( [ 0, - 0.8, 9 ], i * 3 );

  }

  /** a station in the upper pool, under the upper falls */
  function upperStation( i: number ): void {

    const z = upperPlunge + 2.2 + rng() * 3, x = ( rng() - 0.5 ) * 5;
    const level = river.levelAt( x, z ), depth = river.depthAt( x, z );
    target.set( [ x, level - Math.min( 0.6, Math.max( 0.15, depth * 0.5 ) ), z ], i * 3 );

  }

  function clearOfStones( x: number, z: number, margin: number ): boolean {

    return stones.every( ( b ) => Math.hypot( x - b.x, z - b.z ) > b.r + margin );

  }

  function place( i: number, x: number, y: number, z: number ): void {

    pos.set( [ x, y, z ], i * 3 );
    prev.set( [ x, y, z ], i * 3 );
    vel.set( [ 0, 0, - 0.2 ], i * 3 );
    head.set( [ 0, 0, - 1 ], i * 3 );

  }

  /** back into the river far downstream, out of sight, to run up to the pool */
  function enter( i: number ): void {

    const z = 46 + rng() * 22;
    const x = river.deepest( 0.006 * ( z - 16 ) ** 2, z ) + ( rng() - 0.5 ) * 1.5;
    place( i, x, - 0.5, z );
    state[ i ] = ENTERING;
    poolStation( i );

  }

  for ( let i = 0; i < COUNT; i ++ ) {

    size[ i ] = between( LENGTH );
    phase[ i ] = rng() * TAU;
    amp[ i ] = 0.06;
    if ( i < 24 ) {

      state[ i ] = HOLD;
      poolStation( i );
      place( i, target[ i * 3 ], target[ i * 3 + 1 ], target[ i * 3 + 2 ] );
      timer[ i ] = wait( 0 ) * ( 0.15 + rng() );

    } else if ( i < 28 ) {

      state[ i ] = RESTING;
      upperStation( i );
      place( i, target[ i * 3 ], target[ i * 3 + 1 ], target[ i * 3 + 2 ] );
      restedAt[ i ] = - rng() * 60;

    } else {

      enter( i );

    }

  }

  // the curtain's z at height y (over the sea) above the crest at x: where a leaping fish meets the
  // falling water
  const end = MAIN_FALLS.lip - MAIN_FALLS.plunge;
  function curtainZ( x: number, y: number ): number {

    const cx = Math.max( - CREST.half, Math.min( CREST.half, x ) );
    const t = Math.sqrt( 2 * Math.min( Math.max( MAIN_FALLS.lip - y, 0 ), end ) / G );
    return crestZ( cx ) + MAIN_FALLS.speed( cx ) * t;

  }

  /** the toward-the-crest direction at x (unit, ground plane) */
  function towardCrest( x: number ): [ number, number ] {

    const slope = ( crestZ( x + 0.01 ) - crestZ( x - 0.01 ) ) / 0.02, l = Math.hypot( slope, 1 );
    return [ slope / l, - 1 / l ];

  }

  function startLeap( i: number ): void {

    let x = 0;
    for ( let k = 0; k < 12; k ++ ) {

      x = ( rng() - 0.5 ) * 2 * ( CREST.half - 0.7 );
      if ( rng() < 0.3 + 0.7 * mainFlow( x ) ) break;

    }
    const z = mainPlunge( x ) + 0.55 + rng() * 1.3;
    target.set( [ x, - 0.75, z ], i * 3 );
    state[ i ] = APPROACH;
    timer[ i ] = 14;

  }

  /** steers a swimming fish toward its target at `speed` (m/s over the ground) */
  function swim( i: number, dt: number, speed: number, holdStill: boolean ): void {

    // the run is shallow: a fish there keeps to less water than in the pool
    const shallow = pos[ i * 3 + 2 ] < crestZ( pos[ i * 3 ] ) + 0.2 ? 0.1 : 0.22;

    const i3 = i * 3;
    let x = pos[ i3 ], y = pos[ i3 + 1 ], z = pos[ i3 + 2 ];
    let vx = vel[ i3 ], vy = vel[ i3 + 1 ], vz = vel[ i3 + 2 ];
    const dx = target[ i3 ] - x, dy = target[ i3 + 1 ] - y, dz = target[ i3 + 2 ] - z;
    const d = Math.hypot( dx, dy, dz ) + 1e-6;
    const want = Math.min( speed, d * ( holdStill ? 0.6 : 1.2 ) + 0.02 );
    let ax = ( dx / d * want - vx ) * 2.2, ay = ( dy / d * want - vy ) * 2.2, az = ( dz / d * want - vz ) * 2.2;
    // the others: keep a body length apart
    for ( let j = 0; j < COUNT; j ++ ) {

      if ( j === i || ! SWIMMING.has( state[ j ] ) ) continue;
      const ex = x - pos[ j * 3 ], ey = y - pos[ j * 3 + 1 ], ez = z - pos[ j * 3 + 2 ];
      const e2 = ex * ex + ey * ey + ez * ez, r = ( size[ i ] + size[ j ] ) * 0.75;
      if ( e2 > r * r || e2 < 1e-8 ) continue;
      const e = Math.sqrt( e2 ), f = ( r - e ) / r * 3;
      ax += ex / e * f;
      ay += ey / e * f * 0.5;
      az += ez / e * f;

    }
    // the stones: round them
    for ( const b of stones ) {

      const ex = x - b.x, ez = z - b.z, e = Math.hypot( ex, ez ), r = b.r + 0.5;
      if ( e > r || e < 1e-6 ) continue;
      ax += ex / e * ( r - e ) * 6;
      az += ez / e * ( r - e ) * 6;

    }
    // a little wander
    ax += Math.sin( clock * 0.7 + i * 3.1 ) * 0.15;
    az += Math.cos( clock * 0.53 + i * 1.7 ) * 0.1;
    const al = Math.hypot( ax, ay, az ), amax = 2.5 + speed * 2;
    if ( al > amax ) {

      ax *= amax / al; ay *= amax / al; az *= amax / al;

    }
    vx += ax * dt; vy += ay * dt; vz += az * dt;
    const s = Math.hypot( vx, vy, vz ), vmax = speed * 1.3 + 0.2;
    if ( s > vmax ) {

      vx *= vmax / s; vy *= vmax / s; vz *= vmax / s;

    }
    let nx = x + vx * dt, nz = z + vz * dt;
    // the banks: never into water shallower than a fish
    if ( river.depthAt( nx, nz ) < shallow ) {

      const mid = river.deepest( x, z, 4 );
      nx = x + Math.sign( mid - x ) * Math.min( Math.abs( mid - x ), 0.6 * dt + 0.01 );
      nz = z;
      vx *= 0.3;
      vz *= 0.3;

    }
    x = nx;
    z = nz;
    y += vy * dt;
    // in the water, clear of its bed
    const level = river.levelAt( x, z ), bed = river.bedAt( x, z );
    y = Math.min( Math.max( y, bed + 0.14 ), level - 0.1 );
    pos[ i3 ] = x; pos[ i3 + 1 ] = y; pos[ i3 + 2 ] = z;
    vel[ i3 ] = vx; vel[ i3 + 1 ] = vy; vel[ i3 + 2 ] = vz;
    effort[ i ] = Math.min( 2.5, Math.hypot( vx - current( x, z ).vx, vz - current( x, z ).vz ) / Math.max( size[ i ], 0.3 ) );

  }

  function splash( x: number, y: number, z: number, level: number, s: number ): void {

    o.onSplash?.( x, y, z, level, s );

  }

  function step( i: number, dt: number, night: number ): void {

    const i3 = i * 3;
    const st = state[ i ];
    switch ( st ) {

      case HOLD: {

        swim( i, dt, 0.5, true );
        timer[ i ] -= dt;
        if ( timer[ i ] <= 0 ) {

          startLeap( i );

        } else if ( rng() < dt / 12 ) {

          poolStation( i );

        }
        break;

      }
      case APPROACH: {

        swim( i, dt, 1.5, false );
        timer[ i ] -= dt;
        const dx = target[ i3 ] - pos[ i3 ], dz = target[ i3 + 2 ] - pos[ i3 + 2 ];
        if ( Math.hypot( dx, dz ) < 0.5 ) {

          // up and out
          const [ tx, tz ] = towardCrest( pos[ i3 ] ), along = between( LEAP.along );
          launch.set( [ tx * along, between( LEAP.up ), tz * along ], i3 );
          vel.set( [ launch[ i3 ] * 0.6, launch[ i3 + 1 ] * 0.7, launch[ i3 + 2 ] * 0.6 ], i3 );
          state[ i ] = RISE;

        } else if ( timer[ i ] <= 0 ) {

          state[ i ] = HOLD;
          timer[ i ] = wait( night );
          poolStation( i );

        }
        break;

      }
      case RISE: {

        pos[ i3 ] += vel[ i3 ] * dt;
        pos[ i3 + 1 ] += vel[ i3 + 1 ] * dt;
        pos[ i3 + 2 ] += vel[ i3 + 2 ] * dt;
        if ( pos[ i3 + 1 ] > - 0.05 ) {

          state[ i ] = AIR;
          through[ i ] = 0;
          vel.set( launch.subarray( i3, i3 + 3 ), i3 );
          leaps ++;
          splash( pos[ i3 ], 0, pos[ i3 + 2 ], 0, 0.55 );

        }
        break;

      }
      case AIR: {

        vel[ i3 + 1 ] -= G * dt;
        pos[ i3 ] += vel[ i3 ] * dt;
        pos[ i3 + 1 ] += vel[ i3 + 1 ] * dt;
        pos[ i3 + 2 ] += vel[ i3 + 2 ] * dt;
        const x = pos[ i3 ], y = pos[ i3 + 1 ], z = pos[ i3 + 2 ];
        const lip = MAIN_FALLS.lip, crest = crestZ( Math.max( - CREST.half, Math.min( CREST.half, x ) ) );
        if ( y > lip + 0.03 && z < crest + 0.1 ) {

          // over the crest: into the run
          state[ i ] = UPSTREAM;
          made ++;
          const level = runLevel( z );
          pos[ i3 + 1 ] = level - 0.15;
          vel.set( [ 0, 0, - 1.2 ], i3 );
          splash( x, level, z, level, 0.45 );
          upperStation( i );
          break;

        }
        if ( ! through[ i ] && y < lip && z < curtainZ( x, y ) + 0.08 ) {

          // into the falling water: through it if its arc would still clear the crest
          const apex = y + Math.max( vel[ i3 + 1 ], 0 ) ** 2 / ( 2 * G );
          const need = lip + 0.1 + mainFlow( x ) * 0.35;
          if ( vel[ i3 + 1 ] > 0 && apex - 0.12 > need ) {

            through[ i ] = 1;
            vel[ i3 + 1 ] -= 0.7;
            vel[ i3 ] *= 0.75;
            vel[ i3 + 2 ] *= 0.75;

          } else {

            state[ i ] = CAUGHT;
            spin[ i ] = ( rng() < 0.5 ? - 1 : 1 ) * ( 5 + rng() * 6 );

          }
          break;

        }
        if ( z < crest + 0.12 && y < lip ) {

          // against the rock under the crest
          state[ i ] = CAUGHT;
          spin[ i ] = ( rng() < 0.5 ? - 1 : 1 ) * 6;
          break;

        }
        if ( y < 0 && vel[ i3 + 1 ] < 0 ) {

          splash( x, 0, z, 0, 1 );
          state[ i ] = RECOVER;
          timer[ i ] = 1.2 + rng() * 2;
          vel.set( [ vel[ i3 ] * 0.3, - 1.2, vel[ i3 + 2 ] * 0.3 ], i3 );

        }
        break;

      }
      case CAUGHT: {

        // carried down and out with the falling water, tumbling
        const x = pos[ i3 ];
        const k = 1 - Math.exp( - dt * 5 );
        const [ tx, tz ] = towardCrest( x ), out = MAIN_FALLS.speed( Math.max( - CREST.half, Math.min( CREST.half, x ) ) ) * 1.1;
        vel[ i3 ] += ( - tx * out - vel[ i3 ] ) * k;
        vel[ i3 + 2 ] += ( - tz * out - vel[ i3 + 2 ] ) * k;
        vel[ i3 + 1 ] -= G * 1.1 * dt;
        pos[ i3 ] += vel[ i3 ] * dt;
        pos[ i3 + 1 ] += vel[ i3 + 1 ] * dt;
        pos[ i3 + 2 ] += vel[ i3 + 2 ] * dt;
        roll[ i ] += spin[ i ] * dt;
        if ( pos[ i3 + 1 ] < 0 ) {

          splash( pos[ i3 ], 0, pos[ i3 + 2 ], 0, 0.8 );
          state[ i ] = RECOVER;
          timer[ i ] = 1.5 + rng() * 2.5;
          vel.set( [ vel[ i3 ] * 0.3, - 1.5, vel[ i3 + 2 ] * 0.5 ], i3 );

        }
        break;

      }
      case RECOVER: {

        // dazed: carried off by the current, then holding again
        const c = current( pos[ i3 ], pos[ i3 + 2 ] );
        target.set( [ pos[ i3 ] + c.vx, Math.min( pos[ i3 + 1 ], - 0.6 ), pos[ i3 + 2 ] + Math.max( c.vz, 0.3 ) * 1.5 ], i3 );
        swim( i, dt, 1.0, false );
        timer[ i ] -= dt;
        if ( timer[ i ] <= 0 ) {

          state[ i ] = HOLD;
          timer[ i ] = wait( night );
          poolStation( i );

        }
        break;

      }
      case UPSTREAM: {

        swim( i, dt, 1.4, false );
        const dx = target[ i3 ] - pos[ i3 ], dz = target[ i3 + 2 ] - pos[ i3 + 2 ];
        if ( Math.hypot( dx, dz ) < 0.6 ) {

          state[ i ] = RESTING;
          restedAt[ i ] = clock;

        }
        break;

      }
      case RESTING: {

        swim( i, dt, 0.45, true );
        if ( rng() < dt / 15 ) upperStation( i );
        break;

      }
      case LEAVING: {

        swim( i, dt, 1.2, false );
        const dx = target[ i3 ] - pos[ i3 ], dz = target[ i3 + 2 ] - pos[ i3 + 2 ];
        if ( Math.hypot( dx, dz ) < 0.5 ) enter( i );
        break;

      }
      case ENTERING: {

        // up the river; the station's x where the channel allows, else its deepest water
        const z = pos[ i3 + 2 ];
        if ( z > 20 ) target.set( [ river.deepest( pos[ i3 ], z - 4, 6 ), - 0.5, z - 6 ], i3 );
        swim( i, dt, 1.1, false );
        if ( z <= 20 ) {

          state[ i ] = HOLD;
          timer[ i ] = wait( night ) * 0.5;
          poolStation( i );

        }
        break;

      }

    }

  }

  // the upper pool's crowd: the one rested longest goes on, in under the upper falls' foam
  function thin(): void {

    let resting = 0, longest = - 1;
    for ( let i = 0; i < COUNT; i ++ ) {

      if ( state[ i ] !== RESTING ) continue;
      resting ++;
      if ( longest < 0 || restedAt[ i ] < restedAt[ longest ] ) longest = i;

    }
    if ( longest >= 0 && ( resting > UPPER_ROOM || clock - restedAt[ longest ] > 150 ) ) {

      state[ longest ] = LEAVING;
      const level = river.levelAt( 0, upperPlunge - 0.4 );
      target.set( [ ( rng() - 0.5 ) * 1.5, level - 0.35, upperPlunge - 0.5 ], longest * 3 );

    }

  }

  const q = new THREE.Quaternion(), euler = new THREE.Euler( 0, 0, 0, 'YXZ' );
  // turns each fish to its course, beats its tail, and writes the instances (`dt` 0: the pose only)
  const write = ( dt: number ) => {

    const pose = attrs.iPose.array as Float32Array, turn = attrs.iTurn.array as Float32Array;
    const motion = attrs.iMotion.array as Float32Array, before = attrs.iBefore.array as Float32Array;
    const kHead = 1 - Math.exp( - dt * 4 ), kRoll = 1 - Math.exp( - dt * 3 );
    for ( let i = 0; i < COUNT; i ++ ) {

      const i3 = i * 3, i4 = i * 4, L = size[ i ], st = state[ i ];
      const air = st === AIR || st === CAUGHT || st === RISE;
      let hx = vel[ i3 ], hy = vel[ i3 + 1 ], hz = vel[ i3 + 2 ];
      if ( ! air ) {

        // a swimming fish heads where it swims through the water: against the current
        const c = current( pos[ i3 ], pos[ i3 + 2 ] );
        hx -= c.vx * 0.9;
        hz -= c.vz * 0.9;
        hy *= 0.5;

      }
      const hs = Math.hypot( hx, hy, hz );
      if ( dt > 0 && hs > 1e-4 ) {

        const kh = air ? 1 - Math.exp( - dt * 14 ) : kHead * Math.min( 1, hs / ( L * 0.4 ) + 0.1 );
        const ohx = head[ i3 ], ohz = head[ i3 + 2 ];
        let nx = ohx + ( hx / hs - ohx ) * kh, ny = head[ i3 + 1 ] + ( hy / hs - head[ i3 + 1 ] ) * kh, nz = ohz + ( hz / hs - ohz ) * kh;
        const pitch = air ? 0.98 : 0.35;
        ny = Math.max( - pitch, Math.min( pitch, ny ) );
        const nl = Math.hypot( nx, ny, nz ) || 1;
        head[ i3 ] = nx / nl; head[ i3 + 1 ] = ny / nl; head[ i3 + 2 ] = nz / nl;
        const yawRate = ( ohz * nx - ohx * nz ) / dt;
        if ( st !== CAUGHT ) roll[ i ] += ( Math.max( - 0.45, Math.min( 0.45, yawRate * 0.1 ) ) - roll[ i ] ) * kRoll;
        bend[ i ] += ( Math.max( - 0.25, Math.min( 0.25, yawRate * 0.06 ) ) - bend[ i ] ) * kRoll;

      }
      if ( dt > 0 ) {

        // the tail: a slow beat holding, harder with the effort; thrashing in the air
        const e = air ? 0 : effort[ i ];
        const freq = air ? 8 : Math.min( 7, 1.4 + 1.6 * e );
        phase[ i ] = ( phase[ i ] + dt * TAU * freq ) % ( TAU * 64 );
        const a = air ? 0.17 : 0.05 + 0.03 * Math.min( 2, e );
        amp[ i ] += ( a - amp[ i ] ) * ( 1 - Math.exp( - dt * 8 ) );

      }
      euler.set( - Math.asin( head[ i3 + 1 ] ), Math.atan2( head[ i3 ], head[ i3 + 2 ] ), roll[ i ] );
      q.setFromEuler( euler );
      pose[ i4 ] = pos[ i3 ]; pose[ i4 + 1 ] = o.sea + pos[ i3 + 1 ]; pose[ i4 + 2 ] = pos[ i3 + 2 ]; pose[ i4 + 3 ] = L;
      turn[ i4 ] = q.x; turn[ i4 + 1 ] = q.y; turn[ i4 + 2 ] = q.z; turn[ i4 + 3 ] = q.w;
      // the paired fins half folded along the flank
      motion[ i4 ] = phase[ i ]; motion[ i4 + 1 ] = amp[ i ]; motion[ i4 + 2 ] = bend[ i ]; motion[ i4 + 3 ] = 0.55;
      before[ i4 ] = prev[ i3 ]; before[ i4 + 1 ] = o.sea + prev[ i3 + 1 ]; before[ i4 + 2 ] = prev[ i3 + 2 ]; before[ i4 + 3 ] = prevPhase[ i ];

    }
    for ( const a of Object.values( attrs ) ) a.needsUpdate = true;

  };
  write( 0 );

  return {
    mesh,
    get leaps() {

      return leaps;

    },
    get made() {

      return made;

    },
    airborne() {

      const out: THREE.Vector3[] = [];
      for ( let i = 0; i < COUNT; i ++ ) if ( state[ i ] === AIR || state[ i ] === CAUGHT ) out.push( new THREE.Vector3( pos[ i * 3 ], o.sea + pos[ i * 3 + 1 ], pos[ i * 3 + 2 ] ) );
      return out;

    },
    leapNow() {

      let best = - 1, bestZ = Infinity;
      for ( let i = 0; i < COUNT; i ++ ) if ( state[ i ] === HOLD && pos[ i * 3 + 2 ] < bestZ ) {

        best = i;
        bestZ = pos[ i * 3 + 2 ];

      }
      if ( best >= 0 ) startLeap( best );

    },
    update( dt, night ) {

      prev.set( pos );
      prevPhase.set( phase );
      if ( dt > 0 ) {

        const steps = Math.min( 4, Math.ceil( dt * 60 ) ), h = Math.min( dt, 0.1 ) / steps;
        for ( let s = 0; s < steps; s ++ ) {

          clock += h;
          for ( let i = 0; i < COUNT; i ++ ) step( i, h, night );

        }
        thin();

      }
      write( dt );

    },
  };

}
