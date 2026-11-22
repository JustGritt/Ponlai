// The valley: a river gorge with two falls, its terrain's shape, mesh and material, its boulders, and
// the fields the render pipeline's water reads.
//
// The river runs toward +z (the noon sun stands toward +z, so the falls face it). Three reaches,
// each a channel with its own water level, carved into the land:
// - the pool and the river below it at the sea's level (the render pipeline's water: the pool is
//   the only water it draws), a deep plunge hole under the main falls and a gentle valley
//   downstream that bends out of sight;
// - the run above the main falls, a shallow rapid between the main crest and the gorge of the
//   upper falls, with a deeper pool under them;
// - the river above the upper falls, in its notch, turning out of sight upstream.
// A reach's channel is a parabolic bed `depth` deep and `half` wide each side of its centre line,
// its banks rising `steep` metres a metre from the water's edge until they meet the land. The
// reaches meet at the falls in steps a few tenths of a metre long: the crest's sill and the face
// under it. The land rises from the gorge's rim to rolling hills and a ring of mountains, which
// hides the sea's plane at the horizon.
//
// The mesh is one grid, 0.3 m apart round the falls and pool, the spacing growing outward to the
// mountains; its steep faces are pushed along their normals by noise so the cliffs read as rock. Each
// vertex carries its sky's visibility (a horizon scan over the heights), its height over the water
// beside it and the falls' spray on it, which the material reads.
import * as THREE from 'three/webgpu';
import { abs, attribute, cameraPosition, cross, dFdx, dFdy, dot, float, length, max, mix, mx_fractal_noise_float, mx_noise_float, mx_noise_vec3, mx_worley_noise_vec2, normalView, normalWorld, normalize, positionView, positionWorld, sign, smoothstep, vec3 } from 'three/tsl';
import { SimplexNoise } from 'three/addons/math/SimplexNoise.js';
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import type { RenderParts } from '../render/index.ts';
import { rgb, surface } from '../surfaces.ts';
import { mulberry32 } from '../wreckage.ts';

/** The main falls: the crest's height over the pool (m), its bend (the crest line z = bend x^2 up to
 * `reach`, a horseshoe whose ends stand downstream), the half width of the water going over it and
 * the water's depth on the sill. */
export const CREST = { height: 2.6, bend: 0.035, reach: 7, half: 7.2, over: 0.2 };
/** The upper falls: where its crest stands (z), its bend, the crest's height over the pool, the half
 * width of the water going over and the water on its sill. */
export const UPPER = { z: - 24, bend: 0.015, reach: 3, height: 11, half: 2.3, over: 0.3 };
/** The run's level at the upper pool, over the main crest's sill. */
const RUN_RISE = 0.65;

const smooth = ( e0: number, e1: number, x: number ) => {

  const t = Math.min( Math.max( ( x - e0 ) / ( e1 - e0 ), 0 ), 1 );
  return t * t * ( 3 - 2 * t );

};
const bump = ( x: number, c: number, w: number ) => Math.exp( - ( ( ( x - c ) / w ) ** 2 ) );

/** A reach's channel at one z. */
interface Reach { level: number; centre: number; half: number; depth: number; steep: number }

/** The main crest's line: its z at x. */
export const crestZ = ( x: number ) => CREST.bend * Math.min( x * x, CREST.reach * CREST.reach );
/** The upper crest's line. */
export const upperZ = ( x: number ) => UPPER.z + UPPER.bend * Math.min( x * x, UPPER.reach * UPPER.reach );
/** The run's water level at z (over the sea). */
export const runLevel = ( z: number ) => CREST.height + CREST.over + RUN_RISE * smooth( - 1, - 14, z );
/** The river's level above the upper falls. */
export const topLevel = UPPER.height + UPPER.over;

export class Valley {

  private readonly noise: SimplexNoise;
  private readonly wiggle: SimplexNoise;

  constructor( seed = 7 ) {

    const rng = mulberry32( seed );
    this.noise = new SimplexNoise( { random: rng } );
    this.wiggle = new SimplexNoise( { random: rng } );

  }

  private fbm( x: number, z: number, octaves: number ): number {

    let sum = 0, amp = 1, f = 1, norm = 0;
    for ( let o = 0; o < octaves; o ++ ) {

      sum += amp * this.noise.noise( x * f, z * f );
      norm += amp;
      amp *= 0.5;
      f *= 2.03;

    }
    return sum / norm;

  }

  /** the pool and the river below it */
  private down( z: number ): Reach {

    const below = Math.max( 0, z - 16 );
    return {
      level: 0,
      centre: 0.0045 * below * below + 1.2 * this.wiggle.noise( z * 0.025, 0.5 ) * smooth( 14, 30, z ),
      half: 6.3 + 2.9 * bump( z, 7, 7.5 ) + 0.6 * this.wiggle.noise( z * 0.06, 3.1 ),
      depth: 1.0 + 0.4 * bump( z, 9, 6 ),
      // cliffs round the pool's head, a gentle valley downstream
      steep: 0.45 + 3.3 * bump( z, 3, 8 ),
    };

  }

  /** the run between the falls, and the upper pool */
  private run( z: number ): Reach {

    return {
      level: runLevel( z ),
      centre: 0.5 * Math.sin( z * 0.22 ),
      half: 7.6 - 1.8 * smooth( - 1, - 10, z ),
      depth: 0.2 + 0.35 * smooth( - 0.3, - 4, z ) + 1.1 * bump( z, - 20.5, 2.8 ),
      steep: 1.1 + 2.9 * smooth( - 9, - 17, z ),
    };

  }

  /** the river above the upper falls */
  private top( z: number ): Reach {

    const above = Math.max( 0, - z - 30 );
    return {
      level: topLevel,
      centre: - 0.006 * above * above,
      half: 2.6 + 1.4 * smooth( - 25, - 36, z ),
      depth: 0.3 + 0.4 * smooth( - 24.5, - 30, z ),
      steep: 1.6,
    };

  }

  /** the share of the run (1 upstream of the main crest) and of the top (1 upstream of the upper crest) */
  private shares( x: number, z: number ): [ number, number ] {

    return [ smooth( 0.2, - 0.2, z - crestZ( x ) ), smooth( 0.25, - 0.25, z - upperZ( x ) ) ];

  }

  private carve( x: number, z: number, r: Reach ): number {

    const d = Math.abs( x - r.centre ), e = d - r.half;
    if ( e < 0 ) return r.level - r.depth * ( 1 - ( d / r.half ) ** 2 );
    return r.level + r.steep * e + 0.03 * e * e;

  }

  /** the land the channels are cut into: the gorge's rim, hills, and the mountains far off */
  private land( x: number, z: number ): number {

    const fromRiver = Math.abs( x - this.down( Math.max( z, 0 ) ).centre );
    let h = 9.5 + 5 * smooth( - 2, - 20, z ) - 0.07 * Math.max( 0, z - 22 );
    h = Math.max( h, 3 );
    h += 0.9 * this.fbm( x / 9, z / 9, 3 );
    h += 6 * this.fbm( x / 70 + 11, z / 70, 3 ) * smooth( 20, 70, fromRiver );
    h += 4 * Math.abs( this.fbm( x / 28 - 7, z / 28, 3 ) ) * smooth( 12, 45, fromRiver );
    const r = Math.hypot( x, z - 20 );
    h += 160 * smooth( 380, 900, r ) * ( 0.6 + 0.4 * this.fbm( x / 160, z / 160 + 5, 4 ) );
    return h;

  }

  /** the water's level at x, z (over the sea): the level of the reach it lies in, water or not */
  levelAt( x: number, z: number ): number {

    const [ s1, s2 ] = this.shares( x, z );
    return ( 0 * ( 1 - s1 ) + runLevel( z ) * s1 ) * ( 1 - s2 ) + topLevel * s2;

  }

  /** the ground's height at x, z (over the sea) */
  height( x: number, z: number ): number {

    const [ s1, s2 ] = this.shares( x, z );
    const down = this.down( z );
    let low = this.carve( x, z, down );
    // the plunge hole under the main falls
    const toCrest = z - crestZ( x );
    if ( Math.abs( x - down.centre ) < down.half ) low -= 2.3 * bump( toCrest, 2.2, 2.4 ) * smooth( 9, 5.5, Math.abs( x ) );
    const run = this.carve( x, z, this.run( z ) );
    const top = this.carve( x, z, this.top( z ) );
    let h = ( low * ( 1 - s1 ) + run * s1 ) * ( 1 - s2 ) + top * s2;
    // the sills stay smooth: the water over them shows their line
    const sill = Math.max( bump( toCrest, - 0.25, 0.7 ) * smooth( 9, 7, Math.abs( x ) ), bump( z - upperZ( x ), - 0.3, 0.8 ) * smooth( 4, 2.6, Math.abs( x ) ) );
    const level = this.levelAt( x, z );
    // the bed's cobbles and bars; the banks' outcrops
    const under = level - h;
    h += ( 1 - sill ) * ( under > 0 ? 0.14 * this.fbm( x * 0.6, z * 0.6, 3 ) : 0.5 * this.fbm( x * 0.3 + 4, z * 0.3, 3 ) * smooth( 0, 1.5, - under ) );
    return Math.min( h, this.land( x, z ) );

  }

  /** the water's surface at x, z (over the sea), or null where the ground stands above it */
  water( x: number, z: number ): number | null {

    const level = this.levelAt( x, z );
    return this.height( x, z ) < level - 0.02 ? level : null;

  }

  /** the falls' spray on the ground at x, y, z: 1 at their feet, falling off over metres */
  spray( x: number, y: number, z: number ): number {

    const main = Math.hypot( Math.max( Math.abs( x ) - CREST.half, 0 ), z - crestZ( x ) - 1.4 );
    const upper = Math.hypot( Math.max( Math.abs( x ) - UPPER.half, 0 ), z - upperZ( x ) - 1.6 );
    return Math.max( Math.exp( - main / 5 ) * smooth( CREST.height + 9, CREST.height + 2, y ), Math.exp( - upper / 4.5 ) * smooth( UPPER.height + 6, UPPER.height, y ) );

  }

}

// ---- the heights on a regular grid, for the horizon scans and the fields

export class HeightGrid {

  readonly n: number;
  readonly data: Float32Array;

  constructor( valley: Valley, readonly half: number, readonly cell: number ) {

    this.n = Math.round( 2 * half / cell ) + 1;
    this.data = new Float32Array( this.n * this.n );
    for ( let j = 0; j < this.n; j ++ ) for ( let i = 0; i < this.n; i ++ ) this.data[ j * this.n + i ] = valley.height( - half + i * cell, - half + j * cell );

  }

  /** the height at x, z (bilinear; the edge's beyond the grid) */
  at( x: number, z: number ): number {

    const n = this.n;
    const fx = Math.min( Math.max( ( x + this.half ) / this.cell, 0 ), n - 1.001 );
    const fz = Math.min( Math.max( ( z + this.half ) / this.cell, 0 ), n - 1.001 );
    const i = Math.floor( fx ), j = Math.floor( fz ), tx = fx - i, tz = fz - j;
    const d = this.data, k = j * n + i;
    return ( d[ k ] * ( 1 - tx ) + d[ k + 1 ] * tx ) * ( 1 - tz ) + ( d[ k + n ] * ( 1 - tx ) + d[ k + n + 1 ] * tx ) * tz;

  }

  /** the sines of the horizon's elevation seen from x, y, z in HORIZON_DIRECTIONS azimuths (0 at +x,
   * turning toward +z) */
  horizon( x: number, y: number, z: number, out = new Float32Array( HORIZON_DIRECTIONS ) ): Float32Array {

    for ( let k = 0; k < HORIZON_DIRECTIONS; k ++ ) {

      const a = k / HORIZON_DIRECTIONS * Math.PI * 2, cx = Math.cos( a ), cz = Math.sin( a );
      let best = - 1;
      for ( const r of HORIZON_STEPS ) {

        const dh = this.at( x + cx * r, z + cz * r ) - y;
        best = Math.max( best, dh / Math.hypot( r, dh ) );

      }
      out[ k ] = best;

    }
    return out;

  }

  /** the sky's share of a level surface's light at x, y, z (cosine weighted: cos^2 of each azimuth's
   * horizon), from its horizon */
  sky( x: number, y: number, z: number ): number {

    const h = this.horizon( x, y, z, scratch );
    let sum = 0;
    for ( let k = 0; k < HORIZON_DIRECTIONS; k ++ ) {

      const s = Math.max( h[ k ], 0 );
      sum += 1 - s * s;

    }
    return sum / HORIZON_DIRECTIONS;

  }

}

export const HORIZON_DIRECTIONS = 8;
const HORIZON_STEPS = [ 0.7, 1.5, 2.5, 4, 6, 9, 13, 19, 28, 40, 60, 90, 140 ];
const scratch = new Float32Array( HORIZON_DIRECTIONS );

// ---- the mesh

/** The grid's coordinates along one axis: `step` apart over [ from, to ], then growing by `grow` a
 * step for `out` steps each way. */
function axis( from: number, to: number, step: number, grow: number, out: number ): number[] {

  const core: number[] = [];
  const n = Math.round( ( to - from ) / step );
  for ( let i = 0; i <= n; i ++ ) core.push( from + ( to - from ) * i / n );
  const before: number[] = [], after: number[] = [];
  let d = step, a = from, b = to;
  for ( let i = 0; i < out; i ++ ) {

    d *= grow;
    a -= d;
    b += d;
    before.unshift( a );
    after.push( b );

  }
  return [ ...before, ...core, ...after ];

}

/** The terrain's mesh (heights over `sea`), with its `aShade` (sky visibility, height over the water
 * beside it, spray). */
export function valleyGeometry( valley: Valley, grid: HeightGrid, sea: number ): THREE.BufferGeometry {

  const xs = axis( - 17, 17, 0.3, 1.045, 110 ), zs = axis( - 32, 24, 0.3, 1.045, 110 );
  const nx = xs.length, nz = zs.length;
  const position = new Float32Array( nx * nz * 3 );
  for ( let j = 0; j < nz; j ++ ) for ( let i = 0; i < nx; i ++ ) {

    const k = ( j * nx + i ) * 3;
    position[ k ] = xs[ i ];
    position[ k + 1 ] = valley.height( xs[ i ], zs[ j ] );
    position[ k + 2 ] = zs[ j ];

  }
  const index = new Uint32Array( ( nx - 1 ) * ( nz - 1 ) * 6 );
  let t = 0;
  for ( let j = 0; j < nz - 1; j ++ ) for ( let i = 0; i < nx - 1; i ++ ) {

    const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
    // split each quad along the diagonal that follows the ground best
    if ( Math.abs( position[ a * 3 + 1 ] - position[ d * 3 + 1 ] ) < Math.abs( position[ b * 3 + 1 ] - position[ c * 3 + 1 ] ) ) {

      index.set( [ a, c, d, a, d, b ], t );

    } else {

      index.set( [ a, c, b, b, c, d ], t );

    }
    t += 6;

  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute( 'position', new THREE.BufferAttribute( position, 3 ) );
  geometry.setIndex( new THREE.BufferAttribute( index, 1 ) );
  geometry.computeVertexNormals();

  // the cliffs: steep faces pushed along their normals, so they read as broken rock
  const rough = new SimplexNoise( { random: mulberry32( 99 ) } );
  const normal = geometry.getAttribute( 'normal' ) as THREE.BufferAttribute;
  for ( let v = 0; v < nx * nz; v ++ ) {

    const ny = normal.getY( v );
    const steep = smooth( 0.85, 0.45, ny );
    if ( steep <= 0 ) continue;
    const x = position[ v * 3 ], y = position[ v * 3 + 1 ], z = position[ v * 3 + 2 ];
    // not under the crests, where the falls' sheets hang close
    const crest = Math.max( smooth( 1.2, 0, Math.abs( z - crestZ( x ) ) ) * smooth( 8.5, 7, Math.abs( x ) ), smooth( 1.2, 0, Math.abs( z - upperZ( x ) ) ) * smooth( 3.6, 2.6, Math.abs( x ) ) );
    const n = 0.55 * rough.noise3d( x * 0.35, y * 0.5, z * 0.35 ) + 0.3 * rough.noise3d( x * 0.9, y * 1.3, z * 0.9 );
    const push = steep * ( 1 - crest ) * n;
    position[ v * 3 ] += normal.getX( v ) * push;
    position[ v * 3 + 1 ] += ny * push;
    position[ v * 3 + 2 ] += normal.getZ( v ) * push;

  }
  geometry.computeVertexNormals();

  const shade = new Float32Array( nx * nz * 3 );
  for ( let v = 0; v < nx * nz; v ++ ) {

    const x = position[ v * 3 ], y = position[ v * 3 + 1 ], z = position[ v * 3 + 2 ];
    shade[ v * 3 ] = Math.abs( x ) < grid.half && Math.abs( z ) < grid.half ? grid.sky( x, y + 0.15, z ) : 1;
    shade[ v * 3 + 1 ] = y - valley.levelAt( x, z );
    shade[ v * 3 + 2 ] = valley.spray( x, y, z );
    position[ v * 3 + 1 ] = y + sea;

  }
  geometry.setAttribute( 'aShade', new THREE.BufferAttribute( shade, 3 ) );
  geometry.computeBoundingSphere();
  geometry.computeBoundingBox();
  return geometry;

}

// ---- the boulders

/** A boulder: where it stands (x, z), its radius across. */
export interface Boulder { x: number; z: number; r: number }

/** The boulders: round the pool's margin and the falls' feet, in the run (its rapids form round
 * them), and scattered over the banks. */
export function placeBoulders( valley: Valley ): Boulder[] {

  const rng = mulberry32( 3131 );
  const out: Boulder[] = [];
  const free = ( x: number, z: number, r: number ) => out.every( ( b ) => Math.hypot( b.x - x, b.z - z ) > ( b.r + r ) * 0.9 );
  // the fish leap from here: keep it clear
  const leapZone = ( x: number, z: number ) => Math.abs( x ) < CREST.half + 0.8 && z < crestZ( x ) + 6.5;
  const tryPlace = ( x: number, z: number, r: number ) => {

    if ( ! leapZone( x, z ) && free( x, z, r ) ) out.push( { x, z, r } );

  };
  // the falls' feet, either side
  tryPlace( - 8.1, 1.6, 1.3 );
  tryPlace( 8.3, 2.2, 1.6 );
  tryPlace( 7.6, 4.4, 0.9 );
  // the pool's margins: at its water's edge, half in the water
  for ( let k = 0; k < 60 && out.length < 18; k ++ ) {

    const z = 2 + rng() * 22, side = rng() < 0.5 ? - 1 : 1;
    let x = side * ( 5 + rng() * 6 );
    // walk to the water's edge
    for ( let s = 0; s < 30; s ++ ) {

      if ( valley.height( x, z ) > 0.1 ) break;
      x += side * 0.4;

    }
    tryPlace( x - side * rng() * 1.2, z, 0.5 + rng() * 1.1 );

  }
  // the run: a few in the stream
  for ( const [ x, z, r ] of [ [ - 3.4, - 6, 0.8 ], [ 2.6, - 9.5, 1.0 ], [ - 1.2, - 12.5, 0.7 ], [ 4.6, - 3.2, 0.6 ], [ - 5.2, - 10.5, 1.1 ], [ 0.8, - 4.2, 0.45 ] ] ) tryPlace( x, z, r );
  // the banks downstream and round the run
  for ( let k = 0; k < 200 && out.length < 52; k ++ ) {

    const x = ( rng() - 0.5 ) * 44, z = - 20 + rng() * 60;
    const h = valley.height( x, z );
    if ( h < 0.3 || h > 9 ) continue;
    tryPlace( x, z, 0.4 + rng() * rng() * 1.6 );

  }
  return out;

}

/** The boulders' mesh: each a lumpy, flattened stone sunk a third into the ground, with the terrain's
 * `aShade`. */
export function boulderGeometry( valley: Valley, grid: HeightGrid, boulders: readonly Boulder[], sea: number ): THREE.BufferGeometry {

  const rng = mulberry32( 77 );
  const lump = new SimplexNoise( { random: rng } );
  const parts: THREE.BufferGeometry[] = [];
  boulders.forEach( ( b, k ) => {

    const solid = new THREE.IcosahedronGeometry( 1, b.r > 0.8 ? 4 : 3 );
    solid.deleteAttribute( 'normal' );
    solid.deleteAttribute( 'uv' );
    // shared vertices, so the stone shades smooth
    const g = mergeVertices( solid );
    const p = g.getAttribute( 'position' ) as THREE.BufferAttribute;
    const sy = 0.5 + rng() * 0.3, sz = 0.75 + rng() * 0.4, yaw = rng() * Math.PI * 2;
    const v = new THREE.Vector3();
    for ( let i = 0; i < p.count; i ++ ) {

      v.fromBufferAttribute( p, i );
      const n = 1 + 0.22 * lump.noise3d( v.x * 1.1 + k * 7, v.y * 1.1, v.z * 1.1 ) + 0.08 * lump.noise3d( v.x * 3 + k, v.y * 3, v.z * 3 );
      // a flatter underside
      v.multiplyScalar( n );
      if ( v.y < 0 ) v.y *= 0.6;
      p.setXYZ( i, v.x * b.r, v.y * b.r * sy, v.z * b.r * sz );

    }
    g.rotateY( yaw );
    const ground = Math.min( valley.height( b.x, b.z ), valley.height( b.x + b.r * 0.7, b.z ), valley.height( b.x - b.r * 0.7, b.z ), valley.height( b.x, b.z + b.r * 0.7 ), valley.height( b.x, b.z - b.r * 0.7 ) );
    g.translate( b.x, ground + b.r * sy * 0.3, b.z );
    g.computeVertexNormals();
    parts.push( g );

  } );
  const geometry = mergeGeometries( parts );
  const position = geometry.getAttribute( 'position' ) as THREE.BufferAttribute;
  const shade = new Float32Array( position.count * 3 );
  for ( let i = 0; i < position.count; i ++ ) {

    const x = position.getX( i ), y = position.getY( i ), z = position.getZ( i );
    shade[ i * 3 ] = 0.15 + 0.85 * grid.sky( x, Math.max( y, grid.at( x, z ) ) + 0.3, z );
    shade[ i * 3 + 1 ] = y - valley.levelAt( x, z );
    shade[ i * 3 + 2 ] = valley.spray( x, y, z );
    position.setY( i, y + sea );

  }
  geometry.setAttribute( 'aShade', new THREE.BufferAttribute( shade, 3 ) );
  geometry.computeBoundingSphere();
  return geometry;

}

// ---- the material

/** A view-space normal bumped by `height` (m), from its screen derivatives (Mikkelsen's surface
 * gradient). */
function bumped( height: THREE.Node<'float'> ): THREE.Node<'vec3'> {

  const dpdx = dFdx( positionView ), dpdy = dFdy( positionView );
  const r1 = cross( dpdy, normalView ), r2 = cross( normalView, dpdx );
  const det = dot( dpdx, r1 );
  const grad = sign( det ).mul( r1.mul( dFdx( height ) ).add( r2.mul( dFdy( height ) ) ) );
  return normalize( abs( det ).mul( normalView ).sub( grad ) );

}

/** The valley's material: dark basalt on the steep faces, moss where the falls' spray keeps it wet,
 * grass on the gentle slopes above the water, cobbles and gravel at and under the water's edge (not
 * on the boulders: `stones`), wet stone dark and glossy. Its ambient light falls with the sky its
 * vertex sees; the stone's relief is bumped in, fading with distance. */
export function valleyMaterial( render: RenderParts, { stones = false } = {} ): THREE.MeshStandardNodeMaterial {

  const shade = attribute( 'aShade', 'vec3' );
  const sky = shade.x, above = shade.y, spray = shade.z;
  const p = positionWorld;
  const up = normalWorld.y;
  const big = mx_fractal_noise_float( p.mul( 0.11 ), 3, 2, 0.5, 1 );
  const grain = mx_fractal_noise_float( p.mul( 1.7 ), 3, 2, 0.5, 1 );
  // basalt: dark grey, browner in beds, pale lichen on the dry upper faces
  let rock = mix( rgb( 0x2c2b29 ), rgb( 0x4f4b45 ), smoothstep( - 0.5, 0.6, big.add( grain.mul( 0.35 ) ) ) );
  const beds = mx_noise_float( vec3( p.x.mul( 0.04 ), p.y.mul( 0.8 ), p.z.mul( 0.04 ) ) );
  rock = mix( rock, rgb( 0x4b3f33 ), smoothstep( 0.1, 0.6, beds ).mul( 0.4 ) );
  const lichen = smoothstep( 0.2, 0.7, grain.add( big.mul( 0.6 ) ) ).mul( smoothstep( 1.5, 4, above ) ).mul( float( 1 ).sub( spray ) );
  rock = mix( rock, rgb( 0x7d7a6c ), lichen.mul( 0.4 ) );
  // moss: on the spray-wet rock round the falls, and on the ledges
  const mossy = smoothstep( 0.25, 0.85, spray.add( grain.mul( 0.25 ) ).add( big.mul( 0.2 ) ) ).mul( smoothstep( 0.0, 0.6, above ) ).mul( smoothstep( - 0.3, 0.35, up ) );
  const moss = mix( rgb( 0x2b3a16 ), rgb( 0x4c5c26 ), smoothstep( - 0.4, 0.5, grain.add( big ) ) );
  // grass on the gentle slopes above the water, browner in patches
  const grassy = smoothstep( 0.66, 0.84, up.add( grain.mul( 0.08 ) ) ).mul( smoothstep( 0.4, 1.1, above.add( big.mul( 0.4 ) ) ) );
  const grass = mix( mix( rgb( 0x46562a ), rgb( 0x6c7038 ), smoothstep( - 0.4, 0.5, big.mul( 0.8 ).add( grain.mul( 0.4 ) ) ) ), rgb( 0x857a4a ), smoothstep( 0.3, 0.8, big ).mul( 0.55 ) );
  // cobbles at and under the water's edge: round stones about the Worley cells' points (warped, so
  // they come out uneven), each a dome darker to its rim, set in sandy grit
  const warp = mx_noise_vec3( vec3( p.xz.mul( 1.3 ), 2.1 ) ).xy.mul( 0.55 );
  const cells = mx_worley_noise_vec2( p.xz.mul( 3.6 ).add( warp ) );
  const fine = mx_worley_noise_vec2( p.xz.mul( 13 ).add( 5.3 ) );
  const gap = smoothstep( 0.5, 0.3, cells.x ).mul( mix( float( 0.82 ), float( 1 ), smoothstep( 0.0, 0.06, cells.y.sub( cells.x ) ) ) )
    .mul( mix( float( 0.72 ), float( 1.08 ), smoothstep( 0.45, 0.05, cells.x ) ) );
  const grit = mix( rgb( 0x3a352c ), rgb( 0x5e574a ), smoothstep( 0.35, 0.05, fine.x ) );
  const tone = mx_noise_float( p.mul( 4.2 ) );
  let stone = mix( rgb( 0x6a655c ), rgb( 0x8f8576 ), smoothstep( - 0.5, 0.5, tone ) );
  stone = mix( stone, rgb( 0x5a5a5c ), smoothstep( 0.2, 0.7, mx_noise_float( p.mul( 1.1 ).add( 9 ) ) ).mul( 0.6 ) );
  stone = mix( stone, rgb( 0x9b7c55 ), smoothstep( 0.45, 0.8, mx_noise_float( p.mul( 3.7 ).add( 3 ) ) ).mul( 0.5 ) );
  const gravel = mix( grit, stone, gap );
  const bed = stones ? float( 0 ) : smoothstep( 0.35, - 0.05, above.add( grain.mul( 0.2 ) ) ).mul( smoothstep( 0.25, 0.6, up ) );
  // green slime on the stones deeper down
  const slime = smoothstep( - 0.4, - 1.6, above ).mul( 0.5 );
  let colour = rock;
  colour = mix( colour, moss, mossy );
  colour = mix( colour, grass, grassy.mul( float( 1 ).sub( mossy.mul( 0.7 ) ) ) );
  colour = mix( colour, mix( gravel, rgb( 0x3f4a26 ), slime ), bed );
  // wet: a band over the water's edge, and everything the spray reaches
  const wet = max( smoothstep( 0.45, 0.05, above.add( grain.mul( 0.15 ) ) ), smoothstep( 0.35, 0.8, spray ) );
  colour = mix( colour, colour.mul( 0.55 ), wet.mul( float( 1 ).sub( grassy ) ) );
  const roughness = mix( mix( float( 0.9 ).sub( grain.mul( 0.05 ) ), float( 0.97 ), grassy ), float( 0.42 ), wet.mul( float( 1 ).sub( grassy ) ) );
  const material = surface( render, { name: stones ? 'boulders' : 'valley', color: colour, roughness } );
  material.aoNode = mix( float( 0.25 ), float( 1 ), sky );
  // the stone's relief: cracks and knobs on the rock, the cobbles' domes; none on the grass
  const near = smoothstep( 40, 8, length( cameraPosition.sub( p ) ) );
  const knobs = mx_fractal_noise_float( p.mul( 2.3 ), 3, 2.2, 0.5, 1 ).mul( 0.035 ).add( mx_noise_float( p.mul( 9 ) ).mul( 0.008 ) );
  const relief = knobs.mul( float( 1 ).sub( grassy.mul( 0.8 ) ) );
  material.normalNode = bumped( relief.mul( near ) );
  return material;

}

// ---- the render pipeline's fields

/** The fields the water reads, over a square of `half` (m) in `cells` cells a side: the sea-level
 * shore's distance (m, negative over land; a chamfer transform of the wet cells) and the floor's
 * height, in the formats NeutralInputs.ts `neutralFields` gives them in. */
export function valleyFields( valley: Valley, { half, cells, sea, xform }: { half: number; cells: number; sea: number; xform: THREE.Vector4 } ) {

  const cell = 2 * half / cells;
  const floor = new Float32Array( cells * cells );
  const wet = new Uint8Array( cells * cells );
  for ( let j = 0; j < cells; j ++ ) for ( let i = 0; i < cells; i ++ ) {

    const h = valley.height( - half + ( i + 0.5 ) * cell, - half + ( j + 0.5 ) * cell );
    floor[ j * cells + i ] = sea + h;
    wet[ j * cells + i ] = h < 0 ? 1 : 0;

  }
  // the distance to the other side, each way: a two-pass chamfer (3-4) transform
  const distanceTo = ( target: number ) => {

    const d = new Float32Array( cells * cells );
    for ( let k = 0; k < d.length; k ++ ) d[ k ] = wet[ k ] === target ? 0 : 1e9;
    const a = cell, b = cell * Math.SQRT2;
    for ( let j = 0; j < cells; j ++ ) for ( let i = 0; i < cells; i ++ ) {

      const k = j * cells + i;
      if ( i > 0 ) d[ k ] = Math.min( d[ k ], d[ k - 1 ] + a );
      if ( j > 0 ) {

        d[ k ] = Math.min( d[ k ], d[ k - cells ] + a );
        if ( i > 0 ) d[ k ] = Math.min( d[ k ], d[ k - cells - 1 ] + b );
        if ( i < cells - 1 ) d[ k ] = Math.min( d[ k ], d[ k - cells + 1 ] + b );

      }

    }
    for ( let j = cells - 1; j >= 0; j -- ) for ( let i = cells - 1; i >= 0; i -- ) {

      const k = j * cells + i;
      if ( i < cells - 1 ) d[ k ] = Math.min( d[ k ], d[ k + 1 ] + a );
      if ( j < cells - 1 ) {

        d[ k ] = Math.min( d[ k ], d[ k + cells ] + a );
        if ( i < cells - 1 ) d[ k ] = Math.min( d[ k ], d[ k + cells + 1 ] + b );
        if ( i > 0 ) d[ k ] = Math.min( d[ k ], d[ k + cells - 1 ] + b );

      }

    }
    return d;

  };
  const toLand = distanceTo( 0 ), toWater = distanceTo( 1 );
  const shore = new Float32Array( cells * cells );
  for ( let k = 0; k < shore.length; k ++ ) shore[ k ] = wet[ k ] ? toLand[ k ] - cell / 2 : - ( toWater[ k ] - cell / 2 );
  const texture = ( data: Float32Array, name: string ) => {

    const tex = new THREE.DataTexture( data, cells, cells, THREE.RedFormat, THREE.FloatType );
    tex.minFilter = tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    tex.name = name;
    return tex;

  };
  xform.set( - half, - half, 1 / ( 2 * half ), 1 / ( 2 * half ) );
  return {
    textures: { shore: texture( shore, 'shore-distance' ), floor: texture( floor, 'seabed-height' ) },
    shoreData: { data: shore, nx: cells, nz: cells, x0: - half, z0: - half, cell },
    bounds: { x0: - half, z0: - half, x1: half, z1: half },
  };

}
