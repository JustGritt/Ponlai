// The boat: a small wooden fishing boat working the water round the rocks under its engine.
//
// It is built from code, as the wreck is (wreckage.ts), from the same parametric hull: a
// double-ender about 8.4 m long, painted (a cream sheer strake over dark blue topsides, a white boot
// top and red antifouling below the waterline), with a planked deck, a wheelhouse aft, a mast with
// its boom and rigging forward, and its gear: fish boxes, a heap of net with its floats, a winch,
// tyres hung over the side as fenders and a lifebuoy. By night (`night`) the wheelhouse windows
// glow and its navigation lights burn: white at the masthead and astern, red to port, green to
// starboard, and two local lights, a floodlight on the mast and the masthead light, light its deck.
//
// It floats on the sea's real waves as the wreckage does (floaters.ts `launch`): sample points
// over its bottom push up by the water they displace, so it heaves, pitches and rolls on the
// swell; its drag is low along the keel and high across it. Its pilot follows a route round the
// stack (`plotRoute`): it looks a few lengths ahead along the route, turns the boat toward that
// point at a limited rate (a torque, as the rudder's), and holds its speed by the propeller's
// thrust at the stern, easing off in a sharp turn.
//
// The boat's frame: x forward (the bow at +x), y up from the gunwale's height amidships, z to
// starboard.
import * as THREE from 'three/webgpu';
import RAPIER from '@dimforge/rapier3d-compat';
import {
  float, fract, frontFacing, mix, mx_fractal_noise_float, positionLocal, select, smoothstep, step, uniform, uv, vec3,
} from 'three/tsl';
import type { LocalSourceSpec } from './render/lighting/LocalLights.ts';
import type { RenderParts } from './render/index.ts';
import { GLOW } from './lighthouse.ts';
import { albedo, rgb, surface } from './surfaces.ts';
import { group, hullGeometry, hullPoint, hullPoints, merge, ropeGeometry, type HullShape } from './wreckage.ts';
import type { VesselSpec } from './floaters.ts';

const HULL: HullShape = { length: 8.4, beam: 2.7, depth: 1.35, sheer: 0.45, ends: 'boat', jag: 0, holes: 0, seed: 41 };
const L = HULL.length;
// the deck's height, and the waterline's when the boat floats level (the boat's frame)
const DECK = - 0.3;
const WATERLINE = - 0.75;
/** where a person on deck stands to look out (the boat's frame): at the starboard rail forward of the
 * wheelhouse, at a standing eye's height, clear of the boom over the deck's middle */
export const EYE = new THREE.Vector3( - 0.15, DECK + 1.6, 0.95 );
// the physics: kg, the water it displaces with every point under (m^3, so that it floats with the
// points' middles on the waterline), the points' vertical extent, and the drag in its frame
const MASS = 5500;
const VOLUME = MASS / ( 1025 * 0.5 );
const EXTENT = 0.6;
const DRAG = { along: 0.12, across: 2.5, vertical: 2.2 };
// the pilot: its speed (m/s), the most it turns (rad/s), how far it looks ahead (m), the engine's
// most thrust (N) and where the propeller pushes (the boat's frame)
const PILOT = { speed: 2.4, turn: 0.22, ahead: 9, thrust: 4500, propeller: new THREE.Vector3( - 3.7, - 1.05, 0 ) };
// the lateral drag's yaw resistance (N m per rad/s), which the rudder's torque adds ahead of time
const YAW_DRAG = 40000;
/** the navigation lights' radiance at full night (render/lighting/Radiometry.ts units): the white
 * ones', and the coloured ones', lower so that they keep their colour rather than clip to white
 * (the green a third of the red's, so that the two are about as bright) */
const NAV_GLOW = { white: 1.2e-5, colour: 5e-6 };

export interface Boat {
  /** what floaters.ts launches */
  vessel: VesselSpec;
  /** 0 by day, 1 by night */
  night: THREE.UniformNode<'float', number>;
  /** the boat's position (its frame's origin, world), kept up by `update` */
  readonly position: THREE.Vector3;
  /** the engine's throttle, 0 to 1 */
  readonly throttle: number;
  /** its local lights by night (world) */
  lights(): LocalSourceSpec[];
  /** after the physics has moved it: its position and lamps */
  update(): void;
}

// ------------------------------------------------------------------------------ the route

/** A closed route round the stack, `clear` metres or more from every obstacle (`clearance`: how far
 * a point at sea level is from the nearest), as near as it can keep to `radius` from the stack,
 * swinging in and out a little. Found by dynamic programming over the radius at each degree round
 * the stack (the cost: the distance from the preferred radius and each change of radius, a
 * change of at most one step per degree), then smoothed where that keeps it clear, and resampled
 * to points a metre apart. Counter-clockwise seen from above (from +x toward +z). */
export function plotRoute( clearance: ( x: number, z: number ) => number, o = { radius: 30, clear: 9.5 } ): THREE.Vector2[] {

  const N = 360, R0 = 15, R1 = 75, DR = 0.5, M = Math.round( ( R1 - R0 ) / DR ) + 1;
  const at = ( k: number, r: number ) => {

    const a = ( k % N ) / N * Math.PI * 2;
    return [ Math.cos( a ) * r, Math.sin( a ) * r ];

  };
  const ok = new Uint8Array( N * M );
  for ( let k = 0; k < N; k ++ ) for ( let m = 0; m < M; m ++ ) {

    const [ x, z ] = at( k, R0 + m * DR );
    ok[ k * M + m ] = clearance( x, z ) >= o.clear ? 1 : 0;

  }
  const prefer = ( k: number ) => {

    const a = k / N * Math.PI * 2;
    return o.radius + 3 * Math.sin( 2 * a + 0.6 ) + 1.5 * Math.sin( 3 * a + 1.9 );

  };
  // two laps, so the first lap's start is forgotten; the second lap is the route
  const steps = 2 * N;
  const cost = new Float64Array( M ).fill( 0 ), next = new Float64Array( M );
  const from = new Int16Array( steps * M );
  for ( let m = 0; m < M; m ++ ) cost[ m ] = ok[ m ] ? 0 : Infinity;
  for ( let s = 1; s < steps; s ++ ) {

    const k = s % N, want = prefer( k );
    for ( let m = 0; m < M; m ++ ) {

      next[ m ] = Infinity;
      if ( ! ok[ k * M + m ] ) continue;
      const r = R0 + m * DR;
      for ( let d = - 1; d <= 1; d ++ ) {

        const c = cost[ m + d ] ?? Infinity;
        const total = c + ( ( r - want ) / 10 ) ** 2 + Math.abs( d ) * 0.05;
        if ( total < next[ m ] ) {

          next[ m ] = total;
          from[ s * M + m ] = m + d;

        }

      }

    }
    cost.set( next );

  }
  let m = 0;
  for ( let j = 1; j < M; j ++ ) if ( cost[ j ] < cost[ m ] ) m = j;
  const radius = new Float32Array( N );
  for ( let s = steps - 1; s >= N; s -- ) {

    radius[ s % N ] = R0 + m * DR;
    m = from[ s * M + m ];

  }
  // smoothed where the smoothed point stays clear
  for ( let pass = 0; pass < 12; pass ++ ) {

    const before = radius.slice();
    for ( let k = 0; k < N; k ++ ) {

      let sum = 0;
      for ( let d = - 6; d <= 6; d ++ ) sum += before[ ( k + d + N ) % N ];
      const r = sum / 13;
      const [ x, z ] = at( k, r );
      if ( clearance( x, z ) >= o.clear ) radius[ k ] = r;

    }

  }
  // resampled a metre apart
  const raw: THREE.Vector2[] = [];
  for ( let k = 0; k < N; k ++ ) raw.push( new THREE.Vector2( ...at( k, radius[ k ] ) ) );
  const out: THREE.Vector2[] = [ raw[ 0 ].clone() ];
  let carry = 0;
  for ( let k = 0; k < N; k ++ ) {

    const a = raw[ k ], b = raw[ ( k + 1 ) % N ], length = a.distanceTo( b );
    let s = 1 - carry;
    while ( s <= length ) {

      out.push( a.clone().lerp( b, s / length ) );
      s += 1;

    }
    carry = length - ( s - 1 );

  }
  if ( out[ out.length - 1 ].distanceTo( out[ 0 ] ) < 0.5 ) out.pop();
  return out;

}

// ------------------------------------------------------------------------------ the boat

/** Where the deck meets the planking at u along the hull: its half width there. */
function deckHalf( u: number ): number {

  let lo = 0, hi = 1;
  const p = new THREE.Vector3();
  if ( hullPoint( HULL, u, 1, p ).y < DECK ) return 0;
  for ( let i = 0; i < 30; i ++ ) {

    const t = ( lo + hi ) / 2;
    if ( hullPoint( HULL, u, t, p ).y < DECK ) lo = t;
    else hi = t;

  }
  return hullPoint( HULL, u, ( lo + hi ) / 2, p ).z;

}

/** The gunwale's half width and height at x (the boat's frame). */
function gunwaleAt( x: number ): THREE.Vector3 {

  return hullPoint( HULL, THREE.MathUtils.clamp( x / L + 0.5, 0, 1 ), 1 ).setX( x );

}

/** A box of size sx, sy, sz centred at x, y, z. */
function box( sx: number, sy: number, sz: number, x: number, y: number, z: number ): THREE.BufferGeometry {

  return new THREE.BoxGeometry( sx, sy, sz ).translate( x, y, z );

}

/** A round rod of radius r from a to b. */
function rod( a: THREE.Vector3, b: THREE.Vector3, r: number, r1 = r ): THREE.BufferGeometry {

  const d = b.clone().sub( a );
  const g = new THREE.CylinderGeometry( r1, r, d.length(), 10 );
  g.translate( 0, d.length() / 2, 0 );
  g.applyQuaternion( new THREE.Quaternion().setFromUnitVectors( new THREE.Vector3( 0, 1, 0 ), d.normalize() ) );
  return g.translate( a.x, a.y, a.z );

}

/** The planked deck: a strip between the bulwarks at the deck's height, facing up. */
function deckGeometry(): THREE.BufferGeometry {

  const n = 60, positions: number[] = [], uvs: number[] = [], index: number[] = [];
  for ( let i = 0; i <= n; i ++ ) {

    const u = 0.004 + 0.992 * i / n, x = ( u - 0.5 ) * L, w = deckHalf( u ) * 0.985;
    positions.push( x, DECK, - w, x, DECK, w );
    uvs.push( x, - w, x, w );

  }
  for ( let i = 0; i < n; i ++ ) {

    const a = i * 2, b = a + 2;
    index.push( a, a + 1, b, a + 1, b + 1, b );

  }
  const g = new THREE.BufferGeometry();
  g.setAttribute( 'position', new THREE.Float32BufferAttribute( positions, 3 ) );
  g.setAttribute( 'uv', new THREE.Float32BufferAttribute( uvs, 2 ) );
  g.setIndex( index );
  g.computeVertexNormals();
  return g;

}

/** A heap of net: a flattened, lumpy dome. */
function netGeometry( radius: number, height: number, seed: number ): THREE.BufferGeometry {

  const g = new THREE.SphereGeometry( radius, 18, 10, 0, Math.PI * 2, 0, Math.PI / 2 );
  const p = g.attributes.position as THREE.BufferAttribute;
  for ( let i = 0; i < p.count; i ++ ) {

    const x = p.getX( i ), y = p.getY( i ), z = p.getZ( i );
    const lump = 1 + 0.16 * Math.sin( x * 9 + seed ) * Math.sin( z * 8 + seed * 2 ) + 0.08 * Math.sin( x * 23 + z * 17 );
    p.setXYZ( i, x * lump, y / radius * height * lump, z * lump );

  }
  g.computeVertexNormals();
  return g;

}

export function createBoat( render: RenderParts, sea: number, route: THREE.Vector2[], startAngle: number ): Boat {

  const night = uniform( 0 );
  const glow = ( r: number, g: number, b: number, k: number ) => vec3( r, g, b ).mul( night.mul( k ) );

  // ---- the hull's paint: u along it, v round it from the gunwale (m); the waterline by height
  const u = uv().x, v = uv().y, y = positionLocal.y;
  const seam = float( 1 ).sub( smoothstep( 0, 0.1, fract( v.div( 0.16 ) ) ) );
  const wear = mx_fractal_noise_float( vec3( u.mul( 1.4 ), v.mul( 3.5 ), 5 ), 3, 2, 0.5, 1 );
  const bottom = float( 1 ).sub( smoothstep( WATERLINE + 0.03, WATERLINE + 0.05, y ) );
  const bootTop = smoothstep( WATERLINE + 0.03, WATERLINE + 0.05, y ).mul( float( 1 ).sub( smoothstep( WATERLINE + 0.11, WATERLINE + 0.13, y ) ) );
  const sheer = float( 1 ).sub( smoothstep( 0.15, 0.17, v ) );
  let outside = mix( rgb( 0x17314a ), rgb( 0xe2ddcf ), sheer );
  outside = mix( outside, rgb( 0xe8e4d8 ), bootTop );
  outside = mix( outside, rgb( 0x7c2b1f ), bottom );
  // scuffed to the wood along the strakes, darker in the laps
  outside = mix( outside, rgb( 0x6a5843 ), smoothstep( 0.42, 0.6, wear ).mul( 0.7 ).mul( float( 1 ).sub( bottom ) ) );
  outside = outside.mul( float( 1 ).sub( seam.mul( 0.28 ) ) );
  // wet and darker toward the water
  const wet = smoothstep( WATERLINE + 0.35, WATERLINE + 0.05, y );
  outside = mix( outside, outside.mul( 0.7 ), wet );
  const inside = rgb( 0xcfcabb ).mul( float( 1 ).sub( seam.mul( 0.15 ) ) );
  const hullPaint = surface( render, {
    name: 'boat hull', color: select( frontFacing, outside, inside ) as unknown as THREE.Node<'vec3'>, side: THREE.DoubleSide,
    roughness: mix( mix( float( 0.62 ), float( 0.82 ), bottom ), float( 0.3 ), wet.mul( float( 1 ).sub( bottom ) ) ),
  } );

  // ---- the deck: weathered planks along the boat, butts staggered
  const dx = uv().x, dz = uv().y;
  const row = dz.div( 0.11 ).floor();
  const deckSeam = float( 1 ).sub( smoothstep( 0, 0.08, fract( dz.div( 0.11 ) ) ) )
    .max( float( 1 ).sub( smoothstep( 0, 0.01, fract( dx.div( 2.6 ).add( fract( row.mul( 0.618 ) ) ) ) ) ) );
  const deckGrain = mx_fractal_noise_float( vec3( dx.mul( 0.7 ), dz.mul( 12 ), row ), 3, 2, 0.5, 1 );
  const deck = surface( render, {
    name: 'boat deck', color: mix( rgb( 0x6c604f ), rgb( 0x9a8b72 ), smoothstep( - 0.6, 0.6, deckGrain ) ).mul( float( 1 ).sub( deckSeam.mul( 0.45 ) ) ),
    roughness: 0.85,
  } );

  const white = surface( render, { name: 'boat paint', color: albedo( 0xe3dfd3 ), roughness: 0.6 } );
  const trim = surface( render, { name: 'boat trim', color: albedo( 0x1c3a55 ), roughness: 0.55 } );
  const roof = surface( render, { name: 'wheelhouse roof', color: albedo( 0xb3b0a6 ), roughness: 0.7 } );
  const varnish = surface( render, {
    name: 'varnished timber',
    color: mix( rgb( 0x5a3d22 ), rgb( 0x8f6a42 ), smoothstep( - 0.5, 0.5, mx_fractal_noise_float( positionLocal.mul( vec3( 9, 1.2, 9 ) ), 3, 2, 0.5, 1 ) ) ),
    roughness: 0.45,
  } );
  const glass = surface( render, { name: 'wheelhouse glass', color: albedo( 0x15191c ), roughness: 0.1, emissive: glow( 1, 0.52, 0.2, GLOW.windows.value * 0.8 ) } );
  const iron = surface( render, { name: 'boat iron', color: albedo( 0x26282a ), roughness: 0.5, metalness: 0.6 } );
  const rubber = surface( render, { name: 'tyre', color: albedo( 0x141414 ), roughness: 0.9 } );
  const rope = surface( render, { name: 'boat rope', color: albedo( 0x8a7656 ), roughness: 0.95 } );
  const net = surface( render, { name: 'net', color: albedo( 0x2c4a3c ), roughness: 0.95 } );
  const orange = surface( render, { name: 'floats', color: albedo( 0xd2551d ), roughness: 0.5 } );
  const boxes = surface( render, { name: 'fish boxes', color: albedo( 0x2b4f66 ), roughness: 0.7 } );
  // the lamps: a coloured lens by day, burning by night
  const lamp = ( name: string, lens: number, r: number, g: number, b: number, k: number ) => surface( render, { name, color: albedo( lens ), roughness: 0.2, emissive: glow( r, g, b, k ) } );
  const masthead = lamp( 'white light', 0xcfcabd, 1, 0.9, 0.75, NAV_GLOW.white );
  const port = lamp( 'port light', 0x5a1010, 1, 0, 0, NAV_GLOW.colour );
  const starboard = lamp( 'starboard light', 0x0e4a22, 0, 1, 0.06, NAV_GLOW.colour * 0.3 );

  // ---- the hull and its ribs and rails, centred along x
  const { shell, frames } = hullGeometry( HULL );
  shell.translate( - L / 2, 0, 0 );
  frames.translate( - L / 2, 0, 0 );

  // ---- the wheelhouse, aft: x from WH.x0 to WH.x1, its walls to WH.top
  const WH = { x0: - 2.9, x1: - 1.2, half: 0.85, top: 1.3 };
  const whMid = ( WH.x0 + WH.x1 ) / 2, whLen = WH.x1 - WH.x0;
  const walls = box( whLen, WH.top - DECK, WH.half * 2, whMid, ( WH.top + DECK ) / 2, 0 );
  const roofTop = merge( [ box( whLen + 0.24, 0.07, WH.half * 2 + 0.2, whMid, WH.top + 0.035, 0 ), box( whLen * 0.6, 0.05, WH.half * 1.6, whMid - 0.1, WH.top + 0.095, 0 ) ] );
  const roofEdge = merge( [
    box( 0.05, 0.12, WH.half * 2 + 0.22, WH.x1 + 0.12, WH.top, 0 ), box( 0.05, 0.12, WH.half * 2 + 0.22, WH.x0 - 0.12, WH.top, 0 ),
    box( whLen + 0.24, 0.12, 0.05, whMid, WH.top, WH.half + 0.1 ), box( whLen + 0.24, 0.12, 0.05, whMid, WH.top, - WH.half - 0.1 ),
    // a band of trim round the walls' foot
    box( whLen + 0.02, 0.16, WH.half * 2 + 0.02, whMid, DECK + 0.08, 0 ),
  ] );
  const panes = merge( [
    ...[ - 0.52, 0, 0.52 ].map( ( z ) => box( 0.03, 0.46, 0.44, WH.x1 + 0.01, 0.9, z ) ),
    ...[ - 1.65, - 2.4 ].flatMap( ( x ) => [ box( 0.48, 0.42, 0.03, x, 0.9, WH.half + 0.01 ), box( 0.48, 0.42, 0.03, x, 0.9, - WH.half - 0.01 ) ] ),
    box( 0.03, 0.4, 0.4, WH.x0 - 0.01, 0.9, - 0.42 ),
  ] );
  const door = box( 0.04, 1.35, 0.6, WH.x0 - 0.02, DECK + 0.75, 0.38 );
  const stack = merge( [ rod( new THREE.Vector3( - 2.6, WH.top, - 0.55 ), new THREE.Vector3( - 2.6, WH.top + 0.95, - 0.55 ), 0.065 ), box( 0.2, 0.04, 0.2, - 2.6, WH.top + 0.97, - 0.55 ) ] );
  const lifebuoy = new THREE.TorusGeometry( 0.24, 0.055, 8, 20 ).rotateY( Math.PI / 2 ).translate( WH.x0 - 0.07, 0.75, - 0.38 );

  // ---- the mast forward, its crosstree and boom, and the rigging
  const MAST = new THREE.Vector3( 1.3, DECK, 0 ), TOP = new THREE.Vector3( 1.3, 5.3, 0 );
  const BOOM = { heel: new THREE.Vector3( 1.22, 0.95, 0 ), end: new THREE.Vector3( - 0.7, 1.75, 0 ) };
  const spars = merge( [
    rod( MAST, TOP, 0.09, 0.055 ),
    rod( new THREE.Vector3( 1.3, 4.2, - 0.5 ), new THREE.Vector3( 1.3, 4.2, 0.5 ), 0.035 ),
    rod( BOOM.heel, BOOM.end, 0.06, 0.045 ),
  ] );
  const stem = gunwaleAt( L / 2 - 0.15 ), shroudAt = gunwaleAt( 1.0 );
  const hook = BOOM.end.clone().setY( 0.6 );
  const rigging = merge( [
    ropeGeometry( [ TOP.clone().setY( 5.15 ), stem.clone().setZ( 0 ).setY( stem.y + 0.05 ) ], 0.012 ),
    ...[ - 1, 1 ].flatMap( ( s ) => [
      ropeGeometry( [ new THREE.Vector3( 1.3, 4.2, 0.5 * s ), shroudAt.clone().setZ( shroudAt.z * s * 0.98 ) ], 0.012 ),
      ropeGeometry( [ TOP.clone().setY( 4.8 ), new THREE.Vector3( 1.3, 4.2, 0.5 * s ) ], 0.01 ),
    ] ),
    // the topping lift and the fall from the boom's end with its hook
    ropeGeometry( [ TOP.clone().setY( 5.0 ), BOOM.end ], 0.012 ),
    ropeGeometry( [ BOOM.end, hook ], 0.014 ),
  ] );
  const hookIron = new THREE.TorusGeometry( 0.06, 0.015, 5, 10, Math.PI * 1.4 ).translate( hook.x, hook.y - 0.06, 0 );

  // ---- the gear on deck: a hatch, a winch, fish boxes, the heap of net aft with its floats
  const hatch = merge( [ box( 0.95, 0.22, 0.95, 0.15, DECK + 0.11, 0 ), box( 1.0, 0.04, 1.0, 0.15, DECK + 0.24, 0 ) ] );
  const winch = merge( [
    rod( new THREE.Vector3( 0.95, DECK + 0.35, - 0.35 ), new THREE.Vector3( 0.95, DECK + 0.35, 0.35 ), 0.13 ),
    box( 0.3, 0.42, 0.05, 0.95, DECK + 0.21, - 0.4 ), box( 0.3, 0.42, 0.05, 0.95, DECK + 0.21, 0.4 ),
  ] );
  const fishBoxes = merge( [
    box( 0.56, 0.27, 0.38, 2.25, DECK + 0.135, - 0.35 ), box( 0.56, 0.27, 0.38, 2.25, DECK + 0.405, - 0.35 ),
    box( 0.56, 0.27, 0.38, 2.28, DECK + 0.675, - 0.33 ), box( 0.56, 0.27, 0.38, 2.22, DECK + 0.135, 0.12 ),
    box( 0.56, 0.27, 0.38, 2.85, DECK + 0.135, - 0.2 ),
  ] );
  const netHeap = netGeometry( 0.62, 0.42, 3 ).translate( - 3.45, DECK, 0 );
  const floats: THREE.BufferGeometry[] = [];
  for ( let k = 0; k < 7; k ++ ) {

    const a = k * 2.4, r = 0.15 + ( k % 3 ) * 0.14;
    floats.push( new THREE.SphereGeometry( 0.075, 10, 8 ).translate( - 3.45 + Math.cos( a ) * r, DECK + 0.34 + 0.05 * Math.sin( k * 3 ), Math.sin( a ) * r ) );

  }

  // ---- tyres hung over each side as fenders, on short ropes from the gunwale
  const tyres: THREE.BufferGeometry[] = [], lanyards: THREE.BufferGeometry[] = [];
  for ( const x of [ - 1.6, 0.35, 2.1 ] ) for ( const s of [ - 1, 1 ] ) {

    const g = gunwaleAt( x );
    const centre = new THREE.Vector3( x, g.y - 0.42, s * ( g.z + 0.1 ) );
    tyres.push( new THREE.TorusGeometry( 0.2, 0.075, 8, 16 ).rotateX( s * 0.12 ).translate( centre.x, centre.y, centre.z ) );
    lanyards.push( ropeGeometry( [ new THREE.Vector3( x, g.y + 0.04, s * g.z ), centre.clone().setY( centre.y + 0.2 ) ], 0.012 ) );

  }

  // ---- the lights: the masthead's, the sidelights on the wheelhouse roof, the stern light
  const LIGHTS = {
    masthead: new THREE.Vector3( 1.3, 5.38, 0 ), stern: new THREE.Vector3( WH.x0 - 0.1, WH.top + 0.2, 0 ),
    port: new THREE.Vector3( WH.x1 - 0.08, WH.top + 0.16, - WH.half - 0.05 ), starboard: new THREE.Vector3( WH.x1 - 0.08, WH.top + 0.16, WH.half + 0.05 ),
  };
  const lampAt = ( p: THREE.Vector3, r = 0.06 ) => new THREE.SphereGeometry( r, 10, 8 ).translate( p.x, p.y, p.z );
  const boards = merge( [
    box( 0.22, 0.16, 0.02, LIGHTS.port.x - 0.02, LIGHTS.port.y, LIGHTS.port.z + 0.07 ),
    box( 0.22, 0.16, 0.02, LIGHTS.starboard.x - 0.02, LIGHTS.starboard.y, LIGHTS.starboard.z - 0.07 ),
    rod( new THREE.Vector3( LIGHTS.stern.x, WH.top, 0 ), LIGHTS.stern.clone().setY( LIGHTS.stern.y - 0.05 ), 0.02 ),
  ] );

  const object = group(
    [ shell, hullPaint ], [ frames, varnish ], [ deckGeometry(), deck ],
    [ walls, white ], [ merge( [ roofTop ] ), roof ], [ merge( [ roofEdge, door, boards ] ), trim ], [ panes, glass ],
    [ merge( [ stack, winch, hookIron ] ), iron ], [ merge( [ spars, hatch ] ), varnish ], [ merge( [ rigging, ...lanyards ] ), rope ],
    [ fishBoxes, boxes ], [ netHeap, net ], [ merge( [ ...floats, lifebuoy ] ), orange ], [ merge( tyres ), rubber ],
    [ lampAt( LIGHTS.masthead, 0.07 ), masthead ], [ lampAt( LIGHTS.stern ), masthead ], [ lampAt( LIGHTS.port ), port ], [ lampAt( LIGHTS.starboard ), starboard ],
  );

  // ---- the physics: points over its bottom, where the hull is full (the ends rise out of the water)
  const points: THREE.Vector3[] = [];
  for ( const uu of [ 0.2, 0.35, 0.5, 0.65, 0.8 ] ) {

    const half = hullPoint( HULL, uu, 1 ).z;
    for ( const s of [ - 0.5, 0, 0.5 ] ) points.push( new THREE.Vector3( ( uu - 0.5 ) * L, WATERLINE, s * half ) );

  }
  const outline = hullPoints( shell );

  // ---- the pilot
  const n = route.length;
  let index = 0;
  for ( let k = 1; k < n; k ++ ) {

    const a = Math.atan2( route[ k ].y, route[ k ].x ), b = Math.atan2( route[ index ].y, route[ index ].x );
    if ( Math.abs( Math.atan2( Math.sin( a - startAngle ), Math.cos( a - startAngle ) ) ) < Math.abs( Math.atan2( Math.sin( b - startAngle ), Math.cos( b - startAngle ) ) ) ) index = k;

  }
  const startDir = route[ ( index + 1 ) % n ].clone().sub( route[ index ] );
  const q = new THREE.Quaternion(), forward = new THREE.Vector3(), thrustAt = new THREE.Vector3(), force = { x: 0, y: 0, z: 0 };
  let throttle = 0;
  const steer = ( rigid: RAPIER.RigidBody ) => {

    const t = rigid.translation(), r = rigid.rotation();
    q.set( r.x, r.y, r.z, r.w );
    forward.set( 1, 0, 0 ).applyQuaternion( q );
    // the nearest point of the route a little ahead of the last
    let best = index, bestD = Infinity;
    for ( let k = 0; k < 40; k ++ ) {

      const i = ( index + k ) % n, d = Math.hypot( route[ i ].x - t.x, route[ i ].y - t.z );
      if ( d < bestD ) {

        bestD = d;
        best = i;

      }

    }
    index = best;
    const look = route[ ( index + PILOT.ahead ) % n ];
    // the heading wanted and the heading now (radians about y: x turns toward -z)
    const want = Math.atan2( - ( look.y - t.z ), look.x - t.x ), now = Math.atan2( - forward.z, forward.x );
    const error = Math.atan2( Math.sin( want - now ), Math.cos( want - now ) );
    const rate = THREE.MathUtils.clamp( error * 1.1, - PILOT.turn, PILOT.turn );
    const spin = rigid.angvel().y;
    rigid.addTorque( { x: 0, y: 4 * 35700 * ( rate - spin ) + YAW_DRAG * rate, z: 0 }, true );
    // the speed: eased off in a sharp turn, and off altogether if the boat is knocked over
    const v = rigid.linvel();
    const ahead = v.x * forward.x + v.y * forward.y + v.z * forward.z;
    const target = PILOT.speed * ( 1 - 0.4 * Math.min( Math.abs( error ) / 0.6, 1 ) );
    const upright = THREE.MathUtils.smoothstep( new THREE.Vector3( 0, 1, 0 ).applyQuaternion( q ).y, 0.5, 0.8 );
    const thrust = THREE.MathUtils.clamp( MASS * 1.5 * ( target - ahead ), - 0.3 * PILOT.thrust, PILOT.thrust ) * upright;
    throttle = Math.max( 0, thrust ) / PILOT.thrust;
    thrustAt.copy( PILOT.propeller ).applyQuaternion( q ).add( t as THREE.Vector3Like );
    force.x = forward.x * thrust;
    force.y = forward.y * thrust;
    force.z = forward.z * thrust;
    rigid.addForceAtPoint( force, thrustAt, true );

  };

  const vessel: VesselSpec = {
    name: 'fishing boat', object, points, mass: MASS,
    // the engine and the ballast low in the hull
    centreOfMass: new THREE.Vector3( 0, WATERLINE - 0.18, 0 ),
    // a solid's of its size: roll (with the water it carries round), yaw, pitch
    inertia: new THREE.Vector3( 6500, 35700, 33500 ),
    volume: VOLUME, extent: EXTENT, drag: DRAG,
    colliders: () => [ RAPIER.ColliderDesc.convexHull( outline )! ],
    start: new THREE.Vector3( route[ index ].x, sea - WATERLINE, route[ index ].y ),
    heading: Math.atan2( - startDir.y, startDir.x ),
    steer,
  };

  const position = new THREE.Vector3();
  // a floodlight on the mast over the working deck, and the masthead light
  const lamps = { deck: new THREE.Vector3( 0.9, 2.7, 0 ), masthead: LIGHTS.masthead.clone() };
  const deckLight = new THREE.Vector3(), mastLight = new THREE.Vector3();
  return {
    vessel, night, position,
    get throttle() {

      return throttle;

    },
    lights: () => [
      { position: deckLight, intensity: new THREE.Vector3( 2.2e-6, 1.7e-6, 1.1e-6 ), range: 8 },
      { position: mastLight, intensity: new THREE.Vector3( 1.2e-6, 1.1e-6, 1e-6 ), range: 12 },
    ],
    update() {

      object.updateMatrixWorld();
      position.setFromMatrixPosition( object.matrixWorld );
      deckLight.copy( lamps.deck ).applyMatrix4( object.matrixWorld );
      mastLight.copy( lamps.masthead ).applyMatrix4( object.matrixWorld );

    },
  };

}
