// The fish: schools of mackerel near the surface round the rocks, that scatter from the boat and
// now and then leap clear of the water.
//
// The mesh and its material are Ponlai's fish (life/FishGeometry.ts, life/FishMaterial.ts, from its
// island's flying fish) given a mackerel's anatomy and skin: slender, two dorsal fins, a deeply
// forked tail, blue-green with dark wavy bars over the back, silver on the flank. Drawn as one
// instanced mesh in the opaque pass only, marked as a swimmer (render/ocean/Refraction.ts
// markSwimmer) and moved to where the eye sees it through the waves (render/ocean/Water.ts
// swimmerSway): the water shows a fish from that pass's image, and reads the floor behind it from
// the refraction image, which leaves it out. It casts no shadow.
//
// How they move: per frame on the CPU, as Ponlai's flying fish do (and tidewater's schooling fish
// before them): separation, alignment and cohesion within the school, and a goal the school swims
// to in long runs, kept within its water, between 3 and 28 m from the rocks and from 0.35 to 1.6 m
// under the surface. A fish the boat comes near turns away from it and dives. Every so often (an
// exponential wait, longer by night) a few fish of a school rise fast and leap: out of the water
// in a short arc and back in, each with its splash.
//
// History: the scene clock's step drives them (none while paused), from a seeded stream, so a
// paused page shows the schools where they started.
import * as THREE from 'three/webgpu';
import { markSwimmer, shared, swimmerSway, type RenderParts } from './render/index.ts';
import { fishGeometry, wingRoot, type Anatomy } from './life/FishGeometry.ts';
import { makeFishMaterial, type Skin } from './life/FishMaterial.ts';
import { mulberry32 } from './wreckage.ts';

const MACKEREL: Anatomy = {
  body: 0.8, sec: 2.2,
  top: [ [ 0, 0.004 ], [ 0.03, 0.02 ], [ 0.08, 0.04 ], [ 0.18, 0.058 ], [ 0.35, 0.066 ], [ 0.55, 0.058 ], [ 0.75, 0.036 ], [ 0.9, 0.018 ], [ 1, 0.013 ] ],
  bot: [ [ 0, 0.004 ], [ 0.03, 0.02 ], [ 0.08, 0.038 ], [ 0.18, 0.054 ], [ 0.35, 0.06 ], [ 0.55, 0.052 ], [ 0.75, 0.032 ], [ 0.9, 0.016 ], [ 1, 0.012 ] ],
  wid: [ [ 0, 0.004 ], [ 0.04, 0.022 ], [ 0.12, 0.04 ], [ 0.3, 0.048 ], [ 0.5, 0.042 ], [ 0.7, 0.028 ], [ 0.9, 0.014 ], [ 1, 0.01 ] ],
  dorsal: [
    { from: 0.3, to: 0.44, rays: 10, h: [ [ 0, 0.05 ], [ 1, 0.012 ] ], rake: [ 0.5, 0.9 ], notch: 0.01 },
    { from: 0.6, to: 0.68, rays: 8, h: [ [ 0, 0.032 ], [ 1, 0.015 ] ], rake: [ 0.6, 0.9 ], notch: 0.01 },
  ],
  anal: [ { from: 0.62, to: 0.7, rays: 8, h: [ [ 0, 0.028 ], [ 1, 0.014 ] ], rake: [ 0.6, 0.9 ], notch: 0.01 } ],
  // small paired fins, carried half folded
  pectoral: { u: 0.22, y: 0.005, rays: 10, lead: 0.9, last: 0.3, len: 0.12, tail: 0.5, dihedral: 0.15 },
  pelvic: { u: 0.3, y: - 0.045, rays: 5, lead: 0.7, last: 0.3, len: 0.07, tail: 0.6, dihedral: 0.05 },
  caudal: { len: 0.2, span: 0.15, fork: 0.25, lower: 1.0, rays: 13 },
};

const SKIN: Skin = {
  back: 0x1f4a4a, marks: 0x0a1716, flank: 0xbcc8cb, belly: 0xf1efe8, head: 0x244a4c, snout: 0x4d6a6c,
  fin: 0x55666a, finEdge: 0x2a3438, iris: 0xc8c0a0,
  headEnd: 0.2, snoutEnd: 0.03, headOver: 'back',
  eye: { u: 0.07, y: 0.015, r: 0.022 },
};

// the water they keep to (m): from the rocks, from the stack, and the band under the surface
const SHORE = [ 3, 28 ] as const, REACH = 75, BAND = [ - 1.6, - 0.35 ] as const;
const LENGTH = [ 0.38, 0.48 ] as const;
const SCHOOLS = 3, PER_SCHOOL = [ 22, 32 ] as const, CAPACITY = SCHOOLS * PER_SCHOOL[ 1 ];
// steering, in body lengths and per second (Ponlai's flying fish's, tidewater's school mode)
const STEER = { cruise: 2.4, max: 4.5, accel: 3, sep: 1.6, nbr: 5, wSep: 5, wAli: 1.8, wCoh: 1.0, wGoal: 0.8 };
// the leaps: the mean wait between a school's bursts by day and by night (s), the fish in a burst,
// the speed out of the water: up and along (m/s)
const LEAP = { day: 12, night: 40, fish: [ 1, 4 ], up: [ 3.2, 4.4 ], along: [ 1.8, 3.0 ] } as const;
// how near the boat (m) a fish turns away from it
const SHY = 8;
const G = 9.81, TAU = Math.PI * 2;
const SWIM = 0, RISE = 1, AIR = 2;

interface School {
  offset: number; count: number;
  home: THREE.Vector3; goal: THREE.Vector3; centre: THREE.Vector3; heading: THREE.Vector3;
  timer: number; burst: number;
}

export interface Fish {
  mesh: THREE.Mesh;
  /** steps the schools by the scene clock's `dt`; `night` 0 by day to 1 by night */
  update( dt: number, night: number ): void;
  /** the fish in the air now, and the leaps so far */
  readonly leaping: number;
  readonly leaps: number;
  /** each school's middle (world) */
  centres(): THREE.Vector3[];
  /** where the fish in the air are (world), for the tools */
  airborne(): THREE.Vector3[];
}

export function createFish( render: RenderParts, o: {
  sea: number;
  /** how far x, z at sea level is from the nearest rock's waterline */
  shore: ( x: number, z: number ) => number;
  /** what the fish shy from (the boat), live */
  threat: THREE.Vector3;
  /** a fish broke the surface at x, y, z (out or in), for the sound */
  onSplash?: ( x: number, y: number, z: number ) => void;
} ): Fish {

  const base = fishGeometry( MACKEREL, 0 );
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = base.index;
  for ( const name of [ 'position', 'normal', 'aData' ] ) geometry.setAttribute( name, base.getAttribute( name ) );
  const attr = () => new THREE.InstancedBufferAttribute( new Float32Array( CAPACITY * 4 ), 4 );
  const attrs = { iPose: attr(), iTurn: attr(), iMotion: attr(), iBefore: attr() };
  for ( const [ name, a ] of Object.entries( attrs ) ) geometry.setAttribute( name, a );
  const roots = { pectoral: wingRoot( MACKEREL, MACKEREL.pectoral ), pelvic: wingRoot( MACKEREL, MACKEREL.pelvic ) };
  const material = makeFishMaterial( MACKEREL, SKIN, shared, render.waveMaps, roots, ( p ) => swimmerSway( render.fft, p, shared.waterLevel ) );
  material.name = 'mackerel';
  markSwimmer( material );
  const mesh = new THREE.Mesh( geometry, material );
  mesh.name = 'mackerel';
  mesh.frustumCulled = false;

  const rng = mulberry32( 4242 );
  const fits = ( x: number, z: number ) => {

    const d = o.shore( x, z );
    return d >= SHORE[ 0 ] && d <= SHORE[ 1 ] && Math.hypot( x, z ) <= REACH;

  };
  // per fish: position (y under the mean sea), velocity, heading, the previous position and phase
  const pos = new Float32Array( CAPACITY * 3 ), vel = new Float32Array( CAPACITY * 3 ), head = new Float32Array( CAPACITY * 3 );
  const prev = new Float32Array( CAPACITY * 3 ), size = new Float32Array( CAPACITY ), roll = new Float32Array( CAPACITY );
  const bend = new Float32Array( CAPACITY ), phase = new Float32Array( CAPACITY ), prevPhase = new Float32Array( CAPACITY );
  const amp = new Float32Array( CAPACITY ), speedMul = new Float32Array( CAPACITY );
  const state = new Uint8Array( CAPACITY ), delay = new Float32Array( CAPACITY ), launch = new Float32Array( CAPACITY * 2 );
  const acc = new Float32Array( CAPACITY * 9 ), nbrs = new Uint16Array( CAPACITY );
  let leaps = 0;

  const wait = ( night: number ) => - Math.log( 1 - rng() * 0.999 ) * ( LEAP.day + ( LEAP.night - LEAP.day ) * night );
  const between = ( [ a, b ]: readonly [ number, number ] ) => a + rng() * ( b - a );

  // the schools' homes: in their water, apart, on the side of the stack the first view looks across
  const schools: School[] = [];
  let n = 0;
  for ( let tries = 0; schools.length < SCHOOLS && tries < 4000; tries ++ ) {

    const a = - 0.6 + rng() * 2.6 + ( tries > 2000 ? rng() * TAU : 0 ), r = 14 + rng() * 40;
    const x = Math.cos( a ) * r, z = Math.sin( a ) * r;
    const d = o.shore( x, z );
    if ( d < 5 || d > 14 || schools.some( ( s ) => Math.hypot( s.home.x - x, s.home.z - z ) < 22 ) ) continue;
    const home = new THREE.Vector3( x, ( BAND[ 0 ] + BAND[ 1 ] ) / 2, z );
    const h = rng() * TAU;
    const count = PER_SCHOOL[ 0 ] + Math.floor( rng() * ( PER_SCHOOL[ 1 ] - PER_SCHOOL[ 0 ] + 1 ) );
    const school: School = {
      offset: n, count, home, goal: home.clone(), centre: home.clone(), heading: new THREE.Vector3( Math.cos( h ), 0, Math.sin( h ) ), timer: 0, burst: wait( 0 ),
    };
    schools.push( school );
    for ( let k = 0; k < count; k ++ ) spawn( school, n + k );
    n += count;
    retarget( school );

  }
  geometry.instanceCount = n;

  // a fish of the school at a free spot near its home, swimming along the school's heading
  function spawn( g: School, i: number ): void {

    size[ i ] = between( LENGTH );
    speedMul[ i ] = 0.88 + rng() * 0.24;
    phase[ i ] = rng() * TAU;
    amp[ i ] = 0.08;
    const h = g.heading;
    let x = g.home.x, z = g.home.z;
    for ( let t = 0; t < 10; t ++ ) {

      const along = ( rng() - 0.5 ) * 5, across = ( rng() - 0.5 ) * 3;
      const tx = g.home.x + h.x * along - h.z * across, tz = g.home.z + h.z * along + h.x * across;
      if ( fits( tx, tz ) ) {

        x = tx;
        z = tz;
        break;

      }

    }
    const y = BAND[ 0 ] + ( BAND[ 1 ] - BAND[ 0 ] ) * rng();
    const s = STEER.cruise * size[ i ] * speedMul[ i ], d = Math.atan2( h.z, h.x ) + ( rng() - 0.5 ) * 0.3;
    pos.set( [ x, y, z ], i * 3 );
    vel.set( [ Math.cos( d ) * s, 0, Math.sin( d ) * s ], i * 3 );
    head.set( [ Math.cos( d ), 0, Math.sin( d ) ], i * 3 );
    prev.set( pos.subarray( i * 3, i * 3 + 3 ), i * 3 );
    prevPhase[ i ] = phase[ i ];

  }

  // the next goal: a long run ahead within the water, or back toward home
  function retarget( g: School ): void {

    for ( let k = 0; k < 24; k ++ ) {

      const a = Math.atan2( g.heading.z, g.heading.x ) + ( rng() - 0.5 ) * ( k < 12 ? 1.6 : 4 );
      const d = 12 + rng() * 14;
      const x = g.centre.x + Math.cos( a ) * d, z = g.centre.z + Math.sin( a ) * d;
      if ( ! fits( x, z ) || ! fits( ( x + g.centre.x ) / 2, ( z + g.centre.z ) / 2 ) || Math.hypot( x - g.home.x, z - g.home.z ) > 30 ) continue;
      g.goal.set( x, BAND[ 0 ] + ( BAND[ 1 ] - BAND[ 0 ] ) * rng(), z );
      g.timer = 10 + rng() * 10;
      return;

    }
    g.goal.copy( g.home );
    g.timer = 8;

  }

  function step( g: School, dt: number, night: number ): void {

    const o9 = g.offset, count = g.count, P = pos, V = vel, A = acc, C = nbrs;
    let cx = 0, cy = 0, cz = 0, swimming = 0;
    for ( let k = 0; k < count; k ++ ) {

      if ( state[ o9 + k ] !== SWIM ) continue;
      const i = ( o9 + k ) * 3;
      cx += P[ i ]; cy += P[ i + 1 ]; cz += P[ i + 2 ];
      swimming ++;

    }
    if ( swimming > 0 ) g.centre.set( cx / swimming, cy / swimming, cz / swimming );
    g.timer -= dt;
    const gx = g.goal.x - g.centre.x, gz = g.goal.z - g.centre.z;
    // the boat near the school: a goal away from it
    const tx = g.centre.x - o.threat.x, tz = g.centre.z - o.threat.z, td = Math.hypot( tx, tz );
    if ( td < SHY * 1.6 && g.timer > 1 ) {

      const away = new THREE.Vector3( g.centre.x + tx / td * 14, BAND[ 0 ], g.centre.z + tz / td * 14 );
      if ( fits( away.x, away.z ) ) {

        g.goal.copy( away );
        g.timer = 1;

      }

    }
    if ( g.timer <= 0 || gx * gx + gz * gz < 4 ) retarget( g );

    // a burst: a few swimming fish, each after its own short delay, rise and leap
    g.burst -= dt;
    if ( g.burst <= 0 ) {

      g.burst = wait( night );
      let take = Math.round( between( LEAP.fish ) );
      for ( let k = 0; k < count && take > 0; k ++ ) {

        const i = o9 + Math.floor( rng() * count );
        if ( state[ i ] !== SWIM ) continue;
        state[ i ] = RISE;
        delay[ i ] = rng() * 1.2;
        launch[ i * 2 ] = between( LEAP.up );
        launch[ i * 2 + 1 ] = between( LEAP.along );
        take --;

      }

    }

    // neighbours within the school (swimming fish only)
    const Lmax = LENGTH[ 1 ];
    const nbr2 = ( STEER.nbr * Lmax ) ** 2, sepR = STEER.sep * Lmax, sep2 = sepR * sepR;
    A.fill( 0, o9 * 9, ( o9 + count ) * 9 );
    C.fill( 0, o9, o9 + count );
    for ( let a = 0; a < count; a ++ ) {

      const i = o9 + a;
      if ( state[ i ] !== SWIM ) continue;
      const ix = P[ i * 3 ], iy = P[ i * 3 + 1 ], iz = P[ i * 3 + 2 ];
      for ( let b = a + 1; b < count; b ++ ) {

        const j = o9 + b;
        if ( state[ j ] !== SWIM ) continue;
        const dx = P[ j * 3 ] - ix, dy = P[ j * 3 + 1 ] - iy, dz = P[ j * 3 + 2 ] - iz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if ( d2 > nbr2 ) continue;
        C[ i ] ++; C[ j ] ++;
        const ai = i * 9, aj = j * 9;
        A[ ai + 3 ] += V[ j * 3 ]; A[ ai + 4 ] += V[ j * 3 + 1 ]; A[ ai + 5 ] += V[ j * 3 + 2 ];
        A[ aj + 3 ] += V[ i * 3 ]; A[ aj + 4 ] += V[ i * 3 + 1 ]; A[ aj + 5 ] += V[ i * 3 + 2 ];
        A[ ai + 6 ] += dx; A[ ai + 7 ] += dy; A[ ai + 8 ] += dz;
        A[ aj + 6 ] -= dx; A[ aj + 7 ] -= dy; A[ aj + 8 ] -= dz;
        if ( d2 < sep2 ) {

          const d = Math.sqrt( d2 ) + 1e-4, f = ( sepR - d ) / ( sepR * d );
          A[ ai ] -= dx * f; A[ ai + 1 ] -= dy * f; A[ ai + 2 ] -= dz * f;
          A[ aj ] += dx * f; A[ aj + 1 ] += dy * f; A[ aj + 2 ] += dz * f;

        }

      }

    }

    for ( let a = 0; a < count; a ++ ) {

      const i = o9 + a, i3 = i * 3, ai = i * 9;
      const L = size[ i ];
      let x = P[ i3 ], y = P[ i3 + 1 ], z = P[ i3 + 2 ];
      let vx = V[ i3 ], vy = V[ i3 + 1 ], vz = V[ i3 + 2 ];

      if ( state[ i ] === AIR ) {

        // a ballistic arc; back in, it dives
        vy -= G * dt;
        x += vx * dt; y += vy * dt; z += vz * dt;
        if ( y < 0 && vy < 0 ) {

          state[ i ] = SWIM;
          o.onSplash?.( x, o.sea, z );
          vx *= 0.5; vz *= 0.5; vy = - 1.4;

        }
        P[ i3 ] = x; P[ i3 + 1 ] = y; P[ i3 + 2 ] = z;
        V[ i3 ] = vx; V[ i3 + 1 ] = vy; V[ i3 + 2 ] = vz;
        continue;

      }

      const cruise = STEER.cruise * L * speedMul[ i ];
      let ax = A[ ai ] * STEER.wSep, ay = A[ ai + 1 ] * STEER.wSep, az = A[ ai + 2 ] * STEER.wSep;
      const c = C[ i ];
      if ( c > 0 ) {

        const inv = 1 / c;
        ax += ( A[ ai + 3 ] * inv - vx ) * STEER.wAli + A[ ai + 6 ] * inv * STEER.wCoh;
        ay += ( A[ ai + 4 ] * inv - vy ) * STEER.wAli + A[ ai + 7 ] * inv * STEER.wCoh;
        az += ( A[ ai + 5 ] * inv - vz ) * STEER.wAli + A[ ai + 8 ] * inv * STEER.wCoh;

      }
      // the goal
      const dx = g.goal.x - x, dz = g.goal.z - z, dy = g.goal.y - y, dl = Math.hypot( dx, dy, dz ) + 1e-6;
      ax += ( dx / dl * cruise - vx ) * STEER.wGoal;
      ay += ( dy / dl * cruise - vy ) * STEER.wGoal;
      az += ( dz / dl * cruise - vz ) * STEER.wGoal;
      // out of the water ahead (a rock, too far out): turn back toward home
      if ( ! fits( x + vx * 1.5, z + vz * 1.5 ) ) {

        ax -= vx * 3;
        az -= vz * 3;
        const hx = g.home.x - x, hz = g.home.z - z, hl = Math.hypot( hx, hz ) + 1e-6;
        ax += hx / hl * cruise * 2;
        az += hz / hl * cruise * 2;

      }
      // the boat: away from it, and down
      let amax = STEER.accel * L * 4, vmax = STEER.max * L;
      const bx = x - o.threat.x, bz = z - o.threat.z, bd = Math.hypot( bx, bz );
      if ( bd < SHY ) {

        const fear = 1 - bd / SHY;
        ax += bx / ( bd + 1e-6 ) * 14 * fear;
        az += bz / ( bd + 1e-6 ) * 14 * fear;
        ay -= 4 * fear;
        amax *= 1 + 2 * fear;
        vmax *= 1 + fear;

      }
      if ( state[ i ] === RISE ) delay[ i ] -= dt;
      const rising = state[ i ] === RISE && delay[ i ] <= 0;
      if ( rising ) {

        // up to the surface, fast, then out
        const vh = Math.hypot( vx, vz ) + 1e-6;
        ax += ( vx / vh * 3 - vx ) * 3;
        az += ( vz / vh * 3 - vz ) * 3;
        ay += 10;
        amax = 30;
        vmax = 6;

      } else {

        // the band, and level swimming
        if ( y > BAND[ 1 ] ) ay -= ( y - BAND[ 1 ] ) * 8;
        if ( y < BAND[ 0 ] ) ay += ( BAND[ 0 ] - y ) * 8;
        ay -= vy * 1.5;

      }
      const al = Math.hypot( ax, ay, az );
      if ( al > amax ) {

        ax *= amax / al; ay *= amax / al; az *= amax / al;

      }
      vx += ax * dt; vy += ay * dt; vz += az * dt;
      const speed = Math.hypot( vx, vy, vz ) + 1e-6, vmin = cruise * 0.3;
      const sc = speed > vmax ? vmax / speed : speed < vmin ? vmin / speed : 1;
      vx *= sc; vy *= sc; vz *= sc;
      x += vx * dt; y += vy * dt; z += vz * dt;
      if ( rising && y > - 0.12 ) {

        // out of the water, along its course
        state[ i ] = AIR;
        leaps ++;
        const vh = Math.hypot( vx, vz ) + 1e-6, along = launch[ i * 2 + 1 ];
        vx = vx / vh * along;
        vz = vz / vh * along;
        vy = launch[ i * 2 ];
        o.onSplash?.( x, o.sea, z );

      } else if ( ! rising ) y = Math.min( y, - 0.25 );
      P[ i3 ] = x; P[ i3 + 1 ] = y; P[ i3 + 2 ] = z;
      V[ i3 ] = vx; V[ i3 + 1 ] = vy; V[ i3 + 2 ] = vz;

    }

    const hx = g.goal.x - g.centre.x, hz = g.goal.z - g.centre.z, hl = Math.hypot( hx, hz );
    if ( hl > 0.5 ) g.heading.set( hx / hl, 0, hz / hl );

  }

  const q = new THREE.Quaternion(), euler = new THREE.Euler( 0, 0, 0, 'YXZ' );
  // turns each fish to its course, beats its tail, and writes the instances (`dt` 0: the pose only)
  const write = ( dt: number ) => {

    const pose = attrs.iPose.array as Float32Array, turn = attrs.iTurn.array as Float32Array;
    const motion = attrs.iMotion.array as Float32Array, before = attrs.iBefore.array as Float32Array;
    const kHead = 1 - Math.exp( - dt * 6 ), kRoll = 1 - Math.exp( - dt * 3 );
    for ( let i = 0; i < n; i ++ ) {

      const i3 = i * 3, i4 = i * 4, L = size[ i ], air = state[ i ] === AIR;
      const vx = vel[ i3 ], vy = vel[ i3 + 1 ], vz = vel[ i3 + 2 ];
      const speed = Math.hypot( vx, vy, vz ) + 1e-6;
      if ( dt > 0 ) {

        // the heading follows the velocity (at once in the air, its nose along the arc)
        const kh = air ? 1 : kHead * Math.min( 1, speed / ( L * 0.5 ) + 0.15 );
        const ohx = head[ i3 ], ohz = head[ i3 + 2 ];
        let hx = ohx + ( vx / speed - ohx ) * kh, hy = head[ i3 + 1 ] + ( vy / speed - head[ i3 + 1 ] ) * kh, hz = ohz + ( vz / speed - ohz ) * kh;
        const pitch = air ? 0.95 : 0.45;
        hy = Math.max( - pitch, Math.min( pitch, hy ) );
        const hl = Math.hypot( hx, hy, hz ) || 1;
        head[ i3 ] = hx / hl; head[ i3 + 1 ] = hy / hl; head[ i3 + 2 ] = hz / hl;
        const yawRate = ( ohz * hx - ohx * hz ) / dt;
        roll[ i ] += ( Math.max( - 0.5, Math.min( 0.5, yawRate * 0.12 ) ) - roll[ i ] ) * kRoll;
        bend[ i ] += ( Math.max( - 0.25, Math.min( 0.25, yawRate * 0.06 ) ) - bend[ i ] ) * kRoll;
        // the tail: its beat grows with the speed; a hard flick in the air
        const bl = speed / L;
        const freq = air ? 9 : Math.min( 10, 1.6 + 0.9 * bl );
        phase[ i ] = ( phase[ i ] + dt * TAU * freq ) % ( TAU * 64 );
        const a = air ? 0.16 : 0.08 * ( 0.55 + 0.45 * Math.min( 2.5, bl / STEER.cruise ) );
        amp[ i ] += ( a - amp[ i ] ) * ( 1 - Math.exp( - dt * 8 ) );

      }
      euler.set( - Math.asin( head[ i3 + 1 ] ), Math.atan2( head[ i3 ], head[ i3 + 2 ] ), roll[ i ] );
      q.setFromEuler( euler );
      pose[ i4 ] = pos[ i3 ]; pose[ i4 + 1 ] = o.sea + pos[ i3 + 1 ]; pose[ i4 + 2 ] = pos[ i3 + 2 ]; pose[ i4 + 3 ] = L;
      turn[ i4 ] = q.x; turn[ i4 + 1 ] = q.y; turn[ i4 + 2 ] = q.z; turn[ i4 + 3 ] = q.w;
      // the paired fins half folded along the flank
      motion[ i4 ] = phase[ i ]; motion[ i4 + 1 ] = amp[ i ]; motion[ i4 + 2 ] = bend[ i ]; motion[ i4 + 3 ] = 0.6;
      before[ i4 ] = prev[ i3 ]; before[ i4 + 1 ] = o.sea + prev[ i3 + 1 ]; before[ i4 + 2 ] = prev[ i3 + 2 ]; before[ i4 + 3 ] = prevPhase[ i ];

    }
    for ( const a of Object.values( attrs ) ) a.needsUpdate = true;

  };
  write( 0 );

  return {
    mesh,
    get leaping() {

      let k = 0;
      for ( let i = 0; i < n; i ++ ) if ( state[ i ] === AIR ) k ++;
      return k;

    },
    get leaps() {

      return leaps;

    },
    centres: () => schools.map( ( g ) => g.centre.clone().setY( o.sea + g.centre.y ) ),
    airborne() {

      const out: THREE.Vector3[] = [];
      for ( let i = 0; i < n; i ++ ) if ( state[ i ] === AIR ) out.push( new THREE.Vector3( pos[ i * 3 ], o.sea + pos[ i * 3 + 1 ], pos[ i * 3 + 2 ] ) );
      return out;

    },
    update( dt, night ) {

      prev.set( pos );
      prevPhase.set( phase );
      if ( dt > 0 ) {

        const steps = Math.min( 3, Math.ceil( dt * 30 ) ), h = Math.min( dt, 0.1 ) / steps;
        for ( let s = 0; s < steps; s ++ ) for ( const g of schools ) step( g, h, night );

      }
      write( dt );

    },
  };

}
