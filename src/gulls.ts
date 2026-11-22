// The gulls: a flock of herring gulls that wheel round the lighthouse and the stacks, follow the
// fishing boat, and settle on the rocks.
//
// One instanced mesh. The bird is built from code in its own frame (+z forward, +y up, the wings
// along x): a lathed body, the head and its yellow bill, a fanned tail, and each wing as a sheet in
// two parts, the arm from the shoulder and the hand from the wrist, pale grey above with black tips
// and a white trailing edge. Its vertex stage flaps the wings (the arm turns about the shoulder,
// the hand about the wrist, lagging it) and holds a gliding gull's crank, the arm raised and the
// hand drooped; and folds them, each wing vertex moving toward where it lies on a resting gull's
// back (`aFolded`), the legs coming down as it does. It then turns and places the bird by its
// instance, as Ponlai's island does its fish (packages/island/life). It casts no shadow.
//
// The flight, on the CPU: each gull flying holds to a circle (an orbit: round the lantern, over a
// stack, or over the boat, which moves) by steering along the circle's tangent, bending in toward
// it from outside and out from inside, its turn rate limited and its bank the turn's; it climbs
// or sinks toward its orbit's height and keeps above the rocks and the tower ahead of it. It
// glides, and flaps in short bouts, more when it climbs or flies slowly. Every so often it takes
// another orbit. Some stand on the rocks, on the ledges and tops where the whitewash is
// (`perches`), and take off after a while; as night falls all of them come down to the rocks and
// stand there until morning. A landing glides down to its perch, slowing and flapping hard over the
// last metres; a take-off climbs away flapping.
//
// History: the scene clock's step drives it (none while paused), from a seeded stream, so a paused
// page shows the flock where it started.
import * as THREE from 'three/webgpu';
import {
  Fn, attribute, cos, cross, float, frontFacing, mix, normalize, normalLocal, positionGeometry, normalGeometry, positionPrevious, select, sin, vec3,
} from 'three/tsl';
import { TEMPORAL, type RenderParts } from './render/index.ts';
import { albedo, surface } from './surfaces.ts';
import { mulberry32 } from './wreckage.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

const COUNT = 16;
const G = 9.81, TAU = Math.PI * 2;
// the wing's joints (the bird's frame, m): the shoulder and the wrist, and the half span
const SHOULDER = { x: 0.055, y: 0.025 };
const WRIST = 0.33, TIP = 0.72;
// a gliding gull's wings: the arm raised (rad), the hand drooped from it
const DIHEDRAL = 0.14, CRANK = - 0.3;
// flight: the cruise speed (m/s), the most it turns (rad/s), climb and sink rates (m/s)
const FLIGHT = { speed: 9, turn: 1.3, climb: 2.6, sink: 2.2 };
// how high its feet are below its origin when it stands
const LEGS = 0.2;

const FLY = 0, LAND = 1, PERCHED = 2, TAKEOFF = 3;

/** A circle the gulls wheel round: its centre (live: the boat's moves), the height above the
 * centre they keep, the radius, and how often it is chosen. */
export interface Orbit {
  centre: THREE.Vector3;
  lift: readonly [ number, number ];
  radius: readonly [ number, number ];
  weight: number;
  /** the speed it is flown at (m/s): slower over the boat, hanging behind it */
  speed?: number;
}

export interface Gulls {
  mesh: THREE.Mesh;
  /** steps the flock by the scene clock's `dt`; `night` 0 by day to 1 by night */
  update( dt: number, night: number ): void;
  /** a gull to call from: its position, live (the sound follows it); null when none is about */
  caller( random: () => number ): THREE.Vector3 | null;
  /** how many stand on the rocks */
  readonly perched: number;
}

interface Gull {
  position: THREE.Vector3;
  before: THREE.Vector3;
  mode: number;
  /** heading (rad about y, 0 along +z), speed, vertical speed, bank and pitch */
  yaw: number; speed: number; climb: number; bank: number; pitch: number;
  /** the wingbeat's phase, its amplitude, the fold (0 spread .. 1 folded) */
  phase: number; beforePhase: number; amp: number; fold: number;
  flapping: boolean; flapFor: number;
  orbit: Orbit; radius: number; lift: number; turn: 1 | - 1;
  /** time left on this orbit, standing, or before a landing at nightfall */
  timer: number;
  perch: THREE.Vector3 | null;
  scale: number;
}

// ------------------------------------------------------------------------------ the bird's mesh

class Builder {

  position: number[] = [];
  normal: number[] = [];
  color: number[] = [];
  shade: number[] = [];
  folded: number[] = [];

  /** a three geometry's triangles, in one colour, the part and side given, folding to themselves */
  add( g: THREE.BufferGeometry, hex: number, part: number, side = 0, fold?: ( p: THREE.Vector3 ) => THREE.Vector3 ): void {

    const flat = g.index ? g.toNonIndexed() : g;
    if ( ! flat.attributes.normal ) flat.computeVertexNormals();
    const p = flat.attributes.position, n = flat.attributes.normal;
    const c = albedo( hex ), v = new THREE.Vector3();
    for ( let i = 0; i < p.count; i ++ ) {

      v.fromBufferAttribute( p, i );
      this.position.push( v.x, v.y, v.z );
      this.normal.push( n.getX( i ), n.getY( i ), n.getZ( i ) );
      this.color.push( c.r, c.g, c.b );
      this.shade.push( part, side, 0, 0 );
      const f = fold ? fold( v.clone() ) : v;
      this.folded.push( f.x, f.y, f.z );

    }

  }

  build(): THREE.BufferGeometry {

    const g = new THREE.BufferGeometry();
    g.setAttribute( 'position', new THREE.Float32BufferAttribute( this.position, 3 ) );
    g.setAttribute( 'normal', new THREE.Float32BufferAttribute( this.normal, 3 ) );
    g.setAttribute( 'aColor', new THREE.Float32BufferAttribute( this.color, 3 ) );
    g.setAttribute( 'aShade', new THREE.Float32BufferAttribute( this.shade, 4 ) );
    g.setAttribute( 'aFolded', new THREE.Float32BufferAttribute( this.folded, 3 ) );
    g.computeBoundingSphere();
    return g;

  }

}

const lerp = ( a: number, b: number, t: number ) => a + ( b - a ) * t;
// the wing's planform along its half span x: the leading edge's z and the chord
const PLAN: [ number, number, number ][] = [ [ SHOULDER.x, 0.13, 0.21 ], [ WRIST, 0.12, 0.19 ], [ 0.5, 0.06, 0.15 ], [ 0.62, 0.0, 0.1 ], [ TIP, - 0.07, 0.035 ] ];
function plan( x: number ): [ number, number ] {

  for ( let i = 1; i < PLAN.length; i ++ ) {

    if ( x <= PLAN[ i ][ 0 ] ) {

      const a = PLAN[ i - 1 ], b = PLAN[ i ], t = ( x - a[ 0 ] ) / ( b[ 0 ] - a[ 0 ] );
      return [ lerp( a[ 1 ], b[ 1 ], t ), lerp( a[ 2 ], b[ 2 ], t ) ];

    }

  }
  return [ PLAN[ PLAN.length - 1 ][ 1 ], PLAN[ PLAN.length - 1 ][ 2 ] ];

}

const GREY = 0x9aa3a9, WHITE = 0xf1f1ec, BLACK = 0x1c1c1e;

function gullGeometry(): THREE.BufferGeometry {

  const B = new Builder();
  // ---- the body, lathed along z and a little flattened; grey on the back (the mantle)
  const profile = [ [ 0.001, - 0.24 ], [ 0.014, - 0.22 ], [ 0.036, - 0.15 ], [ 0.062, - 0.05 ], [ 0.075, 0.04 ], [ 0.07, 0.11 ], [ 0.048, 0.165 ], [ 0.04, 0.19 ], [ 0.001, 0.2 ] ]
    .map( ( [ r, z ] ) => new THREE.Vector2( r, z ) );
  const body = new THREE.LatheGeometry( profile, 14 ).rotateX( Math.PI / 2 ).scale( 1, 0.85, 1 );
  // a lathe about z after the turn: split its triangles into the mantle and the rest by height
  const flat = body.toNonIndexed();
  const p = flat.attributes.position, n = flat.attributes.normal;
  const mantle = { p: [] as number[], n: [] as number[] }, rest = { p: [] as number[], n: [] as number[] };
  for ( let i = 0; i < p.count; i += 3 ) {

    const y = ( p.getY( i ) + p.getY( i + 1 ) + p.getY( i + 2 ) ) / 3, z = ( p.getZ( i ) + p.getZ( i + 1 ) + p.getZ( i + 2 ) ) / 3;
    const into = y > 0.03 && z < 0.12 && z > - 0.17 ? mantle : rest;
    for ( let k = 0; k < 3; k ++ ) {

      into.p.push( p.getX( i + k ), p.getY( i + k ), p.getZ( i + k ) );
      into.n.push( n.getX( i + k ), n.getY( i + k ), n.getZ( i + k ) );

    }

  }
  const part = ( list: { p: number[]; n: number[] } ) => {

    const g = new THREE.BufferGeometry();
    g.setAttribute( 'position', new THREE.Float32BufferAttribute( list.p, 3 ) );
    g.setAttribute( 'normal', new THREE.Float32BufferAttribute( list.n, 3 ) );
    return g;

  };
  B.add( part( mantle ), GREY, 0 );
  B.add( part( rest ), WHITE, 0 );
  // ---- the head, its bill and eyes
  B.add( new THREE.SphereGeometry( 0.048, 12, 8 ).translate( 0, 0.03, 0.2 ), WHITE, 0 );
  B.add( new THREE.ConeGeometry( 0.013, 0.07, 8 ).rotateX( Math.PI / 2 ).translate( 0, 0.022, 0.275 ), 0xe2b93a, 0 );
  for ( const s of [ - 1, 1 ] ) B.add( new THREE.SphereGeometry( 0.008, 6, 4 ).translate( s * 0.036, 0.045, 0.222 ), 0x16140f, 0 );
  // ---- the tail: a white fan
  const tail = new THREE.BufferGeometry();
  tail.setAttribute( 'position', new THREE.Float32BufferAttribute( [
    - 0.03, 0.006, - 0.15, 0.03, 0.006, - 0.15, 0.065, 0.004, - 0.3, - 0.03, 0.006, - 0.15, 0.065, 0.004, - 0.3, - 0.065, 0.004, - 0.3,
  ], 3 ) );
  B.add( tail, WHITE, 0 );
  // ---- the legs: tucked into the body in flight, down to the perch when folded
  for ( const s of [ - 1, 1 ] ) {

    const hip = new THREE.Vector3( s * 0.028, - 0.045, 0.0 );
    const leg = new THREE.CylinderGeometry( 0.007, 0.006, LEGS - 0.05, 5 ).translate( 0, - ( LEGS - 0.05 ) / 2, 0 );
    const foot = new THREE.ConeGeometry( 0.025, 0.06, 3 ).rotateX( Math.PI / 2 ).scale( 1, 0.15, 1 ).translate( 0, - ( LEGS - 0.045 ), 0.02 );
    for ( const g of [ leg, foot ] ) {

      g.translate( hip.x, hip.y, hip.z );
      // tucked: drawn up into the body at the hip
      B.add( g, 0xd49a8c, 3, s, ( v ) => v.clone() );

    }

  }
  // the legs' rest position is the tucked one: their vertices drawn up to the hip, the extended
  // pose kept as the fold's target
  {

    const n = B.position.length / 3;
    for ( let i = 0; i < n; i ++ ) {

      if ( B.shade[ i * 4 ] !== 3 ) continue;
      B.position[ i * 3 + 1 ] = - 0.04;
      B.position[ i * 3 + 2 ] = - 0.02;

    }

  }
  // ---- the wings: arm (part 1) and hand (part 2), a grid over span and chord, cambered
  const spans = { 1: [ SHOULDER.x, 0.12, 0.2, 0.27, WRIST ], 2: [ WRIST, 0.4, 0.47, 0.54, 0.6, 0.66, TIP ] } as const;
  const chords = [ 0, 0.25, 0.5, 0.8, 1 ];
  for ( const side of [ - 1, 1 ] ) for ( const wing of [ 1, 2 ] as const ) {

    const xs = spans[ wing ];
    const vertex = ( x: number, c: number ) => {

      const [ lead, chord ] = plan( x );
      const s = ( x - SHOULDER.x ) / ( TIP - SHOULDER.x );
      const at = new THREE.Vector3( side * x, SHOULDER.y + 0.014 * Math.sin( Math.PI * c ) * ( 1 - 0.6 * s ), lead - chord * c );
      // folded along the back: the span laid back along the body, narrowing to the tips crossed
      // over the tail, the chord down the upper flank
      const narrow = 1 - 0.75 * s * s;
      const fold = new THREE.Vector3( side * ( 0.012 + ( 0.058 + 0.008 * c ) * narrow ), 0.04 - 0.04 * c * narrow - 0.012 * s, 0.07 - s * 0.41 - 0.025 * c * narrow );
      // black at the tip, a white trailing edge inboard, grey elsewhere
      const tip = x > 0.5 ? 1 : 0;
      const hex = tip ? BLACK : c >= 0.8 ? WHITE : GREY;
      return { at, fold, hex, tip };

    };
    for ( let i = 0; i < xs.length - 1; i ++ ) for ( let j = 0; j < chords.length - 1; j ++ ) {

      const a = vertex( xs[ i ], chords[ j ] ), b = vertex( xs[ i + 1 ], chords[ j ] );
      const c = vertex( xs[ i + 1 ], chords[ j + 1 ] ), d = vertex( xs[ i ], chords[ j + 1 ] );
      // wound so the top faces up on either wing
      const quad = side > 0 ? [ a, b, c, a, c, d ] : [ a, c, b, a, d, c ];
      for ( const q of quad ) {

        const colour = albedo( q.tip ? BLACK : q.hex );
        B.position.push( q.at.x, q.at.y, q.at.z );
        B.normal.push( 0, 1, 0 );
        B.color.push( colour.r, colour.g, colour.b );
        B.shade.push( wing, side, q.tip, 0 );
        B.folded.push( q.fold.x, q.fold.y, q.fold.z );

      }

    }

  }
  return B.build();

}

// ------------------------------------------------------------------------------ the material

// rotates v by the unit quaternion q
const rotateQ = ( q: V4, v: V3 ): V3 => v.add( cross( q.xyz, cross( q.xyz, v ).add( v.mul( q.w ) ) ).mul( 2 ) );
// turns v about the z axis by a
const turnZ = ( v: V3, a: F ): V3 => vec3( v.x.mul( cos( a ) ).sub( v.y.mul( sin( a ) ) ), v.x.mul( sin( a ) ).add( v.y.mul( cos( a ) ) ), v.z );

function gullMaterial( render: RenderParts ): THREE.MeshStandardNodeMaterial {

  const aColor = attribute( 'aColor', 'vec3' ), shade = attribute( 'aShade', 'vec4' ), aFolded = attribute( 'aFolded', 'vec3' );
  const pose = attribute( 'iPose', 'vec4' ), turn = attribute( 'iTurn', 'vec4' ), motion = attribute( 'iMotion', 'vec4' ), before = attribute( 'iBefore', 'vec4' );
  const part = shade.x, side = shade.y;
  const isArm = part.equal( 1 ), isHand = part.equal( 2 ), isWing = isArm.or( isHand );

  // the bird at a phase of its wingbeat: motion ( phase, amplitude, fold, - )
  const posed = ( phase: F ) => {

    const amp = motion.y, fold = motion.z;
    const arm = float( DIHEDRAL ).add( amp.mul( sin( phase ) ) ).mul( side );
    const hand = float( CRANK ).add( amp.mul( 0.7 ).mul( sin( phase.sub( 0.7 ) ) ) ).mul( side );
    const shoulder = vec3( side.mul( SHOULDER.x ), SHOULDER.y, 0 ), wrist = vec3( side.mul( WRIST ), SHOULDER.y, 0 );
    const p = positionGeometry, n = normalGeometry;
    // the hand turns about the wrist, then the whole wing about the shoulder
    const fromWrist = select( isHand, turnZ( p.sub( wrist ), hand ).add( wrist ), p );
    const flapped = select( isWing, turnZ( fromWrist.sub( shoulder ), arm ).add( shoulder ), p );
    const flappedN = select( isWing, turnZ( select( isHand, turnZ( n, hand ), n ), arm ), n );
    // folding: toward the resting pose; the body bobs against the wingbeat
    const foldedN = select( isWing, normalize( vec3( side.mul( 0.8 ), 0.6, 0 ) ), n );
    const bob = vec3( 0, amp.mul( sin( phase ) ).mul( - 0.02 ), 0 );
    return { p: mix( flapped, aFolded, fold ).add( bob ), n: normalize( mix( flappedN, foldedN, fold ) ) };

  };
  const place = ( p: V3, at: V3 ) => rotateQ( turn, p.mul( pose.w ) ).add( at );

  // the wings' undersides are paler, the tips stay dark
  const under = select( shade.z.greaterThan( 0.5 ), aColor.mul( 1.6 ).add( 0.03 ), mix( aColor, vec3( 0.78, 0.8, 0.8 ), 0.7 ) );
  const colour = select( isWing.and( frontFacing.not() ), under, aColor ) as unknown as V3;
  const material = surface( render, { name: 'gull', color: colour, roughness: 0.75, side: THREE.DoubleSide } );
  material.positionNode = Fn( () => {

    const now = posed( motion.x );
    normalLocal.assign( rotateQ( turn, now.n ) );
    if ( TEMPORAL ) positionPrevious.assign( place( posed( before.w ).p, before.xyz ) );
    return place( now.p, pose.xyz );

  } )();
  if ( TEMPORAL ) material.userData.previousPosition = true;
  return material;

}

// ------------------------------------------------------------------------------ the flock

/** Perches on the rocks: points of their surfaces that face up, near their tops (so a gull comes down
 * to one from above), above the spray, `spacing` apart. */
export function findPerches( geometries: THREE.BufferGeometry[], sea: number, o = { above: 1.8, nearTop: 2.5, beyond: 14, spacing: 1.6, max: 48 } ): THREE.Vector3[] {

  const random = mulberry32( 77 );
  const candidates: THREE.Vector3[] = [];
  const v = new THREE.Vector3();
  for ( const g of geometries ) {

    const p = g.attributes.position, n = g.attributes.normal;
    g.computeBoundingBox();
    const top = g.boundingBox!.max.y;
    for ( let i = 0; i < p.count; i ++ ) {

      if ( n.getY( i ) < 0.86 ) continue;
      v.fromBufferAttribute( p, i );
      if ( v.y < sea + o.above || v.y < top - o.nearTop || Math.hypot( v.x, v.z ) < o.beyond ) continue;
      candidates.push( v.clone() );

    }

  }
  // shuffled, then taken where none taken lies within the spacing
  for ( let i = candidates.length - 1; i > 0; i -- ) {

    const j = Math.floor( random() * ( i + 1 ) );
    [ candidates[ i ], candidates[ j ] ] = [ candidates[ j ], candidates[ i ] ];

  }
  const out: THREE.Vector3[] = [];
  for ( const c of candidates ) {

    if ( out.length >= o.max ) break;
    if ( out.every( ( q ) => q.distanceTo( c ) >= o.spacing ) ) out.push( c );

  }
  return out;

}

export function createGulls( render: RenderParts, o: {
  sea: number;
  orbits: Orbit[];
  perches: THREE.Vector3[];
  /** the least height (world) a gull may fly at over x, z: the rocks' and the tower's tops */
  ceiling: ( x: number, z: number ) => number;
} ): Gulls {

  const base = gullGeometry();
  const geometry = new THREE.InstancedBufferGeometry();
  for ( const name of [ 'position', 'normal', 'aColor', 'aShade', 'aFolded' ] ) geometry.setAttribute( name, base.getAttribute( name ) );
  const attr = () => new THREE.InstancedBufferAttribute( new Float32Array( COUNT * 4 ), 4 );
  const attrs = { iPose: attr(), iTurn: attr(), iMotion: attr(), iBefore: attr() };
  for ( const [ name, a ] of Object.entries( attrs ) ) geometry.setAttribute( name, a );
  geometry.instanceCount = COUNT;
  const mesh = new THREE.Mesh( geometry, gullMaterial( render ) );
  mesh.name = 'gulls';
  mesh.frustumCulled = false;
  // no shadow: the wings are single sheets, which would shadow themselves in speckles, and a gull's
  // shadow is lost at this size
  mesh.castShadow = false;

  const random = mulberry32( 1907 );
  const range = ( [ a, b ]: readonly [ number, number ] ) => a + random() * ( b - a );
  const taken = new Set<THREE.Vector3>();
  const totalWeight = o.orbits.reduce( ( s, orbit ) => s + orbit.weight, 0 );

  const chooseOrbit = ( gull: Gull ) => {

    let w = random() * totalWeight;
    gull.orbit = o.orbits.find( ( orbit ) => ( w -= orbit.weight ) <= 0 ) ?? o.orbits[ 0 ];
    gull.radius = range( gull.orbit.radius );
    gull.lift = range( gull.orbit.lift );
    gull.turn = random() < 0.5 ? 1 : - 1;
    gull.timer = 18 + random() * 40;

  };
  const freePerch = ( near: THREE.Vector3 ) => {

    let best: THREE.Vector3 | null = null, bestD = Infinity;
    for ( const perch of o.perches ) {

      if ( taken.has( perch ) ) continue;
      // the nearest free ones, with some chance in the choice
      const d = perch.distanceTo( near ) * ( 0.6 + random() * 0.8 );
      if ( d < bestD ) {

        bestD = d;
        best = perch;

      }

    }
    return best;

  };

  const gulls: Gull[] = [];
  for ( let i = 0; i < COUNT; i ++ ) {

    const gull: Gull = {
      position: new THREE.Vector3(), before: new THREE.Vector3(), mode: FLY,
      yaw: random() * TAU, speed: FLIGHT.speed, climb: 0, bank: 0, pitch: 0,
      phase: random() * TAU, beforePhase: 0, amp: 0, fold: 0, flapping: false, flapFor: random() * 4,
      orbit: o.orbits[ 0 ], radius: 0, lift: 0, turn: 1, timer: 0, perch: null, scale: 0.92 + random() * 0.18,
    };
    chooseOrbit( gull );
    // a quarter stand on the rocks; the rest are on their circles, heading along them
    const perch = i % 4 === 3 ? freePerch( new THREE.Vector3( random() * 80 - 40, o.sea, random() * 80 - 40 ) ) : null;
    if ( perch ) {

      taken.add( perch );
      gull.perch = perch;
      gull.mode = PERCHED;
      gull.fold = 1;
      gull.speed = 0;
      gull.timer = 20 + random() * 90;
      gull.position.copy( perch ).setY( perch.y + LEGS );

    } else {

      const a = random() * TAU, c = gull.orbit.centre;
      gull.position.set( c.x + Math.cos( a ) * gull.radius, Math.max( c.y + gull.lift, o.ceiling( c.x, c.z ) + 1 ), c.z + Math.sin( a ) * gull.radius );
      // along the tangent, in the gull's direction round
      gull.yaw = Math.atan2( - Math.sin( a ) * gull.turn, Math.cos( a ) * gull.turn );

    }
    gull.before.copy( gull.position );
    gull.beforePhase = gull.phase;
    gulls.push( gull );

  }

  const ahead = new THREE.Vector3(), rest = new THREE.Vector3();
  let clock = 0;
  // the least height on the way ahead: here and where it will be over the next seconds
  const clearance = ( gull: Gull ) => {

    let c = o.ceiling( gull.position.x, gull.position.z );
    for ( const t of [ 0.8, 1.6, 2.6 ] ) {

      ahead.set( Math.sin( gull.yaw ), 0, Math.cos( gull.yaw ) ).multiplyScalar( gull.speed * t ).add( gull.position );
      c = Math.max( c, o.ceiling( ahead.x, ahead.z ) );

    }
    return c;

  };

  const fly = ( gull: Gull, dt: number, night: number ) => {

    const p = gull.position;
    let wantYaw: number, wantHeight: number, wantSpeed = FLIGHT.speed;
    if ( gull.mode === LAND && gull.perch ) {

      // to the perch, gliding down a slope, slowing over the last metres
      const target = gull.perch, dx = target.x - p.x, dz = target.z - p.z, d = Math.hypot( dx, dz );
      wantYaw = Math.atan2( dx, dz );
      wantHeight = target.y + LEGS + Math.max( 0, d - 1 ) * 0.4;
      if ( d > 7 ) wantHeight = Math.max( wantHeight, clearance( gull ) + 1 );
      wantSpeed = THREE.MathUtils.clamp( 1.5 + d * 0.45, 1.5, FLIGHT.speed );
      if ( d < 1.6 ) {

        // the flare: it hangs on its wings and settles onto the perch
        rest.set( target.x, target.y + LEGS, target.z );
        p.lerp( rest, 1 - Math.exp( - dt * 3 ) );
        gull.speed += ( 0.5 - gull.speed ) * ( 1 - Math.exp( - dt * 4 ) );
        gull.amp += ( 0.8 - gull.amp ) * ( 1 - Math.exp( - dt * 6 ) );
        gull.phase += dt * TAU * 4;
        gull.pitch += ( - 0.5 - gull.pitch ) * ( 1 - Math.exp( - dt * 4 ) );
        gull.bank += ( 0 - gull.bank ) * ( 1 - Math.exp( - dt * 4 ) );
        gull.fold += ( 0.2 - gull.fold ) * ( 1 - Math.exp( - dt * 4 ) );
        if ( p.distanceTo( rest ) < 0.12 ) {

          gull.mode = PERCHED;
          gull.timer = night > 0.5 ? Infinity : 30 + random() * 120;

        }
        return;

      }

    } else {

      // along the orbit's tangent, bent toward the circle
      const c = gull.orbit.centre, dx = p.x - c.x, dz = p.z - c.z, r = Math.hypot( dx, dz ) || 1;
      const rx = dx / r, rz = dz / r;
      const tx = - rz * gull.turn, tz = rx * gull.turn;
      const bend = Math.atan( ( r - gull.radius ) / ( 0.5 * gull.radius ) );
      const wx = tx * Math.cos( bend ) - rx * Math.sin( bend ), wz = tz * Math.cos( bend ) - rz * Math.sin( bend );
      wantYaw = Math.atan2( wx, wz );
      wantHeight = c.y + gull.lift + 1.5 * Math.sin( clock * 0.21 + gull.scale * 40 );
      if ( gull.mode === TAKEOFF ) {

        wantHeight = Math.max( wantHeight, ( gull.perch?.y ?? p.y ) + 6 );
        if ( p.y > ( gull.perch?.y ?? 0 ) + 4 ) {

          gull.mode = FLY;
          gull.perch = null;

        }

      }
      wantHeight = Math.max( wantHeight, clearance( gull ) + 1 );
      wantSpeed = gull.orbit.speed ?? FLIGHT.speed;

    }
    // turn, bank into the turn, climb or sink
    const error = Math.atan2( Math.sin( wantYaw - gull.yaw ), Math.cos( wantYaw - gull.yaw ) );
    const rate = THREE.MathUtils.clamp( error * 2, - FLIGHT.turn, FLIGHT.turn );
    gull.yaw += rate * dt;
    gull.speed += ( wantSpeed - gull.speed ) * ( 1 - Math.exp( - dt * 1.5 ) );
    const climb = THREE.MathUtils.clamp( ( wantHeight - p.y ) * 0.8, - FLIGHT.sink, FLIGHT.climb );
    gull.climb += ( climb - gull.climb ) * ( 1 - Math.exp( - dt * 2 ) );
    const bank = Math.atan( gull.speed * rate / G );
    gull.bank += ( THREE.MathUtils.clamp( bank, - 0.8, 0.8 ) - gull.bank ) * ( 1 - Math.exp( - dt * 3 ) );
    gull.pitch = Math.atan2( gull.climb, Math.max( gull.speed, 1 ) ) * 0.7;
    p.x += Math.sin( gull.yaw ) * gull.speed * dt;
    p.z += Math.cos( gull.yaw ) * gull.speed * dt;
    p.y += gull.climb * dt;
    // the wingbeat: bouts of flapping, more when climbing, slow or landing
    gull.flapFor -= dt;
    const hard = gull.climb > 1 || gull.speed < 6 || gull.mode === TAKEOFF;
    if ( gull.flapFor <= 0 ) {

      gull.flapping = hard || random() < 0.35;
      gull.flapFor = gull.flapping ? 0.8 + random() * 2 : 2 + random() * 8;

    }
    const flapping = gull.flapping || hard;
    const amp = flapping ? ( gull.mode === FLY ? 0.55 : 0.75 ) : 0;
    gull.amp += ( amp - gull.amp ) * ( 1 - Math.exp( - dt * 5 ) );
    gull.phase += dt * TAU * ( flapping ? ( gull.mode === FLY ? 3 : 3.8 ) : 0.4 );
    gull.fold += ( ( gull.mode === LAND && gull.speed < 2.5 ? 0.25 : 0 ) - gull.fold ) * ( 1 - Math.exp( - dt * 6 ) );

  };

  const stand = ( gull: Gull, dt: number, night: number ) => {

    const perch = gull.perch!;
    gull.position.lerp( rest.set( perch.x, perch.y + LEGS, perch.z ), 1 - Math.exp( - dt * 8 ) );
    gull.fold += ( 1 - gull.fold ) * ( 1 - Math.exp( - dt * 5 ) );
    gull.amp += ( 0 - gull.amp ) * ( 1 - Math.exp( - dt * 6 ) );
    gull.bank += ( 0 - gull.bank ) * ( 1 - Math.exp( - dt * 4 ) );
    gull.pitch += ( - 0.12 - gull.pitch ) * ( 1 - Math.exp( - dt * 4 ) );
    gull.speed = 0;
    gull.climb = 0;
    // one that came down for the night leaves in the morning, each in its own time
    if ( gull.timer === Infinity && night < 0.35 ) gull.timer = 5 + random() * 60;
    gull.timer -= dt;
    // by day it leaves after a while; by night it stays
    if ( night < 0.35 && gull.timer <= 0 ) {

      gull.mode = TAKEOFF;
      taken.delete( perch );
      chooseOrbit( gull );
      gull.speed = 3;
      gull.climb = 2;
      gull.flapping = true;
      gull.flapFor = 1.5;
      gull.amp = 0.5;

    }

  };

  const write = () => {

    const pose = attrs.iPose.array as Float32Array, turn = attrs.iTurn.array as Float32Array;
    const motion = attrs.iMotion.array as Float32Array, before = attrs.iBefore.array as Float32Array;
    const euler = new THREE.Euler( 0, 0, 0, 'YXZ' ), q = new THREE.Quaternion();
    gulls.forEach( ( gull, i ) => {

      const k = i * 4;
      // yaw about y (0 along +z), pitch (nose up is a turn about -x), bank (a right turn rolls right)
      euler.set( - gull.pitch, gull.yaw, - gull.bank );
      q.setFromEuler( euler );
      pose[ k ] = gull.position.x; pose[ k + 1 ] = gull.position.y; pose[ k + 2 ] = gull.position.z; pose[ k + 3 ] = gull.scale;
      turn[ k ] = q.x; turn[ k + 1 ] = q.y; turn[ k + 2 ] = q.z; turn[ k + 3 ] = q.w;
      motion[ k ] = gull.phase; motion[ k + 1 ] = gull.amp; motion[ k + 2 ] = gull.fold; motion[ k + 3 ] = 0;
      before[ k ] = gull.before.x; before[ k + 1 ] = gull.before.y; before[ k + 2 ] = gull.before.z; before[ k + 3 ] = gull.beforePhase;

    } );
    for ( const a of Object.values( attrs ) ) a.needsUpdate = true;

  };
  write();

  return {
    mesh,
    get perched() {

      return gulls.reduce( ( n, g ) => n + ( g.mode === PERCHED ? 1 : 0 ), 0 );

    },
    update( dt, night ) {

      for ( const gull of gulls ) {

        gull.before.copy( gull.position );
        gull.beforePhase = gull.phase;

      }
      if ( dt <= 0 ) return write();
      dt = Math.min( dt, 0.1 );
      clock += dt;
      for ( const gull of gulls ) {

        if ( gull.mode === PERCHED ) {

          stand( gull, dt, night );
          continue;

        }
        // at nightfall, each in its own time, down to a free perch; by day now and then one lands
        if ( night > 0.55 && gull.mode === FLY && gull.timer > 20 ) gull.timer = random() * 20;
        gull.timer -= dt;
        if ( gull.mode === FLY && gull.timer <= 0 ) {

          const perched = gulls.filter( ( g ) => g.perch ).length;
          const perch = night > 0.55 || ( perched < COUNT / 4 && random() < 0.4 ) ? freePerch( gull.position ) : null;
          if ( perch ) {

            taken.add( perch );
            gull.perch = perch;
            gull.mode = LAND;

          } else {

            chooseOrbit( gull );
            // at night, those still up try again soon
            if ( night > 0.55 ) gull.timer = random() * 20;

          }

        }
        fly( gull, dt, night );

      }
      write();

    },
    caller( pick ) {

      if ( gulls.length === 0 ) return null;
      return gulls[ Math.floor( pick() * gulls.length ) ].position;

    },
  };

}
