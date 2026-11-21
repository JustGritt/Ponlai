// The floating wreckage: what the storm tore from the ship, riding the sea's actual waves under Rapier.
//
// Each body carries a few sample points in its own frame, each standing for an equal share of the
// volume it can displace. Every physics step, a point under the surface pushes up with the weight
// of the water its share displaces (Archimedes), scaled by how deep it is in, and drags against
// the water's orbital motion, so the pieces heave, roll and drift with the waves rather than bob on
// a flat plane. The surface comes from the GPU (waveProbe.ts) a frame or two late; between results
// each point keeps the last height read for it.
//
// The rocks and the wreck wedged against the stack are fixed triangle meshes in the same world, so
// the wreckage knocks against them. A piece dropped from above reports its splash as it meets the
// water, for the sound.
//
// A vessel (`launch`: the fishing boat, boat.ts) floats the same way, with its own mass, centre of
// mass and inertia, and drag that differs along its keel and across it, so it runs ahead and
// resists going sideways; its pilot adds its engine's thrust and its rudder's turn every step. It
// is kept whatever the count of pieces and however far it goes, and it shoves the wreckage aside.
import * as THREE from 'three/webgpu';
import RAPIER from '@dimforge/rapier3d-compat';
import type { WaveProbe } from './waveProbe.ts';
import {
  barrelGeometry, canvasGeometry, gratingGeometry, group, hullGeometry, hullPoints, merge, mulberry32, plankGeometry, ropeGeometry, sparGeometry, wheelGeometry,
  type HullShape, type WreckMaterials,
} from './wreckage.ts';

const WATER_DENSITY = 1025;
const G = 9.81;
const STEP = 1 / 60;
const MAX_BODIES = 56;
// drag against the water, per kilogram of displaced water ( 1/s ): vertical, horizontal
const DRAG = { vertical: 2.2, horizontal: 0.9 };
// how far from the stack a piece may drift before it is taken away (m)
const DRIFT_LIMIT = 150;

export type Kind = 'plank' | 'hull' | 'mast' | 'lifeboat' | 'wheel' | 'grating' | 'barrel';

interface Variant {
  parts: [ THREE.BufferGeometry, THREE.Material ][];
  colliders: () => RAPIER.ColliderDesc[];
  /** the sample points in the body's frame */
  points: THREE.Vector3[];
}

interface KindSpec {
  variants: Variant[];
  /** kg */
  mass: number;
  /** the water it displaces fully under (m^3): the wood's, and the air a capsized boat traps */
  volume: number;
  /** the vertical extent a sample point stands for (m) */
  extent: number;
  /** how a new piece of this kind is turned: any way, or only about the vertical */
  tumble: boolean;
}

/** Drag against the water per kilogram of displaced water ( 1/s ), in a body's own frame. */
export interface HullDrag {
  /** along its x (the keel) */
  along: number;
  /** across it (z) */
  across: number;
  vertical: number;
}

/** A vessel under its own power (`Floaters.launch`). Its frame: x forward, y up, z to starboard. */
export interface VesselSpec {
  name: string;
  /** its meshes, in its own frame */
  object: THREE.Object3D;
  /** its colliders, in its own frame; their density is ignored (the mass is given) */
  colliders: () => RAPIER.ColliderDesc[];
  /** the sample points, as a piece's (each an equal share of `volume`) */
  points: THREE.Vector3[];
  /** kg, the centre of mass and the principal moments of inertia ( kg m^2 ) in its frame */
  mass: number;
  centreOfMass: THREE.Vector3;
  inertia: THREE.Vector3;
  /** the water it displaces with every point under (m^3), and a point's vertical extent (m) */
  volume: number;
  extent: number;
  drag: HullDrag;
  /** where it starts: its frame's origin (world) and its heading (radians about y) */
  start: THREE.Vector3;
  heading: number;
  /** every physics step, after the water's forces: the pilot's own (thrust, rudder) */
  steer( rigid: RAPIER.RigidBody, step: number ): void;
}

interface Body {
  kind: Kind | 'vessel';
  rigid: RAPIER.RigidBody;
  object: THREE.Object3D;
  /** the sample points in its frame, the share of the volume each displaces and its extent */
  points: THREE.Vector3[];
  share: number;
  extent: number;
  /** drag in its own frame; null: the wreckage's, the same in every direction (DRAG) */
  drag: HullDrag | null;
  /** a vessel: never taken away, steered every step */
  steer: VesselSpec[ 'steer' ] | null;
  /** the batch after which its samples are its own */
  since: number;
  /** the latest surface under each sample point: height, vx, vy, vz */
  water: Float32Array;
  /** dropped from above and not yet in the water */
  falling: boolean;
}

export interface Floaters {
  /** steps the physics by the scene clock's `dt` and moves the pieces */
  update( dt: number ): void;
  /** drops a random piece of wreckage above the sea at x, z */
  drop( x: number, z: number ): void;
  /** puts a vessel on the sea (its object joins the group) and returns its body */
  launch( vessel: VesselSpec ): RAPIER.RigidBody;
  /** the pieces of wreckage afloat (the vessels not counted) */
  readonly count: number;
  /** the probe's batches and every body's height against the mean water under its points */
  stats(): { sent: number; received: number; bodies: { kind: Kind | 'vessel'; x: number; y: number; z: number; water: number }[] };
}

// ------------------------------------------------------------------------------ the kinds

/** Moves every part so the parts' joint box is centred on the origin. */
function centre( parts: THREE.BufferGeometry[] ): THREE.Box3 {

  const box = new THREE.Box3();
  for ( const g of parts ) {

    g.computeBoundingBox();
    box.union( g.boundingBox! );

  }
  const c = box.getCenter( new THREE.Vector3() );
  for ( const g of parts ) g.translate( - c.x, - c.y, - c.z );
  return box.translate( c.negate() );

}

/** nx x ny x nz points over `box`, pulled in by `inset`. */
function boxPoints( box: THREE.Box3, nx: number, ny: number, nz: number, inset = 0.75 ): THREE.Vector3[] {

  const out: THREE.Vector3[] = [];
  const at = ( lo: number, hi: number, k: number, n: number ) => n === 1 ? ( lo + hi ) / 2 : ( lo + hi ) / 2 + ( hi - lo ) / 2 * inset * ( 2 * k / ( n - 1 ) - 1 );
  for ( let i = 0; i < nx; i ++ ) for ( let j = 0; j < ny; j ++ ) for ( let k = 0; k < nz; k ++ ) {

    out.push( new THREE.Vector3( at( box.min.x, box.max.x, i, nx ), at( box.min.y, box.max.y, j, ny ), at( box.min.z, box.max.z, k, nz ) ) );

  }
  return out;

}

const alongX = new THREE.Quaternion().setFromAxisAngle( new THREE.Vector3( 0, 0, 1 ), Math.PI / 2 );
const alongZ = new THREE.Quaternion().setFromAxisAngle( new THREE.Vector3( 1, 0, 0 ), Math.PI / 2 );

function kinds( m: WreckMaterials ): Record<Kind, KindSpec> {

  // ---- splintered planks
  const planks: Variant[] = [ [ 3.4, 0.26, 101 ], [ 2.6, 0.22, 102 ], [ 1.9, 0.28, 103 ], [ 2.9, 0.24, 104 ] ].map( ( [ length, width, seed ] ) => {

    const g = plankGeometry( length, width, 0.065, seed );
    const box = centre( [ g ] );
    return {
      parts: [ [ g, m.timber ] ],
      colliders: () => [ RAPIER.ColliderDesc.cuboid( length / 2 - 0.15, 0.033, width / 2 ) ],
      points: boxPoints( box, 3, 1, 1 ),
    };

  } );

  // ---- torn-off pieces of the hull: planking on its ribs, broken at both ends, one torn from a side
  const hullPieces: Variant[] = ( [
    { length: 3.4, beam: 1.9, depth: 1.1, sheer: 0, ends: 'broken', jag: 0.18, holes: 0.12, span: [ - 1, 0.35 ], seed: 11 },
    { length: 2.7, beam: 1.6, depth: 0.95, sheer: 0, ends: 'broken', jag: 0.22, holes: 0.1, span: [ - 0.25, 1 ], seed: 12 },
  ] as HullShape[] ).map( ( shape ) => {

    const { shell, frames } = hullGeometry( shape );
    const box = centre( [ shell, frames ] );
    const outline = hullPoints( shell );
    return {
      parts: [ [ shell, m.hull ], [ frames, m.timber ] ],
      colliders: () => [ RAPIER.ColliderDesc.convexHull( outline )! ],
      points: boxPoints( box, 3, 2, 2 ),
    };

  } );

  // ---- the capsized lifeboat, keel up: thwarts inside, the air under it keeping it afloat
  const lifeboat = ( () => {

    const shape: HullShape = { length: 4.2, beam: 1.5, depth: 0.65, sheer: 0.28, ends: 'boat', jag: 0, holes: 0, seed: 21 };
    const { shell, frames } = hullGeometry( shape );
    const thwarts: THREE.BufferGeometry[] = [ 1.2, 2.1, 3.0 ].map( ( x ) => {

      const t = new THREE.BoxGeometry( 0.22, 0.05, shape.beam * 0.86 );
      t.translate( x, - 0.22, 0 );
      return t;

    } );
    const timber = merge( [ frames, ...thwarts ] );
    for ( const g of [ shell, timber ] ) g.rotateX( Math.PI );
    const box = centre( [ shell, timber ] );
    const outline = hullPoints( shell );
    return {
      parts: [ [ shell, m.hull ], [ timber, m.timber ] ],
      colliders: () => [ RAPIER.ColliderDesc.convexHull( outline )! ],
      points: boxPoints( box, 3, 2, 2 ),
    } as Variant;

  } )();

  // ---- the broken mast: the mast snapped off, its yard across it, the torn sail spread on the water,
  // rope trailing from the yard
  const mast = ( () => {

    // the mast along x about the origin, its yard across it at YARD (the body's frame)
    const YARD = new THREE.Vector3( 1.7, 0.26, 0.3 );
    const spar = sparGeometry( 6.6, 0.19, 31 );
    const yard = sparGeometry( 4.8, 0.1, 32 );
    yard.rotateY( Math.PI / 2 );
    yard.translate( YARD.x, YARD.y, YARD.z );
    const sail = canvasGeometry( 2.9, 4.4, 33 );
    sail.translate( YARD.x - 1.5, 0.12, YARD.z );
    const ropes = merge( [
      ropeGeometry( [ new THREE.Vector3( YARD.x, 0.3, 2.6 ), new THREE.Vector3( YARD.x + 0.8, 0.05, 3.1 ), new THREE.Vector3( YARD.x + 2.2, - 0.05, 3.3 ), new THREE.Vector3( YARD.x + 3.4, 0, 2.7 ) ] ),
      ropeGeometry( [ new THREE.Vector3( 2.9, 0.18, 0 ), new THREE.Vector3( 3.6, 0.05, - 0.9 ), new THREE.Vector3( 4.6, - 0.05, - 1.3 ), new THREE.Vector3( 5.5, 0, - 0.8 ) ] ),
      ropeGeometry( [ new THREE.Vector3( YARD.x, 0.3, - 2.0 ), new THREE.Vector3( YARD.x - 0.6, 0.08, - 2.7 ), new THREE.Vector3( YARD.x - 1.8, - 0.02, - 3.0 ) ] ),
    ] );
    return {
      parts: [ [ merge( [ spar, yard ] ), m.timber ], [ sail, m.canvas ], [ ropes, m.rope ] ],
      colliders: () => [
        RAPIER.ColliderDesc.cylinder( 3.2, 0.19 ).setRotation( alongX ),
        RAPIER.ColliderDesc.cylinder( 2.3, 0.1 ).setRotation( alongZ ).setTranslation( YARD.x, YARD.y, YARD.z ),
      ],
      // along the mast, the yard's ends and the sail's corners (the cloth holds some air)
      points: [ - 2.6, - 1.3, 0, 1.3, 2.6 ].map( ( x ) => new THREE.Vector3( x, 0, 0 ) ).concat( [
        new THREE.Vector3( YARD.x, YARD.y, YARD.z + 2.1 ), new THREE.Vector3( YARD.x, YARD.y, YARD.z - 2.1 ),
        new THREE.Vector3( YARD.x - 2.4, 0.12, YARD.z + 1.6 ), new THREE.Vector3( YARD.x - 2.4, 0.12, YARD.z - 1.6 ),
      ] ),
    } as Variant;

  } )();

  // ---- the ship's wheel
  const wheel = ( () => {

    const { timber, brass } = wheelGeometry();
    return {
      parts: [ [ timber, m.timber ], [ brass, m.brass ] ],
      colliders: () => [ RAPIER.ColliderDesc.cylinder( 0.06, 0.8 ) ],
      points: [ 0, 1, 2, 3 ].map( ( k ) => new THREE.Vector3( Math.cos( k * Math.PI / 2 ) * 0.5, 0, Math.sin( k * Math.PI / 2 ) * 0.5 ) ),
    } as Variant;

  } )();

  // ---- a hatch grating
  const grating = ( () => {

    const g = gratingGeometry( 1.3, 0.9 );
    return {
      parts: [ [ g, m.timber ] ],
      colliders: () => [ RAPIER.ColliderDesc.cuboid( 0.65, 0.05, 0.45 ) ],
      points: boxPoints( new THREE.Box3( new THREE.Vector3( - 0.65, 0, - 0.45 ), new THREE.Vector3( 0.65, 0, 0.45 ) ), 2, 1, 2 ),
    } as Variant;

  } )();

  // ---- the cargo's barrels: three rings of four points along the axis
  const barrel = ( () => {

    const points: THREE.Vector3[] = [];
    for ( const y of [ - 0.3, 0, 0.3 ] ) for ( let k = 0; k < 4; k ++ ) points.push( new THREE.Vector3( Math.cos( k * Math.PI / 2 ) * 0.18, y, Math.sin( k * Math.PI / 2 ) * 0.18 ) );
    return { parts: [ [ barrelGeometry(), m.barrel ] ], colliders: () => [ RAPIER.ColliderDesc.cylinder( 0.45, 0.3 ) ], points } as Variant;

  } )();

  return {
    plank: { variants: planks, mass: 34, volume: 0.058, extent: 0.1, tumble: false },
    hull: { variants: hullPieces, mass: 380, volume: 0.85, extent: 0.35, tumble: true },
    lifeboat: { variants: [ lifeboat ], mass: 420, volume: 1.9, extent: 0.25, tumble: false },
    mast: { variants: [ mast ], mass: 520, volume: 1.0, extent: 0.2, tumble: false },
    wheel: { variants: [ wheel ], mass: 30, volume: 0.06, extent: 0.08, tumble: false },
    grating: { variants: [ grating ], mass: 28, volume: 0.055, extent: 0.08, tumble: false },
    barrel: { variants: [ barrel ], mass: 70, volume: 0.25, extent: 0.3, tumble: true },
  };

}

// what a click drops, by weight
const DROPS: [ Kind, number ][] = [ [ 'plank', 5 ], [ 'hull', 2 ], [ 'barrel', 2 ], [ 'grating', 1.5 ], [ 'wheel', 0.6 ], [ 'mast', 0.6 ], [ 'lifeboat', 0.5 ] ];

/** The floating wreckage, its physics world and its pieces (added to `opts.group`). */
export async function createFloaters( opts: {
  group: THREE.Group; probe: WaveProbe; sea: number; materials: WreckMaterials;
  /** fixed triangle meshes in world space: the rocks, the wreck */
  fixed: THREE.BufferGeometry[];
  /** where the wreck lies (radians about y from +x toward +z): the debris field spreads from there */
  wreckAngle: number;
  /** whether the debris field may start a piece at x, z (clear of the rocks) */
  clear: ( x: number, z: number ) => boolean;
  onObject: ( object: THREE.Object3D ) => void;
  /** a dropped piece struck the water at x, y, z, falling at `speed` (m/s) */
  onSplash?: ( x: number, y: number, z: number, speed: number ) => void;
} ): Promise<Floaters> {

  await RAPIER.init();
  const world = new RAPIER.World( { x: 0, y: - G, z: 0 } );
  world.timestep = STEP;
  const { probe, sea } = opts;

  for ( const geometry of opts.fixed ) {

    const positions = new Float32Array( geometry.attributes.position.array );
    const indices = new Uint32Array( geometry.index!.array );
    world.createCollider( RAPIER.ColliderDesc.trimesh( positions, indices ).setFriction( 0.8 ), world.createRigidBody( RAPIER.RigidBodyDesc.fixed() ) );

  }

  const specs = kinds( opts.materials );
  const bodies: Body[] = [];
  let accumulator = 0;
  const xz = new Float32Array( probe.capacity * 2 );
  const p = new THREE.Vector3(), q = new THREE.Quaternion();
  const random = mulberry32( 2026 );

  const spawn = ( kind: Kind, x: number, y: number, z: number, falling = false ) => {

    const pieces = bodies.filter( ( b ) => ! b.steer );
    if ( pieces.length >= MAX_BODIES ) remove( pieces[ 0 ] );
    const spec = specs[ kind ];
    const variant = Math.floor( random() * spec.variants.length );
    const v = spec.variants[ variant ];
    const yaw = random() * Math.PI * 2;
    const rotation = spec.tumble
      ? new THREE.Quaternion().setFromEuler( new THREE.Euler( random() * Math.PI * 2, yaw, random() * Math.PI * 2 ) )
      : new THREE.Quaternion().setFromEuler( new THREE.Euler( ( random() - 0.5 ) * 0.3, yaw, ( random() - 0.5 ) * 0.3 ) );
    const rigid = world.createRigidBody( RAPIER.RigidBodyDesc.dynamic()
      .setTranslation( x, y, z ).setRotation( rotation )
      .setAngvel( { x: ( random() - 0.5 ), y: ( random() - 0.5 ), z: ( random() - 0.5 ) } )
      .setLinearDamping( 0.05 ).setAngularDamping( 0.4 ).setCanSleep( false ) );
    const colliders = v.colliders();
    for ( const collider of colliders ) world.createCollider( collider.setMass( spec.mass / colliders.length ).setFriction( 0.7 ).setRestitution( 0.1 ), rigid );
    const object = group( ...v.parts );
    object.name = kind;
    opts.group.add( object );
    opts.onObject( object );
    bodies.push( {
      kind, rigid, object, points: v.points, share: spec.volume / v.points.length, extent: spec.extent, drag: null, steer: null,
      since: probe.sent, water: still( v.points.length ), falling,
    } );

  };

  const remove = ( body: Body ) => {

    world.removeRigidBody( body.rigid );
    opts.group.remove( body.object );
    bodies.splice( bodies.indexOf( body ), 1 );

  };

  // the surface under a new body's points, until its first samples: the mean sea, still
  const still = ( n: number ) => {

    const water = new Float32Array( n * 4 );
    for ( let k = 0; k < n; k ++ ) water[ k * 4 ] = sea;
    return water;

  };

  // the batch in flight's layout: which body held which slots when it was sent
  let sentLayout: { body: Body; slot: number }[] = [];
  const takeResult = ( batch: number ) => {

    for ( const { body, slot } of sentLayout ) {

      if ( ! bodies.includes( body ) || batch <= body.since ) continue;
      body.water.set( probe.result.subarray( slot * 4, ( slot + body.points.length ) * 4 ) );

    }

  };

  const sendPoints = () => {

    if ( ! probe.ready || bodies.length === 0 ) return;
    let n = 0;
    const layout: { body: Body; slot: number }[] = [];
    for ( const body of bodies ) {

      const points = body.points;
      if ( n + points.length > probe.capacity ) break;
      layout.push( { body, slot: n } );
      const t = body.rigid.translation(), r = body.rigid.rotation();
      q.set( r.x, r.y, r.z, r.w );
      for ( const local of points ) {

        p.copy( local ).applyQuaternion( q ).add( t as THREE.Vector3Like );
        xz[ n * 2 ] = p.x;
        xz[ n * 2 + 1 ] = p.z;
        n ++;

      }

    }
    sentLayout = layout;
    probe.request( xz, n, takeResult );

  };

  const force = { x: 0, y: 0, z: 0 }, at = { x: 0, y: 0, z: 0 };
  const relative = new THREE.Vector3(), qInverse = new THREE.Quaternion();
  const applyWater = () => {

    for ( const body of bodies ) {

      const { points, share, extent, drag } = body;
      const t = body.rigid.translation(), r = body.rigid.rotation();
      q.set( r.x, r.y, r.z, r.w );
      body.rigid.resetForces( true );
      body.rigid.resetTorques( true );
      let wet = 0;
      points.forEach( ( local, k ) => {

        p.copy( local ).applyQuaternion( q ).add( t as THREE.Vector3Like );
        const height = body.water[ k * 4 ];
        // how much of the point's share is under the surface, 0 to 1
        const f = THREE.MathUtils.clamp( ( height - p.y ) / ( 2 * extent ) + 0.5, 0, 1 );
        if ( f <= 0 ) return;
        wet += f;
        const displaced = WATER_DENSITY * share * f;
        // the point's own velocity: the body's plus its spin's, against the water's
        const v = body.rigid.velocityAtPoint( p );
        relative.set( v.x - body.water[ k * 4 + 1 ], v.y - body.water[ k * 4 + 2 ], v.z - body.water[ k * 4 + 3 ] );
        if ( drag ) {

          // a hull's drag: in its own frame, then back to the world's
          relative.applyQuaternion( qInverse.copy( q ).invert() );
          relative.set( relative.x * drag.along, relative.y * drag.vertical, relative.z * drag.across ).applyQuaternion( q );

        } else {

          relative.set( relative.x * DRAG.horizontal, relative.y * DRAG.vertical, relative.z * DRAG.horizontal );

        }
        force.x = - displaced * relative.x;
        force.y = displaced * G - displaced * relative.y;
        force.z = - displaced * relative.z;
        at.x = p.x; at.y = p.y; at.z = p.z;
        body.rigid.addForceAtPoint( force, at, true );

      } );
      // water damps spin far more than air (a hull's roll is damped by its own points' drag)
      body.rigid.setAngularDamping( wet > 0 ? ( drag ? 0.5 : 1.2 ) : 0.2 );
      if ( body.falling && wet > 0 ) {

        body.falling = false;
        opts.onSplash?.( t.x, t.y, t.z, Math.abs( body.rigid.linvel().y ) );

      }

    }

  };

  const sync = () => {

    for ( const body of bodies ) {

      const t = body.rigid.translation(), r = body.rigid.rotation();
      body.object.position.set( t.x, t.y, t.z );
      body.object.quaternion.set( r.x, r.y, r.z, r.w );
      body.object.updateMatrixWorld();

    }

  };

  // The debris field: spread from the wreck down the stack's side and out to sea, already afloat.
  const field: [ Kind, number ][] = [
    [ 'mast', 1 ], [ 'lifeboat', 1 ], [ 'hull', 3 ], [ 'wheel', 1 ], [ 'grating', 2 ], [ 'plank', 12 ], [ 'barrel', 4 ],
  ];
  for ( const [ kind, count ] of field ) for ( let i = 0; i < count; i ++ ) {

    // clear of the rocks (the whaleback and the skerries lie in the field)
    let x = 0, z = 0;
    for ( let tries = 0; tries < 20; tries ++ ) {

      const a = opts.wreckAngle + ( random() - 0.5 ) * 2.2, r = 15 + random() * 36;
      x = Math.cos( a ) * r;
      z = Math.sin( a ) * r;
      if ( opts.clear( x, z ) ) break;

    }
    spawn( kind, x, sea + 0.2, z );

  }
  sync();

  const pick = () => {

    let w = random() * DROPS.reduce( ( s, [ , weight ] ) => s + weight, 0 );
    for ( const [ kind, weight ] of DROPS ) if ( ( w -= weight ) <= 0 ) return kind;
    return 'plank';

  };

  return {
    get count() {

      return bodies.reduce( ( n, body ) => n + ( body.steer ? 0 : 1 ), 0 );

    },
    stats() {

      return {
        sent: probe.sent, received: probe.received,
        bodies: bodies.map( ( body ) => {

          const t = body.rigid.translation(), n = body.points.length;
          let water = 0;
          for ( let k = 0; k < n; k ++ ) water += body.water[ k * 4 ];
          return { kind: body.kind, x: + t.x.toFixed( 2 ), y: + t.y.toFixed( 3 ), z: + t.z.toFixed( 2 ), water: + ( water / n ).toFixed( 3 ) };

        } ),
      };

    },
    update( dt: number ) {

      sendPoints();
      if ( dt <= 0 ) return;
      accumulator = Math.min( accumulator + dt, STEP * 4 );
      while ( accumulator >= STEP ) {

        applyWater();
        for ( const body of bodies ) body.steer?.( body.rigid, STEP );
        world.step();
        accumulator -= STEP;

      }
      for ( const body of [ ...bodies ] ) {

        const t = body.rigid.translation();
        if ( ! body.steer && ( Math.hypot( t.x, t.z ) > DRIFT_LIMIT || t.y < sea - 40 ) ) remove( body );

      }
      sync();

    },
    drop( x: number, z: number ) {

      spawn( pick(), x, sea + 5 + random() * 3, z, true );
      sync();

    },
    launch( vessel ) {

      const { centreOfMass: c, inertia: i } = vessel;
      const rigid = world.createRigidBody( RAPIER.RigidBodyDesc.dynamic()
        .setTranslation( vessel.start.x, vessel.start.y, vessel.start.z )
        .setRotation( new THREE.Quaternion().setFromAxisAngle( new THREE.Vector3( 0, 1, 0 ), vessel.heading ) )
        .setAdditionalMassProperties( vessel.mass, { x: c.x, y: c.y, z: c.z }, { x: i.x, y: i.y, z: i.z }, { x: 0, y: 0, z: 0, w: 1 } )
        .setLinearDamping( 0.02 ).setAngularDamping( 0.5 ).setCanSleep( false ) );
      for ( const collider of vessel.colliders() ) world.createCollider( collider.setDensity( 0 ).setFriction( 0.5 ).setRestitution( 0.05 ), rigid );
      vessel.object.name = vessel.name;
      opts.group.add( vessel.object );
      opts.onObject( vessel.object );
      // first in the probe's batch, so its points are always sampled
      bodies.unshift( {
        kind: 'vessel', rigid, object: vessel.object, points: vessel.points, share: vessel.volume / vessel.points.length, extent: vessel.extent,
        drag: vessel.drag, steer: vessel.steer, since: probe.sent, water: still( vessel.points.length ), falling: false,
      } );
      sync();
      return rigid;

    },
  };

}
