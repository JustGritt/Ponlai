// Floating wreckage: barrels, crates and a buoy that ride the sea's actual waves, under Rapier.
//
// Each body carries a few sample points in its own frame, each standing for an equal share of its
// volume. Every physics step, a point under the surface pushes up with the weight of the water its
// share displaces (Archimedes), scaled by how deep it is in, and drags against the water's orbital
// motion, so bodies heave, roll and drift with the waves rather than bob on a flat plane. The
// surface comes from the GPU (waveProbe.ts) a frame or two late; between results each point keeps
// the last height read for it.
//
// The stack is a static triangle mesh in the same world, so wreckage knocks against the rock.
import * as THREE from 'three/webgpu';
import RAPIER from '@dimforge/rapier3d-compat';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { float, fract, mix, positionLocal, smoothstep, step, uv, vec3, abs, min, mx_fractal_noise_float, positionWorld } from 'three/tsl';
import type { RenderParts } from './render/index.ts';
import { rgb, surface } from './surfaces.ts';
import type { WaveProbe } from './waveProbe.ts';

const WATER_DENSITY = 1025;
const G = 9.81;
const STEP = 1 / 60;
const MAX_BODIES = 48;
// drag against the water, per kilogram of displaced water ( 1/s ): vertical, horizontal
const DRAG = { vertical: 2.2, horizontal: 0.9 };
// how far from the stack a body may drift before it is taken away (m)
const DRIFT_LIMIT = 140;

type Kind = 'barrel' | 'crate' | 'buoy';

interface KindSpec {
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
  collider: () => RAPIER.ColliderDesc;
  /** the sample points in the body's frame, and the vertical extent each point stands for (m) */
  points: THREE.Vector3[];
  extent: number;
  volume: number;
}

interface Body {
  kind: Kind;
  rigid: RAPIER.RigidBody;
  mesh: THREE.Mesh;
  /** the first of its sample slots */
  slot: number;
  /** the batch after which its samples are its own */
  since: number;
  /** the latest surface under each sample point: height, vx, vy, vz */
  water: Float32Array;
}

export interface Floaters {
  /** steps the physics by the scene clock's `dt` and moves the meshes */
  update( dt: number ): void;
  /** drops a random piece of wreckage above the sea at x, z */
  drop( x: number, z: number ): void;
  readonly count: number;
  /** the probe's batches and every body's height against the mean water under its points */
  stats(): { sent: number; received: number; bodies: { kind: Kind; x: number; y: number; z: number; water: number }[] };
}

function barrelGeometry(): THREE.BufferGeometry {

  // a barrel's bulging stave profile, closed at both ends
  const R = 0.3, H = 0.9;
  const points: THREE.Vector2[] = [ new THREE.Vector2( 0, - H / 2 ) ];
  for ( let i = 0; i <= 12; i ++ ) {

    const t = i / 12, y = - H / 2 + t * H;
    points.push( new THREE.Vector2( R * ( 0.88 + 0.12 * Math.sin( Math.PI * t ) ), y ) );

  }
  points.push( new THREE.Vector2( 0, H / 2 ) );
  return new THREE.LatheGeometry( points, 24 );

}

function kinds( render: RenderParts ): Record<Kind, KindSpec> {

  const grain = mx_fractal_noise_float( positionWorld.mul( vec3( 2, 9, 2 ) ), 2, 2, 0.5, 1 );
  // barrel: oak staves with two dark iron hoops near each end
  const y = abs( positionLocal.y );
  const hoop = step( 0.25, y ).mul( step( y, 0.31 ) ).add( step( 0.4, y ).mul( step( y, 0.44 ) ) );
  const oak = mix( rgb( 0x5c3b22 ), rgb( 0x7a5232 ), grain.mul( 0.5 ).add( 0.5 ) );
  const barrelColour = mix( oak, rgb( 0x2b2b2b ), hoop );
  const barrel = surface( render, { name: 'barrel', color: barrelColour, roughness: 0.7 } );
  // crate: planks with a darker frame along every face's border
  const u = uv();
  const edge = min( min( u.x, float( 1 ).sub( u.x ) ), min( u.y, float( 1 ).sub( u.y ) ) );
  const frame = float( 1 ).sub( smoothstep( 0.1, 0.12, edge ) );
  const seam = smoothstep( 0.9, 0.97, fract( u.y.mul( 4 ) ) );
  const pine = mix( rgb( 0x8a6a42 ), rgb( 0xa8865a ), grain.mul( 0.5 ).add( 0.5 ) );
  const crateColour = mix( mix( pine, pine.mul( 0.55 ), seam ), rgb( 0x5e4528 ), frame );
  const crate = surface( render, { name: 'crate', color: crateColour, roughness: 0.85 } );
  // buoy: red and white bands
  const band = step( 0.5, fract( positionLocal.y.mul( 1.6 ).add( 0.25 ) ) );
  const buoy = surface( render, { name: 'buoy', color: mix( rgb( 0xb3261e ), rgb( 0xe6e2da ), band ), roughness: 0.45 } );

  const grid = ( hx: number, hy: number, hz: number ) => {

    const out: THREE.Vector3[] = [];
    for ( const sx of [ - 1, 1 ] ) for ( const sy of [ - 1, 1 ] ) for ( const sz of [ - 1, 1 ] ) out.push( new THREE.Vector3( sx * hx / 2, sy * hy / 2, sz * hz / 2 ) );
    return out;

  };
  // a cylinder along y: three rings of four
  const rings = ( r: number, h: number ) => {

    const out: THREE.Vector3[] = [];
    for ( const yy of [ - h / 3, 0, h / 3 ] ) for ( let k = 0; k < 4; k ++ ) out.push( new THREE.Vector3( Math.cos( k * Math.PI / 2 ) * r * 0.6, yy, Math.sin( k * Math.PI / 2 ) * r * 0.6 ) );
    return out;

  };
  const top = new THREE.ConeGeometry( 0.3, 0.5, 16 );
  top.translate( 0, 0.95, 0 );
  const buoyGeometry = mergeGeometries( [ new THREE.CylinderGeometry( 0.42, 0.55, 1.4, 24 ), top ] );
  return {
    barrel: {
      geometry: barrelGeometry(), material: barrel,
      // a light barrel: empty, sealed
      collider: () => RAPIER.ColliderDesc.cylinder( 0.45, 0.3 ).setDensity( 280 ),
      points: rings( 0.3, 0.9 ), extent: 0.3, volume: Math.PI * 0.3 * 0.3 * 0.9,
    },
    crate: {
      geometry: new THREE.BoxGeometry( 0.8, 0.8, 0.8 ), material: crate,
      collider: () => RAPIER.ColliderDesc.cuboid( 0.4, 0.4, 0.4 ).setDensity( 470 ),
      points: grid( 0.8, 0.8, 0.8 ), extent: 0.4, volume: 0.512,
    },
    buoy: {
      geometry: buoyGeometry, material: buoy,
      // ballasted: its mass low, so it floats upright
      collider: () => RAPIER.ColliderDesc.cylinder( 0.7, 0.5 ).setMassProperties( 220, { x: 0, y: - 0.45, z: 0 }, { x: 30, y: 20, z: 30 }, { x: 0, y: 0, z: 0, w: 1 } ),
      points: rings( 0.5, 1.4 ), extent: 0.47, volume: Math.PI * 0.5 * 0.5 * 1.4,
    },
  };

}

/** The floating wreckage, its physics world and its meshes (added to `group`). */
export async function createFloaters( render: RenderParts, opts: {
  group: THREE.Group; probe: WaveProbe; sea: number; stack: THREE.BufferGeometry; onMesh: ( mesh: THREE.Mesh ) => void;
} ): Promise<Floaters> {

  await RAPIER.init();
  const world = new RAPIER.World( { x: 0, y: - G, z: 0 } );
  world.timestep = STEP;
  const { probe, sea, group } = opts;

  // the stack, as a fixed triangle mesh
  const positions = opts.stack.attributes.position.array as Float32Array;
  const indices = new Uint32Array( opts.stack.index!.array );
  world.createCollider( RAPIER.ColliderDesc.trimesh( new Float32Array( positions ), indices ).setFriction( 0.8 ), world.createRigidBody( RAPIER.RigidBodyDesc.fixed() ) );

  const specs = kinds( render );
  const bodies: Body[] = [];
  let nextSlot = 0;
  let accumulator = 0;
  const xz = new Float32Array( probe.capacity * 2 );
  const p = new THREE.Vector3(), q = new THREE.Quaternion();
  let seed = 12345;
  const random = () => {

    seed = ( seed * 16807 ) % 2147483647;
    return seed / 2147483647;

  };

  // the slots each body's points take in the probe's batch, packed in the bodies' order
  const repack = () => {

    nextSlot = 0;
    for ( const body of bodies ) {

      body.slot = nextSlot;
      nextSlot += specs[ body.kind ].points.length;

    }

  };

  const spawn = ( kind: Kind, x: number, y: number, z: number ) => {

    if ( bodies.length >= MAX_BODIES ) remove( bodies[ 0 ] );
    const spec = specs[ kind ];
    const rotation = new THREE.Quaternion().setFromEuler( new THREE.Euler( random() * Math.PI, random() * Math.PI, random() * Math.PI ) );
    if ( kind === 'buoy' ) rotation.identity();
    const rigid = world.createRigidBody( RAPIER.RigidBodyDesc.dynamic()
      .setTranslation( x, y, z ).setRotation( rotation )
      .setAngvel( { x: ( random() - 0.5 ) * 2, y: ( random() - 0.5 ) * 2, z: ( random() - 0.5 ) * 2 } )
      .setLinearDamping( 0.05 ).setAngularDamping( 0.4 ).setCanSleep( false ) );
    world.createCollider( spec.collider().setFriction( 0.6 ).setRestitution( 0.15 ), rigid );
    const mesh = new THREE.Mesh( spec.geometry, spec.material );
    mesh.name = kind;
    mesh.castShadow = mesh.receiveShadow = true;
    group.add( mesh );
    opts.onMesh( mesh );
    const water = new Float32Array( spec.points.length * 4 );
    for ( let k = 0; k < spec.points.length; k ++ ) water[ k * 4 ] = sea;
    bodies.push( { kind, rigid, mesh, slot: 0, since: probe.sent, water } );
    repack();

  };

  const remove = ( body: Body ) => {

    world.removeRigidBody( body.rigid );
    group.remove( body.mesh );
    bodies.splice( bodies.indexOf( body ), 1 );
    repack();

  };

  // the batch in flight's layout: which body held which slots when it was sent
  let sentLayout: { body: Body; slot: number }[] = [];
  const takeResult = ( batch: number ) => {

    for ( const { body, slot } of sentLayout ) {

      if ( ! bodies.includes( body ) || batch <= body.since ) continue;
      body.water.set( probe.result.subarray( slot * 4, ( slot + specs[ body.kind ].points.length ) * 4 ) );

    }

  };

  const sendPoints = () => {

    if ( ! probe.ready || bodies.length === 0 ) return;
    let n = 0;
    const layout: { body: Body; slot: number }[] = [];
    for ( const body of bodies ) {

      layout.push( { body, slot: n } );
      const t = body.rigid.translation(), r = body.rigid.rotation();
      q.set( r.x, r.y, r.z, r.w );
      for ( const local of specs[ body.kind ].points ) {

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
  const applyWater = () => {

    for ( const body of bodies ) {

      const spec = specs[ body.kind ];
      const share = spec.volume / spec.points.length;
      const t = body.rigid.translation(), r = body.rigid.rotation();
      q.set( r.x, r.y, r.z, r.w );
      body.rigid.resetForces( true );
      body.rigid.resetTorques( true );
      let wet = 0;
      spec.points.forEach( ( local, k ) => {

        p.copy( local ).applyQuaternion( q ).add( t as THREE.Vector3Like );
        const height = body.water[ k * 4 ];
        // how much of the point's share is under the surface, 0 to 1
        const f = THREE.MathUtils.clamp( ( height - p.y ) / ( 2 * spec.extent ) + 0.5, 0, 1 );
        if ( f <= 0 ) return;
        wet += f;
        const displaced = WATER_DENSITY * share * f;
        // the point's own velocity: the body's plus its spin's
        const v = body.rigid.velocityAtPoint( p );
        const rx = v.x - body.water[ k * 4 + 1 ], ry = v.y - body.water[ k * 4 + 2 ], rz = v.z - body.water[ k * 4 + 3 ];
        force.x = - displaced * DRAG.horizontal * rx;
        force.y = displaced * G - displaced * DRAG.vertical * ry;
        force.z = - displaced * DRAG.horizontal * rz;
        at.x = p.x; at.y = p.y; at.z = p.z;
        body.rigid.addForceAtPoint( force, at, true );

      } );
      // water damps spin far more than air
      body.rigid.setAngularDamping( wet > 0 ? 1.2 : 0.2 );

    }

  };

  const sync = () => {

    for ( const body of bodies ) {

      const t = body.rigid.translation(), r = body.rigid.rotation();
      body.mesh.position.set( t.x, t.y, t.z );
      body.mesh.quaternion.set( r.x, r.y, r.z, r.w );
      body.mesh.updateMatrixWorld();

    }

  };

  const DROP_KINDS: Kind[] = [ 'barrel', 'barrel', 'crate', 'crate', 'crate' ];
  // the first wreckage, scattered round the stack, already afloat
  for ( let i = 0; i < 14; i ++ ) {

    const a = random() * Math.PI * 2, r = 16 + random() * 22;
    spawn( DROP_KINDS[ i % DROP_KINDS.length ], Math.cos( a ) * r, sea + 0.3, Math.sin( a ) * r );

  }
  spawn( 'buoy', 26, sea, - 14 );
  spawn( 'buoy', - 30, sea, 18 );
  sync();

  return {
    get count() {

      return bodies.length;

    },
    stats() {

      return {
        sent: probe.sent, received: probe.received,
        bodies: bodies.map( ( body ) => {

          const t = body.rigid.translation(), n = specs[ body.kind ].points.length;
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
        world.step();
        accumulator -= STEP;

      }
      for ( const body of [ ...bodies ] ) {

        const t = body.rigid.translation();
        if ( Math.hypot( t.x, t.z ) > DRIFT_LIMIT || t.y < sea - 40 ) remove( body );

      }
      sync();

    },
    drop( x: number, z: number ) {

      spawn( DROP_KINDS[ Math.floor( random() * DROP_KINDS.length ) ], x, sea + 6 + random() * 3, z );
      sync();

    },
  };

}
