import * as THREE from 'three/webgpu';
import { texture } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  startScene, neutralBounceGround, neutralSeaLattice, refractedGroup, showUnderwater, duskForce, nightGlowShare, DAY_PRESETS,
  type RenderParts,
} from './render/index.ts';
import { browserHost } from './render/browser.ts';
import { ISLAND_LOOK } from './looks/ponlai/IslandLook.ts';
import { FLOOR, ROCKS, RockShape, STACK_TOP, rockFields, rockMaterial, shoreDistance } from './rocks.ts';
import { createLighthouse, type Lighthouse } from './lighthouse.ts';
import { WaveProbe } from './waveProbe.ts';
import { createFloaters, type Floaters } from './floaters.ts';
import { albedo, surface } from './surfaces.ts';
import { bowWreck, wreckMaterials, type WreckMaterials } from './wreckage.ts';
import { createSoundscape, type Soundscape, type SoundSources } from './sound.ts';
import { createBoat, plotRoute, EYE, type Boat } from './boat.ts';
import { createGulls, findPerches, type Gulls, type Orbit } from './gulls.ts';
import { graphicsControls, readSound, setUpPage, showStatus, soundControls, timeControls, viewControls } from './page.ts';

// The scene: a sea stack with a lighthouse among lesser rocks, far out in deep water, a wreck on the
// stack and its wreckage floating round it, a fishing boat working round the rocks, gulls over it
// all, and the sound of it all.
// Ponlai's own look (exposure, haze, grading) is used as the island has it.

// the fields round the stack: their half size and cells a side (1 m cells)
const FIELD = { half: 96, cells: 192 };
// the camera orbits this point (m above the sea)
const TARGET_HEIGHT = STACK_TOP - 1.5;
// how close (m) the camera may come to a rock's waterline below its top
const CAMERA_CLEARANCE = 2;
// whether the sound was turned off, kept for the next visit
// whether the sound was turned off and its volume, kept for the next visit under this prefix
const SOUND_KEY = 'lighthouse';
// where the ship struck: the angle about the stack of the point its stem hit (radians from +x toward
// +z; the first view looks at the stack from about 0.75), and how far the line it came in on turns
// from straight at the rock (a glancing blow, so the first view sees the hull from the side)
const WRECK = { angle: 0.95, glance: - 1.05 };
// where the boat starts on its route: the angle about the stack (it then comes round in front of the
// first view's stack, counter-clockwise seen from above)
const BOAT_START = 0.3;
// the HUD's help line round the lighthouse and aboard the ship
const HELP = { lighthouse: 'Drag to orbit · scroll to zoom · click the sea to drop wreckage', ship: 'Drag to look around · click the sea to drop wreckage' };
// aboard the ship: a person's field of view (degrees, vertical; the orbit's is a long lens), the near
// plane (m: the orbit's 1 m cut the boom open at the view's edge, as the plane's depth there is
// well under the distance), and the share of the boat's roll and pitch the view takes (a person's
// balance takes out the rest)
const SHIP_FOV = 55;
const SHIP_NEAR = 0.2;
const SHIP_TILT = 0.5;

export async function startProject() {

  setUpPage( HELP.lighthouse );
  const host = browserHost( { canvasHost: document.getElementById( 'app' )!, loading: document.getElementById( 'loading' ) } );
  const rocks = ROCKS.map( ( rock ) => new RockShape( rock ) );
  const main = rocks[ 0 ];
  let fields!: ReturnType<typeof rockFields>;
  let fieldNodes!: { shore: THREE.TextureNode; floor: THREE.TextureNode };
  let world!: THREE.Group;
  let refracted!: THREE.ClippingGroup;
  let rockMeshes!: THREE.Mesh[];
  let lighthouse!: Lighthouse;
  let floaters: Floaters | null = null;
  let materials!: WreckMaterials;
  let wreck!: ReturnType<typeof bowWreck>;
  let controls!: OrbitControls;
  let ship: Aboard | null = null;
  let sound: Soundscape | null = null;
  let boat: Boat | null = null;
  let gulls: Gulls | null = null;
  let sea = 0;

  const running = await startScene( host, { look: ISLAND_LOOK, defaults: { hour: DAY_PRESETS.dusk, dayLength: 2880 } }, {
    afterSky( { render, shared } ) {

      sea = shared.waterLevel.value;
      render.camera.position.set( 64, sea + 9, 60 );
      render.camera.lookAt( 0, sea + TARGET_HEIGHT, 0 );

    },
    passes: ( { shared } ) => ( { rainCeiling: shared.waterLevel.value + 40, extraPasses: [] } ),
    ground( { shared } ) {

      fields = rockFields( { rocks, ...FIELD, sea, xform: shared.shoreXform.value } );
      fieldNodes = { shore: texture( fields.textures.shore ), floor: texture( fields.textures.floor ) };
      return { fields: fieldNodes, shoreData: fields.shoreData, bounceGround: neutralBounceGround( { ground: sea + FLOOR, albedo: [ 0.2, 0.2, 0.18 ] } ) };

    },
    afterBounce( { render } ) {

      world = new THREE.Group();
      world.name = 'scene';
      // what reaches under the sea: the refraction's camera sees it below the waterline
      refracted = refractedGroup( sea );
      refracted.name = 'under and over the sea';
      const rockSurface = rockMaterial( render, sea );
      rockMeshes = rocks.map( ( rock ) => {

        const mesh = new THREE.Mesh( rock.geometry( sea ), rockSurface );
        mesh.name = rock.rock.name;
        mesh.castShadow = mesh.receiveShadow = true;
        showUnderwater( mesh );
        refracted.add( mesh );
        return mesh;

      } );
      const floor = new THREE.Mesh( new THREE.CircleGeometry( 600, 64 ), surface( render, { name: 'sea floor', color: albedo( 0x6a6656 ), roughness: 0.95 } ) );
      floor.rotation.x = - Math.PI / 2;
      floor.position.y = sea + FLOOR;
      floor.name = 'sea floor';
      floor.receiveShadow = true;
      showUnderwater( floor );
      refracted.add( floor );
      lighthouse = createLighthouse( render, new THREE.Vector3( - 0.6, sea + STACK_TOP - 0.15, - 0.8 ) );
      // the ship the storm drove onto the stack: its bow wedged against the rock
      materials = wreckMaterials( render, sea );
      const stem = new THREE.Vector3( Math.cos( WRECK.angle ), 0, Math.sin( WRECK.angle ) ).multiplyScalar( main.waterlineRadius( WRECK.angle ) - 0.3 ).setY( sea + 2.2 );
      wreck = bowWreck( materials, stem, Math.PI - WRECK.angle + WRECK.glance );
      wreck.group.traverse( showUnderwater );
      refracted.add( wreck.group );
      world.add( refracted, lighthouse.group );
      render.scene.add( world );
      world.updateMatrixWorld( true );

    },
    ambient: () => ( { ambientData: neutralSeaLattice( { sea } ), world: { group: world } } ),
    afterAmbient( { render, host } ) {

      host.status( 'Floating the wreckage…' );
      // what casts into the sun's shadow map: the stack, the lighthouse, the rocks near them and the
      // water round them (the far stacks shade only themselves)
      const casters = lighthouse.bounds();
      rockMeshes.forEach( ( mesh, i ) => {

        if ( Math.hypot( rocks[ i ].rock.x, rocks[ i ].rock.z ) < 36 ) casters.union( new THREE.Box3().setFromObject( mesh ) );

      } );
      casters.min.y = sea - 2;
      casters.expandByVector( new THREE.Vector3( 6, 0, 6 ) );
      render.sunShadow.setCasters( casters );
      // the lamp warms the gallery and the tower's top by night
      const lamp = lighthouse.lamp;
      render.localLights.add( {
        name: 'lighthouse', cost: 'none', testMode: 'stateless', land: 'independent',
        sources: () => [ { position: lamp.clone(), intensity: new THREE.Vector3( 6e-6, 5e-6, 3.5e-6 ), range: 16 } ],
      } );

    },
    water: () => ( {
      fields: fieldNodes,
      landBox: { x0: - 12, z0: - 12, x1: 12, z1: 12 },
      // no shelf: the floor's depth holds out to the horizon
      seaShape: { openSea: { from: 1e5, to: 2e5 }, reflectorTop: lighthouse.lamp.y + 4 },
      // the sea reflects the wreck, the wreckage and the rocks within these boxes
      heldOpaque: [
        new THREE.Box3( new THREE.Vector3( - 70, sea - 1, - 70 ), new THREE.Vector3( 70, sea + 3, 70 ) ), new THREE.Box3().setFromObject( wreck.group ),
        ...rockMeshes.slice( 1 ).map( ( mesh ) => new THREE.Box3().setFromObject( mesh ) ),
      ],
    } ),
    async afterWater( { render, sizing, host } ) {

      sizing.resize();
      const probe = new WaveProbe( render, fieldNodes );
      floaters = await createFloaters( {
        group: refracted, probe, sea, materials, fixed: [ ...rockMeshes.map( ( mesh ) => mesh.geometry ), wreck.collider ], wreckAngle: WRECK.angle,
        clear: ( x, z ) => shoreDistance( rocks, x, z ) > 2.5,
        onObject: ( object ) => object.traverse( showUnderwater ),
        onSplash: ( x, y, z, speed ) => sound?.splash( x, y, z, speed ),
      } );
      // the fishing boat on its route round the stack, clear of the rocks and the wreck
      const wreckBox = new THREE.Box3().setFromObject( wreck.group );
      const wreckAt = wreckBox.getCenter( new THREE.Vector3() ), wreckReach = Math.hypot( wreckBox.max.x - wreckBox.min.x, wreckBox.max.z - wreckBox.min.z ) / 2;
      const route = plotRoute( ( x, z ) => Math.min( shoreDistance( rocks, x, z ), Math.hypot( x - wreckAt.x, z - wreckAt.z ) - wreckReach ) );
      boat = createBoat( render, sea, route, BOAT_START );
      floaters.launch( boat.vessel );
      boat.update();
      const theBoat = boat;
      render.localLights.add( { name: 'boat', cost: 'none', testMode: 'stateless', land: 'independent', sources: () => theBoat.lights() } );
      // the gulls over the lantern, the stacks and the boat, standing on the rocks
      gulls = createGulls( render, {
        sea, orbits: gullOrbits( rocks, sea, boat ), perches: findPerches( rockMeshes.map( ( mesh ) => mesh.geometry ), sea ),
        ceiling: ceiling( rocks, sea, lighthouse.lamp, wreckBox, boat ),
      } );
      world.add( gulls.mesh );
      sound = createSoundscape( render.camera, soundSources( rocks, sea, wreck.group, gulls, boat ), readSound( SOUND_KEY ) );
      soundControls( sound, SOUND_KEY );
      controls = orbit( render, sea );
      ship = aboard( render, boat, controls );
      dropOnClick( render, sea, ( x, z ) => floaters?.drop( x, z ) );
      // for the console and the tools: `__LIGHTHOUSE__.floaters.stats()`
      ( window as unknown as { __LIGHTHOUSE__: unknown } ).__LIGHTHOUSE__ = {
        floaters, probe, controls, camera: render.camera, boat, route, gulls, shore: ( x: number, z: number ) => shoreDistance( rocks, x, z ),
      };
      // next: the haze's bake and every shader's first build, the longest wait on a phone
      host.status( 'Compiling shaders…' );

    },
    afterShared: ( { host } ) => host.status( 'Almost there…' ),
    // the sound, quiet while the page loaded, comes up as the scene shows
    onRevealed: () => sound?.reveal(),
    frame: ( { render, shared } ) => {

      let night = 0;
      return {
        content: () => ( {
          localLights: () => night,
          afterDay() {

            // aboard, the camera follows the boat once the physics has moved it (afterWeather)
            if ( ship?.on ) return;
            controls?.update();
            const camera = render.camera.position;
            // never under the sea, nor inside a rock
            const floorY = sea + 1.2;
            if ( camera.y < floorY ) camera.y = floorY;
            for ( const rock of rocks ) {

              const d = rock.distance( camera.x, camera.z );
              if ( d >= CAMERA_CLEARANCE || camera.y > sea + rock.top + CAMERA_CLEARANCE ) continue;
              const dx = camera.x - rock.rock.x, dz = camera.z - rock.rock.z, l = Math.hypot( dx, dz ) || 1;
              camera.x += dx / l * ( CAMERA_CLEARANCE - d );
              camera.z += dz / l * ( CAMERA_CLEARANCE - d );

            }

          },
          afterWeather( dt: number ) {

            const { sky } = render;
            night = duskForce( 'nightGlow', nightGlowShare( sky.dayWindows, sky.elevation, sky.rising, shared.clouded.value ) );
            lighthouse.night.value = night;
            lighthouse.update( shared.time.value );
            floaters?.update( dt );
            if ( boat ) {

              boat.night.value = night;
              boat.update();
              ship?.update();
              render.localLights.refresh();

            }
            gulls?.update( dt, night );
            sound?.update( dt, night, sea );
            showStatus( `${ floaters?.count ?? 0 } afloat` );

          },
        } ),
      };

    },
  } );

  viewControls( ( view ) => {

    ship?.board( view === 'ship' );
    return view === 'ship' ? HELP.ship : HELP.lighthouse;

  } );
  timeControls( running.render );
  graphicsControls( running.render, running.frame.pacer, sound );
  return running;

}

/** Where the scene's sounds come from: surf at each rock that has its own (round the main stack, at
 * four points), the gulls themselves, the boat's engine, the wreck's timbers. */
function soundSources( rocks: readonly RockShape[], sea: number, wreck: THREE.Object3D, gulls: Gulls, boat: Boat ): SoundSources {

  const surf: SoundSources[ 'surf' ] = [];
  for ( const shape of rocks ) {

    const { rock } = shape;
    if ( rock.surf <= 0 ) continue;
    // on the waterline: round the main stack, and on each other rock the side facing it (where the
    // camera mostly is)
    const angles = shape === rocks[ 0 ] ? [ 0.3, 1.9, 3.5, 5.0 ] : [ Math.atan2( - rock.z, - rock.x ) ];
    for ( const angle of angles ) {

      const r = shape.waterlineRadius( angle );
      surf.push( { position: new THREE.Vector3( rock.x + Math.cos( angle ) * r, sea + 0.5, rock.z + Math.sin( angle ) * r ), gain: rock.surf } );

    }

  }
  return {
    surf,
    gull: () => gulls.caller( Math.random ),
    boat,
    wreck: new THREE.Box3().setFromObject( wreck ).getCenter( new THREE.Vector3() ),
  };

}

/** The circles the gulls wheel round: high round the lantern, over the outer stacks and the stump,
 * and low over the boat, following it. */
function gullOrbits( rocks: readonly RockShape[], sea: number, boat: Boat ): Orbit[] {

  const above = ( name: string ) => {

    const shape = rocks.find( ( r ) => r.rock.name === name )!;
    return new THREE.Vector3( shape.rock.x, sea + shape.top + 6, shape.rock.z );

  };
  return [
    { centre: new THREE.Vector3( 0, sea + STACK_TOP + 10, 0 ), lift: [ - 2, 8 ], radius: [ 16, 30 ], weight: 3 },
    { centre: above( 'sister stack' ), lift: [ - 2, 5 ], radius: [ 9, 16 ], weight: 2 },
    { centre: above( 'far stack' ), lift: [ - 2, 5 ], radius: [ 8, 14 ], weight: 1 },
    { centre: above( 'stump' ), lift: [ 0, 6 ], radius: [ 8, 12 ], weight: 1 },
    { centre: boat.position, lift: [ 6, 11 ], radius: [ 6, 11 ], weight: 2.5, speed: 7.5 },
  ];

}

/** The least height a gull may fly at over x, z: above every rock near it, the tower, the wreck and
 * the boat's mast. */
function ceiling( rocks: readonly RockShape[], sea: number, lamp: THREE.Vector3, wreck: THREE.Box3, boat: Boat ): ( x: number, z: number ) => number {

  const wreckNear = wreck.clone().expandByScalar( 3 );
  return ( x, z ) => {

    let c = sea + 1.5;
    for ( const rock of rocks ) if ( rock.distance( x, z ) < 6 ) c = Math.max( c, sea + rock.top + 2.5 );
    if ( Math.hypot( x - lamp.x, z - lamp.z ) < 6 ) c = Math.max( c, lamp.y + 4 );
    if ( x > wreckNear.min.x && x < wreckNear.max.x && z > wreckNear.min.z && z < wreckNear.max.z ) c = Math.max( c, wreck.max.y + 2 );
    if ( Math.hypot( x - boat.position.x, z - boat.position.z ) < 6 ) c = Math.max( c, boat.position.y + 6.5 );
    return c;

  };

}

/** Drag to orbit the stack, scroll to zoom. */
function orbit( render: RenderParts, sea: number ): OrbitControls {

  const controls = new OrbitControls( render.camera, render.renderer.domElement );
  controls.target.set( 0, sea + TARGET_HEIGHT, 0 );
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 12;
  controls.maxDistance = 260;
  controls.maxPolarAngle = Math.PI * 0.53;
  controls.update();
  return controls;

}

interface Aboard {
  /** whether the camera is aboard */
  readonly on: boolean;
  /** boards the ship (true) or goes back to the orbit where it was left (false) */
  board( on: boolean ): void;
  /** aboard: the camera at the eye, after the boat has moved */
  update(): void;
}

/** The view aboard the ship: the camera at a person's eye on deck, moving with the boat, looking
 * ahead, turned by dragging. */
function aboard( render: RenderParts, boat: Boat, controls: OrbitControls ): Aboard {

  const { camera } = render;
  const canvas = render.renderer.domElement;
  const hull = boat.vessel.object;
  const orbitLens = { fov: camera.fov, near: camera.near }, left = new THREE.Vector3();
  // the look about the boat's heading: pitch (x, up positive) and yaw (y, to port positive)
  const look = new THREE.Euler( - 0.05, 0, 0, 'YXZ' ), turn = new THREE.Euler( 0, 0, 0, 'YXZ' );
  const tilt = new THREE.Quaternion(), level = new THREE.Quaternion(), view = new THREE.Quaternion(), ahead = new THREE.Vector3();
  const UP = new THREE.Vector3( 0, 1, 0 );
  let on = false, held: { x: number; y: number } | null = null;

  // a drag carries the view with the pointer, a canvas's height turning it by the field of view
  canvas.addEventListener( 'pointerdown', ( e ) => {

    if ( on ) held = { x: e.clientX, y: e.clientY };

  } );
  canvas.addEventListener( 'pointermove', ( e ) => {

    if ( ! held ) return;
    const k = THREE.MathUtils.degToRad( SHIP_FOV ) / Math.max( canvas.clientHeight, 1 );
    look.y += ( e.clientX - held.x ) * k;
    look.x = THREE.MathUtils.clamp( look.x + ( e.clientY - held.y ) * k, - 1.3, 1.3 );
    held = { x: e.clientX, y: e.clientY };

  } );
  for ( const type of [ 'pointerup', 'pointercancel' ] ) addEventListener( type, () => { held = null; } );

  return {
    get on() {

      return on;

    },
    board( value ) {

      if ( value === on ) return;
      on = value;
      controls.enabled = ! on;
      if ( on ) left.copy( camera.position );
      else camera.position.copy( left );
      camera.fov = on ? SHIP_FOV : orbitLens.fov;
      camera.near = on ? SHIP_NEAR : orbitLens.near;
      camera.updateProjectionMatrix();

    },
    update() {

      if ( ! on ) return;
      camera.position.copy( EYE ).applyMatrix4( hull.matrixWorld );
      hull.getWorldQuaternion( tilt );
      ahead.set( 1, 0, 0 ).applyQuaternion( tilt );
      level.setFromAxisAngle( UP, Math.atan2( - ahead.z, ahead.x ) );
      // the camera looks down its -z, the boat's bow is its +x
      turn.set( look.x, look.y - Math.PI / 2, 0 );
      camera.quaternion.copy( level.slerp( tilt, SHIP_TILT ) ).multiply( view.setFromEuler( turn ) );
      camera.updateMatrixWorld();

    },
  };

}

/** A click (not a drag) on the sea drops a piece of wreckage there. */
function dropOnClick( render: RenderParts, sea: number, drop: ( x: number, z: number ) => void ): void {

  const canvas = render.renderer.domElement;
  const raycaster = new THREE.Raycaster();
  const plane = new THREE.Plane( new THREE.Vector3( 0, 1, 0 ), - sea );
  const hit = new THREE.Vector3();
  let down: { x: number; y: number; t: number } | null = null;
  canvas.addEventListener( 'pointerdown', ( e ) => {

    down = { x: e.clientX, y: e.clientY, t: performance.now() };

  } );
  canvas.addEventListener( 'pointerup', ( e ) => {

    if ( ! down || Math.hypot( e.clientX - down.x, e.clientY - down.y ) > 6 || performance.now() - down.t > 400 ) return;
    const rect = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2( ( e.clientX - rect.left ) / rect.width * 2 - 1, - ( e.clientY - rect.top ) / rect.height * 2 + 1 );
    raycaster.setFromCamera( ndc, render.camera );
    if ( ! raycaster.ray.intersectPlane( plane, hit ) ) return;
    if ( Math.hypot( hit.x, hit.z ) > 110 ) return;
    drop( hit.x, hit.z );

  } );

}
