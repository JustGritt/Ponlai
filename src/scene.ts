import * as THREE from 'three/webgpu';
import { texture } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  startScene, neutralBounceGround, neutralSeaLattice, refractedGroup, showUnderwater, duskForce, nightGlowShare, DAY_PRESETS,
  type DayPreset, type RenderParts,
} from './render/index.ts';
import { browserHost } from './render/browser.ts';
import { ISLAND_LOOK } from './looks/ponlai/IslandLook.ts';
import { STACK, stackFields, stackGeometry, stackMaterial } from './stack.ts';
import { createLighthouse, type Lighthouse } from './lighthouse.ts';
import { WaveProbe } from './waveProbe.ts';
import { createFloaters, type Floaters } from './floaters.ts';
import { albedo, surface } from './surfaces.ts';

// The scene: a sea stack with a lighthouse, far out in deep water, wreckage floating round it.
// Ponlai's own look (exposure, haze, grading) is used as the island has it.

// the fields round the stack: their half size and cells a side (1 m cells)
const FIELD = { half: 96, cells: 192 };
// the camera orbits this point (m above the sea)
const TARGET_HEIGHT = 17;

export async function startProject() {

  const host = browserHost( { canvasHost: document.getElementById( 'app' )!, loading: document.getElementById( 'loading' ) } );
  let fields!: ReturnType<typeof stackFields>;
  let fieldNodes!: { shore: THREE.TextureNode; floor: THREE.TextureNode };
  let world!: THREE.Group;
  let refracted!: THREE.ClippingGroup;
  let stack!: THREE.Mesh;
  let lighthouse!: Lighthouse;
  let floaters: Floaters | null = null;
  let controls!: OrbitControls;
  let sea = 0;

  const running = await startScene( host, { look: ISLAND_LOOK, defaults: { hour: DAY_PRESETS.dusk, dayLength: 2880 } }, {
    afterSky( { render, shared } ) {

      sea = shared.waterLevel.value;
      render.camera.position.set( 70, sea + 10, 66 );
      render.camera.lookAt( 0, sea + TARGET_HEIGHT, 0 );

    },
    passes: ( { shared } ) => ( { rainCeiling: shared.waterLevel.value + 40, extraPasses: [] } ),
    ground( { shared } ) {

      fields = stackFields( { ...FIELD, sea, xform: shared.shoreXform.value } );
      fieldNodes = { shore: texture( fields.textures.shore ), floor: texture( fields.textures.floor ) };
      return { fields: fieldNodes, shoreData: fields.shoreData, bounceGround: neutralBounceGround( { ground: sea + STACK.floor, albedo: [ 0.2, 0.2, 0.18 ] } ) };

    },
    afterBounce( { render } ) {

      world = new THREE.Group();
      world.name = 'scene';
      // what reaches under the sea: the refraction's camera sees it below the waterline
      refracted = refractedGroup( sea );
      refracted.name = 'under and over the sea';
      stack = new THREE.Mesh( stackGeometry( sea ), stackMaterial( render, sea ) );
      stack.name = 'sea stack';
      stack.castShadow = stack.receiveShadow = true;
      showUnderwater( stack );
      refracted.add( stack );
      const floor = new THREE.Mesh( new THREE.CircleGeometry( 600, 64 ), surface( render, { name: 'sea floor', color: albedo( 0x6a6656 ), roughness: 0.95 } ) );
      floor.rotation.x = - Math.PI / 2;
      floor.position.y = sea + STACK.floor;
      floor.name = 'sea floor';
      floor.receiveShadow = true;
      showUnderwater( floor );
      refracted.add( floor );
      lighthouse = createLighthouse( render, new THREE.Vector3( - 0.6, sea + STACK.top - 0.15, - 0.8 ) );
      world.add( refracted, lighthouse.group );
      render.scene.add( world );
      world.updateMatrixWorld( true );

    },
    ambient: () => ( { ambientData: neutralSeaLattice( { sea } ), world: { group: world } } ),
    afterAmbient( { render } ) {

      // what casts into the sun's shadow map: the stack, the lighthouse and the water round them
      const casters = new THREE.Box3().setFromObject( stack ).union( lighthouse.bounds() );
      casters.min.y = sea - 2;
      casters.expandByVector( new THREE.Vector3( 14, 0, 14 ) );
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
      // the sea reflects the wreckage within this box
      heldOpaque: [ new THREE.Box3( new THREE.Vector3( - 70, sea - 1, - 70 ), new THREE.Vector3( 70, sea + 3, 70 ) ) ],
    } ),
    async afterWater( { render, sizing } ) {

      sizing.resize();
      const probe = new WaveProbe( render, fieldNodes );
      floaters = await createFloaters( render, {
        group: refracted, probe, sea, stack: stack.geometry,
        onMesh: ( mesh ) => showUnderwater( mesh ),
      } );
      controls = orbit( render, sea );
      dropOnClick( render, sea, ( x, z ) => floaters?.drop( x, z ) );
      // for the console and the tools: `__LIGHTHOUSE__.floaters.stats()`
      ( window as unknown as { __LIGHTHOUSE__: unknown } ).__LIGHTHOUSE__ = { floaters, probe, controls, camera: render.camera };

    },
    frame: ( { render, shared } ) => {

      let night = 0;
      return {
        content: () => ( {
          localLights: () => night,
          afterDay() {

            controls?.update();
            // never under the sea
            const floorY = sea + 1.2;
            if ( render.camera.position.y < floorY ) render.camera.position.y = floorY;

          },
          afterWeather( dt: number ) {

            const { sky } = render;
            night = duskForce( 'nightGlow', nightGlowShare( sky.dayWindows, sky.elevation, sky.rising, shared.clouded.value ) );
            lighthouse.night.value = night;
            lighthouse.update( shared.time.value );
            floaters?.update( dt );
            hud( floaters?.count ?? 0 );

          },
        } ),
      };

    },
  } );

  presets( running.render );
  return running;

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

let shownCount = - 1;
function hud( count: number ): void {

  if ( count === shownCount ) return;
  shownCount = count;
  const el = document.getElementById( 'count' );
  if ( el ) el.textContent = `${ count } afloat`;

}

/** The time-of-day buttons. */
function presets( render: RenderParts ): void {

  for ( const button of document.querySelectorAll<HTMLButtonElement>( '[data-preset]' ) ) {

    button.addEventListener( 'click', () => void render.setPreset( button.dataset.preset as DayPreset ) );

  }

}
