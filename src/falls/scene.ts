import * as THREE from 'three/webgpu';
import { texture } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  startScene, neutralBounceGround, neutralLattice, refractedGroup, showUnderwater, duskForce, nightGlowShare, DAY_PRESETS,
  type RenderParts,
} from '../render/index.ts';
import type { Look } from '../render/look/Look.ts';
import { browserHost } from '../render/browser.ts';
import { ISLAND_LOOK } from '../looks/ponlai/IslandLook.ts';
import { SEA } from '../looks/ponlai/Sea.ts';
import { presetButtons, readMuted, setUpPage, showStatus, soundButton } from '../page.ts';
import { CREST, HeightGrid, Valley, runLevel, upperZ, boulderGeometry, placeBoulders, valleyFields, valleyGeometry, valleyMaterial } from './valley.ts';
import { MAIN_FALLS, UPPER_FALLS, createRiver, type River } from './river.ts';
import { createFallsSound, type FallsSound } from './sound.ts';
import { createSalmon, type Salmon } from './salmon.ts';

// The waterfall: a river gorge where salmon leap at a falls, a taller falls above it, and the sound
// of it all. Ponlai's own look, but for the sea's distant swell, which a pool has none of.

// the pool's water: clear river water, a little stained (1 / m, per channel: the render pipeline's
// sea optics, ocean/SeaOptics.ts). Fewer particles than the sea's, so it is clearer; more dissolved
// matter, which takes the blue first, so the deep water is green.
const RIVER_WATER = {
  absorption: [ 0.32, 0.075, 0.12 ], backscatter: [ 0.0012, 0.0014, 0.0016 ], scatter: [ 0.06, 0.065, 0.07 ], forwardScatter: 0.05,
} as const;
// a pool has no distant swell: the look's swell given a breath of wind over a few metres
const LOOK: Look = { ...ISLAND_LOOK, sea: { ...SEA, swell: { ...SEA.swell, windSpeed: 0.3, fetch: 30 } } };
// the fields round the falls: their half size and cells a side (0.5 m cells)
const FIELD = { half: 64, cells: 256 };
// the camera orbits this point (over the pool, m)
const TARGET = new THREE.Vector3( 0, CREST.height * 0.6, 3 );
// whether the sound was turned off, kept for the next visit
const MUTED_KEY = 'waterfall.muted';

export async function startWaterfall() {

  setUpPage( 'waterfall', 'Drag to orbit · scroll to zoom' );
  const host = browserHost( { canvasHost: document.getElementById( 'app' )!, loading: document.getElementById( 'loading' ) } );
  const valley = new Valley();
  const grid = new HeightGrid( valley, 160, 1 );
  const boulders = placeBoulders( valley );
  let fields!: ReturnType<typeof valleyFields>;
  let fieldNodes!: { shore: THREE.TextureNode; floor: THREE.TextureNode };
  let world!: THREE.Group;
  let refracted!: THREE.ClippingGroup;
  let ground!: THREE.Mesh;
  let controls!: OrbitControls;
  let river: River | null = null;
  let salmon: Salmon | null = null;
  let sound: FallsSound | null = null;
  let sea = 0;

  const running = await startScene( host, { look: LOOK, defaults: { hour: DAY_PRESETS.noon - 1.5, dayLength: 2880 } }, {
    afterScenes( { shared } ) {

      shared.sigA.value.set( ...RIVER_WATER.absorption );
      shared.bb.value.set( ...RIVER_WATER.backscatter );
      shared.scatter.value.set( ...RIVER_WATER.scatter );
      shared.forwardScatter.value = RIVER_WATER.forwardScatter;

    },
    afterSky( { render, shared } ) {

      sea = shared.waterLevel.value;
      render.camera.position.set( - 9, sea + 4.2, 33 );
      render.camera.lookAt( TARGET.x, sea + TARGET.y, TARGET.z );

    },
    passes: ( { shared } ) => ( { rainCeiling: shared.waterLevel.value + 40, extraPasses: [] } ),
    ground( { shared } ) {

      fields = valleyFields( valley, { ...FIELD, sea, xform: shared.shoreXform.value } );
      fieldNodes = { shore: texture( fields.textures.shore ), floor: texture( fields.textures.floor ) };
      return { fields: fieldNodes, shoreData: fields.shoreData, bounceGround: neutralBounceGround( { ground: sea, albedo: [ 0.11, 0.12, 0.08 ] } ) };

    },
    afterBounce( { render } ) {

      world = new THREE.Group();
      world.name = 'scene';
      // the ground under the pool: the refraction's camera sees it below the waterline
      refracted = refractedGroup( sea );
      refracted.name = 'the valley';
      const material = valleyMaterial( render );
      ground = new THREE.Mesh( valleyGeometry( valley, grid, sea ), material );
      ground.name = 'valley';
      ground.castShadow = ground.receiveShadow = true;
      showUnderwater( ground );
      const stones = new THREE.Mesh( boulderGeometry( valley, grid, boulders, sea ), valleyMaterial( render, { stones: true } ) );
      stones.name = 'boulders';
      stones.castShadow = stones.receiveShadow = true;
      showUnderwater( stones );
      refracted.add( ground, stones );
      world.add( refracted );
      render.scene.add( world );
      world.updateMatrixWorld( true );

    },
    ambient: () => ( { ambientData: neutralLattice( { ground: sea, albedo: [ 0.11, 0.12, 0.08 ] } ), world: { group: world } } ),
    afterAmbient( { render } ) {

      // the gorge round the falls casts into the sun's map
      render.sunShadow.setCasters( new THREE.Box3( new THREE.Vector3( - 34, sea - 6, - 44 ), new THREE.Vector3( 34, sea + 22, 40 ) ) );

    },
    water: () => ( {
      fields: fieldNodes,
      landBox: { x0: - 40, z0: - 60, x1: 40, z1: 60 },
      seaShape: { openSea: { from: 1e5, to: 2e5 }, reflectorTop: 16 },
      // the pool reflects the gorge round it
      heldOpaque: [ new THREE.Box3( new THREE.Vector3( - 30, sea - 4, - 30 ), new THREE.Vector3( 30, sea + 16, 45 ) ) ],
    } ),
    async afterWater( { render, sizing } ) {

      sizing.resize();
      river = createRiver( render, valley, grid, boulders, sea );
      render.waterScene.add( ...river.meshes );
      const sprays = river.splashes;
      salmon = createSalmon( render, {
        sea, valley, boulders,
        onSplash: ( x, y, z, level, size ) => {

          sprays.splash( x, sea + y, z, sea + level, size );
          sound?.splash( x, sea + y, z, size );

        },
      } );
      world.add( salmon.mesh );
      sound = createFallsSound( render.camera, {
        main: [ - 4, 0, 4 ].map( ( x ) => new THREE.Vector3( x, sea + MAIN_FALLS.lip * 0.5, CREST.bend * x * x + 0.8 ) ),
        upper: new THREE.Vector3( 0, sea + UPPER_FALLS.plunge + 1.5, upperZ( 0 ) + 2.5 ),
        run: [ new THREE.Vector3( - 2, sea + runLevel( - 6 ), - 6 ), new THREE.Vector3( 2, sea + runLevel( - 12 ), - 12 ) ],
        tail: new THREE.Vector3( 1, sea + 0.3, 22 ),
      }, readMuted( MUTED_KEY ) );
      soundButton( sound, MUTED_KEY );
      controls = orbit( render, sea );
      ( window as unknown as { __FALLS__: unknown } ).__FALLS__ = { controls, camera: render.camera, valley, grid, boulders, ground, river, salmon, sound };

    },
    frame: ( { render, shared } ) => {

      let night = 0;
      return {
        content: () => ( {
          localLights: () => night,
          afterDay() {

            controls?.update();
            // never under the ground or the water
            const camera = render.camera.position;
            const floorY = sea + Math.max( grid.at( camera.x, camera.z ), valley.levelAt( camera.x, camera.z ) ) + 0.8;
            if ( camera.y < floorY ) camera.y = floorY;

          },
          afterWeather( dt: number ) {

            const { sky } = render;
            night = duskForce( 'nightGlow', nightGlowShare( sky.dayWindows, sky.elevation, sky.rising, shared.clouded.value ) );
            if ( salmon ) {

              salmon.update( dt, night );
              showStatus( `${ salmon.leaps } leaps · ${ salmon.made } over` );

            }
            sound?.update( dt, sea );

          },
        } ),
      };

    },
  } );

  presetButtons( running.render );
  return running;

}

/** Drag to orbit the pool, scroll to zoom. */
function orbit( render: RenderParts, sea: number ): OrbitControls {

  const controls = new OrbitControls( render.camera, render.renderer.domElement );
  controls.target.set( TARGET.x, sea + TARGET.y, TARGET.z );
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 4;
  controls.maxDistance = 70;
  controls.maxPolarAngle = Math.PI * 0.55;
  controls.update();
  return controls;

}
