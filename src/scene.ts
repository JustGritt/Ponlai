import * as THREE from 'three/webgpu';
import { startScene, neutralFields, neutralBounceGround, neutralSeaLattice, OPEN_SEA_SHORE } from './render/index.ts';
import { browserHost } from './render/browser.ts';
import { NEUTRAL_LOOK } from './render/setup.ts';
import { ISLAND_LOOK as PONLAI_LOOK } from './looks/ponlai/IslandLook.ts';

// The look the scene starts from, one of two:
// - PONLAI_LOOK (recommended): Ponlai's own light, air and image values, as the island is drawn.
// - NEUTRAL_LOOK: every gain 1, no haze, bloom or grade: a plain start for a look of your own.
const LOOK = PONLAI_LOOK;

export async function startProject() {

  const host = browserHost( { canvasHost: document.getElementById( 'app' )!, loading: document.getElementById( 'loading' ) } );
  const rock = new THREE.Mesh( new THREE.DodecahedronGeometry( 2 ), new THREE.MeshStandardNodeMaterial( { color: 0x6b665e, roughness: 0.9 } ) );
  const world = new THREE.Group();
  world.add( rock );
  let fields!: ReturnType<typeof neutralFields>;

  return startScene( host, { look: LOOK, defaults: { hour: 15, dayLength: 2880 } }, {
    afterSky( { render } ) {

      render.camera.position.set( 8, 3, 10 );
      render.camera.lookAt( 0, 0.5, 0 );

    },
    passes: ( { shared } ) => ( { rainCeiling: shared.waterLevel.value + 8, extraPasses: [] } ),
    ground( { shared } ) {

      const floor = shared.waterLevel.value - 3;
      fields = neutralFields( { half: 64, shore: OPEN_SEA_SHORE, floor, xform: shared.shoreXform.value } );
      return { fields: fields.fields, shoreData: fields.shoreData, bounceGround: neutralBounceGround( { ground: floor, albedo: [ 0.3, 0.3, 0.3 ] } ) };

    },
    afterBounce( { render, shared } ) {

      rock.position.y = shared.waterLevel.value - 1;
      render.scene.add( world );
      world.updateMatrixWorld( true );

    },
    ambient: ( { shared } ) => ( { ambientData: neutralSeaLattice( { sea: shared.waterLevel.value } ), world: { group: world } } ),
    afterAmbient( { render } ) {

      render.sunShadow.setCasters( new THREE.Box3().setFromObject( rock ) );

    },
    water: () => ( {
      fields: fields.fields,
      landBox: { x0: - 2, z0: - 2, x1: 2, z1: 2 },
      seaShape: { openSea: { from: 1e5, to: 2e5 }, reflectorTop: 3 },
      heldOpaque: [],
    } ),
    afterWater( { sizing } ) {

      sizing.resize();

    },
  } );

}
