import * as THREE from 'three/webgpu';
import { float, uniform, vec3 } from 'three/tsl';
import { createWindUniforms, windResponse } from './weather/Wind.ts';
import { displacementUniforms } from './vegetation/Displacement.ts';
import { ABSORPTION, BACKSCATTER, FORWARD_ANGLE, FORWARD_SCATTER, SCATTER } from './ocean/SeaOptics.ts';
import type { Look } from './look/Look.ts';
import { onSceneDispose } from './SceneDispose.ts';

// Uniforms shared by the water, the terrain and the vegetation, installed by one scene at a time.
//
// Uniforms shared by the water, the terrain and the vegetation (one node each, read by every
// material that uses it).
// - The app's entry module creates the object (`createShared`) and installs it (`installShared`) before
//   it builds anything: no module makes the uniforms as it loads, and a tool that builds materials
//   under node installs its own (tools/render/wgsl-build.mjs). The modules that take it as a parameter
//   are handed the same object; `shared` below is the installed one, for those that read it as
//   they build or run. Nothing reads it as a module loads.
// - One scene holds it at a time: a second install throws until the scene that installed it is
//   disposed (SceneDispose.ts), and a scene built after it then installs its own. A read before
//   the install throws too, at the read, naming the field
//   (its stack names the reader): unlike the settings and the look, the uniforms have no defaults
//   a reader could take instead.
// - The look's numbers every material's shader reads take their tables here, as an argument
//   (look/Look.ts; the entry hands over the set it installs): the ambient occlusion's floor, the
//   meadow canopy's media and the wind's look, which the wind's states carry (weather/Wind.ts
//   `windResponse`). So two objects made from two sets hold their own
//   (tools/render/verify-look-params.mjs).
export const createShared = ( look: SharedTables ) => ( {
	time: uniform( 0 ),
	wind: createWindUniforms( windResponse( look.wind ) ),
	// the displacement field the islanders part the meadow through (world/Trample.ts): the
	// base nodes the blades sample it by (vegetation/Displacement.ts)
	displacement: displacementUniforms,
	// the trees' leaves' albedo against their atlas (vegetation/TreeSurfaces.ts leafMaterial:
	// the bamboo's crowns); 1 the atlas as it is: the leaves about 1.5 times the meadow's albedo
	// (about 0.12), so a crown parts from the sward under it under overcast and by night
	leafGain: uniform( 1 ),
	// the weather's share of a cloudy or overcast sky: 0 under a clear or fair one, 1 under a closed one
	clouded: uniform( 0 ),
	// what the sward's ambient diffuse is multiplied by (materials/SwardTint.ts); white under a clear
	// or fair sky
	swardTint: uniform( new THREE.Vector3( 1, 1, 1 ) ),
	rain: uniform( 0 ),
	wetness: uniform( 0 ),
	waterLevel: uniform( 0 ),
	// The sea's inherent optical properties, per channel (1 / m), from ocean/SeaOptics.ts (its
	// sources there): the absorption `sigA` and the backscattering coefficient `bb`. The same
	// coefficients serve the light under the water (Underwater.ts), the water body (ocean/Water.ts),
	// the floor's bounce (GroundBounce.ts) and the ambient probes (world/AmbientProbes.ts).
	sigA: uniform( new THREE.Vector3( ...ABSORPTION ) ),
	bb: uniform( new THREE.Vector3( ...BACKSCATTER ) ),
	// the total scattering coefficient `scatter`, and the particles' forward scattering that blurs the
	// floor seen through the water, its coefficient and rms angle along one axis (ocean/Water.ts)
	scatter: uniform( new THREE.Vector3( ...SCATTER ) ),
	forwardScatter: uniform( FORWARD_SCATTER ),
	forwardAngle: uniform( FORWARD_ANGLE ),
	sunIntoWater: uniform( 1 ), // 1 - Fresnel of the sun at the surface
	// Storm strength: whitecaps on breaking crests (0 calm).
	storm: uniform( 0 ),
	muSun: uniform( 1 ), // cosine of the refracted sun below the surface
	shoreXform: uniform( new THREE.Vector4() ), // x0, z0, 1 / width, 1 / depth of the shoreline field
	sunDir: uniform( new THREE.Vector3( 0, 1, 0 ) ),
	// the sky's key, linear rgb (RenderStages.ts publishLighting): the sea's and the water's, which the
	// clouds' own transmittance over the land's floor (Sky.ts `seaKeyVisibility`) cuts per point
	// where it is read
	sunColor: uniform( new THREE.Vector3() ),
	// the key on the land (RenderStages.ts publishLighting): sunColor, the clouds' shadow over the land's
	// floor (Sky.ts `keyDirectVisibility`, the deck's key gain on that floor) cutting it per point
	// where it is read
	landKey: uniform( new THREE.Vector3() ),
	skyIrradiance: uniform( new THREE.Vector3() ), // linear rgb
	// The live sky reduces hemispherical irradiance on the GPU, so shading never waits for
	// a CPU readback. The uniform above remains a readback diagnostic for explicit snapshots.
	skyFill: uniform( 1 ),
	// a low sun's lift of the sky's light on the water: the sea's bounce onto the land (the probes'
	// `seaScale`), the sky's light in the water (`skyIrradianceSource`) and the sea's reflection of the
	// sky, never the land's own sky light, the sky drawn or its meter (lighting/AmbientRelight.ts
	// `lowSunSkyLift`); 1 from its top up and under cloud.
	// Whatever else reads `skyIrradianceSource` takes it too: the rain's light (RenderStages.ts
	// `rain.setLightNode`), the storm's spray (StormSea's `sky`) and its whitecaps (ocean/Water.ts).
	// Those draw only under rain and storm, which count as overcast (1), so only through a front from
	// a clear or fair sky into them under a low sun, by the share of the clear sky left
	skyLift: uniform( 1 ),
	// the night sky as the dome draws it, a share of the light it sheds (sky/Sky.ts `track`,
	// Weather.ts `seenNight`): 1 by day. Its readers: the far sea's sky reflection as it meets the
	// horizon (ocean/Water.ts `SEEN_SKY`), the far air's moon's sky and deck light
	// (weather/VolumetricHaze.ts `prepareMedium`, `SEEN_AIR`) and the exposure's meter on the sky's
	// pixels (post/Metering.ts)
	seenNight: uniform( 1 ),
	skyIrradianceSource: vec3( 0 ) as THREE.Node<'vec3'>,
	// The same as lent to the land after sunset (Sky.landIrradianceNode): the figures' fill
	// under `flags=ambientVolume:0` only (the ambient volume lights them otherwise). It is the probe's
	// panorama, which takes the fog layer's haze as the dome does, not the land's source, which
	// leaves the fog layer out (Sky.ts `fogEnvironment`).
	skyFillIrradiance: vec3( 0 ) as THREE.Node<'vec3'>,
	// log2 of the sky's geometric-mean luminance (Sky.meterLog2Node, scene units): the exposure's
	// sky term, the dome's metering illuminance being this luminance's
	skyMeterLog2: float( 0 ) as THREE.Node<'float'>,
	// the least ambient occlusion a surface keeps where its own and the screen's multiply (the
	// look's floor: post/GTAO.ts `floorOcclusion`), a number, which the shaders take as a constant
	aoFloor: look.occlusion.floor,
	// the media the key crosses inside a meadow (lighting/MeadowCanopy.ts): the level canopy over
	// every point of it and the crowns that stand over their neighbours, each its extinction per
	// metre along the key's ray and the share of the key that passes between the blades whatever the
	// path (the look's)
	meadowCanopy: { density: uniform( look.meadowCanopy.canopy.density ), open: uniform( look.meadowCanopy.canopy.open ) },
	meadowCast: { density: uniform( look.meadowCanopy.cast.density ), open: uniform( look.meadowCanopy.cast.open ) },
} );

/** The look's tables the shared object takes (`createShared`). */
export type SharedTables = Pick<Look, 'occlusion' | 'meadowCanopy' | 'wind'>;

export type Shared = ReturnType<typeof createShared>;

// what `shared` holds until the install: any field read throws
const NOT_INSTALLED = new Proxy( {} as Shared, {
	get: ( _, field ) => { throw new Error( `shared.${ String( field ) } was read before installShared: the entry installs the scene's uniforms before anything builds` ); },
} );

/** The installed uniforms (a read of any field throws until the entry installs them). */
export let shared: Shared = NOT_INSTALLED;

/** Installs the scene's uniforms before anything is built: an install while a scene holds them throws. */
export function installShared( created: Shared ): void {

	if ( shared !== NOT_INSTALLED ) throw new Error( 'installShared: a scene holds the uniforms: dispose it before another scene installs its own' );
	shared = created;

}
onSceneDispose( 'shared', () => { shared = NOT_INSTALLED; } );
