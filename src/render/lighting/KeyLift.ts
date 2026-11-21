import * as THREE from 'three/webgpu';
import { cameraViewMatrix, uniform, vec3, vec4 } from 'three/tsl';
import { debugFlags } from '../debug/DebugFlags.ts';
import { MOON, moonDirection } from '../sky/DayCycle.ts';
import type { KeyLiftShading } from '../look/AmbientLook.ts';

type V3 = THREE.Node<'vec3'>;

// The low key's lift: the direction a low key's light is taken from climbs above the key's own.
//
// The low key's lift: under a key low over the horizon, the direction the surfaces take its light
// from (every term of how a surface receives it) and the sun's shadows fall along climbs above the
// key's own. Under a true sun a few degrees high, level ground takes a twentieth of the light a
// surface facing the sun takes, and the sky lights its shade as much as the sun lights the rest:
// lit and shaded ground merge into one dim plane, with a lit to shade ratio under 3 on flat bare
// ground in the lighting layer at 3 degrees, and the shadows run out across the whole island. The
// golden hour's ground stays clearly lit, stained by the low sun's colour, under shadows a few
// times their casters' height. So the shading direction's elevation never falls
// below 14 degrees: it follows the key down to there, its corner rounded over a degree either side
// (`join`), holds at 14 while the key sinks to 6, climbs to 18 over the next degree (the golden
// hour, from the key at about 5.5: a monotone cubic over the lift's rows, one degree wide, so no
// step shows and no frame jumps) and holds at 18 down to the horizon, at the key's own azimuth.
// The shading elevation is a function of the key's elevation alone, so a time of day and a weather
// give one direction whichever way the clock ran, mornings as evenings.
// What moves to the lifted direction is everything about how a surface receives the key: the whole
// direct term (GroundBounceLightingModel `direct`: the diffuse, the specular with its Fresnel and
// the disk's representative point, lighting/SurfaceBRDF.ts), the vegetation's wrap and light
// through to the far side (VegetationLighting), the feathers' (characters/FigureSurfaces.ts), the
// terrain's terminator
// (lighting/Underwater.ts), and the shadows that bound them: the sun's map is drawn along the
// lifted direction (main.ts passes it to SunShadow, and the figures' map follows that map:
// characters/CharacterShadow.ts), and the meadow's layer shades its blades and the ground under
// them along it (`liftKeyWorld`, Grass.ts `meadowLayerDepth`), as the meadow's screen-space shadows
// do (post/MeadowShadows.ts). Lit faces, highlights and shadows so turn to one side together;
// with only the diffuse lifted, the dusk's highlights and back light still faced the grazing sun,
// apart from the ground's shading. What keeps the key's true direction is everything about where
// the light comes from: the key's colour and brightness (Sky.ts: the atmosphere's transmittance,
// dimmed under 6 degrees), the drawn sun, the sky and its harmonics, the aerial
// perspective, the clouds' shadows, the haze's scattering, the water's own glint (it must meet the
// drawn sun or moon), the refracted sun under the water and its map (UnderwaterLightingModel), and
// the bounces' directions (the ambient volume's lobes, GroundBounce.ts's map); the ambient volume
// weighs its bounce by the lifted cosine, the light level ground takes (AmbientRelight.ts). The
// haze's shadow is the sun's map, so its shafts follow the lifted shadows. Nothing else brightens
// the dusk: the rest is the exposure's. The metering illuminance a lit material writes
// (Pipeline.ts `meterLight`) is its incident light, the lifted cosine included, and the sea's
// takes the same cosine for its key term (ocean/Water.ts).
// The moon, the night's key, stands fixed in the sky (sky/DayCycle.ts MOON, `moonDirection`) and
// shades from its own direction for as long as it is the key, so the night keeps one shading and
// one level from dusk to dawn. It is drawn there, and the key's own direction is that one too
// (DayCycle `keyDir`), so everything that keeps the key's true direction above (the water's glint
// and so the moon's path on the sea, the refracted light under the water, the clouds, the haze)
// sees the moon where it is drawn: the disk, the glow of the air around it, its path on the water
// and the island's shading and shadows stand in one place. Over the handoff (from 0.7 to 2.5
// degrees under the horizon, through no key light) the key turns from the sun to that direction,
// and arrives there as the moon becomes the key: a key above the lift's floor shades from its own
// direction, so with the moon above 15 degrees (MOON's least) the shading and the shadows turn with
// it and nothing jumps when the moon takes over.
// The lifted direction is the key's L scale + U up + W side, U the world's up and W the moon's
// direction in view space, with `scale`, `up` and `side` set on the CPU from the key
// (`updateKeyLift`, every publish of the lighting, which reads the moon's direction from DayCycle
// first): with no lift they are 1, 0 and 0, and the lifted direction is the key's bit for bit;
// under a sun key side is 0, under the moon scale and up are 0 and side 1. The shadow's is
// `liftedKeyDirection`, the same direction in world space.
// Off with `flags=keyLift:false` (at run time from the next publish of the lighting): the sun's
// lift only, the moon stays where it stands. Ours.
// The rows and the shading elevation they give are a look's, made into the lift where the
// render pipeline's sky stage creates it (look/AmbientLook.ts `createKeyLift`, RenderStages.ts
// `createSky`); the sky carries it beside the key's direction (`Sky.keyLift`), and every function
// here that turns a key into its lifted direction takes it as its first argument. This file keeps
// the lifted direction and its uniforms, which no look changes.

const scale = uniform( 1 ), up = uniform( 0 ), side = uniform( 0 );
// the moon's direction (world, unit; DayCycle `moonDirection`), on the CPU and for the shaders,
// read again at every `updateKeyLift`
const MOON_SHADING_DIRECTION = moonDirection( new THREE.Vector3() );
const moonShadingDirection = uniform( MOON_SHADING_DIRECTION.clone() );

// how the key's world direction is lifted by the lift `shading`: under the sun its elevation and the
// lifted one (radians); under the moon its own direction (MOON_SHADING_DIRECTION); neither where nothing lifts it
function lifted( shading: KeyLiftShading, key: THREE.Vector3, moon: boolean ): { elevation: number; lifted: number; lift: boolean; moon: boolean } {

	const across = Math.hypot( key.x, key.z );
	const elevation = THREE.MathUtils.radToDeg( Math.atan2( key.y, across ) );
	const off = debugFlags.keyLift === false;
	if ( moon && ! off ) return { elevation: THREE.MathUtils.degToRad( elevation ), lifted: THREE.MathUtils.degToRad( MOON.elevation ), lift: true, moon: true };
	const shade = off || across === 0 ? elevation : shading( elevation );
	return { elevation: THREE.MathUtils.degToRad( elevation ), lifted: THREE.MathUtils.degToRad( shade ), lift: shade !== elevation, moon: false };

}

/** The sine of the elevation the surfaces take the key's light from under the key's world
 * direction `key` (the moon's if `moon`) and the lift `shading` (Sky.ts `keyLift`): the cosine
 * level ground takes (the ambient volume's key bounce, AmbientRelight.ts). None from a key below
 * the horizon, which lights no level ground. */
export function liftedKeyHeight( shading: KeyLiftShading, key: THREE.Vector3, moon = false ): number {

	return key.y > 0 ? Math.sin( lifted( shading, key, moon ).lifted ) : 0;

}

/** The key's world direction `key` (the moon's if `moon`) lifted by the lift `shading` (Sky.ts
 * `keyLift`), into `target`: the direction the surfaces take its light from and the sun's map is
 * drawn along (main.ts, SunShadow.ts); `key` itself where nothing lifts it. */
export function liftedKeyDirection( shading: KeyLiftShading, key: THREE.Vector3, moon: boolean, target: THREE.Vector3 ): THREE.Vector3 {

	const { lifted: l, lift, moon: fixed } = lifted( shading, key, moon );
	if ( fixed ) return target.copy( MOON_SHADING_DIRECTION );
	if ( ! lift ) return target.copy( key );
	const across = Math.cos( l ) / Math.hypot( key.x, key.z );
	return target.set( key.x * across, Math.sin( l ), key.z * across );

}

/** Sets the lift for the key's world direction, the moon's if `moon`, by the lift `shading` (Sky.ts
 * `keyLift`; RenderStages.ts `publishLighting`). */
export function updateKeyLift( shading: KeyLiftShading, key: THREE.Vector3, moon = false ): void {

	moonShadingDirection.value.copy( moonDirection( MOON_SHADING_DIRECTION ) );
	const { elevation: e, lifted: l, lift, moon: fixed } = lifted( shading, key, moon );
	if ( fixed ) {

		scale.value = 0;
		up.value = 0;
		side.value = 1;
		return;

	}
	side.value = 0;
	if ( ! lift ) {

		scale.value = 1;
		up.value = 0;
		return;

	}
	const k = Math.cos( l ) / Math.cos( e );
	scale.value = k;
	up.value = Math.sin( l ) - Math.sin( e ) * k;

}

/** The key's world direction L lifted: `liftedKeyDirection` on the GPU, the direction the sun's
 * map is drawn along (the meadow layer's shadowing, Grass.ts `meadowLayerDepth`, takes it, and the
 * sea's metering illuminance, ocean/Water.ts, takes its level-ground cosine from it). */
export function liftKeyWorld( L: V3 ): V3 {

	return L.mul( scale ).add( vec3( 0, up, 0 ) ).add( moonShadingDirection.mul( side ) );

}

/** The key's view-space direction L lifted: the direction the surfaces take its light from. */
export function liftKey( L: V3 ): V3 {

	const U = cameraViewMatrix.mul( vec4( 0, 1, 0, 0 ) ).xyz;
	const W = cameraViewMatrix.mul( vec4( moonShadingDirection, 0 ) ).xyz;
	return L.mul( scale ).add( U.mul( up ) ).add( W.mul( side ) );

}
