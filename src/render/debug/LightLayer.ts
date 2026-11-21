import * as THREE from 'three/webgpu';
import { cameraViewMatrix, diffuseColor, diffuseContribution, normalView, vec3, vec4 } from 'three/tsl';
import { debugFlags } from './DebugFlags.ts';
import { CLASSES, useClasses } from './Classes.ts';
import { diffuseOcclusion } from '../lighting/AmbientGrade.ts';

// The lighting layer's debug views, for measuring and tuning the light on its own.
//
// The lighting layer's debug views, for measuring and tuning the light on its own (debug builds,
// read at load from `flags=`):
//   lightOnly:1    every lit surface's diffuse albedo is 0.5 grey (the terrain above and under the
//                  water and the sea floor, the stones, the meadow's blades and flowers, the bamboo's
//                  culms and leaves, the yunpeng, the islanders, the game's props, the fish), so the
//                  image is the light itself: the radiance a grey card of albedo 0.5 returns there.
//                  The specular stays.
//   lightOnly:2    the same with the specular off too (direct and ambient).
//   grey:<albedo>  the grey's albedo under lightOnly instead of 0.5: `grey:0.18` turns every lit
//                  surface into an 18 % grey card, whose display values the display chain's
//                  exposure and curve are set against (Pipeline.ts, Exposure.ts).
//   ambientOnly    the key light (the sun or the moon) taken out of the lit materials, with the
//                  vegetation's transmitted light and the feathers' (all of them its
//                  direct terms), so only the sky's light and the ground's bounce remain (and the
//                  local lights, lighting/LocalLights.ts, which are not the key); combines with
//                  lightOnly.
//   opaqueVegetation  the vegetation lit as opaque surfaces: no light through it (`extra`, a leaf's
//                  thin layer), a blade's key unwrapped, its ambient unweighted by its translucency
//                  (vegetation/VegetationLighting.ts), so a leaf or blade takes only the diffuse light an
//                  opaque card in its place would; combines with the others.
//   albedoOnly     every lit surface drawn unlit in its own diffuse albedo (what its lighting takes,
//                  a darkening folded into it included), none of the light: the image is the
//                  albedo itself, so a material's albedo is read where it lies on screen (the turf
//                  between the meadow's blades: tools/island/meadow-light.mjs). Overrides the two above.
//   occlusionOnly  every lit surface drawn unlit in the occlusion its ambient diffuse takes, grey:
//                  the material's own (vertex AO, the turf's roots, a pocket's) times the screen's
//                  where it receives it, graded and floored (lighting/AmbientGrade.ts
//                  `diffuseOcclusion`), times the model's `ambientScale` (a plant's root-to-tip
//                  occlusion), so a cell's open sky is the one the light uses, not the screen's alone
//                  (the `occlusion` page of a script that reads cells). Overrides the three above.
//   normalOnly     every lit surface drawn unlit in its shading normal (world, as the lights take
//                  it: a blade's leaned and flattened one), encoded ( n + 1 ) / 2, so a pixel's
//                  multisampled resolve and a cell's mean hold the normals' mean unnormalised: its
//                  length says how far they agree (the `shading` page of a script that reads cells). Overrides all
//                  of the above.
//   classes        every lit surface drawn unlit in a flat colour naming its kind (debug/Classes.ts).
//                  Overrides all of the above.
// The water, the sky, the clouds and the haze are left as they are (they are not lit by these
// models), and so is what a material draws as emission (the yunpeng's glow). What a material folds
// into its albedo goes with it (the meadow's darkening toward its roots, a wet surface's darkening):
// those are surface colour, not light. The one exception is the sea's metering illuminance
// (ocean/Water.ts): `ambientOnly` writes it without its key term, so the meter's two terms read
// apart; the sea's colour is left as it is.
//
// One hook, `useLightLayer`, run by the constructor of the lighting model root every lit material
// builds on (GroundBounceLightingModel, which the terrain's, the vegetation's and the figures'
// models extend). It sets instance overrides of the
// model's methods, so a subclass's own `direct` is replaced as well, without a check in each. With
// no flag set it does nothing: the shaders, pipelines and frames are those without the module.
// Being load-time, a change of view reloads the page (and gives its own pipelines).

type V3 = THREE.Node<'vec3'>;
type Reflected = { directDiffuse: V3; directSpecular: V3; indirectDiffuse: V3; indirectSpecular: V3 };

const level = ( value: unknown ): number => value === true ? 1 : typeof value === 'number' ? value : 0;
/** 0 off, 1 grey albedo, 2 grey albedo and no specular. */
export const LIGHT_ONLY = Math.max( 0, Math.min( 2, Math.round( level( debugFlags.lightOnly ) ) ) ) as 0 | 1 | 2;
/** The key light out of the lit materials. */
export const AMBIENT_ONLY = level( debugFlags.ambientOnly ) > 0;
/** The vegetation lit as opaque surfaces, no light through it. */
export const OPAQUE_VEGETATION = level( debugFlags.opaqueVegetation ) > 0;
/** Every lit surface unlit in its own diffuse albedo. */
export const ALBEDO_ONLY = level( debugFlags.albedoOnly ) > 0;
/** Every lit surface unlit in its shading normal, world, as ( n + 1 ) / 2: the normal its lights
 * take, so a pixel's resolve averages its samples' normals unnormalised. */
export const NORMAL_ONLY = level( debugFlags.normalOnly ) > 0;
/** Every lit surface unlit in the occlusion its ambient diffuse takes. */
export const OCCLUSION_ONLY = level( debugFlags.occlusionOnly ) > 0;
// the grey card's albedo
const GREY = typeof debugFlags.grey === 'number' && debugFlags.grey > 0 ? Math.min( debugFlags.grey, 1 ) : 0.5;

/** Applies the lighting layer's debug views to a lighting model (see the header): called in the
 * constructor of each lighting model root, before the model builds. */
export function useLightLayer( model: THREE.LightingModel ): void {

	const m = model as unknown as {
		start( builder: THREE.NodeBuilder ): void;
		direct( input: unknown, builder?: THREE.NodeBuilder ): void;
		indirect( builder: THREE.NodeBuilder ): void;
	};
	if ( CLASSES ) {

		useClasses( model );
		return;

	}
	if ( NORMAL_ONLY ) {

		// as albedoOnly, the shading normal (view space, as the lights take it) turned to the world
		m.direct = () => {};
		m.indirect = ( builder ) => {

			const reflected = ( builder.context as unknown as { reflectedLight: Reflected } ).reflectedLight;
			reflected.directDiffuse.assign( vec4( normalView, 0 ).mul( cameraViewMatrix ).xyz.mul( 0.5 ).add( 0.5 ) );
			reflected.directSpecular.assign( vec3( 0 ) );
			reflected.indirectDiffuse.assign( vec3( 0 ) );
			reflected.indirectSpecular.assign( vec3( 0 ) );

		};
		return;

	}
	if ( OCCLUSION_ONLY ) {

		// as albedoOnly, the reflected term the occlusion three's `indirect` would have multiplied
		// the ambient diffuse by (its occlusion turn does not run: nothing else reads it)
		m.direct = () => {};
		m.indirect = ( builder ) => {

			const reflected = ( builder.context as unknown as { reflectedLight: Reflected } ).reflectedLight;
			const scale = ( model as unknown as { ambientScale?(): THREE.Node<'float'> | undefined } ).ambientScale?.();
			const occlusion = diffuseOcclusion( builder );
			reflected.directDiffuse.assign( vec3( scale ? occlusion.mul( scale ) : occlusion ) );
			reflected.directSpecular.assign( vec3( 0 ) );
			reflected.indirectDiffuse.assign( vec3( 0 ) );
			reflected.indirectSpecular.assign( vec3( 0 ) );

		};
		return;

	}
	if ( ALBEDO_ONLY ) {

		// no light at all: the ambient's turn (the last in LightingModel.start) leaves the albedo as
		// the one reflected term, which three sums after `start` returns
		m.direct = () => {};
		m.indirect = ( builder ) => {

			const reflected = ( builder.context as unknown as { reflectedLight: Reflected } ).reflectedLight;
			reflected.directDiffuse.assign( diffuseColor.rgb );
			reflected.directSpecular.assign( vec3( 0 ) );
			reflected.indirectDiffuse.assign( vec3( 0 ) );
			reflected.indirectSpecular.assign( vec3( 0 ) );

		};
		return;

	}
	// the key's terms are all added in `direct` (three calls it once per light, inside `start`)
	if ( AMBIENT_ONLY ) m.direct = () => {};
	if ( LIGHT_ONLY > 0 ) {

		// before the lights and the ambient read the albedo (three's physical model reads
		// diffuseContribution, the vegetation's transmission and the islanders' model diffuseColor)
		const start = m.start.bind( model );
		m.start = ( builder ) => {

			diffuseColor.rgb.assign( vec3( GREY ) );
			diffuseContribution.assign( vec3( GREY ) );
			start( builder );

		};

	}
	if ( LIGHT_ONLY === 2 ) {

		// the direct lights have run by the ambient's turn (LightingModel.start), and three sums the
		// reflected light only after `start` returns: zero both specular terms last
		const indirect = m.indirect.bind( model );
		m.indirect = ( builder ) => {

			indirect( builder );
			const reflected = ( builder.context as unknown as { reflectedLight: Reflected } ).reflectedLight;
			reflected.directSpecular.assign( vec3( 0 ) );
			reflected.indirectSpecular.assign( vec3( 0 ) );

		};

	}

}
