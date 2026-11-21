import * as THREE from 'three/webgpu';
import { Fn, luminance, property, vec3 } from 'three/tsl';
import { meterOutput } from '../post/Motion.ts';

type V3 = THREE.Node<'vec3'>;
type F = THREE.Node<'float'>;

// The metering illuminance a lit material writes, the exposure's input.
//
// The metering illuminance a lit material writes (post/Motion.ts `meterLight`, the exposure's
// input): M = lum( E ) / pi, the luminance a white Lambertian surface would show under the light
// falling on the point, E the RGB irradiance the lighting model shades with. It is gathered beside
// the colour from the same terms, never approximated apart:
// - the key: the light's irradiance as the model receives it (its shadow map, the screen-space
//   shade and the clouds already in it, under the water the refracted sun down the column) times
//   the cosine the model's diffuse takes (the lifted key's clamped N.L; a blade's wrap and light
//   through it, a leaf's thin layer in its Lambert's place, a feather's light through it, as each
//   model takes it: `meterKey`);
// - the local lights' illuminance (lighting/LocalLights.ts `shadeLocalLights`), beside the key;
// - the ambient: the irradiance the model's ambient diffuse reads (the ambient lights and the
//   ground's bounce, lighting/GroundBounce.ts, the environment's or the ambient volume's diffuse,
//   a leaf's thin layer under it), with the model's own scales, then times the occlusion the
//   ambient diffuse takes (lighting/AmbientGrade.ts: the material's and the screen's) and, under
//   the water, the column's attenuation (`meterAmbient`).
// Neither the albedo, the Fresnel weight on the diffuse, the specular, emission nor the haze. The
// two sums are fragment-stage properties (zero-initialised, set to zero again in the model's
// `start`), so they add no varying; the material's MRT node turns them into M through one laid-out
// function after the lighting. A material writes M once `writeMeter` marks it: its MRT node merged
// over the scene pass's (post/Motion.ts `meterOutput`), whatever MRT node it is given later (its
// velocity under the temporal modes). It is drawn only in passes with an MRT (the scene's and the
// refraction's, which drops the names its target lacks), as Motion.ts's note asks. The water writes
// its own (ocean/Water.ts: a share of the white card's, the key by level ground's lifted cosine) and the
// sky dome is the fog and sky's.

/** The key's and the local lights' irradiance on the point (rgb, scene units). */
export const meterKey = property( 'vec3', 'MeterKey' ) as unknown as V3;
/** The ambient irradiance on the point, the occlusion applied once the model's ambient is done. */
export const meterAmbient = property( 'vec3', 'MeterAmbient' ) as unknown as V3;

/** Clears both sums: at the start of a lighting model's `start`. */
export function meterStart(): void {

	meterKey.assign( vec3( 0 ) );
	meterAmbient.assign( vec3( 0 ) );

}

/** M from the two sums. */
const meterOf = /*@__PURE__*/ Fn( ( [ key, ambient ]: [ V3, V3 ] ) => luminance( key.add( ambient ) ).mul( 1 / Math.PI ) )
	.setLayout( { name: 'meterOf', type: 'float', inputs: [ { name: 'key', type: 'vec3' }, { name: 'ambient', type: 'vec3' } ] } );

const meter = meterOf( meterKey, meterAmbient ) as unknown as F;
const meterMRT = meterOutput( meter );

/** Makes `material` write its M (see the header): its MRT node, now or set later, merged with the
 * metering output. For a material lit by a model that fills the sums. */
export function writeMeter( material: THREE.NodeMaterial ): void {

	const own = material.mrtNode;
	let merged: THREE.MRTNode = own ? own.merge( meterMRT ) : meterMRT;
	Object.defineProperty( material, 'mrtNode', {
		configurable: true,
		enumerable: true,
		get: () => merged,
		set: ( value: THREE.MRTNode | null ) => {

			merged = value ? value.merge( meterMRT ) : meterMRT;

		},
	} );

}
