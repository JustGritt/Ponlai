import type * as THREE from 'three/webgpu';
import { Fn, If, modelWorldMatrix, positionPrevious, positionWorld, select, uniform, varyingProperty, vec4 } from 'three/tsl';
import { TEMPORAL } from '../post/Antialiasing.ts';
import { motionNode, velocityOutput } from '../post/Motion.ts';
import { windDisplacement, type WindUniforms } from '../weather/Wind.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

// The vegetation's shadows: cast and received in the plants' swayed pose.
//
// The plants cast their shadows in their swayed pose, so the shadows move with them, and receive
// them in the same pose: a leaf cast in one pose and looked up in another would put its own shadow
// beside itself (acne and flicker as it sways). `flags.shadowWind = false` casts and receives in the
// rest pose instead (see below), which keeps thin leaves from moving across shadow texels.
// Returns the rest pose in world space for any other light term that must stay as still.
// castMask (per vertex, 0 or 1) keeps parts out of the shadow map: stems and stalks are a few
// millimetres across, well under a texel, so the map could only show them as dashes that come
// and go with each redraw; the sun's own disk would blur such a thin shadow away within a few
// decimetres anyway. A masked vertex collapses to the origin, so its triangles have no area;
// keep the mask the same on all three corners (one flag per part), or the triangle stretches.
//
// Motion vectors (the temporal modes, packages/render/post/Motion.ts): `previousPosition` is where the
// animated vertex was in the previous rendered frame, in the same space (the wind's sway evaluated
// with `shared.wind.previous`, `previousSway`). The shaded draw assigns it to three's
// `positionPrevious` in its position stage, after the rest inputs are read and before three moves
// `positionLocal` to the swayed pose, and writes its velocity as class vegetation. Only the
// material itself does: the vegetation prepass's depth copies share this position node but write
// no motion (VegetationPrepass.ts), the shadow passes write no velocity, and a draw without a
// velocity target needs no previous data, so none of them evaluates the sway twice. Without the
// temporal modes nothing is built for it (MSAA's shaders are unchanged).
// `flags.mv.wind = false` (WindMotion.ts sets `userData.windMotion = false`) builds the material
// with the current pose as the previous one: no wind motion and no second sway, the A/B of its
// cost and the MV-UNIT check's "source off".
//
// The pose is a switch: each caster's shadow position, and the position each receiver looks its
// shadow up at, take the swayed pose while `shadowWindOn` is 1, which SunShadow.ts sets every frame from `flags.shadowWind` (on unless false,
// debug/DebugFlags.ts) and which also makes it redraw the map every frame. Both poses share one
// pipeline, so --toggle can time the swayed shadows within a page. The rest pose stays in
// `userData.castRest` for the passes that must keep it whatever the flag (the rain's collision map,
// RainCollision.ts).
export const shadowWindOn = uniform( 1 );

export function useRestPoseShadows( material: THREE.NodeMaterial, restPosition: V3, animatedPosition: V3, castMask?: F, previousPosition?: V3 ): V3 {

	const restWorld = varyingProperty( 'vec3', 'vVegetationShadowRestWorld' );
	const previous = TEMPORAL ? previousPosition : undefined;
	// This is a material-local vertex stack, not a shared layout function: it writes a varying
	// and uses this object's matrix. Rest positions already include the instance transform.
	material.positionNode = Fn( ( builder: THREE.NodeBuilder ) => {

		const rest = restPosition.toVar();
		restWorld.assign( modelWorldMatrix.mul( vec4( rest, 1 ) ).xyz );
		const b = builder as unknown as { material: THREE.Material | null; needsPreviousData(): boolean };
		if ( previous && b.material === material && b.needsPreviousData() ) {

			positionPrevious.assign( material.userData.windMotion === false ? animatedPosition : previous );

		}
		return animatedPosition;

	} )();
	const castRest = castMask ? restPosition.mul( castMask ) : restPosition;
	material.userData.castRest = castRest;
	material.castShadowPositionNode = Fn( () => {

		const cast = restPosition.toVar();
		If( shadowWindOn.greaterThan( 0.5 ), () => {

			cast.assign( animatedPosition );

		} );
		return castMask ? cast.mul( castMask ) : cast;

	} )();
	// both poses' world positions are varyings already; the uniform picks one (no branch)
	material.receivedShadowPositionNode = select( shadowWindOn.greaterThan( 0.5 ), positionWorld, restWorld );
	if ( previous ) {

		material.userData.motion = 'animated';
		material.userData.previousPosition = true;
		material.mrtNode = velocityOutput( motionNode( 'vegetation' ) );
		// three's cache key leaves out userData: the build without the wind's motion needs its own
		const key = material.customProgramCacheKey;
		material.customProgramCacheKey = function ( this: THREE.NodeMaterial ) {

			return key.call( this ) + ( this.userData.windMotion === false ? '|windStill' : '' );

		};

	} else delete material.userData.previousPosition; // (a clone of a material that had one)
	// Renderer caches shadow overrides by the source material's version.
	material.needsUpdate = true;
	return restWorld;

}

/** Under the temporal modes, the material's registered wind displacement (`setWindDisplacement`)
 * evaluated with the previous rendered frame's wind: the `previousPosition` of
 * `useRestPoseShadows`. Undefined otherwise, or for a material without one (the old sway). */
export function previousSway( material: THREE.Material, wind: WindUniforms ): V3 | undefined {

	const displacement = windDisplacement( material );
	return TEMPORAL && displacement ? displacement( wind.previous ) : undefined;

}
