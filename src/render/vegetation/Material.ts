import * as THREE from 'three/webgpu';
import { Fn, materialColor, specularColor, specularColorBlended, vec3, vec4, mix, dot, clamp, float, smoothstep, uniform } from 'three/tsl';
import { setWindDisplacement, type WindDisplacement, type WindUniforms } from '../weather/Wind.ts';
import { showSway } from '../debug/WindView.ts';
import { followWindMotionFlag } from './WindMotion.ts';
import { VegetationLighting, useLighting } from './VegetationLighting.ts';
import { previousSway, useRestPoseShadows } from './VegetationShadow.ts';
import { ALPHA_TO_COVERAGE, coverageDither, markVegetation, useLeafCoverage } from './Coverage.ts';
import { LEAF_ALPHA_TEST, leafAlphaTexture } from './LeafTexture.ts';
import type { Shared } from '../shared.ts';
import type { AmbientReceiver } from '../lighting/AmbientSample.ts';
import { writeMeter } from '../lighting/MeterLight.ts';
import { tuneApply, tuneUniform } from '../debug/Tune.ts';

// The vegetation's material factory: every plant kind's surface built by one sequence.
//
// The vegetation's material factory: every plant kind's surface is built here by one sequence of
// calls, the one each kind used to write by hand, in the same order, so the node graphs (and the
// shaders, the pipelines and the image) are the ones the kinds built before:
//   the surface (a leaf's map, alpha map and cut-out, or a plain double-sided surface)
//   -> the vegetation tag (`markVegetation`: leaves and blades, the depth prepass and the shadow
//      filter read it) -> the leaf's coverage band under the temporal modes (`useLeafCoverage`)
//   -> wetness (a tint and a roughness toward the wet surface's)
//   -> the wind (`useVegetationMotion`: the displacement, the swayed shadow received at rest, the
//      previous sway for motion vectors, a kind's own vertex work before its position, the
//      `flags.mv.wind` follower, the wind view)
//   -> the normal and ambient occlusion -> the light through thin leaves (VegetationLighting: a
//      blade's diffuse wrapped round by its translucency and `transmission`; a translucent leaf's
//      thin layer in its Lambert's place, without `transmission`)
//   -> a leaf's specular (`LEAF_F0`, `useSpecular`).
// Families: 'leaf' (alpha-tested leaves: the bamboo's sprays, BambooMaterials.ts, on a map and
// alpha map of their own; the folded leaf, vegetation/LeafTexture.ts, by default), 'blade' (the meadow's blades: tagged as vegetation, no leaf map, drawn
// whole or not at all: alpha to coverage at 4x, else `coverageDither`'s threshold) and 'sod' (not
// tagged, so it takes the ordinary shadow filter and stays out of the depth prepass: kept for the
// kinds to come; no kind uses it today, the grass over the cliff's edge being the meadow's own). A leaf's `alphaTest: 'alphaOnly'` keeps no numeric cut-out (as a material cloned with
// three's NodeMaterial.copy has, which skips Material's alphaTest setter): the shadow passes and 4x
// cut it by its alpha alone.
// The meadow (Grass.ts) instances itself from the cull's storage buffers and widens its blades on
// screen (`pixelWideBlade`) inside its own displacement; the factory only takes its hooks: its own
// previous position (`previous`) and its normal and instance colour (`beforePosition`).

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

/** A leaf's specular reflectance, the sky's reflection and the key's glint alike: a waxy cuticle's,
 * as the meadow's blades take (Grass.ts SURFACE.f0). Under three's default 0.04 the rough leaf's
 * broad lobe outweighed its diffuse in blue, 2.1 times on a crown's lit half at noon and 2.9 on its
 * dark half, where the albedo holds almost no blue: a pale mint sheen. At 0.02 the sheen still
 * gave a sunlit leaf 12 to 20 % of its light, whitish and bluer than the diffuse, so a sunlit crown
 * read pale cyan; 0.008, near the blades', leaves a faint sheen and the leaf's own colour. On the family, so
 * the crowns and the falling leaves take it alike, and any tree to come. Tunable on debug pages
 * (debug/Tune.ts: `vegetation.leaf.f0`). */
const LEAF_F0 = /*@__PURE__*/ tuneUniform( 'vegetation.leaf.f0', 0.008, { min: 0.004, max: 0.04 } );

/** A surface's specular reflectance F0, for the sky's reflection and the key's glint alike. F0
 * alone: F90 follows it in the lighting model (GroundBounce.ts `start`, lighting/SurfaceBRDF.ts
 * `fresnel90`), whatever is written here. */
export function useSpecular( material: THREE.MeshStandardNodeMaterial, f0: F ): void {

	( material as unknown as { setupSpecular: () => void } ).setupSpecular = () => {

		specularColor.assign( vec3( f0 ) );
		specularColorBlended.assign( vec3( f0 ) );

	};

}

/** Light through a thin surface, beside the usual diffuse term (view space; VegetationLighting). */
export type Transmission = ( L: V3, V: V3, N: V3 ) => F | V3;

/** The light through a thin surface (`vegetationTransmission`): the strength of the key's back
 * light rises over `edge` either side of the terminator (a smoothstep, Hermite, in -N.L) to the
 * clamped cosine on the far face, so a curved blade, its normal turning ten degrees or so across
 * it, lights whole as the key comes behind it rather than in strips along its folds; and past half
 * translucency the light goes forward, into a view toward the key, by Schlick's phase function
 * (Blasi, Le Saëc and Schlick 1993) ( 1 - kappa^2 ) / ( 4 pi ( 1 - kappa cos )^2 ), which
 * integrates to 1 over the sphere, times 4 pi so it is 1 where it scatters evenly. Both tunable
 * (`lighting.vegetation.edge`, `.kappa`): the shader reads uniforms of the terms worked out here,
 * the numbers it read as constants at the defaults. */
export const TRANSMISSION = { edge: 0.2, kappa: 0.971 };

// (passed to the shader function as arguments: a laid-out function that read them itself would be
// emitted once, with the first material's uniform names, for every material that calls it)
const transmissionTerms = { edgeLow: uniform( 0 ), edgeHigh: uniform( 0 ), negKappa: uniform( 0 ), phaseScale: uniform( 0 ) };
tuneApply( 'lighting.vegetation.edge', TRANSMISSION.edge, { min: 0.05, max: 0.5 }, ( e ) => { transmissionTerms.edgeLow.value = - e; transmissionTerms.edgeHigh.value = e; } );
tuneApply( 'lighting.vegetation.kappa', TRANSMISSION.kappa, { min: 0.5, max: 0.99 }, ( k ) => { transmissionTerms.negKappa.value = - k; transmissionTerms.phaseScale.value = 1 - k * k; } );

/** The light through a thin surface of translucency t, by the cosines N.L and dot( L, -V ):
 * t of the diffuse, from behind the face (the smooth back light), times
 * mix( 1, the phase, saturate( 2 t - 1 ) ). One shader function for every kind. */
const vegetationTransmission = /*@__PURE__*/ Fn( ( [ t, NL, forward, edgeLow, edgeHigh, negKappa, phaseScale ]: [ F, F, F, F, F, F, F ] ) => {

	const x = NL.negate();
	const back = smoothstep( edgeLow, edgeHigh, x ).mul( clamp( x, 0, 1 ) );
	const d = forward.mul( negKappa ).add( 1 );
	const phase = phaseScale.div( d.mul( d ) );
	return t.mul( back ).mul( mix( float( 1 ), phase, clamp( t.mul( 2 ).sub( 1 ), 0, 1 ) ) );

} ).setLayout( { name: 'vegetationTransmission', type: 'float', inputs: [
	{ name: 't', type: 'float' }, { name: 'NL', type: 'float' }, { name: 'forward', type: 'float' },
	{ name: 'edgeLow', type: 'float' }, { name: 'edgeHigh', type: 'float' }, { name: 'negKappa', type: 'float' }, { name: 'phaseScale', type: 'float' },
] } );

/** Light through a thin surface of translucency `t` (VegetationLighting's): see
 * `vegetationTransmission`. */
export function transmission( t: F | number ): ( L: V3, V: V3, N: V3 ) => F {

	return ( L, V, N ) => vegetationTransmission( typeof t === 'number' ? float( t ) : t, dot( N, L ), dot( L, V.negate() ), transmissionTerms.edgeLow, transmissionTerms.edgeHigh, transmissionTerms.negKappa, transmissionTerms.phaseScale );

}

/** A plant's wind: its swayed pose (`displacement`, Wind.ts), the shadow cast and received in the
 * rest pose (`rest`, with `castMask` keeping parts out of the sun's map), the previous frame's sway
 * for the temporal modes' motion vectors, the `flags.mv.wind` follower (WindMotion.ts) and the
 * wind view's colour (`sway`, read once the displacement has been built). The shadow is cast in
 * the swayed pose unless `flags.shadowWind = false` (useRestPoseShadows). Returns the rest pose in world space. */
export function useVegetationMotion( material: THREE.NodeMaterial, wind: WindUniforms, motion: VegetationMotion ): V3 {

	const swayed = setWindDisplacement( material, motion.displacement, wind );
	const previous = motion.previous ? motion.previous( swayed ) : previousSway( material, wind );
	const rest = useRestPoseShadows( material, motion.rest, swayed, motion.castMask, previous );
	const before = motion.beforePosition;
	if ( before ) {

		const placed = material.positionNode!;
		material.positionNode = Fn( () => {

			before();
			return placed;

		} )();

	}
	followWindMotionFlag( material );
	const sway = motion.sway?.();
	if ( sway ) showSway( material, sway, wind );
	return rest;

}

export interface VegetationMotion {
	/** The rest pose, in the space of the material's position (world space here). */
	rest: V3;
	/** The swayed pose as a function of the wind's state. */
	displacement: WindDisplacement;
	/** Per vertex, 0 keeps the vertex out of the sun's map (VegetationShadow.ts). */
	castMask?: F;
	/** The plant's sway for the wind view (packages/render/debug/WindView.ts), once the displacement is built. */
	sway?: () => V2 | undefined;
	/** The previous frame's position for the motion vectors, from the swayed pose; default the
	 * displacement under the previous frame's wind (`previousSway`). */
	previous?: ( swayed: V3 ) => V3 | undefined;
	/** Vertex work run before the position is returned (assignments to `normalLocal`,
	 * `instanceColor`): a kind that instances itself does three's instancing there. */
	beforePosition?: () => void;
}

export interface VegetationLight {
	/** The occlusion from root to tip, on the ambient terms alone (VegetationLighting): a node
	 * shared by every build, or one built per build. */
	scale: F | ( () => F );
	/** How much the surface lets light through, 0..1, a number or a node per pixel: a blade's
	 * diffuse wrapped round by smoothstep( 0, 0.55, t ), and the strength of the light through it
	 * (VegetationLighting). */
	translucency: F | number;
	/** Light through the surface, by default `transmission` by its translucency (a translucent
	 * leaf takes none: its thin layer stands in its Lambert's place, VegetationLighting). */
	extra?: Transmission;
	/** How it reads the ambient volume (VegetationLighting's `ambient`; default degree 2 in front
	 * of the surface). */
	ambient?: AmbientReceiver;
}

export interface VegetationOptions extends VegetationMotion {
	family: 'leaf' | 'blade' | 'sod';
	name?: string;
	/** The dry surface's roughness. */
	roughness: number;
	/** A plain surface's vertex colours (a leaf always takes them). */
	vertexColors?: boolean;
	/** A leaf's colour map, its alpha map (default: the folded leaf's, `leafAlphaTexture`) and its
	 * cut-out (default LEAF_ALPHA_TEST). */
	leaf?: { map: THREE.Texture; alphaMap?: THREE.Texture; alphaTest?: number | 'alphaOnly' };
	/** Wetness: the colour's tint at full wetness (or the tint as a function of the wetness) and the
	 * wet surface's roughness (or the roughness as a function of the wetness). */
	wet: { tint: number | ( ( wetness: F ) => F ); roughness: number | ( ( wetness: F ) => F ) };
	/** The colour from the wet tint; default the material's colour (map and vertex colours) tinted. */
	color?: ( tint: F ) => THREE.Node<'vec3'> | THREE.Node<'vec4'>;
	normal?: V3;
	ao?: F;
	/** The direct light's scale and transmission, or a function of the rest pose (world space) that
	 * gives them. */
	light: VegetationLight | ( ( rest: V3 ) => VegetationLight );
}

/** A plant kind's material, built by the factory's one sequence of calls (see the header). */
export function vegetationMaterial( shared: Shared, options: VegetationOptions ): THREE.MeshStandardNodeMaterial {

	const { leaf } = options;
	let mat: THREE.MeshStandardNodeMaterial;
	if ( leaf ) {

		const alphaTest = leaf.alphaTest === 'alphaOnly' ? 0 : leaf.alphaTest ?? LEAF_ALPHA_TEST;
		mat = new THREE.MeshStandardNodeMaterial( {
			map: leaf.map, alphaTest, alphaToCoverage: ALPHA_TO_COVERAGE, side: THREE.DoubleSide, roughness: options.roughness, metalness: 0, vertexColors: true,
		} );
		mat.alphaMap = leaf.alphaMap ?? leafAlphaTexture( ( leaf.map.image as { width: number } ).width );

	} else {

		const blade = options.family === 'blade';
		mat = new THREE.MeshStandardNodeMaterial( { vertexColors: options.vertexColors ?? false, side: THREE.DoubleSide, roughness: options.roughness, metalness: 0, alphaToCoverage: blade && ALPHA_TO_COVERAGE } );
		if ( blade && ! ALPHA_TO_COVERAGE ) mat.alphaTestNode = coverageDither();

	}
	if ( options.family === 'leaf' || options.family === 'blade' ) markVegetation( mat );
	if ( leaf ) useLeafCoverage( mat );
	const wetness = clamp( shared.wetness, 0, 1 );
	const { tint, roughness } = options.wet;
	const wetTint = typeof tint === 'number' ? mix( 1, tint, wetness ) : tint( wetness );
	// materialColor retains the leaf map and its alpha; three applies vertex and instance colours
	// afterward. Wetness must not darken the cutout silhouette itself.
	mat.colorNode = options.color ? options.color( wetTint ) : materialColor.mul( vec4( wetTint, wetTint, wetTint, 1 ) );
	mat.roughnessNode = typeof roughness === 'number' ? mix( options.roughness, roughness, wetness ) : roughness( wetness );
	const rest = useVegetationMotion( mat, shared.wind, options );
	if ( options.normal ) mat.normalNode = options.normal;
	if ( options.ao ) mat.aoNode = options.ao;
	const light = typeof options.light === 'function' ? options.light( rest ) : options.light;
	const extra = light.extra ?? transmission( light.translucency );
	const scale = light.scale;
	useLighting( mat, () => new VegetationLighting( typeof scale === 'function' ? scale() : scale, light.translucency, options.family, extra, light.ambient ) );
	if ( options.family === 'leaf' ) useSpecular( mat, LEAF_F0 );
	writeMeter( mat );
	if ( options.name ) mat.name = options.name;
	return mat;

}
