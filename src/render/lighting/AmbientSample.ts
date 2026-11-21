import * as THREE from 'three/webgpu';
import {
	Fn, If, cameraPosition, clamp, dot, faceDirection, float, luminance, max, min, mix, normalWorld, normalWorldGeometry, normalize,
	positionWorld, reflect, smoothstep, vec3, vec4,
} from 'three/tsl';
import { debugFlags } from '../debug/DebugFlags.ts';
import { tune } from '../debug/Tune.ts';
import { IRRADIANCE_SLABS, type AmbientVolume } from './AmbientVolume.ts';
import { onSceneDispose } from '../SceneDispose.ts';

// How the island's lit materials read the ambient volume for their ambient diffuse light.
//
// How the island's lit materials read the ambient volume (AmbientVolume.ts, relit every frame by
// AmbientRelight.ts) for their ambient diffuse light, in place of the environment's irradiance: the
// probes see the land, so a grove's floor darkens, light reaches it from its open side, and a face
// turned from the sky takes the ground's warm bounce. The environment map keeps the specular, which
// the volume's local irradiance occludes.
//
// - The lookup (`ambientField`): the point's lattice coordinates (`latticeAt`), then each slab of
//   the output texture sampled once, filtered (trilinear between the eight probes round the point),
//   and the harmonics evaluated at the normal (`irradianceL1`, `irradianceL2`), clamped at zero per
//   channel (the relight's deringing holds the luminance, not each channel). Degree 1 reads three
//   slabs, degree 2 seven (`AmbientReceiver.bands`). The island's receivers (the ground, the
//   meadow, the culms, the leaves, the Hanming islanders) take degree 2, as Patry's probes are
//   second-order harmonics throughout ("Real-Time Samurai Cinema", p. 16): at degree 1 the
//   relight's directionality boost turns light that comes from below (the lit meadow's bounce)
//   against a face turned up, which degree 2's quadratic band gives back (AmbientRelight.ts, known
//   limits).
// - The seam: outside the box in x or z every point reads the open sea's probe of its level (the
//   output's last column). Over the cell beyond the box's border probes the coefficients blend
//   from the border's to the sea's, so the light does not step at the box's edge; each side's
//   fetches sit in a branch taken only where its share is above zero, so a point well inside the
//   box reads one set.
// - The point read (`ambientPoint`): the surface's point moved half a metre along its normal and a
//   quarter toward the eye, so a sample at a wall or on the ground is taken in the air in front of
//   it rather than between probes on both sides of a solid (a lattice of 2 m cells, filtered, leaks
//   the far side's light through anything thinner than a cell otherwise; the offsets along the
//   normal and toward the eye as in Majercik et al., "Scaling Probe-Based Real-Time Dynamic Global
//   Illumination for Production", JCGT 2021). A blade of the meadow reads straight above its point instead (its normals turn every
//   way across a tuft, and a sample offset along them would scatter over the cell). The probes' own
//   placement (moved out of solids, the invalid ones filled from their neighbours:
//   world/AmbientProbes.ts) does the rest. No per-probe visibility weights (eight fetches by hand
//   instead of the filter's one): they would answer a leak these offsets leave, and none is named
//   yet.
// - The texture: the volume's one `irradiance` node, whose `.value` a growth swaps (AmbientVolume
//   `apply`); every fetch here is a sample of it, so it follows that node: new bind groups, no
//   pipeline. The fetches name an explicit level (the texture has none other), so they may sit in
//   branches. The lattice helpers are layout functions taking the volume's uniforms as arguments:
//   r186 emits a layout function that reads uniforms with the first material's binding names in
//   the next one (SeaDetail.ts). The texture holds the irradiance over its storage scale
//   (AmbientRelight.ts `storageGain`, a power of two from the sky's geometric mean luminance, read
//   from the sky's harmonics buffer: no texture binding), which the lookup multiplies back, times
//   the receiver's gain tier (`AmbientReceiver.tier`, `AMBIENT_TIERS`: 'normal' 1, every
//   receiver's; a tier of another gain would be a parameter to give a few materials). The
//   last slab's w holds the probe's sky view factor (the share of a level surface's light under a
//   uniform sky that reaches it: AmbientRelight.ts) as one number, unscaled
//   (`ambientSkyVisibility`, at least 0.01), for what needs only how much sky a point sees. The
//   volumetric haze reads it (weather/VolumetricHaze.ts, its cover's `skyVisibility`, given by
//   Pipeline.ts) as the sky its froxels see; no material reads it.
// - The receivers (`AmbientReceiver`), through the one place each lighting model takes its ambient
//   diffuse: GroundBounceLightingModel and every model built on it hand theirs to the ambient grade
//   (AmbientGrade.ts `useAmbientGrade`), which replaces three's environment irradiance with
//   `ambientEnvironment`'s. The cliffs and the stones keep the volume under the water, carried down
//   the column (lighting/Underwater.ts); the sea floor takes the environment's light. The
//   flying fish (life/FishMaterial.ts) read the volume through UnderwaterLightingModel's default,
//   degree 1 weighted by depth: in flight above the water they take the volume's light, under it the
//   environment's carried down the water column.
// - A translucent receiver (`translucency` t, a number or a node per pixel: the meadow's blades
//   alone, vegetation/VegetationLighting.ts) takes more light and less of its direction as it lets light
//   through, each band weighted whole, multiplications on the coefficients before the evaluation:
//   the constant by 1 + t, the linear band by w1( t ) and the five of degree 2 by w2( t ), its m = 0
//   basis 0.315392 ( 3 z^2 - 1 ) with its constant part (degree 2 is one zero-mean band, not split).
//   The weights are the blade's own diffuse lobe projected on the bands (`translucentBands`): the
//   key's diffuse on a blade (vegetation/VegetationLighting.ts) is the clamped cosine C( mu ) wrapped by
//   the share s( t ) (`translucentWrap`) toward 3/4 of the squared half-Lambert,
//   W( mu ) = 3/16 ( 1 + mu )^2, with t of the clamped cosine from behind, C( -mu ), for the light
//   through it. A zonal kernel's band l scales the irradiance by its Legendre coefficient
//   2 pi int k( mu ) P_l( mu ) dmu (Ramamoorthi and Hanrahan, "An Efficient Representation for
//   Irradiance Environment Maps", SIGGRAPH 2001): C's are pi, 2 pi / 3, pi / 4; W's pi, pi / 2,
//   pi / 10, so against C's 1, 3/4 and 2/5; C( -mu )'s 1, -1 and 1. The kernel
//   ( 1 - s ) C + s W + t C( -mu ) takes, against the clamped cosine the volume is stored for,
//   w0 = 1 + t, w1 = 1 - s / 4 - t and w2 = 1 - 3 s / 5 + t. The blades' t runs 0.55 to 0.83
//   (world/Grass.ts `TRANSLUCENCY`: 0.6 to 0.9 under its mask of 0.92), wrapped whole: the
//   constant by 1.55 to 1.83, the linear band 0.2 to -0.08, degree 2 0.95 to 1.23, so a blade's
//   tip takes its ambient almost without direction, a little from behind.
// - A thin layer (`thin`, the leaves': vegetation/VegetationLighting.ts) takes the volume's light for
//   its single scattering, E01( 3/4 n ) at the geometric normal n of the face toward the viewer,
//   from the coefficients without any weighting (`ambientEnvironment`'s `thin`, `AmbientField.low`):
//   the constant once and the linear band at three quarters, so the light from every side counts,
//   without degree 2; for a uniform sky pi times its radiance. VegetationLighting blends it in by
//   the leaf's translucency, times the layer's response under a uniform sky.
// - The specular's relight (`ambientEnvironment`): the environment's specular, a reflection of the
//   open sky, scaled by the local irradiance over the environment's own at the same normal, their
//   luminances' ratio, lum( E_px( n ) ) / lum( E_env( n ) ), at most RELIGHT_MAX: a grove's floor
//   and a cliff's foot stop mirroring a sky they do not see, and open ground by the bright sea,
//   which takes more than the sky alone gives, reflects a little more. Measured over every lit
//   receiver of the island views (fair noon, golden hour, overcast, the meadow at 24 m, an islander
//   close up, the cliffs from the water): the median 0.3 to 1.2, the 99th percentile at most 1.8
//   (2.7 for the leaves close up), the culms at most 2.9 and an islander's jaw 2.0; above 4 only
//   about twenty texels of a golden hour's frame, at a cliff's foot by the sunlit sea, up to 4.3,
//   with no highlight to see there. The cap (our own choice) keeps a point lit by far more than the
//   sky from a mirror brighter than any reflection of it. E_px is the volume's irradiance at the
//   point read without a translucent receiver's weighting; E_env the harmonics of the source the
//   environment's reflections are filtered from (AmbientRelight.ts `environmentSlabs`, at the
//   volume's scale: `installEnvironmentHarmonics`), never another sky's, or the ratio in the open
//   would drift from 1 and every reflection with it. The denominator keeps a floor of
//   ENVIRONMENT_FLOOR times the environment's mean irradiance, which only keeps the night from a
//   division by zero. The idea, a reflection normalised by the irradiance where it is read, is
//   Patry's ("Real-Time Samurai Cinema", SIGGRAPH 2021 Advances in Real-Time Rendering, p. 63: the
//   reflection probes' luminance normalised by a single sample of the harmonics' irradiance,
//   converted to luminance); the form is ours.
// - The reflection cone against the horizon (`specularOcclusion`, which the ambient grade applies
//   with the ambient occlusion to every receiver: AmbientGrade.ts): a cone round the reflected
//   direction r = reflect( -V, N ), its half-angle the GGX lobe's (CONE_FIT), meets the surface's
//   own horizon, the geometric normal's (a normal map that tips r below the surface loses that
//   part): s = sat( ( sin theta - r.n_geo ) / ( 2 sin theta ) ) is how far the cone reaches under
//   it, and the specular keeps sat( 1 - k s^2 ( 2 - s^2 ) ) times the ambient occlusion. The idea,
//   a cone of the reflection's lobe against an occluder, is Lagarde and de Rousiers' ("Moving
//   Frostbite to PBR", SIGGRAPH 2014); this form and its constants are ours. It replaces three's
//   specular occlusion from the AO and N.V and the stones' own horizon band.
// - The bamboo's canopy: its probes see the groves (world/AmbientProbes.ts), so the volume darkens
//   a grove's floor and its culms' feet itself, and the canopy's own shadowing of their sky light (the
//   island's and the meadow's baked sky visibility, world/Chunks.ts `underCanopy` and the
//   anchors'; the culms' by their height, world/BambooMaterials.ts) is left out where materials
//   read the volume (`CANOPY_SHADE`), or the floor would darken twice: on a fair noon, in the
//   shadow, a grove's floor took 0.45 of an open meadow's light before the volume, 0.51 from the
//   volume alone and 0.22 with both. The culms keep only a soft darkening at their feet
//   (BambooMaterials.ts CULM_SHADE). The contact terms (a stone's foot, a culm's) stay. The
//   meadow's blades are the one receiver that keeps the canopy beside the volume, taken relative
//   to their tips, which stand about where the probe they read sees it: a root under a full grove
//   takes about 0.56 of what a root on the open meadow takes against its tip, 0.8 stop
//   (world/Grass.ts `aoCanopy`, `grass.ao.canopy`).
//
// Load flags (debug builds, `flags=`): `ambientVolume:0` (or false) builds every material on the
// environment's irradiance as before the volume, the ground bounce whole (the shaders those
// materials built before this module); `groundBounceLand` keeps the ground bounce's land
// (GroundBounce.ts) beside the volume, which carries the key's bounce off the land itself;
// `canopyShadow` keeps the canopy's shadowing beside it (the A/B of the double darkening; the world's
// chunks are baked with it, so the flag is part of their cache key: world/WorldChunks.ts);
// `ambientVolumeOff:<kinds>` sends the receivers named (`/` between them; `all` every one) back to
// the environment's irradiance, the rest unchanged, to time each one's read of the volume
// (`AmbientReceiver.kind`: terrain, stones, meadow, culms, leaves, figures (both looks), else the
// material's name, such as yunpeng or fish). It is for timing only: the chunks' bakes and the
// ground bounce's land stay the volume path's (`CANOPY_SHADE`, `GROUND_BOUNCE_LAND`), so a
// receiver sent back is lit as neither path lights it; `ambientVolume:0` gives the old path.

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

/** Materials read the ambient volume (off with `flags=ambientVolume:0`, at load). */
export const AMBIENT_VOLUME = debugFlags.ambientVolume !== false && debugFlags.ambientVolume !== 0;
/** The ground bounce keeps its land: without the volume, or with `flags=groundBounceLand`. */
export const GROUND_BOUNCE_LAND = ! AMBIENT_VOLUME || debugFlags.groundBounceLand === true || debugFlags.groundBounceLand === 1;
/** The bamboo's canopy darkens the sky light of what stands under it on its own (the island's and
 * the meadow's baked sky visibility, world/Chunks.ts; the culms' by their height,
 * world/BambooMaterials.ts): without the volume, or with `flags=canopyShadow`. */
export const CANOPY_SHADE = ! AMBIENT_VOLUME || debugFlags.canopyShadow === true || debugFlags.canopyShadow === 1;

// receivers sent back to the environment's irradiance (`flags=ambientVolumeOff:terrain/meadow`)
const OFF = new Set( typeof debugFlags.ambientVolumeOff === 'string' ? debugFlags.ambientVolumeOff.split( '/' ) : [] );

// the point read: along the normal and toward the eye (m), or straight up for a blade
const NORMAL_OFFSET = 0.5, VIEW_OFFSET = 0.25, LIFT = 0.5;
// The reflection cone's half-angle (radians) for a GGX alpha: P( alpha ), the least-squares cubic
// through the half-angle of the cone round the reflected direction that holds 90 % of the lobe's
// energy (D x the Schlick-GGX visibility x cos, viewed along the normal: lighting/SurfaceBRDF.ts),
// integrated numerically for alpha from 0 to 0.9 (the fit stays within 0.05 rad, and leaves a
// mirror a cone of 0.05 rad, not a ray). CONE_OCCLUSION (k) is how much of the reflection a cone
// wholly under the horizon loses: 1 - 0.09. Such a cone sees the land in place of the sky, and
// under an even sky a diffuse surface sends back its albedo times the sky's radiance (one bounce);
// 0.09 is the land's mean albedo (0.093 over our views under `flags=albedoOnly`: the leaves 0.07,
// the grass 0.08, the ground 0.09, the culms 0.10, the rock 0.16). Tunable (debug/Tune.ts
// `lighting.specularOcclusion.strength`), read as the app loads: `specularOcclusion` takes it as a
// constant.
const CONE_FIT = [ 0.051, 5.612, - 8.899, 4.728 ];
const CONE_OCCLUSION = tune( 'lighting.specularOcclusion.strength', 0.91, { min: 0.5, max: 1, load: true } )();
// the specular relight's floor on its denominator, a share of the environment's mean irradiance
const ENVIRONMENT_FLOOR = 1e-4;
// the specular relight's cap (see the header)
const RELIGHT_MAX = 4;

let installed: AmbientVolume | null = null;
let environment: { environmentSlabs(): V4[] } | null = null;

/** Makes `volume` the ambient volume every lit island material reads: before any of them builds. */
export function installAmbientVolume( volume: AmbientVolume ): void {

	installed = volume;

}

/** Makes `source` (AmbientRelight) the environment's own harmonics the specular's relight divides
 * by: before any material builds. */
export function installEnvironmentHarmonics( source: { environmentSlabs(): V4[] } ): void {

	environment = source;

}
onSceneDispose( 'ambient volume', () => { installed = null; environment = null; } );

/** Whether materials built now read the volume. */
export function ambientVolumeOn( kind?: string ): boolean {

	return AMBIENT_VOLUME && installed !== null && ! ( OFF.has( 'all' ) || ( kind !== undefined && OFF.has( kind ) ) );

}

/** How a lighting model reads the volume. */
export interface AmbientReceiver {
	/** the harmonics' degree: 1 (three slabs) or 2 (seven) */
	bands: 1 | 2;
	/** where it reads: 'normal' (default) in front of the surface, 'up' straight above it (the
	 * meadow's blades) */
	offset?: 'normal' | 'up';
	/** the volume's share, the rest the environment's (under the water: Underwater.ts) */
	weight?: F;
	/** the receiver's name for `flags=ambientVolumeOff` (default: the material's name) */
	kind?: string;
	/** how much light it lets through, 0..1, a number or a node (see the header) */
	translucency?: F | number;
	/** a thin layer: `ambientEnvironment` also returns its single scattering's light (see the
	 * header) */
	thin?: boolean;
	/** its gain tier (`AMBIENT_TIERS`; default 'normal') */
	tier?: AmbientTier;
}

/** The ambient gain tiers a receiver picks from (see the header). */
export const AMBIENT_TIERS = { normal: 1 };
export type AmbientTier = keyof typeof AMBIENT_TIERS;

/** The translucency at which a blade's diffuse has wrapped whole (`translucentWrap`): the least
 * a blade has, its root's 0.6 under the mask of 0.92 (world/Grass.ts `TRANSLUCENCY`), so every
 * blade wraps whole and the ramp is as wide as that allows, for a kind less translucent than the
 * blades to ease into the wrap. */
const WRAP_WHOLE = 0.55;

/** A blade's wrap share s( t ): how far its key's diffuse has moved from the clamped cosine toward
 * the squared half-Lambert at translucency t, rising smoothly from 0 to whole at WRAP_WHOLE
 * (vegetation/VegetationLighting.ts, and the ambient bands' weights here). */
export const translucentWrap = ( t: F ): F => smoothstep( 0, WRAP_WHOLE, t );

/** A translucent receiver's weights on the constant, the linear band and degree 2 at translucency
 * t: the blade's diffuse lobe projected on the bands (see the header). */
const translucentBands = ( t: F ): [ F, F, F ] => {

	const s = translucentWrap( t ).toVar();
	return [ t.add( 1 ), float( 1 ).sub( s.mul( 1 / 4 ) ).sub( t ), float( 1 ).sub( s.mul( 3 / 5 ) ).add( t ) ];

};

/** Whether a receiver of translucency `t` is weighted at all. */
const translucent = ( t: F | number ): boolean => typeof t !== 'number' || t > 0;

/** ( u of the box, v of slab 0, w, the open sea's share ) for a world point. */
const latticeAt = /*@__PURE__*/ Fn( ( [ P, origin, dims, spacing ]: [ V3, V3, V3, F ] ) => {

	const g = P.sub( origin ).div( spacing ).toVar();
	const last = dims.sub( 1 ).toVar();
	// how far beyond the border probes, in cells (negative inside)
	const outside = max( max( g.x.sub( last.x ), g.x.negate() ), max( g.z.sub( last.z ), g.z.negate() ) );
	const c = clamp( g, vec3( 0 ), last );
	return vec4(
		c.x.add( 0.5 ).div( dims.x.add( 1 ) ),
		c.y.add( 0.5 ).div( dims.y.mul( IRRADIANCE_SLABS ) ),
		c.z.add( 0.5 ).div( dims.z ),
		smoothstep( 0, 1, outside ),
	);

} ).setLayout( {
	name: 'ambientLattice',
	type: 'vec4',
	inputs: [ { name: 'P', type: 'vec3' }, { name: 'origin', type: 'vec3' }, { name: 'dims', type: 'vec3' }, { name: 'spacing', type: 'float' } ],
} );

/** The degree 0 and 1 basis ( Y0, Y1, Y2, Y3 ) at n (three.js's order: y, z, x). */
const basis1 = ( n: V3 ): V4 => vec4( 0.282095, n.y.mul( 0.488603 ), n.z.mul( 0.488603 ), n.x.mul( 0.488603 ) );

/** E( n ) from the degree 1 slabs ( c0 .. c3 ) of r, g and b, clamped at zero. */
const irradianceL1 = /*@__PURE__*/ Fn( ( [ r, g, b, n ]: [ V4, V4, V4, V3 ] ) => {

	const y = basis1( n ).toVar();
	return max( vec3( dot( r, y ), dot( g, y ), dot( b, y ) ), vec3( 0 ) );

} ).setLayout( {
	name: 'ambientIrradianceL1',
	type: 'vec3',
	inputs: [ { name: 'r', type: 'vec4' }, { name: 'g', type: 'vec4' }, { name: 'b', type: 'vec4' }, { name: 'n', type: 'vec3' } ],
} );

/** E( n ) from all seven slabs, clamped at zero. */
const irradianceL2 = /*@__PURE__*/ Fn( ( [ r, g, b, r2, g2, b2, last, n ]: [ V4, V4, V4, V4, V4, V4, V4, V3 ] ) => {

	const y = basis1( n ).toVar();
	const y2 = vec4( n.x.mul( n.y ).mul( 1.092548 ), n.y.mul( n.z ).mul( 1.092548 ), n.z.mul( n.z ).mul( 3 ).sub( 1 ).mul( 0.315392 ), n.x.mul( n.z ).mul( 1.092548 ) ).toVar();
	const y8 = n.x.mul( n.x ).sub( n.y.mul( n.y ) ).mul( 0.546274 );
	const e = vec3( dot( r, y ).add( dot( r2, y2 ) ), dot( g, y ).add( dot( g2, y2 ) ), dot( b, y ).add( dot( b2, y2 ) ) ).add( last.xyz.mul( y8 ) );
	return max( e, vec3( 0 ) );

} ).setLayout( {
	name: 'ambientIrradianceL2',
	type: 'vec3',
	inputs: [
		{ name: 'r', type: 'vec4' }, { name: 'g', type: 'vec4' }, { name: 'b', type: 'vec4' },
		{ name: 'r2', type: 'vec4' }, { name: 'g2', type: 'vec4' }, { name: 'b2', type: 'vec4' },
		{ name: 'last', type: 'vec4' }, { name: 'n', type: 'vec3' },
	],
} );

/** The volume's light round one point: its irradiance at any normal. */
export interface AmbientField {
	/** E( n ) (world space, unit), in the scene's units, as three's `iblIrradiance` */
	at( n: V3 ): V3;
	/** E( n ) without the translucent receiver's weighting (the front face's alone) */
	front( n: V3 ): V3;
	/** E01( n ): the constant and the linear band alone, without the weighting (a thin layer's) */
	low( n: V3 ): V3;
}

/** E( n ) from seven slabs of degree 2 coefficients, or the first three at degree 1. */
function evaluate( c: V4[], n: V3 ): V3 {

	return c.length === IRRADIANCE_SLABS ? irradianceL2( c[ 0 ], c[ 1 ], c[ 2 ], c[ 3 ], c[ 4 ], c[ 5 ], c[ 6 ], n ) as V3 : irradianceL1( c[ 0 ], c[ 1 ], c[ 2 ], n ) as V3;

}

/** The volume's coefficients at world point P (see the header), of degree `bands`, for a receiver
 * of translucency `t` (see the header). Called while a material's fragment stage builds (its seam's
 * branches), with the volume installed. */
export function ambientField( P: V3, bands: 1 | 2, t: F | number = 0, tier: AmbientTier = 'normal' ): AmbientField {

	const volume = installed!;
	const slabs = bands === 2 ? IRRADIANCE_SLABS : 3;
	const at = latticeAt( P, volume.origin, volume.dims, volume.spacing ).toVar();
	const fetch = ( u: F, s: number ): V4 => volume.irradiance.sample( vec3( u, at.y.add( s / IRRADIANCE_SLABS ), at.z ) ).level( float( 0 ) ) as unknown as V4;
	const sea = volume.dims.x.add( 0.5 ).div( volume.dims.x.add( 1 ) );
	const c = Array.from( { length: slabs }, () => vec4( 0 ).toVar() );
	If( at.w.lessThan( 1 ), () => {

		const share = at.w.oneMinus();
		c.forEach( ( v, s ) => v.assign( fetch( at.x, s ).mul( share ) ) );

	} );
	If( at.w.greaterThan( 0 ), () => {

		c.forEach( ( v, s ) => v.addAssign( fetch( sea, s ).mul( at.w ) ) );

	} );
	// the storage scale back, times the tier's gain (see the header)
	const gain = AMBIENT_TIERS[ tier ] === 1 ? volume.storageGain() : volume.storageGain().mul( AMBIENT_TIERS[ tier ] );
	c.forEach( ( v ) => v.mulAssign( gain ) );
	// each band weighted whole: the constant by 1 + t, the linear band by w1( t ), degree 2 by
	// w2( t ) (see the header)
	let weighted: V4[] = c;
	if ( translucent( t ) ) {

		const tn = ( typeof t === 'number' ? float( t ) : t ).toVar();
		const [ b0, b1, b2 ] = translucentBands( tn );
		const w0 = b0.toVar(), w1 = b1.toVar(), w2 = b2.toVar();
		weighted = c.map( ( v, s ) => s >= 3 ? v.mul( w2 ) as V4 : vec4( v.x.mul( w0 ), v.yzw.mul( w1 ) ) );

	}
	return {
		at: ( n ) => evaluate( weighted, n ),
		front: ( n ) => evaluate( c, n ),
		low: ( n ) => irradianceL1( c[ 0 ], c[ 1 ], c[ 2 ], n ) as V3,
	};

}

/** The sky view factor v_sky at world point P (the volume's last slab's w: AmbientRelight.ts), at
 * least 0.01 (see the header). Called while a shader builds, with the volume installed: the
 * volumetric haze's ambient visibility reads it (weather/VolumetricHaze.ts HazeCover). */
export function ambientSkyVisibility( P: V3 ): F {

	const volume = installed!;
	const at = latticeAt( P, volume.origin, volume.dims, volume.spacing ).toVar();
	const fetch = ( u: F ): F => ( volume.irradiance.sample( vec3( u, at.y.add( 6 / IRRADIANCE_SLABS ), at.z ) ).level( float( 0 ) ) as unknown as V4 ).w;
	const sea = volume.dims.x.add( 0.5 ).div( volume.dims.x.add( 1 ) );
	return max( mix( fetch( at.x ), fetch( sea ), at.w ), 0.01 ).toVar();

}

/** The point a surface at P with world normal N reads the volume at (see the header). */
export function ambientPoint( P: V3, N: V3 ): V3 {

	return P.add( N.mul( NORMAL_OFFSET ) ).add( normalize( cameraPosition.sub( P ) ).mul( VIEW_OFFSET ) );

}

/** The environment specular's occlusion by the reflection cone against the geometric normal's
 * horizon (see the header), before the ambient occlusion: N the shading normal, Ng the geometric
 * one, V the view (one space), alpha GGX's. */
export const specularOcclusion = /*@__PURE__*/ Fn( ( [ N, Ng, V, alpha ]: [ V3, V3, V3, F ] ) => {

	const a = clamp( alpha, 0, 1 );
	const fit = float( CONE_FIT[ 3 ] ).mul( a ).add( CONE_FIT[ 2 ] ).mul( a ).add( CONE_FIT[ 1 ] ).mul( a ).add( CONE_FIT[ 0 ] );
	const sinCone = min( fit, Math.PI / 2 ).sin().toVar();
	const s = clamp( sinCone.sub( dot( reflect( V.negate(), N ), Ng ) ).div( sinCone.mul( 2 ) ), 0, 1 ).toVar();
	const s2 = s.mul( s );
	return clamp( float( 1 ).sub( s2.mul( s2.oneMinus().add( 1 ) ).mul( CONE_OCCLUSION ) ), 0, 1 );

} ).setLayout( {
	name: 'specularOcclusion',
	type: 'float',
	inputs: [ { name: 'N', type: 'vec3' }, { name: 'Ng', type: 'vec3' }, { name: 'V', type: 'vec3' }, { name: 'alpha', type: 'float' } ],
} );

/** A lighting model's ambient from the volume: `irradiance`, three's `iblIrradiance` in its place
 * (graded: the relight applies the grade's sky and bounce parameters), and `occlusion`, the factor on
 * the environment's specular. `env` is the environment's irradiance at the surface (three's
 * `iblIrradiance`). */
export function ambientEnvironment( env: V3, receiver: AmbientReceiver ): { irradiance: V3; occlusion: F; thin?: V3 } {

	const sky = env.toVar();
	const P = positionWorld, N = normalWorld, t = receiver.translucency ?? 0;
	const point = receiver.offset === 'up' ? P.add( vec3( 0, LIFT, 0 ) ) : ambientPoint( P, N );
	// a thin layer's light, E01( 3/4 n ) at the face toward the viewer (see the header); the
	// environment's stand-in is a uniform sky's, its irradiance
	const thinAt = ( field: AmbientField ): V3 => field.low( normalWorldGeometry.mul( faceDirection ).mul( 0.75 ) );
	let E: V3, front: V3, thin: V3 | undefined;
	if ( receiver.weight ) {

		const w = receiver.weight.toVar();
		const e = sky.toVar(), f = sky.toVar();
		const h = receiver.thin ? sky.toVar() : undefined;
		If( w.greaterThan( 0 ), () => {

			const field = ambientField( point, receiver.bands, t, receiver.tier );
			e.assign( mix( sky, field.at( N ), w ) );
			f.assign( translucent( t ) ? mix( sky, field.front( N ), w ) : e );
			if ( h ) h.assign( mix( sky, thinAt( field ), w ) );

		} );
		E = e;
		front = f;
		thin = h;

	} else {

		const field = ambientField( point, receiver.bands, t, receiver.tier );
		E = field.at( N ).toVar();
		front = translucent( t ) ? field.front( N ) : E;
		thin = receiver.thin ? thinAt( field ).toVar() : undefined;

	}
	// the specular's relight: the point's irradiance (its front face's) over the environment's own
	// at the same normal, with a floor at a share of the environment's mean, at most RELIGHT_MAX
	// (see the header)
	const slabs = environment!.environmentSlabs();
	const own = luminance( evaluate( slabs, N ) );
	const mean = luminance( vec3( slabs[ 0 ].x, slabs[ 1 ].x, slabs[ 2 ].x ) ).mul( 0.282095 );
	const occlusion = min( luminance( front ).div( max( own, max( mean.mul( ENVIRONMENT_FLOOR ), 1e-12 ) ) ), RELIGHT_MAX );
	return { irradiance: E, occlusion: occlusion.toVar(), thin };

}
