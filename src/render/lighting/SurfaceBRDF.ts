import * as THREE from 'three/webgpu';
import { Fn, float, vec3, max, min, clamp, dot, length, reflect, uniform } from 'three/tsl';
import { SUN_GLINT_DISC } from './SunDisc.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

// The specular BRDF every physical model shares: GGX, Smith visibility and Schlick's Fresnel.
//
// The specular BRDF every physical model shares (lighting/GroundBounce.ts and the models built on
// it): a GGX distribution, a Smith visibility and Schlick's Fresnel, the terms Patry's shading talk
// lists ("Samurai Shading in Ghost of Tsushima", Jasmin Patry, SIGGRAPH 2020); the Smith term
// here takes its Schlick-GGX form.
// - The materials' `roughness` r is the perceptual roughness and GGX's alpha is r squared, the
//   common mapping (Burley, "Physically-Based Shading at Disney", SIGGRAPH 2012; Karis, "Real
//   Shading in Unreal Engine 4", SIGGRAPH 2013). The models read it as gloss, g = 1 - r. How tight
//   or broad a surface's highlight is comes from its material's value alone.
// - Visibility: Schlick-GGX with k = alpha / 2 (Karis 2013, to fit Smith's GGX),
//   V = 1 / ( ( N.L ( 1 - k ) + k ) ( N.V ( 1 - k ) + k ) ) / 4.
// - D keeps three's 1 / pi: the light colours here are not premultiplied by pi (three's Lambert
//   divides the albedo by pi), so the specular takes the same convention as the diffuse.
// - F90 follows F0 (`fresnel90`): saturate( 50 F0 ) on F0's mean, Schüler's rule as Lagarde and
//   de Rousiers give it.

/** GGX's alpha for a gloss g: the perceptual roughness 1 - g, squared. */
export const glossAlpha = /*@__PURE__*/ Fn( ( [ g ]: [ F ] ) => {

	const r = max( float( 1 ).sub( clamp( g, 0, 1 ) ), 1e-3 );
	return r.mul( r );

} ).setLayout( {
	name: 'glossAlpha',
	type: 'float',
	inputs: [ { name: 'g', type: 'float' } ],
} );

/** GGX D x Schlick-GGX visibility x Schlick's F for the given cosines (all clamped to 0..1). */
export const specularGGX = /*@__PURE__*/ Fn( ( [ NL, NV, NH, VH, alpha, f0, f90 ]: [ F, F, F, F, F, V3, F ] ) => {

	const a2 = alpha.mul( alpha );
	const d = NH.mul( NH ).mul( a2.sub( 1 ) ).add( 1 );
	const D = a2.div( max( d.mul( d ), 1e-12 ) ).mul( 1 / Math.PI );
	const k = alpha.mul( 0.5 );
	const vis = float( 0.25 ).div( NL.mul( k.oneMinus() ).add( k ).mul( NV.mul( k.oneMinus() ).add( k ) ) );
	const fresnel = float( 1 ).sub( VH ).pow( 5 );
	const Fr = f0.add( vec3( f90 ).sub( f0 ).mul( fresnel ) );
	return Fr.mul( D.mul( vis ) );

} ).setLayout( {
	name: 'specularGGX',
	type: 'vec3',
	inputs: [
		{ name: 'NL', type: 'float' }, { name: 'NV', type: 'float' }, { name: 'NH', type: 'float' },
		{ name: 'VH', type: 'float' }, { name: 'alpha', type: 'float' }, { name: 'f0', type: 'vec3' },
		{ name: 'f90', type: 'float' },
	],
} );

// F90 follows F0: no real material reflects less than 2 % at normal incidence, so a lower F0 is
// read as occlusion baked into it (a map that means "no reflection here") and the white rim at
// grazing angles goes with it, F90 = saturate( 50 mean( F0 ) ) (Schüler, ShaderX7, 2009, as given
// by Lagarde and de Rousiers, "Moving Frostbite to Physically Based Rendering", SIGGRAPH 2014,
// listing 27). A dielectric's 0.04 keeps 1.

/** F90 for a reflectance F0 (lighting/SurfaceBRDF.ts). */
export const fresnel90 = /*@__PURE__*/ Fn( ( [ f0 ]: [ V3 ] ) => {

	return clamp( dot( f0, vec3( 1 / 3 ) ).mul( 50 ), 0, 1 );

} ).setLayout( {
	name: 'fresnel90',
	type: 'float',
	inputs: [ { name: 'f0', type: 'vec3' } ],
} );

// The key is a disk (Karis 2013's representative point; Lagarde and de Rousiers, "Moving Frostbite
// to PBR", SIGGRAPH 2014): the specular takes its light direction at the point of the disk nearest
// the reflected view ray. L is the key's shading direction (lifted under a low key:
// lighting/KeyLift.ts), the disk's centre. The reflection r = reflect( -V, N ) projected on the
// plane across L, p = r - ( r.L ) L, clamped to length tan R, offsets L: L' = normalize( L + o ). The diffuse
// keeps L. At grazing views ( 2 N.V < 1 ) the offset's component along the geometric normal is
// clamped to tan R sat( 2 N.V ), so L' is not pushed under the horizon to draw a bright band over
// far level ground (ours). The specular is scaled by ( a' / ( R ( 1 - a' ) + a' ) )^2,
// a' = mix( alpha, 1, spread): a smooth surface's glint widens and dims at the same energy, a rough
// one barely changes. `radius` is the key's (main.ts: the sun's or the moon's, Radiometry.ts),
// `spread` the weather's `glintSpread` (sky/Weather.ts: 0 in every weather so far).
export const keyGlint = {
	radius: uniform( SUN_GLINT_DISC ),
	spread: uniform( 0 ),
};

/** The disk's representative light direction for a key L, shading normal N, view V and geometric
 * normal Ng (one space), tanR the tangent of the disk's angular radius. */
export const glintDirection = /*@__PURE__*/ Fn( ( [ L, N, V, Ng, tanR ]: [ V3, V3, V3, V3, F ] ) => {

	const r = reflect( V.negate(), N );
	const p = r.sub( L.mul( dot( r, L ) ) ).toVar();
	const len = length( p );
	const o = p.mul( min( float( 1 ), tanR.div( max( len, 1e-6 ) ) ) ).toVar();
	const limit = tanR.mul( clamp( dot( N, V ).mul( 2 ), 0, 1 ) );
	const oN = dot( o, Ng );
	o.addAssign( Ng.mul( clamp( oN, limit.negate(), limit ).sub( oN ) ) );
	return L.add( o ).normalize();

} ).setLayout( {
	name: 'glintDirection',
	type: 'vec3',
	inputs: [
		{ name: 'L', type: 'vec3' }, { name: 'N', type: 'vec3' }, { name: 'V', type: 'vec3' },
		{ name: 'Ng', type: 'vec3' }, { name: 'tanR', type: 'float' },
	],
} );

/** The disk's energy normalisation for GGX alpha, angular radius R and spread s. */
export const glintNormalisation = /*@__PURE__*/ Fn( ( [ alpha, R, s ]: [ F, F, F ] ) => {

	const a = alpha.add( float( 1 ).sub( alpha ).mul( s ) );
	const k = a.div( R.mul( a.oneMinus() ).add( a ) );
	return k.mul( k );

} ).setLayout( {
	name: 'glintNormalisation',
	type: 'float',
	inputs: [ { name: 'alpha', type: 'float' }, { name: 'R', type: 'float' }, { name: 's', type: 'float' } ],
} );
