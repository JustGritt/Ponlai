import * as THREE from 'three/webgpu';
import { Fn, abs, clamp, exp, float, max, min, select, sqrt } from 'three/tsl';

// The sea's means over the slopes no pixel resolves, from their one Gaussian distribution.
//
// The water (Water.ts) reads its reflection's lobe here: the visible facets' cosine, their mean
// Fresnel, the lobe's lift, width and spread in elevation, and Smith's shadowing, all from the
// variance of the pixel's unresolved slopes along the view; and the normal integral and Smith's G1 its sky and its glint share.
// `verify-slope-variance` (tools/render) restates these formulas and checks them against a
// numerical integration over the same distribution.

type F = THREE.Node<'float'>;

// the unit normal density, and its integral through erf by Abramowitz and Stegun's 7.1.26 (an
// absolute error under 1.5e-7)
export const normalPdf = ( x: F ) => exp( x.mul( x ).mul( - 0.5 ) ).mul( 1 / Math.sqrt( 2 * Math.PI ) );
export const normalCdf = Fn( ( [ x ]: [ F ] ) => {

	const z = abs( x ).mul( Math.SQRT1_2 ).toVar();
	const t = float( 1 ).div( z.mul( 0.3275911 ).add( 1 ) ).toVar();
	const poly = t.mul( t.mul( t.mul( t.mul( t.mul( 1.061405429 ).add( - 1.453152027 ) ).add( 1.421413741 ) ).add( - 0.284496736 ) ).add( 0.254829592 ) );
	const half = float( 1 ).sub( poly.mul( exp( z.mul( z ).negate() ) ) ).mul( 0.5 ).toVar();
	return select( x.lessThan( 0 ), float( 0.5 ).sub( half ), float( 0.5 ).add( half ) );

} ).setLayout( { name: 'normalCdf', type: 'float', inputs: [ { name: 'x', type: 'float' } ] } );

// Smith's shadowing of a ray at elevation sine `y` over Gaussian slopes of spread `s` along its
// azimuth (the Beckmann distribution's G1 in the rational form of Walter, Marschner, Li & Torrance
// 2007, "Microfacet Models for Refraction through Rough Surfaces", EGSR), 0 at or under the horizon
export const smithG1 = Fn( ( [ y, s ]: [ F, F ] ) => {

	const c = clamp( y, 0, 0.9999 ).toVar();
	const a = c.div( sqrt( float( 1 ).sub( c.mul( c ) ) ).mul( s ).mul( Math.SQRT2 ) ).toVar();
	const g = a.mul( 3.535 ).add( a.mul( a ).mul( 2.181 ) ).div( a.mul( 2.276 ).add( a.mul( a ).mul( 2.577 ) ).add( 1 ) ).toVar();
	return select( a.lessThan( 1.6 ), g, float( 1 ) );

} ).setLayout( { name: 'smithG1', type: 'float', inputs: [ { name: 'y', type: 'float' }, { name: 's', type: 'float' } ] } );

// The mean surface's reflection over the slopes no pixel resolves, all from one spread: the slopes
// along the view a Gaussian of spread sigmaV, `slopeScale` times the square root of their variance
// along the view (Water.ts `surface`'s `viewVar`, UnresolvedSlopes.ts), before
// the waves' damping (the damped one falls away where the fine waves do, and the lee stood out as a
// smooth pale tongue).
// - The facets the eye sees, weighted by their projected area (the visible normals' distribution:
//   Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs", JCGT),
//   meet the view at a cosine of mean cosVisible = E2 / E1 and spread sqrt( E3 / E1 - cosVisible^2 )
//   over the flat sea's cos (the moments of cos + sigmaV z over z > -m, z normal, m = cos / sigmaV:
//   E1 = cos Phi( m ) + sigmaV phi( m ), E2 = ( cos^2 + sigmaV^2 ) Phi( m ) + cos sigmaV phi( m ),
//   E3 = ( cos^2 + 3 sigmaV^2 ) cos Phi( m ) + ( cos^2 + 2 sigmaV^2 ) sigmaV phi( m )), and a
//   facet's reflectance climbs so steeply at grazing that the sheet reflects as they do. Its mean
//   over them is taken at the mean less and plus one spread, averaged (a two-point rule for a spread
//   of cosines; F at the mean alone fell short of the integrated mean by up to 15 % at sigmaV 0.1,
//   the two points within 4.4 %, `verify-slope-variance`'s measure). The mean Fresnel is the pixel's
//   own times that mean over F( cos ) (a lifted cosine in the pixel's Fresnel alone would even the
//   waves out),
//   times the share of the reflected rays no other wave stops: Smith's shadowing for those slopes,
//   at the lobe's centre and its two edges, weighted 1/4, 1/2, 1/4.
// - The reflected rays rise with the facets' tilt toward the eye: the lobe's centre stands `lift`
//   times ( cosVisible - cos ) over the mirror (an elevation's sine), and its elevation spreads
//   `lift` times the visible cosines' spread. `lift` is 2, the geometry's own: a facet tilted toward
//   the eye by a small angle raises the view's cosine on it by that angle times the grazing angle's
//   cosine, and the reflected ray by twice the angle, so its sine by twice as much (1.5 left the far
//   sea of a low sun too dark). The pixel's own mirror rises by as much, so the waves it resolves
//   keep their relief. The sky is read there over the lobe's width, 2 sigmaV across (a level of the
//   panorama, never wider than the lobe stands over the horizon), and the lobe's share under the
//   horizon, a normal integral over its elevation's spread, at least `edge` wide (the waves near the
//   eye, which leave the lobe no width, keep a soft edge), takes the horizon's colour by
//   BELOW_HORIZON.
// (A mean over the slopes finer than the pixel: what the pixel would show if it resolved them.)
export const REFLECTION_LOBE = { slopeScale: 1, lift: 2, edge: 0.05 } as const;

export interface ReflectionLobe {
	/** The slopes' spread along the view. */
	sigmaV: F;
	/** The mean Fresnel over the facets the eye sees. */
	facetFresnel: F;
	/** The lobe's centre over the mirror (an elevation's sine). */
	lift: F;
	/** The lobe's width either way, across. */
	lobeW: F;
	/** The spread of the lobe's elevation, at least `edge`: the width of its share over the horizon. */
	rise: F;
	/** The share of the reflected rays no other wave stops. */
	lobeShadow: F;
}

/** The lobe of the mean surface over the unresolved slopes (REFLECTION_LOBE) for their variance
 * `viewVar` along the view and the flat sea's view cosine `cos`, with the reflectance `fresnel`. */
export function reflectionLobe( viewVar: F, cos: F, slopeScale: F, lift: F, fresnel: ( c: F ) => F ): ReflectionLobe {

	const sigmaV = max( sqrt( viewVar ).mul( slopeScale ), 1e-5 ).toVar();
	const m = cos.div( sigmaV ).toVar();
	const cdfM = normalCdf( m ).toVar(), pdfM = normalPdf( m ).toVar();
	const e1 = cos.mul( cdfM ).add( sigmaV.mul( pdfM ) ).toVar();
	// (each moment from the two before it: E_n = cos E_n-1 + ( n - 1 ) sigmaV^2 E_n-2, E_0 = Phi( m ))
	const e2 = cos.mul( e1 ).add( sigmaV.mul( sigmaV ).mul( cdfM ) );
	const e3 = cos.mul( e2 ).add( sigmaV.mul( sigmaV ).mul( 2 ).mul( e1 ) );
	const cosVisible = min( e2.div( e1 ), 1 ).toVar();
	const cosSpread = sqrt( max( e3.div( e1 ).sub( cosVisible.mul( cosVisible ) ), 0 ) ).toVar();
	const facetFresnel = fresnel( max( cosVisible.sub( cosSpread ), 1e-4 ) ).add( fresnel( min( cosVisible.add( cosSpread ), 1 ) ) ).mul( 0.5 );
	const raised = cosVisible.sub( cos ).mul( lift ).toVar();
	const lobeY = min( cos.add( raised ), 1 ).toVar();
	const lobeW = sigmaV.mul( 2 ).toVar();
	const lobeShadow = smithG1( lobeY.sub( lobeW ), sigmaV ).mul( 0.25 ).add( smithG1( lobeY, sigmaV ).mul( 0.5 ) )
		.add( smithG1( lobeY.add( lobeW ), sigmaV ).mul( 0.25 ) ).toVar();
	return { sigmaV, facetFresnel, lift: raised, lobeW, rise: max( cosSpread.mul( lift ), REFLECTION_LOBE.edge ), lobeShadow };

}
