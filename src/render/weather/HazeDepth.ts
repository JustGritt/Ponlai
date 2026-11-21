// The haze's medium in closed form, shared by the haze and the clouds' march.
//
// The haze's medium in closed form, shared by the haze (VolumetricHaze.ts, whose header describes
// the medium) and the clouds' march (sky/Clouds.ts, which ends where the haze closes):
//   - `meanExtinction`: the medium's mean density over a span of heights, so an optical depth is a
//     ray's length times it;
//   - `hazeReach`: the distance along a ray at which the haze lets less than 1/500 through
//     (`REACH_DEPTH`), 8-bit's floor: past it nothing on the ray can show through the haze;
//   - `hazeFormKeep` (the haze's alone): the share of a column's scattered light that still keeps
//     its form (VolumetricHaze.ts's header: a deck's light along a long ray);
//   - `HazeMedium`: the medium's uniforms a reader outside the haze holds (the clouds build their
//     march before the haze exists), which the haze writes whenever it sets its own
//     (VolumetricHaze.ts `prepareMedium`).
// Rays here are straight and climb with the direction's y alone, as the haze's surfaces take them
// (VolumetricHaze.ts `skyHaze`).

import * as THREE from 'three/webgpu';
import { Fn, If, Loop, exp, float, log, max, min, select, sqrt, uniform } from 'three/tsl';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

// (1 - e^-x) / x, the share of a slab of optical depth x that scatters, per unit of that depth, with
// its series near 0 (where the difference loses its digits)
export const expShare = Fn( ( [ x ]: [ F ] ) => {

	const series = float( 1 ).sub( x.mul( 0.5 ) ).add( x.mul( x ).div( 6 ) );
	return select( x.lessThan( 1e-3 ), series, float( 1 ).sub( exp( x.negate() ) ).div( max( x, 1e-30 ) ) );

} ).setLayout( { name: 'hazeExpShare', type: 'float', inputs: [ { name: 'x', type: 'float' } ] } );

// The medium's mean density (1/m, before each band's sigma) over the heights between a and b
// (metres, either order; below the sea it holds its value at sea level, which only rays hidden
// under the water reach): the exponential layer (density at sea level, scale height, top) and the
// ground layer (rho_f, top, width). An optical depth is a ray's length times the mean over the
// heights it spans. The exponential layer ends at its top (a low deck's base: HazeKeys.ts
// `aerosolTop`; far above the atmosphere without one): its mean over the span's part under the top,
// times that part's share of the span. A span wholly under the top keeps its depth to the bit, so a
// sky without a deck does; a point (a level ray) over the top has none of it.
export const meanExtinction = Fn( ( [ a0, b0, layer, ground ]: [ F, F, V3, V3 ] ) => {

	const a = max( min( a0, b0 ), 0 ).toVar();
	const b = max( max( a0, b0 ), 0 ).toVar();
	const span = b.sub( a ).toVar();
	// the exponential layer over the span's part under the top, its density scaled by that part's
	// share of the span where the span reaches over the top: below it the same operations on the
	// same values as without a top, so they round alike
	const low = min( a, layer.z ), high = min( b, layer.z );
	const under = high.sub( low ).toVar();
	const density = select( b.greaterThan( layer.z ), layer.x.mul( under.div( max( span, 1e-30 ) ) ), layer.x ).uniformFlow();
	const exponential = density.mul( exp( low.div( layer.y ).negate() ) ).mul( expShare( under.div( layer.y ) ) );
	// the ground layer's weight 1 - u / r integrates to u - r, r = sqrt( u^2 + w^2 ); its mean,
	// written so that no difference cancels: 1 - ( ua + ub ) / ( ra + rb )
	const ua = a.sub( ground.y ).toVar(), ub = b.sub( ground.y ).toVar();
	const w2 = ground.z.mul( ground.z );
	const r = sqrt( ua.mul( ua ).add( w2 ) ).add( sqrt( ub.mul( ub ).add( w2 ) ) );
	const weight = float( 1 ).sub( ua.add( ub ).div( max( r, 1e-6 ) ) );
	return exponential.add( ground.x.mul( weight ) );

} ).setLayout( { name: 'hazeMeanExtinction', type: 'float', inputs: [
	{ name: 'a0', type: 'float' }, { name: 'b0', type: 'float' }, { name: 'layer', type: 'vec3' }, { name: 'ground', type: 'vec3' },
] } );

// The share of the light a column of the air's optical depth `depth` sends the eye that still keeps
// its form, the air's anisotropy `g` (see the header: a deck's light along a long ray):
// ( 1 - e^-( 2 - g ) T ) / ( ( 2 - g ) ( 1 - e^-T ) ), 1 on a short column.
export const hazeFormKeep = Fn( ( [ depth, g ]: [ F, F ] ) => {

	const a = float( 2 ).sub( g );
	return expShare( a.mul( depth ) ).div( max( expShare( depth ), 1e-30 ) );

} ).setLayout( { name: 'hazeFormKeep', type: 'float', inputs: [ { name: 'depth', type: 'float' }, { name: 'g', type: 'float' } ] } );

// The exponential layer's top where nothing ends it (no deck): over any height a ray reaches.
export const AEROSOL_OPEN = 1e9;

// The optical depth at which a ray lets 1/500 through.
export const REACH_DEPTH = Math.log( 500 );

// The distance (m) along a ray from height y (m) climbing dy a metre at which the medium's optical
// depth, times `sigma` (the least extinguished band's), reaches REACH_DEPTH, searched between t0 and
// t1: t1 where it stays under it there, and about t0 where it is past it already. The depth grows along the ray, so a
// bisection in the log of the distance finds it, 12 halvings to under 0.2 % of the distance between
// 100 m and 250 km.
export const hazeReach = Fn( ( [ dy, y, layer, ground, sigma, t0, t1 ]: [ F, F, V3, V3, F, F, F ] ) => {

	const depth = ( t: F ) => t.mul( meanExtinction( y, y.add( dy.mul( t ) ), layer, ground ) ).mul( sigma );
	const reach = float( t1 ).toVar();
	If( depth( t1 ).greaterThan( REACH_DEPTH ), () => {

		const low = log( max( t0, 1 ) ).toVar(), high = log( max( t1, 1 ) ).toVar();
		Loop( 12, () => {

			const middle = low.add( high ).mul( 0.5 ).toVar();
			If( depth( exp( middle ) ).greaterThan( REACH_DEPTH ), () => {

				high.assign( middle );

			} ).Else( () => {

				low.assign( middle );

			} );

		} );
		reach.assign( exp( high ) );

	} );
	return reach;

} ).setLayout( { name: 'hazeReach', type: 'float', inputs: [
	{ name: 'dy', type: 'float' }, { name: 'y', type: 'float' }, { name: 'layer', type: 'vec3' }, { name: 'ground', type: 'vec3' },
	{ name: 'sigma', type: 'float' }, { name: 't0', type: 'float' }, { name: 't1', type: 'float' },
] } );

/** The haze's medium as uniforms of a reader's own, set from the haze's (VolumetricHaze.ts
 * `prepareMedium`), and the reach along a ray in it. */
export class HazeMedium {

	readonly layer = uniform( new THREE.Vector3( 0, 1400, AEROSOL_OPEN ) );
	readonly ground = uniform( new THREE.Vector3( 0, 14, 9.431 ) );
	// the least extinguished band's sigma, the last to close: 0 (no haze over the sky, or no haze
	// yet) reaches every distance
	readonly sigma = uniform( 0 );

	/** `hazeReach` from height y (m) along a ray climbing dy a metre, searched between t0 and t1 (m). */
	reach( dy: F, y: F, t0: F, t1: F ): F {

		return hazeReach( dy, y, this.layer, this.ground, this.sigma, t0, t1 );

	}

}
