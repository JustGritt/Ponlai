import * as THREE from 'three/webgpu';
import { Fn, abs, cross, dot, float, length, mat3, max, min, normalize, select, sqrt, vec3 } from 'three/tsl';
import { shBasis } from './ProbeFormat.ts';

// Degree-2 spherical harmonics for the ambient light's relight, in three.js's basis.
//
// Degree-2 spherical harmonics for the ambient light's relight (AmbientRelight.ts): the products,
// the cosine lobe, the deringing window and the directionality boost, in three.js's
// SphericalHarmonics3 basis and order (y up; ProbeFormat.ts `shBasis`):
//   Y0 = 0.282095, Y1 = 0.488603 y, Y2 = 0.488603 z, Y3 = 0.488603 x, Y4 = 1.092548 x y,
//   Y5 = 1.092548 y z, Y6 = 0.315392 ( 3 z^2 - 1 ), Y7 = 1.092548 x z, Y8 = 0.546274 ( x^2 - y^2 ).
// On the GPU a function's nine coefficients travel as a mat3, the coefficient i in the column
// floor( i / 3 ), the row i mod 3, so the helpers are `Fn( ... ).setLayout( ... )` functions of
// mat3 (each one shader function, and pure: they take no uniform).
//
// - The product (`shProduct`): the projection of the product of two functions, truncated back to
//   degree 2, through the triple integrals of the basis (`GAUNT`), which the module computes when
//   it loads by a quadrature exact for these polynomials (Gauss-Legendre in z, uniform in the
//   azimuth): 83 of the 729 are not zero. The sky's radiance times a probe's sky visibility is
//   Patry's ("Real-Time Samurai Cinema: Lighting, Atmosphere, and Tonemapping in Ghost of
//   Tsushima", SIGGRAPH 2021 Advances in Real-Time Rendering in Games, p. 24); the triple product
//   is Sloan's ("Stupid Spherical Harmonics (SH) Tricks", GDC 2008).
// - The cosine lobe (`shConvolve`, Ramamoorthi and Hanrahan, "An efficient representation for
//   irradiance environment maps", SIGGRAPH 2001): radiance coefficients times pi, 2 pi / 3, pi / 4
//   per band are the irradiance's, E( n ) = sum c_i Y_i( n ), which is what three.js's
//   `shGetIrradianceAt` returns from radiance coefficients.
// - The deringing (`shDeringWindow`, `shWindow`): Sloan's window ("Deringing Spherical Harmonics",
//   SIGGRAPH Asia 2017), h_l = 1 / ( 1 + lambda l^2 ( l + 1 )^2 ), with nearly the smallest lambda
//   that leaves the function nowhere below a margin beta times its mean (beta 0: nowhere negative;
//   the caller picks beta), found on the GPU as Patry derings (p. 46), who finds the minimum with
//   one Newton step and lambda by a depth-3 search. Here the minimum's search starts from the
//   lowest of the direction opposite the linear band and rings of candidates round it, then takes
//   two Newton steps on the sphere (`shMinimumDirection`), and lambda comes in closed form at that
//   direction, over three rounds that follow the minimum as the window moves it. On sets of
//   irradiance functions (convolved sums of deltas) nearly all end no more than 1.6 % of the mean
//   below zero (AmbientRelight.ts DERING_MARGIN); an exact search would need a 3 x 3 eigensolve per
//   function.
// - The directionality boost (`shTowardDelta`, Patry p. 43): a lerp toward the projection of a
//   delta in the direction of the linear band's maximum, scaled to keep the constant band.
// - The sun's bounce lobes (`bounceLobe`, CPU): a direction projected as it is (the basis at it,
//   a constant band of Y0), windowed (Sloan's window) just enough to be nowhere negative, as Patry
//   windows the reflected light's directions (p. 33).

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
/** A function's nine coefficients on the GPU (see the header). */
export type M3 = THREE.Node<'mat3'>;

const Y00 = 0.282095, K1 = 0.488603, KXY = 1.092548, KZZ = 0.315392, KXX = 0.546274;

/** The cosine lobe's factor per coefficient. */
export const COSINE_LOBE = [ Math.PI, 2 * Math.PI / 3, 2 * Math.PI / 3, 2 * Math.PI / 3, Math.PI / 4, Math.PI / 4, Math.PI / 4, Math.PI / 4, Math.PI / 4 ];
/** The degree of each coefficient. */
export const BAND = [ 0, 1, 1, 1, 2, 2, 2, 2, 2 ];

// ------------------------------------------------------------------ the triple products

/** The triple integrals of the basis that are not zero: the integral of Y_i Y_j Y_k over the
 * sphere, so the projection of f g holds sum c f_i g_j Y_k. */
export const GAUNT: ReadonlyArray<{ i: number; j: number; k: number; c: number }> = tripleIntegrals();

function tripleIntegrals(): { i: number; j: number; k: number; c: number }[] {

	// Gauss-Legendre in z (8 nodes: exact to degree 15) times 16 azimuths (exact for trigonometric
	// degree below 16): exact for the products' degree 6
	const NODES = 8, AZIMUTHS = 16;
	const z = new Float64Array( NODES ), w = new Float64Array( NODES );
	for ( let n = 0; n < NODES; n ++ ) {

		let x = Math.cos( Math.PI * ( n + 0.75 ) / ( NODES + 0.5 ) ), dp = 1;
		for ( let it = 0; it < 64; it ++ ) {

			let p0 = 1, p1 = x;
			for ( let l = 2; l <= NODES; l ++ ) {

				const p2 = ( ( 2 * l - 1 ) * x * p1 - ( l - 1 ) * p0 ) / l;
				p0 = p1; p1 = p2;

			}
			dp = NODES * ( x * p1 - p0 ) / ( x * x - 1 );
			const dx = p1 / dp;
			x -= dx;
			if ( Math.abs( dx ) < 1e-15 ) break;

		}
		z[ n ] = x; w[ n ] = 2 / ( ( 1 - x * x ) * dp * dp );

	}
	const C = new Float64Array( 729 ), Y = new Float64Array( 9 );
	for ( let n = 0; n < NODES; n ++ ) for ( let a = 0; a < AZIMUTHS; a ++ ) {

		const r = Math.sqrt( 1 - z[ n ] * z[ n ] ), phi = 2 * Math.PI * a / AZIMUTHS, weight = w[ n ] * 2 * Math.PI / AZIMUTHS;
		shBasis( r * Math.cos( phi ), r * Math.sin( phi ), z[ n ], Y, 0 );
		for ( let i = 0; i < 9; i ++ ) for ( let j = 0; j < 9; j ++ ) for ( let k = 0; k < 9; k ++ ) C[ ( i * 9 + j ) * 9 + k ] += weight * Y[ i ] * Y[ j ] * Y[ k ];

	}
	const out: { i: number; j: number; k: number; c: number }[] = [];
	for ( let i = 0; i < 9; i ++ ) for ( let j = 0; j < 9; j ++ ) for ( let k = 0; k < 9; k ++ ) {

		const c = C[ ( i * 9 + j ) * 9 + k ];
		if ( Math.abs( c ) > 1e-6 ) out.push( { i, j, k, c } );

	}
	return out;

}

// ------------------------------------------------------------------ the sun's bounce lobes

// the lobe's window (Sloan's, the smallest lambda that leaves a delta nowhere negative)
const LOBE = bounceLobeShape();

function bounceLobeShape(): { h1: number; h2: number } {

	// a windowed delta along its axis, in t = cos( angle ): ( 1 + 3 h1 t + 5 h2 ( 3 t^2 - 1 ) / 2 ) / 4 pi
	const lowest = ( h1: number, h2: number ) => {

		const f = ( t: number ) => 1 + 3 * h1 * t + 2.5 * h2 * ( 3 * t * t - 1 );
		let m = Math.min( f( - 1 ), f( 1 ) );
		const vertex = h2 > 0 ? - h1 / ( 5 * h2 ) : 2;
		if ( Math.abs( vertex ) <= 1 ) m = Math.min( m, f( vertex ) );
		return m;

	};
	let lo = 0, hi = 16;
	for ( let it = 0; it < 64; it ++ ) {

		const lambda = ( lo + hi ) / 2;
		if ( lowest( 1 / ( 1 + 4 * lambda ), 1 / ( 1 + 36 * lambda ) ) >= 0 ) hi = lambda; else lo = lambda;

	}
	return { h1: 1 / ( 1 + 4 * hi ), h2: 1 / ( 1 + 36 * hi ) };

}

const lobeBasis = new Float64Array( 9 );

/** Adds `weight` times the sun's bounce lobe toward the unit direction ( x, y, z ) to out[ 0 .. 8 ]
 * (radiance coefficients; see the header). */
export function bounceLobe( x: number, y: number, z: number, weight: number, out: Float64Array ): void {

	shBasis( x, y, z, lobeBasis, 0 );
	for ( let i = 0; i < 9; i ++ ) out[ i ] += weight * ( BAND[ i ] === 0 ? 1 : BAND[ i ] === 1 ? LOBE.h1 : LOBE.h2 ) * lobeBasis[ i ];

}

// ------------------------------------------------------------------ the GPU helpers

const ROWS = [ 'x', 'y', 'z' ] as const;

/** The nine coefficients of a mat3 (see the header). */
export function coefficients( m: M3 ): F[] {

	const columns = [ 0, 1, 2 ].map( ( c ) => ( m as unknown as { element( i: number ): V3 } ).element( c ) );
	return Array.from( { length: 9 }, ( _, i ) => columns[ Math.floor( i / 3 ) ][ ROWS[ i % 3 ] ] as F );

}

/** Nine coefficients as a mat3 (see the header). */
export function toMat3( c: ReadonlyArray<F | number> ): M3 {

	const f = ( i: number ) => typeof c[ i ] === 'number' ? float( c[ i ] as number ) : c[ i ] as F;
	return mat3( vec3( f( 0 ), f( 1 ), f( 2 ) ), vec3( f( 3 ), f( 4 ), f( 5 ) ), vec3( f( 6 ), f( 7 ), f( 8 ) ) ) as unknown as M3;

}

// a mat3 scaled coefficient by coefficient by per-band factors
function scaleBands( m: M3, b0: F | number, b1: F | number, b2: F | number ): M3 {

	const c = coefficients( m );
	return toMat3( c.map( ( v, i ) => v.mul( [ b0, b1, b2 ][ BAND[ i ] ] ) ) );

}

/** The basis at a unit direction. */
export const shBasisAt = Fn( ( [ d ]: [ V3 ] ) => {

	return toMat3( [
		float( Y00 ), d.y.mul( K1 ), d.z.mul( K1 ), d.x.mul( K1 ),
		d.x.mul( d.y ).mul( KXY ), d.y.mul( d.z ).mul( KXY ), d.z.mul( d.z ).mul( 3 ).sub( 1 ).mul( KZZ ), d.x.mul( d.z ).mul( KXY ),
		d.x.mul( d.x ).sub( d.y.mul( d.y ) ).mul( KXX ),
	] );

} ).setLayout( { name: 'shBasisAt', type: 'mat3', inputs: [ { name: 'd', type: 'vec3' } ] } );

/** The function at a unit direction: sum c_i Y_i( n ). */
export const shEvaluate = Fn( ( [ m, n ]: [ M3, V3 ] ) => {

	const c = coefficients( m ), y = coefficients( shBasisAt( n ).toVar() as unknown as M3 );
	let sum: F = c[ 0 ].mul( y[ 0 ] );
	for ( let i = 1; i < 9; i ++ ) sum = sum.add( c[ i ].mul( y[ i ] ) );
	return sum;

} ).setLayout( { name: 'shEvaluate', type: 'float', inputs: [ { name: 'm', type: 'mat3' }, { name: 'n', type: 'vec3' } ] } );

/** The projection of a b, truncated to degree 2 (`GAUNT`). */
export const shProduct = Fn( ( [ a, b ]: [ M3, M3 ] ) => {

	const A = coefficients( a ), B = coefficients( b );
	const out: F[] = [];
	for ( let k = 0; k < 9; k ++ ) {

		// grouped by the first factor: sum_i a_i ( sum_j C_ijk b_j )
		let sum: F | null = null;
		for ( let i = 0; i < 9; i ++ ) {

			let inner: F | null = null;
			for ( const g of GAUNT ) {

				if ( g.i !== i || g.k !== k ) continue;
				const term = B[ g.j ].mul( g.c );
				inner = inner ? inner.add( term ) : term;

			}
			if ( inner ) sum = sum ? sum.add( A[ i ].mul( inner ) ) : A[ i ].mul( inner );

		}
		out.push( sum ?? float( 0 ) );

	}
	return toMat3( out );

} ).setLayout( { name: 'shProduct', type: 'mat3', inputs: [ { name: 'a', type: 'mat3' }, { name: 'b', type: 'mat3' } ] } );

/** Radiance coefficients convolved with the cosine lobe: the irradiance's (see the header). */
export const shConvolve = Fn( ( [ m ]: [ M3 ] ) => {

	return scaleBands( m, COSINE_LOBE[ 0 ], COSINE_LOBE[ 1 ], COSINE_LOBE[ 4 ] );

} ).setLayout( { name: 'shConvolve', type: 'mat3', inputs: [ { name: 'm', type: 'mat3' } ] } );

/** Sloan's window with factor lambda: degree 1 over 1 + 4 lambda, degree 2 over 1 + 36 lambda. */
export const shWindow = Fn( ( [ m, lambda ]: [ M3, F ] ) => {

	return scaleBands( m, 1, float( 1 ).div( lambda.mul( 4 ).add( 1 ) ), float( 1 ).div( lambda.mul( 36 ).add( 1 ) ) );

} ).setLayout( { name: 'shWindow', type: 'mat3', inputs: [ { name: 'm', type: 'mat3' }, { name: 'lambda', type: 'float' } ] } );

// the degree-2 band as d^T Q d on the unit sphere (Y6's constant folded in by x^2 + y^2 + z^2 = 1)
function quadraticForm( c: F[] ): M3 {

	const qxy = c[ 4 ].mul( KXY / 2 ), qyz = c[ 5 ].mul( KXY / 2 ), qxz = c[ 7 ].mul( KXY / 2 );
	const qxx = c[ 8 ].mul( KXX ).sub( c[ 6 ].mul( KZZ ) ), qyy = c[ 8 ].mul( - KXX ).sub( c[ 6 ].mul( KZZ ) ), qzz = c[ 6 ].mul( 2 * KZZ );
	return mat3( vec3( qxx, qxy, qxz ), vec3( qxy, qyy, qyz ), vec3( qxz, qyz, qzz ) ) as unknown as M3;

}

const times = ( Q: M3, d: V3 ) => ( Q as unknown as { mul( v: V3 ): V3 } ).mul( d );

// the minimum's search: candidates on rings round the direction opposite the linear band, at these
// cosines from it (a function whose degree-2 band dominates has its minimum on a ring, where the
// opposite direction is a maximum along the ring's meridian), RING_AZIMUTHS each
const RINGS = [ - 0.8, - 0.55, - 0.3, 0 ], RING_AZIMUTHS = 6, NEWTON_STEPS = 2;

/** The direction of the function's minimum on the sphere, nearly: the lowest of the direction
 * opposite the linear band and the rings round it, then NEWTON_STEPS Newton steps on the sphere. */
export const shMinimumDirection = Fn( ( [ m ]: [ M3 ] ) => {

	const c = coefficients( m );
	const b = vec3( c[ 3 ], c[ 1 ], c[ 2 ] ).mul( K1 ).toVar();
	const Q = ( quadraticForm( c ) as unknown as { toVar(): M3 } ).toVar();
	const f = ( d: V3 ) => dot( b, d ).add( dot( d, times( Q, d ) ) );
	// the frame: a along the linear band (up where it vanishes), u and v across it
	const bl = length( b ).toVar();
	const along = b.div( max( bl, 1e-20 ) ).toVar(), up = vec3( 0, 1, 0 ).toVar(), side = vec3( 1, 0, 0 ).toVar();
	const a = select( bl.greaterThan( 1e-20 ), along, up ).toVar();
	const u = normalize( cross( a, select( abs( a.y ).lessThan( 0.9 ), up, side ) ) ).toVar();
	const v = cross( a, u ).toVar();
	const best = a.negate().toVar();
	const lowest = f( best ).toVar();
	for ( const t of RINGS ) for ( let k = 0; k < RING_AZIMUTHS; k ++ ) {

		const phi = 2 * Math.PI * k / RING_AZIMUTHS, s = Math.sqrt( 1 - t * t );
		const e = a.mul( t ).add( u.mul( s * Math.cos( phi ) ) ).add( v.mul( s * Math.sin( phi ) ) ).toVar();
		const fe = f( e ).toVar();
		const lower = fe.lessThan( lowest );
		best.assign( select( lower, e, best ) );
		lowest.assign( select( lower, fe, lowest ) );

	}
	for ( let step = 0; step < NEWTON_STEPS; step ++ ) {

		// the gradient and Hessian in the tangent plane (the Hessian's second term is the sphere's
		// curvature: - ( G . d ) on the plane)
		const G = b.add( times( Q, best ).mul( 2 ) ).toVar();
		const gd = dot( G, best ).toVar();
		const t1 = normalize( cross( best, select( abs( best.y ).lessThan( 0.9 ), up, side ) ) ).toVar();
		const t2 = cross( best, t1 ).toVar();
		const g1 = dot( G, t1 ).toVar(), g2 = dot( G, t2 ).toVar();
		const Qt1 = times( Q, t1 ).toVar(), Qt2 = times( Q, t2 ).toVar();
		const h11 = dot( t1, Qt1 ).mul( 2 ).sub( gd ).toVar(), h22 = dot( t2, Qt2 ).mul( 2 ).sub( gd ).toVar(), h12 = dot( t1, Qt2 ).mul( 2 ).toVar();
		const det = h11.mul( h22 ).sub( h12.mul( h12 ) ).toVar();
		// (no step where the Hessian is not positive definite)
		const convex = select( h11.greaterThan( 0 ).and( det.greaterThan( 1e-30 ) ), float( 1 ), float( 0 ) ).toVar();
		const inverse = convex.div( max( det, 1e-30 ) ).toVar();
		const su = h22.mul( g1 ).sub( h12.mul( g2 ) ).mul( inverse ).negate().toVar();
		const sv = h11.mul( g2 ).sub( h12.mul( g1 ) ).mul( inverse ).negate().toVar();
		// (a step of at most a radian, kept only if it goes lower)
		const reach = min( float( 1 ), float( 1 ).div( max( sqrt( su.mul( su ).add( sv.mul( sv ) ) ), 1e-20 ) ) ).toVar();
		const next = normalize( best.add( t1.mul( su.mul( reach ) ) ).add( t2.mul( sv.mul( reach ) ) ) ).toVar();
		const fn = f( next ).toVar();
		const lower = fn.lessThan( lowest );
		best.assign( select( lower, next, best ) );
		lowest.assign( select( lower, fn, lowest ) );

	}
	return best;

} ).setLayout( { name: 'shMinimumDirection', type: 'vec3', inputs: [ { name: 'm', type: 'mat3' } ] } );

// the window's rounds (see shDeringWindow)
const DERING_ROUNDS = 3;

/** The smallest lambda of Sloan's window that leaves the function nowhere below `beta` times its
 * mean A, nearly (0 where it is nowhere below that, or where its mean is not positive, which no
 * window can lift). DERING_ROUNDS rounds: the minimum of the function windowed with the lambda so
 * far (`shMinimumDirection`), then the largest lambda that brings the function at that direction
 * to beta A, A + B / ( 1 + 4 lambda ) + C / ( 1 + 36 lambda ) = beta A (A, B, C the bands' values
 * there), in closed form with A' = ( 1 - beta ) A: 144 A' l^2 + ( 40 A' + 36 B + 4 C ) l + A' + B
 * + C = 0. A window moves the minimum (round the ring of a lobe's back), so the rounds follow
 * it. */
export const shDeringWindow = Fn( ( [ m, beta ]: [ M3, F ] ) => {

	const c = coefficients( m );
	const mean = c[ 0 ].mul( Y00 ).toVar();
	const A = mean.mul( float( 1 ).sub( beta ) ).toVar();
	const lambda = float( 0 ).toVar();
	for ( let round = 0; round < DERING_ROUNDS; round ++ ) {

		const d = shMinimumDirection( shWindow( m, lambda ) ).toVar();
		const B = d.y.mul( c[ 1 ] ).add( d.z.mul( c[ 2 ] ) ).add( d.x.mul( c[ 3 ] ) ).mul( K1 ).toVar();
		const C = shEvaluate( m, d ).sub( mean ).sub( B ).toVar();
		const qa = A.mul( 144 ).toVar(), qb = A.mul( 40 ).add( B.mul( 36 ) ).add( C.mul( 4 ) ).toVar(), qc = A.add( B ).add( C ).toVar();
		const disc = qb.mul( qb ).sub( qa.mul( qc ).mul( 4 ) ).toVar();
		const root = sqrt( max( disc, 0 ) ).toVar();
		// the roots q / qa and qc / q, q = - ( qb + sign( qb ) root ) / 2 (the stable form)
		const q = qb.add( select( qb.greaterThanEqual( 0 ), root, root.negate() ) ).mul( - 0.5 ).toVar();
		const safe = select( abs( q ).greaterThan( 1e-30 ), q, float( - 1e-30 ) ).toVar();
		const largest = max( q.div( max( qa, 1e-30 ) ), qc.div( safe ) ).toVar();
		const zeroes = A.greaterThan( 0 ).and( disc.greaterThanEqual( 0 ) );
		lambda.assign( max( lambda, select( zeroes, largest, float( 0 ) ) ) );

	}
	return lambda;

} ).setLayout( { name: 'shDeringWindow', type: 'float', inputs: [ { name: 'm', type: 'mat3' }, { name: 'beta', type: 'float' } ] } );

/** The directionality boost: a lerp by `amount` toward the projection of a delta in the unit
 * direction `d`, scaled to keep the constant band (see the header). */
export const shTowardDelta = Fn( ( [ m, d, amount ]: [ M3, V3, F ] ) => {

	const c = coefficients( m ), y = coefficients( shBasisAt( d ).toVar() as unknown as M3 );
	const scale = c[ 0 ].div( Y00 ).toVar();
	return toMat3( c.map( ( v, i ) => i === 0 ? v : v.add( y[ i ].mul( scale ).sub( v ).mul( amount ) ) ) );

} ).setLayout( { name: 'shTowardDelta', type: 'mat3', inputs: [ { name: 'm', type: 'mat3' }, { name: 'd', type: 'vec3' }, { name: 'amount', type: 'float' } ] } );

/** The direction of the linear band's maximum of three channels' coefficients, by luminance, and
 * whether it has one (1, or 0 where the linear band vanishes). */
export function linearMaximum( r: M3, g: M3, b: M3 ): { direction: V3; has: F } {

	const band = ( m: M3 ) => { const c = coefficients( m ); return vec3( c[ 3 ], c[ 1 ], c[ 2 ] ); };
	const v = band( r ).mul( 0.2126 ).add( band( g ).mul( 0.7152 ) ).add( band( b ).mul( 0.0722 ) ).toVar();
	const l = length( v ).toVar();
	return { direction: v.div( max( l, 1e-20 ) ), has: select( l.greaterThan( 1e-20 ), float( 1 ), float( 0 ) ) };

}
