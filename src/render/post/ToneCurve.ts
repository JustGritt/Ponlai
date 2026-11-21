// The display's tone curve: a rational function per channel, solved from six constraints.
//
// The display's tone curve, one per channel in the curve's space (ToneSpace.ts), on the exposed
// colour directly: a rational function of the third degree over the third,
//   y = clamp( ( a x^3 + b x^2 + c x ) / ( d x^3 + e x^2 + f x + 1 ), 0, 1 )
// a smooth toe from black (its slope there c), mid grey where `mid` puts it and a long shoulder
// into white at `white`. Its six coefficients are solved on the CPU from six constraints, at load
// and when the GUI changes one:
//   1. the slope at black is `toe` (c = s0);
//   2. y( 0.18 ) = `mid`;
//   3. the log-log slope at 0.18 is MID_SLOPE, 1: neutral local contrast at mid grey;
//   4. y( `white` ) = 1;
//   5. white y'( white ) = WHITE_SLOPE: nearly flat into white;
//   6. y( 1 ) = ONE.
// The value constraints are linear in the coefficients once multiplied by the denominator. A slope
// constraint at a point whose value is fixed is linear too: y' = ( N' - y D' ) / D, so x y' = k
// reads N'( x ) - y D'( x ) = ( k / x ) D( x ). The six are solved as one linear system, and a few
// Newton steps on the constraints as written then take out the rounding. The solution is checked:
// the denominator stays positive and y rises over [ 0, white ], and a / d > 1 (the curve's
// asymptote over white, so it reaches 1 before it flattens). A failure of the default set
// (look/DisplayLook.ts `toneCurve`) throws in development; any other set that fails (a GUI
// combination) warns and the pipeline keeps the last good curve (Pipeline.ts).
// ONE was measured once on the curve before this one (Lottes's form with a bent toe, two and a half
// stops over its mid grey) and fixed, so the lights keep the level they had. That curve's log-log
// slope at mid grey, 1.10, leaves no solution without a pole with a slope of 2.2 at black (the
// denominator crosses zero in the toe; the system is linear, so its solution is the only one), and
// a slope of 1 or under does; 1 is the neutral choice, tunable. Pure TypeScript, so
// node runs it (strip types).

export interface ToneCurve { toe: number; mid: number; white: number }

// the log-log slope at mid grey, and y( 1 ), the previous curve's (see above)
const MID_SLOPE = 1;
const ONE = 0.666;
// white times the slope at white
const WHITE_SLOPE = 0.02;
const MID_GREY = 0.18;

/** The coefficients [ a, b, c, d, e, f ] (see the header). */
export function solveToneCurve( { toe, mid, white }: ToneCurve ): number[] {

	// each row: the coefficients of a, b, d, e, f (c fixed at `toe`) and the right-hand side
	// y( x ) = y: N( x ) - y D( x ) = 0
	const value = ( x: number, y: number ) => [ x ** 3, x ** 2, - y * x ** 3, - y * x ** 2, - y * x, y - toe * x ];
	// x y'( x ) = k where y( x ) = y: N'( x ) - y D'( x ) - ( k / x ) D( x ) = 0
	const slope = ( x: number, y: number, k: number ) => {

		const r = k / x;
		return [ 3 * x ** 2, 2 * x, - 3 * y * x ** 2 - r * x ** 3, - 2 * y * x - r * x ** 2, - y - r * x, r - toe ];

	};
	const rows = [
		value( MID_GREY, mid ),
		// (a log-log slope s at mid grey: x y' = s y)
		slope( MID_GREY, mid, MID_SLOPE * mid ),
		value( white, 1 ),
		slope( white, 1, WHITE_SLOPE ),
		value( 1, ONE ),
	];
	const [ a, b, d, e, f ] = solve( rows );
	const k = [ a, b, toe, d, e, f ];
	// Newton on the five constraints as written (c stays the slope at black)
	for ( let step = 0; step < 4; step ++ ) {

		const r0 = residuals( k, mid, white );
		const J = [ 0, 1, 3, 4, 5 ].map( ( i ) => {

			const h = 1e-7 * Math.max( 1, Math.abs( k[ i ] ) ), kk = k.slice();
			kk[ i ] += h;
			return residuals( kk, mid, white ).map( ( v, j ) => ( v - r0[ j ] ) / h );

		} );
		// rows: constraints; columns: the five free coefficients
		const sys = r0.map( ( v, j ) => [ ...J.map( ( col ) => col[ j ] ), - v ] );
		const dk = solve( sys );
		[ 0, 1, 3, 4, 5 ].forEach( ( i, n ) => k[ i ] += dk[ n ] );

	}
	return k;

}

/** y at `x` for the coefficients `k`. */
export function toneAt( k: readonly number[], x: number ): number {

	const [ a, b, c, d, e, f ] = k;
	return Math.min( Math.max( ( ( a * x + b ) * x + c ) * x / ( ( ( d * x + e ) * x + f ) * x + 1 ), 0 ), 1 );

}

/** What fails the checks (see the header), or null. */
export function checkToneCurve( k: readonly number[], white: number ): string | null {

	const [ a, , , d, e, f ] = k;
	if ( ! ( a / d > 1 ) ) return `a / d = ${ a / d } is not over 1`;
	let last = - Infinity;
	for ( let i = 0; i <= 4096; i ++ ) {

		const x = white * i / 4096;
		if ( ! ( ( ( d * x + e ) * x + f ) * x + 1 > 0 ) ) return `the denominator is not positive at ${ x }`;
		const y = raw( k, x );
		if ( y < last - 1e-9 ) return `y falls at ${ x }`;
		last = y;

	}
	return null;

}

function raw( k: readonly number[], x: number ): number {

	const [ a, b, c, d, e, f ] = k;
	return ( ( a * x + b ) * x + c ) * x / ( ( ( d * x + e ) * x + f ) * x + 1 );

}

// the five constraints after the slope at black, as residuals
function residuals( k: readonly number[], mid: number, white: number ): number[] {

	const h = 1e-5;
	const logSlope = ( x: number ) => ( Math.log( raw( k, x * Math.exp( h ) ) ) - Math.log( raw( k, x * Math.exp( - h ) ) ) ) / ( 2 * h );
	const slopeAt = ( x: number ) => ( raw( k, x * ( 1 + h ) ) - raw( k, x * ( 1 - h ) ) ) / ( 2 * h * x );
	return [
		raw( k, MID_GREY ) - mid,
		logSlope( MID_GREY ) - MID_SLOPE,
		raw( k, white ) - 1,
		white * slopeAt( white ) - WHITE_SLOPE,
		raw( k, 1 ) - ONE,
	];

}

// Gaussian elimination with partial pivoting on an augmented n x ( n + 1 ) system
function solve( rows: number[][] ): number[] {

	const m = rows.map( ( r ) => r.slice() ), n = m.length;
	for ( let i = 0; i < n; i ++ ) {

		let p = i;
		for ( let j = i + 1; j < n; j ++ ) if ( Math.abs( m[ j ][ i ] ) > Math.abs( m[ p ][ i ] ) ) p = j;
		[ m[ i ], m[ p ] ] = [ m[ p ], m[ i ] ];
		for ( let j = i + 1; j < n; j ++ ) {

			const t = m[ j ][ i ] / m[ i ][ i ];
			for ( let c = i; c <= n; c ++ ) m[ j ][ c ] -= t * m[ i ][ c ];

		}

	}
	const x = new Array( n ).fill( 0 );
	for ( let i = n - 1; i >= 0; i -- ) {

		let s = m[ i ][ n ];
		for ( let c = i + 1; c < n; c ++ ) s -= m[ i ][ c ] * x[ c ];
		x[ i ] = s / m[ i ][ i ];

	}
	return x;

}
