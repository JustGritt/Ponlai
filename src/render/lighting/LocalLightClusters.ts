import { REFRACTION_GUARD_BOUND } from '../ocean/Refraction.ts';

// The local lights' cluster grid and its CPU side: the depth slices and each frame's sort.
//
// The local lights' cluster grid and its CPU side (lighting/LocalLights.ts has the why): the grid,
// the depth slices, and each frame's sort of the sources in view with every slice's range in it.
// Plain TypeScript with no TSL, so node runs it (tools/render/verify-local-lights.mjs).

/** The screen tile, px at the 2K baseline, and the cluster grid: tiles across, down, depth slices. */
export const TILE = 32;
export const GRID = { x: 2560 / TILE, y: 1440 / TILE, z: 24 } as const;
export const CLUSTERS = GRID.x * GRID.y * GRID.z;
/** The most sources a cluster lists; more are truncated in depth order. */
export const PER_CLUSTER = 32;
/** A cluster's words: the count of the sources touching it, then its list, two 16-bit indices a word. */
export const CLUSTER_WORDS = 1 + PER_CLUSTER / 2;
/** The first slice's far end, m: from the camera to here, the rest exponential beyond. */
export const FIRST_SLICE = 5;

/** The refraction's guard band in NDC units on each side at its widest (ocean/Refraction.ts), which
 * the edge tiles take in: the band below follows the camera, and the clusters cover all it can reach. */
export const GUARD = REFRACTION_GUARD_BOUND;

/** The slices' three numbers: the first's far end, the last's, and k = ( GRID.z - 1 ) / ln( last / first ),
 * for the farthest reach of a source in view `reach` (m). */
export function slicesFor( reach: number ): [ number, number, number ] {

	const first = FIRST_SLICE, last = Math.max( reach, 2 * first );
	return [ first, last, ( GRID.z - 1 ) / Math.log( last / first ) ];

}

/** Where slice s starts (m of view depth): 0 for the first, first e^( ( s - 1 ) / k ) after it. */
export function sliceStart( s: number, [ first, , k ]: readonly [ number, number, number ] ): number {

	return s === 0 ? 0 : first * Math.exp( ( s - 1 ) / k );

}

/** The slice of a view depth (m). */
export function sliceAt( depth: number, [ first, , k ]: readonly [ number, number, number ] ): number {

	return depth < first ? 0 : Math.min( Math.max( Math.floor( Math.log( Math.max( depth, 1e-6 ) / first ) * k ) + 1, 1 ), GRID.z - 1 );

}

export interface InView {
	/** the sources in view, by view depth, then index */
	order: number[];
	/** each slice's range in `order`, ( start, end ) */
	ranges: Uint32Array;
	slices: [ number, number, number ];
}

let depths = new Float64Array( 1024 );

/** The sources whose spheres meet the main view widened by the guard band and lie past `near`,
 * sorted by view depth, then index (a deterministic order), and each slice's range in that order.
 * `view` holds each source's view-space position and range ( x, y, z, r ); `e` the projection's
 * elements (column-major: P00 e[ 0 ], P11 e[ 5 ], P20 e[ 8 ], P21 e[ 9 ]). */
export function sortInView( view: ArrayLike<number>, n: number, e: ArrayLike<number>, near: number ): InView {

	if ( depths.length < n ) depths = new Float64Array( n );
	const order: number[] = [];
	// the side planes through the eye: x = d ( ndc + P20 ) / P00 at the band's edges (y alike)
	const ax0 = ( - 1 - GUARD.left + e[ 8 ] ) / e[ 0 ], ax1 = ( 1 + GUARD.right + e[ 8 ] ) / e[ 0 ];
	const ay0 = ( - 1 - GUARD.bottom + e[ 9 ] ) / e[ 5 ], ay1 = ( 1 + GUARD.top + e[ 9 ] ) / e[ 5 ];
	let reach = 0;
	for ( let i = 0; i < n; i ++ ) {

		const x = view[ i * 4 ], y = view[ i * 4 + 1 ], d = - view[ i * 4 + 2 ], r = view[ i * 4 + 3 ];
		depths[ i ] = d;
		if ( d + r < near ) continue;
		// the sphere's signed distances to the four planes, inside positive
		if ( ( x - d * ax0 ) / Math.hypot( 1, ax0 ) < - r || ( d * ax1 - x ) / Math.hypot( 1, ax1 ) < - r ) continue;
		if ( ( y - d * ay0 ) / Math.hypot( 1, ay0 ) < - r || ( d * ay1 - y ) / Math.hypot( 1, ay1 ) < - r ) continue;
		order.push( i );
		reach = Math.max( reach, d + r );

	}
	order.sort( ( a, b ) => depths[ a ] - depths[ b ] || a - b );
	const slices = slicesFor( reach );
	// slice s spans [ start( s ), start( s + 1 ) ], the last to infinity. (In doubles where the kernel
	// takes floats: a source a slice's range leaves out by that rounding reaches into the slice by no
	// more than it, where its window is 0.)
	const ranges = new Uint32Array( 2 * GRID.z );
	for ( let s = 0; s < GRID.z; s ++ ) {

		const z0 = sliceStart( s, slices ), z1 = s === GRID.z - 1 ? Infinity : sliceStart( s + 1, slices );
		let a = order.length, b = 0;
		for ( let q = 0; q < order.length; q ++ ) {

			const i = order[ q ], d = depths[ i ], r = view[ i * 4 + 3 ];
			if ( d + r >= z0 && d - r <= z1 ) {

				a = Math.min( a, q );
				b = q + 1;

			}

		}
		if ( a < b ) {

			ranges[ 2 * s ] = a;
			ranges[ 2 * s + 1 ] = b;

		}

	}
	return { order, ranges, slices };

}
