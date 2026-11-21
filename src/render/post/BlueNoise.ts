import * as THREE from 'three/webgpu';

// A tile of blue noise for the output's dither, made at load by void-and-cluster.
//
// A tile of blue noise for the output's dither (PipelineDisplay.ts `finish`): BLUE_NOISE x
// BLUE_NOISE thresholds n in [ 0, 1 ), each a pixel's rank among the tile's, made at load by
// Ulichney's void-and-cluster ("The void-and-cluster method for dither array generation", SPIE
// 1993) from a fixed seed, so it is the same tile on every load and no asset. Blue noise breaks the
// 8-bit output's bands with the least visible grain; static, it does not shimmer and a paused frame
// repeats. 64 is the smallest tile whose repetition does not show at 4K.
// The method: an initial pattern of a tenth of the pixels (seeded), relaxed by moving the pixel in
// its tightest cluster to its largest void until that move would put it back; ranks below the
// initial count by removing the tightest cluster's pixel one at a time, and above it by filling the
// largest void one at a time (past half the tile the largest void of the ones is the tightest
// cluster of the zeros, Ulichney's third phase). Clusters and voids are measured by a Gaussian of
// sigma 1.5 pixels, wrapped around the tile, so the tile repeats seamlessly. About 30 ms at load.

export const BLUE_NOISE = 64;
const SIGMA = 1.5;
const SEED = 0x9e3779b9;

/** The tile's thresholds, row by row ( x + y BLUE_NOISE ), each rank / BLUE_NOISE^2. */
export function blueNoise( size = BLUE_NOISE, seed = SEED ): Float32Array {

	const n = size * size;
	// the wrapped Gaussian's weights by offset
	const kernel = new Float32Array( n );
	for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

		const dx = Math.min( x, size - x ), dy = Math.min( y, size - y );
		kernel[ x + y * size ] = Math.exp( - ( dx * dx + dy * dy ) / ( 2 * SIGMA * SIGMA ) );

	}
	const on = new Uint8Array( n );
	const energy = new Float32Array( n );
	const add = ( i: number, sign: number ) => {

		const ix = i % size, iy = ( i / size ) | 0;
		for ( let y = 0; y < size; y ++ ) {

			const row = ( ( y - iy + size ) % size ) * size, out = y * size;
			for ( let x = 0; x < size; x ++ ) energy[ out + x ] += sign * kernel[ row + ( x - ix + size ) % size ];

		}

	};
	// the tightest cluster: the set pixel of the most energy; the largest void: the unset pixel of
	// the least (ties to the first, so the result is fixed)
	const tightest = () => {

		let best = - 1, e = - Infinity;
		for ( let i = 0; i < n; i ++ ) if ( on[ i ] && energy[ i ] > e ) { e = energy[ i ]; best = i; }
		return best;

	};
	const largestVoid = () => {

		let best = - 1, e = Infinity;
		for ( let i = 0; i < n; i ++ ) if ( ! on[ i ] && energy[ i ] < e ) { e = energy[ i ]; best = i; }
		return best;

	};

	// the initial pattern, from a seeded stream (mulberry32)
	let s = seed >>> 0;
	const random = () => {

		s = ( s + 0x6d2b79f5 ) >>> 0;
		let t = s;
		t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
		t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};
	const initial = Math.round( n / 10 );
	for ( let placed = 0; placed < initial; ) {

		const i = Math.floor( random() * n );
		if ( on[ i ] ) continue;
		on[ i ] = 1; add( i, 1 ); placed ++;

	}
	for ( let guard = 0; guard < n; guard ++ ) {

		const c = tightest();
		on[ c ] = 0; add( c, - 1 );
		const v = largestVoid();
		on[ v ] = 1; add( v, 1 );
		if ( v === c ) break;

	}

	const rank = new Int32Array( n );
	const start = on.slice(), startEnergy = energy.slice();
	// ranks below the initial count: remove the tightest cluster's pixel
	for ( let r = initial - 1; r >= 0; r -- ) {

		const c = tightest();
		on[ c ] = 0; add( c, - 1 );
		rank[ c ] = r;

	}
	// ranks from the initial count up: fill the largest void
	on.set( start ); energy.set( startEnergy );
	for ( let r = initial; r < n; r ++ ) {

		const v = largestVoid();
		on[ v ] = 1; add( v, 1 );
		rank[ v ] = r;

	}
	const out = new Float32Array( n );
	for ( let i = 0; i < n; i ++ ) out[ i ] = rank[ i ] / n;
	return out;

}

/** The tile as a texture (single channel, float, nearest), read by textureLoad at a pixel modulo
 * BLUE_NOISE. */
export function blueNoiseTexture(): THREE.DataTexture {

	const t = new THREE.DataTexture( blueNoise(), BLUE_NOISE, BLUE_NOISE, THREE.RedFormat, THREE.FloatType );
	t.minFilter = t.magFilter = THREE.NearestFilter;
	t.wrapS = t.wrapT = THREE.RepeatWrapping;
	t.generateMipmaps = false;
	t.colorSpace = THREE.NoColorSpace;
	t.needsUpdate = true;
	return t;

}
