import * as THREE from 'three/webgpu';

// Work in steps of a few tenths of a millisecond: typed arrays and bounds, sliced.
//
// Work in steps of a few tenths of a millisecond, for builders that run as generators (the growth's
// idle-time steps, Growth.ts): typed arrays from long plain arrays a slice at a time, and a
// geometry's bounds. Each gives the same values as the one-shot call three makes
// (`Float32BufferAttribute`, `setIndex` with an array, `computeBoundingBox`, `computeBoundingSphere`).
// An app's first build runs the same generators to the end in one go (`runSteps`).

// values a step (a slice of a long array)
const SLICE = 1 << 14;
// a Growing's block: 2^16 values
const SHIFT = 16, BLOCK = 1 << SHIFT, MASK = BLOCK - 1;

/** A list of numbers (doubles, as a plain array holds them) that grows a block at a time: a plain
 * array copies everything it holds each time it outgrows its store, which for the hundreds of
 * thousands of values a builder pushes took milliseconds in one go (several arrays growing at
 * the same push). */
export class Growing {

	length = 0;
	private readonly blocks: Float64Array[] = [];
	private last = new Float64Array( 0 );
	private at = BLOCK;

	push( ...values: number[] ): void {

		for ( let k = 0; k < values.length; k ++ ) {

			if ( this.at === BLOCK ) {

				this.blocks.push( this.last = new Float64Array( BLOCK ) );
				this.at = 0;

			}
			this.last[ this.at ++ ] = values[ k ];
			this.length ++;

		}

	}

	get( i: number ): number {

		return this.blocks[ i >> SHIFT ][ i & MASK ];

	}

	/** Its values from `start` up to `end` (within one block). */
	slice( start: number, end: number ): Float64Array {

		return this.blocks[ start >> SHIFT ].subarray( start & MASK, ( ( start & MASK ) + end - start ) );

	}

}

/** `new Float32Array( values )` in steps. */
export function* float32Steps( values: Growing ): Generator<void, Float32Array> {

	const out = new Float32Array( values.length );
	for ( let k = 0; k < values.length; k += SLICE ) {

		// (a block holds whole slices; the copy rounds each double to a float, as an assignment does)
		out.set( values.slice( k, Math.min( k + SLICE, values.length ) ), k );
		yield;

	}
	return out;

}

/** The index three's `setIndex( array )` makes (16-bit unless a value needs 32), in steps. */
export function* indexSteps( values: Growing ): Generator<void, THREE.BufferAttribute> {

	// (three's arrayNeedsUint32)
	let wide = false;
	for ( let k = 0; k < values.length && ! wide; k += SLICE ) {

		const part = values.slice( k, Math.min( k + SLICE, values.length ) );
		for ( let i = 0; i < part.length; i ++ ) if ( part[ i ] >= 65535 ) { wide = true; break; }
		yield;

	}
	const out = wide ? new Uint32Array( values.length ) : new Uint16Array( values.length );
	for ( let k = 0; k < values.length; k += SLICE ) {

		out.set( values.slice( k, Math.min( k + SLICE, values.length ) ), k );
		yield;

	}
	return new THREE.BufferAttribute( out, 1 );

}

/** Runs a builder's steps to the end at once, and returns its result. */
export function runSteps<T>( steps: Generator<unknown, T> ): T {

	let next = steps.next();
	while ( ! next.done ) next = steps.next();
	return next.value;

}

const _v = new THREE.Vector3();

/** `geometry`'s bounding box and sphere from its positions, in steps: what its computeBoundingBox
 * and computeBoundingSphere find (the box's centre, then the farthest point from it), for a
 * geometry without morph targets. */
export function* boundsSteps( geometry: THREE.BufferGeometry ): Generator<void, void> {

	const position = geometry.getAttribute( 'position' ) as THREE.BufferAttribute;
	const box = new THREE.Box3(), sphere = new THREE.Sphere();
	const n = position.count, per = SLICE / 2;
	for ( let k = 0; k < n; k += per ) {

		for ( let i = k, end = Math.min( k + per, n ); i < end; i ++ ) box.expandByPoint( _v.fromBufferAttribute( position, i ) );
		yield;

	}
	box.getCenter( sphere.center );
	let maxRadiusSq = 0;
	for ( let k = 0; k < n; k += per ) {

		for ( let i = k, end = Math.min( k + per, n ); i < end; i ++ ) maxRadiusSq = Math.max( maxRadiusSq, sphere.center.distanceToSquared( _v.fromBufferAttribute( position, i ) ) );
		yield;

	}
	sphere.radius = Math.sqrt( maxRadiusSq );
	geometry.boundingBox = box;
	geometry.boundingSphere = sphere;

}
