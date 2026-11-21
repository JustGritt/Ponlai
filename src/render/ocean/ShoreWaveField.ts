/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import type { ShoreField } from '../RenderTypes.ts';

// The rocky shore's wave fields as plain data, a function of the shore's distance field.
//
// The rocky shore's wave fields as plain data (no three, no DOM): a function of the shore's
// distance field alone, so the page builds them at load and a grown island's in a worker
// (shoreFieldWorker.ts, World growth in Growth.ts); ShoreWaves.ts makes their textures.

/** A field's RGBA float texels, nx by nz. */
export interface FieldPixels { pixels: Float32Array; nx: number; nz: number }

const MAX_RES = 256;

// The shore's distance is clamped to LAND_REACH (the chunks' shore tiles), but the shelter's
// recovery runs from 18 to 36 m. Beyond the clamp the distance to the coast is the clamp plus the
// distance to the region under it (that region is the coast grown by the clamp), so the clamped
// cells take the distance to an unclamped cell plus that cell's own distance, an upper bound
// that is exact for a cell on the line to the coast: the least over the cells round the nearest. One exact Euclidean feature transform (Felzenszwalb and Huttenlocher, "Distance
// Transforms of Sampled Functions", 2012), separable: columns, then rows.
const SHORE_CLAMP = 24, REFINE = 2;
function unclampShore( distances: Float32Array, nx: number, nz: number, cellX: number, cellZ: number ): void {

	const count = nx * nz;
	// per cell, the squared z distance to the nearest site in its column, and that site's row
	const column = new Float64Array( count ).fill( Infinity ), row = new Int32Array( count ).fill( - 1 );
	let sites = 0;
	for ( let i = 0; i < nx; i ++ ) {

		let last = - 1;
		for ( let j = 0; j < nz; j ++ ) {

			if ( distances[ j * nx + i ] < SHORE_CLAMP ) { last = j; sites ++; }
			if ( last >= 0 ) { column[ j * nx + i ] = ( ( j - last ) * cellZ ) ** 2; row[ j * nx + i ] = last; }

		}
		last = - 1;
		for ( let j = nz - 1; j >= 0; j -- ) {

			if ( distances[ j * nx + i ] < SHORE_CLAMP ) last = j;
			const d = last >= 0 ? ( ( last - j ) * cellZ ) ** 2 : Infinity;
			if ( d < column[ j * nx + i ] ) { column[ j * nx + i ] = d; row[ j * nx + i ] = last; }

		}

	}
	if ( sites === 0 || sites === count ) return;
	// rows: the lower envelope of the parabolas ( x - x_q )^2 + column( q ) over the finite ones
	const v = new Int32Array( nx ), z = new Float64Array( nx + 1 );
	const out = distances.slice();
	for ( let j = 0; j < nz; j ++ ) {

		const o = j * nx;
		let n = - 1;
		for ( let q = 0; q < nx; q ++ ) {

			const fq = column[ o + q ];
			if ( fq === Infinity ) continue;
			const xq = q * cellX;
			let s = - Infinity;
			while ( n >= 0 ) {

				const xv = v[ n ] * cellX;
				s = ( fq + xq * xq - column[ o + v[ n ] ] - xv * xv ) / ( 2 * ( xq - xv ) );
				if ( s > z[ n ] ) break;
				n --;

			}
			n ++;
			v[ n ] = q; z[ n ] = n === 0 ? - Infinity : s; z[ n + 1 ] = Infinity;

		}
		if ( n < 0 ) continue;
		let e = 0;
		for ( let i = 0; i < nx; i ++ ) {

			const k = j * nx + i;
			while ( z[ e + 1 ] < i * cellX ) e ++;
			if ( distances[ k ] < SHORE_CLAMP ) continue;
			// the nearest site is close to the line to the coast, not on it: its neighbours may
			// lie nearer that line, so the least bound round it is taken
			const si = v[ e ], sj = row[ j * nx + si ];
			const a0 = Math.max( 0, si - REFINE ), a1 = Math.min( nx - 1, si + REFINE );
			const b0 = Math.max( 0, sj - REFINE ), b1 = Math.min( nz - 1, sj + REFINE );
			let best = Infinity;
			for ( let b = b0; b <= b1; b ++ ) {

				const ez = ( j - b ) * cellZ;
				for ( let a = a0; a <= a1; a ++ ) {

					const d = distances[ b * nx + a ], ex = ( i - a ) * cellX;
					if ( d < SHORE_CLAMP ) {

						const bound = Math.sqrt( ex * ex + ez * ez ) + d;
						if ( bound < best ) best = bound;

					}

				}

			}
			out[ k ] = best;

		}

	}
	distances.set( out );

}

function sampleShore( field: ShoreField, x: number, z: number ): number {

	const fx = Math.max( 0, Math.min( field.nx - 1, ( x - field.x0 ) / field.cell - 0.5 ) );
	const fz = Math.max( 0, Math.min( field.nz - 1, ( z - field.z0 ) / field.cell - 0.5 ) );
	const i = Math.floor( fx ), j = Math.floor( fz ), u = fx - i, v = fz - j;
	const ii = Math.min( i + 1, field.nx - 1 ), jj = Math.min( j + 1, field.nz - 1 );
	const a = field.data[ j * field.nx + i ], b = field.data[ j * field.nx + ii ];
	const c = field.data[ jj * field.nx + i ], d = field.data[ jj * field.nx + ii ];
	return ( a + ( b - a ) * u ) * ( 1 - v ) + ( c + ( d - c ) * u ) * v;

}


// A geometric boundary only: confidence and distance gradient. Wave timing comes from FFT,
// so crossing wave systems never inherit the old single-period arrival-time field.
export function shoreBoundaryPixels( shore: ShoreField ): FieldPixels {

	const scale = Math.min( 1, MAX_RES / Math.max( shore.nx, shore.nz ) );
	const nx = Math.max( 2, Math.round( shore.nx * scale ) ), nz = Math.max( 2, Math.round( shore.nz * scale ) );
	const width = shore.nx * shore.cell, height = shore.nz * shore.cell;
	const pixels = new Float32Array( nx * nz * 4 );
	for ( let j = 0; j < nz; j ++ ) for ( let i = 0; i < nx; i ++ ) {

		const x = shore.x0 + ( i + 0.5 ) * width / nx, z = shore.z0 + ( j + 0.5 ) * height / nz;
		const sd = sampleShore( shore, x, z );
		const gx = ( sampleShore( shore, x + 0.4, z ) - sampleShore( shore, x - 0.4, z ) ) / 0.8;
		const gz = ( sampleShore( shore, x, z + 0.4 ) - sampleShore( shore, x, z - 0.4 ) ) / 0.8;
		const k = ( j * nx + i ) * 4;
		pixels[ k ] = sd;
		pixels[ k + 1 ] = Math.min( 1, Math.max( 0, ( Math.hypot( gx, gz ) - 0.45 ) / 0.45 ) );
		pixels[ k + 2 ] = gx; pixels[ k + 3 ] = gz;

	}
	return { pixels, nx, nz };

}

/**
 * Directional energy shelter for the dominant swell. Our closed-island
 * extension: Tidewater's local direction alignment alone does not test intervening land.
 * R: swell amplitude transmission, G: central-ray visibility, BA: spatial derivatives of R.
 * This is an art-directed soft wave shadow, not a diffraction or fluid solver.
 */
export function shoreShelterPixels( shoreGrid: ShoreField, direction: { x: number; y: number }, spreadDegrees = 18 ): FieldPixels & { info: Record<string, unknown> } {

	// eslint-disable-next-line ponlai/scene-clock -- the field's build time, a readout
	const started = performance.now();
	const scale = Math.min( 1, MAX_RES / Math.max( shoreGrid.nx, shoreGrid.nz ) );
	const nx = Math.max( 2, Math.round( shoreGrid.nx * scale ) ), nz = Math.max( 2, Math.round( shoreGrid.nz * scale ) );
	const info = { x0: shoreGrid.x0, z0: shoreGrid.z0, width: shoreGrid.nx * shoreGrid.cell, height: shoreGrid.nz * shoreGrid.cell,
		cellX: shoreGrid.nx * shoreGrid.cell / nx, cellZ: shoreGrid.nz * shoreGrid.cell / nz };
	const dx = direction.x, dz = direction.y;
	const shore = shoreGrid, count = nx * nz;
	const visible = new Float32Array( count ), direct = new Float32Array( count );
	const water = new Uint8Array( count ), distances = new Float32Array( count );
	const angles = [ - spreadDegrees, - spreadDegrees / 2, 0, spreadDegrees / 2, spreadDegrees ], weights = [ 1, 2, 6, 2, 1 ];
	// Each cell turns the fan by its own fraction of a step (a hash of its world position): five
	// fixed rays cut the penumbra into five steps, which read as streaks across the lee's calm.
	const jitter = ( x: number, z: number ): number => {

		let h = Math.imul( Math.round( x * 64 ) | 0, 0x27d4eb2d ) ^ Math.imul( Math.round( z * 64 ) | 0, 0x165667b1 );
		h = Math.imul( h ^ ( h >>> 15 ), 0x85ebca6b );
		h = Math.imul( h ^ ( h >>> 13 ), 0xc2b2ae35 );
		return ( ( h ^ ( h >>> 16 ) ) >>> 0 ) / 4294967296 - 0.5;

	};
	const rayAt = ( degrees: number, weight: number ) => {

		const a = degrees * Math.PI / 180;
		return { x: - dx * Math.cos( a ) + dz * Math.sin( a ), z: - dx * Math.sin( a ) - dz * Math.cos( a ), weight };

	};
	const x1 = info.x0 + info.width, z1 = info.z0 + info.height;
	const minimumStep = Math.min( info.cellX, info.cellZ, shore.cell ) * 0.5;
	const trace = ( x: number, z: number, rx: number, rz: number ): number => {

		let distance = minimumStep;
		// The shore SDF is sampled more densely than this field. Conservative, capped steps
		// retain narrow rock shoulders without marching at fixed resolution across open sea.
		while ( distance < Math.hypot( info.width, info.height ) ) {

			const px = x + rx * distance, pz = z + rz * distance;
			if ( px < info.x0 || pz < info.z0 || px > x1 || pz > z1 ) return 1;
			const clearance = sampleShore( shore, px, pz );
			// A millimetre tolerance covers grazing contacts below the shore-grid precision.
			if ( clearance <= 0.001 ) return 0;
			distance += Math.max( minimumStep, Math.min( 2, clearance * 0.65 ) );

		}
		return 1;

	};
	for ( let j = 0; j < nz; j ++ ) for ( let i = 0; i < nx; i ++ ) {

		const k = j * nx + i, x = info.x0 + ( i + 0.5 ) * info.cellX, z = info.z0 + ( j + 0.5 ) * info.cellZ;
		const sd = sampleShore( shore, x, z );
		distances[ k ] = sd;
		if ( sd <= 0 ) continue;
		water[ k ] = 1;
		let sum = 0;
		const turn = jitter( x, z ) * spreadDegrees / 2;
		for ( let ray = 0; ray < angles.length; ray ++ ) {

			const r = rayAt( angles[ ray ] + turn, weights[ ray ] ), visibility = trace( x, z, r.x, r.z );
			sum += visibility * r.weight;
			if ( ray === 2 ) direct[ k ] = visibility;

		}
		visible[ k ] = sum / 12;

	}
	unclampShore( distances, nx, nz, info.cellX, info.cellZ );
	// Blur water neighbours only. This removes fan/grid steps (and the jitter's noise) at penumbra
	// boundaries without bleeding a fully exposed wave through the solid island into a small inlet.
	for ( let pass = 0; pass < 8; pass ++ ) {

		const previous = visible.slice();
		for ( let j = 0; j < nz; j ++ ) for ( let i = 0; i < nx; i ++ ) {

			const k = j * nx + i;
			if ( ! water[ k ] ) continue;
			let sum = previous[ k ] * 4, weight = 4;
			for ( const next of [ i > 0 ? k - 1 : - 1, i < nx - 1 ? k + 1 : - 1, j > 0 ? k - nx : - 1, j < nz - 1 ? k + nx : - 1 ] ) {

				if ( next >= 0 && water[ next ] ) { sum += previous[ next ]; weight ++; }

			}
			visible[ k ] = sum / weight;

		}

	}
	const pixels = new Float32Array( count * 4 );
	const amplitude = ( k: number ): number => {

		if ( ! water[ k ] ) return 0;
		// Recover distant wind sea before the finite field edge; clamp-to-edge samples
		// must not carry an infinite, straight shadow strip across the offshore FFT.
		const t = Math.max( 0, Math.min( 1, ( distances[ k ] - 18 ) / 18 ) );
		const recovery = t * t * ( 3 - 2 * t );
		const i = k % nx, j = Math.floor( k / nx );
		const border = Math.min( ( i + 0.5 ) * info.cellX, ( nx - i - 0.5 ) * info.cellX,
			( j + 0.5 ) * info.cellZ, ( nz - j - 0.5 ) * info.cellZ );
		const edge = Math.max( 0, Math.min( 1, ( border - 1 ) / 4 ) );
		const edgeWeight = edge * edge * ( 3 - 2 * edge );
		// The cached correction has an exact zero rim at every heading. Only the local
		// energy shadow is bounded; the incident FFT waves continue over the whole sea.
		return 1 - ( 1 - ( 0.06 + 0.94 * visible[ k ] ) ) * ( 1 - recovery ) * edgeWeight;

	};
	for ( let j = 0; j < nz; j ++ ) for ( let i = 0; i < nx; i ++ ) {

		const k = j * nx + i, out = k * 4;
		pixels[ out ] = amplitude( k ); pixels[ out + 1 ] = direct[ k ];
		if ( ! water[ k ] ) continue;
		// One-sided water differences avoid an artificial damping-gradient seam at land.
		const left = i > 0 && water[ k - 1 ] ? k - 1 : k, right = i < nx - 1 && water[ k + 1 ] ? k + 1 : k;
		const down = j > 0 && water[ k - nx ] ? k - nx : k, up = j < nz - 1 && water[ k + nx ] ? k + nx : k;
		pixels[ out + 2 ] = ( amplitude( right ) - amplitude( left ) ) / Math.max( ( right - left ) * info.cellX, info.cellX );
		pixels[ out + 3 ] = ( amplitude( up ) - amplitude( down ) ) / Math.max( ( up - down ) / nx * info.cellZ, info.cellZ );

	}
	// Extend the water-side amplitude and gradient into land for linear filtering. A zero
	// land amplitude would add a false shoreline ramp that its one-sided gradient omits.
	// This extension never participates in tracing or smoothing.
	const queue = new Int32Array( count ), reached = water.slice();
	let head = 0, tail = 0;
	for ( let k = 0; k < count; k ++ ) if ( water[ k ] ) queue[ tail ++ ] = k;
	while ( head < tail ) {

		const k = queue[ head ++ ], i = k % nx, j = Math.floor( k / nx );
		for ( const next of [ i > 0 ? k - 1 : - 1, i < nx - 1 ? k + 1 : - 1, j > 0 ? k - nx : - 1, j < nz - 1 ? k + nx : - 1 ] ) {

			if ( next < 0 || reached[ next ] ) continue;
			reached[ next ] = 1;
			pixels[ next * 4 ] = pixels[ k * 4 ];
			pixels[ next * 4 + 2 ] = pixels[ k * 4 + 2 ]; pixels[ next * 4 + 3 ] = pixels[ k * 4 + 3 ];
			queue[ tail ++ ] = next;

		}

	}
	// eslint-disable-next-line ponlai/scene-clock -- the field's build time, a readout
	return { pixels, nx, nz, info: { direction: [ dx, dz ], fanDegrees: angles, residualAmplitude: 0.06, buildMs: performance.now() - started } };

}
