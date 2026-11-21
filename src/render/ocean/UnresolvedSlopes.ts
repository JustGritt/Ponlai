import * as THREE from 'three/webgpu';
import { clamp, floor, int, min, mix } from 'three/tsl';
import { slopeDensity, squareShare, type WaveSpectrumSettings } from './OceanSpectrum.ts';

type F = THREE.Node<'float'>;
type V4 = THREE.Node<'vec4'>;

// The slope variance of the sea that the FFT cascades' filtered reads leave out, by the footprint.
//
// A small table built from the spectrum on the CPU: the slopes a pixel does not resolve.
//
// LEAN mapping (Olano & Baker, "LEAN Mapping", I3D 2010) shades the bumps a filtered read cannot show
// as a distribution of slopes: the slopes combine linearly, so any linear filter (MIP levels,
// trilinear and anisotropic reads) gives their mean, and their covariance, the filtered second moments
// less the mean's square, is an off-centre Gaussian (Beckmann) lobe about it. LEAN stores the second
// moments in a texture beside the slopes. Here the covariance is taken from the spectrum instead: a
// filtered read of a field of slopes keeps the variance of the filtered field, the field's variance
// less what the filter takes out, so what the read loses follows from the spectrum and the filter's
// response alone, with no texture of moments. The resolved normal is the mean, the lost variance the
// covariance. Every term of the sea's light that depends on the slopes reads that one distribution
// (Water.ts): the sky's reflection, the mean Fresnel, the sun's and the moon's glints, the rays the
// screen's reflection traces (SeaReflection.ts) and the refracted floor's blur (the glints and the
// blur damped as the waves are, the reflection not: Water.ts `surface`).
//
// The variance a filtered field keeps is the spectrum times the filter's response, summed over the
// waves. The response is our model of the filter the water's reads take:
// - box mips: a level's texel is the mean of 2^m x 2^m texels of the field (three's mip chain of a
//   power-of-two target), a wave of theta radians per texel scaled by sin( 2^m theta / 2 ) /
//   ( 2^m sin( theta / 2 ) ) per axis;
// - bilinear reads between a level's texel centres, a wave's variance over the positions scaled by
//   ( 2 + cos Theta ) / 3 per axis (Theta its phase step between the level's texels);
// - trilinear, two levels mixed by the level's fraction f: ( 1 - f )^2 and f^2 of each level's power
//   and 2 f ( 1 - f ) of their cross power, the mean over the positions of one level's read times the
//   other's (exact: both reads are linear between their texel centres);
// - the anisotropic filter (at most MAX_ANISOTROPY reads): n = 2^delta trilinear reads, evenly spread
//   along the footprint's long axis over its length, delta = log2 of the ratio of the axes (0 to 2,
//   fractional between the table's columns), at the level of max( short axis, long axis /
//   MAX_ANISOTROPY ); its response along the long axis, the n reads' Dirichlet kernel, taken as if
//   the bilinear field were the same at every position (it varies within a texel). The anisotropic extension
//   (EXT_texture_filter_anisotropic) leaves the scheme to the implementation and describes one
//   possible approach, N = min( ceil( Pmax / Pmin ), maxAniso ) reads at the level of Pmax / N along
//   the long axis; the hardware's own rule is not known here, and this one is ours.
// The waves are taken as even over the directions at each wavenumber (the spectrum's spread is broad
// at the short waves that a pixel loses; the wind's own anisotropy of the slopes is left out, as the
// sea's shading takes one isotropic slope distribution of the wind), over the ring of each cascade's
// band, and the finest cascade's square of modes with its corners. The response is averaged over the
// directions of the waves and of the footprint against the texture's axes, and split into the slopes
// along the footprint's long axis and across it (by that symmetry the two are the covariance's axes).
//
// The table: rows by the footprint, log2 of that level's footprint in metres (SLOPE_ROWS, a quarter
// level apart: half a level apart, reading between rows was off by up to 2.3 % of a cascade's variance),
// SLOPE_COLUMNS columns by delta, and two entries per column, the lost variance of the slope along the
// long axis and across it, a vec4 of the four cascades each (the entries' order: `fillSlopeEntries`).
// The finest cascade's column at delta 0 is read again for the water's near ripples (OceanFFT.ts
// NEAR_RIPPLES), its field read at finer scales with an explicit level: a ripple of scale s at
// footprint F loses what the cascade loses at s F. What the table holds is in the cascades' own units
// (their amplitude, the finest one's `fineScale` in it); the water scales it by the gusts and slicks it
// applies to the slopes no pixel resolves.
//
// `verify-slope-variance` (tools/render) builds the cascades' fields on the CPU from the bands the
// spectrum fills (OceanFFT.ts `cascadeBands`), filters them by the model above and checks that what
// the filtered field keeps plus what the table, read as the water reads it, says is lost is the
// field's variance, at the levels and ratios its header lists.

/** The table's rows: log2 of the footprint (m), from `first` by `step`. */
export const SLOPE_ROWS = { first: - 12, step: 0.25, count: 81 } as const;
/** The sampler's most anisotropic reads: the FFT targets' anisotropy (OceanFFT.ts, Water.ts
 * `WaterQuality`, typed to this value) and the table's last column. */
export const MAX_ANISOTROPY = 4;
/** The table's columns: delta = log2 of the footprint's ratio of axes, 0 to log2( MAX_ANISOTROPY ). */
export const SLOPE_COLUMNS = Math.log2( MAX_ANISOTROPY ) + 1;
/** The table's entries, each a vec4 of the four cascades (`fillSlopeEntries`). */
export const SLOPE_ENTRIES = SLOPE_ROWS.count * SLOPE_COLUMNS * 2;

// the mip levels of a cascade of 256 texels: 0 to 8
const LAST_LEVEL = 8;
// the response's grid: log2 of the wavenumber in radians per texel, the level (quarters; below 0 the
// footprint is under a texel, read at level 0, its anisotropic reads closer together), the waves'
// and the footprint's directions
const KT = { first: - 6, step: 1 / 8, count: 67 } as const;
const FIRST_LEVEL = - 4, LEVEL_STEP = 0.25, LEVEL_COUNT = ( LAST_LEVEL - FIRST_LEVEL ) / LEVEL_STEP + 1;
const DIRECTIONS = 32, FOOTPRINTS = 4;

// a box mip's gain on a wave of theta radians per texel at level m
function boxGain( m: number, theta: number ): number {

	const s = 2 ** m, d = Math.sin( theta / 2 );
	return Math.abs( d ) < 1e-9 ? 1 : Math.sin( s * theta / 2 ) / ( s * d );

}

// a level's bilinear power, over the positions, of a unit wave
function levelPower( m: number, theta: number ): number {

	return boxGain( m, theta ) ** 2 * ( 2 + Math.cos( 2 ** m * theta ) ) / 3;

}

// the mean over the positions of level m's bilinear read of a unit wave times the conjugate of level
// m + 1's, ( re, im ): in units of level m's texel, y over the common period [ 0, 2 ), each read linear
// between its texel centres (level m's at j + 1/2, level m + 1's at 2 j + 1), so the product is a
// quadratic on each of four pieces, which Simpson's rule takes exactly
function levelCross( m: number, theta: number ): [ number, number ] {

	const T = 2 ** m * theta;
	const at = ( y: number ): [ number, number ] => {

		const j = Math.floor( y - 0.5 ), u = y - 0.5 - j;
		const j2 = Math.floor( ( y - 1 ) / 2 ), u2 = ( y - 1 - 2 * j2 ) / 2;
		const a0 = T * ( j + 0.5 ), a1 = T * ( j + 1.5 ), b0 = T * ( 2 * j2 + 1 ), b1 = T * ( 2 * j2 + 3 );
		const r1 = [ ( 1 - u ) * Math.cos( a0 ) + u * Math.cos( a1 ), ( 1 - u ) * Math.sin( a0 ) + u * Math.sin( a1 ) ];
		const r2 = [ ( 1 - u2 ) * Math.cos( b0 ) + u2 * Math.cos( b1 ), ( 1 - u2 ) * Math.sin( b0 ) + u2 * Math.sin( b1 ) ];
		// r1 times the conjugate of r2
		return [ r1[ 0 ] * r2[ 0 ] + r1[ 1 ] * r2[ 1 ], r1[ 1 ] * r2[ 0 ] - r1[ 0 ] * r2[ 1 ] ];

	};
	let re = 0, im = 0;
	for ( const a of [ 0, 0.5, 1, 1.5 ] ) {

		// (the pieces' ends nudged inside, where the reads' texel indices change)
		const e = 1e-9, p = at( a + e ), q = at( a + 0.25 ), r = at( a + 0.5 - e );
		re += ( p[ 0 ] + 4 * q[ 0 ] + r[ 0 ] ) / 6 * 0.5;
		im += ( p[ 1 ] + 4 * q[ 1 ] + r[ 1 ] ) / 6 * 0.5;

	}
	const g = boxGain( m, theta ) * boxGain( m + 1, theta ) / 2;
	return [ re * g, im * g ];

}

/** The trilinear read's power, over the positions, of a unit wave of ( tx, tz ) radians per texel
 * at level `level` (0 to 8). */
export function trilinearPower( level: number, tx: number, tz: number ): number {

	const m = Math.min( Math.floor( level ), LAST_LEVEL - 1 ), f = level - m;
	const cx = levelCross( m, tx ), cz = levelCross( m, tz );
	return ( 1 - f ) ** 2 * levelPower( m, tx ) * levelPower( m, tz ) + f * f * levelPower( m + 1, tx ) * levelPower( m + 1, tz )
		+ 2 * f * ( 1 - f ) * ( cx[ 0 ] * cz[ 0 ] - cx[ 1 ] * cz[ 1 ] );

}

/** The anisotropic filter's reads along the footprint's long axis: n of them over `length` texels,
 * their power on a wave of `t` radians per texel along that axis. */
export function linePower( n: number, length: number, t: number ): number {

	if ( n <= 1 ) return 1;
	const a = t * length / n, d = Math.sin( a / 2 );
	return Math.abs( d ) < 1e-9 ? 1 : ( Math.sin( n * a / 2 ) / ( n * d ) ) ** 2;

}

// The filter's response, kept of the slope variance along the long axis and across it (1 unfiltered),
// by the wavenumber, the level and delta, for the waves even over the directions present (a corner's
// ring only the directions inside the square of modes): [ kt ][ level ][ delta ][ axis ].
let response: Float32Array | null = null;

function filterResponse(): Float32Array {

	if ( response ) return response;
	const out = new Float32Array( KT.count * LEVEL_COUNT * SLOPE_COLUMNS * 2 );
	const power = new Float64Array( DIRECTIONS );
	// per direction and level m: the level's power, and its cross power with level m + 1 (re, im),
	// the products of the two axes' (trilinearPower, taken apart so that each level's terms are made once)
	const terms = new Float64Array( DIRECTIONS * ( LAST_LEVEL + 1 ) * 3 );
	for ( let i = 0; i < KT.count; i ++ ) {

		const kt = 2 ** ( KT.first + KT.step * i );
		for ( let p = 0; p < DIRECTIONS; p ++ ) {

			const phi = 2 * Math.PI * ( p + 0.5 ) / DIRECTIONS, tx = kt * Math.cos( phi ), tz = kt * Math.sin( phi );
			for ( let m = 0; m <= LAST_LEVEL; m ++ ) {

				const at = ( p * ( LAST_LEVEL + 1 ) + m ) * 3;
				terms[ at ] = levelPower( m, tx ) * levelPower( m, tz );
				if ( m === LAST_LEVEL ) continue;
				const cx = levelCross( m, tx ), cz = levelCross( m, tz );
				terms[ at + 1 ] = cx[ 0 ] * cz[ 0 ] - cx[ 1 ] * cz[ 1 ];

			}

		}
		for ( let l = 0; l < LEVEL_COUNT; l ++ ) {

			const level = FIRST_LEVEL + l * LEVEL_STEP;
			const m = Math.min( Math.floor( Math.max( level, 0 ) ), LAST_LEVEL - 1 ), f = Math.max( level, 0 ) - m;
			for ( let p = 0; p < DIRECTIONS; p ++ ) {

				const phi = 2 * Math.PI * ( p + 0.5 ) / DIRECTIONS;
				const at = ( p * ( LAST_LEVEL + 1 ) + m ) * 3;
				power[ p ] = Math.max( Math.abs( Math.cos( phi ) ), Math.abs( Math.sin( phi ) ) ) * kt > Math.PI ? - 1
					: ( 1 - f ) ** 2 * terms[ at ] + f * f * terms[ at + 3 ] + 2 * f * ( 1 - f ) * terms[ at + 1 ];

			}
			for ( let delta = 0; delta < SLOPE_COLUMNS; delta ++ ) {

				const n = 2 ** delta, length = 2 ** ( level + delta );
				let along = 0, across = 0, count = 0;
				for ( let q = 0; q < FOOTPRINTS; q ++ ) {

					const psi = Math.PI / 2 * ( q + 0.5 ) / FOOTPRINTS;
					for ( let p = 0; p < DIRECTIONS; p ++ ) {

						if ( power[ p ] < 0 ) continue;
						const c = Math.cos( 2 * Math.PI * ( p + 0.5 ) / DIRECTIONS - psi );
						const kept = power[ p ] * linePower( n, length, kt * c );
						along += 2 * c * c * kept;
						across += 2 * ( 1 - c * c ) * kept;
						count ++;

					}

				}
				const at = ( ( i * LEVEL_COUNT + l ) * SLOPE_COLUMNS + delta ) * 2;
				out[ at ] = count ? along / count : 0;
				out[ at + 1 ] = count ? across / count : 0;

			}

		}

	}
	response = out;
	return out;

}

/** A cascade's band: its tile (m), the ring of wavenumbers it holds (rad/m) and, for the finest, the
 * square of modes out to `k1` with its corners; `gain` scales its slopes (the finest's `fineScale`
 * squared, the spectrum's amplitude squared). */
export interface CascadeBand { size: number; k0: number; k1: number; square: boolean; gain: number }

/** The table and the totals the water reads with it (in the cascades' units). */
export interface SlopeTable {
	/** [ row ][ delta ][ along, across ] x 4 cascades: the slope variance lost, per axis. */
	lost: Float32Array;
	/** Each cascade's slope variance over both axes. */
	total: number[];
}

/** A cascade's level (of `size` metres over `n` texels) at a footprint of log2 `row` metres, below 0
 * where the footprint is under a texel (read at level 0), held at FIRST_LEVEL and the last level. */
export function cascadeLevel( size: number, n: number, row: number ): number {

	return Math.min( Math.max( row - Math.log2( size / n ), FIRST_LEVEL ), LAST_LEVEL );

}

/** Builds the table from the wave systems (the spectrum OceanFFT sums) and the cascades' bands. */
export function slopeTable( systems: WaveSpectrumSettings[], depth: number, bands: CascadeBand[], n: number ): SlopeTable {

	const G = filterResponse();
	const densities = systems.map( ( s ) => slopeDensity( s, depth ) );
	const density = ( k: number ) => densities.reduce( ( sum, d ) => sum + d( k ), 0 );
	const C = bands.length;
	// each cascade's slope variance per bin of the response's wavenumbers
	const bins = bands.map( ( b ) => {

		const texel = b.size / n, out = new Float64Array( KT.count );
		const top = b.square ? b.k1 * Math.SQRT2 : b.k1;
		for ( let i = 0; i < KT.count; i ++ ) {

			const lo = Math.max( 2 ** ( KT.first + KT.step * ( i - 0.5 ) ) / texel, b.k0 );
			const hi = Math.min( 2 ** ( KT.first + KT.step * ( i + 0.5 ) ) / texel, top );
			if ( hi <= lo ) continue;
			// Simpson's rule over log k, 8 intervals
			const span = Math.log( hi / lo );
			let sum = 0;
			for ( let j = 0; j <= 8; j ++ ) {

				const k = lo * Math.exp( span * j / 8 );
				const share = b.square && k > b.k1 ? squareShare( k, b.k1 ) : 1;
				sum += ( j === 0 || j === 8 ? 1 : j % 2 ? 4 : 2 ) * density( k ) * k * share;

			}
			out[ i ] = sum * span / 24 * b.gain;

		}
		return out;

	} );
	const total = bins.map( ( b ) => b.reduce( ( s, v ) => s + v, 0 ) );
	const lost = new Float32Array( SLOPE_ROWS.count * SLOPE_COLUMNS * 2 * 4 );
	for ( let r = 0; r < SLOPE_ROWS.count; r ++ ) {

		const row = SLOPE_ROWS.first + SLOPE_ROWS.step * r;
		for ( let c = 0; c < C; c ++ ) {

			const q = ( cascadeLevel( bands[ c ].size, n, row ) - FIRST_LEVEL ) / LEVEL_STEP;
			const l0 = Math.min( Math.floor( q ), LEVEL_COUNT - 2 ), t = q - l0;
			for ( let delta = 0; delta < SLOPE_COLUMNS; delta ++ ) {

				for ( let axis = 0; axis < 2; axis ++ ) {

					let kept = 0;
					for ( let i = 0; i < KT.count; i ++ ) {

						const g = ( l: number ) => G[ ( ( i * LEVEL_COUNT + l ) * SLOPE_COLUMNS + delta ) * 2 + axis ];
						kept += bins[ c ][ i ] * ( g( l0 ) * ( 1 - t ) + g( l0 + 1 ) * t );

					}
					lost[ ( ( r * SLOPE_COLUMNS + delta ) * 2 + axis ) * 4 + c ] = Math.max( total[ c ] - kept, 0 ) / 2;

				}

			}

		}

	}
	return { lost, total };

}

/** Something that holds four numbers, as THREE.Vector4 does. */
export interface SlopeEntry { x: number; y: number; z: number; w: number; set( x: number, y: number, z: number, w: number ): unknown }

/** Writes the table into `entries` (SLOPE_ENTRIES of them), as the water's uniform array holds it:
 * entry ( row SLOPE_COLUMNS + delta ) 2 + axis, axis 0 along the footprint's long axis and 1 across,
 * the four cascades in x, y, z, w. */
export function fillSlopeEntries( table: SlopeTable, entries: SlopeEntry[] ): void {

	for ( let i = 0; i < SLOPE_ENTRIES; i ++ ) entries[ i ].set( table.lost[ i * 4 ], table.lost[ i * 4 + 1 ], table.lost[ i * 4 + 2 ], table.lost[ i * 4 + 3 ] );

}

/** The entries read as the water's shader reads them (`slopeReader`): the lost variance of each
 * cascade along the footprint's long axis and across it, at log2 footprint `row` (m) and ratio
 * `delta` (0 to SLOPE_COLUMNS - 1), bilinear in both. */
export function readSlopeEntries( entries: SlopeEntry[], row: number, delta: number ): { along: number[]; across: number[] } {

	const q = Math.min( Math.max( ( row - SLOPE_ROWS.first ) / SLOPE_ROWS.step, 0 ), SLOPE_ROWS.count - 1 );
	const r0 = Math.min( Math.floor( q ), SLOPE_ROWS.count - 2 ), t = q - r0;
	const d0 = Math.min( Math.floor( delta ), SLOPE_COLUMNS - 2 ), s = delta - d0;
	const read = ( axis: number ) => {

		const at = ( r: number, d: number ) => entries[ ( r * SLOPE_COLUMNS + d ) * 2 + axis ];
		const v = ( e: SlopeEntry ) => [ e.x, e.y, e.z, e.w ];
		const a = v( at( r0, d0 ) ), b = v( at( r0, d0 + 1 ) ), c = v( at( r0 + 1, d0 ) ), d = v( at( r0 + 1, d0 + 1 ) );
		return a.map( ( _, k ) => ( a[ k ] * ( 1 - s ) + b[ k ] * s ) * ( 1 - t ) + ( c[ k ] * ( 1 - s ) + d[ k ] * s ) * t );

	};
	return { along: read( 0 ), across: read( 1 ) };

}

/** The entries read in a shader, as `readSlopeEntries` reads them on the CPU: `lostAt` the lost
 * variance of the four cascades along the footprint's long axis and across it, at log2 footprint
 * `row` (m) and ratio `delta`; `fineLostAt` the finest cascade's over both axes at a ratio of 1 (the
 * water's near ripples). `entries` is OceanFFT.ts `unresolved`. */
export function slopeReader( entries: { element( i: THREE.Node<'int'> ): unknown } ) {

	const slopeRow = ( row: F ) => {

		const q = clamp( row.sub( SLOPE_ROWS.first ).div( SLOPE_ROWS.step ), 0, SLOPE_ROWS.count - 1 );
		const r0 = min( floor( q ), SLOPE_ROWS.count - 2 );
		return { r0, r1: r0.add( 1 ), t: q.sub( r0 ) };

	};
	const entry = ( r: F, d: F | number, axis: number ) => entries.element( int( r.mul( SLOPE_COLUMNS ).add( d ).mul( 2 ).add( axis ) ) ) as V4;
	const lostAt = ( row: F, delta: F ) => {

		const { r0, r1, t } = slopeRow( row );
		const d0 = min( floor( delta ), SLOPE_COLUMNS - 2 ), d1 = d0.add( 1 ), s = delta.sub( d0 );
		const read = ( axis: number ) => mix( mix( entry( r0, d0, axis ), entry( r0, d1, axis ), s ), mix( entry( r1, d0, axis ), entry( r1, d1, axis ), s ), t );
		return { along: read( 0 ), across: read( 1 ) };

	};
	const fineLostAt = ( row: F ): F => {

		const { r0, r1, t } = slopeRow( row );
		const both = ( r: F ) => entry( r, 0, 0 ).w.add( entry( r, 0, 1 ).w );
		return mix( both( r0 ), both( r1 ), t );

	};
	return { lostAt, fineLostAt };

}
