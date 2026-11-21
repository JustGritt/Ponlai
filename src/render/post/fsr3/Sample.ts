/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_sample.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { Fn, float, int, vec2, ivec2, abs, min, max, sin, clamp, floor, select } from './Tsl.ts';
import { ffxFract, type N } from './Core.ts';
import { FSR3UPSCALER_EPSILON } from './Common.ts';
import type { Fsr3Bindings } from './Bindings.ts';

// FSR 3's resampling kernels: bilinear and Lanczos 2 over a 4 x 4 fetch with deringing.
//
// The upscaler's resampling kernels: bilinear, Lanczos 2 (reference, LUT and FSR 1's
// approximation) over a 4 x 4 fetch with deringing, and the macros that declare a fetch and a
// sample function for a texture.
//
// Differences from upstream:
// - The FFX_HALF / min16 variants are not ported.
// - The `Declare*` macros become functions that take the texel load (and, for the sample, the
//   interpolator and fetch) as arguments; the 4 x 4 fetch is an object of 16 variables.
// - `Lanczos2_UseLUT` reads the LUT through the bindings (`SampleLanczos2Weight`).
// - The 2D Lanczos filters compute each axis's four weights once (`lanczos2D`), not at each of
//   the five 1D calls: the same values from 8 weight evaluations instead of 20.
// - `FFX_FSR3UPSCALER_GET_LANCZOS_SAMPLER1D( x )` is `getLanczosSampler1D( x )`.

export interface FetchedBilinearSamples {
	fColor00: N; fColor10: N; fColor01: N; fColor11: N;
}

export interface FetchedBicubicSamples {
	fColor00: N; fColor10: N; fColor20: N; fColor30: N;
	fColor01: N; fColor11: N; fColor21: N; fColor31: N;
	fColor02: N; fColor12: N; fColor22: N; fColor32: N;
	fColor03: N; fColor13: N; fColor23: N; fColor33: N;
}

export const Linear = /*@__PURE__*/ Fn( ( [ A, B, t ]: N[] ) => {

	return A.add( B.sub( A ).mul( t ) );

} ).setLayout( { name: 'Linear', type: 'vec4', inputs: [ { name: 'A', type: 'vec4' }, { name: 'B', type: 'vec4' }, { name: 't', type: 'float' } ] } );

export function Bilinear( BilinearSamples: FetchedBilinearSamples, fPxFrac: N ): N {

	const fColorX0 = Linear( BilinearSamples.fColor00, BilinearSamples.fColor10, fPxFrac.x );
	const fColorX1 = Linear( BilinearSamples.fColor01, BilinearSamples.fColor11, fPxFrac.x );
	const fColorXY = Linear( fColorX0, fColorX1, fPxFrac.y );
	return fColorXY;

}

export const Lanczos2NoClamp = /*@__PURE__*/ Fn( ( [ x ]: N[] ) => {

	const PI = 3.141592653589793; // TODO: share SDK constants
	const px = x.mul( PI ).toVar();
	const hpx = x.mul( 0.5 * PI ).toVar();
	return select( abs( x ).lessThan( FSR3UPSCALER_EPSILON ), float( 1.0 ), sin( px ).div( px ).mul( sin( hpx ).div( hpx ) ) );

} ).setLayout( { name: 'Lanczos2NoClamp', type: 'float', inputs: [ { name: 'x', type: 'float' } ] } );

export const Lanczos2 = /*@__PURE__*/ Fn( ( [ x ]: N[] ) => {

	return Lanczos2NoClamp( min( abs( x ), float( 2.0 ) ) );

} ).setLayout( { name: 'Lanczos2', type: 'float', inputs: [ { name: 'x', type: 'float' } ] } );

// FSR1 lanczos approximation. Input is x*x and must be <= 4.
export const Lanczos2ApproxSqNoClamp = /*@__PURE__*/ Fn( ( [ x2 ]: N[] ) => {

	const a = float( 2.0 / 5.0 ).mul( x2 ).sub( 1 ).toVar();
	const b = float( 1.0 / 4.0 ).mul( x2 ).sub( 1 ).toVar();
	return float( 25.0 / 16.0 ).mul( a ).mul( a ).sub( 25.0 / 16.0 - 1 ).mul( b.mul( b ) );

} ).setLayout( { name: 'Lanczos2ApproxSqNoClamp', type: 'float', inputs: [ { name: 'x2', type: 'float' } ] } );

export const Lanczos2ApproxSq = /*@__PURE__*/ Fn( ( [ x2 ]: N[] ) => {

	return Lanczos2ApproxSqNoClamp( min( x2, float( 4.0 ) ) );

} ).setLayout( { name: 'Lanczos2ApproxSq', type: 'float', inputs: [ { name: 'x2', type: 'float' } ] } );

export const Lanczos2ApproxNoClamp = /*@__PURE__*/ Fn( ( [ x ]: N[] ) => {

	return Lanczos2ApproxSqNoClamp( x.mul( x ) );

} ).setLayout( { name: 'Lanczos2ApproxNoClamp', type: 'float', inputs: [ { name: 'x', type: 'float' } ] } );

export const Lanczos2Approx = /*@__PURE__*/ Fn( ( [ x ]: N[] ) => {

	return Lanczos2ApproxSq( x.mul( x ) );

} ).setLayout( { name: 'Lanczos2Approx', type: 'float', inputs: [ { name: 'x', type: 'float' } ] } );

export function Lanczos2_UseLUT( b: Fsr3Bindings, x: N ): N {

	return b.SampleLanczos2Weight( abs( x ) );

}

/** The four weights of the 1D filter at t (upstream's Lanczos2( fColor0..3, t ) computes them at each
 * call: the 2D filter calls it five times, four of them at the same t). */
function weights4( weight: ( x: N ) => N, t: N ): N[] {

	return [
		weight( float( - 1.0 ).sub( t ) ).toVar(),
		weight( float( - 0.0 ).sub( t ) ).toVar(),
		weight( float( + 1.0 ).sub( t ) ).toVar(),
		weight( float( + 2.0 ).sub( t ) ).toVar(),
	];

}

/** The four-tap 1D filter with its weights: upstream's sums, in its order. */
export const Filter4 = /*@__PURE__*/ Fn( ( [ fColor0, fColor1, fColor2, fColor3, fWeight0, fWeight1, fWeight2, fWeight3 ]: N[] ) => {

	return fWeight0.mul( fColor0 ).add( fWeight1.mul( fColor1 ) ).add( fWeight2.mul( fColor2 ) ).add( fWeight3.mul( fColor3 ) )
		.div( fWeight0.add( fWeight1 ).add( fWeight2 ).add( fWeight3 ) );

} ).setLayout( { name: 'Filter4', type: 'vec4', inputs: [ 'fColor0', 'fColor1', 'fColor2', 'fColor3' ].map( ( name ) => ( { name, type: 'vec4' } ) ).concat( [ 'fWeight0', 'fWeight1', 'fWeight2', 'fWeight3' ].map( ( name ) => ( { name, type: 'float' } ) ) ) } );

/** The 2D separable filter and upstream's deringing clamp to the four nearest texels. The weights of
 * x are computed once for the four rows and those of y once (the same values upstream's five 1D
 * calls compute: the reference Lanczos's sines were most of the accumulation's time). */
function lanczos2D( weight: ( x: N ) => N, Samples: FetchedBicubicSamples, fPxFrac: N ): N {

	const wx = weights4( weight, fPxFrac.x );
	const wy = weights4( weight, fPxFrac.y );
	const fColorX0 = Filter4( Samples.fColor00, Samples.fColor10, Samples.fColor20, Samples.fColor30, ...wx );
	const fColorX1 = Filter4( Samples.fColor01, Samples.fColor11, Samples.fColor21, Samples.fColor31, ...wx );
	const fColorX2 = Filter4( Samples.fColor02, Samples.fColor12, Samples.fColor22, Samples.fColor32, ...wx );
	const fColorX3 = Filter4( Samples.fColor03, Samples.fColor13, Samples.fColor23, Samples.fColor33, ...wx );
	const fColorXY = Filter4( fColorX0, fColorX1, fColorX2, fColorX3, ...wy ).toVar();

	// Deringing

	// TODO: only use 4 by checking jitter
	const fDeringingSamples = [
		Samples.fColor11,
		Samples.fColor21,
		Samples.fColor12,
		Samples.fColor22,
	];

	let fDeringingMin = fDeringingSamples[ 0 ];
	let fDeringingMax = fDeringingSamples[ 0 ];

	for ( let iSampleIndex = 1; iSampleIndex < 4; ++ iSampleIndex ) {

		fDeringingMin = min( fDeringingMin, fDeringingSamples[ iSampleIndex ] );
		fDeringingMax = max( fDeringingMax, fDeringingSamples[ iSampleIndex ] );

	}

	return clamp( fColorXY, fDeringingMin, fDeringingMax );

}

/** Lanczos2( FetchedBicubicSamples, fPxFrac ) */
export function Lanczos2Samples( Samples: FetchedBicubicSamples, fPxFrac: N ): N {

	return lanczos2D( Lanczos2, Samples, fPxFrac );

}

/** Lanczos2LUT( FetchedBicubicSamples, fPxFrac ) */
export function Lanczos2LUT( b: Fsr3Bindings, Samples: FetchedBicubicSamples, fPxFrac: N ): N {

	return lanczos2D( ( x ) => Lanczos2_UseLUT( b, x ), Samples, fPxFrac );

}

/** Lanczos2Approx( FetchedBicubicSamples, fPxFrac ) */
export function Lanczos2ApproxSamples( Samples: FetchedBicubicSamples, fPxFrac: N ): N {

	return lanczos2D( Lanczos2ApproxNoClamp, Samples, fPxFrac );

}

// Clamp by offset direction. Assuming iPxSample is already in range and iPxOffset is compile time constant.
export const ClampCoord = /*@__PURE__*/ Fn( ( [ iPxSample, iPxOffset, iTextureSize ]: N[] ) => {

	const result = iPxSample.add( iPxOffset ).toVar();
	result.x.assign( max( int( 1 ), min( result.x, iTextureSize.x.sub( 2 ) ) ) );
	result.y.assign( max( int( 1 ), min( result.y, iTextureSize.y.sub( 2 ) ) ) );
	return result;

} ).setLayout( { name: 'ClampCoord', type: 'ivec2', inputs: [ { name: 'iPxSample', type: 'ivec2' }, { name: 'iPxOffset', type: 'ivec2' }, { name: 'iTextureSize', type: 'ivec2' } ] } );

/** DeclareCustomFetchBicubicSamples( Name, LoadTexture ): the 4 x 4 texels around iPxSample. */
export function FetchBicubicSamples( LoadTexture: ( p: N ) => N, iPxSample: N, iTextureSize: N ): FetchedBicubicSamples {

	const at = ( x: number, y: number ) => LoadTexture( ClampCoord( iPxSample, ivec2( x, y ), iTextureSize ) ).toVar();
	return {
		fColor00: at( - 1, - 1 ), fColor10: at( + 0, - 1 ), fColor20: at( + 1, - 1 ), fColor30: at( + 2, - 1 ),
		fColor01: at( - 1, + 0 ), fColor11: at( + 0, + 0 ), fColor21: at( + 1, + 0 ), fColor31: at( + 2, + 0 ),
		fColor02: at( - 1, + 1 ), fColor12: at( + 0, + 1 ), fColor22: at( + 1, + 1 ), fColor32: at( + 2, + 1 ),
		fColor03: at( - 1, + 2 ), fColor13: at( + 0, + 2 ), fColor23: at( + 1, + 2 ), fColor33: at( + 2, + 2 ),
	};

}

/** DeclareCustomFetchBilinearSamples( Name, LoadTexture ) */
export function FetchBilinearSamples( LoadTexture: ( p: N ) => N, iPxSample: N, iTextureSize: N ): FetchedBilinearSamples {

	const at = ( x: number, y: number ) => LoadTexture( ClampCoord( iPxSample, ivec2( x, y ), iTextureSize ) ).toVar();
	return { fColor00: at( + 0, + 0 ), fColor10: at( + 1, + 0 ), fColor01: at( + 0, + 1 ), fColor11: at( + 1, + 1 ) };

}

// BE CAREFUL: there is some precision issues and (3253, 125) leading to (3252.9989778, 125.001102)
// is common, so iPxSample can "jitter"
/** DeclareCustomTextureSample( Name, InterpolateSamples, FetchSamples ): a sample at fUvSample. */
export function CustomTextureSample<S>( InterpolateSamples: ( s: S, fPxFrac: N ) => N, FetchSamples: ( iPxSample: N, iTextureSize: N ) => S, fUvSample: N, iTextureSize: N ): N {

	const fPxSample = fUvSample.mul( vec2( iTextureSize ) ).sub( vec2( 0.5, 0.5 ) ).toVar();
	const fPxFrac = ffxFract( fPxSample ).toVar();
	/* Clamp base coords */
	fPxSample.x.assign( max( float( 0.0 ), min( float( iTextureSize.x.sub( 1 ) ), fPxSample.x ) ) );
	fPxSample.y.assign( max( float( 0.0 ), min( float( iTextureSize.y.sub( 1 ) ), fPxSample.y ) ) );
	/* */
	const iPxSample = ivec2( floor( fPxSample ) ).toVar();
	const fColorXY = InterpolateSamples( FetchSamples( iPxSample, iTextureSize ), fPxFrac );
	return fColorXY;

}

/** FFX_FSR3UPSCALER_GET_LANCZOS_SAMPLER1D( x ): 0 Lanczos2, 1 Lanczos2LUT, 2 Lanczos2Approx. */
export function getLanczosSampler1D( b: Fsr3Bindings, type: 0 | 1 | 2 ): ( s: FetchedBicubicSamples, fPxFrac: N ) => N {

	if ( type === 0 ) return Lanczos2Samples;
	if ( type === 1 ) return ( s, f ) => Lanczos2LUT( b, s, f );
	return Lanczos2ApproxSamples;

}
