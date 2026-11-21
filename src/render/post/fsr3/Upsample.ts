/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_upsample.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { If, float, int, vec2, vec3, ivec2, min, max, clamp, floor, dot, length, exp, select } from './Tsl.ts';
import { ffxLerp, type N } from './Core.ts';
import {
	ClampLoad, IsOnScreen, RGBToYCoCg, YCoCgToRGB, Tonemap, InverseTonemap, RectificationBoxAddSample,
	RectificationBoxComputeVarianceBoxData, FSR3UPSCALER_EPSILON, fAverageLanczosWeightPerFrame,
	type AccumulationPassCommonParams, type AccumulationPassData, type RectificationBox,
} from './Common.ts';
import { Lanczos2, Lanczos2_UseLUT, Lanczos2ApproxSq } from './Sample.ts';
import type { Fsr3Bindings } from './Bindings.ts';
import type { Fsr3ShaderOptions } from './Host.ts';

// FSR 3's upsampling: this frame's Lanczos contribution and the history's rectification box.
//
// The current frame's contribution: a sliced 2-lobe Lanczos over the 3 x 3 render texels nearest
// the output pixel (the other slices accumulate over the jitter sequence), weighted by how much
// the history is trusted, and the rectification box (the samples' weighted mean and deviation in
// YCoCg) the history is clipped against.
//
// Upstream's HDR branches (FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT) are complete: on an initial
// sample the nine samples go YCoCg -> RGB -> Tonemap -> YCoCg before the box, and the box centre
// comes back through InverseTonemap.
//
// Differences from upstream: the Xbox paired 16-bit path
// (FFX_FSR3UPSCALER_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS) and the FFX_HALF weight are not
// ported. The 3 x 3 loops are unrolled at build time, as upstream's FFX_UNROLL asks; the flips stay
// runtime selects.
//
// Every lerp is `ffxLerp` ( x, y, t ), never TSL's method form: three's `a.mix( b, t )` is
// mix( b, t, a ), not mix( a, b, t ). The kernel bias once went through it and came out as
// max + min ( weight - max ): at a 1.5x upscale 1 instead of 1.5 while accumulating (a Lanczos
// two render pixels wide, softening every frame's contribution) and 0, a 3 x 3 box, on new or
// disoccluded pixels; at a scale of 1 it equals upstream's only while the weight is 1.

export function Deringing( clippingBox: RectificationBox, fColor: N ): void {

	fColor.assign( clamp( fColor, clippingBox.aabbMin, clippingBox.aabbMax ) );

}

export function GetUpsampleLanczosWeight( b: Fsr3Bindings, o: Fsr3ShaderOptions, fSrcSampleOffset: N, fKernelWeight: N ): N {

	const fSrcSampleOffsetBiased = fSrcSampleOffset.mul( vec2( fKernelWeight ) ).toVar();
	const type = o.FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE;
	if ( type === 0 ) return Lanczos2( length( fSrcSampleOffsetBiased ) ); // LANCZOS_TYPE_REFERENCE
	if ( type === 1 ) return Lanczos2_UseLUT( b, length( fSrcSampleOffsetBiased ) ); // LANCZOS_TYPE_LUT
	return Lanczos2ApproxSq( dot( fSrcSampleOffsetBiased, fSrcSampleOffsetBiased ) ); // LANCZOS_TYPE_APPROXIMATE

}

export function ComputeMaxKernelWeight( b: Fsr3Bindings ): N {

	const fKernelSizeBias = float( 1.0 ).add( float( 1.0 ).div( vec2( b.DownscaleFactor() ) ).sub( 1.0 ).x );

	return min( float( 1.99 ), fKernelSizeBias );

}

export function LoadPreparedColor( b: Fsr3Bindings, iSamplePos: N ): N {

	const fRgb = max( vec3( 0, 0, 0 ), b.LoadInputColor( iSamplePos ) ).mul( b.Exposure() );
	const fPreparedYCoCg = RGBToYCoCg( fRgb );

	return fPreparedYCoCg;

}

export function ComputeUpsampledColorAndWeight( b: Fsr3Bindings, o: Fsr3ShaderOptions, params: AccumulationPassCommonParams, data: AccumulationPassData ): void {

	// We compute a sliced lanczos filter with 2 lobes (other slices are accumulated temporaly)
	const fDstOutputPos = vec2( params.iPxHrPos ).add( vec2( 0.5 ) ).toVar();
	const fSrcOutputPos = fDstOutputPos.mul( b.DownscaleFactor() ).toVar();
	const iSrcInputPos = ivec2( floor( fSrcOutputPos ) ).toVar();
	const fSrcUnjitteredPos = vec2( iSrcInputPos ).add( vec2( 0.5, 0.5 ) ).sub( b.Jitter() ).toVar(); // This is the un-jittered position of the sample at offset 0,0
	const fBaseSampleOffset = fSrcUnjitteredPos.sub( fSrcOutputPos ).toVar();

	const offsetTL = ivec2(
		select( fSrcUnjitteredPos.x.greaterThan( fSrcOutputPos.x ), int( - 2 ), int( - 1 ) ),
		select( fSrcUnjitteredPos.y.greaterThan( fSrcOutputPos.y ), int( - 2 ), int( - 1 ) ) ).toVar();

	//Load samples
	// If fSrcUnjitteredPos.y > fSrcOutputPos.y, indicates offsetTL.y = -2, sample offset Y will be [-2, 1], clipbox will be rows [1, 3].
	// Flip row# for sampling offset in this case, so first 0~2 rows in the sampled array can always be used for computing the clipbox.
	// This reduces branch or cmove on sampled colors, but moving this overhead to sample position / weight calculation time which apply to less values.
	const bFlipRow = fSrcUnjitteredPos.y.greaterThan( fSrcOutputPos.y ).toVar();
	const bFlipCol = fSrcUnjitteredPos.x.greaterThan( fSrcOutputPos.x ).toVar();
	const fOffsetTL = vec2( offsetTL ).toVar();

	const bIsInitialSample = params.fAccumulation.equal( 0.0 ).toVar();

	const sampleColRow = ( row: number, col: number ) => ivec2( select( bFlipCol, int( 3 - col ), int( col ) ), select( bFlipRow, int( 3 - row ), int( row ) ) );

	const fSamples: N[] = [];
	for ( let row = 0; row < 3; row ++ ) {

		for ( let col = 0; col < 3; col ++ ) {

			const iSampleColRow = sampleColRow( row, col );
			const iSrcSamplePos = iSrcInputPos.add( offsetTL ).add( iSampleColRow );
			const iSampleCoord = ClampLoad( iSrcSamplePos, ivec2( 0, 0 ), ivec2( b.RenderSize() ) );
			fSamples.push( LoadPreparedColor( b, iSampleCoord ).toVar() );

		}

	}

	if ( o.FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT ) {

		If( bIsInitialSample, () => {

			for ( let iSampleIndex = 0; iSampleIndex < 9; ++ iSampleIndex ) {

				//YCoCg -> RGB -> Tonemap -> YCoCg (Use RGB tonemapper to avoid color desaturation)
				fSamples[ iSampleIndex ].assign( RGBToYCoCg( Tonemap( YCoCgToRGB( fSamples[ iSampleIndex ] ) ) ) );

			}

		} );

	}

	// Identify how much of each upsampled color to be used for this frame
	const fKernelBiasMax = ComputeMaxKernelWeight( b ).toVar();
	const fKernelBiasMin = max( float( 1.0 ), float( 1.0 ).add( fKernelBiasMax ).mul( 0.3 ) ).toVar();

	const fKernelBiasWeight =
		min( float( 1.0 ).sub( params.fDisocclusion.mul( 0.5 ) ),
			min( float( 1.0 ).sub( params.fShadingChange ),
				data.fHistoryWeight.mul( 5.0 ).saturate(),
			) ).toVar();

	const fKernelBias = ffxLerp( fKernelBiasMin, fKernelBiasMax, fKernelBiasWeight ).toVar();

	let iSampleIndex = 0;
	for ( let row = 0; row < 3; row ++ ) {

		for ( let col = 0; col < 3; col ++ ) {

			const sampleColRowV = sampleColRow( row, col );
			const fOffset = fOffsetTL.add( vec2( sampleColRowV ) );
			const fSrcSampleOffset = fBaseSampleOffset.add( fOffset ).toVar();

			const iSrcSamplePos = iSrcInputPos.add( offsetTL ).add( sampleColRowV );

			const fOnScreenFactor = float( IsOnScreen( iSrcSamplePos, ivec2( b.RenderSize() ) ) ).toVar();

			const sample = fSamples[ iSampleIndex ];
			If( bIsInitialSample.not(), () => {

				const fSampleWeight = fOnScreenFactor.mul( float( GetUpsampleLanczosWeight( b, o, fSrcSampleOffset, fKernelBias ) ) ).toVar();

				data.fUpsampledColor.addAssign( sample.mul( fSampleWeight ) );
				data.fUpsampledWeight.addAssign( fSampleWeight );

			} );

			// Update rectification box
			{

				const fRectificationCurveBias = - 2.3;
				const fSrcSampleOffsetSq = dot( fSrcSampleOffset, fSrcSampleOffset );
				const fBoxSampleWeight = exp( fSrcSampleOffsetSq.mul( fRectificationCurveBias ) ).mul( fOnScreenFactor ).toVar();

				const bInitialSample = ( row === 0 ) && ( col === 0 );
				RectificationBoxAddSample( bInitialSample, data.clippingBox, sample, fBoxSampleWeight );

			}
			++ iSampleIndex;

		}

	}

	RectificationBoxComputeVarianceBoxData( data.clippingBox );

	data.fUpsampledWeight.mulAssign( float( data.fUpsampledWeight.greaterThan( FSR3UPSCALER_EPSILON ) ) );

	If( data.fUpsampledWeight.greaterThan( FSR3UPSCALER_EPSILON ), () => {

		// Normalize for deringing (we need to compare colors)
		data.fUpsampledColor.assign( data.fUpsampledColor.div( data.fUpsampledWeight ) );
		data.fUpsampledWeight.mulAssign( fAverageLanczosWeightPerFrame );

		Deringing( data.clippingBox, data.fUpsampledColor );

	} );

	// Initial samples using tonemapped upsampling
	If( bIsInitialSample, () => {

		if ( o.FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT ) {

			data.fUpsampledColor.assign( RGBToYCoCg( InverseTonemap( YCoCgToRGB( data.clippingBox.boxCenter ) ) ) );

		} else {

			data.fUpsampledColor.assign( data.clippingBox.boxCenter );

		}
		data.fUpsampledWeight.assign( 1.0 );
		data.fHistoryWeight.assign( 0.0 );

	} );

}
