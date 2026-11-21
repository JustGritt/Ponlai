/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_prepare_reactivity.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { If, float, uint, vec2, vec4, ivec2, bool, abs, min, max, length, round, select } from './Tsl.ts';
import { ffxLerp, ffxSaturate, type N } from './Core.ts';
import {
	FSR3UPSCALER_FP16_MAX, FSR3UPSCALER_FP32_MAX, FSR3UPSCALER_FP32_MIN, fReconstructedDepthBilinearWeightThreshold,
	ReconstructedDepthMvPxThreshold, Get4KVelocity, GetBilinearSamplingData, ClampLoad, IsOnScreen, GetViewSpaceDepth,
	GetViewSpaceDepthInMeters, MinDividedByMax, ClampUv, IsUvInside, ShadingChangeRenderSize, ShadingChangeMaxRenderSize,
	ComputeHrPosFromLrPos,
} from './Common.ts';
import type { Fsr3Bindings } from './Bindings.ts';

// FSR 3's reactivity pass: disocclusions, shading change, reactiveness and new locks.
//
// Prepares what the accumulation weighs by, at render size: disocclusions (this depth against the
// reconstructed previous depth), the shading change (the dilated reactive mask or the
// shading-change pass), the reactiveness (motion divergence or the transparency mask), the
// accumulation count (stored for the next frame), and new locks on thin features.
//
// Differences from upstream: none in the maths. The disocclusion loop's early exit
// (`&& bPotentialDisocclusion`) wraps each unrolled step in that condition; the thin-feature
// test's bit masks are build-time constants and its pattern a uint variable.

export function ComputeDisocclusions( b: Fsr3Bindings, fUv: N, fMotionVectorIn: N, fCurrentDepthViewSpace: N ): N {

	const fNearestDepthInMeters = min( fCurrentDepthViewSpace.mul( b.ViewSpaceToMetersFactor() ), float( FSR3UPSCALER_FP16_MAX ) );
	const fReconstructedDeptMvThreshold = ReconstructedDepthMvPxThreshold( fNearestDepthInMeters );

	const fMotionVector = vec2( fMotionVectorIn ).toVar();
	fMotionVector.mulAssign( float( Get4KVelocity( fMotionVector ).greaterThan( fReconstructedDeptMvThreshold ) ) );

	const fReprojectedUv = fUv.add( fMotionVector );
	const bilinearInfo = GetBilinearSamplingData( fReprojectedUv, b.RenderSize() );

	const fDisocclusion = float( 0.0 ).toVar( 'fDisocclusion' );
	const fWeightSum = float( 0.0 ).toVar( 'fWeightSum' );
	const bPotentialDisocclusion = bool( true ).toVar( 'bPotentialDisocclusion' );

	for ( let iSampleIndex = 0; iSampleIndex < 4; iSampleIndex ++ ) {

		If( bPotentialDisocclusion, () => {

			const iOffset = bilinearInfo.iOffsets[ iSampleIndex ];
			const iSamplePos = ClampLoad( bilinearInfo.iBasePos, iOffset, ivec2( b.RenderSize() ) ).toVar();

			If( IsOnScreen( iSamplePos, b.RenderSize() ), () => {

				const fWeight = bilinearInfo.fWeights[ iSampleIndex ];
				If( fWeight.greaterThan( fReconstructedDepthBilinearWeightThreshold ), () => {

					const fPrevNearestDepthViewSpace = GetViewSpaceDepth( b, b.LoadReconstructedPrevDepth( iSamplePos ) ).toVar();
					const fDepthDifference = fCurrentDepthViewSpace.sub( fPrevNearestDepthViewSpace ).toVar();

					bPotentialDisocclusion.assign( bPotentialDisocclusion.and( fDepthDifference.greaterThan( FSR3UPSCALER_FP32_MIN ) ) );

					If( bPotentialDisocclusion, () => {

						const fHalfViewportWidth = length( vec2( b.RenderSize() ).mul( 0.5 ) );
						const fDepthThreshold = max( fCurrentDepthViewSpace, fPrevNearestDepthViewSpace );

						const Ksep = 1.37e-05;
						const fRequiredDepthSeparation = float( Ksep ).mul( fHalfViewportWidth ).mul( fDepthThreshold );

						fDisocclusion.addAssign( ffxSaturate( float( fRequiredDepthSeparation.div( fDepthDifference ) ) ).mul( fWeight ) );
						fWeightSum.addAssign( fWeight );

					} );

				} );

			} );

		} );

	}

	fDisocclusion.assign( select( bPotentialDisocclusion.and( fWeightSum.greaterThan( 0 ) ), ffxSaturate( float( 1.0 ).sub( fDisocclusion.div( fWeightSum ) ) ), float( 0.0 ) ) );

	return fDisocclusion;

}

export function ComputeMotionDivergence( b: Fsr3Bindings, fUv: N, fMotionVector: N, fCurrentDepthSample: N ): N {

	const iPxReprojectedPos = ivec2( fUv.add( fMotionVector ).mul( vec2( b.RenderSize() ) ) ).toVar();
	const fReprojectedDepth = b.LoadDilatedDepth( iPxReprojectedPos ).toVar();
	const fReprojectedMotionVector = b.LoadDilatedMotionVector( iPxReprojectedPos ).toVar();

	const fReprojectedVelocity = Get4KVelocity( fReprojectedMotionVector ).toVar();
	const f4KVelocity = Get4KVelocity( fMotionVector ).toVar();

	// ( upstream computes fMaxLen = max( length( mv ), length( reprojected mv ) ) and leaves it unused )

	const fNucleusDepthInMeters = GetViewSpaceDepthInMeters( b, fReprojectedDepth );
	const fCurrentDepthInMeters = GetViewSpaceDepthInMeters( b, fCurrentDepthSample );

	const fDistanceFactor = MinDividedByMax( fNucleusDepthInMeters, fCurrentDepthInMeters );
	const fVelocityFactor = ffxSaturate( f4KVelocity.div( 10.0 ) );
	const fMotionVectorFieldConfidence = float( 1.0 ).sub( ffxSaturate( fReprojectedVelocity.div( f4KVelocity ) ) ).mul( fDistanceFactor ).mul( fVelocityFactor ).toVar();

	// a still pixel divides by 0: HLSL's saturate flushes the NaN to 0 and the velocity factor
	// makes the result 0, while WGSL leaves a NaN or infinity indeterminate, so take upstream's 0
	return select( f4KVelocity.greaterThan( 0.0 ), fMotionVectorFieldConfidence, float( 0.0 ) );

}

export function DilateReactiveMasks( b: Fsr3Bindings, iPxPos: N ): N {

	const fDilatedReactiveMasks = float( 0.0 ).toVar( 'fDilatedReactiveMasks' );

	for ( let y = - 1; y <= 1; y ++ ) {

		for ( let x = - 1; x <= 1; x ++ ) {

			const sampleCoord = ClampLoad( iPxPos, ivec2( x, y ), ivec2( b.RenderSize() ) );
			fDilatedReactiveMasks.assign( max( fDilatedReactiveMasks, b.LoadReactiveMask( sampleCoord ) ) );

		}

	}

	return fDilatedReactiveMasks;

}

export function DilateTransparencyAndCompositionMasks( b: Fsr3Bindings, fUv: N ): N {

	const fUvTransparencyAndCompositionMask = ClampUv( fUv, b.RenderSize(), b.GetTransparencyAndCompositionMaskResourceDimensions() );
	return b.SampleTransparencyAndCompositionMask( fUvTransparencyAndCompositionMask );

}

export function ComputeThinFeatureConfidence( b: Fsr3Bindings, iPxPos: N ): N {

	/*
	 1 2 3
	 4 0 5
	 6 7 8
	*/

	const iNucleusIndex = 0;
	const iSampleCount = 9;
	const iSampleOffsets = [
		[ + 0, + 0 ],
		[ - 1, - 1 ],
		[ + 0, - 1 ],
		[ + 1, - 1 ],
		[ - 1, + 0 ],
		[ + 1, + 0 ],
		[ - 1, + 1 ],
		[ + 0, + 1 ],
		[ + 1, + 1 ],
	];

	const result = float( 0 ).toVar( 'fThinFeatureConfidence' );

	const fSamples: N[] = [];

	const fLumaMin = float( 0 ).toVar( 'fLumaMin' );
	fLumaMin.assign( FSR3UPSCALER_FP32_MAX() );
	const fLumaMax = float( FSR3UPSCALER_FP32_MIN ).toVar( 'fLumaMax' );

	for ( let iSampleIndex = 0; iSampleIndex < iSampleCount; ++ iSampleIndex ) {

		const iPxSamplePos = ClampLoad( iPxPos, ivec2( iSampleOffsets[ iSampleIndex ][ 0 ], iSampleOffsets[ iSampleIndex ][ 1 ] ), ivec2( b.RenderSize() ) );
		fSamples[ iSampleIndex ] = b.LoadCurrentLuma( iPxSamplePos ).mul( b.Exposure() ).toVar();

		fLumaMin.assign( min( fLumaMin, fSamples[ iSampleIndex ] ) );
		fLumaMax.assign( max( fLumaMax, fSamples[ iSampleIndex ] ) );

	}

	const fThreshold = 0.9;
	const fDissimilarLumaMin = float( 0 ).toVar( 'fDissimilarLumaMin' );
	fDissimilarLumaMin.assign( FSR3UPSCALER_FP32_MAX() );
	const fDissimilarLumaMax = float( 0 ).toVar( 'fDissimilarLumaMax' );

	const SETBIT = ( x: number ) => ( 1 << x ) >>> 0;
	const uPatternMask = uint( SETBIT( iNucleusIndex ) ).toVar( 'uPatternMask' ); // Flag nucleus as similar

	const uNumRejectionMasks = 4;
	const uRejectionMasks = [
		SETBIT( 1 ) | SETBIT( 2 ) | SETBIT( 4 ) | SETBIT( iNucleusIndex ), // Upper left
		SETBIT( 2 ) | SETBIT( 3 ) | SETBIT( 5 ) | SETBIT( iNucleusIndex ), // Upper right
		SETBIT( 4 ) | SETBIT( 6 ) | SETBIT( 7 ) | SETBIT( iNucleusIndex ), // Lower left
		SETBIT( 5 ) | SETBIT( 7 ) | SETBIT( 8 ) | SETBIT( iNucleusIndex ), // Lower right
	];

	let iBitIndex = 1;
	for ( let iSampleIndex = 1; iSampleIndex < iSampleCount; ++ iSampleIndex, ++ iBitIndex ) {

		const fDifference = abs( fSamples[ iSampleIndex ].sub( fSamples[ iNucleusIndex ] ) ).div( fLumaMax.sub( fLumaMin ) );

		const bit = SETBIT( iBitIndex );
		If( fDifference.lessThan( fThreshold ), () => {

			uPatternMask.assign( uPatternMask.bitOr( uint( bit ) ) );

		} ).Else( () => {

			fDissimilarLumaMin.assign( min( fDissimilarLumaMin, fSamples[ iSampleIndex ] ) );
			fDissimilarLumaMax.assign( max( fDissimilarLumaMax, fSamples[ iSampleIndex ] ) );

		} );

	}

	const bIsRidge = fSamples[ iNucleusIndex ].greaterThan( fDissimilarLumaMax ).or( fSamples[ iNucleusIndex ].lessThan( fDissimilarLumaMin ) ).toVar();

	// the early returns become one flag: the confidence is 0 unless the nucleus is a ridge and no
	// rejection mask is fully set
	const bRejected = bIsRidge.not().toVar( 'bRejected' );
	for ( let i = 0; i < uNumRejectionMasks; i ++ ) {

		const m = uint( uRejectionMasks[ i ] >>> 0 );
		bRejected.assign( bRejected.or( uPatternMask.bitAnd( m ).equal( m ) ) );

	}

	result.assign( select( bRejected, float( 0.0 ), float( 1.0 ).sub( fLumaMin.div( fLumaMax ) ) ) );
	return result;

}

export function UpdateAccumulation( b: Fsr3Bindings, iPxPos: N, fUv: N, fMotionVector: N, fDisocclusion: N, fShadingChange: N ): N {

	const fReprojectedUv = fUv.add( fMotionVector ).toVar();
	const fAccumulation = float( 0.0 ).toVar( 'fAccumulation' );

	If( IsUvInside( fReprojectedUv ), () => {

		const fReprojectedUv_HW = ClampUv( fReprojectedUv, b.PreviousFrameRenderSize(), b.MaxRenderSize() );
		fAccumulation.assign( ffxSaturate( b.SampleAccumulation( fReprojectedUv_HW ) ) );

	} );

	const fAccumulationAddedPerFrame = b.AccumulationAddedPerFrame(); //default is 0.333

	// Assume at frame N+0 fShadingChange is 1.0, and all subsequent frames fShadingChange is 0.0 and fDisocclusion is 0.0. Then,
	// frame N+0 fAccumulation will be 0.000
	// frame N+2 fAccumulation will be 0.000 + 0.333 * 1 == 0.333
	// frame N+3 fAccumulation will be 0.000 + 0.333 * 2 == 0.666
	// frame N+4 fAccumulation will be 0.000 + 0.333 * 3 == 0.999
	fAccumulation.assign( ffxLerp( fAccumulation, float( 0.0 ), fShadingChange ) );

	const fMinDisocclusionAccumulation = b.MinDisocclusionAccumulation(); //default is -0.333

	// Assume at frame N+0 fDisocclusion is 1.0, and all subsequent frames fShadingChange is 0.0 and fDisocclusion is 0.0. Then,
	// frame N+0 fAccumulation will be -0.333f (but normalized to store in unorm)
	// frame N+1 fAccumulation will be -0.333f + 0.333 * 1 == 0.000
	// frame N+2 fAccumulation will be -0.333f + 0.333 * 2 == 0.333
	// frame N+3 fAccumulation will be -0.333f + 0.333 * 3 == 0.666
	// frame N+4 fAccumulation will be -0.333f + 0.333 * 4 == 0.999
	fAccumulation.assign( ffxLerp( fAccumulation, min( fMinDisocclusionAccumulation, fAccumulation ), fDisocclusion ) );

	fAccumulation.mulAssign( float( round( fAccumulation.mul( 100.0 ) ).greaterThan( 1.0 ) ) );

	// Update for next frame, normalize to store in unorm
	const fAccumulatedFramesToStore = ffxSaturate( fAccumulation.add( fAccumulationAddedPerFrame ) );
	b.StoreAccumulation( iPxPos, fAccumulatedFramesToStore );

	return fAccumulation;

}

export function ComputeShadingChange( b: Fsr3Bindings, fUv: N ): N {

	// NOTE: Here we re-apply jitter, will be reverted again when sampled in accumulation pass
	const fShadingChangeUv = ClampUv( fUv.sub( b.Jitter().div( vec2( b.RenderSize() ) ) ), ShadingChangeRenderSize( b ), ShadingChangeMaxRenderSize( b ) );
	const fShadingChange = ffxSaturate( b.SampleShadingChange( fShadingChangeUv ) );

	return fShadingChange;

}

export function PrepareReactivity( b: Fsr3Bindings, iPxPos: N ): void {

	const fUv = vec2( iPxPos ).add( 0.5 ).div( vec2( b.RenderSize() ) ).toVar( 'fUv' );
	const fMotionVector = b.LoadDilatedMotionVector( iPxPos ).toVar( 'fMotionVector' );

	// Discard small mvs
	// ( upstream computes f4KVelocity and fDilatedUv here and leaves them unused )

	const fDilatedDepth = b.LoadDilatedDepth( iPxPos ).toVar( 'fDilatedDepth' );
	// ( and fDepthInMeters, unused )

	const fDisocclusion = ComputeDisocclusions( b, fUv, fMotionVector, GetViewSpaceDepth( b, fDilatedDepth ).toVar() ).toVar( 'fDisocclusionOut' );
	const fShadingChange = max( DilateReactiveMasks( b, iPxPos ), ComputeShadingChange( b, fUv ) ).toVar( 'fShadingChange' );

	const fMotionDivergence = ComputeMotionDivergence( b, fUv, fMotionVector, fDilatedDepth ).toVar( 'fMotionDivergence' );
	const fDilatedTransparencyAndComposition = DilateTransparencyAndCompositionMasks( b, fUv );
	const fFinalReactiveness = max( fMotionDivergence, fDilatedTransparencyAndComposition );

	const fAccumulation = UpdateAccumulation( b, iPxPos, fUv, fMotionVector, fDisocclusion, fShadingChange );

	// fOutput[ REACTIVE ], [ DISOCCLUSION ], [ SHADING_CHANGE ], [ ACCUMULAION ]
	const fOutput = vec4( fFinalReactiveness, fDisocclusion, fShadingChange, fAccumulation );

	b.StoreDilatedReactiveMasks( iPxPos, fOutput );

	const fLockStrength = ComputeThinFeatureConfidence( b, iPxPos ).toVar( 'fLockStrength' );
	If( fLockStrength.greaterThan( 1.0 / 100.0 ), () => {

		b.StoreNewLocks( ComputeHrPosFromLrPos( b, ivec2( iPxPos ) ), fLockStrength );

	} );

}
