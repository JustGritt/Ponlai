/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_luma_instability.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { If, float, vec2, vec4, max, sign } from './Tsl.ts';
import { ffxSaturate, component, type N } from './Core.ts';
import {
	MinDividedByMax, IsUvInside, ClampUv, Get4KVelocity, REACTIVE, DISOCCLUSION, SHADING_CHANGE, ACCUMULAION,
} from './Common.ts';
import type { Fsr3Bindings } from './Bindings.ts';

// FSR 3's luma instability: pixels whose luma flips back toward a recent frame's.
//
// Luma instability at render size: a luma that flips back towards one of the last three frames'
// (an oscillation the accumulation should not chase) marks the pixel unstable, unless it moves,
// was disoccluded, is reactive or changed its shading. Also shifts the four-frame luma history.
//
// Differences from upstream: none in the maths. The history loop is unrolled; the farthest-depth
// sample ComputeLumaInstabilityFactor takes and never reads is left out (and its parameter).

const N_MINUS_1 = 0;
const N_MINUS_2 = 1;
const N_MINUS_3 = 2;
const N_MINUS_4 = 3;

export interface LumaInstabilityFactorData {
	fLumaHistory: N;
	fLumaInstabilityFactor: N;
}

export function ComputeLumaInstabilityFactor( b: Fsr3Bindings, data: LumaInstabilityFactorData, fCurrentFrameLuma: N ): void {

	const h = ( i: number ) => component( data.fLumaHistory, i );

	const fLumaInstability = float( 0.0 ).toVar( 'fLumaInstability' );
	const fDiffs0 = fCurrentFrameLuma.sub( h( N_MINUS_1 ) ).toVar();
	const fSimilarity0 = MinDividedByMax( fCurrentFrameLuma, h( N_MINUS_1 ), float( 1.0 ) ).toVar();

	const fMaxSimilarity = fSimilarity0.toVar( 'fMaxSimilarity' );

	If( fSimilarity0.lessThan( 1.0 ), () => {

		for ( let i = N_MINUS_2; i <= N_MINUS_4; i ++ ) {

			const fDiffs1 = fCurrentFrameLuma.sub( h( i ) );
			const fSimilarity1 = MinDividedByMax( fCurrentFrameLuma, h( i ) );

			If( sign( fDiffs0 ).equal( sign( fDiffs1 ) ), () => {

				fMaxSimilarity.assign( max( fMaxSimilarity, fSimilarity1 ) );

			} );

		}

		fLumaInstability.assign( float( fMaxSimilarity.greaterThan( fSimilarity0 ) ) );

	} );

	// Shift history
	data.fLumaHistory.assign( vec4( fCurrentFrameLuma, h( N_MINUS_1 ), h( N_MINUS_2 ), h( N_MINUS_3 ) ) );

	data.fLumaHistory.divAssign( b.Exposure() );

	data.fLumaInstabilityFactor.assign( fLumaInstability.mul( float( h( N_MINUS_4 ).notEqual( 0 ) ) ) );

}

export function LumaInstability( b: Fsr3Bindings, iPxPos: N ): void {

	const data: LumaInstabilityFactorData = {
		fLumaInstabilityFactor: float( 0.0 ).toVar( 'fLumaInstabilityFactor' ),
		fLumaHistory: vec4( 0.0, 0.0, 0.0, 0.0 ).toVar( 'fLumaHistory' ),
	};

	const fDilatedMotionVector = b.LoadDilatedMotionVector( iPxPos ).toVar();
	const fUv = vec2( iPxPos ).add( 0.5 ).div( vec2( b.RenderSize() ) ).toVar();
	const fUvCurrFrameJittered = fUv.add( b.Jitter().div( vec2( b.RenderSize() ) ) ).toVar();
	const fUvPrevFrameJittered = fUv.add( b.PreviousFrameJitter().div( vec2( b.PreviousFrameRenderSize() ) ) );
	const fReprojectedUv = fUvPrevFrameJittered.add( fDilatedMotionVector ).toVar();

	If( IsUvInside( fReprojectedUv ), () => {

		const fUvReactive_HW = ClampUv( fUvCurrFrameJittered, b.RenderSize(), b.MaxRenderSize() );

		const fDilatedReactiveMasks = b.SampleDilatedReactiveMasks( fUvReactive_HW ).toVar();
		const fReactiveMask = ffxSaturate( component( fDilatedReactiveMasks, REACTIVE ) ).toVar();
		const fDisocclusion = ffxSaturate( component( fDilatedReactiveMasks, DISOCCLUSION ) ).toVar();
		const fShadingChange = ffxSaturate( component( fDilatedReactiveMasks, SHADING_CHANGE ) ).toVar();
		const fAccumulation = ffxSaturate( component( fDilatedReactiveMasks, ACCUMULAION ) );

		const bAccumulationFactor = fAccumulation.greaterThan( 0.9 );

		const bComputeInstability = bAccumulationFactor;

		If( bComputeInstability, () => {

			const fUv_HW = ClampUv( fUvCurrFrameJittered, b.RenderSize(), b.MaxRenderSize() );
			const fCurrentFrameLuma = b.SampleCurrentLuma( fUv_HW ).mul( b.Exposure() ).toVar();

			const fReprojectedUv_HW = ClampUv( fReprojectedUv, b.PreviousFrameRenderSize(), b.MaxRenderSize() );
			data.fLumaHistory.assign( b.SampleLumaHistory( fReprojectedUv_HW ).mul( b.DeltaPreExposure() ).mul( b.Exposure() ) );

			// ( upstream samples the farthest depth's mip 1 here and passes it to
			// ComputeLumaInstabilityFactor, which does not read it: left out )

			ComputeLumaInstabilityFactor( b, data, fCurrentFrameLuma );

			const fVelocityWeight = float( 1.0 ).sub( ffxSaturate( Get4KVelocity( fDilatedMotionVector ).div( 20.0 ) ) );
			data.fLumaInstabilityFactor.mulAssign( fVelocityWeight.mul( float( 1.0 ).sub( fDisocclusion ) ).mul( float( 1.0 ).sub( fReactiveMask ) ).mul( float( 1.0 ).sub( fShadingChange ) ) );

		} );

	} );

	b.StoreLumaHistory( iPxPos, data.fLumaHistory );
	b.StoreLumaInstability( iPxPos, data.fLumaInstabilityFactor );

}
