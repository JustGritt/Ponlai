/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_accumulate.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { If, float, vec2, vec3, vec4, ivec2, bool, max, min, length, normalize, round } from './Tsl.ts';
import { ffxLerp, ffxSaturate, ffxPow, component, type N } from './Core.ts';
import {
	FSR3UPSCALER_FP16_MIN, FSR3UPSCALER_EPSILON, fLockThreshold, fLockMax, REACTIVE, DISOCCLUSION, SHADING_CHANGE, ACCUMULAION,
	RGBToYCoCg, YCoCgToRGB, Tonemap, InverseTonemap, IsUvInside, ClampUv, Get4KVelocity, newRectificationBox,
	type AccumulationPassCommonParams, type AccumulationPassData,
} from './Common.ts';
import { ComputeReprojectedUVs, GetMotionVector, ReprojectHistoryColor } from './Reproject.ts';
import { ComputeUpsampledColorAndWeight } from './Upsample.ts';
import type { Fsr3Bindings } from './Bindings.ts';
import type { Fsr3ShaderOptions } from './Host.ts';

// FSR 3's accumulation: the history reprojected, clipped and blended with this frame's colour.
//
// The accumulation: reprojects the history, updates the locks (thin-feature protection), weighs
// the history against this frame's upsampled colour, clips the history to the rectification box
// and blends, storing colour and lock for the next frame (and the output when RCAS is off).
//
// Upstream's HDR branches (FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT) are complete: the upsampled and
// history colours go YCoCg -> RGB -> Tonemap -> YCoCg before the blend and the result through
// InverseTonemap after it. Exposure and pre-exposure: the history is multiplied by
// DeltaPreExposure() and Exposure() when reprojected (Reproject.ts), the inputs by Exposure() when
// loaded (Upsample.ts), and the stored history divided by Exposure() here, as upstream.
//
// Differences from upstream: none in the maths. The two `Accumulate` overloads are `Accumulate`
// (the blend) and `AccumulatePixel` (the pass body); the per-pixel structs are TSL variables.

export function Accumulate( o: Fsr3ShaderOptions, data: AccumulationPassData ): void {

	// Avoid invalid values when accumulation and upsampled weight is 0
	data.fHistoryWeight.mulAssign( float( data.fHistoryWeight.greaterThan( FSR3UPSCALER_FP16_MIN ) ) );
	data.fHistoryWeight.assign( max( float( FSR3UPSCALER_EPSILON ), data.fHistoryWeight.add( data.fUpsampledWeight ) ) );

	if ( o.FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT ) {

		//YCoCg -> RGB -> Tonemap -> YCoCg (Use RGB tonemapper to avoid color desaturation)
		data.fUpsampledColor.assign( RGBToYCoCg( Tonemap( YCoCgToRGB( data.fUpsampledColor ) ) ) );
		data.fHistoryColor.assign( RGBToYCoCg( Tonemap( YCoCgToRGB( data.fHistoryColor ) ) ) );

	}

	const fAlpha = ffxSaturate( data.fUpsampledWeight.div( data.fHistoryWeight ) ).toVar();
	data.fHistoryColor.assign( ffxLerp( data.fHistoryColor, data.fUpsampledColor, fAlpha ) );
	data.fHistoryColor.assign( YCoCgToRGB( data.fHistoryColor ) );

	if ( o.FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT ) {

		data.fHistoryColor.assign( InverseTonemap( data.fHistoryColor ) );

	}

}

export function RectifyHistory( params: AccumulationPassCommonParams, data: AccumulationPassData ): void {

	const f4kVelocityFactor = ffxSaturate( params.f4KVelocity.div( 20.0 ) );
	const fDistanceFactor = ffxSaturate( float( 0.75 ).sub( params.fFarthestDepthInMeters.div( 20.0 ) ) );
	const fAccumulationFactor = float( 1.0 ).sub( params.fAccumulation );
	const fReactiveFactor = ffxPow( params.fReactiveMask, float( 1.0 / 2.0 ) );
	const fShadingChangeFactor = params.fShadingChange;
	const fBoxScaleT = max( f4kVelocityFactor, max( fDistanceFactor, max( fAccumulationFactor, max( fReactiveFactor, fShadingChangeFactor ) ) ) ).toVar();

	const fBoxScale = ffxLerp( float( 3.0 ), float( 1.0 ), fBoxScaleT ).toVar();

	const fScaledBoxVec = data.clippingBox.boxVec.mul( vec3( 1.7, 1.0, 1.0 ) ).mul( fBoxScale ).toVar();
	const fClampedScaledBoxVec = max( fScaledBoxVec, vec3( 1.193e-7, 1.193e-7, 1.193e-7 ) );
	const fTransformedHistoryColor = data.fHistoryColor.sub( data.clippingBox.boxCenter ).div( fClampedScaledBoxVec ).toVar();

	If( length( fTransformedHistoryColor ).greaterThan( 1.0 ), () => {

		const fClampedHistoryColor = normalize( fTransformedHistoryColor );
		const fFinalClampedHistoryColor = fClampedHistoryColor.mul( fScaledBoxVec ).add( data.clippingBox.boxCenter ).toVar();

		// Scale history color using rectification info, also using accumulation mask to avoid potential invalid color protection
		const fHistoryContribution = max( params.fLumaInstabilityFactor, data.fLockContributionThisFrame ).mul( params.fAccumulation ).mul( float( 1 ).sub( params.fDisocclusion ) );
		data.fHistoryColor.assign( ffxLerp( fFinalClampedHistoryColor, data.fHistoryColor, ffxSaturate( fHistoryContribution ) ) );

	} );

}

export function UpdateLockStatus( b: Fsr3Bindings, params: AccumulationPassCommonParams, data: AccumulationPassData ): void {

	data.fLock.mulAssign( float( params.bIsNewSample.equal( false ) ) );

	const fLifetimeDecreaseFactor = max( ffxSaturate( params.fShadingChange ), max( params.fReactiveMask, params.fDisocclusion ) ).toVar();
	data.fLock.assign( max( float( 0.0 ), data.fLock.sub( fLifetimeDecreaseFactor.mul( fLockMax ) ) ) );

	// Compute this frame lock contribution
	data.fLockContributionThisFrame.assign( ffxSaturate( ffxSaturate( data.fLock.sub( fLockThreshold ) ).mul( fLockMax - fLockThreshold ) ) );

	const fNewLockIntensity = b.LoadRwNewLocks( params.iPxHrPos ).mul( float( 1.0 ).sub( max( params.fShadingChange.mul( 0 ), params.fReactiveMask ) ) );
	data.fLock.assign( max( float( 0.0 ), min( data.fLock.add( fNewLockIntensity ), float( fLockMax ) ) ) );

	// Preparing for next frame
	const fLifetimeDecrease = float( 0.1 ).div( b.JitterSequenceLength() ).mul( float( 1.0 ).sub( fLifetimeDecreaseFactor ) );
	data.fLock.assign( max( float( 0.0 ), data.fLock.sub( fLifetimeDecrease ) ) );

	// we expect similar motion for next frame
	// kill lock if that location is outside screen, avoid locks to be clamped to screen borders
	const fEstimatedUvNextFrame = params.fHrUv.sub( params.fMotionVector );
	data.fLock.mulAssign( float( IsUvInside( fEstimatedUvNextFrame ).equal( true ) ) );

}

export function ComputeBaseAccumulationWeight( b: Fsr3Bindings, params: AccumulationPassCommonParams, data: AccumulationPassData ): void {

	const fBaseAccumulation = params.fAccumulation.toVar();

	fBaseAccumulation.assign( min( fBaseAccumulation, ffxLerp( fBaseAccumulation, float( 0.15 ), ffxSaturate( max( float( 0.0 ), params.f4KVelocity.mul( b.VelocityFactor() ).div( 0.5 ) ) ) ) ) );

	data.fHistoryWeight.assign( fBaseAccumulation );

}

export function InitPassData( b: Fsr3Bindings, o: Fsr3ShaderOptions, iPxHrPos: N ): { params: AccumulationPassCommonParams, data: AccumulationPassData } {

	// Init constant params
	const fHrUv = vec2( iPxHrPos ).add( 0.5 ).div( vec2( b.UpscaleSize() ) ).toVar( 'fHrUv' );
	const fLrUvJittered = fHrUv.add( b.Jitter().div( vec2( b.RenderSize() ) ) ).toVar( 'fLrUvJittered' );
	const fLrUv_HwSampler = ClampUv( fLrUvJittered, b.RenderSize(), b.MaxRenderSize() ).toVar( 'fLrUv_HwSampler' );
	const fMotionVector = vec2( GetMotionVector( b, o, iPxHrPos, fHrUv ) ).toVar( 'fMotionVector' );
	const f4KVelocity = Get4KVelocity( fMotionVector ).toVar( 'f4KVelocity' );

	const params: AccumulationPassCommonParams = {
		iPxHrPos,
		fHrUv,
		fLrUvJittered,
		fLrUv_HwSampler,
		fMotionVector,
		fReprojectedHrUv: vec2( 0 ).toVar( 'fReprojectedHrUv' ),
		f4KVelocity,
		fDisocclusion: float( 0 ).toVar( 'fDisocclusion' ),
		fReactiveMask: float( 0 ).toVar( 'fReactiveMask' ),
		fShadingChange: float( 0 ).toVar( 'fShadingChange' ),
		fAccumulation: float( 0 ).toVar( 'fAccumulation' ),
		fLumaInstabilityFactor: float( 0 ).toVar( 'fLumaInstabilityFactor' ),
		fFarthestDepthInMeters: float( 0 ).toVar( 'fFarthestDepthInMeters' ),
		bIsExistingSample: bool( false ).toVar( 'bIsExistingSample' ),
		bIsNewSample: bool( false ).toVar( 'bIsNewSample' ),
	};

	ComputeReprojectedUVs( params );

	const fLumaInstabilityUv_HW = ClampUv( fHrUv, b.RenderSize(), b.MaxRenderSize() );
	params.fLumaInstabilityFactor.assign( b.SampleLumaInstability( fLumaInstabilityUv_HW ) );

	const fFarthestDepthUv = ClampUv( params.fLrUvJittered, ivec2( b.RenderSize() ).div( 2 ), b.GetFarthestDepthMip1ResourceDimensions() );
	params.fFarthestDepthInMeters.assign( b.SampleFarthestDepthMip1( fFarthestDepthUv ) );
	params.bIsNewSample.assign( params.bIsExistingSample.equal( false ).or( b.FrameIndex().equal( 0 ) ) );

	const fDilatedReactiveMasks = b.SampleDilatedReactiveMasks( params.fLrUv_HwSampler ).toVar();
	params.fReactiveMask.assign( ffxSaturate( component( fDilatedReactiveMasks, REACTIVE ) ) );
	params.fDisocclusion.assign( ffxSaturate( component( fDilatedReactiveMasks, DISOCCLUSION ) ) );
	params.fShadingChange.assign( ffxSaturate( component( fDilatedReactiveMasks, SHADING_CHANGE ) ) );
	params.fAccumulation.assign( ffxSaturate( component( fDilatedReactiveMasks, ACCUMULAION ) ) );
	params.fAccumulation.mulAssign( float( round( params.fAccumulation.mul( 100.0 ) ).greaterThan( 1.0 ) ) );

	// Init variable data
	const data: AccumulationPassData = {
		clippingBox: newRectificationBox(),
		fUpsampledColor: vec3( 0.0, 0.0, 0.0 ).toVar( 'fUpsampledColor' ),
		fHistoryColor: vec3( 0.0, 0.0, 0.0 ).toVar( 'fHistoryColor' ),
		fHistoryWeight: float( 1.0 ).toVar( 'fHistoryWeight' ),
		fUpsampledWeight: float( 0.0 ).toVar( 'fUpsampledWeight' ),
		fLock: float( 0.0 ).toVar( 'fLock' ),
		fLockContributionThisFrame: float( 0.0 ).toVar( 'fLockContributionThisFrame' ),
	};

	return { params, data };

}

/** Accumulate( FfxInt32x2 iPxHrPos ): the pass body. */
export function AccumulatePixel( b: Fsr3Bindings, o: Fsr3ShaderOptions, iPxHrPos: N ): void {

	const { params, data } = InitPassData( b, o, iPxHrPos );

	If( params.bIsExistingSample.and( params.bIsNewSample.not() ), () => {

		ReprojectHistoryColor( b, o, params, data );

	} );

	UpdateLockStatus( b, params, data );

	ComputeBaseAccumulationWeight( b, params, data );

	ComputeUpsampledColorAndWeight( b, o, params, data );

	RectifyHistory( params, data );

	Accumulate( o, data );

	data.fHistoryColor.divAssign( b.Exposure() );

	data.fHistoryColor.assign( max( data.fHistoryColor, vec3( 0.0, 0.0, 0.0 ) ) );

	b.StoreInternalColorAndWeight( iPxHrPos, vec4( data.fHistoryColor, data.fLock ) );

	// Output final color when RCAS is disabled
	if ( ! o.FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING ) {

		b.StoreUpscaledOutput( iPxHrPos, data.fHistoryColor );

	}
	b.StoreNewLocks( iPxHrPos, float( 0 ) );

}
