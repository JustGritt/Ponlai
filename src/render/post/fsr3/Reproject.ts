/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_reproject.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { vec2, ivec2 } from './Tsl.ts';
import type { N } from './Core.ts';
import { IsUvInside, RGBToYCoCg, type AccumulationPassCommonParams, type AccumulationPassData } from './Common.ts';
import { CustomTextureSample, FetchBicubicSamples, getLanczosSampler1D } from './Sample.ts';
import type { Fsr3Bindings } from './Bindings.ts';
import type { Fsr3ShaderOptions } from './Host.ts';

// FSR 3's history reprojection: last frame's colour and lock, sampled with Lanczos.
//
// The history's reprojection: the previous frame's upscaled colour and lock, sampled with Lanczos
// at the reprojected position, and the motion vector the accumulation uses.
//
// Differences from upstream: the FFX_HALF history sampler (FFX_FSR3UPSCALER_OPTION_REPROJECT_
// SAMPLERS_USE_DATA_HALF) is not ported. FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE is the
// options' (0 reference by default, 1 the LUT).

export function WrapHistory( b: Fsr3Bindings, iPxSample: N ): N {

	return b.LoadHistory( iPxSample );

}

/** DeclareCustomFetchBicubicSamples( FetchHistorySamples, WrapHistory ) */
export function FetchHistorySamples( b: Fsr3Bindings, iPxSample: N, iTextureSize: N ) {

	return FetchBicubicSamples( ( p ) => WrapHistory( b, p ), iPxSample, iTextureSize );

}

/** DeclareCustomTextureSample( HistorySample, GET_LANCZOS_SAMPLER1D( REPROJECT_USE_LANCZOS_TYPE ), FetchHistorySamples ) */
export function HistorySample( b: Fsr3Bindings, o: Fsr3ShaderOptions, fUvSample: N, iTextureSize: N ): N {

	return CustomTextureSample( getLanczosSampler1D( b, o.FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE ), ( p, s ) => FetchHistorySamples( b, p, s ), fUvSample, iTextureSize );

}

export function GetMotionVector( b: Fsr3Bindings, o: Fsr3ShaderOptions, iPxHrPos: N, fHrUv: N ): N {

	if ( o.FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS ) {

		return b.LoadDilatedMotionVector( ivec2( fHrUv.mul( vec2( b.RenderSize() ) ) ) );

	}
	return b.LoadInputMotionVector( o, iPxHrPos );

}

export function ComputeReprojectedUVs( params: AccumulationPassCommonParams ): void {

	params.fReprojectedHrUv.assign( params.fHrUv.add( params.fMotionVector ) );

	params.bIsExistingSample.assign( IsUvInside( params.fReprojectedHrUv ) );

}

export function ReprojectHistoryColor( b: Fsr3Bindings, o: Fsr3ShaderOptions, params: AccumulationPassCommonParams, data: AccumulationPassData ): void {

	const fReprojectedHistory = HistorySample( b, o, params.fReprojectedHrUv, b.PreviousFrameUpscaleSize() ).toVar();

	data.fHistoryColor.assign( fReprojectedHistory.rgb );
	data.fHistoryColor.mulAssign( b.DeltaPreExposure() );
	data.fHistoryColor.mulAssign( b.Exposure() );

	data.fHistoryColor.assign( RGBToYCoCg( data.fHistoryColor ) );

	data.fLock.assign( fReprojectedHistory.w );

}
