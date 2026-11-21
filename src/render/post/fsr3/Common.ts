/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_common.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import {
	Fn, If, float, int, uint, vec2, vec3, ivec2, uvec2, abs, max, min, sqrt, dot, length, normalize, cross, exp, log2, floor,
	select,
} from './Tsl.ts';
import { ffxLerp, ffxSaturate, ffxPow, ffxFract, ffxAsFloat, allNotEqualZero3, type N } from './Core.ts';
import type { Fsr3Bindings } from './Bindings.ts';
import type { Fsr3ShaderOptions } from './Host.ts';

// The upscaler's shared constants, structs and helpers, in upstream's order.
//
// Differences from upstream:
// - Helpers that read the constant buffer (`ReconstructedDepthMvPxThreshold` does not, but
//   `GetViewSpaceDepth`, `ComputeHrPosFromLrPos`, `ShadingChangeRenderSize`, ... do) take the
//   bindings as their first argument and are expanded where they are called: three r186 can emit
//   a `Fn( ... ).setLayout( ... )` function with another kernel's uniform names when two compute
//   kernels share it (docs/reference/three-behaviour.md, Shader functions), so only the pure helpers are
//   `setLayout` functions (one WGSL function each).
// - HLSL's structs with `inout` (RectificationBox, AccumulationPassData, ...) are JavaScript
//   objects of TSL variables, the functions that take them `inout` expand in their one caller.
// - `FSR3UPSCALER_FP32_MAX` is built from its bits (0x7f7fffff): the literal's decimal is exact
//   in HLSL but three prints large floats in a form WGSL may round past the largest finite f32.
// - `IsOnScreen` compares as unsigned, as upstream's `FfxUInt32x2( pos ) < FfxUInt32x2( size )`.

export const FSR3UPSCALER_FP16_MIN = 6.10e-05;
export const FSR3UPSCALER_FP16_MAX = 65504.0;
export const FSR3UPSCALER_EPSILON = FSR3UPSCALER_FP16_MIN;
export const FSR3UPSCALER_TONEMAP_EPSILON = FSR3UPSCALER_FP16_MIN;
export const FSR3UPSCALER_FP32_MAX = (): N => ffxAsFloat( uint( 0x7f7fffff ) );
export const FSR3UPSCALER_FP32_MIN = 1.175494351e-38;

// Reconstructed depth usage
export const fReconstructedDepthBilinearWeightThreshold = FSR3UPSCALER_EPSILON * 10;

export const ReconstructedDepthMvPxThreshold = /*@__PURE__*/ Fn( ( [ fNearestDepthInMeters ]: N[] ) => {

	return ffxLerp( float( 0.25 ), float( 0.75 ), ffxSaturate( fNearestDepthInMeters.div( 100.0 ) ) );

} ).setLayout( { name: 'ReconstructedDepthMvPxThreshold', type: 'float', inputs: [ { name: 'fNearestDepthInMeters', type: 'float' } ] } );

// Accumulation
export const fUpsampleLanczosWeightScale = 1.0 / 16.0;
export const fAverageLanczosWeightPerFrame = 0.74 * fUpsampleLanczosWeightScale; // Average lanczos weight for jitter accumulated samples
export const fAccumulationMaxOnMotion = 3.0 * fUpsampleLanczosWeightScale;

export const SHADING_CHANGE_SET_SIZE = 5;
export const iShadingChangeMipStart = 0;
export const fShadingChangeSamplePow = 1.0 / 1.0;

export const fLockThreshold = 1.0;
export const fLockMax = 2.0;

export const REACTIVE = 0;
export const DISOCCLUSION = 1;
export const SHADING_CHANGE = 2;
export const ACCUMULAION = 3;

export const FRAME_INFO_EXPOSURE = 0;
export const FRAME_INFO_LOG_LUMA = 1;
export const FRAME_INFO_SCENE_AVERAGE_LUMA = 2;

export function TonemapFirstFrame( b: Fsr3Bindings ): N {

	// ( upstream's bEnabled is a constant true )
	return b.FrameIndex().equal( 0 );

}

export function AverageLanczosWeightPerFrame(): N {

	return float( 0.74 );

}

export function ShadingChangeRenderSize( b: Fsr3Bindings ): N {

	return ivec2( vec2( b.RenderSize() ).mul( 0.5 ) );

}

export function ShadingChangeMaxRenderSize( b: Fsr3Bindings ): N {

	return ivec2( vec2( b.MaxRenderSize() ).mul( 0.5 ) );

}

export function PreviousFrameShadingChangeRenderSize( b: Fsr3Bindings ): N {

	return ivec2( vec2( b.PreviousFrameRenderSize() ).mul( 0.5 ) );

}

/** (defined upstream when the pass binds FRAME_INFO as an SRV) */
export function SceneAverageLuma( b: Fsr3Bindings ): N {

	return b.FrameInfo().element( FRAME_INFO_SCENE_AVERAGE_LUMA );

}

// Auto exposure
export const resetAutoExposureAverageSmoothing = 1e4;

/** AccumulationPassCommonParams: the accumulation's per-pixel inputs. */
export interface AccumulationPassCommonParams {
	iPxHrPos: N;
	fHrUv: N;
	fLrUvJittered: N;
	fLrUv_HwSampler: N;
	fMotionVector: N;
	fReprojectedHrUv: N;
	f4KVelocity: N;
	fDisocclusion: N;
	fReactiveMask: N;
	fShadingChange: N;
	fAccumulation: N;
	fLumaInstabilityFactor: N;
	fFarthestDepthInMeters: N;
	bIsExistingSample: N;
	bIsNewSample: N;
}

export const Get4KVelocity = /*@__PURE__*/ Fn( ( [ fMotionVector ]: N[] ) => {

	return length( fMotionVector.mul( vec2( 3840.0, 2160.0 ) ) );

} ).setLayout( { name: 'Get4KVelocity', type: 'float', inputs: [ { name: 'fMotionVector', type: 'vec2' } ] } );

/** RectificationBox: TSL variables, mutated in place. */
export interface RectificationBox {
	boxCenter: N;
	boxVec: N;
	aabbMin: N;
	aabbMax: N;
	fBoxCenterWeight: N;
}

export function newRectificationBox(): RectificationBox {

	return {
		boxCenter: vec3( 0 ).toVar( 'boxCenter' ),
		boxVec: vec3( 0 ).toVar( 'boxVec' ),
		aabbMin: vec3( 0 ).toVar( 'aabbMin' ),
		aabbMax: vec3( 0 ).toVar( 'aabbMax' ),
		fBoxCenterWeight: float( 0 ).toVar( 'fBoxCenterWeight' ),
	};

}

/** AccumulationPassData: TSL variables, mutated in place. */
export interface AccumulationPassData {
	clippingBox: RectificationBox;
	fUpsampledColor: N;
	fUpsampledWeight: N;
	fHistoryColor: N;
	fHistoryWeight: N;
	fLock: N;
	fLockContributionThisFrame: N;
}

export function RectificationBoxAddInitialSample( rectificationBox: RectificationBox, colorSample: N, fSampleWeight: N ): void {

	rectificationBox.aabbMin.assign( colorSample );
	rectificationBox.aabbMax.assign( colorSample );
	const weightedSample = colorSample.mul( fSampleWeight ).toVar();
	rectificationBox.boxCenter.assign( weightedSample );
	rectificationBox.boxVec.assign( colorSample.mul( weightedSample ) );
	rectificationBox.fBoxCenterWeight.assign( fSampleWeight );

}

/** `bInitialSample` is a build-time boolean at upstream's one call site ( row == 0 && col == 0,
 * unrolled ). */
export function RectificationBoxAddSample( bInitialSample: boolean, rectificationBox: RectificationBox, colorSample: N, fSampleWeight: N ): void {

	if ( bInitialSample ) {

		RectificationBoxAddInitialSample( rectificationBox, colorSample, fSampleWeight );

	} else {

		rectificationBox.aabbMin.assign( min( rectificationBox.aabbMin, colorSample ) );
		rectificationBox.aabbMax.assign( max( rectificationBox.aabbMax, colorSample ) );
		const weightedSample = colorSample.mul( fSampleWeight ).toVar();
		rectificationBox.boxCenter.addAssign( weightedSample );
		rectificationBox.boxVec.addAssign( colorSample.mul( weightedSample ) );
		rectificationBox.fBoxCenterWeight.addAssign( fSampleWeight );

	}

}

export function RectificationBoxComputeVarianceBoxData( rectificationBox: RectificationBox ): void {

	rectificationBox.fBoxCenterWeight.assign( select( abs( rectificationBox.fBoxCenterWeight ).greaterThan( float( FSR3UPSCALER_FP32_MIN ) ), rectificationBox.fBoxCenterWeight, float( 1.0 ) ) );
	rectificationBox.boxCenter.divAssign( rectificationBox.fBoxCenterWeight );
	rectificationBox.boxVec.divAssign( rectificationBox.fBoxCenterWeight );
	const stdDev = sqrt( abs( rectificationBox.boxVec.sub( rectificationBox.boxCenter.mul( rectificationBox.boxCenter ) ) ) );
	rectificationBox.boxVec.assign( stdDev );

}

export const SafeRcp3 = /*@__PURE__*/ Fn( ( [ v ]: N[] ) => {

	return select( allNotEqualZero3( v ), vec3( 1, 1, 1 ).div( v ), vec3( 0, 0, 0 ) );

} ).setLayout( { name: 'SafeRcp3', type: 'vec3', inputs: [ { name: 'v', type: 'vec3' } ] } );

const MinDividedByMax3 = /*@__PURE__*/ Fn( ( [ v0, v1, fOnZeroReturnValue ]: N[] ) => {

	const m = max( v0, v1 ).toVar();
	return select( m.notEqual( 0 ), min( v0, v1 ).div( m ), fOnZeroReturnValue );

} ).setLayout( { name: 'MinDividedByMax3', type: 'float', inputs: [ { name: 'v0', type: 'float' }, { name: 'v1', type: 'float' }, { name: 'fOnZeroReturnValue', type: 'float' } ] } );

const MinDividedByMax2 = /*@__PURE__*/ Fn( ( [ v0, v1 ]: N[] ) => {

	const m = max( v0, v1 ).toVar();
	return select( m.notEqual( 0 ), min( v0, v1 ).div( m ), float( 0 ) );

} ).setLayout( { name: 'MinDividedByMax', type: 'float', inputs: [ { name: 'v0', type: 'float' }, { name: 'v1', type: 'float' } ] } );

/** MinDividedByMax( v0, v1 [, fOnZeroReturnValue ] ): upstream's two overloads. */
export function MinDividedByMax( v0: N, v1: N, fOnZeroReturnValue?: N ): N {

	return fOnZeroReturnValue === undefined ? MinDividedByMax2( v0, v1 ) : MinDividedByMax3( v0, v1, fOnZeroReturnValue );

}

export const YCoCgToRGB = /*@__PURE__*/ Fn( ( [ fYCoCg ]: N[] ) => {

	return vec3(
		fYCoCg.x.add( fYCoCg.y ).sub( fYCoCg.z ),
		fYCoCg.x.add( fYCoCg.z ),
		fYCoCg.x.sub( fYCoCg.y ).sub( fYCoCg.z ) );

} ).setLayout( { name: 'YCoCgToRGB', type: 'vec3', inputs: [ { name: 'fYCoCg', type: 'vec3' } ] } );

export const RGBToYCoCg = /*@__PURE__*/ Fn( ( [ fRgb ]: N[] ) => {

	return vec3(
		fRgb.r.mul( 0.25 ).add( fRgb.g.mul( 0.5 ) ).add( fRgb.b.mul( 0.25 ) ),
		fRgb.r.mul( 0.5 ).sub( fRgb.b.mul( 0.5 ) ),
		fRgb.r.mul( - 0.25 ).add( fRgb.g.mul( 0.5 ) ).sub( fRgb.b.mul( 0.25 ) ) );

} ).setLayout( { name: 'RGBToYCoCg', type: 'vec3', inputs: [ { name: 'fRgb', type: 'vec3' } ] } );

export const RGBToLuma = /*@__PURE__*/ Fn( ( [ fLinearRgb ]: N[] ) => {

	return dot( fLinearRgb, vec3( 0.2126, 0.7152, 0.0722 ) );

} ).setLayout( { name: 'RGBToLuma', type: 'float', inputs: [ { name: 'fLinearRgb', type: 'vec3' } ] } );

export const RGBToPerceivedLuma = /*@__PURE__*/ Fn( ( [ fLinearRgb ]: N[] ) => {

	const fLuminance = RGBToLuma( fLinearRgb ).toVar();
	const fPercievedLuminance = float( 0 ).toVar();
	If( fLuminance.lessThanEqual( 216.0 / 24389.0 ), () => {

		fPercievedLuminance.assign( fLuminance.mul( 24389.0 / 27.0 ) );

	} ).Else( () => {

		fPercievedLuminance.assign( ffxPow( fLuminance, float( 1.0 / 3.0 ) ).mul( 116.0 ).sub( 16.0 ) );

	} );
	return fPercievedLuminance.mul( 0.01 );

} ).setLayout( { name: 'RGBToPerceivedLuma', type: 'float', inputs: [ { name: 'fLinearRgb', type: 'vec3' } ] } );

export const Tonemap = /*@__PURE__*/ Fn( ( [ fRgb ]: N[] ) => {

	return fRgb.div( vec3( max( max( float( 0.0 ), fRgb.r ), max( fRgb.g, fRgb.b ) ).add( 1.0 ) ) );

} ).setLayout( { name: 'Tonemap', type: 'vec3', inputs: [ { name: 'fRgb', type: 'vec3' } ] } );

export const InverseTonemap = /*@__PURE__*/ Fn( ( [ fRgb ]: N[] ) => {

	return fRgb.div( vec3( max( float( FSR3UPSCALER_TONEMAP_EPSILON ), float( 1.0 ).sub( max( fRgb.r, max( fRgb.g, fRgb.b ) ) ) ) ) );

} ).setLayout( { name: 'InverseTonemap', type: 'vec3', inputs: [ { name: 'fRgb', type: 'vec3' } ] } );

export const IsUvInside = /*@__PURE__*/ Fn( ( [ fUv ]: N[] ) => {

	return fUv.x.greaterThanEqual( 0.0 ).and( fUv.x.lessThanEqual( 1.0 ) ).and( fUv.y.greaterThanEqual( 0.0 ).and( fUv.y.lessThanEqual( 1.0 ) ) );

} ).setLayout( { name: 'IsUvInside', type: 'bool', inputs: [ { name: 'fUv', type: 'vec2' } ] } );

export const ClampLoad = /*@__PURE__*/ Fn( ( [ iPxSample, iPxOffset, iTextureSize ]: N[] ) => {

	const result = iPxSample.add( iPxOffset ).toVar();
	result.x.assign( max( int( 0 ), min( result.x, iTextureSize.x.sub( 1 ) ) ) );
	result.y.assign( max( int( 0 ), min( result.y, iTextureSize.y.sub( 1 ) ) ) );
	return result;

} ).setLayout( { name: 'ClampLoad', type: 'ivec2', inputs: [ { name: 'iPxSample', type: 'ivec2' }, { name: 'iPxOffset', type: 'ivec2' }, { name: 'iTextureSize', type: 'ivec2' } ] } );

export const ClampUv = /*@__PURE__*/ Fn( ( [ fUv, iTextureSize, iResourceSize ]: N[] ) => {

	const fSampleLocation = fUv.mul( vec2( iTextureSize ) ).toVar();
	const fClampedLocation = max( vec2( 0.5, 0.5 ), min( fSampleLocation, vec2( iTextureSize ).sub( vec2( 0.5, 0.5 ) ) ) ).toVar();
	const fClampedUv = fClampedLocation.div( vec2( iResourceSize ) );
	return fClampedUv;

} ).setLayout( { name: 'ClampUv', type: 'vec2', inputs: [ { name: 'fUv', type: 'vec2' }, { name: 'iTextureSize', type: 'ivec2' }, { name: 'iResourceSize', type: 'ivec2' } ] } );

export const IsOnScreen = /*@__PURE__*/ Fn( ( [ pos, size ]: N[] ) => {

	const p = uvec2( pos ).toVar(), s = uvec2( size ).toVar();
	return p.x.lessThan( s.x ).and( p.y.lessThan( s.y ) );

} ).setLayout( { name: 'IsOnScreen', type: 'bool', inputs: [ { name: 'pos', type: 'ivec2' }, { name: 'size', type: 'ivec2' } ] } );

export const ComputeAutoExposureFromLavg = /*@__PURE__*/ Fn( ( [ LavgIn ]: N[] ) => {

	const Lavg = exp( LavgIn ).toVar();

	const S = 100.0; //ISO arithmetic speed
	const K = 12.5;
	const ExposureISO100 = log2( Lavg.mul( S ).div( K ) ).toVar();

	const q = 0.65;
	const Lmax = float( 78.0 / ( q * S ) ).mul( ffxPow( float( 2.0 ), ExposureISO100 ) ).toVar();

	return float( 1.0 ).div( Lmax );

} ).setLayout( { name: 'ComputeAutoExposureFromLavg', type: 'float', inputs: [ { name: 'Lavg', type: 'float' } ] } );

export function ComputeHrPosFromLrPos( b: Fsr3Bindings, iPxLrPos: N ): N {

	const fSrcJitteredPos = vec2( iPxLrPos ).add( 0.5 ).sub( b.Jitter() );
	const fLrPosInHr = fSrcJitteredPos.div( vec2( b.RenderSize() ) ).mul( vec2( b.UpscaleSize() ) );
	const iPxHrPos = ivec2( floor( fLrPosInHr ) );
	return iPxHrPos;

}

export const ComputeNdc = /*@__PURE__*/ Fn( ( [ fPxPos, iSize ]: N[] ) => {

	return fPxPos.div( vec2( iSize ) ).mul( vec2( 2.0, - 2.0 ) ).add( vec2( - 1.0, 1.0 ) );

} ).setLayout( { name: 'ComputeNdc', type: 'vec2', inputs: [ { name: 'fPxPos', type: 'vec2' }, { name: 'iSize', type: 'ivec2' } ] } );

export function GetViewSpaceDepth( b: Fsr3Bindings, fDeviceDepth: N ): N {

	const fDeviceToViewDepth = b.DeviceToViewSpaceTransformFactors();

	// fDeviceToViewDepth details found in ffx_fsr3upscaler.cpp
	return fDeviceToViewDepth.y.div( float( fDeviceDepth ).sub( fDeviceToViewDepth.x ) );

}

export function GetViewSpaceDepthInMeters( b: Fsr3Bindings, fDeviceDepth: N ): N {

	return GetViewSpaceDepth( b, fDeviceDepth ).mul( b.ViewSpaceToMetersFactor() );

}

export function GetViewSpacePosition( b: Fsr3Bindings, iViewportPos: N, iViewportSize: N, fDeviceDepth: N ): N {

	const fDeviceToViewDepth = b.DeviceToViewSpaceTransformFactors();

	const Z = GetViewSpaceDepth( b, fDeviceDepth ).toVar();

	const fNdcPos = ComputeNdc( vec2( iViewportPos ), iViewportSize ).toVar();
	const X = fDeviceToViewDepth.z.mul( fNdcPos.x ).mul( Z );
	const Y = fDeviceToViewDepth.w.mul( fNdcPos.y ).mul( Z );

	return vec3( X, Y, Z );

}

export function GetViewSpacePositionInMeters( b: Fsr3Bindings, iViewportPos: N, iViewportSize: N, fDeviceDepth: N ): N {

	return GetViewSpacePosition( b, iViewportPos, iViewportSize, fDeviceDepth ).mul( b.ViewSpaceToMetersFactor() );

}

export function GetMaxDistanceInMeters( b: Fsr3Bindings, o: Fsr3ShaderOptions ): N {

	if ( o.FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH ) {

		return GetViewSpaceDepth( b, float( 0.0 ) ).mul( b.ViewSpaceToMetersFactor() );

	}
	return GetViewSpaceDepth( b, float( 1.0 ) ).mul( b.ViewSpaceToMetersFactor() );

}

/** BilinearSamplingData: the four offsets are constants; the weights and base are variables. */
export interface BilinearSamplingData {
	iOffsets: N[];
	fWeights: N[];
	iBasePos: N;
}

export function GetBilinearSamplingData( fUv: N, iSize: N ): BilinearSamplingData {

	const fPxSample = fUv.mul( vec2( iSize ) ).sub( vec2( 0.5, 0.5 ) ).toVar();
	const iBasePos = ivec2( floor( fPxSample ) ).toVar();
	const fPxFrac = ffxFract( fPxSample ).toVar();

	const iOffsets = [ ivec2( 0, 0 ), ivec2( 1, 0 ), ivec2( 0, 1 ), ivec2( 1, 1 ) ];

	const fWeights = [
		float( 1 ).sub( fPxFrac.x ).mul( float( 1 ).sub( fPxFrac.y ) ).toVar(),
		fPxFrac.x.mul( float( 1 ).sub( fPxFrac.y ) ).toVar(),
		float( 1 ).sub( fPxFrac.x ).mul( fPxFrac.y ).toVar(),
		fPxFrac.x.mul( fPxFrac.y ).toVar(),
	];

	return { iOffsets, fWeights, iBasePos };

}

export interface PlaneData {
	fNormal: N;
	fDistanceFromOrigin: N;
}

export function GetPlaneFromPoints( fP0: N, fP1: N, fP2: N ): PlaneData {

	const v0 = fP0.sub( fP1 );
	const v1 = fP0.sub( fP2 );
	const fNormal = normalize( cross( v0, v1 ) ).toVar();
	const fDistanceFromOrigin = dot( fP0, fNormal ).negate().toVar();
	return { fNormal, fDistanceFromOrigin };

}

export function PointToPlaneDistance( plane: PlaneData, fPoint: N ): N {

	return abs( dot( plane.fNormal, fPoint ).add( plane.fDistanceFromOrigin ) );

}
