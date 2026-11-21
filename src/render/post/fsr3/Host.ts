/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/internal/
 * ffx_fsr3upscaler.cpp, ffx_fsr3upscaler_private.h, include/ffx_fsr3upscaler.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { FsrRcasCon } from './Fsr1Rcas.ts';
import { ffxSpdSetup } from './Spd.ts';
import {
	FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_1,
	FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_2, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR,
	FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_1, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_2,
	FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_1,
	FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_2, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_1,
	FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_2, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_CURRENT_LUMA,
	FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_PREVIOUS_LUMA, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RCAS_INPUT,
	FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_INSTABILITY,
	FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERMEDIATE_FP16x1, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS,
	FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FRAME_INFO, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH,
} from './Resources.ts';

// FSR 3's host side in plain TypeScript: flags, quality modes, jitter and per-frame constants.
//
// The FSR 3 upscaler's host side, in plain TypeScript (no three: `node` runs it by stripping types
// for the host check against upstream): the context flags, the quality modes, the jitter, the
// per-frame constants `fsr3upscalerDispatch` computes before its dispatches, the device-depth
// parameters, the ping-pong parity, the clears and the dispatch grid. Every float is rounded to 32
// bits after each operation (`Math.fround`), as upstream's floats are, so the constants match
// upstream's to the last bit (measured against upstream's functions built natively: jitter phase
// counts and offsets, device-depth factors, RCAS and SPD constants, quality-mode sizes, the half
// conversion, all bit-identical).
//
// `Fsr3UpscalerContext.dispatch` returns what the C++ schedules instead of scheduling it: the
// constant buffers' values, the clear jobs in upstream's order and the passes with upstream's
// workgroup counts. The integration copies the constants into `Bindings.ts`'s uniforms, runs the
// clear kernels and the pass kernels of `Passes.ts` in that order, and dispatches each kernel
// over the canvas's fixed grid; the kernels skip the workgroups and threads outside the counts
// given here, which they read from the same constants.
//
// Differences from upstream:
// - The backend, resource registration, barriers, aliasing, shader-blob permutations,
//   `fsr3upscalerDebugCheckDispatch` and the watermark are not ported: the frame graph and TSL
//   take their place. `getPipelinePermutationFlags` is ported to say which options each pass is
//   built with.
// - The SPD atomic counter and its two clears are gone: the luminance and shading-change pyramids'
//   last workgroup tail is a second dispatch (`Spd.ts`), so nothing counts workgroups.
// - Upstream picks the Lanczos LUT for the history reprojection on GPUs whose wave size ranges
//   32 to 64 (AMD's), the reference Lanczos otherwise; there are no wave sizes in WebGPU, so the
//   choice is an option (`useLut`, default false: the reference, what other vendors get).
// - The reconstructed previous depth is cleared to the bits of 1.0 (0 when inverted). Upstream
//   schedules a float clear of 1.0 on its R32_UINT target; the backends clear that resource with the
//   raw bits of the colour union (Vulkan reads the float's bits as the uint), the far depth that
//   `InterlockedMin` over `asuint( depth )` needs, so the port clears to those bits directly.
// - `FFX_EPSILON` and `FFX_PI` (used only by the Lanczos LUT's generator) come from the SDK's
//   ffx_util.h, which the download leaves out: 1e-6 and 3.141592653589793.

// ffx_fsr3upscaler.h: FfxFsr3UpscalerInitializationFlagBits
export const FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE = 1 << 0;
export const FFX_FSR3UPSCALER_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS = 1 << 1;
export const FFX_FSR3UPSCALER_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION = 1 << 2;
export const FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED = 1 << 3;
export const FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE = 1 << 4;
export const FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE = 1 << 5;
export const FFX_FSR3UPSCALER_ENABLE_DYNAMIC_RESOLUTION = 1 << 6;
export const FFX_FSR3UPSCALER_ENABLE_TEXTURE1D_USAGE = 1 << 7; // deprecated upstream
export const FFX_FSR3UPSCALER_ENABLE_DEBUG_CHECKING = 1 << 8;

// FfxFsr3UpscalerDispatchFlags
export const FFX_FSR3UPSCALER_DISPATCH_DRAW_DEBUG_VIEW = 1 << 0;

// FfxFsr3UpscalerPass
export const FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS = 0;
export const FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID = 1;
export const FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID = 2;
export const FFX_FSR3UPSCALER_PASS_SHADING_CHANGE = 3;
export const FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY = 4;
export const FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY = 5;
export const FFX_FSR3UPSCALER_PASS_ACCUMULATE = 6;
export const FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN = 7;
export const FFX_FSR3UPSCALER_PASS_RCAS = 8;
export const FFX_FSR3UPSCALER_PASS_DEBUG_VIEW = 9;
export const FFX_FSR3UPSCALER_PASS_GENERATE_REACTIVE = 10;
export const FFX_FSR3UPSCALER_PASS_TCR_AUTOGENERATE = 11; // deprecated upstream
export const FFX_FSR3UPSCALER_PASS_COUNT = 12;

// ffx_upscale.h: FfxApiUpscaleQualityMode
export const FFX_UPSCALE_QUALITY_MODE_NATIVEAA = 0;
export const FFX_UPSCALE_QUALITY_MODE_QUALITY = 1;
export const FFX_UPSCALE_QUALITY_MODE_BALANCED = 2;
export const FFX_UPSCALE_QUALITY_MODE_PERFORMANCE = 3;
export const FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE = 4;

// ffx_fsr3upscaler_private.h: Fs3UpscalerShaderPermutationOptions
export const FSR3UPSCALER_SHADER_PERMUTATION_USE_LANCZOS_TYPE = 1 << 0;
export const FSR3UPSCALER_SHADER_PERMUTATION_HDR_COLOR_INPUT = 1 << 1;
export const FSR3UPSCALER_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS = 1 << 2;
export const FSR3UPSCALER_SHADER_PERMUTATION_JITTER_MOTION_VECTORS = 1 << 3;
export const FSR3UPSCALER_SHADER_PERMUTATION_DEPTH_INVERTED = 1 << 4;
export const FSR3UPSCALER_SHADER_PERMUTATION_ENABLE_SHARPENING = 1 << 5;
export const FSR3UPSCALER_SHADER_PERMUTATION_FORCE_WAVE64 = 1 << 6;
export const FSR3UPSCALER_SHADER_PERMUTATION_ALLOW_FP16 = 1 << 7;

const FSR3UPSCALER_MAX_QUEUED_FRAMES = 16;
const FLT_EPSILON = 1.1920928955078125e-7; // 2^-23
const FFX_EPSILON = 1e-6;
const FFX_PI = 3.141592653589793;
const fr = Math.fround;

/** The `FFX_FSR3UPSCALER_OPTION_*` defines a pass is compiled with (ffx_fsr3upscaler_shaderblobs.cpp's
 * key from the permutation flags). The shader modules read these at build time. */
export interface Fsr3ShaderOptions {
	FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT: boolean;
	FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS: boolean;
	FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS: boolean;
	FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH: boolean;
	FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING: boolean;
	/** 0 the reference Lanczos, 1 the LUT (the permutation's USE_LANCZOS_TYPE bit). */
	FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE: 0 | 1;
	/** ffx_fsr3upscaler_upsample.h's default, 2 (the approximation); no permutation sets it. */
	FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE: 0 | 1 | 2;
}

/** ffx_fsr3upscaler.cpp: the permutation a pass is built with. `fp16` and `force64` are kept for
 * the signature; the port has no FP16 or wave64 variants and ignores those bits. */
export function getPipelinePermutationFlags( contextFlags: number, passId: number, fp16: boolean, force64: boolean, useLut: boolean ): number {

	let flags = 0;
	flags |= ( contextFlags & FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE ) ? FSR3UPSCALER_SHADER_PERMUTATION_HDR_COLOR_INPUT : 0;
	flags |= ( contextFlags & FFX_FSR3UPSCALER_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS ) ? 0 : FSR3UPSCALER_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS;
	flags |= ( contextFlags & FFX_FSR3UPSCALER_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION ) ? FSR3UPSCALER_SHADER_PERMUTATION_JITTER_MOTION_VECTORS : 0;
	flags |= ( contextFlags & FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED ) ? FSR3UPSCALER_SHADER_PERMUTATION_DEPTH_INVERTED : 0;
	flags |= ( passId === FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN ) ? FSR3UPSCALER_SHADER_PERMUTATION_ENABLE_SHARPENING : 0;
	flags |= ( useLut ) ? FSR3UPSCALER_SHADER_PERMUTATION_USE_LANCZOS_TYPE : 0;
	flags |= ( force64 ) ? FSR3UPSCALER_SHADER_PERMUTATION_FORCE_WAVE64 : 0;
	flags |= ( fp16 && ( passId !== FFX_FSR3UPSCALER_PASS_RCAS ) ) ? FSR3UPSCALER_SHADER_PERMUTATION_ALLOW_FP16 : 0;
	return flags;

}

/** The options a permutation's flags define (ffx_fsr3upscaler_shaderblobs.cpp). */
export function getShaderOptions( permutationFlags: number ): Fsr3ShaderOptions {

	return {
		FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE: ( permutationFlags & FSR3UPSCALER_SHADER_PERMUTATION_USE_LANCZOS_TYPE ) ? 1 : 0,
		FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT: ( permutationFlags & FSR3UPSCALER_SHADER_PERMUTATION_HDR_COLOR_INPUT ) !== 0,
		FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS: ( permutationFlags & FSR3UPSCALER_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS ) !== 0,
		FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS: ( permutationFlags & FSR3UPSCALER_SHADER_PERMUTATION_JITTER_MOTION_VECTORS ) !== 0,
		FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH: ( permutationFlags & FSR3UPSCALER_SHADER_PERMUTATION_DEPTH_INVERTED ) !== 0,
		FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING: ( permutationFlags & FSR3UPSCALER_SHADER_PERMUTATION_ENABLE_SHARPENING ) !== 0,
		FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE: 2,
	};

}

/** A pass's options for a context (no FP16, no wave64; `useLut` see the header). */
export function getPassOptions( contextFlags: number, passId: number, useLut = false ): Fsr3ShaderOptions {

	return getShaderOptions( getPipelinePermutationFlags( contextFlags, passId, false, false, useLut ) );

}

// ---------------------------------------------------------------------------------------------
// Lanczos and Halton

/** ffx_fsr3upscaler.cpp: the Lanczos 2 kernel (the LUT's generator). */
export function lanczos2( value: number ): number {

	if ( Math.abs( value ) < FFX_EPSILON ) return 1;
	// ( FFX_PI is a double: the products are doubles, sinf takes and returns floats )
	const a = fr( Math.sin( fr( FFX_PI * value ) ) ) / ( FFX_PI * value );
	const b = fr( Math.sin( fr( 0.5 * FFX_PI * value ) ) ) / ( 0.5 * FFX_PI * value );
	return fr( a * b );

}

/** ffx_fsr3upscaler.cpp: the Halton number of `index` in `base`. */
export function halton( index: number, base: number ): number {

	let f = 1.0, result = 0.0;
	for ( let currentIndex = index; currentIndex > 0; ) {

		f = fr( f / base );
		result = fr( result + fr( f * ( currentIndex % base ) ) );
		currentIndex = Math.floor( fr( fr( currentIndex ) / fr( base ) ) ) >>> 0;

	}
	return result;

}

/** ffx_fsr3upscaler.cpp: the jitter sequence's length for a render and display width. */
export function ffxFsr3UpscalerGetJitterPhaseCount( renderWidth: number, displayWidth: number ): number {

	const basePhaseCount = 8.0;
	const ratio = fr( fr( displayWidth ) / renderWidth );
	return Math.trunc( fr( basePhaseCount * fr( Math.pow( ratio, 2.0 ) ) ) ) | 0;

}

/** ffx_fsr3upscaler.cpp: the jitter offset of `index` in a sequence of `phaseCount`, in render
 * pixels ( -0.5 .. 0.5 ). */
export function ffxFsr3UpscalerGetJitterOffset( index: number, phaseCount: number ): { x: number, y: number } {

	if ( ! ( phaseCount > 0 ) ) throw new Error( 'FSR3: phaseCount must be positive' );
	const x = fr( halton( ( index % phaseCount ) + 1, 2 ) - 0.5 );
	const y = fr( halton( ( index % phaseCount ) + 1, 3 ) - 0.5 );
	return { x, y };

}

/** ffx_fsr3upscaler.cpp: the upscale ratio of a quality mode. */
export function ffxFsr3UpscalerGetUpscaleRatioFromQualityMode( qualityMode: number ): number {

	switch ( qualityMode ) {

		case FFX_UPSCALE_QUALITY_MODE_NATIVEAA: return 1.0;
		case FFX_UPSCALE_QUALITY_MODE_QUALITY: return 1.5;
		case FFX_UPSCALE_QUALITY_MODE_BALANCED: return fr( 1.7 );
		case FFX_UPSCALE_QUALITY_MODE_PERFORMANCE: return 2.0;
		case FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE: return 3.0;
		default: return 0.0;

	}

}

/** ffx_fsr3upscaler.cpp: the render size of a quality mode for a display size. */
export function ffxFsr3UpscalerGetRenderResolutionFromQualityMode( displayWidth: number, displayHeight: number, qualityMode: number ): { renderWidth: number, renderHeight: number } {

	if ( ! ( FFX_UPSCALE_QUALITY_MODE_NATIVEAA <= qualityMode && qualityMode <= FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE ) ) throw new Error( 'FSR3: invalid quality mode' );
	const ratio = ffxFsr3UpscalerGetUpscaleRatioFromQualityMode( qualityMode );
	const renderWidth = Math.trunc( fr( fr( displayWidth ) / ratio ) ) >>> 0;
	const renderHeight = Math.trunc( fr( fr( displayHeight ) / ratio ) ) >>> 0;
	return { renderWidth, renderHeight };

}

// ---------------------------------------------------------------------------------------------
// The context

/** ffx_fsr3upscaler_private.h: Fsr3UpscalerConstants (cbFSR3Upscaler), field for field. */
export interface Fsr3UpscalerConstants {
	renderSize: [ number, number ];
	previousFrameRenderSize: [ number, number ];
	upscaleSize: [ number, number ];
	previousFrameUpscaleSize: [ number, number ];
	maxRenderSize: [ number, number ];
	maxUpscaleSize: [ number, number ];
	deviceToViewDepth: [ number, number, number, number ];
	jitterOffset: [ number, number ];
	previousFrameJitterOffset: [ number, number ];
	motionVectorScale: [ number, number ];
	downscaleFactor: [ number, number ];
	motionVectorJitterCancellation: [ number, number ];
	tanHalfFOV: number;
	jitterPhaseCount: number;
	deltaTime: number;
	deltaPreExposure: number;
	viewSpaceToMetersFactor: number;
	frameIndex: number;
	velocityFactor: number;
	reactivenessScale: number;
	shadingChangeScale: number;
	accumulationAddedPerFrame: number;
	minDisocclusionAccumulation: number;
}

/** ffx_fsr3upscaler_private.h: Fsr3UpscalerSpdConstants (cbSPD). */
export interface Fsr3UpscalerSpdConstants {
	mips: number;
	numWorkGroups: number;
	workGroupOffset: [ number, number ];
	renderSize: [ number, number ];
}

/** cbRCAS: four uints. */
export interface Fsr3UpscalerRcasConstants {
	rcasConfig: [ number, number, number, number ];
}

/** cbGenerateReactive of the reactive-mask generator. */
export interface Fsr3UpscalerGenerateReactiveConstants {
	scale: number;
	threshold: number;
	binaryValue: number;
	flags: number;
}

/** ffx_fsr3upscaler.h: FfxFsr3UpscalerContextDescription (the backend left out). */
export interface FfxFsr3UpscalerContextDescription {
	flags: number;
	maxRenderSize: { width: number, height: number };
	maxUpscaleSize: { width: number, height: number };
}

/** ffx_fsr3upscaler.h: FfxFsr3UpscalerDispatchDescription, without the resources (the integration
 * binds those through `Bindings.ts`). Angles in radians, `frameTimeDelta` in milliseconds. */
export interface FfxFsr3UpscalerDispatchDescription {
	/** whether `exposure` is a resource; when false the internal default (exposure 1) is used */
	hasExposure?: boolean;
	jitterOffset: { x: number, y: number };
	motionVectorScale: { x: number, y: number };
	renderSize: { width: number, height: number };
	upscaleSize: { width: number, height: number };
	enableSharpening: boolean;
	sharpness: number;
	frameTimeDelta: number;
	preExposure: number;
	reset: boolean;
	cameraNear: number;
	cameraFar: number;
	cameraFovAngleVertical: number;
	viewSpaceToMetersFactor: number;
	flags: number;
}

/** ffx_fsr3upscaler.h: FfxFsr3UpscalerGenerateReactiveDescription, without the resources. */
export interface FfxFsr3UpscalerGenerateReactiveDescription {
	renderSize: { width: number, height: number };
	scale: number;
	cutoffThreshold: number;
	binaryValue: number;
	flags: number;
}

/** A clear job: `target` a resource identifier, the four values its texels take. */
export interface Fsr3ClearJob {
	label: string;
	target: number;
	color: [ number, number, number, number ];
}

/** A compute job: the pass and upstream's workgroup counts. */
export interface Fsr3ComputeJob {
	pass: number;
	dispatchX: number;
	dispatchY: number;
}

/** What one `fsr3upscalerDispatch` schedules, in upstream's order: the clears, then the passes. */
export interface Fsr3FramePlan {
	constants: Fsr3UpscalerConstants;
	spd: Fsr3UpscalerSpdConstants;
	rcas: Fsr3UpscalerRcasConstants;
	/** the resource frame index this dispatch ran with (its parity picks the ping-pong sides) */
	resourceFrameIndex: number;
	/** the aliases bound this frame: an alias identifier -> the concrete resource for SRVs / UAVs */
	srvAliases: Map<number, number>;
	uavAliases: Map<number, number>;
	clears: Fsr3ClearJob[];
	jobs: Fsr3ComputeJob[];
	/** the luminance pyramid's tail dispatch and the shading-change pyramid's (see `Spd.ts`) run
	 * after their pyramid when `spd.mips > 6`: one workgroup each */
	spdTail: boolean;
}

function newConstants(): Fsr3UpscalerConstants {

	return {
		renderSize: [ 0, 0 ], previousFrameRenderSize: [ 0, 0 ], upscaleSize: [ 0, 0 ], previousFrameUpscaleSize: [ 0, 0 ],
		maxRenderSize: [ 0, 0 ], maxUpscaleSize: [ 0, 0 ], deviceToViewDepth: [ 0, 0, 0, 0 ], jitterOffset: [ 0, 0 ],
		previousFrameJitterOffset: [ 0, 0 ], motionVectorScale: [ 0, 0 ], downscaleFactor: [ 0, 0 ],
		motionVectorJitterCancellation: [ 0, 0 ], tanHalfFOV: 0, jitterPhaseCount: 0, deltaTime: 0, deltaPreExposure: 0,
		viewSpaceToMetersFactor: 0, frameIndex: 0, velocityFactor: 0, reactivenessScale: 0, shadingChangeScale: 0,
		accumulationAddedPerFrame: 0, minDisocclusionAccumulation: 0,
	};

}

/** ffx_fsr3upscaler.cpp: the device-to-view-depth factors in `constants.deviceToViewDepth`, from the
 * camera's near / far / vertical fov and the context's inverted / infinite flags. */
export function setupDeviceDepthToViewSpaceDepthParams( contextFlags: number, constants: Fsr3UpscalerConstants, params: Pick<FfxFsr3UpscalerDispatchDescription, 'cameraNear' | 'cameraFar' | 'cameraFovAngleVertical' | 'renderSize'> ): void {

	const bInverted = ( contextFlags & FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED ) === FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED;
	const bInfinite = ( contextFlags & FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE ) === FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;

	// make sure it has no impact if near and far plane values are swapped in dispatch params
	// the flags "inverted" and "infinite" will decide what transform to use
	let fMin = fr( Math.min( params.cameraNear, params.cameraFar ) );
	let fMax = fr( Math.max( params.cameraNear, params.cameraFar ) );

	if ( bInverted ) {

		const tmp = fMin;
		fMin = fMax;
		fMax = tmp;

	}

	// a 0 0 0   x
	// 0 b 0 0   y
	// 0 0 c d   z
	// 0 0 e 0   1

	const fQ = fr( fMax / fr( fMin - fMax ) );
	const d = - 1.0; // for clarity

	const matrix_elem_c = [
		[ fQ, // non reversed, non infinite
			fr( - 1.0 - FLT_EPSILON ) ], // non reversed, infinite
		[ fQ, // reversed, non infinite
			fr( 0.0 + FLT_EPSILON ) ], // reversed, infinite
	];

	const matrix_elem_e = [
		[ fr( fQ * fMin ), // non reversed, non infinite
			fr( - fMin - FLT_EPSILON ) ], // non reversed, infinite
		[ fr( fQ * fMin ), // reversed, non infinite
			fMax ], // reversed, infinite
	];

	const i = bInverted ? 1 : 0, j = bInfinite ? 1 : 0;
	constants.deviceToViewDepth[ 0 ] = fr( d * matrix_elem_c[ i ][ j ] );
	constants.deviceToViewDepth[ 1 ] = matrix_elem_e[ i ][ j ];

	// revert x and y coords
	const aspect = fr( params.renderSize.width / fr( params.renderSize.height ) );
	const halfFov = fr( 0.5 * fr( params.cameraFovAngleVertical ) );
	const cotHalfFovY = fr( fr( Math.cos( halfFov ) ) / fr( Math.sin( halfFov ) ) );
	const a = fr( cotHalfFovY / aspect );
	const b = cotHalfFovY;

	constants.deviceToViewDepth[ 2 ] = fr( 1.0 / a );
	constants.deviceToViewDepth[ 3 ] = fr( 1.0 / b );

}

/** The resource aliases of a frame (fsr3upscalerDispatch's "per frame descriptor tables"): which
 * concrete resource each ping-pong or aliased identifier names for SRVs and for UAVs. */
export function getResourceAliases( resourceFrameIndex: number ): { srv: Map<number, number>, uav: Map<number, number> } {

	const isOddFrame = ( resourceFrameIndex & 1 ) !== 0;
	const accumulationSrvResourceIndex = isOddFrame ? FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_2 : FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_1;
	const accumulationUavResourceIndex = isOddFrame ? FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_1 : FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_2;
	const upscaledColorSrvResourceIndex = isOddFrame ? FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_2 : FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_1;
	const upscaledColorUavResourceIndex = isOddFrame ? FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_1 : FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_2;
	const lumaHistorySrvResourceIndex = isOddFrame ? FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_2 : FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_1;
	const lumaHistoryUavResourceIndex = isOddFrame ? FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_1 : FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_2;
	const currentLumaSrvResourceIndex = isOddFrame ? FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_2 : FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_1;
	const currentLumaUavResourceIndex = currentLumaSrvResourceIndex;
	const previousLumaSrvResourceIndex = isOddFrame ? FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_1 : FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_2;

	const srv = new Map<number, number>();
	const uav = new Map<number, number>();
	srv.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION, accumulationSrvResourceIndex );
	srv.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR, upscaledColorSrvResourceIndex );
	uav.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION, accumulationUavResourceIndex );
	uav.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR, upscaledColorUavResourceIndex );
	srv.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RCAS_INPUT, upscaledColorUavResourceIndex );
	srv.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_CURRENT_LUMA, currentLumaSrvResourceIndex );
	uav.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_CURRENT_LUMA, currentLumaUavResourceIndex );
	srv.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_PREVIOUS_LUMA, previousLumaSrvResourceIndex );
	uav.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY, lumaHistoryUavResourceIndex );
	srv.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY, lumaHistorySrvResourceIndex );
	srv.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERMEDIATE_FP16x1 );
	uav.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERMEDIATE_FP16x1 );
	srv.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_INSTABILITY, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERMEDIATE_FP16x1 );
	uav.set( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_INSTABILITY, FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERMEDIATE_FP16x1 );
	return { srv, uav };

}

/** ffx_fsr3upscaler.cpp's context, its host state and `fsr3upscalerDispatch` up to the dispatches. */
export class Fsr3UpscalerContext {

	readonly contextDescription: FfxFsr3UpscalerContextDescription;
	readonly constants: Fsr3UpscalerConstants;
	firstExecution: boolean;
	resourceFrameIndex: number;
	readonly previousJitterOffset: [ number, number ];
	preExposure: number;
	previousFramePreExposure: number;

	/** fsr3upscalerCreate's host part. */
	constructor( contextDescription: FfxFsr3UpscalerContextDescription ) {

		this.contextDescription = contextDescription;
		this.constants = newConstants();
		this.previousJitterOffset = [ 0, 0 ];
		this.preExposure = 0;
		this.previousFramePreExposure = 0;

		this.firstExecution = true;
		this.resourceFrameIndex = 0;

		this.constants.maxUpscaleSize[ 0 ] = contextDescription.maxUpscaleSize.width;
		this.constants.maxUpscaleSize[ 1 ] = contextDescription.maxUpscaleSize.height;
		this.constants.velocityFactor = 1.0;
		this.constants.reactivenessScale = 1.0;
		this.constants.shadingChangeScale = 1.0;
		this.constants.accumulationAddedPerFrame = fr( 1.0 / 3.0 );
		this.constants.minDisocclusionAccumulation = fr( - 1.0 / 3.0 );

	}

	/** ffxFsr3UpscalerContextConfigure's upscale keys (FfxFsr3UpscalerConfigureKey 0..4). */
	configure( key: number, value: number ): void {

		const v = fr( value );
		switch ( key ) {

			case 0: this.constants.velocityFactor = Math.min( 1, Math.max( 0, v ) ); break;
			case 1: this.constants.reactivenessScale = Math.max( 0, v ); break;
			case 2: this.constants.shadingChangeScale = Math.max( 0, v ); break;
			case 3: this.constants.accumulationAddedPerFrame = Math.min( 1, Math.max( 0, v ) ); break;
			case 4: this.constants.minDisocclusionAccumulation = Math.min( 1.0, Math.max( - 1.0, v ) ); break;

		}

	}

	/** fsr3upscalerDispatch: the constants, clears and dispatches of one frame. */
	dispatch( params: FfxFsr3UpscalerDispatchDescription ): Fsr3FramePlan {

		const context = this;
		const constants = this.constants;
		const flags = this.contextDescription.flags;
		const clears: Fsr3ClearJob[] = [];
		const jobs: Fsr3ComputeJob[] = [];

		if ( context.firstExecution ) {

			clears.push( { label: 'Clear Accumulation 1', target: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_1, color: [ 0, 0, 0, 0 ] } );
			clears.push( { label: 'Clear Accumulation 2', target: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_2, color: [ 0, 0, 0, 0 ] } );
			clears.push( { label: 'Clear Temporal Luma 1', target: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_1, color: [ 0, 0, 0, 0 ] } );
			clears.push( { label: 'Clear Temporal Luma 2', target: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_2, color: [ 0, 0, 0, 0 ] } );

		}

		// Prepare per frame descriptor tables
		const aliases = getResourceAliases( context.resourceFrameIndex );
		const accumulationSrvResourceIndex = aliases.srv.get( FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION )!;

		const resetAccumulation = params.reset || context.firstExecution;
		context.firstExecution = false;

		constants.previousFrameJitterOffset[ 0 ] = constants.jitterOffset[ 0 ];
		constants.previousFrameJitterOffset[ 1 ] = constants.jitterOffset[ 1 ];
		constants.jitterOffset[ 0 ] = fr( params.jitterOffset.x );
		constants.jitterOffset[ 1 ] = fr( params.jitterOffset.y );

		constants.previousFrameRenderSize[ 0 ] = constants.renderSize[ 0 ];
		constants.previousFrameRenderSize[ 1 ] = constants.renderSize[ 1 ];
		// ( upstream falls back to the colour resource's size when the render size is 0; the port
		// has no resource here and takes the render size as given )
		constants.renderSize[ 0 ] = params.renderSize.width | 0;
		constants.renderSize[ 1 ] = params.renderSize.height | 0;
		constants.maxRenderSize[ 0 ] = this.contextDescription.maxRenderSize.width | 0;
		constants.maxRenderSize[ 1 ] = this.contextDescription.maxRenderSize.height | 0;

		// compute the horizontal FOV for the shader from the vertical one.
		const aspectRatio = fr( fr( params.renderSize.width ) / fr( params.renderSize.height ) );
		const fov = fr( params.cameraFovAngleVertical );
		const cameraAngleHorizontal = fr( fr( Math.atan( fr( fr( Math.tan( fr( fov / 2 ) ) ) * aspectRatio ) ) ) * 2 );
		constants.tanHalfFOV = fr( Math.tan( fr( cameraAngleHorizontal * 0.5 ) ) );
		constants.viewSpaceToMetersFactor = ( params.viewSpaceToMetersFactor > 0.0 ) ? fr( params.viewSpaceToMetersFactor ) : 1.0;

		// compute params to enable device depth to view space depth computation in shader
		setupDeviceDepthToViewSpaceDepthParams( flags, constants, params );

		constants.previousFrameUpscaleSize[ 0 ] = constants.upscaleSize[ 0 ];
		constants.previousFrameUpscaleSize[ 1 ] = constants.upscaleSize[ 1 ];
		if ( params.upscaleSize.height === 0 && params.upscaleSize.width === 0 ) {

			constants.upscaleSize[ 0 ] = this.contextDescription.maxUpscaleSize.width;
			constants.upscaleSize[ 1 ] = this.contextDescription.maxUpscaleSize.height;

		} else {

			constants.upscaleSize[ 0 ] = params.upscaleSize.width;
			constants.upscaleSize[ 1 ] = params.upscaleSize.height;

		}

		// To be updated if resource is larger than the actual image size
		constants.downscaleFactor[ 0 ] = fr( fr( constants.renderSize[ 0 ] ) / fr( constants.upscaleSize[ 0 ] ) );
		constants.downscaleFactor[ 1 ] = fr( fr( constants.renderSize[ 1 ] ) / fr( constants.upscaleSize[ 1 ] ) );

		// calculate pre-exposure relevant factors
		constants.deltaPreExposure = 1.0;
		context.previousFramePreExposure = context.preExposure;
		context.preExposure = ( params.preExposure !== 0.0 ) ? fr( params.preExposure ) : 1.0;

		if ( context.previousFramePreExposure > 0.0 ) {

			constants.deltaPreExposure = fr( context.preExposure / context.previousFramePreExposure );

		}

		// motion vector data
		const motionVectorsTargetSize = ( flags & FFX_FSR3UPSCALER_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS ) ? constants.upscaleSize : constants.renderSize;

		constants.motionVectorScale[ 0 ] = fr( fr( params.motionVectorScale.x ) / fr( motionVectorsTargetSize[ 0 ] ) );
		constants.motionVectorScale[ 1 ] = fr( fr( params.motionVectorScale.y ) / fr( motionVectorsTargetSize[ 1 ] ) );

		// compute jitter cancellation
		if ( flags & FFX_FSR3UPSCALER_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION ) {

			constants.motionVectorJitterCancellation[ 0 ] = fr( fr( context.previousJitterOffset[ 0 ] - constants.jitterOffset[ 0 ] ) / fr( motionVectorsTargetSize[ 0 ] ) );
			constants.motionVectorJitterCancellation[ 1 ] = fr( fr( context.previousJitterOffset[ 1 ] - constants.jitterOffset[ 1 ] ) / fr( motionVectorsTargetSize[ 1 ] ) );

			context.previousJitterOffset[ 0 ] = constants.jitterOffset[ 0 ];
			context.previousJitterOffset[ 1 ] = constants.jitterOffset[ 1 ];

		}

		// lock data, assuming jitter sequence length computation for now
		const jitterPhaseCount = ffxFsr3UpscalerGetJitterPhaseCount( params.renderSize.width, constants.upscaleSize[ 0 ] );

		// init on first frame
		if ( resetAccumulation || constants.jitterPhaseCount === 0 ) {

			constants.jitterPhaseCount = jitterPhaseCount;

		} else {

			const jitterPhaseCountDelta = Math.trunc( jitterPhaseCount - constants.jitterPhaseCount );
			if ( jitterPhaseCountDelta > 0 ) {

				constants.jitterPhaseCount ++;

			} else if ( jitterPhaseCountDelta < 0 ) {

				constants.jitterPhaseCount --;

			}

		}

		// convert delta time to seconds and clamp to [0, 1].
		constants.deltaTime = Math.max( 0.0, Math.min( 1.0, fr( fr( params.frameTimeDelta ) / 1000.0 ) ) );

		if ( resetAccumulation ) {

			constants.frameIndex = 0.0;

		} else {

			constants.frameIndex = fr( constants.frameIndex + 1.0 );

		}

		// reactive mask bias
		const threadGroupWorkRegionDim = 8;
		const dispatchSrcX = Math.trunc( ( constants.renderSize[ 0 ] + ( threadGroupWorkRegionDim - 1 ) ) / threadGroupWorkRegionDim );
		const dispatchSrcY = Math.trunc( ( constants.renderSize[ 1 ] + ( threadGroupWorkRegionDim - 1 ) ) / threadGroupWorkRegionDim );
		const dispatchDstX = Math.trunc( ( constants.upscaleSize[ 0 ] + ( threadGroupWorkRegionDim - 1 ) ) / threadGroupWorkRegionDim );
		const dispatchDstY = Math.trunc( ( constants.upscaleSize[ 1 ] + ( threadGroupWorkRegionDim - 1 ) ) / threadGroupWorkRegionDim );
		const dispatchShadingChangePassX = Math.trunc( ( Math.trunc( fr( constants.renderSize[ 0 ] * 0.5 ) ) + ( threadGroupWorkRegionDim - 1 ) ) / threadGroupWorkRegionDim );
		const dispatchShadingChangePassY = Math.trunc( ( Math.trunc( fr( constants.renderSize[ 1 ] * 0.5 ) ) + ( threadGroupWorkRegionDim - 1 ) ) / threadGroupWorkRegionDim );

		// Clear reconstructed depth for max depth store.
		if ( resetAccumulation ) {

			clears.push( { label: 'Clear Resource', target: accumulationSrvResourceIndex, color: [ 0, 0, 0, 0 ] } );
			clears.push( { label: 'Clear Scene Luminance', target: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS, color: [ 0, 0, 0, 0 ] } );
			// Auto exposure always used to track luma changes in locking logic
			clears.push( { label: 'Clear Frame Info', target: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FRAME_INFO, color: [ - 1, 1, 0, 0 ] } );

		}

		{

			// FSR3: need to clear here since we need the content of this surface for frameinterpolation
			// so clearing in the lock pass is not an option
			const bInverted = ( flags & FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED ) === FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED;
			const v = bInverted ? 0 : 1;
			clears.push( { label: 'Clear Reconstructed Previous Nearest Depth', target: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH, color: [ v, v, v, v ] } );

		}

		// ( upstream's "Clear Spd Atomic Count": no counter, see the header )

		// Auto exposure
		const rectInfo: [ number, number, number, number ] = [ 0, 0, params.renderSize.width, params.renderSize.height ];
		const setup = ffxSpdSetup( rectInfo );

		// downsample
		const spd: Fsr3UpscalerSpdConstants = {
			numWorkGroups: setup.numWorkGroupsAndMips[ 0 ],
			mips: setup.numWorkGroupsAndMips[ 1 ],
			workGroupOffset: [ setup.workGroupOffset[ 0 ], setup.workGroupOffset[ 1 ] ],
			renderSize: [ params.renderSize.width, params.renderSize.height ],
		};

		// compute the constants.
		const sharpenessRemapped = fr( fr( - 2.0 * fr( params.sharpness ) ) + 2.0 );
		const rcas: Fsr3UpscalerRcasConstants = { rcasConfig: FsrRcasCon( sharpenessRemapped ) };

		// the aliased resources are discarded (their contents undefined, nothing to do here); SPD_MIPS
		// is cleared every frame "to prevent reading pixels that have never been written"
		clears.push( { label: 'Clear Spd Mips', target: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS, color: [ 0, 0, 0, 0 ] } );

		jobs.push( { pass: FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS, dispatchX: dispatchSrcX, dispatchY: dispatchSrcY } );
		jobs.push( { pass: FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID, dispatchX: setup.dispatchThreadGroupCountXY[ 0 ], dispatchY: setup.dispatchThreadGroupCountXY[ 1 ] } );
		jobs.push( { pass: FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID, dispatchX: setup.dispatchThreadGroupCountXY[ 0 ], dispatchY: setup.dispatchThreadGroupCountXY[ 1 ] } );
		jobs.push( { pass: FFX_FSR3UPSCALER_PASS_SHADING_CHANGE, dispatchX: dispatchShadingChangePassX, dispatchY: dispatchShadingChangePassY } );
		jobs.push( { pass: FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY, dispatchX: dispatchSrcX, dispatchY: dispatchSrcY } );
		jobs.push( { pass: FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY, dispatchX: dispatchSrcX, dispatchY: dispatchSrcY } );
		jobs.push( { pass: params.enableSharpening ? FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN : FFX_FSR3UPSCALER_PASS_ACCUMULATE, dispatchX: dispatchDstX, dispatchY: dispatchDstY } );

		// RCAS
		if ( params.enableSharpening ) {

			// dispatch RCAS
			const threadGroupWorkRegionDimRCAS = 16;
			const dispatchX = Math.trunc( ( constants.upscaleSize[ 0 ] + ( threadGroupWorkRegionDimRCAS - 1 ) ) / threadGroupWorkRegionDimRCAS );
			const dispatchY = Math.trunc( ( constants.upscaleSize[ 1 ] + ( threadGroupWorkRegionDimRCAS - 1 ) ) / threadGroupWorkRegionDimRCAS );
			jobs.push( { pass: FFX_FSR3UPSCALER_PASS_RCAS, dispatchX, dispatchY } );

		}

		if ( params.flags & FFX_FSR3UPSCALER_DISPATCH_DRAW_DEBUG_VIEW ) {

			jobs.push( { pass: FFX_FSR3UPSCALER_PASS_DEBUG_VIEW, dispatchX: dispatchDstX, dispatchY: dispatchDstY } );

		}

		const plan: Fsr3FramePlan = {
			constants: structuredClone( constants ), spd, rcas, resourceFrameIndex: context.resourceFrameIndex,
			srvAliases: aliases.srv, uavAliases: aliases.uav, clears, jobs, spdTail: spd.mips > 6,
		};

		context.resourceFrameIndex = ( context.resourceFrameIndex + 1 ) % FSR3UPSCALER_MAX_QUEUED_FRAMES;

		return plan;

	}

	/** ffxFsr3UpscalerContextGenerateReactiveMask: the generator's constants and workgroup counts. */
	generateReactiveMask( params: FfxFsr3UpscalerGenerateReactiveDescription ): { constants: Fsr3UpscalerGenerateReactiveConstants, dispatchX: number, dispatchY: number } {

		const threadGroupWorkRegionDim = 8;
		const dispatchSrcX = Math.trunc( ( params.renderSize.width + ( threadGroupWorkRegionDim - 1 ) ) / threadGroupWorkRegionDim );
		const dispatchSrcY = Math.trunc( ( params.renderSize.height + ( threadGroupWorkRegionDim - 1 ) ) / threadGroupWorkRegionDim );
		return {
			constants: { scale: fr( params.scale ), threshold: fr( params.cutoffThreshold ), binaryValue: fr( params.binaryValue ), flags: params.flags >>> 0 },
			dispatchX: dispatchSrcX, dispatchY: dispatchSrcY,
		};

	}

}
