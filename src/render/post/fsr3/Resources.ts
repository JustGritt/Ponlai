/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_resources.h, internal/ffx_fsr3upscaler.cpp's
 * getInternalResourceDescriptions and getSharedResourceDescriptions).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

// FSR 3's resource identifiers and the description of every internal resource.
//
// The upscaler's resource identifiers (upstream's numbers, so a binding table maps one to one) and
// the description of every internal resource: its size class, upstream's format, the WebGPU format
// the port uses, its mips, its ping-pong partner, and the aliasing and clear rules. The kernels
// never create textures: the integration creates these from `FSR3_INTERNAL_RESOURCES` and hands
// them to `Bindings.ts` by identifier.
//
// Formats. With the device feature `texture-formats-tier1` (`formatFor`), every resource upstream
// keeps in R8_UNORM, R16_FLOAT or R16G16_FLOAT takes that format (r8unorm, r16float, rg16float:
// storable, filterable, the same values as the wide formats below hold in their first channels,
// at a quarter or half of their bytes), except NEW_LOCKS, which stays r32float (read-write: as
// r8unorm under `texture-formats-tier2` the accumulation took 2.2 ms more on the high tier at the baseline). WebGPU
// core can store to rgba8unorm, rgba16float, r32float / r32uint / r32sint, rg32float, rgba32float
// and a few integer formats, and can filter rgba8unorm and rgba16float but not the 32-bit floats.
// So without the feature (`format`):
// - a single- or two-channel 16-bit float that is written by a kernel and sampled with the linear
//   sampler takes rgba16float: the same half precision as upstream's R16_FLOAT / R16G16_FLOAT
//   (stores round to half as upstream's do) and filterable;
// - an R8_UNORM / R8G8B8A8_UNORM one takes rgba8unorm (the same 8-bit quantisation), except
//   NEW_LOCKS, which the accumulation reads and writes in one pass: only r32float, r32uint and
//   r32sint are read-write storage formats, so it is r32float and its stores round to 8 bits by
//   hand (`Bindings.ts` `StoreNewLocks`) to keep upstream's values;
// - R32_FLOAT stays r32float (DILATED_DEPTH; its one linear sample, in the debug view, filters by
//   hand);
// - the reconstructed previous nearest depth (R32_UINT, written with `InterlockedMin` / `Max`)
//   is a storage buffer of u32 at the render-max size, row-major ( y * maxRenderWidth + x ):
//   WebGPU has no texture atomics;
// - FRAME_INFO (one RGBA32F texel the luminance pyramid's last workgroup reads and writes) is a
//   storage buffer of 4 floats: rgba32float is no read-write storage format;
// - the Lanczos LUT (R16_SNORM, 128 x 1) is an r32float texture holding the snorm values exactly
//   as the hardware converts them ( c / 32767 ), filtered by hand: there is no 16-bit snorm format
//   in WebGPU core and r32float does not filter;
// - SPD_ATOMIC_COUNT is gone (SPD's tail is a second dispatch: `Spd.ts`); the default reactivity
//   and default exposure resources are the accessors' defaults in `Bindings.ts` (0 and 0, which
//   `Exposure()` turns into 1).
// SPD_MIPS: upstream allocates the full chain of the half render-max size (mip count 0) and binds
// levels 0..5 as UAVs; nothing reads or writes a level above 5 (the pyramids' stores of levels 6+
// fall through `StorePyramid`), so the port allocates the 6 levels. three r186 binds one mip level
// of a StorageTexture for writing (`textureStore( ... ).setMipLevel( i )` binds a view of that
// level), so it stays one texture; the integration gives it 6 `mipmaps` entries (three then
// allocates 6 levels and generates none) and samples it with a mip filter.
// Aliasing: upstream aliases INTERMEDIATE_FP16x1 as FARTHEST_DEPTH (written by prepare inputs, read
// by the luminance pyramid) and as LUMA_INSTABILITY (written by luma instability, read by the
// accumulation), and discards the aliasable resources each frame (their contents undefined). WebGPU
// has no aliasing: one texture serves both names (`Host.ts` `getResourceAliases`), and a discard
// needs no work.

export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NULL = 0;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_OPAQUE_ONLY = 1;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_COLOR = 2;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS = 3;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_DEPTH = 4;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_EXPOSURE = 5;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK = 6;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK = 7;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH = 8;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS = 9;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_DEPTH = 10;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR = 11;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION = 12;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NEW_LOCKS = 13;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY = 14;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DEBUG_OUTPUT = 15;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LANCZOS_LUT = 16;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_ATOMIC_COUNT = 17;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT = 18;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RCAS_INPUT = 19;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_1 = 20;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_2 = 21;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_1 = 22;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_2 = 23;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_REACTIVITY = 24;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_TRANSPARENCY_AND_COMPOSITION = 25;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS = 26;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS = 27; // same as SPD_MIPS_LEVEL_0
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS_LEVEL_0 = 27;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS_LEVEL_1 = 28;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS_LEVEL_2 = 29;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS_LEVEL_3 = 30;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS_LEVEL_4 = 31;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS_LEVEL_5 = 32;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_EXPOSURE = 33;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FRAME_INFO = 34;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_AUTOREACTIVE = 35;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_AUTOCOMPOSITION_DEPRECATED = 36;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_1 = 37;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_2 = 38;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_1 = 40;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_2 = 41;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SHADING_CHANGE = 42;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH = 43;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH_MIP1 = 44;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_CURRENT_LUMA = 45;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_PREVIOUS_LUMA = 46;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_INSTABILITY = 48;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERMEDIATE_FP16x1 = 49;
export const FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_COUNT = 60;

export const FFX_FSR3UPSCALER_CONSTANTBUFFER_IDENTIFIER_FSR3UPSCALER = 0;
export const FFX_FSR3UPSCALER_CONSTANTBUFFER_IDENTIFIER_SPD = 1;
export const FFX_FSR3UPSCALER_CONSTANTBUFFER_IDENTIFIER_RCAS = 2;
export const FFX_FSR3UPSCALER_CONSTANTBUFFER_IDENTIFIER_GENREACTIVE = 3;
export const FFX_FSR3UPSCALER_CONSTANTBUFFER_COUNT = 4;

export const FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_TONEMAP = 1;
export const FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_INVERSETONEMAP = 2;
export const FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_THRESHOLD = 4;
export const FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_USE_COMPONENTS_MAX = 8;

/** The size a resource is allocated at: the render-max size (upstream's maxRenderSize, the canvas's
 * size in the integration), half of it (integer division), the upscale-max size, or fixed. */
export type Fsr3SizeClass = 'renderMax' | 'renderMaxHalf' | 'upscaleMax' | 'fixed';

/** The WebGPU format of a texture, or a storage buffer's element. */
export type Fsr3Format = 'r8unorm' | 'r16float' | 'rg16float' | 'rgba8unorm' | 'rgba16float' | 'r32float' | 'r32uint' | 'buffer-u32' | 'buffer-vec4f';

export interface Fsr3ResourceDescription {
	id: number;
	name: string;
	sizeClass: Fsr3SizeClass;
	/** for `fixed` */
	width?: number;
	height?: number;
	/** upstream's surface format */
	upstreamFormat: string;
	format: Fsr3Format;
	mips: number;
	/** the other side of a ping-pong pair */
	pingPong?: number;
	/** upstream marks it aliasable and discards it every frame (contents undefined) */
	aliasable: boolean;
	/** read-write in one pass (a read-write storage texture) */
	readWrite: boolean;
	/** sampled with the linear sampler somewhere (the integration sets LinearFilter, ClampToEdge) */
	linear: boolean;
	/** when upstream clears it (`Host.ts` lists the clear jobs per frame) */
	clears: string;
}

/** The format a resource takes on the device: upstream's own where `texture-formats-tier1` makes it
 * storable (see the header), else `format`. */
export function formatFor( d: Fsr3ResourceDescription, tier1: boolean ): Fsr3Format {

	if ( ! tier1 || d.readWrite ) return d.format;
	if ( d.upstreamFormat === 'R8_UNORM' ) return 'r8unorm';
	if ( d.upstreamFormat === 'R16_FLOAT' ) return 'r16float';
	if ( d.upstreamFormat.startsWith( 'R16G16_FLOAT' ) ) return 'rg16float';
	return d.format;

}

/** getInternalResourceDescriptions and getSharedResourceDescriptions, in upstream's order. */
export const FSR3_INTERNAL_RESOURCES: readonly Fsr3ResourceDescription[] = [
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_1, name: 'FSR3UPSCALER_Accumulation1', sizeClass: 'renderMax', upstreamFormat: 'R8_UNORM', format: 'rgba8unorm', mips: 1, pingPong: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_2, aliasable: false, readWrite: false, linear: true, clears: 'first execution; the SRV side on reset' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_2, name: 'FSR3UPSCALER_Accumulation2', sizeClass: 'renderMax', upstreamFormat: 'R8_UNORM', format: 'rgba8unorm', mips: 1, pingPong: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION_1, aliasable: false, readWrite: false, linear: true, clears: 'first execution; the SRV side on reset' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_1, name: 'FSR3UPSCALER_Luma1', sizeClass: 'renderMax', upstreamFormat: 'R16_FLOAT', format: 'rgba16float', mips: 1, pingPong: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_2, aliasable: false, readWrite: false, linear: true, clears: 'first execution' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_2, name: 'FSR3UPSCALER_Luma2', sizeClass: 'renderMax', upstreamFormat: 'R16_FLOAT', format: 'rgba16float', mips: 1, pingPong: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_1, aliasable: false, readWrite: false, linear: true, clears: 'first execution' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERMEDIATE_FP16x1, name: 'FSR3UPSCALER_IntermediateFp16x1', sizeClass: 'renderMax', upstreamFormat: 'R16_FLOAT', format: 'rgba16float', mips: 1, aliasable: true, readWrite: false, linear: true, clears: 'never (discarded each frame)' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SHADING_CHANGE, name: 'FSR3UPSCALER_ShadingChange', sizeClass: 'renderMaxHalf', upstreamFormat: 'R8_UNORM', format: 'rgba8unorm', mips: 1, aliasable: true, readWrite: false, linear: true, clears: 'never (discarded each frame)' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NEW_LOCKS, name: 'FSR3UPSCALER_NewLocks', sizeClass: 'upscaleMax', upstreamFormat: 'R8_UNORM', format: 'r32float', mips: 1, aliasable: true, readWrite: true, linear: false, clears: 'never (the accumulation zeroes each texel it reads)' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_1, name: 'FSR3UPSCALER_InternalUpscaled1', sizeClass: 'upscaleMax', upstreamFormat: 'R16G16B16A16_FLOAT', format: 'rgba16float', mips: 1, pingPong: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_2, aliasable: false, readWrite: false, linear: true, clears: 'never' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_2, name: 'FSR3UPSCALER_InternalUpscaled2', sizeClass: 'upscaleMax', upstreamFormat: 'R16G16B16A16_FLOAT', format: 'rgba16float', mips: 1, pingPong: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_1, aliasable: false, readWrite: false, linear: true, clears: 'never' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS, name: 'FSR3UPSCALER_SpdMips', sizeClass: 'renderMaxHalf', upstreamFormat: 'R16G16_FLOAT (full chain)', format: 'rgba16float', mips: 6, aliasable: true, readWrite: false, linear: true, clears: 'every frame (and on reset), all levels' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH_MIP1, name: 'FSR3UPSCALER_FarthestDepthMip1', sizeClass: 'renderMaxHalf', upstreamFormat: 'R16_FLOAT', format: 'rgba16float', mips: 1, aliasable: true, readWrite: false, linear: true, clears: 'never (discarded each frame)' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_1, name: 'FSR3UPSCALER_LumaHistory1', sizeClass: 'renderMax', upstreamFormat: 'R16G16B16A16_FLOAT', format: 'rgba16float', mips: 1, pingPong: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_2, aliasable: false, readWrite: false, linear: true, clears: 'never' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_2, name: 'FSR3UPSCALER_LumaHistory2', sizeClass: 'renderMax', upstreamFormat: 'R16G16B16A16_FLOAT', format: 'rgba16float', mips: 1, pingPong: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY_1, aliasable: false, readWrite: false, linear: true, clears: 'never' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS, name: 'FSR3UPSCALER_DilatedReactiveMasks', sizeClass: 'renderMax', upstreamFormat: 'R8G8B8A8_UNORM', format: 'rgba8unorm', mips: 1, aliasable: true, readWrite: false, linear: true, clears: 'never (discarded each frame)' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LANCZOS_LUT, name: 'FSR3UPSCALER_LanczosLutData', sizeClass: 'fixed', width: 128, height: 1, upstreamFormat: 'R16_SNORM (read only, initialised)', format: 'r32float', mips: 1, aliasable: false, readWrite: false, linear: false, clears: 'never (`createLanczos2LutData`)' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FRAME_INFO, name: 'FSR3UPSCALER_FrameInfo', sizeClass: 'fixed', width: 1, height: 1, upstreamFormat: 'R32G32B32A32_FLOAT', format: 'buffer-vec4f', mips: 1, aliasable: false, readWrite: true, linear: false, clears: 'on reset to ( -1, 1, 0, 0 )' },
	// shared resources
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_DEPTH, name: 'FSR3UPSCALER_DilatedDepth', sizeClass: 'renderMax', upstreamFormat: 'R32_FLOAT', format: 'r32float', mips: 1, aliasable: false, readWrite: false, linear: false, clears: 'never' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS, name: 'FSR3UPSCALER_DilatedVelocity', sizeClass: 'renderMax', upstreamFormat: 'R16G16_FLOAT', format: 'rgba16float', mips: 1, aliasable: false, readWrite: false, linear: true, clears: 'never' },
	{ id: FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH, name: 'FSR3UPSCALER_ReconstructedPrevNearestDepth', sizeClass: 'renderMax', upstreamFormat: 'R32_UINT', format: 'buffer-u32', mips: 1, aliasable: false, readWrite: true, linear: false, clears: 'every frame to the bits of 1.0 (0 when inverted)' },
];

/** A resource's size for the context's max render and upscale sizes. */
export function getResourceSize( desc: Fsr3ResourceDescription, maxRenderSize: { width: number, height: number }, maxUpscaleSize: { width: number, height: number } ): { width: number, height: number } {

	switch ( desc.sizeClass ) {

		case 'renderMax': return { width: maxRenderSize.width, height: maxRenderSize.height };
		case 'renderMaxHalf': return { width: Math.trunc( maxRenderSize.width / 2 ), height: Math.trunc( maxRenderSize.height / 2 ) };
		case 'upscaleMax': return { width: maxUpscaleSize.width, height: maxUpscaleSize.height };
		case 'fixed': return { width: desc.width!, height: desc.height! };

	}

}

/** getInternalResourceDescriptions's LUT: Lanczos 2 over 0..2 in 128 texels, rounded to 16-bit snorm,
 * then converted back as the hardware reads an snorm texel ( max( c / 32767, -1 ) ) into the
 * r32float the port stores. */
export function createLanczos2LutData( lanczos2: ( x: number ) => number ): Float32Array {

	const lanczos2LutWidth = 128;
	const data = new Float32Array( lanczos2LutWidth );
	for ( let currentLanczosWidthIndex = 0; currentLanczosWidthIndex < lanczos2LutWidth; currentLanczosWidthIndex ++ ) {

		const x = Math.fround( Math.fround( 2.0 * currentLanczosWidthIndex ) / Math.fround( lanczos2LutWidth - 1 ) );
		const y = lanczos2( x );
		// roundf rounds halves away from zero; int16_t keeps the sign
		const v = Math.fround( y * 32767.0 );
		const c = Math.sign( v ) * Math.round( Math.abs( v ) );
		data[ currentLanczosWidthIndex ] = Math.max( c / 32767, - 1 );

	}
	return data;

}
