/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (api/internal/gpu/ffx_core_hlsl.h,
 * ffx_core_gpu_common.h, ffx_core_cpu.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import {
	Fn, float, uint, min, max, mix, floor, pow, saturate, select, floatBitsToUint, uintBitsToFloat, uvec2,
} from './Tsl.ts';

// The ffx_core helpers the FSR 3 upscaler calls, in upstream's names and maths.
//
// The ffx_core helpers the FSR 3 upscaler's files call, in upstream's names, in the order the
// upscaler's headers first need them. Only what the port calls is here (ffx_core holds several
// hundred helpers); each keeps upstream's maths, and the approximations by bitcast are bit-exact
// (`ffxApproximateReciprocalMedium`'s magic constant, the bitfield remaps).
//
// Differences from upstream:
// - 32-bit only: the FFX_HALF and wave-operation helpers are not ported (the port runs SPD's
//   workgroup-memory path and FSR's 32-bit path).
// - HLSL's overloads by vector width become one helper per use: `ffxMin` / `ffxMax` / `ffxSaturate`
//   / `ffxLerp` / `ffxPow` are WGSL's `min` / `max` / `saturate` / `mix` / `pow`, which take any
//   width, so callers use those directly; `ffxFract` keeps upstream's definition (x - floor( x ))
//   rather than WGSL's `fract`, which is defined the same way.
// - `ffxReciprocal` is `rcp` in HLSL (a hardware approximation there, about 1 ulp); WGSL has no rcp,
//   so it is 1 / x, the exact value the approximation stands for.
// - The CPU helpers FsrRcasCon needs (`ffxAsUInt32`, `ffxF32ToF16`, `ffxPackHalf2x16`) are plain
//   TypeScript, `ffxF32ToF16`'s two 512-entry tables built by the rule they were generated from
//   (bit-identical to upstream's literal tables on every exponent).
//
// The port types its nodes loosely (`N`): HLSL's implicit int / float conversions have no
// counterpart in TSL's typings, and the shader compiler checks the types of what is built.

export type N = any;

// ---------------------------------------------------------------------------------------------
// GPU (ffx_core_hlsl.h, ffx_core_gpu_common.h)

/** ffx_core_hlsl.h: the bits of a float as an unsigned int. */
export const ffxAsUInt32 = ( x: N ): N => floatBitsToUint( x );

/** ffx_core_hlsl.h: an unsigned int's bits as a float. */
export const ffxAsFloat = ( x: N ): N => uintBitsToFloat( x );

export const ffxSaturate = ( x: N ): N => saturate( x );

export const ffxLerp = ( x: N, y: N, t: N ): N => mix( x, y, t );

/** ffx_core_hlsl.h: x - floor( x ). */
export const ffxFract = ( x: N ): N => x.sub( floor( x ) );

export const ffxPow = ( x: N, y: N ): N => pow( x, y );

/** ffx_core_hlsl.h: rcp( x ) (see the header). */
export const ffxReciprocal = ( x: N ): N => float( 1.0 ).div( x );

export const ffxMax3 = ( x: N, y: N, z: N ): N => max( x, max( y, z ) );

export const ffxMin3 = ( x: N, y: N, z: N ): N => min( x, min( y, z ) );

/** ffx_core_gpu_common.h: a medium-quality reciprocal from a bitcast estimate and one Newton step. */
export const ffxApproximateReciprocalMedium = /*@__PURE__*/ Fn( ( [ value ]: N[] ) => {

	const b = ffxAsFloat( uint( 0x7ef19fff ).sub( ffxAsUInt32( value ) ) ).toVar();
	return b.mul( b.negate().mul( value ).add( 2.0 ) );

} ).setLayout( { name: 'ffxApproximateReciprocalMedium', type: 'float', inputs: [ { name: 'value', type: 'float' } ] } );

/** ffx_core_hlsl.h: `bits` bits of `src` from bit `off`. */
export const ffxBitfieldExtract = ( src: N, off: number, bits: number ): N => {

	const mask = ( ( 1 << bits ) - 1 ) >>> 0;
	return src.shiftRight( uint( off ) ).bitAnd( uint( mask ) );

};

/** ffx_core_hlsl.h: the low `bits` bits of `ins` over `src`. */
export const ffxBitfieldInsertMask = ( src: N, ins: N, bits: number ): N => {

	const mask = ( ( 1 << bits ) - 1 ) >>> 0;
	return ins.bitAnd( uint( mask ) ).bitOr( src.bitAnd( uint( ~ mask >>> 0 ) ) );

};

/** ffx_core_gpu_common.h: remaps a 64x1 index to 8x8 in quads (RCAS's swizzle). */
export const ffxRemapForQuad = /*@__PURE__*/ Fn( ( [ a ]: N[] ) => {

	return uvec2( ffxBitfieldExtract( a, 1, 3 ), ffxBitfieldInsertMask( ffxBitfieldExtract( a, 3, 3 ), a, 1 ) );

} ).setLayout( { name: 'ffxRemapForQuad', type: 'uvec2', inputs: [ { name: 'a', type: 'uint' } ] } );

/** ffx_core_gpu_common.h: remaps a 64x1 index to 8x8 for 2D reductions (SPD's swizzle). */
export const ffxRemapForWaveReduction = /*@__PURE__*/ Fn( ( [ a ]: N[] ) => {

	return uvec2(
		ffxBitfieldInsertMask( ffxBitfieldExtract( a, 2, 3 ), a, 1 ),
		ffxBitfieldInsertMask( ffxBitfieldExtract( a, 3, 3 ), ffxBitfieldExtract( a, 1, 2 ), 2 ),
	);

} ).setLayout( { name: 'ffxRemapForWaveReduction', type: 'uvec2', inputs: [ { name: 'a', type: 'uint' } ] } );

/** HLSL's `all( a != b )` for a vec3 against zero, as SafeRcp3 uses it. */
export const allNotEqualZero3 = ( v: N ): N => v.x.notEqual( 0 ).and( v.y.notEqual( 0 ) ).and( v.z.notEqual( 0 ) );

/** HLSL's `sign`: -1, 0 or 1 (WGSL's `sign` is the same; spelled out where a select reads clearer). */
export const ffxSign = ( x: N ): N => select( x.greaterThan( 0 ), float( 1 ), select( x.lessThan( 0 ), float( - 1 ), float( 0 ) ) );

// ---------------------------------------------------------------------------------------------
// CPU (ffx_core_cpu.h)

const f32 = new Float32Array( 1 );
const u32 = new Uint32Array( f32.buffer );

/** ffx_core_cpu.h: the bits of a 32-bit float. */
export function ffxAsUInt32Cpu( x: number ): number {

	f32[ 0 ] = x;
	return u32[ 0 ];

}

/** ffx_core_cpu.h: a 32-bit float from its bits. */
export function ffxAsFloatCpu( x: number ): number {

	u32[ 0 ] = x >>> 0;
	return f32[ 0 ];

}

// ffxF32ToF16's tables (the fast conversion by J. van der Zijp, "Fast Half Float Conversions",
// which ffx_core_cpu.h cites), with upstream's two changes: every overflow, infinity and NaN
// clamps to the largest finite half (0x7bff / 0xfbff) rather than to infinity, shifted out
// entirely (shift 24).
const F16_BASE = new Uint16Array( 512 );
const F16_SHIFT = new Uint8Array( 512 );
for ( let i = 0; i < 256; i ++ ) {

	const e = i - 127;
	let base: number, shift: number;
	if ( e < - 24 ) {

		base = 0x0000; shift = 24;

	} else if ( e < - 14 ) {

		base = 0x0400 >> ( - e - 14 ); shift = - e - 1;

	} else if ( e <= 15 ) {

		base = ( e + 15 ) << 10; shift = 13;

	} else {

		base = 0x7bff; shift = 24;

	}
	F16_BASE[ i ] = base; F16_BASE[ i | 0x100 ] = base | 0x8000;
	F16_SHIFT[ i ] = shift; F16_SHIFT[ i | 0x100 ] = shift;

}

/** The conversion tables (exported to compare them with upstream's literal ones). */
export const ffxF32ToF16Tables = { base: F16_BASE, shift: F16_SHIFT };

/** ffx_core_cpu.h: the nearest half below a float, by table (denormals included). */
export function ffxF32ToF16( f: number ): number {

	const u = ffxAsUInt32Cpu( f );
	const i = u >>> 23;
	return ( F16_BASE[ i ] + ( ( u & 0x7fffff ) >>> F16_SHIFT[ i ] ) ) >>> 0;

}

/** ffx_core_cpu.h: two halves in one uint, x low. */
export function ffxPackHalf2x16( x: [ number, number ] ): number {

	return ( ffxF32ToF16( x[ 0 ] ) + ( ffxF32ToF16( x[ 1 ] ) << 16 ) ) >>> 0;

}

/** HLSL's `v[ i ]` on a vector with a constant index, as a swizzle. */
export const component = ( v: N, i: number ): N => v[ 'xyzw'[ i ] ];
