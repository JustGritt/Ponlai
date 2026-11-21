/*!
 * Part of the port of the AMD FidelityFX SDK 2.3.0's FSR 3 upscaler 3.1.5.
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import * as tsl from 'three/tsl';

// The TSL functions the FSR 3 port calls, loosely typed for HLSL's implicit conversions.
//
// The TSL functions the port calls, loosely typed: HLSL mixes int, uint and float vectors with
// implicit conversions that TSL's typings cannot follow, and the WGSL compiler checks what the
// port builds. The port's modules import TSL from here rather than from 'three/tsl'.

const t = tsl as any;

export const {
	Fn, If, Loop, Return, float, int, uint, bool, vec2, vec3, vec4, ivec2, ivec3, ivec4, uvec2, uvec3, uvec4,
	uniform, texture, textureLoad, storageTexture, textureStore, textureSize, storage, workgroupArray, workgroupBarrier,
	workgroupId, localId, globalId, atomicMin, atomicMax, atomicStore,
	abs, min, max, mix, clamp, saturate, floor, ceil, round, fract, sqrt, exp, exp2, log, log2, pow, sin, cos, sign,
	dot, length, normalize, cross, select, all, any, floatBitsToUint, uintBitsToFloat,
} = t;
