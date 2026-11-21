/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_rcas.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { ivec2, uvec2, vec4 } from './Tsl.ts';
import { ffxRemapForQuad, type N } from './Core.ts';
import { FsrRcasF } from './Fsr1Rcas.ts';
import type { Fsr3Bindings } from './Bindings.ts';

// FSR 3's sharpening pass: RCAS on the accumulated colour, after the accumulation.
//
// The sharpening pass after the accumulation: RCAS (FSR_RCAS_F with FSR_RCAS_DENOISE) on the
// accumulated colour, each thread filtering four pixels of a 16 x 16 tile in a quad swizzle.
// RCAS works on exposed colour: the input is multiplied by Exposure() when loaded and the result
// divided by it, which is all of upstream's HDR handling here (FSR 3.1.5's RCAS has no
// HDR_COLOR_INPUT branch; its limits assume the 0..1 range the exposure brings HDR colour near).
//
// Differences from upstream: the paired 16-bit path (FSR_RCAS_PREFER_PAIRED_VERSION, Xbox) is not
// ported. GROUP_SIZE is defined upstream and unused.

function WriteUpscaledOutput( b: Fsr3Bindings, iPxHrPos: N, fUpscaledColor: N ): void {

	b.StoreUpscaledOutput( ivec2( iPxHrPos ), fUpscaledColor );

}

function FsrRcasLoadF( b: Fsr3Bindings, p: N ): N {

	const fColor = vec4( b.LoadRCAS_Input( p ) ).toVar();
	fColor.rgb.mulAssign( b.Exposure() );
	return fColor;

}

function FsrRcasInputF( c: N ): N {

	return c;

}

function CurrFilter( b: Fsr3Bindings, pos: N ): void {

	const c = FsrRcasF( ivec2( pos ), b.RCASConfig(), ( p ) => FsrRcasLoadF( b, p ), FsrRcasInputF ).toVar();

	c.divAssign( b.Exposure() );

	WriteUpscaledOutput( b, pos, c );

}

export function RCAS( b: Fsr3Bindings, LocalThreadId: N, WorkGroupId: N ): void {

	// Do remapping of local xy in workgroup for a more PS-like swizzle pattern.
	const gxy = ffxRemapForQuad( LocalThreadId.x ).add( uvec2( WorkGroupId.x.shiftLeft( 4 ), WorkGroupId.y.shiftLeft( 4 ) ) ).toVar();

	CurrFilter( b, gxy );
	gxy.x.addAssign( 8 );
	CurrFilter( b, gxy );
	gxy.y.addAssign( 8 );
	CurrFilter( b, gxy );
	gxy.x.subAssign( 8 );
	CurrFilter( b, gxy );

}
