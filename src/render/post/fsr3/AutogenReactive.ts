/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/internal/shaders/
 * ffx_fsr3upscaler_autogen_reactive_pass.hlsl).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { float, uint, vec3, abs, max, length, select } from './Tsl.ts';
import type { N } from './Core.ts';
import { Tonemap, InverseTonemap } from './Common.ts';
import {
	FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_TONEMAP, FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_INVERSETONEMAP,
	FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_THRESHOLD, FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_USE_COMPONENTS_MAX,
} from './Resources.ts';
import type { Fsr3Bindings } from './Bindings.ts';

// FSR 3's reactive-mask generator: how much transparent content changed each pixel.
//
// The reactive-mask generator (ffxFsr3UpscalerContextGenerateReactiveMask): the reactive value of a
// pixel is how much the transparent and composited content changed it, the difference between
// the colour before (opaque only) and after, optionally tonemapped, scaled and thresholded.
//
// Differences from upstream: the pass file's body is this function; the flags are a uniform as
// upstream's (cbGenerateReactive), tested with bitwise ands, so one kernel serves every flag set.

export function GenerateReactive( b: Fsr3Bindings, uDispatchThreadId: N ): void {

	const flags = b.GenReactiveFlags();
	const has = ( bit: number ) => flags.bitAnd( uint( bit ) ).notEqual( uint( 0 ) );

	const ColorPreAlpha = vec3( b.LoadOpaqueOnly( uDispatchThreadId ) ).toVar( 'ColorPreAlpha' );
	const ColorPostAlpha = vec3( b.LoadInputColor( uDispatchThreadId ) ).toVar( 'ColorPostAlpha' );

	// if ( flags & APPLY_TONEMAP ): both tonemapped
	ColorPreAlpha.assign( select( has( FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_TONEMAP ), Tonemap( ColorPreAlpha ), ColorPreAlpha ) );
	ColorPostAlpha.assign( select( has( FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_TONEMAP ), Tonemap( ColorPostAlpha ), ColorPostAlpha ) );

	// if ( flags & APPLY_INVERSETONEMAP ): both inverse tonemapped
	ColorPreAlpha.assign( select( has( FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_INVERSETONEMAP ), InverseTonemap( ColorPreAlpha ), ColorPreAlpha ) );
	ColorPostAlpha.assign( select( has( FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_INVERSETONEMAP ), InverseTonemap( ColorPostAlpha ), ColorPostAlpha ) );

	const out_reactive_value = float( 0.0 ).toVar( 'out_reactive_value' );
	const delta = abs( ColorPostAlpha.sub( ColorPreAlpha ) ).toVar( 'delta' );

	out_reactive_value.assign( select( has( FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_USE_COMPONENTS_MAX ), max( delta.x, max( delta.y, delta.z ) ), length( delta ) ) );
	out_reactive_value.mulAssign( b.GenReactiveScale() );

	out_reactive_value.assign( select( has( FFX_FSR3UPSCALER_AUTOREACTIVEFLAGS_APPLY_THRESHOLD ),
		select( out_reactive_value.lessThan( b.GenReactiveThreshold() ), float( 0 ), b.GenReactiveBinaryValue() ),
		out_reactive_value ) );

	b.StoreAutoReactiveValue( uDispatchThreadId, out_reactive_value );

}
