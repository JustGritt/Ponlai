/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_shading_change.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { If, float, vec2, abs, max } from './Tsl.ts';
import { ffxSaturate, type N } from './Core.ts';
import { iShadingChangeMipStart, ShadingChangeRenderSize, ClampUv, IsOnScreen } from './Common.ts';
import type { Fsr3Bindings } from './Bindings.ts';

// FSR 3's shading change at half the render size, from the shading-change pyramid.
//
// The shading change at half the render size: the largest | difference x sign sum | over the
// shading-change pyramid's levels 0..2 at the jittered position.
//
// Differences from upstream: none in the maths. `1.0f + iShadingChangeMipStart / s_MipLevelsToUse`
// is HLSL's integer division (0 / 3 = 0), so the scale is 1; the loops are unrolled.

/** The SPD levels the pass reads ( iShadingChangeMipStart .. s_MipLevelsToUse - 1 ). */
export const s_MipLevelsToUse = 3;

export function ComputeShadingChangeLuma( b: Fsr3Bindings, fUv: N ): N[] {

	const fSamples: N[] = [];
	const fMipUv = ClampUv( fUv, ShadingChangeRenderSize( b ), b.GetSPDMipDimensions( 0 ) ).toVar();
	for ( let iMipLevel = iShadingChangeMipStart; iMipLevel < s_MipLevelsToUse; iMipLevel ++ ) {

		const fSample = b.SampleSPDMipLevel( fMipUv, iMipLevel ).toVar();

		fSamples[ iMipLevel ] = abs( fSample.x.mul( fSample.y ) ).toVar();

	}

	return fSamples;

}

export function ShadingChange( b: Fsr3Bindings, iPxPos: N ): void {

	If( IsOnScreen( iPxPos, ShadingChangeRenderSize( b ) ), () => {

		const fUv = vec2( iPxPos ).add( 0.5 ).div( vec2( ShadingChangeRenderSize( b ) ) );
		const fUvJittered = fUv.add( b.Jitter().div( vec2( b.RenderSize() ) ) );

		const fSamples = ComputeShadingChangeLuma( b, fUvJittered );

		const fScale = 1.0 + Math.trunc( iShadingChangeMipStart / s_MipLevelsToUse );
		const fShadingChange = float( 0.0 ).toVar( 'fShadingChange' );
		for ( let iMipLevel = iShadingChangeMipStart; iMipLevel < s_MipLevelsToUse; iMipLevel ++ ) {

			If( fSamples[ iMipLevel ].greaterThan( 0 ), () => {

				fShadingChange.assign( max( fShadingChange, fSamples[ iMipLevel ] ).mul( fScale ) );

			} );

		}

		b.StoreShadingChange( iPxPos, ffxSaturate( fShadingChange ) );

	} );

}
