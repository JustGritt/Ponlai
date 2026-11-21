/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_prepare_inputs.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { If, float, vec2, vec3, ivec2, min, max } from './Tsl.ts';
import type { N } from './Core.ts';
import {
	FSR3UPSCALER_FP16_MAX, fReconstructedDepthBilinearWeightThreshold, ReconstructedDepthMvPxThreshold, Get4KVelocity,
	GetBilinearSamplingData, IsOnScreen, GetViewSpaceDepthInMeters, ComputeHrPosFromLrPos, RGBToLuma,
} from './Common.ts';
import type { Fsr3Bindings } from './Bindings.ts';
import type { Fsr3ShaderOptions } from './Host.ts';

// FSR 3's first pass: dilated depth and motion, reconstructed previous depth and luma.
//
// The first pass at render size: the nearest depth of each 3 x 3 neighbourhood (the dilated depth)
// and the motion vector there (the dilated motion), the nearest depth scattered to where it was
// last frame (the reconstructed previous depth, an atomic min), the farthest depth in metres and
// the luma of the input colour.
//
// Differences from upstream: none in the maths. The nine depth loads and the loop over them are
// unrolled at build time (FFX_UNROLL); the bilinear scatter's four taps too.

export function ReconstructPrevDepth( b: Fsr3Bindings, o: Fsr3ShaderOptions, iPxPos: N, fDepth: N, fMotionVectorIn: N ): void {

	const fNearestDepthInMeters = min( GetViewSpaceDepthInMeters( b, fDepth ), float( FSR3UPSCALER_FP16_MAX ) );
	const fReconstructedDeptMvThreshold = ReconstructedDepthMvPxThreshold( fNearestDepthInMeters );

	// Discard small mvs
	const fMotionVector = vec2( fMotionVectorIn ).toVar();
	fMotionVector.mulAssign( float( Get4KVelocity( fMotionVector ).greaterThan( fReconstructedDeptMvThreshold ) ) );

	const fUv = vec2( iPxPos ).add( float( 0.5 ) ).div( vec2( b.RenderSize() ) );
	const fReprojectedUv = fUv.add( fMotionVector );
	const bilinearInfo = GetBilinearSamplingData( fReprojectedUv, b.RenderSize() );

	// Project current depth into previous frame locations.
	// Push to all pixels having some contribution if reprojection is using bilinear logic.
	for ( let iSampleIndex = 0; iSampleIndex < 4; iSampleIndex ++ ) {

		const iOffset = bilinearInfo.iOffsets[ iSampleIndex ];
		const fWeight = bilinearInfo.fWeights[ iSampleIndex ];

		If( fWeight.greaterThan( fReconstructedDepthBilinearWeightThreshold ), () => {

			const iStorePos = bilinearInfo.iBasePos.add( iOffset ).toVar();
			If( IsOnScreen( iStorePos, b.RenderSize() ), () => {

				b.StoreReconstructedDepth( o, iStorePos, fDepth );

			} );

		} );

	}

}

export interface DepthExtents {
	fNearest: N;
	fNearestCoord: N;
	fFarthest: N;
}

export function FindDepthExtents( b: Fsr3Bindings, o: Fsr3ShaderOptions, iPxPos: N ): DepthExtents {

	const iSampleCount = 9;
	const iSampleOffsets = [
		[ + 0, + 0 ],
		[ + 1, + 0 ],
		[ + 0, + 1 ],
		[ + 0, - 1 ],
		[ - 1, + 0 ],
		[ - 1, + 1 ],
		[ + 1, + 1 ],
		[ - 1, - 1 ],
		[ + 1, - 1 ],
	];

	// pull out the depth loads to allow SC to batch them
	const depth: N[] = [];
	for ( let iSampleIndex = 0; iSampleIndex < iSampleCount; ++ iSampleIndex ) {

		const iPos = iPxPos.add( ivec2( iSampleOffsets[ iSampleIndex ][ 0 ], iSampleOffsets[ iSampleIndex ][ 1 ] ) );
		depth[ iSampleIndex ] = float( b.LoadInputDepth( iPos ) ).toVar();

	}

	// find closest depth
	const extents: DepthExtents = {
		fNearestCoord: ivec2( iPxPos ).toVar( 'fNearestCoord' ),
		fNearest: depth[ 0 ].toVar( 'fNearest' ),
		fFarthest: depth[ 0 ].toVar( 'fFarthest' ),
	};
	for ( let iSampleIndex = 1; iSampleIndex < iSampleCount; ++ iSampleIndex ) {

		const iPos = iPxPos.add( ivec2( iSampleOffsets[ iSampleIndex ][ 0 ], iSampleOffsets[ iSampleIndex ][ 1 ] ) ).toVar();
		If( IsOnScreen( iPos, b.RenderSize() ), () => {

			const fNdDepth = depth[ iSampleIndex ];
			const nearer = o.FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH ? fNdDepth.greaterThan( extents.fNearest ) : fNdDepth.lessThan( extents.fNearest );
			If( nearer, () => {

				if ( o.FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH ) {

					extents.fFarthest.assign( min( extents.fFarthest, fNdDepth ) );

				} else {

					extents.fFarthest.assign( max( extents.fFarthest, fNdDepth ) );

				}
				extents.fNearestCoord.assign( iPos );
				extents.fNearest.assign( fNdDepth );

			} );

		} );

	}

	return extents;

}

export function DilateMotionVector( b: Fsr3Bindings, o: Fsr3ShaderOptions, _iPxPos: N, depthExtents: DepthExtents ): N {

	let iMotionVectorPos: N;
	if ( o.FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS ) {

		iMotionVectorPos = depthExtents.fNearestCoord;

	} else {

		iMotionVectorPos = ComputeHrPosFromLrPos( b, depthExtents.fNearestCoord );

	}
	// ( upstream also derives the sample position, iPxPos or its HR position, and leaves it unused )

	const fDilatedMotionVector = b.LoadInputMotionVector( o, iMotionVectorPos );

	return fDilatedMotionVector;

}

export function GetCurrentFrameLuma( b: Fsr3Bindings, iPxPos: N ): N {

	//We assume linear data. if non-linear input (sRGB, ...),
	//then we should convert to linear first and back to sRGB on output.
	const fRgb = max( vec3( 0, 0, 0 ), b.LoadInputColor( iPxPos ) );
	const fLuma = RGBToLuma( fRgb );
	return fLuma;

}

export function PrepareInputs( b: Fsr3Bindings, o: Fsr3ShaderOptions, iPxPos: N ): void {

	const depthExtents = FindDepthExtents( b, o, iPxPos );
	const fDilatedMotionVector = vec2( DilateMotionVector( b, o, iPxPos, depthExtents ) ).toVar();

	ReconstructPrevDepth( b, o, iPxPos, depthExtents.fNearest, fDilatedMotionVector );

	b.StoreDilatedMotionVector( iPxPos, fDilatedMotionVector );
	b.StoreDilatedDepth( iPxPos, depthExtents.fNearest );

	const fFarthestDepthInMeters = min( GetViewSpaceDepthInMeters( b, depthExtents.fFarthest ), float( FSR3UPSCALER_FP16_MAX ) );
	b.StoreFarthestDepth( iPxPos, fFarthestDepthInMeters );

	const fLuma = GetCurrentFrameLuma( b, iPxPos );
	b.StoreCurrentLuma( iPxPos, fLuma );

}
