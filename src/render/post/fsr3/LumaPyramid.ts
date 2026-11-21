/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_luma_pyramid.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { Fn, If, float, uint, vec4, ivec2, max, exp, log, workgroupArray } from './Tsl.ts';
import { component, type N } from './Core.ts';
import {
	FSR3UPSCALER_EPSILON, FRAME_INFO_EXPOSURE, FRAME_INFO_LOG_LUMA, FRAME_INFO_SCENE_AVERAGE_LUMA,
	resetAutoExposureAverageSmoothing, ClampLoad, ComputeAutoExposureFromLavg,
} from './Common.ts';
import { SpdDownsample, SpdDownsampleTail, type SpdCallbacks } from './Spd.ts';
import type { Fsr3Bindings } from './Bindings.ts';

// FSR 3's luminance pyramid: SPD over the luma and farthest depth, with the auto exposure.
//
// The luminance pyramid: SPD over the current luma, its log and the farthest depth, which stores
// the farthest depth's mip 1 (SPD's mip 0, half the render size), mip 5 of (log luma, luma) for its
// tail, and at the last mip (1 x 1) the frame info: the log luma averaged over time (the auto
// exposure FSR always tracks for its locks), the exposure from it and the scene's average luma.
//
// Differences from upstream: the FFX_HALF callbacks (all empty upstream) are not ported; the atomic
// counter callbacks go with SPD's counter (Spd.ts: the tail is `ComputeAutoExposureTail`, its own
// one-workgroup dispatch). The workgroup arrays `spdIntermediateR..A[ 16 ][ 16 ]` are flat arrays
// of 256 indexed x * 16 + y.

const LOG_LUMA = 0;
const LUMA = 1;
const DEPTH_IN_METERS = 2;

function callbacks( b: Fsr3Bindings ): SpdCallbacks {

	const spdIntermediateR = workgroupArray( 'float', 256 );
	const spdIntermediateG = workgroupArray( 'float', 256 );
	const spdIntermediateB = workgroupArray( 'float', 256 );
	const spdIntermediateA = workgroupArray( 'float', 256 );
	const at = ( x: N, y: N ) => uint( x ).mul( 16 ).add( uint( y ) );

	const SpdLoadSourceImageFn = Fn( ( [ iPxPos ]: N[] ) => {

		//We assume linear data. if non-linear input (sRGB, ...),
		//then we should convert to linear first and back to sRGB on output.
		const iPxSamplePos = ClampLoad( ivec2( iPxPos ), ivec2( 0, 0 ), ivec2( b.RenderSize() ) ).toVar();

		const fLuma = b.LoadCurrentLuma( iPxSamplePos ).toVar();
		const fLogLuma = max( float( FSR3UPSCALER_EPSILON ), log( fLuma ) );
		const fFarthestDepthInMeters = b.LoadFarthestDepth( iPxSamplePos );

		// fOutput[ LOG_LUMA ] x, [ LUMA ] y, [ DEPTH_IN_METERS ] z
		return vec4( fLogLuma, fLuma, fFarthestDepthInMeters, 0.0 );

	} ).setLayout( { name: 'SpdLoadSourceImage', type: 'vec4', inputs: [ { name: 'iPxPos', type: 'ivec2' } ] } );

	return {

		// one WGSL function per kernel (SPD calls it sixteen times per thread); built per kernel, so
		// r186's shared-layout uniform naming cannot cross kernels
		SpdLoadSourceImage: ( p: N ) => SpdLoadSourceImageFn( ivec2( p ) ),

		SpdLoad( tex: N ): N {

			return vec4( b.RWLoadPyramid( tex, 5 ), 0, 0 );

		},

		SpdReduce4( v0: N, v1: N, v2: N, v3: N ): N {

			return v0.add( v1 ).add( v2 ).add( v3 ).mul( 0.25 );

		},

		SpdStore( pix: N, outValue: N, index: number ): void {

			if ( index === 5 ) {

				b.StorePyramid( pix, outValue.xy, index );

			} else if ( index === 0 ) {

				b.StoreFarthestDepthMip1( pix, component( outValue, DEPTH_IN_METERS ) );

			}

			If( uint( index ).equal( b.MipCount().sub( 1 ) ), () => { //accumulate on 1x1 level

				If( pix.x.equal( 0 ).and( pix.y.equal( 0 ) ), () => {

					const frameInfo = vec4( b.LoadFrameInfo() ).toVar();
					const fSceneAvgLuma = component( outValue, LUMA );
					const fPrevLogLuma = component( frameInfo, FRAME_INFO_LOG_LUMA ).toVar();
					const fLogLuma = float( component( outValue, LOG_LUMA ) ).toVar();

					If( fPrevLogLuma.lessThan( resetAutoExposureAverageSmoothing ), () => { // Compare Lavg, so small or negative values

						fLogLuma.assign( fPrevLogLuma.add( fLogLuma.sub( fPrevLogLuma ).mul( float( 1.0 ).sub( exp( b.DeltaTime().negate() ) ) ) ) );
						fLogLuma.assign( max( float( 0.0 ), fLogLuma ) );

					} );

					component( frameInfo, FRAME_INFO_EXPOSURE ).assign( ComputeAutoExposureFromLavg( fLogLuma ) );
					component( frameInfo, FRAME_INFO_LOG_LUMA ).assign( fLogLuma );
					component( frameInfo, FRAME_INFO_SCENE_AVERAGE_LUMA ).assign( fSceneAvgLuma );

					b.StoreFrameInfo( frameInfo );

				} );

			} );

		},

		SpdLoadIntermediate( x: N, y: N ): N {

			const i = at( x, y );
			return vec4(
				spdIntermediateR.element( i ),
				spdIntermediateG.element( i ),
				spdIntermediateB.element( i ),
				spdIntermediateA.element( i ) );

		},

		SpdStoreIntermediate( x: N, y: N, value: N ): void {

			const i = at( x, y ).toVar();
			spdIntermediateR.element( i ).assign( value.x );
			spdIntermediateG.element( i ).assign( value.y );
			spdIntermediateB.element( i ).assign( value.z );
			spdIntermediateA.element( i ).assign( value.w );

		},

	};

}

/** ComputeAutoExposure: SPD's per-tile part (mips 0..5). */
export function ComputeAutoExposure( b: Fsr3Bindings, WorkGroupId: N, LocalThreadIndex: N ): void {

	SpdDownsample( callbacks( b ), WorkGroupId.xy, LocalThreadIndex, b.MipCount(), uint( WorkGroupId.z ), b.WorkGroupOffset() );

}

/** The last workgroup's part, as its own dispatch (mips 6.. and the frame info). */
export function ComputeAutoExposureTail( b: Fsr3Bindings, LocalThreadIndex: N ): void {

	SpdDownsampleTail( callbacks( b ), LocalThreadIndex, b.MipCount(), uint( 0 ) );

}
