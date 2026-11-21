/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_shading_change_pyramid.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { Fn, If, float, int, uint, vec2, vec4, ivec2, abs, min, max, floor, sign, pow, select, bool, workgroupArray } from './Tsl.ts';
import type { N } from './Core.ts';
import {
	SHADING_CHANGE_SET_SIZE, iShadingChangeMipStart, fShadingChangeSamplePow, FSR3UPSCALER_FP16_MAX, FSR3UPSCALER_FP16_MIN,
	FSR3UPSCALER_FP32_MIN, FSR3UPSCALER_EPSILON, MinDividedByMax, ClampLoad, IsUvInside,
} from './Common.ts';
import { SpdDownsample, SpdDownsampleTail, type SpdCallbacks } from './Spd.ts';
import { s_MipLevelsToUse } from './ShadingChange.ts';
import type { Fsr3Bindings } from './Bindings.ts';

// FSR 3's shading-change pyramid: the luma's signed change since last frame, reduced by SPD.
//
// The shading-change pyramid: for each render pixel, the smallest signed relative difference
// between this frame's luma and last frame's around the reprojected position (five-tap crosses,
// sorted, walked together), then SPD over ( difference, its sign, 1 ) into SPD_MIPS levels 0..5,
// which the shading-change pass samples.
//
// Differences from upstream:
// - A SampleSet is five TSL variables; `SortSet`'s network of CompareSwap runs on them unrolled.
//   `ComputeMinimumDifference` indexes the sets with runtime indices `a` and `b` through a select
//   chain. Its loop (at most five steps while fMinDiff < FP16_MAX) is unrolled, each step behind
//   that condition. On the last step upstream's `b` update reads fSet0.fSamples[ a ] with `a` up to
//   5, past the array (undefined in HLSL); the select chain reads element 4 there, and that `b` is
//   never read again.
// - `ffxPow( x, fShadingChangeSamplePow )` with the constant 1.0 is x (the folding the shader
//   compiler does upstream; WGSL's pow( x, 1 ) need not return x exactly).
// - The atomic counter and FFX_HALF callbacks are gone (Spd.ts). The tail
//   (`ComputeShadingChangePyramidTail`) stores nothing: every store of a level above 5 falls
//   through `StorePyramid`, as upstream's. It is ported for completeness; running it changes no
//   resource.
// - Only the levels the shading-change pass reads are stored ( iShadingChangeMipStart ..
//   s_MipLevelsToUse - 1, ShadingChange.ts: 0..2 ). Upstream also stores levels 3..5, which nothing
//   reads before the next frame's clear (the luminance pyramid's level 5 is read by its tail before
//   this pass runs). Six storage bindings in one kernel pass WebGPU's default limit of four storage
//   textures per stage (each level is a binding of its own: Bindings.ts `uav`).

const DIFFERENCE = 0;
const SIGN_SUM = 1;
const MIP0_INDICATOR = 2;

const iSampleOffsets = [ [ + 0, + 0 ], [ - 1, + 0 ], [ + 1, + 0 ], [ + 0, - 1 ], [ + 0, + 1 ] ];

type SampleSet = N[];

export function Sort2( v: N ): N {

	return vec2( min( v.x, v.y ), max( v.x, v.y ) );

}

function CompareSwap( fSet: SampleSet, i: number, j: number ): void {

	const fTmp = min( fSet[ i ], fSet[ j ] ).toVar();
	fSet[ j ].assign( max( fSet[ i ], fSet[ j ] ) );
	fSet[ i ].assign( fTmp );

}

export function SortSet( fSet: SampleSet ): void {

	// SHADING_CHANGE_SET_SIZE == 5
	CompareSwap( fSet, 0, 3 );
	CompareSwap( fSet, 1, 4 );
	CompareSwap( fSet, 0, 2 );
	CompareSwap( fSet, 1, 3 );
	CompareSwap( fSet, 0, 1 );
	CompareSwap( fSet, 2, 4 );
	CompareSwap( fSet, 1, 2 );
	CompareSwap( fSet, 3, 4 );
	CompareSwap( fSet, 2, 3 );

}

/** fSet.fSamples[ i ] for a runtime index (see the header). */
function pick( fSet: SampleSet, i: N ): N {

	let v = fSet[ SHADING_CHANGE_SET_SIZE - 1 ];
	for ( let k = SHADING_CHANGE_SET_SIZE - 2; k >= 0; k -- ) v = select( i.equal( k ), fSet[ k ], v );
	return v;

}

export function ComputeMinimumDifference( fSet0In: SampleSet, fSet1In: SampleSet ): N {

	const fMinDiff = float( FSR3UPSCALER_FP16_MAX - 1 ).toVar( 'fMinDiff' );
	const a = int( 0 ).toVar( 'a' );
	const b = int( 0 ).toVar( 'b' );

	// ( the sets are sorted in place: copies, as upstream's are passed by value )
	const fSet0 = fSet0In.map( ( v ) => float( v ).toVar() );
	const fSet1 = fSet1In.map( ( v ) => float( v ).toVar() );
	SortSet( fSet0 );
	SortSet( fSet1 );

	const fMax = min( fSet0[ SHADING_CHANGE_SET_SIZE - 1 ], fSet1[ SHADING_CHANGE_SET_SIZE - 1 ] );

	If( fMax.greaterThan( FSR3UPSCALER_FP32_MIN ), () => {

		for ( let i = 0; i < SHADING_CHANGE_SET_SIZE; i ++ ) {

			If( fMinDiff.lessThan( FSR3UPSCALER_FP16_MAX ), () => {

				const fDiff = pick( fSet0, a ).sub( pick( fSet1, b ) ).toVar();

				If( abs( fDiff ).greaterThan( FSR3UPSCALER_FP16_MIN ), () => {

					fDiff.assign( sign( fDiff ).mul( float( 1.0 ).sub( MinDividedByMax( pick( fSet0, a ), pick( fSet1, b ) ) ) ) );

					fMinDiff.assign( select( abs( fDiff ).lessThan( abs( fMinDiff ) ), fDiff, fMinDiff ) );

					a.addAssign( int( pick( fSet0, a ).lessThan( pick( fSet1, b ) ) ) );
					b.addAssign( int( pick( fSet0, a ).greaterThanEqual( pick( fSet1, b ) ) ) );

				} ).Else( () => {

					fMinDiff.assign( FSR3UPSCALER_FP16_MAX );

				} );

			} );

		}

	} );

	return fMinDiff.mul( float( fMinDiff.lessThan( FSR3UPSCALER_FP16_MAX - 1 ) ) );

}

function samplePow( x: N ): N {

	return fShadingChangeSamplePow === 1.0 ? x : pow( x, float( fShadingChangeSamplePow ) );

}

export function GetCurrentLumaBilinearSamples( b: Fsr3Bindings, fUv: N ): SampleSet {

	const fUvJittered = fUv.add( b.Jitter().div( vec2( b.RenderSize() ) ) ).toVar();
	const iBasePos = ivec2( floor( fUvJittered.mul( vec2( b.RenderSize() ) ) ) ).toVar();

	const fSet: SampleSet = [];
	for ( let iSampleIndex = 0; iSampleIndex < SHADING_CHANGE_SET_SIZE; iSampleIndex ++ ) {

		const iSamplePos = ClampLoad( iBasePos, ivec2( iSampleOffsets[ iSampleIndex ][ 0 ], iSampleOffsets[ iSampleIndex ][ 1 ] ), b.RenderSize() );
		let v = b.LoadCurrentLuma( iSamplePos ).mul( b.Exposure() );
		v = samplePow( v );
		v = max( v, float( FSR3UPSCALER_EPSILON ) );
		fSet.push( float( v ).toVar() );

	}

	return fSet;

}

export function GetPreviousLumaBilinearSamples( b: Fsr3Bindings, fUv: N, fMotionVector: N ): { fSet: SampleSet, bIsExistingSample: N } {

	const fSet: SampleSet = [];
	for ( let i = 0; i < SHADING_CHANGE_SET_SIZE; i ++ ) fSet.push( float( 0 ).toVar() );

	const fUvJittered = fUv.add( b.PreviousFrameJitter().div( vec2( b.PreviousFrameRenderSize() ) ) );
	const fReprojectedUv = fUvJittered.add( fMotionVector ).toVar();

	const bIsExistingSample = bool( IsUvInside( fReprojectedUv ) ).toVar();
	If( bIsExistingSample, () => {

		const iBasePos = ivec2( floor( fReprojectedUv.mul( vec2( b.PreviousFrameRenderSize() ) ) ) ).toVar();

		for ( let iSampleIndex = 0; iSampleIndex < SHADING_CHANGE_SET_SIZE; iSampleIndex ++ ) {

			const iSamplePos = ClampLoad( iBasePos, ivec2( iSampleOffsets[ iSampleIndex ][ 0 ], iSampleOffsets[ iSampleIndex ][ 1 ] ), b.PreviousFrameRenderSize() );
			let v = b.LoadPreviousLuma( iSamplePos ).mul( b.DeltaPreExposure() ).mul( b.Exposure() );
			v = samplePow( v );
			v = max( v, float( FSR3UPSCALER_EPSILON ) );
			fSet[ iSampleIndex ].assign( v );

		}

	} );

	return { fSet, bIsExistingSample };

}

export function ComputeDiff( b: Fsr3Bindings, fUv: N, fMotionVector: N ): N {

	const fMinDiff = float( 0.0 ).toVar( 'fMinDiffOut' );

	const fCurrentSamples = GetCurrentLumaBilinearSamples( b, fUv );
	const previousData = GetPreviousLumaBilinearSamples( b, fUv, fMotionVector );

	If( previousData.bIsExistingSample, () => {

		fMinDiff.assign( ComputeMinimumDifference( fCurrentSamples, previousData.fSet ) );

	} );

	return fMinDiff;

}

function callbacks( b: Fsr3Bindings ): SpdCallbacks {

	const spdIntermediateR = workgroupArray( 'float', 256 );
	const spdIntermediateG = workgroupArray( 'float', 256 );
	const spdIntermediateB = workgroupArray( 'float', 256 );
	const spdIntermediateA = workgroupArray( 'float', 256 );
	const at = ( x: N, y: N ) => uint( x ).mul( 16 ).add( uint( y ) );

	const SpdLoadSourceImageFn = Fn( ( [ iPxPos ]: N[] ) => {

		const iPxSamplePos = ClampLoad( ivec2( iPxPos ), ivec2( 0, 0 ), ivec2( b.RenderSize() ) ).toVar();

		const fDilatedMotionVector = b.LoadDilatedMotionVector( iPxSamplePos ).toVar();
		const fUv = vec2( iPxSamplePos ).add( 0.5 ).div( vec2( b.RenderSize() ) );

		const fScaledAndSignedLumaDiff = ComputeDiff( b, fUv, fDilatedMotionVector ).toVar();

		const fOutput = vec4( 0.0, 0.0, 0.0, 0.0 ).toVar();
		const channel = ( i: number ) => fOutput[ 'xyzw'[ i ] ];
		channel( DIFFERENCE ).assign( fScaledAndSignedLumaDiff );
		channel( SIGN_SUM ).assign( select( fScaledAndSignedLumaDiff.notEqual( 0.0 ), sign( fScaledAndSignedLumaDiff ), float( 0.0 ) ) );
		channel( MIP0_INDICATOR ).assign( 1.0 );

		return fOutput;

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

			if ( index >= iShadingChangeMipStart && index < s_MipLevelsToUse ) {

				b.StorePyramid( pix, outValue.xy, index );

			}

		},

		SpdLoadIntermediate( x: N, y: N ): N {

			const i = at( x, y );
			return vec4( spdIntermediateR.element( i ), spdIntermediateG.element( i ), spdIntermediateB.element( i ), spdIntermediateA.element( i ) );

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

/** ComputeShadingChangePyramid: SPD's per-tile part (levels 0..5). */
export function ComputeShadingChangePyramid( b: Fsr3Bindings, WorkGroupId: N, LocalThreadIndex: N ): void {

	SpdDownsample( callbacks( b ), WorkGroupId.xy, LocalThreadIndex, b.MipCount(), uint( WorkGroupId.z ), b.WorkGroupOffset() );

}

/** The last workgroup's part (stores nothing, see the header). */
export function ComputeShadingChangePyramidTail( b: Fsr3Bindings, LocalThreadIndex: N ): void {

	SpdDownsampleTail( callbacks( b ), LocalThreadIndex, b.MipCount(), uint( 0 ) );

}
