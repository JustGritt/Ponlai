/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/spd/
 * ffx_spd.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { If, ivec2, uint, uvec2, workgroupBarrier } from './Tsl.ts';
import { ffxRemapForWaveReduction, type N } from './Core.ts';

// The Single Pass Downsampler (SPD): a texture's mips 0..11 in one dispatch.
//
// Single Pass Downsampler (SPD): one 256-thread workgroup reduces a 64 x 64 tile to mips 0..5
// (mip 0 is half the source), and the last workgroup to finish reduces the 64 x 64 texels of mip 5
// to mips 6..11. The user (the luminance and shading-change pyramids) supplies the callbacks, as
// upstream's includer defines them: `SpdLoadSourceImage`, `SpdLoad` (mip 5 for the tail),
// `SpdStore`, `SpdLoadIntermediate` / `SpdStoreIntermediate` (its workgroup memory) and
// `SpdReduce4`.
//
// Differences from upstream:
// - The non-packed 32-bit path with FFX_SPD_NO_WAVE_OPERATIONS (the workgroup-memory reductions:
//   `SpdDownsampleMips_0_1_LDS` and the LDS branches of mips 2..5); the wave-operation and FFX_HALF
//   paths are not ported (subgroups are optional in WebGPU). The upscaler's own build uses wave
//   operations where it can: the reductions add the same four values in another order, so a mip
//   may differ from upstream's in the last bit.
// - The last-workgroup tail (`SpdExitWorkgroup`'s global atomic counter and globallycoherent mip 5)
//   is a second dispatch of one workgroup (`SpdDownsampleTail`), run after the first: WebGPU makes
//   no storage write of one workgroup visible to another inside a dispatch. It computes the same
//   mips from the same mip 5. `SpdIncreaseAtomicCounter` / `SpdResetAtomicCounter` are gone.
// - Upstream's early `return`s on the mip count (a uniform) become nested branches; the barriers
//   stay in uniform control flow, as upstream's.
// - The store's mip index is a build-time number (every call site passes a constant), so a
//   callback picks its binding without a runtime switch.
// - HLSL's `[x][y]` workgroup arrays are flattened to x * 16 + y in the callbacks (LumaPyramid.ts,
//   ShadingChangePyramid.ts).

export interface SpdCallbacks {
	SpdLoadSourceImage: ( p: N, slice: N ) => N;
	SpdLoad: ( p: N, slice: N ) => N;
	SpdStore: ( p: N, value: N, mip: number, slice: N ) => void;
	SpdLoadIntermediate: ( x: N, y: N ) => N;
	SpdStoreIntermediate: ( x: N, y: N, value: N ) => void;
	SpdReduce4: ( v0: N, v1: N, v2: N, v3: N ) => N;
}

// ---------------------------------------------------------------------------------------------
// CPU

/** ffxSpdSetup (CPU): the workgroup counts, offset, workgroup total and mip count for a rect
 * ( left, top, width, height ); `mips` < 0 derives the count from the rect. */
export function ffxSpdSetup( rectInfo: [ number, number, number, number ], mips = - 1 ): {
	dispatchThreadGroupCountXY: [ number, number ], workGroupOffset: [ number, number ], numWorkGroupsAndMips: [ number, number ],
} {

	const workGroupOffset: [ number, number ] = [ 0, 0 ];
	const dispatchThreadGroupCountXY: [ number, number ] = [ 0, 0 ];
	const numWorkGroupsAndMips: [ number, number ] = [ 0, 0 ];

	// determines the offset of the first tile to downsample based on
	// left (rectInfo[0]) and top (rectInfo[1]) of the subregion.
	workGroupOffset[ 0 ] = Math.trunc( rectInfo[ 0 ] / 64 );
	workGroupOffset[ 1 ] = Math.trunc( rectInfo[ 1 ] / 64 );

	const endIndexX = Math.trunc( ( ( rectInfo[ 0 ] + rectInfo[ 2 ] - 1 ) >>> 0 ) / 64 ); // rectInfo[0] = left, rectInfo[2] = width
	const endIndexY = Math.trunc( ( ( rectInfo[ 1 ] + rectInfo[ 3 ] - 1 ) >>> 0 ) / 64 ); // rectInfo[1] = top, rectInfo[3] = height

	// we only need to dispatch as many thread groups as tiles we need to downsample
	// number of tiles per slice depends on the subregion to downsample
	dispatchThreadGroupCountXY[ 0 ] = ( endIndexX + 1 - workGroupOffset[ 0 ] ) >>> 0;
	dispatchThreadGroupCountXY[ 1 ] = ( endIndexY + 1 - workGroupOffset[ 1 ] ) >>> 0;

	// number of thread groups per slice
	numWorkGroupsAndMips[ 0 ] = ( dispatchThreadGroupCountXY[ 0 ] * dispatchThreadGroupCountXY[ 1 ] ) >>> 0;

	if ( mips >= 0 ) {

		numWorkGroupsAndMips[ 1 ] = mips >>> 0;

	} else {

		// calculate based on rect width and height
		const resolution = Math.max( rectInfo[ 2 ], rectInfo[ 3 ] );
		numWorkGroupsAndMips[ 1 ] = Math.trunc( Math.min( Math.floor( Math.fround( Math.log2( Math.fround( resolution ) ) ) ), 12 ) ) >>> 0;

	}

	return { dispatchThreadGroupCountXY, workGroupOffset, numWorkGroupsAndMips };

}

// ---------------------------------------------------------------------------------------------
// GPU, non-packed version

function ffxSpdWorkgroupShuffleBarrier(): void {

	workgroupBarrier();

}

function SpdReduceIntermediate( cb: SpdCallbacks, i0: N, i1: N, i2: N, i3: N ): N {

	const v0 = cb.SpdLoadIntermediate( i0.x, i0.y );
	const v1 = cb.SpdLoadIntermediate( i1.x, i1.y );
	const v2 = cb.SpdLoadIntermediate( i2.x, i2.y );
	const v3 = cb.SpdLoadIntermediate( i3.x, i3.y );
	return cb.SpdReduce4( v0, v1, v2, v3 );

}

function SpdReduceLoad4( cb: SpdCallbacks, base: N, slice: N ): N {

	const v0 = cb.SpdLoad( ivec2( base.add( uvec2( 0, 0 ) ) ), slice );
	const v1 = cb.SpdLoad( ivec2( base.add( uvec2( 0, 1 ) ) ), slice );
	const v2 = cb.SpdLoad( ivec2( base.add( uvec2( 1, 0 ) ) ), slice );
	const v3 = cb.SpdLoad( ivec2( base.add( uvec2( 1, 1 ) ) ), slice );
	return cb.SpdReduce4( v0, v1, v2, v3 );

}

function SpdReduceLoadSourceImage( cb: SpdCallbacks, base: N, slice: N ): N {

	// ( SPD_LINEAR_SAMPLER is not defined by the upscaler: four loads )
	const v0 = cb.SpdLoadSourceImage( ivec2( base.add( ivec2( 0, 0 ) ) ), slice );
	const v1 = cb.SpdLoadSourceImage( ivec2( base.add( ivec2( 0, 1 ) ) ), slice );
	const v2 = cb.SpdLoadSourceImage( ivec2( base.add( ivec2( 1, 0 ) ) ), slice );
	const v3 = cb.SpdLoadSourceImage( ivec2( base.add( ivec2( 1, 1 ) ) ), slice );
	return cb.SpdReduce4( v0, v1, v2, v3 );

}

function SpdDownsampleMips_0_1_LDS( cb: SpdCallbacks, x: N, y: N, workGroupID: N, localInvocationIndex: N, mip: N, slice: N ): void {

	const wg64 = ivec2( workGroupID.mul( 64 ) ), wg32 = ivec2( workGroupID.mul( 32 ) );
	const ix = ivec2( x, y );
	const v: N[] = [];

	let tex = wg64.add( ix.mul( 2 ) );
	let pix = wg32.add( ix );
	v[ 0 ] = SpdReduceLoadSourceImage( cb, tex, slice ).toVar();
	cb.SpdStore( pix, v[ 0 ], 0, slice );

	tex = wg64.add( ix.mul( 2 ).add( ivec2( 32, 0 ) ) );
	pix = wg32.add( ix.add( ivec2( 16, 0 ) ) );
	v[ 1 ] = SpdReduceLoadSourceImage( cb, tex, slice ).toVar();
	cb.SpdStore( pix, v[ 1 ], 0, slice );

	tex = wg64.add( ix.mul( 2 ).add( ivec2( 0, 32 ) ) );
	pix = wg32.add( ix.add( ivec2( 0, 16 ) ) );
	v[ 2 ] = SpdReduceLoadSourceImage( cb, tex, slice ).toVar();
	cb.SpdStore( pix, v[ 2 ], 0, slice );

	tex = wg64.add( ix.mul( 2 ).add( ivec2( 32, 32 ) ) );
	pix = wg32.add( ix.add( ivec2( 16, 16 ) ) );
	v[ 3 ] = SpdReduceLoadSourceImage( cb, tex, slice ).toVar();
	cb.SpdStore( pix, v[ 3 ], 0, slice );

	// if ( mip <= 1 ) return;
	If( mip.greaterThan( 1 ), () => {

		for ( let i = 0; i < 4; i ++ ) {

			cb.SpdStoreIntermediate( x, y, v[ i ] );
			ffxSpdWorkgroupShuffleBarrier();
			If( localInvocationIndex.lessThan( 64 ), () => {

				v[ i ].assign( SpdReduceIntermediate( cb, uvec2( x.mul( 2 ).add( 0 ), y.mul( 2 ).add( 0 ) ), uvec2( x.mul( 2 ).add( 1 ), y.mul( 2 ).add( 0 ) ), uvec2( x.mul( 2 ).add( 0 ), y.mul( 2 ).add( 1 ) ), uvec2( x.mul( 2 ).add( 1 ), y.mul( 2 ).add( 1 ) ) ) );
				cb.SpdStore( ivec2( workGroupID.mul( 16 ) ).add( ivec2( x.add( ( i % 2 ) * 8 ), y.add( Math.trunc( i / 2 ) * 8 ) ) ), v[ i ], 1, slice );

			} );
			ffxSpdWorkgroupShuffleBarrier();

		}

		If( localInvocationIndex.lessThan( 64 ), () => {

			cb.SpdStoreIntermediate( x.add( 0 ), y.add( 0 ), v[ 0 ] );
			cb.SpdStoreIntermediate( x.add( 8 ), y.add( 0 ), v[ 1 ] );
			cb.SpdStoreIntermediate( x.add( 0 ), y.add( 8 ), v[ 2 ] );
			cb.SpdStoreIntermediate( x.add( 8 ), y.add( 8 ), v[ 3 ] );

		} );

	} );

}

function SpdDownsampleMip_2( cb: SpdCallbacks, x: N, y: N, workGroupID: N, localInvocationIndex: N, mip: number, slice: N ): void {

	If( localInvocationIndex.lessThan( 64 ), () => {

		const v = SpdReduceIntermediate( cb, uvec2( x.mul( 2 ).add( 0 ), y.mul( 2 ).add( 0 ) ), uvec2( x.mul( 2 ).add( 1 ), y.mul( 2 ).add( 0 ) ), uvec2( x.mul( 2 ).add( 0 ), y.mul( 2 ).add( 1 ) ), uvec2( x.mul( 2 ).add( 1 ), y.mul( 2 ).add( 1 ) ) ).toVar();
		cb.SpdStore( ivec2( workGroupID.mul( 8 ) ).add( ivec2( x, y ) ), v, mip, slice );
		// store to LDS, try to reduce bank conflicts
		// x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
		// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		// 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0 x
		// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		// x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
		// ...
		// x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
		cb.SpdStoreIntermediate( x.mul( 2 ).add( y.mod( 2 ) ), y.mul( 2 ), v );

	} );

}

function SpdDownsampleMip_3( cb: SpdCallbacks, x: N, y: N, workGroupID: N, localInvocationIndex: N, mip: number, slice: N ): void {

	If( localInvocationIndex.lessThan( 16 ), () => {

		// x 0 x 0
		// 0 0 0 0
		// 0 x 0 x
		// 0 0 0 0
		const v = SpdReduceIntermediate( cb,
			uvec2( x.mul( 4 ).add( 0 + 0 ), y.mul( 4 ).add( 0 ) ), uvec2( x.mul( 4 ).add( 2 + 0 ), y.mul( 4 ).add( 0 ) ),
			uvec2( x.mul( 4 ).add( 0 + 1 ), y.mul( 4 ).add( 2 ) ), uvec2( x.mul( 4 ).add( 2 + 1 ), y.mul( 4 ).add( 2 ) ) ).toVar();
		cb.SpdStore( ivec2( workGroupID.mul( 4 ) ).add( ivec2( x, y ) ), v, mip, slice );
		// store to LDS
		// x 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0
		// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		// 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0 0
		// ...
		// 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0
		// ...
		// 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x
		// ...
		cb.SpdStoreIntermediate( x.mul( 4 ).add( y ), y.mul( 4 ), v );

	} );

}

function SpdDownsampleMip_4( cb: SpdCallbacks, x: N, y: N, workGroupID: N, localInvocationIndex: N, mip: number, slice: N ): void {

	If( localInvocationIndex.lessThan( 4 ), () => {

		// x 0 0 0 x 0 0 0
		// ...
		// 0 x 0 0 0 x 0 0
		const v = SpdReduceIntermediate( cb,
			uvec2( x.mul( 8 ).add( 0 + 0 ).add( y.mul( 2 ) ), y.mul( 8 ).add( 0 ) ),
			uvec2( x.mul( 8 ).add( 4 + 0 ).add( y.mul( 2 ) ), y.mul( 8 ).add( 0 ) ),
			uvec2( x.mul( 8 ).add( 0 + 1 ).add( y.mul( 2 ) ), y.mul( 8 ).add( 4 ) ),
			uvec2( x.mul( 8 ).add( 4 + 1 ).add( y.mul( 2 ) ), y.mul( 8 ).add( 4 ) ) ).toVar();
		cb.SpdStore( ivec2( workGroupID.mul( 2 ) ).add( ivec2( x, y ) ), v, mip, slice );
		// store to LDS
		// x x x x 0 ...
		// 0 ...
		cb.SpdStoreIntermediate( x.add( y.mul( 2 ) ), uint( 0 ), v );

	} );

}

function SpdDownsampleMip_5( cb: SpdCallbacks, workGroupID: N, localInvocationIndex: N, mip: number, slice: N ): void {

	If( localInvocationIndex.lessThan( 1 ), () => {

		// x x x x 0 ...
		// 0 ...
		const v = SpdReduceIntermediate( cb, uvec2( 0, 0 ), uvec2( 1, 0 ), uvec2( 2, 0 ), uvec2( 3, 0 ) ).toVar();
		cb.SpdStore( ivec2( workGroupID ), v, mip, slice );

	} );

}

function SpdDownsampleMips_6_7( cb: SpdCallbacks, x: N, y: N, mips: N, slice: N ): void {

	let tex = ivec2( x.mul( 4 ).add( 0 ), y.mul( 4 ).add( 0 ) );
	let pix = ivec2( x.mul( 2 ).add( 0 ), y.mul( 2 ).add( 0 ) );
	const v0 = SpdReduceLoad4( cb, uvec2( tex ), slice ).toVar();
	cb.SpdStore( pix, v0, 6, slice );

	tex = ivec2( x.mul( 4 ).add( 2 ), y.mul( 4 ).add( 0 ) );
	pix = ivec2( x.mul( 2 ).add( 1 ), y.mul( 2 ).add( 0 ) );
	const v1 = SpdReduceLoad4( cb, uvec2( tex ), slice ).toVar();
	cb.SpdStore( pix, v1, 6, slice );

	tex = ivec2( x.mul( 4 ).add( 0 ), y.mul( 4 ).add( 2 ) );
	pix = ivec2( x.mul( 2 ).add( 0 ), y.mul( 2 ).add( 1 ) );
	const v2 = SpdReduceLoad4( cb, uvec2( tex ), slice ).toVar();
	cb.SpdStore( pix, v2, 6, slice );

	tex = ivec2( x.mul( 4 ).add( 2 ), y.mul( 4 ).add( 2 ) );
	pix = ivec2( x.mul( 2 ).add( 1 ), y.mul( 2 ).add( 1 ) );
	const v3 = SpdReduceLoad4( cb, uvec2( tex ), slice ).toVar();
	cb.SpdStore( pix, v3, 6, slice );

	// if ( mips <= 7 ) return;
	If( mips.greaterThan( 7 ), () => {

		// no barrier needed, working on values only from the same thread
		const v = cb.SpdReduce4( v0, v1, v2, v3 ).toVar();
		cb.SpdStore( ivec2( x, y ), v, 7, slice );
		cb.SpdStoreIntermediate( x, y, v );

	} );

}

function SpdDownsampleNextFour( cb: SpdCallbacks, x: N, y: N, workGroupID: N, localInvocationIndex: N, baseMip: number, mips: N, slice: N ): void {

	// each `if ( mips <= baseMip + k ) return;` becomes a branch around the rest
	If( mips.greaterThan( baseMip ), () => {

		ffxSpdWorkgroupShuffleBarrier();
		SpdDownsampleMip_2( cb, x, y, workGroupID, localInvocationIndex, baseMip, slice );
		If( mips.greaterThan( baseMip + 1 ), () => {

			ffxSpdWorkgroupShuffleBarrier();
			SpdDownsampleMip_3( cb, x, y, workGroupID, localInvocationIndex, baseMip + 1, slice );
			If( mips.greaterThan( baseMip + 2 ), () => {

				ffxSpdWorkgroupShuffleBarrier();
				SpdDownsampleMip_4( cb, x, y, workGroupID, localInvocationIndex, baseMip + 2, slice );
				If( mips.greaterThan( baseMip + 3 ), () => {

					ffxSpdWorkgroupShuffleBarrier();
					SpdDownsampleMip_5( cb, workGroupID, localInvocationIndex, baseMip + 3, slice );

				} );

			} );

		} );

	} );

}

/** The thread's x, y in its 64 x 64 tile's 16 x 16 quarter, from SV_GroupIndex. */
function SpdThreadXY( localInvocationIndex: N ): { x: N, y: N } {

	const sub_xy = ffxRemapForWaveReduction( localInvocationIndex.mod( 64 ) ).toVar();
	const x = sub_xy.x.add( uint( 8 ).mul( localInvocationIndex.shiftRight( 6 ).mod( 2 ) ) ).toVar();
	const y = sub_xy.y.add( uint( 8 ).mul( localInvocationIndex.shiftRight( 7 ) ) ).toVar();
	return { x, y };

}

/** SpdDownsample (with the work group offset) up to the last workgroup's exit: mips 0..5 of this
 * workgroup's 64 x 64 tile. `mips` is the uniform mip count. */
export function SpdDownsample( cb: SpdCallbacks, workGroupID: N, localInvocationIndex: N, mips: N, slice: N, workGroupOffset: N ): void {

	const wg = uvec2( workGroupID ).add( workGroupOffset ).toVar();

	// compute MIP level 0 and 1
	const { x, y } = SpdThreadXY( localInvocationIndex );
	SpdDownsampleMips_0_1_LDS( cb, x, y, wg, localInvocationIndex, mips, slice );

	// compute MIP level 2, 3, 4, 5
	SpdDownsampleNextFour( cb, x, y, wg, localInvocationIndex, 2, mips, slice );

}

/** The last workgroup's part of SpdDownsample, as its own one-workgroup dispatch after the first
 * (see the header): mips 6..11 from mip 5. */
export function SpdDownsampleTail( cb: SpdCallbacks, localInvocationIndex: N, mips: N, slice: N ): void {

	// if ( mips <= 6 ) return;
	If( mips.greaterThan( 6 ), () => {

		const { x, y } = SpdThreadXY( localInvocationIndex );

		// After mip 5 there is only a single workgroup left that downsamples the remaining up to 64x64 texels.
		// compute MIP level 6 and 7
		SpdDownsampleMips_6_7( cb, x, y, mips, slice );

		// compute MIP level 8, 9, 10, 11
		SpdDownsampleNextFour( cb, x, y, uvec2( 0, 0 ), localInvocationIndex, 8, mips, slice );

	} );

}
