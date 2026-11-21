/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_debug_view.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { If, float, int, vec2, vec3, vec4, ivec2 } from './Tsl.ts';
import { ffxSaturate, component, type N } from './Core.ts';
import { ClampUv, GetViewSpaceDepthInMeters, fLockThreshold, DISOCCLUSION, REACTIVE, SHADING_CHANGE } from './Common.ts';
import type { Fsr3Bindings } from './Bindings.ts';

// FSR 3's debug view: a 3 x 3 grid of the upscaler's inputs and masks over the output.
//
// The debug view: a 3 x 3 grid over the output, its top row the dilated motion vectors, the
// protected (locked) areas and the dilated depth in metres, its bottom row the disocclusion, the
// shading change ("reactiveness") and the detail-protection takedown (the reactive mask).
//
// Differences from upstream: the viewport grid is computed at build time per cell (upstream fills
// an array of nine and draws six); `SampleDilatedDepth` filters by hand (Bindings.ts).

interface FfxDebugViewport {
	offset: N;
	size: N;
}

function getTransformedUv( iPxPos: N, vp: FfxDebugViewport ): N {

	const fUv = vec2( iPxPos.sub( vp.offset ) ).add( 0.5 ).div( vec2( vp.size ) );

	return fUv;

}

function getMotionVectorColor( b: Fsr3Bindings, fMotionVector: N ): N {

	return vec3( vec2( 0.5 ).add( fMotionVector.mul( vec2( b.RenderSize() ) ).mul( 0.5 ) ), 0.5 );

}

/** (defined upstream and not drawn) */
export function getUnusedIndicationColor( iPxPos: N, vp: FfxDebugViewport ): N {

	const basePos = iPxPos.sub( vp.offset );
	const ar = float( vp.size.x ).div( float( vp.size.y ) );
	return vec4( float( basePos.x.equal( int( float( basePos.y ).mul( ar ) ) ) ), 0, 0, 1 );

}

function drawDilatedMotionVectors( b: Fsr3Bindings, iPxPos: N, vp: FfxDebugViewport ): void {

	const fUv = getTransformedUv( iPxPos, vp );
	const fUv_HW = ClampUv( fUv, b.RenderSize(), b.MaxRenderSize() );

	const fMotionVector = b.SampleDilatedMotionVector( fUv_HW );

	b.StoreUpscaledOutput( iPxPos, getMotionVectorColor( b, fMotionVector ) );

}

function drawDisocclusionMask( b: Fsr3Bindings, iPxPos: N, vp: FfxDebugViewport ): void {

	const fUv = getTransformedUv( iPxPos, vp );
	const fUv_HW = ClampUv( fUv, b.RenderSize(), b.MaxRenderSize() );

	const fDisocclusionFactor = ffxSaturate( component( b.SampleDilatedReactiveMasks( fUv_HW ), DISOCCLUSION ) );

	b.StoreUpscaledOutput( iPxPos, vec3( 0, fDisocclusionFactor, 0 ) );

}

function drawDetailProtectionTakedown( b: Fsr3Bindings, iPxPos: N, vp: FfxDebugViewport ): void {

	const fUv = getTransformedUv( iPxPos, vp );
	const fUv_HW = ClampUv( fUv, b.RenderSize(), b.MaxRenderSize() );

	const fProtectionTakedown = ffxSaturate( component( b.SampleDilatedReactiveMasks( fUv_HW ), REACTIVE ) );

	b.StoreUpscaledOutput( iPxPos, vec3( 0, fProtectionTakedown, 0 ) );

}

function drawReactiveness( b: Fsr3Bindings, iPxPos: N, vp: FfxDebugViewport ): void {

	const fUv = getTransformedUv( iPxPos, vp );
	const fUv_HW = ClampUv( fUv, b.RenderSize(), b.MaxRenderSize() );

	const fShadingChange = ffxSaturate( component( b.SampleDilatedReactiveMasks( fUv_HW ), SHADING_CHANGE ) );

	b.StoreUpscaledOutput( iPxPos, vec3( 0, fShadingChange, 0 ) );

}

function drawProtectedAreas( b: Fsr3Bindings, iPxPos: N, vp: FfxDebugViewport ): void {

	const fUv = getTransformedUv( iPxPos, vp );

	const fProtection = ffxSaturate( b.SampleHistory( fUv ).w.sub( fLockThreshold ) );

	b.StoreUpscaledOutput( iPxPos, vec3( fProtection, 0, 0 ) );

}

function drawDilatedDepthInMeters( b: Fsr3Bindings, iPxPos: N, vp: FfxDebugViewport ): void {

	const fUv = getTransformedUv( iPxPos, vp );
	const fUv_HW = ClampUv( fUv, b.RenderSize(), b.MaxRenderSize() );

	const fDilatedDepth = b.SampleDilatedDepth( fUv_HW );
	const fDepthInMeters = GetViewSpaceDepthInMeters( b, fDilatedDepth );

	b.StoreUpscaledOutput( iPxPos, vec3( ffxSaturate( fDepthInMeters.div( 25.0 ) ), 0, 0 ) );

}

function pointIsInsideViewport( iPxPos: N, vp: FfxDebugViewport ): N {

	const extent = vp.offset.add( vp.size );

	return iPxPos.x.greaterThanEqual( vp.offset.x ).and( iPxPos.x.lessThan( extent.x ) ).and( iPxPos.y.greaterThanEqual( vp.offset.y ).and( iPxPos.y.lessThan( extent.y ) ) );

}

export function DebugView( b: Fsr3Bindings, iPxPosIn: N ): void {

	const iPxPos = ivec2( iPxPosIn ).toVar( 'iPxPos' );
	const VIEWPORT_GRID_SIZE_X = 3;
	const VIEWPORT_GRID_SIZE_Y = 3;

	const fViewportScale = vec2( 1.0 / VIEWPORT_GRID_SIZE_X, 1.0 / VIEWPORT_GRID_SIZE_Y );
	const iViewportSize = ivec2( vec2( b.UpscaleSize() ).mul( fViewportScale ) ).toVar( 'iViewportSize' );

	// compute grid [y][x] for easier placement of viewports
	const vp = ( x: number, y: number ): FfxDebugViewport => ( { offset: iViewportSize.mul( ivec2( x, y ) ), size: iViewportSize } );

	const DRAW_VIEWPORT = ( fn: ( b: Fsr3Bindings, p: N, v: FfxDebugViewport ) => void, v: FfxDebugViewport ) => {

		If( pointIsInsideViewport( iPxPos, v ), () => {

			fn( b, iPxPos, v );

		} );

	};

	// top row
	DRAW_VIEWPORT( drawDilatedMotionVectors, vp( 0, 0 ) );
	DRAW_VIEWPORT( drawProtectedAreas, vp( 1, 0 ) );
	DRAW_VIEWPORT( drawDilatedDepthInMeters, vp( 2, 0 ) );

	// bottom row
	DRAW_VIEWPORT( drawDisocclusionMask, vp( 0, 2 ) );
	DRAW_VIEWPORT( drawReactiveness, vp( 1, 2 ) );
	DRAW_VIEWPORT( drawDetailProtectionTakedown, vp( 2, 2 ) );

}
