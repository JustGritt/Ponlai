/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_sample.h and ffx_fsr3upscaler_reproject.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { Fn, float, vec2, vec4, ivec2, texture, textureLoad, floor, min, max, clamp, sin, cos, select } from 'three/tsl';

// The running resolve's history fetch: FSR 3.1's clamped Lanczos 2 sample.
//
// The running resolve's history fetch (TemporalAA.ts: taa, taau): FSR 3.1's history sample, a
// Lanczos 2 over the 4 x 4 texels around the reprojected position, clamped to the 2 x 2 nearest
// texels (upstream's deringing). All four channels go through it, the accumulated weight in alpha
// too.
//
// Why: a history reprojected at a fractional offset every frame (a pan, the swaying meadow, a
// walking islander) is resampled again each frame, and the kernel's spread accumulates over the
// history's frames into a soft trail of where things were, which the clip box (built from the
// current frame) does not cut off. The Catmull-Rom in five bilinear taps this replaces (Jimenez,
// "Filmic SMAA", SIGGRAPH 2016) spreads each resample a little more than a Lanczos 2 does. Measured
// against the ground truth at high taau in motion (the islander walk and orbit, both slow pans and
// the sun-shadow clip): ghosting 0.171 -> 0.136 (fsr3's 0.144, MSAA's 0.150), the walking islander's
// 0.419 -> 0.322 (MSAA's 0.320), the shimmer (TI) 1 % lower and the high-pass flicker 0.5 % lower;
// the per-frame error (FLIP) rose 3 % over the clips, 4-13 % on the pans' crops (the island pan's
// islander head 0.075 -> 0.085), still below fsr3's and MSAA's on every crop (upstream's 16 texels
// gave two thirds of that rise). Without the deringing the ghosting was the same but the shimmer and
// flicker went back to the Catmull-Rom's. A
// still pixel's fetch is exact (the nearest weight 1, every other 0), so a still camera's image and
// the rest do not change.
//
// Cost (profile-frame on the high tier at the baseline, the resolve at the 4K canvas): upstream's 16 texel loads
// took about 0.9 ms more than the five bilinear taps, whatever the weights cost (FSR 1's polynomial
// approximation saved nothing there): the loads bound it. Here the 2 x 2 nearest texels are loaded
// (the deringing needs them) and each edge's two middle texels are one bilinear tap at their weights'
// ratio (both weights are positive), as Jimenez's layout does; the four corners are left out (their
// weights, products of two negative lobes, are at most 0.75 % each). Eight fetches, and the same
// ghosting, shimmer and flicker as the 16 within the clips' noise, at about 0.25 ms over the
// Catmull-Rom. FSR 1's approximation is not used: its lobes are deeper (-0.105 against -0.064 at
// 1.5 texels), and in five taps without the deringing it raised the shimmer by a tenth and FLIP by a
// quarter.
//
// Differences from upstream:
// - Only the history sample is ported (`HistorySample` with `Lanczos2`, through
//   `CustomTextureSample` and `FetchBicubicSamples`); no FP16 variants, no LUT, no approximation.
// - Eight fetches instead of sixteen (above).
// - Each axis's four weights come from one sine and one cosine (`lanczos2Weights`), the same values
//   as upstream's eight sines.

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V4 = THREE.Node<'vec4'>;

// upstream's FSR3UPSCALER_EPSILON (half float's smallest normal)
const EPSILON = 6.10e-05;

/** Lanczos 2's four weights at the taps -1, 0, 1, 2 texels from a sample a fraction f (0..1) past
 * the second: L( x ) = sin( pi x ) sin( pi x / 2 ) / ( pi^2 x^2 / 2 ) at x = k - f, whose sines are,
 * for every k, +-sin( pi f ) = +-2 s c and +-s or +-c (s and c the sine and cosine of pi f / 2). */
const lanczos2Weights = /*@__PURE__*/ Fn( ( [ f ]: [ F ] ) => {

	const h = f.mul( 0.5 * Math.PI ).toVar();
	const s = sin( h ).toVar(), c = cos( h ).toVar();
	const k = s.mul( c ).mul( 4 / ( Math.PI * Math.PI ) ).toVar();
	const x = vec4( float( - 1 ).sub( f ), f.negate(), float( 1 ).sub( f ), float( 2 ).sub( f ) ).toVar();
	const w = vec4( c.negate(), s, c, s.negate() ).mul( k ).div( x.mul( x ) ).toVar();
	// at f = 0 the nearest tap is the sample itself (sinc's limit)
	w.y.assign( select( f.lessThan( EPSILON ), float( 1 ), w.y ) );
	return w;

} ).setLayout( { name: 'taaLanczos2Weights', type: 'vec4', inputs: [ { name: 'f', type: 'float' } ] } );

/** The history at `uv` (0..1 over a history of `size` texels): Lanczos 2 over the 4 x 4 texels
 * around it (the corners left out) and clamped to the 2 x 2 nearest. The history's sampler filters
 * linearly. */
export function lanczosHistory( history: THREE.TextureNode, uv: V2, size: V2 ): V4 {

	const p = uv.mul( size ).sub( 0.5 ).toVar();
	const f = p.sub( floor( p ) ).toVar();
	const last = size.sub( 1 ).toVar();
	const base = floor( clamp( p, vec2( 0 ), last ) ).toVar();
	const wx = lanczos2Weights( f.x ).toVar(), wy = lanczos2Weights( f.y ).toVar();
	// the 2 x 2 nearest texels (upstream's ClampCoord at the history's edges)
	const load = ( x: number, y: number ) => ( textureLoad( history, ivec2( clamp( base.add( vec2( x, y ) ), vec2( 0 ), last ) ) ) as unknown as V4 ).toVar();
	const c00 = load( 0, 0 ), c10 = load( 1, 0 ), c01 = load( 0, 1 ), c11 = load( 1, 1 );
	// each edge's middle pair, one bilinear tap between its texel centres (base + k + 0.5)
	const mx = wx.y.add( wx.z ).toVar(), my = wy.y.add( wy.z ).toVar();
	const px = base.x.add( 0.5 ).add( wx.z.div( mx ) ).toVar(), py = base.y.add( 0.5 ).add( wy.z.div( my ) ).toVar();
	const at = ( x: F, y: F ) => texture( history, vec2( x, y ).div( size ) ).level( float( 0 ) ) as unknown as V4;
	const kTop = mx.mul( wy.x ), kBottom = mx.mul( wy.w ), kLeft = wx.x.mul( my ), kRight = wx.w.mul( my );
	const sum = c00.mul( wx.y.mul( wy.y ) ).add( c10.mul( wx.z.mul( wy.y ) ) ).add( c01.mul( wx.y.mul( wy.z ) ) ).add( c11.mul( wx.z.mul( wy.z ) ) )
		.add( at( px, base.y.sub( 0.5 ) ).mul( kTop ) ).add( at( px, base.y.add( 2.5 ) ).mul( kBottom ) )
		.add( at( base.x.sub( 0.5 ), py ).mul( kLeft ) ).add( at( base.x.add( 2.5 ), py ).mul( kRight ) );
	const norm = mx.mul( my ).add( kTop ).add( kBottom ).add( kLeft ).add( kRight );
	// the deringing
	return clamp( sum.div( norm ), min( min( c00, c10 ), min( c01, c11 ) ), max( max( c00, c10 ), max( c01, c11 ) ) );

}
