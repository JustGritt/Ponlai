/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/fsr1/
 * ffx_fsr1.h, the RCAS part).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import { float, vec2, vec3, ivec2, abs, max, min } from './Tsl.ts';
import {
	ffxAsFloat, ffxSaturate, ffxApproximateReciprocalMedium, ffxReciprocal, ffxMax3, ffxMin3,
	ffxAsUInt32Cpu, ffxPackHalf2x16, type N,
} from './Core.ts';

// FSR 1's robust contrast-adaptive sharpening (RCAS) as the FSR 3 upscaler includes it.
//
// FSR 1's robust contrast-adaptive sharpening (RCAS) as the FSR 3 upscaler includes it: the host's
// `FsrRcasCon` and the 32-bit `FsrRcasF` (FSR_RCAS_F), with FSR_RCAS_DENOISE on as
// ffx_fsr3upscaler_rcas.h defines it, and FSR_RCAS_PASSTHROUGH_ALPHA off (the upscaler does not
// define it). RCAS solves for the most sharpening a 5-tap cross allows before the result clips,
// limits it on what looks like noise, and resolves with the medium-precision reciprocal "to avoid
// visible tonality changes".
//
// Differences from upstream:
// - The 16-bit and paired versions (FsrRcasH, FsrRcasHx2) and the EASU part of ffx_fsr1.h are not
//   ported (32-bit only).
// - `FsrRcasF`'s out parameters become a returned vec3; its callbacks (`FsrRcasLoadF`,
//   `FsrRcasInputF`) are passed in, as the including file defines them (Rcas.ts).
// - HLSL's `rcp` in `ffxReciprocal` is an exact 1 / x here (Core.ts).

export const FSR_RCAS_LIMIT = 0.25 - ( 1.0 / 16.0 );

/** FsrRcasCon: the constants for a sharpness in stops ( 0 the most, N > 0 the stops of reduction ). */
export function FsrRcasCon( sharpness: number ): [ number, number, number, number ] {

	// Transform from stops to linear value.
	sharpness = Math.fround( Math.pow( 2, Math.fround( - sharpness ) ) );
	const hSharp: [ number, number ] = [ sharpness, sharpness ];
	return [ ffxAsUInt32Cpu( sharpness ), ffxPackHalf2x16( hSharp ), 0, 0 ];

}

/** FsrRcasF: the sharpened colour at `ip`. `FsrRcasLoadF( p )` returns the input texel (vec4) and
 * `FsrRcasInputF( c )` the optional input transform of a colour (vec3). */
export function FsrRcasF( ip: N, con: N, FsrRcasLoadF: ( p: N ) => N, FsrRcasInputF: ( c: N ) => N, denoise = true ): N {

	// Algorithm uses minimal 3x3 pixel neighborhood.
	//    b
	//  d e f
	//    h
	const sp = ip;
	const b = FsrRcasInputF( FsrRcasLoadF( sp.add( ivec2( 0, - 1 ) ) ).rgb ).toVar();
	const d = FsrRcasInputF( FsrRcasLoadF( sp.add( ivec2( - 1, 0 ) ) ).rgb ).toVar();
	const e = FsrRcasInputF( FsrRcasLoadF( sp ).rgb ).toVar();
	const f = FsrRcasInputF( FsrRcasLoadF( sp.add( ivec2( 1, 0 ) ) ).rgb ).toVar();
	const h = FsrRcasInputF( FsrRcasLoadF( sp.add( ivec2( 0, 1 ) ) ).rgb ).toVar();

	// Rename (32-bit) or regroup (16-bit).
	const bR = b.r, bG = b.g, bB = b.b;
	const dR = d.r, dG = d.g, dB = d.b;
	const eR = e.r, eG = e.g, eB = e.b;
	const fR = f.r, fG = f.g, fB = f.b;
	const hR = h.r, hG = h.g, hB = h.b;

	// Luma times 2.
	const bL = bB.mul( 0.5 ).add( bR.mul( 0.5 ).add( bG ) ).toVar();
	const dL = dB.mul( 0.5 ).add( dR.mul( 0.5 ).add( dG ) ).toVar();
	const eL = eB.mul( 0.5 ).add( eR.mul( 0.5 ).add( eG ) ).toVar();
	const fL = fB.mul( 0.5 ).add( fR.mul( 0.5 ).add( fG ) ).toVar();
	const hL = hB.mul( 0.5 ).add( hR.mul( 0.5 ).add( hG ) ).toVar();

	// Noise detection.
	let nz = bL.mul( 0.25 ).add( dL.mul( 0.25 ) ).add( fL.mul( 0.25 ) ).add( hL.mul( 0.25 ) ).sub( eL ).toVar();
	nz = ffxSaturate( abs( nz ).mul( ffxApproximateReciprocalMedium( ffxMax3( ffxMax3( bL, dL, eL ), fL, hL ).sub( ffxMin3( ffxMin3( bL, dL, eL ), fL, hL ) ) ) ) ).toVar();
	nz = nz.mul( - 0.5 ).add( 1.0 ).toVar();

	// Min and max of ring.
	const mn4R = min( ffxMin3( bR, dR, fR ), hR ).toVar();
	const mn4G = min( ffxMin3( bG, dG, fG ), hG ).toVar();
	const mn4B = min( ffxMin3( bB, dB, fB ), hB ).toVar();
	const mx4R = max( ffxMax3( bR, dR, fR ), hR ).toVar();
	const mx4G = max( ffxMax3( bG, dG, fG ), hG ).toVar();
	const mx4B = max( ffxMax3( bB, dB, fB ), hB ).toVar();

	// Immediate constants for peak range.
	const peakC = vec2( 1.0, - 1.0 * 4.0 );

	// Limiters, these need to be high precision RCPs.
	const lowerLimiterMultiplier = ffxSaturate( eL.div( min( ffxMin3( bL, dL, fL ), hL ) ) ).toVar();
	const hitMinR = mn4R.mul( ffxReciprocal( float( 4.0 ).mul( mx4R ) ) ).mul( lowerLimiterMultiplier );
	const hitMinG = mn4G.mul( ffxReciprocal( float( 4.0 ).mul( mx4G ) ) ).mul( lowerLimiterMultiplier );
	const hitMinB = mn4B.mul( ffxReciprocal( float( 4.0 ).mul( mx4B ) ) ).mul( lowerLimiterMultiplier );
	const hitMaxR = peakC.x.sub( mx4R ).mul( ffxReciprocal( float( 4.0 ).mul( mn4R ).add( peakC.y ) ) );
	const hitMaxG = peakC.x.sub( mx4G ).mul( ffxReciprocal( float( 4.0 ).mul( mn4G ).add( peakC.y ) ) );
	const hitMaxB = peakC.x.sub( mx4B ).mul( ffxReciprocal( float( 4.0 ).mul( mn4B ).add( peakC.y ) ) );
	const lobeR = max( hitMinR.negate(), hitMaxR );
	const lobeG = max( hitMinG.negate(), hitMaxG );
	const lobeB = max( hitMinB.negate(), hitMaxB );
	let lobe = max( float( - FSR_RCAS_LIMIT ), min( ffxMax3( lobeR, lobeG, lobeB ), float( 0.0 ) ) ).mul( ffxAsFloat( con.x ) ).toVar();

	// Apply noise removal.
	if ( denoise ) lobe = lobe.mul( nz ).toVar();

	// Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
	const rcpL = ffxApproximateReciprocalMedium( lobe.mul( 4.0 ).add( 1.0 ) ).toVar();
	const pixR = lobe.mul( bR ).add( lobe.mul( dR ) ).add( lobe.mul( hR ) ).add( lobe.mul( fR ) ).add( eR ).mul( rcpL );
	const pixG = lobe.mul( bG ).add( lobe.mul( dG ) ).add( lobe.mul( hG ) ).add( lobe.mul( fG ) ).add( eG ).mul( rcpL );
	const pixB = lobe.mul( bB ).add( lobe.mul( dB ) ).add( lobe.mul( hB ) ).add( lobe.mul( fB ) ).add( eB ).mul( rcpL );
	return vec3( pixR, pixG, pixB );

}
