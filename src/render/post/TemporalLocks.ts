/*!
 * Adapted from tidewater's TemporalUpscale.js (https://github.com/dgreenheck/tidewater, commit 4811ba4),
 * itself the accumulation of AMD FidelityFX Super Resolution 2 2.2 (ffx_fsr2_accumulate.h).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { float, max, min, mix, abs, select, saturate } from 'three/tsl';

// FSR 2's accumulation measures for moving pixels, as the running resolve uses them.
//
// FSR 2's accumulation measures for moving pixels, as TemporalAA.ts's running resolve uses them
// (TemporalAA.ts's header: "Movers"). Ours keeps its history's weight in frames' worth of a frame's
// weight (FSR 2: in Lanczos sample weights), so FSR 2's `upsampledWeight` is one frame here and its
// accumulation cap a number of frames. The resolve works on display values in 0..1 (FSR 2 before
// tone mapping), which none of these terms depend on.
// - The motion cap (ComputeBaseAccumulationWeight): a moving pixel, or one whose history was
//   written by a moving pixel (the temporal reactive's sign, below), keeps at most `capFrames`; the
//   cap eases to one frame at 20 output pixels a frame. The carried reactive scales the capped
//   frames (FSR 2 scales the frames before the cap, so a reactive pixel's cap stayed whole there).
// - The temporal reactive (ComputeTemporalReactiveFactor): a per-pixel factor carried to the next
//   frame (0.4 at a pixel a frame or more, squared as it is carried, so a steady mover keeps 0.84 of
//   its frames), negative where the pixel moved 0.1 output pixels or more. FSR 2 sets it to 1 where
//   the pixel has no history (a new sample), which drops the next frame's history too; ours keeps
//   the disocclusion's tenth there. Ours leaves the per-pixel reactive inputs (bolts, the flash, the
//   veil) out of it: they already cut the frames where they are, and carried over they would hold
//   a bolt's cut a frame longer.
// - The clamp-drop (RectifyHistory): a clipped history drops its weight, so the frames that follow
//   take over. FSR 2 drops it to a tenth of a frame on any clip, and needs its locks and luma
//   instability to keep a still camera's thin detail; ours drops it to one frame, by how far outside
//   the box the history lay, and only where the pixel moves (the caller's `moving`), so a still
//   camera's jitter clips keep their weight as before.

type F = THREE.Node<'float'>;

/** The frames' worth `frames` capped while the pixel moves (`moving` 0..1: this frame or the last),
 * at `speed` output pixels a frame; `capFrames` FSR 2's ten, scaled by the caller's class share. */
export const motionCap = ( frames: F, capFrames: F, moving: F, speed: F, temporalReactive: F ): F => {

	const a = min( frames, mix( frames, capFrames, moving ) ).toVar();
	const b = min( a, mix( a, float( 1 ), saturate( speed.div( 20 ) ) ) );
	return b.mul( float( 1 ).sub( temporalReactive ) );

};

/** The temporal reactive to carry to the next frame, from the one carried here (`previous`, its
 * absolute value), the pixel's speed (output pixels a frame) and whether its history went
 * (`disoccluded` 0..1); negative where the pixel moved 0.1 output pixels or more. */
export const temporalReactive = ( previous: F, speed: F, disoccluded: F ): F => {

	const f = min( previous, 0.99 ).toVar();
	f.assign( max( f, mix( f, float( 0.4 ), saturate( speed ) ) ) );
	f.assign( max( f.mul( f ), disoccluded.mul( 0.1 ) ) );
	// (the sign survives half float: at least 1e-3)
	return select( speed.greaterThanEqual( 0.1 ), max( f, 1e-3 ).negate(), f );

};

/** The history's weight after a clip, dropped by `amount` (0..1: the caller's, from how far outside
 * the box the history lay and whether the pixel moves) to at most `floor`. */
export const clampDrop = ( weight: F, floor: F, amount: F ): F => mix( weight, min( weight, floor ), amount );

/** Whether the carried temporal reactive says the pixel moved last frame (1) or not (0). */
export const movedLastFrame = ( carried: F ): F => select( carried.lessThan( 0 ), float( 1 ), float( 0 ) );

/** The carried temporal reactive's size. */
export const carriedReactive = ( carried: F ): F => abs( carried );
