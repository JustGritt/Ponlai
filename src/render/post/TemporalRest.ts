// The temporal modes' rest convergence: a scene at rest converges once, from scratch.
//
// The temporal modes' rest convergence (Antialiasing.ts: aa=taa|taau|gt, and aa=fsr3, whose upscaler
// drives a rest of its own the same way: post/fsr3/Fsr3Upscaler.ts). A running temporal
// resolve is a function of the frames before it, so a paused frame would depend on how the page got
// there: the history's weights, the jitter's place in its sequence, the camera's last moves. A scene
// at rest (a paused clock, a still camera, nothing else changing) therefore converges once, from
// scratch, the way the clouds settle (Clouds.ts): after `settle` frames at rest, for everything that
// settles on its own (the clouds' view, the refracted sun's map, a bake landing) to be done, a fixed
// sequence of `sequence` jitters (Halton indices 1..N centred on their mean, Jitter.ts `restIndex`;
// the vegetation's dither takes N mod 16) is averaged with exact weights into an image of its own (TemporalAA.ts: the
// rest resolve, no clip, no cap, nothing reprojected), shown from the sequence's last frame on and
// held while the rest lasts, the jitter staying at the sequence's last index. The running history
// keeps updating underneath, so the frame after the rest ends continues from it. What the image shows then depends only on the scene: a paused frame
// is bit-identical, however it was reached.
//
// The rest is the canvas's image: what the page lays over the canvas is not in it. A rest captured while
// the loading screen still faded out (index.html) carried up to 2 % of that screen's colour over the
// whole frame and read as a darker rest; a paused page now lifts the loading screen at once
// (assemble/StartScene.ts `reveal`), and the page's settled signal waits for it in any case (Settled.ts `shown`).
// The settle frames do cover the scene: rests started one frame after a view's eval and 300 frames
// later were byte-identical once nothing lay over them (no sun map redraw, bake or pending refresh
// fell inside either rest).
//
// The rest is tested every frame on a key of what the frame draws from: the camera's unjittered view
// and projection, the scene clock, the render's and the output's sizes, the epoch, the resolve's live
// settings and the app's own signature of the rest (main.ts: a forced lightning strike, which a
// review sets while paused). The frame is not at rest while the app says it is unstable (a sky bake
// in flight, a sky transition still running, the haze's own rest not done: VolumetricHaze.ts).
// Anything else that changes the image without changing
// the key (a GUI control, a weather jump, a seek to the same time) bumps the epoch
// (`restEpoch.bump( reason )`): the scene events (SceneEvents.ts `renderEvent`) call it from every
// site that changes the scene, `__PONLAI__.taa.restart()` among them.
//
// The key is built by `restKey`, which the page's settled signal (Settled.ts) shares under every
// anti-aliasing mode, MSAA included.

/** Fills `key` with what a frame draws from: the camera's unjittered view and projection, the
 * scene clock, the render's and the output's sizes, the epoch (`restEpoch.value` unless given:
 * PreExposure.ts leaves its own bumps out), then `extra` (the caller's own signature). */
export function restKey( key: number[], camera: { matrixWorld: { elements: readonly number[] }; projectionMatrix: { elements: readonly number[] } },
	renderWidth: number, renderHeight: number, outputWidth: number, outputHeight: number, time: number, extra: readonly number[], epoch = restEpoch.value ): number[] {

	key.length = 0;
	key.push( ...camera.matrixWorld.elements, ...camera.projectionMatrix.elements, time, renderWidth, renderHeight, outputWidth, outputHeight,
		epoch, ...extra );
	return key;

}

/** The epoch of the scene's state: a bump restarts the rest (and nothing else). `scene` counts
 * only the bumps that may change what a still frame draws: a restart of the frame loop or the page
 * shown again leaves it (a rest of the scene alone, the haze's, survives the capture tools'
 * stopping and starting the loop for each grab; the temporal resolve's own rest starts over). */
export const restEpoch = {
	value: 0,
	scene: 0,
	// the last bump's reason, for the diagnostics
	reason: null as string | null,
	bump( reason: string, scene = true ): void {

		this.value ++;
		if ( scene ) this.scene ++;
		this.reason = reason;

	},
};

export class TemporalRest {

	/** Frames at rest before the sequence starts, and the sequence's length (0: no rest; the
	 * settings' `restSettle` and `restSequence`, or `gtJitters` for the ground truth). */
	settle = 8;
	sequence = 32;
	/** Consecutive frames rendered with the same key, this one included, up to `target` (0 while the
	 * page is unstable). */
	frames = 0;
	/** The sequence's index this frame (0-based), or null outside it. */
	index: number | null = null;
	/** Whether this frame completes the sequence (its image is shown from this frame on). */
	completing = false;
	/** The rest's image is complete and shown. */
	converged = false;
	/** Rests completed, and restarts (keys that changed while at rest) since the page loaded. */
	completed = 0;
	restarts = 0;
	private key: number[] = [];
	private keyed = false;

	/** The frames a rest takes until its image shows (`frames` stays there while it holds). */
	get target(): number {

		return this.sequence > 0 ? this.settle + this.sequence : 0;

	}

	/** Once per rendered frame, before its jitter is chosen: the frame's key and whether the app is
	 * stable. Sets `index` (the jitter's rest index is index + 1), `completing` and `converged`. */
	observe( key: readonly number[], stable: boolean ): void {

		const same = this.keyed && stable && key.length === this.key.length && key.every( ( v, i ) => Object.is( v, this.key[ i ] ) );
		this.key.length = key.length;
		for ( let i = 0; i < key.length; i ++ ) this.key[ i ] = key[ i ];
		this.keyed = true;
		if ( ! same && this.frames > 1 ) this.restarts ++;
		const target = this.target;
		this.frames = ! stable ? 0 : same ? Math.min( this.frames + 1, Math.max( target, 1 ) ) : 1;
		if ( ! same ) this.converged = false;
		this.completing = false;
		this.index = null;
		if ( this.sequence <= 0 || this.converged || this.frames <= this.settle ) return;
		this.index = this.frames - this.settle - 1;
		if ( this.index === this.sequence - 1 ) {

			this.completing = true;
			this.converged = true;
			this.completed ++;

		}

	}

	/** Starts over (the next frame is the first at rest). */
	restart(): void {

		this.keyed = false;
		this.frames = 0;
		this.index = null;
		this.completing = false;
		this.converged = false;

	}

}
