import { PRE_EXPOSURE_PENDING } from './PreExposure.ts';
import { TemporalRest, restKey } from './TemporalRest.ts';
import { onSceneDispose } from '../SceneDispose.ts';

// The page's settled signal: true once a paused page will draw the same frame from now on.
//
// The page's settled signal (`__THREE_GAME_DIAGNOSTICS__.settled`, debug builds): true once a paused
// page will draw the same frame from now on, under every anti-aliasing mode, so a capture can be
// taken as soon as it holds instead of after a fixed wait or frame count (a bake that lands late
// on a slow or shared GPU defeats any count).
//
// It reuses the temporal rest's machinery (TemporalRest.ts): a TemporalRest of its own (no settle
// frames, a "sequence" as long as the run it needs) observes, every frame, the same key the rest
// is tested on (`restKey`: the camera's unjittered view and projection, the scene clock, the
// render's and the output's sizes, the epoch) plus the page's signature, and counts the frames in a
// row that share it while the page is stable and quiet:
// - stable: no sky bake in flight, no sky transition running (the temporal rest asks that too, and
//   that the haze's rest has ended: RenderStages.ts `temporalFrame`);
// - quiet: nothing that will change the frame without changing the key is pending (main.ts names
//   them in `pending`: a periodic sky bake due, a periodic bake's irradiance still being read back
//   after it has shown, the refracted sun's map behind its light, a sun shadow redraw queued), and no render or compute pipeline is being compiled asynchronously
//   (`watchPipelines`, installed by main.ts before anything compiles, counts the device's
//   `create*PipelineAsync` calls in flight: three skips a draw whose pipeline is not ready yet, so
//   a frame drawn meanwhile lacks it);
// - the signature holds the redraw counts of the stepped and refracted sun maps, so a redraw
//   (after a bake, an invalidation) starts the run over too.
// Under MSAA (and aa=off) the page is settled once that run reaches the tier's `restSettle` frames
// (8 on every tier): what settles on its own after the key stops changing (the clouds' view on its
// second still frame, a bake's products landing) has then settled, as the temporal rest assumes
// for its own settle frames. Under the temporal modes (fsr3's rest included: Pipeline passes the
// one its resolver owns) it is settled when the temporal rest has
// converged and the quiet run began no later than the rest's sequence did (the rest's start plus
// its settle frames), so every frame the rest's image averaged was drawn from the same, quiet
// scene. The haze's rest is such a pending refresh, one every epoch brings (its 15 to 20 frames
// would put the quiet run past the sequence's start for good), so the temporal rest itself waits
// for it, and both runs start on the same frame. A refresh that lands under a held rest image, or during its sequence, leaves the signal
// false until the rest starts over (the held image is no longer the scene's): the diagnostics then
// report `stale` (the rest converged, the quiet run long enough, the run too late), and a tool bumps
// the epoch (`__PONLAI__.taa.epoch.bump()`; the capture tools bump it before they wait anyway).
// With `restSequence` 0 (no rest) the temporal modes never settle.
// Whatever the mode, the page also waits for what covers the canvas without changing its frames
// (`shown`: the loading screen until its fade has ended; a paused page lifts it at once, main.ts).
// A converged rest alone is not enough for a capture: rests that converged about 40 frames after a
// view's eval were captured through the fade's last percent, and only this flag told them apart.
//
// It turns false on the frame anything it watches changes: a camera move, a resize or a render
// scale change, a seek or any scene event (each bumps the epoch: SceneEvents.ts), `establishScene`
// and every other bake (the sky is busy), a forced strike, a redraw of either sun map. A running
// clock changes the key every frame, so a running page never settles.
//
// The pre-exposure (PreExposure.ts) is pending in its own way: it waits for a reading of a frame
// drawn the settle frames into this run, so the run counts the frames that wait for nothing else
// (`sceneQuiet`: stable, no pipeline compiling, nothing but the pre-exposure in `pending`), and the
// signal holds only once the pre-exposure is not pending as well. Were the run to wait for it too,
// it would start only after a reading of a frame already the settle frames into it had come back:
// under the temporal modes then later than the rest's sequence began, so never `covered`, and a
// tool's bump on `stale` started both over again. A new P bumps the epoch, so the run (and under
// the temporal modes the rest) starts again on frames of the P it settles on.
//
// It reads state and changes nothing the frame draws (the pipeline counter wraps the device's
// two async creation calls and passes them through unchanged).

/** The app's per-frame inputs (main.ts). */
export interface SettleFrame {
	// the app's own signature: a forced strike's version, the sun maps' redraw counts
	signature: readonly number[];
	// false while a sky bake is in flight or a sky transition runs (the temporal rest's `stable`)
	stable: boolean;
	// the refreshes that will change the frame without changing the key (empty: none)
	pending: readonly string[];
	// nothing covers the canvas (the loading screen has faded out)
	shown: boolean;
}

type AsyncCreate = ( ...args: unknown[] ) => Promise<unknown>;

// the device's async pipeline creations in flight (null: not watched)
let compiling: number | null = null;
// the next scene's renderer has a device of its own to watch
onSceneDispose( 'pipelines watched', () => { compiling = null; } );

/** Counts the device's async pipeline creations in flight, from now on (once per page). The two
 * calls are passed through unchanged. */
export function watchPipelines( device: unknown ): void {

	if ( compiling !== null ) return;
	compiling = 0;
	const d = device as Record<'createRenderPipelineAsync' | 'createComputePipelineAsync', AsyncCreate>;
	for ( const name of [ 'createRenderPipelineAsync', 'createComputePipelineAsync' ] as const ) {

		const original = d[ name ];
		if ( typeof original !== 'function' ) continue;
		d[ name ] = ( ...args: unknown[] ) => {

			const created = original.apply( device, args );
			compiling = ( compiling ?? 0 ) + 1;
			const done = () => { compiling = ( compiling ?? 1 ) - 1; };
			created.then( done, done );
			return created;

		};

	}

}

export class Settled {

	/** The run MSAA and aa=off need (the tier's `restSettle`). */
	settle = 8;
	/** The quiet run: consecutive frames with the same key while stable and quiet. */
	readonly rest = new TemporalRest();
	/** The signal. */
	value = false;
	/** The frame (frames observed since the loop started, as `diag.frame` counts them) it last
	 * turned true on, and the times it turned true. */
	since: number | null = null;
	turns = 0;
	/** Under the temporal modes: the rest converged on frames that were not all quiet (see above). */
	stale = false;
	/** The last frame was stable and waited for nothing but the pre-exposure: no other refresh
	 * pending, no pipeline compiling. The quiet run counts such frames, and PreExposure.ts takes
	 * the readings that end its `pending` from them. */
	sceneQuiet = false;
	private frame = 0;
	private quietStart = 0;
	private restStart = 0;
	private pending: readonly string[] = [];
	private stable = false;
	private shown = false;
	private readonly key: number[] = [];

	/** Once per rendered frame, after the temporal rest observed it (`temporal`: that rest, under the
	 * temporal modes). */
	observe( camera: Parameters<typeof restKey>[ 1 ], renderWidth: number, renderHeight: number, outputWidth: number, outputHeight: number, time: number,
		input: SettleFrame, temporal: TemporalRest | null ): void {

		this.frame ++;
		this.pending = input.pending;
		this.stable = input.stable;
		this.shown = input.shown;
		const key = restKey( this.key, camera, renderWidth, renderHeight, outputWidth, outputHeight, time, input.signature );
		const r = this.rest;
		r.settle = 0;
		r.sequence = Math.max( 1, temporal ? temporal.sequence : this.settle );
		// (the run leaves the pre-exposure out: it waits for frames of this run, see the header)
		this.sceneQuiet = input.stable && ! compiling && input.pending.every( ( p ) => p === PRE_EXPOSURE_PENDING );
		r.observe( key, this.sceneQuiet );
		if ( r.frames === 1 ) this.quietStart = this.frame;
		if ( temporal && temporal.frames === 1 ) this.restStart = this.frame;
		const was = this.value;
		const covered = temporal !== null && this.quietStart <= this.restStart + temporal.settle;
		this.stale = temporal !== null && temporal.converged && r.converged && ! covered;
		this.value = input.shown && ! input.pending.includes( PRE_EXPOSURE_PENDING ) && ( temporal ? temporal.converged && r.converged && covered : r.converged );
		if ( this.value && ! was ) {

			this.since = this.frame;
			this.turns ++;

		}

	}

	/** The diagnostics' `settle` block. */
	diagnostics(): Record<string, unknown> {

		return {
			settled: this.value,
			// the quiet run and the run it needs (under the temporal modes the rest's sequence, and the
			// rest's own convergence besides)
			frames: this.rest.frames,
			need: this.rest.sequence,
			since: this.since,
			turns: this.turns,
			stable: this.stable,
			shown: this.shown,
			stale: this.stale,
			pending: compiling ? [ ...this.pending, `compiling ${ compiling }` ] : this.pending,
		};

	}

}
