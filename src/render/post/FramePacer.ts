// Frame pacing: renders on a steady divisor of the display's refresh.
//
// Frame pacing: renders on a steady divisor of the display's refresh, so a display faster than the
// frame rate the scene holds shows every frame for the same number of refreshes.
//
// Chrome runs `requestAnimationFrame` at the display's refresh (120 Hz on a ProMotion display). A
// frame that keeps the GPU busy for 12-16 ms then finishes one, two or three refreshes after the
// previous one, as the queue happens to drain: on such a display the frames came 8.3, 16.7 and
// 25 ms apart in an irregular mix (about a sixth at 8.3 ms, a third of the intervals differing
// from the one before), which reads as stutter while the camera moves even where the average is
// above 60 fps. Here the loop renders only on the refreshes that keep the interval at the target
// (every second refresh at 120 Hz for 60 fps) and skips the others before three's animation loop
// sees them, so the scene clock, the camera's damping and the GPU all advance in equal steps.
//
// The refresh period is estimated from the animation frames' own timestamps (the vblank times the
// browser hands to rAF): the median of the shortest intervals between consecutive callbacks,
// rendered or skipped. The divisor is the whole number of refreshes nearest the target interval, never
// more than a quarter above it (so 144 Hz holds 72 fps, 165 Hz 55 and 90 Hz renders every refresh); at
// 60 Hz or a target of 0 every callback renders, as without the pacer. The estimate moves by a fraction
// of a percent from frame to frame, and 165 Hz sits right on a boundary of that rule (2.75 refreshes
// per 60 fps frame, plus the quarter, is 3): the divisor changes only once the estimate puts it at
// least BOUNDARY (1 %) past a boundary, and the first estimate (or a new rate) counts a boundary
// within BOUNDARY as reached, so 165 Hz holds 55 from the start instead of flipping 55 <-> 82.5.
//
// High and medium hold 60 (Quality.ts `frameRate`). Low renders on every refresh (`frameRate: 0`):
// its light frames, held to 60 Hz, finished right at the compositor's deadline and were shown one
// and three refreshes apart in turn. The scene's clock steps by the frames' vblank times (rAF's
// timestamp), not by when the tick starts (a few milliseconds into the frame, by a varying
// amount), and the camera applies its input once per rendered frame (IslandCameraControls'
// `advance`): three's OrbitControls damped on every event, so a frame took one to three damping
// steps by chance. A hidden page renders nothing (main.ts stops the loop).
//
// The rate (`mode`): 'auto' (the default wherever the tier has a target) holds the tier's divisor
// (`base`) and may fall back to the next one, `base + 1`, when the frames cannot make the faster
// rate: on a 144 Hz display 72 fps falls back to 48, on 120 Hz 60 to 40, on 165 Hz 55 to 41.25. A
// 4K canvas on a 144 Hz display missed 72 fps at the dynamic scale's floor once the camera moved
// (30-70 % of the frames shown a refresh late, the main source of late frames): 48 fps shown at an
// even three refreshes each reads steadier than 72 with a third of its frames held for three. The
// fallback is a whole divisor, so the frames stay evenly paced (fixed-refresh consoles do the same:
// a 40 fps mode on 120 Hz displays); browsers offer no variable refresh to fall back on instead.
// There is no fallback below MIN_FALLBACK_FPS (40): under it the camera's motion reads as steps,
// so a 60 Hz display stays at 60 (its next divisor is 30) and the scale's floor takes the load.
// The pacer does not decide when to fall: the dynamic scale does (DynamicResolution.ts, which
// knows its floor, its frame times and its cost model), through `fallback()`, with hysteresis
// (it falls when frames still miss at its floor and returns only once the faster interval is
// predicted to fit with room, for some seconds, waiting longer each time a return failed). No fall
// happens, and `mode` still reads 'auto', where the scale does not steer: paused, pinned (`?dpr`,
// the GUI's full resolution), stepped (`loop.step`), without GPU timestamps (the scale then steers
// by frame intervals and has no cost model), on a tier without the dynamic scale (medium), or under
// `aa=gt`; a page already fallen stays at the fallback while it does not steer.
// A fixed rate (`setRate( fps )`, `?frameRate=<fps>`) never steps. It is a cap: the fastest
// whole-divisor rate not above fps (1 % of tolerance), every refresh when fps is at or above the
// refresh, picked again when the refresh's estimate changes (a fixed 72 on a window moved to a
// 120 Hz display holds 60). For a settings menu: `choices()` lists the rates the display holds
// evenly (read it when the menu opens: until about 30 animation frames have measured the refresh it
// lists a 60 Hz display's [60, 30]), `mode` and `rate` the choice ('auto', or the fixed fps), `fps`
// the rate held now, `setRate()` sets it at any time between frames. The pacer counts its own
// changes (`generation`: a new rate, a new divisor from the refresh, a reset), so the dynamic scale
// drops what it measured at the old pace; `falls` and `returns` count only the scale's decisions.
//
// A paused page's queue. A paused page in the test mode (debug/TestMode.ts) keeps at most
// PAUSED_QUEUE (2) rendered frames outstanding on the GPU: main.ts hands each such frame's
// `queue.onSubmittedWorkDone()` to `submitted()` and sets `queueLimit`, and a due frame waits
// (`held`) while that many have not completed. Nothing else bounds the queue in the tools' headless
// browser (no vsync, no frame-rate limit): Chrome's own back-pressure sets in only seconds of frames
// later. One page alone kept the GPU up with it (less than a frame of work queued); two such pages
// asked more of the GPU than it has, and each page's queue grew to a few hundred times what one page
// alone kept queued. Every read-back then waited behind it (the pre-exposure's readings, the sky's,
// a probe's radiance), so a page took up to ten times as many frames to settle (60-300 instead of
// 31), a radiance read took 30 to 100 times as long, and two probes sweeping six points each ran 12
// times as long as one (a sixth of one probe's throughput), a probe beside a capture page 4 times.
// Bounded, the queue holds four to five times one page's alone with two pages (about two frames of
// work), a page settles in 31-33 frames, and two probes take about 1.4 times one probe's time for
// twice the points (1.45 times its throughput), a probe beside a capture page 1.1 times its time
// alone; one page alone is unchanged, and paused frames and readings are the same bytes (only how
// many frames a second the page draws changes, and a paused frame does not depend on it). A running
// page, and a paused one outside the test mode (capture-clip's stepped clips), is left alone:
// holding a running page's ticks until the GPU caught up bunched them and made the frame intervals
// jitter (debug/FrameProfiler.ts), since Chrome reports finished work in batches, and a player's
// browser has its vsync to bound the queue.

// callbacks in the window the refresh period is taken from, and its quantile (low, as a callback
// delayed by the main thread or a skipped vblank only lengthens an interval)
const WINDOW = 240, QUANTILE = 0.05;
// intervals outside this range (ms) are not refresh periods (a hidden page, a stall)
const MIN_PERIOD = 2, MAX_PERIOD = 50;
// the lowest rate the automatic fallback goes to (fps), and the lowest a player may pick (choices())
export const MIN_FALLBACK_FPS = 40;
const MIN_CHOICE_FPS = 30;
// how far (share of the period) the refresh's estimate must pass a divisor's boundary to change it
const BOUNDARY = 0.01;
// the frames a paused page may keep queued on the GPU (see the header, "A paused page's queue")
export const PAUSED_QUEUE = 2;

/** 'auto': the tier's rate with the automatic fallback; 'fixed': a rate held as it is. */
export type FrameRateMode = 'auto' | 'fixed';

/** The display's next frame (its time): the code's other wait on the page's frames, beside the
 * pacer's own, so `requestAnimationFrame` is called here alone (tools/repo/verify-layers.mjs). */
export const nextFrame = (): Promise<number> => new Promise( ( resolve ) => requestAnimationFrame( resolve ) );

export class FramePacer {

	/** The tier's frame rate (frames per second, or `?frameRate`'s); 0 renders on every refresh. */
	readonly target: number;
	/** The display's refresh period (ms), as measured. */
	period = 1000 / 60;
	/** Refreshes per rendered frame, as rendered now (the base, or the fallback's one more). */
	divisor = 1;
	/** Refreshes per rendered frame at the held rate (the tier's in 'auto', the chosen one if fixed). */
	base = 1;
	/** Bumped by every change of pace the dynamic scale did not ask for: `setRate`, a new divisor from
	 * the refresh's estimate, `reset`. */
	generation = 0;
	/** Whether 'auto' may fall back at all (`?frameFallback=0` turns it off: A/B runs stay comparable). */
	readonly allowFallback: boolean;
	/** Whether it runs at the fallback now, and how often it fell back and returned. */
	fallen = false;
	falls = 0;
	returns = 0;
	/** Animation frames the pacer let through and skipped (for diagnostics). */
	rendered = 0;
	skipped = 0;
	/** The queue's bound (see the header, "A paused page's queue"): the most rendered frames whose
	 * GPU work may be outstanding before a due frame waits (0: unbounded, the browser's own
	 * backpressure alone); the frames outstanding now; the animation frames held for them. */
	queueLimit = 0;
	queued = 0;
	held = 0;
	/** A `requestAnimationFrame` context for three's animation loop (`Animation.setContext`): the
	 * callback runs only on the refreshes the pacer renders. */
	readonly context: { requestAnimationFrame( callback: FrameRequestCallback ): number; cancelAnimationFrame( id: number ): void };
	private readonly deltas: number[] = [];
	private previous = NaN;
	private last = - Infinity;
	private callback: FrameRequestCallback | null = null;
	private pending = 0;
	// the rate a fixed mode holds (fps; 0: every refresh)
	private fixed = 0;
	private current: FrameRateMode;
	// whether the refresh period has been measured yet
	private measured = false;

	/** `target`: the tier's rate; `rate`: 'auto' or a fixed rate (fps, 0 every refresh) to start at. */
	constructor( target: number, { rate = 'auto', allowFallback = true }: { rate?: number | 'auto'; allowFallback?: boolean } = {} ) {

		this.target = Number.isFinite( target ) && target > 0 ? target : 0;
		this.allowFallback = allowFallback;
		// without a target, 'auto' has nothing to hold: every refresh renders, fixed
		this.current = this.target > 0 ? 'auto' : 'fixed';
		if ( rate !== 'auto' ) this.setRate( rate );
		else this.setPeriod( this.period, true );
		const onFrame = ( time: number ) => {

			const callback = this.callback;
			if ( ! callback ) return;
			if ( this.frame( time ) ) {

				// three's loop asks for its next frame first thing, through this context again
				this.callback = null;
				callback( time );

			} else this.pending = requestAnimationFrame( onFrame );

		};
		this.context = {
			requestAnimationFrame: ( callback ) => {

				this.callback = callback;
				this.pending = requestAnimationFrame( onFrame );
				return this.pending;

			},
			cancelAnimationFrame: () => {

				cancelAnimationFrame( this.pending );
				this.callback = null;

			},
		};

	}

	/** The interval (ms) between rendered frames at the rate it runs at now. */
	get interval(): number {

		return this.divisor * this.period;

	}

	/** 'auto' (the tier's rate, falling back when frames miss) or 'fixed' (set through `setRate`). */
	get mode(): FrameRateMode {

		return this.current;

	}

	/** The rate it holds now (fps): the refresh over the divisor in use (the target, or the fixed rate,
	 * while the refresh is not measured yet). */
	get fps(): number {

		if ( ! this.measured && this.rate > 0 ) return this.rate;
		return 1000 / this.interval;

	}

	/** The rate chosen (fps): the tier's in 'auto', the chosen one if fixed; 0 renders on every
	 * refresh, unpaced. */
	get rate(): number {

		return this.current === 'auto' ? this.target : this.fixed;

	}

	/** The interval (ms) at the held rate, the faster one while it runs at the fallback. */
	get baseInterval(): number {

		return this.base * this.period;

	}

	/** Whether it may step to the fallback now: 'auto', allowed, and the next divisor's rate at
	 * MIN_FALLBACK_FPS or above. */
	get canFall(): boolean {

		return this.current === 'auto' && this.allowFallback && this.target > 0 && 1000 / ( ( this.base + 1 ) * this.period ) >= MIN_FALLBACK_FPS - 0.5;

	}

	/** The rates (fps) the display can hold evenly: the refresh over each whole divisor, from the
	 * refresh down to MIN_CHOICE_FPS (a settings menu's list; the refresh rounded to a whole Hz; a 60 Hz
	 * display's until the refresh is measured). */
	choices(): number[] {

		const hz = Math.round( 1000 / this.period );
		const list: number[] = [];
		for ( let d = 1; hz / d >= MIN_CHOICE_FPS - 1e-6; d ++ ) list.push( Math.round( hz / d * 100 ) / 100 );
		return list;

	}

	/** Holds `fps` (fixed: a cap, the fastest whole-divisor rate of the refresh not above it; 0 renders
	 * on every refresh) or returns to 'auto'. Leaves the fallback (not a return: a new pace). */
	setRate( rate: number | 'auto' ): void {

		this.fallen = false;
		if ( rate === 'auto' && this.target > 0 ) this.current = 'auto';
		else {

			this.current = 'fixed';
			this.fixed = rate === 'auto' ? 0 : Number.isFinite( rate ) && rate > 0 ? rate : 0;

		}
		this.setPeriod( this.period, true );
		this.generation ++;

	}

	/** Back at the held rate, as a fresh load starts (a scene jump); the counts stay. */
	reset(): void {

		if ( ! this.fallen ) return;
		this.fallen = false;
		this.divisor = this.base;
		this.generation ++;

	}

	/** Steps to the fallback rate (true) or back to the held one (false), in 'auto' only (the dynamic
	 * scale decides: DynamicResolution.ts). Returns whether the rate changed. */
	fallback( on: boolean ): boolean {

		if ( on === this.fallen || ( on && ! this.canFall ) ) return false;
		this.fallen = on;
		if ( on ) this.falls ++;
		else this.returns ++;
		this.divisor = this.base + ( on ? 1 : 0 );
		return true;

	}

	/** Counts a rendered frame as outstanding until `done` (the queue's `onSubmittedWorkDone` after
	 * its last submit) settles. */
	submitted( done: Promise<unknown> ): void {

		this.queued ++;
		const settle = () => { this.queued --; };
		done.then( settle, settle );

	}

	/** Whether the animation frame at `time` (its rAF timestamp, ms) renders. */
	frame( time: number ): boolean {

		const delta = time - this.previous;
		this.previous = time;
		if ( delta >= MIN_PERIOD && delta <= MAX_PERIOD ) {

			this.deltas.push( delta );
			if ( this.deltas.length > WINDOW ) this.deltas.shift();
			if ( this.deltas.length >= 30 ) {

				this.setPeriod( this.measure(), ! this.measured );
				this.measured = true;

			}

		}
		// a frame is due once the interval has passed, give or take half a refresh (the timestamps
		// jitter by a fraction of a millisecond); a late frame restarts the count from itself
		if ( time - this.last < this.interval - this.period * 0.5 ) {

			this.skipped ++;
			return false;

		}
		// a due frame waits while the GPU is the bound's frames behind (the count restarts from the
		// frame that renders once it has caught up)
		if ( this.queueLimit > 0 && this.queued >= this.queueLimit ) {

			this.held ++;
			return false;

		}
		this.last = time;
		this.rendered ++;
		return true;

	}

	// The refresh period: the median of the shortest intervals (the timestamps jitter by up to a
	// millisecond either way, so a low quantile alone would read a period that short).
	private measure(): number {

		const low = quantile( this.deltas, QUANTILE );
		const single = this.deltas.filter( ( d ) => d < low * 1.5 );
		return quantile( single, 0.5 );

	}

	// The divisor at a period: in 'auto' the nearest to the target, never more than a quarter above
	// it; fixed, the fewest refreshes whose rate is not above the chosen one (0: every refresh).
	private divisorAt( period: number ): number {

		const rate = this.rate;
		if ( ! ( rate > 0 ) ) return 1;
		if ( this.current === 'auto' ) return Math.max( 1, Math.floor( 1000 / rate / period + 0.25 ) );
		return Math.max( 1, Math.ceil( 1000 / ( rate * 1.01 ) / period - 1e-9 ) );

	}

	// A new period estimate: the divisor moves only once the estimate is BOUNDARY past a boundary;
	// `fresh` (the first estimate, a new rate) takes a boundary within BOUNDARY as reached (the
	// slower side). A divisor changed here leaves the fallback: a new pace, not a return.
	private setPeriod( period: number, fresh = false ): void {

		this.period = period;
		const slower = this.divisorAt( period / ( 1 + BOUNDARY ) );
		let base = this.base;
		if ( fresh ) base = slower;
		else {

			const faster = this.divisorAt( period * ( 1 + BOUNDARY ) );
			if ( faster > base ) base = faster;
			else if ( slower < base ) base = slower;

		}
		const changed = base !== this.base;
		this.base = base;
		// a refresh measured anew may also leave no fallback above MIN_FALLBACK_FPS
		const left = this.fallen && ( changed || ! this.canFall );
		if ( left ) this.fallen = false;
		this.divisor = this.base + ( this.fallen ? 1 : 0 );
		if ( changed || left ) this.generation ++;

	}

}

function quantile( values: readonly number[], q: number ): number {

	const s = [ ...values ].sort( ( a, b ) => a - b );
	const i = ( s.length - 1 ) * q, lo = Math.floor( i ), hi = Math.ceil( i );
	return s[ lo ] + ( s[ hi ] - s[ lo ] ) * ( i - lo );

}
