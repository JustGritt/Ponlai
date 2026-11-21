import type * as THREE from 'three/webgpu';
import type { FrameGraph } from './graph/FrameGraph.ts';
import type { DynamicRange } from './RenderScale.ts';
import { gpuAdapterInfo, gpuDevice, pendingTimestamps, resolveTimestamps, takeTimestamps, timestampCapacity, timestampPools, tracksTimestamps, type TimestampPool } from '../three/Backend.ts';

// Dynamic resolution: the render scale that keeps the GPU's frame time inside the paced frame.
//
// Dynamic resolution (high tier): the scene's render scale moves between the tier's maximum and a
// floor so that the GPU's frame time stays inside the paced frame (a 60 Hz frame, `FramePacer.ts`),
// changing as seldom as it can: every change softens or sharpens the whole image at once.
//
// Its interface to the frame graph is one number per frame: `scale`, the render's pixel ratio,
// read after `update()` and applied to the next frame (main.ts sizes the frame graph's targets
// from it and upscales the render to the canvas). In return it takes, per frame, the GPU's busy
// time, the part of it in passes whose cost follows the render's pixel count (`PassScaling`: the
// frame graph's class of each pass, `currentPass.scales`, so an upscaler's canvas-class passes, FSR 1's
// or the TAAU resolve's, count as fixed whatever upscaler it is; by the target's size outside it)
// and tags for periodic work. Nothing else about the frame graph is assumed.
//
// - The GPU time comes from timestamp queries (`GpuFrameTimer`, or the frame profiler's own when it
//   runs), as the time the GPU was busy with each frame: the union of its passes' intervals since
//   the last frame's work ended (in the order they start, each charged from the latest end so far,
//   or its own start, if later, to its end). A tile-based GPU stamps a pass's start when its vertex
//   stage begins, long before the passes it reads have finished, so the raw intervals overlap;
//   summed they come to several times the frame. Without timestamp queries the
//   frame intervals steer it instead (`sampleInterval`).
// - Each frame's time is split by pass: passes into targets sized from the render (the scene, its
//   resolves and copies, the post chain, bloom's mips) and the rest (the canvas-sized upscale and
//   output, shadow maps, the FFT, every compute pass, the cloud view sized from the display, sky
//   bakes), which cost the same at any scale. One fixed share for every view was half the
//   horizon's (whose cloud view and sky take about four fifths of its frame), so every step down
//   there fell short and a single drop took three or four. The render-sized passes do not all
//   follow the pixel count either: their vertex work (the meadow's blades, the terrain, the sea's
//   grid) is the same at any scale. Timed at fixed scales they cost a share of their time at
//   scale 1 plus the rest in proportion to the pixel count (s^2): the share, `nonScaling`, was
//   0.4 - 0.5 over the island and the horizon between scales 0.8 and 1.5, under MSAA and TAAU alike
//   (TAAU: 0.39 - 0.48 on fair, horizon, storm and close, its fixed part, the resolve among it,
//   flat across the scales). It starts there and is measured again at every change of scale (the
//   render-sized passes' median just before and just after), the median of the last SHARE_KEEP
//   measurements.
// - Every sample keeps the scale it was drawn at, so frames at any scale predict the cost at any
//   other: fixed + render-sized x g(s') / g(s), g(s) = k + (1 - k) s^2. Decisions read a quantile
//   of those predictions.
// - It steers a high quantile (the 90th percentile) by what the display shows. With frames paced
//   to every second refresh, the share of frames presented a refresh late stayed at its floor
//   (1-2 %) until that quantile passed about seven eighths of the paced interval, then rose steeply
//   (to about a third of the frames once it ran 5 % over the interval).
// - Periodic work (sky bake slices, the sun's shadow redraws, the refracted sun's redraws, the fog
//   bank's re-bakes) is tagged by frame (`sample`'s `heavy`) and left out of the quantiles: it hardly
//   follows the pixel count, and a quantile sitting on it stepped the scale down while most frames
//   had room. A kind of work that runs on more than HEAVY_SHARE of the frames is steady cost and
//   stays in. Tagged frames still count where frames take the whole interval (the emergency), and
//   in a second quantile: every frame, periodic work included, must also fit at ALL_QUANTILE, so
//   periodic work alone leaves at most that share of the frames late. (Left out altogether, the
//   sky's bake slices, on a sixth of the frames at noon, were each shown a refresh late.)
// - Down: when the quantile of the frames at the scale passes the paced interval less
//   DOWN_HEADROOM (at most every HOLD seconds; after EMERGENCY_HOLD when the last frames take the
//   whole interval), straight to the largest scale predicted to land under the interval less
//   TARGET_HEADROOM, in one change. A step that would save under MIN_SAVING ms is not taken: the
//   view is bound by its fixed work, and a softer image would not make it faster. A frame whose
//   pass split finds no render-sized pass (a frame graph that renders into targets of another size,
//   such as max-size targets drawn through a viewport) is taken as wholly render-sized, with a
//   warning (`onWarning`): otherwise every scale would predict the same cost.
// - Up: after UP_HOLD seconds at the scale, in one jump to the largest scale that the last UP_SHORT
//   seconds predict lands under the target with UP_MARGIN ms to spare, once the last UP_LONG
//   seconds (every frame moved to the candidate scale) allow at least as much: a view whose cost
//   swings (the camera turning from the horizon back to the island) waits until the heavier frames
//   have left the long window and then steps once. A scale it stepped up to and had to leave within
//   PROBE seconds is barred, with every scale above it, for BAR seconds: not outright, but by the
//   error that step showed (the quantile its frames reached over the one predicted, at least
//   BAR_MARGIN ms), added to their predictions; the scales between the one the step left and the
//   barred one take that error in proportion to their predicted growth (the model's error grows
//   with the jump: in the meadow a step from 0.8 to 1.3 came out 3 ms over its prediction and one
//   to 1.25 just under the bar then failed as badly). The same view does not try the same step
//   again, while a view that has turned lighter since (the camera off the horizon, the rain gone)
//   steps up as soon as its frames show room for that error too: an outright bar held a light view
//   at the heavy one's scale for up to BAR seconds. Leaving it, the step goes back to the scale it
//   left if it must, below the quality floor too and without the judged steps there: that scale
//   held. (Stopped at the quality floor, a failed step from 0.65 to 0.95 held 0.8 with twice the
//   late frames of 0.65.)
// - Without GPU times (no timestamp queries, `gpuTimer=off`) the frame intervals steer it, one step
//   at a time: down while more than MISS_SHARE of a window's frames missed their vblank, up after
//   UP_AFTER seconds without a miss. Every step down bars the scale it left from a step back up
//   for FRAME_BAR seconds, doubling each time it is left again soon after (up to FRAME_BAR_MAX),
//   so a load that misses just above a scale settles below it instead of flipping.
// - A quality floor (QUALITY_FLOOR x the canvas's pixel ratio: a 40 % render per axis, already
//   below FSR 1's Performance mode at 50 %; chosen from captures, where 30 % looked soft). A step
//   down stops there, and it goes below only while late frames persist (the quantile of the last
//   LATE_SPAN seconds still over its limit after FLOOR_DWELL seconds at the floor), by the regular
//   rule, never the emergency's, one judged step at a time: to the nearest level predicted to save
//   PROBE_SAVING (a single level saves too little to tell from the frames' noise, so this is often
//   two or three), never further, and only once the step before it was judged to have saved.
//   Each step down is judged once
//   frames at the new scale are in: if it saved under RESPONSE of the time predicted, the GPU's time
//   does not follow the pixel count (another page or app drawing: its work lands inside this page's
//   passes, or a view bound by work that does not scale), and it holds at the quality floor for
//   CONTENTION_HOLD seconds, twice as long each time a step below fails again, rather than sinking
//   towards the floor, where the image is soft and the frames are still late; one failed step alone
//   is not taken for it, a second within CONTENTION_HOLD seconds is (the view moving under a
//   player's drags failed single steps). Held there with late
//   frames for SUSTAINED seconds, it tells `onSustainedContention` (a hook, not set: the frame
//   pacer's fallback below covers frames still late there).
// - The frame rate's fallback (FramePacer.ts) is decided here, from the GPU's times, since only the
//   scale knows its floor, its frames and its cost model; the pacer only steps. It falls back (72 ->
//   48 fps on a 144 Hz display) when the scale is at its floor (the floor, the quality floor while
//   contention holds it there, or a scale whose step down would save too little: below the quality
//   floor a step too small to judge, where a simulated 4K orbit stayed at 0.65, never reaching the
//   floor, with its frames over the interval; or a view bound by its fixed work) and more than
//   FALL_LATE of its last FALL_SPAN seconds of steered frames came late (a rendered interval over
//   the paced one plus half a refresh: shown a refresh late). Seconds of frames, not of the clock:
//   the frames count only at the current scale (after SETTLE) and pace, and only once their
//   intervals add up to FALL_SPAN; main.ts passes no interval that spans a break in the steering
//   (paused, pinned, stepped, a loop start: `resume`), so the first frame after an unpause, often
//   late, is one frame of FALL_SPAN's, not all of them. It also falls on the emergency there (the
//   median of the last EMERGENCY_FRAMES steered since the break over the interval). The scale's
//   budget follows the pacer's interval, so at the fallback it rises. It
//   returns once the floor's GPU time, predicted from the frames at whatever scale (`level`, as a
//   step down reads it), stays under RETURN_FIT of the faster interval for RETURN_SPAN seconds of
//   running time (a stopped loop keeps the count: a profiler's A/B runs each page for slices of a
//   few seconds, in which a count restarted at each start could not reach RETURN_SPAN), and
//   no sooner than RETURN_WAIT seconds after the fall; a return that falls again within RETURN_WAIT
//   doubles the wait before the next (up to RETURN_MAX), and the wait goes back to RETURN_WAIT only
//   after RETURN_MAX seconds back at the faster rate without a fall (a return that held between
//   RETURN_WAIT and RETURN_MAX keeps the wait), so a view on the edge settles at the slower rate rather
//   than flipping between the two. (At 72 fps the return asks for 11.1 ms at the floor: the
//   steering target there is 11.5, so a view returns only with room to step back up.) A change of
//   pace it did not decide (the pacer's `generation`: the player's `setRate`, a new divisor from the
//   refresh's estimate, a scene jump's reset) drops the frames judged at the old pace, the return's
//   running time and its wait. While fallen it remembers no scale (the store holds the held rate's).
//   A known gap: a view that rests one step above the floor, its frames late from work outside the
//   GPU's times (the main thread, the compositor), takes no step down (its GPU times fit) and so is
//   not at its floor, and does not fall (a simulated orbit at 0.65 with 13-17 % presented late).
// - Until its first decision (`settling`) it waits for SETTLE_FRAMES clean frames and then takes
//   the largest scale their median predicts fits, up or down, with room for the first frames
//   running lighter than the scene will (SETTLE_LIGHT). It starts from the scale it last held
//   on this display (`restore`: the store, keyed by tier, AA mode, frame rate, canvas size, GPU
//   and whether GPU times or frame intervals steer it), so a returning player's first frames are near
//   it and the page keeps its loading screen up over the first jump only. A scale is remembered
//   once it has held STORE_AFTER seconds at the current key (main.ts leaves the memory off in an
//   automated browser, so a measurement does not start where the last one ended; `drsStore=off`
//   turns it off anywhere, `on` on there). The key holds the AA mode: a TAAU frame (one sample
//   per pixel, the resolve at the canvas's size) costs less per render pixel than a 4x MSAA one,
//   so the two settle at different scales.
// - TAAU (aa=taau) upscales in its resolve, which accumulates jittered renders into a history at
//   the canvas's size: a change of scale resizes only the render-class targets (no pipeline, no
//   graph rebuild, the history kept and read by uv; the jitter sequence's length follows the new
//   ratio while its frame count runs on; RCAS's sharpness eases to the new ratio's over a quarter
//   second: TemporalAA.ts, Jitter.ts). Its floor rises so the canvas is at most TAAU_MAX_UPSCALE
//   times the render per axis (`maxUpscale`): below that a moving image has too few samples per
//   output pixel. The quality floor is then the floor, and nothing goes below it. MSAA (FSR 1)
//   keeps the tier's floor, the quality floor and the steps below it.
// High's baseline (Quality.ts, RenderScale.ts `dynamicRange`): 1440p's pixel count with MSAA, the
// level 60 fps is guaranteed at, with about 14 ms of GPU time a frame in the normal scenes: scale
// 4/3 on the standard's 4K output (1920x1080 CSS at 2x, FSR 1's Quality 1.5x upscale), native on a
// 2K display (2560x1440 at 1x). A load starts there; the
// page renders there whenever the scene is paused, so paused frames and captures render at the
// baseline whatever scale the running page had reached (pinned, `?dpr` or the GUI's full
// resolution, at that ratio instead). The range runs from the display's ratio (up to the canvas's
// cap, 2: native 4K) down to the ratio that renders 1080p's pixel count (0.75 at 2K output, 1 at 4K
// output), at most three quarters of the maximum (`floorShare`: a 1x display of 1080p or less still
// keeps a floor below its maximum), in steps of 0.05, the baseline one of the levels. TAAU's upscale
// bound (0.4 on a 1x display, 0.8 on a 2x one) lies below that floor, and so does the quality floor
// (40 %), so neither moves it at 2K or 4K output.
// The loading screen waits for the first decision (at most 3 s), so the first jump from the
// nominal ratio happens behind it. A change only resizes the frame graph's targets: no pipeline is
// built and no frame is dropped (the cloud view on high is sized from the display, so its kernels
// never rebuild). Anything sized by the render must stay a render pass or be sized from something
// fixed: a compute kernel dispatched by a target's size is rebuilt at every step.
// `?gpuTimer=off` steers by the frame intervals. With `trackTimestamp` three resolves its queries
// into one buffer and drops what it resolves while that is mapped; `GpuFrameTimer` and the frame
// profiler each take over the pools' `resolveQueriesAsync`, so only one may own them: while the
// profiler times passes it feeds this from its own readings.
// Its holds, bars and windows run on its own `clock`, performance.now's in every page a player
// sees. The profiler's A/B (profile-frame --compare) stops one page while the other runs its slice;
// through a debug hook (assemble/SceneHandles.ts `dynamicClock`) the stopped page's clock stands still meanwhile, so
// a hold or bar does not run out while the page draws nothing and each page's controller sees its
// slices as one continuous run.

// The quantile of the frames' GPU times it steers (QUANTILE), and the time (ms) left in the paced
// interval for the main thread's share of the frame and the compositor: past the interval less
// DOWN_HEADROOM it steps down, and it steps to scales predicted to land under the interval less
// TARGET_HEADROOM (14.9 and 14.3 ms at 60 fps, 23.2 and 22.6 ms at 40 fps).
const QUANTILE = 0.9, DOWN_HEADROOM = 1.8, TARGET_HEADROOM = 2.4;
// seconds between steps down, after a change before frames at the new scale count, the seconds a
// step down reads and the least clean frames a decision takes
const HOLD = 1.5, SETTLE = 0.4, DOWN_SPAN = 2, MIN_FRAMES = 45;
// the least a step down must save (ms); less, and the view is bound by its fixed work
const MIN_SAVING = 0.5;
// a step up: seconds after any change, its two windows (s) and the short one's extra headroom (ms)
const UP_HOLD = 3, UP_LONG = 6, UP_SHORT = 2, UP_MARGIN = 0.3;
// seconds between evaluations of a step up
const UP_EVERY = 0.25;
// frames that take the whole interval: the median of the last EMERGENCY_FRAMES does
const EMERGENCY_HOLD = 0.5, EMERGENCY_FRAMES = 12;
// a step down within PROBE seconds of a step up bars the scale it left, and those above, for BAR s,
// by the step's error and at least BAR_MARGIN ms (the frames' noise: as PROBE_SAVING below)
const PROBE = 8, BAR = 30, BAR_MARGIN = 1;
// the quality floor's share of the canvas's pixel ratio
const QUALITY_FLOOR = 0.4;
// late frames persist at the quality floor: the quantile of the last LATE_SPAN seconds over its limit
// after FLOOR_DWELL seconds there. (Under another page's load the GPU's busy time understates how
// late the frames are shown: frames whose GPU time this page put at a level that has room alone
// were shown a refresh late four times as often.)
const LATE_SPAN = 2, FLOOR_DWELL = 3;
// a step down that saved under RESPONSE of the predicted time reads as contention, held at the
// quality floor CONTENTION_HOLD seconds (doubling up to CONTENTION_MAX); held there with late frames
// for SUSTAINED seconds, it is sustained. The time before a step is the median of its last
// PROBE_SPAN seconds at the old scale (the frames that called for it, not those of a lighter view
// before them), and only steps predicted to save PROBE_SAVING ms or more are judged: a smaller one
// drowns in the frames' noise.
const RESPONSE = 0.4, CONTENTION_HOLD = 20, CONTENTION_MAX = 160, SUSTAINED = 10, PROBE_SPAN = 0.5, PROBE_SAVING = 1;
// the render-sized passes' share that does not follow the pixel count: its first estimate, the
// range it is kept in, the measurements it is the median of and the least change of the squared
// scale a measurement takes (a smaller step drowns in the frames' noise)
const NON_SCALING = 0.45, SHARE_MIN = 0, SHARE_MAX = 0.85, SHARE_KEEP = 7, SHARE_SPAN = 0.12;
// settling: the clean frames it reads, the usual lead of the quantile over their median (ms) and
// how much lighter the first frames run than the scene will (until the key light's first shadow step
// its receivers read one shadow map, not two: about 7 % of a frame)
const SETTLE_FRAMES = 45, SETTLE_SPREAD = 0.4, SETTLE_LIGHT = 0.07;
// seconds a scale holds before it is remembered
const STORE_AFTER = 4;
// a kind of periodic work on more than this share of the frames is steady cost, part of the frames'
// usual time (at noon on high the sky's bake slices ran on about a sixth of the frames and the
// refracted sun's redraws on one in fifteen)
const HEAVY_SHARE = 0.2;
// Periodic work stays out of the quantile, not out of the late frames: every frame, periodic work
// included, must also fit at this quantile, so periodic work may leave at most this share of the
// frames over the limit. (Left out altogether, the sky's bake slices on a sixth of the frames at noon
// on high were each shown a refresh late: on a 144 Hz display paced to 72 fps the sky view held 0.8
// with 16 % of its frames late, where the earlier controller, which read every frame, held 0.6 with
// 8 %.)
const ALL_QUANTILE = 0.95;
// samples kept (s)
const KEEP = Math.max( UP_LONG, DOWN_SPAN, LATE_SPAN ) + 0.5;
// frame-interval fallback: a frame over 1.5 refresh periods missed its vblank; steps down when
// more than MISS_SHARE of a window missed (below the quality floor, more than MISS_LATE of a whole
// window, and each step must cut the share to MISS_CUT of it), up after UP_AFTER seconds without a
// miss
const MISS_SHARE = 0.02, MISS_LATE = 0.06, MISS_CUT = 0.6, UP_AFTER = 6, INTERVALS = 150;
// frame-interval fallback: seconds a scale it stepped down from is barred from a step back up,
// doubled each time it is left again within FRAME_BAR_MAX seconds of the last bar's end
const FRAME_BAR = 8, FRAME_BAR_MAX = 64;
// the stored scales (in the page's storage, `store`) and how many displays it remembers
const STORE_KEY = 'soomi.renderScale', STORE_ENTRIES = 8;
// the frame rate's fallback: the share of late frames at the floor over FALL_SPAN seconds that falls
// back; a return's fit (share of the faster interval), how long it must hold (s), and the wait after
// a fall (s), doubled up to RETURN_MAX after a return that fell again within it
const FALL_LATE = 0.1, FALL_SPAN = 2, RETURN_FIT = 0.8, RETURN_SPAN = 5, RETURN_WAIT = 10, RETURN_MAX = 80;

/** Periodic work in a frame (bits of `sample`'s `heavy`). */
export const PeriodicWork = { skyBake: 1, sunShadow: 2, refractedSun: 4 } as const;
const WORK_KINDS = Object.values( PeriodicWork );

export interface DynamicResolutionOptions extends DynamicRange {
	step: number;
	// the most the canvas may be upscaled from the render, per axis (Infinity: no bound; TAAU's
	// TAAU_MAX_UPSCALE): raises the floor to the canvas's ratio over it
	maxUpscale?: number;
	// where the scales are remembered between visits (the page's storage: main.ts); none, nowhere
	store?: ScaleStore | null;
}

/** TAAU's bound on the upscale (canvas over render, per axis): the floor rises to the canvas's ratio
 * over it (0.4 at 2K output and 0.8 at 4K, below the high tier's floor there, 0.75 and 1.0). TAAU
 * reconstructs the canvas from jittered renders, about eight samples per output pixel over its
 * sequence (Jitter.ts: L = 8 r^2, up to 64), so far below the canvas a moving image has too few
 * samples per pixel to hold its detail (FSR 2 and DLSS stop at 3x, their Ultra Performance mode,
 * and recommend 2x at most for quality); 2.5 is our choice. MSAA's FSR 1 path keeps the tier's floor and the quality
 * floor below. */
export const TAAU_MAX_UPSCALE = 2.5;

/** One frame's GPU time (ms) and how it was drawn. */
export interface FrameSampleOptions {
	/** the part of `ms` in passes whose targets are sized from the render (NaN: not known; the
	 * whole frame is then taken as render-sized) */
	pixel?: number;
	/** periodic work in the frame (`PeriodicWork` bits) */
	heavy?: number;
	/** when it arrived (s of its `clock`) */
	now?: number;
}

interface Sample {
	t: number;
	scale: number;
	total: number;
	// the part in fixed passes and in render-sized ones
	fixed: number;
	pixel: number;
	heavy: number;
}

/** A change of scale (seconds of its `clock`), the rule that made it and, for the GPU's rules,
 * the 90th percentile predicted at the new scale and (once its frames are in) the one measured. */
export interface ScaleChange { at: number; scale: number; why: string; from?: number; predicted?: number; measured?: number }

// A change whose frames at the new scale are still to come: the median of the render-sized passes'
// time in the frames that called for it and, for a step down, the saving predicted there (judged
// when at least PROBE_SAVING).
interface Response { change: ScaleChange; beforePixel: number; saving: number }

// A failed step up's bar (see `barPenalty`).
interface Bar { scale: number; from: number; until: number; error: number }

/** The frame pacer's side of the frame rate's fallback (FramePacer.ts). */
export interface PacedRate {
	/** the refresh period and the paced interval now (ms) */
	readonly period: number;
	readonly interval: number;
	/** the paced interval at the held (faster) rate (ms) */
	readonly baseInterval: number;
	/** whether it runs at the fallback, and whether it may step there */
	readonly fallen: boolean;
	/** bumped by each change of pace it did not ask for (a new rate, a new divisor, a reset) */
	readonly generation: number;
	readonly canFall: boolean;
	fallback( on: boolean ): boolean;
}

export class DynamicResolution {

	max: number;
	floor: number;
	/** The scale it goes below only while frames are really late and steps down save time. */
	qualityFloor: number;
	readonly step: number;
	// the tier's own baseline and floor for the view (RenderScale.ts `dynamicRange`), and the
	// upscale bound
	private limitBaseline: number;
	private limitFloor: number;
	private readonly maxUpscale: number;
	/** The scale the scene renders at while the resolution is dynamic. */
	scale: number;
	/** Changes so far, newest last. */
	readonly changes: ScaleChange[] = [];
	/** 'gpu' once a GPU time arrived, 'frames' while it steers by frame intervals. */
	source: 'gpu' | 'frames' = 'frames';
	/** false holds the scale where it is (debug: with `force`, to time one scale). */
	enabled = true;
	/** true until its first decision from the GPU's times (the page keeps its loading screen up). */
	settling = true;
	/** The last frames' GPU busy times (ms), whatever their scale (shown in the GUI). */
	readonly observed: number[] = [];
	/** The render-sized passes' share of their time at scale 1 that does not follow the pixel count. */
	nonScaling = NON_SCALING;
	/** Until when (s) a step down that did not save time holds it at the quality floor. */
	contendedUntil = - Infinity;
	/** Whether the last step down it judged saved the time predicted (null: none judged since the
	 * last step up). */
	lastStepSaved: boolean | null = null;
	/** Whether the last step down it considered would have saved too little (a view bound by its
	 * fixed work). */
	fixedBound = false;
	/**
	 * Hook: called with true once the frames stay late at the quality floor for SUSTAINED seconds
	 * after a step down failed to save time (contention), and with false when that ends. Nothing
	 * sets it: the frame pacer's fallback (`pacer`) takes frames still late there.
	 */
	onSustainedContention: ( ( sustained: boolean ) => void ) | null = null;
	/** Whether the sustained-contention hook was last called with true. */
	sustained = false;
	/** The frame pacer whose fallback it decides (null: none). */
	pacer: PacedRate | null = null;
	/** Called once with a message when something it reads looks wrong (main.ts sets it in debug
	 * builds). */
	onWarning: ( ( message: string ) => void ) | null = null;
	/** Its clock (s), which every hold, bar and window reads: `performance.now`'s. Only the profiler
	 * replaces it (assemble/SceneHandles.ts's debug hook `dynamicClock`): a page it keeps stopped during another
	 * page's slices (profile-frame --compare) holds its clock still, so its timers do not run out
	 * while it draws nothing. */
	// eslint-disable-next-line ponlai/scene-clock -- the controller's clock: it steers the render scale by real frame times (see the header)
	clock: () => number = () => performance.now() / 1000;
	private samples: Sample[] = [];
	private intervals: number[] = [];
	private lastChange = - Infinity;
	private lastMiss = - Infinity;
	// the paced frame interval (ms)
	private interval = 1000 / 60;
	// a step up's bar: until `until`, scales at or above `scale` are predicted `error` ms dearer, and
	// those between `from` (the scale the step left) and `scale` a share of it (`barPenalty`)
	private bar: Bar = { scale: Infinity, from: Infinity, until: - Infinity, error: 0 };
	private lastWhy = '';
	private response: Response | null = null;
	// measurements of the non-scaling share (the prior counts once)
	private readonly shares: number[] = [ NON_SCALING ];
	private lateAtFloorSince = NaN;
	// the last contention hold (s), doubled on each failed step until a step up or one that saves
	private contentionHold = 0;
	// when a judged step last failed without being taken for contention (one failure alone is not)
	private doubted = - Infinity;
	// periodic work steady enough to stay in the quantiles (PeriodicWork bits)
	private steady = 0;
	private nextUp = - Infinity;
	// the frame-interval fallback: the share of frames that missed their vblank before a step below
	// the quality floor, until the step is judged
	private missedBefore: number | null = null;
	private storeKey: string | null = null;
	private readonly store: ScaleStore | null;
	// the scale last remembered under the store key, and since when the key holds (s)
	private stored = NaN;
	private keySince = - Infinity;
	// the canvas's pixel ratio
	private canvasRatio = NaN;
	// the frame-interval fallback's bars: a scale it stepped down from, until when it is barred and
	// for how long it was
	private readonly frameBars = new Map<number, { until: number; seconds: number }>();
	private warned = false;
	// the fallback: the last FALL_SPAN seconds of steered frames (each one's interval, ms, whether it
	// came late, its scale and pace), since when they count, the pacer's generation they were taken
	// at, when the steering last resumed after a break, the last fall and return (s), the wait before
	// a return (s), and the running time (s) the floor has been predicted to fit the faster interval,
	// with the last update it counted
	private paced: { t: number; ms: number; late: boolean; scale: number; pace: number }[] = [];
	private pacedMs = 0;
	private pacedSince = - Infinity;
	private pacerGeneration = NaN;
	private resumedAt = - Infinity;
	private fellAt = - Infinity;
	private returnedAt = - Infinity;
	private returnWait = RETURN_WAIT;
	private fit = 0;
	private fitSeen = - Infinity;

	constructor( { max, baseline, floor: least, step, maxUpscale = Infinity, store = null }: DynamicResolutionOptions ) {

		if ( ! ( step > 0 && maxUpscale >= 1 ) ) throw new RangeError( 'Dynamic resolution needs a positive step and an upscale bound of at least 1' );
		DynamicResolution.check( { max, baseline, floor: least } );
		this.max = max;
		this.limitBaseline = baseline;
		const floor = this.floor = this.limitFloor = this.round( least );
		this.maxUpscale = maxUpscale;
		this.store = store;
		this.qualityFloor = floor;
		this.step = step;
		this.scale = this.baseline;

	}

	/** The scale a load starts from and a paused or pinned page renders at: the tier's baseline for
	 * the view, within the range. Not rounded (4/3 x 1920 is 2560 pixels; 1.333 x 1920 is not), and
	 * one of the levels (`steps`). */
	get baseline(): number {

		return Math.min( Math.max( this.limitBaseline, this.floor ), this.max );

	}

	/** The tier's range for the view (RenderScale.ts `dynamicRange`), whenever the window changes
	 * size or display: the levels run down from its maximum, and the floor never rises above it. */
	setRange( { max, baseline, floor }: DynamicRange ): void {

		DynamicResolution.check( { max, baseline, floor } );
		const least = this.round( floor );
		if ( max === this.max && baseline === this.limitBaseline && least === this.limitFloor ) return;
		// a scale at the baseline (a load's start) follows the new baseline
		const atBaseline = Math.abs( this.scale - this.baseline ) < 1e-6;
		this.max = max;
		this.limitBaseline = baseline;
		this.limitFloor = least;
		this.floor = this.boundedFloor();
		this.qualityFloor = Math.min( Math.max( this.qualityFloor, this.floor ), max );
		this.change( atBaseline ? this.baseline : Math.min( Math.max( this.scale, this.floor ), max ), this.clock(), 'range' );

	}

	private static check( { max, baseline, floor }: DynamicRange ): void {

		if ( ! ( floor > 0 && floor <= baseline && baseline <= max ) ) throw new RangeError( `Dynamic resolution needs 0 < floor <= baseline <= max (${ floor }, ${ baseline }, ${ max })` );

	}

	/** The canvas's pixel ratio (the display's, up to its cap): the quality floor follows it, and so
	 * does the floor under an upscale bound. */
	setCanvasRatio( ratio: number ): void {

		if ( ! ( ratio > 0 ) ) return;
		this.canvasRatio = ratio;
		this.floor = this.boundedFloor();
		// the lowest level at or above the share
		let floor = this.max;
		for ( const s of this.steps ) if ( s >= QUALITY_FLOOR * ratio - 1e-6 ) floor = s;
		this.qualityFloor = Math.max( floor, this.floor );
		if ( this.scale < this.floor - 1e-6 ) this.change( this.floor, this.clock(), 'floor' );

	}

	// The tier's floor, raised to the lowest level at or above the canvas's ratio over the upscale
	// bound (the levels run down from the maximum in steps), never above the maximum.
	private boundedFloor(): number {

		const least = this.canvasRatio > 0 ? this.canvasRatio / this.maxUpscale : 0;
		if ( least <= this.limitFloor ) return Math.min( this.limitFloor, this.max );
		const n = Math.floor( ( this.max - least ) / this.step + 1e-6 );
		return Math.min( Math.max( this.round( this.max - n * this.step ), this.limitFloor ), this.max );

	}

	/** The interval (ms) the frames are paced to (FramePacer.ts); the GPU's budget follows it. */
	setFrameInterval( ms: number ): void {

		if ( Number.isFinite( ms ) && ms > 0 ) this.interval = Math.min( Math.max( ms, 4 ), 67 );

	}

	/** The scales it can take, from the maximum down to the floor, the baseline among them. */
	get steps(): number[] {

		const list: number[] = [];
		for ( let s = this.max; s > this.floor - 1e-6; s = this.round( s - this.step ) ) list.push( s );
		if ( list[ list.length - 1 ] > this.floor + 1e-6 ) list.push( this.floor );
		const baseline = this.baseline;
		if ( ! list.some( ( s ) => Math.abs( s - baseline ) < 1e-6 ) ) {

			list.push( baseline );
			list.sort( ( a, b ) => b - a );

		}
		return list;

	}

	// A scale rounded to the nearest level when it is within rounding of one (a remembered or forced
	// 1.333 is the baseline 4/3), else to a thousandth.
	private snap( scale: number ): number {

		const near = this.steps.find( ( s ) => Math.abs( s - scale ) < 1e-3 );
		return near ?? this.round( scale );

	}

	/** The key the scales it holds are remembered under (`renderScaleKey`: tier, AA mode, frame
	 * rate, canvas size, GPU and steering), or null to remember nothing; set again whenever the canvas changes
	 * size. A new key gets its entry once the scale has held STORE_AFTER seconds under it. */
	setStoreKey( key: string | null, now = this.clock() ): void {

		if ( key === this.storeKey ) return;
		this.storeKey = key;
		this.stored = NaN;
		this.keySince = now;

	}

	/** Starts from the scale last held under the store key, if any (before the first frame). */
	restore(): void {

		const entry = this.storeKey === null ? undefined : readStore( this.store )[ this.storeKey ];
		if ( ! entry || ! Number.isFinite( entry.scale ) ) return;
		// before the first frame, behind the loading screen: not a change anyone sees
		this.scale = this.stored = this.snap( Math.min( Math.max( entry.scale, this.floor ), this.max ) );
		this.lastWhy = 'restore';

	}

	/** One frame's GPU busy time (ms), rendered at `scale` (NaN: a frame that does not steer it, while
	 * paused or pinned). */
	sample( ms: number, scale: number, { pixel = NaN, heavy = 0, now = this.clock() }: FrameSampleOptions = {} ): void {

		if ( ! Number.isFinite( ms ) || ms <= 0 ) return;
		this.source = 'gpu';
		this.observed.push( ms );
		if ( this.observed.length > 120 ) this.observed.shift();
		if ( ! ( scale > 0 ) ) return;
		// the first frames at a new scale use its targets for the first time
		if ( scale === this.scale && now - this.lastChange < SETTLE ) return;
		// A split that finds no render-sized pass at all (the scene pass always is one) means the
		// targets no longer match the render's size: every scale would then predict the same cost,
		// and no step down would ever save. The frame counts as wholly render-sized instead.
		if ( pixel <= 0 ) {

			pixel = NaN;
			if ( ! this.warned ) {

				this.warned = true;
				this.onWarning?.( 'Dynamic resolution: no pass of a frame renders into a target sized from the render (PassScaling); its whole GPU time is taken to follow the pixel count' );

			}

		}
		const p = Number.isFinite( pixel ) ? Math.min( pixel, ms ) : ms;
		this.samples.push( { t: now, scale, total: ms, fixed: ms - p, pixel: p, heavy } );
		while ( this.samples.length && now - this.samples[ 0 ].t > KEEP ) this.samples.shift();

	}

	/** One rendered frame's interval (ms) since the previous steered frame: whether it came late (the
	 * frame rate's fallback), and the steering for a browser without timestamp queries. An interval
	 * that spans a break in the steering is not passed: `resume` instead. */
	sampleInterval( ms: number, now = this.clock() ): void {

		if ( ! Number.isFinite( ms ) || ms <= 0 || ms > 250 ) return;
		// an interval over a change of pace belongs to neither pace
		if ( this.pacer && this.syncPacer( now ) ) {

			const pace = this.pacer.interval;
			this.paced.push( { t: now, ms, late: ms > pace + this.pacer.period * 0.5, scale: this.scale, pace } );
			this.pacedMs += ms;
			while ( this.paced.length > 1 && this.pacedMs - this.paced[ 0 ].ms >= FALL_SPAN * 1000 ) this.pacedMs -= this.paced.shift()!.ms;

		}
		if ( this.source === 'gpu' ) return;
		if ( now - this.lastChange < SETTLE ) return;
		this.intervals.push( ms );
		if ( this.intervals.length > INTERVALS ) this.intervals.shift();

	}

	/** Moves the scale when its window says so. Returns whether it changed. */
	update( now = this.clock() ): boolean {

		if ( ! this.enabled ) return false;
		const changed = this.source === 'gpu' ? this.steerByGpu( now ) : this.steerByIntervals( now );
		if ( this.source === 'gpu' && ! this.settling ) this.steerPacing( now );
		this.remember( now );
		return changed;

	}

	/** The frame loop starts (or restarts after a stop): its first frames (first-use work, whole
	 * bakes) do not count, and it settles again from the next clean frames. */
	begin( now = this.clock() ): void {

		this.lastChange = now;
		this.samples.length = 0;
		this.intervals.length = 0;
		this.response = null;
		this.missedBefore = null;
		this.settling = true;
		this.clearPaced( now );
		// the time the loop stood still is not time the floor fitted (the fit so far stands)
		this.fitSeen = now;

	}

	/** A steered frame after frames that did not steer (paused, pinned, stepped) or after a loop start:
	 * its interval spans the break and is not passed; the emergency counts frames from here. */
	resume( now = this.clock() ): void {

		this.resumedAt = now;

	}

	/** Sets the scale (a debug override, or a seek to a known settle point). */
	force( scale: number, now = this.clock() ): void {

		this.change( Math.min( Math.max( this.snap( scale ), this.floor ), this.max ), now, 'force' );

	}

	/** The lowest scale it may step down to now: the quality floor, and below it only while frames
	 * are really late, never past it in one step (the step to it is judged first), not while a
	 * failed step holds it there, and from below it only once the step there saved time. */
	lowestAllowed( now = this.clock() ): number {

		if ( this.scale > this.qualityFloor + 1e-6 || now < this.contendedUntil ) return this.qualityFloor;
		if ( this.scale < this.qualityFloor - 1e-6 && ( this.response !== null || this.lastStepSaved !== true ) ) return this.scale;
		if ( this.scale >= this.qualityFloor - 1e-6 && now - this.lastChange < FLOOR_DWELL ) return this.qualityFloor;
		return this.overLimit( now ) ? this.floor : this.qualityFloor;

	}

	private steerByGpu( now: number ): boolean {

		this.steady = this.steadyWork( now );
		this.measureResponse( now );
		const since = now - this.lastChange;
		const target = this.interval - TARGET_HEADROOM;
		const clean = ( x: Sample ) => this.clean( x );
		// the frames drawn at the scale since it was set
		const current = this.samples.filter( ( x ) => x.scale === this.scale && x.t - this.lastChange >= SETTLE );
		if ( this.settling ) {

			const frames = current.filter( clean );
			if ( frames.length < SETTLE_FRAMES ) return false;
			this.settling = false;
			const s = this.largestFitting( frames, ( target - SETTLE_SPREAD ) / ( 1 + SETTLE_LIGHT ), 0.5, this.qualityFloor );
			if ( s < this.scale ) return this.down( s, now, 'settle', frames );
			return this.to( s, now, 'settle', frames );

		}
		this.watchSustained( now );
		const lowest = this.lowestAllowed( now );
		// Frames that take the whole interval, periodic work included: no waiting for the window. It
		// stops at the quality floor: a dozen frames do not show late frames persisting.
		const last = current.slice( - EMERGENCY_FRAMES );
		// a step up it has to take back may return to the scale it left, below the quality floor too
		// (that scale held: no judged step is needed to go back there)
		const back = this.lastWhy === 'up' && since < PROBE ? this.changes[ this.changes.length - 1 ]?.from ?? NaN : NaN;
		const stop = Math.max( lowest, Math.min( this.qualityFloor, this.scale ) );
		const urgent = Number.isFinite( back ) ? Math.min( stop, back ) : stop;
		if ( this.scale > urgent && since >= EMERGENCY_HOLD && last.length === EMERGENCY_FRAMES && quantile( last.map( ( x ) => x.total ), 0.5 ) > this.interval ) {

			const s = this.largestFitting( last, target, 0.5, urgent, this.scale );
			if ( s < this.scale && this.down( s, now, 'emergency', current.filter( clean ) ) ) return true;

		}
		if ( since < HOLD ) return false;
		const window = current.filter( ( x ) => now - x.t <= DOWN_SPAN );
		const recent = window.filter( clean );
		if ( recent.length >= MIN_FRAMES ) {

			const high = this.level( recent, window, this.scale );
			if ( high > this.interval - DOWN_HEADROOM ) {

				const least = Number.isFinite( back ) ? Math.min( lowest, back ) : lowest;
				if ( this.scale <= least ) return false;
				const s = this.largestFitting( recent, target, QUANTILE, least, this.scale, undefined, window );
				// a step that saves too little: the view is bound by its fixed work
				this.fixedBound = high - this.level( recent, window, s ) < MIN_SAVING;
				return this.fixedBound ? false : this.down( s, now, 'down', recent );

			}
			this.fixedBound = false;

		}
		// (a few times a second: it reads every frame of the last seconds at a few scales)
		if ( since >= UP_HOLD && this.scale < this.max && now >= this.nextUp ) {

			this.nextUp = now + UP_EVERY;
			const longAll = this.samples.filter( ( x ) => now - x.t <= UP_LONG );
			const shortAll = longAll.filter( ( x ) => now - x.t <= UP_SHORT );
			const long = longAll.filter( clean );
			const short = shortAll.filter( clean );
			if ( short.length >= MIN_FRAMES ) {

				const bar = now < this.bar.until ? this.bar : undefined;
				const a = this.largestFitting( long, target, QUANTILE, this.scale, Infinity, bar, longAll );
				const b = this.largestFitting( short, target - UP_MARGIN, QUANTILE, this.scale, Infinity, bar, shortAll );
				// Up to the short window's level once the long one allows at least as much (waiting for
				// the two to be equal would never step up where the margin keeps them a level apart);
				// while the long one allows less, the view has just turned lighter: wait, then step once.
				if ( b > this.scale && a >= b ) return this.to( b, now, 'up', short );

			}

		}
		return false;

	}

	// The frame rate's fallback (see the header): falls back at the floor while frames still come
	// late, returns once the floor is predicted to fit the faster interval with room.
	private steerPacing( now: number ): void {

		const pacer = this.pacer;
		if ( ! pacer || ! this.syncPacer( now ) ) return;
		if ( ! pacer.fallen ) {

			if ( ! pacer.canFall || ! this.atFloor( now ) ) return;
			// FALL_SPAN seconds of steered frames rendered at this scale (after SETTLE) and pace
			const pace = pacer.interval, from = this.lastChange + SETTLE;
			const frames = this.paced.filter( ( x ) => x.scale === this.scale && x.t >= from && Math.abs( x.pace - pace ) < pacer.period * 0.5 );
			const span = frames.reduce( ( sum, x ) => sum + x.ms, 0 );
			const late = span >= FALL_SPAN * 1000 && frames.filter( ( x ) => x.late ).length > frames.length * FALL_LATE;
			const last = this.samples.filter( ( x ) => x.scale === this.scale && x.t - this.lastChange >= SETTLE && x.t >= this.resumedAt ).slice( - EMERGENCY_FRAMES );
			const emergency = now - Math.max( this.lastChange, this.pacedSince ) >= EMERGENCY_HOLD && last.length === EMERGENCY_FRAMES && quantile( last.map( ( x ) => x.total ), 0.5 ) > this.interval;
			if ( ! ( late || emergency ) || ! pacer.fallback( true ) ) return;
			// a return that fell again soon: wait twice as long before the next; back to RETURN_WAIT
			// only after RETURN_MAX seconds at the faster rate without a fall
			if ( now - this.returnedAt >= RETURN_MAX ) this.returnWait = RETURN_WAIT;
			else if ( now - this.returnedAt < RETURN_WAIT ) this.returnWait = Math.min( this.returnWait * 2, RETURN_MAX );
			this.fellAt = now;
			this.clearPaced( now );
			this.fit = 0;
			this.fitSeen = now;
			return;

		}
		// RETURN_SPAN seconds of running time (a loop stopped and started again, `begin`, keeps what it
		// had: the frames before and after the stop both fitted), at most a tenth of a second per update
		const dt = Math.min( Math.max( now - this.fitSeen, 0 ), 0.1 );
		this.fitSeen = now;
		const all = this.samples.filter( ( x ) => now - x.t <= DOWN_SPAN );
		const clean = all.filter( ( x ) => this.clean( x ) );
		// too few frames since a start say nothing either way
		if ( clean.length < MIN_FRAMES ) return;
		if ( this.level( clean, all, this.floor ) > RETURN_FIT * pacer.baseInterval ) {

			this.fit = 0;
			return;

		}
		this.fit += dt;
		if ( this.fit < RETURN_SPAN || now - this.fellAt < this.returnWait || ! pacer.fallback( false ) ) return;
		this.returnedAt = now;
		this.clearPaced( now );
		this.fit = 0;

	}

	// Follows the pacer's own changes (see the header): a change it did not decide drops the frames
	// judged at the old pace, the return's running time and its wait. Returns false at a change.
	private syncPacer( now: number ): boolean {

		const generation = this.pacer!.generation;
		if ( generation === this.pacerGeneration ) return true;
		const first = Number.isNaN( this.pacerGeneration );
		this.pacerGeneration = generation;
		if ( first ) return true;
		this.clearPaced( now );
		this.fit = 0;
		this.fitSeen = now;
		this.returnWait = RETURN_WAIT;
		return false;

	}

	private clearPaced( now: number ): void {

		this.paced.length = 0;
		this.pacedMs = 0;
		this.pacedSince = now;

	}

	// At the lowest scale that helps: the floor; the quality floor while contention holds it there; or
	// any scale whose step down would save too little (below the quality floor, one too small to judge).
	private atFloor( now: number ): boolean {

		return this.scale <= this.floor + 1e-6 || this.fixedBound || ( now < this.contendedUntil && this.scale <= this.qualityFloor + 1e-6 );

	}

	// Not periodic work, or only work that runs on so many frames it is steady cost.
	private clean( x: Sample ): boolean {

		return ( x.heavy & ~ this.steady ) === 0;

	}

	// The render-sized passes' time at `scale` relative to scale 1.
	private growth( scale: number ): number {

		return this.nonScaling + ( 1 - this.nonScaling ) * scale * scale;

	}

	// The GPU time (ms) the `q` quantile of `frames` is predicted to take at `scale`.
	private predict( frames: readonly Sample[], scale: number, q: number ): number {

		const g = this.growth( scale );
		return quantile( frames.map( ( x ) => x.fixed + x.pixel * g / this.growth( x.scale ) ), q );

	}

	// The level the rules steer at `scale`: the QUANTILE of the clean frames, or the ALL_QUANTILE of
	// every frame (periodic work included), whichever is higher.
	private level( clean: readonly Sample[], all: readonly Sample[], scale: number ): number {

		const high = this.predict( clean, scale, QUANTILE );
		return all.length >= MIN_FRAMES ? Math.max( high, this.predict( all, scale, ALL_QUANTILE ) ) : high;

	}

	// The error a bar adds to the prediction at `scale`: all of it at and above the barred scale, none
	// at or below the scale the failed step left, and between the two in proportion to the predicted
	// growth of the render-sized passes (rising with the scale).
	private barPenalty( bar: Bar, scale: number ): number {

		if ( scale <= bar.from + 1e-6 ) return 0;
		if ( scale >= bar.scale - 1e-6 ) return bar.error;
		const g = this.growth( bar.from );
		return bar.error * ( this.growth( scale ) - g ) / ( this.growth( bar.scale ) - g );

	}

	// The largest scale between `lowest` and below `ceiling` (the maximum without one) whose
	// predicted `q` quantile (plus a bar's error, `barPenalty`) lands under `limit`; `lowest` if none
	// does (the cost rises with the scale, the bar's error too, so a bisection over the levels finds
	// it).
	private largestFitting( frames: readonly Sample[], limit: number, q: number, lowest: number, ceiling = Infinity, bar?: Bar, all?: readonly Sample[] ): number {

		const levels = this.steps.filter( ( s ) => s >= lowest - 1e-6 && s < ceiling - 1e-6 );
		if ( levels.length === 0 ) return lowest;
		// levels run from the largest down: find the first that fits
		let lo = 0, hi = levels.length;
		while ( lo < hi ) {

			const mid = ( lo + hi ) >> 1;
			const penalty = bar ? this.barPenalty( bar, levels[ mid ] ) : 0;
			const cost = all ? this.level( frames, all, levels[ mid ] ) : this.predict( frames, levels[ mid ], q );
			if ( cost + penalty <= limit ) hi = mid;
			else lo = mid + 1;

		}
		return lo < levels.length ? levels[ lo ] : Math.max( lowest, levels[ levels.length - 1 ] );

	}

	// Steps down to `scale` (`frames`: the clean frames at the scale it leaves that called for it).
	// Returns false when the step is not taken.
	private down( scale: number, now: number, why: string, frames: readonly Sample[] ): boolean {

		const from = this.scale;
		const response = this.before( frames, now );
		// the render-sized passes' saving: the fixed ones (the cloud view above all) change with the
		// view, so a step taken as the camera turns would be judged by the turn
		const saving = ( s: number ) => response ? response.beforePixel * ( 1 - this.growth( s ) / this.growth( from ) ) : NaN;
		const up = this.lastWhy === 'up' && now - this.lastChange < PROBE ? this.changes[ this.changes.length - 1 ] : undefined;
		const returning = up?.from !== undefined && scale >= up.from - 1e-6;
		if ( scale < this.qualityFloor - 1e-6 && ! returning ) {

			// Below the quality floor every step is judged, one at a time: it goes to the nearest level
			// below that is predicted to save enough to judge (often two or three levels: one saves too
			// little to tell from the frames' noise), never further even when the fit asks for more,
			// and is not taken if none does (a run of small steps would sink to the floor under
			// contention, none of them judged).
			const s = this.steps.find( ( l ) => l < from - 1e-6 && saving( l ) >= PROBE_SAVING );
			if ( s === undefined ) {

				this.fixedBound = true;
				return false;

			}
			scale = s;

		}
		// a step up it has to take back at once: bar it, and every scale above, for a while, by the
		// error the step showed (the quantile its frames reached over the one predicted for them)
		if ( this.lastWhy === 'up' && now - this.lastChange < PROBE ) {

			const up = this.changes[ this.changes.length - 1 ];
			const predicted = up?.predicted ?? NaN;
			const error = frames.length && Number.isFinite( predicted ) ? this.predict( frames, from, QUANTILE ) - predicted : 0;
			this.bar = { scale: from, from: Math.min( up?.from ?? from, from ), until: now + BAR, error: Math.max( error, BAR_MARGIN ) };

		}
		const changed = this.to( scale, now, why, frames, saving( scale ) );
		// returning below the quality floor: that scale was reached by steps judged there, or held
		// (a step too small to judge leaves it so, rather than stuck there unable to step lower)
		if ( changed && returning && scale < this.qualityFloor - 1e-6 && this.lastStepSaved === null ) this.lastStepSaved = true;
		return changed;

	}

	// The frames that called for a change (its last PROBE_SPAN seconds at the scale it leaves, not
	// those of a lighter view before them) and their medians.
	private before( frames: readonly Sample[], now: number ): { beforePixel: number } | null {

		const last = frames.filter( ( x ) => now - x.t <= PROBE_SPAN );
		const list = last.length >= EMERGENCY_FRAMES ? last : frames.slice( - EMERGENCY_FRAMES );
		if ( list.length < EMERGENCY_FRAMES ) return null;
		return { beforePixel: quantile( list.map( ( x ) => x.pixel ), 0.5 ) };

	}

	// Changes to `scale` by one of the GPU's rules, and arms the measurement of its response.
	private to( scale: number, now: number, why: string, frames: readonly Sample[], saving = NaN ): boolean {

		const response = this.before( frames, now );
		const changed = this.change( scale, now, why, frames.length ? this.predict( frames, scale, QUANTILE ) : NaN );
		if ( changed && response ) this.response = { change: this.changes[ this.changes.length - 1 ], beforePixel: response.beforePixel, saving };
		return changed;

	}

	// Once frames at the scale a change went to are in: the quantile it reached, what the
	// render-sized passes' time did (the non-scaling share) and, for a step down, whether it saved
	// what was predicted.
	private measureResponse( now: number ): void {

		const response = this.response;
		if ( ! response ) return;
		const change = response.change;
		const frames = this.samples.filter( ( x ) => x.scale === change.scale && x.t - change.at >= SETTLE && this.clean( x ) );
		if ( frames.length < MIN_FRAMES ) return;
		this.response = null;
		change.measured = quantile( frames.map( ( x ) => x.total ), QUANTILE );
		const afterPixel = quantile( frames.map( ( x ) => x.pixel ), 0.5 );
		let saved = true, doubt = false;
		if ( response.saving >= PROBE_SAVING ) {

			saved = response.beforePixel - afterPixel >= RESPONSE * response.saving;
			// One failed step is not taken for contention unless another failed within CONTENTION_HOLD
			// seconds: the view moves under the judgement (under a player's drags a step from 0.8 to 0.7
			// in overcast measured 3 ms over its prediction as the camera turned, and the hold kept 0.8
			// for 25 s with a fifth more late frames), while another page's load fails every step. The
			// step stands, the next one is judged again, and the measurement teaches nothing.
			doubt = ! saved && now - this.doubted > CONTENTION_HOLD;
			if ( doubt ) this.doubted = now;
			this.lastStepSaved = saved || doubt;

		}
		if ( doubt ) return;
		if ( saved ) {

			if ( response.saving >= PROBE_SAVING ) {

				this.contentionHold = 0;
				this.doubted = - Infinity;

			}
			// Where the maximum is the canvas's own ratio, leaving it adds the canvas-sized upscale and
			// moves the output pass to the fixed side: that change's ratio says nothing of the share.
			const atCanvas = Math.abs( this.max - this.canvasRatio ) < 1e-6 && ( change.from === this.max || change.scale === this.max );
			if ( ! atCanvas ) this.learnShare( change.from!, change.scale, response.beforePixel, afterPixel );
			return;

		}
		// The GPU's time did not follow the pixel count: something else is drawing. Hold at the
		// quality floor, twice as long each time a step below it fails again.
		this.contentionHold = this.contentionHold > 0 ? Math.min( this.contentionHold * 2, CONTENTION_MAX ) : CONTENTION_HOLD;
		this.contendedUntil = now + this.contentionHold;
		if ( this.scale < this.qualityFloor ) this.change( this.qualityFloor, now, 'contention' );

	}

	// One measurement of the non-scaling share from the render-sized passes' medians before (at
	// `from`) and after (at `to`) a change: after / before = g( to ) / g( from ).
	private learnShare( from: number, to: number, before: number, after: number ): void {

		const f = from * from, t = to * to;
		if ( Math.abs( f - t ) < SHARE_SPAN || ! ( before > 0 ) || ! ( after > 0 ) ) return;
		const r = after / before;
		const d = r * ( 1 - f ) - ( 1 - t );
		if ( Math.abs( d ) < 1e-3 ) return;
		const k = ( t - r * f ) / d;
		// far outside the range: the view changed under the step
		if ( ! ( k > SHARE_MIN - 0.3 && k < 1.2 ) ) return;
		this.shares.push( Math.min( Math.max( k, SHARE_MIN ), SHARE_MAX ) );
		if ( this.shares.length > SHARE_KEEP ) this.shares.shift();
		this.nonScaling = quantile( this.shares, 0.5 );

	}

	// Whether the quantile of the last LATE_SPAN seconds' clean frames (moved to the scale) is over
	// the limit for a step down.
	private overLimit( now: number ): boolean {

		const all = this.samples.filter( ( x ) => now - x.t <= LATE_SPAN );
		const frames = all.filter( ( x ) => this.clean( x ) );
		return frames.length >= MIN_FRAMES && this.level( frames, all, this.scale ) > this.interval - DOWN_HEADROOM;

	}

	// The kinds of periodic work that run on more than HEAVY_SHARE of the recent frames.
	private steadyWork( now: number ): number {

		const frames = this.samples.filter( ( x ) => now - x.t <= UP_LONG );
		let steady = 0;
		if ( frames.length === 0 ) return steady;
		for ( const kind of WORK_KINDS ) if ( frames.filter( ( x ) => x.heavy & kind ).length > frames.length * HEAVY_SHARE ) steady |= kind;
		return steady;

	}

	// Late frames held at the quality floor while its steps down did not save time: tells the hook.
	private watchSustained( now: number ): void {

		const held = this.scale <= this.qualityFloor + 1e-6 && this.lastStepSaved === false && this.overLimit( now );
		if ( ! held ) this.lateAtFloorSince = NaN;
		else if ( Number.isNaN( this.lateAtFloorSince ) ) this.lateAtFloorSince = now;
		const sustained = held && now - this.lateAtFloorSince >= SUSTAINED;
		if ( sustained === this.sustained ) return;
		this.sustained = sustained;
		this.onSustainedContention?.( sustained );

	}

	private steerByIntervals( now: number ): boolean {

		const v = this.intervals;
		if ( v.length < MIN_FRAMES ) return false;
		// its first decision (its scales are remembered from here on)
		this.settling = false;
		const period = quantile( v, 0.1 );
		const missed = v.filter( ( x ) => x > period * 1.5 ).length / v.length;
		if ( missed > 0 ) this.lastMiss = Math.max( this.lastMiss, now );
		const since = now - this.lastChange;
		const full = v.length >= INTERVALS;
		// Below the quality floor only while frames keep missing their vblank, one step at a time, each
		// judged once on a whole window after it: it must have cut the missed frames clearly (to
		// MISS_CUT of the window that called for it, which was picked for its misses, so the next one
		// has fewer by chance alone). One that did not reads as contention: it returns to the floor
		// and holds there, twice as long each time a step fails again.
		if ( this.missedBefore !== null ) {

			if ( ! full ) return false;
			const helped = missed <= this.missedBefore * MISS_CUT;
			this.missedBefore = null;
			this.lastStepSaved = helped;
			if ( ! helped ) {

				this.contentionHold = this.contentionHold > 0 ? Math.min( this.contentionHold * 2, CONTENTION_MAX ) : CONTENTION_HOLD;
				this.contendedUntil = now + this.contentionHold;
				return this.change( this.qualityFloor, now, 'contention' );

			}
			this.contentionHold = 0;

		}
		if ( since >= HOLD && missed > MISS_SHARE ) {

			// a step below the quality floor reads a whole window, after FLOOR_DWELL at the scale
			const late = missed > MISS_LATE && full && since >= FLOOR_DWELL && now >= this.contendedUntil;
			const lowest = late && this.scale <= this.qualityFloor + 1e-6 ? this.floor : this.qualityFloor;
			if ( this.scale > lowest + 1e-6 ) {

				const scale = this.round( Math.max( this.scale - this.step, lowest ) );
				// the scale it leaves missed: keep away from it for a while, longer each time
				const bar = this.frameBars.get( this.scale );
				const seconds = bar && now - bar.until < FRAME_BAR_MAX ? Math.min( bar.seconds * 2, FRAME_BAR_MAX ) : FRAME_BAR;
				this.frameBars.set( this.scale, { until: now + seconds, seconds } );
				const changed = this.change( scale, now, 'frames' );
				if ( scale < this.qualityFloor - 1e-6 ) this.missedBefore = missed;
				return changed;

			}

		}
		if ( since >= UP_AFTER && now - this.lastMiss >= UP_AFTER && this.scale < this.max ) {

			const up = this.round( Math.min( this.scale + this.step, this.max ) );
			const bar = this.frameBars.get( up );
			if ( ! ( bar && now < bar.until ) ) return this.change( up, now, 'frames up' );

		}
		return false;

	}

	private change( scale: number, now: number, why: string, predicted = NaN ): boolean {

		const from = this.scale;
		const changed = scale !== from;
		if ( changed ) {

			const entry: ScaleChange = { at: now, scale, why, from };
			if ( Number.isFinite( predicted ) ) entry.predicted = predicted;
			this.changes.push( entry );
			if ( this.changes.length > 200 ) this.changes.shift();
			this.response = null;
			this.missedBefore = null;
			// whether the view is bound by its fixed work is judged again at the new scale
			this.fixedBound = false;
			// room to step up: whatever held the time up has gone (below the quality floor the steps
			// that saved time down there still count)
			if ( why === 'up' ) {

				this.contentionHold = 0;
				if ( scale >= this.qualityFloor - 1e-6 ) this.lastStepSaved = null;

			}

		}
		this.lastWhy = why;
		this.scale = scale;
		this.lastChange = now;
		this.intervals.length = 0;
		return changed;

	}

	// Remembers a scale once it has held STORE_AFTER seconds under the current key.
	private remember( now: number ): void {

		// at the frame rate's fallback the scale is the slower rate's: the store is the held rate's
		if ( this.storeKey === null || this.settling || this.pacer?.fallen || this.scale === this.stored || now - Math.max( this.lastChange, this.keySince ) < STORE_AFTER ) return;
		this.stored = this.scale;
		writeStore( this.store, this.storeKey, this.scale );

	}

	private round( x: number ): number {

		return Math.round( x * 1000 ) / 1000;

	}

}

function quantile( values: readonly number[], q: number ): number {

	const s = [ ...values ].sort( ( a, b ) => a - b );
	const i = ( s.length - 1 ) * q, lo = Math.floor( i ), hi = Math.ceil( i );
	return s[ lo ] + ( s[ hi ] - s[ lo ] ) * ( i - lo );

}

// ------------------------------------------------------------------ the remembered scale

type Store = Record<string, { scale: number; at: number }>;
export type ScaleStore = Pick<Storage, 'getItem' | 'setItem'>;

// Storage can be missing or refuse (a private window, blocked site data): the page then starts from
// the nominal scale, as on a first visit.
function readStore( storage: ScaleStore | null ): Store {

	try {

		const text = storage?.getItem( STORE_KEY );
		const store = text ? JSON.parse( text ) as unknown : null;
		return store && typeof store === 'object' ? store as Store : {};

	} catch {

		return {};

	}

}

function writeStore( storage: ScaleStore | null, key: string, scale: number ): void {

	try {

		const store = readStore( storage );
		// eslint-disable-next-line ponlai/scene-clock -- when a display's scale was stored, to keep the most recent ones
		store[ key ] = { scale, at: Date.now() };
		// the displays used most recently (where it fails, the next visit starts at the nominal scale)
		storage?.setItem( STORE_KEY, JSON.stringify( Object.fromEntries( Object.entries( store ).sort( ( a, b ) => b[ 1 ].at - a[ 1 ].at ).slice( 0, STORE_ENTRIES ) ) ) );

	} catch { /* not remembered */ }

}

/** The key a scale is remembered under: the tier, the antialiasing mode (a TAAU frame's render costs
 * less than a 4x MSAA one's, so the two settle at different scales), its frame rate, the canvas's
 * size, the GPU and what steers it ('gpu' times or frame intervals: the fallback's scales are not
 * the GPU rule's). */
export function renderScaleKey( renderer: THREE.WebGPURenderer, tier: string, aa: string, frameRate: number, canvasWidth: number, canvasHeight: number, steering: 'gpu' | 'frames' ): string {

	let gpu = '';
	try {

		const info = gpuAdapterInfo( renderer );
		if ( info ) gpu = [ info.vendor, info.architecture, info.device, info.description ].filter( Boolean ).join( '/' );

	} catch {

		// no adapter info: the key still holds the canvas and tier

	}
	return `${ tier }|${ aa }|${ frameRate }|${ canvasWidth }x${ canvasHeight }|${ gpu }` + ( steering === 'gpu' ? '' : '|frames' );

}

// ------------------------------------------------------------------ which passes follow the pixels

// the passes whose class is kept until their timestamps are read (several seconds of frames)
const PASSES_KEPT = 1 << 16;

/** Sorts the renderer's passes, as they begin, into those whose cost follows the render's pixel
 * count and the rest. Inside a frame graph pass, a render into a target of the pass's own size class
 * takes the graph's word (`currentPass.scales`: a render-class pass follows it; a canvas-class one,
 * such as the TAAU resolve, the upscale and the output, does not, even when the canvas and the render
 * are the same size); any other render (a shadow map redrawn inside the scene pass, a chain's
 * reduced mips, work outside the graph) is sorted by its target's size: at the render's size or a
 * power-of-two fraction of it (the scene, its resolves and copies, the post chain, bloom's mips) it
 * follows, otherwise (the canvas, shadow maps, the FFT's and the sky's fixed targets) it does not.
 * Every compute pass is fixed (the cloud view's dispatch follows the display). Reads three's
 * inspector hooks, wrapping whichever inspector the renderer has. */
export class PassScaling {

	/** The frame graph whose passes it asks first (main.ts sets it once the pipeline is built). */
	graph: Pick<FrameGraph<unknown>, 'currentPass' | 'extent'> | null = null;
	private readonly pixel = new Map<string, boolean>();
	private width = 0;
	private height = 0;

	private constructor( renderer: THREE.WebGPURenderer ) {

		type Hooks = { beginRender( uid: string, scene: unknown, camera: unknown, target: { width: number; height: number } | null ): void; beginCompute( uid: string, nodes: unknown ): void };
		const inspector = renderer.inspector as unknown as Hooks;
		const beginRender = inspector.beginRender.bind( inspector ), beginCompute = inspector.beginCompute.bind( inspector );
		inspector.beginRender = ( uid, scene, camera, target ) => {

			this.note( uid, target !== null && this.classify( target.width, target.height ) );
			beginRender( uid, scene, camera, target );

		};
		inspector.beginCompute = ( uid, nodes ) => {

			this.note( uid, false );
			beginCompute( uid, nodes );

		};

	}

	static install( renderer: THREE.WebGPURenderer ): PassScaling {

		return new PassScaling( renderer );

	}

	/** The render's size in pixels (call whenever it changes). */
	setSize( width: number, height: number ): void {

		this.width = width;
		this.height = height;

	}

	/** Whether the pass with timestamp uid `uid` followed the pixel count (and forgets it). */
	take( uid: string ): boolean {

		const pixel = this.pixel.get( uid ) ?? false;
		this.pixel.delete( uid );
		return pixel;

	}

	private note( uid: string, pixel: boolean ): void {

		this.pixel.set( uid, pixel );
		// Uids whose timestamps never arrive (lost read-backs) must not pile up, but the ones still
		// to be read must stay: a frame has about a hundred passes and computes (the sky's bake
		// slices among them), and the frame profiler's read-backs ran about forty frames behind under
		// load, so 4096 entries were evicted before they were read (every pass then counted as fixed)
		if ( this.pixel.size > PASSES_KEPT ) this.pixel.delete( this.pixel.keys().next().value! );

	}

	// The graph's class for a render into a target of its pass's size, else the target's size.
	private classify( w: number, h: number ): boolean {

		const pass = this.graph?.currentPass;
		if ( pass?.size ) {

			const e = this.graph!.extent( pass.size ).value;
			if ( e.x === w && e.y === h ) return pass.scales === 'pixel';

		}
		return this.follows( w, h );

	}

	// at the render's size or a fraction 1 / 2^k of it (floored or rounded), in both axes alike
	private follows( w: number, h: number ): boolean {

		for ( let k = 0, d = 1; k <= 6; k ++, d *= 2 ) {

			if ( Math.abs( w - this.width / d ) <= 1.01 && Math.abs( h - this.height / d ) <= 1.01 ) return true;

		}
		return false;

	}

}

// ------------------------------------------------------------------ GPU frame time

// The few WebGPU types used here (the project builds without @webgpu/types).
type GPUBuffer = { size: number; mapAsync( mode: number, offset: number, size: number ): Promise<void>; getMappedRange( offset: number, size: number ): ArrayBuffer; unmap(): void };
type GPUDevice = {
	createBuffer( descriptor: { label: string; size: number; usage: number } ): GPUBuffer;
	createCommandEncoder( descriptor: { label: string } ): { finish(): object };
	queue: { submit( buffers: object[] ): void };
};

// The frame (three's `info.frame`) a timestamp uid belongs to (`<prefix>:<id>:f<frame>`).
function frameOf( uid: string ): number {

	return Number( uid.slice( uid.lastIndexOf( ':f' ) + 2 ) );

}

// GPUBufferUsage.MAP_READ | COPY_DST, GPUMapMode.READ
const READBACK_USAGE = 0x0001 | 0x0008, MAP_READ = 0x0001;
const RING = 8;

/** The GPU's busy time per frame from three's timestamp queries (`trackTimestamp`), read back
 * through a small ring of buffers (three's pool maps a single one and drops the frames it
 * resolves while that one is mapped), and the part of it in passes that follow the render's pixel
 * count (`PassScaling`). Call `resolve()` once per frame, after its passes. */
export class GpuFrameTimer {

	/** Called with each frame's id (three's `info.frame`), its GPU busy time in ms and the part of it
	 * in passes that follow the pixel count. */
	onFrame: ( ( frame: number, ms: number, pixel: number ) => void ) | null = null;
	private readonly renderer: THREE.WebGPURenderer;
	private readonly passes: PassScaling;
	private readonly buffers: { buffer: GPUBuffer; busy: boolean }[] = [];
	private readonly frames = new Map<number, [ number, number, boolean ][]>();
	private origin: bigint | null = null;
	private lastEnd = - Infinity;
	private newest = - 1;
	// the newest frame reported, and frames some of whose passes went unread
	private reported = - 1;
	private readonly lost = new Set<number>();

	private constructor( renderer: THREE.WebGPURenderer, passes: PassScaling ) {

		this.renderer = renderer;
		this.passes = passes;

	}

	/** A timer, or null where the device has no timestamp queries (or the renderer does not track them). */
	static create( renderer: THREE.WebGPURenderer, passes: PassScaling ): GpuFrameTimer | null {

		return tracksTimestamps( renderer ) ? new GpuFrameTimer( renderer, passes ) : null;

	}

	resolve(): void {

		// three resolves a pool on its own only when it overflows: route that through here too
		const pools = timestampPools( this.renderer, 'frameTimer', ( pool ) => this.resolvePool( pool ) ).filter( ( pool ) => pendingTimestamps( pool ) > 0 );
		// The render and compute pools are read together or not at all: a frame read from one of them
		// alone (its compute passes, the cloud view among them, left out) would pass for a light one.
		const free = this.buffers.filter( ( b ) => ! b.busy ).length + RING - this.buffers.length;
		for ( const pool of pools ) this.resolvePool( pool, free < pools.length );

	}

	private resolvePool( pool: TimestampPool, drop = false ): void {

		if ( pendingTimestamps( pool ) === 0 ) return;
		const taken = takeTimestamps( pool ), count = taken.count, capacity = timestampCapacity( pool );
		const offsets = [ ...taken.offsets ].map( ( [ uid, base ] ) => [ uid, base, this.passes.take( uid ) ] as const );
		const device = gpuDevice<GPUDevice>( this.renderer );
		let slot = drop ? undefined : this.buffers.find( ( b ) => ! b.busy && b.buffer.size >= capacity * 8 );
		if ( ! slot && ! drop && this.buffers.length < RING ) {

			slot = { buffer: device.createBuffer( { label: 'frameTimer.timestamps', size: capacity * 8, usage: READBACK_USAGE } ), busy: false };
			this.buffers.push( slot );

		}
		if ( ! slot ) {

			// every read-back buffer is in flight: these frames go unmeasured, with any of their passes
			// read in another batch
			for ( const [ uid ] of offsets ) this.lost.add( frameOf( uid ) );
			return;

		}
		const target = slot;
		target.busy = true;
		const bytes = count * 8;
		const encoder = device.createCommandEncoder( { label: 'frameTimer.resolve' } );
		resolveTimestamps( encoder, pool, count, target.buffer );
		device.queue.submit( [ encoder.finish() ] );
		target.buffer.mapAsync( MAP_READ, 0, bytes ).then( () => {

			const times = new BigUint64Array( target.buffer.getMappedRange( 0, bytes ).slice( 0 ) );
			target.buffer.unmap();
			target.busy = false;
			for ( const [ uid, base, pixel ] of offsets ) {

				const start = times[ base ], end = times[ base + 1 ];
				if ( start === 0n || end < start ) continue;
				const frame = frameOf( uid );
				// (a pass of a frame already reported arrived late: that frame is gone)
				if ( ! Number.isFinite( frame ) || frame <= this.reported ) continue;
				if ( this.origin === null ) this.origin = start;
				const list = this.frames.get( frame ) ?? [];
				list.push( [ Number( start - this.origin ) / 1e6, Number( end - this.origin ) / 1e6, pixel ] );
				this.frames.set( frame, list );
				this.newest = Math.max( this.newest, frame );

			}
			this.settle();

		}, () => { target.busy = false; } );

	}

	// A frame is complete once passes of two later frames arrived (work issued after a tick, such as
	// a sky bake resuming after a read-back, carries the frame id of the tick before it).
	private settle(): void {

		const ready = [ ...this.frames.keys() ].filter( ( f ) => f < this.newest - 1 ).sort( ( a, b ) => a - b );
		for ( const frame of ready ) {

			const passes = this.frames.get( frame )!.sort( ( a, b ) => a[ 0 ] - b[ 0 ] );
			this.frames.delete( frame );
			this.reported = Math.max( this.reported, frame );
			let cursor = this.lastEnd, busy = 0, pixel = 0;
			for ( const [ start, end, follows ] of passes ) {

				const ms = Math.max( end - Math.max( start, cursor ), 0 );
				busy += ms;
				if ( follows ) pixel += ms;
				cursor = Math.max( cursor, end );

			}
			this.lastEnd = cursor;
			if ( ! this.lost.delete( frame ) ) this.onFrame?.( frame, busy, pixel );

		}
		// frames lost whole never come up here
		for ( const frame of this.lost ) if ( frame < this.newest - 64 ) this.lost.delete( frame );

	}

}
