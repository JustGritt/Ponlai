import * as THREE from 'three/webgpu';
import {
	Fn, If, float, vec2, vec3, vec4, ivec2, texture, textureLoad, uniform, screenCoordinate, floor, exp, dot, max, min,
	clamp, abs, sqrt, select, mix, smoothstep, length, round, saturate, sRGBTransferOETF, sRGBTransferEOTF, outputStruct, mrt, mat4,
} from 'three/tsl';
import { debugFlags } from '../debug/DebugFlags.ts';
import type { TemporalSettings } from '../Quality.ts';
import { jitter } from './Jitter.ts';
import { motionState, motionUniforms, onTaaCut, reactiveUniforms } from './Motion.ts';
import { TemporalRest, restEpoch, restKey } from './TemporalRest.ts';
import { sceneEvents } from './SceneEvents.ts';
import { FSR, rcas } from './FSR.ts';
import { lanczosHistory } from './TemporalHistory.ts';
import { carriedReactive, clampDrop, motionCap, movedLastFrame, temporalReactive } from './TemporalLocks.ts';
import { temporalSharpness } from './Antialiasing.ts';
import type { FrameGraph, ResourceHandle, SizeClass } from './graph/FrameGraph.ts';
import { resolveSize, type Extents } from './graph/GraphCompiler.ts';
import { compileChain, compileQuad } from './graph/Compile.ts';

// The temporal modes' resolve: jittered frames accumulated into a reprojected history.
//
// The temporal modes' resolve (Antialiasing.ts: aa=taa|taau, and the ground truth aa=gt). The scene
// renders single-sampled with its projection shifted by a sub-pixel jitter each frame (Jitter.ts),
// writing motion vectors beside its colour (Motion.ts); the post chain grades that frame as it does
// any other (order B: the resolve comes after the grade and the look, on the `display` image), and
// this pass accumulates the graded frames into a history, each output pixel reprojected along its
// motion vector and its history clipped to what the current frame shows around it. After Brian
// Karis, "High Quality Temporal Supersampling" (SIGGRAPH 2014), and Lasse Jon Fuglsang Pedersen's
// TAA for Playdead's INSIDE ("Temporal Reprojection Anti-Aliasing in INSIDE", GDC 2016): the motion of
// the nearest surface in a 3x3 neighbourhood, FSR 3.1's history fetch (a Lanczos 2 over 4 x 4
// texels with its deringing: TemporalHistory.ts, whose header has why and what it measured) and
// variance clipping (Marco Salvi, "An Excursion in Temporal Supersampling", GDC 2016) in YCoCg toward
// the neighbourhood's mean.
//
// - Accumulation domain: linear light. Display values (the grade's sRGB encoding) average darker at
//   partial coverage (a thin blade over bright ground came out darker than either), so under
//   `linear` the display pass stops before the encoding and writes the graded colour in linear
//   light, clamped to 0..1, in half float (Pipeline.ts); the resolve reads it as it is and the canvas
//   pass encodes the resolved image (the sRGB transfer). The resolve once undid the encoding per tap
//   (nine taps a pixel at the canvas's size): the same maths at about four times the cost.
// - Reconstruction (TAAU after AMD's FSR 2 and Unreal's TSR): the current frame's samples are
//   weighted by their distance to the output pixel, a Gaussian in output pixels at their jittered
//   positions (sigma 0.42, 0.30 at rest), plus 5 % of a wide Gaussian (0.6 input pixels) so a pixel
//   with no sample near it is not left to one. The history keeps the accumulated weight in its
//   alpha, as FSR 2 does, so an upscaled image gains detail over frames. The sky does not jitter
//   (Sky.ts draws it unjittered and the clouds accumulate in their own history): its taps count at
//   their pixel's centre, and a neighbourhood of sky alone is the frame's own sky, a Catmull-Rom of
//   the input at the output pixel, without a history (so pure sky does not wobble or average twice).
// - Reprojection: the motion of the nearest tap (Karis' depth dilation); the sky's own, where only
//   sky is near, from the camera's rotation alone (a direction, w = 0). A point the previous frame
//   did not show is found from the previous frame's nearest surface (`surfaceDepthPrev`) at the
//   reprojected uv, rebuilt into the world with the previous camera: more than 2 % (plus 5 cm) away
//   from the point is a disocclusion, and its history goes (unless the 3x3 straddles an edge, where
//   the jitter alone moves what the pixel shows and the clip box handles it).
// - The history's weight is capped at a number of frames' worth: more still, fewer moving, scaled by
//   the surface's class (the sea and the islanders keep less; the sea a whole history again with
//   distance, `waterFar`, where its waves' phase motion, which the motion vectors leave out, is under a
//   pixel a frame: a short history there left the jitter as a flickering line where a moon's or sun's
//   glitter meets the horizon, and the glints' reactive weight fades over the same distance), by the
//   reactive inputs (the velocity's
//   alpha: a bolt, and under the ground truth the rain and spray; a strike's flash; under the ground
//   truth the rain veil's per-pixel reactive `veilReactive`; after FSR 2's reactive and
//   transparency-and-composition masks).
// - Movers (taa, taau; not lite, not the ground truth's frames; each a debug flag read every frame,
//   `taaVelocityReject`, `taaClampDrop`, `taaMotionCap`, `taaWiden`, all four `taaMovers`; on by
//   default): where things move the history must go faster than the frames' count alone lets it.
//   The state (`taaMotion`, beside the history and ping-ponged with it, its size; reset with it by a
//   cut, and not the previous frame's after a resize or a graph rebuild, `stateValid`) keeps per
//   output pixel the nearest surface's own motion (its dilated motion less the camera's at its depth)
//   and FSR 2's temporal reactive (TemporalLocks.ts). Velocity rejection (after Unity HDRP's TAA and
//   Unreal's TSR): the own motion now against the one the history at uvPrev was written with, from
//   none at 0.5 to all at 2 output pixels apart, drops the history and narrows the box to its moving
//   width; behind a walking islander the history is the islander's (the disocclusion test skips the
//   depth edge next to it, where the jitter alone moves what a pixel shows). The camera's motion is
//   left out because its acceleration changes every pixel's motion at once: compared whole, a flick
//   rejected the whole frame's history (its ghosting doubled). FSR 2's motion cap (at most 10 frames' worth times the class's share
//   while the pixel moves or moved last frame, one frame at 20 pixels a frame, less by the carried
//   reactive) and its clamp-drop (a clipped history of a moving or rejected pixel drops its weight
//   toward a frame, by how far outside the box it lay; still pixels keep theirs, so a still camera's
//   jitter clips do not flicker). Where the history goes (disoccluded, rejected, dropped) the frame's
//   own estimate is the wide Gaussian's, softer instead of aliased (FSR 2's kernel bias). The sea and
//   the sky are left out (the sea's motion vectors leave out its waves' phase). Measured at high taau
//   against the ground truth: the islander orbit's ghosting about a third lower and its sharpness at
//   the ground truth's, the slow pan's ghosting lower, for 5-10 % more high-pass flicker on moving
//   crops; the islander walk's ghosting is mostly the swaying meadow behind it (its shading changes,
//   which no motion vector carries), which the movers barely touch. The flags switch the measures
//   by uniforms in the same shader, so `profile-frame --toggle` on them shows no cost: time the
//   second stage by a load with `flags=taaSmooth:false` (built without its target) and the movers
//   with `--compare` against a build without them.
// - The second stage (taa, taau; not lite, not the ground truth; under `linear`, where the canvas
//   pass reads the encoded image, and TAA's encode below the canvas reads its image; `taaSmooth =
//   false` builds without it):
//   a still camera's thin detail flickered at 2-8 times MSAA's high-pass flicker on a sunlit meadow
//   (the sun-shadow clip), and the history's accumulation, not its clip, let it through: the
//   jitter's sub-pixel hits and misses of blades and the vegetation's coverage dither
//   (vegetation/Coverage.ts) change a pixel every frame, and an exponential average of N frames passes about
//   1/N of each frame's change (with no jitter and the dither held the meadow measured below MSAA;
//   without any clip on still pixels it barely moved; reordering the jitter and the dither in time
//   bought a few per cent, 48 still frames a third). The image shown is a second exponential
//   average of the history's (`taaSmooth`, per output pixel, ping-ponged beside the history and reset
//   with it): it takes the new image by the history's own share of the new frame, so it drops
//   whatever the history drops (a cut, a disocclusion, a rejection, a clamp-drop, the reactive
//   inputs), and it runs only on still opaque and vegetation pixels (it gives way from 0.02 to 0.1
//   output pixels a frame, and to a strike's flash), its previous image clipped to twice the
//   history's box so a change of light is not held back. It also gives way where the first stage's
//   history was clipped (from the box's edge to twice its width): shading that changes with no
//   motion vector (rain ripples on a puddle, a moving shadow on still ground) is what the clip
//   catches, and without this the puddles' ghosting rose by three quarters (by a sixth with it).
//   Two averages in a row pass about 1/N^2 of a frame's change at the cost of a second N frames'
//   lag on slow changes the box lets through (a shadow edge drifting with the sun: its difference
//   from MSAA rose by about a fifth; a slow change of the whole view's light shows about twice as
//   late). Measured at high taau: on the sun-shadow clip the meadow's high-pass flicker from 8.1
//   and 3.7 to 1.9 and 1.2 times MSAA's, the cliffs from 2.0 and 1.7 to 1.2 and 0.8; with the wind
//   running at the same view the meadow's from 1.0 to 0.5 (its blades there sway under the exit
//   speed); the islander walk's ghosting unchanged. The history itself (what reprojects and
//   accumulates) stays the first stage's, and so does everything the rest shows.
// - TAAU resolves straight to the canvas's size and keeps that history across render-scale steps
//   (read by uv); TAA resolves at the render's size and resamples its history once when that changes.
//   The history's weight is counted in render samples, so after a step up (more samples a pixel)
//   the same history would count as fewer frames' worth and the new, sharper frame would take a
//   larger share at once: the first frame showed most of the step (on the sea, whose class keeps
//   fewest frames, a sharpness step 0.3x MSAA+FSR's, over the 0.25x allowed). For the frame after a
//   step up the history's weight is scaled by the new density over the old (`historyGain`), so it
//   keeps its frames' worth and the step spreads over the frames as a step down's does (whose
//   history the cap already holds at its frames' worth).
// - TAAU's canvas-size work costs the same at every render scale, so at a low scale it is most of
//   what the upscale costs: what depends only on the input pixel nearest an output pixel is taken
//   once per input pixel, by a pass at the render's size (`taaNeighbourhood`, after FSR 2's
//   reconstruct-and-dilate pass): the 3x3's clip box (its mean and half-widths), the history's
//   frames' worth there, whether it straddles an edge, and the nearest surface's motion and depth
//   (its view z over NEAR_Z_SCALE, in half float). The running resolve (and its debug views) reads
//   them at that pixel and reconstructs the frame from the 2x2 nearest samples (the third on each
//   axis lies an input pixel or more away, where the narrow Gaussian keeps under 0.7 % of its weight
//   at TAAU's largest ratio and none at its floor's; it held 3 - 6 % of the wide one's, which only
//   a pixel far from every sample leans on). From the floor at a 4K canvas the resolve and that
//   pass together took about two thirds of the 3x3 resolve's time. The rest's resolve, TAA's (the
//   same size in and out) and lite keep the 3x3 in one pass.
// - Rest (TemporalRest.ts): a resting scene runs a fixed jitter sequence through the rest resolve
//   (no clip, no cap, nothing reprojected, exact weights) into an image of its own, shown once it is
//   complete; the ground truth's rest is a box filter over 64 jitters (centred on their mean:
//   Jitter.ts) at the canvas's resolution.
// - The canvas pass sharpens the encoded image and adds what must not be accumulated: the vignette
//   (in linear, before the encoding, as the grade did), RCAS sharpening (FSR.ts) and the
//   dither. A class with `sharpenMoving` stops is sharpened more where it moves on screen (by its
//   speed from the velocity under the pixel): the sea, whose short history, resampled each frame,
//   came out at 0.7 of the ground truth's sharpness under a pan (vegetation keeps the frame's: more
//   sharpening there raised its crawl). Under `linear` the running resolve writes its image encoded as a second target
//   (`taaEncoded`, the vignette at each pixel), so RCAS's five taps are not encoded one by one
//   (they were half the canvas pass's time at 3840x2160; the extra half-float write costs less
//   than that). The rest's image and the ground truth (one tap) are encoded in the canvas pass.
//   TAA below the canvas encodes at the render's size (the second stage's image where it runs)
//   and upscales with EASU first. The running resolve's attachments, the history, `taaEncoded`,
//   the movers' state and the second stage's image, are four half-float RGBA targets: 32 bytes a
//   sample, WebGPU's limit for one pass's colour attachments, so a fifth needs packing (the
//   reactive into the second stage's alpha, the state into RG16F).
// - After the resolve (taa, taau: AfterResolve.ts) the rain's streaks and the storm's spray are
//   drawn unjittered into an overlay at the canvas's size, between the resolve's passes and the
//   canvas pass (`after`), and the canvas pass adds the change the rain veil and that overlay make
//   to the graded colour (`layer`) after RCAS, before the dither: the pixels it changes are decoded
//   to linear light (the sRGB transfer undone), changed and encoded
//   again; the vignette applies to the change as to the image. Nothing of it enters a history. A
//   running frame's canvas pass draws the change itself (`layerInline`: a build of its own, used
//   only while the layer shows, so in fair weather the pass holds none of its registers); the
//   frame that completes the rest has a pass draw it into a target (`after`), which the canvas
//   pass reads as it is while the rest holds (`drawsLayer`), as does TAA's upscaled canvas pass.
//   Drawn by its own pass every frame, the change cost a canvas-size target written and read.

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;
type Uniform2 = THREE.UniformNode<'vec2', THREE.Vector2>;

export type TemporalMode = 'taa' | 'taau' | 'gt';

const RENDER: SizeClass = { of: 'render' };
const CANVAS: SizeClass = { of: 'canvas' };
// view z beyond which a pixel is sky (the resolved sky is -1e5)
const SKY_Z = - 9e4;
// the surface classes (Motion.ts MOTION_CLASS: opaque, vegetation, water, character, sky), each with
// its share of the history's frames and its clip box's extra width (TemporalSettings classFrames,
// classGamma)
const CLASSES = 5;
// one value per class (a uniform each), picked by a class index node (0 ... 4)
const byClassOf = ( cls: THREE.Node<'float'>, values: THREE.UniformNode<'float', number>[] ): THREE.Node<'float'> => {

	let v: THREE.Node<'float'> = values[ CLASSES - 1 ];
	for ( let i = CLASSES - 2; i >= 0; i -- ) v = select( cls.lessThan( i + 0.5 ), values[ i ], v );
	return v;

};
// the wide Gaussian's sigma (input pixels) and its share of a frame's weight
const WIDE_SIGMA = 0.6, WIDE_SHARE = 0.05;
// the sharpening's easing after a change of render scale (seconds)
const SHARPNESS_EASE = 0.25;
// the 3x3 around an input pixel, row by row
const TAPS_3X3 = [ - 1, 0, 1 ].flatMap( ( dy ) => [ - 1, 0, 1 ].map( ( dx ) => [ dx, dy ] ) );
// TAAU's neighbourhood pass keeps the nearest surface's view z over this in half float (the sky's
// -1e5 lies beyond half float's range; the disocclusion's tolerance, 2 %, far above its precision)
const NEAR_Z_SCALE = 1000;
// The movers (see the header), each a debug flag read every frame (unset: `MOVERS`), and the classes
// they act on (opaque, vegetation, character; not the water, whose motion vectors leave out its
// waves' phase, nor the sky)
const MOVER_FLAGS = [ 'taaVelocityReject', 'taaClampDrop', 'taaMotionCap', 'taaWiden' ] as const;
const MOVERS = true;
const MOVER_CLASSES = [ 1, 1, 0, 1, 0 ];
// the second stage (see the header): the classes it smooths (opaque, vegetation) and the speed
// (output pixels a frame) over which it gives way to the first stage's image
const SMOOTH_CLASSES = [ 1, 1, 0, 0, 0 ];
const SMOOTH_SPEED_FROM = 0.02, SMOOTH_SPEED_TO = 0.1;
// its previous image's clip box, in widths of the history's
const SMOOTH_BOX = 2;
// velocity rejection: the change of the dilated motion (output pixels a frame) from none to all
const REJECT_FROM = 0.5, REJECT_TO = 2;
// FSR 2's frames' worth for a moving pixel (times the class's share)
const CAP_FRAMES = 10;

/** The graph's resources the resolve reads (their versions after the passes before it). */
export interface TemporalInputs {
	// the graded image at the render's size, before the vignette and dither (in linear light under
	// `linear`, else display values)
	display: ResourceHandle;
	velocity: ResourceHandle;
	// the nearest surface's view z, this frame's and the previous frame's
	surfaceDepth: ResourceHandle;
	surfaceDepthPrev: ResourceHandle;
	// the rain veil's per-pixel reactive weight (the composite writes it: Pipeline.ts; none when the
	// veil is drawn after the resolve)
	veilReactive: ResourceHandle | null;
	canvas: ResourceHandle;
	// after the resolve (AfterResolve.ts): declares the pass that draws the overlay, after the
	// resolve's and before the canvas pass, and returns what the canvas passes read (`optional`:
	// written by a pass that may not run)
	after?: () => { optional: ResourceHandle[]; required: ResourceHandle[] };
}

export interface TemporalOptions {
	mode: TemporalMode;
	settings: TemporalSettings;
	// the grade's vignette factor at the pixel drawn into an image of `size`
	vignette: ( size: Uniform2 ) => F;
	// the canvas's dither (`on` 0 leaves them out)
	finish: ( s: V3, size: Uniform2, on: F ) => V4;
	// a fresh node (one per shader): what the canvas pass adds to its image at its pixel, after the
	// resolve (the change to the graded colour, in linear light under `linear`, without the vignette;
	// alpha 1 where there is one, else 0)
	layer?: () => V4;
	// the layer itself (a fresh node per shader), which the canvas pass of a running frame draws while
	// `on` (it rains, or the after pass drew), instead of reading it from a pass of its own
	layerInline?: { node: () => V4; on: () => boolean };
}

// The textures the resolve reads, and under TAAU the neighbourhood pass's targets.
interface ResolveTextures {
	display: THREE.Texture; velocity: THREE.Texture; depth: THREE.Texture; depthPrev: THREE.Texture; veil: THREE.Texture | null;
	near: { stats: THREE.Texture; clip: THREE.Texture; nearest: THREE.Texture } | null;
}

/** Per-frame inputs from the app: what else keys the rest, and whether the app is stable. */
export interface TemporalFrame {
	// the app's own signature of the rest (a forced strike), as numbers
	signature?: readonly number[];
	// false while something changes the image on its own (a sky bake in flight, a transition)
	stable?: boolean;
	// the frame's delta (seconds), for the sharpening's easing
	dt?: number;
	// the scene clock's step this frame (seconds, 0 while paused) and the pacer's nominal interval
	// (ms): FSR 3's frame time (Fsr3Upscaler.ts)
	step?: number;
	interval?: number;
}

/** The debug views of the resolve (graph/DebugView.ts). */
export type TemporalDebugView = 'weight' | 'disocclusion' | 'clip';

// linear to display values (the sRGB transfer), and back
const toDisplay = ( c: V3 ) => sRGBTransferOETF( clamp( c, 0, 1 ) ) as V3;
const fromDisplay = ( s: V3 ) => sRGBTransferEOTF( clamp( s, 0, 1 ) ) as V3;

const toYCoCg = ( c: V3 ) => vec3( dot( c, vec3( 0.25, 0.5, 0.25 ) ), dot( c, vec3( 0.5, 0, - 0.5 ) ), dot( c, vec3( - 0.25, 0.5, - 0.25 ) ) );
const fromYCoCg = ( c: V3 ) => vec3( c.x.add( c.y ).sub( c.z ), c.x.add( c.z ), c.x.sub( c.y ).sub( c.z ) );

// Catmull-Rom weights of the four taps around a sample at fraction t past the second
const catmullRom = ( t: F ) => {

	const t2 = t.mul( t ), t3 = t2.mul( t );
	return [ t3.mul( - 0.5 ).add( t2 ).sub( t.mul( 0.5 ) ), t3.mul( 1.5 ).sub( t2.mul( 2.5 ) ).add( 1 ), t3.mul( - 1.5 ).add( t2.mul( 2 ) ).add( t.mul( 0.5 ) ), t3.mul( 0.5 ).sub( t2.mul( 0.5 ) ) ];

};

/** A full-screen pass built once per texture it reads (a ping-pong history's two, a held image):
 * the graph renders the variant `pick` names. */
interface Variants { pass: string; kind: 'quad' | 'chain'; quads: THREE.QuadMesh[]; pipelines: THREE.RenderPipeline[] }

export class TemporalAA {

	/** The app's resolve, once built (the debug view asks for it while the graph compiles). */
	static current: TemporalAA | null = null;
	readonly mode: TemporalMode;
	/** The live settings (the GUI tunes them; the uniforms follow every frame). */
	readonly settings: TemporalSettings;
	/** The rest convergence. */
	readonly rest = new TemporalRest();
	/** Times the running resolve started over from nothing after its first frame: cuts, resizes
	 * that could not carry the history, graph rebuilds (the diagnostics' `historyResets`). */
	historyResets = 0;
	private resolved = false;
	/** The sharpening this frame (RCAS stops) and the one it eases toward. */
	readonly sharpness = uniform( 1 );
	sharpnessTarget = 1;
	/** Live parameters of the resolve outside the tier's table. */
	readonly params = {
		// disocclusion: relative and absolute depth tolerance (m)
		disoccludeRelative: uniform( 0.02 ), disoccludeAbsolute: uniform( 0.05 ),
	};

	private readonly graph: FrameGraph<unknown>;
	private readonly renderer: THREE.WebGPURenderer;
	private readonly options: TemporalOptions;
	private readonly historyClass: SizeClass;
	private readonly history: ResourceHandle;
	private readonly historyPrev: ResourceHandle;
	private readonly restImage: ResourceHandle;
	private readonly restPrev: ResourceHandle;
	private readonly restShown: ResourceHandle;
	// the movers' state (taa, taau; not lite, not the ground truth), per output pixel and ping-pong
	// beside the history: the surface's own motion (output pixels a frame, the camera's left out) and
	// FSR 2's temporal reactive (TemporalLocks.ts), written by the running resolve
	private readonly state: ResourceHandle | null = null;
	private readonly statePrev: ResourceHandle | null = null;
	// the second stage's image (`taaSmooth`), per output pixel and ping-pong beside the history
	private readonly smooth: ResourceHandle | null = null;
	private readonly smoothPrev: ResourceHandle | null = null;
	// taa below the canvas: the resolved image encoded at the render's size, and EASU's
	private readonly encoded: ResourceHandle | null = null;
	private readonly easuImage: ResourceHandle | null = null;
	private fsr: FSR | null = null;
	// the history of the size before a resize, read once by the next resolve (resampled)
	private readonly carry: THREE.RenderTarget;
	private carryValid = false;
	private readonly carrySize = new THREE.Vector2();
	private readonly variants: Variants[] = [];
	// the resolve's builds by parity and their history nodes (the carry swaps one in for a frame)
	private resolveQuads: THREE.QuadMesh[] = [];
	private resolveHistory: THREE.TextureNode[] = [];
	// TAAU's neighbourhood pass and its targets' versions (see the header)
	private neighbourhoodQuad: THREE.QuadMesh | null = null;
	private nearHandles: ResourceHandle[] | null = null;
	private restQuads: THREE.QuadMesh[] = [];
	private canvasPipelines: THREE.RenderPipeline[] = [];
	// the canvas pass of a running frame that draws the rain's layer itself, by the history's parity
	private inlinePipelines: THREE.RenderPipeline[] = [];
	private encodeQuads: THREE.QuadMesh[] = [];
	private upscaledPipeline: THREE.RenderPipeline | null = null;
	// the debug view's builds by parity (graph/DebugView.ts asks for one)
	private debugQuads: THREE.QuadMesh[] = [];
	private debugTarget: ResourceHandle | null = null;
	private sharpnessFrom = 1;
	private sharpnessTime = SHARPNESS_EASE;
	// render pixels per output pixel in the last frame (TAAU's history gain)
	private lastDensity = NaN;
	private settingsKey = '';
	private settingsVersion = 0;
	private readonly lastCameraWorld = new THREE.Matrix4();
	private readonly lastProjectionInverse = new THREE.Matrix4();
	private hasLast = false;
	private readonly key: number[] = [];
	// set by Pipeline while the frame is upscaled (taa: EASU after the resolve)
	upscaling = false;
	private readonly u = {
		historySize: uniform( new THREE.Vector2( 1, 1 ) ),
		reset: uniform( 1 ),
		previousDepthValid: uniform( 0 ),
		// TAAU, the frame after a step up of the render scale: the history's weight times the new
		// render density over the old (1 otherwise)
		historyGain: uniform( 1 ),
		framesStill: uniform( 12 ), framesStillWater: uniform( 12 ), framesMoving: uniform( 5 ), speedFull: uniform( 16 ),
		gammaStill: uniform( 1.25 ), gammaMove: uniform( 0.75 ), sigma: uniform( 0.42 ), sigmaRest: uniform( 0.3 ),
		classFrames: Array.from( { length: CLASSES }, () => uniform( 1 ) ), classGamma: Array.from( { length: CLASSES }, () => uniform( 0 ) ),
		skyPassThrough: uniform( 1 ),
		// the movers: whether the state holds the previous frame's, each measure on (1) or off, and
		// the classes they act on
		stateValid: uniform( 0 ), velocityReject: uniform( 1 ), clampDrop: uniform( 1 ), motionCap: uniform( 1 ), widen: uniform( 1 ), smoothOn: uniform( 1 ), smoothValid: uniform( 0 ), smoothClass: SMOOTH_CLASSES.map( ( v ) => uniform( v ) ),
		moverClass: MOVER_CLASSES.map( ( v ) => uniform( v ) ),
		sharpenMoving: Array.from( { length: CLASSES }, () => uniform( 0 ) ), sharpenSpeed: uniform( 4 ), waterFar0: uniform( 1e6 ), waterFar1: uniform( 2e6 ),
		// the camera's world matrix this frame and the previous rendered frame's, with its unjittered
		// inverse projection (the disocclusion test and the sky's rotation)
		cameraWorld: uniform( new THREE.Matrix4() ),
		previousCameraWorld: uniform( new THREE.Matrix4() ),
		previousProjectionInverse: uniform( new THREE.Matrix4() ),
	};

	constructor( graph: FrameGraph<unknown>, renderer: THREE.WebGPURenderer, options: TemporalOptions ) {

		this.graph = graph;
		this.renderer = renderer;
		this.options = options;
		this.mode = options.mode;
		this.settings = structuredClone( options.settings );
		const g = graph;
		// TAAU's history at the canvas's size, kept across render-scale steps; TAA's and the ground
		// truth's at the render's (the ground truth renders at the canvas's resolution)
		this.historyClass = this.mode === 'taau' ? CANVAS : RENDER;
		const desc = { size: this.historyClass, format: THREE.RGBAFormat, type: THREE.HalfFloatType };
		this.history = g.color( 'taaHistory', desc );
		this.historyPrev = g.previous( this.history );
		this.restImage = g.color( 'taaRest', desc );
		this.restPrev = g.previous( this.restImage );
		this.restShown = g.color( 'taaRestShown', desc, true );
		if ( this.mode !== 'gt' && ! this.settings.lite ) {

			this.state = g.color( 'taaMotion', desc );
			this.statePrev = g.previous( this.state );
			if ( debugFlags.taaSmooth !== false && this.settings.linear ) {

				this.smooth = g.color( 'taaSmooth', desc );
				this.smoothPrev = g.previous( this.smooth );

			}

		}
		if ( this.mode === 'taa' ) {

			// (half float: filterable, so EASU gathers)
			this.encoded = g.color( 'taaDisplay', { size: RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType } );
			this.easuImage = g.color( 'TAA.easu', { size: CANVAS, format: THREE.RGBAFormat, type: THREE.HalfFloatType } );

		}
		this.carry = new THREE.RenderTarget( 1, 1, { type: THREE.HalfFloatType, depthBuffer: false, generateMipmaps: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter } );
		this.carry.texture.name = 'taaHistory.carry';
		// across a cut the history is no longer this view's
		onTaaCut( () => {

			g.resetHistory( this.history );
			if ( this.state ) g.resetHistory( this.state );
			if ( this.smooth ) g.resetHistory( this.smooth );
			this.carryValid = false;
			// (no camera motion across it either)
			this.u.previousCameraWorld.value.copy( this.u.cameraWorld.value );
			this.u.previousProjectionInverse.value.copy( jitter.uniforms.unjitteredProjectionInverse.value );

		} );
		this.applySettings();
		TemporalAA.current = this;

	}

	/** The resolved image's size class: the canvas's for TAAU, else the render's. */
	get historySizeClass(): SizeClass {

		return this.historyClass;

	}

	/** Whether the image shown is the rest's. */
	get converged(): boolean {

		return this.rest.converged;

	}

	/** Whether the rest holds an image an earlier frame completed: what is drawn after the resolve
	 * is then kept from that frame (Pipeline.ts `afterHeld`), whose jitter is the rest sequence's
	 * last, rather than drawn again over this frame's running jitter. */
	get holding(): boolean {

		return this.rest.converged && ! this.rest.completing;

	}

	/** Whether the canvas pass draws the rain's layer itself this frame (`layerInline`): a running
	 * frame's, while the layer shows. The frame that completes the rest draws it into its target,
	 * which the frames after it read as it is (Pipeline.ts `afterHeld`), as does TAA's upscaled canvas
	 * pass. */
	get drawsLayer(): boolean {

		return this.options.layerInline?.on() === true && ! this.rest.converged && ! ( this.mode === 'taa' && this.upscaling );

	}

	// ---- the passes

	/** Declares the passes after `display`: the resolve, the rest and its hand-over, and the canvas
	 * pass (TAA below the canvas: the encoding, EASU and the sharpening canvas pass). */
	declare( inputs: TemporalInputs ): void {

		const g = this.graph;
		const O = this.options;
		const { display, velocity, surfaceDepth, surfaceDepthPrev, veilReactive } = inputs;
		let canvas = inputs.canvas;
		const renderSize = g.extent( RENDER ), outputSize = g.extent( this.historyClass ), canvasSize = g.extent( CANVAS );
		// TAAU (but lite, and with the veil's reactive weight, which the resolve reads per output pixel):
		// the 3x3's statistics at the render's size, from the neighbourhood pass (see the header)
		const near = this.mode === 'taau' && ! this.settings.lite && ! veilReactive ? ( [ 'taaStats', 'taaClip', 'taaNearest' ] as const ).map( ( name ) =>
			g.color( name, { size: RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType } ) ) : null;
		const textures: ResolveTextures = {
			display: g.texture( display ), velocity: g.texture( velocity ), depth: g.texture( surfaceDepth ),
			depthPrev: g.texture( surfaceDepthPrev ), veil: veilReactive ? g.texture( veilReactive ) : null,
			near: near ? { stats: g.texture( near[ 0 ] ), clip: g.texture( near[ 1 ] ), nearest: g.texture( near[ 2 ] ) } : null,
		};
		if ( near ) {

			const material = new THREE.NodeMaterial();
			material.name = 'TAA.Neighbourhood';
			material.fragmentNode = this.neighbourhood( textures, renderSize, outputSize );
			const quad = this.neighbourhoodQuad = new THREE.QuadMesh( material );
			quad.name = 'TAA.Neighbourhood';
			g.addPass( 'taaNeighbourhood', 'quad', ( b ) => {

				b.read( display );
				b.readOptional( velocity );
				b.read( surfaceDepth );
				// (the attachments in the struct's order; the MRT only declares them, as the resolve's)
				for ( let i = 0; i < near.length; i ++ ) near[ i ] = b.write( near[ i ] );
				this.nearHandles = near;
				return { label: 'TAA resolve', quad, mrt: mrt( {} ) };

			} );

		}
		// Under `linear` the running resolve also writes its image encoded (the vignette, the sRGB
		// transfer), so the canvas pass's RCAS reads encoded taps instead of encoding
		// five a pixel. The ground truth is not sharpened (one encode a pixel already).
		const gt = this.mode === 'gt';
		let canvasImage = this.settings.linear && ! gt ? g.color( 'taaEncoded', { size: this.historyClass, format: THREE.RGBAFormat, type: THREE.HalfFloatType } ) : null;
		// the resolve, built once per parity of the history it reads
		const build = ( history: THREE.Texture, state: THREE.Texture | null, kind: 'running' | 'rest' | TemporalDebugView, name: string, smooth: THREE.Texture | null = null ) => {

			const node = texture( history );
			const material = new THREE.NodeMaterial();
			material.name = name;
			material.fragmentNode = this.resolve( textures, node, state, kind, renderSize, outputSize, kind === 'running' && canvasImage !== null, smooth );
			const quad = new THREE.QuadMesh( material );
			quad.name = name;
			return { quad, node };

		};
		// (the state's parity is the history's: the resolve writes both every frame)
		const running = [ 0, 1 ].map( ( p ) => build( g.texture( this.historyPrev, p as 0 | 1 ), this.statePrev ? g.texture( this.statePrev, p as 0 | 1 ) : null, 'running', 'TAA.Resolve', this.smoothPrev && canvasImage ? g.texture( this.smoothPrev, p as 0 | 1 ) : null ) );
		this.resolveQuads = running.map( ( r ) => r.quad );
		this.resolveHistory = running.map( ( r ) => r.node );
		let history = this.history;
		let state = this.state;
		let smooth = canvasImage ? this.smooth : null;
		g.addPass( 'taaResolve', 'quad', ( b ) => {

			b.read( display );
			b.readOptional( velocity );
			b.read( surfaceDepth );
			b.read( surfaceDepthPrev );
			if ( veilReactive ) b.readOptional( veilReactive );
			b.read( this.historyPrev );
			if ( this.statePrev ) b.read( this.statePrev );
			if ( smooth && this.smoothPrev ) b.read( this.smoothPrev );
			for ( const h of near ?? [] ) b.read( h );
			history = b.write( history );
			// (the attachments in this order: the struct's)
			if ( canvasImage ) canvasImage = b.write( canvasImage );
			if ( state ) state = b.write( state );
			if ( smooth ) smooth = b.write( smooth );
			// (its material writes them through an output struct, which ignores the MRT: the MRT only
			// declares the attachments to the graph, and leaves them unblended)
			return { label: 'TAA resolve', quad: this.resolveQuads[ 0 ], mrt: canvasImage || state ? mrt( {} ) : null, run: () => this.runResolve() };

		} );
		this.variants.push( { pass: 'taaResolve', kind: 'quad', quads: this.resolveQuads, pipelines: [] } );

		// the rest's resolve, once per parity of the rest image it reads
		this.restQuads = [ 0, 1 ].map( ( p ) => build( g.texture( this.restPrev, p as 0 | 1 ), null, 'rest', 'TAA.Rest' ).quad );
		let restImage = this.restImage;
		g.addPass( 'taaRest', 'quad', ( b ) => {

			b.read( display );
			b.read( surfaceDepth );
			b.read( this.restPrev );
			restImage = b.write( restImage );
			return {
				label: 'TAA rest', when: () => this.rest.index !== null, whenLabel: 'rest sequence', quad: this.restQuads[ 0 ],
				run: () => {

					this.u.reset.value = this.rest.index === 0 ? 1 : 0;
					this.restQuads[ g.parityOf( restImage ) ].render( this.renderer );

				},
			};

		} );
		this.variants.push( { pass: 'taaRest', kind: 'quad', quads: this.restQuads, pipelines: [] } );
		// the complete rest image, handed to the shown image once
		let restShown = this.restShown;
		g.addPass( 'taaRestShow', 'callback', ( b ) => {

			b.readOptional( restImage );
			restShown = b.write( restShown, { clear: true } );
			return {
				label: 'TAA rest', when: () => this.rest.completing, whenLabel: 'rest complete', copy: true,
				run: () => this.renderer.copyTextureToTexture( g.texture( restImage ), g.texture( restShown ) ),
			};

		} );

		// what is drawn after the resolve, and what the canvas passes then read
		const afterReads = inputs.after?.() ?? { optional: [], required: [] };
		const layer = O.layer;
		// the layer's change to a canvas pixel's encoded colour `s`, whose vignette is `v` (`inline`: the
		// layer drawn here)
		const over = ( s: V3, v: F, inline = false ): V3 => {

			if ( ! layer ) return s;
			const l = ( inline ? O.layerInline!.node() : layer() ).toVar();
			const out = s.toVar();
			If( l.a.greaterThan( 0 ), () => {

				out.assign( this.settings.linear
					? toDisplay( fromDisplay( s ).add( l.rgb.mul( v ) ) )
					: s.add( l.rgb.mul( v ) ) );

			} );
			return out;

		};

		// What the canvas shows: the history this frame wrote (by its parity) or its encoded image, or
		// the rest's image.
		const sources = [ g.texture( history, 0 ), g.texture( history, 1 ), g.texture( this.restShown ) ];
		const sourceIndex = () => this.rest.converged ? 2 : g.parityOf( history );
		// the resolved image (linear) to display values, with the vignette (in linear, as the grade
		// applied it); accumulated in display values (`linear` off), the vignette applies to those
		const encode = ( c: V3, v: F ) => this.settings.linear ? toDisplay( c.mul( v ) ) : c.mul( v );
		// the sharpening at a canvas pixel `p`: the frame's, plus up to its class's `sharpenMoving` stops
		// where the surface there moves on screen (its speed in canvas pixels from the velocity at the
		// render pixel under it; still pixels, the sky and the rest keep the frame's)
		const moving = this.settings.sharpenMoving.stops.some( ( v ) => v > 0 ) && ! gt;
		const sharpnessAt = ( p: V2 ): F => {

			if ( ! moving ) return this.sharpness;
			const cell = ivec2( clamp( floor( p.mul( renderSize ).div( canvasSize ) ), vec2( 0 ), renderSize.sub( 1 ) ) );
			const v = ( textureLoad( textures.velocity, cell ) as unknown as V4 ).toVar();
			const speed = length( v.xy.mul( 0.5 ).mul( canvasSize ) );
			const stops = byClassOf( round( clamp( v.z, 0, 1 ).mul( 4 ) ), this.u.sharpenMoving );
			return max( this.sharpness.sub( stops.mul( smoothstep( 0, this.u.sharpenSpeed, speed ) ) ), 0 );

		};
		// one canvas pixel: the image encoded (with the vignette at that pixel), sharpened by RCAS; the
		// resolve's encoded image as it is (its taps carry the vignette at their own pixels)
		const canvasNode = ( source: THREE.Texture, isEncoded: boolean, inline = false ) => Fn( () => {

			const p = screenCoordinate.xy.floor().toConst();
			const last = outputSize.sub( 1 ).toConst();
			const v = O.vignette( canvasSize ).toConst();
			const load = ( dx: number, dy: number ) => ( textureLoad( source, ivec2( clamp( p.add( vec2( dx, dy ) ), vec2( 0 ), last ) ) ) as unknown as V4 ).rgb;
			const tap = ( dx: number, dy: number ) => vec4( isEncoded ? load( dx, dy ) : encode( load( dx, dy ), v ), 1 );
			// (the ground truth is not sharpened)
			const s = gt ? tap( 0, 0 ).rgb : rcas( tap, sharpnessAt( p.add( 0.5 ) ) );
			return O.finish( over( s.toVar(), v, inline ), canvasSize, float( 1 ) );

		} )();
		const canvasPass = ( b: Parameters<Parameters<FrameGraph<unknown>[ 'addPass' ]>[ 2 ]>[ 0 ] ) => {

			b.read( canvasImage ?? history );
			b.readOptional( restShown );
			if ( moving ) b.readOptional( velocity );
			for ( const h of afterReads.required ) b.read( h );
			for ( const h of afterReads.optional ) b.readOptional( h );
			canvas = b.write( canvas );

		};
		const canvasPipeline = ( source: THREE.Texture, i: number, inline: boolean ) => {

			const shown = i < 2 && canvasImage !== null;
			const pipeline = new THREE.RenderPipeline( this.renderer, canvasNode( shown ? g.texture( canvasImage! ) : source, shown, inline ) );
			pipeline.outputColorTransform = false;
			return pipeline;

		};
		this.canvasPipelines = sources.map( ( source, i ) => canvasPipeline( source, i, false ) );
		// (a running frame's: the history this frame wrote)
		this.inlinePipelines = layer && O.layerInline ? sources.slice( 0, 2 ).map( ( source, i ) => canvasPipeline( source, i, true ) ) : [];
		const taa = this.mode === 'taa';
		g.addPass( 'taaOutput', 'chain', ( b ) => {

			canvasPass( b );
			return {
				label: 'output', pipeline: this.canvasPipelines[ 0 ],
				...( taa ? { when: () => ! this.upscaling, whenLabel: '!upscaling' } : {} ),
				run: () => ( this.drawsLayer ? this.inlinePipelines : this.canvasPipelines )[ sourceIndex() ].render(),
			};

		} );
		this.variants.push( { pass: 'taaOutput', kind: 'chain', quads: [], pipelines: [ ...this.canvasPipelines, ...this.inlinePipelines ] } );
		if ( ! taa ) return;

		// TAA below the canvas: the resolved image encoded at the render's size (the vignette in
		// linear, as the grade had it), EASU to the canvas, and RCAS with the dither there.
		let encoded = this.encoded!, easuImage = this.easuImage!;
		const encodeNode = ( source: THREE.Texture ) => Fn( () => {

			const c = ( textureLoad( source, ivec2( screenCoordinate.xy ) ) as unknown as V4 ).rgb;
			return vec4( encode( c, O.vignette( renderSize ) ), 1 );

		} )();
		// (a running frame's image is the second stage's where it runs, as the canvas pass shows it)
		const encodeSources = smooth ? [ g.texture( smooth, 0 ), g.texture( smooth, 1 ), sources[ 2 ] ] : sources;
		this.encodeQuads = encodeSources.map( ( source ) => {

			const material = new THREE.NodeMaterial();
			material.name = 'TAA.Encode';
			material.fragmentNode = encodeNode( source );
			const quad = new THREE.QuadMesh( material );
			quad.name = 'TAA.Encode';
			return quad;

		} );
		g.addPass( 'taaEncode', 'quad', ( b ) => {

			b.read( smooth ?? history );
			b.readOptional( restShown );
			encoded = b.write( encoded );
			return {
				label: 'output & FSR', when: () => this.upscaling, whenLabel: 'upscaling', quad: this.encodeQuads[ 0 ],
				run: () => this.encodeQuads[ sourceIndex() ].render( this.renderer ),
			};

		} );
		this.variants.push( { pass: 'taaEncode', kind: 'quad', quads: this.encodeQuads, pipelines: [] } );
		const fsr = this.fsr = new FSR( g.texture( encoded ), g.texture( easuImage ), true, g );
		g.addPass( 'taaEasu', 'quad', ( b ) => {

			b.readOptional( encoded );
			easuImage = b.write( easuImage );
			return { label: 'output & FSR', when: () => this.upscaling, whenLabel: 'upscaling', quad: fsr.quad };

		} );
		const upscaled = this.upscaledPipeline = new THREE.RenderPipeline( this.renderer, Fn( () => O.finish( over( fsr.rcas( sharpnessAt( screenCoordinate.xy ) ).toVar(), O.vignette( canvasSize ) ), canvasSize, float( 1 ) ) )() );
		upscaled.outputColorTransform = false;
		g.addPass( 'taaUpscaledOutput', 'chain', ( b ) => {

			b.readOptional( easuImage );
			if ( moving ) b.readOptional( velocity );
			for ( const h of afterReads.required ) b.read( h );
			for ( const h of afterReads.optional ) b.readOptional( h );
			canvas = b.write( canvas );
			return { label: 'output & FSR', when: () => this.upscaling, whenLabel: 'upscaling', pipeline: upscaled };

		} );

	}

	/** Debug builds (graph/DebugView.ts): a pass after the resolve that draws one of its debug
	 * values (`view`) into a target of the history's size, and the target, for the view to show. */
	declareDebugView( view: TemporalDebugView, inputs: { display: ResourceHandle; velocity: ResourceHandle; surfaceDepth: ResourceHandle; surfaceDepthPrev: ResourceHandle; veilReactive: ResourceHandle | null } ): ResourceHandle {

		const g = this.graph;
		const renderSize = g.extent( RENDER ), outputSize = g.extent( this.historyClass );
		// (under TAAU the running resolve's own: with the neighbourhood pass's statistics)
		const near = this.nearHandles;
		const textures: ResolveTextures = {
			display: g.texture( inputs.display ), velocity: g.texture( inputs.velocity ), depth: g.texture( inputs.surfaceDepth ),
			depthPrev: g.texture( inputs.surfaceDepthPrev ), veil: inputs.veilReactive ? g.texture( inputs.veilReactive ) : null,
			near: near ? { stats: g.texture( near[ 0 ] ), clip: g.texture( near[ 1 ] ), nearest: g.texture( near[ 2 ] ) } : null,
		};
		let target = this.debugTarget ??= g.color( 'taaDebug', { size: this.historyClass, format: THREE.RGBAFormat, type: THREE.HalfFloatType } );
		this.debugQuads = [ 0, 1 ].map( ( p ) => {

			const material = new THREE.NodeMaterial();
			material.name = 'TAA.Debug';
			material.fragmentNode = this.resolve( textures, texture( g.texture( this.historyPrev, p as 0 | 1 ) ), this.statePrev ? g.texture( this.statePrev, p as 0 | 1 ) : null, view, renderSize, outputSize );
			return new THREE.QuadMesh( material );

		} );
		const history = this.history;
		g.addPass( 'taaDebug', 'quad', ( b ) => {

			b.read( inputs.display );
			b.readOptional( inputs.velocity );
			b.read( inputs.surfaceDepth );
			b.read( inputs.surfaceDepthPrev );
			if ( inputs.veilReactive ) b.readOptional( inputs.veilReactive );
			for ( const h of near ?? [] ) b.read( h );
			b.read( this.historyPrev );
			if ( this.statePrev ) b.read( this.statePrev );
			target = b.write( target );
			// (the history's parity is the frame's, as for the resolve)
			return { label: 'debug view', quad: this.debugQuads[ 0 ], run: () => this.debugQuads[ g.parityOf( history ) ].render( this.renderer ) };

		} );
		this.variants.push( { pass: 'taaDebug', kind: 'quad', quads: this.debugQuads, pipelines: [] } );
		return target;

	}

	// The resolve's shader: the running resolve, the rest's, or one of its debug values. Under TAAU
	// (`t.near`: the neighbourhood pass's targets) the running resolve and its debug views read the
	// 3x3's statistics from that pass and reconstruct the frame from the 2x2 nearest samples.
	private resolve( t: ResolveTextures, history: THREE.TextureNode, state: THREE.Texture | null, kind: 'running' | 'rest' | TemporalDebugView, inputSize: Uniform2, outputSize: Uniform2, encoded = false, smoothTex: THREE.Texture | null = null ): V4 {

		const O = this.options, S = this.settings, u = this.u, K = this.params;
		const rest = kind === 'rest';
		const box = rest && this.mode === 'gt';
		const lite = S.lite;
		const linear = S.linear;
		const near = rest ? null : t.near;
		// (the movers: the running resolve and its debug views, where the state exists)
		const movers = rest ? null : state;
		const out = Fn( () => {

			const q = screenCoordinate.xy; // output pixel centres, at .5
			// input pixels per output pixel
			const scale = inputSize.div( outputSize ).toVar();
			const pIn = q.mul( scale ).toVar();
			const j = jitter.uniforms.px;
			const last = inputSize.sub( 1 ).toVar();
			// the input pixel whose jittered sample lies nearest this output pixel, and its neighbours
			const base = floor( pIn.sub( j ) ).toVar();
			const sigma = rest ? u.sigmaRest : u.sigma;
			const kS = float( - 0.5 ).div( sigma.mul( sigma ) ).toVar();
			const sumS = vec3( 0 ).toVar(), wS = float( 0 ).toVar();
			const sumW = vec3( 0 ).toVar(), wW = float( 0 ).toVar();
			// (the 3x3's statistics, here unless the neighbourhood pass took them)
			const m1 = near ? null : vec3( 0 ).toVar(), m2 = near ? null : vec3( 0 ).toVar();
			// (the nearest and farthest taps: the rest needs neither)
			const nearZ = rest || near ? null : float( - 1e30 ).toVar(), farZ = rest || near ? null : float( 1e30 ).toVar();
			const nearCell = rest || near ? null : vec2( 0 ).toVar();
			const skies = near ? null : float( 0 ).toVar();
			// Under TAAU the 2x2 nearest samples: the nearest on each axis (base) and its neighbour on
			// the side the output pixel lies (the third lies at least an input pixel away, where the
			// narrow Gaussian keeps under 0.7 % of its weight at TAAU's largest ratio, 0.75, and none at
			// its floor's).
			const side = near ? pIn.sub( j ).sub( base ).toVar() : null;
			const taps = near ? [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ] : lite ? [ [ 0, 0 ], [ - 1, 0 ], [ 1, 0 ], [ 0, - 1 ], [ 0, 1 ] ] : TAPS_3X3;
			for ( const [ dx, dy ] of taps ) {

				const cell = ( side
					? base.add( vec2( dx ? select( side.x.lessThan( 0.5 ), float( - 1 ), float( 1 ) ) : float( 0 ), dy ? select( side.y.lessThan( 0.5 ), float( - 1 ), float( 1 ) ) : float( 0 ) ) )
					: base.add( vec2( dx, dy ) ) ).toVar();
				const tap = ivec2( clamp( cell, vec2( 0 ), last ) );
				const c = toYCoCg( ( textureLoad( t.display, tap ) as unknown as V4 ).rgb ).toVar();
				const z = ( textureLoad( t.depth, tap ) as unknown as V4 ).r.toVar();
				const sky = z.lessThan( SKY_Z );
				// the sample's offset from this output pixel (output pixels): a surface's at its jittered
				// position, the sky's at its pixel's centre (it does not jitter)
				const d = cell.add( 0.5 ).add( select( sky, vec2( 0 ), j ) ).div( scale ).sub( q ).toVar();
				const ws = box
					? select( abs( d.x ).lessThanEqual( 0.5 ).and( abs( d.y ).lessThanEqual( 0.5 ) ), float( 1 ), float( 0 ) )
					: exp( dot( d, d ).mul( kS ) );
				sumS.addAssign( c.mul( ws ) );
				wS.addAssign( ws );
				if ( ! box ) {

					const dIn = d.mul( scale );
					const ww = exp( dot( dIn, dIn ).mul( - 0.5 / ( WIDE_SIGMA * WIDE_SIGMA ) ) );
					sumW.addAssign( c.mul( ww ) );
					wW.addAssign( ww );

				}
				if ( m1 && m2 && skies ) {

					m1.addAssign( c );
					m2.addAssign( c.mul( c ) );
					skies.addAssign( select( sky, float( 1 ), float( 0 ) ) );

				}
				if ( nearZ && farZ && nearCell ) {

					If( z.greaterThan( nearZ ), () => {

						nearZ.assign( z );
						nearCell.assign( cell );

					} );
					farZ.assign( min( farZ, z ) );

				}

			}
			const n = taps.length;
			// the weight one frame brings on average: the Gaussian's integral over the samples' density
			// (the box's area, for the ground truth)
			const perFrame = ( box ? float( 1 ) : sigma.mul( sigma ).mul( 2 * Math.PI ) ).mul( scale.x.mul( scale.y ) ).toVar();
			// this frame's estimate: the near samples, plus a little of the wide one
			const wide = box ? float( 0 ) : perFrame.mul( WIDE_SHARE );
			const current = vec3( 0 ).toVar();
			if ( box ) current.assign( sumS.div( max( wS, 1e-6 ) ) );
			else current.assign( sumS.add( sumW.div( max( wW, 1e-6 ) ).mul( wide ) ).div( wS.add( wide ) ) );
			const currentWeight = box ? wS : wS.add( wide );
			// (TAAU: the 3x3's statistics at the nearest input pixel, from the neighbourhood pass)
			const at = near ? ivec2( clamp( base, vec2( 0 ), last ) ) : null;
			const hood = near && at ? {
				stats: ( textureLoad( near.stats, at ) as unknown as V4 ).toVar(),
				clip: ( textureLoad( near.clip, at ) as unknown as V4 ).toVar(),
				nearest: ( textureLoad( near.nearest, at ) as unknown as V4 ).toVar(),
			} : null;
			// (the 3x3 is all sky exactly when its nearest tap is)
			const zNear = hood ? hood.nearest.z.mul( NEAR_Z_SCALE ).toVar() : nearZ;
			const allSky = ( hood && zNear ? zNear.lessThan( SKY_Z ) : skies!.greaterThan( n - 0.5 ) ).toVar();
			// A neighbourhood of sky alone, without a history: the frame's own sky at the output pixel,
			// a Catmull-Rom of the input (16 loads; clamped to 0..1 in linear light, as the input is)
			const passThrough = allSky.and( u.skyPassThrough.greaterThan( 0.5 ) ).toVar();
			If( passThrough, () => {

				const sp = pIn.sub( 0.5 ).toVar();
				const b0 = floor( sp ).toVar();
				const f = sp.sub( b0 ).toVar();
				const wx = catmullRom( f.x ), wy = catmullRom( f.y );
				const acc = vec3( 0 ).toVar();
				for ( let y = 0; y < 4; y ++ ) for ( let x = 0; x < 4; x ++ ) {

					const texel = textureLoad( t.display, ivec2( clamp( b0.add( vec2( x - 1, y - 1 ) ), vec2( 0 ), last ) ) ) as unknown as V4;
					acc.addAssign( texel.rgb.mul( wx[ x ].mul( wy[ y ] ) ) );

				}
				current.assign( toYCoCg( linear ? clamp( acc, 0, 1 ) : acc ) );

			} );

			// ---- the history
			const historyWeight = float( 0 ).toVar();
			const clipped = vec3( 0 ).toVar();
			const disoccluded = float( 0 ).toVar();
			const reach = float( 0 ).toVar();
			// (the movers' state this pixel writes: its motion, and the temporal reactive)
			const stateOut = movers ? vec4( 0 ).toVar() : null;
			// (the second stage's image this pixel shows and writes: the first stage's unless it smooths)
			const smoothed = smoothTex && movers ? vec3( 0 ).toVar() : null;
			const beta = smoothed ? float( 1 ).toVar() : null;
			if ( rest ) {

				// at rest nothing moves: the image accumulated so far, at this pixel, every weight exact (a
				// frame whose neighbourhood was all sky adds its sky to the average: near the horizon the
				// jitter decides which frames those are, and the rest averages them all)
				const hist = ( textureLoad( history, ivec2( q ) ) as unknown as V4 ).toVar();
				clipped.assign( toYCoCg( max( hist.rgb, vec3( 0 ) ) ) );
				historyWeight.assign( select( u.reset.lessThan( 0.5 ), max( hist.a, 0 ), float( 0 ) ) );

			} else if ( zNear ) {

				// where the point was: the nearest surface's motion, or the sky's rotation
				const velocity = hood ? hood.nearest : ( textureLoad( t.velocity, ivec2( clamp( nearCell!, vec2( 0 ), last ) ) ) as unknown as V4 ).toVar();
				const uvNow = q.div( outputSize ).toVar();
				const uvPrev = uvNow.sub( velocity.xy.mul( vec2( 0.5, - 0.5 ) ) ).toVar();
				const sky = zNear.lessThan( SKY_Z ).toVar();
				const P = jitter.uniforms.unjitteredProjectionInverse;
				If( sky, () => {

					const ndc = vec2( uvNow.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uvNow.y.mul( 2 ) ) );
					const ray = P.mul( vec4( ndc, 1, 1 ) ).toVar();
					const dir = u.cameraWorld.mul( vec4( ray.xyz.div( ray.w ), 0 ) ).xyz;
					const clip = motionUniforms.previousViewProjection.mul( vec4( dir, 0 ) ).toVar();
					uvPrev.assign( vec2( clip.x.div( clip.w ).mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( clip.y.div( clip.w ).mul( 0.5 ) ) ) );

				} );
				const inside = uvPrev.x.greaterThanEqual( 0 ).and( uvPrev.x.lessThanEqual( 1 ) ).and( uvPrev.y.greaterThanEqual( 0 ) ).and( uvPrev.y.lessThanEqual( 1 ) ).toVar();
				// The movers (see the header): this pixel's motion in output pixels, its class (the
				// nearest surface's), and the state its history was written with, at the output pixel
				// nearest uvPrev (none outside, or when the state is not the previous frame's).
				const mv = movers ? ( () => {

					const motion = uvNow.sub( uvPrev ).mul( outputSize ).toVar();
					const speed = length( motion ).toVar();
					// the surface's own motion: where a still surface at its depth was (the camera's motion
					// alone) to where it was, so the camera's own acceleration (a flick) rejects nothing
					const ndc = vec2( uvNow.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uvNow.y.mul( 2 ) ) );
					const ray = P.mul( vec4( ndc, 1, 1 ) ).toVar();
					const dir = ray.xyz.div( ray.w ).toVar();
					const still = motionUniforms.previousViewProjection.mul( u.cameraWorld.mul( vec4( dir.mul( zNear.div( dir.z ) ), 1 ) ) ).toVar();
					const uvStill = vec2( still.x.div( still.w ).mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( still.y.div( still.w ).mul( 0.5 ) ) );
					const own = select( sky, vec2( 0 ), uvStill.sub( uvPrev ).mul( outputSize ) ).toVar();
					const cls = round( clamp( hood ? hood.nearest.w : velocity.z, 0, 1 ).mul( 4 ) ).toVar();
					const mask = byClassOf( cls, u.moverClass ).toVar();
					const known = select( inside, u.stateValid, float( 0 ) ).toVar();
					const prevAt = ivec2( clamp( floor( uvPrev.mul( outputSize ) ), vec2( 0 ), outputSize.sub( 1 ) ) );
					const carried = ( textureLoad( movers, prevAt ) as unknown as V4 ).mul( known ).toVar();
					// velocity rejection: the surface's own motion now against the one the history was
					// written with (behind a mover the history is the mover's)
					const reject = smoothstep( REJECT_FROM, REJECT_TO, length( own.sub( carried.xy ) ) ).mul( known ).mul( mask ).mul( u.velocityReject ).toVar();
					// moving this frame (from 0.1 output pixels a frame) or last frame (FSR 2's sign)
					const moving = max( saturate( speed.mul( 10 ) ), movedLastFrame( carried.z ) ).mul( mask ).toVar();
					return { own, speed, cls, mask, carried, reject, moving };

				} )() : null;
				// Disocclusion: the previous frame's nearest surface at uvPrev, rebuilt into the world
				// with the previous camera and seen from this one, against the nearest tap's depth.
				const tolerance = abs( zNear ).mul( K.disoccludeRelative ).add( K.disoccludeAbsolute ).toVar();
				const edge = hood ? hood.clip.w.greaterThan( 0.5 ) : zNear.sub( farZ! ).greaterThan( tolerance );
				If( sky.not().and( inside ).and( u.previousDepthValid.greaterThan( 0.5 ) ).and( edge.not() ), () => {

					const was = ( textureLoad( t.depthPrev, ivec2( clamp( floor( uvPrev.mul( inputSize ) ), vec2( 0 ), last ) ) ) as unknown as V4 ).r;
					const ndcPrev = vec2( uvPrev.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uvPrev.y.mul( 2 ) ) );
					const ray = u.previousProjectionInverse.mul( vec4( ndcPrev, 1, 1 ) ).toVar();
					const dir = ray.xyz.div( ray.w ).toVar();
					const world = u.previousCameraWorld.mul( vec4( dir.mul( was.div( dir.z ) ), 1 ) );
					// (a perspective clip's w is minus the view z)
					const zNow = motionUniforms.viewProjection.mul( world ).w.negate();
					disoccluded.assign( select( abs( zNow.sub( zNear ) ).greaterThan( tolerance ), float( 1 ), float( 0 ) ) );

				} );
				// the history, FSR 3.1's Lanczos 2 over 4 x 4 texels with its deringing (TemporalHistory.ts),
				// or one bilinear tap (lite)
				const hist = vec4( 0 ).toVar();
				if ( lite ) hist.assign( texture( history, uvPrev ).level( float( 0 ) ) as unknown as V4 );
				else hist.assign( lanczosHistory( history, uvPrev, u.historySize ) );
				const histC = toYCoCg( max( hist.rgb, vec3( 0 ) ) ).toVar();
				// the clip box and the history's frames' worth (`hoodStats`), from the 3x3 here or from
				// the neighbourhood pass
				let mean: V3, extent: V3, frames: F;
				if ( hood ) {

					mean = hood.stats.xyz;
					extent = hood.clip.xyz;
					frames = hood.stats.w;

				} else {

					const veil = t.veil;
					const reactive = lite ? reactiveUniforms.flash : max( velocity.w, veil ? max( reactiveUniforms.flash,
						( textureLoad( veil, ivec2( clamp( floor( pIn ), vec2( 0 ), last ) ) ) as unknown as V4 ).r ) : reactiveUniforms.flash );
					( { mean, extent, frames } = this.hoodStats( m1!, m2!, n, velocity, zNear, reactive, outputSize ) );

				}
				if ( mv ) {

					// the box goes to its moving width where the history is rejected (FSR 2 widens it where its
					// depth clip fires): the hood's gamma by speed and class, scaled toward the moving one's
					const classGamma = byClassOf( mv.cls, u.classGamma );
					const now = max( mix( u.gammaStill, u.gammaMove, smoothstep( 0, 2, mv.speed ) ).add( classGamma ), 0.05 );
					const moved = max( u.gammaMove.add( classGamma ), 0.05 );
					extent = extent.mul( mix( float( 1 ), moved.div( now ), mv.reject ) );
					// FSR 2's motion cap and temporal reactive (TemporalLocks.ts), the class's share of its frames
					const capped = motionCap( frames, float( CAP_FRAMES ).mul( byClassOf( mv.cls, u.classFrames ) ), mv.moving, mv.speed, carriedReactive( mv.carried.z ) );
					frames = mix( frames, capped, u.motionCap.mul( mv.mask ) );

				}
				// variance clipping toward the neighbourhood's mean (Salvi), wider still than moving
				const v = histC.sub( mean ).toVar();
				const unit = abs( v ).div( extent );
				reach.assign( max( unit.x, max( unit.y, unit.z ) ) );
				clipped.assign( select( reach.greaterThan( 1 ), mean.add( v.div( reach ) ), histC ) );
				// the weight the history may keep, in frames' worth
				const valid = inside.and( u.reset.lessThan( 0.5 ) ).and( disoccluded.lessThan( 0.5 ) ).and( passThrough.not() );
				historyWeight.assign( select( valid, min( max( hist.a, 0 ).mul( u.historyGain ), frames.mul( perFrame ) ), float( 0 ) ) );
				const drop = mv ? saturate( reach.sub( 1 ) ).mul( max( mv.moving, mv.reject ) ).mul( u.clampDrop ).toVar() : null;
				if ( mv && drop ) {

					// a rejected history goes, and a moving pixel's clip drops its weight (FSR 2's clamp-drop,
					// TemporalLocks.ts), down to a frame at twice the box
					historyWeight.assign( historyWeight.mul( float( 1 ).sub( mv.reject ) ) );
					historyWeight.assign( clampDrop( historyWeight, perFrame, drop ) );

				}
				If( inside.not().or( u.reset.greaterThan( 0.5 ) ), () => {

					disoccluded.assign( select( u.reset.greaterThan( 0.5 ), float( 3 ), float( 2 ) ) );

				} );
				if ( mv && drop && stateOut ) {

					// Where the history goes (disoccluded, rejected, dropped) the frame's own estimate is the
					// wide Gaussian's, softer instead of aliased (FSR 2's kernel bias): the GT is a box filter.
					const wideNow = max( saturate( disoccluded ), max( mv.reject, drop ) ).mul( mv.mask ).mul( u.widen ).mul( select( passThrough, float( 0 ), float( 1 ) ) );
					current.assign( mix( current, sumW.div( max( wW, 1e-6 ) ), wideNow ) );
					stateOut.assign( vec4( mv.own, temporalReactive( carriedReactive( mv.carried.z ), mv.speed, select( disoccluded.equal( 1 ), float( 1 ), float( 0 ) ) ), 0 ) );
					if ( smoothed && beta && smoothTex ) {

						// The second stage (see the header): the image shown before, at this pixel, clipped to
						// a box twice as wide as the history's, takes the first stage's new image by the
						// first stage's own share of the new frame, so it drops whatever the first stage
						// drops (a cut, a disocclusion, a rejection, a clamp-drop, a reactive input); still
						// opaque and vegetation pixels only, never while a strike flashes, and less where the
						// first stage's history was clipped.
						const gate = float( 1 ).sub( smoothstep( SMOOTH_SPEED_FROM, SMOOTH_SPEED_TO, mv.speed ) ).mul( byClassOf( mv.cls, u.smoothClass ) )
							.mul( float( 1 ).sub( smoothstep( 1, 2, reach ) ) )
							.mul( u.smoothOn ).mul( u.smoothValid ).mul( select( inside.and( passThrough.not() ), float( 1 ), float( 0 ) ) )
							.mul( float( 1 ).sub( saturate( reactiveUniforms.flash ) ) );
						const prev = toYCoCg( max( ( textureLoad( smoothTex, ivec2( q ) ) as unknown as V4 ).rgb, vec3( 0 ) ) ).sub( mean ).toVar();
						const unit2 = abs( prev ).div( extent.mul( SMOOTH_BOX ) ).toVar();
						smoothed.assign( mean.add( prev.div( max( max( unit2.x, max( unit2.y, unit2.z ) ), 1 ) ) ) );
						beta.assign( mix( float( 1 ), float( 0 ), gate ) );

					}

				}

			}
			const total = historyWeight.add( currentWeight ).toVar();
			const resolved = clipped.mul( historyWeight ).add( current.mul( currentWeight ) ).div( total );
			if ( kind === 'weight' ) return vec4( vec3( historyWeight.div( total ) ), 1 );
			if ( kind === 'disocclusion' ) return vec4( select( disoccluded.equal( 1 ), float( 1 ), float( 0 ) ), select( disoccluded.equal( 2 ), float( 1 ), float( 0 ) ), select( disoccluded.equal( 3 ), float( 1 ), float( 0 ) ), 1 );
			if ( kind === 'clip' ) return vec4( vec3( clamp( reach.sub( 1 ), 0, 1 ) ), 1 );
			const color = vec4( max( fromYCoCg( resolved ), vec3( 0 ) ), total );
			// (the second stage's share of the new image: the first stage's, or all of it)
			const shown = smoothed && beta ? vec4( max( fromYCoCg( mix( smoothed, resolved, max( beta, currentWeight.div( total ) ) ) ), vec3( 0 ) ), 1 ) : null;
			// (with the state, one body for its attachments: a matrix's columns, as the neighbourhood's)
			return stateOut ? mat4( color, stateOut, shown ?? vec4( 0 ), vec4( 0 ) ) : color;

		} )();
		const stateful = movers !== null && kind === 'running';
		const color = stateful ? ( out as unknown as { element( i: number ): V4 } ).element( 0 ) : out as V4;
		const smoothing = stateful && smoothTex !== null && encoded;
		const shown = smoothing ? ( out as unknown as { element( i: number ): V4 } ).element( 2 ) : color;
		const written = stateful ? [ ( out as unknown as { element( i: number ): V4 } ).element( 1 ), ...( smoothing ? [ shown ] : [] ) ] : [];
		// (and the image encoded for the canvas pass, with the vignette at this pixel: the second
		// stage's where it runs)
		const encodedImage = encoded ? [ vec4( toDisplay( shown.rgb.mul( O.vignette( outputSize ) ) ), 1 ) ] : [];
		return encoded || stateful ? outputStruct( color, ...encodedImage, ...written ) as unknown as V4 : color;

	}

	// The 3x3's statistics the history needs, from its moments (`m1`, `m2` over `n` taps) and the
	// nearest surface's motion (`velocity`, at input pixel `nearCell`; its class in b): the clip
	// box's centre and half-widths (variance clipping toward the mean, wider while moving, by
	// class), and the frames' worth the history may keep (fewer while it moves, by class, less where
	// it is `reactive`).
	private hoodStats( m1: V3, m2: V3, n: number, velocity: V4, z: F, reactive: F, outputSize: Uniform2 ): { mean: V3; extent: V3; frames: F } {

		const u = this.u;
		// the surface's class (the velocity's b: 0, 0.25 ... 1)
		const cls = round( clamp( velocity.z, 0, 1 ).mul( 4 ) ).toVar();
		const byClass = ( values: THREE.UniformNode<'float', number>[] ) => byClassOf( cls, values );
		const speed = length( velocity.xy.mul( 0.5 ).mul( outputSize ) ).toVar();
		const mean = m1.div( n ).toVar();
		const sd = sqrt( max( m2.div( n ).sub( mean.mul( mean ) ), vec3( 0 ) ) );
		const gamma = mix( u.gammaStill, u.gammaMove, smoothstep( 0, 2, speed ) ).add( byClass( u.classGamma ) );
		const extent = sd.mul( max( gamma, 0.05 ) ).add( 1e-5 ).toVar();
		// (the sea's share grows to a whole history with distance, and its glints' reactive weight fades:
		// its normals move at the waves' phase speed, which the motion vectors leave out, by under a pixel
		// a frame far off; a strike's flash still counts)
		const share = byClass( u.classFrames ).toVar();
		const react = reactive.toVar();
		If( cls.equal( 2 ), () => {

			const far = smoothstep( u.waterFar0, u.waterFar1, abs( z ) ).toVar();
			share.assign( mix( share, 1, far ) );
			react.assign( max( react.mul( float( 1 ).sub( far ) ), reactiveUniforms.flash ) );

		} );
		const still = select( cls.equal( 2 ), u.framesStillWater, u.framesStill );
		const frames = mix( still, u.framesMoving, smoothstep( 0, u.speedFull, speed ) ).mul( share )
			.mul( clamp( float( 1 ).sub( react ), 0, 1 ) );
		return { mean, extent, frames };

	}

	// TAAU's neighbourhood pass, one input pixel each: the 3x3 around it, as the resolve read it
	// around the input pixel nearest an output pixel (`hoodStats`; its mean and frames in `stats`,
	// its half-widths and whether it straddles an edge in `clip`, the nearest surface's motion and
	// depth in `nearest`).
	private neighbourhood( t: { display: THREE.Texture; velocity: THREE.Texture; depth: THREE.Texture }, inputSize: Uniform2, outputSize: Uniform2 ): V4 {

		const K = this.params;
		// (one body for the three attachments: an output struct is the material's own node, so the
		// three are a matrix's columns, as the resolve's second attachment is its first's)
		const m = Fn( () => {

			const base = floor( screenCoordinate.xy ).toVar();
			const last = inputSize.sub( 1 ).toVar();
			const m1 = vec3( 0 ).toVar(), m2 = vec3( 0 ).toVar();
			const nearZ = float( - 1e30 ).toVar(), farZ = float( 1e30 ).toVar();
			const nearCell = vec2( 0 ).toVar();
			for ( const [ dx, dy ] of TAPS_3X3 ) {

				const cell = base.add( vec2( dx, dy ) ).toVar();
				const tap = ivec2( clamp( cell, vec2( 0 ), last ) );
				const c = toYCoCg( ( textureLoad( t.display, tap ) as unknown as V4 ).rgb ).toVar();
				const z = ( textureLoad( t.depth, tap ) as unknown as V4 ).r.toVar();
				m1.addAssign( c );
				m2.addAssign( c.mul( c ) );
				If( z.greaterThan( nearZ ), () => {

					nearZ.assign( z );
					nearCell.assign( cell );

				} );
				farZ.assign( min( farZ, z ) );

			}
			const velocity = ( textureLoad( t.velocity, ivec2( clamp( nearCell, vec2( 0 ), last ) ) ) as unknown as V4 ).toVar();
			const tolerance = abs( nearZ ).mul( K.disoccludeRelative ).add( K.disoccludeAbsolute );
			const edge = select( nearZ.sub( farZ ).greaterThan( tolerance ), float( 1 ), float( 0 ) );
			const { mean, extent, frames } = this.hoodStats( m1, m2, TAPS_3X3.length, velocity, nearZ, max( velocity.w, reactiveUniforms.flash ), outputSize );
			return mat4( vec4( mean, frames ), vec4( extent, edge ), vec4( velocity.xy, nearZ.div( NEAR_Z_SCALE ), velocity.z ), vec4( 0 ) );

		} )() as unknown as { element( i: number ): THREE.Node<'vec4'> };
		return outputStruct( m.element( 0 ), m.element( 1 ), m.element( 2 ) ) as unknown as V4;

	}

	// The running resolve: its build for the history's parity, reading the history (or, once after a
	// resize, the carried history of the size before).
	private runResolve(): void {

		const g = this.graph;
		const parity = g.parityOf( this.history );
		const valid = g.historyValid( this.history );
		const carry = ! valid && this.carryValid;
		const node = this.resolveHistory[ parity ];
		const own = node.value;
		this.u.reset.value = valid || carry ? 0 : 1;
		if ( ! valid && ! carry && this.resolved ) this.historyResets ++;
		this.resolved = true;
		if ( carry ) {

			node.value = this.carry.texture;
			this.u.historySize.value.copy( this.carrySize );

		} else this.u.historySize.value.copy( g.extent( this.historyClass ).value );
		this.u.previousDepthValid.value = this.previousDepthValid ? 1 : 0;
		this.u.stateValid.value = this.state && g.historyValid( this.state ) ? 1 : 0;
		this.u.smoothValid.value = this.smooth && g.historyValid( this.smooth ) ? 1 : 0;
		try {

			this.resolveQuads[ parity ].render( this.renderer );

		} finally {

			node.value = own;
			this.carryValid = false;

		}

	}

	private previousDepthValid = false;

	/** Before a resize (Pipeline.setSize, between frames): a history whose size changes is kept in
	 * the carry, which the next resolve reads once instead of starting over. */
	beforeResize( extents: Extents ): void {

		const g = this.graph;
		const now = g.extent( this.historyClass ).value;
		const next = resolveSize( this.historyClass, {
			render: { width: Math.max( extents.render.width, 1 ), height: Math.max( extents.render.height, 1 ) },
			canvas: { width: Math.max( extents.canvas.width, extents.render.width, 1 ), height: Math.max( extents.canvas.height, extents.render.height, 1 ) },
		} );
		if ( next.width === now.x && next.height === now.y ) return;
		this.carryValid = false;
		if ( ! g.historyValid( this.history ) || now.x <= 1 || now.y <= 1 ) return;
		// (outside a frame, the previous version is the newest history)
		this.carry.setSize( now.x, now.y );
		this.renderer.initRenderTarget( this.carry );
		// eslint-disable-next-line ponlai/history-on-graph -- a resize, between frames: the history of the old size goes into the carry, which the graph cannot declare (its targets follow their size class), and the next resolve reads it once
		this.renderer.copyTextureToTexture( g.texture( this.historyPrev ), this.carry.texture );
		this.carrySize.copy( now );
		this.carryValid = true;

	}

	// ---- the frame

	/** Before the frame's jitter is chosen (the camera unjittered): the live settings, the
	 * sharpening, the rest test (which sets the jitter's rest index) and the camera's matrices. */
	beginFrame( camera: THREE.Camera, renderWidth: number, renderHeight: number, outputWidth: number, outputHeight: number, time: number, frame: TemporalFrame = {} ): void {

		this.applySettings();
		// the movers' flags (`taaMovers` sets the four's default)
		const all = debugFlags.taaMovers;
		const [ reject, drop, cap, widen ] = MOVER_FLAGS.map( ( name ) => ( debugFlags[ name ] ?? all ?? MOVERS ) !== false ? 1 : 0 );
		this.u.velocityReject.value = reject; this.u.clampDrop.value = drop; this.u.motionCap.value = cap; this.u.widen.value = widen;
		this.u.smoothOn.value = debugFlags.taaSmooth === false ? 0 : 1;
		// the rest: what the frame draws from, unjittered
		const key = restKey( this.key, camera, renderWidth, renderHeight, outputWidth, outputHeight, time, [ this.settingsVersion, ...( frame.signature ?? [] ) ] );
		this.rest.settle = Math.max( 0, Math.round( this.settings.restSettle ) );
		this.rest.sequence = Math.max( 0, Math.round( this.mode === 'gt' ? this.settings.gtJitters : this.settings.restSequence ) );
		this.rest.observe( key, frame.stable !== false );
		// (while converged the jitter stays at the sequence's last index, as under FSR 3: what the display
		// chain reads of the live frame beside the held image is then bit-identical too)
		jitter.restIndex = this.rest.index !== null ? this.rest.index + 1 : this.rest.converged ? this.rest.sequence : null;
		jitter.restLength = this.rest.sequence;
		// the sharpening, eased to its new value over SHARPNESS_EASE seconds after a scale change; at
		// rest it is the new value at once, so the rest's image does not depend on the frames' deltas
		const target = temporalSharpness( this.settings, this.mode, outputWidth / Math.max( renderWidth, 1 ) );
		if ( target !== this.sharpnessTarget ) {

			this.sharpnessFrom = this.sharpness.value;
			this.sharpnessTarget = target;
			this.sharpnessTime = 0;

		}
		const resting = this.rest.index !== null || this.rest.converged;
		// after a step up of the render scale, the history keeps its frames' worth (see the header)
		const density = renderWidth * renderHeight / Math.max( outputWidth * outputHeight, 1 );
		this.u.historyGain.value = this.mode === 'taau' && density > this.lastDensity ? density / this.lastDensity : 1;
		this.lastDensity = density;
		this.sharpnessTime = resting ? SHARPNESS_EASE : Math.min( this.sharpnessTime + Math.max( frame.dt ?? 0, 0 ), SHARPNESS_EASE );
		const k = this.sharpnessTime / SHARPNESS_EASE;
		this.sharpness.value = k >= 1 ? this.sharpnessTarget : this.sharpnessFrom + ( this.sharpnessTarget - this.sharpnessFrom ) * k;
		// the camera this frame and in the previous rendered frame (a cut makes them equal: onTaaCut)
		const u = this.u;
		if ( this.hasLast ) {

			u.previousCameraWorld.value.copy( this.lastCameraWorld );
			u.previousProjectionInverse.value.copy( this.lastProjectionInverse );

		} else {

			u.previousCameraWorld.value.copy( camera.matrixWorld );
			u.previousProjectionInverse.value.copy( camera.projectionMatrixInverse );

		}
		u.cameraWorld.value.copy( camera.matrixWorld );
		this.lastCameraWorld.copy( camera.matrixWorld );
		this.lastProjectionInverse.copy( camera.projectionMatrixInverse );
		this.hasLast = true;

	}

	/** After the motion's frame began (cuts applied): whether the previous frame's nearest surface may
	 * be compared (it is invalid after a cut or a resize). */
	setPreviousDepthValid( valid: boolean ): void {

		this.previousDepthValid = valid && ! motionState.cut;

	}

	// the live settings into the uniforms; a change restarts the rest (its key holds the version)
	private applySettings(): void {

		const S = this.settings, u = this.u;
		const k = JSON.stringify( S ) + JSON.stringify( [ this.params.disoccludeRelative.value, this.params.disoccludeAbsolute.value ] );
		if ( k === this.settingsKey ) return;
		this.settingsKey = k;
		this.settingsVersion ++;
		u.framesStill.value = S.framesStill; u.framesStillWater.value = S.framesStillWater; u.framesMoving.value = S.framesMoving; u.speedFull.value = S.speedFull;
		u.gammaStill.value = S.gammaStill; u.gammaMove.value = S.gammaMove; u.sigma.value = S.sigma; u.sigmaRest.value = S.sigmaRest;
		u.skyPassThrough.value = S.skyPassThrough ? 1 : 0;
		for ( let i = 0; i < CLASSES; i ++ ) {

			u.classFrames[ i ].value = S.classFrames[ i ] ?? 1;
			u.classGamma[ i ].value = S.classGamma[ i ] ?? 0;

		}
		for ( let i = 0; i < CLASSES; i ++ ) u.sharpenMoving[ i ].value = S.sharpenMoving.stops[ i ] ?? 0;
		u.sharpenSpeed.value = S.sharpenMoving.speed;
		u.waterFar0.value = S.waterFar[ 0 ]; u.waterFar1.value = Math.max( S.waterFar[ 1 ], S.waterFar[ 0 ] + 1e-3 );
		const glint = S.glintReactive;
		reactiveUniforms.glintFrom.value = glint.from; reactiveUniforms.glintTo.value = glint.to; reactiveUniforms.glintMax.value = glint.max;

	}

	/** Restarts the rest (the next frame is the first at rest). */
	restartRest(): void {

		this.rest.restart();

	}

	// ---- warm-up

	/** Compiles the builds the graph's warm-up does not see (each pass's other variants), each under
	 * its pass's binding. */
	warm(): Promise<void> {

		const g = this.graph, r = this.renderer;
		const pending: Promise<unknown>[] = [];
		for ( const v of this.variants ) {

			const index = g.decl.passes.findIndex( ( p ) => p.name === v.pass );
			if ( index < 0 ) continue;
			g.bind( index );
			try {

				for ( const quad of v.quads.slice( 1 ) ) pending.push( compileQuad( r, quad ).done );
				for ( const pipeline of v.pipelines.slice( 1 ) ) pending.push( compileChain( r, pipeline ).done );

			} finally {

				g.unbind();

			}

		}
		return Promise.all( pending ).then( () => undefined );

	}

	/** The diagnostics' `taa` block. */
	diagnostics(): Record<string, unknown> {

		const g = this.graph;
		const input = g.extent( RENDER ).value, output = g.extent( this.historyClass ).value;
		return {
			mode: this.mode,
			converged: this.rest.converged,
			restFrames: this.rest.frames,
			restTarget: this.rest.target,
			restIndex: this.rest.index,
			restEpoch: restEpoch.value,
			restReason: restEpoch.reason,
			// the last scene event (SceneEvents.ts): its reason, whether it cut, the frame it followed
			lastEvent: sceneEvents.last,
			jitterIndex: jitter.k + 1,
			jitterLength: jitter.restIndex !== null ? this.rest.sequence : jitter.length,
			// the running history started over (cuts, resizes it could not be carried across), and the
			// cuts alone (Motion.ts taaCut)
			historyResets: this.historyResets,
			cuts: motionState.historyResets,
			inputSize: [ input.x, input.y ],
			outputSize: [ output.x, output.y ],
			sharpness: + this.sharpness.value.toFixed( 4 ),
			// the per-pixel reactive inputs are read (the velocity's alpha and the veil; not by lite)
			reactiveOn: ! this.settings.lite,
			flash: reactiveUniforms.flash.value,
		};

	}

	dispose(): void {

		this.carry.dispose();
		for ( const quad of [ ...this.resolveQuads, ...this.restQuads, ...this.encodeQuads, ...this.debugQuads, ...( this.neighbourhoodQuad ? [ this.neighbourhoodQuad ] : [] ) ] ) ( quad.material as THREE.Material ).dispose();
		for ( const pipeline of [ ...this.canvasPipelines, ...this.inlinePipelines, ...( this.upscaledPipeline ? [ this.upscaledPipeline ] : [] ) ] ) pipeline.dispose();
		this.fsr?.dispose();

	}

}
