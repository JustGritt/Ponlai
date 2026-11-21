/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, texture, textureLoad, uniform, screenCoordinate, ivec2, int, vec2, vec3, vec4, float,
	clamp, max, min, exp2, floor, select, step, Discard, Loop, normalize, refract,
} from 'three/tsl';
import { VolumetricHaze, type HazeFrame } from '../weather/VolumetricHaze.ts';
import { LightShafts } from './LightShafts.ts';
import { FSR } from './FSR.ts';
import type { PhysicalBloom } from './PhysicalBloom.ts';
import type { Metering } from './Metering.ts';
import { cloudedShare } from '../sky/Weather.ts';
import { AA_MODE, AFTER_RESOLVE, FSR3, RESOLVE, SCENE_SAMPLES, TEMPORAL } from './Antialiasing.ts';
import { beginMotionFrame, METER_LIGHT, onTaaCut } from './Motion.ts';
import { preExposure } from './PreExposure.ts';
import { RAIN_OVERLAY } from './AfterResolve.ts';
import { jitter } from './Jitter.ts';
import { MEADOW_SHADOWS, SS_SHADOW, meadowDepthMap, meadowDepthMaterial, meadowGroundView, meadowShadows, meadowShadowMaterial, screenShadowView } from './MeadowShadows.ts';
import { GTAO, gtaoBlurMaterial, gtaoDepthMaterial, gtaoMap, gtaoMaterial, gtaoProjection, gtaoView } from './GTAO.ts';
import { TemporalAA, type TemporalFrame } from './TemporalAA.ts';
import { Settled, type SettleFrame } from './Settled.ts';
import type { Fsr3Upscaler } from './fsr3/Fsr3Upscaler.ts';
import { QUALITY, type TemporalSettings } from '../Quality.ts';
import { FrameGraph, type ResourceHandle, type SizeClass } from './graph/FrameGraph.ts';
import type { Sky } from '../sky/Sky.ts';
import type { SunShadow } from '../lighting/SunShadow.ts';
import { ambientSkyVisibility } from '../lighting/AmbientSample.ts';
import { liftedKeyHeight } from '../lighting/KeyLift.ts';
import { duskForce } from '../sky/DayCycle.ts';
import type { Rain } from '../weather/Rain.ts';
import { CLIP_MARGIN, RefractionView, REFRACTION_SIZE } from '../ocean/Refraction.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { whiteBalance } from './ToneSpace.ts';
import { checkToneCurve, solveToneCurve, type ToneCurve } from './ToneCurve.ts';
import { fillGradeLut, gradeOf, lutNodes, mergeGrade, type GradeChanges, type GradeParams } from './GradeLut.ts';
import { LocalTone } from './LocalTone.ts';
import { keyed } from './DayKeys.ts';
import type { ExposureKey } from '../look/ExposureLook.ts';
import { restEpoch, restKey } from './TemporalRest.ts';
import { tuneCheck, tuneObject } from '../debug/Tune.ts';
import type { ExtraPass } from './ExtraPass.ts';
import type { WaveMaps } from '../ocean/WaveMaps.ts';
import { SeaReflection } from '../ocean/SeaReflection.ts';
import { DEFAULT_WATER_QUALITY } from '../ocean/Water.ts';
import { settings } from '../Settings.ts';
import { DEFAULT_POST_OPTIONS, postOptionsFromURL, type PostOptions } from './PostOptions.ts';
import type { DisplayTables } from '../look/DisplayLook.ts';
import { declarePasses } from './PipelinePasses.ts';
import { DepthPyramid } from './DepthPyramid.ts';
import { buildDisplay, GRADE_TABLE, NIGHT, TONE_SPACE } from './PipelineDisplay.ts';

// The frame's pipeline: the frame graph of its targets and passes, from scene to canvas.
//
// Frame graph:
//   opaque scene ── sceneRT (MSAA, SCENE_SAMPLES, half float, multisampled depth): the colour and
//     the metering illuminance (meterLight, an MRT: Motion.ts's sceneMRT)
//   underwater pyramid ── underwater (the opaque colour of what lies under the sea, premultiplied by
//     that weight, and its mip chain: what the water refracts at each pixel's footprint)
//   refraction reach ── refractionReach (64 texels: how far past each edge of the screen the
//     water's refracted lookups reach in this view)
//   refraction mask ── refractionDepth at the nearest depth wherever the opaque image already holds
//     what lies under the sea, and in the guard band beyond that reach, so the refraction pass
//     shades only what it alone can show
//   depth resolve ── depthRT (view-space z of the opaque pass: sample 0, and the farthest sample)
//   depth pyramid ── the nearest and farthest depth per cell over ten levels, in a storage buffer
//     (DepthPyramid.ts; in a frame with water): what the sea's reflection is traced over
//   opaque copy ── opaqueRT (the resolved opaque colour, with mips: what the water refracts where
//     the refraction image holds nothing, and what the sea's reflection meets)
//   sea reflection ── the water's surface drawn alone (its normal, roughness and view depth), one
//     stochastic reflected ray per pixel traced over the depth pyramid, and its denoiser
//     (reprojection, prefilter, temporal resolve) into seaResolved, what the water reads at its pixel
//     (ocean/SeaReflection.ts; in a frame with water, held once a still view has converged)
//   refraction ── refraction and refractionDepth: the world under the sea through a wider view, at
//     half the render's pixel density (ocean/Refraction.ts; the water's refraction source where the
//     opaque image does not hold the floor: behind what stands above the water, and off the screen)
//   water ── back into sceneRT, depth-tested per sample against the opaque samples
//   surface resolve ── surfaceRT (view-space z of the nearest surface, the sea's included: sample
//     0's, or the nearest sample's where sample 0 sees the sky, so a pixel is sky only where all its
//     samples are)
//   rain ── into sceneRT (its multisampled colour and depth are then discarded: nothing reads them;
//     without rain the water discards the colour and the surface resolve's read keeps the depth)
//   haze (weather/VolumetricHaze.ts): its froxel grid's shadow kernel (skipped while a rest holds it)
//   and its light kernel, which integrates each column into the light volume
//   sky constants: the clouds' mean transmittance over the sun's and the moon's discs, which the
//   flare's visibility reads (Sky.ts discCloudTransmittance, LightShafts.ts `occlusion`), evaluated
//   once into a texel ahead of the scene
//   the scene's extra passes (`ExtraPass`, the constructor's list): passes the pipeline does not
//   know, declared here in the list's order, ahead of the scene passes that draw what they
//   prepare. The island gives the falling leaves' births and fixed steps (world/FallingLeaves.ts:
//   one compute pass over its fixed ring, its own buffers only; a reset's frame prewarms it
//   instead; idle in the test mode)
//   wave maps (ocean/WaveMaps.ts): the long waves' maps round the camera and the caustics (while the
//   sun lights them) into one array target outside the graph's resources, which the materials lit
//   under the water read, its mip chain after the last layer drawn
//   light shafts: the open sky near the sun, the rays marched toward it (half resolution, while the
//   sun faces the view): the crepuscular rays through the clouds' gaps until the haze reaches them
//   constants: what the post passes take that is the same for every pixel (the flare's visibility,
//   the sun's or the moon's while it is the key: depth taps over its disk times the sky constants'
//   cloud transmittance over it; the rods' colour's retinal scale, Purkinje.ts `purkinjeScale`),
//   evaluated once into a texel
//   composite (haze, rain veil, shafts, the rods' colour by night: Purkinje.ts)
//   ── hdr, once (the veil after the resolve under taa, taau)
//   metering: the histograms of hdr's luminance and of meterLight (Metering.ts: one compute pass at
//   a fixed 32 x 16 tiles, its buffer cleared by a small pass before the composite), before the
//   bloom, which reads this frame's exposure
//   meter: the exposure state from the histograms and the sky (Metering.ts, Exposure.ts), one texel,
//   adapted from the previous frame's over the scene clock's step (reset by every cut, in every
//   mode: SceneEvents.ts `onSceneCut`; taken at once on a paused clock), copied at the frame's end
//   for the next frame's meter, and read back (one reading in flight) for the next
//   pre-exposure (PreExposure.ts); while it has not reached its target (debug builds) the page is
//   not settled (`exposurePending`)
//   bloom: the bloom from hdr over its threshold (PhysicalBloom.ts: eight 13-tap downsamples from
//   half the render to 1/256 of it, the first through the threshold, then seven tent upsamples
//   adding each level back at its weight), the flare (the sun's, or the moon's while it is the
//   key) joining its first level (LightShafts.ts flare)
//   sharp (hdr + the bloom at its strength)
//   local tone: the sharp image's log luminance (localTone.luma), its bilateral grid and the grid
//   blurred across (localTone.grid, gridX, gridXY: LocalTone.ts), at fixed sizes
//   display: the exposure, the local tone mapping, tidewater's grade (the night's grade, the
//   tone, the vignette), the white balance and the tone curve in their
//   curve space (ToneSpace.ts), the grade's two tables (GradeLut.ts, indexed by the square root of linear light),
//   overcast's highlights toward grey (`highlightGrey`), the BT.709 OETF, dither ── displayRT
//   ── copied to the screen
// Under the temporal modes (Antialiasing.ts: aa=taa|taau) the scene half is single-sampled and
// jittered, and writes the motion vectors beside the colour:
//   opaque scene ── sceneColor, velocity and meterLight (an MRT: Motion.ts's sceneMRT), sceneDepth
//   depth resolve ── depthRT (sample 0 twice: one sample)
//   depth pyramid ── as above, jittered with the scene
//   underwater pyramid, refraction reach and mask ── as above, jittered with the scene
//   refraction ── as above, jittered with the scene
//   water ── sceneColor and velocity (sceneMRT), depth-tested against the opaque depth
//   surface resolve ── surfaceRT, copied at the frame's end into surfaceDepthPrev (the next
//     frame's history of the nearest surface)
//   overlay (the ground truth: taa and taau draw these after the resolve) ── the rain and the
//     storm's spray (moved from the water's scene) over the scene, with Motion.ts's overlayMRT: the
//     velocity under them kept
// and the post passes follow as above, but for the vignette and the dither, which the display pass
// leaves to the canvas pass, and (accumulating in linear light, Quality.ts `taa.linear`) for the
// output transfer: displayRT holds the BT.709 code's sRGB decode (half float),
// which the resolve averages as it is and encodes once for its canvas pass; then (TemporalAA.ts):
//   TAA neighbourhood (TAAU) ── taaStats, taaClip, taaNearest: each input pixel's 3x3 clip box,
//     history weight and nearest motion and depth, at the render's size
//   TAA resolve ── taaHistory (ping-pong: the frames accumulated, at the canvas's size for TAAU)
//     and taaEncoded (the same image encoded with the vignette, under `taa.linear`; not for aa=gt)
//   TAA rest ── taaRest (ping-pong), while a resting scene runs its jitter sequence, then copied
//     into taaRestShown once complete (TemporalRest.ts)
//   rain after resolve (taa, taau: AfterResolve.ts) ── rainOverlay: the rain's streaks and splashes
//     and the storm's spray, unjittered, at the canvas's size (premultiplied radiance, coverage), while
//     either shows
//   rain input ── rainInput, rainBehind: the grade's input at each render texel and the air there
//     (its transmittance, and the light it adds), at the render's size, while it rains
//   rain grade ── rainGrade: the grade as a table, for the canvas pass of a running frame, while it
//     rains
//   rain layer ── rainLayer: the change the rain veil and the overlay make to the graded image, at
//     the canvas's size (the overlay through the composite at the pixel), drawn here only by the
//     frame that completes the rest and for TAA's upscaled canvas pass; a running frame's canvas
//     pass draws it itself (neither rain pass runs while the rest holds its image: the frame that
//     completed it drew them, `afterHeld`)
//   canvas pass: taaEncoded (or the history or the rest's image, encoded there with the vignette),
//     RCAS, the rain layer added, dither; TAA below the canvas encodes it at the render's
//     size (taaDisplay), EASU, then RCAS, the rain layer and the dither
// (`taaResolve=off`, debug builds: the jittered frame shown as it is, through the output passes
// above, with the grade's vignette and dither, as before the resolve landed).
// Under fsr3 (Antialiasing.ts) the scene half is the temporal modes' (the sky dome jittered too,
// with its motion: Sky.ts), the composite writes veilReactive beside hdr, and FSR 3 resolves the
// frame where AMD places it (fsr3/Fsr3Upscaler.ts), after the composite and before the rest of the
// post chain:
//   FSR 3 ── its compute passes (the clears, prepare inputs, the luma and shading-change pyramids,
//     shading change, prepare reactivity, luma instability, accumulate, RCAS) on hdr, the nearest
//     surface's depth, the velocity and veilReactive, into its output: scene radiance at the
//     canvas's size (an import written through storage); the rest's held copy of the last running
//     output (fsr3Held) is read instead while the rest's sequence runs
//   bloom ── from FSR's output, its levels at their render-relative sizes
//   sharp (FSR's output + the bloom) ── at the canvas's size
//   display ── at the canvas's size, the full grade with vignette, tone curve, BT.709 OETF and dither
//     (half float), copied to the canvas by the output pass
// The rain veil's per-pixel reactive weight (veilReactive, R16F at the render's size) is a resource
// of its own the resolve reads: under the resolve the composite pass writes it as a second target
// beside hdr, the veil's opacity at the pixel (Rain.veil) x 1.5, at most 0.6, so the resolve trusts
// its history less only where the veil is dense, and the meadow beside it keeps its history. (The
// legacy inline composite, `post=` without compositeOnce, leaves it at zero.) Under taa and taau the
// veil is drawn after the resolve instead (AfterResolve.ts) and the composite writes, in place of
// veilReactive, the air's transmittance (rainAir, R16F at the render's size), which it hands over:
// the composite is affine in the scene colour at a pixel, hdr = A scene + B, so the streaks drawn
// after the resolve take the haze and the shafts of what lies behind them as they did drawn before
// the composite.
// Every pass runs at the scene's render size or a share of it. When the canvas is larger (a quality
// preset below the display's resolution), displayRT goes through FSR 1 (edge-adaptive upsampling,
// then contrast-adaptive sharpening) to the screen instead, and the dither is added there, at
// display pixels.
// The targets are the frame graph's (graph/FrameGraph.ts): each is declared with its format,
// sample count and size class (the render, a share of it, the canvas or fixed), and the graph sizes
// them all from `setSize`'s render and canvas sizes. The passes read their sizes from the graph's
// extent uniforms.
// The graph runs the whole frame as passes that declare what they read and write: the scene half
// (at 4x outside the temporal modes first the vegetation's depth, its resolve and the meadow's
// screen-space shadows, MeadowShadows.ts; then opaque over that depth, the depth and copy resolves, water, the
// surface resolve, rain), the haze's two kernels, the light shafts' two passes, the composite's
// constants, the
// composite, the metering histograms (and their clear before the composite), the meter, the bloom (PhysicalBloom.ts: a pass per level), the sharp image, the display pass,
// and the canvas pass, or EASU and the sharpening canvas pass while upscaling. Every pass but the
// scene passes and the canvas passes is a full-screen quad of its own, and no pass renders another
// inside it (but three's FSR1Node, `post=three-fsr`, in its canvas pass). Each pass that runs only
// now and then (rain, the haze's shadow, the light shafts, either output) says when by a predicate
// the graph evaluates at the frame's start. A pass whose output weighs nothing in what reads it (the
// shafts without the sun toward the view) is skipped rather than drawn and multiplied by 0; its
// reader reads it optionally and gets what it last held, finite, times 0. The
// `post=` legacy switches are variants of the graph, chosen as it is built.
// At startup the graph's warm-up (`graph.warm`, from main.ts) compiles every pass, so one that
// first runs later (the rain, the upscale) builds nothing then.
// Where the parts are: the graph's structural choices (`PostOptions`, `?post=`) in PostOptions.ts;
// the display chain's materials, built in the constructor's place for them, in PipelineDisplay.ts
// (`buildDisplay`), and their numbers chosen for the look in the display's tables the pipeline is
// made with (`look`: look/DisplayLook.ts `createDisplayTables`); the passes above,
// declared in order once the materials are built, in PipelinePasses.ts (`declarePasses`). This file
// keeps the targets, the constructor's steps, the sizes and the per-frame updates.

//
// Rules (the look itself is in docs/design/art-direction.md):
// - Sizes are owned by the graph (graph/FrameGraph.ts): it sizes every target by its size class
//   from `setSize` (the render; a share of it: the shafts' halves, the bloom's levels; the canvas: FSR's EASU target; fixed: the haze's volumes), and every pass reads
//   its sizes from the graph's shared extent uniforms (`graph.extent( cls )`, `graph.texel( cls )`
//   for one over it), never from a copy of its own. Declare a new target on the graph with its
//   class, and new post work as a pass of its own drawing into it, not as an rtt() inside another
//   pass's material (three sizes an rtt() to the canvas, not the render, and runs it out of the
//   graph's sight). A target three makes itself (a bloom chain's) is imported into the graph and
//   follows the render through `graph.onResize`.
// - The graph's attachment ops are derived each frame from the passes that run and the reads they
//   declare (graph/AttachmentOps.ts): with rain the water stores `sceneRT`'s samples and leaves the
//   resolve to the rain, which then discards colour and depth; without it the water discards the
//   multisampled colour and keeps the depth the surface resolve reads. So every texture a pass
//   binds from the scene target (colour, multisampled depth) must be declared (`read`,
//   `readOptional`): an undeclared read of a discarded attachment shows garbage, and one between
//   the water and the rain sees the scene without the sea.
// - `render` is `graph.execute`: the haze's kernels and the
//   output and FSR passes are graph passes too (the `post=` switches choose which are declared at
//   load), so new frame work goes in as a pass, not as a call around the graph. A pass that
//   declares no depth clears whatever depth its target holds outside the graph (the canvas's, and
//   the targets three's nodes draw into inside a pass), as three's defaults do; declare the depth
//   to keep it.
// - The water cannot sample the target it draws into: it reads `opaqueRT` and `depthRT` (sample 0,
//   and the farthest sample for what lies behind the sea at an edge pixel), and the underwater
//   pyramid (`textures.underwater`). Post passes read
//   `surfaceRT`, the nearest surface's depth with the sea's included (sky only where every sample
//   of the pixel is: the surface resolve). Multisampled depth is read
//   with `textureLoad`, one sample at a time (the sample index in its level argument);
//   single-sampled, that argument is a mip level, so load sample 0 only. View z below -9e4 is sky.
// - In a full-screen pass the camera nodes are the pass's own camera: the scene camera's matrices
//   come in as uniforms.
// - The metering illuminance (`meterLight`, METER_LIGHT: R16F at the render's size, the scene
//   target's attachment after the colour and, under the temporal modes, the velocity) holds per
//   pixel M = Y( E ) / pi in scene units of luminance, unexposed: the luminance a white Lambertian
//   surface would show there under the pixel's incident light E (the key with its shadow and
//   cosine, the sky's and the ambient volume's diffuse light with its occlusion, the ground bounce;
//   no albedo, specular, emission or air in front). Who writes it: every lit material its own M,
//   gathered from its lighting model's terms (lighting/MeterLight.ts, marked by `writeMeter`, its
//   MRT node merged through Motion.ts `meterOutput`), and the water the M of its surface (its
//   `outputStruct`'s meter: the sky's upper hemisphere and the sun on it), and the sky dome one
//   value on all its pixels, the sky's geometric-mean luminance times METER_SKY (Sky.ts, the same
//   log the exposure's sky term reads: shared.ts `skyMeterLog2`); every pixel nothing draws keeps
//   the clear value 0. Only an exact 0 is no reading, which the readers skip. The rain and the overlay pass leave it as it
//   is; a transparent draw in the opaque or the water pass (the lightning's bolt, the spray in
//   msaa's water pass) writes 0 there, accepted: a one-channel attachment cannot blend (Motion.ts),
//   and those few pixels drop out of the meter. Its reader is the metering histograms
//   (Metering.ts), whose read keeps the resolve. At 4x the target's multisample resolve averages
//   the samples into the texture the readers sample, with the coverage alpha to coverage gave the
//   colour. Like the colour it is stored pre-exposed (PreExposure.ts: every writer multiplies by
//   2^P, one uniform per frame, P a whole number of stops from the last exposure read back), and
//   the meter divides 2^P back out, so it works in absolute log2 of these units. Half float is
//   normal from 2^-14 and at most 65504 (`__PONLAI__.radiance` with `range`, debug/Radiance.ts,
//   reads the share outside over the non-sky pixels). Unscaled, on a clear night 62 % of `hdr`'s
//   non-sky pixels fell below 2^-14 in the island view and 45 % in the horizon view, and 2.3 % and
//   0.6 % of meterLight's (the night's M runs about 2^-16 to 2^-7), and they were stored a little
//   dark: the night's and the dusk's images came out up to about 0.8 of an 8-bit level brighter
//   once pre-exposed. With pre-exposure none of either falls below 2^-14 there (the night's
//   smallest stored values sit about 3.5 stops above it in hdr, 7 in meterLight), and at noon the
//   largest non-sky hdr value is stored about 2.5 stops below the top; the sun's disc is held at
//   65504.
// - The display chain is tidewater's with our tone curve (ToneCurve.ts) and, over it, the local
//   tone mapping, curve space and white balance after Patry, "Real-Time Samurai Cinema" (SIGGRAPH
//   2021), in this order: the desaturation (`desaturation`) and the local tone mapping first
//   (LocalTone.ts: a function of the frame, nothing kept between frames), one
//   curve and grade (the shadows' contrast, `shadowContrast`; the tone, TONE, and the night's grade, NIGHT, over it are off by default), then
//   the grade's tables (GradeLut.ts, indexed by the square root of the curve's linear output: the first generated from the grade's parameters, `grade` and
//   `setGrade`, the second the identity), then overcast's highlights toward grey (`highlightGrey`).
//   The curve maps each channel in the
//   space of ToneSpace.ts, AP1 with its red primary moved out, and the white balance is folded
//   into its input matrix (`whiteBalance`, keyed by the sun and the weather; rebuilt only when it changes). The exposure is metered
//   from the frame by the meter pass (Metering.ts, Exposure.ts: the metering illuminance's and the
//   luminance's histograms and the sky's mean luminance, against keyed targets) into the exposure
//   state's texel, which every pass after it loads (`exposure`, 2^( EV - P ) for the pre-exposed
//   hdr; each declares the state as a read), so they all read the same value in the same frame (the composite, before the meter, reads none:
//   the rods' colour takes its retinal scale from the night's light, Purkinje.ts); the clock's and the weather's
//   exposure (`settings.exposure`, DayCycle and Sky) no longer sets the frame's: it stays the
//   level the night sky's own light is tuned against.

// The graph's dev checks (every target at its class's size, the attachment ops that reached
// WebGPU): on the dev server and with `?debug`.
const DEV = import.meta.env.DEV || settings.has( 'debug' );
// The graph's warm-up renders the passes it cannot compile once at startup (`fgPrime=0`: not, for
// the check that priming changes no frame).
const PRIME = settings.get( 'fgPrime' ) !== '0';
// The refraction's reach past the screen's edges (refractionReach): texels per edge, the pixels each
// follows toward it, and the margin the mask grows it by (screen uv: the reach times scale, plus add)
// for the floor's slopes, which the floor seen straight behind leaves out, and the waves' beyond.
const REACH = 64, REACH_SAMPLES = 16;
// the surface's tilt the reach follows besides the flat sea (a slope), about twice the slopes'
// spread across one axis in a strong wind
const REACH_SLOPE = 0.3;
const REACH_MARGIN = { scale: 1.3, add: 0.02 };
const RENDER: SizeClass = { of: 'render' };
// the exposure state's one texel (Metering.ts)
const EXPOSURE_TEXEL: SizeClass = { of: 'fixed', width: 1, height: 1 };
const CANVAS: SizeClass = { of: 'canvas' };

/** What a frame draws (`Pipeline.render`'s argument). `rainScene`: the rain, while it shows (under the
 * temporal modes with the storm's spray in it, but for what draws after the resolve: see `overlay`).
 * `overlay`: whether the temporal modes' overlay pass draws `rainScene` this frame (rain or spray
 * showing; not under taa and taau). `afterScene`: under taa and taau, the rain (streaks, splashes,
 * rings) and the storm's spray, drawn after the resolve (AfterResolve.ts), while either shows. */
export interface PipelineFrame {
	scene: THREE.Scene; waterScene: THREE.Scene; rainScene?: THREE.Scene; overlay?: boolean; afterScene?: THREE.Scene; camera: THREE.Camera;
	// the vegetation's depth copies with the meadow's screen-space shadows (VegetationPrepass.ts
	// `prepassScene`)
	prepassScene?: THREE.Scene;
	// the opaque scene's depth copies with the ambient occlusion or the screen-space shadows over the
	// whole scene (post/OpaquePrepass.ts)
	opaquePrepassScene?: THREE.Scene;
	water: { prepare( sun: THREE.DirectionalLight ): void } | null; sun: THREE.DirectionalLight; time: number;
	// the key's shading direction (world, toward the light: lighting/KeyLift.ts), which the
	// meadow's screen-space shadows march along; left out, the light's own
	keyDirection?: THREE.Vector3;
	// the temporal resolve's per-frame inputs from the app (the rest's signature and stability, the
	// frame's delta)
	temporal?: TemporalFrame;
	// the settled signal's per-frame inputs (Settled.ts; debug builds): left out, it is not observed
	settle?: SettleFrame;
	// the haze's rest inputs (VolumetricHaze.ts: the app's signature and stability); left out, the
	// haze never rests
	haze?: HazeFrame;
}

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V4 = THREE.Node<'vec4'>;
// what a pass's setup declares its reads and writes on
export type PassBuilder = Parameters<Parameters<FrameGraph<PipelineFrame>[ 'addPass' ]>[ 2 ]>[ 0 ];

export interface PipelineSettings {
	// the clock's and the weather's exposure (DayCycle, Sky): the level the night sky's own light
	// is tuned against, no longer the frame's (Exposure.ts meters that)
	exposure: number;
	// stops over the exposure's target (the GUI's)
	exposureOffset: number;
	// the tone curve's slope at black, the mid grey it shows and the input that shows white
	// (look/DisplayLook.ts `toneCurve`)
	curve: ToneCurve;
	// the grade's two tables (GradeLut.ts, `Pipeline.luts`): 0 the first, 1 the second
	lut: number;
	// tidewater's grade: the bloom's strength and the vignette's
	bloom: number; vignette: number;
	// the lighting's tone over tidewater's grade (0 none, 1 full): highlights a touch golden,
	// shadows a touch teal, the contrast round mid grey softer and the bloom a little stronger
	tone: number;
	// 0 by day, 1 at night (from the clock, main.ts): how far the night's grade has come in (NIGHT)
	night: number;
	// 0 by day, 1 at night: how far the exposure's night key has come in (sky/DayCycle.ts
	// `exposureNight`, apart from the display's)
	exposureNight: number;
	// the four skies' shares in the sky as it stands (Weather.ts `skyMix`, main.ts): the clouded
	// day's keys (DayKeys.ts `cloudy`, the cloudy and overcast skies' share)
	skyMix: number[];
}

// The lighting's tone over tidewater's grade (docs/design/art-direction.md, Lighting): a
// photographic image pushed a little toward a painted one. Highlights turn a touch golden and
// shadows a touch teal (by the exposed luminance, so the night takes it as the day), the contrast
// round mid grey softens by a tenth and the bloom strengthens by a quarter. It lifts no black: a
// constant fill in the grade greyed everything below mid grey alike and veiled the frame; the
// shadows' openness comes from the sky light the island takes instead (look/SkyLook.ts
// `clearSkyFill`). 0.5 of the full step was chosen over 0 and 1 by eye. It is off by default
// (docs/design/art-direction.md, Lighting: the lighting keeps the sun's and the sky's own
// balance): with the clear-sky fill light (`clearSkyFill`) it took the light's direction out of the
// meadow. `flags=tone:<t>` sets it at load for A/Bs.
const TONE = Number( debugFlags.tone ?? 0 );


export class Pipeline {

	renderer: THREE.WebGPURenderer;
	width = 1;
	height = 1;
	// (from the display's tables: the constructor makes them)
	settings: PipelineSettings;

	// The canvas's size; larger than width x height while the frame is upscaled.
	outputWidth = 1;
	outputHeight = 1;
	// RCAS sharpening after the upscale, in stops (0 the strongest, 2 none). AMD's default 0.2
	// sharpened the meadow past the native render's detail; at 0.6 the upscaled cliff matches
	// native and the meadow keeps most of it, without ringing.
	readonly sharpness = uniform( 0.6 );

	/** The frame graph: it owns the targets below and sizes them. */
	readonly graph: FrameGraph<PipelineFrame>;
	/** The graph's textures that materials outside the pipeline read: the resolved opaque colour
	 * and linear depth (the water, the spray, the rain), the nearest surface's depth, the
	 * underwater pyramid and the refraction pass's colour and depth (the water). */
	readonly textures: {
		readonly opaqueColor: THREE.Texture; readonly linearDepth: THREE.Texture; readonly surfaceDepth: THREE.Texture;
		readonly underwater: THREE.Texture; readonly refraction: THREE.Texture; readonly refractionDepth: THREE.Texture;
	};
	/** The linear depth's pyramid of nearest and farthest depths, which the sea's reflection traces
	 * over (DepthPyramid.ts). */
	readonly depthPyramid: DepthPyramid;
	/** The sea's reflection of what stands above it, traced and denoised ahead of the water
	 * (ocean/SeaReflection.ts), and its trace's most steps (WaterQuality `ssrIterations`). */
	readonly seaReflection: SeaReflection;
	readonly seaIterations: number;
	/** The sea's mean level (m), which the underwater pyramid keeps what lies under (main.ts sets it
	 * from the shared uniforms'). */
	readonly seaLevel = uniform( 0 );
	/** The refraction pass's camera (ocean/Refraction.ts). */
	readonly refraction = new RefractionView();
	readonly resources: {
		sceneColor: ResourceHandle; sceneDepth: ResourceHandle; linearDepth: ResourceHandle; opaqueColor: ResourceHandle;
		meadowCoverage: ResourceHandle | null; meadowDepth: ResourceHandle | null; meadowShadows: ResourceHandle | null;
		gtaoOpaque: ResourceHandle | null; gtaoOpaqueDepth: ResourceHandle | null; gtaoDepth: ResourceHandle | null; gtaoRaw: ResourceHandle | null; gtaoBlur: ResourceHandle | null; gtao: ResourceHandle | null;
		underwater: ResourceHandle; refractionReach: ResourceHandle; refraction: ResourceHandle; refractionDepth: ResourceHandle;
		surfaceDepth: ResourceHandle; velocity: ResourceHandle | null; meterLight: ResourceHandle; exposureState: ResourceHandle; exposureStatePrev: ResourceHandle; surfaceDepthPrev: ResourceHandle | null; veilReactive: ResourceHandle | null;
		rainAir: ResourceHandle | null; rainOverlay: ResourceHandle | null; rainLayer: ResourceHandle | null; rainInput: ResourceHandle | null; rainBehind: ResourceHandle | null; rainGrade: ResourceHandle | null;
		hdr: ResourceHandle | null; sharp: ResourceHandle; display: ResourceHandle;
		easu: ResourceHandle | null; canvas: ResourceHandle;
	};
	readonly air: VolumetricHaze;
	readonly shafts = new LightShafts();
	readonly options: Readonly<PostOptions>;
	/** The temporal modes' resolve (null under msaa and off, and with `taaResolve=off`). */
	readonly temporal: TemporalAA | null = null;
	/** FSR 3 under fsr3 (Fsr3Upscaler.ts), else null. */
	readonly fsr3: Fsr3Upscaler | null = null;
	/** Whether the paused page will draw the same frame from now on (Settled.ts). */
	readonly settled = new Settled();

	depthQuad: THREE.QuadMesh;
	// (msaa and off: under the temporal modes the opaque copy is a texture copy)
	copyQuad: THREE.QuadMesh | null = null;
	underwaterQuad: THREE.QuadMesh;
	refractionReachQuad: THREE.QuadMesh;
	refractionMaskQuad: THREE.QuadMesh;
	surfaceQuad: THREE.QuadMesh;
	// the meadow's screen-space shadows: the depth's nearest view z and the march (MeadowShadows.ts)
	meadowDepthQuad: THREE.QuadMesh | null = null;
	meadowShadowQuad: THREE.QuadMesh | null = null;
	// the ambient occlusion: the half-resolution view z, the slices and the blur (GTAO.ts)
	gtaoQuads: { depth: THREE.QuadMesh; slices: THREE.QuadMesh; blurX: THREE.QuadMesh; blurY: THREE.QuadMesh } | null = null;
	// (msaa, off and the temporal modes with `taaResolve=off`: the resolve has a canvas pass of its own)
	output: THREE.RenderPipeline | null = null;
	// FSR.ts's sharpening canvas pass after EASU (the fsrPort option), or three's FSR1Node (post=three-fsr)
	upscaledOutput: THREE.RenderPipeline | null = null;
	upscaledOutputThree: THREE.RenderPipeline | null = null;
	fsr: FSR | null = null;
	// the composite of the haze, the rain veil and the shafts over the scene, into hdr (the
	// compositeOnce option)
	compositeQuad: THREE.QuadMesh | null = null;
	// the post passes' per-frame constants, into a texel (`postConstants`)
	constants = {} as { quad: THREE.QuadMesh; output: ResourceHandle };
	// the sky's per-frame constants (the discs' cloud transmittance), into a texel before the scene
	// (`skyConstants`)
	skyConstants = {} as { quad: THREE.QuadMesh; output: ResourceHandle };
	// after the resolve, the rain's layer over the graded image (AfterResolve.ts), and the rain
	layerQuad: THREE.QuadMesh | null = null;
	rainInputQuad: THREE.QuadMesh | null = null;
	rainGradeQuad: THREE.QuadMesh | null = null;
	readonly rain: Rain | undefined;
	// the composite and the bloom (the sharp image)
	sharpQuad!: THREE.QuadMesh;
	// the graded image at the render's size, before the copy or upscale to the canvas
	displayQuad!: THREE.QuadMesh;
	// tidewater's physical bloom (PhysicalBloom.ts)
	bloom!: PhysicalBloom;
	private readonly exposureKeys: ExposureKey = { key: 0, highlight: 0, sky: 0, adaptUp: 0.05, adaptDown: 0.05 };
	// the adaptation: reset at the next frame (a cut), the scene clock at the last frame
	exposureReset = true;
	private exposureTime: number | null = null;
	/** Debug builds: the exposure has not yet reached its target (more than 0.01 stop off, as last
	 * read back), which keeps the page's settled signal false (assemble/SceneFrame.ts `pending`). */
	exposurePending = false;
	// a readback of the exposure state is in flight (one at a time)
	private exposureReading = false;
	// what the frame draws from, for the pre-exposure's readings (PreExposure.ts `viewFrame`)
	private readonly viewKey: number[] = [];
	/** The metering histograms and the meter (Metering.ts); without the composite pass
	 * (`post=inline-composite`) no histogram is taken and the meter reads none. */
	metering!: Metering;
	// the local tone mapping (LocalTone.ts; null with `flags=localTone:false`)
	readonly localTone: LocalTone | null = null;
	/** The grade's two tables' base nodes (GradeLut.ts): a table of LUT_SIZE^3 set as a node's
	 * `value` replaces it; `settings.lut` blends from the first to the second. */
	readonly luts = lutNodes();
	/** The first table's grade (GradeLut.ts): `setGrade` changes it and fills the table anew. */
	readonly grade: GradeParams;
	// the shadows' contrast this frame (`shadowContrast`)
	private readonly shade = { pivot: 0, slope: 0 };
	private readonly desaturation = { amount: 0, cool: 0 };
	private readonly white = { r: 1, g: 1, b: 1 };
	// the white the tone curve's input matrix was last built for
	private toneWhite = [ NaN, NaN, NaN ];
	// the tone curve whose coefficients were last solved (ToneCurve.ts)
	private toneCurveKey = '';
	u = {
		// tidewater's grade (the bloom's strength the display's: the constructor sets it)
		bloom: uniform( 0 ), tone: uniform( 0 ), night: uniform( 0 ),
		vignette: uniform( 1 ),
		// the shadows' contrast (`shadowContrast`): its pivot as an exposed luminance, and its slope
		shadePivot: uniform( 0.8 ), shadeSlope: uniform( 0 ),
		// the desaturation before the local gain (`desaturation`)
		desaturate: uniform( 0 ), desaturateCool: uniform( 0 ),
		// overcast's highlights toward grey (`highlightGrey`): overcast's share by day
		highlightShare: uniform( 0 ),
		// the white balance alone (`whiteBalance`), through which the desaturation's cool share reads
		// its colours
		balance: uniform( new THREE.Matrix3() ),
		// the Purkinje shift's weight: the night's times its strength, the key's share of the light
		// its retinal scale is taken from and the retinal illuminance that light is taken to
		// (Purkinje.ts `purkinjeScale`)
		purkinje: uniform( 0 ), purkinjeKey: uniform( 0 ), purkinjeRetina: uniform( 1 ),
		// the tone curve's input matrix (the white balance in it) and output matrix
		toneIn: uniform( new THREE.Matrix3() ), toneOut: uniform( TONE_SPACE.output ),
		// the tone curve's numerator's a, b, c and its denominator's d, e, f (ToneCurve.ts)
		toneNum: uniform( new THREE.Vector3() ), toneDen: uniform( new THREE.Vector3() ),
		// the blend toward the grade's second table
		lut: uniform( 0 ),
		// the scene camera's (inside a full-screen pass the camera nodes refer to the pass's own camera)
		projInv: uniform( new THREE.Matrix4() ), cameraWorld: uniform( new THREE.Matrix4() ),
		proj: uniform( new THREE.Matrix4() ), view: uniform( new THREE.Matrix4() ),
		// the key light's direction in the scene camera's view space (the meadow's screen-space shadows)
		keyView: uniform( new THREE.Vector3( 0, 1, 0 ) ),
		// 1 where the display pass adds the dither (drawn at the canvas's size, not upscaled after)
		ditherOn: uniform( 1 ),
		// 1 while the after pass draws (AfterResolve.ts): else its overlay holds an earlier frame's;
		// and while the rain's layer pass runs
		afterOn: uniform( 0 ), layerOn: uniform( 0 ),
	};


	/** `look`: the display's tables (look/DisplayLook.ts `createDisplayTables`); `temporal`: the
	 * tier's resolve settings (Quality.ts `taa`); `extraPasses`: ExtraPass.ts. */
	constructor( renderer: THREE.WebGPURenderer, readonly sky: Sky, readonly look: DisplayTables, sunShadow: SunShadow, rain?: Rain, options: Partial<PostOptions> = postOptionsFromURL(), temporal?: TemporalSettings, readonly extraPasses: readonly ExtraPass[] = [], seaIterations = DEFAULT_WATER_QUALITY.ssrIterations ) {

		// (the bloom's and the vignette's strengths tunable on debug pages, debug/Tune.ts:
		// `post.bloom.strength`, `post.vignette.strength`)
		this.settings = tuneObject( {
			// (the exposure starts at the sky's exposure by day)
			exposure: sky.exposure.day, exposureOffset: 0,
			// (tunable on debug pages, debug/Tune.ts: `post.tone.blackSlope`, `midGrey`, `white`)
			curve: tuneObject( { ...look.toneCurve }, { toe: 'post.tone.blackSlope', mid: 'post.tone.midGrey', white: 'post.tone.white' },
				{ toe: { min: 1.5, max: 3 }, mid: { min: 0.12, max: 0.22 }, white: { min: 8, max: 32, scale: 'log' } } ),
			lut: 0, ...look.strengths, tone: TONE, night: 0, exposureNight: 0, skyMix: [ 0, 1, 0, 0 ],
		} as PipelineSettings, { bloom: 'post.bloom.strength', vignette: 'post.vignette.strength' },
		{ bloom: { min: 0, max: 2 }, vignette: { min: 0, max: 1.5 } } );
		this.u.bloom.value = look.strengths.bloom;
		this.grade = gradeOf( {}, look.baseGrade );
		this.renderer = renderer;
		this.rain = rain;
		// (a tone curve the solve cannot meet keeps the last good one and reports it, debug/Tune.ts)
		const curve = this.settings.curve;
		tuneCheck( [ 'post.tone.blackSlope', 'post.tone.midGrey', 'post.tone.white' ], () => checkToneCurve( solveToneCurve( curve ), curve.white ) );
		// (the exposed mid grey as the default curve shows it, in linear display values: the grade's
		// contrast pivots there, GradeLut.ts)
		fillGradeLut( this.grade, look.toneCurve.mid, this.luts[ 0 ].value as THREE.Data3DTexture );
		this.options = { ...DEFAULT_POST_OPTIONS, ...options };
		this.settled.settle = ( temporal ?? QUALITY.high.taa ).restSettle;
		const O = this.options;
		this.air = new VolumetricHaze( sunShadow, sky.atmosphere, O.hazeGrid, sky.sunBeamIlluminance, sky.moonBeamIlluminance );
		// the clouds' march ends where this haze closes (sky/CloudKernels.ts `makeMarch`): it reads the medium
		// from uniforms of its own, built before the haze, which the haze writes with its own
		this.air.cloudMedium = sky.clouds.haze;
		// the island's cover hides the haze from the open sky: the rain's collision map (its top,
		// drawn at load and on every Grow, in every weather), times the ambient probes' sky
		// visibility, which exists only once main.ts installs the volume with the world, so the
		// shadow kernels that read it compile then (`compileHazeShadow`), not in the warm-up
		if ( rain ) {

			const collision = rain.collision;
			this.air.setCover( { topAt: collision.topAt, seaLevel: collision.waterLevel, get version() { return collision.version; }, skyVisibility: ambientSkyVisibility, late: true } );

		}
		// the sky fogs itself (the composite leaves its pixels), its own ray with the dome's thin haze,
		sky.fogDome( ( d, distance ) => this.air.skyHaze( d, distance, undefined, true ) );
		// and its panoramas from the sea's level: the seen one and the probe's as the dome takes the
		// haze, the land's light without the fog layer (Sky.ts `fogEnvironment`)
		sky.fogEnvironment( ( d, distance ) => this.air.openSky( d, 0, sky, distance ), ( d, distance ) => this.air.openSky( d, 0, sky, distance, false ) );
		// The scene's target: half-float colour and float depth at the scene's sample count, then
		// the resolves the water and the post passes read (single-sampled).
		const g = this.graph = new FrameGraph<PipelineFrame>( renderer, { sceneSamples: SCENE_SAMPLES, legacyDiscard: O.discardScene ? [ 'rain' ] : [], dev: DEV, prime: PRIME } );
		const sceneColor = g.color( 'sceneColor', { size: RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType, samples: 'scene' } );
		const sceneDepth = g.depth( 'sceneDepth', { size: RENDER, samples: 'scene', type: THREE.FloatType } );
		const linearDepth = g.color( 'linearDepth', { size: RENDER, format: THREE.RGFormat, type: THREE.FloatType } );
		// (with mips, which the water's refraction reads; the sea's reflection reads level 0, ocean/ScreenReflection.ts)
		const opaqueColor = g.color( 'opaqueColor', { size: RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType, mips: true } );
		const surfaceDepth = g.color( 'surfaceDepth', { size: RENDER, format: THREE.RedFormat, type: THREE.FloatType } );
		// what the water refracts at its footprint: the opaque colour under the sea and its mips
		const underwater = g.color( 'underwater', { size: RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType, mips: true } );
		// (read along the footprint's long axis: a surface seen at a slant stretches it)
		g.texture( underwater ).anisotropy = 8;
		// how far past each edge of the screen the water's refracted lookups reach (Pipeline's
		// refraction mask): r below the screen per column, g left and b right of it per row
		const refractionReach = g.color( 'refractionReach', { size: { of: 'fixed', width: REACH, height: 1 }, format: THREE.RGBAFormat, type: THREE.HalfFloatType, filter: 'nearest' } );
		// the water's refraction source: the world under the sea, lit, and its depth (single-sampled)
		const refraction = g.color( 'refraction', { size: REFRACTION_SIZE, format: THREE.RGBAFormat, type: THREE.HalfFloatType } );
		const refractionDepth = g.depth( 'refractionDepth', { size: REFRACTION_SIZE, type: THREE.FloatType } );
		// The temporal modes' motion vectors (rg: the NDC motion since the previous frame, b: the
		// surface's class, a: its reactive weight; Motion.ts), and the nearest surface's depth of the
		// previous frame, for the resolve's disocclusion test (a copy at the frame's end).
		// (in the scene's target: its sample count, one under the temporal modes)
		const velocity = TEMPORAL ? g.color( 'velocity', { size: RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType, samples: 'scene' } ) : null;
		// The metering illuminance M (the header's rules), the scene target's last attachment; at 4x
		// its texture is the target's multisample resolve, the average of the samples.
		const meterLight = g.color( METER_LIGHT, { size: RENDER, format: THREE.RedFormat, type: THREE.HalfFloatType, samples: 'scene' } );
		// The exposure state ( EV, its velocity, EV_t, moved: Exposure.ts ), one texel the meter writes,
		// and the previous frame's, copied at the frame's end
		const exposureState = g.color( 'exposureState', { size: EXPOSURE_TEXEL, format: THREE.RGBAFormat, type: THREE.FloatType } );
		const exposureStatePrev = g.color( 'exposureStatePrev', { size: EXPOSURE_TEXEL, format: THREE.RGBAFormat, type: THREE.FloatType }, true );
		const surfaceDepthPrev = TEMPORAL && ! FSR3 ? g.color( 'surfaceDepthPrev', { size: RENDER, format: THREE.RedFormat, type: THREE.FloatType }, true ) : null;
		// the rain veil's reactive weight per pixel, for the resolve, written by the composite
		// (persistent: zero until it is written, so the resolve's binding never changes)
		const veilReactive = ( RESOLVE || FSR3 ) && ! AFTER_RESOLVE ? g.color( 'veilReactive', { size: RENDER, format: THREE.RedFormat, type: THREE.HalfFloatType }, true ) : null;
		// After the resolve (AfterResolve.ts): the air's transmittance at each pixel, which the
		// composite writes in place of veilReactive (the same cost: rainy or fair, nothing more), for the
		// streaks drawn later; and the streaks and spray themselves, drawn at the canvas's size
		// (premultiplied scene radiance, coverage).
		const rainAir = AFTER_RESOLVE && O.compositeOnce ? g.color( 'rainAir', { size: RENDER, format: THREE.RedFormat, type: THREE.HalfFloatType } ) : null;
		const rainOverlay = AFTER_RESOLVE ? g.color( RAIN_OVERLAY, { size: CANVAS, format: THREE.RGBAFormat, type: THREE.HalfFloatType } ) : null;
		// (and the change all the rain makes to the graded image, which the canvas pass adds, and what
		// it reads at a render texel: the grade's input and the air's transmittance, and the light the
		// air adds)
		const rainLayer = AFTER_RESOLVE ? g.color( 'rainLayer', { size: CANVAS, format: THREE.RGBAFormat, type: THREE.HalfFloatType } ) : null;
		const rainInput = AFTER_RESOLVE ? g.color( 'rainInput', { size: RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType } ) : null;
		const rainBehind = rainInput && rainAir ? g.color( 'rainBehind', { size: RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType } ) : null;
		// (the grade as a table, for the canvas pass that draws the layer)
		const rainGrade = AFTER_RESOLVE ? g.color( 'rainGrade', { size: { of: 'fixed', width: GRADE_TABLE.size ** 2, height: GRADE_TABLE.size }, format: THREE.RGBAFormat, type: THREE.HalfFloatType } ) : null;
		// the composite and the sharp image (half float, as three's rtt() made them)
		const hdr = O.compositeOnce ? g.color( 'hdr', { size: RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType } ) : null;
		// (under fsr3 the sharp image and the display pass run on FSR's output, at the canvas's size)
		const sharp = g.color( 'sharp', { size: FSR3 ? CANVAS : RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType } );
		// (linear light under the temporal resolve: no per-tap decode in the resolve, half the bytes)
		const LINEAR_DISPLAY = RESOLVE && ( temporal ?? QUALITY.high.taa ).linear;
		// (under fsr3 at the canvas's size in half float: display values, which the copy to the canvas
		// rounds to 8 bits; a float target there would take twice the memory)
		const display = g.color( 'display', { size: FSR3 ? CANVAS : RENDER, format: THREE.RGBAFormat, type: LINEAR_DISPLAY || FSR3 ? THREE.HalfFloatType : THREE.FloatType } );
		// the upscale's EASU image (the canvas's size) and the canvas (the resolve has its own)
		const easu = O.fsrPort && ! RESOLVE && ! FSR3 ? g.color( 'FSR.easu', { size: { of: 'canvas' }, format: THREE.RGBAFormat, type: THREE.HalfFloatType } ) : null;
		const canvas = g.import( 'canvas', null, { of: 'canvas' } );
		// The meadow's screen-space shadows (MeadowShadows.ts): the colour the vegetation's depth pass
		// draws beside the scene's depth (three's targets carry one; never read: with an alpha channel,
		// which alpha to coverage needs), the depth's nearest view z and the shadow.
		const meadowCoverage = MEADOW_SHADOWS ? g.color( 'meadowCoverage', { size: RENDER, format: THREE.RGBAFormat, type: THREE.UnsignedByteType, samples: 'scene' } ) : null;
		const meadowDepth = MEADOW_SHADOWS ? g.color( 'meadowDepth', { size: RENDER, format: SS_SHADOW ? THREE.RGFormat : THREE.RedFormat, type: THREE.FloatType } ) : null;
		// (over the whole scene the depth's nearest view z and its surface's thickness, and the shadow
		// with both beside it, which the ground marches over: MeadowShadows.ts)
		const meadowShadowMap = MEADOW_SHADOWS ? g.color( 'meadowShadows', SS_SHADOW ? { size: RENDER, format: THREE.RGBAFormat, type: THREE.HalfFloatType, filter: 'nearest' } : { size: RENDER, format: THREE.RedFormat, type: THREE.UnsignedByteType, filter: 'nearest' } ) : null;
		// The ambient occlusion (GTAO.ts): at half the render's size, the opaque copies' distance (and
		// over the whole scene their surfaces' thickness, post/OpaquePrepass.ts) and
		// their depth (post/OpaquePrepass.ts; the meadow's screen-space shadows read them too over the whole scene,
		// MeadowShadows.ts `SS_SHADOW`), the view z, the slices' and the blur's ( occlusion, view z )
		const half: SizeClass = { of: 'render', divisor: 2 };
		const OPAQUE_PREPASS = GTAO || SS_SHADOW;
		const gtaoOpaque = OPAQUE_PREPASS ? g.color( 'gtaoOpaque', { size: half, format: SS_SHADOW ? THREE.RGFormat : THREE.RedFormat, type: THREE.FloatType, filter: 'nearest' } ) : null;
		const gtaoOpaqueDepth = OPAQUE_PREPASS ? g.depth( 'gtaoOpaqueDepth', { size: half, type: THREE.FloatType } ) : null;
		const gtaoDepth = GTAO ? g.color( 'gtaoDepth', { size: half, format: THREE.RedFormat, type: THREE.FloatType, filter: 'nearest' } ) : null;
		const gtaoRaw = GTAO ? g.color( 'gtaoRaw', { size: half, format: THREE.RGFormat, type: THREE.HalfFloatType, filter: 'nearest' } ) : null;
		const gtaoBlur = GTAO ? g.color( 'gtaoBlur', { size: half, format: THREE.RGFormat, type: THREE.HalfFloatType, filter: 'nearest' } ) : null;
		const gtao = GTAO ? g.color( 'gtao', { size: half, format: THREE.RGFormat, type: THREE.HalfFloatType, filter: 'nearest' } ) : null;
		this.resources = { meadowCoverage, meadowDepth, meadowShadows: meadowShadowMap, gtaoOpaque, gtaoOpaqueDepth, gtaoDepth, gtaoRaw, gtaoBlur, gtao, sceneColor, sceneDepth, linearDepth, opaqueColor, underwater, refractionReach, refraction, refractionDepth, surfaceDepth, velocity, meterLight, exposureState, exposureStatePrev, surfaceDepthPrev, veilReactive, rainAir, rainOverlay, rainLayer, rainInput, rainBehind, rainGrade, hdr, sharp, display, easu, canvas };
		// the targets the frame draws into, made now (the probes, the warm-ups and the water's
		// textures need them before the first frame)
		g.renderTarget( velocity ? [ sceneColor, velocity, meterLight ] : [ sceneColor, meterLight ], sceneDepth );
		for ( const h of [ linearDepth, opaqueColor, underwater, refractionReach, display ] ) g.renderTarget( [ h ] );
		g.renderTarget( [ refraction ], refractionDepth );
		g.renderTarget( [ surfaceDepth ] );
		if ( meadowCoverage && meadowDepth && meadowShadowMap ) {

			g.renderTarget( [ meadowCoverage ], sceneDepth );
			g.renderTarget( [ meadowDepth ] );
			g.renderTarget( [ meadowShadowMap ] );
			meadowShadows.value = g.texture( meadowShadowMap );
			meadowDepthMap.value = g.texture( meadowDepth );

		}
		if ( gtaoOpaque && gtaoOpaqueDepth ) g.renderTarget( [ gtaoOpaque ], gtaoOpaqueDepth );
		if ( gtaoDepth && gtaoRaw && gtaoBlur && gtao ) {

			for ( const h of [ gtaoDepth, gtaoRaw, gtaoBlur, gtao ] ) g.renderTarget( [ h ] );
			gtaoMap.value = g.texture( gtao );

		}
		this.depthPyramid = new DepthPyramid( g.texture( linearDepth ) );
		this.seaReflection = new SeaReflection( g );
		this.seaIterations = seaIterations;
		this.textures = {
			opaqueColor: g.texture( opaqueColor ), linearDepth: g.texture( linearDepth ), surfaceDepth: g.texture( surfaceDepth ),
			underwater: g.texture( underwater ), refraction: g.texture( refraction ), refractionDepth: g.texture( refractionDepth ),
		};
		// the sizes the passes read, the graph's shared extent uniforms (set with the targets' own
		// sizes): the render's and the canvas's
		const renderSize = g.extent( RENDER ), canvasSize = g.extent( CANVAS );
		const u = this.u;
		// the frame's exposure of what hdr holds: 2^EV from the exposure state the meter writes
		// (Metering.ts) over the pre-exposure 2^P hdr already carries (PreExposure.ts), read by every
		// pass after the meter (a new read at each call: FSR 3's accessors take it so, Bindings.ts)
		// (the state's x: its EV, Exposure.ts)
		const exposureRead = () => exp2( ( textureLoad( texture( g.texture( exposureState ) ), ivec2( 0 ) ) as unknown as V4 ).x ).mul( preExposure.inverse );
		const exposure = exposureRead();
		// Multisampled depth can only be loaded, a sample at a time. The sky stays far away (-1e5).
		const viewZ = ( coord: V2, d: F ) => {

			const z = float( - 1e5 ).toVar();
			If( d.lessThan( 0.999999 ), () => {

				const q = coord.div( renderSize );
				const v = u.projInv.mul( vec4( q.x.mul( 2 ).sub( 1 ), float( 1 ).sub( q.y ).mul( 2 ).sub( 1 ), d, 1 ) ).toVar();
				z.assign( v.z.div( v.w ) );

			} );
			return z;

		};
		const msDepth = texture( g.texture( sceneDepth ) );
		const depthSample = ( coord: V2, i: number ) => textureLoad( msDepth, ivec2( coord ), int( i ) ) as unknown as F;
		// Sample 0 for what reads the depth at a point (the water's traces, rain, spray and the haze),
		// and the farthest of the samples for the water's view of what lies behind it.
		// The water is shaded once per pixel, but drawn only into the samples it is in front of: at an
		// edge pixel where a cliff, a figure or a leaf covers sample 0, the samples the water does
		// cover hold what is behind it, the farthest one.
		const resolveMat = new THREE.NodeMaterial();
		resolveMat.fragmentNode = Fn( () => {

			const coord = screenCoordinate.xy;
			const d0 = depthSample( coord, 0 ).toVar();
			// Single-sampled, the depth texture has no samples to load past 0 (the third argument
			// would read a mip level instead).
			const far = SCENE_SAMPLES === 4 ? max( max( d0, depthSample( coord, 1 ) ), max( depthSample( coord, 2 ), depthSample( coord, 3 ) ) ) : d0;
			return vec4( viewZ( coord, d0 ), viewZ( coord, far ), 0, 0 );

		} )();
		this.depthQuad = new THREE.QuadMesh( resolveMat );
		if ( meadowDepth ) {

			// (over the whole scene with the opaque copies' distance, MeadowShadows.ts `SS_SHADOW`)
			const opaque = SS_SHADOW && gtaoOpaque ? { map: g.texture( gtaoOpaque ), halfSize: g.extent( half ) } : undefined;
			this.meadowDepthQuad = new THREE.QuadMesh( meadowDepthMaterial( g.texture( sceneDepth ), renderSize, u.projInv, opaque ) );
			this.meadowShadowQuad = new THREE.QuadMesh( meadowShadowMaterial( g.texture( meadowDepth ), renderSize, u.proj, u.projInv, u.keyView ) );
			// the ground's march (TerrainMaterial.ts) reads the same values
			meadowGroundView.size.value = renderSize.value;
			meadowGroundView.proj.value = u.proj.value;
			meadowGroundView.key.value = u.keyView.value;

		}
		if ( gtaoOpaque && gtaoDepth && gtaoRaw && gtaoBlur ) {

			// (the passes and the receivers read the graph's own extents)
			gtaoView.size.value = g.extent( half ).value;
			gtaoView.render.value = renderSize.value;
			this.gtaoQuads = {
				depth: new THREE.QuadMesh( gtaoDepthMaterial( g.texture( sceneDepth ), g.texture( gtaoOpaque ), u.projInv ) ),
				slices: new THREE.QuadMesh( gtaoMaterial( g.texture( gtaoDepth ) ) ),
				blurX: new THREE.QuadMesh( gtaoBlurMaterial( g.texture( gtaoRaw ), g.texture( gtaoDepth ), 1, 0 ) ),
				blurY: new THREE.QuadMesh( gtaoBlurMaterial( g.texture( gtaoBlur ), g.texture( gtaoDepth ), 0, 1 ) ),
			};

		}
		if ( ! TEMPORAL ) {

			const copyMat = new THREE.NodeMaterial();
			copyMat.fragmentNode = textureLoad( texture( g.texture( sceneColor ) ), ivec2( screenCoordinate.xy ) );
			this.copyQuad = new THREE.QuadMesh( copyMat );

		}
		// The underwater pyramid's top level: the opaque colour where every sample of the pixel lies
		// below the refraction's clip height (the floor, the stones, the cliffs' feet: what a ray
		// through the sea can reach, as the refraction image holds it), premultiplied by that weight
		// (alpha), and nothing where one lies above it (a cliff, the rim's grass, a figure) or is
		// sky. Its mips, each a box filter of the level above, are then the average colour of what
		// lies under the sea over any footprint, the surfaces above it left out: the water divides
		// by the alpha. What swims in the water (marked by Refraction.ts `markSwimmer`: an alpha of 0)
		// is left out too, as are the pixels it shares: the water finds it along its refracted ray
		// in the opaque image (Water.ts), and the floor behind it comes from the refraction image,
		// which never draws it.
		const underwaterMat = new THREE.NodeMaterial();
		underwaterMat.fragmentNode = Fn( () => {

			const coord = screenCoordinate.xy;
			const q = coord.div( renderSize );
			const clip = this.seaLevel.add( CLIP_MARGIN );
			const below = ( d: F ) => {

				const v = u.projInv.mul( vec4( q.x.mul( 2 ).sub( 1 ), float( 1 ).sub( q.y ).mul( 2 ).sub( 1 ), d, 1 ) ).toVar();
				const y = u.cameraWorld.mul( vec4( v.xyz.div( v.w ), 1 ) ).y;
				return step( y, clip ).mul( step( d, 0.999999 ) );

			};
			let w = below( depthSample( coord, 0 ) );
			for ( let i = 1; i < SCENE_SAMPLES; i ++ ) w = w.mul( below( depthSample( coord, i ) ) );
			const color = ( textureLoad( texture( g.texture( sceneColor ) ), ivec2( coord ) ) as unknown as V4 ).toVar();
			const weight = w.mul( step( 0.999, color.a ) ).toVar();
			return vec4( color.rgb.mul( weight ), weight );

		} )();
		this.underwaterQuad = new THREE.QuadMesh( underwaterMat );
		// How far past each edge of the screen the water's refracted lookups reach. Light bends down
		// into the water, so a refracted ray lands on floor lower on the screen than the pixel it is
		// seen through: from the island camera's height about a fiftieth of the sea's pixels look
		// past the screen's edge, a few per cent of its height at most, but seen low over the sea a
		// quarter to half of them land under the near water, below the edge, as far as the guard
		// band reaches and beyond. Each texel follows REACH_SAMPLES pixels toward one edge (denser
		// near it) along its column (below) or row (left, right): the camera's ray to the sea,
		// refracted, down to the floor the opaque pixel shows straight behind (where that pixel drew
		// what lies under the sea), its end projected; the farthest past the edge. The flat sea and
		// its ends below the screen alone missed a few in a thousand of the lookups past the edge
		// low over a strong wind's sea; with the tilts none in ten thousand.
		const tUnderwater = texture( g.texture( underwater ) );
		const reachMat = new THREE.NodeMaterial();
		reachMat.name = 'refractionReach';
		reachMat.fragmentNode = Fn( () => {

			const eye = u.cameraWorld.mul( vec4( 0, 0, 0, 1 ) ).xyz.toVar();
			const sea = this.seaLevel;
			// How far past the edges the refracted rays seen through screen uv q end: ( below, left,
			// right, the sideways spill of an end below ), or 0 where q shows no floor under the sea.
			// The sea's slopes tilt the rays, sideways most of all, which the flat sea leaves out
			// (low over a gale's sea a tenth of the screen and more): the surface tilted by
			// REACH_SLOPE across the view and along it as well.
			const reachAt = ( q: V2 ) => {

				const texel = ivec2( clamp( q.mul( renderSize ), vec2( 0 ), renderSize.sub( 1 ) ) );
				const d = depthSample( vec2( texel ), 0 );
				const v = u.projInv.mul( vec4( q.x.mul( 2 ).sub( 1 ), float( 1 ).sub( q.y.mul( 2 ) ), d, 1 ) ).toVar();
				const floor = u.cameraWorld.mul( vec4( v.xyz.div( v.w ), 1 ) ).xyz.toVar();
				const dir = normalize( floor.sub( eye ) ).toVar();
				const out = vec4( 0 ).toVar();
				If( textureLoad( tUnderwater, texel, int( 0 ) ).a.greaterThan( 0.999 ).and( floor.y.lessThan( sea ) ).and( dir.y.lessThan( - 1e-3 ) ).and( eye.y.greaterThan( sea ) ), () => {

					const surface = eye.add( dir.mul( eye.y.sub( sea ).div( dir.y.negate() ) ) ).toVar();
					const along = normalize( vec3( dir.x, 0, dir.z ).add( vec3( 1e-5, 0, 0 ) ) ).toVar();
					const across = vec3( along.z.negate(), 0, along.x ).toVar();
					for ( const tilt of [ [ 0, 0 ], [ 1, 0 ], [ - 1, 0 ], [ 0, 1 ], [ 0, - 1 ] ] ) {

						const n = normalize( vec3( 0, 1, 0 ).add( across.mul( tilt[ 0 ] * REACH_SLOPE ) ).add( along.mul( tilt[ 1 ] * REACH_SLOPE ) ) );
						const t = refract( dir, n, 1 / 1.333 );
						// (as the water keeps a ray heading down and bounds its path: Water.ts)
						const down = normalize( vec3( t.x, min( t.y, - 0.08 ), t.z ) ).toVar();
						const end = surface.add( down.mul( min( sea.sub( floor.y ).div( max( down.y.negate(), 0.04 ) ), 80 ) ) );
						const c = u.proj.mul( u.view.mul( vec4( end, 1 ) ) ).toVar();
						const e = vec2( c.x.div( c.w ).mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( c.y.div( c.w ).mul( 0.5 ) ) ).toVar();
						const sideways = max( e.x.negate(), e.x.sub( 1 ) );
						out.assign( max( out, vec4( e.y.sub( 1 ), e.x.negate(), e.x.sub( 1 ), select( e.y.greaterThan( 1 ), sideways, float( 0 ) ) ) ) );

					}

				} );
				return out;

			};
			const at = screenCoordinate.x.div( REACH );
			// (a: how far the ends below the screen spill past its sides, for the corners)
			const below = float( 0 ).toVar(), left = float( 0 ).toVar(), right = float( 0 ).toVar(), spill = float( 0 ).toVar();
			Loop( REACH_SAMPLES, ( { i } ) => {

				// from the middle of the screen to its edge, the steps shrinking toward the edge
				const f = float( 1 ).sub( float( i ).add( 0.5 ).div( REACH_SAMPLES ) );
				const inset = f.mul( f ).mul( 0.5 );
				const b = reachAt( vec2( at, float( 1 ).sub( inset ) ) ).toVar();
				below.assign( max( below, b.x ) );
				spill.assign( max( spill, b.w ) );
				left.assign( max( left, reachAt( vec2( inset, at ) ).y ) );
				right.assign( max( right, reachAt( vec2( float( 1 ).sub( inset ), at ) ).z ) );

			} );
			return vec4( below, left, right, spill );

		} )();
		this.refractionReachQuad = new THREE.QuadMesh( reachMat );
		// The refraction mask: the full-screen triangle lies at the nearest depth, so where it is drawn
		// the refraction pass's fragments fail the depth test before they are shaded. On the screen
		// it is drawn where the opaque image already holds what lies under the sea, from which the
		// water then reads (Water.ts): every opaque pixel within some eight of the refraction texel's
		// place on the screen under the sea (the pyramid's weight of 1 at level 3, read bilinearly: a
		// block of about 16 pixels), so the refraction image is kept for the water's filtered reads
		// beside what stands above the water, and behind it (the four texels a bilinear read there
		// takes: a lone pixel above the water at a small weight let a lookup meet a masked texel). In
		// the guard band off the screen it is drawn beyond the lookups' reach past that edge
		// (refractionReach, the largest of the columns or rows around; below a corner, the sideways
		// spill of the ends below the screen as well), grown by REACH_MARGIN for the floor's slopes
		// and the waves' beyond the reach's tilt.
		const G = this.refraction.guard; // ( sx, sy, cx, cy ), this frame's (Refraction.ts setGuard)
		const refractionSize = g.extent( REFRACTION_SIZE );
		const tReach = texture( g.texture( refractionReach ) );
		const maskMat = new THREE.NodeMaterial();
		maskMat.name = 'refractionMask';
		maskMat.colorWrite = false;
		maskMat.fragmentNode = Fn( () => {

			const r = screenCoordinate.xy.div( refractionSize );
			const sx = r.x.mul( 2 ).sub( 1 ).mul( G.x ).add( G.z ), sy = float( 1 ).sub( r.y.mul( 2 ) ).mul( G.y ).add( G.w );
			const q = vec2( sx.mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( sy.mul( 0.5 ) ) ).toVar();
			const inside = q.x.greaterThan( 0 ).and( q.x.lessThan( 1 ) ).and( q.y.greaterThan( 0 ) ).and( q.y.lessThan( 1 ) );
			// the reach past an edge around texel index k (a column for below, a row for the sides)
			const reach = ( k: F ) => {

				let m: V4 = vec4( 0 );
				for ( let o = - 2; o <= 2; o ++ ) m = max( m, textureLoad( tReach, ivec2( int( clamp( k.add( o ), 0, REACH - 1 ) ), int( 0 ) ), int( 0 ) ) as unknown as V4 );
				return m.mul( REACH_MARGIN.scale ).add( REACH_MARGIN.add );

			};
			const column = reach( floor( clamp( q.x, 0, 0.9999 ).mul( REACH ) ) ), row = reach( floor( clamp( q.y, 0, 0.9999 ).mul( REACH ) ) );
			// (below a corner, the ends that left the screen through its bottom edge spill sideways)
			const side = select( q.y.greaterThan( 1 ), column.w, float( 0 ) );
			const within = q.y.sub( 1 ).lessThanEqual( column.x ).and( q.x.negate().lessThanEqual( max( row.y, side ) ) ).and( q.x.sub( 1 ).lessThanEqual( max( row.z, side ) ) );
			// (the four level-3 texels around q, each 8 pixels square, all under the sea)
			const level3 = renderSize.div( 8 ).floor().max( 1 );
			const corner = ivec2( q.mul( level3 ).sub( 0.5 ).floor() );
			let all: F = float( 1 );
			for ( const [ dx, dy ] of [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ] ) {

				const texel = ivec2( clamp( vec2( corner ).add( vec2( dx, dy ) ), vec2( 0 ), level3.sub( 1 ) ) );
				all = min( all, ( textureLoad( tUnderwater, texel, int( 3 ) ) as unknown as V4 ).a );

			}
			Discard( select( inside, all.lessThan( 0.999 ), within ) );
			return vec4( 0 );

		} )();
		this.refractionMaskQuad = new THREE.QuadMesh( maskMat );
		// The nearest surface's view z: sample 0's, but the nearest sample's where sample 0 sees the sky
		// and another sample a surface, so a pixel is sky only where all its samples are. Such an edge
		// pixel's colour holds the surface's samples unhazed (the haze composite fogs everything but
		// the sky, which fogs itself): the composite fogs it at the surface's distance, and the sky's
		// share then takes the air it already has once more, which leaves the hazy sky at the horizon
		// nearly as it was. Read by sample 0 alone, the sea's far edge (MSAA's sample 0 lies near the
		// pixel's top, in the sky) stayed unhazed along the horizon's row, a dark line.
		const surfaceMat = new THREE.NodeMaterial();
		surfaceMat.fragmentNode = Fn( () => {

			const coord = screenCoordinate.xy;
			const d0 = depthSample( coord, 0 ).toVar();
			if ( SCENE_SAMPLES !== 4 ) return vec4( viewZ( coord, d0 ) );
			const near = min( min( d0, depthSample( coord, 1 ) ), min( depthSample( coord, 2 ), depthSample( coord, 3 ) ) ).toVar();
			return vec4( viewZ( coord, select( d0.lessThan( 0.999999 ), d0, near ) ) );

		} )();
		this.surfaceQuad = new THREE.QuadMesh( surfaceMat );

		buildDisplay( this, { temporal, renderSize, canvasSize, exposureRead, exposure, linearDisplay: LINEAR_DISPLAY, dev: DEV } );
		declarePasses( this );
		if ( TEMPORAL ) {

			// The jitter shifts the camera's projection through the scene half (the graph's jittered
			// span) and is undone after it; the post passes reconstruct positions with the inverse
			// the frame was drawn with, set once the jitter is on (Jitter.ts): the composite's own, the
			// haze's rays (its composite: the froxel grid itself stays unjittered) and the shafts' (only
			// the sun's position on screen stays unjittered, from `update`).
			jitter.mode = AA_MODE === 'taau' ? 'taau' : FSR3 ? 'fsr3' : 'taa';
			g.hooks.jitterBegin = ( f ) => {

				jitter.begin( f.camera );
				const inverse = f.camera.projectionMatrixInverse;
				this.u.projInv.value.copy( inverse );
				this.u.proj.value.copy( f.camera.projectionMatrix );
				this.air.setProjectionInverse( inverse );
				this.shafts.setProjectionInverse( inverse );
				this.seaReflection.setProjection( f.camera.projectionMatrix );

			};
			g.hooks.jitterEnd = ( f ) => jitter.end( f.camera );
			// across a cut the nearest surface's history is no longer the previous frame's
			const surfaceDepthPrev = this.resources.surfaceDepthPrev;
			if ( surfaceDepthPrev ) onTaaCut( () => g.resetHistory( surfaceDepthPrev ) );

		}
		g.compile();
		// The scene target's multisampled colour and depth are resolved, then never read: the
		// graph's attachment ops discard them in the last pass that draws into them and drop a
		// resolve nothing samples before the next draw, derived per frame from the passes that run
		// (the legacy table, `fgOps=legacy`, discards in the rain only: the discardScene option,
		// `legacyDiscard` above). WebGPU lets a pass load its attachments and still discard them
		// once resolved, which three r186 offers only together with a clear, so AttachmentOps
		// rewrites the stores as three hands each pass's descriptor to WebGPU.
		g.attachmentOps.install();

	}

	/** The wave maps and the caustics (ocean/WaveMaps.ts), baked by their pass while they are set. */
	waveMaps: WaveMaps | null = null;

	// the displacement field as its pass leaves it, and the previous frame's (null without one)
	displacement: { field: ResourceHandle; previous: ResourceHandle | null } | null = null;

	/** Whether the clip tools step a clip (their global is set: assemble/StartScene.ts): the
	 * exposure then snaps every frame. */
	clipping: () => boolean = () => false;

	/** The scene's render size, and the canvas's when it is larger (the frame is then upscaled). */
	setSize( w: number, h: number, outputWidth = w, outputHeight = h ): void {

		// (a history whose size changes is carried over to the next resolve, before the graph resizes it)
		this.temporal?.beforeResize( { render: { width: w, height: h }, canvas: { width: outputWidth, height: outputHeight } } );
		this.width = w; this.height = h;
		this.outputWidth = Math.max( outputWidth, w ); this.outputHeight = Math.max( outputHeight, h );
		// every target by its size class (the bloom's levels, the fixed constants', the upscale's at
		// the canvas), the extent uniforms the passes read their sizes from, and the chain's own
		// passes through their resize hook
		this.graph.setExtents( { render: { width: w, height: h }, canvas: { width: outputWidth, height: outputHeight } } );

	}

	/** The scene's multisampled colour and depth (the opaque scene, the water and the rain draw
	 * into it). */
	get sceneRT(): THREE.RenderTarget {

		return this.graph.target( this.resources.sceneColor );

	}

	/** The opaque pass's linear view z: sample 0 (r) and the farthest sample (g). */
	get depthRT(): THREE.RenderTarget {

		return this.graph.target( this.resources.linearDepth );

	}

	/** The resolved opaque colour (what the water reflects, and refracts where the refraction image
	 * holds nothing). */
	get opaqueRT(): THREE.RenderTarget {

		return this.graph.target( this.resources.opaqueColor );

	}

	/** The nearest surface's linear view z, the sea's included. */
	get surfaceRT(): THREE.RenderTarget {

		return this.graph.target( this.resources.surfaceDepth );

	}

	/** The graded image at the render's size (the profiler labels its pass by it). */
	get displayRT(): THREE.RenderTarget {

		return this.graph.target( this.resources.display );

	}

	// whether the rain's layer shows this frame (AfterResolve.ts): while it rains (the veil) or the
	// after pass draws (its pass then runs, unless the rest holds it)
	layerRuns( f: PipelineFrame ): boolean {

		return f.afterScene !== undefined || ( this.rain?.uniforms.rain.value ?? 0 ) > 0.001;

	}

	// Whether the rain's after pass and layer pass keep what the frame that completed the rest drew
	// (TemporalAA.ts `holding`): both draw over this frame's jittered surface depth and grade input,
	// at its jitter, and once the rest has converged the jitter runs on through the running sequence
	// (Jitter.ts), whose place depends on the frames rendered since the page loaded. Drawn again,
	// a paused frame's rain then depended on how it was reached (and only the rain did: the resolved
	// image is the rest's). The frame that completes the rest draws at the rest sequence's last
	// jitter, and nothing it draws from changes while the rest holds (the rest's key), so its
	// overlay and layer stand for every frame after it; the canvas pass reads them as they are.
	afterHeld(): boolean {

		return this.temporal?.holding === true;

	}

	/** What resolves the frame against its history and owns its rest: the temporal resolve, or FSR 3
	 * (null under msaa, off, and `taaResolve=off`). */
	get resolver(): TemporalAA | Fsr3Upscaler | null {

		return this.temporal ?? this.fsr3;

	}

	/** Whether the frame is rendered below the canvas's resolution and upscaled by the output passes
	 * (never under fsr3, whose output already has the canvas's size). */
	get upscaling(): boolean {

		return ! FSR3 && ( this.outputWidth !== this.width || this.outputHeight !== this.height );

	}

	/** Renders the frame: its uniforms from the camera and the settings, then the frame graph's
	 * passes (`graph.execute`). */
	render( frame: PipelineFrame ): void {

		const camera = frame.camera;
		const u = this.u;
		// the P the last reading asks for, before the rests read the epoch it may bump
		preExposure.beginFrame();
		camera.updateMatrixWorld();
		// the refraction's band below the screen, before the passes that read it
		this.refraction.setGuard( camera, this.seaLevel.value );
		u.projInv.value.copy( ( camera as THREE.PerspectiveCamera ).projectionMatrixInverse );
		u.cameraWorld.value.copy( camera.matrixWorld );
		u.proj.value.copy( camera.projectionMatrix );
		u.view.value.copy( camera.matrixWorldInverse );
		// the resolve takes the opaque scene's depth while the screen-space shadows over the whole scene are on
		// (MeadowShadows.ts `SS_SHADOW`, `flags.ssShadow`)
		screenShadowView.opaque.value = SS_SHADOW && debugFlags.ssShadow !== false ? 1 : 0;
		if ( frame.keyDirection ) u.keyView.value.copy( frame.keyDirection );
		else u.keyView.value.copy( frame.sun.position ).sub( frame.sun.target.position );
		u.keyView.value.transformDirection( camera.matrixWorldInverse );
		if ( GTAO ) gtaoProjection( camera, gtaoView.proj.value );
		if ( TEMPORAL ) {

			// this frame's rest (which picks the rest's jitter), jitter and motion, from the camera as
			// it is (unjittered)
			this.temporal?.beginFrame( camera, this.width, this.height, this.outputWidth, this.outputHeight, frame.time, frame.temporal );
			this.fsr3?.beginFrame( camera, this.width, this.height, this.outputWidth, this.outputHeight, frame.time, frame.temporal );
			jitter.prepare( camera, this.width, this.height, this.outputWidth / this.width, this.outputWidth );
			beginMotionFrame( camera );
			// (FSR's plan after the cuts, from this frame's jitter)
			this.fsr3?.prepareFrame( camera as THREE.PerspectiveCamera, this.width, this.height, this.outputWidth, this.outputHeight );
			if ( this.temporal ) {

				// (after a cut or a resize the previous frame's nearest surface is gone)
				this.temporal.setPreviousDepthValid( this.graph.historyValid( this.resources.surfaceDepthPrev! ) );
				this.temporal.upscaling = this.upscaling;

			}

		}
		if ( frame.settle ) this.settled.observe( camera, this.width, this.height, this.outputWidth, this.outputHeight, frame.time, frame.settle, this.resolver?.rest ?? null );
		// (what the frame draws from, its epoch without the pre-exposure's own bumps, and whether the
		// page waits for nothing but the pre-exposure: its readings end `pending` only from a frame of
		// the settled view, PreExposure.ts)
		preExposure.viewFrame( restKey( this.viewKey, camera, this.width, this.height, this.outputWidth, this.outputHeight, frame.time, frame.settle?.signature ?? [], restEpoch.value - preExposure.changes ),
			frame.settle !== undefined && this.settled.sceneQuiet, this.settled.settle );
		const projection = TEMPORAL ? jitter.uniforms.unjitteredProjection.value : camera.projectionMatrix;
		const haze = frame.haze ?? { time: frame.time, signature: [], stable: false };
		this.air.update( camera, projection, this.width, this.height, this.outputWidth, this.outputHeight, haze );
		// (the sea's reflection rests once the haze's has: the opaque image it reflects is then still)
		this.seaReflection.update( camera, projection, this.width, this.height, { ...haze, stable: haze.stable && ! this.air.pending } );
		this.shafts.update( camera as THREE.PerspectiveCamera, this.air.seaLevelExtinction, this.air.transmittance( camera.position.y, this.shafts.flareDirection ), projection );
		this.updateDisplay( frame );
		this.u.afterOn.value = frame.afterScene !== undefined ? 1 : 0;
		this.u.layerOn.value = this.layerRuns( frame ) ? 1 : 0;
		this.graph.execute( frame );
		// The exposure this frame ended with, read back for the next pre-exposure (PreExposure.ts),
		// and in debug builds whether it has reached its target, for the settled signal. (The copy is
		// queued after this frame's passes, so the reading is this frame's, made at its P.)
		if ( ! this.exposureReading ) {

			this.exposureReading = true;
			const ticket = preExposure.ticket(), settle = frame.settle !== undefined;
			this.readExposure().then( ( e ) => {

				preExposure.observe( ticket, e.ev );
				if ( settle ) this.exposurePending = Math.abs( e.ev - e.target ) > 0.01;

			}, () => {} ).finally( () => { this.exposureReading = false; } );

		}

	}

	/** Sets `changes` over the first table's grade and fills the table anew (GradeLut.ts); the
	 * caller raises the scene event (main.ts `setGrade`, the GUI's). */
	setGrade( changes: GradeChanges = {} ): void {

		mergeGrade( this.grade, changes );
		fillGradeLut( this.grade, this.look.toneCurve.mid, this.luts[ 0 ].value as THREE.Data3DTexture );

	}

	/** Compiles the haze's shadow kernels, which read the ambient probes' sky visibility: main.ts
	 * calls it once it has installed the ambient volume (lighting/AmbientSample.ts), and the loop
	 * waits for it with the warm-up. Registered as kernels of the opaque pass, they compile at once
	 * (the warm-up has started by then). */
	compileHazeShadow(): Promise<void> {

		return this.graph.latent( 'opaque', [], this.air.shadowKernels );

	}

	/** Where the frame's linear scene radiance lies, for its readout (debug/Radiance.ts): `opaque`,
	 * the resolved opaque colour (the lit surfaces and the sky before the sea, the haze and the post
	 * chain; at 4x the samples' box resolve), or `hdr`, the composite's output (the sea and the air
	 * over them, before the bloom and the display chain; null where the composite does not run once,
	 * `post=legacy`). Both are stored to the frame's end, which their later readers need. Also
	 * `meterLight`, the metering illuminance (at 4x resolved and stored while a pass reads it:
	 * the metering histograms, or the debug view, `fgview=meterLight`), and `surfaceDepth`, the nearest surface's view z
	 * (below -9e4: sky), which tells the sky's pixels apart, and `sharp`, the composite with the
	 * bloom added (before the local tone, the tone curve and the grade). */
	radianceSource( source: 'opaque' | 'hdr' | 'sharp' | 'meterLight' | 'surfaceDepth' ): { target: THREE.RenderTarget; index: number } | null {

		const r = this.resources;
		const h = source === 'opaque' ? r.opaqueColor : source === 'hdr' ? r.hdr : source === 'sharp' ? r.sharp : source === 'meterLight' ? r.meterLight : r.surfaceDepth;
		if ( ! h ) return null;
		const target = this.graph.target( h );
		return { target, index: target.textures.indexOf( this.graph.texture( h ) ) };

	}

	/** The exposure state as the last frame left it (Metering.ts, Exposure.ts), read back: the
	 * exposure EV and the target EV_t (stops), EV's velocity (stops per second), `moved` (1 where
	 * that frame's EV differed from the frame's before), and the factor 2^EV that brings scene
	 * radiance to the exposed image (hdr, pre-exposed, takes 2^( EV - P ): PreExposure.ts). */
	async readExposure(): Promise<{ ev: number; velocity: number; target: number; moved: number; exposure: number }> {

		const target = this.graph.target( this.resources.exposureState );
		const index = target.textures.indexOf( this.graph.texture( this.resources.exposureState ) );
		const px = await this.renderer.readRenderTargetPixelsAsync( target, 0, 0, 1, 1, index ) as Float32Array;
		return { ev: px[ 0 ], velocity: px[ 1 ], target: px[ 2 ], moved: px[ 3 ], exposure: 2 ** px[ 0 ] };

	}

	// The display chain's and the canvas pass's uniforms from the settings (the exposure's keys and
	// offset, the grade, the dither).
	private updateDisplay( { time }: PipelineFrame ): void {

		const S = this.settings;
		const u = this.u;
		// the night's weight for the display (the tone, the grade, the bloom) and the exposure's own
		// for its keys (sky/DayCycle.ts `displayNight`, `exposureNight`), each of which a debug page
		// may force (`duskForce`)
		const night = duskForce( 'displayNight', S.night ), exposureNight = duskForce( 'exposureNight', S.exposureNight );
		u.tone.value = S.tone; u.night.value = night; u.lut.value = S.lut;
		u.vignette.value = S.vignette;
		u.ditherOn.value = this.upscaling ? 0 : 1;
		// the clouded day's share: the cloudy and overcast skies' (DayKeys.ts `cloudy`)
		const cloud = cloudedShare( S.skyMix );
		this.localTone?.update( this.sky.elevation, night, cloud );
		this.bloom.update( this.sky.elevation, night, cloud );
		const ex = this.sky.exposure.keysAt( this.sky.elevation, exposureNight, S.skyMix, this.exposureKeys, this.sky.rising ), mu = this.metering.uniforms;
		mu.key.value = ex.key; mu.highlight.value = ex.highlight; mu.skyShare.value = ex.sky; mu.offset.value = S.exposureOffset;
		// the adaptation's step: the scene clock's, once per rendered frame; a paused clock (and the
		// test mode, a paused page), a reset or a lost history take the target at once
		const dt = this.exposureTime === null ? 0 : time - this.exposureTime;
		this.exposureTime = time;
		// (a stepped clip snaps every frame: Exposure.ts, capture-clip.mjs)
		const clip = this.clipping();
		const reset = this.exposureReset || clip || dt <= 0 || ! this.graph.historyValid( this.resources.exposureStatePrev );
		this.exposureReset = false;
		mu.snap.value = reset ? 1 : 0;
		mu.dt.value = reset ? 0 : dt;
		mu.up.value = ex.adaptUp; mu.down.value = ex.adaptDown;
		u.bloom.value = S.bloom * ( 1 + S.tone * 0.5 ) * ( 1 + night * NIGHT.bloom );
		const D = this.look;
		const shade = keyed( D.shadowContrast, this.sky.elevation, night, this.shade, cloud );
		u.shadePivot.value = 0.18 * 2 ** shade.pivot; u.shadeSlope.value = shade.slope;
		const desaturation = keyed( D.desaturation, this.sky.elevation, night, this.desaturation, cloud, this.sky.rising );
		u.desaturate.value = desaturation.amount; u.desaturateCool.value = desaturation.cool;
		u.purkinje.value = night * D.purkinje.strength;
		u.purkinjeRetina.value = D.purkinje.retina;
		const keyColor = this.sky.keyColor;
		u.purkinjeKey.value = ( 0.2126 * keyColor.r + 0.7152 * keyColor.g + 0.0722 * keyColor.b ) * liftedKeyHeight( this.sky.keyLift, this.sky.keyDir, this.sky.keyIsMoon ) / Math.PI;
		// the day's and the clouded white (DayKeys.ts `keyed`, without the night), overcast's
		// (`whiteOvercast`) over them by overcast's share, then the night's over all by the night's
		// weight, as `keyed` blends it: under no overcast the white is `keyed`'s to the bit
		const white = keyed( D.whiteBalance, this.sky.elevation, 0, this.white, cloud, this.sky.rising );
		const overcast = S.skyMix[ 3 ] ?? 0, atNight = D.whiteBalance.night!;
		for ( const k of [ 'r', 'g', 'b' ] as const ) {

			const v = white[ k ] + ( D.whiteOvercast[ k ] - white[ k ] ) * overcast;
			white[ k ] = v + ( atNight[ k ] - v ) * night;

		}
		u.highlightShare.value = overcast * ( 1 - night );
		const { r, g, b } = white;
		if ( r !== this.toneWhite[ 0 ] || g !== this.toneWhite[ 1 ] || b !== this.toneWhite[ 2 ] ) {

			this.toneWhite = [ r, g, b ];
			u.balance.value.copy( whiteBalance( [ r, g, b ] ) );
			u.toneIn.value.copy( TONE_SPACE.input ).multiply( u.balance.value );

		}
		const C = S.curve;
		const key = `${ C.toe } ${ C.mid } ${ C.white }`;
		if ( key !== this.toneCurveKey ) {

			this.toneCurveKey = key;
			const k = solveToneCurve( C );
			// a set the solve cannot meet (a pole or a fold in range: settings far from the
			// default, as the GUI can reach) keeps the last good curve; the default must pass
			const failed = checkToneCurve( k, C.white );
			if ( failed && DEV && key === `${ D.toneCurve.toe } ${ D.toneCurve.mid } ${ D.toneCurve.white }` ) throw new Error( `tone curve ${ key }: ${ failed }` );
			if ( failed ) console.warn( `tone curve ${ key }: ${ failed }; the last good curve stays` );
			else {

				u.toneNum.value.set( k[ 0 ], k[ 1 ], k[ 2 ] );
				u.toneDen.value.set( k[ 3 ], k[ 4 ], k[ 5 ] );

			}

		}

	}

}
