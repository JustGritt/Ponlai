import * as THREE from 'three/webgpu';
import {
	clamp, floor, float, int, ivec2, max, mrt, output, screenCoordinate, screenSize, texture, textureLoad, textureSize, vec2,
} from 'three/tsl';
import { jitter } from './Jitter.ts';
import { preExposedRGBA } from './PreExposure.ts';

// Rain after the resolve: the temporal modes draw the rain unjittered after the resolve.
//
// Rain after the resolve (Antialiasing.ts `AFTER_RESOLVE`: aa=taa|taau). The rain's streaks, its
// splashes, its veil and the storm's spray are thin, fast or brief, and translucent: drawn into the jittered frame they leave
// no motion of their own a history can follow, so the resolve either kept its history under them
// (they faded and trailed) or, told by a reactive mask, took them from one jittered frame (the faint
// distant streaks vanished and the near ones broke into dashes). Under the temporal modes they are
// drawn after the resolve instead, at the canvas's resolution and unjittered, as engines draw
// translucency after their temporal upscaler (Unreal's after-motion-blur translucency): every frame
// shows each streak whole, at output pixels. MSAA keeps its rain as it was, and the ground truth
// (aa=gt) keeps it before its resolve, the supersampled image the streaks are measured against.
//
// - The pass (`rainAfter`, Pipeline.ts): after the resolve's passes, at the canvas's size,
//   unjittered, a scene pass that draws the streaks, splashes and rings and the spray into `rainOverlay` (RGBA16F,
//   cleared to zero through its MRT, `afterMRT`): premultiplied scene radiance in rgb and their
//   coverage in alpha, as each material blends. It has no depth buffer. One sample a pixel and
//   nothing averaging it: the streaks are prefiltered (RainDrops.ts: widened to their profile
//   convolved with a pixel, faded by the ratio), as is the veil (at least 0.8 of a pixel
//   half-wide).
// - Occlusion: the scene's nearest surface (`surfaceDepth`, the sea's included) is at the render's
//   size and was drawn jittered. A fragment finds the four render texels around the point it shows
//   (its canvas position scaled to the render, less this frame's jitter, so the texels line up with
//   the resolved image: `afterTexels`) and tests its depth against each; the tests are weighted
//   bilinearly (`afterResolveCover`). An edge then covers a streak by a fraction over about a render
//   pixel, as the resolve antialiased the edge itself, rather than in the render's steps; a streak
//   in front of both sides is untouched.
// - Haze and shafts: the composite is affine in the scene colour at each pixel,
//   hdr = A scene + B (A the haze's transmittance; B the light the haze and the shafts add). Under
//   AFTER_RESOLVE the composite writes A (`rainAir`, R16F at the render's size, in place of
//   veilReactive: the same cost in fair weather) and no veil; B is the grade's input less A times
//   the scene colour, so a streak takes the air of what lies behind it, as it did drawn into the
//   scene before the composite.
// - The veil (Rain.veil): evaluated at the canvas pixel's unjittered ray (interpolated from the
//   pass's vertices), in front of what lies at the render texel nearest the point it shows, as the
//   composite laid it over the air (the shafts the composite added over it now lie under it); a
//   shell's work only past it (none at all for what lies before the nearest shell, as the near
//   meadow does), and the rest of it only where a streak may show. Evaluated in the composite
//   instead, along the render texels' unjittered rays and read bilinearly, it cost as much at equal
//   sizes and, below the canvas, lost its faint streaks against the ground truth (ghosting 1.5 x
//   MSAA's, from 0.2 x).
// - Grade: while it rains or the overlay holds something, the canvas pass adds the change the rain
//   makes to the graded colour after RCAS and before the dither (TemporalAA.ts `layer`). At the
//   render texel nearest the point the pixel shows, the display pass's grade input (the sharp image,
//   taken once a texel by a pass at the render's size, `rainInput`: per canvas pixel it took a fifth
//   of the layer's time) has the veil, then the overlay, laid over it in scene radiance, as they were
//   before the grade, and that goes through the display pass's grade (exposure, grade, curve, in
//   linear light); the change is its difference from what the display pass graded at the texel. The
//   change is added to the resolved image decoded to linear light (the sRGB transfer undone, on
//   changed pixels only), which is encoded again; the vignette applies to it as to the image. It is taken at the
//   rain's coverage of a single sample (PipelineDisplay.ts AFTER_COVERAGE) and spread over the
//   pixel, as a supersampled image grades each sample before it averages them. Laid over the graded
//   image instead, faint rain over the dark sea and meadow, in the curve's toe, came out brighter
//   and harder (the rings on the sea most); laid over it before the grade at its full coverage,
//   thin streaks came out fainter than the ground truth's. The rain's bloom is left out.
// - Where: a running frame's canvas pass draws the change itself, in a build of its own used only
//   while the layer shows (built into the one canvas pass for every frame, the rain's registers
//   slowed it in fair weather too; drawn by a pass of its own, it cost a canvas-size target written
//   and read), and grades by a table (PipelineDisplay.ts `gradeTable`, baked each frame from the
//   grade itself: the change is the difference of two readings, so the table's error cancels to
//   first order; in a stepped rain clip a few hundredths of a level from the full grade's on
//   average, at most one in 99.9 % of the pixels). The full grade at every canvas pixel was a third
//   of the layer's time.
//   The frame that completes the rest draws the change with the full grade into `rainLayer`, which
//   the canvas pass reads as it is while the rest holds (below), as TAA's upscaled canvas pass does.
// - At rest (TemporalRest.ts): the passes draw over the frame's own jitter and jittered depth, and
//   once the rest has converged the jitter runs on through the running sequence, whose place
//   depends on the frames rendered. So while the rest holds its image neither pass runs: the
//   overlay and the layer the completing frame drew (at the rest sequence's last jitter) stay, and a
//   paused frame's rain repeats however it was reached (Pipeline.ts `afterHeld`).
// - Nothing of the rain stays before the resolve: the overlay pass does not run, no rain or spray
//   writes a velocity or a reactive weight, and there is no `veilReactive`. The splashes stayed there
//   at first, on the surfaces they lie on, with their reactive weight: a glint of a few frames, they
//   still lost most of themselves to the history (the storm's remaining ghosting, against the
//   ground truth, was theirs).

type F = THREE.Node<'float'>;
type IV2 = THREE.Node<'ivec2'>;

/** The overlay's resource name (the MRT finds its output by it). */
export const RAIN_OVERLAY = 'rainOverlay';

/** The after pass's MRT: the overlay, cleared to zero, blended as each material says, pre-exposed
 * as hdr is (PreExposure.ts), whose units the rain layer adds it in. */
export const afterMRT: THREE.MRTNode = mrt( { [ RAIN_OVERLAY ]: preExposedRGBA( output ) } )
	.setBlendMode( RAIN_OVERLAY, new THREE.BlendMode( THREE.MaterialBlending ) )
	.setClearColor( RAIN_OVERLAY, new THREE.Color( 0, 0, 0 ), 0 );

/** The four render texels around the point a canvas pixel shows, and their bilinear weights: the
 * pixel's centre scaled to the render (`renderSize`, over `outputSize`), less this frame's jitter
 * (a texel's centre shows the unjittered point a jitter's offset from it). */
export function afterTexels( renderSize: THREE.Node<'vec2'>, outputSize: THREE.Node<'vec2'> = screenSize ): { texel: IV2; weight: F }[] {

	const t = screenCoordinate.xy.mul( renderSize.div( outputSize ) ).sub( jitter.uniforms.px ).sub( 0.5 ).toVar();
	const base = floor( t ).toVar();
	const f = t.sub( base ).toVar();
	const last = renderSize.sub( 1 );
	return [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ].map( ( [ x, y ] ) => ( {
		texel: ivec2( clamp( base.add( vec2( x, y ) ), vec2( 0 ), last ) ),
		weight: ( x ? f.x : float( 1 ).sub( f.x ) ).mul( y ? f.y : float( 1 ).sub( f.y ) ) as F,
	} ) );

}

/** In a material of the after pass: how much of the fragment shows, from the scene's nearest
 * surface (`surfaceDepth`, linear view z at the render's size). `visible` tests one render texel
 * (its coordinate and its nearest surface's view z); the four tests are weighted bilinearly. */
export function afterResolveCover( surfaceDepth: THREE.Texture, visible: ( texel: IV2, surfaceZ: F ) => F ): F {

	const depth = texture( surfaceDepth );
	const size = vec2( textureSize( depth, int( 0 ) ) as unknown as IV2 );
	let cover: F = float( 0 );
	for ( const { texel, weight } of afterTexels( size ) ) cover = cover.add( visible( texel, textureLoad( depth, texel ) as unknown as F ).mul( weight ) ) as F;
	return max( cover, 0 );

}

