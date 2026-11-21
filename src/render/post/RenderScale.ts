import type { RenderScaleRule } from '../Quality.ts';

// Render scale per quality tier: the scene's pixel ratio, upscaled to the canvas.
//
// Render scale per quality tier (`QualityProfile.render` in Quality.ts). The scene renders at a
// pixel ratio of its own and the frame graph upscales the graded image to the canvas, which always
// matches the display: a smaller canvas stretched by the browser (bilinear) lost a third of the
// meadow's, cliff's and ivy's fine detail at 1.5 DPR.
//
// Pixel ratios of the scene's render, relative to CSS pixels, never above the display's or the
// cap (ocean optics, clouds and the post passes scale with the pixel count):
// - high: dynamic, steered by the GPU's frame time (DynamicResolution.ts), between a floor and the
//   display's own ratio (up to OUTPUT_PIXEL_RATIO_CAP). Its baseline and floor are counts of
//   rendered pixels, not shares of the display's ratio, so that a frame costs about the same on any
//   display: the standard, 4K output (1920x1080 CSS at 2x), renders 1440p (scale 4/3, FSR 1's
//   Quality 1.5x upscale), drops to 1080p and climbs to native 4K with headroom, and a 2K display
//   (2560x1440 at 1x) renders natively and drops to about 1080p when busy. A load starts at,
//   and a paused or pinned page holds, the baseline, capped by the maximum (a display of 2K or fewer
//   pixels renders natively there). The floor is at most `floorShare` of the maximum per axis, so a
//   small display's floor stays below its maximum. `dynamicRange` gives the three; this returns the
//   baseline as the nominal;
// - medium and low (parked, Quality.ts): a budget of rendered pixels (1600x900 and 1280x720 today);
//   `?dpr` and the GUI's full resolution lift it.
// The canvas always matches the display (up to OUTPUT_PIXEL_RATIO_CAP): a render below it is
// upscaled inside the frame graph with FSR 1 and the dither added after, never stretched by the
// browser. Every pass is sized to the render, not the canvas (the frame graph sizes each target by
// its size class). At the display's ratio nothing is upscaled, so `dpr=1` captures are unchanged. The
// tiers never save through multisampling: every one draws the scene at 4x (Quality.ts).

// The canvas follows the display up to this ratio. Past it only the upscale and grade passes
// would grow (a phone's 3x), and the browser's stretch of a 2x canvas is no longer visible.
export const OUTPUT_PIXEL_RATIO_CAP = 2;

/** The scene's nominal render pixel ratio for a `width` x `height` CSS-pixel view: the tier's
 * budget, or a dynamic tier's baseline. `full` (an explicit `?dpr` or the GUI's full-resolution
 * control) lifts the budget and pins the ratio at the display's (or `?dpr`'s). */
export function renderPixelRatio( width: number, height: number, rule: RenderScaleRule, full: boolean, cap: number ): number {

	const requested = Math.min( devicePixelRatio, cap );
	if ( full ) return requested;
	if ( 'dynamic' in rule ) return dynamicRange( width, height, devicePixelRatio, rule, cap ).baseline;
	return Math.min( requested, Math.sqrt( rule.budget / Math.max( width * height, 1 ) ) );

}

/** A dynamic tier's range of render pixel ratios for a `width` x `height` CSS-pixel view. */
export interface DynamicRange {
	// the display's `ratio`, up to the cap and the tier's `max`
	max: number;
	// the ratio that renders the tier's `baseline` pixels, within the floor and the maximum
	baseline: number;
	// the ratio that renders the tier's `floor` pixels, at most `floorShare` of the maximum
	floor: number;
}

/** A dynamic tier's range on a display of pixel ratio `ratio` (pure: the tools compute the pin of
 * the high tier's performance target with it). */
export function dynamicRange( width: number, height: number, ratio: number, rule: Extract<RenderScaleRule, { dynamic: unknown }>, cap: number ): DynamicRange {

	const { dynamic } = rule;
	const area = Math.max( width * height, 1 );
	const max = Math.min( ratio, cap, dynamic.max );
	const floor = Math.min( Math.sqrt( dynamic.floor / area ), dynamic.floorShare * max );
	const baseline = Math.min( Math.max( Math.sqrt( dynamic.baseline / area ), floor ), max );
	return { max, baseline, floor };

}

/** The canvas's pixel ratio: the display's, and never below the render's (an explicit `?dpr`
 * above the cap renders, and shows, at that ratio unscaled). */
export function outputPixelRatio( render: number ): number {

	return Math.max( Math.min( devicePixelRatio, OUTPUT_PIXEL_RATIO_CAP ), render );

}
