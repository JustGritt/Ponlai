import type { DayKeys } from '../post/DayKeys.ts';
import { tuneKeys, tuneObject } from '../debug/Tune.ts';
import type { PipelineSettings } from '../post/Pipeline.ts';
import { gradeOf, type GradeParams } from '../post/GradeLut.ts';
import type { ToneCurve } from '../post/ToneCurve.ts';
import type { BloomKey, DesaturationKey, DisplayLook, Rgb } from './Look.ts';

// The display's tables built from a look's table: tone curve, grade, bloom and vignette.
//
// The display's tables built from a look's table (look/Look.ts `DisplayLook`): the shadows'
// contrast, the desaturation, overcast's highlights toward grey, the bloom's and the vignette's
// strengths, the vignette's shape, the white balance, the tone curve's three settings, the base
// grade, the bloom's keys and the rods' colour at night (`createDisplayTables`). Like the exposure
// (look/ExposureLook.ts), they are taken as an argument where the render pipeline's stage creates
// them (RenderStages.ts `createSeaAndPipeline`, which hands them to the frame's passes:
// post/Pipeline.ts `look`), so two made from two tables hold their own numbers
// (tools/render/verify-look-params.mjs) and this module holds none at module level.
// post/PipelineDisplay.ts builds the display chain's shader from them, post/Pipeline.ts
// `updateDisplay` keys them by the sun and the weather each frame, and the methods that read them
// are their modules' (post/ToneCurve.ts, post/GradeLut.ts, post/PhysicalBloom.ts,
// post/Purkinje.ts).
// - The tables register their tunables (debug/Tune.ts) as they are created, under their names
//   (`post.shadowContrast.*`, `post.desaturate.*`, `post.white.*`, `post.bloom.*`,
//   `post.purkinje.<retina|strength>@night`); the cool share's edges and overcast's highlights
//   register where the display chain builds the uniforms that hold them (post/PipelineDisplay.ts),
//   and the tone curve and the two strengths on the pipeline's settings (post/Pipeline.ts
//   `settings`). The tune table is one per process: on a debug page a second set made from another
//   table reads the first one's tunables.

export type { BloomKey };

/** The display's tables, made from a look's (`createDisplayTables`). */
export interface DisplayTables {
	// The shadows' contrast: under a pivot each stop of the exposed luminance (the local tone's gain
	// in it) darkens by `slope` of a stop more (a negative slope lifts it instead), over it nothing
	// changes; after the local tone, before the tone curve. The pivot (stops over mid grey) and slope
	// are set per key (keyed by the sun's elevation, DayKeys.ts).
	// Tunable on debug pages (debug/Tune.ts): `post.shadowContrast.<value>@<key>`.
	readonly shadowContrast: DayKeys<{ pivot: number; slope: number }>;
	// The desaturation before the local tone's gain: each colour drawn toward its own luminance by
	// `amount` (0 keeps it, 1 grey), c + ( Y( c ) - c ) amount, keyed by the sun and the weather
	// (DayKeys.ts); below 0 it saturates. `cool` is the share of the fade that only cool colours take
	// (`coolShare`, post/PipelineDisplay.ts).
	// Tunable on debug pages (debug/Tune.ts): `post.desaturate.<amount|cool>@<key>`.
	readonly desaturation: DayKeys<DesaturationKey>;
	// The edges of the desaturation's cool share (`coolShare`, post/PipelineDisplay.ts): its blue less
	// its red as a share of its spread, from `from` (none of it cool) to `to` (all of it).
	// Tunable on debug pages (debug/Tune.ts): `post.desaturate.coolFrom`, `post.desaturate.coolTo`.
	readonly coolShare: { from: number; to: number };
	// Overcast's highlights toward grey: the grade's tables' linear output c becomes
	// c + ( Y( c ) - c ) amount smoothstep( from, to, Y( c ) ) share, `share` overcast's share in the
	// sky times one less the night's weight (as its white, `whiteOvercast`: none at night, and
	// nothing at all where the sky holds no overcast). `from` and `to` are linear luminances of the
	// tables' output.
	// Tunable on debug pages (debug/Tune.ts): `post.highlight.amount`, `from`, `to`.
	readonly highlightGrey: { amount: number; from: number; to: number };
	/** The bloom's strength (times the time's and the weather's, and its share is the threshold's:
	 * `bloom`) and the vignette's strength (`vignette`). No grain: a dither of half a code value is
	 * all the noise the frame takes. */
	readonly strengths: Readonly<Pick<PipelineSettings, 'bloom' | 'vignette'>>;
	// The vignette's shape: with r the distance from the frame's centre in its own uv (round in uv,
	// not in pixels), the factor mix( 1, 1 - depth smoothstep( inner, outer, r ), strength ), a
	// radial smoothstep.
	readonly vignette: { inner: number; outer: number; depth: number };
	// The white balance, as the colour white becomes (linear Rec.709), by the sun's elevation, under
	// a clouded sky and at night (DayKeys.ts), set with the light. Overcast takes a white of its own
	// (`whiteOvercast`), which comes in over the clouded day's white by overcast's share in the sky,
	// before the night's, so an overcast night takes the night's white as any other.
	// Tunable on debug pages (debug/Tune.ts): `post.white.<r|g|b>@<key>`, overcast's `@overcast`.
	readonly whiteBalance: DayKeys<Rgb>;
	readonly whiteOvercast: Rgb;
	// The display's tone curve (post/ToneCurve.ts solves its coefficients from these three): the
	// slope at black, where mid grey lands and the white it reaches 1 at. The pipeline's settings
	// take a copy (post/Pipeline.ts `settings.curve`, tunable there).
	readonly toneCurve: Readonly<ToneCurve>;
	/** The base grade (post/GradeLut.ts builds its tables): the day's, which the other grades build
	 * on, a look's changes over no grade. */
	readonly baseGrade: Readonly<GradeParams>;
	/** The key the bloom's uniforms start from (post/PhysicalBloom.ts). */
	readonly bloomStart: BloomKey;
	/** The bloom by the sun's elevation, under a clouded sky and at night (DayKeys.ts), on the
	 * exposed image (mid grey 0.18): the land's threshold T (where its glow starts), knee K (the
	 * knee's half width), share k and cap, the sky's (`sky...`), and the levels' weights from the
	 * finest (w0, half the render) to the widest (w7, 1/256), whose sum is the strength. Tunable on
	 * debug pages (debug/Tune.ts `post.bloom.<value>@<key>`); a tuned sky threshold leaves its knee
	 * where the table puts it. */
	readonly bloom: DayKeys<BloomKey>;
	/** The rods' colour at night (post/Purkinje.ts): the retinal illuminance (trolands) white on open
	 * level ground under the night's light is taken to, which sets where the rods' lift starts, and
	 * the strength the change is taken back by at night. Read every frame. Tunable on debug pages
	 * (debug/Tune.ts): `post.purkinje.retina@night`, `post.purkinje.strength@night`. */
	readonly purkinje: { retina: number; strength: number };
}

/** The display's tables from `display`, a look's (see the header): its tunables register here. */
export function createDisplayTables( display: DisplayLook ): DisplayTables {

	const shadowContrast = tuneKeys( 'post.shadowContrast', display.shadowContrast, { pivot: { min: - 3, max: 4 }, slope: { min: - 0.3, max: 0.5 } } );
	const desaturation = tuneKeys( 'post.desaturate', display.desaturation, { amount: { min: - 0.5, max: 1 }, cool: { min: 0, max: 1 } } );
	const whiteRange = { r: { min: 0.2, max: 1.5 }, g: { min: 0.5, max: 1.5 }, b: { min: 0.5, max: 3 } };
	const whiteBalance = tuneKeys( 'post.white', display.whiteBalance, whiteRange );
	const whiteOvercast = tuneObject( { ...display.whiteOvercast },
		{ r: 'post.white.r@overcast', g: 'post.white.g@overcast', b: 'post.white.b@overcast' }, whiteRange );
	const bloom = tuneKeys( 'post.bloom', display.bloom, { skyThreshold: { min: 3, max: 40, scale: 'log' }, skyCap: { min: 100, max: 3000, scale: 'log' } } );
	const purkinje = tuneObject( { ...display.purkinje }, { retina: 'post.purkinje.retina@night', strength: 'post.purkinje.strength@night' },
		{ retina: { min: 1, max: 2000, scale: 'log' }, strength: { min: 0, max: 0.4 } } );
	return {
		shadowContrast, desaturation,
		coolShare: { ...display.coolShare },
		highlightGrey: { ...display.highlightGrey },
		strengths: display.strengths,
		vignette: { ...display.vignette },
		whiteBalance, whiteOvercast,
		toneCurve: display.toneCurve,
		baseGrade: gradeOf( display.baseGrade ),
		bloomStart: display.bloomStart,
		bloom, purkinje,
	};

}
