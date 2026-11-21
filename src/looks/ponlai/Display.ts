import type { DayKey } from '../../render/index.ts';
import type { BloomKey, DesaturationKey, DisplayLook, Rgb } from '../../render/setup.ts';

// The island's display look: contrast, grade, bloom, vignette, white balance and tone curve.
//
// The island's display look (packages/island/look/: the island's numbers for the render pipeline's
// look tables, look/Look.ts; look/DisplayLook.ts makes the display's tables from these): the
// shadow contrast, the desaturation, overcast's highlights toward grey, the bloom's and the
// vignette's strengths, the vignette's shape, the white balance, the tone curve's three settings,
// the base grade, the bloom's keyframes and the rods' colour at night. The methods that read them are
// their modules' (post/ToneCurve.ts, post/GradeLut.ts, post/PhysicalBloom.ts, post/Purkinje.ts,
// post/PipelineDisplay.ts).

const DESATURATION_DAY: DayKey<DesaturationKey>[] = [ { at: 4, name: 'golden', values: { amount: - 0.163, cool: 0 } }, { at: 9, name: 'day', values: { amount: 0.0122, cool: 0 } } ];
const WHITE_FROM_DAY: DayKey<Rgb>[] = [ { at: 9, name: 'day', values: { r: 0.5, g: 1, b: 1.36 } }, { at: 40, name: 'high', values: { r: 0.593, g: 1, b: 1.248 } } ];

// the levels' weights from the finest to the widest, scaled so their sum is `strength`
const weights = ( strength: number, w: number[] ) => {

	const sum = w.reduce( ( a, b ) => a + b );
	const [ w0, w1, w2, w3, w4, w5, w6, w7 ] = w.map( ( v ) => Math.round( v / sum * strength * 1e4 ) / 1e4 );
	return { w0, w1, w2, w3, w4, w5, w6, w7 };

};
// The bloom by the sun's elevation, under a clouded sky and at night (DayKeys.ts), on the exposed
// image (mid grey 0.18): the land's threshold T (where its glow starts), knee K (the knee's half
// width), share k and cap, the sky's (`sky...`), and the levels' weights from the finest (w0, half
// the render) to the widest (w7, 1/256), whose sum is the strength. Tunable on debug pages
// (debug/Tune.ts `post.bloom.<value>@<key>`, the keys `low`, `high`, `cloudy`, `night`); a tuned
// sky threshold leaves its knee where the table puts it.
const BLOOM_HIGH_SUN: BloomKey = {
	threshold: 1.76, knee: 0.66, share: 0.2, cap: 3.89,
	skyThreshold: 18.3, skyKnee: 0.4 * 18.3, skyShare: 0.17, skyCap: 2250,
	w0: 0, w1: 0.38, w2: 0.74, w3: 0.185, w4: 0.395, w5: 0.371, w6: 0.0944, w7: 0.13,
};

export const DISPLAY: DisplayLook = {

	// The shadow contrast: under a pivot each stop of the exposed luminance (the local tone's gain in
	// it) darkens by `slope` of a stop more (a negative slope lifts it), over it nothing changes;
	// after the local tone, before the tone curve. The pivot (stops over mid grey) and slope are set
	// per keyframe (keyed by the sun's elevation, DayKeys.ts). By day it is light, its pivot 2 stops over
	// mid grey and its slope 0.05: on the island view a surface 1, 2, 3 and 4 stops under the sunlit
	// meadow shows at Oklab L 0.46, 0.32, 0.26 and 0.23 (0.56 for the meadow itself). A day slope of
	// 0.45 over a pivot 1.6 stops over mid grey took nearly the whole frame under its pivot, the lit
	// ground too, and showed those stops at 0.39, 0.24, 0.18 and 0.16: the shadowed areas, the shaded grass
	// and the rock under the sea read near black and lost half their chroma. The golden hour keeps
	// most of the slope with its pivot nearly a stop under mid grey, so only what lies in the long
	// shadows drops, and the sunlit cliff stays as it is. The clouded sky deepens its darks too, from
	// about 0.9 of a stop over mid grey. The night's slope is negative, a lift: under a pivot 2 stops
	// below mid grey, below the moonlit grass, each stop darker is lifted by a quarter of a stop. The
	// moon's shadows on the island are mostly narrow, the culms', under the local tone's scale, so the
	// local tone cannot set them apart from the moonlit land; this lift does it per pixel, the grass in
	// the moon's shadow about half a stop nearer the moonlit grass while that grass and the sea move by
	// an eighth of a stop or less.
	// Tunable on debug pages (debug/Tune.ts): `post.shadowContrast.<value>@golden`, `@day`, `@cloudy`, `@night`.
	shadowContrast: {
		day: [ { at: 4, name: 'golden', values: { pivot: - 0.86, slope: 0.37 } }, { at: 9, name: 'day', values: { pivot: 2, slope: 0.05 } } ],
		cloudy: { pivot: 0.902, slope: 0.18 },
		night: { pivot: - 2, slope: - 0.25 },
	},

	// The desaturation before the local tone's gain: each colour drawn toward its own luminance by
	// `amount` (0 keeps it, 1 grey), c + ( Y( c ) - c ) amount, keyed by the sun and the weather
	// (DayKeys.ts); below 0 it saturates. Keyed as the white balance is, so each key sets its own: the
	// golden hour saturates by a sixth, the day keeps its colours, the clouded sky fades them by a
	// twelfth, the night by about a half. The rising sun's `dawn` key, held from -1 degree down,
	// fades the cool colours by 0.6 and passes into the golden key's by 1.5 (the golden key's own
	// values, shared): before the sun the clear morning's land, sea and sky took the dawn's white
	// (`dawn`, `whiteBalance`) on top of their own blue, and the upper sky and the far sea turned violet
	// (h 271 to 278 from -4 to 0 degrees). A fade of every colour brought them back to h 250 to 262,
	// but it took the warm colours with them: the rose band over the horizon (h 327 to 353, C 0.03 to
	// 0.05 from -8 to -1 degrees) went to a grey-blue band (h 263, the white's cast), and the meadow's
	// green, under it, to h 231 to 241. So the key's `cool` share of 1 fades only the colours that lie
	// toward blue once the white is applied (`coolShare`: blue over red as a share of the colour's
	// spread, 0 none to 0.6 all), and the warm band and the meadow keep their own colour under the same
	// white: the sky's top stays at h 248 to 257, the band at h 327 to 353, the meadow at h 214 to 226
	// from -7 to -4.5 degrees.
	// Tunable on debug pages (debug/Tune.ts): `post.desaturate.<amount|cool>@golden`, `@dawn`, `@day`,
	// `@cloudy`, `@night`; the cool share's edges `post.desaturate.coolFrom`, `post.desaturate.coolTo`.
	// amount: the fade (0 keeps the colour, 1 grey, below 0 saturates); cool: the share of the fade
	// that only cool colours take (`coolShare`)
	desaturation: {
		day: DESATURATION_DAY,
		rising: [ { at: - 1, name: 'dawn', values: { amount: 0.6, cool: 1 } }, { at: 1.5, values: DESATURATION_DAY[ 0 ].values }, ...DESATURATION_DAY ],
		cloudy: { amount: 0.08, cool: 0 },
		night: { amount: 0.481, cool: 0 },
	},
	// The edges of the desaturation's cool share (`coolShare`, post/PipelineDisplay.ts): its blue less its
	// red as a share of its spread, from `from` (none of it cool) to `to` (all of it).
	// Tunable on debug pages (debug/Tune.ts): `post.desaturate.coolFrom`, `post.desaturate.coolTo`.
	coolShare: { from: 0, to: 0.6 },
	// Overcast's highlights toward grey: the grade's tables' linear output c becomes
	// c + ( Y( c ) - c ) amount smoothstep( from, to, Y( c ) ) share, `share` overcast's share in the sky
	// times one less the night's weight (as its white, `whiteOvercast`: none at night, and nothing at all
	// where the sky holds no overcast). Overcast's white, which turns the deck away from the sun
	// blue-green grey and the meadow green, turned the bright sky about the sun cyan too (OKLab a -0.033
	// at a lightness of 0.74, 10 to 30 degrees from it), and a warmer white turns the deck and the
	// meadow back. The deck lies under `from` (a display lightness of about 0.35, the deck about 0.32)
	// and keeps its colour; the meadow straddles it: its hue stays, its median chroma falls by about a
	// fifth (0.083 to 0.065), and the lit grass above a lightness of 0.55 keeps under half its chroma.
	// `from` and `to` are linear luminances of the tables' output (display lightness about 0.35 and
	// 0.70); amount 0.58 brings the sky 10 to 30 degrees from the sun to a -0.014 and 30 to 60 degrees
	// to about a -0.024 (-0.024 at a sun of 22 degrees, -0.025 at 15), and the glow within 10 degrees
	// of it to a chroma under 0.01.
	// Tunable on debug pages (debug/Tune.ts): `post.highlight.amount`, `from`, `to`.
	highlightGrey: { amount: 0.58, from: 0.067, to: 0.4 },

	// The bloom's strength (times the time's and the weather's, and its share is the threshold's:
	// `bloom`) and the vignette's strength (`vignette`). No grain: a dither of half a
	// code value is all the noise the frame takes.
	strengths: { bloom: 1.36, vignette: 1 },
	// The vignette's shape: with r the distance from the frame's centre in its own uv (round in uv,
	// not in pixels), the factor mix( 1, 1 - depth smoothstep( inner, outer, r ), strength ), a radial
	// smoothstep. Nothing changes out to the middle of each edge (r = 0.5); the smoothstep leaves 1
	// with no slope there, so the darkening starts with no visible edge, and at strength 1 the corners
	// (r = 0.71) end at about 0.73, about 0.45 stops (the falloff spans to r = 1, past the corners, so
	// they sit on its steepening first half: 0.98 at r = 0.55, 0.92 at 0.6, 0.84 at 0.65).
	vignette: { inner: 0.5, outer: 1, depth: 0.73 },
	// The white balance, as the colour white becomes (linear Rec.709), by the sun's elevation, under a
	// clouded sky and at night (DayKeys.ts), set with the light. A clear day pushes the display past
	// neutral toward cool, the sun's warmth on the lit ground held back: furthest at a sun of 9 degrees
	// (red half of green, blue 1.36), which keeps the low sun's rock and meadow, and less by 40 degrees
	// (red 0.59, blue 1.25, the `high` key), where the stronger cast would turn sunlit rock blue. The
	// high key moves with the sky's (look/SkyLook.ts `skyLight`), which takes the inverse, so the sky holds its
	// colour. The golden hour warms a little further (red 1.05, blue 0.9); the clouded skies take a cool
	// white too (red 0.78, blue 1.16), their grey light kept from reading warm. Overcast takes a cooler
	// white of its own (red 0.49, blue 1.14, `whiteOvercast`): under the clouded key its slightly warm
	// grey deck read as a neutral grey and the meadow under it as yellow-green (hue 120); its own key
	// brings the deck to a blue-green grey (OKLab a -0.022, b -0.008) and the meadow to green (hue
	// 132), the lightness all but unchanged. It comes in over the clouded day's white by overcast's share in
	// the sky, before the night's, so an overcast night takes the night's white as any other.
	// The night takes a cool white too (red 0.53, blue 1.2), on top of the moon's light and the rods'
	// (Purkinje.ts).
	// The rising sun takes a cold white where the setting sun takes the warm one (`dawn`, in place of
	// the golden key while the sun climbs: DayKeys.ts `rising`; red 0.4, blue 2, tuned on our own
	// views): all that the sun does not reach at dawn is blue, sky, water and rock alike, and a face turned
	// to the sun takes a pale warm colour, far less orange than the same face at the same height in
	// the evening. It is what moves the rose band over the horizon, which the sky's own light
	// (look/SkyLook.ts `skyLight`, its `dawn` key) cannot reach. The sides are one from the `day` key up.
	// Tunable on debug pages (debug/Tune.ts): `post.white.<r|g|b>@golden`, `@dawn`, `@day`, `@high`,
	// `@cloudy`, `@overcast`, `@night`.
	whiteBalance: {
		day: [ { at: 4, name: 'golden', values: { r: 1.05, g: 1, b: 0.9 } }, ...WHITE_FROM_DAY ],
		rising: [ { at: 4, name: 'dawn', values: { r: 0.4, g: 1, b: 2.0 } }, ...WHITE_FROM_DAY ],
		cloudy: { r: 0.775, g: 1, b: 1.16 },
		night: { r: 0.527, g: 1, b: 1.2 },
	},
	// overcast's own white (see the white balance above), over the clouded day's by overcast's share
	whiteOvercast: { r: 0.49, g: 1, b: 1.14 },

	// The display's tone curve (post/ToneCurve.ts solves its coefficients from these three): the slope
	// at black, where mid grey lands and the white it reaches 1 at. The pipeline's settings take a copy
	// (post/Pipeline.ts `settings.curve`, tunable there).
	toneCurve: { toe: 2.46, mid: 0.168, white: 15.1 },

	// The base grade (post/GradeLut.ts builds its tables from these changes over no grade): the day's, which the other grades build
	// on. No contrast and no saturation over the whole frame (the contrast is the shadow
	// contrast's, `shadowContrast`, and the tone curve's): each hue moved on its own, lightly, the neutral axis left
	// where it is but for a touch of cool in the highlights.
	// - The warm hues (red to yellow) times 1.15: the earth, the bark and the low sun's light keep
	//   their colour under the softer curve; they also turn 8 degrees toward red, so the golden hour's
	//   light reads orange where the curve's shoulder turns it yellow (a degree or so on the day's
	//   rock and meadow). The `flowers` band (the red the parked flowers had; the meadow draws
	//   none now) times 1.15 over that, which the cooler white balance and the softer curve had cut by
	//   about a fifth; it also turns red-orange 4 degrees toward red.
	// - The sky's blue times 1.2: the noon sky read pale and grey over the brighter curve.
	// - The green times 0.95 and turned 3.5 degrees toward yellow: the cool noon white balance
	//   (`whiteBalance`) already takes the sunlit meadow from yellow-green to
	//   green, and a turn toward blue-green over it took the lit blades past it (on screen, in linear
	//   light, their mean red to green about 0.53 to 0.60 at 4 degrees toward blue-green, 0.61 to 0.66
	//   at this turn, from 0.74 to 0.81 under a neutral white), and a touch less loud than the sky over
	//   it.
	// - The sea's blue-green times 1.05 and turned 7 degrees toward blue; purple turned 6 degrees
	//   toward red.
	// - The highlights' white turned a touch cool (an Oklab chroma of about 0.008 at a hue of 260).
	baseGrade: {
		highlights: { tint: [ 0.97, 1, 1.05 ] },
		bands: {
			grass: { saturation: 0.95, shift: - 3.5 }, sky: { saturation: 1.2 }, sea: { saturation: 1.05, shift: 7 },
			flowers: { saturation: 1.15, shift: - 4 }, warm: { saturation: 1.15, shift: - 8 }, purple: { shift: - 6 },
		},
	},

	// The bloom's keyframes (post/PhysicalBloom.ts, whose header gives the method they set).
	// The clear high sun's key (from 9 degrees up) is set with the day's exposure and grade, its
	// levels' weights on the halo about the sun out to some 50 degrees. The bloom stands for the
	// lens's scatter, and the lens does not change with the sky: on the exposed image the same light
	// glows the same, so a clouded sky takes the high sun's key whole (under cloud the land and the sky
	// stay under its thresholds, and what glows glows as on a clear day: the sea's glints, a sun seen
	// through a gap), and every key's sky cap is the high sun's. By a high sun the land's threshold
	// sits over three stops above mid grey, so the lit ground stays out; in the golden hour and by
	// night near mid grey, so a low sun's whites glow, with a wider knee than the day's in the golden
	// hour and about the day's by night. By night the land's share is small by choice (0.05, falling
	// from the low sun's 0.2 as the night comes in): at a threshold under the moon's glints a large
	// share spread them over the wide levels into a pale pedestal under the whole field, and the water
	// between the glints stood lit. The sky keeps its own share, so the moon's disc and the bright
	// clouds glow as before. The weights carry the strength: about 2.3 by a high sun; 1.25 in the
	// golden hour and by night, whose thresholds let far more in, toward the wide levels (a broad
	// soft glow about the moon and bright clouds). The high sun's weights rise and fall from level to
	// level, less at 1/16 and 1/128 than at the levels either side, which sets the halo's outer rings
	// (all positive, so the summed kernel still falls off from its centre). The finest level takes
	// none: its spread, a pixel or two, is the rendered image's own sharpness. The sky's share is a little
	// smaller by a clear high sun (0.17) than by a low sun (0.2), so the glow round the sun stays
	// close to it rather than spreading over the sky far from it.
	bloom: {
		day: [
			{ at: 4, name: 'low', values: {
				threshold: 0.2, knee: 1.1, share: 0.2, cap: 2,
				skyThreshold: 6, skyKnee: 0.4 * 6, skyShare: 0.2, skyCap: BLOOM_HIGH_SUN.skyCap,
				...weights( 1.25, [ 0, 0.17, 0.44, 0.54, 0.64, 0.74, 0.88, 0.58 ] ),
			} },
			{ at: 9, name: 'high', values: BLOOM_HIGH_SUN },
		],
		cloudy: BLOOM_HIGH_SUN,
		night: {
			threshold: 0.18, knee: 0.6, share: 0.05, cap: 1,
			skyThreshold: 1, skyKnee: 0.4, skyShare: 0.5, skyCap: BLOOM_HIGH_SUN.skyCap,
			...weights( 1.25, [ 0, 0.13, 0.34, 0.44, 0.58, 0.78, 0.94, 0.58 ] ),
		},
	},
	bloomStart: BLOOM_HIGH_SUN,

	// The rods' colour at night (post/Purkinje.ts, whose header gives the shape `retina` sets): the
	// retinal illuminance (trolands) white on open level ground under the night's light is taken
	// to, and the strength the change is taken back by at night. The darker a colour is against lit
	// white, the more the rods lift it and the cooler it turns, so the darkest land turns cooler than
	// the lit meadow. Tunable on debug pages (debug/Tune.ts): `post.purkinje.retina@night`,
	// `post.purkinje.strength@night`.
	purkinje: { retina: 250, strength: 0.13 },

};
