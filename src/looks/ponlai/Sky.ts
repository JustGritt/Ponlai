import type { DayKey } from '../../render/index.ts';
import type { LandSkyRow, Rgb, SkyLook } from '../../render/setup.ts';

// The island's sky look: its numbers for the sky's tables.
//
// The island's sky look (packages/island/look/: the island's numbers for the render pipeline's look
// tables, look/Look.ts; look/SkyLook.ts makes the sky's tables from these, and its header says what
// each table is). Upper-case names are this file's; a name in backticks with no file is
// sky/Sky.ts's or a field of the sky's tables (look/SkyLook.ts `SkyTables`).

// Art direction: the island stays readable after sunset. Through the minutes after it the sun's
// skylight has faded while the moon's key and the night ambient are not yet in, which left the
// meadow at about half its night brightness. The light the sky lends the island takes a gain per
// sun elevation, and a cast about the sunset and in the night, ISLAND_SKY: the environment's
// source panorama, which lights the terrain and plants, in its bake (its haze without the fog
// layer: `fogEnvironment`), and the islanders' sky fill
// (`landIrradianceNode`, the irradiance probe graded the same way). The seen sky, the panorama
// the sea reflects and the irradiance probe itself, which lights the sea's body, keep the
// atmosphere's own, so the evening's sky and sea look as they did. Each row is an elevation
// (degrees, descending): the colour the light is drawn toward (linear, at the light's own
// luminance), how far, and the gain; rows interpolate linearly and the end rows hold beyond them.
// From 2 degrees up the light is the atmosphere's alone, and from -13.2 down it is the moonlit
// sky's own again, with no cast: the night's blue on the display is the eye's (post/Purkinje.ts),
// given once, so the land lit by this sky takes the sky's colour and no deeper blue of its own
// (drawn 0.6 of the way to a deep blue, the land was blued by the key, by this sky and by the eye,
// all in the sky's own hue, and read as one blue with the sky over it). The gains set the
// island's exposed brightness (the top view's land, with the moon's key; the metering illuminance
// holds this light too) through the twilight.
// A clear evening keeps its day exposure on the golden hour's key until the night's comes in from
// -7 degrees (post/Exposure.ts, DayCycle.ts `displayNight`), and the land keeps the probes' golden
// gain on the sky's light (lighting/AmbientRelight.ts PROBE_SKY), both held from the golden hour
// down; so past a lift of about 1.2 through the sunset itself, the light lent from -1.3 degrees
// down stands a little under the sky's own (gains of 0.85 to 0.96 down to -5.3 degrees), set on
// the meadow on the plateau in the island view, which reads about L 0.2 at -3.3 and -5.3 degrees.
// The rows at -1.3, -2.6 and -4 take a scale near their neighbours', the row at -8.6 keeps the
// 0.76 it was set with, and the gains return to none as the night's exposure comes in (by -13.2).
// The rows at 1, 0.5 and 0 degrees take the colour the rows around them gave there. Below the
// horizon the setting side's rows hold gains only, so the land takes the colour of the sky's own
// light: the dusk key (SKY_LIGHT) warms it (its irradiance red over green about 1.6 at -3 degrees
// and 1.45 at -5, blue over green 0.7 and 0.8), and the afterglow's air over the land keeps the
// glow's colour (Haze.ts AFTER_SUNSET). The meadow on the plateau in the island view reads
// Oklab hue 75 - 83 and chroma 0.029 - 0.034 from -3 to -5.3 degrees, the cliffs chroma 0.045 -
// 0.047. Drawn most of the way to an orange as well (red over green about 2, blue over green 0.1),
// the land took that warmth twice: the meadow at hue 67 - 74 and chroma 0.037 - 0.042, the cliffs
// at chroma 0.06. With the dusk key's colour kept from the land as well, the lent light near
// grey, the meadow turned an olive green (hue 105 - 112): its warmth after sunset is the sky's.
// The afterglow is a clear sky's: under a deck there is no glow, and an orange cast laid on the
// cliffs of a cloudy or overcast evening read a warm brown under a blue deck and sea (the rock at
// -5 degrees red over green 0.83, against 0.5 and a blue grey under the clouded rows). A clouded
// sky takes rows of its own (ISLAND_SKY_CLOUDED: a light warm cast, the gains of 0.65 and 0.5 at
// -4 and -5.3 degrees) by its clouded share (`landSky`).
// The rising sun has rows of its own (ISLAND_SKY_RISING).
// By day a clouded sky lends the island its own light: a deck of cloud passes on little of the blue
// of the sky's high key (SKY_LIGHT), so from 12 to 40 degrees, as that key climbs, the light lent to
// the island is drawn most of the way to a warm grey (CLOUD_LIGHT_TINT, Weather.ts `cloudedHighSun`). The land
// keeps its green under a high sun; the meter's level is the same, the tint keeping the luminance.
const CLOUD_LIGHT_TINT = [ 1.2, 1, 0.5, 0.6 ] as const;
const ISLAND_SKY: readonly LandSkyRow[] = [
	{ at: 2 },
	{ at: 1, tint: [ 1.085, 0.988, 0.872 ], amount: 0.15, gain: 1.16 },
	{ at: 0.5, tint: [ 1.128, 0.982, 0.807 ], amount: 0.225, gain: 1.24 },
	{ at: 0, tint: [ 1.17, 0.976, 0.743 ], amount: 0.3, gain: 1.16 },
	{ at: - 1.3, gain: 0.89 },
	{ at: - 2.6, gain: 0.96 },
	{ at: - 3.3, gain: 0.85 },
	{ at: - 4, gain: 0.87 },
	{ at: - 5.3, gain: 0.92 },
	{ at: - 8.6, gain: 0.76 },
	{ at: - 13.2 },
];
const ISLAND_SKY_CLOUDED: readonly LandSkyRow[] = [
	...ISLAND_SKY.slice( 0, 4 ),
	{ at: - 1.3, tint: [ 1, 0.75, 0.45 ], amount: 0.5, gain: 0.89 },
	{ at: - 2.6, tint: [ 0.95, 0.8, 0.58 ], amount: 0.5, gain: 0.96 },
	{ at: - 3.3, tint: [ 1.11, 0.99, 0.767 ], amount: 0.5, gain: 0.85 },
	{ at: - 4, tint: [ 0.9, 0.85, 0.7 ], amount: 0.5, gain: 0.65 },
	{ at: - 5.3, tint: [ 1.04, 1, 0.874 ], amount: 0.36, gain: 0.5 },
	...ISLAND_SKY.slice( 9 ),
];

// The rising sun's: none of the evening's warm cast and no cast at all, gains of its own. A clear
// morning's day exposure is metered by the sky and the land together (post/Exposure.ts, the `dawn`
// key), so the land's brightness is this light against the sky's: the
// gains keep the land from about 0.55 at -8 degrees, where the sky's light is still deep, and 0.66
// at -6.5, where the land still takes part of the night's gain on its sky light (lighting/
// AmbientRelight.ts PROBE_SKY: about a quarter of it there), through a lift of about 2 at -2.6,
// where the sky's has fallen behind the sun's coming, to none from 2 up.
// A blue cast of its own turned the rock's brightest tenth to C 0.08 to 0.09 at -6.5 degrees,
// against 0.07 to 0.08 without: the land takes the sky's own colour. A clouded
// morning keeps these tints but takes the evening's gains back by its clouded share (`landSky`):
// its exposure stays on the clouded keys, which these gains were not set under.
const ISLAND_SKY_RISING: readonly LandSkyRow[] = [
	{ at: 2 },
	{ at: 1, gain: 1.3 },
	{ at: 0, gain: 1.6 },
	{ at: - 1.3, gain: 2 },
	{ at: - 2.6, gain: 2.2 },
	{ at: - 3.3, gain: 2 },
	{ at: - 4, gain: 1.8 },
	{ at: - 5.3, gain: 1.3 },
	{ at: - 6.5, gain: 0.66 },
	{ at: - 8, gain: 0.55 },
	{ at: - 8.6, gain: 0.6 },
	{ at: - 10, gain: 0.8 },
	ISLAND_SKY[ ISLAND_SKY.length - 1 ],
];

// Off by default (docs/design/art-direction.md, Lighting: the lighting keeps the sun's and the sky's
// own balance): with the meadow's own soft light the lifted shadows left a shaded clump within a
// quarter of a lit one and the meadow showed no side the light came from. `flags=clearSkyFill:<k>`
// (or `true`) brings it back for A/Bs. What it did: shadowed areas stayed clear, never black. Shadow is lit by
// the sky alone, so the sky light lent to the island (as ISLAND_SKY's: the environment's source
// and the islanders' fill) takes a gain and turns a touch cooler: the skylit faces of the cliffs
// and the meadow's shaded side open up while what the sky cannot reach (cracks, the meadow's
// floor, what sky occlusion darkens) stays dark. A constant lift in the grade instead greyed
// everything below mid grey alike and veiled the frame. The gain's extra share per sun elevation
// (degrees, descending; linear between rows, the end rows held beyond): a low sun's light already
// holds a larger share of sky, so the same gain brightened a morning island about twice as much as
// a noon one, and the share falls with the sun to none at 2 degrees, where ISLAND_SKY's evening
// rows begin; past them, in the night, the night sky's fill takes a share of its own, so the
// moonlit island's shadowed areas stay readable (a film's night, not a dark one). Cloud leaves the
// whole island in shadow, where a gain only brightens it: the share fades with the direct
// visibility, none by overcast. A noon gain of 1.8 was chosen over 2.5 by eye (at 2.5 the shadows
// turned cold).
const CLEAR_SKY_FILL = [
	{ at: 45, share: 0.8 },
	{ at: 26, share: 0.55 },
	{ at: 13, share: 0.35 },
	{ at: 5, share: 0.15 },
	{ at: 2, share: 0 },
	{ at: - 13.2, share: 0 },
	{ at: - 18, share: 0.6 },
];
// the sky light's tint and how far it is drawn toward it at a share of 1
const CLEAR_SKY_FILL_TINT = [ 0.86, 1, 1.1, 0.44 ] as const;

// Off by default with the clear shadows (`flags=lowSun:<k>` or `true` for A/Bs). What it did: a low
// sun's light golden and its shadows cool blue-violet. The shadows' half: from about 12 degrees down to the horizon the sky light lent to the
// island turns from CLEAR_SKY_FILL's cool cyan toward blue-violet, drawn toward it by up to
// LOW_SUN.amount whatever the gain (a low sun's share of the gain is small), so only the island's
// shade takes it, never the sea or the sky. It fades out above LOW_SUN.top, and below the horizon
// into ISLAND_SKY's warm evening cast; cloud takes it out with the direct sun. (The light's half,
// a warmer grade of what the low sun lit, is gone.)
const LOW_SUN = { full: 12, top: 28, amount: 0.65 };
const LOW_SUN_TINT = [ 0.95, 0.85, 1.2 ] as const;
// How far a deck draws the key toward white at the same luminance: overcast's sun is whiter than a
// clear day's at the same height but keeps some of its warmth; rain's closed deck draws it all the
// way.
const KEY_DECK_WHITE = 0.3;

// The sky's light against the sun's. As the atmosphere's march gave it, the clear day's sky lit
// level shade too brightly beside the sun. By day it takes one constant gain (SKY_LIGHT, an
// illuminance colour per channel, keyed by the sun's elevation: DayKeys.ts) on the light the sun
// lends it: the atmosphere's in-scattered light (Atmosphere.ts `skyRadiance`). So everything that
// reads the sky takes it, the seen sky, the environment's bakes and the irradiance probe that light
// the island, the sea's reflection and the clouds' and the haze's sky light, and the sky the eye
// sees stays the sky that lights. The clouds' sunlight takes none: it is the sun's beam at their
// height (sky/CloudKernels.ts `sunLight`; the cirrus's at each of its points, `streakSunColour`). A
// weather's day sky (Weather.ts `daySkyAt`) scales the day keyframes' share and the clouds' sunlight
// alike (`skyGain`, `trackClouds`), so the sky, the clouds and the haze it lends sky light to keep
// their balance; the sun's beam the haze scatters keeps the keyframes' light without it
// (`sunBeamIlluminance`), as the land's key does. How the sky's light falls with the sun is the atmosphere's
// own (the aerosol's wide base and the multiple scattering that grows along long paths:
// Atmosphere.ts), not rows by the sun's elevation. Under cloud the gain moves to the night key from
// 2 degrees to -6 under the horizon (a smoothstep); a clear sky's holds the day's keys through the
// twilight and moves from -7 to -11 after sunset (by stops), from -13 to -3 before sunrise
// (DayCycle.ts `skyNight`), so the sky after sunset dims with the sun's light it lends
// without the jump to the atmosphere's own light that fading the gain to none gave (several times
// brighter than the golden key, its blue more so). The sun's own light
// (its transmittance, the disk) and the cumulus it lights are untouched. The night key is the sun's
// deep twilight key; the moonlit sky takes the day's keys at the moon's height on the moon's light,
// at a hue of its own (MOON_SKY_HUE, `track`). The keys set
// the sky's brightness and colour in bands of elevation and of angle from the sun, together with
// the atmosphere's phase and aerosol and the haze's keys, once the environment's bake took the
// haze's light: the golden key takes the sky warm (red over blue about 2.9), the low key near grey,
// the high key blue (blue over red about 1.5), the atmosphere's own colour doing the rest, and the
// night key is about 0.03, its red about twice its green, which the night's white balance takes
// back (the display's `whiteBalance`, look/Display.ts): the land's dark colours stay near neutral
// under it, the clear night's dome is dark, its low sky the haze's moonlit light (Haze.ts NIGHT) and
// its blue the moonlit atmosphere's own and the rods' (Purkinje.ts).
// The golden, low and morning keys keep the colour they lent the sky before the ozone's absorption
// was integrated over each channel's band (Atmosphere.ts): with it the bare sky lost red and blue
// against green, by 0.75 and 0.88 times at a sun of 3 degrees, 0.89 and 0.95 at 12 and 0.97 and
// 0.99 at noon, and these keys take the first two back at their luminance (the golden hour's sky,
// far bay and meadow had turned yellow by 10 - 16 degrees of hue, the morning's sea toward cyan).
// The high key, whose sky hardly moved, keeps its own, its red at 0.264, set when the least-squares
// tuning of the clear day's sky gains (CLEAR_SKY) ran: against the 0.235 it had, it takes the sky over a
// sun at 50 degrees, seen away from it, about 7 degrees of hue toward violet (Oklab hue 241 to 248).
// Under cloud the high key has a twin (`cloudyHigh`, taken by the weather's clouded share, Weather.ts
// `cloudedShare`): a deck of cloud gives the sky and the haze no such blue, and the haze over the
// land takes this light whole (VolumetricHaze.ts), so at the day's blue the crowns under a clouded sky
// turned blue grey, most of all at 30 degrees. The twin stays near grey, blue over red about 1.1,
// at about the day's luminance, so the haze's hue moves and the land's level and the exposure do
// not. Its golden and low keys are the day's own, so a clouded sky parts from a clear one only
// over 12 degrees.
// On a clear evening the sky takes a key below the golden one (`dusk`, at -1 degrees, not under
// cloud), held through the twilight until the night key takes it from -7 degrees, at the
// luminance the warm key before it had. Its colour, green 0.55 and blue 0.33 of red, is
// set by least squares with the afterglow's haze weight (HazeKeys.ts) against the twilight's sky toward
// the glow at -3.3 and -5.3 degrees, over the atmosphere with its multiple scattering whole
// (Atmosphere.ts): under the near-neutral key it had (0.89 and 0.86) that sky stood near grey
// over the glow (Oklab chroma about 0.01 at -5.3), and the key takes it to orange from the glow to
// the top of a view toward it (red over green about 1.7, blue over green 0.4 - 0.6). When the
// multiple scattering stood at 1 / 4 pi of its level, a warm key (blue under a third of red)
// turned the sky 20 degrees up magenta (red over green about 2.1, blue over green 1.5, with the
// sun 5.6 degrees under the horizon). The sea under the glow takes less of the key's warmth than
// the sky over it: it stays lighter and greyer, by the water's reflection, not the sky's light.
// The land takes its twilight warmth from this key as well: below the horizon the light lent to it
// (ISLAND_SKY) keeps the sky's colour and takes only a gain.
// The morning is not the evening turned round. These keyframes hold in every direction, so a warm
// keyframe warms the whole rendered image: the golden keyframe is the evening's, where the whole sky
// is alight, and the clear morning has keyframes of its own (`dawn`, `morning`, read while the sun
// rises: DayKeys.ts `rising`). At sunrise the haze is already blue and the warm colour stays in the sun's road and the
// low streaks, so `dawn` takes a near-neutral light at the luminance the cooler key before it had,
// green 0.79 and blue 0.86 of red: set by least squares, with the dawn's haze weights
// (HazeKeys.ts), against the dawn's horizon views from -8 to 3 degrees over the atmosphere with its
// multiple scattering whole (it was the dusk key's near-neutral colour before, 0.89 and 0.86), and the
// display's dawn white balance (`whiteBalance`, look/Display.ts) cools it from there: the band
// over the horizon, nearly all the haze's light, and the sea under it stay a quiet blue grey. By 12
// degrees the rendered image has gone cool and quiet, and `morning` is the day's light at about two thirds
// of the low key with less green (red a little over blue, green about 0.7 of them), which turns the
// morning's sky and sea from cyan to blue, so the sea stands lighter under the same sky, the meter
// holding the sky's level. Both are tuned on our own views, at the luminance they had. From the `high` key up
// the two sides are one. A clouded morning keeps the clouded keys.
const SKY_LIGHT_LOW_SUN: readonly DayKey<Rgb>[] = [
	{ at: 3, name: 'golden', values: { r: 0.36, g: 0.162, b: 0.1227 } },
	{ at: 12, name: 'low', values: { r: 0.2409, g: 0.1857, b: 0.1979 } },
];
const SKY_LIGHT_HIGH: DayKey<Rgb> = { at: 40, name: 'high', values: { r: 0.264, g: 0.232, b: 0.346 } };
const SKY_LIGHT = {
	day: [ { at: - 1, name: 'dusk', values: { r: 0.2263, g: 0.125, b: 0.0751 } }, ...SKY_LIGHT_LOW_SUN, SKY_LIGHT_HIGH ],
	rising: [
		{ at: 3, name: 'dawn', values: { r: 0.2008, g: 0.159, b: 0.1727 } },
		{ at: 12, name: 'morning', values: { r: 0.1809, g: 0.1207, b: 0.1683 } },
		SKY_LIGHT_HIGH,
	],
	cloudy: [ ...SKY_LIGHT_LOW_SUN, { at: 40, name: 'cloudyHigh', values: { r: 0.25, g: 0.232, b: 0.28 } } ],
	night: { r: 0.045, g: 0.023, b: 0.028 },
};

// The clear day's sky against the key. With SKY_LIGHT alone the key's illuminance at normal
// incidence stood 50 - 72 times the sky's irradiance on level ground (the irradiance probe's: the
// environment's sky with its clouds and the haze over it, before any gain of the land's own) from
// 50 down to 6 degrees in the evening, 68 - 91 from 26 down to 7 in the morning, 30 - 45 at 4 and
// 20 - 24 at 3: the island's shadowed areas were nearly black against its lit side, and the sky stood dark
// against the key. Under a clear or fair day's sun the sky's light takes a gain of its own over
// SKY_LIGHT (CLEAR_SKY), keyed by the sun's elevation, on every path SKY_LIGHT reaches (`skyGain`):
// the sky the eye sees, the sea's reflection and its body, the clouds' and the haze's
// sky light, the land's (the environment's source: the probes' harmonics and the specular atlas,
// the haze over it without the fog layer, `fogEnvironment`; the islanders' fill, from the probe's
// panorama with the fog layer) and the exposure's sky meter. So the sky the eye sees stays the sky
// that lights and that the exposure meters (post/Exposure.ts). The sky's irradiance takes the gain
// whole (the sky's light, the sun's in-scattered light and the haze's are all SKY_LIGHT's), so the
// keys are the ratio with no gain over the one they hold, 12.1 - 15 from 6 degrees up on either
// side of the day and at the morning's 3 degrees, but for the day side's low and high gains (3.51
// and 2.48, 0.78 and 0.73 of the ratio's 4.48 and 3.4) and SKY_LIGHT's high red: those are set on
// the clear day's views from 6 degrees up, together with the haze (Weather.ts `haze`) and under the
// exposure keyframes of the time (look/ExposureLook.ts), which the cirrus (Clouds.ts), faint by day,
// hardly changes. At the ratio's gains the sky over a sun at 50 degrees, seen away from it, reads
// at Oklab lightness 0.71 and chroma 0.099 and the far sea at 0.63 and 0.102, against 0.68 and
// 0.10 and 0.62 and 0.097 at the tuned gains, and the views from 6 degrees up that take the day side's
// gains stand 0.16 - 0.33 stops darker. The rising side's own keys keep their gains (its high key
// is the day's). At the evening's 3 degrees the sky stands brighter against the key, about
// 7.4 - 8.9: it lights the land's shadowed areas and the far bay, as far as a view along the horizon toward
// the sun holds the far sea and the sun's strip on the water (look/ExposureLook.ts, the highlight
// key). The smoothstep between keys (DayKeys.ts) puts the ratio's day high midway, at 21 degrees in
// the evening. Under 3 degrees the key falls faster than the sky by itself (the ratio 12.4 at 1.6
// degrees with no gain), so the gain eases to none at the horizon (`horizon`): the twilight and the
// night keep SKY_LIGHT's light, the ratio falling to about 6 at 2 degrees on the way. The gains
// hold these ratios over the sky with its multiple scattering whole (Atmosphere.ts), whose
// irradiance stands higher than when the multiple scattering stood at 1 / 4 pi of its level:
// 0.85 - 0.89 times the gains that held them then, the exposure over the horizon within 0.03 stops
// of the one it had on either side of the day. The gain leaves by the weather's clouded share
// (Weather.ts `cloudedShare`): a cloudy or overcast sky keeps the light it had. The open sky seen
// through a deck's gaps is the clear day's sky all the same (most of the atmosphere's Rayleigh scattering
// lies above a deck of 1 - 3 km), so where the dome, the environment and the deck's irradiance see
// it past the clouds, by their transmittance, it takes the clear day's gain back (`gapSky`: the
// clear day's gain over the clouded sky's, exactly 1 under a clear or fair sky; each body's share of
// the sky takes it at that body's height, the moon's at the moon's). Without it the
// sky behind a cloudy deck's gaps would stand at the clouded sky's lower radiance, and the gaps would
// show the haze in front of them rather than their blue.
const CLEAR_SKY_HORIZON: DayKey<{ gain: number }> = { at: 0, name: 'horizon', values: { gain: 1 } };
const CLEAR_SKY_HIGH: DayKey<{ gain: number }> = { at: 40, name: 'high', values: { gain: 2.48 } };
const CLEAR_SKY = {
	day: [ CLEAR_SKY_HORIZON, { at: 3, name: 'golden', values: { gain: 2.41 } }, { at: 6, name: 'late', values: { gain: 3.39 } }, { at: 12, name: 'low', values: { gain: 3.51 } }, CLEAR_SKY_HIGH ],
	rising: [ CLEAR_SKY_HORIZON, { at: 3, name: 'dawn', values: { gain: 1.76 } }, { at: 6, name: 'early', values: { gain: 4.08 } }, { at: 12, name: 'morning', values: { gain: 5.83 } }, CLEAR_SKY_HIGH ],
	night: { gain: 1 },
};

// The clouds are one model, and a kind of sky is a row of its parameters: one row per sky of the blend
// (CLOUD_SKIES, in MIX_SKIES' order), blended each frame by the weather's shares of the four skies
// (`trackClouds`), so a change of weather crosses them by those shares. The parked weathers count
// as overcast in those shares (Weather.ts), so they take overcast's row. The coverage and the
// density are the weather state's (Weather.ts WEATHER_PRESETS), interpolated with it; nothing here
// points into the presets, whose fields the weather interpolates as numbers.
//
// `base`, `top` (m): the shell the clouds stand in (Clouds.ts `setShell`). Clear's and fair's
// cumulus stand high; cloudy's and overcast's layer low and shallow, as a stratocumulus or a
// stratus does, its masses nearer and so larger in the sky.
// `type`, `spread`: the cloud type (0 a stratus, 1 a cumulus: sky/CloudModel.ts STRATUS, CUMULUS) and how
// far it varies from place to place, either way. Fair's are cumulus; cloudy's run from
// stratocumulus to cumulus side by side; overcast's layer is a stratus, low and flat.
// `farFade`: how far the clouds thin with distance (Clouds.ts `farFade`). A fair sky's heaps melt
// into the haze before the horizon, where they would pile into bands; a layer runs to the horizon.
const CLOUD_SKIES = [
	{ type: 1, spread: 0, farFade: 1 }, // clear, in the cloud model's own shell
	{ type: 1, spread: 0, farFade: 1 }, // fair, in the cloud model's own shell
	{ base: 1500, top: 4500, type: 0.9, spread: 0.3, farFade: 0 }, // cloudy
	{ base: 600, top: 3000, type: 0.05, spread: 0.1, farFade: 0 }, // overcast
];

// The range each store holds under the moon, per sky of the blend (MIX_SKIES' order), in stops over the
// luminance of the moonlight on the clouds (`storeLight`): the brightest value and the first
// percentile (the cloud dome's and the layer's over the texels at least a tenth covered). Under
// the moon alone every store's radiance follows the moonlight, so the range over it is the clouds'
// own: coverage and density do not order it alone, the clouds' form does as much. A fair sky has
// the darkest thin edges (its
// small heaps); a cloudy sky the brightest moonlit sides (its broken layer); overcast's closed deck
// shows its dark base, the narrowest range. The rows are estimates that centre the range, measured
// on the skies' nights and kept through later changes of the clouds' forms, with several stops to
// spare on either side (NIGHT_TOP, NIGHT_FLOOR). The panoramas' group
// (the seen panoramas, the environment's source and its atlases) takes the sky's light too, its
// darkest the open sky low down. A weather between skies takes the rows by its shares of them,
// as its shell and form do (CLOUD_SKIES).
const STORE_RANGE: SkyLook[ 'storeRange' ] = {
	dome: [ [ - 1.7, - 8.5 ], [ 0.2, - 11.8 ], [ 2.1, - 6.2 ], [ - 1.7, - 6.5 ] ],
	layer: [ [ - 1.7, - 8.5 ], [ - 1.0, - 10.5 ], [ 2.1, - 6.3 ], [ - 1.8, - 6.2 ] ],
	panorama: [ [ - 1.2, - 12.0 ], [ - 1.1, - 12.0 ], [ 2.4, - 8.0 ], [ 0.1, - 6.7 ] ],
};

// The low sun's key. Through twenty air masses and more the atmosphere's transmittance (Atmosphere.ts,
// its aerosol dimming the low sun by about half at the golden hour) still leaves the key too bright
// under the lifted shading (lighting/KeyLift.ts): with the sun at 3 degrees the lit rock's whole face
// goes to a pale gold and the land reads as a bright afternoon. The key (the island's and the sea's
// sunlight) takes a gain under 6 degrees (degrees, descending; linear between rows, the last row
// held down to the horizon), so that face stays a saturated orange under the sky. The shadowed
// areas, lit by the sky, do not move; the disk and the sky's own sun are untouched.
const LOW_KEY = [
	{ at: 6, gain: 1 },
	{ at: 4, gain: 0.78 },
	{ at: 2.6, gain: 0.46 },
];

// The distant sea's body (the environment's lower half under its reflection, `seaPanorama`): its
// colour's share of the light that reaches the sea, the sky's irradiance with its clouds (the bake's
// deck irradiance) and the sun's direct light on the level water, in luminance, up to a knee at
// SEA_KNEE times the sun's illuminance (scene units, both at the weather's ambient gain), where it is
// the colour times that illuminance. The knee lies under every kept weather's noon (the light there
// is 0.258 of the illuminance under a clear sky, 0.251 under a fair one), so a high sun's body is
// what it was; under it the body falls with the light: to 0.57 of it at 30 degrees, 0.06 at 5, and
// under a hundredth after sunset, where the sky alone lights it. It followed a smoothstep of the
// sun's height from -7 to 8 degrees before, which held it 10 to 55 times the sky's E / pi through
// the twilight (0.46 at noon); the water's horizon term reads this half at its mip 5, whose row under
// the horizon takes 0.35 to 0.49 of the far sea's read, and the far sea stood grey over the glow it
// mirrors. Tunable: `sky.sea.knee`.
const SEA_KNEE = 0.24;

// Art direction: within a few degrees of the horizon the sun shines through twenty to forty air
// masses and its light turns an almost pure red (red/blue above 100 at one degree), which paints
// the island like lava. The optical depth that sets the key's hue is capped softly past about
// six degrees (`sky.key.hueDepth`), while its luminance keeps the full extinction: a low sun stays
// orange red (red/green about 1.85 at six degrees, 2.2 at the golden hour and 2.25 at the horizon,
// red/blue at most about 13) and still dims as it sets. The cap stands where it holds the golden
// hour's sunlit faces at a full orange without the extra chroma the clean air aloft gives the key
// (Atmosphere.ts). The rising sun's cap is lower (`sky.key.dawnHueDepth`, binding from about ten
// degrees down): the morning's warm colour lies in the sun's road and the low streaks, and a face
// turned to the rising sun takes a pale warm light, far less orange than the evening's at the same
// height (red/green at most about 1.6, red/blue about 4). Above that the key's hue is
// chosen rather than the atmosphere's own course, which reddens evenly with the air mass: a warm
// white while the sun is high (KEY_WHITE: red/green 1.15, blue/green 0.95 at noon, 1.17 and 0.9
// at 30 degrees) that loses its blue first (0.8 at 20 degrees, red/green 1.2) and turns orange
// between 15 and 10 degrees (1.45 and 0.55 at 11.5). So each of the two ratios takes the atmosphere's
// change of it in part, by the atmosphere's optical depth in green (`sky.key.reddens` for red/green, `.yellows` for
// blue/green: about none at noon, two thirds at that depth, all of it past twice it). The ozone's
// absorption integrated over each channel's band (Atmosphere.ts) takes red off the atmosphere's
// transmittance (red/green 1.45 to 1.29 at 12 degrees), so the red/green share's onset is 0.52
// where it was 0.75, which brings the key back to about 1.45 at 12 degrees (0.75 left it at 1.29,
// a lit face yellow green) and leaves the lower sun a little less red than before. The sky and
// clouds keep their own colours, and the key's luminance is the sun tint's.
const KEY_HUE = {
	// the caps of the depth that sets the hue: the setting sun's and the rising sun's
	depth: 1.6, dawnDepth: 0.8,
	// the key's white: linear red/green and blue/green while the sun is high
	white: [ 1.15, 0.96 ],
	reddens: 0.52, yellows: 0.52,
} as const;

// The overall colour on each of the cirrus's two shares (Clouds.ts `keyDirect`, `keyAmbient`),
// by the sun's elevation, the morning's apart, the night's coming in with the clouds' moonlight.
// Under a low sun (afterglow, golden, low, predawn, sunrise, morning) the two shares take the
// light's colour apart: the direct share warm, red over green 2.2, the ambient share cool, blue
// over red 1.5, their luminances 0.38 and 0.59 of a white key's; together they read red plus blue
// less green, a rose. White keys left the layer the colour of the sun's light at its height alone,
// a yellow-white 0.17 above the sky round it in Oklab L and paler than it. With these, in the
// golden hour, it reads rose toward the sun and away from it alike (hue 35 - 43), 0.03 - 0.04 more
// saturated than that sky and 0.08 - 0.09 above it in L. The direct share's luminance moves that
// L little (0.38 to 0.44 of white: 0.085 to 0.10); a redder direct share turns the side toward the
// sun past rose, a less red one leaves the side away from it as pale as the sky. From 8 degrees up
// to 30 both shares turn back to white, the morning's as the evening's; only the high sun's keys
// and the night's are white, so the layer is unchanged under a sun above 30 degrees and by night.
const white = (): Rgb => ( { r: 1, g: 1, b: 1 } );
const warm = (): Rgb => ( { r: 0.66, g: 0.3, b: 0.3 } );
const cool = (): Rgb => ( { r: 0.6, g: 0.55, b: 0.9 } );

// The moon's key's colour as written (red over green 0.8, blue over green 1.4; the island's moonlight
// takes the tuned hue over it, `moonHue`, near neutral), its level the
// moon's light above the atmosphere through the atmosphere toward the moon (`moonAboveAir`), as the
// sun's key light is the sun's through the atmosphere toward the sun; how bright the night shows is
// the exposure's (post/Exposure.ts).
// The island's moonlight takes a hue of its own (`moonHue`), and the clouds' direct moonlight the
// same; the moonlit sky, the clouds' sky light and the haze take the moon's light above the atmosphere at
// their own hue (MOON_SKY_HUE).
// The red and blue over green are as written (the quotients of the channels differ by an ulp), so
// the hue below at these is exactly one.
const MOON_KEY = { colour: [ 0.0108, 0.0135, 0.0189 ], red: 0.8, blue: 1.4 } as const;
// The moonlit sky's colour at unit luminance: MOON_KEY times the sky's night key (SKY_LIGHT's
// `night`), normalised, the hue the night sky had while its level was that key's. The moonlit sky,
// the haze's moonlight and the clouds' sky light keep it; their level is the sun's sky at the moon's
// height (`track`). The sky's night key itself stays the sun's deep twilight key.
const MOON_SKY_HUE = [ 1.3366, 0.8540, 1.4554 ] as const;

// The night's ambient's colour at unit luminance (`displayNightAmbient`).
const NIGHT_AMBIENT_HUE = [ 0.6248, 1.0154, 1.9526 ] as const;
// The night's ambient in the clouds and the distant sea (`displayNightAmbient`), a radiance: a
// uniform sky giving level ground a fiftieth of the moon's key light on it (its light through the atmosphere at
// normal incidence). Starlight and the airglow give level ground a few thousandths of a lux, the
// full moon some hundredths at its height.
const NIGHT_AMBIENT_SHARE = 1 / 50;

// The island's moonlight's red and blue over green. They move the moonlit ground's colour without
// its level (the hue keeps MOON_KEY's luminance) and without the moonlit sky's, clouds' and haze's,
// which keep theirs. At MOON_KEY's own (MOON_KEY_RED, MOON_KEY_BLUE) the hue is exactly white.
// The moonlight itself is near neutral, a touch cool: the night's blue on the display is the eye's
// (post/Purkinje.ts), given once, so a blue key under it blued the land twice, in the sky's own hue.
// Under it moonlit rock reads a cool grey, the island's brightest thing, and the meadow nearly
// black. The clouds take the same moonlight (`trackClouds`). The two were set once on the clear
// night's views: moonlit rock's chroma rose by about .004 in Oklab between (.97, 1.05) and these,
// its hue a few degrees toward blue, the meadow's by under a thousandth (its chroma is its albedo's).
const MOON_HUE = { red: 0.93, blue: 1.16 };

// The key under cloud, one rule for every sky. Where cloud stands along the key the land does not
// lose the key whole: a cloud passes on a part of the sunlight it takes, most of it scattered
// forward, so the ground under it keeps a soft light from the sun's side that still casts the
// island's shadows. The land's key keeps mix( floor, 1, T ), T the clouds' transmittance along the
// key (`keyDirectVisibility`), and the floor is the diffuse transmission of a layer of the sky's
// shell and density (Clouds.ts `layerTransmission`: about 0.21 under fair's shell, 0.28 under
// cloudy's, 0.25 under overcast's) times this share, at most 1: how much of what the layer passes
// down the land keeps as a light along the key, the one parameter for how deep the clouds shade the
// land, as engines give the cloud shadow on surfaces a strength of its own. A thicker or denser
// layer leaves less, and where the clouds thin to a gap the key goes back to the sky's own, so a
// gap's lit ground is never brighter than a clear day's under the same sun. The sea keeps the same
// floor (`seaKeyVisibility`; its glint only the direct beam, `cloudDirectVisibility`), and under a
// closed sky the haze does (`deckKeyFloor`, by overcast's share). The floor
// rides on whichever light keys the scene, the moon's by night as the sun's by day; under a closed
// sky the sun's part of it leaves over the last 2.5 degrees before the horizon (Weather.ts
// `overcastFloorSun`), so the land there keeps the layer's diffuse light through sunset, with no
// flash of the setting beam. Raise it for a stronger light from the sun's side and darker cast
// shadows under cloud; lower it for deeper cloud shadows and a flatter land under a closed sky.
const KEY_THROUGH_CLOUD = 3;

export const SKY: SkyLook = {
	cloudLightTint: CLOUD_LIGHT_TINT,
	landSky: { setting: ISLAND_SKY, clouded: ISLAND_SKY_CLOUDED, rising: ISLAND_SKY_RISING },
	clearSkyFill: CLEAR_SKY_FILL,
	clearSkyFillTint: CLEAR_SKY_FILL_TINT,
	lowSun: LOW_SUN,
	lowSunTint: LOW_SUN_TINT,
	keyDeckWhite: KEY_DECK_WHITE,
	skyLight: SKY_LIGHT,
	clearSky: CLEAR_SKY,
	cloudSkies: CLOUD_SKIES,
	storeRange: STORE_RANGE,
	lowKey: LOW_KEY,
	seaKnee: SEA_KNEE,
	keyHue: KEY_HUE,
	streakDirect: { afterglow: warm(), golden: warm(), low: warm(), predawn: warm(), sunrise: warm(), morning: warm(), high: white(), night: white() },
	streakAmbient: { afterglow: cool(), golden: cool(), low: cool(), predawn: cool(), sunrise: cool(), morning: cool(), high: white(), night: white() },
	moonKey: MOON_KEY,
	moonSkyHue: MOON_SKY_HUE,
	nightAmbientHue: NIGHT_AMBIENT_HUE,
	nightAmbientShare: NIGHT_AMBIENT_SHARE,
	moonHue: MOON_HUE,
	keyThroughCloud: KEY_THROUGH_CLOUD,
};
