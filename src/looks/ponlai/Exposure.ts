import type { ExposureKey, ExposureLook, SharingKey } from '../../render/setup.ts';

// The island's exposure look: the exposure's keys by the sun's elevation, cloud and night.
//
// The island's exposure look (packages/island/look/: the island's numbers for the render pipeline's
// look tables, look/Look.ts; look/ExposureLook.ts reads these): the exposure's keys by the sun's
// elevation, the rising sun, the clouded day and the night, and overcast's keys on top of them.
// post/Exposure.ts's header says what each key does.

// The exposure's keys (DayKeys.ts): the evening's dusk (held from -0.4 degrees down) and sunset
// (0.4), the golden hour (a sun at 3 degrees), a low sun (12), a high one (40), the rising sun's
// dawn (held from -2 degrees down) and its golden hour (3), a clouded day's (-1, 0, 3, 6 and 12)
// and the night. They are set against the exposed white of a diffuser under
// the sun's direct light alone (by night the shaded one), seen toward and away from the sun, with
// the sky's light as look/SkyLook.ts `skyLight` and `clearSky` set it, together with the haze and the grading.
// The day's κ follow the sea's share of the meter (ocean/Water.ts): the sea is metered as a share
// of what a white card laid level at its point would meter, so a frame's sea no longer moves its
// meter up under a clear sky and down under a closed deck. Each key was lowered in steps on the
// views of its state (the clear day's from 6 degrees up for the low and high keys, the golden
// hour's, the rising sun's, the clouded day's), to the step at which those views' land, sky and sea
// read as they did before the change, within a third of a stop in lightness: the high key went down
// by 0.8 stops, the low by 0.9, the golden by 0.7, the rising sun's dawn by 0.8 and its golden hour
// by 0.6, the clouded by 0.9. The sky's share β is 0.4 under a clear day's keys, the dawn's and the
// night's, 0.2 under the golden hours' and the clouded ones: the land's light sets most of a view's
// exposure, the sky's own enough that a view follows the sky's light through the day. The highlight
// key holds a view toward a low sun (the sea's glitter toward a sun at 12 degrees, a view along the
// horizon toward one at 3 degrees, one into the glow after sunset) under one away from it on the
// scene's share; with the keys lower, such a view stands 0.14 to 0.36 stops darker than before, the
// golden hour's the least. The rising sun's keys share it (shared, so tuning `@golden` moves them
// all). The clouded keys' κ_H lie well over their κ, so the guard holds a clouded view down only
// where its brightest few percent lie far over its metered light (under a sky with no sun, rarely).
// A clouded day stands under the clear one by the light its clouds take, overcast's key
// OVERCAST_KEY on top of it by its share. `cloudy` sets the cloudy sky toward a high sun, where the
// land in the clouds' shade is lit about as the clear day's metered frame is and the key alone
// holds the frame under it: at -0.6 the broken cloud over a meadow reads at a display lightness of
// 0.63 and the sunlit rock at 0.55, and the meadow in the clouds' shade at 0.26 with the sun 15
// degrees up and, on toward `cloudyLow`, at 0.19 with it 8 up. It sets the bright cloud's
// lightness and share, its tint (Oklab b, which the tone curve moves) and the gaps between the
// clouds. A lower key darkens the gaps faster than the bright cloud, and the clouds' own contrast
// widens a little as the frame darkens, which no key mends; the land under the deck goes down with
// the sky. The keys from the golden hour down were set again once the sea's metering and the
// dusk's light had settled, on the island's default view through the evening's and the morning's
// dusk, each to where that view's land reads as it did before the sea's metering, within a third of
// a stop (the night's own change apart, which the night key carries; on screen the land moves about
// one and a half times a key's change there, in the tone curve's toe). The clear evening's land
// under the horizon read 0.6 stops over that, so the dusk key lies 0.35 stops under the golden
// hour's, reached from `sunset` at 0.4 degrees, which shares the golden hour's. The clouded day's
// land had lost the fall from 12 to 6 degrees its old meter made and read 1.6 stops over it from 7
// degrees to the horizon, so `cloudyLow`, `cloudyGolden` and `cloudySunset` lie 1.05, 0.95 and 1.2
// stops under the keys they had and `cloudyDusk` holds 1.6 under from -1 degree down, where the
// clouded exposure's night weight takes over (DayCycle.ts `exposureNight`); the cloudy morning
// shares them and reads within half a stop of that, its land brighter just under the horizon and
// darker where gaps in the cloud let the low sun through. One night key serves every night's sky. It lies about 2.3
// stops under the day's high key and 2.1 under its low: the moonlit open sky is the brightest large
// thing in a night's frame, and the sea's smaller share in the meter raised the night's exposure
// by about 0.8 stops at the key it had, so the key is raised to the lowest value that holds its gap
// to the day's high key within a third of a stop of two stops. The meter would take back the
// moonlight a deck takes, so overcast's night keeps a share of that darkness on screen
// (OVERCAST_NIGHT). The rising sun's dawn key meters the morning's land before the
// sun by both the sky and its own light, so the land keeps its level while the sky brightens toward
// the sun (a morning's day exposure comes in long before the sun: DayCycle.ts `displayNight`), and
// from -2 to 3 degrees it passes into the morning's golden key, about 0.4 stops under the
// evening's: at 3 degrees the island view meters mostly the sea on either side of the day (about
// seven tenths of its metered cells), but in the morning its land lies nearly all in shade (about a
// tenth of it sunlit, against four tenths in the evening). Under cloud the clouded keys
// take over from it by the cloud's share, as from the day's. The adaptation's time constants are
// the same in every key (the header, Adaptation). Tunable on debug pages (debug/Tune.ts):
// `post.exposure.<value>@<key>`. A value another key shares is written with that key's number and named in
// `shares`: the reader makes the two one number and one tunable.
const GOLDEN: ExposureKey = { key: - 1.3, highlight: 0.45, sky: 0.2, adaptUp: 0.05, adaptDown: 0.05 };
const FROM_LOW: readonly SharingKey<ExposureKey>[] = [
	{ at: 12, name: 'low', values: { key: - 0.46, highlight: - 0.97, sky: 0.4, adaptUp: 0.05, adaptDown: 0.05 } },
	{ at: 40, name: 'high', values: { key: - 0.24, highlight: 0.11, sky: 0.4, adaptUp: 0.05, adaptDown: 0.05 } },
];

export const EXPOSURE: ExposureLook = {
	keys: {
		day: [
			{ at: - 0.4, name: 'dusk', values: { key: - 1.65, highlight: GOLDEN.highlight, sky: 0.2, adaptUp: 0.05, adaptDown: 0.05 }, shares: { highlight: 'golden' } },
			{ at: 0.4, name: 'sunset', values: { key: GOLDEN.key, highlight: GOLDEN.highlight, sky: 0.2, adaptUp: 0.05, adaptDown: 0.05 }, shares: { key: 'golden', highlight: 'golden' } },
			{ at: 3, name: 'golden', values: GOLDEN },
			...FROM_LOW,
		],
		rising: [
			{ at: - 2, name: 'dawn', values: { key: - 1.79, highlight: GOLDEN.highlight, sky: 0.4, adaptUp: 0.05, adaptDown: 0.05 }, shares: { highlight: 'golden' } },
			{ at: 3, name: 'morningGolden', values: { key: - 1.71, highlight: GOLDEN.highlight, sky: 0.2, adaptUp: 0.05, adaptDown: 0.05 }, shares: { highlight: 'golden' } },
			...FROM_LOW,
		],
			cloudy: [
				{ at: - 1, name: 'cloudyDusk', values: { key: - 2.9, highlight: 3.05, sky: 0.2, adaptUp: 0.05, adaptDown: 0.05 } },
				{ at: 0, name: 'cloudySunset', values: { key: - 2.5, highlight: 3.05, sky: 0.2, adaptUp: 0.05, adaptDown: 0.05 } },
				{ at: 3, name: 'cloudyGolden', values: { key: - 2.25, highlight: 3.05, sky: 0.2, adaptUp: 0.05, adaptDown: 0.05 } },
				{ at: 6, name: 'cloudyLow', values: { key: - 2.04, highlight: 3.05, sky: 0.2, adaptUp: 0.05, adaptDown: 0.05 } },
				{ at: 12, name: 'cloudy', values: { key: - 0.6, highlight: 3.05, sky: 0.2, adaptUp: 0.05, adaptDown: 0.05 } },
			],
		night: { key: - 2.57, highlight: - 1.32, sky: 0.4, adaptUp: 0.05, adaptDown: 0.05 },
	},
	// Overcast's key on top of the clouded one toward a high sun, stops, by overcast's share of the
	// sky (Weather.ts `skyMix`, a parked sky counting as overcast), whole by day and handing over to the
	// night by the exposure's own night weight, as the keys do. It enters here alone, never in the
	// keys' own blend. It stands with overcast's own day sky (sky/Weather.ts, overcast's `daySky`): the
	// sky dome meters at its bake's geometric-mean luminance, so a darker sky drags the meter and the
	// exposure brightens, the more sky a frame holds the more. At -0.5, with overcast's day sky at
	// 0.092 and the sun at 30 degrees, the meadow away from the sun reads at a display lightness of
	// 0.28 and the deck behind it at 0.33 to 0.34. The parked rain, storm and fog
	// count as overcast in the sky's blend and take this key but keep a day sky of 1, so they read
	// darker than overcast; they are parked and left so.
	// Toward a low sun it passes to OVERCAST_KEY_LOW (OVERCAST_KEY_SPAN). Tunable:
	// `post.exposure.overcastKey`.
	overcastKey: - 0.5,
	// Overcast's key toward a low sun, stops, in place of OVERCAST_KEY. Once the sea was metered as a
	// share of the white card's light, the overcast evening's land read from 0.4 stops over what it
	// read before at 12 degrees to 1.0 to 1.3 over from 7 degrees to the horizon, where the clouded keys
	// it holds (OVERCAST_HELD) stay flat; this key takes that back, set on the island's default view
	// between the evening's land and the morning's, which reads up to half a stop brighter under a deck
	// whatever the key. Under the horizon the land no longer follows how it read before: that fell two
	// stops within a degree, the exposure falling with it, and the exposure may not fall as the light
	// does (its night weight comes in no faster than the meter's light goes). It stands 0.14 stops
	// under OVERCAST_KEY: each 3-degree step's fall of the key across OVERCAST_KEY_SPAN is at most
	// 0.34 of that gap, under 0.05 stops. Tunable: `post.exposure.overcastKeyLow`.
	overcastKeyLow: - 0.64,
	// Where overcast's key passes from its low sun's value (OVERCAST_KEY_LOW, at and under the first)
	// to its high sun's (OVERCAST_KEY, at and over the second), degrees: a smoothstep.
	overcastKeySpan: [ 6, 18 ],
	// The share of a clear sky's light the overcast deck lets through onto the land by night: the
	// open land's illuminance under overcast against a clear sky's with a key at the moon's height,
	// measured on our island with the deck in its night form (sky/CloudModel.ts MOON_DECK_GLOW, the whole
	// deck seen) and the moon's fixed land gain (look/Ambient.ts `overcastMoonLand`), the
	// form the night draws. So in scene units an overcast night lies log2( 1 / 0.72 ), about 0.47
	// stops, under a clear one. The day's deck form reads about 0.79 by the same measure; the night
	// never takes it.
	deckLandThrough: 0.72,
	// Overcast night's share f of that darkness the display keeps: the night key goes down by f times
	// it, so the meter takes back only 1 - f of it and an overcast night reads darker than a clear one
	// on screen, not only in scene units. f is a choice of the look (0: the meter takes it all back,
	// 1: none). It comes in by overcast's share and the exposure's night weight, a function of the
	// clock and the weather alone. Tunable: `post.exposure.overcastNight`.
	overcastNight: 0.5,
	// Where overcast holds the clouded keys from, degrees (`cloudy`): its closed deck does not break
	// up at a low sun (sky/Weather.ts `overcastDeckLight`), so below it overcast keeps that key, its own
	// key toward a low sun (OVERCAST_KEY_LOW) taking the low sun's light, and hands over only to the
	// night's, while a cloudy sky's keys go on to the low sun's and the evening's under 12 degrees.
	overcastHeld: 12,
};
