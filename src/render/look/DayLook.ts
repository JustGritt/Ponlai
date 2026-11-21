import { tuneObject } from '../debug/Tune.ts';
import { markSkyRetune } from '../sky/SkyRetune.ts';
import type { DayLook, Span } from './Look.ts';

// The day's windows built from a look's table: the key's handoff, dusk and the night's layers.
//
// The day's windows built from a look's table (look/Look.ts `DayLook`): where the key turns from
// the sun to the moon, the dusk's windows, how deep under all of it each night layer comes in and
// the ease at the window's night end. Like the exposure (look/ExposureLook.ts), they are taken as
// an argument where the render pipeline's sky stage creates them (RenderStages.ts `createSky`), so
// two made from two tables hold their own numbers (tools/render/verify-look-params.mjs) and this
// module holds none at module level. sky/DayCycle.ts reads them for the night's shares, each
// share's function taking them as its first argument; its header says how the day runs. The day
// cycle (`DayCycle.windows`) and the sky (`Sky.dayWindows`) carry them to the sky's track, the
// ambient light's relight, the haze's keys and the island's lamps. The day's table also holds the
// exposure by the sun's elevation with the night sky's own exposure: those the exposure takes where
// it is created (look/ExposureLook.ts), not this module.
// - They register their tunables (debug/Tune.ts) as they are created, under the names they always
//   had (`sky.twilight.depth.<layer>`, `sky.twilight.ease`). The tune table is one per process: on
//   a debug page a second set made from another table reads the first one's tunables.

export type { Span };

/** The day's table the windows are made from (`createDayWindows`). */
export type DayWindowTable = Pick<DayLook, 'handoffTop' | 'moonKeyTop' | 'duskWindows' | 'cloudedDaylight' | 'duskDepth' | 'duskEase'>;

/** The day's windows a look's table makes (`createDayWindows`). */
export interface DayWindows {
	/** Highest elevation (degrees) at which the key may turn from the sun toward the moon. */
	readonly handoffTop: number;
	/** Elevation (degrees) at which the moon's key starts to rise, in full two degrees lower: the
	 * handoff has just turned the key to the moon, so it turns through zero light (`handoffTop` to
	 * here). */
	readonly moonKeyTop: number;
	/** Where each of the dusk's shares comes in, the sun's elevation (degrees): [ whole, none ], all
	 * of the share at or under the first, none at or over the second (DayCycle.ts `rampIn`, or
	 * `duskIn` by stops for a clear evening's), the clear evening's and morning's and the clouded
	 * sky's apart. A clouded sky's share is the clear one's moved toward the clouded one by the
	 * clouded share (Weather.ts `cloudedShare`). `display` is the display's night (tone, grade,
	 * bloom), `exposure` the exposure's own night weight, `stars` the drawn stars' clear evening
	 * (their morning and a clouded sky's are the display's, `starNight`), `sky` the sky's night key,
	 * `ambient` the night's ambient (and the leaves' night key and the sward's), `nightGlow` the
	 * night glow (what glows by night) and the local lights, `cloudMoon` the clouds' own moonlight, `moonKey` the
	 * moon's key and its sky, `handoff` the key's turn from the sun to the moon. A clouded sky's
	 * display and stars follow the daylight instead (`fromDaylight`); its exposure takes its own
	 * window by elevation (`exposure.clouded`). */
	readonly dusk: DayLook[ 'duskWindows' ] & { readonly moonKey: Span; readonly handoff: Span };
	/** The daylight's window in which a clouded sky's display and stars take the night (the
	 * snapshot's `daylight`): all of it under the first, none over the second. */
	readonly cloudedDaylight: Span;
	/** How deep under all of it (stops) each night layer comes in at the evening's window's day end,
	 * rising by even stops a degree toward all of it at the window's night end (`duskIn`, which
	 * eases the last `ease` degrees): the night key in the sky's light (`sky`), the night ambient and
	 * the night glow and its light (`ambient`), the stars, the haze's night key and the clouds' own
	 * moonlight. Tunable on debug pages (debug/Tune.ts): `sky.twilight.depth.<layer>`. */
	readonly depth: { sky: number; ambient: number; stars: number; haze: number; cloudMoon: number };
	/** Over how many degrees at the night's end of the evening's window (`dusk`'s evening windows)
	 * its night layers ease into all of their light (`duskIn`). Tunable: `sky.twilight.ease`. */
	readonly ease: { degrees: number };
}

/** The day's windows made from a look's table: registers their tunables (see the header). */
export function createDayWindows( day: DayWindowTable ): DayWindows {

	const options = { min: 1, max: 24, onChange: markSkyRetune };
	const depth = tuneObject( { ...day.duskDepth }, {
		sky: 'sky.twilight.depth.sky', ambient: 'sky.twilight.depth.ambient',
		stars: 'sky.twilight.depth.stars', haze: 'sky.twilight.depth.haze', cloudMoon: 'sky.twilight.depth.cloudMoon',
	}, { sky: options, ambient: options, stars: options, haze: options, cloudMoon: options } );
	const ease = tuneObject( { degrees: day.duskEase }, { degrees: 'sky.twilight.ease' }, { degrees: { min: 0, max: 4, onChange: markSkyRetune } } );
	return {
		handoffTop: day.handoffTop,
		moonKeyTop: day.moonKeyTop,
		dusk: {
			...day.duskWindows,
			moonKey: [ day.moonKeyTop - 2, day.moonKeyTop ],
			handoff: [ day.moonKeyTop, day.handoffTop ],
		},
		cloudedDaylight: day.cloudedDaylight,
		depth,
		ease,
	};

}
