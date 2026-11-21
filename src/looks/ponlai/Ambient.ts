import type { AmbientLook, AmbientRowLook } from '../../render/setup.ts';

// The island's environment-light look: the ambient grade's table and the low key's lift.
//
// The island's environment-light look (packages/island/look/: the island's numbers for the render
// pipeline's look tables, look/Look.ts; look/AmbientLook.ts makes the grade's tables and the lift
// from these): the ambient grade's table by the sun's elevation and the weather's skies, with the
// gains under overcast's deck, and the low key's lift. lighting/AmbientGrade.ts's and
// lighting/KeyLift.ts's headers say what the numbers do on screen. A row spells only its changes
// over the parameters at no grade (look/AmbientLook.ts DEFAULT_AMBIENT_SETTINGS).

// The key's bounce by day under the clouded skies, against 1 at no grade: the cloudy sky's about
// half (8 / 15), overcast's a third (the header). Overcast's keeps 0.33 from 2 degrees down.
const CLOUDY_BOUNCE = 8 / 15, OVERCAST_BOUNCE = 1 / 3;

// The clear and the fair sky's bounce of the key: 1 at the 60 degree row, so about 1.3 at the
// sun's noon of 52 degrees, 2 at 30, 2.3 from 20 down to 9 and back to 1 by 4, the morning's as
// the evening's (GRADE_RISING keeps these rows), so the grass in a wide shadow beside the lit
// meadow stays green and turns only a little blue (lighting/AmbientGrade.ts's header).

// The rows: the high sun, the afternoon, the low sun (13 to 7 degrees), the sky's reflection letting go (5.5
// to 4 degrees), golden hour,
// where the sky's gain starts toward the night key (2 degrees), sunset, the afterglow (-1.3 and -2.6 degrees), the
// blue hour, the night
const GRADE: readonly AmbientRowLook[] = [
	{ at: 60, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: OVERCAST_BOUNCE } },
	{ at: 30, clear: { sunBounce: 2 }, fair: { sunBounce: 2 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: OVERCAST_BOUNCE } },
	{ at: 20, clear: { sunBounce: 2.3 }, fair: { sunBounce: 2.3 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: OVERCAST_BOUNCE } },
	{ at: 13, clear: { sunBounce: 2.3 }, fair: { sunBounce: 2.3 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: OVERCAST_BOUNCE } },
	{ at: 10, clear: { sunBounce: 2.3 }, fair: { sunBounce: 2.3 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: OVERCAST_BOUNCE } },
	{ at: 9, clear: { sunBounce: 2.3 }, fair: { sunBounce: 2.3 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: OVERCAST_BOUNCE } },
	{ at: 7, clear: { sunBounce: 1.8 }, fair: { sunBounce: 1.8 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: OVERCAST_BOUNCE } },
	{ at: 5.5, clear: { skySpecular: 0.5, sunBounce: 1.4 }, fair: { skySpecular: 0.5, sunBounce: 1.4 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: OVERCAST_BOUNCE } },
	{ at: 4, clear: { skySpecular: 0.5 }, fair: { skySpecular: 0.5 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: OVERCAST_BOUNCE } },
	{ at: 2, clear: { skySpecular: 0.5 }, fair: { skySpecular: 0.5 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: 0.33 } },
	{ at: 0, clear: { skySpecular: 0.5 }, fair: { skySpecular: 0.5 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: 0.33 } },
	{ at: - 1.3, clear: { skySpecular: 0.5 }, fair: { skySpecular: 0.5 }, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: 0.33 } },
	{ at: - 2.6, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: 0.33 } },
	{ at: - 4, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: 0.33 } },
	{ at: - 18, cloudy: { sunBounce: CLOUDY_BOUNCE }, overcast: { sunBounce: 0.33 } },
];

// The rising sun's (the header): the clear and the fair sky's reflection at 0.1 from -2.6 to -12
// degrees, the night's from -13 down; the clouded columns are the evening's.
const DAWN_SPECULAR = 0.1;
const dawn = ( row: AmbientRowLook, skySpecular: number ): AmbientRowLook => ( { ...row, at: row.at, clear: { ...row.clear, skySpecular }, fair: { ...row.fair, skySpecular } } );
const GRADE_RISING: readonly AmbientRowLook[] = [
	...GRADE.filter( ( row ) => row.at > - 2.6 ),
	dawn( GRADE[ GRADE.length - 3 ], DAWN_SPECULAR ),
	dawn( GRADE[ GRADE.length - 2 ], DAWN_SPECULAR ),
	dawn( { ...GRADE[ GRADE.length - 2 ], at: - 12 }, DAWN_SPECULAR ),
	{ ...GRADE[ GRADE.length - 1 ], at: - 13 },
	GRADE[ GRADE.length - 1 ],
];

export const AMBIENT: AmbientLook = {
	grade: GRADE,
	gradeRising: GRADE_RISING,
	// a gain on the table's `sunBounce` at every hour and in every weather
	sunBounceGain: 1,
	// The sun's part of the weather's gain under overcast's deck, per unit, on the land's ambient
	// light: on the sky's light per unit of the bake (after the unit that makes it a clear day's in the
	// same place: AmbientRelight.ts OVERCAST_LAND_UNIT), its bounce with it, by overcast's share and its
	// deck's light (`overcastLandWeight`), and on the key's bounce per unit of the table's (`update`),
	// by overcast's share and the deck's light at the sun's height (the moon's at its own height takes
	// `overcastMoonLand`). It is not the land's whole ambient light under the deck over a clear day's:
	// that ratio also holds the bake's ratio of the two skies and the sun's bounce's share of the
	// light. 1 leaves the land as the sky and the table give it. At 1.35, with overcast's dimmer day
	// sky (Weather.ts `daySky` 0.074), on lit open ground (a white card's light, the brighter half of
	// the ground and meadow within 40 m on two eye-height views beside and away from the sun), the
	// ambient light reads about 0.95 of the sky's irradiance over pi and the card's whole light 5.3
	// times it, the direct light making 0.82 of it with the sun 22 degrees up and 0.81 at 15. Tunable:
	// `lighting.relight.overcastLand`.
	overcastLand: 1.35,
	// The moon's part of the weather's land gain under overcast's deck (`overcastLand` is the sun's):
	// the moonlit land's sky light and the moon's bounce under the deck.
	overcastMoonLand: 1.75,
	// the low key's lift (lighting/KeyLift.ts): the shading direction's elevation at the sun key's
	keyLift: {
		rows: [ { at: 0, shading: 18 }, { at: 5, shading: 18 }, { at: 6, shading: 14 } ],
		join: 1,
	},
};
