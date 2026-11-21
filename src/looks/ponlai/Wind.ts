import type { WindLook } from '../../render/setup.ts';

// The island's wind look: how the island's plants answer the weather's wind.
//
// The island's wind look (packages/island/look/: the island's numbers for the render pipeline's
// look tables, look/Look.ts; weather/Wind.ts `windResponse` reads these): how the island's plants
// answer the weather's wind.

export const WIND: WindLook = {
	// the gusts' share of the wind's strength: a full gust doubles it
	gustGain: 1,
	// about half a clump
	gustJitter: 0.5,
	// The plants' speed by the wind 10 m up, ( 0, 0 ) to the strong level's 11 m/s: the levels
	// (light, gentle, fresh, strong: 1.5, 3.8, 7, 11) take 4, 9.5, 13 and 17.
	plantWind: [ [ 0, 0 ], [ 1.5, 4 ], [ 3.8, 9.5 ], [ 7, 13 ], [ 11, 17 ] ],
	// The strong level's full gust, 17 ( 1 + gustGain ): no level blows past it, and a fair day's
	// mean gust gives a limb a load of 0.42.
	plantSpeed: 34,
	// plants one band apart fall half a swing apart in ten swings, so a band is 0.05, and of two
	// neighbours three in four sit on different bands
	plantRateSpread: 0.15,
};
