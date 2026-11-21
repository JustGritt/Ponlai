import type { LocalLightsLook } from '../../render/setup.ts';

// The island's local lights look: its numbers for the local lights' table.
//
// The island's local lights look (packages/island/look/: the island's numbers for the render
// pipeline's look tables, look/Look.ts; the local lights take it where they are made,
// render/lighting/LocalLights.ts `createLocalLights`).

export const LOCAL_LIGHTS: LocalLightsLook = {
	// The art gain over the sources' own intensity. The yunpeng's light is its glow's radiance times
	// its glowing area (Yunpeng.ts), the light such a clump would really give. At a glow that keeps the
	// stalks a soft white by night (a brighter one turned them into a blown-out flare, a lamp rather
	// than a luminous plant), that light adds about a fourteenth of the moonlit ground's brightness a
	// metre off, no pool at all. The gain is on the light alone: about fourteen times makes the ground a
	// metre from a clump about twice as bright as the moonlit open ground round it (the light adding
	// about as much as the moon gives), fading to nothing within about 6 m, while the glow stays soft.
	gain: 14,
};
