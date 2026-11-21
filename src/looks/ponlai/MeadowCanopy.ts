import type { MeadowCanopyLook } from '../../render/setup.ts';

// The island's meadow canopy look: its numbers for the meadow canopy's shared uniforms.
//
// The island's meadow canopy look (packages/island/look/: the island's numbers for the render
// pipeline's look tables, look/Look.ts; the shared uniforms take them where they are made,
// render/shared.ts `createShared`).

export const MEADOW_CANOPY: MeadowCanopyLook = {
	canopy: {
		density: 5, // extinction per metre of canopy along the ray
		// The share of the key light that reaches any depth between the blades. Low, so the roots and
		// the turf between them lie deep under lit tips; the blades' tips take the light back through
		// their transmission (Grass.ts), so the lawn's mean brightness holds (at half, the former gap
		// field's mean open share, the meadow read flat).
		open: 0.15,
	},
	cast: {
		density: 3, // extinction per metre of crowns along the ray
		open: 0.25, // the share of the key that passes between the crowns' blades
	},
};
