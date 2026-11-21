import type { OcclusionLook } from '../../render/setup.ts';

// The island's ambient occlusion look: its numbers for the occlusion's shared uniforms.
//
// The island's ambient occlusion look (packages/island/look/: the island's numbers for the render
// pipeline's look tables, look/Look.ts; the shared uniforms take it where they are made,
// render/shared.ts `createShared`).

export const OCCLUSION: OcclusionLook = {
	// the deepest crevice or contact keeps a quarter of the ambient light and never turns black
	floor: 0.25,
};
