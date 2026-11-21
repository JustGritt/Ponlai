import type { SeaLook } from '../../render/setup.ts';

// The island's sea look: the sea's large-scale detail, the distant swell and the finest slopes' share.
//
// The island's sea look (packages/island/look/: the island's numbers for the render pipeline's look
// tables, look/Look.ts; the render pipeline's stage hands them to the sea as it makes it,
// render/RenderStages.ts `createSeaAndPipeline`): the strengths of the sea's large-scale detail, the
// distant swell, and how much of the surface's slope lies in waves finer than any pixel. The sea's optical properties are physical quantities, not look, and stay in
// ocean/SeaOptics.ts.

export const SEA: SeaLook = {
	// the gusts' contrast about their mean, the slicks' and the windrows' (ocean/SeaDetail.ts)
	detail: { gust: 1, slick: 1, streak: 0.3 },
	// the distant swell under the weather's own (ocean/OceanFFT.ts): its scale under the sea
	// state's, its wind and fetch, its heading against the local sea's, its spread and its
	// spectrum's shape
	swell: { scale: 0.48, windSpeed: 6, fetch: 1200e3, directionOffset: - 20, spread: 1, swell: 0.9, peakEnhancement: 3.3, shortWavesFade: 0.1 },
	// the share of Cox and Munk's mean square slope held below every pixel, the rest in the cascades
	// (ocean/OceanFFT.ts `slopeBudget`)
	subPixelSlope: 0,
};
