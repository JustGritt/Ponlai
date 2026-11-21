// What the sun shadow and the refracted sun take from a mesh, as plain data.
//
// What the render pipeline's sun shadow and refracted sun take from a mesh, as data: the caster
// columns (SunShadow.ts fits its casters' box and grid from them) and the subset of triangles that
// reach under the water (RefractedSun.ts draws only those). The render pipeline reads this format;
// whoever derives it from its meshes writes it (the island's world/ChunkDerived.ts, in the world's
// workers). Plain numbers that import nothing, so the workers and node can run it: a file the chunk
// worker imports must not reach three.js (tools/repo/verify-layers.mjs).

// Caster triangles reaching no lower than this above the water level cannot write the refracted map
// (its material discards everything above the level), so the proxies leave them out. The margin only
// guards a level that moves: RefractedSun cuts again once the level has moved by half of it.
export const LEVEL_MARGIN = 0.5;

/** The lowest and highest point in each cell of a world-anchored lattice of `cell` (m): cells
 * [ x0, x0 + nx cell ) x [ z0, z0 + nz cell ), row by row; +-Infinity where no point falls. */
export interface CasterColumns { x0: number; z0: number; cell: number; nx: number; nz: number; low: Float32Array; high: Float32Array }

/** The index of a mesh's triangles that reach below `level` + LEVEL_MARGIN, in their order; null
 * when every triangle does (the map then draws the whole mesh). */
export interface SubmergedSubset { level: number; index: Uint32Array | null }
