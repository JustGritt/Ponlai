// Plain data types the render pipeline reads and the island fills: a ground box and a ground field.
//
// Plain data types the render pipeline reads and the island fills: a box on the ground and a field
// over a grid of the ground. Types only, importing nothing, so the world's workers and node can use
// them: a file the chunk worker imports must not reach three.js (tools/repo/verify-layers.mjs).

/** A box on the ground (m). */
export interface Bounds { x0: number; z0: number; x1: number; z1: number }

/** A field over a grid of the ground: nx x nz cells of `cell` (m) from ( x0, z0 ), row by row
 * (the shore's distance field and the fields laid out like it). */
export interface ShoreField { data: Float32Array; nx: number; nz: number; x0: number; z0: number; cell: number }
