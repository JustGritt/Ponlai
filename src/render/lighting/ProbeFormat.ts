// The ambient light's probes as data: a probe's layout, the lattice and the harmonics' basis.
//
// The ambient light's probes as data: the layout of a probe in the lattice's buffer, the lattice
// itself and the harmonics' basis the probes are projected in. The render pipeline reads this
// format (AmbientVolume.ts uploads the lattice, AmbientRelight.ts relights it, Harmonics.ts
// projects onto the same basis); whoever bakes the probes writes it (the island's
// world/AmbientProbes.ts, in the world's workers). Plain numbers that import nothing, so the
// workers and node can run it: a file the chunk worker imports must not reach three.js
// (tools/repo/verify-layers.mjs).

/** Floats per probe: nine vec4 of ( sky, bounced rgb ) coefficients, seven of the groves'
 * in-scatter (rgb per coefficient in turn, then a spare 0), three of the sea's visibility (nine
 * coefficients, then three spare 0), then ( offset xyz, valid ). */
export const PROBE_STRIDE = 80;
/** The vec4 a probe's in-scatter starts at, its sea visibility's, and its ( offset xyz, valid ). */
export const SCATTER_VEC4 = 9, SEA_VEC4 = 16, META_VEC4 = 19;

// The ambient light's probes merged into one lattice (world/AmbientProbes.ts `mergeProbes`): the box
// of the chunks that baked probes, nx x ny x nz probes of PROBE_STRIDE floats in rows ( z, then y,
// then x ), the probe ( i, j, k ) at ( x0, y0, z0 ) + ( i, j, k ) spacing; then ny more, the open
// sea's probe at each level, for every point outside the box (and the gaps inside it, which hold
// copies of it). Invalid probes (deep in a solid) hold their neighbours' mean, filled by the merge.
export interface AmbientLattice { x0: number; y0: number; z0: number; spacing: number; nx: number; ny: number; nz: number; probes: Float32Array }

/** The nine real spherical harmonics up to degree 2 at a unit direction, in three.js's
 * SphericalHarmonics3 order and normalisation (y up), into out[ o .. o + 8 ]. */
export function shBasis( x: number, y: number, z: number, out: Float64Array | Float32Array, o: number ): void {

	out[ o ] = 0.282095;
	out[ o + 1 ] = 0.488603 * y;
	out[ o + 2 ] = 0.488603 * z;
	out[ o + 3 ] = 0.488603 * x;
	out[ o + 4 ] = 1.092548 * x * y;
	out[ o + 5 ] = 1.092548 * y * z;
	out[ o + 6 ] = 0.315392 * ( 3 * z * z - 1 );
	out[ o + 7 ] = 1.092548 * x * z;
	out[ o + 8 ] = 0.546274 * ( x * x - y * y );

}
