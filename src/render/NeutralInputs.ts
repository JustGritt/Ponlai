// The render pipeline's neutral inputs: plain worlds for a scene that has no world of its own.
//
// The render pipeline's neutral inputs: what a scene hands the render pipeline's stages where it
// has no world of its own to derive them from, in the formats a world gives them in (the ambient
// light's probes: lighting/ProbeFormat.ts; the ground the bounce reads: lighting/GroundBounce.ts
// `BounceGround`; the scene's fields and the shore's field: RenderStages.ts `SceneFields`,
// RenderTypes.ts `ShoreField`). They describe the plainest worlds there are: level ground of one
// albedo under an open sky, or the open sea far from any shore over a level floor, and fields that
// hold one value everywhere. The test scenes are built on them (app/scenes/); a world of its own
// (the island) gives its own and loads none of this.
//
// - The probes (`neutralLattice`): every probe sees the whole sky over the horizon and the level
//   ground under it, which sees the whole sky itself, so under a uniform white sky it bounces its
//   albedo in every direction below the horizon. Both are a hemisphere's indicator, whose
//   projection on the harmonics is exact: of the nine coefficients only the constant and the
//   vertical one are not zero (the three of the second band integrate to zero over a hemisphere
//   about the y axis). No probe sees the sea, none is moved and all are valid. A probe does not
//   depend on where it stands, so the lattice is as small as the format allows and the probes of
//   the points outside it are the same: the ground has no end.
// - The ground (`neutralBounceGround`): its albedo, on the land and under the sea alike, and its
//   height (the open sea's: its floor's).
// - The fields (`neutralFields`): the shore's sea-level distance and the floor's height, one value
//   each over a square round the origin (far inland: a large negative distance; open water: a
//   large positive one), as textures of the formats a world's are, with the transform that maps
//   the ground to them and the shore's field as plain data. The open sea is the same call: a shore
//   farther than any wave's reach (OPEN_SEA_SHORE), so no shore wave, lee or coastal damping is
//   anywhere, and the floor at one depth, so the water is equally deep everywhere.
// - The open sea's probes (`neutralSeaLattice`): what the island's probes hold far from its land,
//   level by level: above the mean level the sky over the horizon and the sea under it (the sea's
//   visibility, which the relight lights as a level bounce surface); under it the sky through the
//   surface and the deep water's body below (its irradiance reflectance, ocean/SeaOptics.ts
//   `bodyReflectance`, as bounced light).
import * as THREE from 'three/webgpu';
import { float, texture, vec3 } from 'three/tsl';
import { META_VEC4, PROBE_STRIDE, SEA_VEC4, type AmbientLattice } from './lighting/ProbeFormat.ts';
import type { BounceGround } from './lighting/GroundBounce.ts';
import { bodyReflectance } from './ocean/SeaOptics.ts';
import type { Bounds, ShoreField } from './RenderTypes.ts';

/** A linear albedo. */
export type Albedo = readonly [ number, number, number ];

// a hemisphere's indicator about the y axis on the harmonics (ProbeFormat.ts `shBasis`): the
// constant's coefficient, 0.282095 x 2 pi, and the vertical one's, 0.488603 x pi (negative for the
// lower half)
const HEMISPHERE_0 = 0.282095 * 2 * Math.PI, HEMISPHERE_Y = 0.488603 * Math.PI;

/** The probes of level ground of `albedo` at the height `ground` (m) under an open sky: a lattice
 * of 2 x 2 columns of `levels` levels, `spacing` (m) apart, its lowest level half a cell over
 * the ground, round the origin. */
export function neutralLattice( { ground, albedo, spacing = 2, levels = 4 }: { ground: number; albedo: Albedo; spacing?: number; levels?: number } ): AmbientLattice {

	const probe = new Float32Array( PROBE_STRIDE );
	// the constant: ( sky visibility, bounced r, g, b ); the vertical one: the sky above, the ground below
	probe.set( [ HEMISPHERE_0, HEMISPHERE_0 * albedo[ 0 ], HEMISPHERE_0 * albedo[ 1 ], HEMISPHERE_0 * albedo[ 2 ] ], 0 );
	probe.set( [ HEMISPHERE_Y, - HEMISPHERE_Y * albedo[ 0 ], - HEMISPHERE_Y * albedo[ 1 ], - HEMISPHERE_Y * albedo[ 2 ] ], 4 );
	probe[ META_VEC4 * 4 + 3 ] = 1;
	const nx = 2, ny = levels, nz = 2;
	// the box's probes, then one per level for the points outside it
	const count = nx * ny * nz + ny;
	const probes = new Float32Array( count * PROBE_STRIDE );
	for ( let k = 0; k < count; k ++ ) probes.set( probe, k * PROBE_STRIDE );
	return { x0: - spacing / 2, y0: ground + spacing / 2, z0: - spacing / 2, spacing, nx, ny, nz, probes };

}

/** The shore's sea-level distance (m) that puts every point of the open sea far from any shore:
 * past the reach of the shore's waves, its lee and its damping of the short waves. */
export const OPEN_SEA_SHORE = 1000;

/** The probes of the open sea, its mean level at `sea` (m): a lattice of 2 x 2 columns of `levels`
 * levels, `spacing` (m) apart, its lowest level a cell under the sea, round the origin. */
export function neutralSeaLattice( { sea = 0, spacing = 2, levels = 4 }: { sea?: number; spacing?: number; levels?: number } = {} ): AmbientLattice {

	const body = bodyReflectance();
	const level = ( y: number ) => {

		const probe = new Float32Array( PROBE_STRIDE );
		// the sky over the horizon, above the water and through its surface under it alike
		probe[ 0 ] = HEMISPHERE_0;
		probe[ 4 ] = HEMISPHERE_Y;
		if ( y >= sea ) {

			// the sea under the horizon: its visibility
			probe[ SEA_VEC4 * 4 ] = HEMISPHERE_0;
			probe[ SEA_VEC4 * 4 + 1 ] = - HEMISPHERE_Y;

		} else {

			// the water's body under the horizon: bounced, at its reflectance
			for ( let c = 0; c < 3; c ++ ) {

				probe[ 1 + c ] = HEMISPHERE_0 * body[ c ];
				probe[ 5 + c ] = - HEMISPHERE_Y * body[ c ];

			}

		}
		probe[ META_VEC4 * 4 + 3 ] = 1;
		return probe;

	};
	const nx = 2, ny = levels, nz = 2, y0 = sea - spacing;
	const probes = new Float32Array( ( nx * ny * nz + ny ) * PROBE_STRIDE );
	// the box's probes in rows ( z, then y, then x ), then one per level for the points outside it
	for ( let k = 0; k < nz; k ++ ) for ( let j = 0; j < ny; j ++ ) for ( let i = 0; i < nx; i ++ ) probes.set( level( y0 + j * spacing ), ( ( k * ny + j ) * nx + i ) * PROBE_STRIDE );
	for ( let j = 0; j < ny; j ++ ) probes.set( level( y0 + j * spacing ), ( nx * ny * nz + j ) * PROBE_STRIDE );
	return { x0: - spacing / 2, y0, z0: - spacing / 2, spacing, nx, ny, nz, probes };

}

/** Level ground of `albedo` at the height `ground` (m), as the ground bounce's bake reads it. */
export function neutralBounceGround( { ground, albedo }: { ground: number; albedo: Albedo } ): BounceGround {

	return {
		land: () => vec3( ...albedo ),
		floor: () => vec3( ...albedo ),
		height: () => float( ground ),
	};

}

/** The scene's fields at one value each over the square of `half` (m) round the origin, in
 * `cells` cells a side: the shore's sea-level distance `shore` (m, negative over the land) and
 * the floor's height `floor` (m). Sets `xform` (shared.shoreXform's value) to the square. */
export function neutralFields( { half, cells = 32, shore, floor, xform }: { half: number; cells?: number; shore: number; floor: number; xform: THREE.Vector4 } ): {
	fields: { shore: THREE.TextureNode; floor: THREE.TextureNode };
	shoreData: ShoreField;
	bounds: Bounds;
} {

	const field = ( value: number, name: string ) => {

		const data = new Float32Array( cells * cells ).fill( value );
		const tex = new THREE.DataTexture( data, cells, cells, THREE.RedFormat, THREE.FloatType );
		tex.minFilter = tex.magFilter = THREE.LinearFilter;
		tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
		tex.generateMipmaps = false;
		tex.needsUpdate = true;
		tex.name = name;
		return { data, node: texture( tex ) };

	};
	const shoreField = field( shore, 'shore-distance' ), floorField = field( floor, 'seabed-height' );
	const cell = 2 * half / cells;
	xform.set( - half, - half, 1 / ( 2 * half ), 1 / ( 2 * half ) );
	return {
		fields: { shore: shoreField.node, floor: floorField.node },
		shoreData: { data: shoreField.data, nx: cells, nz: cells, x0: - half, z0: - half, cell },
		bounds: { x0: - half, z0: - half, x1: half, z1: half },
	};

}
