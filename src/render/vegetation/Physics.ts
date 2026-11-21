// The vegetation's shapes under load: a thin elastic strand bent by its own weight.
//
// The vegetation's shapes under load: a blade or a stem is a thin elastic strand clamped at its
// root, bent by its own weight (and, to come, by the wind's pressure) against its stiffness, and
// held off whatever it lies on. A kind that takes a physical shape would read its numbers and
// curves here. No module imports this today: the meadow's blades (Grass.ts) are a quadratic
// Bezier with a random arch (its `BEND`), not a physical shape. The family of free shapes
// (`drapeTable`) and a shape solved against a wall (`strandShape`, `drape` with a wall) are kept
// for the hanging kinds to come.
//
// Model (planar elastica of a cantilever, the large-deflection beam; Jahrmann & Wimmer, "Responsive
// Real-Time Grass Rendering for General 3D Scenes", 2017, take the same forces for grass):
// - A strand of length L (m) leaves its root at an angle (rad above the horizontal) and is
//   inextensible. At arc length s its curvature is M(s) / EI(s): M (N m) the moment about the
//   point at s of every load beyond it (its weight, contact forces), EI (N m^2) its flexural
//   rigidity there. It bends in the vertical plane of its root's direction (gravity has no
//   sideways part).
// - Its cross-section is a blade of width b(u) = b0 sqrt( 1 - u^2 ) (sword-shaped, as the meadow's
//   and the rim's blades are drawn; u = s / L) and thickness h(u) = h0 ( 1 - u + taper u ), so
//   EI = E f b h^3 / 12 and the weight per metre w = rho g b h. f stiffens a blade folded along its
//   midrib (a V section against a flat strip). b0 cancels out of the shape: it is governed by the
//   gravity number G = w0 L^3 / EI0 = 12 rho g L^3 / ( E f h0^2 ) alone, with the taper. G grows
//   with the cube of the length, so a long blade droops to hang and a short one barely bends:
//   at the defaults G is about 8 at half a metre, 0.5 at 0.2 m and 60 at a metre.
// - Shapes are equilibria, not simulations: a function of G, the root angle and what it lies on,
//   so they are the same whatever came before (ARCHITECTURE.md, rule 4). A free strand's shape is
//   read from a family of curves solved once (`drapeTable`, over G and the root angle) and
//   interpolated; one that meets the wall is solved again with the wall pushing it out
//   (`drape`), from that shape, so it lies on the stone along a smooth arc, bridging small ledges
//   as a stiff blade does, without passing into it.
// - Units: SI (m, kg, N, Pa). Shapes come out in units of the strand's length, x along its root's
//   horizontal direction and y up, from the root.

/** Standard gravity (m/s^2). */
export const GRAVITY = 9.81;

/** A strand's material and section: Young's modulus E (Pa), density rho (kg/m^3), thickness h0 at
 * the root (m), the fold's stiffening f and the thickness at the tip as a share of the root's. */
export interface StrandMaterial { modulus: number; density: number; thickness: number; stiffening: number; taper: number }

/** A grass blade: a fresh leaf's modulus and density (about 1.5 GPa, 900 kg/m^3), 0.4 mm thick at
 * the root, thinning to 40 % at the tip, its midrib's fold seven times as stiff as a flat strip. */
export const GRASS_BLADE: StrandMaterial = { modulus: 1.5e9, density: 900, thickness: 4e-4, stiffening: 7, taper: 0.4 };

/** The strand's gravity number G = w0 L^3 / EI0 at `length` (m); `stiffness` scales E f (a blade's
 * own variation). */
export function gravityNumber( material: StrandMaterial, length: number, stiffness = 1 ): number {

	return 12 * material.density * GRAVITY * length ** 3 / ( material.modulus * material.stiffening * stiffness * material.thickness ** 2 );

}

/** EI at `u` (0 root, 1 tip) over EI at the root. */
export function rigidityShare( u: number, taper: number ): number {

	const h = 1 - u + taper * u;
	return Math.sqrt( Math.max( 1 - u * u, 0 ) ) * h * h * h;

}

/** The weight per metre at `u` over the root's. */
export function weightShare( u: number, taper: number ): number {

	return Math.sqrt( Math.max( 1 - u * u, 0 ) ) * ( 1 - u + taper * u );

}

/** Where a strand may lie: the least x (in units of its length, along its root's direction) at
 * height y (the same units, from the root), or -Infinity where nothing bounds it. */
export type Wall = ( y: number ) => number;

// How fast the wall's push on a node grows per unit of its depth in the stone, per step (in units
// of the strand's weight).
const CONTACT = 4;

/** Solves a strand's shape in place: `phi` holds its `phi.length` segments' angles (rad above the
 * horizontal), from a first guess to the equilibrium, by a relaxed fixed-point iteration on the
 * curvature (the moment of the loads beyond each node over its rigidity), until no angle is
 * more than 1e-4 rad from the one its curvature gives and no node lies more than a thousandth of
 * the length inside `wall`, or `iterations` steps. Returns the last step's largest such gap. */
export function drape( gravity: number, angle: number, taper: number, phi: Float64Array, wall?: Wall, iterations = 80 ): number {

	const n = phi.length, ds = 1 / n;
	const x = new Float64Array( n + 1 ), y = new Float64Array( n + 1 ), kappa = new Float64Array( n + 1 ), push = new Float64Array( n + 1 );
	// A full step overshoots for a floppy strand (a lower angle shortens the arms, which raises it
	// again): the angles approach the ones their curvature gives a share at a time.
	const relax = Math.max( 1 / ( 1 + 0.08 * gravity ), 0.03 );
	let change = 0;
	for ( let it = 0; it < iterations; it ++ ) {

		for ( let i = 0; i < n; i ++ ) {

			x[ i + 1 ] = x[ i ] + Math.cos( phi[ i ] ) * ds;
			y[ i + 1 ] = y[ i ] + Math.sin( phi[ i ] ) * ds;

		}
		// The wall's push on each node (horizontal, out) grows while the node is inside it and
		// eases while it is clear, never pulling: it settles at the force that holds the node on the
		// stone, as a support does, so the strand lies on it without a stiff spring's chatter.
		let inside = 0;
		if ( wall ) for ( let i = 1; i <= n; i ++ ) {

			const depth = wall( y[ i ] ) - x[ i ];
			push[ i ] = Math.max( 0, push[ i ] + CONTACT * relax * depth * gravity * ds );
			inside = Math.max( inside, depth );

		}
		// the moment about each node of the loads beyond it, by sums from the tip: each segment's
		// weight at its middle, each node's push
		let fy = 0, fx = 0, xfy = 0, yfx = 0;
		kappa[ n ] = 0;
		for ( let i = n - 1; i >= 0; i -- ) {

			const w = - weightShare( ( i + 0.5 ) * ds, taper ) * ds * gravity;
			fy += w; xfy += ( x[ i ] + x[ i + 1 ] ) * 0.5 * w;
			fx += push[ i + 1 ]; yfx += y[ i + 1 ] * push[ i + 1 ];
			kappa[ i ] = ( xfy - x[ i ] * fy - ( yfx - y[ i ] * fx ) ) / Math.max( rigidityShare( i * ds, taper ), 1e-3 );

		}
		let theta = angle;
		change = 0;
		for ( let i = 0; i < n; i ++ ) {

			const next = theta + ds * ( kappa[ i ] + kappa[ i + 1 ] ) * 0.5;
			const d = ( theta + next ) * 0.5 - phi[ i ];
			phi[ i ] += d * relax;
			change = Math.max( change, Math.abs( d ) );
			theta = next;

		}
		if ( change < 1e-4 && inside < 1e-3 ) break;

	}
	return change;

}

// The family of free shapes: the segments' angles over G (log spaced) and the root angle.
const TABLE_G = [ 0.05, 400 ], TABLE_GN = 37;
const TABLE_A = [ - 1.2, 1.4 ], TABLE_AN = 27;
/** The family's grid (`drapeTable`): gravity numbers from g[ 0 ] to g[ 1 ], gn of them log spaced,
 * by root angles (rad) from a[ 0 ] to a[ 1 ], an of them evenly spaced. */
export const DRAPE_TABLE = { g: TABLE_G, gn: TABLE_GN, a: TABLE_A, an: TABLE_AN } as const;
const tables = new Map<string, Float64Array>();

/** The free shapes of a strand of `segments` and `taper`, solved once per pair: TABLE_GN gravity
 * numbers by TABLE_AN root angles, each its segments' angles. */
export function drapeTable( segments: number, taper: number ): Float64Array {

	const key = `${ segments }:${ taper }`;
	let table = tables.get( key );
	if ( table ) return table;
	table = new Float64Array( TABLE_GN * TABLE_AN * segments );
	const phi = new Float64Array( segments );
	for ( let a = 0; a < TABLE_AN; a ++ ) {

		const angle = TABLE_A[ 0 ] + ( TABLE_A[ 1 ] - TABLE_A[ 0 ] ) * a / ( TABLE_AN - 1 );
		phi.fill( angle );
		// from stiff to floppy, each from the last's shape
		for ( let g = 0; g < TABLE_GN; g ++ ) {

			const gravity = TABLE_G[ 0 ] * ( TABLE_G[ 1 ] / TABLE_G[ 0 ] ) ** ( g / ( TABLE_GN - 1 ) );
			drape( gravity, angle, taper, phi, undefined, 4000 );
			table.set( phi, ( g * TABLE_AN + a ) * segments );

		}

	}
	tables.set( key, table );
	return table;

}

/** A free strand's segment angles into `phi` (its length is the segment count), interpolated in
 * the family of shapes (`drapeTable`) at gravity number `gravity` and root angle `angle` (rad). */
export function freeDrape( gravity: number, angle: number, taper: number, phi: Float64Array ): void {

	const n = phi.length, table = drapeTable( n, taper );
	const gl = Math.log( Math.min( Math.max( gravity, TABLE_G[ 0 ] ), TABLE_G[ 1 ] ) / TABLE_G[ 0 ] ) / Math.log( TABLE_G[ 1 ] / TABLE_G[ 0 ] ) * ( TABLE_GN - 1 );
	const al = ( Math.min( Math.max( angle, TABLE_A[ 0 ] ), TABLE_A[ 1 ] ) - TABLE_A[ 0 ] ) / ( TABLE_A[ 1 ] - TABLE_A[ 0 ] ) * ( TABLE_AN - 1 );
	const g0 = Math.min( Math.floor( gl ), TABLE_GN - 2 ), a0 = Math.min( Math.floor( al ), TABLE_AN - 2 );
	const fg = gl - g0, fa = al - a0;
	const at = ( g: number, a: number, i: number ) => table[ ( g * TABLE_AN + a ) * n + i ];
	// the root angle outside the table's range turns the whole shape by the difference
	const shift = angle - ( TABLE_A[ 0 ] + ( TABLE_A[ 1 ] - TABLE_A[ 0 ] ) * al / ( TABLE_AN - 1 ) );
	for ( let i = 0; i < n; i ++ ) {

		const lo = at( g0, a0, i ) + ( at( g0, a0 + 1, i ) - at( g0, a0, i ) ) * fa;
		const hi = at( g0 + 1, a0, i ) + ( at( g0 + 1, a0 + 1, i ) - at( g0 + 1, a0, i ) ) * fa;
		phi[ i ] = lo + ( hi - lo ) * fg + shift;

	}

}

/** A strand's rest shape: its segments' angles into `phi` (its length is the segment count) and
 * its nodes' ( x, y ) into `nodes` (2 ( segments + 1 ) values, in units of its length, from the
 * root), free (`freeDrape`) or, where the free shape passes into `wall`, solved against it
 * (`drape`). Returns whether it touches the wall. */
export function strandShape( gravity: number, angle: number, taper: number, phi: Float64Array, nodes: Float64Array, wall?: Wall ): boolean {

	freeDrape( gravity, angle, taper, phi );
	const n = phi.length;
	const place = () => {

		let inside = false;
		nodes[ 0 ] = 0; nodes[ 1 ] = 0;
		for ( let i = 0; i < n; i ++ ) {

			nodes[ i * 2 + 2 ] = nodes[ i * 2 ] + Math.cos( phi[ i ] ) / n;
			nodes[ i * 2 + 3 ] = nodes[ i * 2 + 1 ] + Math.sin( phi[ i ] ) / n;
			if ( wall && nodes[ i * 2 + 2 ] < wall( nodes[ i * 2 + 3 ] ) ) inside = true;

		}
		return inside;

	};
	if ( ! place() ) return false;
	drape( gravity, angle, taper, phi, wall );
	place();
	return true;

}
