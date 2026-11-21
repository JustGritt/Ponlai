/*!
 * Adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater), src/world/fish/FishGeometry.js.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';

// A fish's mesh from its anatomy table: a lofted body and fins as membranes between rays.
//
// Copied unchanged from Ponlai's island (packages/island/life/FishGeometry.ts at d5ad84b6), for the
// waterfall's salmon (src/falls/salmon.ts).
//
// A fish's mesh from its anatomy table (tidewater's FishGeometry, 4811ba4: the body lofted from
// superellipse rings, the fins as membranes between rays), cut to what a swimming fish seen from
// the island needs: no mouth cavity, eye domes, cut faces or poses.
//
// Local frame, the total length 1: the snout at z = +0.5, the tip of the tail fin at z = -0.5,
// the back at +y, the body's axis on y = 0.
//
// Paired fins can be wings (`Wing`): rays fanning out sideways in one plane, spread as in a glide,
// which the material folds back along the flank about the fin's root (FishMaterial.ts). Tidewater's
// paired fins fan in the vertical plane and stay spread.
//
// aData per vertex:
//   x  the position along the fish (0 snout .. 1 tail tip): the swimming wave
//   y  the part (PART)
//   z, w  body: arc length round the girth from the back's midline, and the height fraction (-1
//         belly .. 1 back); fins: the position along the ray (0 root .. 1 edge) and the ray
//         coordinate (an integer on the rays, 0 the leading ray)

export const PART = { BODY: 0, DORSAL: 1, ANAL: 2, CAUDAL: 3, PECTORAL: 4, PELVIC: 5 } as const;

type Profile = readonly ( readonly [ number, number ] )[];

/** A midline fin: rays from `from` to `to` along the body (u), each `h` high over its segment. */
export interface MidlineFin { from: number; to: number; rays: number; h: Profile; rake: readonly [ number, number ]; notch: number }

/** A paired fin spread as a wing: `rays` from a root at `u` and height `y` (fractions of the total
 * length), the leading ray at `lead` and the last at `last` (rad, from the body's axis toward the
 * tail, in the wing's plane), `len` the leading ray's length, falling to `len * tail` at the
 * last; `dihedral` tilts the plane up from the horizontal (rad). */
export interface Wing { u: number; y: number; rays: number; lead: number; last: number; len: number; tail: number; dihedral: number }

/** The anatomy, the total length 1 (tidewater's FishSpecies fields, those kept here). */
export interface Anatomy {
	// fraction of the total length up to the tail fin's base
	body: number;
	// the dorsal height, ventral depth and half width over u (0 snout .. 1 the tail fin's base)
	top: Profile; bot: Profile; wid: Profile;
	// the cross-section's exponent (2 an ellipse, > 2 boxier)
	sec: number;
	dorsal: readonly MidlineFin[]; anal: readonly MidlineFin[];
	pectoral: Wing; pelvic: Wing;
	// forked tail: the lobes' length and span, the middle rays' length relative to the lobes, and
	// the lower lobe's length relative to the upper
	caudal: { len: number; span: number; fork: number; lower: number; rays: number };
}

const TAU = Math.PI * 2;

/** A smooth piecewise profile through [ u, value ] pairs of increasing u. */
export function prof( pts: Profile, u: number ): number {

	if ( u <= pts[ 0 ][ 0 ] ) return pts[ 0 ][ 1 ];
	for ( let i = 1; i < pts.length; i ++ ) {

		if ( u <= pts[ i ][ 0 ] ) {

			const a = pts[ i - 1 ], b = pts[ i ];
			const t = ( u - a[ 0 ] ) / ( b[ 0 ] - a[ 0 ] );
			const s = t * t * ( 3 - 2 * t );
			return a[ 1 ] + ( b[ 1 ] - a[ 1 ] ) * ( 0.5 * t + 0.5 * s );

		}

	}
	return pts[ pts.length - 1 ][ 1 ];

}

const lerp = ( a: number, b: number, t: number ) => a + ( b - a ) * t;

class MeshData {

	pos: number[] = [];
	dat: number[] = [];
	idx: number[] = [];

	v( x: number, y: number, z: number, d0: number, d1: number, d2: number, d3: number ): number {

		this.pos.push( x, y, z );
		this.dat.push( d0, d1, d2, d3 );
		return this.pos.length / 3 - 1;

	}

	tri( a: number, b: number, c: number ): void {

		this.idx.push( a, b, c );

	}

	// a b / d c, counter-clockwise seen from the front
	quad( a: number, b: number, c: number, d: number ): void {

		this.idx.push( a, b, d, b, c, d );

	}

	build(): THREE.BufferGeometry {

		const g = new THREE.BufferGeometry();
		g.setAttribute( 'position', new THREE.Float32BufferAttribute( this.pos, 3 ) );
		g.setAttribute( 'aData', new THREE.Float32BufferAttribute( this.dat, 4 ) );
		g.setIndex( this.idx );
		g.computeVertexNormals();
		g.computeBoundingSphere();
		return g;

	}

}

// ------------------------------------------------------------------ body

interface Section { T: number; B: number; W: number; e: number }

// the cross-section at u (0 snout .. 1 the tail fin's base)
function section( S: Anatomy, u: number ): Section {

	return { T: prof( S.top, u ), B: prof( S.bot, u ), W: prof( S.wid, u ), e: 2 / S.sec };

}

// the point on the outline at angle phi (0 the back, PI / 2 the right flank (+x), PI the belly)
function outline( c: Section, phi: number ): [ number, number ] {

	const s = Math.sin( phi ), k = Math.cos( phi );
	return [ c.W * Math.sign( s ) * Math.pow( Math.abs( s ), c.e ), ( k >= 0 ? c.T : c.B ) * Math.sign( k ) * Math.pow( Math.abs( k ), c.e ) ];

}

// the arc length along the outline from the back's midline to angle phi (0 .. PI)
function arcTo( c: Section, phi: number ): number {

	const n = 16;
	let [ px, py ] = outline( c, 0 ), s = 0;
	for ( let i = 1; i <= n; i ++ ) {

		const [ qx, qy ] = outline( c, phi * i / n );
		s += Math.hypot( qx - px, qy - py );
		px = qx; py = qy;

	}
	return s;

}

/** The half width of the body's surface at height y on the section (0 outside). */
function surfaceX( c: Section, y: number ): number {

	const H = y >= 0 ? c.T : c.B;
	const r = Math.min( 1, Math.abs( y ) / Math.max( H, 1e-4 ) );
	const n = 2 / c.e;
	return c.W * Math.pow( Math.max( 0, 1 - Math.pow( r, n ) ), 1 / n );

}

function buildBody( M: MeshData, S: Anatomy, rings: number, around: number ): void {

	const L = S.body;
	const us: number[] = [];
	for ( let i = 0; i <= rings; i ++ ) {

		const t = i / rings;
		us.push( 0.6 * Math.pow( t, 1.55 ) + 0.4 * t ); // denser at the head

	}
	us[ 0 ] = 0.02;
	const ids: number[][] = [];
	for ( const u of us ) {

		const c = section( S, u ), z = 0.5 - u * L;
		const ring: number[] = [];
		for ( let j = 0; j < around; j ++ ) {

			const phi = TAU * j / around;
			const [ x, y ] = outline( c, phi );
			const h = y >= 0 ? y / Math.max( c.T, 1e-4 ) : y / Math.max( c.B, 1e-4 );
			ring.push( M.v( x, y, z, u * L, PART.BODY, arcTo( c, phi > Math.PI ? TAU - phi : phi ), h ) );

		}
		ids.push( ring );

	}
	for ( let i = 0; i < ids.length - 1; i ++ ) for ( let j = 0; j < around; j ++ ) {

		const j1 = ( j + 1 ) % around;
		M.quad( ids[ i ][ j ], ids[ i ][ j1 ], ids[ i + 1 ][ j1 ], ids[ i + 1 ][ j ] );

	}
	// the snout's tip and the end under the tail fin
	const first = ids[ 0 ], last = ids[ ids.length - 1 ];
	const tip = M.v( 0, 0, 0.5, 0, PART.BODY, 0, 0 );
	const end = M.v( 0, 0, 0.5 - L - 0.004, L, PART.BODY, 0, 0 );
	for ( let j = 0; j < around; j ++ ) {

		const j1 = ( j + 1 ) % around;
		M.tri( tip, first[ j1 ], first[ j ] );
		M.tri( end, last[ j ], last[ j1 ] );

	}

}

// ------------------------------------------------------------------ fins

interface Ray { b: number[]; t: number[]; ub: number; ut: number }

// a membrane stretched between rays, dipping between their tips; front and back faces
function membrane( M: MeshData, part: number, rays: Ray[], notch: number, rows: number, rayIds: number[], flip: boolean ): void {

	const cols: ( Ray & { dip: number; id: number } )[] = [];
	const mid = ( p: number[], q: number[] ) => [ ( p[ 0 ] + q[ 0 ] ) / 2, ( p[ 1 ] + q[ 1 ] ) / 2, ( p[ 2 ] + q[ 2 ] ) / 2 ];
	for ( let k = 0; k < rays.length; k ++ ) {

		cols.push( { ...rays[ k ], dip: 0, id: rayIds[ k ] } );
		if ( k < rays.length - 1 ) {

			const a = rays[ k ], b = rays[ k + 1 ];
			cols.push( { b: mid( a.b, b.b ), t: mid( a.t, b.t ), ub: ( a.ub + b.ub ) / 2, ut: ( a.ut + b.ut ) / 2, dip: notch, id: ( rayIds[ k ] + rayIds[ k + 1 ] ) / 2 } );

		}

	}
	for ( const back of [ false, true ] ) {

		const grid = cols.map( ( c ) => {

			const col: number[] = [];
			for ( let r = 0; r <= rows; r ++ ) {

				const t = ( r / rows ) * ( 1 - c.dip );
				col.push( M.v( lerp( c.b[ 0 ], c.t[ 0 ], t ), lerp( c.b[ 1 ], c.t[ 1 ], t ), lerp( c.b[ 2 ], c.t[ 2 ], t ), lerp( c.ub, c.ut, t ), part, t, c.id ) );

			}
			return col;

		} );
		for ( let k = 0; k < grid.length - 1; k ++ ) for ( let r = 0; r < rows; r ++ ) {

			const a = grid[ k ][ r ], b = grid[ k + 1 ][ r ], c = grid[ k + 1 ][ r + 1 ], d = grid[ k ][ r + 1 ];
			if ( back !== flip ) M.quad( b, a, d, c );
			else M.quad( a, b, c, d );

		}

	}

}

// a subset of a fin's rays (always the first and the last)
function pickRays( n: number, max: number ): number[] {

	if ( n <= max ) return [ ...Array( n ).keys() ];
	const out: number[] = [];
	for ( let i = 0; i < max; i ++ ) out.push( Math.round( i * ( n - 1 ) / ( max - 1 ) ) );
	return out;

}

// the dorsal (sign 1) and anal (sign -1) fins on the midline
function midlineFins( M: MeshData, S: Anatomy, fins: readonly MidlineFin[], sign: number, maxRays: number ): void {

	const L = S.body, zOf = ( u: number ) => 0.5 - u * L;
	for ( const fin of fins ) {

		const ids = pickRays( fin.rays, maxRays );
		const rays = ids.map( ( k ) => {

			const t = fin.rays > 1 ? k / ( fin.rays - 1 ) : 0;
			const u = lerp( fin.from, fin.to, t );
			const c = section( S, u );
			const y0 = sign > 0 ? c.T * 0.9 : - c.B * 0.9;
			const h = prof( fin.h, t ) + ( sign > 0 ? c.T : c.B ) * 0.1;
			const rake = lerp( fin.rake[ 0 ], fin.rake[ 1 ], t );
			return { b: [ 0, y0, zOf( u ) ], t: [ 0, y0 + sign * h * Math.cos( rake ), zOf( u ) - h * Math.sin( rake ) ], ub: u * L, ut: u * L + h * Math.sin( rake ) };

		} );
		membrane( M, sign > 0 ? PART.DORSAL : PART.ANAL, rays, fin.notch, 1, ids, sign < 0 );

	}

}

// the forked tail fin, its lower lobe `lower` times the upper's length (flying fish)
function caudalFin( M: MeshData, S: Anatomy, maxRays: number ): void {

	const C = S.caudal, L = S.body;
	const zb = 0.5 - L + 0.014;
	const end = section( S, 1 );
	const hp = Math.min( end.T, end.B ) * 0.9;
	const ids = pickRays( C.rays, maxRays );
	const rays = ids.map( ( k ) => {

		const s = - 1 + 2 * k / ( C.rays - 1 ); // -1 the lower lobe .. 1 the upper
		const a = Math.abs( s );
		const len = C.len * ( C.fork + ( 1 - C.fork ) * Math.pow( a, 1.3 ) ) * ( s < 0 ? lerp( 1, C.lower, a ) : 1 );
		return { b: [ 0, hp * s, zb ], t: [ 0, C.span * s * ( s < 0 ? lerp( 1, C.lower, a * 0.6 ) : 1 ), zb - 0.014 - len ], ub: L - 0.014, ut: L + len };

	} );
	membrane( M, PART.CAUDAL, rays, 0.05, 2, ids, false );

}

// A paired fin spread as a wing: rays fanning out from the root in a plane tilted up by the
// dihedral, the leading ray nearly square to the body and the last along it.
function wing( M: MeshData, S: Anatomy, F: Wing, part: number, maxRays: number ): void {

	const L = S.body;
	const zb = 0.5 - F.u * L;
	const ids = pickRays( F.rays, maxRays );
	for ( const side of [ 1, - 1 ] ) {

		const xb = side * wingRoot( S, F );
		const rays = ids.map( ( k ) => {

			const t = F.rays > 1 ? k / ( F.rays - 1 ) : 0;
			const angle = lerp( F.lead, F.last, t );
			const len = F.len * lerp( 1, F.tail, Math.pow( t, 0.8 ) );
			// out to the side and back, lifted by the dihedral
			const out = Math.sin( angle ) * len, back = Math.cos( angle ) * len;
			const tip = [ xb + side * out * Math.cos( F.dihedral ), F.y + out * Math.sin( F.dihedral ), zb - back ];
			return { b: [ xb, F.y, zb ], t: tip, ub: F.u * L, ut: F.u * L + back };

		} );
		membrane( M, part, rays, 0.03, 2, ids, side < 0 );

	}

}

/** The distance of a wing's root from the body's axis (the pivot its rays fold about). */
export function wingRoot( S: Anatomy, F: Wing ): number {

	return surfaceX( section( S, F.u ), F.y ) * 0.92;

}

// ------------------------------------------------------------------ the fish

/** The fish's mesh. `detail` 0 is the near level (more rings and rays), 1 the far one. */
export function fishGeometry( S: Anatomy, detail: 0 | 1 = 0 ): THREE.BufferGeometry {

	const M = new MeshData();
	buildBody( M, S, detail === 0 ? 18 : 9, detail === 0 ? 14 : 8 );
	const rays = detail === 0 ? 9 : 4;
	midlineFins( M, S, S.dorsal, 1, rays );
	midlineFins( M, S, S.anal, - 1, rays );
	caudalFin( M, S, detail === 0 ? 9 : 5 );
	wing( M, S, S.pectoral, PART.PECTORAL, detail === 0 ? 9 : 5 );
	wing( M, S, S.pelvic, PART.PELVIC, detail === 0 ? 5 : 3 );
	return M.build();

}
