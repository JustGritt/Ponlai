import * as THREE from 'three/webgpu';
import { hash3i, mulberry32, type Rng } from '../util/noise.ts';
import { Growing, boundsSteps, float32Steps, runSteps } from '../util/Steps.ts';

// What every tree's mesh is built from: tubes for the wood and clumps of leaf cards.
//
// What every tree's mesh is built from (Bamboo.ts; the tubes and clumps for the trees to come): a
// mesh builder with named attributes, tubes for the wood and clumps of leaf cards for the crown.
//
// - The wood: a tube along a curve carries uv around the branch and along it (in the bark tile's
//   metres), tangents around it and a `wood` attribute (deadwood, hollow: the bark materials mix
//   by it, vegetation/TreeSurfaces.ts barkMaterial).
// - The leaves: a clump of leaf cards over a branch tip, the cards filling the clump toward its
//   shell and their normals bent out from the clump and the crown, so the clumps shade as one canopy
//   with lumps (the canopy cards of tidewater's trees, PlantGeometry.js addCard and lobeCards; the
//   technique, not its code), the inside of each clump and of the crown darker (in the cards'
//   vertex colours). A card uvs into a 2x2 atlas of sprays.
// - Determinism: a tree's form draws from one stream in a fixed order; what must not move the form
//   (a clump's cards, a tube's uv offset) draws from a stream of its own seeded by its place
//   (`streamAt`).

export const smooth = ( a: number, b: number, x: number ) => {

	const t = Math.min( Math.max( ( x - a ) / ( b - a ), 0 ), 1 );
	return t * t * ( 3 - 2 * t );

};
export const wrap = ( a: number ) => Math.atan2( Math.sin( a ), Math.cos( a ) );
/** A stream seeded by a place (cm), for what must not draw from a form's stream. */
export const streamAt = ( p: THREE.Vector3, salt: number ) => mulberry32( ( hash3i( Math.round( p.x * 100 ), Math.round( p.z * 100 ), salt ) * 4294967296 ) >>> 0 );

export const UP = new THREE.Vector3( 0, 1, 0 );

/** Indexed triangles with named per-vertex attributes (`layout`: name and width). Normals are given
 * (a `normal` in the layout) or computed over shared vertices, the pairs in `seams` (one point
 * twice, as a tube's seam has for its uv) sharing theirs. The attributes named in `interleave`
 * share one interleaved buffer: WebGPU binds at most eight vertex buffers to a pipeline, and the
 * branch's four (BRANCH_ATTRIBUTES) would take a leaf card's mesh past it. Its values grow a block
 * at a time and its geometry is made in steps (`geometrySteps`, for the growth's idle time:
 * Steps.ts), the same bits as three's one-shot calls (`geometry` runs the steps to the end). */
export class MeshBuilder {

	readonly positions = new Growing();
	readonly data = new Map<string, { size: number; values: Growing }>();
	readonly indices = new Growing();
	readonly seams = new Growing();

	private readonly interleave: readonly string[];

	constructor( layout: Record<string, number>, interleave: readonly string[] = [] ) {

		for ( const name in layout ) this.data.set( name, { size: layout[ name ], values: new Growing() } );
		this.interleave = interleave;

	}

	get count(): number {

		return this.positions.length / 3;

	}

	vertex( p: THREE.Vector3, values: Record<string, ArrayLike<number>> ): number {

		this.positions.push( p.x, p.y, p.z );
		for ( const [ name, { size, values: out } ] of this.data ) {

			const v = values[ name ];
			for ( let k = 0; k < size; k ++ ) out.push( v[ k ] );

		}
		return this.count - 1;

	}

	geometry( name: string ): THREE.BufferGeometry {

		return runSteps( this.geometrySteps( name ) );

	}

	/** The geometry in steps of a few tenths of a millisecond: what three's Float32BufferAttribute,
	 * Uint32BufferAttribute, computeVertexNormals, computeBoundingBox and computeBoundingSphere make. */
	* geometrySteps( name: string ): Generator<void, THREE.BufferGeometry> {

		const g = new THREE.BufferGeometry();
		g.setAttribute( 'position', new THREE.BufferAttribute( yield* float32Steps( this.positions ), 3 ) );
		for ( const [ key, { size, values } ] of this.data ) if ( ! this.interleave.includes( key ) ) g.setAttribute( key, new THREE.BufferAttribute( yield* float32Steps( values ), size ) );
		if ( this.interleave.length > 0 ) {

			const parts = this.interleave.map( ( key ) => ( { key, ...this.data.get( key )! } ) );
			const stride = parts.reduce( ( sum, p ) => sum + p.size, 0 );
			const array = new Float32Array( this.count * stride );
			let offset = 0;
			const buffer = new THREE.InterleavedBuffer( array, stride );
			for ( const { key, size, values } of parts ) {

				for ( let v0 = 0; v0 < this.count; v0 += STEP ) {

					for ( let v = v0, end = Math.min( v0 + STEP, this.count ); v < end; v ++ ) for ( let k = 0; k < size; k ++ ) array[ v * stride + offset + k ] = values.get( v * size + k );
					yield;

				}
				g.setAttribute( key, new THREE.InterleavedBufferAttribute( buffer, size, offset ) );
				offset += size;

			}

		}
		const index = new Uint32Array( this.indices.length );
		for ( let k = 0; k < index.length; k += STEP * 4 ) {

			for ( let i = k, end = Math.min( k + STEP * 4, index.length ); i < end; i ++ ) index[ i ] = this.indices.get( i );
			yield;

		}
		g.setIndex( new THREE.BufferAttribute( index, 1 ) );
		if ( ! this.data.has( 'normal' ) ) {

			yield* normalSteps( g );
			const n = g.getAttribute( 'normal' ) as THREE.BufferAttribute;
			const a = new THREE.Vector3(), b = new THREE.Vector3();
			for ( let k = 0; k < this.seams.length; k += 2 ) {

				const i = this.seams.get( k ), j = this.seams.get( k + 1 );
				a.fromBufferAttribute( n, i ).add( b.fromBufferAttribute( n, j ) ).normalize();
				n.setXYZ( i, a.x, a.y, a.z );
				n.setXYZ( j, a.x, a.y, a.z );
				if ( k % ( STEP * 2 ) === 0 ) yield;

			}

		}
		yield* boundsSteps( g );
		g.name = name;
		return g;

	}

}

// vertices (or triangles) a step
const STEP = 1 << 10;
const _pA = new THREE.Vector3(), _pB = new THREE.Vector3(), _pC = new THREE.Vector3();
const _nA = new THREE.Vector3(), _nB = new THREE.Vector3(), _nC = new THREE.Vector3();
const _cb = new THREE.Vector3(), _ab = new THREE.Vector3();

// An indexed geometry's vertex normals in steps: three r186's computeVertexNormals (its face
// normals summed per vertex in the triangles' order, then normalised), the same operations in the
// same order.
function* normalSteps( g: THREE.BufferGeometry ): Generator<void, void> {

	const position = g.getAttribute( 'position' ) as THREE.BufferAttribute, index = g.getIndex()!;
	const normal = new THREE.BufferAttribute( new Float32Array( position.count * 3 ), 3 );
	g.setAttribute( 'normal', normal );
	for ( let i0 = 0; i0 < index.count; i0 += STEP * 3 ) {

		for ( let i = i0, end = Math.min( i0 + STEP * 3, index.count ); i < end; i += 3 ) {

			const vA = index.getX( i ), vB = index.getX( i + 1 ), vC = index.getX( i + 2 );
			_pA.fromBufferAttribute( position, vA );
			_pB.fromBufferAttribute( position, vB );
			_pC.fromBufferAttribute( position, vC );
			_cb.subVectors( _pC, _pB );
			_ab.subVectors( _pA, _pB );
			_cb.cross( _ab );
			_nA.fromBufferAttribute( normal, vA );
			_nB.fromBufferAttribute( normal, vB );
			_nC.fromBufferAttribute( normal, vC );
			_nA.add( _cb );
			_nB.add( _cb );
			_nC.add( _cb );
			normal.setXYZ( vA, _nA.x, _nA.y, _nA.z );
			normal.setXYZ( vB, _nB.x, _nB.y, _nB.z );
			normal.setXYZ( vC, _nC.x, _nC.y, _nC.z );

		}
		yield;

	}
	for ( let v0 = 0; v0 < normal.count; v0 += STEP ) {

		for ( let v = v0, end = Math.min( v0 + STEP, normal.count ); v < end; v ++ ) {

			_nA.fromBufferAttribute( normal, v ).normalize();
			normal.setXYZ( v, _nA.x, _nA.y, _nA.z );

		}
		yield;

	}

}

/** A branch's attributes, on its wood's vertices and its leaf cards' (Deform.ts `branchBend`,
 * `sprayBend`): `branchFrom` ( B0, the branch's start in the mesh's space; 0 ) and `branchTo` ( B1, its end; its yield to the wind, 1 - its stiffness ), and the
 * sub-branch's (a card's spray) `sprayFrom` ( C0, 0 ) and `sprayTo` ( C1, its yield ). A vertex on
 * no branch (a trunk's, a card on no branch) takes zeros, and yields nothing;
 * one on no sub-branch (the wood) zeros. The hashes are taken in the shader from the ends, so a
 * limb's vertices share them. */
export const BRANCH_ATTRIBUTES = { branchFrom: 4, branchTo: 4, sprayFrom: 4, sprayTo: 4 } as const;
/** A branch's attribute values ( `branchFrom`, `branchTo` ) from B0 to B1 by its `give`
 * ( 1 - stiffness ); no branch with `from` and `to` omitted; no sub-branch. */
export function branchValues( from?: THREE.Vector3, to?: THREE.Vector3, give = 0 ): Record<keyof typeof BRANCH_ATTRIBUTES, number[]> {

	const none = [ 0, 0, 0, 0 ];
	return from && to
		? { branchFrom: [ from.x, from.y, from.z, 0 ], branchTo: [ to.x, to.y, to.z, give ], sprayFrom: none, sprayTo: none }
		: { branchFrom: none, branchTo: none, sprayFrom: none, sprayTo: none };

}

/** `branch`'s values on a card whose spray runs from C0 to C1 by its `give`. */
export function sprayValues( branch: Record<keyof typeof BRANCH_ATTRIBUTES, number[]>, from: THREE.Vector3, to: THREE.Vector3, give: number ): Record<keyof typeof BRANCH_ATTRIBUTES, number[]> {

	return { ...branch, sprayFrom: [ from.x, from.y, from.z, 0 ], sprayTo: [ to.x, to.y, to.z, give ] };

}

/** The wood's mesh layout (tube) and the leaf cards' (leafClump), with `extra` attributes. */
export const woodLayout = () => new MeshBuilder( { uv: 2, tangent: 4, wood: 2 } );
export const leafLayout = ( extra: Record<string, number> = {}, interleave: readonly string[] = [] ) => new MeshBuilder( { normal: 3, uv: 2, color: 3, leafSway: 4, ...extra }, interleave );

/** How a tube's surface departs from a round one at a point: its radius's scale, and how far it is
 * deadwood and the hollow's dark. */
export type Surface = ( theta: number, height: number ) => { k: number; dead: number; hollow: number };

export interface TubeOptions {
	r0: number; r1: number; // radius at the start and the end
	sides: number;
	step: number; // ring spacing (m)
	tile: { readonly around: number; readonly along: number }; // the bark's tile (m)
	surface?: Surface;
	twist?: number; // with a surface: the grain's turn (rad per m of height), which the uv follows
	dead?: number; // without a surface: how far it is deadwood (at its start, and at its end with `deadEnd`)
	deadEnd?: number;
	// open; closed by a rounded cap (a branch whose children carry on from inside it, so no open
	// rim shows where they leave it); drawn to a point; or broken off, ragged
	end?: 'open' | 'round' | 'point' | 'broken';
	rng?: Rng;
}

const _t = new THREE.Vector3(), _b = new THREE.Vector3(), _d = new THREE.Vector3(), _p = new THREE.Vector3(), _a = new THREE.Vector3();

/** A tube along `curve`: rings carried along it by parallel transport. `surface` reads the ring
 * angle in the world's x-z plane and the height above `foot`, so grain and bands run on across
 * tubes. Its uv wraps the bark's tile a whole number of times round it (from its first radius) and
 * runs along it, from an offset hashed from its start; a ring's last vertex repeats its first (the
 * uv's seam), their normals shared. */
export function tube( mesh: MeshBuilder, curve: THREE.Curve<THREE.Vector3>, foot: number, o: TubeOptions ): void {

	const length = curve.getLength();
	const rings = Math.max( 2, Math.ceil( length / o.step ) ) + 1;
	const first = mesh.count;
	const t0 = curve.getTangent( 0 ).normalize();
	const start = curve.getPoint( 0 );
	const wraps = Math.max( 1, Math.round( 2 * Math.PI * o.r0 / o.tile.around ) );
	const offset = streamAt( start, 31 );
	const u0 = offset(), v0 = offset();
	// a start normal off the tangent, then carried along
	const n = new THREE.Vector3( 0, 0, 1 ).cross( t0 );
	if ( n.lengthSq() < 1e-4 ) n.set( 1, 0, 0 ).cross( t0 );
	n.normalize();
	const prev = t0.clone();
	const centre = new THREE.Vector3();
	const row = o.sides + 1;
	const deadAt = ( s: number ) => ( { k: 1, dead: ( o.dead ?? 0 ) + ( ( o.deadEnd ?? o.dead ?? 0 ) - ( o.dead ?? 0 ) ) * s, hollow: 0 } );
	let along = 0;
	const ring = ( r: number, v: number, ragged: boolean ) => {

		const seam = mesh.count;
		// the bark's furrows follow the grain's twist where the surface does
		const grain = o.surface ? ( o.twist ?? 0 ) * ( centre.y - foot ) / ( 2 * Math.PI ) : 0;
		for ( let j = 0; j <= o.sides; j ++ ) {

			const a = ( j % o.sides ) / o.sides * Math.PI * 2;
			_d.copy( n ).multiplyScalar( Math.cos( a ) ).addScaledVector( _b, Math.sin( a ) );
			_a.copy( n ).multiplyScalar( - Math.sin( a ) ).addScaledVector( _b, Math.cos( a ) );
			const values = { uv: [ u0 + ( j / o.sides + grain ) * wraps, v ], tangent: [ _a.x, _a.y, _a.z, 1 ], wood: [ 0, 0 ] };
			if ( j === o.sides ) {

				// the seam: the first vertex again, with the uv's far edge
				const k = seam * 3;
				_p.set( mesh.positions.get( k ), mesh.positions.get( k + 1 ), mesh.positions.get( k + 2 ) );
				const w = mesh.data.get( 'wood' )!.values;
				values.wood = [ w.get( seam * 2 ), w.get( seam * 2 + 1 ) ];
				mesh.seams.push( seam, mesh.vertex( _p, values ) );
				continue;

			}
			const theta = Math.atan2( _d.z, _d.x );
			const f = o.surface ? o.surface( theta, centre.y - foot ) : deadAt( along );
			_p.copy( centre ).addScaledVector( _d, r * f.k );
			if ( ragged && o.rng ) _p.addScaledVector( _t, ( o.rng() - 0.3 ) * r * 1.2 );
			values.wood = [ f.dead, f.hollow ];
			mesh.vertex( _p, values );

		}

	};

	for ( let i = 0; i < rings; i ++ ) {

		const s = i / ( rings - 1 );
		along = s;
		centre.copy( curve.getPointAt( s ) );
		_t.copy( curve.getTangentAt( s ) ).normalize();
		// parallel transport: turn the normal by the tangent's turn
		const axis = _d.crossVectors( prev, _t );
		const turn = Math.asin( Math.min( 1, axis.length() ) );
		if ( turn > 1e-6 ) n.applyAxisAngle( axis.normalize(), turn );
		n.addScaledVector( _t, - n.dot( _t ) ).normalize();
		prev.copy( _t );
		_b.crossVectors( _t, n );
		let r = o.r0 + ( o.r1 - o.r0 ) * s;
		if ( o.end === 'point' ) r *= Math.max( 0.06, 1 - Math.pow( s, 3 ) );
		// a broken end's last ring is ragged along the tangent
		ring( r, v0 + s * length / o.tile.along, o.end === 'broken' && i === rings - 1 );

	}

	// a rounded cap: three more rings closing over the end, a quarter circle in profile
	let count = rings, v = v0 + length / o.tile.along;
	if ( o.end === 'round' ) {

		const end = centre.clone();
		for ( let q = 1; q <= 3; q ++ ) {

			const phi = q / 3 * Math.PI / 2 * 0.9;
			centre.copy( end ).addScaledVector( _t, o.r1 * Math.sin( phi ) * 0.8 );
			v += o.r1 * 0.4 / o.tile.along;
			ring( o.r1 * Math.cos( phi ), v, false );

		}
		count += 3;

	}

	for ( let i = 0; i < count - 1; i ++ ) for ( let j = 0; j < o.sides; j ++ ) {

		const a = first + i * row + j, b = a + 1, c = a + row, d = b + row;
		mesh.indices.push( a, b, c, b, d, c );

	}

	if ( o.end === 'broken' || o.end === 'point' || o.end === 'round' ) {

		// a cap's middle: a little sunk (a broken end), on the axis (a point), or the round cap's top
		const tip = o.end === 'round' ? centre.clone().addScaledVector( _t, o.r1 * 0.1 ) : curve.getPointAt( 1 ).addScaledVector( _t, o.end === 'broken' ? - o.r1 * 0.3 : 0 );
		const last = first + ( count - 1 ) * row;
		const f = o.end === 'round' && o.surface ? o.surface( 0, tip.y - foot ) : deadAt( 1 );
		const c = mesh.vertex( tip, { uv: [ u0 + wraps / 2, v + 0.05 ], tangent: [ n.x, n.y, n.z, 1 ], wood: [ f.dead, f.hollow ] } );
		for ( let j = 0; j < o.sides; j ++ ) mesh.indices.push( c, last + j, last + j + 1 );

	}

}

/** A crown's leaf cards: how many cards a clump holds and their side (m), the atlas's size (px, for
 * the cards' inset), how far a card faces up rather than out, how much of a clump below its middle
 * holds cards, and the sway they carry (bend: the crown's top's yield ( 1 - stiffness ) to the
 * main bend, Deform.ts mainBend, falling as the square of the height share eta below it, eta in
 * `leafSway` w; flutter: m along a card's normal, its whole card, Deform.ts leafFlutter's amount). The crown: the point under it the canopy's normals spread
 * from, its radius and its top above `foot`. `extra`: more attributes, the same on every card. */
export interface CardStyle {
	cards: number; card: number; atlas: number;
	up: readonly [ number, number ]; below: number;
	bend: number; flutter: number;
	crown: THREE.Vector3; crownRadius: number; crownTop: number; foot: number;
	extra?: Record<string, ArrayLike<number>>;
}

const _c = new THREE.Vector3(), _out = new THREE.Vector3(), _normal = new THREE.Vector3();
const _X = new THREE.Vector3(), _Y = new THREE.Vector3(), _dl = new THREE.Vector3(), _dc = new THREE.Vector3(), _nn = new THREE.Vector3();

/** A leaf clump: `style.cards` cards in a flat ellipsoid over a branch tip, toward its shell and
 * few below its middle, facing out of it and up; their normals bent out from the clump and from
 * the crown, their colour its exposure (inner and lower leaves darker), times the clump's `shade`. */
export function leafClump( leaves: MeshBuilder, centre: THREE.Vector3, rh: number, rv: number, shade: number, style: CardStyle ): void {

	const rng = streamAt( centre, 17 );
	const scale = new THREE.Vector3( rh, rv, rh );
	const inset = 2 / style.atlas;
	for ( let k = 0; k < style.cards; k ++ ) {

		let x: number, y: number, z: number;
		do {

			x = rng() * 2 - 1; y = rng() * 2 - 1; z = rng() * 2 - 1;

		} while ( x * x + y * y + z * z > 1 || y < - style.below );
		const l = Math.hypot( x, y, z ) || 1, shell = 0.45 + 0.55 * Math.sqrt( l );
		_c.set( x / l, y / l, z / l ).multiplyScalar( shell * 0.85 ).multiply( scale ).add( centre );
		_out.set( x, y, z ).normalize();
		_normal.copy( _out ).lerp( UP, style.up[ 0 ] + style.up[ 1 ] * rng() ).normalize();
		const size = style.card * ( 0.8 + 0.45 * rng() ), yaw = rng() * Math.PI * 2;
		const tile = Math.floor( rng() * 4 ), tu = ( tile % 2 ) * 0.5, tv = Math.floor( tile / 2 ) * 0.5;
		const phase = rng();
		_X.crossVectors( Math.abs( _normal.y ) < 0.95 ? UP : _a.set( 1, 0, 0 ), _normal ).normalize();
		_Y.crossVectors( _normal, _X );
		_X.multiplyScalar( Math.cos( yaw ) ).addScaledVector( _Y.clone(), Math.sin( yaw ) );
		_Y.crossVectors( _normal, _X );
		const first = leaves.count;
		for ( const [ cx, cy, u, v ] of [ [ - 0.5, - 0.5, 0, 0 ], [ 0.5, - 0.5, 1, 0 ], [ 0.5, 0.5, 1, 1 ], [ - 0.5, 0.5, 0, 1 ] ] ) {

			_p.copy( _c ).addScaledVector( _X, cx * size ).addScaledVector( _Y, cy * size );
			_dl.subVectors( _p, centre ).divide( scale );
			_dc.subVectors( _p, style.crown );
			_nn.copy( _dl ).normalize().multiplyScalar( 0.5 ).addScaledVector( _dc.clone().normalize(), 0.4 ).addScaledVector( _normal, 0.2 ).normalize();
			const exposure = Math.min( 1, 0.5 * Math.min( 1, _dl.length() ) + 0.4 * Math.min( 1, Math.hypot( _dc.x, _dc.z ) / style.crownRadius ) + 0.25 * Math.max( 0, _dc.y / _dc.length() ) );
			const k = shade * ( 0.3 + 0.7 * exposure * exposure );
			const eta = Math.min( 1, Math.max( 0, ( _p.y - style.foot ) / style.crownTop ) );
			const bend = style.bend * eta * eta;
			leaves.vertex( _p, {
				normal: [ _nn.x, _nn.y, _nn.z ],
				uv: [ tu + inset + u * ( 0.5 - 2 * inset ), tv + inset + v * ( 0.5 - 2 * inset ) ],
				color: [ k, k, k ],
				leafSway: [ bend, style.flutter, phase, eta ],
				...style.extra,
			} );

		}
		leaves.indices.push( first, first + 1, first + 2, first, first + 2, first + 3 );

	}

}

export const centroid = ( points: readonly THREE.Vector3[] ) => {

	const c = new THREE.Vector3();
	for ( const p of points ) c.add( p );
	return c.divideScalar( points.length );

};

/** Splits the tips a branch carries in two across their longer horizontal axis, a little off the
 * middle. */
export function split( points: THREE.Vector3[], rng: Rng ): [ THREE.Vector3[], THREE.Vector3[] ] {

	const c = centroid( points );
	let sxx = 0, szz = 0, sxz = 0;
	for ( const p of points ) {

		const x = p.x - c.x, z = p.z - c.z;
		sxx += x * x; szz += z * z; sxz += x * z;

	}

	const angle = 0.5 * Math.atan2( 2 * sxz, sxx - szz );
	const ax = Math.cos( angle ), az = Math.sin( angle );
	const sorted = points.slice().sort( ( a, b ) => ( a.x * ax + a.z * az ) - ( b.x * ax + b.z * az ) );
	const at = Math.min( points.length - 1, Math.max( 1, Math.round( points.length * ( 0.5 + ( rng() - 0.5 ) * 0.16 ) ) ) );
	return [ sorted.slice( 0, at ), sorted.slice( at ) ];

}
