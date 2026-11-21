import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, uniform, storage, workgroupArray, workgroupBarrier, workgroupId, localId, uint, float, select, dot, min, max, ceil,
} from 'three/tsl';
import { TEMPORAL } from '../post/Antialiasing.ts';
import { jitter } from '../post/Jitter.ts';
import { runSteps } from '../util/Steps.ts';
import { hasGpuFeature } from '../three/Backend.ts';

// GPU-driven culling for an instanced vegetation kind, in a compute pass before it draws.
//
// GPU-driven culling for an instanced vegetation kind (today the plants drawn as instances of a few
// prototypes, PlantInstances.ts: the bamboo; the meadow generates its blades per view instead and
// takes only `unjitteredPadded` from here). Every record of every batch lives in one set of storage
// buffers: the kind's payload (`CullPayload`: its buffers and what it writes into a slot; a plant's
// place, form and level band) and the cull's own (a bounding sphere the kind gives, the visible
// list, the bands). Before the view camera draws the kind, a compute pass tests each record's sphere against
// the view frustum (and, where the kind folds records away by distance, `CullPayload.fold`, drops
// those the vertex stage would collapse to a point anyway), and writes the indices of the records
// that pass, in their original order, plus each batch's indirect draw arguments. The draw then
// runs the vertex stage only for those records. The compaction is a scan, not an atomic append:
// the surviving records keep the order every record drew in before, so depth ties and the
// alpha-to-coverage result are those of the full draw, and a paused frame is bit-identical.
//
// Two dispatches: the first counts each workgroup's visible records, the second sums the counts of
// the batch's earlier workgroups, scans its own and scatters. Each batch is padded to whole
// workgroups, so no workgroup spans two batches.
//
// Bands: a record far from the camera keeps only some of its parts (the meadow's blades, Grass.ts
// `tuftLod`), the rest narrowed to degenerate triangles that would still run the vertex stage. So
// each batch's visible records are sorted into a few bands by the parts they keep
// (`CullBatchInput.caps`, say 16, 12, 8, 6, 4 and 2 of a lawn tuft's 16 blades), each band drawn by its
// own indirect arguments with only its cap's parts: the band's records follow one another in the
// batch's visible list, in slot order within the band, and its draw starts at its first through
// `firstInstance` (the instance index the vertex stage reads includes it). The count pass decides
// each record's band from the same count the vertex stage computes (`CullLod.count`) and keeps it
// per slot (`bands`), so the scatter places every record by the decision the counts were made from.
// Each workgroup counts its bands in words of three 10-bit fields (a workgroup has at most 256
// records in a band), as many words as its bands need, scanned together; the scatter sums the
// earlier workgroups' counts one reduction at a time, so its workgroup memory stays under WebGPU's
// default 16 KiB with up to MAX_BANDS bands. Without `indirect-first-instance` every batch has one band,
// its first: its draw starts at 0.
//
// A kind is one pool: a batch per sort of record (the meadow's lawn, its
// thin extra tufts, each batch over its own geometry) holds every chunk's records in the chunks' order ( cj, ci ), as the
// anchors come (ChunkMerge.ts), each drawn in one draw. The buffers hold a capacity, the next power
// of two of the slots the batches fill; the dispatches cover the filled workgroups, and the slots
// past them are never visible. A grown island's pool (World growth, Growth.ts) is a new pool: its
// arrays are filled in steps (`poolSteps`) and its buffers made and uploaded ahead
// (`poolBuffers`), and `setPool` points every storage node at them (one node per buffer, type and
// access: `node`): new bind groups, no pipeline (docs/reference/three-behaviour.md, Storage buffers). The
// kernels read the batches' bounds, the last workgroup and the scatter's passes from uniforms, and
// every batch keeps its place even while empty (an empty batch starts where the next one does and
// writes no arguments), so no pool changes their code.
//
// Rules:
// - Each record's sphere bounds every pose its vertex stage can give it (`CullPayload.sphere`: the
//   meadow pads its tufts' rest boxes for the widening, the sway and standing water's sink). Where
//   the LOD widens the kept parts of a far record beyond what its sphere holds, the kernel adds
//   that reach by its count (`CullLod.pad`); it is exactly 0 near, so near records test as before.
// - The dispatch is skipped while the view and the part selection are unchanged.
// - Without the prepass, or with coverage kept in its shaded draw, the image is that of a draw of
//   every record to the byte; behind the prepass's opaque shaded draw a few isolated pixels
//   differ from it (the set depends on the drawn instance list).
// - Indirect arguments are set on a batch's geometry for its own draw only: other draws over the
//   same geometry (the rain's collision proxies, orthographic cameras) draw every record. A batch's
//   bands are its geometry's indirect offsets (an array: three issues a draw for each), so the
//   batch stays one render object and its depth copy draws the same bands.
// - A band's cap never cuts a part its records keep any of: a record goes into the band of the
//   least cap at or above its count (less 0.001, a hair against the vertex stage's rounding), and
//   the first band, which also takes any record over its cap (the meadow without its LOD, the default), draws the
//   batch's index count of the view (`CullRange.indexCount`, its nearest record's parts). The image
//   is the one-draw image but for the split (docs/reference/three-behaviour.md, Split draws).
//   `flags.grassCull = false` draws every tuft through the same storage path, and
//   `flags.grassBands = false` puts every visible tuft in its batch's first band (the one draw of
//   before), for A/Bs.
// - A batch's padding to whole workgroups repeats its first record's transform in the payload's
//   `matrices` buffer, where it has one: never drawn (its sphere is never visible), but three's InstancedMesh box
//   reads the matrices from slot 0 up to a batch's count, and a zero matrix there divides by a zero
//   w (a NaN box: Water.ts's reflection check). Slots past the last batch are never read.
// - A material that applies instancing itself from storage buffers must hold the placed position
//   in a variable (`toVar`), as three's instancing assigns it to `positionLocal`: an inline
//   expression rounds differently and flips a few hundred edge samples. `instanceColor` multiplies
//   the diffuse colour whenever the object has a truthy `instanceColor`: assign it in the vertex
//   stage.

type UintMax = ( a: THREE.Node<'uint'>, b: THREE.Node<'uint'> ) => THREE.Node<'uint'>;

const GROUP = 256;
/** The cull's workgroup size: a kernel over the pool's slots dispatched by `workgroups` takes it. */
export const CULL_GROUP = GROUP;
const LOG2_GROUP = 8;
// A batch's bands at most, counted per workgroup in BAND_WORDS words of three 10-bit fields.
export const MAX_BANDS = 9;
const BAND_WORDS = 3;
const BAND_BITS = 10;

type Storage<T extends StorageType = StorageType> = THREE.StorageBufferNode<T>;
type StorageType = 'uint' | 'vec4' | 'mat4';

/** A kind's data per slot, in storage buffers of its own (beside the cull's), and what it gives
 * the cull. */
export interface CullPayload<R> {
	/** The kind's buffers, in the order they are made: their names, floats per slot and the
	 * attributes' item size. */
	buffers: { name: string; floats: number; itemSize: number }[];
	/** The buffer holding each slot's anchor in its first vec4's x, y, z: the distance the LOD
	 * counts parts by (`CullLod.count`). */
	origin: string;
	/** The buffer of the instance transforms, for a kind drawn as three's InstancedMesh (which reads
	 * its box from it): a batch's padding repeats its first record's there. */
	matrices?: string;
	/** Buffers of vec4s the kind's own kernels write per slot (`floats` per slot), zeros in a new
	 * pool (no kind keeps one today). */
	scratch?: { name: string; floats: number }[];
	/** Writes record `r` into slot `i` of the kind's arrays. */
	write( r: R, i: number, arrays: Record<string, Float32Array> ): void;
	/** Writes record `r`'s bounding sphere (world space) into `spheres` at `i * 4`: every pose the
	 * vertex stage can give it over `geometry`, the batch's. */
	sphere( r: R, geometry: THREE.BufferGeometry, spheres: Float32Array, i: number ): void;
	/** Records folded away at a distance: whether slot `slot`, anchored at `origin`, is folded for
	 * a camera at `camera` (`node` reads the kind's buffers); `far` is a uniform the test reads,
	 * whose change culls again. */
	fold?: { far: THREE.UniformNode<'float', number>; folded( slot: THREE.Node<'uint'>, origin: THREE.Node<'vec3'>, camera: THREE.Node<'vec3'>, node: ( buffer: string ) => Storage<'vec4'> ): THREE.Node<'bool'> };
}

export interface CullBatchInput<R> {
	records: R[]; geometry: THREE.BufferGeometry;
	/** The record's parts and the least share of them it keeps (`CullLod.count`). */
	blades?: number; floor?: number;
	/** The part counts of the batch's bands, descending (at most MAX_BANDS), and the indices of
	 * one part: a band's draw takes its cap's parts, after `baseIndices` drawn by every band (a
	 * part every record keeps whole: a plant's flowers and scapes). */
	caps?: number[]; bladeIndices?: number; baseIndices?: number;
}

/** The part count the vertex stage keeps a record at (Grass.ts `tuftLod`), as a node: its anchor,
 * the camera's position, its parts and floor; and whether the view thins by distance at all. */
export interface CullLod {
	count: ( origin: THREE.Node<'vec3'>, camera: THREE.Node<'vec3'>, blades: THREE.Node<'float'>, floor: THREE.Node<'float'>, on: THREE.Node<'float'> ) => THREE.Node<'float'>;
	on: THREE.UniformNode<'float', number>;
	/** 0 puts every visible record in its batch's first band (`flags.grassBands = false`, for A/Bs). */
	bands: THREE.UniformNode<'uint', number>;
	/** How far (m) a record's kept parts reach past its sphere when it keeps `count` of its `blades`
	 * (the meadow's far tufts widen theirs further than their sphere's pad holds); 0 near. */
	pad?: ( count: THREE.Node<'float'>, blades: THREE.Node<'float'> ) => THREE.Node<'float'>;
}

export interface CullRange {
	/** First slot of the batch in the shared buffers (and in the visible list). */
	start: number;
	count: number;
	groupStart: number;
	groupEnd: number;
	/** The batch's index count for this view (its nearest record's parts): its first band's, and
	 * the most any band draws. */
	indexCount: THREE.UniformNode<'uint', number>;
	/** Byte offsets of the five indirect arguments of each of the batch's bands. */
	indirectOffsets: number[];
}

/** A pool's contents: the slots (a power of two of workgroups), the workgroups its batches fill,
 * each batch's range, the kind's per-slot arrays and the spheres. */
export interface CullPool {
	slots: number;
	groups: number;
	ranges: { start: number; count: number; groupStart: number; groupEnd: number }[];
	arrays: Record<string, Float32Array>;
	spheres: Float32Array;
}

/** A pool's storage buffers, made ahead of `setPool` so they can be uploaded before the frame that
 * takes them: the kind's by name, then `spheres`, `visible` (every slot's own index), `bands` (each
 * slot's band) and `counts` (a workgroup's band counts). */
export type CullBuffers = Record<string, THREE.StorageBufferAttribute>;

/** The pool's arrays for `batches`, filled `per` records at a step (a generator, so the growth can
 * spread the work over idle time: World growth, Growth.ts); a first build runs it to the end
 * (Steps.ts `runSteps`). The result is the same however it is stepped. */
export function* poolSteps<R>( payload: CullPayload<R>, batches: CullBatchInput<R>[], per = Infinity ): Generator<void, CullPool> {

	const ranges: CullPool[ 'ranges' ] = [];
	let slot = 0;
	for ( const b of batches ) {

		const count = b.records.length;
		const groups = Math.ceil( count / GROUP );
		ranges.push( { start: slot, count, groupStart: slot / GROUP, groupEnd: slot / GROUP + groups } );
		slot += groups * GROUP;

	}
	const groups = Math.max( slot, GROUP ) / GROUP;
	const n = 2 ** Math.ceil( Math.log2( groups ) ) * GROUP;
	const arrays: Record<string, Float32Array> = {};
	for ( const b of payload.buffers ) arrays[ b.name ] = new Float32Array( n * b.floats );
	const spheres = new Float32Array( n * 4 ).fill( - 1 );
	let done = 0;
	for ( let k = 0; k < batches.length; k ++ ) {

		const b = batches[ k ], range = ranges[ k ];
		for ( let j = 0; j < b.records.length; j ++ ) {

			const r = b.records[ j ], i = range.start + j;
			payload.write( r, i, arrays );
			payload.sphere( r, b.geometry, spheres, i );
			if ( ++ done % per === 0 ) yield;

		}

	}
	// (the rules, above: a batch's padding repeats its first record's transform)
	const matrices = payload.matrices ? arrays[ payload.matrices ] : null;
	if ( matrices ) for ( const range of ranges ) for ( let i = range.start + range.count; i < range.groupEnd * GROUP && range.count > 0; i ++ ) matrices.copyWithin( i * 16, range.start * 16, range.start * 16 + 16 );
	return { slots: n, groups, ranges, arrays, spheres };

}

/** The storage buffers of `pool`, for upload before `setPool` takes them. */
export function poolBuffers<R>( payload: CullPayload<R>, pool: CullPool ): CullBuffers {

	const n = pool.slots, visible = new Uint32Array( n );
	for ( let i = 0; i < n; i ++ ) visible[ i ] = i;
	const buffers: CullBuffers = {};
	for ( const b of payload.buffers ) buffers[ b.name ] = new THREE.StorageBufferAttribute( pool.arrays[ b.name ], b.itemSize );
	for ( const b of payload.scratch ?? [] ) buffers[ b.name ] = new THREE.StorageBufferAttribute( new Float32Array( n * b.floats ), 4 );
	buffers.spheres = new THREE.StorageBufferAttribute( pool.spheres, 4 );
	buffers.visible = new THREE.StorageBufferAttribute( visible, 1 );
	buffers.bands = new THREE.StorageBufferAttribute( new Uint32Array( n ), 1 );
	buffers.counts = new THREE.StorageBufferAttribute( new Uint32Array( n / GROUP * BAND_WORDS ), 1 );
	return buffers;

}

const _frustum = new THREE.Frustum(), _projScreen = new THREE.Matrix4(), _viewInverse = new THREE.Matrix4(), _camera = new THREE.Vector3();
const _padded = new THREE.Matrix4(), _size = new THREE.Vector2();

// Under the temporal modes the view camera draws the kind with its projection jittered by up to
// half a render pixel (Jitter.ts), a different one every frame. Keyed and culled by it, the cull
// would dispatch every frame of a still view. It takes the unjittered projection instead, widened
// by one render pixel on each side: the jittered frustum lies inside it whatever the offset. Any
// other camera keeps its own projection. Null when `camera` is not the jittered view camera.
export function unjitteredPadded( renderer: THREE.Renderer, camera: THREE.Camera ): THREE.Matrix4 | null {

	const q = jitter.uniforms.unjitteredProjection.value.elements, p = camera.projectionMatrix.elements;
	const d = jitter.uniforms.ndc.value;
	// The jitter adds a multiple of the projection's last row to its first two (`Jitter.begin`): for
	// a perspective camera only elements 8 and 9 move, by the offset in NDC.
	for ( let i = 0; i < 16; i ++ ) if ( i !== 8 && i !== 9 && p[ i ] !== q[ i ] ) return null;
	if ( Math.abs( p[ 8 ] - q[ 8 ] ) > Math.abs( d.x ) + 1e-9 || Math.abs( p[ 9 ] - q[ 9 ] ) > Math.abs( d.y ) + 1e-9 ) return null;
	const target = renderer.getRenderTarget();
	if ( target ) _size.set( target.width, target.height );
	else renderer.getDrawingBufferSize( _size );
	// Scaling the clip x (y) row by w / ( w + 2 ) moves the frustum's sides out by a pixel (2 / w in NDC).
	const sx = _size.x / ( _size.x + 2 ), sy = _size.y / ( _size.y + 2 );
	const e = _padded.copy( jitter.uniforms.unjitteredProjection.value ).elements;
	for ( const i of [ 0, 4, 8, 12 ] ) e[ i ] *= sx;
	for ( const i of [ 1, 5, 9, 13 ] ) e[ i ] *= sy;
	return _padded;

}

// (the map holds every type; each node is read back as the type it was made with)
type AnyStorage = { toReadOnly(): unknown; value: THREE.StorageBufferAttribute };

/** The cull's options: the kind's payload, its LOD and its kernels' label (`<name>.CullCount`). */
export interface VegetationCullOptions<R> { name: string; payload: CullPayload<R>; lod: CullLod }

export class VegetationCull<R> {

	/** The buffers' slots: the capacity, a power of two. */
	slots: number;
	/** The workgroups dispatched: those the batches fill. */
	groups: number;
	readonly ranges: CullRange[] = [];
	readonly indirect: THREE.IndirectStorageBufferAttribute;
	private buffers: CullBuffers;
	// every storage node over the pool's buffers, by buffer, type and access: `setPool` points them
	// at new buffers (a new bind group, no pipeline: docs/reference/three-behaviour.md, Storage buffers)
	private readonly nodes = new Map<string, { buffer: string; node: AnyStorage }>();
	// the batches' bounds, as the kernels read them (uniforms, so a new pool builds no pipeline)
	private readonly bounds: { start: THREE.UniformNode<'uint', number>; groupStart: THREE.UniformNode<'uint', number>; groupEnd: THREE.UniformNode<'uint', number> }[];
	private readonly lastGroup = uniform( 0, 'uint' );
	// the scatter's passes over a batch's earlier workgroups: its longest batch's workgroups / GROUP
	private readonly sumPasses = uniform( 0, 'uint' );
	private readonly dispatch: [ number, number, number ] = [ 1, 1, 1 ];
	private readonly planes = Array.from( { length: 5 }, () => uniform( new THREE.Vector4() ) );
	private readonly cameraPosition = uniform( new THREE.Vector3() );
	// built at the first update, which knows whether the device draws from a first instance
	private kernels: THREE.ComputeNode[] | null = null;
	private readonly lastKey: Float64Array;
	// each batch's record (parts, floor), band caps and indices per part
	private readonly batches: { blades: number; floor: number; caps: number[]; bladeIndices: number; baseIndices: number; first: number }[];
	/** Debug diagnostics: views culled (dispatches) and updates skipped as unchanged, since load. */
	readonly stats = { dispatches: 0, skipped: 0 };

	private readonly name: string;
	readonly payload: CullPayload<R>;
	private readonly lod: CullLod;

	constructor( options: VegetationCullOptions<R>, batches: CullBatchInput<R>[] ) {

		this.name = options.name;
		this.payload = options.payload;
		this.lod = options.lod;
		const pool = runSteps( poolSteps( this.payload, batches ) );
		this.slots = pool.slots;
		this.groups = pool.groups;
		this.buffers = poolBuffers( this.payload, pool );
		let draws = 0;
		this.batches = batches.map( ( b ) => {

			const blades = b.blades ?? 16, caps = ( b.caps ?? [ blades ] ).slice( 0, MAX_BANDS );
			const batch = { blades, floor: b.floor ?? 0, caps, bladeIndices: b.bladeIndices ?? 1, baseIndices: b.baseIndices ?? 0, first: draws };
			draws += caps.length;
			return batch;

		} );
		this.indirect = new THREE.IndirectStorageBufferAttribute( new Uint32Array( draws * 5 ), 1 );
		this.bounds = batches.map( () => ( { start: uniform( 0, 'uint' ), groupStart: uniform( 0, 'uint' ), groupEnd: uniform( 0, 'uint' ) } ) );
		pool.ranges.forEach( ( r, k ) => this.ranges.push( { ...r, indexCount: uniform( 0, 'uint' ),
			indirectOffsets: this.batches[ k ].caps.map( ( _, j ) => ( this.batches[ k ].first + j ) * 20 ) } ) );
		this.lastKey = new Float64Array( 16 + 9 + batches.length ).fill( NaN );
		this.setRanges( pool );

	}

	/** Each batch's band caps as drawn (blades), known from the first update on. */
	get bands(): number[][] {

		return this.batches.map( ( b, k ) => b.caps.slice( 0, this.ranges[ k ].indirectOffsets.length ) );

	}

	/** The workgroups of CULL_GROUP slots the pool fills: an array a kernel over the slots takes as
	 * its `dispatchSize`, so a new pool's size reaches it. */
	get workgroups(): [ number, number, number ] {

		return this.dispatch;

	}

	/** The pool's buffer `name` (the kind's, or the cull's own). */
	buffer( name: string ): THREE.StorageBufferAttribute {

		return this.buffers[ name ];

	}

	/** A storage node over one of the pool's buffers (one per buffer, type and access), which
	 * follows the pool when `setPool` replaces it. */
	node<T extends StorageType>( buffer: string, type: T, readOnly = false ): Storage<T> {

		const key = `${ buffer } ${ type } ${ readOnly }`;
		let entry = this.nodes.get( key );
		if ( ! entry ) {

			const attribute = this.buffers[ buffer ];
			const node = storage( attribute, type, attribute.count ) as unknown as AnyStorage;
			if ( readOnly ) node.toReadOnly();
			this.nodes.set( key, entry = { buffer, node } );

		}
		return entry.node as unknown as Storage<T>;

	}

	/** Takes `pool` (the same batches in the same order), with its buffers made by `buffers`: the
	 * nodes are pointed at them, the bounds and the dispatch follow, and the next update culls
	 * again. Returns the buffers it replaced, to dispose of once nothing draws with them. */
	setPool( pool: CullPool, buffers: CullBuffers = poolBuffers( this.payload, pool ) ): CullBuffers {

		const old = this.buffers;
		this.buffers = buffers;
		for ( const { buffer, node } of this.nodes.values() ) node.value = buffers[ buffer ];
		this.slots = pool.slots;
		this.groups = pool.groups;
		pool.ranges.forEach( ( r, k ) => Object.assign( this.ranges[ k ], r ) );
		this.setRanges( pool );
		this.invalidate();
		return old;

	}

	private setRanges( pool: CullPool ): void {

		const args = this.indirect.array as Uint32Array;
		pool.ranges.forEach( ( r, k ) => {

			const b = this.bounds[ k ];
			b.start.value = r.start;
			b.groupStart.value = r.groupStart;
			b.groupEnd.value = r.groupEnd;
			// (a batch's arguments are the scatter's from its first dispatch on; an empty batch's stay 0,
			// as do the first index and base vertex)
			const first = this.batches[ k ].first * 5;
			args.fill( 0, first, first + this.batches[ k ].caps.length * 5 );

		} );
		this.indirect.needsUpdate = true;
		this.lastGroup.value = pool.groups - 1;
		this.sumPasses.value = Math.ceil( Math.max( ...pool.ranges.map( ( r ) => r.groupEnd - r.groupStart ) ) / GROUP );
		this.dispatch[ 0 ] = pool.groups;

	}

	private buildKernels(): THREE.ComputeNode[] {

		const spheres = this.node( 'spheres', 'vec4', true );
		const origins = this.node( this.payload.origin, 'vec4', true );
		const visible = this.node( 'visible', 'uint' );
		const bandOf = this.node( 'bands', 'uint' );
		const counts = this.node( 'counts', 'uint' );
		const args = storage( this.indirect, 'uint', this.indirect.count );
		const planes = this.planes, cameraPosition = this.cameraPosition, bounds = this.bounds, fold = this.payload.fold;
		// the bands drawn: every batch's, or its first alone (`update`)
		const caps = this.bands;
		const K = Math.max( ...caps.map( ( c ) => c.length ) );
		type Bound = typeof bounds[ number ];
		// Per workgroup: a value of its batch (its first workgroup, first slot, a constant). The
		// batches lie in order, so the last batch starting at or before the workgroup holds it (an
		// empty batch starts where the next one does, and loses to it).
		const pick = <T extends 'uint' | 'float'>( g: THREE.Node<'uint'>, value: ( b: Bound, k: number ) => THREE.Node<T> ) => {

			let node: THREE.Node<T> = value( bounds[ 0 ], 0 );
			for ( let k = 1; k < bounds.length; k ++ ) node = select( g.greaterThanEqual( bounds[ k ].groupStart ), value( bounds[ k ], k ), node ) as THREE.Node<T>;
			return node;

		};

		const isVisible = ( gi: THREE.Node<'uint'>, pad: THREE.Node<'float'> | null ) => {

			const s = spheres.element( gi ).toVar();
			let inside = s.w.greaterThanEqual( 0 );
			const radius = pad ? s.w.add( pad ).toVar() : s.w;
			for ( const p of planes ) inside = inside.and( dot( p.xyz, s.xyz ).add( p.w ).greaterThanEqual( radius.negate() ) );
			// (records the kind has folded away at this distance: a plant's level outside its band)
			if ( ! fold ) return inside;
			const folded = fold.folded( gi, origins.element( gi ).xyz, cameraPosition, ( buffer ) => this.node( buffer, 'vec4', true ) );
			return inside.and( folded.not() );

		};
		// A record's band: the last of its batch's bands whose cap holds the parts it keeps (the
		// first when none does); K for a record not drawn.
		const bandOfTuft = ( g: THREE.Node<'uint'>, gi: THREE.Node<'uint'> ) => {

			const blades = pick( g, ( _, k ) => float( this.batches[ k ].blades ) );
			const floor = pick( g, ( _, k ) => float( this.batches[ k ].floor ) );
			const count = this.lod.count( origins.element( gi ).xyz, cameraPosition, blades, floor, this.lod.on ).toVar();
			const n = uint( ceil( count.sub( 0.001 ) ) ).toVar();
			let band: THREE.Node<'uint'> = uint( 0 );
			for ( let j = 1; j < K; j ++ ) band = band.add( select( pick( g, ( _, k ) => uint( caps[ k ][ j ] ?? 0 ) ).greaterThanEqual( n ), uint( 1 ), uint( 0 ) ) );
			const pad = this.lod.pad ? this.lod.pad( count, blades ) : null;
			return select( isVisible( gi, pad ), select( this.lod.bands.equal( uint( 1 ) ), band, uint( 0 ) ), uint( K ) );

		};
		// The band as a one in its word's field (BAND_BITS each), one node per word the bands use.
		const W = Math.ceil( K / 3 );
		const flags = ( band: THREE.Node<'uint'> ) => Array.from( { length: W }, ( _, w ) => {

			let word: THREE.Node<'uint'> = uint( 0 );
			for ( let k = w * 3; k < Math.min( K, w * 3 + 3 ); k ++ ) word = word.add( select( band.equal( uint( k ) ), uint( 1 << ( BAND_BITS * ( k - w * 3 ) ) ), uint( 0 ) ) );
			return word;

		} );
		const field = ( word: THREE.Node<'uint'>, k: number ) => word.shiftRight( uint( BAND_BITS * ( k % 3 ) ) ).bitAnd( uint( ( 1 << BAND_BITS ) - 1 ) );
		type Lanes = { element: ( i: THREE.Node<'uint'> ) => THREE.Node<'uint'> };
		// Inclusive scan of the words of each thread over the workgroup (Hillis-Steele, ping-ponged
		// between two arrays per word so each step needs one barrier). Returns the arrays holding the
		// results; a field never carries into the next (at most GROUP < 2^BAND_BITS).
		const scan = ( words: THREE.Node<'uint'>[] ) => {

			const lanes = words.map( () => [ workgroupArray( 'uint', GROUP ), workgroupArray( 'uint', GROUP ) ] as unknown as Lanes[] );
			const i = localId.x;
			words.forEach( ( word, w ) => lanes[ w ][ 0 ].element( i ).assign( word ) );
			for ( let s = 0; s < LOG2_GROUP; s ++ ) {

				workgroupBarrier();
				const off = 1 << s;
				for ( const pair of lanes ) {

					const from = pair[ s & 1 ], to = pair[ ( s + 1 ) & 1 ];
					const before = select( i.greaterThanEqual( uint( off ) ), from.element( ( max as unknown as UintMax )( i, uint( off ) ).sub( off ) ), uint( 0 ) );
					to.element( i ).assign( from.element( i ).add( before ) );

				}

			}
			workgroupBarrier();
			return lanes.map( ( pair ) => pair[ LOG2_GROUP & 1 ] );

		};

		const count = Fn( () => {

			const g = workgroupId.x, i = localId.x;
			const gi = g.mul( GROUP ).add( i );
			const band = bandOfTuft( g, gi ).toVar();
			bandOf.element( gi ).assign( band );
			const result = scan( flags( band ) );
			If( i.equal( uint( GROUP - 1 ) ), () => {

				result.forEach( ( r, w ) => counts.element( g.mul( BAND_WORDS ).add( w ) ).assign( r.element( uint( GROUP - 1 ) ) ) );

			} );

		} )().computeKernel( [ GROUP ] );

		const scatter = Fn( () => {

			const g = workgroupId.x, i = localId.x;
			const gi = g.mul( GROUP ).add( i );
			// (the band the count pass decided, so the counts and the places agree)
			const band = bandOf.element( gi ).toVar();
			const result = scan( flags( band ) );
			const inclusive = result.map( ( r ) => r.element( i ).toVar() );
			// Per band, the records of the batch's earlier workgroups and of all of them: each thread
			// sums every GROUP-th workgroup's counts, then the workgroup reduces the sums.
			const first = pick( g, ( b ) => b.groupStart ).toVar();
			const end = pick( g, ( b ) => b.groupEnd ).toVar();
			const earlier = Array.from( { length: K }, () => uint( 0 ).toVar() );
			const total = Array.from( { length: K }, () => uint( 0 ).toVar() );
			Loop( { start: uint( 0 ), end: this.sumPasses, type: 'uint', condition: '<' }, ( { i: pass } ) => {

				const j = first.add( i ).add( pass.mul( GROUP ) ).toVar();
				const at = ( min as unknown as UintMax )( j, this.lastGroup ).mul( BAND_WORDS );
				const words = Array.from( { length: W }, ( _, w ) => select( j.lessThan( end ), counts.element( at.add( w ) ), uint( 0 ) ).toVar() );
				for ( let k = 0; k < K; k ++ ) {

					const n = field( words[ Math.floor( k / 3 ) ], k ).toVar();
					total[ k ].addAssign( n );
					earlier[ k ].addAssign( select( j.lessThan( g ), n, uint( 0 ) ) );

				}

			} );
			// (one reduction at a time over one array: both at once took 2 K words a thread, past the
			// default workgroup memory with seven bands or more; the barrier first lets every thread read the
			// last reduction's sums before they are overwritten)
			const sum = workgroupArray( 'uint', GROUP * K ) as unknown as Lanes;
			const reduce = ( values: THREE.Node<'uint'>[] ) => {

				workgroupBarrier();
				for ( let k = 0; k < K; k ++ ) sum.element( i.mul( K ).add( k ) ).assign( values[ k ] );
				for ( let s = GROUP >> 1; s >= 1; s >>= 1 ) {

					workgroupBarrier();
					If( i.lessThan( uint( s ) ), () => {

						for ( let k = 0; k < K; k ++ ) sum.element( i.mul( K ).add( k ) ).addAssign( sum.element( i.add( s ).mul( K ).add( k ) ) );

					} );

				}
				workgroupBarrier();
				return values.map( ( _, k ) => sum.element( uint( k ) ).toVar() );

			};
			const offset = reduce( earlier );
			const tufts = reduce( total );
			// each band's first place in the batch's list: after the bands before it
			const bandStart: THREE.Node<'uint'>[] = [ uint( 0 ) ];
			for ( let k = 1; k < K; k ++ ) bandStart.push( bandStart[ k - 1 ].add( tufts[ k - 1 ] ).toVar() );
			If( band.lessThan( uint( K ) ), () => {

				let place: THREE.Node<'uint'> = uint( 0 );
				for ( let k = 0; k < K; k ++ ) place = place.add( select( band.equal( uint( k ) ), bandStart[ k ].add( offset[ k ] ).add( field( inclusive[ Math.floor( k / 3 ) ], k ) ).sub( 1 ), uint( 0 ) ) );
				visible.element( pick( g, ( b ) => b.start ).add( place ) ).assign( gi );

			} );
			// The batch's last workgroup writes its bands' draw arguments: index count, instance
			// count and first instance (first index and base vertex stay 0). An empty batch has none.
			bounds.forEach( ( b, k ) => {

				If( g.add( 1 ).equal( b.groupEnd ).and( b.groupEnd.greaterThan( b.groupStart ) ).and( i.equal( 0 ) ), () => {

					const { first: draw, bladeIndices, baseIndices } = this.batches[ k ];
					caps[ k ].forEach( ( cap, j ) => {

						const indexCount = j === 0 ? this.ranges[ k ].indexCount : ( min as unknown as UintMax )( uint( baseIndices + cap * bladeIndices ), this.ranges[ k ].indexCount );
						args.element( uint( ( draw + j ) * 5 ) ).assign( indexCount );
						args.element( uint( ( draw + j ) * 5 + 1 ) ).assign( tufts[ j ] );
						args.element( uint( ( draw + j ) * 5 + 4 ) ).assign( bandStart[ j ] );

					} );

				} );

			} );

		} )().computeKernel( [ GROUP ] );

		for ( const kernel of [ count, scatter ] ) {

			// (the same array, so a new pool's size reaches both: an array is read at each dispatch,
			// never part of the pipeline)
			kernel.dispatchSize = this.dispatch;
			kernel.setName( `${ this.name }.${ kernel === count ? 'CullCount' : 'CullScatter' }` );

		}
		return [ count, scatter ];

	}

	/** Cull for `camera` unless the last dispatch was for the same view and the same draws. The
	 * caller has set every batch's `indexCount` and the LOD's `on`. */
	update( renderer: THREE.Renderer, camera: THREE.Camera ): void {

		if ( ! this.kernels ) {

			// A band's draw starts at its first record's place through `firstInstance`, which an
			// indirect draw honours only with this feature: without it each batch keeps its first band.
			if ( ! hasGpuFeature( renderer, 'indirect-first-instance' ) ) for ( const r of this.ranges ) r.indirectOffsets.length = 1;
			this.kernels = this.buildKernels();

		}
		const key = this.lastKey;
		const projection = ( TEMPORAL ? unjitteredPadded( renderer, camera ) : null ) ?? camera.projectionMatrix;
		const e = camera.matrixWorld.elements, p = projection.elements;
		let same = true;
		const check = ( i: number, v: number ) => {

			if ( key[ i ] !== v ) {

				key[ i ] = v;
				same = false;

			}

		};
		for ( let i = 0; i < 16; i ++ ) check( i, e[ i ] );
		check( 16, p[ 0 ] ); check( 17, p[ 5 ] ); check( 18, p[ 8 ] );
		check( 19, p[ 9 ] ); check( 20, p[ 10 ] ); check( 21, p[ 14 ] );
		check( 22, this.payload.fold?.far.value ?? 0 );
		check( 23, this.lod.on.value );
		check( 24, this.lod.bands.value );
		this.ranges.forEach( ( r, k ) => check( 25 + k, r.indexCount.value ) );
		if ( same ) {

			this.stats.skipped ++;
			return;

		}
		this.stats.dispatches ++;
		_projScreen.multiplyMatrices( projection, _viewInverse.copy( camera.matrixWorld ).invert() );
		_frustum.setFromProjectionMatrix( _projScreen, camera.coordinateSystem, camera.reversedDepth );
		// left, right, bottom, top, near: the meadow never reaches the far plane
		for ( let k = 0; k < 4; k ++ ) this.writePlane( k, _frustum.planes[ k ] );
		this.writePlane( 4, _frustum.planes[ 5 ] );
		this.cameraPosition.value.copy( _camera.setFromMatrixPosition( camera.matrixWorld ) );
		renderer.compute( this.kernels );

	}

	/** Forget the last view: the next update dispatches again. */
	invalidate(): void {

		this.lastKey.fill( NaN );

	}

	private writePlane( k: number, plane: THREE.Plane ): void {

		this.planes[ k ].value.set( plane.normal.x, plane.normal.y, plane.normal.z, plane.constant );

	}

}
