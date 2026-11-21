// The frame graph's declarations and their compiler: order checks, culling and attachment ops.
//
// The frame graph's declarations and their compiler, after the FrameGraph of O'Donnell (Frostbite,
// GDC 2017); Unreal's RDG and Unity's RenderGraph are implementations of the same idea. Passes
// declare what they read and write, each write makes a new version of its resource, and the
// compiler checks the declared order against those versions, culls passes nothing live reads and
// derives each attachment's load, store and multisample resolve from the passes that run.
//
// Unlike Frostbite's graph this one does not alias memory, insert barriers or rebuild every frame:
// WebGPU tracks its own hazards, and the passes are declared once, when the pipeline is built. A
// pass may be skipped at runtime by its `when` predicate; the attachment ops are derived for every
// combination of skippable passes (the enabled mask).
//
// A compute or callback pass may write a texture through storage (`storage( h )`: a kernel's
// stores, a copy into it) rather than as an attachment: the write makes a new version and its pass
// is that version's producer for culling and the version checks, but it has no render target and
// the attachment ops never see it. Such a write is partial (what the version before held stays
// alive), and a texture the graph does not own (an import) written that way roots its pass, as any
// write of a persistent resource does.
//
// Pure TypeScript with no runtime import, so Node runs it directly (tools/render/verify-framegraph.mjs);
// FrameGraph.ts executes the compiled graph with three's renderer.

/** The size of a resource: a share of the scene's render, of the canvas, or fixed. The rounding is
 * the one the passes had before the graph: `floor( w * scale )` (rtt passes and bloom chains, at
 * least `min` when given), `max( floor( w / divisor ), 1 )` (a reduced pass), and the canvas
 * never smaller than the render. */
export type SizeClass =
	| { of: 'render'; scale?: number; divisor?: number; min?: number }
	| { of: 'canvas'; scale?: number }
	| { of: 'fixed'; width: number; height: number };

export interface Size { width: number; height: number }
export interface Extents { render: Size; canvas: Size }

export type PassKind = 'scene' | 'quad' | 'chain' | 'compute' | 'callback';
/** 'sample': through a sampler or a load of the resolved texture; 'samples': the multisampled
 * samples themselves (a load per sample). */
export type ReadMode = 'sample' | 'samples';
/** 'derived' (the default) follows the passes that run: a store is discarded where no later pass
 * loads or reads the attachment, a multisample resolve dropped where nothing samples it before the
 * next draw into the same samples. 'legacy' is the graph before derived ops: the multisampled stores
 * discarded in the rain pass only (as Pipeline discarded them by hand before the graph ran its
 * passes). 'store' keeps every store and resolve, as three records them (`post=store-scene`). */
export type OpsMode = 'legacy' | 'derived' | 'store';
export type Samples = number | 'scene';

export interface ResourceHandle {
	readonly resource: number;
	readonly version: number;
	readonly name: string;
	// the previous frame's content of a ping-pong resource (see `previous`)
	readonly previous?: boolean;
}

export interface ResourceDesc {
	size: SizeClass;
	// three's format and type constants at runtime; names in the CPU tables
	format?: string | number;
	type?: string | number;
	samples?: Samples;
	filter?: 'linear' | 'nearest';
	// a full mip chain, which three makes after each pass that draws the resource, every level a
	// box filter of the one above (WebGPUTexturePassUtils): what a pass reads at a footprint wider
	// than a texel (single-sampled colour only)
	mips?: boolean;
}

export type ResourceKind = 'color' | 'depth' | 'import';

export interface ResourceDecl {
	readonly id: number;
	readonly name: string;
	readonly kind: ResourceKind;
	readonly desc: Readonly<ResourceDesc>;
	// content kept across frames: never culled, never discarded
	readonly persistent: boolean;
	// the canvas (an import without a texture)
	readonly canvas: boolean;
	// two physical textures swapped at the end of each frame (`previous`)
	pingPong: boolean;
	// the latest version declared so far
	latest: number;
	// versions[ v ] = the index of the pass that wrote version v (-1: before the frame)
	readonly producers: number[];
}

export interface ReadDecl { readonly resource: number; readonly version: number; readonly mode: ReadMode; readonly optional: boolean; readonly previous: boolean }
export interface WriteDecl { readonly resource: number; readonly version: number; readonly clear: boolean; readonly blend?: string }
export interface DepthDecl { readonly resource: number; readonly version: number; readonly write: boolean; readonly clear: boolean }

/** What a pass's setup returns besides its executor (FrameGraph.ts adds the executor fields). */
export interface PassFlags {
	label: string;
	when?: unknown;
	// a name for the `when` predicate, for tables and introspection
	whenLabel?: string;
	jittered?: boolean;
	neverCull?: boolean;
	// the pass's size class when it has no attachment of its own (compute, callback)
	size?: SizeClass;
	mrt?: unknown;
}

export interface PassDecl {
	readonly index: number;
	readonly name: string;
	readonly kind: PassKind;
	label: string;
	readonly reads: ReadDecl[];
	// colour attachments in attachment order (several: an MRT)
	readonly writes: WriteDecl[];
	// versions written through storage (compute and callback passes): no attachment, no render target
	readonly storage: WriteDecl[];
	depth: DepthDecl | null;
	conditional: boolean;
	whenLabel: string | null;
	jittered: boolean;
	neverCull: boolean;
	hasMrt: boolean;
	size: SizeClass | null;
}

export interface PassBuilder {
	/** Reads a version written earlier in the frame, or a persistent or imported resource's. */
	read( h: ResourceHandle, mode?: ReadMode ): ResourceHandle;
	/** Reads a version whose producer may be skipped (`when`): the pass then reads what the
	 * texture held before, which its shader must tolerate. */
	readOptional( h: ResourceHandle, mode?: ReadMode ): ResourceHandle;
	/** Writes a colour attachment (in attachment order) and returns the new version. */
	write( h: ResourceHandle, options?: { clear?: boolean; blend?: string } ): ResourceHandle;
	/** The depth attachment: tested and written (`write`), or tested only. */
	depth( h: ResourceHandle, options: { write: boolean; clear?: boolean } ): ResourceHandle;
	/** Writes a colour or imported texture through storage (compute and callback passes: a kernel's
	 * store, a copy into it) and returns the new version: no attachment and no render target, so the
	 * targets and the attachment ops ignore it, but it makes a version like any write (its pass is
	 * its producer for culling and the version checks). Partial: what the version before held is
	 * kept alive. */
	storage( h: ResourceHandle ): ResourceHandle;
}

export class GraphError extends Error {

	constructor( message: string ) {

		super( message );
		this.name = 'GraphError';

	}

}

const fail = ( message: string ): never => {

	throw new GraphError( message );

};

/** Whether a colour write replaces everything under it: an explicit clear, or a full-screen quad or
 * chain that does not blend over what it loads. Such a write needs nothing earlier passes left in
 * the attachment, so it clears on load, and the passes before it are not kept alive for it. A
 * blended full-screen pass (`blend` set) loads and draws over its input. Culling, the ops tables
 * and the runtime's clear flags all decide with this one predicate. */
export function drawsOver( pass: { kind: PassKind }, write: { clear: boolean; blend?: string } ): boolean {

	return write.clear || ( ( pass.kind === 'quad' || pass.kind === 'chain' ) && ! write.blend );

}

const sizeKey = ( s: SizeClass ): string => s.of === 'fixed' ? `fixed:${ s.width }x${ s.height }`
	: s.of === 'canvas' ? `canvas:${ s.scale ?? 1 }`
		: `render:${ s.scale ?? 1 }:${ s.divisor ?? 0 }:${ s.min ?? 0 }`;

/** A stable key per size class (structurally equal classes share it). */
export function sizeClassKey( s: SizeClass ): string {

	return sizeKey( s );

}

/** The key of a render target: its colour attachments in order and its depth attachment, by
 * resource name. The compiler makes one target per key; FrameGraph.renderTarget makes the same
 * target ahead of the passes that will draw into it. */
export function targetKey( colors: readonly string[], depth: string | null ): string {

	return `${ colors.join( '+' ) }|${ depth ?? '-' }`;

}

/** The pixel size of a class at the given extents, with the rounding the passes had before. */
export function resolveSize( s: SizeClass, e: Extents ): Size {

	if ( s.of === 'fixed' ) return { width: s.width, height: s.height };
	if ( s.of === 'canvas' ) {

		const w = Math.max( e.canvas.width, e.render.width ), h = Math.max( e.canvas.height, e.render.height );
		const k = s.scale ?? 1;
		return k === 1 ? { width: w, height: h } : { width: Math.floor( w * k ), height: Math.floor( h * k ) };

	}
	const { width, height } = e.render;
	if ( s.divisor ) return { width: Math.max( Math.floor( width / s.divisor ), 1 ), height: Math.max( Math.floor( height / s.divisor ), 1 ) };
	const k = s.scale ?? 1;
	let w = Math.floor( width * k ), h = Math.floor( height * k );
	if ( s.min !== undefined ) {

		w = Math.max( w, s.min );
		h = Math.max( h, s.min );

	}
	return { width: w, height: h };

}

/** The declarations: resources, passes in execution order, frame-end copies, the debug view. */
export class GraphDecl {

	readonly resources: ResourceDecl[] = [];
	readonly passes: PassDecl[] = [];
	readonly copies: { src: ResourceHandle; dst: ResourceHandle }[] = [];
	debugView: number | null = null;
	private building: PassDecl | null = null;
	private names = new Set<string>();

	private resource( name: string, kind: ResourceKind, desc: ResourceDesc, persistent: boolean, canvas: boolean ): ResourceHandle {

		if ( this.names.has( name ) ) fail( `resource '${ name }' declared twice` );
		if ( this.building ) fail( `resource '${ name }' declared inside the setup of '${ this.building.name }'` );
		this.names.add( name );
		const id = this.resources.length;
		this.resources.push( { id, name, kind, desc: { ...desc }, persistent, canvas, pingPong: false, latest: 0, producers: [ - 1 ] } );
		return { resource: id, version: 0, name };

	}

	/** A colour resource (`persistent`: its content survives the frame). */
	color( name: string, desc: ResourceDesc, persistent = false ): ResourceHandle {

		if ( desc.mips && ( desc.samples ?? 1 ) !== 1 ) fail( `resource '${ name }': mips need a single-sampled target` );
		return this.resource( name, 'color', desc, persistent, false );

	}

	depth( name: string, desc: { size: SizeClass; samples?: Samples; type?: string | number; format?: string | number } ): ResourceHandle {

		return this.resource( name, 'depth', desc, false, false );

	}

	/** A texture made outside the graph (or the canvas: `canvas` true), read or written as is. */
	import( name: string, desc: ResourceDesc, canvas = false ): ResourceHandle {

		return this.resource( name, 'import', desc, true, canvas );

	}

	/** The previous frame's version of a resource kept in two textures that swap at the frame's
	 * end. Only for resources read solely by passes built once per texture (the TAA history):
	 * a TSL texture node binds one texture. */
	previous( h: ResourceHandle ): ResourceHandle {

		const r = this.get( h );
		if ( r.kind !== 'color' ) fail( `previous( '${ r.name }' ): only colour resources ping-pong` );
		r.pingPong = true;
		return { resource: r.id, version: - 1, name: r.name, previous: true };

	}

	/** Copies `src`'s last version into the persistent `dst` at the end of the frame
	 * (copyTextureToTexture): a one-frame history any number of passes may read. */
	copyToPersistent( src: ResourceHandle, dst: ResourceHandle ): void {

		const d = this.get( dst );
		if ( ! d.persistent || d.kind === 'import' ) fail( `copyToPersistent: '${ d.name }' is not a persistent resource of the graph` );
		const s = this.get( src );
		if ( s.id === d.id ) fail( `copyToPersistent: '${ s.name }' onto itself` );
		if ( this.copies.some( ( c ) => c.dst.resource === d.id ) ) fail( `copyToPersistent: '${ d.name }' has two sources` );
		this.copies.push( { src, dst } );

	}

	/** The resource the debug view shows (kept live and stored). */
	setDebugView( h: ResourceHandle | null ): void {

		this.debugView = h ? this.get( h ).id : null;

	}

	get( h: ResourceHandle ): ResourceDecl {

		return this.resources[ h.resource ] ?? fail( `unknown resource ${ h.name }` );

	}

	/** Declares a pass; `setup` declares its reads and writes through the builder and returns its
	 * flags (and, in FrameGraph, its executor), which are returned as given. */
	addPass<T extends PassFlags>( name: string, kind: PassKind, setup: ( b: PassBuilder ) => T ): T {

		if ( this.passes.some( ( p ) => p.name === name ) ) fail( `pass '${ name }' declared twice` );
		const pass: PassDecl = {
			index: this.passes.length, name, kind, label: name, reads: [], writes: [], storage: [], depth: null,
			conditional: false, whenLabel: null, jittered: false, neverCull: false, hasMrt: false, size: null,
		};
		const read = ( h: ResourceHandle, mode: ReadMode, optional: boolean ): ResourceHandle => {

			const r = this.get( h );
			if ( h.previous ) {

				if ( ! r.pingPong ) fail( `'${ name }' reads the previous '${ r.name }' without previous()` );

			} else {

				if ( h.version > r.latest ) fail( `'${ name }' reads '${ r.name }' v${ h.version }, not declared yet` );
				if ( h.version < r.latest ) fail( `'${ name }' reads '${ r.name }' v${ h.version }, overwritten by v${ r.latest }` );
				if ( h.version === 0 && ! r.persistent ) fail( `'${ name }' reads '${ r.name }' before any pass writes it` );
				const producer = r.producers[ h.version ];
				if ( producer >= 0 && producer === pass.index ) fail( `'${ name }' reads its own write of '${ r.name }'` );
				if ( producer >= 0 && this.passes[ producer ].conditional && ! optional )
					fail( `'${ name }' reads '${ r.name }' from '${ this.passes[ producer ].name }', which may be skipped: use readOptional` );

			}
			pass.reads.push( { resource: r.id, version: h.version, mode, optional, previous: h.previous === true } );
			return h;

		};
		const next = ( r: ResourceDecl, h: ResourceHandle, what: string ): ResourceHandle => {

			if ( h.previous ) fail( `'${ name }' ${ what } the previous '${ r.name }'` );
			if ( h.version !== r.latest ) fail( `'${ name }' ${ what } '${ r.name }' v${ h.version }, a stale version (latest v${ r.latest })` );
			r.latest ++;
			r.producers.push( pass.index );
			return { resource: r.id, version: r.latest, name: r.name };

		};
		const builder: PassBuilder = {
			read: ( h, mode = 'sample' ) => read( h, mode, false ),
			readOptional: ( h, mode = 'sample' ) => read( h, mode, true ),
			write: ( h, options = {} ) => {

				const r = this.get( h );
				if ( r.kind === 'depth' ) fail( `'${ name }' writes depth '${ r.name }' as colour: use depth()` );
				if ( pass.writes.some( ( w ) => w.resource === r.id ) ) fail( `'${ name }' writes '${ r.name }' twice` );
				const v = next( r, h, 'writes' );
				pass.writes.push( { resource: r.id, version: v.version, clear: options.clear === true, blend: options.blend } );
				return v;

			},
			depth: ( h, options ) => {

				const r = this.get( h );
				if ( r.kind !== 'depth' ) fail( `'${ name }' uses colour '${ r.name }' as its depth attachment` );
				if ( pass.depth ) fail( `'${ name }' has two depth attachments` );
				if ( options.write ) {

					const v = next( r, h, 'writes' );
					pass.depth = { resource: r.id, version: v.version, write: true, clear: options.clear === true };
					return v;

				}
				if ( h.previous || h.version !== r.latest ) fail( `'${ name }' tests against '${ r.name }' v${ h.version }, a stale version` );
				if ( h.version === 0 ) fail( `'${ name }' tests against '${ r.name }' before any pass writes it` );
				if ( options.clear ) fail( `'${ name }' clears '${ r.name }' without writing it` );
				const producer = r.producers[ h.version ];
				if ( producer >= 0 && this.passes[ producer ].conditional ) fail( `'${ name }' tests against '${ r.name }' from '${ this.passes[ producer ].name }', which may be skipped` );
				pass.depth = { resource: r.id, version: h.version, write: false, clear: false };
				return h;

			},
			storage: ( h ) => {

				const r = this.get( h );
				if ( kind !== 'compute' && kind !== 'callback' ) fail( `'${ name }' writes '${ r.name }' through storage: only compute and callback passes do` );
				if ( r.kind === 'depth' || r.canvas ) fail( `'${ name }' writes '${ r.name }' through storage: depth and the canvas have no storage` );
				if ( pass.storage.some( ( w ) => w.resource === r.id ) ) fail( `'${ name }' writes '${ r.name }' through storage twice` );
				const v = next( r, h, 'writes' );
				pass.storage.push( { resource: r.id, version: v.version, clear: false } );
				return v;

			},
		};
		this.building = pass;
		this.passes.push( pass );
		// a refused pass leaves no trace: the versions its setup made are taken back
		const marks = this.resources.map( ( r ) => r.latest );
		try {

			const flags = setup( builder );
			pass.label = flags.label ?? name;
			pass.conditional = typeof flags.when === 'function';
			pass.whenLabel = flags.whenLabel ?? ( pass.conditional ? 'when' : null );
			pass.jittered = flags.jittered === true;
			pass.neverCull = flags.neverCull === true;
			pass.hasMrt = flags.mrt != null;
			pass.size = flags.size ?? null;
			this.checkPass( pass );
			return flags;

		} catch ( error ) {

			this.passes.pop();
			this.resources.forEach( ( r, i ) => {

				r.latest = marks[ i ];
				r.producers.length = marks[ i ] + 1;

			} );
			throw error;

		} finally {

			this.building = null;

		}

	}

	// Checks that need the whole setup of one pass.
	private checkPass( pass: PassDecl ): void {

		const name = pass.name;
		// no read and write of one physical texture in a pass (a ping-pong's previous texture is another)
		const written = new Set( pass.writes.map( ( w ) => w.resource ) );
		if ( pass.depth ) written.add( pass.depth.resource );
		for ( const w of pass.storage ) {

			if ( written.has( w.resource ) ) fail( `'${ name }' writes '${ this.resources[ w.resource ].name }' as an attachment and through storage` );
			written.add( w.resource );

		}
		for ( const r of pass.reads ) if ( ! r.previous && written.has( r.resource ) )
			fail( `'${ name }' reads and writes '${ this.resources[ r.resource ].name }'` );
		if ( pass.writes.length > 1 && ! pass.hasMrt ) fail( `'${ name }' writes ${ pass.writes.length } attachments without an mrt` );
		// three r186 sets one clear flag for every colour attachment of a pass (renderer.autoClearColor)
		if ( pass.writes.length > 1 && new Set( pass.writes.map( ( w ) => drawsOver( pass, w ) ) ).size > 1 )
			fail( `'${ name }' clears some of its attachments and loads others: three clears all or none` );
		// a three r186 RenderTarget always carries a colour attachment (its textures[ 0 ]), which a
		// depth-only pass would draw into, store and key its pipelines by; the depth-only draws
		// planned (the vegetation's depth copies) stay inside their colour pass with colorWrite off
		if ( pass.depth && pass.writes.length === 0 && pass.kind !== 'compute' )
			fail( `'${ name }' has a depth attachment but no colour attachment: depth-only passes are not supported (three's render targets carry a colour texture)` );
		if ( pass.kind === 'compute' && ( pass.writes.length > 0 || pass.depth ) )
			fail( `'${ name }': a compute pass has no attachments (declare what it writes with storage())` );
		const attachments = [ ...pass.writes.map( ( w ) => w.resource ), ...( pass.depth ? [ pass.depth.resource ] : [] ) ].map( ( id ) => this.resources[ id ] );
		if ( attachments.length > 1 ) {

			const key = sizeKey( attachments[ 0 ].desc.size );
			const samples = attachments[ 0 ].desc.samples ?? 1;
			for ( const a of attachments ) {

				if ( a.canvas ) fail( `'${ name }': the canvas cannot share a pass with other attachments` );
				if ( sizeKey( a.desc.size ) !== key ) fail( `'${ name }': attachments of different size classes ('${ attachments[ 0 ].name }', '${ a.name }')` );
				if ( ( a.desc.samples ?? 1 ) !== samples ) fail( `'${ name }': attachments of different sample counts ('${ attachments[ 0 ].name }', '${ a.name }')` );

			}

		}

	}

}

// ---- compiled graph

export interface CompileOptions {
	// the scene's sample count (`samples: 'scene'`)
	sceneSamples: number;
	// the passes whose multisampled stores the legacy table discards (today's: the rain)
	legacyDiscard?: readonly string[];
}

export interface TargetInfo {
	readonly key: string;
	readonly colors: number[];
	readonly depth: number | null;
	readonly size: SizeClass;
	readonly samples: number;
	readonly pingPong: boolean;
	readonly canvas: boolean;
}

export type LoadOp = 'clear' | 'load';
export type StoreOp = 'store' | 'discard';
export interface ColorOps { resource: string; load: LoadOp; store: StoreOp; resolve: boolean | null }
export interface DepthOps { resource: string; write: boolean; load: LoadOp; store: StoreOp }
export interface PassOps {
	pass: string;
	index: number;
	target: string | null;
	colors: ColorOps[];
	depth: DepthOps | null;
	// whether the descriptor three records differs (a store discarded or a resolve dropped)
	rewrite: boolean;
}

export interface OpsTable {
	readonly mode: OpsMode;
	readonly mask: number;
	readonly running: string[];
	readonly passes: PassOps[];
	// descriptors the runtime rewrites in a frame with this mask
	readonly rewrites: number;
}

/** The compiled graph: live passes in order, their targets, and the ops per enabled mask. */
export class CompiledGraph {

	readonly decl: GraphDecl;
	readonly options: Readonly<Required<CompileOptions>>;
	/** Pass indices that execute (not culled), in order. */
	readonly live: number[];
	readonly culled: string[];
	/** The live passes with a `when` predicate, in order: bit i of a mask enables the i-th. */
	readonly conditional: number[];
	/** Per pass index: its render target (null: none, or the canvas through `canvasTarget`). */
	readonly targetOf: ( TargetInfo | null )[];
	readonly targets: TargetInfo[];
	readonly jitterSpan: [ number, number ] | null;
	// the derived tables, per mode and enabled mask: a frame looks its mask up without allocating
	private tables: Record<OpsMode, Map<number, OpsTable>> = { legacy: new Map(), derived: new Map(), store: new Map() };
	// resources whose last version in a frame is read by a later frame
	private crossFrame: Set<number>;

	constructor( decl: GraphDecl, options: CompileOptions ) {

		this.decl = decl;
		this.options = { legacyDiscard: [ 'rain' ], ...options };
		const { passes, resources } = decl;
		// ---- whole-graph validation
		const jittered = passes.filter( ( p ) => p.jittered ).map( ( p ) => p.index );
		if ( jittered.length > 0 && jittered[ jittered.length - 1 ] - jittered[ 0 ] !== jittered.length - 1 ) {

			fail( `jittered passes must be contiguous: ${ jittered.map( ( i ) => passes[ i ].name ).join( ', ' ) }` );

		}
		for ( const c of decl.copies ) {

			const d = resources[ c.dst.resource ];
			for ( const p of passes ) if ( [ ...p.writes, ...p.storage ].some( ( w ) => w.resource === d.id ) ) fail( `'${ p.name }' writes '${ d.name }', the frame-end copy's target` );
			const s = resources[ c.src.resource ];
			const ds = d.desc, ss = s.desc;
			if ( sizeKey( ds.size ) !== sizeKey( ss.size ) || ds.format !== ss.format || ds.type !== ss.type ) fail( `copyToPersistent: '${ s.name }' and '${ d.name }' differ in size class, format or type` );
			if ( this.samplesOf( s ) !== 1 || this.samplesOf( d ) !== 1 ) fail( `copyToPersistent: '${ s.name }' -> '${ d.name }' must be single-sampled` );

		}
		for ( const r of resources ) if ( r.kind === 'import' && r.canvas && r.producers.length === 1 ) fail( `nothing writes the canvas '${ r.name }'` );

		this.crossFrame = new Set();
		for ( const r of resources ) {

			const readsBefore = passes.some( ( p ) => p.reads.some( ( x ) => x.resource === r.id && ( x.previous || ( x.version === 0 && r.persistent ) ) ) );
			const optionalLate = passes.some( ( p ) => p.reads.some( ( x ) => x.resource === r.id && x.optional ) )
				&& r.producers.some( ( i ) => i >= 0 && passes[ i ].conditional );
			if ( r.persistent || r.pingPong || readsBefore || optionalLate || decl.copies.some( ( c ) => c.src.resource === r.id ) || decl.debugView === r.id ) this.crossFrame.add( r.id );

		}

		// ---- culling by liveness
		const live = new Set<number>();
		const work: number[] = [];
		const mark = ( i: number ) => {

			if ( i >= 0 && ! live.has( i ) ) {

				live.add( i );
				work.push( i );

			}

		};
		// the producers a reader of version v may see: v's own, and while that one may be skipped
		// (or loads what it draws over), the ones before
		const producersOf = ( r: ResourceDecl, v: number ) => {

			for ( let u = v; u > 0; u -- ) {

				const p = passes[ r.producers[ u ] ];
				mark( p.index );
				const w = p.writes.find( ( x ) => x.resource === r.id );
				const clears = w ? drawsOver( p, w ) : p.depth?.clear === true;
				if ( ! p.conditional && clears ) break;

			}

		};
		for ( const p of passes ) {

			const rooted = p.neverCull || [ ...p.writes, ...p.storage, ...( p.depth?.write ? [ p.depth ] : [] ) ].some( ( w ) => {

				const r = resources[ w.resource ];
				return r.canvas || r.persistent || r.pingPong || decl.debugView === r.id;

			} );
			if ( rooted ) mark( p.index );

		}
		for ( const c of decl.copies ) producersOf( resources[ c.src.resource ], resources[ c.src.resource ].latest );
		for ( const r of resources ) if ( r.pingPong ) producersOf( r, r.latest );
		while ( work.length > 0 ) {

			const p = passes[ work.pop()! ];
			for ( const x of p.reads ) if ( ! x.previous ) producersOf( resources[ x.resource ], x.version );
			// a write that loads keeps what it draws over alive; one that clears (or a full-screen
			// pass that overwrites) needs nothing before it. Should the pass itself be skipped, the
			// readers of its version walk back past it (producersOf), so a conditional pass needs no
			// exception here.
			for ( const w of p.writes ) if ( ! drawsOver( p, w ) ) producersOf( resources[ w.resource ], w.version - 1 );
			// (a storage write may leave texels as they were)
			for ( const w of p.storage ) producersOf( resources[ w.resource ], w.version - 1 );
			if ( p.depth && ! ( p.depth.write && p.depth.clear ) ) producersOf( resources[ p.depth.resource ], p.depth.write ? p.depth.version - 1 : p.depth.version );

		}
		this.live = passes.filter( ( p ) => live.has( p.index ) ).map( ( p ) => p.index );
		this.culled = passes.filter( ( p ) => ! live.has( p.index ) ).map( ( p ) => p.name );
		this.conditional = this.live.filter( ( i ) => passes[ i ].conditional );
		if ( this.conditional.length > 30 ) fail( 'more than 30 conditional passes: the enabled mask is a 32-bit integer' );
		const liveJittered = this.live.filter( ( i ) => passes[ i ].jittered );
		this.jitterSpan = liveJittered.length > 0 ? [ liveJittered[ 0 ], liveJittered[ liveJittered.length - 1 ] ] : null;

		// ---- one render target per distinct attachment tuple
		this.targets = [];
		this.targetOf = passes.map( () => null );
		const byKey = new Map<string, TargetInfo>();
		for ( const i of this.live ) {

			const p = passes[ i ];
			const colors = p.writes.map( ( w ) => w.resource );
			const depth = p.depth ? p.depth.resource : null;
			if ( colors.length === 0 && depth === null ) continue;
			const first = resources[ colors[ 0 ] ?? depth! ];
			const key = targetKey( colors.map( ( c ) => resources[ c ].name ), depth === null ? null : resources[ depth ].name );
			let t = byKey.get( key );
			if ( ! t ) {

				t = {
					key, colors, depth, size: first.desc.size, samples: this.samplesOf( first ),
					pingPong: colors.some( ( c ) => resources[ c ].pingPong ), canvas: first.canvas,
				};
				byKey.set( key, t );
				this.targets.push( t );

			}
			this.targetOf[ i ] = t;

		}
		// the runtime keeps one parity per ping-pong resource and swaps it only when a pass wrote the
		// resource; ping-pong colours sharing a target must therefore always be written together, or
		// their parities part and the target's two render targets fit neither
		for ( const t of this.targets ) {

			const pp = t.colors.filter( ( c ) => resources[ c ].pingPong );
			if ( pp.length < 2 ) continue;
			for ( const p of passes ) {

				const n = p.writes.filter( ( w ) => pp.includes( w.resource ) ).length;
				if ( n > 0 && n !== pp.length ) fail( `'${ p.name }' writes some of the ping-pong attachments of '${ t.key }' but not all` );

			}

		}

	}

	samplesOf( r: ResourceDecl ): number {

		const s = r.desc.samples ?? 1;
		return s === 'scene' ? this.options.sceneSamples : s;

	}

	get passNames(): string[] {

		return this.live.map( ( i ) => this.decl.passes[ i ].name );

	}

	/** The mask with exactly the named conditional passes enabled. */
	maskOf( enabled: Iterable<string> ): number {

		const names = new Set( enabled );
		let mask = 0;
		this.conditional.forEach( ( index, bit ) => {

			if ( names.has( this.decl.passes[ index ].name ) ) mask |= 1 << bit;

		} );
		return mask;

	}

	/** The mask with every conditional pass enabled. */
	get fullMask(): number {

		return this.conditional.length === 0 ? 0 : ( 2 ** this.conditional.length ) - 1;

	}

	/** The attachment ops of every pass that runs under `mask`: derived on the first frame with that
	 * mask, then cached (the same object every frame; deterministic). */
	ops( mode: OpsMode, mask: number ): OpsTable {

		const cache = this.tables[ mode ] ?? fail( `unknown ops mode '${ mode }'` );
		let table = cache.get( mask );
		if ( ! table ) {

			table = this.deriveOps( mode, mask );
			cache.set( mask, table );

		}
		return table;

	}

	/** The pass indices that run under `mask`, in order. */
	running( mask: number ): number[] {

		const off = new Set( this.conditional.filter( ( _, bit ) => ( mask & ( 1 << bit ) ) === 0 ) );
		return this.live.filter( ( i ) => ! off.has( i ) );

	}

	private deriveOps( mode: OpsMode, mask: number ): OpsTable {

		const { passes, resources } = this.decl;
		const run = this.running( mask );
		const clearsColour = drawsOver;
		const legacy = new Set( this.options.legacyDiscard );
		const lastWriter = new Map<number, number>();
		for ( const i of run ) {

			const p = passes[ i ];
			for ( const w of p.writes ) lastWriter.set( w.resource, i );
			if ( p.depth?.write ) lastWriter.set( p.depth.resource, i );

		}
		const out: PassOps[] = [];
		run.forEach( ( index, k ) => {

			const p = passes[ index ];
			const later = run.slice( k + 1 ).map( ( i ) => passes[ i ] );
			const t = this.targetOf[ index ];
			const colors: ColorOps[] = p.writes.map( ( w ) => {

				const r = resources[ w.resource ];
				const msaa = this.samplesOf( r ) > 1;
				const load: LoadOp = clearsColour( p, w ) ? 'clear' : 'load';
				const last = lastWriter.get( r.id ) === index;
				// scan the later passes up to the next write of this resource
				let loaded = false, samplesRead = false, sampled = false;
				for ( const q of later ) {

					for ( const x of q.reads ) if ( x.resource === r.id && ! x.previous ) {

						if ( x.mode === 'samples' ) samplesRead = true;
						else sampled = true;

					}
					const qw = q.writes.find( ( x ) => x.resource === r.id );
					if ( qw ) {

						if ( ! clearsColour( q, qw ) ) loaded = true;
						break;

					}

				}
				const keep = r.canvas || r.kind === 'import' || this.decl.debugView === r.id;
				const carried = this.crossFrame.has( r.id ) && last;
				let store: StoreOp, resolve: boolean | null = null;
				if ( mode === 'store' ) {

					store = 'store';
					if ( msaa ) resolve = true;

				} else if ( mode === 'legacy' ) {

					store = msaa && legacy.has( p.name ) ? 'discard' : 'store';
					if ( msaa ) resolve = true;

				} else if ( msaa ) {

					store = loaded || samplesRead || keep || ( carried && r.persistent ) ? 'store' : 'discard';
					resolve = sampled || last;

				} else {

					store = loaded || samplesRead || sampled || keep || carried ? 'store' : 'discard';

				}
				return { resource: r.name, load, store, resolve };

			} );
			let depth: DepthOps | null = null;
			if ( p.depth ) {

				const d = p.depth, r = resources[ d.resource ];
				const msaa = this.samplesOf( r ) > 1;
				let used = false;
				for ( const q of later ) {

					if ( q.reads.some( ( x ) => x.resource === r.id ) ) {

						used = true;
						break;

					}
					if ( q.depth?.resource === r.id ) {

						used = ! q.depth.clear;
						break;

					}

				}
				const carried = this.crossFrame.has( r.id ) && lastWriter.get( r.id ) === index;
				const store: StoreOp = mode === 'store' ? 'store'
					: mode === 'legacy' ? ( msaa && legacy.has( p.name ) ? 'discard' : 'store' )
						: ( used || carried || this.decl.debugView === r.id ? 'store' : 'discard' );
				depth = { resource: r.name, write: d.write, load: d.clear ? 'clear' : 'load', store };

			}
			const rewrite = colors.some( ( c ) => c.store === 'discard' || c.resolve === false ) || depth?.store === 'discard';
			out.push( { pass: p.name, index, target: t ? t.key : null, colors, depth, rewrite } );

		} );
		return {
			mode, mask, running: run.map( ( i ) => passes[ i ].name ), passes: out,
			rewrites: out.filter( ( o ) => o.rewrite ).length,
		};

	}

	/** A plain description for tables and tests (deterministic). */
	summary(): object {

		const { passes, resources } = this.decl;
		const desc = ( id: number ) => {

			const r = resources[ id ];
			return { name: r.name, size: sizeKey( r.desc.size ), format: r.desc.format ?? null, type: r.desc.type ?? null, samples: this.samplesOf( r ) };

		};
		return {
			passes: passes.map( ( p ) => ( {
				name: p.name, kind: p.kind, label: p.label, when: p.whenLabel, jittered: p.jittered, culled: ! this.live.includes( p.index ),
				target: this.targetOf[ p.index ]?.key ?? null,
				size: sizeKey( p.size ?? ( p.writes[ 0 ] ? resources[ p.writes[ 0 ].resource ].desc.size : p.depth ? resources[ p.depth.resource ].desc.size : { of: 'fixed', width: 0, height: 0 } ) ),
				reads: p.reads.map( ( r ) => `${ resources[ r.resource ].name }${ r.previous ? '@prev' : `@${ r.version }` }:${ r.mode }${ r.optional ? '?' : '' }` ),
				writes: p.writes.map( ( w ) => desc( w.resource ) ),
				...( p.storage.length > 0 ? { storage: p.storage.map( ( w ) => desc( w.resource ) ) } : {} ),
				depth: p.depth ? { ...desc( p.depth.resource ), write: p.depth.write } : null,
			} ) ),
			targets: this.targets.map( ( t ) => ( { key: t.key, size: sizeKey( t.size ), samples: t.samples, pingPong: t.pingPong } ) ),
			copies: this.decl.copies.map( ( c ) => `${ c.src.name }->${ c.dst.name }` ),
			jitterSpan: this.jitterSpan && this.jitterSpan.map( ( i ) => passes[ i ].name ),
		};

	}

}

export function compileGraph( decl: GraphDecl, options: CompileOptions ): CompiledGraph {

	return new CompiledGraph( decl, options );

}
