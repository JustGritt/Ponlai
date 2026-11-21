// The three.js WebGPU backend and its device: each use of its internals is a named function.
//
// three's WebGPU backend and the device it holds (three r186): `renderer.backend` is public, but
// what it carries (the device, the GPU objects behind three's textures and attributes, the
// timestamp queries' pools, the features, the methods we wrap) is not typed and changes between
// versions. Each use of these three.js internals is a function here, named for what it does; the
// rest of the app calls these and never names the backend. The entries they depend on are in `three-internals.json`;
// `checkRenderer` (Internals.ts) checks the backend's members once the renderer is initialised.
//
// The WebGPU types are the few members used (the project builds without @webgpu/types): a caller
// with its own narrower type of the device names it as the type argument.

type Callable = ( this: unknown, ...args: unknown[] ) => unknown;
type Backend = Record<string, unknown> & {
	isWebGPUBackend?: boolean;
	device: { adapterInfo?: AdapterInfo } | null;
	context: unknown;
	trackTimestamp: boolean;
	timestampQueryPool: Record<string, Pool | null>;
	hasFeature( name: string ): boolean;
	get( object: unknown ): { texture?: object; buffer?: object; _paddedItemSize?: number };
};
type AdapterInfo = { vendor?: string; architecture?: string; device?: string; description?: string };
type Pool = {
	querySet: object; resolveBuffer: object; queryOffsets: Map<string, number>;
	currentQueryIndex: number; maxQueries: number; resolveQueriesAsync: () => Promise<number>;
};

function backendOf( renderer: unknown ): Backend {

	return ( renderer as { backend: Backend } ).backend;

}

// the method to call through: an instance's own wrapper installed before ours, else the
// prototype's at call time (so later prototype patches are seen)
function through( host: object, own: Callable | null, method: string ): Callable {

	return own ?? ( Object.getPrototypeOf( host ) as Record<string, Callable> )[ method ];

}

/** Whether the renderer runs on three's WebGPU backend (not its WebGL fallback). */
export function isWebGPU( renderer: unknown ): boolean {

	return backendOf( renderer ).isWebGPUBackend === true;

}

/** The GPUDevice three renders with. */
export function gpuDevice<Device = unknown>( renderer: unknown ): Device {

	return backendOf( renderer ).device as Device;

}

/** The adapter's description (vendor, architecture, device), where the device reports one. */
export function gpuAdapterInfo( renderer: unknown ): AdapterInfo | undefined {

	return backendOf( renderer ).device?.adapterInfo;

}

/** Whether the device was created with the WebGPU feature `name`. */
export function hasGpuFeature( renderer: unknown, name: string ): boolean {

	return backendOf( renderer ).hasFeature( name ) === true;

}

/** The canvas's GPUCanvasContext (configured on first use). */
export function canvasContext<Context = unknown>( renderer: unknown ): Context {

	return backendOf( renderer ).context as Context;

}

/** The GPUTexture three made for `texture` (undefined while three has made none). */
export function gpuTexture<Texture = object>( renderer: unknown, texture: object ): Texture | undefined {

	return backendOf( renderer ).get( texture ).texture as Texture | undefined;

}

/** The GPUBuffer three made for `attribute` (a BufferAttribute or an interleaved buffer; undefined
 * while three has made none). */
export function gpuBuffer<Buffer = object>( renderer: unknown, attribute: object ): Buffer | undefined {

	return backendOf( renderer ).get( attribute ).buffer as Buffer | undefined;

}

/** Sets the GPUBuffer three keeps for `attribute`, as three's createAttribute sets the one it
 * makes: three then finds it made, and owns it from then on. */
export function setGpuBuffer( renderer: unknown, attribute: object, buffer: object ): void {

	backendOf( renderer ).get( attribute ).buffer = buffer;

}

/** Wraps the backend's `beginRender` (an instance property): `wrapper` runs at the start of each
 * render pass three begins, with the render context and the call to the method it wraps (an
 * instance's own wrapper installed before, else the prototype's at call time). Returns what puts
 * back what was there, where this wrapper is still the property. */
export function wrapBeginRender<Context>( renderer: unknown, wrapper: ( renderContext: Context, begin: () => unknown ) => unknown ): () => void {

	const backend = backendOf( renderer );
	const own = Object.prototype.hasOwnProperty.call( backend, 'beginRender' ) ? backend.beginRender as Callable : null;
	const wrapped = function ( this: unknown, renderContext: Context ) {

		return wrapper( renderContext, () => through( backend, own, 'beginRender' ).call( backend, renderContext ) );

	};
	backend.beginRender = wrapped;
	return () => {

		if ( backend.beginRender !== wrapped ) return;
		if ( own ) backend.beginRender = own;
		else delete backend.beginRender;

	};

}

/** Wraps the backend's render and compute pipeline creation once per `owner` (a pipeline counter, a
 * profiler), each wrapper calling through to what the method was when it was installed:
 * `around( create, kind )` returns what runs in its place. Every pipeline three makes, synchronous
 * or asynchronous, goes through one of the two. Returns false where `owner` had already wrapped
 * them. */
export function wrapPipelineCreation( renderer: unknown, owner: symbol, around: ( create: ( ...args: unknown[] ) => unknown, kind: 'render' | 'compute' ) => ( ...args: unknown[] ) => unknown ): boolean {

	const backend = backendOf( renderer ) as Backend & Record<symbol, unknown>;
	if ( backend[ owner ] ) return false;
	backend[ owner ] = true;
	for ( const [ method, kind ] of [ [ 'createRenderPipeline', 'render' ], [ 'createComputePipeline', 'compute' ] ] as const ) {

		const original = backend[ method ] as Callable;
		backend[ method ] = around( ( ...args: unknown[] ) => original.apply( backend, args ), kind );

	}
	return true;

}

// The staged attribute writes a backend's `updateAttribute` takes, once wrapped.
const attributeWrites = new WeakMap<object, AttributeWrite>();

/** What decides an attribute write three's backend is about to make: called with the attribute's
 * data (the attribute, or an interleaved one's buffer), its GPUBuffer (undefined while none is made,
 * and for data three pads) and three's own write; returns nothing (the write is its business). */
export type AttributeWrite = ( source: object, buffer: object | undefined, write: () => void ) => void;

/** Routes every attribute write of the backend (`updateAttribute`) through `write`, installed once
 * per backend: a second call keeps the first's. */
export function wrapAttributeWrites( renderer: unknown, write: AttributeWrite ): void {

	const backend = backendOf( renderer );
	if ( attributeWrites.has( backend ) ) return;
	attributeWrites.set( backend, write );
	const update = backend.updateAttribute as Callable;
	backend.updateAttribute = function ( this: Backend, attribute: unknown ) {

		const source = ( attribute as { isInterleavedBufferAttribute?: boolean; data: object } ).isInterleavedBufferAttribute ? ( attribute as { data: object } ).data : attribute as object;
		const gpu = this.get( source );
		// (a buffer three pads, its items rounded up to 4 bytes, is written by three only)
		return write( source, gpu._paddedItemSize !== undefined ? undefined : gpu.buffer, () => update.call( this, attribute ) );

	};

}

// ---- timestamp queries

/** Whether three records the passes' GPU timestamps (`trackTimestamp`, and the device has them). */
export function tracksTimestamps( renderer: unknown ): boolean {

	return backendOf( renderer ).trackTimestamp === true;

}

/** One of three's timestamp query pools (render, compute). */
export type TimestampPool = { readonly kind: 'TimestampPool' };

/** three's timestamp query pools that exist, each one's own read-back (which three makes only when
 * a pool overflows) routed to `overflow` once per `owner`. */
export function timestampPools( renderer: unknown, owner: string, overflow: ( pool: TimestampPool ) => void ): TimestampPool[] {

	const pools: TimestampPool[] = [];
	for ( const pool of Object.values( backendOf( renderer ).timestampQueryPool ) ) {

		if ( ! pool ) continue;
		const hooked = pool as Pool & Record<string, unknown>;
		if ( ! hooked[ owner ] ) {

			hooked[ owner ] = true;
			pool.resolveQueriesAsync = () => { overflow( pool as unknown as TimestampPool ); return Promise.resolve( 0 ); };

		}
		pools.push( pool as unknown as TimestampPool );

	}
	return pools;

}

/** How many queries `pool` holds since it was last taken. */
export function pendingTimestamps( pool: TimestampPool ): number {

	return ( pool as unknown as Pool ).currentQueryIndex;

}

/** The most queries `pool` holds (each 8 bytes once resolved). */
export function timestampCapacity( pool: TimestampPool ): number {

	return ( pool as unknown as Pool ).maxQueries;

}

/** Takes `pool`'s queries: their count, and each pass's uid with the index of its first query (the
 * second follows it); three's pool starts over. */
export function takeTimestamps( pool: TimestampPool ): { count: number; offsets: Map<string, number> } {

	const p = pool as unknown as Pool;
	const count = p.currentQueryIndex, offsets = new Map( p.queryOffsets );
	p.currentQueryIndex = 0;
	p.queryOffsets.clear();
	return { count, offsets };

}

/** Records on `encoder` the resolve of `pool`'s first `count` queries into its own buffer and their
 * copy into `destination` (`count` × 8 bytes from 0). */
export function resolveTimestamps( encoder: unknown, pool: TimestampPool, count: number, destination: object ): void {

	const p = pool as unknown as Pool;
	const e = encoder as { resolveQuerySet( set: object, first: number, count: number, destination: object, offset: number ): void; copyBufferToBuffer( source: object, sourceOffset: number, destination: object, destinationOffset: number, size: number ): void };
	e.resolveQuerySet( p.querySet, 0, count, p.resolveBuffer, 0 );
	e.copyBufferToBuffer( p.resolveBuffer, 0, destination, 0, count * 8 );

}
