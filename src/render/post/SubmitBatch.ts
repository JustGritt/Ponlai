import { debugFlags } from '../debug/DebugFlags.ts';
import { settings } from '../Settings.ts';

// Submit batching: three's command buffers submitted a few at a time, in order.
//
// Submit batching: the command buffers three hands to `queue.submit` are held for a moment and
// submitted a few at a time, in the order given, so a frame reaches the GPU in a handful of
// submits instead of one for each.
//
// three's WebGPU backend records every render pass, clear, copy, compute group and mipmap chain
// on a command encoder of its own and submits it at once: dozens of submits a frame here, many of
// them mip levels of the ocean's maps. In Chrome on macOS
// each submit becomes a Metal command buffer, and a Metal queue holds at most 64 uncompleted
// ones: whenever the GPU ran a frame or so behind, the next submit waited for one to complete, in
// the GPU process's main thread, which also presents the page. It stalled for 9-20 ms with almost
// no CPU time, and the display held the previous frame for 3 to 10 refreshes: on a 120 Hz display
// both hitches over 25 ms of a still 20 s trace began with such a stall. Batched, a frame's
// submits stay far from that limit: headless at 60 Hz, no submit blocked for a millisecond
// (before, 4 to 6 a second did, up to 21 ms), and the GPU process spent a third less time decoding
// the page's commands. A test page paced as this one and drawing the same GPU load in one pass was
// presented late 0.1-0.4 % of the time on that display, the scene 1.5-2 %; batched, about two
// fifths fewer of the scene's frames came late (against `submits=each`, in a window and full
// screen), and most of those left follow a major GC.
//
// Batching does not change what a major GC (every few seconds) costs: the page's sweeper then
// releases the WebGPU objects of hundreds of frames at once (encoders, passes and command buffers,
// about 340 a frame), and the GPU process spends 25-50 ms of the next 50 on them either way.
//
// Order: command buffers submitted together run in the order given, exactly as one submit at a
// time. A queue write, though, takes effect before every command buffer submitted after it, so a
// write must not overtake a held command buffer that uses the buffer it writes (three rewrites
// shared uniform buffers between passes). Such a write is staged: its data goes to a staging
// buffer at once, and a copy into place joins the held list at the write's position, behind the
// command buffers that read the old contents and ahead of those submitted after it. A buffer
// counts as used when a held command buffer names it (vertex, index or indirect buffer, a copy, a
// clear, a query resolve), binds it in a bind group or executes a render bundle that does; what
// the encoders still open will use is submitted after the write anyway. A write to a buffer that
// no held command buffer uses goes straight to the queue: nothing held can tell it from one made
// after them. Texture writes, `mapAsync`, `onSubmittedWorkDone`, every `destroy`, error scopes
// and a canvas reconfiguration submit the held command buffers first, as do the end of the
// frame's tick and of any task or microtask that left some held (the canvas's image must be
// submitted before the browser presents it). The GPU runs the same commands in the same order,
// so the image is unchanged; staged writes add a small copy each.
//
// Holding a command buffer delays the GPU's start on it, so one submitted after a pause (the
// tick's first, while the GPU may have run dry) goes at once, and the rest are held a fraction of
// a millisecond at most: batching that held them for most of the tick made the frames late more
// often, not less.
//
// One invalid command buffer makes the queue reject every command buffer submitted with it, and
// the staged copies among them: a pass whose pipeline failed to build (three records it once
// before it skips the pipeline) takes the rest of its batch with it, where a submit of its own
// lost that pass alone. Debug builds watch for the device's uncaptured errors and, after the
// first, submit each command buffer on its own, with one warning.
//
// `debugFlags.batchSubmits = false` submits each command buffer as three hands it over (the old
// path), at runtime, for an A/B in one page; the URL option `submits=each` sets it at load, and
// `submitBatch=hold,wait,idle` overrides the policy below (for display A/Bs). Either way the
// tracking wrappers stay installed, so their JS cost shows only against a build without them.
//
// For other code: the wrappers on the device's, queue's and encoders' methods are installed when
// this module loads, before the renderer. Raw WebGPU work outside three goes through them; a new
// kind of queue operation that must not overtake held command buffers needs a flush here, and a
// buffer a new encoder method reads must join the tracking, or a write may overtake a held command
// buffer that reads it. A profiler or tool that times the queue's calls wraps the natives under
// the batching (`QUEUE_NATIVES` on the queue), not `queue.submit`, or it times them twice.

// command buffers held at most, the longest the oldest one waits (ms of the main thread's time,
// checked as three submits and writes), and the pause after which a command buffer goes at once
// (ms since the last submit)
const HOLD = 8, WAIT_MS = 0.2, IDLE_MS = 1;
// staging buffers are taken in chunks of this size (bytes) and reused after each submit: queue
// operations run in order, so a later write cannot reach a copy already submitted. A larger write
// to a buffer a held command buffer uses submits what is held and goes straight to the queue.
const CHUNK = 1 << 16;

{

	const params = settings;
	if ( params.get( 'submits' ) === 'each' ) debugFlags.batchSubmits = false;
	const policy = params.get( 'submitBatch' );
	if ( policy !== null ) {

		// up to three numbers, none negative; an empty or unreadable one would disable a limit
		const values = policy.split( ',' ).map( ( v ) => v.trim() === '' ? NaN : Number( v ) );
		if ( values.length <= 3 && values.every( ( v ) => v >= 0 ) ) [ debugFlags.submitHold, debugFlags.submitWait, debugFlags.submitIdle ] = values;
		else console.warn( `submitBatch=${ policy }: expected hold,wait,idle (numbers, 0 or more); keeping the defaults` );

	}

}

type Callable = ( this: unknown, ...args: unknown[] ) => unknown;
// a replacement method: any parameter list (the methods' own types are not declared here)
type Method = ( this: unknown, ...args: never[] ) => unknown;
type Host = Record<PropertyKey, unknown>;
type Constructor = abstract new ( ...args: never[] ) => object;
/** The members of the GPU device the batching wraps (three's backend's `device`). */
export type SubmitDevice = { queue: unknown; addEventListener?: ( type: string, listener: ( event: Event ) => void ) => void };

const globals = globalThis as unknown as Record<string, Constructor | undefined>;
const BufferClass = globals.GPUBuffer;
const prototypeOf = ( name: string ) => globals[ name ]?.prototype as unknown as Host | undefined;
const isBuffer = ( x: unknown ): x is Host => BufferClass !== undefined && x instanceof BufferClass;

// on encoders, their passes and command buffers, bundle encoders and bundles: the buffers their
// commands use; on bind groups: the buffers they bind; on buffers: the serial of the batch
// whose held commands use them
const REFS = Symbol( 'submitBatch.refs' );
const GROUP = Symbol( 'submitBatch.group' );
const USED = Symbol( 'submitBatch.used' );

const BUFFER_USAGE_COPY = 0x0004 | 0x0008; // COPY_SRC | COPY_DST
const COPIES = { label: 'submitBatch.copies' };

let active: SubmitBatch | null = null;

// arrays for the encoders' lists, reused once their command buffer is held
const pool: Host[][] = [];

function refsOf( host: Host ): Host[] {

	let refs = host[ REFS ] as Host[] | undefined;
	if ( ! refs ) host[ REFS ] = refs = pool.pop() ?? [];
	return refs;

}

function patch( host: Host | undefined, method: string, wrap: ( original: Callable ) => Method ): void {

	const original = host?.[ method ];
	if ( typeof original === 'function' ) host![ method ] = wrap( original as Callable );

}

// the native methods the staging copies use, taken before the tracking wraps them
const native = {
	copy: prototypeOf( 'GPUCommandEncoder' )?.copyBufferToBuffer as Callable | undefined,
	finish: prototypeOf( 'GPUCommandEncoder' )?.finish as Callable | undefined,
};

// What every encoder's commands use. Installed on the prototypes when this module loads, ahead of
// the renderer, so every bind group is known; encoders and passes are new objects every frame.
{

	patch( prototypeOf( 'GPUDevice' ), 'createBindGroup', ( original ) => function ( this: unknown, descriptor ) {

		const group = original.call( this, descriptor ) as Host;
		let buffers: Host[] | null = null;
		for ( const entry of ( descriptor as { entries?: Iterable<{ resource?: unknown }> } | undefined )?.entries ?? [] ) {

			const r = entry.resource;
			const b = isBuffer( r ) ? r : ( r as { buffer?: unknown } | null )?.buffer;
			if ( isBuffer( b ) ) ( buffers ??= [] ).push( b );

		}
		if ( buffers ) group[ GROUP ] = buffers;
		return group;

	} );
	const command = prototypeOf( 'GPUCommandEncoder' );
	const render = prototypeOf( 'GPURenderPassEncoder' );
	const compute = prototypeOf( 'GPUComputePassEncoder' );
	const bundle = prototypeOf( 'GPURenderBundleEncoder' );
	// a pass adds to its command encoder's list (which it creates, so the pass can share it)
	for ( const method of [ 'beginRenderPass', 'beginComputePass' ] ) {

		patch( command, method, ( original ) => function ( this: unknown ) {

			const pass = original.apply( this, arguments as unknown as unknown[] ) as Host;
			pass[ REFS ] = refsOf( this as Host );
			return pass;

		} );

	}
	// a buffer given directly, as `{ buffer }` (a copy's location) or bound in a bind group
	const add = ( host: Host, x: unknown ) => {

		if ( isBuffer( x ) ) refsOf( host ).push( x );
		else if ( x !== null && typeof x === 'object' ) {

			const group = ( x as Host )[ GROUP ] as Host[] | undefined;
			if ( group ) {

				const refs = refsOf( host );
				for ( let i = 0; i < group.length; i ++ ) refs.push( group[ i ] );

			} else if ( isBuffer( ( x as { buffer?: unknown } ).buffer ) ) refsOf( host ).push( ( x as { buffer: Host } ).buffer );

		}

	};
	const first = ( original: Callable ) => function ( this: unknown, a: unknown ) {

		add( this as Host, a );
		return original.apply( this, arguments as unknown as unknown[] );

	};
	const second = ( original: Callable ) => function ( this: unknown ) {

		add( this as Host, arguments[ 1 ] );
		return original.apply( this, arguments as unknown as unknown[] );

	};
	// ( source, sourceOffset, destination, ... ) or ( source, destination, size )
	const firstThree = ( original: Callable ) => function ( this: unknown ) {

		add( this as Host, arguments[ 0 ] );
		add( this as Host, arguments[ 1 ] );
		add( this as Host, arguments[ 2 ] );
		return original.apply( this, arguments as unknown as unknown[] );

	};
	const fourth = ( original: Callable ) => function ( this: unknown ) {

		add( this as Host, arguments[ 3 ] );
		return original.apply( this, arguments as unknown as unknown[] );

	};
	for ( const host of [ render, compute, bundle ] ) patch( host, 'setBindGroup', second );
	for ( const host of [ render, bundle ] ) {

		patch( host, 'setVertexBuffer', second );
		for ( const method of [ 'setIndexBuffer', 'drawIndirect', 'drawIndexedIndirect' ] ) patch( host, method, first );

	}
	patch( compute, 'dispatchWorkgroupsIndirect', first );
	patch( render, 'executeBundles', ( original ) => function ( this: unknown, bundles: unknown ) {

		// any iterable: read it once, and hand the same list on
		const list = Array.from( bundles as Iterable<Host> );
		for ( const b of list ) {

			const refs = b[ REFS ] as Host[] | undefined;
			if ( refs ) for ( let i = 0; i < refs.length; i ++ ) refsOf( this as Host ).push( refs[ i ] );

		}
		return original.call( this, list );

	} );
	patch( command, 'copyBufferToBuffer', firstThree );
	patch( command, 'copyBufferToTexture', first );
	patch( command, 'copyTextureToBuffer', second );
	patch( command, 'clearBuffer', first );
	patch( command, 'resolveQuerySet', fourth );
	// the list passes on to the command buffer (a bundle keeps its own: it is drawn again)
	for ( const host of [ command, bundle ] ) {

		patch( host, 'finish', ( original ) => function ( this: unknown ) {

			const result = original.apply( this, arguments as unknown as unknown[] ) as Host;
			const refs = ( this as Host )[ REFS ] as Host[] | undefined;
			if ( refs ) {

				result[ REFS ] = host === bundle ? refs.slice() : refs;
				( this as Host )[ REFS ] = undefined;

			}
			return result;

		} );

	}
	// operations that must not overtake a held command buffer
	const flushFirst = ( original: Callable ) => function ( this: unknown ) {

		active?.flush( 'order' );
		return original.apply( this, arguments as unknown as unknown[] );

	};
	patch( prototypeOf( 'GPUBuffer' ), 'mapAsync', flushFirst );
	for ( const name of [ 'GPUBuffer', 'GPUTexture', 'GPUQuerySet' ] ) patch( prototypeOf( name ), 'destroy', flushFirst );
	for ( const method of [ 'configure', 'unconfigure' ] ) patch( prototypeOf( 'GPUCanvasContext' ), method, flushFirst );
	for ( const method of [ 'pushErrorScope', 'popErrorScope' ] ) patch( prototypeOf( 'GPUDevice' ), method, flushFirst );
	const queue = prototypeOf( 'GPUQueue' );
	for ( const method of [ 'writeTexture', 'copyExternalImageToTexture', 'onSubmittedWorkDone' ] ) patch( queue, method, flushFirst );

}

// the byte length of a queue write's data
function byteSize( data: unknown, dataOffset = 0, size?: number ): number {

	if ( ArrayBuffer.isView( data ) && ! ( data instanceof DataView ) ) {

		const view = data as unknown as { length: number; BYTES_PER_ELEMENT: number };
		return ( size ?? view.length - dataOffset ) * view.BYTES_PER_ELEMENT;

	}
	return size ?? ( data as { byteLength: number } ).byteLength - dataOffset;

}

/** On the queue, under the batching: its own `submit` and `writeBuffer`, which every real submit
 * and write the batching makes goes through (a registered symbol, so a lazily loaded module reads
 * it without importing this one). A tool that times the queue's calls (the frame profiler) wraps
 * these, so it times each native call once and not the batching's own work. */
export const QUEUE_NATIVES = Symbol.for( 'ponlai.submitBatch.natives' );

export class SubmitBatch {

	/** Command buffers three submitted, the real submits they went out in, the writes staged, and
	 * why each real submit happened. */
	readonly stats = { held: 0, submits: 0, staged: 0, why: {} as Record<string, number> };
	private pending: Host[] = [];
	private spare: Host[] = [];
	private serial = 1;
	private pendingSince = 0;
	private lastSubmit = - Infinity;
	private scheduled = false;
	private flushing = false;
	// set after a device error in a debug build: every command buffer goes on its own
	private fallback = false;
	private copies: Host | null = null;
	private readonly chunks: Host[] = [];
	private chunk = 0;
	private chunkOffset = 0;
	private staged: Host | null = null;
	private stagedOffset = 0;
	private readonly device: Host;
	private readonly queue: Host;
	private readonly native: { submit: Callable; writeBuffer: Callable };
	private readonly createEncoder: Callable;
	private readonly createBuffer: Callable;
	private readonly copy: Callable;
	private readonly finish: Callable;
	private readonly onMicrotask = () => {

		this.scheduled = false;
		this.flush( 'task' );

	};

	/** `debug`: watch the device's uncaptured errors and stop batching after the first (see above). */
	constructor( device: SubmitDevice, { debug = false } = {} ) {

		this.device = device as unknown as Host;
		const queue = this.queue = device.queue as Host;
		const deviceProto = prototypeOf( 'GPUDevice' );
		// the native methods, past any wrapper (the staging copies need no tracking)
		this.createEncoder = deviceProto?.createCommandEncoder as Callable;
		this.createBuffer = deviceProto?.createBuffer as Callable;
		this.copy = native.copy as Callable;
		this.finish = native.finish as Callable;
		this.native = { submit: queue.submit as Callable, writeBuffer: queue.writeBuffer as Callable };
		if ( ! this.createEncoder || ! this.createBuffer || ! this.copy || ! this.finish || ! this.native.submit || ! this.native.writeBuffer ) return;
		active = this;
		queue[ QUEUE_NATIVES ] = this.native;
		const self = this;
		if ( debug ) device.addEventListener?.( 'uncapturederror', ( event ) => {

			if ( self.fallback || debugFlags.batchSubmits === false ) return;
			self.fallback = true;
			self.flush( 'error' );
			const message = ( event as Event & { error?: { message?: string } } ).error?.message ?? 'unknown';
			console.warn( `SubmitBatch: a WebGPU error (${ message.split( '\n' )[ 0 ] }); from now on each command buffer is submitted on its own, as an invalid one rejects every command buffer submitted with it` );

		} );
		patch( queue, 'submit', () => function ( this: unknown, buffers: unknown ) {

			if ( debugFlags.batchSubmits === false || self.fallback ) {

				self.flush( 'off' );
				return self.native.submit.call( queue, buffers );

			}
			self.closeCopies();
			const pending = self.pending;
			// eslint-disable-next-line ponlai/scene-clock -- submission pacing: when command buffers go to the GPU, not what they draw
			const now = performance.now();
			const idle = pending.length === 0 && now - self.lastSubmit >= Number( debugFlags.submitIdle ?? IDLE_MS );
			if ( pending.length === 0 ) self.pendingSince = now;
			// read once: any iterable may be given
			for ( const cb of buffers as Iterable<Host> ) {

				const refs = cb[ REFS ] as Host[] | undefined;
				if ( refs ) {

					for ( let i = 0; i < refs.length; i ++ ) refs[ i ][ USED ] = self.serial;
					refs.length = 0;
					pool.push( refs );
					cb[ REFS ] = undefined;

				}
				pending.push( cb );
				self.stats.held ++;

			}
			if ( idle ) self.flush( 'idle' );
			else if ( pending.length >= Number( debugFlags.submitHold ?? HOLD ) ) self.flush( 'full' );
			// eslint-disable-next-line ponlai/scene-clock -- submission pacing: when command buffers go to the GPU, not what they draw
			else if ( performance.now() - self.pendingSince >= Number( debugFlags.submitWait ?? WAIT_MS ) ) self.flush( 'time' );
			else self.schedule();

		} );
		patch( queue, 'writeBuffer', () => function ( this: unknown, buffer: Host, bufferOffset: number, data: unknown, dataOffset?: number, size?: number ) {

			// three writes uniforms all through a pass's recording: the time limit holds here too
			// eslint-disable-next-line ponlai/scene-clock -- submission pacing: when command buffers go to the GPU, not what they draw
			if ( self.pending.length > 0 && performance.now() - self.pendingSince >= Number( debugFlags.submitWait ?? WAIT_MS ) ) self.flush( 'time' );
			if ( buffer[ USED ] === self.serial ) {

				const bytes = byteSize( data, dataOffset, size );
				// too large to stage: what is held goes first, and the write after it
				if ( bytes > CHUNK ) self.flush( 'large' );
				else {

					// a held command buffer uses this buffer: write a staging copy now, copy it in its turn
					self.stage( bytes );
					self.native.writeBuffer.call( queue, self.staged, self.stagedOffset, data, dataOffset ?? 0, size );
					self.copies ??= self.createEncoder.call( self.device, COPIES ) as Host;
					self.copy.call( self.copies, self.staged, self.stagedOffset, buffer, bufferOffset, bytes );
					self.stats.staged ++;
					self.schedule();
					return;

				}

			}
			return self.native.writeBuffer.apply( queue, arguments as unknown as unknown[] );

		} );

	}

	/** Submits the held command buffers, in order (`why` counts the reason); the end of the tick
	 * calls it with 'tick'. */
	flush( why = 'tick' ): void {

		if ( this.flushing ) return;
		this.closeCopies();
		if ( this.pending.length === 0 ) return;
		// the held list is handed over and a fresh one started before the submit, so a submit that
		// throws leaves nothing held twice
		const list = this.pending;
		this.pending = this.spare;
		this.flushing = true;
		try {

			this.native.submit.call( this.queue, list );

		} finally {

			this.flushing = false;
			list.length = 0;
			this.spare = list;
			// eslint-disable-next-line ponlai/scene-clock -- submission pacing: when command buffers go to the GPU, not what they draw
			this.lastSubmit = performance.now();
			this.serial ++;
			this.chunk = 0;
			this.chunkOffset = 0;
			this.stats.submits ++;
			this.stats.why[ why ] = ( this.stats.why[ why ] ?? 0 ) + 1;

		}

	}

	// the staged writes' copies join the held list here, ahead of what is submitted next
	private closeCopies(): void {

		if ( this.copies === null ) return;
		this.pending.push( this.finish.call( this.copies ) as Host );
		this.copies = null;

	}

	// staging space for a write of `bytes` bytes (CHUNK at most): sets the buffer and the offset
	private stage( bytes: number ): void {

		const aligned = ( bytes + 15 ) & ~ 15;
		if ( this.chunkOffset + aligned > CHUNK ) {

			this.chunk ++;
			this.chunkOffset = 0;

		}
		let buffer = this.chunks[ this.chunk ];
		if ( ! buffer ) {

			buffer = this.createBuffer.call( this.device, { label: 'submitBatch.staging', size: CHUNK, usage: BUFFER_USAGE_COPY } ) as Host;
			this.chunks.push( buffer );

		}
		this.staged = buffer;
		this.stagedOffset = this.chunkOffset;
		this.chunkOffset += aligned;

	}

	// what is held at the end of a task or microtask goes out before the browser moves on
	private schedule(): void {

		if ( this.scheduled ) return;
		this.scheduled = true;
		queueMicrotask( this.onMicrotask );

	}

}
