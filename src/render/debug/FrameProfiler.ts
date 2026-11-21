// The frame profiler for `?profile`: each pass's GPU time and the frame's CPU time.

// three is imported for its types only: the app hands in the inspector's base class, so this
// lazily loaded module does not split three out of the main bundle into a chunk of its own.
import type * as THREE from 'three/webgpu';
import { canvasContext, gpuDevice, pendingTimestamps, resolveTimestamps, takeTimestamps, timestampPools, tracksTimestamps, wrapPipelineCreation, type TimestampPool } from '../three/Backend.ts';

// Frame profiler for `?profile` (loaded only then; without it the app builds the same renderer,
// pipelines and image). Three's inspector hooks see every render and compute pass with the uid of
// its timestamp query; the renderer's `trackTimestamp` writes each pass's GPU start and end, which
// are resolved here every frame into a ring of read-back buffers (three's own pool resolves into
// one buffer and drops a frame's queries while that buffer is mapped).
//
// A pass's GPU time is the time it adds to the GPU's timeline, not its end minus its start: a
// tile-based GPU stamps a render pass's start when its vertex stage begins, and the tiler runs a
// full-screen pass's few vertices long before the passes it reads have finished, so the raw
// intervals overlap and summed to several times the frame. Each instant of the frame's busy time
// (the union of its passes' intervals since the last frame's work ended, which on a GPU-bound frame
// is the frame interval) is charged to the pass issued first (the order of the inspector's begin
// calls) whose interval covers it. Where passes run one after another this is simply each one's
// duration. Two earlier rules failed:
// - in the order they complete, each from the previous completion, a pass that ran inside a longer
//   one hid the longer one's time before it started: in rain under TAAU the grade table's bake ran
//   inside the opaque scene and the frames' sum fell 1.5 ms short of their GPU time;
// - in the order they start, each from the latest end so far, a render pass whose vertex stage
//   started early took the time of the passes it waited on: under fsr3 the bloom's high pass was
//   stamped 4 ms before the FSR kernels it reads and took their 7 ms (the compute passes' stamps,
//   which start when their dispatches do, came out at 0).
// The previous frame's end is the floor only while it lies before this frame's last end: the
// first frames' stamps can jump by seconds (seen at startup on a Metal GPU), and a floor taken
// from such a frame charged nothing to the frames after it until the clock caught up.
//
// The tick's wall time includes the browser's backpressure: once the GPU falls behind, Chrome
// blocks the calls that hand work to its GPU process until earlier frames drain, so a GPU-bound
// tick looks as long as the frame. Not only `queue.submit`: a `queue.writeBuffer` of a few dozen
// kilobytes waited 10-25 ms for Chrome's transfer buffer at night and on the horizon, the other
// queue writes (`writeTexture`, `copyExternalImageToTexture`) go through the same buffer, and
// `getCurrentTexture` may wait for a free swap-chain image. Nor only the queue: the block lands on
// whichever call finds Chrome's command buffer full, which depends on the frame's command stream.
// On main it was mostly a writeBuffer (and createCommandEncoder); on the chunked island it was the
// first setPipeline of the opaque pass and of the vegetation prepass, 8-10 ms a GPU-bound frame
// that showed as JS. So the encoding calls are timed too, patched on their prototypes: the
// device's createCommandEncoder; the command encoder's pass, copy, clear, resolve and finish calls;
// the passes' state calls, draws and dispatches. Only a call whose own time reaches
// `ENCODE_BLOCK_MS` counts, as `encode`, so their own encoding work (about a thousand calls a
// frame, a few tenths of a millisecond) stays JS. Time in these calls is measured apart, by call
// (`waits`), and the tick's own JS work is the rest. That time includes the calls' own
// copying: about 0.15 ms a frame for the uniform uploads, more for a large upload, which then
// shows as a long write wait rather than as JS. The same holds for the work issued after the
// tick (the sky's bakes), whose submits block as well.
//
// With the frame rate uncapped (no vsync, as the profiling tool runs Chrome by default), nothing
// but that backpressure bounds how far the page runs ahead of the GPU: on the island at noon
// headless Chrome queued about 120 frames (the GPU took 3.5 s to drain once the loop stopped). A
// GPU spike then surfaces in the frame interval that many frames later, so the tool attributes GPU
// hitches on the GPU's own timeline (`gpuInterval`), and the timestamps take as long to read back:
// each frame's queries are resolved into a read-back buffer of a ring at its end, and the frame
// waits for its own read-backs, however long the GPU runs behind (`settle`). A wall-clock cutoff
// (4 s) dropped every pass of every frame once a frame took much over 33 ms (46 ms in the A/B
// runs), since 120 of them then took longer than that to drain.
// Holding the tick until fewer frames are queued was tried and dropped: Chrome reports finished
// work in batches, which bunched the ticks and made the intervals jitter. (A paused page holds its
// ticks so, at two frames: post/FramePacer.ts, "A paused page's queue"; profiled scenes run their
// clocks and are not held.)
//
// Work issued outside the tick (the sky's bakes resume in microtasks after a GPU read-back) is
// timed apart: the JS time of its render and compute calls, and the gap between two ticks.
//
// Each pass is labelled by what it draws or computes:
//   1. a registered render target (`target()`), else
//   2. a registered scene, quad or compute node (`object()`, optionally only into one target), or a
//      name pattern on the compute node, quad or target (`pattern()`, its label fixed or made from
//      the name it matched: FSR 3's kernels each get their own), else
//   3. the pass it is nested in (an rtt() pass rendered from another pass's node graph), else
//   4. the frame graph's pass running (`graphPass()`, which the graph's pass hooks call with the
//      pass's label), else
//   5. the innermost wrapped method on the call stack (`scope()`), else
//   6. a wrapped async method still running (`scope( ..., { async: true } )`: the sky's bakes run
//      in microtasks after the frame's tick), else 'other'.
// So the graph labels its own passes and what three's nodes render as part of them (the bloom
// chains' passes); what runs outside the graph (the sky, the shadow maps, the ocean) is labelled
// by hand.
// The app (app/main.ts, app/scenes/ScenePage.ts) exposes `api` as `window.__PONLAI_PROFILE__`, read by tools/render/profile-frame.mjs.
// `?profile=frames` keeps the frame and tick timing but skips the GPU timestamps. Measure frame
// rates that way: with timestamp queries on, Chrome ran the same frames up to a quarter faster
// (light scenes most; a scene that keeps the GPU saturated hardly at all), so the per-pass split
// shows where the time goes, not the frame rate a player sees.

interface Pass {
	uid: string;
	group: string;
	name: string;
	parent: Pass | null;
	start: number;
	end: number;
	done: boolean;
}

interface Frame {
	id: number;
	// rAF timestamp and performance.now() when the frame's callback started
	time: number;
	now: number;
	js: number;
	jsWall: number;
	wait: number;
	waits: { submit: number; write: number; texture: number; encode: number };
	outside: number;
	outsideWait: number;
	gap: number;
	passes: Pass[];
	pending: number;
	pipelines: number;
	pipelineMs: number;
	calls: number;
	triangles: number;
	finishedAt: number;
	// the read-backs submitted up to its end (`readbacks`): once all of them have returned, a pass
	// still without timestamps never had a query written and will not get one
	readbacks: number;
	scale: number;
}

/** A finished frame, as read by the profiling tool. Times in milliseconds. */
interface FrameRecord {
	id: number;
	t: number; // since navigation start
	interval: number; // rAF time since the previous frame
	nowInterval: number; // performance.now() since the previous frame's callback
	js: number; // the synchronous tick's own work (its wall time less the waits)
	jsWall: number; // the tick's wall time
	submitWait: number; // time the tick spent in queue.submit, the queue writes, getCurrentTexture and the encoding calls' blocks (GPU backpressure)
	waits: { submit: number; write: number; texture: number; encode: number }; // submitWait by call
	outsideJs: number; // JS time of render and compute calls issued after the tick (async bakes), less their waits
	outsideWait: number; // time those calls spent in the queue calls and the encoding calls' blocks
	gap: number; // from the end of the previous tick to the start of this one
	gpu: number; // GPU busy time: the passes' charges (see above)
	pixel: number; // the part of gpu in passes that follow the render's pixel count (`pixelPass`; NaN without it)
	gpuInterval: number; // the GPU's completion of this frame since the previous frame's
	span: number; // first pass start to last pass end on the GPU (may overlap the previous frame)
	passes: number;
	missing: number; // passes without timestamps (no query written, a read-back lost or failed)
	pipelines: number; // render and compute pipelines created this frame
	pipelineMs: number; // JS time spent creating them
	calls: number;
	triangles: number;
	scale: number; // the scene's render scale the frame was drawn at (`renderScale` then)
	groups: Record<string, number>;
	detail?: Record<string, number>;
	raw?: [ string, string, number, number, number ][]; // group, what, start, end (ms from the frame's first start), charge
}

// The few WebGPU types used here (the project builds without @webgpu/types).
type GPUBuffer = { size: number; mapAsync( mode: number, offset: number, size: number ): Promise<void>; getMappedRange( offset: number, size: number ): ArrayBuffer; unmap(): void };
type GPUDevice = {
	createBuffer( descriptor: { label: string; size: number; usage: number } ): GPUBuffer;
	createCommandEncoder( descriptor: { label: string } ): { finish(): object };
	queue: { submit( buffers: object[] ): void };
};
// the calls whose time is counted as waiting (see above), by key in `Frame.waits`
type WaitCall = 'submit' | 'write' | 'texture' | 'encode';
// GPUBufferUsage.MAP_READ | COPY_DST, GPUMapMode.READ
const READBACK_USAGE = 0x0001 | 0x0008;
const MAP_READ = 0x0001;

const PROFILED = Symbol( 'frameProfiler' );
type Named = { name?: string; isQuadMesh?: boolean; isScene?: boolean; isComputeNode?: boolean; texture?: THREE.Texture; width?: number; height?: number };

// read-back buffers in flight: one per pool (render, compute) per frame. In an uncapped run their
// maps resolve as late as the GPU runs behind (about 120 frames on the island at noon), so a ring
// of 128 ran out and lost a pool's passes in about a third of the frames; a buffer holds one
// frame's queries, so a deeper ring costs little memory
const RING = 1024;
// a read-back buffer's size: its frame's queries, rounded up so most frames reuse any free buffer
const readbackBytes = ( queries: number ) => Math.max( 4096, 2 ** Math.ceil( Math.log2( queries * 8 ) ) );
// an encoding call counts as a wait only from this long (ms): its own work (well under a
// microsecond; the coarsened timer reads it as 0 or 0.1 ms) stays JS, a block takes milliseconds
const ENCODE_BLOCK_MS = 0.25;

/** The profiler, built on the renderer's inspector base (`THREE.InspectorBase`). */
export function createFrameProfiler( Base: typeof THREE.InspectorBase, options: { timestamps?: boolean } = {} ) {

	class FrameProfiler extends Base {

		readonly timestamps: boolean;
		/** The app's render scale, recorded with each frame as it starts. */
		renderScale = NaN;
		/** Called with each frame as it finishes (the app's dynamic resolution reads its GPU time). */
		onRecord: ( ( record: FrameRecord ) => void ) | null = null;
		/** Whether the pass with a timestamp uid follows the render's pixel count (the dynamic
		 * resolution's split, `PassScaling`), asked once per timed pass. */
		pixelPass: ( ( uid: string ) => boolean ) | null = null;
		private targets = new WeakMap<object, string>();
		private objects = new WeakMap<object, { label: string; target?: object }[]>();
		private patterns: [ RegExp, string | ( ( name: string ) => string ) ][] = [];
		private stack: string[] = [];
		private graphLabel: string | null = null;
		private asyncScopes = new Map<number, string>();
		private asyncId = 0;
		private frame: Frame | null = null;
		private context: Pass | null = null;
		private open: Frame[] = [];
		private done: FrameRecord[] = [];
		private byUid = new Map<string, Pass>();
		private lastTime = 0;
		private lastNow = 0;
		private lastEnd = - Infinity;
		private lastComplete = false;
		private waits = { submit: 0, write: 0, texture: 0, encode: 0 };
		private outsideWait = 0;
		private inTick = false;
		private lastTickEnd = NaN;
		private outsideDepth = 0;
		private outsideStart = 0;
		private queueWrapped = false;
		private origin: bigint | null = null;
		private buffers: { buffer: GPUBuffer; busy: boolean }[] = [];
		private pipelineCount = 0;
		private pipelineMs = 0;
		private detail = false;
		private raw = false;
		private lost = 0;
		private failed = { map: 0, zero: 0, order: 0, unread: 0 };
		// read-backs submitted so far, and the serials (1, 2, ...) of those still in flight
		private readbacks = 0;
		private inFlight = new Set<number>();
		private latency: number[] = [];
		readonly marks: Record<string, number> = {};
		// the first frames: their ticks, and when the GPU finished each (`done`, since navigation: the
		// first one waits for the GPU process to compile every pipeline it uses)
		readonly startup: { firstFrame?: Record<string, number>; frames: { at: number; interval: number; js: number; wait: number; pipelines: number; pipelineMs: number; done?: number }[] } = { frames: [] };

		constructor( { timestamps = true }: { timestamps?: boolean } = {} ) {

			super();
			this.timestamps = timestamps;

		}

		// ------------------------------------------------------------------ labels

		/** Every pass into `target` belongs to `label`. */
		target( target: object | null | undefined, label: string ): void {

			if ( target && ! this.targets.has( target ) ) this.targets.set( target, label );

		}

		/** Passes drawing `object` (a scene, quad or compute node), optionally only into `target`. */
		object( object: object | null | undefined, label: string, target?: object ): void {

			if ( ! object ) return;
			const list = this.objects.get( object ) ?? [];
			list.push( { label, target } );
			this.objects.set( object, list );

		}

		/** Passes whose compute node, quad or target name matches (the label may be made from the
		 * name). */
		pattern( match: RegExp, label: string | ( ( name: string ) => string ) ): void {

			this.patterns.push( [ match, label ] );

		}

		/** Passes issued until the next call belong to `label` (the frame graph's pass running; null
		 * when it ends), unless a target, object, pattern or parent pass labels them. */
		graphPass( label: string | null ): void {

			this.graphLabel = label;

		}

		/** Passes issued while `object[ method ]` runs belong to `label` (unless labelled otherwise).
		 * An async method keeps its label for passes issued outside any tick until it settles. */
		scope( object: unknown, method: string, label: string, { async = false }: { async?: boolean } = {} ): void {

			const host = object as Record<string, ( ...args: unknown[] ) => unknown> | null;
			const original = host?.[ method ];
			if ( typeof original !== 'function' ) return;
			const profiler = this;
			host![ method ] = function ( this: unknown, ...args: unknown[] ) {

				profiler.stack.push( label );
				let result: unknown;
				try {

					result = original.apply( this, args );

				} finally {

					profiler.stack.pop();

				}
				if ( async && result instanceof Promise ) {

					const id = ++ profiler.asyncId;
					profiler.asyncScopes.set( id, label );
					result.finally( () => profiler.asyncScopes.delete( id ) ).catch( () => {} );

				}
				return result;

			};

		}

		private labelFor( drawn: object | object[], target: Named | null, parent: Pass | null ): string {

			if ( target && this.targets.has( target ) ) return this.targets.get( target )!;
			for ( const object of Array.isArray( drawn ) ? drawn : [ drawn ] ) {

				for ( const entry of this.objects.get( object ) ?? [] ) if ( ! entry.target || entry.target === target ) return entry.label;

			}
			const names = this.names( drawn, target );
			for ( const [ match, label ] of this.patterns ) {

				const name = names.find( ( n ) => match.test( n ) );
				if ( name !== undefined ) return typeof label === 'string' ? label : label( name );

			}
			if ( parent ) return parent.group;
			if ( this.graphLabel !== null ) return this.graphLabel;
			if ( this.stack.length ) return this.stack[ this.stack.length - 1 ];
			if ( this.asyncScopes.size ) return [ ...this.asyncScopes.values() ].pop()!;
			return 'other';

		}

		private names( drawn: object | object[], target: Named | null ): string[] {

			const names: string[] = [];
			for ( const object of Array.isArray( drawn ) ? drawn : [ drawn ] ) {

				const o = object as Named;
				if ( o.name ) names.push( o.name );
				const material = ( o as { material?: { name?: string } } ).material;
				if ( material?.name ) names.push( material.name );

			}
			if ( target?.texture?.name ) names.push( target.texture.name );
			return names;

		}

		private describe( drawn: object | object[], target: Named | null ): string {

			const what = ( Array.isArray( drawn ) ? drawn : [ drawn ] ).map( ( object ) => {

				const o = object as Named;
				return o.name || ( o.isScene ? 'scene' : o.isQuadMesh ? 'quad' : o.isComputeNode ? 'compute' : 'object' );

			} ).join( '+' );
			const into = target ? target.texture?.name || `${ target.width }x${ target.height }` : 'canvas';
			return ( Array.isArray( drawn ) || ( drawn as Named ).isComputeNode ) ? what : `${ what } > ${ into }`;

		}

		// ------------------------------------------------------------------ inspector hooks

		override setRenderer( renderer: THREE.Renderer ): this {

			super.setRenderer( renderer );
			// pipelines created per frame: a compile shows up as a hitch
			if ( renderer ) wrapPipelineCreation( renderer, PROFILED, ( create ) => ( ...args: unknown[] ) => {

				const t0 = performance.now();
				try {

					return create( ...args );

				} finally {

					this.pipelineCount ++;
					this.pipelineMs += performance.now() - t0;

				}

			} );
			return this;

		}

		/** Times the waits in `queue.submit`, the queue writes and the canvas's `getCurrentTexture`
		 * (Chrome's backpressure once the GPU falls behind): in the tick, and outside it while a
		 * render or compute call issued after the tick runs. Under the submit batching it times the
		 * queue's own submit and writeBuffer beneath it (`QUEUE_NATIVES`), so each real call counts
		 * once and the batching's bookkeeping counts as JS. A timed call made inside another (a
		 * texture write that first submits what is held) counts under its own key, and the outer
		 * one only for the rest of its time, so no time counts twice. The encoding calls count only
		 * from `ENCODE_BLOCK_MS` (`encode` in the tick, the outside wait after it). */
		private wrapQueue(): void {

			type Callable = ( ...args: unknown[] ) => unknown;
			const renderer = this.getRenderer();
			const queue = renderer ? gpuDevice<{ queue: Record<string, Callable> } | null>( renderer )?.queue : undefined;
			if ( ! renderer || ! queue ) return;
			this.queueWrapped = true;
			// time of the timed calls made inside the one running
			let inner = 0;
			// `target`: the object the method is called on (the queue, for its natives)
			const wrap = ( host: Record<string, Callable> | null | undefined, method: string, key: WaitCall, target: object | null | undefined = host ) => {

				const original = host?.[ method ];
				if ( ! host || typeof original !== 'function' ) return;
				host[ method ] = ( ...args: unknown[] ) => {

					const outside = ! this.inTick && this.outsideDepth > 0;
					if ( ! this.inTick && ! outside ) return original.apply( target, args );
					const outer = inner;
					inner = 0;
					const t0 = performance.now();
					try {

						return original.apply( target, args );

					} finally {

						const ms = performance.now() - t0;
						const own = ms - inner;
						inner = outer + ms;
						if ( outside ) this.outsideWait += own;
						else this.waits[ key ] += own;

					}

				};

			};
			// SubmitBatch's QUEUE_NATIVES (read by its registered symbol: importing it would split it
			// out of the main bundle)
			const natives = ( queue as unknown as Record<symbol, Record<string, Callable> | undefined> )[ Symbol.for( 'ponlai.submitBatch.natives' ) ];
			wrap( natives ?? queue, 'submit', 'submit', queue );
			wrap( natives ?? queue, 'writeBuffer', 'write', queue );
			wrap( queue, 'writeTexture', 'write' );
			wrap( queue, 'copyExternalImageToTexture', 'write' );
			// the canvas context (it hands out the swap chain's image, the render's final target)
			wrap( canvasContext<Record<string, Callable> | null>( renderer ), 'getCurrentTexture', 'texture' );
			// The encoding calls, on their prototypes: Chrome's backpressure lands on whichever call finds
			// its command buffer full, and on the chunked island that was a render pass's setPipeline,
			// 8-10 ms a frame counted as JS. AttachmentOps' instance wrappers call through to the
			// prototype at call time, so this one runs beneath them; SubmitBatch's prototype wrappers are
			// installed first (its module loads before this one), so this one runs above them, and their
			// tracking is inside the timed call, below the threshold.
			const prototypes: [ string, string[] ][] = [
				[ 'GPUDevice', [ 'createCommandEncoder' ] ],
				[ 'GPUCommandEncoder', [ 'beginRenderPass', 'beginComputePass', 'copyBufferToBuffer', 'copyBufferToTexture', 'copyTextureToBuffer', 'copyTextureToTexture', 'clearBuffer', 'resolveQuerySet', 'finish' ] ],
				[ 'GPURenderPassEncoder', [ 'setPipeline', 'setBindGroup', 'setVertexBuffer', 'setIndexBuffer', 'setViewport', 'setScissorRect', 'setStencilReference', 'setBlendConstant', 'draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect', 'executeBundles', 'end' ] ],
				[ 'GPUComputePassEncoder', [ 'setPipeline', 'setBindGroup', 'dispatchWorkgroups', 'dispatchWorkgroupsIndirect', 'end' ] ],
			];
			for ( const [ type, methods ] of prototypes ) {

				const proto = ( globalThis as unknown as Record<string, { prototype: Record<string, Callable> } | undefined> )[ type ]?.prototype;
				if ( ! proto ) continue;
				for ( const method of methods ) {

					const original = proto[ method ];
					if ( typeof original !== 'function' ) continue;
					const self = this;
					// no rest parameter: about a thousand calls a frame would each allocate an array
					proto[ method ] = function ( this: unknown ) {

						const outside = ! self.inTick && self.outsideDepth > 0;
						if ( ! self.inTick && ! outside ) return original.apply( this, arguments as unknown as unknown[] );
						const outer = inner;
						inner = 0;
						const t0 = performance.now();
						try {

							return original.apply( this, arguments as unknown as unknown[] );

						} finally {

							const ms = performance.now() - t0;
							const nested = inner;
							const own = ms - nested;
							const blocked = own >= ENCODE_BLOCK_MS ? own : 0;
							// the parent subtracts only the time counted somewhere: a call below the threshold
							// stays in its parent's (or the tick's) own time
							inner = outer + nested + blocked;
							if ( blocked > 0 ) {

								if ( outside ) self.outsideWait += blocked;
								else self.waits.encode += blocked;

							}

						}

					};

				}

			}

		}

		override begin(): void {

			super.begin();
			if ( ! this.queueWrapped ) this.wrapQueue();
			this.context = null;
			this.frame = {
				id: this.nodeFrame.frameId, time: 0, now: performance.now(), js: 0, jsWall: 0, wait: 0, waits: { submit: 0, write: 0, texture: 0, encode: 0 }, outside: 0, outsideWait: 0, gap: NaN, passes: [], pending: 0,
				pipelines: 0, pipelineMs: 0, calls: 0, triangles: 0, finishedAt: 0, readbacks: 0, scale: this.renderScale,
			};
			this.pipelineCount = 0;
			this.pipelineMs = 0;

		}

		override finish(): void {

			super.finish();
			const frame = this.frame;
			this.frame = null;
			this.context = null;
			if ( ! frame ) return;
			frame.pipelines = this.pipelineCount;
			frame.pipelineMs = this.pipelineMs;
			frame.finishedAt = performance.now();
			this.open.push( frame );
			if ( this.timestamps ) this.resolve();
			frame.readbacks = this.readbacks;
			this.settle();

		}

		override beginRender( uid: string, scene: THREE.Scene, _camera: THREE.Camera, renderTarget: THREE.RenderTarget ): void {

			this.beginPass( uid, scene, renderTarget as unknown as Named | null );

		}

		override finishRender(): void {

			this.context = this.context?.parent ?? null;
			this.endOutside();

		}

		override beginCompute( uid: string, computeNodes: THREE.ComputeNode ): void {

			// (a list of kernels dispatched in one pass arrives as an array)
			this.beginPass( uid, computeNodes as THREE.ComputeNode | THREE.ComputeNode[], null );

		}

		override finishCompute(): void {

			this.context = this.context?.parent ?? null;
			this.endOutside();

		}

		private endOutside(): void {

			if ( this.inTick || this.outsideDepth === 0 ) return;
			if ( -- this.outsideDepth === 0 ) {

				// the call's wall time less what it spent blocked in the queue calls
				if ( this.frame ) {

					this.frame.outside += Math.max( performance.now() - this.outsideStart - this.outsideWait, 0 );
					this.frame.outsideWait += this.outsideWait;

				}
				this.outsideWait = 0;

			}

		}

		private beginPass( uid: string, drawn: object | object[], target: Named | null ): void {

			if ( ! this.inTick && this.outsideDepth ++ === 0 ) {

				this.outsideStart = performance.now();
				this.outsideWait = 0;

			}
			const parent = this.context;
			const pass: Pass = { uid, group: this.labelFor( drawn, target, parent ), name: this.describe( drawn, target ), parent, start: 0, end: 0, done: ! this.timestamps };
			this.context = pass;
			const frame = this.frame;
			if ( ! frame ) return; // before the loop starts (startup passes)
			frame.passes.push( pass );
			if ( this.timestamps ) {

				frame.pending ++;
				this.byUid.set( uid, pass );

			}

		}

		// ------------------------------------------------------------------ timestamps

		private resolve(): void {

			const renderer = this.getRenderer();
			if ( ! renderer || ! tracksTimestamps( renderer ) ) return;
			const device = gpuDevice<GPUDevice>( renderer );
			// three resolves on its own only when the pool overflows; route that through here too
			for ( const pool of timestampPools( renderer, 'profiled', ( pool ) => this.resolvePool( device, pool ) ) ) this.resolvePool( device, pool );

		}

		private resolvePool( device: GPUDevice, pool: TimestampPool ): void {

			if ( pendingTimestamps( pool ) === 0 ) return;
			const { count, offsets } = takeTimestamps( pool );
			const bytes = count * 8;
			let slot = this.buffers.find( ( b ) => ! b.busy && b.buffer.size >= bytes );
			if ( ! slot && this.buffers.length < RING ) {

				slot = { buffer: device.createBuffer( { label: 'profiler.timestamps', size: readbackBytes( count ), usage: READBACK_USAGE } ), busy: false };
				this.buffers.push( slot );

			}
			if ( ! slot ) {

				// every read-back buffer is still in flight: these passes go unmeasured
				this.lost += offsets.size;
				for ( const uid of offsets.keys() ) this.deliver( uid, null, null );
				return;

			}
			slot.busy = true;
			const serial = ++ this.readbacks;
			this.inFlight.add( serial );
			const encoder = device.createCommandEncoder( { label: 'profiler.resolve' } );
			resolveTimestamps( encoder, pool, count, slot.buffer );
			device.queue.submit( [ encoder.finish() ] );
			const buffer = slot.buffer;
			const submitted = performance.now();
			buffer.mapAsync( MAP_READ, 0, bytes ).then( () => {

				this.latency.push( performance.now() - submitted );
				if ( this.latency.length > 600 ) this.latency.shift();

				const times = new BigUint64Array( buffer.getMappedRange( 0, bytes ).slice( 0 ) );
				buffer.unmap();
				slot!.busy = false;
				this.inFlight.delete( serial );
				for ( const [ uid, base ] of offsets ) this.deliver( uid, times[ base ], times[ base + 1 ] );
				this.settle();

			}, () => {

				slot!.busy = false;
				this.inFlight.delete( serial );
				this.failed.map += offsets.size;
				for ( const uid of offsets.keys() ) this.deliver( uid, null, null );
				this.settle();

			} );

		}

		private deliver( uid: string, start: bigint | null, end: bigint | null ): void {

			const pass = this.byUid.get( uid );
			if ( ! pass ) return;
			this.byUid.delete( uid );
			pass.done = true;
			if ( start !== null && end !== null && end >= start && start > 0n ) {

				if ( this.origin === null ) this.origin = start;
				pass.start = Number( start - this.origin ) / 1e6;
				pass.end = Number( end - this.origin ) / 1e6;

			} else {

				if ( start === 0n ) this.failed.zero ++;
				else if ( start !== null && end !== null ) this.failed.order ++;
				pass.start = pass.end = NaN;

			}

		}

		/** Moves frames to `done`, in order, once their passes all reported or every read-back
		 * submitted up to their end has returned (a pass still without timestamps then never had a
		 * query written). No wall-clock cutoff: a read-back returns when the GPU reaches it, which in
		 * an uncapped run is as many seconds as the GPU runs behind. */
		private settle(): void {

			let oldest = Infinity;
			for ( const serial of this.inFlight ) oldest = Math.min( oldest, serial );
			while ( this.open.length ) {

				const frame = this.open[ 0 ];
				const pending = frame.passes.filter( ( p ) => ! p.done );
				if ( pending.length && oldest <= frame.readbacks ) break;
				this.open.shift();
				for ( const pass of pending ) this.byUid.delete( pass.uid );
				this.failed.unread += pending.length;
				const record = this.record( frame );
				this.done.push( record );
				this.onRecord?.( record );
				if ( this.done.length > 20000 ) this.done.splice( 0, this.done.length - 20000 );

			}

		}

		private record( frame: Frame ): FrameRecord {

			// (passes that never reported are not done, and counted here once)
			let missing = 0;
			const groups: Record<string, number> = {};
			const detail: Record<string, number> = {};
			const charges = new Map<Pass, number>();
			const timed: Pass[] = [];
			let gpu = 0, first = Infinity;
			for ( const pass of frame.passes ) {

				if ( ! pass.done || ! Number.isFinite( pass.start ) ) {

					if ( this.timestamps ) missing ++;
					continue;

				}
				if ( ! this.timestamps ) continue;
				timed.push( pass );
				first = Math.min( first, pass.start );

			}
			// each instant after the floor (the end of the last frame's work still running, unless the
			// stamps jumped) to the first-issued pass covering it (`timed` is in issue order)
			let last = - Infinity;
			for ( const pass of timed ) last = Math.max( last, pass.end );
			const floor = this.lastEnd < last ? this.lastEnd : - Infinity;
			const points = new Set<number>();
			for ( const pass of timed ) {

				charges.set( pass, 0 );
				if ( pass.end > floor ) points.add( Math.max( pass.start, floor ) ).add( pass.end );

			}
			const cuts = [ ...points ].sort( ( a, b ) => a - b );
			for ( let i = 0; i + 1 < cuts.length; i ++ ) {

				const a = cuts[ i ], b = cuts[ i + 1 ];
				const owner = timed.find( ( p ) => p.start <= a && p.end >= b );
				if ( owner ) charges.set( owner, charges.get( owner )! + b - a );

			}
			let pixel = 0;
			for ( const pass of timed ) {

				const ms = charges.get( pass )!;
				gpu += ms;
				if ( this.pixelPass?.( pass.uid ) ) pixel += ms;
				groups[ pass.group ] = ( groups[ pass.group ] ?? 0 ) + ms;
				if ( this.detail ) {

					const key = `${ pass.group } | ${ pass.name }`;
					detail[ key ] = ( detail[ key ] ?? 0 ) + ms;

				}

			}
			const end = timed.length ? Math.max( last, floor ) : NaN;
			// the GPU's interval only between two frames that both reported every pass
			const complete = timed.length > 0 && missing === 0;
			const record: FrameRecord = {
				id: frame.id, t: frame.now, interval: frame.time - this.lastTime, nowInterval: frame.now - this.lastNow,
				js: frame.js, jsWall: frame.jsWall, submitWait: frame.wait, waits: frame.waits, outsideJs: frame.outside, outsideWait: frame.outsideWait, gap: frame.gap,
				gpu, pixel: this.pixelPass ? pixel : NaN, gpuInterval: complete && this.lastComplete ? end - this.lastEnd : NaN,
				span: timed.length ? end - first : 0, passes: frame.passes.length, missing: this.timestamps ? missing : 0,
				pipelines: frame.pipelines, pipelineMs: frame.pipelineMs, calls: frame.calls, triangles: frame.triangles, scale: frame.scale, groups,
			};
			if ( this.detail ) record.detail = detail;
			if ( this.raw ) record.raw = frame.passes.map( ( p ) => [ p.group, p.name, p.start - first, p.end - first, charges.get( p ) ?? NaN ] );
			if ( timed.length ) this.lastEnd = end;
			this.lastComplete = complete;
			this.lastTime = frame.time;
			this.lastNow = frame.now;
			return record;

		}

		// ------------------------------------------------------------------ loop and startup

		/** Times the synchronous tick (its pipeline creation included, its submit waits apart) and
		 * stamps the frame. */
		wrapLoop( tick: ( time: number ) => void ): ( time: number ) => void {

			return ( time: number ) => {

				const frame = this.frame;
				const waits = this.waits = { submit: 0, write: 0, texture: 0, encode: 0 };
				this.inTick = true;
				const t0 = performance.now();
				try {

					tick( time );

				} finally {

					this.inTick = false;

				}
				const t1 = performance.now();
				const wall = t1 - t0;
				const gap = t0 - this.lastTickEnd;
				this.lastTickEnd = t1;
				const wait = waits.submit + waits.write + waits.texture + waits.encode;
				const js = Math.max( wall - wait, 0 );
				if ( frame ) {

					frame.time = time;
					frame.now = t0;
					frame.js = js;
					frame.jsWall = wall;
					frame.wait = wait;
					frame.waits = waits;
					frame.gap = gap;
					const info = ( this.getRenderer() as THREE.Renderer | null )?.info;
					frame.calls = info?.render.drawCalls ?? 0;
					frame.triangles = info?.render.triangles ?? 0;

				}
				const frames = this.startup.frames;
				if ( frames.length < 30 ) {

					const entry: ( typeof frames )[ number ] = { at: t0, interval: frames.length ? t0 - this.startupLast : 0, js, wait, pipelines: this.pipelineCount, pipelineMs: this.pipelineMs };
					frames.push( entry );
					if ( frames.length === 1 ) this.startup.firstFrame = { at: t0, js, wait, pipelines: this.pipelineCount, pipelineMs: this.pipelineMs };
					this.startupLast = t0;
					const renderer = this.getRenderer();
					const queue = renderer ? gpuDevice<{ queue: { onSubmittedWorkDone(): Promise<void> } } | null>( renderer )?.queue : undefined;
					queue?.onSubmittedWorkDone().then( () => {

						entry.done = performance.now();
						if ( frames[ 0 ] === entry && this.startup.firstFrame ) this.startup.firstFrame.done = entry.done;

					} ).catch( () => {} );

				}

			};

		}

		private startupLast = 0;

		/** Milliseconds since navigation start for a named point of the startup. */
		mark( name: string ): void {

			this.marks[ name ] = performance.now();

		}

		// ------------------------------------------------------------------ tool API

		readonly api = {
			/** Drops the frames measured so far (a warm-up's). */
			reset: () => {

				this.settle();
				this.done.length = 0;

			},
			/** The frames finished since the last take, oldest first. */
			take: (): FrameRecord[] => {

				this.settle();
				return this.done.splice( 0, this.done.length );

			},
			/** Whether frames carry each pass's detail (group | what > target) besides the groups. */
			setDetail: ( on: boolean ) => { this.detail = on; },
			/** Whether frames carry every pass's GPU start and end (ms from the frame's first start). */
			setRaw: ( on: boolean ) => { this.raw = on; },
			/** Frames still waiting for their timestamps (started before `before`, if given). */
			pending: ( before = Infinity ) => this.open.filter( ( f ) => f.now < before ).length,
			info: () => ( {
				timestamps: this.timestamps && Boolean( this.getRenderer() && tracksTimestamps( this.getRenderer() ) ),
				lostPasses: this.lost, failedPasses: this.failed, readbackMs: [ ...this.latency ].sort( ( a, b ) => a - b ).filter( ( _, i, a ) => i === 0 || i === a.length >> 1 || i === a.length - 1 ), marks: this.marks, startup: this.startup,
			} ),
		};

	}

	return new FrameProfiler( options );

}

export type FrameProfiler = ReturnType<typeof createFrameProfiler>;

/** The renderer's passes by subsystem. `handles` are the renderer's parts, named as in
 * RenderStages.ts `RenderParts`, as the app hands them in (app/main.ts); an app labels its own
 * parts with `profiler.scope` beside this call. */
export function labelPasses( profiler: FrameProfiler, handles: Record<string, unknown> ): void {

	const h = handles as Record<string, Record<string, unknown>>;
	// the frame graph labels its passes itself (`graphPass`), the bloom's levels with them
	// FSR 3's kernels (aa=fsr3), each its own group: FSR3.<kernel> -> 'FSR 3 <kernel>'
	profiler.pattern( /^FSR3\./, ( name ) => `FSR 3 ${ name.slice( 5 ) }` );
	profiler.pattern( /^FSR/, 'output & FSR' ); // three's FSR1 passes inside its chain (post=three-fsr)
	// the sky's per-frame compute and its bakes
	profiler.pattern( /^Clouds\.View/, 'clouds view march' );
	profiler.pattern( /^Clouds\.Panorama/, 'clouds panorama' );
	profiler.pattern( /^Clouds\.Shadow/, 'cloud shadow map' );
	profiler.pattern( /^PMREM/, 'sky bake' );
	profiler.pattern( /^Rain\./, 'rain drops' );
	profiler.pattern( /^(StormSea|Spray)\./, 'storm sea' );
	const sky = h.sky;
	profiler.scope( sky, 'track', 'atmosphere LUT' );
	profiler.scope( sky, 'advance', 'sky env blend' );
	profiler.scope( sky, 'updateClouds', 'clouds other' );
	profiler.scope( sky, 'update', 'sky bake', { async: true } );
	profiler.scope( sky.lightning, 'update', 'lightning' );
	profiler.scope( h.rain, 'update', 'rain drops' );
	profiler.scope( h.rain, 'bakeCollision', 'rain collision (startup)' );
	profiler.scope( h.sunShadow, 'update', 'sun shadow map' );
	profiler.scope( h.refractedSun, 'update', 'refracted sun map' );
	profiler.scope( h.fft, 'update', 'ocean FFT' );
	profiler.scope( h.stormSea, 'update', 'storm sea' );
	profiler.scope( h.shoreWaves, 'update', 'shore waves' );
	profiler.scope( h.groundBounce, 'update', 'ground bounce map' );

}
