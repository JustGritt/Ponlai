import type { OpsMode, OpsTable, PassOps } from './GraphCompiler.ts';
import { settings, type SettingsQuery } from '../../Settings.ts';
import { gpuDevice, wrapBeginRender } from '../../three/Backend.ts';

// The frame graph's attachment ops: stores discarded and resolves dropped where unread.
//
// The frame graph's attachment ops applied to the render passes three records: a store discarded
// where nothing later reads the attachment, a multisample resolve dropped where nothing samples it
// before the next draw into the same samples. The table is derived from the passes that run in the
// frame (GraphCompiler's enabled mask), so it follows the weather: under rain the water stores its
// samples for the rain to load and leaves the resolve to it, and the rain discards the samples and
// the depth once resolved; without rain the water itself discards its multisampled colour once
// resolved, and stores the depth the surface resolve then reads. Its legacy table is what Pipeline
// did by hand before the graph ran its passes: the scene's multisampled colour and depth discarded
// in the rain pass only, the water's stores and resolve kept.
//
// A store is discarded only where no pass DECLARES a later read, so every texture a pass binds must
// be declared (`read`, `readOptional`): an undeclared read of a discarded attachment shows garbage,
// and one of a resolve the water left to the rain shows the scene without the sea.
//
// three r186 offers a discard only together with a clear (`storeMultisampled*Buffer = false`) and
// always resolves a multisampled target, so the descriptor is rewritten as it reaches WebGPU:
// the backend's `beginRender` (wrapped through three/Backend.ts `wrapBeginRender`) notes the render
// target of the context it begins, and the command encoder it then creates gets a
// `beginRenderPass` of its own that hands the rewritten descriptor on.
// This relies on r186's WebGPUBackend beginning every render pass on a command encoder of its own,
// created inside `beginRender` right before the pass (WebGPUBackend.beginRender); re-check on a
// three upgrade. r186 does begin a second pass on the same encoder in two cases: a viewport
// texture node copying the framebuffer mid-pass (copyFramebufferToTexture) and layered
// array-camera targets (finishRender). A discard on the first of those would drop what the copy or
// the next layer needs, and only the last can be told apart once it comes, so only an encoder's
// first pass is rewritten: a re-begin is passed on as three made it and counted (`rebegins`, an
// error in dev). The graph's passes use neither feature.
//
// It composes with prototype patches (a wrapper of GPUCommandEncoder.prototype.beginRenderPass):
// the device's `createCommandEncoder` and each encoder's `beginRenderPass` are wrapped on the
// INSTANCE and call through to whatever the prototype method is at call time, so a prototype patch
// installed before or after this one runs under it and sees the rewritten descriptor. An instance
// wrapper of `createCommandEncoder` installed before it is called through as well. One installed
// AFTER it that returns encoders of its own without calling the earlier wrapper bypasses it
// silently. The frame's submit batching (`SubmitBatch.ts`) patches the encoder prototypes and makes
// its own staging encoders from the native method, so it composes in either order. A command
// recorder (tried for the frame pacing as CommandBatch, not shipped: it hands three a recording
// encoder and replays it later through the device method it saved) is the case that bypasses it,
// so such a wrapper must be installed before AttachmentOps: AttachmentOps then wraps the recording
// encoder's `beginRenderPass`, and the recording keeps the rewritten descriptor. A wrapper
// installed later is flagged (`overwrapped`), and the graph's dev check compares the rewrites that
// reached the encoder with the table's (frames that differ counted in `opsMismatchFrames`,
// the diagnostics' `graph`). Re-check all of this on a three upgrade. The copied descriptor leaves three's cached one as it was.
//
// It acts only while the graph executes (`begin` / `end`) and only on the executing pass's own
// target: a pass that renders other targets inside it (a shadow map, a bloom chain) is untouched.
// The table comes from the URL option `fgOps`: derived (the default), legacy (the graph's ops before
// they were derived), store (every store and resolve kept, as three records them) or off (not
// applied at all). Without `fgOps`, `post=store-scene` (and `post=legacy`, which includes it)
// keeps every store: the A/B without the discards.

export type AttachmentOpsMode = OpsMode | 'off';

const MODES: readonly AttachmentOpsMode[] = [ 'derived', 'legacy', 'store', 'off' ];

/** `fgOps` from the app's options or `params` (derived, legacy, store or off); without it `store`
 * under `post=store-scene` or `post=legacy`, else `derived`. */
export function opsModeFromURL( params: SettingsQuery | null = settings ): AttachmentOpsMode {

	if ( params === null ) return 'derived';
	const value = params.get( 'fgOps' ) as AttachmentOpsMode | null;
	if ( value !== null && MODES.includes( value ) ) return value;
	const post = ( params.get( 'post' ) ?? '' ).split( ',' );
	return post.includes( 'store-scene' ) || post.includes( 'legacy' ) ? 'store' : 'derived';

}

type Attachment = { resolveTarget?: unknown; storeOp?: string } | null;
type Descriptor = { colorAttachments: Iterable<Attachment> & ArrayLike<Attachment>; depthStencilAttachment?: { depthStoreOp?: string } | null };
type Encoder = { beginRenderPass( descriptor: Descriptor ): unknown };
type Device = { createCommandEncoder( descriptor?: unknown ): Encoder };
type RenderContext = { renderTarget?: unknown; textures?: ArrayLike<unknown> | null };
type Callable = ( this: unknown, ...args: unknown[] ) => unknown;

// the method to call through: an instance's own wrapper installed before ours, else the
// prototype's at call time (so later prototype patches are seen)
function through( host: object, own: Callable | null, method: string ): Callable {

	return own ?? ( Object.getPrototypeOf( host ) as Record<string, Callable> )[ method ];

}

export class AttachmentOps {

	readonly mode: AttachmentOpsMode;
	/** Descriptors rewritten since `begin` (the graph's `opsApplied`). */
	applied = 0;
	/** Descriptors whose attachment count did not match the pass's ops (dev check). */
	mismatches = 0;
	/** Second render passes begun on an encoder already rewritten (passed on unchanged). */
	rebegins = 0;
	/** The device's createCommandEncoder was replaced after install (by a wrapper that may not
	 * call through: see the note at the top). */
	overwrapped = false;
	private readonly renderer: unknown;
	private readonly dev: boolean;
	// the device whose createCommandEncoder `install` wrapped
	private device: Device | null = null;
	private active = false;
	private table: OpsTable | null = null;
	// the executing pass's own target and its ops
	private target: unknown = undefined;
	private passOps: PassOps | null = null;
	// the render target of the context `beginRender` is beginning (undefined: none)
	private noted: unknown = undefined;
	private notedCount = - 1;
	private restore: ( () => void ) | null = null;
	// one per install: wrappers of an earlier install that a later wrapper kept (see uninstall)
	// only pass calls on
	private generation = 0;
	private createWrapper: unknown = null;

	constructor( renderer: unknown, options: { mode?: AttachmentOpsMode; dev?: boolean } = {} ) {

		this.renderer = renderer;
		this.mode = options.mode ?? opsModeFromURL();
		this.dev = options.dev === true;
		// (dev: a misspelt fgOps falls back to the default, which would make an A/B compare the
		// default with itself)
		const asked = this.dev && options.mode === undefined ? settings.get( 'fgOps' ) : null;
		if ( asked !== null && asked !== this.mode ) console.warn( `fgOps=${ asked } is not one of ${ MODES.join( ', ' ) }: the ops are ${ this.mode }` );

	}

	get installed(): boolean {

		return this.restore !== null;

	}

	/** Wraps the renderer's backend's `beginRender` and the device's `createCommandEncoder` (instance
	 * properties; `uninstall` restores what was there). Install after any wrapper of
	 * createCommandEncoder that does not call through (a command recorder). */
	install(): void {

		if ( this.restore ) return;
		this.overwrapped = false;
		const generation = ++ this.generation;
		const device = this.device = gpuDevice<Device>( this.renderer );
		const ownCreate = Object.prototype.hasOwnProperty.call( device, 'createCommandEncoder' ) ? device.createCommandEncoder as unknown as Callable : null;
		const self = this;
		const restoreBegin = wrapBeginRender<RenderContext>( this.renderer, ( renderContext, begin ) => {

			if ( self.generation !== generation || ! self.restore ) return begin();
			const outer = self.noted, outerCount = self.notedCount;
			self.noted = renderContext?.renderTarget ?? null;
			self.notedCount = renderContext?.textures ? renderContext.textures.length : 1;
			try {

				return begin();

			} finally {

				self.noted = outer;
				self.notedCount = outerCount;

			}

		} );
		const createWrapper = function ( this: unknown, descriptor?: unknown ) {

			const encoder = through( device, ownCreate, 'createCommandEncoder' ).call( device, descriptor ) as Encoder;
			if ( self.generation !== generation || ! self.restore ) return encoder;
			const ops = self.opsForNoted();
			if ( ops ) self.wrapEncoder( encoder, ops, self.notedCount );
			return encoder;

		};
		device.createCommandEncoder = createWrapper;
		this.createWrapper = createWrapper;
		this.restore = () => {

			// put back what was there only where ours is still the property: a wrapper installed
			// later holds a reference to ours, so ours stays there, passing every call on (its
			// generation is over once `restore` is cleared)
			restoreBegin();
			if ( device.createCommandEncoder === createWrapper ) {

				if ( ownCreate ) device.createCommandEncoder = ownCreate as unknown as Device[ 'createCommandEncoder' ];
				else delete ( device as Partial<Device> ).createCommandEncoder;

			}

		};

	}

	uninstall(): void {

		this.restore?.();
		this.restore = null;
		this.createWrapper = null;

	}

	/** The frame's table (the graph's enabled mask); the ops act until `end`. */
	begin( table: OpsTable | null ): void {

		this.active = this.mode !== 'off' && table !== null;
		this.table = table;
		this.applied = 0;
		if ( this.restore && ! this.overwrapped && this.device!.createCommandEncoder !== this.createWrapper ) {

			this.overwrapped = true;
			if ( this.dev ) console.warn( 'AttachmentOps: the device\'s createCommandEncoder was wrapped after AttachmentOps; a wrapper that does not call through bypasses it (install AttachmentOps last)' );

		}

	}

	end(): void {

		this.active = false;
		this.table = null;
		this.leavePass();

	}

	/** The executing pass: its own render target (null: the canvas) and its ops. */
	enterPass( target: unknown, ops: PassOps | null ): void {

		this.target = target;
		this.passOps = ops;

	}

	leavePass(): void {

		this.target = undefined;
		this.passOps = null;

	}

	private opsForNoted(): PassOps | null {

		if ( ! this.active || this.noted === undefined || this.target === undefined ) return null;
		if ( this.noted !== this.target ) return null;
		const ops = this.passOps;
		return ops && ops.rewrite ? ops : null;

	}

	private wrapEncoder( encoder: Encoder, ops: PassOps, count: number ): void {

		const own = Object.prototype.hasOwnProperty.call( encoder, 'beginRenderPass' ) ? encoder.beginRenderPass as unknown as Callable : null;
		const self = this;
		let begun = false;
		encoder.beginRenderPass = function ( this: unknown, descriptor: Descriptor ) {

			let out = descriptor;
			if ( ! begun ) out = self.rewrite( descriptor, ops, count );
			else {

				// a pass re-begun on its encoder (see the note at the top): left as three made it
				self.rebegins ++;
				if ( self.dev ) console.error( `AttachmentOps: '${ ops.pass }' began a second render pass on its encoder; its first pass was already rewritten` );

			}
			begun = true;
			return through( encoder, own, 'beginRenderPass' ).call( encoder, out );

		};

	}

	private rewrite( descriptor: Descriptor, ops: PassOps, count: number ): Descriptor {

		const colors = Array.from( descriptor.colorAttachments );
		if ( colors.length !== ops.colors.length || ( count >= 0 && count !== colors.length ) ) {

			// the pass's target is not the one the ops were compiled for: leave it as three made it
			this.mismatches ++;
			if ( this.dev ) console.error( `AttachmentOps: '${ ops.pass }' has ${ colors.length } colour attachments, its ops ${ ops.colors.length }` );
			return descriptor;

		}
		this.applied ++;
		const depth = descriptor.depthStencilAttachment;
		return {
			...descriptor,
			colorAttachments: colors.map( ( a, i ) => {

				const o = ops.colors[ i ];
				if ( ! a ) return a;
				let out = a;
				if ( o.store === 'discard' ) out = { ...out, storeOp: 'discard' };
				if ( o.resolve === false && out.resolveTarget ) out = { ...out, resolveTarget: undefined };
				return out;

			} ),
			depthStencilAttachment: depth && ops.depth?.store === 'discard' ? { ...depth, depthStoreOp: 'discard' } : depth,
		};

	}

	/** The table in effect (for diagnostics). */
	get current(): OpsTable | null {

		return this.table;

	}

}
