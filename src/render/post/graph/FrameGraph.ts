import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import {
	GraphDecl, GraphError, compileGraph, drawsOver, resolveSize, sizeClassKey, targetKey,
	type CompiledGraph, type Extents, type OpsTable, type PassBuilder, type PassFlags, type PassKind,
	type ResourceDecl, type ResourceDesc, type ResourceHandle, type Samples, type Size, type SizeClass, type TargetInfo,
} from './GraphCompiler.ts';
import { AttachmentOps } from './AttachmentOps.ts';
import { compileBound, compileChain, compileLatent, compileNodes, compileQuad, sceneOf } from './Compile.ts';
import { ownDepthClears } from '../../three/ThreePatches.ts';

// The frame graph's runtime: its textures and targets, and the compiled passes run in order.
// It owns the graph's textures and render targets, their sizes and lifetimes, and executes the
// compiled passes in order with three's own executors (a scene render, a QuadMesh, a RenderPipeline
// and the rtt / bloom passes it updates on the way, three's nodes that render passes of their own
// (a bloom chain), a compute dispatch, or a callback). The graph decides which passes run (`when`),
// binds each pass's target and MRT, clears or loads its attachments, and rewrites the stores and
// resolves three records (AttachmentOps); at the frame's end it swaps ping-pong histories and
// makes the persistent copies. See GraphCompiler.ts for the model.
//
// Textures exist from their declaration (TSL nodes bind them while the passes are built), at 1x1
// until `setExtents` sizes them. Each distinct tuple of attachments a pass writes is one
// THREE.RenderTarget whose textures are the resources' own; a depth resource always lives in a
// render target (a DepthTexture outside one takes the renderer's sample count). `renderTarget`
// makes the target of a tuple ahead of the passes (code that still binds its targets by hand,
// probes and warm-ups); the passes that later draw into the same tuple use that target. Every
// target the graph makes follows its size class through `setExtents`, and targets the graph does
// not own (three's display nodes in a chain, which make their own) follow it through `onResize`.
//
// A texture may sit in several of those render targets (a resource written alone by one pass and
// with a velocity attachment by another). In three r186 `RenderTarget.setSize` disposes the target,
// which destroys every texture in it, and each target's first `updateRenderTarget` after that
// recreates all its textures (its own record of the size is gone): a second target's first use in
// a frame would wipe what the first had drawn into the shared texture. So after any resize the
// targets that share a texture are initialised, in declaration order, before the next frame's
// first pass draws; the recreations then happen while nothing holds content. So are the targets
// no pass draws into (a frame-end copy's destination, a texture written by storage), whose first
// use is not a draw. Any other target initialises itself when a pass first draws into it, as
// before the graph: a pass that runs only now and then (the rain, the upscale) allocates
// nothing until it does. A depth texture's size is set by the graph itself, since three resizes it
// only through the one target that owns it. A recreated target is also new to three's clear of a
// target's depth on its first render, which would wipe a depth a pass loads (the vegetation's
// prepass under the opaque scene): the graph owns those clears (three/ThreePatches.ts `ownDepthClears`).
//
// A pass that declares no depth clears whatever depth its target holds outside the graph (the
// canvas's, and the targets three's nodes draw into inside a pass), as three's defaults do; declare
// the depth to keep it.
//
// The warm-up (`warm`, with Compile.ts) compiles every pass and the latents registered with it at
// startup; the world, vegetation, water, character and shadow-map materials still build in their
// first frames (the warm-up runs before the world exists, the water's pass on an empty scene). It
// compiles each pass under its own binding (target, MRT, depth flags), the passes three's nodes
// render inside a pass (a bloom chain), and the objects and kernels owners register as run only at
// times (`latent( pass, objects, kernels )`: the rain, the bolts, the storm's spray and foam, the game's
// props and islanders). A new
// pass, or a draw that first shows later, joins it the same way; `tools/render/verify-first-use.mjs`
// counts the pipelines any step still creates. A miss is not harmless: a material built twice under
// two keys can be emitted with the first build's uniform names in a layout function (r186), an
// invalid shader. Priming (rendering a pass once at startup) covers only what the walker cannot
// reach (three's FSR1Node, `post=three-fsr`), never leaves a persistent resource written, skips a
// node that may keep state of its own (a history) unless the pass's owner resets it (`invalidate`),
// and is off with `fgPrime=0`, for the check that it changes no frame. A latent registered for a
// pass the graph does not have is reported (`warmStats.failed`) and builds on first use.

export type { SizeClass, ResourceHandle, ResourceDesc, PassKind } from './GraphCompiler.ts';

type Uniform2 = THREE.UniformNode<'vec2', THREE.Vector2>;
// a resize hook and the size it last had (null: not called yet)
type ResizeHook = { size: SizeClass; hook: ( width: number, height: number ) => void; last: Size | null };

/** What a pass's setup returns: its flags and its executor. */
export interface PassDef<F> extends PassFlags {
	when?: ( f: F ) => boolean;
	mrt?: THREE.MRTNode | null;
	// scene: what to draw and from where, with a hook before the draw
	scene?: THREE.Object3D | ( ( f: F ) => THREE.Object3D );
	camera?: THREE.Camera | ( ( f: F ) => THREE.Camera );
	before?( f: F ): void;
	// scene: objects drawn only later (weather, effects), for the warm-up
	latent?(): THREE.Object3D[];
	// after the warm-up primed the pass (see `warm`): forget what of its owner's the priming wrote,
	// so the first real frame redoes it as on a cold load (a map its owner bakes when due)
	invalidate?(): void;
	// quad: a full-screen material
	quad?: THREE.QuadMesh;
	// chain: a render pipeline (with the rtt and bloom passes its output node updates), or three's
	// nodes that render passes of their own (`updateBefore`: a bloom chain), run in order. The
	// pass's clear flags, and any attachment ops derived for it, apply to every pass the nodes draw
	// into its target: a bloom draws into its first target twice (that mip's blur, then the
	// composite), so keep such a pass out of derived store rewrites, or give it a target drawn once.
	pipeline?: THREE.RenderPipeline;
	nodes?: THREE.Node[];
	// compute: the kernels to dispatch (dispatched by `renderer.compute( nodes )`); with a `run` of
	// its own (a 2D dispatch, a kernel picked per frame) every kernel the pass may dispatch, which the
	// warm-up compiles
	compute?: THREE.ComputeNode | THREE.ComputeNode[] | ( ( f: F ) => THREE.ComputeNode | THREE.ComputeNode[] | null );
	// any kind: replaces the default executor (callback: required)
	run?( f: F, graph: FrameGraph<F> ): void;
	// callback: it writes its attachments by copying into their textures, not by drawing (its target
	// is then initialised after each resize, which three does on a target's next draw)
	copy?: boolean;
}

/** The pass executing: its name, profiler label and size class, and whether its cost follows the
 * render's pixel count ('pixel': a pass at the render's size or a share of it, a compute pass that
 * declares such a `size`) or not ('fixed': the canvas, a fixed size, a compute or callback pass
 * without a render-sized attachment or `size`). */
export interface CurrentPass { readonly name: string; readonly label: string; readonly size: SizeClass | null; readonly scales: 'pixel' | 'fixed' }

const passScales = ( size: SizeClass | null ): 'pixel' | 'fixed' => size?.of === 'render' ? 'pixel' : 'fixed';

/** What the owners of passes (the haze, the upscale) declare on the graph while it is built. */
export interface GraphResources {
	color( name: string, desc: ResourceDesc, persistent?: boolean ): ResourceHandle;
	renderTarget( colors: readonly ResourceHandle[], depth?: ResourceHandle | null ): THREE.RenderTarget;
	texture( h: ResourceHandle, parity?: 0 | 1 ): THREE.Texture;
	import( name: string, target: THREE.Texture | THREE.RenderTarget | null, size: SizeClass ): ResourceHandle;
	extent( size: SizeClass ): Uniform2;
	texel( size: SizeClass ): Uniform2;
	onResize( size: SizeClass, hook: ( width: number, height: number ) => void ): void;
}

/** How the owners of draws and kernels that run only at times (the rain, the bolts, the storm)
 * register them for the graph's warm-up (`FrameGraph.latent`). */
export interface LatentRegistry {
	latent( pass: string, objects: readonly THREE.Object3D[], kernels?: readonly THREE.ComputeNode[] ): Promise<void>;
}

export interface FrameGraphHooks<F> {
	onPassBegin?( pass: CurrentPass, f: F ): void;
	onPassEnd?( pass: CurrentPass, f: F ): void;
	// around the contiguous span of jittered passes that run (the TAA jitter)
	jitterBegin?( f: F ): void;
	jitterEnd?( f: F ): void;
}

export interface FrameGraphOptions {
	// the scene's sample count, for `samples: 'scene'`
	sceneSamples: number;
	// the passes whose multisampled stores the legacy ops discard (default: the rain)
	legacyDiscard?: string[];
	// dev checks: target sizes against their classes, attachment counts
	dev?: boolean;
	// the warm-up renders the passes it cannot compile (default true; `fgPrime=0` turns it off)
	prime?: boolean;
}

/** What the warm-up did: the passes it compiled, the latent draws and kernels, the passes it
 * primed (rendered once, since a pass it renders before itself was out of the compiler's reach)
 * and those it left to their first use, and its wall time. */
export interface WarmStats {
	passes: number; latents: number; kernels: number;
	primed: string[]; skipped: string[];
	// compiles that failed (left to their pass's first frame)
	failed: string[];
	// from its start until the compiled pipelines exist, and the priming after (which waits for
	// the real extents)
	ms: number; primeMs: number;
}

// objects a scene pass draws only at times, and their owner's compute kernels (see `latent`)
interface Latent {
	index: number; objects: THREE.Object3D[]; kernels: THREE.ComputeNode[];
	resolve?: () => void;
}

interface Physical {
	// the resource's texture, and the second one of a ping-pong resource
	textures: THREE.Texture[];
	// the render target (per ping-pong parity) that owns each texture
	owners: ( THREE.RenderTarget | null )[];
	// history valid (ping-pong, persistent copies): false after resetHistory (or a resize) until
	// written again
	valid: boolean;
	// a ping-pong resource: which of its two textures holds the newest content. It moves only at
	// the end of a frame that wrote the resource, so after a frame whose writer was skipped both
	// `previous` and the current version still read the newest history, and the next write goes
	// to the other texture instead of overwriting it.
	newest: 0 | 1;
	// imported render target or texture (not sized by the graph)
	imported: boolean;
	// the size it was last given (null: not yet)
	size?: { width: number; height: number } | null;
}

const filterOf = ( f: ResourceDesc[ 'filter' ] ) => f === 'nearest' ? THREE.NearestFilter : THREE.LinearFilter;

// Nodes rendering in `updateBefore` that keep nothing from one frame to the next, so priming
// (rendering a pass once at startup) leaves no stale state in them: FSR1Node (`post=three-fsr`),
// out of the walker's reach, and the blooms it reaches anyway. Any other node (an rtt() inside a
// pass's material, which the graph's own passes no longer use) is not primed.
const STATELESS_NODES = new Set( [ 'FSR1Node', 'BloomNode' ] );

export class FrameGraph<F = unknown> implements LatentRegistry {

	readonly renderer: THREE.WebGPURenderer;
	readonly decl = new GraphDecl();
	readonly hooks: FrameGraphHooks<F> = {};
	/** The pass executing, for the profiler and the dynamic resolution's pass scaling. */
	currentPass: CurrentPass | null = null;
	/** Dev counter: render pass descriptors whose store or resolve was rewritten in the last frame. */
	opsApplied = 0;
	/** The descriptors the last frame's ops table rewrites (`opsApplied` equals it while
	 * AttachmentOps is installed and nothing bypasses it). */
	opsExpected = 0;
	/** Dev counter: targets found at a size other than their class's (an executed pass's, or any
	 * graph target's in `validateSizes`). */
	sizeErrors = 0;
	/** Dev counter: frames whose rewritten descriptors (`opsApplied`) differed from the compiled
	 * table's `rewrites` while AttachmentOps was installed (bypassed by a later wrapper, a target
	 * of another layout, a pass re-begun on its encoder). */
	opsMismatchFrames = 0;
	/** Render targets made after the first frame (the graph compiles once and makes its targets
	 * then, resizing them in place: anything made later, such as a target remade at a resize, is
	 * a rebuild that costs allocations and, for passes keyed by their target, pipelines). */
	rebuilds = 0;
	/** The enabled mask of the last frame executed (null: none yet). */
	lastMask: number | null = null;
	readonly attachmentOps: AttachmentOps;
	private readonly options: Required<FrameGraphOptions>;
	private readonly defs: PassDef<F>[] = [];
	private readonly physical: Physical[] = [];
	private compiled: CompiledGraph | null = null;
	// per target key: its render target(s), one per ping-pong parity
	private readonly targets = new Map<string, THREE.RenderTarget[]>();
	// every render target the graph made, with the size class it follows
	private readonly targetSizes = new Map<THREE.RenderTarget, SizeClass>();
	// the targets of resources no live pass draws into (initialised after every resize)
	private readonly soloTargets = new Set<THREE.RenderTarget>();
	// passes the graph does not own, sized through onResize (`last`: the size each last had)
	private readonly resizeHooks: ResizeHook[] = [];
	private extentsSet = false;
	private extents: Extents = { render: { width: 1, height: 1 }, canvas: { width: 1, height: 1 } };
	// the shared extent uniforms per size class: its pixel size, or one over it (`texel`)
	private readonly extentNodes = new Map<string, { size: SizeClass; node: Uniform2; texel: boolean }>();
	private saved: {
		target: THREE.RenderTarget | null; mrt: THREE.MRTNode | null; depth: boolean; stencil: boolean;
		autoClear: boolean; autoClearColor: boolean; autoClearDepth: boolean; autoClearStencil: boolean;
	} | null = null;
	private executing = false;
	// the targets were resized (or made): initialise them all before the next frame draws
	private needsInit = true;
	// during execute: the resources some running pass writes this frame
	private frameWritten: Set<number> | null = null;
	private readonly drawingBuffer = new THREE.Vector2();
	// what a chain pass of three's nodes hands their updateBefore
	private readonly nodeFrame: THREE.NodeFrame;

	constructor( renderer: THREE.WebGPURenderer, options: FrameGraphOptions ) {

		this.renderer = renderer;
		this.options = { legacyDiscard: [ 'rain' ], dev: false, prime: true, ...options };
		this.attachmentOps = new AttachmentOps( renderer, { dev: this.options.dev } );
		this.nodeFrame = { renderer } as unknown as THREE.NodeFrame;

	}

	// ---- resources

	private makeTexture( desc: ResourceDesc ): THREE.Texture {

		// as THREE.RenderTarget makes its textures (the render target is set when one owns it)
		const texture = new THREE.Texture( { width: 1, height: 1, depth: 1 } as unknown as HTMLImageElement );
		texture.isRenderTargetTexture = true;
		texture.flipY = false;
		texture.generateMipmaps = desc.mips === true;
		texture.minFilter = texture.magFilter = filterOf( desc.filter );
		if ( desc.mips ) texture.minFilter = desc.filter === 'nearest' ? THREE.NearestMipmapNearestFilter : THREE.LinearMipmapLinearFilter;
		if ( desc.format !== undefined ) texture.format = desc.format as THREE.AnyPixelFormat;
		if ( desc.type !== undefined ) texture.type = desc.type as THREE.TextureDataType;
		return texture;

	}

	private add( h: ResourceHandle, textures: THREE.Texture[], imported = false ): ResourceHandle {

		this.physical[ h.resource ] = { textures, owners: textures.map( () => null ), valid: true, imported, newest: 1 };
		return h;

	}

	/** A colour resource; `persistent` keeps its content across frames. */
	color( name: string, desc: ResourceDesc, persistent = false ): ResourceHandle {

		this.mutable( name );
		const h = this.decl.color( name, desc, persistent );
		const texture = this.makeTexture( desc );
		texture.name = name;
		return this.add( h, [ texture ] );

	}

	/** A depth resource (float depth by default), always in a render target of its own size. */
	depth( name: string, desc: { size: SizeClass; samples?: Samples; type?: THREE.TextureDataType } ): ResourceHandle {

		this.mutable( name );
		const h = this.decl.depth( name, { ...desc, type: desc.type ?? THREE.FloatType } );
		const texture = new THREE.DepthTexture( 1, 1 );
		texture.type = desc.type ?? THREE.FloatType;
		texture.name = name;
		return this.add( h, [ texture ] );

	}

	/** A texture or render target made outside the graph, or the canvas (`null`). */
	import( name: string, target: THREE.Texture | THREE.RenderTarget | null, size: SizeClass ): ResourceHandle {

		this.mutable( name );
		const texture = target === null ? null : ( target as THREE.RenderTarget ).isRenderTarget ? ( target as THREE.RenderTarget ).texture : target as THREE.Texture;
		const h = this.decl.import( name, { size, format: texture?.format, type: texture?.type, samples: ( target as THREE.RenderTarget | null )?.isRenderTarget ? Math.max( ( target as THREE.RenderTarget ).samples, 1 ) : 1 }, target === null );
		this.add( h, texture ? [ texture ] : [], true );
		if ( target && ( target as THREE.RenderTarget ).isRenderTarget ) this.physical[ h.resource ].owners = [ target as THREE.RenderTarget ];
		return h;

	}

	/** The resource's texture for TSL: stable for the graph's life. A ping-pong resource has two:
	 * `parity` picks one for the current version and the other for its `previous` handle (default:
	 * `parityOf( h )`). */
	texture( h: ResourceHandle, parity?: 0 | 1 ): THREE.Texture {

		const p = this.physical[ h.resource ];
		if ( ! p || p.textures.length === 0 ) throw new GraphError( `'${ h.name }' has no texture (the canvas)` );
		if ( p.textures.length === 1 ) return p.textures[ 0 ];
		const current = parity ?? this.parityOf( h );
		return p.textures[ h.previous ? 1 - current : current ];

	}

	/** Which texture holds a ping-pong resource's current version (0 for any other resource): a
	 * pass built once per texture runs the build of this parity. During a frame that writes the
	 * resource it is the texture written (the other holds `previous`); during a frame whose writer
	 * is skipped, the newest history, which the current version then is; outside a frame, the
	 * texture the next write goes to. */
	parityOf( h: ResourceHandle ): 0 | 1 {

		const p = this.physical[ h.resource ];
		if ( p.textures.length < 2 ) return 0;
		const writes = this.frameWritten ? this.frameWritten.has( h.resource ) : true;
		return ( writes ? 1 - p.newest : p.newest ) as 0 | 1;

	}

	/** The previous frame's version of a colour resource, kept in a second texture (see
	 * GraphDecl.previous: only for passes built once per texture). */
	previous( h: ResourceHandle ): ResourceHandle {

		this.mutable( h.name );
		if ( this.physical[ h.resource ]?.owners.some( Boolean ) ) throw new GraphError( `previous( '${ h.name }' ): a render target already holds it (declare previous() before renderTarget())` );
		const prev = this.decl.previous( h );
		const p = this.physical[ h.resource ];
		if ( p.textures.length === 1 ) {

			// a texture of its own (Texture.clone would share the first one's Source and image)
			const second = this.makeTexture( this.decl.get( h ).desc );
			second.name = `${ h.name }.previous`;
			p.textures.push( second );
			p.owners.push( null );

		}
		return prev;

	}

	/** Copies `src` into the persistent `dst` at the frame's end (copyTextureToTexture). */
	copyToPersistent( src: ResourceHandle, dst: ResourceHandle ): void {

		this.mutable( dst.name );
		this.decl.copyToPersistent( src, dst );

	}

	/** Marks a history (ping-pong or persistent) invalid until the graph writes it again. */
	resetHistory( h: ResourceHandle ): void {

		this.physical[ h.resource ].valid = false;

	}

	historyValid( h: ResourceHandle ): boolean {

		return this.physical[ h.resource ].valid;

	}

	/** A vec2 uniform holding a size class's pixel size, shared by every caller of that class: the
	 * passes read their sizes from the graph, which sets them with the targets' (`setExtents`). */
	extent( size: SizeClass ): Uniform2 {

		return this.extentNode( size, false );

	}

	/** A vec2 uniform holding one over a size class's pixel size (a texel's size in uv), shared like
	 * `extent`. Computed on the CPU: a shader's own division rounds differently. */
	texel( size: SizeClass ): Uniform2 {

		return this.extentNode( size, true );

	}

	private extentNode( size: SizeClass, texel: boolean ): Uniform2 {

		const key = `${ sizeClassKey( size ) }${ texel ? '/texel' : '' }`;
		let e = this.extentNodes.get( key );
		if ( ! e ) {

			e = { size, node: uniform( new THREE.Vector2() ) as unknown as Uniform2, texel };
			this.setExtentNode( e );
			this.extentNodes.set( key, e );

		}
		return e.node;

	}

	private setExtentNode( e: { size: SizeClass; node: Uniform2; texel: boolean } ): void {

		const s = resolveSize( e.size, this.extents );
		if ( e.texel ) e.node.value.set( 1 / Math.max( s.width, 1 ), 1 / Math.max( s.height, 1 ) );
		else e.node.value.set( s.width, s.height );

	}

	/** Calls `hook` with a size class's pixel size whenever `setExtents` changes it (and once
	 * extents are first set): for render targets the graph does not own, such as three's bloom
	 * chains', so they follow the graph's extents with its rounding. */
	onResize( size: SizeClass, hook: ( width: number, height: number ) => void ): void {

		const entry: ResizeHook = { size, hook, last: null };
		this.resizeHooks.push( entry );
		if ( this.extentsSet ) this.runHook( entry );

	}

	private runHook( entry: ResizeHook ): void {

		const s = resolveSize( entry.size, this.extents );
		if ( entry.last && entry.last.width === s.width && entry.last.height === s.height ) return;
		entry.last = s;
		entry.hook( s.width, s.height );

	}

	/** The render target that holds these attachments: colour resources in attachment order and
	 * a depth resource, of one size class and sample count, none ping-pong or imported. It is the
	 * target the passes drawing into the same attachments use; made now if no pass or earlier call
	 * made it, for code that binds its targets by hand, probes and warm-ups. */
	renderTarget( colors: readonly ResourceHandle[], depth: ResourceHandle | null = null ): THREE.RenderTarget {

		const decls = colors.map( ( h ) => this.decl.get( h ) );
		const depthDecl = depth ? this.decl.get( depth ) : null;
		const all = depthDecl ? [ ...decls, depthDecl ] : decls;
		const what = `renderTarget( ${ all.map( ( r ) => `'${ r.name }'` ).join( ', ' ) } )`;
		if ( decls.length === 0 ) throw new GraphError( `${ what }: no colour attachment (three's render targets carry one)` );
		if ( decls.some( ( r ) => r.kind !== 'color' ) || ( depthDecl && depthDecl.kind !== 'depth' ) ) throw new GraphError( `${ what }: colour resources, then a depth resource` );
		if ( all.some( ( r ) => r.pingPong ) ) throw new GraphError( `${ what }: a ping-pong resource has a target per parity` );
		const key = targetKey( decls.map( ( r ) => r.name ), depthDecl ? depthDecl.name : null );
		const made = this.targets.get( key );
		if ( made ) return made[ 0 ];
		const size = all[ 0 ].desc.size, samples = this.samplesOfDecl( all[ 0 ] );
		for ( const r of all ) {

			if ( sizeClassKey( r.desc.size ) !== sizeClassKey( size ) ) throw new GraphError( `${ what }: attachments of different size classes` );
			if ( this.samplesOfDecl( r ) !== samples ) throw new GraphError( `${ what }: attachments of different sample counts` );

		}
		const rt = this.makeTarget( decls.map( ( r ) => this.physical[ r.id ].textures[ 0 ] ), depthDecl ? this.physical[ depthDecl.id ].textures[ 0 ] as THREE.DepthTexture : null, samples );
		this.targets.set( key, [ rt ] );
		this.targetSizes.set( rt, size );
		for ( const r of all ) this.physical[ r.id ].owners[ 0 ] ??= rt;
		if ( this.sizeTarget( rt, size ) ) this.needsInit = true;
		return rt;

	}

	/** The render target that holds a resource (the first that took it: a pass's, or one made by
	 * `renderTarget`); the current or previous parity's for a ping-pong resource. */
	target( h: ResourceHandle ): THREE.RenderTarget {

		const p = this.physical[ h.resource ];
		const parity = p && p.textures.length > 1 ? ( h.previous ? 1 - this.parityOf( h ) : this.parityOf( h ) ) : 0;
		const rt = p?.owners[ parity ];
		if ( ! rt ) throw new GraphError( `'${ h.name }' has no render target yet (compile the graph, or make one with renderTarget())` );
		return rt;

	}

	private samplesOfDecl( r: ResourceDecl ): number {

		const s = r.desc.samples ?? 1;
		return s === 'scene' ? this.options.sceneSamples : s;

	}

	// ---- passes

	/** Declares a pass (in execution order); `setup` declares its reads and writes and returns
	 * its flags and executor. */
	addPass( name: string, kind: PassKind, setup: ( b: PassBuilder ) => PassDef<F> ): PassDef<F> {

		this.mutable( name );
		const def = this.decl.addPass( name, kind, setup );
		const ok = kind === 'scene' ? def.scene && def.camera : kind === 'quad' ? def.quad : kind === 'chain' ? def.pipeline ?? def.nodes?.length
			: kind === 'compute' ? def.compute : def.run;
		if ( ! ok && ! def.run ) throw new GraphError( `pass '${ name }' (${ kind }) has no executor` );
		this.defs.push( def );
		return def;

	}

	private mutable( what: string ): void {

		if ( this.compiled ) throw new GraphError( `'${ what }': the graph is compiled (declare everything before compile())` );

	}

	/** Validates, culls and derives the ops; makes the render targets. Once. */
	compile(): CompiledGraph {

		if ( this.compiled ) return this.compiled;
		for ( const extend of FrameGraph.beforeCompile ) extend( this as FrameGraph<unknown> );
		const compiled = compileGraph( this.decl, { sceneSamples: this.options.sceneSamples, legacyDiscard: this.options.legacyDiscard } );
		// (a target made ahead by renderTarget() holds the same attachments, size class and samples)
		for ( const t of compiled.targets ) if ( ! t.canvas && ! this.targets.has( t.key ) ) this.targets.set( t.key, this.makeTargets( t ) );
		// A pass that writes its attachments by a copy (`copy`: the temporal modes' opaque copy) never
		// draws into its target, which three would then leave at its old size after a resize (it
		// recreates a target's textures on the target's next use, and a copy is none): such a target
		// is initialised after every resize, like the targets no pass draws into.
		for ( const index of compiled.live ) {

			const t = compiled.targetOf[ index ];
			if ( ! this.defs[ index ].copy || ! t || t.canvas ) continue;
			for ( const rt of this.targets.get( t.key )! ) {

				if ( ! this.targetSizes.has( rt ) ) continue; // imported
				this.soloTargets.add( rt );
				this.needsInit = true;

			}

		}
		// a resource no live pass draws into still gets a render target of its own (a copy's
		// destination, a texture written by storage): its textures need one to be allocated
		// (a depth resource no live pass draws into needs none: nothing copies into depth)
		this.decl.resources.forEach( ( r, id ) => {

			const p = this.physical[ id ];
			if ( p.imported || r.kind === 'depth' ) return;
			p.textures.forEach( ( texture, parity ) => {

				if ( p.owners[ parity ] ) return;
				const solo = this.makeTarget( [ texture ], null, compiled.samplesOf( r ) );
				p.owners[ parity ] = solo;
				this.targets.set( `${ r.name }#solo${ parity }`, [ solo ] );
				this.targetSizes.set( solo, r.desc.size );
				this.soloTargets.add( solo );
				this.needsInit = true;

			} );

		} );
		this.compiled = compiled;
		this.applyExtents();
		return compiled;

	}

	private makeTarget( colors: THREE.Texture[], depth: THREE.DepthTexture | null, samples: number ): THREE.RenderTarget {

		if ( this.lastMask !== null ) this.rebuilds ++;
		const rt = new THREE.RenderTarget( 1, 1, { count: Math.max( colors.length, 1 ), samples: samples > 1 ? samples : 0, depthBuffer: depth !== null } );
		colors.forEach( ( texture, i ) => {

			rt.textures[ i ] = texture;
			if ( ! texture.renderTarget ) ( texture as { renderTarget: THREE.RenderTarget | null } ).renderTarget = rt;

		} );
		if ( depth ) rt.depthTexture = depth;
		return rt;

	}

	private makeTargets( t: TargetInfo ): THREE.RenderTarget[] {

		const parities = t.pingPong ? [ 0, 1 ] : [ 0 ];
		return parities.map( ( parity ) => {

			const colors = t.colors.map( ( id ) => {

				const p = this.physical[ id ];
				return p.textures[ p.textures.length > 1 ? parity : 0 ];

			} );
			const depth = t.depth === null ? null : this.physical[ t.depth ].textures[ 0 ] as THREE.DepthTexture;
			const imported = t.colors.length === 1 && this.physical[ t.colors[ 0 ] ].imported ? this.physical[ t.colors[ 0 ] ].owners[ 0 ] : null;
			const rt = imported ?? this.makeTarget( colors, depth, t.samples );
			if ( ! imported ) this.targetSizes.set( rt, t.size );
			for ( const id of t.depth === null ? t.colors : [ ...t.colors, t.depth ] ) {

				const p = this.physical[ id ];
				const slot = p.textures.length > 1 ? parity : 0;
				p.owners[ slot ] ??= rt;

			}
			return rt;

		} );

	}

	/** The render and canvas sizes (the canvas is never smaller than the render). */
	setExtents( extents: Extents ): void {

		this.extents = {
			render: { width: Math.max( extents.render.width, 1 ), height: Math.max( extents.render.height, 1 ) },
			canvas: { width: Math.max( extents.canvas.width, extents.render.width, 1 ), height: Math.max( extents.canvas.height, extents.render.height, 1 ) },
		};
		this.extentsSet = true;
		this.applyExtents();

	}

	// a target to its class's size at the current extents; true when it changed
	private sizeTarget( rt: THREE.RenderTarget, size: SizeClass ): boolean {

		const s = resolveSize( size, this.extents );
		const width = Math.max( s.width, 1 ), height = Math.max( s.height, 1 );
		if ( rt.width === width && rt.height === height ) return false;
		rt.setSize( width, height );
		return true;

	}

	private applyExtents(): void {

		for ( const e of this.extentNodes.values() ) this.setExtentNode( e );
		if ( this.extentsSet ) for ( const entry of this.resizeHooks ) this.runHook( entry );
		for ( const [ rt, size ] of this.targetSizes ) if ( this.sizeTarget( rt, size ) ) this.needsInit = true;
		// the depth textures' own size (three sets it only through the target that owns one) and the
		// histories a resize emptied
		this.decl.resources.forEach( ( r, id ) => {

			const p = this.physical[ id ];
			if ( p.imported ) return;
			const s = resolveSize( r.desc.size, this.extents );
			const width = Math.max( s.width, 1 ), height = Math.max( s.height, 1 );
			for ( const texture of p.textures ) {

				const image = texture.image as { width: number; height: number };
				if ( r.kind === 'depth' && ( image.width !== width || image.height !== height ) ) {

					image.width = width;
					image.height = height;
					texture.needsUpdate = true;
					this.needsInit = true;

				}

			}
			if ( ( r.persistent || r.pingPong ) && p.size && ( p.size.width !== width || p.size.height !== height ) ) p.valid = false;
			p.size = { width, height };

		} );

	}

	// After a resize: the targets that share a texture with another, and those no pass draws into,
	// have their textures (re)created now, in declaration order, before any pass of the frame
	// draws (see the note at the top). The others initialise themselves on their next draw.
	private initTargets(): void {

		if ( ! this.needsInit ) return;
		this.needsInit = false;
		const attachments = ( rt: THREE.RenderTarget ) => rt.depthTexture ? [ ...rt.textures, rt.depthTexture ] : rt.textures;
		const holders = new Map<THREE.Texture, number>();
		for ( const rt of this.targetSizes.keys() ) for ( const texture of attachments( rt ) ) holders.set( texture, ( holders.get( texture ) ?? 0 ) + 1 );
		for ( const rts of this.targets.values() ) for ( const rt of rts ) {

			if ( ! this.targetSizes.has( rt ) ) continue; // imported
			if ( this.soloTargets.has( rt ) || attachments( rt ).some( ( texture ) => holders.get( texture )! > 1 ) ) this.renderer.initRenderTarget( rt );

		}

	}

	/** Dev check: every render target the graph made (and its depth texture) has its class's size
	 * at the current extents, so nothing resized one behind the graph's back. Counted in
	 * `sizeErrors`, with the first few reported. */
	validateSizes(): number {

		let errors = 0;
		for ( const [ rt, size ] of this.targetSizes ) {

			const s = resolveSize( size, this.extents );
			const width = Math.max( s.width, 1 ), height = Math.max( s.height, 1 );
			const depth = rt.depthTexture?.image as { width: number; height: number } | undefined;
			if ( rt.width === width && rt.height === height && ( ! depth || ( depth.width === width && depth.height === height ) ) ) continue;
			errors ++;
			if ( this.sizeErrors ++ < 8 ) console.error( `FrameGraph: the target of '${ rt.textures.map( ( t ) => t.name ).join( '+' ) }' is ${ rt.width }x${ rt.height }${ depth ? ` (depth ${ depth.width }x${ depth.height })` : '' }, its class ${ sizeClassKey( size ) } is ${ width }x${ height }` );

		}
		return errors;

	}

	/** The render target a pass draws into this frame (null: the canvas, or no attachment). */
	targetOf( index: number ): THREE.RenderTarget | null {

		const t = this.compiled?.targetOf[ index ];
		if ( ! t || t.canvas ) return null;
		const rts = this.targets.get( t.key )!;
		if ( rts.length === 1 ) return rts[ 0 ];
		// a ping-pong target: the parity of its ping-pong colours (written together, so equal)
		const id = t.colors.find( ( c ) => this.decl.resources[ c ].pingPong )!;
		return rts[ this.parityOf( { resource: id, version: 0, name: this.decl.resources[ id ].name } ) ];

	}

	// ---- execution

	/** Binds a pass's target, MRT, depth flags and clears; `unbind` restores the renderer. The
	 * warm-up uses the same binding. */
	bind( index: number, table: OpsTable | null = null ): void {

		const r = this.renderer;
		const decl = this.decl.passes[ index ];
		const def = this.defs[ index ];
		this.saved ??= {
			target: r.getRenderTarget(), mrt: r.getMRT(), depth: r.depth, stencil: r.stencil,
			autoClear: r.autoClear, autoClearColor: r.autoClearColor, autoClearDepth: r.autoClearDepth, autoClearStencil: r.autoClearStencil,
		};
		const t = this.compiled?.targetOf[ index ] ?? null;
		if ( ! t ) {

			// compute and callback passes without attachments keep the renderer's target
			r.setMRT( def.mrt ?? null );
			return;

		}
		const rt = this.targetOf( index );
		r.setRenderTarget( rt );
		r.setMRT( def.mrt ?? null );
		if ( rt ) {

			r.depth = rt.depthBuffer;
			r.stencil = rt.stencilBuffer;
			// (three would clear a depth the pass loads on the target's first render after a resize)
			if ( rt.depthBuffer ) ownDepthClears( r, rt );

		}
		// clear or load, as the compiled ops say (the same in both tables; a pass's colour writes
		// all clear or all load, as the compiler checks)
		const ops = table?.passes.find( ( p ) => p.index === index ) ?? null;
		const colourClear = ops ? ops.colors.some( ( c ) => c.load === 'clear' ) : decl.writes.some( ( w ) => drawsOver( decl, w ) );
		// A pass without a depth resource clears whatever depth its target holds that the graph
		// does not own: the canvas's depth buffer, and that of any target three's nodes draw into
		// inside the pass (they inherit the renderer's flags; a bloom chain's have none), as three's
		// defaults do; with the flag off each of those passes loaded a depth nothing reads. A graph target with depth always
		// holds a depth resource, which a pass that does not declare it keeps (loads).
		const foreignDepth = ! decl.depth && ! rt?.depthBuffer;
		const depthClear = ops?.depth ? ops.depth.load === 'clear' : decl.depth?.clear === true || foreignDepth;
		r.autoClear = colourClear || depthClear;
		r.autoClearColor = colourClear;
		r.autoClearDepth = depthClear;
		r.autoClearStencil = depthClear;

	}

	unbind(): void {

		const s = this.saved;
		if ( ! s ) return;
		const r = this.renderer;
		r.setRenderTarget( s.target );
		r.setMRT( s.mrt );
		r.depth = s.depth;
		r.stencil = s.stencil;
		r.autoClear = s.autoClear;
		r.autoClearColor = s.autoClearColor;
		r.autoClearDepth = s.autoClearDepth;
		r.autoClearStencil = s.autoClearStencil;
		this.saved = null;

	}

	/** The enabled mask: each conditional pass's `when`, evaluated once at the frame's start. */
	maskFor( f: F ): number {

		const compiled = this.compiled!;
		let mask = 0;
		compiled.conditional.forEach( ( index, bit ) => {

			if ( ( this.defs[ index ].when as ( f: F ) => boolean )( f ) ) mask |= 1 << bit;

		} );
		return mask;

	}

	/** Runs the frame: the passes the mask enables, in order, then the frame-end copies and swaps. */
	execute( f: F ): void {

		const compiled = this.compiled ?? this.compile();
		if ( this.executing ) throw new GraphError( 'execute() re-entered' );
		this.executing = true;
		const mask = this.lastMask = this.maskFor( f );
		const ops = this.attachmentOps;
		const table = ops.mode === 'off' ? null : compiled.ops( ops.mode, mask );
		this.opsExpected = table ? table.rewrites : 0;
		const running = compiled.running( mask );
		const span = running.filter( ( i ) => this.decl.passes[ i ].jittered );
		const written = new Set<number>();
		for ( const i of running ) for ( const w of [ ...this.decl.passes[ i ].writes, ...this.decl.passes[ i ].storage ] ) written.add( w.resource );
		const first = span[ 0 ], last = span[ span.length - 1 ];
		if ( this.options.dev ) this.validateSizes();
		this.initTargets();
		this.frameWritten = written;
		ops.begin( table );
		// whether the jitter is on the camera (a pass that throws inside the span must not leave it
		// there: the next frame would take the shifted projection for the camera's own)
		let jittering = false;
		try {

			for ( const index of running ) {

				if ( index === first ) { this.hooks.jitterBegin?.( f ); jittering = true; }
				this.runPass( index, f, table );
				if ( index === last ) { jittering = false; this.hooks.jitterEnd?.( f ); }

			}

		} catch ( error ) {

			this.frameWritten = null;
			throw error;

		} finally {

			if ( jittering ) this.hooks.jitterEnd?.( f );
			this.opsApplied = ops.applied;
			// dev: every rewrite the table expects reached WebGPU (AttachmentOps must be installed
			// after any wrapper of the device's createCommandEncoder that does not call through,
			// such as a command recorder; see AttachmentOps.ts)
			if ( this.options.dev && ops.installed && table && ops.applied !== table.rewrites ) {

				if ( this.opsMismatchFrames ++ < 8 ) console.warn( `FrameGraph: ${ ops.applied } descriptors rewritten, the ops table expects ${ table.rewrites }${ ops.overwrapped ? ' (createCommandEncoder was wrapped after AttachmentOps)' : '' }` );

			}
			ops.end();
			this.currentPass = null;
			this.unbind();
			this.executing = false;

		}
		this.endFrame( written );

	}

	private runPass( index: number, f: F, table: OpsTable | null ): void {

		const decl = this.decl.passes[ index ];
		const def = this.defs[ index ];
		const target = this.compiled!.targetOf[ index ];
		const size = decl.size ?? target?.size ?? null;
		const current: CurrentPass = { name: decl.name, label: decl.label, size, scales: passScales( size ) };
		this.currentPass = current;
		this.bind( index, table );
		const rt = target ? this.targetOf( index ) : undefined;
		if ( this.options.dev && target ) this.checkSize( current, rt ?? null, target.size );
		this.attachmentOps.enterPass( target ? rt ?? null : undefined, table?.passes.find( ( p ) => p.index === index ) ?? null );
		this.hooks.onPassBegin?.( current, f );
		try {

			this.executePass( decl.kind, def, f );

		} finally {

			this.hooks.onPassEnd?.( current, f );
			this.attachmentOps.leavePass();
			this.unbind();
			this.currentPass = null;

		}

	}

	private executePass( kind: PassKind, def: PassDef<F>, f: F ): void {

		const r = this.renderer;
		if ( def.run ) {

			def.run( f, this );
			return;

		}
		switch ( kind ) {

			case 'scene': {

				def.before?.( f );
				const scene = typeof def.scene === 'function' ? def.scene( f ) : def.scene!;
				const camera = typeof def.camera === 'function' ? ( def.camera as ( f: F ) => THREE.Camera )( f ) : def.camera!;
				r.render( scene, camera );
				break;

			}
			case 'quad': def.quad!.render( r ); break;
			case 'chain': {

				if ( def.pipeline ) def.pipeline.render();
				// (three's nodes take the frame of the node system: they read its renderer)
				else for ( const node of def.nodes! ) node.updateBefore( this.nodeFrame );
				break;

			}
			case 'compute': {

				const nodes = typeof def.compute === 'function' ? def.compute( f ) : def.compute!;
				if ( nodes ) r.compute( nodes );
				break;

			}
			default: throw new GraphError( `callback pass without run()` );

		}

	}

	// dev: the target a pass draws into must have its class's size
	private checkSize( pass: CurrentPass, rt: THREE.RenderTarget | null, size: SizeClass ): void {

		const want = resolveSize( size, this.extents );
		let width: number, height: number;
		if ( rt ) {

			width = rt.width;
			height = rt.height;

		} else {

			this.renderer.getDrawingBufferSize( this.drawingBuffer );
			width = this.drawingBuffer.x;
			height = this.drawingBuffer.y;

		}
		if ( width !== Math.max( want.width, 1 ) || height !== Math.max( want.height, 1 ) ) {

			if ( this.sizeErrors ++ < 8 ) console.error( `FrameGraph: '${ pass.name }' draws into ${ width }x${ height }, its class ${ sizeClassKey( size ) } is ${ want.width }x${ want.height }` );

		}

	}

	// the persistent copies, then the ping-pong swaps: only a history some pass wrote this frame
	// swaps (and becomes valid); one whose writer was skipped keeps its parity, so `previous` still
	// reads the newest history and the next write does not overwrite it
	private endFrame( written: Set<number> ): void {

		const r = this.renderer;
		try {

			for ( const { src, dst } of this.decl.copies ) {

				r.copyTextureToTexture( this.texture( src ), this.texture( dst ) );
				this.physical[ dst.resource ].valid = true;

			}

		} finally {

			this.decl.resources.forEach( ( res, id ) => {

				if ( ! res.pingPong || ! written.has( id ) ) return;
				const p = this.physical[ id ];
				p.newest = p.newest === 0 ? 1 : 0;
				p.valid = true;

			} );
			this.frameWritten = null;

		}

	}

	/** The ops table of a mask (introspection and the verifier). */
	opsTable( mask: number ): OpsTable | null {

		const mode = this.attachmentOps.mode;
		return this.compiled && mode !== 'off' ? this.compiled.ops( mode, mask ) : null;

	}

	// ---- warm-up: every pass's pipelines before its first frame

	/** What the warm-up did (null until it starts; its times once it has finished). */
	warmStats: WarmStats | null = null;
	private warming: Promise<void> | null = null;
	// the frame the scene passes compiled for (their scene and camera), and whether the warm-up
	// started: latents registered after it compile at once
	private warmFrame: F | undefined = undefined;
	private warmStarted = false;
	private readonly latents: Latent[] = [];
	// failures reported before the warm-up started (a latent for a pass the graph does not have)
	private readonly earlyFailures: string[] = [];

	/** Registers objects the scene pass `pass` draws only at times (rain, a bolt, a storm's spray),
	 * and the compute kernels their owner dispatches with them, for the warm-up to compile before
	 * their first use, shown or not. An object compiles as part of the scene it is in (the pass's
	 * own when it is in none). Registered once the warm-up has started, they compile at once.
	 * Resolves when their pipelines exist (or their compile failed, or the graph has no such scene
	 * pass: see `warmStats.failed` and the warning). */
	latent( pass: string, objects: readonly THREE.Object3D[], kernels: readonly THREE.ComputeNode[] = [] ): Promise<void> {

		const index = this.decl.passes.findIndex( ( p ) => p.name === pass );
		if ( index < 0 || this.decl.passes[ index ].kind !== 'scene' ) {

			// (a graph variant without that pass: its objects build on first use, as without a
			// warm-up, rather than stopping the page that registers them)
			this.warmFailed( `latent( '${ pass }' )`, new GraphError( 'the graph has no scene pass of that name' ) );
			return Promise.resolve();

		}
		const latent: Latent = { index, objects: [ ...objects ], kernels: [ ...kernels ] };
		if ( this.warmStarted ) return this.compileLatents( latent );
		return new Promise<void>( ( resolve ) => this.latents.push( { ...latent, resolve } ) );

	}

	/** Compiles the pipelines of every compiled pass, those that run only at times included (fog,
	 * rain, the view distance, either output), so that no frame builds one. Each pass compiles
	 * under its own binding (`bind`: its target, MRT and depth flags), restored before anything
	 * is awaited: a scene pass compiles its scene for its camera (from `f`, the frame it would
	 * draw) and its latent objects; a quad its material; a chain its pipeline's material and the
	 * passes three's nodes render before it (a bloom; see Compile.ts), or its nodes' own
	 * passes (a bloom's high pass, blurs and composite, each into its target); a compute pass its
	 * kernels. A chain with a pass out of the compiler's reach is primed instead: rendered once at
	 * the real extents (once they are set), whatever its `when`. Priming never leaves a persistent
	 * resource written: a primed pass that writes one has its history reset (`resetHistory`) and
	 * its owner's `invalidate` called, and is not primed when it has none; nor is one whose
	 * unreached nodes may keep state of their own (only FSR1Node is known not to). The graph's option
	 * `prime: false` (`fgPrime=0`) leaves such passes to their first use. A compile that fails is
	 * reported (`warmStats.failed`, a warning) and left to the pass's first frame. Runs once; later
	 * calls return the same promise. */
	warm( f?: F ): Promise<void> {

		this.warming ??= this.warmAll( f );
		return this.warming;

	}

	private async warmAll( f: F | undefined ): Promise<void> {

		const started = performance.now(); // eslint-disable-line ponlai/scene-clock -- the graph's build and warm-up times, a readout
		const compiled = this.compiled ?? this.compile();
		if ( this.executing ) throw new GraphError( 'warm() inside execute()' );
		this.warmFrame = f;
		this.warmStarted = true;
		const stats: WarmStats = this.warmStats = { passes: 0, latents: 0, kernels: 0, primed: [], skipped: [], failed: this.earlyFailures.splice( 0 ), ms: 0, primeMs: 0 };
		const pending: Promise<unknown>[] = [];
		const unreached: { index: number; why: string[] }[] = [];
		for ( const index of compiled.live ) {

			const name = this.decl.passes[ index ].name;
			this.bind( index );
			try {

				const why = this.warmPass( index, pending );
				if ( why.length > 0 ) unreached.push( { index, why } );

			} catch ( error ) {

				this.warmFailed( name, error );

			} finally {

				this.unbind();

			}

		}
		// the latents registered so far, each under its pass's binding
		for ( const latent of this.latents.splice( 0 ) ) pending.push( this.compileLatents( latent ).then( latent.resolve ) );
		await Promise.all( pending );
		stats.ms = performance.now() - started; // eslint-disable-line ponlai/scene-clock -- the graph's build and warm-up times, a readout
		if ( unreached.length > 0 ) {

			const priming = performance.now(); // eslint-disable-line ponlai/scene-clock -- the graph's build and warm-up times, a readout
			await this.primePasses( unreached, f, stats );
			stats.primeMs = performance.now() - priming; // eslint-disable-line ponlai/scene-clock -- the graph's build and warm-up times, a readout

		}

	}

	// a pass's scene, camera or kernels: as declared, or of the warm-up's frame
	private ofFrame<T>( value: T | ( ( f: F ) => T ) | undefined ): T | undefined {

		if ( typeof value !== 'function' ) return value;
		return this.warmFrame === undefined ? undefined : ( value as ( f: F ) => T )( this.warmFrame );

	}

	private warmFailed( what: string, error: unknown ): void {

		const entry = `${ what }: ${ String( ( error as Error )?.message ?? error ).split( '\n' )[ 0 ] }`;
		( this.warmStats?.failed ?? this.earlyFailures ).push( entry );
		console.warn( `FrameGraph.warm: '${ what }' did not precompile (it will compile on its first frame)`, error );

	}

	// a compile's promise, its failure reported instead of rejected
	private warmPromise( what: string, promise: Promise<unknown> ): Promise<void> {

		return promise.then( () => undefined, ( error ) => this.warmFailed( what, error ) );

	}

	// Compiles one pass under its binding (already bound); returns what of it the compiler could
	// not reach (a chain to prime).
	private warmPass( index: number, pending: Promise<unknown>[] ): string[] {

		const decl = this.decl.passes[ index ];
		const def = this.defs[ index ];
		const stats = this.warmStats!;
		const r = this.renderer;
		switch ( decl.kind ) {

			case 'scene': {

				const scene = this.ofFrame( def.scene ), camera = this.ofFrame( def.camera );
				if ( ! scene || ! camera ) {

					stats.skipped.push( `${ decl.name }: no frame to take its scene from` );
					return [];

				}
				pending.push( this.warmPromise( decl.name, compileBound( r, scene, camera ).done ) );
				const latent = def.latent?.() ?? [];
				if ( latent.length > 0 ) pending.push( this.warmPromise( `${ decl.name } (latent)`, this.compileObjects( latent, scene as THREE.Scene, camera ) ) );
				stats.passes ++;
				return [];

			}
			case 'quad': {

				if ( ! def.quad ) break;
				const c = compileQuad( r, def.quad );
				pending.push( this.warmPromise( decl.name, c.done ) );
				stats.passes ++;
				return c.unreached;

			}
			case 'chain': {

				const c = def.pipeline ? compileChain( r, def.pipeline ) : def.nodes ? compileNodes( r, def.nodes ) : null;
				if ( ! c ) break;
				pending.push( this.warmPromise( decl.name, c.done ) );
				stats.passes ++;
				return c.unreached;

			}
			case 'compute': {

				const nodes = this.ofFrame( def.compute );
				if ( ! nodes ) break;
				pending.push( this.warmPromise( decl.name, r.compileComputeAsync( nodes ) ) );
				stats.passes ++;
				return [];

			}

		}
		// a callback (or a pass whose executor is its own run()): nothing to compile
		stats.skipped.push( `${ decl.name }: no draw or kernel to compile` );
		return [];

	}

	// latent objects, as part of the scene each is in, else the pass's (already bound)
	private compileObjects( objects: readonly THREE.Object3D[], scene: THREE.Scene, camera: THREE.Camera ): Promise<void> {

		const groups = new Map<THREE.Scene, THREE.Object3D[]>();
		for ( const object of objects ) {

			const own = sceneOf( object ) ?? scene;
			groups.set( own, [ ...( groups.get( own ) ?? [] ), object ] );

		}
		return Promise.all( [ ...groups ].map( ( [ s, list ] ) => compileLatent( this.renderer, list, s, camera ).done ) ).then( () => undefined );

	}

	// a registered latent under its pass's binding, with its owner's kernels
	private compileLatents( latent: Latent ): Promise<void> {

		const name = `${ this.decl.passes[ latent.index ].name } (latent)`;
		const def = this.defs[ latent.index ];
		const scene = this.ofFrame( def.scene ) as THREE.Scene | undefined, camera = this.ofFrame( def.camera );
		const pending: Promise<void>[] = [];
		if ( latent.objects.length > 0 ) {

			if ( this.executing ) this.warmFailed( name, new GraphError( 'registered inside execute()' ) );
			else if ( ! camera ) this.warmFailed( name, new GraphError( 'no frame to take the camera from (warm( f ))' ) );
			else {

				this.bind( latent.index );
				try {

					pending.push( this.warmPromise( name, this.compileObjects( latent.objects, scene ?? new THREE.Scene(), camera ) ) );

				} catch ( error ) {

					this.warmFailed( name, error );

				} finally {

					this.unbind();

				}

			}

		}
		if ( latent.kernels.length > 0 ) pending.push( this.warmPromise( `${ name } kernels`, this.renderer.compileComputeAsync( latent.kernels ) ) );
		if ( this.warmStats ) {

			this.warmStats.latents += latent.objects.length;
			this.warmStats.kernels += latent.kernels.length;

		}
		return Promise.all( pending ).then( () => undefined );

	}

	// Renders the passes the compiler could not reach, once each, behind the loading screen at the
	// real extents, and invalidates what persistent of theirs they wrote.
	private async primePasses( list: { index: number; why: string[] }[], f: F | undefined, stats: WarmStats ): Promise<void> {

		if ( this.options.prime === false ) {

			for ( const { index, why } of list ) stats.skipped.push( `${ this.decl.passes[ index ].name }: not primed (fgPrime=0; ${ why.join( ', ' ) })` );
			return;

		}
		// the real extents: at once when they are set, else when Pipeline.setSize first sets them
		await new Promise<void>( ( resolve ) => this.onResize( { of: 'render' }, () => resolve() ) );
		for ( const { index, why } of list ) {

			const decl = this.decl.passes[ index ];
			const def = this.defs[ index ];
			// (the canvas holds the last frame only for the display, which redraws it every frame)
			const persistent = [ ...decl.writes, ...( decl.depth?.write ? [ decl.depth ] : [] ) ]
				.map( ( w ) => this.decl.resources[ w.resource ] ).filter( ( res ) => ( res.persistent || res.pingPong ) && ! res.canvas );
			if ( persistent.length > 0 && ! def.invalidate ) {

				stats.skipped.push( `${ decl.name }: not primed, it writes ${ persistent.map( ( res ) => res.name ).join( ', ' ) } (${ why.join( ', ' ) })` );
				continue;

			}
			// What the walker could not reach may keep state of its own that the graph does not see
			// (a node's history targets: TRAANode, AfterImageNode): prime only nodes known to keep
			// none, or a pass whose owner resets what priming wrote (`invalidate`).
			const unknown = why.filter( ( reason ) => ! STATELESS_NODES.has( reason.replace( / \(not set up\)$/, '' ) ) );
			if ( unknown.length > 0 && ! def.invalidate ) {

				stats.skipped.push( `${ decl.name }: not primed, it may keep state (${ unknown.join( ', ' ) })` );
				continue;

			}
			if ( decl.kind === 'scene' && f === undefined ) {

				stats.skipped.push( `${ decl.name }: not primed, no frame (${ why.join( ', ' ) })` );
				continue;

			}
			this.initTargets();
			this.bind( index );
			try {

				this.executePass( decl.kind, def, f as F );

			} catch ( error ) {

				this.warmFailed( `${ decl.name } (priming)`, error );
				continue;

			} finally {

				this.unbind();

			}
			for ( const res of persistent ) this.resetHistory( { resource: res.id, version: 0, name: res.name } );
			def.invalidate?.();
			stats.primed.push( `${ decl.name } (${ why.join( ', ' ) })` );

		}

	}

	// ---- introspection

	/** Called by `compile()` before it compiles any graph, with the graph: what a debug build
	 * appends to every graph (the debug view's pass, packages/render/post/graph/DebugView.ts). */
	static beforeCompile: ( ( graph: FrameGraph<unknown> ) => void )[] = [];

	/** The graph as plain JSON: the passes in order (kind, label, `when`, reads and writes with
	 * their versions, size class and size, the attachment ops of the last frame's mask), the
	 * resources (format, samples, persistence, size, bytes) and the totals. For diagnostics, the
	 * debug view's list and tests; deterministic for a graph and its extents. */
	describe(): GraphDescription {

		const { passes, resources } = this.decl;
		const compiled = this.compiled;
		const live = compiled ? new Set( compiled.live ) : null;
		const mask = compiled ? this.lastMask ?? compiled.fullMask : 0;
		const running = compiled ? new Set( compiled.running( mask ) ) : null;
		const table = compiled ? this.opsTable( mask ) : null;
		const size = ( cls: SizeClass ) => {

			const s = resolveSize( cls, this.extents );
			return { width: Math.max( s.width, 1 ), height: Math.max( s.height, 1 ) };

		};
		const name = ( id: number ) => resources[ id ].name;
		const describedPasses = passes.map( ( p ): PassDescription => {

			const target = compiled?.targetOf[ p.index ] ?? null;
			const cls = p.size ?? target?.size ?? null;
			return {
				index: p.index, name: p.name, kind: p.kind, label: p.label,
				culled: live ? ! live.has( p.index ) : false, runs: running ? running.has( p.index ) : false,
				when: p.whenLabel, jittered: p.jittered, mrt: p.hasMrt,
				reads: p.reads.map( ( r ) => ( { resource: name( r.resource ), version: r.previous ? 'previous' : r.version, mode: r.mode, optional: r.optional } ) ),
				writes: p.writes.map( ( w ) => ( { resource: name( w.resource ), version: w.version, load: drawsOver( p, w ) ? 'clear' : 'load', blend: w.blend ?? null } ) ),
				...( p.storage.length > 0 ? { storage: p.storage.map( ( w ) => ( { resource: name( w.resource ), version: w.version } ) ) } : {} ),
				depth: p.depth ? { resource: name( p.depth.resource ), version: p.depth.version, write: p.depth.write, clear: p.depth.clear } : null,
				target: target ? ( target.canvas ? 'canvas' : target.key ) : null,
				size: cls ? sizeClassKey( cls ) : null, actual: cls ? size( cls ) : null, scales: passScales( cls ),
				// (a copy: the table is the one execute() applies)
				ops: structuredClone( table?.passes.find( ( o ) => o.index === p.index ) ?? null ),
			};

		} );
		const describedResources = resources.map( ( r ): ResourceDescription => {

			const p = this.physical[ r.id ];
			const samples = compiled ? compiled.samplesOf( r ) : this.samplesOfDecl( r );
			const actual = size( r.desc.size );
			const owned = ! r.canvas && r.kind !== 'import';
			// a multisampled colour also has its resolve texture; a ping-pong resource two textures; a
			// mip chain a third more
			const perTexel = texelBytes( r.kind, r.desc.format, r.desc.type ) * ( r.desc.mips ? 4 / 3 : 1 );
			const copies = ( r.kind === 'color' && samples > 1 ? samples + 1 : samples ) * ( p?.textures.length ?? 1 );
			return {
				name: r.name, kind: r.kind, format: constantName( FORMATS, r.desc.format ) ?? ( r.kind === 'depth' ? 'Depth' : null ), type: constantName( TYPES, r.desc.type ),
				samples, mips: r.desc.mips === true, persistent: r.persistent, pingPong: r.pingPong, imported: ! owned, canvas: r.canvas,
				historyValid: r.persistent || r.pingPong ? p?.valid ?? false : null,
				size: sizeClassKey( r.desc.size ), actual,
				versions: r.producers.map( ( i ) => i < 0 ? null : passes[ i ].name ),
				bytes: owned ? actual.width * actual.height * perTexel * copies : 0,
				debugView: this.decl.debugView === r.id,
			};

		} );
		const signature = JSON.stringify( [
			this.options.sceneSamples, this.attachmentOps.mode,
			describedPasses.filter( ( p ) => ! p.culled ).map( ( p ) => [ p.name, p.kind, p.when, p.target, p.size ] ),
		] );
		return {
			ready: compiled !== null && this.extentsSet,
			variant: `s${ this.options.sceneSamples }-${ this.attachmentOps.mode }-${ fnv1a( signature ) }`,
			sceneSamples: this.options.sceneSamples, opsMode: this.attachmentOps.mode, mask,
			extents: { render: { ...this.extents.render }, canvas: { ...this.extents.canvas } },
			passes: describedPasses,
			resources: describedResources,
			culled: compiled ? [ ...compiled.culled ] : [],
			copies: this.decl.copies.map( ( c ) => ( { from: c.src.name, to: c.dst.name } ) ),
			jitterSpan: compiled?.jitterSpan ? compiled.jitterSpan.map( ( i ) => passes[ i ].name ) as [ string, string ] : null,
			bytes: describedResources.reduce( ( sum, r ) => sum + r.bytes, 0 ),
			rebuilds: this.rebuilds,
		};

	}

	/** The graph in Graphviz's dot language (`dot -Tsvg`): passes as boxes in order (dashed when
	 * culled, their `when` beside the label), each resource version as an ellipse, writes from a
	 * pass to the version it makes and reads from a version to its pass (dashed: optional; bold:
	 * the multisampled samples; dotted: a depth test, or the version a write loads and draws
	 * over). */
	dot(): string {

		const d = this.describe();
		// a quoted ID: quotes and backslashes escaped, line breaks as dot's own \n
		const q = ( text: string ) => `"${ text.replace( /["\\]/g, '\\$&' ).replace( /\n/g, '\\n' ) }"`;
		const lines = [ 'digraph FrameGraph {', '\trankdir=LR;', '\tnode [fontname="Helvetica", fontsize=10];', '\tedge [fontname="Helvetica", fontsize=9];' ];
		const resources = new Map( d.resources.map( ( r ) => [ r.name, r ] ) );
		const versions = new Set<string>();
		const version = ( resource: string, v: number | 'previous' ) => {

			const id = `${ resource }@${ v }`;
			if ( ! versions.has( id ) ) {

				versions.add( id );
				const r = resources.get( resource )!;
				const what = [ r.format ?? r.kind, r.type, r.samples > 1 ? `x${ r.samples }` : null, r.mips ? 'mips' : null, r.persistent && ! r.canvas ? 'persistent' : null, r.pingPong ? 'ping-pong' : null ].filter( Boolean ).join( ' ' );
				lines.push( `\t${ q( id ) } [shape=ellipse, label=${ q( `${ resource } ${ v === 'previous' ? 'previous' : `v${ v }` }\n${ r.size }${ what ? ` ${ what }` : '' }` ) }${ r.canvas ? ', style=bold' : '' }];` );

			}
			return q( id );

		};
		for ( const p of d.passes ) {

			const pass = q( `pass:${ p.name }` );
			const label = `${ p.index }. ${ p.name } (${ p.kind })\n${ p.label }${ p.when ? `\nwhen ${ p.when }` : '' }`;
			lines.push( `\t${ pass } [shape=box, label=${ q( label ) }${ p.culled ? ', style=dashed' : '' }];` );
			for ( const r of p.reads ) lines.push( `\t${ version( r.resource, r.version ) } -> ${ pass }${ r.mode === 'samples' ? ' [style=bold' + ( r.optional ? ', color=gray40' : '' ) + ']' : r.optional ? ' [style=dashed]' : '' };` );
			if ( p.depth && ! p.depth.write ) lines.push( `\t${ version( p.depth.resource, p.depth.version ) } -> ${ pass } [style=dotted];` );
			// a write that loads (draws over) the version before it depends on that version too
			for ( const w of p.storage ?? [] ) lines.push( `\t${ pass } -> ${ version( w.resource, w.version ) } [style=dashed, label="storage"];` );
			for ( const w of p.writes ) {

				if ( w.load === 'load' && w.version > 0 ) lines.push( `\t${ version( w.resource, w.version - 1 ) } -> ${ pass } [style=dotted, label="load"];` );
				lines.push( `\t${ pass } -> ${ version( w.resource, w.version ) };` );

			}
			if ( p.depth?.write ) {

				if ( ! p.depth.clear && p.depth.version > 0 ) lines.push( `\t${ version( p.depth.resource, p.depth.version - 1 ) } -> ${ pass } [style=dotted];` );
				lines.push( `\t${ pass } -> ${ version( p.depth.resource, p.depth.version ) } [label="depth"];` );

			}

		}
		for ( const c of d.copies ) lines.push( `\t${ q( `${ c.from }@${ resources.get( c.from )!.versions.length - 1 }` ) } -> ${ q( `${ c.to }@0` ) } [label="frame-end copy", style=dashed];` );
		lines.push( '}' );
		return lines.join( '\n' );

	}
}

export interface PassDescription {
	index: number; name: string; kind: PassKind; label: string;
	// not live (nothing it writes is read), and whether it ran in the last frame's mask
	culled: boolean; runs: boolean;
	when: string | null; jittered: boolean; mrt: boolean;
	reads: { resource: string; version: number | 'previous'; mode: string; optional: boolean }[];
	writes: { resource: string; version: number; load: 'clear' | 'load'; blend: string | null }[];
	// the versions written through storage (left out when none)
	storage?: { resource: string; version: number }[];
	depth: { resource: string; version: number; write: boolean; clear: boolean } | null;
	target: string | null; size: string | null; actual: Size | null; scales: 'pixel' | 'fixed';
	ops: OpsTable[ 'passes' ][ number ] | null;
}

export interface ResourceDescription {
	name: string; kind: string; format: string | null; type: string | null; samples: number; mips: boolean;
	persistent: boolean; pingPong: boolean; imported: boolean; canvas: boolean;
	// a history's validity (persistent and ping-pong resources), else null
	historyValid: boolean | null;
	size: string; actual: Size;
	// the pass that writes each version (null: before the frame)
	versions: ( string | null )[];
	// the memory of its textures (multisampled samples, resolve, both ping-pong textures); 0 for
	// what the graph does not own (the canvas, imports)
	bytes: number;
	debugView: boolean;
}

export interface GraphDescription {
	ready: boolean;
	// the graph's structure as a short key: scene samples, ops mode and a hash of its live passes
	// (names, kinds, predicates, targets and size classes)
	variant: string;
	sceneSamples: number; opsMode: string; mask: number;
	extents: Extents;
	passes: PassDescription[];
	resources: ResourceDescription[];
	culled: string[];
	copies: { from: string; to: string }[];
	jitterSpan: [ string, string ] | null;
	bytes: number;
	rebuilds: number;
}

// three's format and type constants by name, for descriptions
const FORMATS: Record<string, number> = {
	RGBA: THREE.RGBAFormat, RGB: THREE.RGBFormat, RG: THREE.RGFormat, Red: THREE.RedFormat,
	RGBAInteger: THREE.RGBAIntegerFormat, RGInteger: THREE.RGIntegerFormat, RedInteger: THREE.RedIntegerFormat,
	Depth: THREE.DepthFormat, DepthStencil: THREE.DepthStencilFormat,
};
const TYPES: Record<string, number> = {
	UnsignedByte: THREE.UnsignedByteType, Byte: THREE.ByteType, HalfFloat: THREE.HalfFloatType, Float: THREE.FloatType,
	UnsignedInt: THREE.UnsignedIntType, Int: THREE.IntType, UnsignedShort: THREE.UnsignedShortType, Short: THREE.ShortType,
};

function constantName( table: Record<string, number>, value: string | number | undefined ): string | null {

	if ( value === undefined ) return null;
	if ( typeof value === 'string' ) return value;
	for ( const key in table ) if ( table[ key ] === value ) return key;
	return String( value );

}

// bytes per texel of a format and type (three's defaults: RGBA, unsigned bytes; float depth)
function texelBytes( kind: string, format: string | number | undefined, type: string | number | undefined ): number {

	const f = constantName( FORMATS, format ) ?? ( kind === 'depth' ? 'Depth' : 'RGBA' );
	const t = constantName( TYPES, type ) ?? ( kind === 'depth' ? 'Float' : 'UnsignedByte' );
	const channels = f.startsWith( 'RGBA' ) ? 4 : f.startsWith( 'RGB' ) ? 3 : f.startsWith( 'RG' ) ? 2 : 1;
	const bytes = t === 'Float' || t === 'Int' || t === 'UnsignedInt' ? 4 : t === 'HalfFloat' || t === 'Short' || t === 'UnsignedShort' ? 2 : 1;
	return f === 'DepthStencil' ? 4 : channels * bytes;

}

// a short stable hash (FNV-1a, 32 bits, hex)
function fnv1a( text: string ): string {

	let h = 0x811c9dc5;
	for ( let i = 0; i < text.length; i ++ ) h = Math.imul( h ^ text.charCodeAt( i ), 0x01000193 );
	return ( h >>> 0 ).toString( 16 ).padStart( 8, '0' );

}
