import { REVISION, Mesh, MeshStandardNodeMaterial, PlaneGeometry } from 'three/webgpu';
import type { Camera, Light, Node, RenderTarget, Scene, WebGPURenderer } from 'three/webgpu';
import { missing } from './Internals.ts';

// Cache-key patches for three's node and pipeline caches: one shader per target layout.
//
// Cache-key patches for three r186's node and pipeline caches, so a material drawn into targets
// of different layouts (another MRT, other attachment formats or names) gets a shader and a
// pipeline per layout. Installed by main.ts right after `renderer.init()`, before any material
// builds.
//
// In r186 a draw's node build is cached by `NodeManager.getForRenderCacheKey( renderObject )`, the
// render object's initial cache key. That key ends in its render context's id, and the contexts are
// keyed by the attachment count, the first attachment's format and type, samples, depth, the MRT
// and the call depth, so those already tell builds apart; the other attachments' formats and the
// textures' names do not, though the node builder reads them while it builds (an MRT picks its
// outputs by name): two targets of one such layout whose later attachments differ shared the first
// one's build. The backend's `getRenderCacheKey` takes only the first colour attachment's format,
// and nothing of the context MRT, which carries the per-attachment blend modes (WebGPUPipelineUtils
// reads `context.mrt.getBlendMode( texture.name )`). Both keys gain the MRT and every attachment's
// format, type and name.
//
// A render object keeps the node key it was first built under: three asks for it again when it
// disposes the object (NodeManager.delete), when its shared context may hold another target's
// textures, and a key read then would miss the build's cache entry (leaking it) or evict another
// layout's. (A render object never changes its build: three makes a new one when its key changes.)
//
// `tools/render/verify-frame-graph-gpu.mjs` checks the patches (both orders of an MRT and a plain draw, a
// swapped layout). On another three revision, or a renderer without the members they patch, they
// throw, naming the entry of `three-internals.json` (Internals.ts `missing`): re-check them on a
// three upgrade (packages/render/three/README.md).
//
// One more patch, of the bindings (`installBindingPatches`): a render object's bind groups are cloned
// from its build's (NodeBuilderState.createBindings), and a cloned texture binding starts with the
// texture its node held when the material was built; `_createBindings` makes that texture ready on
// the GPU before the first update moves the binding to the node's current one. Once the app has
// put another texture in the node and disposed the first (the island's growth replaces its fields
// and wave fields: world/Growth.ts), every new render object of the material (a grown chunk's
// shadow draw) uploads the disposed texture again, and it stays on the GPU for good, since its
// dispose event has already fired (13.6 MB from a twelve-step growth). The patch starts each
// texture binding at its node's current texture, which is where the first update would move it.
// There is no public path: the build's bindings are three's internal state.
//
// And one build ahead of the frame (`primeSharedNodes`). r186 makes some nodes every lit material
// shares inside the first build that needs them, and that build's code differs from every later
// one's, though they draw the same: the light's colour times its shadow and the shadow node's own
// graph (AnalyticLightNode.setupShadow, ShadowNode.setup) are made once, in the first receiver's
// build, which declares the shadow's variable where the graph was made and keeps the product in a
// variable of its own; the camera's world matrix (`cameraWorldMatrix`, made on first use, by the
// environment's reflection) takes its node id there, and a shared uniform group lists its
// uniforms by node id (NodeBuilder._getBindGroup), so the render group's order differs between a
// build that made its own screen-size uniform before that matrix existed and one after. Which
// receiver built first followed the land (the meadow and the vegetation then drawn beside it at render
// order -1, by their distance from the camera): a page loaded at grow=6 built the meadow's first batch
// apart from its other two, one pipeline more than a page grown to it. A throwaway receiver of the
// scene's lights and environment builds first, before the loop's first frame, so every receiver
// the frame builds has the later code whatever its order. It makes no pipeline; three keeps the
// shadow map it creates for the real receivers (so the map's first redraw moves to the loop's first
// frame, SunShadow.ts). Paused views at grow 0, 6 and 12 drew the same pixels with it as without.
//
// And the graph's depth clears (`ownDepthClears`). r186 clears a render target's depth before the
// first render into it whenever the renderer's flags clear colour but not depth
// (`Renderer._renderScene`, its `depthInitialized` record of the target), and a resize disposes the
// target, which drops that record. So on the frame after every change of the render scale the
// opaque pass (colour cleared, the depth loaded) wiped the depth the vegetation's prepass had just
// drawn into the shared depth (VegetationPrepass.ts), and every shaded draw testing depth for
// equality against it failed: the meadow, the bamboo's leaves and the yunpeng vanished for one
// frame. The graph declares every clear and load of the depth it owns, so it marks each target
// with depth initialised as it binds it, and three adds no clear of its own.

type Texture = { format: unknown; type: unknown; name: string };
type RenderContext = { mrt?: { id: number } | null; textures: Texture[] | null };
type RenderObject = { initialCacheKey: unknown; context: RenderContext };
type Callable = ( this: unknown, ...args: unknown[] ) => unknown;
type Host = Record<string, unknown>;
type BindGroup = { bindings: { isSampledTexture?: boolean; textureNode?: { value: unknown } | null; texture: unknown }[] };

// the attachment layout: their count, and each texture's format, type and name
function layoutKey( context: RenderContext ): string {

	const textures = context.textures;
	if ( textures === null ) return 'canvas';
	return `${ textures.length }:${ textures.map( ( t ) => `${ t.format }/${ t.type }/${ t.name }` ).join( ',' ) }`;

}

const patched = new WeakSet<object>();

// three's revision, which the patches were written against (r186): another throws
function checkRevision( user: string, revision: string ): void {

	if ( ! /^186(\.|$)/.test( revision ) ) throw missing( 'revision', `${ user } (written for r186, this is r${ revision })` );

}

/** Patches the renderer's node manager and backend (instances) so their cache keys include the
 * context MRT and the whole attachment layout (the node key fixed per render object). Call after
 * `renderer.init()`. Throws on a three revision other than 186, whose internals it was written
 * against, or on a renderer without the members it patches (not initialised, or not WebGPU). Returns
 * true (a patched renderer is left as it is). */
export function installCacheKeyPatches( renderer: unknown, revision: string = REVISION ): true {

	checkRevision( 'installCacheKeyPatches', revision );
	const r = renderer as { _nodes?: Host | null; backend?: Host & { utils?: { getCurrentColorFormats?( context: RenderContext ): string[] } } };
	const nodes = r._nodes, backend = r.backend;
	if ( typeof nodes?.getForRenderCacheKey !== 'function' ) throw missing( 'NodeManager.getForRenderCacheKey', 'installCacheKeyPatches' );
	if ( typeof backend?.getRenderCacheKey !== 'function' ) throw missing( 'WebGPUBackend.getRenderCacheKey', 'installCacheKeyPatches' );
	if ( typeof backend.utils?.getCurrentColorFormats !== 'function' ) throw missing( 'WebGPUUtils.getCurrentColorFormats', 'installCacheKeyPatches' );
	if ( patched.has( nodes ) ) return true;
	patched.add( nodes );

	const nodeKey = nodes.getForRenderCacheKey as Callable;
	const keys = new WeakMap<RenderObject, string>();
	nodes.getForRenderCacheKey = function ( this: unknown, renderObject: RenderObject ) {

		let key = keys.get( renderObject );
		if ( key === undefined ) {

			const context = renderObject.context;
			key = `${ nodeKey.call( this, renderObject ) }|${ context.mrt?.id ?? '-' }|${ layoutKey( context ) }`;
			keys.set( renderObject, key );

		}
		return key;

	};
	const pipelineKey = backend.getRenderCacheKey as Callable;
	backend.getRenderCacheKey = function ( this: unknown, renderObject: RenderObject ) {

		const context = renderObject.context;
		const formats = backend.utils!.getCurrentColorFormats!( context );
		return `${ pipelineKey.call( this, renderObject ) },${ formats.join( '/' ) },${ context.mrt?.id ?? '-' }`;

	};
	return true;

}

/** Patches the renderer's bindings (an instance) so that a new bind group's texture bindings start at
 * their node's current texture, not the one the material was built with (see the header). Call after
 * `renderer.init()`. Throws on a three revision other than 186, or on a renderer without the member
 * it patches. */
export function installBindingPatches( renderer: unknown, revision: string = REVISION ): void {

	checkRevision( 'installBindingPatches', revision );
	const bindings = ( renderer as { _bindings?: Host | null } )._bindings;
	if ( typeof bindings?._createBindings !== 'function' ) throw missing( 'Bindings._createBindings', 'installBindingPatches' );
	if ( patched.has( bindings ) ) return;
	patched.add( bindings );

	const create = bindings._createBindings as Callable;
	bindings._createBindings = function ( this: unknown, groups: BindGroup[] ) {

		for ( const group of groups ) for ( const binding of group.bindings ) {

			// (the setter resets the binding's version when the texture changes)
			if ( binding.isSampledTexture && binding.textureNode ) binding.texture = binding.textureNode.value;

		}
		return create.call( this, groups );

	};

}

/** Builds one throwaway receiver of `scene`'s lights and environment through the backend's node
 * builder (no pipeline), so the nodes r186 makes inside the first lit build exist before any material
 * the frame draws is built (see the header). Call once, after the lights, their shadows' filters
 * and the scene's environment are in place and before the first frame. Throws on a three revision
 * other than 186, or on a renderer without the members it builds through. */
export function primeSharedNodes( renderer: WebGPURenderer, scene: Scene, camera: Camera, revision: string = REVISION ): void {

	checkRevision( 'primeSharedNodes', revision );
	type Builder = { scene: Scene; material: unknown; camera: Camera; context: Record<string, unknown>; lightsNode: unknown; environmentNode: unknown; build(): void };
	const r = renderer as unknown as { lighting?: { createNode?( lights: Light[] ): unknown }; backend: { createNodeBuilder?( object: Mesh, renderer: unknown ): Builder } };
	if ( typeof r.lighting?.createNode !== 'function' ) throw missing( 'Lighting.createNode', 'primeSharedNodes' );
	if ( typeof r.backend.createNodeBuilder !== 'function' ) throw missing( 'WebGPUBackend.createNodeBuilder', 'primeSharedNodes' );
	const lights: Light[] = [];
	scene.traverseVisible( ( object ) => {

		if ( ( object as Light ).isLight ) lights.push( object as Light );

	} );
	const material = new MeshStandardNodeMaterial();
	const mesh = new Mesh( new PlaneGeometry(), material );
	mesh.receiveShadow = true;
	const builder = r.backend.createNodeBuilder( mesh, renderer );
	builder.scene = scene;
	builder.material = material;
	builder.camera = camera;
	builder.context.material = material;
	builder.lightsNode = r.lighting.createNode( lights );
	builder.environmentNode = ( scene as unknown as { environmentNode?: Node | null } ).environmentNode ?? null;
	builder.build();
	mesh.geometry.dispose();
	material.dispose();

}

/** Marks `target`'s depth as initialised for three, so r186 never clears it on its own: the frame
 * graph decides every clear and load of the depth it owns (see the header). Call as the graph binds
 * a target with depth, before it renders into it. A renderer without three's texture records (the
 * frame graph's tables run it on stand-in renderers) is left alone: in the app `checkRenderer`
 * (Internals.ts) has checked them at start. */
export function ownDepthClears( renderer: unknown, target: RenderTarget ): void {

	// (`get` makes the record when the target has none yet; three's first render then fills it in)
	const textures = ( renderer as { _textures?: { get( target: RenderTarget ): { depthInitialized?: boolean } } } )._textures;
	if ( textures ) textures.get( target ).depthInitialized = true;

}
