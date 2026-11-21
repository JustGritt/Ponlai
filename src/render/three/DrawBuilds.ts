import type { Camera, Material, MRTNode, Mesh, Node, NodeMaterial, Object3D, QuadMesh, RenderPipeline, RenderTarget, Scene, Texture, WebGPURenderer } from 'three/webgpu';
import type BloomNode from 'three/addons/tsl/display/BloomNode.js';
import { missing } from './Internals.ts';

// The three.js steps for building a draw, which the frame graph's ahead-of-time compiles take.
//
// three r186's steps for building a draw, which the frame graph's ahead-of-time compiles
// (post/graph/Compile.ts) take in place of `renderer.compileAsync`'s own: what a draw's node state
// and pipeline are made from, and the passes a bloom draws. Each is three's own code or state,
// mirrored or read as r186 has it (`three-internals.json`); post/graph/Compile.ts says why the
// compiles need them.

/** A draw's node state: the nodes that render passes of their own before the draw. */
export type NodeState = { updateBeforeNodes: Node[] };

type RenderObject = { drawRange: unknown; group: unknown };
type RenderContext = { textures: unknown; depthTexture: unknown; depth: boolean; stencil: boolean };
type Internals = {
	_objects: { get( ...args: unknown[] ): RenderObject };
	_nodes: { getForRender( renderObject: RenderObject ): NodeState; updateForRender( renderObject: RenderObject ): void };
	_geometries: { updateForRender( renderObject: RenderObject ): void };
	_bindings: { updateForRender( renderObject: RenderObject ): void };
	_pipelines: PipelineCache;
	_currentRenderContext: RenderContext;
	_renderContexts: { get( target: RenderTarget | null, mrt: MRTNode | null, callDepth: number ): RenderContext };
	_isPreCompiling: boolean;
	_createObjectPipeline?: ( ...args: unknown[] ) => void;
	_compilationPromises: unknown;
};
type PipelineCache = { getForRender( renderObject: RenderObject, promises: Promise<unknown>[] | null ): unknown; updateForRender?( renderObject: RenderObject ): void };
type CreateObjectPipeline = ( this: Internals, object: Object3D, material: Material, scene: Scene, camera: Camera,
	lightsNode: unknown, group: unknown, clippingContext: unknown, passId: unknown ) => void;

// the renderer's members the draw builds go through (a renderer of another three throws, naming the
// first one missing)
function internals( renderer: WebGPURenderer ): Internals {

	const r = renderer as unknown as Partial<Internals>;
	const members: [ string, boolean ][] = [
		[ 'Renderer._createObjectPipeline', typeof r._createObjectPipeline === 'function' ],
		[ 'RenderObjects.get', typeof r._objects?.get === 'function' ],
		[ 'NodeManager.getForRender', typeof r._nodes?.getForRender === 'function' ],
		[ 'NodeManager.updateForRender', typeof r._nodes?.updateForRender === 'function' ],
		[ 'Geometries.updateForRender', typeof r._geometries?.updateForRender === 'function' ],
		[ 'Bindings.updateForRender', typeof r._bindings?.updateForRender === 'function' ],
		[ 'Pipelines.getForRender', typeof r._pipelines?.getForRender === 'function' ],
		[ 'Renderer._compilationPromises', '_compilationPromises' in r ],
		[ 'RenderContexts.get', typeof r._renderContexts?.get === 'function' ],
	];
	for ( const [ entry, holds ] of members ) if ( ! holds ) throw missing( entry, 'buildDrawsNow (packages/render/three/DrawBuilds.ts)' );
	return r as Internals;

}

/** Builds the draws of `object` as `renderer.compileAsync( object, camera, scene )` collects them,
 * each with three's own steps for a draw (its compileAsync loop's and `_createObjectPipeline`'s),
 * in this call instead of after an await, and without `updateBefore`, which would render the passes
 * a material holds. A draw at `callDepth` > 0 takes its call depth's render context, filled as a
 * render fills it (compileAsync collects in the top level's). Returns the draws' node states, and
 * a promise that resolves once their pipelines exist. */
export function buildDrawsNow( renderer: WebGPURenderer, object: Object3D, camera: Camera, scene: Scene | null, callDepth: number ): { states: NodeState[]; done: Promise<void> } {

	const r = internals( renderer );
	const states: NodeState[] = [];
	const pending: Promise<unknown>[] = [];
	const own = Object.prototype.hasOwnProperty.call( r, '_createObjectPipeline' ) ? r._createObjectPipeline! : null;
	let nested: RenderContext | null = null;
	const build: CreateObjectPipeline = function ( object, material, scene, camera, lightsNode, group, clippingContext, passId ) {

		// (compileAsync's own context, filled for this collection)
		let context = r._currentRenderContext;
		if ( callDepth > 0 ) {

			if ( ! nested ) {

				nested = r._renderContexts.get( renderer.getRenderTarget(), renderer.getMRT(), callDepth );
				nested.textures = context.textures;
				nested.depthTexture = context.depthTexture;
				nested.depth = context.depth;
				nested.stencil = context.stencil;

			}
			context = nested;

		}
		const ro = r._objects.get( object, material, scene, camera, lightsNode, context, clippingContext, passId );
		ro.drawRange = ( object as Mesh ).geometry.drawRange;
		ro.group = group;
		r._isPreCompiling = true;
		try {

			states.push( r._nodes.getForRender( ro ) );
			r._geometries.updateForRender( ro );
			r._nodes.updateForRender( ro );
			r._bindings.updateForRender( ro );

		} finally {

			r._isPreCompiling = false;

		}
		r._pipelines.getForRender( ro, pending );

	};
	r._createObjectPipeline = build as unknown as ( ...args: unknown[] ) => void;
	let collected: Promise<void>;
	try {

		// collects synchronously (its own queue stays empty: `build` did the work)
		collected = renderer.compileAsync( object, camera, scene ) as Promise<void>;

	} finally {

		if ( own ) r._createObjectPipeline = own;
		else delete r._createObjectPipeline;

	}
	return { states, done: Promise.all( [ collected, ...pending ] ).then( () => undefined ) };

}

/** Runs `render` with the pipelines its draws need created asynchronously, as `compileAsync`
 * creates them (a draw whose pipeline is not ready draws nothing), standing in for three's
 * `Pipelines.updateForRender` for the call. Resolves once the pipelines exist. */
export function renderWithAsyncPipelines( renderer: WebGPURenderer, render: () => void ): Promise<void> {

	const pipelines = ( renderer as unknown as { _pipelines?: PipelineCache } )._pipelines;
	if ( typeof pipelines?.getForRender !== 'function' ) throw missing( 'Pipelines.getForRender', 'renderWithAsyncPipelines (packages/render/three/DrawBuilds.ts)' );
	if ( typeof pipelines.updateForRender !== 'function' ) throw missing( 'Pipelines.updateForRender', 'renderWithAsyncPipelines (packages/render/three/DrawBuilds.ts)' );
	const promises: Promise<unknown>[] = [];
	const own = Object.prototype.hasOwnProperty.call( pipelines, 'updateForRender' ) ? pipelines.updateForRender : null;
	pipelines.updateForRender = ( renderObject ) => void pipelines.getForRender( renderObject, promises );
	try {

		render();

	} finally {

		if ( own ) pipelines.updateForRender = own;
		else delete pipelines.updateForRender;

	}
	return Promise.all( promises ).then( () => undefined );

}

type Bloom = BloomNode & {
	_highPassFilterMaterial: NodeMaterial | null; _separableBlurMaterials: NodeMaterial[]; _compositeMaterial: NodeMaterial | null;
	_renderTargetBright: RenderTarget; _renderTargetsHorizontal: RenderTarget[]; _renderTargetsVertical: RenderTarget[]; _nMips: number;
};

/** The passes a BloomNode's `updateBefore` draws, in its order: each one's material (null where the
 * node is not set up), its target and the colour input it sets on a blur (null: none). The high
 * pass, each mip's two blur passes, the composite; only the first mips' blur materials are drawn
 * (setup appends a set per build of the material that holds the bloom). */
export function bloomDraws( bloom: BloomNode ): [ NodeMaterial | null, RenderTarget, Texture | null ][] {

	const b = bloom as Bloom;
	const members: [ string, boolean ][] = [
		[ 'BloomNode._highPassFilterMaterial', '_highPassFilterMaterial' in b ],
		[ 'BloomNode._separableBlurMaterials', Array.isArray( b._separableBlurMaterials ) ],
		[ 'BloomNode._compositeMaterial', '_compositeMaterial' in b ],
		[ 'BloomNode._renderTargetBright', Boolean( b._renderTargetBright ) ],
		[ 'BloomNode._renderTargetsHorizontal', Array.isArray( b._renderTargetsHorizontal ) ],
		[ 'BloomNode._renderTargetsVertical', Array.isArray( b._renderTargetsVertical ) ],
		[ 'BloomNode._nMips', typeof b._nMips === 'number' ],
	];
	for ( const [ entry, holds ] of members ) if ( ! holds ) throw missing( entry, 'bloomDraws (packages/render/three/DrawBuilds.ts)' );
	const draws: [ NodeMaterial | null, RenderTarget, Texture | null ][] = [ [ b._highPassFilterMaterial, b._renderTargetBright, null ] ];
	let input = b._renderTargetBright;
	for ( let i = 0; i < b._nMips; i ++ ) {

		const material = b._separableBlurMaterials[ i ] ?? null;
		draws.push( [ material, b._renderTargetsHorizontal[ i ], input.texture ], [ material, b._renderTargetsVertical[ i ], b._renderTargetsHorizontal[ i ].texture ] );
		input = b._renderTargetsVertical[ i ];

	}
	draws.push( [ b._compositeMaterial, b._renderTargetsHorizontal[ 0 ], null ] );
	return draws;

}

/** A render pipeline's full-screen quad, its material set from the output node as its first render
 * sets it (three's `_update`, which compares the renderer's tone mapping and colour space with its
 * own: call it before changing them). */
export function pipelineQuad( pipeline: RenderPipeline ): QuadMesh {

	const p = pipeline as unknown as { _update?(): void; _quadMesh?: QuadMesh };
	if ( typeof p._update !== 'function' ) throw missing( 'RenderPipeline._update', 'pipelineQuad (packages/render/three/DrawBuilds.ts)' );
	if ( ! p._quadMesh ) throw missing( 'RenderPipeline._quadMesh', 'pipelineQuad (packages/render/three/DrawBuilds.ts)' );
	p._update();
	return p._quadMesh;

}
