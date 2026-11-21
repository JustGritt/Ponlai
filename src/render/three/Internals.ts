import { REVISION } from 'three/webgpu';

// The checks of the three.js internals the app depends on beyond three's public interface.
//
// The three.js internals the app depends on beyond three's public interface are listed in `three-internals.json`
// (an entry per member, method or mirrored code: where it lives in three, how it is used, which of
// our functions needs it), and `tools/repo/verify-three-internals.mjs` checks the list against the three
// installed. A dependency that no longer holds is an error naming its entry, never a fallback to
// another path: `missing` makes that error, and `checkRenderer` checks the members of the
// renderer's instance that the functions of this directory use once the renderer is initialised
// (the static check cannot see what three assigns at run time). An upgrade of three follows
// packages/render/three/README.md.

/** The error for a dependency on three.js internals that does not hold: `entry` is the list's id
 * (`three-internals.json`), `user` the function of ours that needs it. */
export function missing( entry: string, user: string ): Error {

	return new Error( `three r${ REVISION } lacks '${ entry }' (packages/render/three/three-internals.json), which ${ user } needs: see packages/render/three/README.md` );

}

// the instance members the functions of this directory use: [ list entry, path from the renderer,
// what it must be ]. (`_createObjectPipeline`, `_getShadowNodes` and the bloom's and the PMREM
// generator's members live on prototypes or other objects: their users check them.)
const MEMBERS: [ string, string, 'object' | 'function' | 'present' ][] = [
	[ 'Renderer._nodes', '_nodes', 'object' ],
	[ 'NodeManager.getForRenderCacheKey', '_nodes.getForRenderCacheKey', 'function' ],
	[ 'NodeManager.getForRender', '_nodes.getForRender', 'function' ],
	[ 'NodeManager.updateForRender', '_nodes.updateForRender', 'function' ],
	[ 'NodeManager.nodeFrame', '_nodes.nodeFrame', 'object' ],
	[ 'Renderer._bindings', '_bindings', 'object' ],
	[ 'Bindings._createBindings', '_bindings._createBindings', 'function' ],
	[ 'Bindings.updateForRender', '_bindings.updateForRender', 'function' ],
	[ 'Renderer._textures', '_textures', 'object' ],
	[ 'DataMap.get', '_textures.get', 'function' ],
	[ 'Renderer._attributes', '_attributes', 'object' ],
	[ 'DataMap.get', '_attributes.get', 'function' ],
	[ 'Attributes.update', '_attributes.update', 'function' ],
	[ 'Attributes.delete', '_attributes.delete', 'function' ],
	[ 'Renderer._objects', '_objects', 'object' ],
	[ 'RenderObjects.get', '_objects.get', 'function' ],
	[ 'Renderer._geometries', '_geometries', 'object' ],
	[ 'Geometries.updateForRender', '_geometries.updateForRender', 'function' ],
	[ 'Renderer._pipelines', '_pipelines', 'object' ],
	[ 'Pipelines.getForRender', '_pipelines.getForRender', 'function' ],
	[ 'Pipelines.updateForRender', '_pipelines.updateForRender', 'function' ],
	[ 'Renderer._renderContexts', '_renderContexts', 'object' ],
	[ 'RenderContexts.get', '_renderContexts.get', 'function' ],
	[ 'Renderer._currentRenderContext', '_currentRenderContext', 'present' ],
	[ 'Renderer._isPreCompiling', '_isPreCompiling', 'present' ],
	[ 'Renderer._compilationPromises', '_compilationPromises', 'present' ],
	[ 'Renderer._createObjectPipeline', '_createObjectPipeline', 'function' ],
	[ 'Renderer._getShadowNodes', '_getShadowNodes', 'function' ],
	[ 'Renderer._animation', '_animation', 'object' ],
	[ 'Animation.start', '_animation.start', 'function' ],
	[ 'Animation.setContext', '_animation.setContext', 'function' ],
	[ 'Renderer._inspector', '_inspector', 'object' ],
	[ 'Renderer.lighting', 'lighting', 'object' ],
	[ 'Lighting.createNode', 'lighting.createNode', 'function' ],
	[ 'WebGPUBackend.device', 'backend.device', 'object' ],
	// (a getter that configures the canvas's context on first read: only its presence is checked)
	[ 'WebGPUBackend.context', 'backend.context', 'present' ],
	[ 'WebGPUBackend.utils', 'backend.utils', 'object' ],
	[ 'WebGPUUtils.getCurrentColorFormats', 'backend.utils.getCurrentColorFormats', 'function' ],
	[ 'WebGPUBackend.getRenderCacheKey', 'backend.getRenderCacheKey', 'function' ],
	[ 'WebGPUBackend.createNodeBuilder', 'backend.createNodeBuilder', 'function' ],
	[ 'WebGPUBackend.beginRender', 'backend.beginRender', 'function' ],
	[ 'WebGPUBackend.updateAttribute', 'backend.updateAttribute', 'function' ],
	[ 'WebGPUBackend.createRenderPipeline', 'backend.createRenderPipeline', 'function' ],
	[ 'WebGPUBackend.createComputePipeline', 'backend.createComputePipeline', 'function' ],
	[ 'WebGPUBackend.hasFeature', 'backend.hasFeature', 'function' ],
	[ 'Backend.get', 'backend.get', 'function' ],
	[ 'Backend.trackTimestamp', 'backend.trackTimestamp', 'present' ],
	[ 'Backend.timestampQueryPool', 'backend.timestampQueryPool', 'object' ],
];

/** Checks that `renderer` (a WebGPURenderer, initialised, on the WebGPU backend) has every member of
 * its instance the functions of this directory use; throws the error naming the first one missing.
 * Call once, right after `renderer.init()`. */
export function checkRenderer( renderer: unknown ): void {

	for ( const [ entry, path, kind ] of MEMBERS ) {

		let value: unknown = renderer, present = true;
		const keys = path.split( '.' );
		for ( let i = 0; i < keys.length && present; i ++ ) {

			present = value !== null && typeof value === 'object' && keys[ i ] in value;
			// (a member only present is never read)
			if ( present && ( kind !== 'present' || i < keys.length - 1 ) ) value = ( value as Record<string, unknown> )[ keys[ i ] ];

		}
		const holds = present && ( kind === 'present' || ( kind === 'function' ? typeof value === 'function' : value !== null && typeof value === 'object' ) );
		if ( ! holds ) throw missing( entry, `checkRenderer (packages/render/three/Internals.ts, renderer.${ path })` );

	}

}
