import type * as THREE from 'three/webgpu';
import type { FrameGraph } from './graph/FrameGraph.ts';
import { checkDeclarations, type Declarations } from '../Extensions.ts';

// The pipeline's two openings to what it does not know: extra passes and deferred draws.
//
// The pipeline's two openings to what it does not know: the scene's extra passes, and the names
// of its scene passes that take draws registered for later.

/** A pass of the scene's own that the pipeline declares without knowing it (Pipeline.ts, the
 * constructor's `extraPasses`). `declare` is called once as the graph is declared and adds its
 * pass or passes to `graph` (`addPass`), with what they read and write declared as any pass's.
 * The pipeline declares its list in order at one place, after the displacement field's pass and
 * before the wave maps', ahead of the scene passes; what a pass needs that exists only later (the
 * world) it takes through a getter of its own and stays idle until then (the pass's `when`).
 * `name` names it in errors (its passes' names are its own, given to `addPass`). It states the
 * three declarations every registered definition does (Extensions.ts), or it is refused.
 * The one place in the order and a `declare` that cannot read the frame's own data (it is generic
 * over `Frame`) are a deliberate limit for now: the one pass the app hands in (the falling
 * leaves) needs neither, and another opening is added when a pass does. */
export interface ExtraPass extends Declarations {
	name: string;
	declare<Frame>( graph: FrameGraph<Frame> ): void;
}

/** Declares the extra passes on `graph`, in the list's order; a pass without its three
 * declarations is refused (it throws). */
export function declareExtraPasses<Frame>( graph: FrameGraph<Frame>, passes: readonly ExtraPass[] ): void {

	for ( const pass of passes ) {

		checkDeclarations( 'ExtraPass', pass.name, pass );
		pass.declare( graph );

	}

}

/** The pipeline's scene passes a draw or a kernel that runs only at times is registered for, so
 * the warm-up compiles it ahead (`latent`): the opaque scene, the sea, the rain's layer, the
 * overlay and what draws after the resolve (PipelinePasses.ts declares them under these names). */
export const SCENE_PASSES = [ 'opaque', 'water', 'rain', 'overlay', 'rainAfter' ] as const;
export type ScenePass = typeof SCENE_PASSES[ number ];

/** The pipeline's latents by pass name: the frame graph's `latent` (graph/FrameGraph.ts
 * `LatentRegistry`) with the pass's name typed, so a misspelt name does not compile. */
export interface SceneLatents {
	latent( pass: ScenePass, objects: readonly THREE.Object3D[], kernels?: readonly THREE.ComputeNode[] ): Promise<void>;
}
