import type { WebGPURenderer } from 'three/webgpu';

// The three.js animation loop and the bookkeeping around each frame, for frames stepped by hand.
//
// three r186's animation loop (`Renderer._animation`), which the app's frame pacer drives, and the
// bookkeeping it does around each frame, which a frame stepped by hand repeats
// (assemble/SceneFrame.ts `loop.step`): the inspector's frame closed and opened, the renderer's info reset, the
// node frame advanced (its id is three's `info.frame`).

type Inspector = { isRunning?: boolean; begin(): void; finish(): void };
type Internals = {
	_animation: { start(): void; stop(): void; setContext( context: unknown ): void };
	_nodes: { nodeFrame: { update(): void; frameId: number } };
	_inspector: Inspector;
};

/** three's animation loop: `setContext` gives it what it asks for animation frames, `start` and
 * `stop` run and halt it. */
export function threeAnimation( renderer: WebGPURenderer ): Internals[ '_animation' ] {

	return ( renderer as unknown as Internals )._animation;

}

/** What three's animation loop does before and after each frame (Animation.start's update), for a
 * frame run outside it: `begin` before the frame's work, `end` after it. */
export function frameBookkeeping( renderer: WebGPURenderer ): { begin(): void; end(): void } {

	const r = renderer as unknown as Internals;
	const nodeFrame = r._nodes.nodeFrame, inspector = r._inspector;
	return {
		begin(): void {

			if ( inspector.isRunning ) inspector.finish();
			if ( renderer.info.autoReset ) renderer.info.reset();
			nodeFrame.update();
			( renderer.info as { frame: number } ).frame = nodeFrame.frameId;
			inspector.begin();

		},
		end(): void {

			inspector.finish();

		},
	};

}
