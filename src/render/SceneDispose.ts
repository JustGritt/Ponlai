// What a scene leaves in the render pipeline's modules, released when the scene is disposed.
//
// A few of the render pipeline's modules hold one scene's state at module level: what the scene
// installed (the shared uniforms, the local lights' table, the ground bounce, the ambient volume),
// what it registered (the cut listeners, the materials the wind follows, the tuned values' nodes
// and listeners) and what its frames counted (the motion's frame, the rest's epoch). Each such
// module registers, once as it loads, the release that puts it back as the page loaded it
// (`onSceneDispose`); the running scene's `dispose` (assemble/StartScene.ts) runs them all
// (`releaseScene`), last registered first, so the next `startScene` on the page builds and draws as
// a fresh load does (tools/render/scenes.mjs `--rebuild` checks it).
// - A release is idempotent: it runs on every dispose, whether or not its module was used.
// - What a module allocates as it loads is reset, not reallocated, unless it grew.
// - A module that must tell what a scene added from what the page loaded with records it as the
//   scene starts (`onSceneStart`; the tuned values, debug/Tune.ts).
// - Caches that hold only what their key determines (the jitter's means, the physics' tables, the
//   star catalogue, the environment BRDF's table) are kept: a fresh load would make the same values.

const releases = new Map<string, () => void>();
const starts = new Map<string, () => void>();

/** Registers, once as the module loads, the release that puts it back as the page loaded it. */
export function onSceneDispose( name: string, release: () => void ): void {

	if ( releases.has( name ) ) throw new Error( `SceneDispose: '${ name }' is registered twice` );
	releases.set( name, release );

}

/** Registers, once as the module loads, what it records as each scene starts building. */
export function onSceneStart( name: string, start: () => void ): void {

	if ( starts.has( name ) ) throw new Error( `SceneDispose: '${ name }' is registered twice` );
	starts.set( name, start );

}

/** Runs every start registered, as a scene starts building (StartScene.ts). */
export function startingScene(): void {

	for ( const start of starts.values() ) start();

}

/** Runs every release registered, last first (StartScene.ts `dispose`). */
export function releaseScene(): void {

	for ( const release of [ ...releases.values() ].reverse() ) release();

}
