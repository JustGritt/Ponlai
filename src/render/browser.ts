// The render pipeline's browser entry point: the host a scene takes from a browser page.
//
// The render pipeline's browser entry point: `browserHost`, the host (assemble/SceneHost.ts) a
// running scene takes from a browser page: the window, the document, the loading screen, the
// globals. Only the app layer imports it (tools/repo/verify-layers.mjs), so no other package reads the
// browser through the render pipeline. It holds re-exports only.
export { browserHost } from './assemble/BrowserHost.ts';
