# The render pipeline's rules

`packages/render/` is the render pipeline, built without the island: lighting, sky, sea, weather,
the parts every plant kind shares, the frame's passes and frame graph, the look's readers, and the
uses of three.js internals. These rules apply to this folder alone. The root `AGENTS.md` holds the
rules every change follows; its section Shaders and GPU work holds the GPU rules this folder shares
with the island.

## Shaders and passes

Each rule names its section of `docs/reference/three-behaviour.md`, which holds the detail.

- Declare a texture's history on the frame graph. Data a frame reads from the previous frame
  advances once per rendered frame (Previous-frame data; partly checked by
  `ponlai/history-on-graph`, `verify-framegraph`).
- A fragment output that writes multiple render targets is the `outputStruct` itself (Multiple
  render targets).
- A shader whose depth must be bit-identical to another's takes `@invariant` on its clip position
  (Bit-identical depth).
- A full-screen pass takes the scene camera's matrices as uniforms (A full-screen pass's camera).
- Read multisampled depth with `textureLoad` (Multisampled depth).

## Checks run by hand

- After a change to the render pipeline, run `tools/render/scenes.mjs` first: it renders the render
  pipeline's test scenes and checks them in seconds.
