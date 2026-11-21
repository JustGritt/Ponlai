# three.js internals

This folder holds every use of three.js internals by the render pipeline, and this page gives the
steps for moving the project to another version of three.js. The render pipeline depends on parts of
three.js that are not its public interface: private members of the renderer and its helpers, the
WebGPU backend and its device, methods it replaces, and three.js code it mirrors. Each of these
dependencies is a function in `packages/render/three/`, and
`packages/render/three/three-internals.json` lists what they depend on. `three` and `@types/three`
are pinned to exact versions, so the list holds until someone changes the pin.

An upgrade is the only time the list can stop holding, and then the code reports an error instead of
taking another path:

- `tools/repo/verify-three-internals.mjs` fails on each entry missing from the installed three.js,
  and on each replaced, wrapped, mirrored or relied-on function whose code changed (its
  fingerprint).
- When the app runs, `checkRenderer` (`packages/render/three/Internals.ts`) checks the renderer's
  members once it is initialised, and each function in `packages/render/three/` throws when a member
  it needs is missing. The error names the list's entry and the function that needs it.

The other dependencies are on three.js's behaviour, with no code that uses its internals: [three.js
and WebGPU behaviour](../../../docs/reference/three-behaviour.md) lists them as the entries marked
r186, each with the check that detects it.

## Upgrade three.js

1. Create a branch. Read three.js's migration guide and release notes for every version between the
   pinned one and the new one, and note what touches an entry of `three-internals.json` or an entry
   marked r186 in `docs/reference/three-behaviour.md`.
2. Change `three` and `@types/three` together, to exact versions (no `^` or `~`), and install:
   `npm install --save-exact three@<version> @types/three@<version>`. Set the list's `three`,
   `types` and `revision` to the new versions.
3. Run `node tools/repo/verify-three-internals.mjs`. For each failure, read three.js's new code,
   change the function in `packages/render/three/` that depends on the entry (or its patch, wrapper
   or mirrored copy), and update the entry. Print the new fingerprints with `--fingerprints` and
   record each only after reading the code it covers. Repeat until the check passes.
4. Run `npm run typecheck` and `npm run build`.
5. Compare the shader code: build the WGSL with `tools/render/wgsl-build.mjs` on the old version and
   the new one (its header), and note which shaders changed and why.
6. Run the checks that need no GPU: `npm run check -- --full`.
7. Run the GPU checks that compare two paths of the same code, which must still pass byte for byte:
   the scene jump (`tools/island/verify-scene-jump.mjs`) and the growth swap
   (`tools/island/verify-growth-swap.mjs`). The app must also start without an error from
   `packages/render/three/`.
8. Compare the rendered output with captures from before the upgrade
   (`tools/kit/parity/capture-set.mjs`). An upgrade need not match byte for byte: judge it within
   the temporal tolerance (`tools/kit/parity/README.md`, How a tolerance is set), explain each
   difference, then capture new reference sets.
9. Check the pipelines: the count the app creates (`diagnostics.pipelines`,
   `packages/render/debug/PipelineCounter.ts`), that no player step creates one
   (`tools/render/verify-first-use.mjs`), and that a growth swap creates none
   (`tools/island/verify-growth-swap.mjs`).
10. Measure the frame time before and after the upgrade
    (`skills/measuring-performance-changes/SKILL.md`).
11. Go through the entries marked r186 in `docs/reference/three-behaviour.md`, each with its check.
    Where three.js no longer behaves as an entry says, remove the code that works around it and the
    entry, and the line of `AGENTS.md` (or `packages/render/AGENTS.md`) or the lint rule that states
    it.
12. Update the comments that name the old version (search for `r186`, the current revision). If
    three.js ships a newer meshoptimizer simplifier, update its notice in
    `packages/island/world/IslandSimplify.ts` and its line in `CREDITS.md`.
