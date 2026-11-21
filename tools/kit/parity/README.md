# Parity captures

Parity captures check that a change leaves the rendered output unchanged. They are paused captures
of a fixed set of views, compared view by view with a reference set. The comparison is pixel for
pixel, or within the reference's own reproducibility. Each capture also has 1:1 crops of named
subjects, for judging a change that is meant to be visible. Clips do the same for motion. This page
holds the facts that span the parity tools. The method of verifying a change with them is the skill
[verifying-rendering-changes](../../../skills/verifying-rendering-changes/SKILL.md) and its
[binding](../../../docs/bindings/verifying-rendering-changes.md). Each tool's manual is its header
(`node <tool> --help`); the tools are listed in `docs/reference/tools.md`.

The tools:

- `tools/island/parity/views.mjs` and `tools/island/parity/clips.mjs`: the views, their crop
  anchors and the clips.
- `tools/kit/parity/capture-set.mjs`: captures a set of views; `tools/kit/diff-set.py` compares two
  sets.
- `tools/kit/reference.mjs`: the store of reference captures, one per commit.
- `tools/kit/parity/capture-clip.mjs` and `tools/kit/clip-metrics.py`: clips and their metrics
  against MSAA and a ground truth.
- `tools/kit/parity/baseline.mjs`, `tools/kit/parity/carry.mjs`, `tools/kit/gates.py` and
  `tools/kit/table.py`: the anti-aliasing gates' baseline, its carry-over, the verdicts and the
  tables.
- `tools/island/verify-scene-jump.mjs`: the views reached by scene jumps in one page, against fresh
  loads.

## Which comparison is byte-exact

MSAA sets are captured in the test mode ([Test mode](../../../docs/reference/test-mode.md)): a
paused page under MSAA, captured once `settled` is true. There, the exact rules apply. A paused MSAA
frame is bit-identical between page loads, so a reference set and a test set must match exactly
(`diff-set.py` with `--max-px1 0`). When both sets were captured with `--repeat`, they must match
within the reference's own A/A difference (`--aa-floor`). A view that is bit-identical across the
reference's captures must also be bit-identical on the test side and match the reference pixel for
pixel.

The sea's reflection is the one exception under MSAA. It averages a fixed sequence of frames and
holds the result (`packages/render/ocean/SeaReflection.ts`, Rest). A few of the opaque frames it
reads can differ in their last bits from one load to the next, so the pixels it reaches can differ
by one or two levels between loads. Each capture records those pixels in its JSON (`held`: cells of
the render, `packages/render/debug/HeldRegion.ts`). `diff-set.py --held` counts two captures as
equal in either of two cases: they are identical, or they are identical outside the union of both
sides' held regions and within the held-region tolerance inside it (`HELD` in
`tools/kit/tolerance.py`). `capture-set.mjs --check` and `verify-scene-jump.mjs --check` pass
`--held` for every MSAA comparison.

A scene jump keeps that contract. Only the load-time options need a new page (quality, aa, size and
device pixel ratio, the debug switches). The views' weather, time, moon and camera are runtime
state, and `__PONLAI__.establishScene( query )` changes them in place. After the jump and one
rendered frame, the paused frame is the one that a fresh load of the view's query draws. It matches
byte for byte wherever the fresh load is bit-identical across loads
(`packages/render/assemble/SceneJump.ts`; `ARCHITECTURE.md`, rule 4: a scene jump clears every
rendering history). `verify-scene-jump.mjs` checks this, and it is the regression check for any
change that adds state a jump must clear.

Temporal frames (`aa=taa`, `taau`, `fsr3`, `gt`) are not bit-identical, and we chose not to make
them so. The GPU ends a render pass on its own (tile overflow, preemption) at points that vary from
draw to draw. A split pass shades a few samples differently (mostly through fragment derivatives),
and the temporal rest averages many such frames and freezes what they held. Under a busy GPU, this
reaches a few hundredths of a percent of the pixels, a few levels each (up to 20 on dense
vegetation). Temporal sets are therefore compared within a tolerance (`diff-set.py --temporal`), the
way engine screenshot tests compare frames (Unity's Graphics Test Framework, Unreal's screenshot
tolerances).

## How a tolerance is set

A tolerance bounds what a frame's own nondeterminism can produce, and nothing more:

- A pixel is off when its largest channel difference exceeds a level. The share of off pixels is
  bounded on the frame, in every small window and on every named crop, and the frame's mean
  difference is bounded (global and local error).
- In every window, only a few pixels may differ by a lot (a missing or recoloured small object).
- The mean signed difference of each channel is bounded in every window and crop. A coherent shift
  that stays under the level everywhere (a 1-level change over a region) therefore still fails.
- One recapture before a failure counts: a view passes when any pairing of the two sides' captures
  passes. Capture temporal sets with `--repeat`.

The thresholds are calibrated on two kinds of evidence. First, they pass every A/A pair of one
build's captures, both quiet and with the GPU shared, with margin (the worst A/A pair uses under
half the tolerance). Second, they fail every synthetic regression cut into those frames (a 1-level
change over a small square, a recoloured flower patch, a quarter-pixel shift of a crop, a missing
small object) by several times the tolerance. A new tolerance must pass the same two tests. The
table and its calibration are in `tools/kit/tolerance.py`. A mean image metric over the frame
(FLIP's, for example) is only reported, never used as a gate: a mean over the frame hides a missing
flower entirely.

## How the reference library is used

References are kept per commit, in a store shared by every task and process that checks against
it (`tools/kit/reference.mjs`). A reference is captured once per commit and capture-tool version. A
task that runs the usual command (`capture-set.mjs --check`, `baseline.mjs`) reuses a stored
reference, or waits while another process captures it.

- A reference is captured only from the commit's own production build, served from its own
  worktree (`tools/kit/serve.mjs preview`).
- A recapture is forced in these cases: when the capture tools change (one reference round after a
  tool change lands); when a browser build changes and the stored samples no longer match; when a
  check needs repeated captures or views that the stored set lacks (only those are captured); and on
  request.
- A browser update is verified, not recaptured: a few stored views of each kind are captured with
  the new build, and the set is reused when they match.
- A commit can take over an earlier commit's MSAA baseline and ground truth without a recapture
  when nothing it changed can affect them (`tools/kit/parity/carry.mjs`). The rule cannot see a
  change that alters motion only, a change outside its watch list, a change in a view or clip that
  its spot check does not cover, or a change to the lower tiers' ground truth; its header lists
  these limits.
- Compare with `--aa-floor` or `--temporal` on sets captured with `--repeat`, so that a view that
  differs once is judged against its own A/A difference.

## Reproducibility observed

- MSAA views are bit-identical between page loads, with readiness judged by stability
  (`tools/kit/capture.mjs`), except where the sea's held reflection reaches. Temporal sets never
  are.
- The sea's held reflection: in 12 fresh loads of each of the sea's parity views, the views whose
  traced sea lies near the camera (island-fair-noon, island-clear-golden, island-night-clear,
  sunrise-long-shadows, meadow-close) are bit-identical. The others (horizon-fair-noon,
  water-skim-cliffs, island-from-sea, night-moon, low-36, rim-sea, cliff-sea-golden) show up to 7
  distinct frames. These differ inside the held region only, by 1 level on up to about 200 pixels
  and by 2 levels on one pixel.
- A clip of a slow pan over the meadow is bit-identical between fresh loads. Vegetation crops
  elsewhere can differ in single frames: with the vegetation's depth prepass on, a few samples pass
  its equal depth test differently from frame to frame (blobs of a few to 150 pixels on leaves and
  blades; none with `vegPrepass=0`).
