# Lighthouse

A sea stack with a lighthouse, far out in deep water, with wreckage floating round it on the real
waves. It is built on [Ponlai](https://github.com/HuberyLL/ponlai)'s physically based render
pipeline for three.js and WebGPU, with [Rapier](https://rapier.rs/) for the physics.

- **The sea** is Ponlai's FFT ocean. The scene describes the stack's shoreline to it
  (`src/stack.ts`, `stackFields`), so the waves break and foam against the rock and calm in its lee.
- **The lighthouse** (`src/lighthouse.ts`) lights up by night: the lamp, the windows, a local light on
  the gallery and two beams that sweep through the air, timed by the sky's own night share.
- **The wreckage** (`src/floaters.ts`) floats by buoyancy at sample points on each body, against the
  same surface the water draws. The waves exist only on the GPU, so a compute pass
  (`src/waveProbe.ts`) samples the ocean's displacement and orbital velocity at those points and reads
  them back, one batch per frame.

## Run it

You need Node.js and a browser with WebGPU, such as a recent Chrome or Edge.

```bash
npm install
npm run dev
```

Open the address Vite prints. Drag to orbit, scroll to zoom, click the sea to drop wreckage. The
buttons set the time of day.

URL options: Ponlai's options work here (see Ponlai's
[URL options](https://github.com/HuberyLL/ponlai/blob/main/docs/reference/url-options.md)), for example
`?preset=midnight`, `?wind=strong` or `?paused&t=12`. The wind is `fresh` unless the query names a
wind or a weather. `?glow.beam=5e-6` (and `glow.lamp`, `glow.lens`, `glow.windows`) tunes the
night's glows.

In the browser console, `__LIGHTHOUSE__.floaters.stats()` lists every body's height against the
water under it.

## Layout

| Path | What it holds |
| --- | --- |
| `src/scene.ts` | The scene: assembles the stack, lighthouse and wreckage through Ponlai's `startScene` |
| `src/stack.ts` | The sea stack's geometry, rock material and shoreline fields |
| `src/lighthouse.ts` | The tower, lantern, cottage, lamp and beams |
| `src/floaters.ts` | The Rapier world and the floating wreckage |
| `src/waveProbe.ts` | The GPU wave sampler the physics reads |
| `src/surfaces.ts` | The material setup every surface shares |
| `src/render/`, `src/looks/ponlai/`, `tools/kit/` | Ponlai's render pipeline, look and tools, copied from Ponlai at `d5ad84b6` |

The captures and tools work as in Ponlai's blank project: `node tools/kit/capture.mjs --help`.

## Licence

Ponlai's code is MIT licensed ([LICENSE.ponlai](LICENSE.ponlai)); its third-party credits are in
Ponlai's [CREDITS.md](https://github.com/HuberyLL/ponlai/blob/main/CREDITS.md).
