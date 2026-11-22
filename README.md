# Lighthouse

A sea stack with a lighthouse, far out in deep water, and the remains of a ship the storm drove onto
it: its bow wedged against the rock, the rest of it afloat on the real waves. A fishing boat works
the water round the rocks and gulls wheel over it all. It is built on [Ponlai](https://github.com/HuberyLL/ponlai)'s physically based render
pipeline for three.js and WebGPU, with [Rapier](https://rapier.rs/) for the physics. (The waterfall
scene that began here is now a project of its own.)

- **The start screen** (`src/page.ts`, `index.html`) asks for the graphics before anything heavy
  loads: the quality tier and a 30 fps cap, set from the last visit or the device (a phone starts on
  Low at 30), with a warning where WebGPU is missing. The physics downloads meanwhile. The loading
  screen behind it is the first view drawn at dusk; its lamp lights and turns while the scene loads,
  under a line naming each stage, and it fades into the scene.

- **The sea** is Ponlai's FFT ocean. The scene describes the rocks' shorelines to it
  (`src/rocks.ts`, `rockFields`): the stack, two lesser stacks, a stump, a whaleback, skerries and a
  reef awash, so the waves break and foam against them and calm in their lee.
- **The lighthouse** (`src/lighthouse.ts`) lights up by night: the lamp, the windows, a local light on
  the gallery and two beams that sweep through the air, timed by the sky's own night share.
- **The wreck** (`src/wreckage.ts`) is built from code: the bow from a parametric hull (planking on
  ribs, a faded painted wale, copper sheathing, a jagged break and stove-in planks), and the debris
  from the same hull and from spars, torn sailcloth, rope, the ship's wheel, hatch gratings,
  splintered planks and the cargo's barrels.
- **The boat** (`src/boat.ts`) is a small double-ended fishing boat built from the wreck's parametric
  hull: painted topsides, boot top and antifouling, a planked deck, a wheelhouse, a mast with its
  boom and rigging, fish boxes, a heap of net, tyres for fenders. It floats on the same buoyancy as
  the wreckage, drag low along its keel and high across it, so it pitches and rolls on the swell,
  and its pilot follows a route round the stack planned clear of the rocks and the wreck
  (`plotRoute`). By night its wheelhouse glows and its navigation lights burn.
- **The gulls** (`src/gulls.ts`) are an instanced flock whose wings flap, glide and fold in the vertex
  stage. They wheel round the lantern and the stacks and follow the boat, keep above the rocks,
  and stand on the stacks' tops; at nightfall they all come down to roost until morning.
- **The sound** (`src/sound.ts`) starts with the Start button (or the first click or key press)
  quietly, creeps up while the scene loads and rises to the HUD's volume as it appears: the open sea, surf on each
  rock (positional, so it swells as you near it), wind that rises with the camera, the gulls calling
  by day (each call follows a gull in the scene), the boat's engine, the wreck's timbers creaking, and
  a splash where dropped wreckage lands. The clips in `public/audio` were generated
  locally with Stable Audio and cut into seamless loops with ffmpeg.
- **The wreckage** (`src/floaters.ts`) floats by buoyancy at sample points on each body, against the
  same surface the water draws, and collides with the rocks and the wedged bow. The waves exist only on the GPU, so a compute pass
  (`src/waveProbe.ts`) samples the ocean's displacement and orbital velocity at those points and reads
  them back, one batch per frame.

## Run it

You need Node.js and a browser with WebGPU, such as a recent Chrome or Edge.

```bash
npm install
npm run dev
```

Open the address Vite prints and pick the graphics on the start screen (an address with `?quality=`
skips it; the HUD's Graphics link returns to it). Drag to orbit, scroll to zoom, click the sea to drop
wreckage. The buttons set the time of day; the slider sets the volume.

URL options: Ponlai's options work here (see Ponlai's
[URL options](https://github.com/HuberyLL/ponlai/blob/main/docs/reference/url-options.md)), for example
`?preset=midnight`, `?wind=strong` or `?paused&t=12`. The wind is `fresh` unless the query names a
wind or a weather. `?glow.beam=5e-6` (and `glow.lamp`, `glow.lens`, `glow.windows`) tunes the
night's glows.

In the browser console, `__LIGHTHOUSE__.floaters.stats()` lists every body's height against the
water under it (the boat is the `vessel`); `__LIGHTHOUSE__.boat`, `.route` and `.gulls` hold
the rest of the scene's life.

## Layout

| Path | What it holds |
| --- | --- |
| `src/scene.ts` | The scene: assembles the stack, lighthouse and wreckage through Ponlai's `startScene` |
| `src/rocks.ts` | The rocks' geometry, material and shoreline fields |
| `src/sound.ts` | The soundscape: sea, surf, wind, gulls, engine, creaks, splashes (clips in `public/audio`) |
| `src/boat.ts` | The fishing boat: its geometry, materials and lights, its route and its pilot |
| `src/gulls.ts` | The gulls: the bird's mesh and material, the flock's flight and roosting |
| `src/main.ts`, `src/page.ts` | The entry, the start screen and the HUD |
| `src/audio.ts` | The audio rig: listener, compressor, volume and the quiet start |
| `src/lighthouse.ts` | The tower, lantern, cottage, lamp and beams |
| `src/wreckage.ts` | The wreck's geometry and materials: the hull, the bow, the loose parts |
| `src/floaters.ts` | The Rapier world, the floating wreckage and the boat's hull in the water |
| `src/waveProbe.ts` | The GPU wave sampler the physics reads |
| `src/surfaces.ts` | The material setup every surface shares |
| `src/render/`, `src/looks/ponlai/`, `tools/kit/` | Ponlai's render pipeline, look and tools, copied from Ponlai at `d5ad84b6` |

The captures and tools work as in Ponlai's blank project: `node tools/kit/capture.mjs --help`.

## Licence

Ponlai's code is MIT licensed ([LICENSE.ponlai](LICENSE.ponlai)); its third-party credits are in
Ponlai's [CREDITS.md](https://github.com/HuberyLL/ponlai/blob/main/CREDITS.md). The title's typeface,
IM Fell English SC by Igino Marini, is under the SIL Open Font License
([public/fonts/OFL.txt](public/fonts/OFL.txt)).
