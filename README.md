# Blank project

**English** | [简体中文](README.zh-CN.md)

A starting point for your own project on Ponlai's render pipeline. It has the render pipeline's sky,
sea, lighting and post-processing, a single rock in the open sea for you to replace with your own
content, and the tools that capture, compare and measure frames. To start a project, copy it as the
repository's [README](https://github.com/HuberyLL/ponlai#use-ponlai-in-your-own-project) shows.

## Run it

Run `npm install`, then `npm run dev`, and open the address Vite prints. The render pipeline's URL
options work here too, for example `?paused&t=12` or `?weather=fair&hour=7`.

To use the tools as well, install Playwright's browser and [uv](https://docs.astral.sh/uv/) (which
runs the Python tools), and commit the project to git at least once, because the tools read it:

```bash
npx playwright install chromium
curl -LsSf https://astral.sh/uv/install.sh | sh
git init && git add -A && git commit -m "Start from Ponlai's blank project"
```

For the agent skills, install Ponlai's Claude Code plugin (see
[Agent skills](https://github.com/HuberyLL/ponlai#agent-skills) in the repository's README).

## What's inside

| File | Purpose |
| --- | --- |
| `package.json` | three.js pinned to exactly 0.186.1 (the render pipeline uses three.js internals), plus Vite, TypeScript and Playwright |
| `vite.config.ts` | Builds the render pipeline's web workers as ES modules |
| `tsconfig.json` | Allows imports that end in `.ts`, which the render pipeline uses |
| `index.html` | The page: the canvas container and a loading screen |
| `src/main.ts` | Applies the page settings, then loads the scene (the render pipeline reads the settings when its modules load) |
| `src/scene.ts` | The scene: a rock, the ground and the water, passed to `startScene` |
| `src/render/` | The render pipeline |
| `src/looks/ponlai/` | Ponlai's look: the lighting, fog and post-processing settings the island uses |
| `public/` | Cloud noise textures, the star catalogue, the Milky Way and the moon (see Assets in [Credits](https://github.com/HuberyLL/ponlai/blob/main/CREDITS.md)) |
| `tools/kit/` | The tools (see [their README](https://github.com/HuberyLL/ponlai/blob/main/tools/kit/README.md)) |
| `tools/project.mjs` | The tools' settings for this project |

`startScene( host, options, content )` runs the render pipeline's stages in a fixed order and calls
your code at hooks between them: `passes`, `ground`, `ambient` and `water` provide the stages'
inputs, and the optional `after...` hooks add your own objects. For a scene without terrain of its
own, the neutral inputs (`neutralFields`, `neutralBounceGround`, `neutralSeaLattice`,
`neutralLattice`) describe a flat sea floor or flat ground. The camera (`render.camera`) has a
vertical field of view of 27 degrees; position it in `afterSky`. The hooks and options are
documented in `packages/render/assemble/StartScene.ts`, and the test scenes in `app/scenes/` are
fuller examples. To load the assets from another URL, pass `assetBase` to `startScene`.

## Choose a look

`LOOK` at the top of `src/scene.ts` picks one of two starting looks:

- `PONLAI_LOOK` (recommended, the default): Ponlai's own exposure, fog, ambient light, bloom and
  colour grading for each weather and time of day, as used on the island.
- `NEUTRAL_LOOK`: all gains at 1, with no fog, bloom or colour grading. A clean slate for your own
  look, built with the skills for building a look and fitting it to your reference images.

In both, the sky, the sun and the sea are physically based; the look only sets the values around
them.

## Capture and compare

Start the dev server through the tools: `serve.mjs` starts Vite in the background on the first free
port from 6400, and prints the address and the command to stop it. Then take two screenshots of the
paused scene and compare them:

```bash
node tools/kit/serve.mjs dev
node tools/kit/gpu-lock.mjs run --shared -- node tools/kit/capture.mjs first --url http://127.0.0.1:6400/
node tools/kit/gpu-lock.mjs run --shared -- node tools/kit/capture.mjs second --url http://127.0.0.1:6400/
uv run tools/kit/img.py diff artifacts/captures/first.png artifacts/captures/second.png
```

The two screenshots are identical, byte for byte. Run any tool with `--help` for its manual.

## Known limitations

- A page runs one render pipeline at a time. To show another scene, call `dispose()` on the running
  one first; the next `startScene` then renders exactly as on a fresh page load.
- Ponlai is tuned and tested only with the quality preset and weathers listed under
  [Status](https://github.com/HuberyLL/ponlai#status) in the repository's README.
- The render pipeline and the tools are not npm packages: you copy them in, and update them by
  copying them again.
