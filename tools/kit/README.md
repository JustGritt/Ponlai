# Ponlai's tools

The tools in this folder capture screenshots of a paused page, compare them byte for byte, read
values from the running page and measure frame time. They work in any project that sets up its scene
with the render pipeline's `startScene`. A project started from [the blank
project](https://github.com/HuberyLL/ponlai/blob/main/templates/blank/README.md) already has them
set up; its Capture and compare section shows them in use.

## Add the tools to an existing project

1. Copy `tools/kit/` to `tools/kit/` in your project, and the blank project's `tools/project.mjs`
   to `tools/project.mjs`; the tools read nothing else from your project. Add `artifacts/`, where
   they write their output, to your `.gitignore`.
2. In your `tools/project.mjs`, set `name`, `url` (your dev server) and `reference.buildInputs` (the
   files your build reads).
3. Install Playwright at the version the tools are tested with, its browser, and
   [uv](https://docs.astral.sh/uv/), which runs the Python tools:

   ```bash
   npm install -D playwright@1.63.0
   npx playwright install chromium
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

4. Commit the project to git at least once: the dev server tool and the store of reference
   screenshots read the commit.

## Manuals and other platforms

Each tool's manual is in its header comment; print it with `node tools/kit/<tool>.mjs --help`.
[Tools](https://github.com/HuberyLL/ponlai/blob/main/docs/reference/tools.md) lists the tools by the
question each answers. The tools were written on macOS. On other systems, the CPU scheduling policy
(`qos.mjs`) does nothing, the GPU lock cannot see other programs' GPU use, and `display-gate.mjs`
shows no notifications. Set `CHROME_PATH` to capture with another Chromium build. The tools'
environment variables start with `machine.envPrefix` from `tools/project.mjs` (`PONLAI_` in this
repository, as in `PONLAI_GPU_SLOTS`).
