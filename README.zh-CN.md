<!-- markdownlint-disable MD013 -->
# 空白项目

[English](README.md) | **简体中文**

基于 Ponlai 渲染管线开始你自己项目的起点。它包含渲染管线的天空、海洋、光照和后处理，大海中一块等你换成自己内容的礁石，以及用来截图、对比和测量画面的工具。复制它就能开始一个项目，做法见仓库 [README](https://github.com/HuberyLL/ponlai/blob/main/README.zh-CN.md#在你自己的项目中使用-ponlai)。

## 运行

先运行 `npm install`，再运行 `npm run dev`，然后打开 Vite 打印出的地址。渲染管线的 URL 选项在这里同样可用，比如 `?paused&t=12` 或 `?weather=fair&hour=7`。

如果还要用工具，需要安装 Playwright 的浏览器和 [uv](https://docs.astral.sh/uv/)（用来运行 Python 工具），并把项目提交到 git 至少一次，因为工具会读取它：

```bash
npx playwright install chromium
curl -LsSf https://astral.sh/uv/install.sh | sh
git init && git add -A && git commit -m "Start from Ponlai's blank project"
```

要使用智能体技能，请安装 Ponlai 的 Claude Code 插件（见仓库 README 的[智能体技能](https://github.com/HuberyLL/ponlai/blob/main/README.zh-CN.md#智能体技能)）。

## 项目内容

| 文件 | 用途 |
| --- | --- |
| `package.json` | 锁定为 0.186.1 的 three.js（渲染管线用到了 three.js 的内部实现），以及 Vite、TypeScript 和 Playwright |
| `vite.config.ts` | 把渲染管线的 Web Worker 构建为 ES 模块 |
| `tsconfig.json` | 允许以 `.ts` 结尾的导入，渲染管线用的就是这种写法 |
| `index.html` | 页面：画布容器和加载画面 |
| `src/main.ts` | 先应用页面设置，再加载场景（渲染管线的模块在加载时读取设置） |
| `src/scene.ts` | 场景：一块礁石、地面和水，交给 `startScene` |
| `src/render/` | 渲染管线 |
| `src/looks/ponlai/` | Ponlai 的画面风格：小岛使用的光照、雾和后处理参数 |
| `public/` | 云的噪声贴图、星表、银河和月亮（见[致谢](https://github.com/HuberyLL/ponlai/blob/main/CREDITS.md)的 Assets 一节） |
| `tools/kit/` | 工具（见[工具的 README](https://github.com/HuberyLL/ponlai/blob/main/tools/kit/README.md)，英文） |
| `tools/project.mjs` | 工具针对本项目的设置 |

`startScene( host, options, content )` 按固定顺序运行渲染管线的各个阶段，并在阶段之间的钩子里调用你的代码：`passes`、`ground`、`ambient` 和 `water` 为各阶段提供输入，可选的 `after...` 钩子用来加入你自己的物体。如果场景没有自己的地形，可以用中性输入（`neutralFields`、`neutralBounceGround`、`neutralSeaLattice`、`neutralLattice`）描述平坦的海底或地面。相机（`render.camera`）的垂直视角是 27 度，在 `afterSky` 里摆放它。钩子和选项的说明在 `packages/render/assemble/StartScene.ts`，`app/scenes/` 中的测试场景是更完整的例子。如果要从别的地址加载资源，给 `startScene` 传入 `assetBase`。

## 选择画面风格

`src/scene.ts` 顶部的 `LOOK` 决定从哪种画面风格开始：

- `PONLAI_LOOK`（推荐，默认）：Ponlai 自己的曝光、雾、环境光、泛光和调色参数，按天气和时段变化，和小岛用的一样。
- `NEUTRAL_LOOK`：所有增益为 1，没有雾、泛光和调色。适合从零做你自己的画面风格，可以配合"电影感风格化写实"和"按参考图调整画面"两个技能使用。

两种风格下，天空、太阳和海洋都是基于物理的；画面风格只调整它们之外的参数。

## 截图与对比

通过工具启动开发服务器：`serve.mjs` 会在后台从 6400 开始找第一个空闲端口启动 Vite，并打印地址和停止它的命令。然后对暂停的场景截两张图并对比：

```bash
node tools/kit/serve.mjs dev
node tools/kit/gpu-lock.mjs run --shared -- node tools/kit/capture.mjs first --url http://127.0.0.1:6400/
node tools/kit/gpu-lock.mjs run --shared -- node tools/kit/capture.mjs second --url http://127.0.0.1:6400/
uv run tools/kit/img.py diff artifacts/captures/first.png artifacts/captures/second.png
```

两张截图逐字节相同。任何工具加上 `--help` 都会打印它的说明。

## 已知限制

- 一个页面同时只能运行一条渲染管线。要切换到另一个场景，先对正在运行的场景调用 `dispose()`；之后的 `startScene` 会和刷新页面后的渲染结果完全一样。
- Ponlai 只在仓库 README [现状](https://github.com/HuberyLL/ponlai/blob/main/README.zh-CN.md#现状)一节列出的画质预设和天气下调校和测试过。
- 渲染管线和工具不是 npm 包：你把它们复制进来，更新时再复制一次。
