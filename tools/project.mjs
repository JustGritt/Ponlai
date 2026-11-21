// The project's values that the tools in tools/kit/ read: the page's globals, where the tools
// write, the names they share on the machine, the dev server and what a stored reference depends
// on. A kit tool imports this file and nothing else outside tools/kit/. Paths are written from the
// project's root. This project has no views' or clips' tables, no list of scene options and no
// pipeline tracer: a tool that needs one fails saying so, and the captures load a page per query.
export default Object.freeze( {

	// the globals the render pipeline mounts on a page with `debug` in its query (a dev server's page
	// has them always), and the element of the page's loading screen
	page: Object.freeze( {
		handle: '__PONLAI__',
		diagnostics: '__THREE_GAME_DIAGNOSTICS__',
		loading: 'loading',
		world: '__PONLAI_WORLD__',
		crops: '__PONLAI_CROPS__',
		gpuGate: '__PONLAI_GPU_GATE__',
		clip: '__PONLAI_CLIP__',
		clipNotes: '__PONLAI_CLIP_NOTES__',
	} ),

	// the title of the notices the tools post
	name: 'Blank project',

	// where the tools write, under the project's root; the project's .gitignore holds `root`
	output: Object.freeze( {
		root: 'artifacts',
		captures: 'captures',
		clips: 'clips',
		comparison: 'comparison',
		jobs: 'jobs',
		parity: 'parity',
		profiles: '.capture-profiles',
		storage: '.storage',
	} ),

	// the names the tools share on this machine. The GPU lock is Ponlai's own, so that this project
	// and a clone of Ponlai on one machine queue for the GPU together.
	machine: Object.freeze( {
		gpuLock: 'ponlai-gpu.lock',
		jobs: 'blank-jobs',
		tempPrefix: 'blank-',
		home: '.blank',
		envPrefix: 'BLANK_',
	} ),

	// the dev server (`npm run dev`) a page tool opens without --url, and the query of a capture
	// without --q: the scene paused at 12 seconds
	url: 'http://127.0.0.1:5173/',
	query: 'paused&t=12&dpr=1',

	reference: Object.freeze( {
		// the tools whose change can change a stored reference's pixels
		toolFiles: Object.freeze( {
			set: Object.freeze( [ 'tools/kit/capture.mjs', 'tools/kit/capture-browser.mjs', 'tools/kit/parity/capture-set.mjs', 'tools/kit/parity/adhoc.mjs' ] ),
			clip: Object.freeze( [ 'tools/kit/parity/capture-clip.mjs', 'tools/kit/capture-browser.mjs' ] ),
		} ),
		// what a production build reads
		buildInputs: Object.freeze( [ 'src', 'public', 'index.html', 'vite.config.ts', 'package.json', 'package-lock.json', 'tsconfig.json' ] ),
		// what can change a stored capture without touching a paused frame: what moves, and the
		// temporal resolve (a folder ends in '/')
		watch: Object.freeze( {
			motion: Object.freeze( [ 'src/', 'package.json', 'package-lock.json' ] ),
			gt: Object.freeze( [ 'src/render/post/', 'package.json', 'package-lock.json' ] ),
		} ),
	} ),

	tables: null,
	sceneOptions: null,
	pipelines: null,

	// what the disk clean ages out
	storage: Object.freeze( {
		folders: Object.freeze( [ 'captures', 'comparison', 'parity' ] ),
		temp: Object.freeze( { runs: Object.freeze( [] ), profiles: Object.freeze( [] ) } ),
	} ),

	// the servers tools/kit/serve.mjs starts
	serve: Object.freeze( {
		build: Object.freeze( [ 'npm', 'run', 'build' ] ),
		vite: Object.freeze( { dev: 'npx vite', preview: 'npx vite preview' } ),
		ports: Object.freeze( [ 6400, 6499 ] ),
		worktrees: '.worktrees',
		prepare: null,
	} ),

} );
