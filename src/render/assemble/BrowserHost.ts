// A running scene's host in a browser page: the window, the document, the loading screen, the globals.
//
// A running scene's host (SceneHost.ts) in a browser page: the canvas in the element the page
// names, the window's size and pixel ratio and its resize events, the document's visibility, the
// page's local storage, `navigator.webdriver`, a loading screen element (a `span` in it takes the
// load's progress; the class `done` lifts it, with a CSS fade the page styles, which has ended once
// its computed opacity is 0), and the page's globals on `window`. It is the render pipeline's one
// module that reads the browser's globals for a scene (tools/repo/verify-layers.mjs, PAGE_NEEDS),
// behind its own entry point (render/browser.ts) that only the app layer imports.
import type { SceneHost, SceneStore } from './SceneHost.ts';

export interface BrowserHostOptions {
	/** the element the canvas goes in */
	canvasHost: HTMLElement;
	/** the loading screen, or null for none */
	loading?: HTMLElement | null;
	/** where the dynamic render scale remembers its scales: the page's local storage by default, null for nowhere */
	store?: SceneStore | null;
}

/** The host of a scene that runs in this browser page. */
export function browserHost( options: BrowserHostOptions ): SceneHost {

	const { canvasHost } = options;
	const loading = options.loading ?? null;
	let store: SceneStore | null = null;
	if ( options.store !== undefined ) store = options.store;
	else {

		// (a page whose storage is blocked keeps no scales)
		try {

			store = window.localStorage;

		} catch {

			store = null;

		}

	}
	const globals = window as unknown as Record<string, unknown>;
	return {
		canvasHost,
		size: () => ( { width: window.innerWidth, height: window.innerHeight } ),
		pixelRatio: () => window.devicePixelRatio,
		onResize( f ) {

			window.addEventListener( 'resize', f );
			return () => window.removeEventListener( 'resize', f );

		},
		onVisibility( f ) {

			const listener = () => f( document.hidden );
			document.addEventListener( 'visibilitychange', listener );
			return () => document.removeEventListener( 'visibilitychange', listener );

		},
		timerDocument: document,
		store,
		automated: navigator.webdriver === true,
		status( text ) {

			const line = loading?.querySelector( 'span' );
			if ( line ) line.textContent = text;

		},
		liftLoading( instant ) {

			if ( ! loading ) return;
			if ( instant ) loading.style.transition = 'none';
			loading.classList.add( 'done' );

		},
		loadingGone: () => loading === null || ( loading.classList.contains( 'done' ) && window.getComputedStyle( loading ).opacity === '0' ),
		publish( name, value ) {

			globals[ name ] = value;

		},
		hasGlobal: ( name ) => globals[ name ] != null,
	};

}
