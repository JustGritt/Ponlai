// What a running scene takes from the page it runs in: the canvas's place, the window, the loading screen.
//
// What a running scene (StartScene.ts) takes from the page it runs in, handed over as one object:
// where the canvas goes, the window's size and pixel ratio and their changes, whether the page is
// hidden, the document a timer follows, the storage the dynamic render scale remembers its scales
// in, whether the browser is automated, the loading screen and the page's globals for the tools.
// The render pipeline reads no browser global itself (ARCHITECTURE.md, rule 13): a page in a
// browser gets this object from `browserHost` (BrowserHost.ts, behind render/browser.ts); a test or
// another host writes its own.

/** The storage the dynamic render scale remembers its scales in (post/DynamicResolution.ts). */
export interface SceneStore {
	getItem( key: string ): string | null;
	setItem( key: string, value: string ): void;
}

export interface SceneHost {
	/** the element the canvas is put in */
	readonly canvasHost: HTMLElement;
	/** the window's size in CSS pixels */
	size(): { width: number; height: number };
	/** the display's pixel ratio */
	pixelRatio(): number;
	/** calls `f` on each change of the window's size; returns the call that stops it */
	onResize( f: () => void ): () => void;
	/** calls `f` with whether the page is hidden on each change of it; returns the call that stops it */
	onVisibility( f: ( hidden: boolean ) => void ): () => void;
	/** the document three's Timer follows to restart its clock when the page shows again, or null */
	readonly timerDocument: Document | null;
	/** where the dynamic render scale remembers its scales, or null */
	readonly store: SceneStore | null;
	/** whether a tool drives the browser (the remembered scale is then off: post/DynamicResolution.ts) */
	readonly automated: boolean;
	/** shows a line of the load's progress on the loading screen */
	status( text: string ): void;
	/** lifts the loading screen, at once when `instant` (a paused page), or with its fade */
	liftLoading( instant: boolean ): void;
	/** whether the loading screen no longer covers the canvas: there is none, or its fade has ended */
	loadingGone(): boolean;
	/** puts `value` on the page as the global `name`, for the tools and the console (debug pages) */
	publish( name: string, value: unknown ): void;
	/** whether the page's global `name` is set (the clip tools' stepped clip: post/Pipeline.ts) */
	hasGlobal( name: string ): boolean;
	/** stops what the host started (its listeners stay with the caller of onResize and onVisibility) */
	dispose?(): void;
}
