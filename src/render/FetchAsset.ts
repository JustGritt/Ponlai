// The render pipeline's and the island's one fetch of the app's data files, and the address they are under.
//
// The render pipeline's and the island's one fetch of the app's assets: the code reads its data
// files (noise volumes, the star catalogue, scans) through `fetchAsset`, never `fetch` itself, so
// what the code takes from the network is in one place (tools/repo/verify-layers.mjs, the page
// check). Textures and models loaded by three's loaders go through their own requests, from the
// same base address (`assetUrl`).
// - The base address is the build's (`import.meta.env.BASE_URL`: Vite serves `public/` there)
//   unless the app set another before anything loads (`setAssetBase`; StartScene.ts `assetBase`).
//   One page holds one renderer, so the base is the module's own.

let base: string | null = null;

/** Sets the address the assets are served from, before anything fetches one (null: the build's). */
export function setAssetBase( address: string | null ): void {

	base = address;

}

/** An asset's address from its path under the assets' base. */
export function assetUrl( path: string ): string {

	return ( base ?? import.meta.env.BASE_URL ) + path;

}

/** Fetches an asset by its path under the address the assets are served from (`assetUrl`). */
export function fetchAsset( path: string ): Promise<Response> {

	return fetch( assetUrl( path ) );

}
