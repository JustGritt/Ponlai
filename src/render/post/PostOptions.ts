import { DEFAULT_HAZE_GRID, type HazeGrid } from '../weather/VolumetricHaze.ts';
import { settings, type SettingsQuery } from '../Settings.ts';

// The frame graph's structural choices, read once when the pipeline builds its graph.
//
// The frame graph's structural choices (`PostOptions`), the quality tier's and the address's
// (`?post=`, declared in Settings.ts): the pipeline (Pipeline.ts) reads them once, when it builds
// its graph.

/** The frame graph's structural choices, fixed when the app loads (the quality tier sets them;
 * the defaults are today's high). `?post=legacy` restores the graph before they were introduced,
 * for A/B timing; `?post=` also takes a comma list of single switches (see `postOptionsFromURL`). */
export interface PostOptions {
	// the haze's froxel grid (VolumetricHaze.ts)
	hazeGrid: HazeGrid;
	// the composite evaluated once into a texture the bloom and the sharp image read (else each
	// re-evaluates it)
	compositeOnce: boolean;
	// FSR 1 from FSR.ts (EASU pass, sharpening folded into the canvas pass), or three's FSR1Node
	fsrPort: boolean;
	// the legacy attachment ops (`fgOps=legacy`) discard the scene target's samples in the rain;
	// false (`post=store-scene`) keeps them (the derived ops follow `fgOps` alone, see
	// AttachmentOps.ts: `post=store-scene` maps to its `store` mode)
	discardScene: boolean;
}

export const DEFAULT_POST_OPTIONS: Readonly<PostOptions> = {
	hazeGrid: DEFAULT_HAZE_GRID, compositeOnce: true, fsrPort: true, discardScene: true,
};

const LEGACY_SWITCHES: Record<string, Partial<PostOptions>> = {
	'inline-composite': { compositeOnce: false },
	'three-fsr': { fsrPort: false },
	'store-scene': { discardScene: false },
};

/** The defaults (with a tier's options over them), with `?post=legacy` (every switch below) or
 * `?post=a,b` applied (the app's options, or `params`). */
export function postOptionsFromURL( params: SettingsQuery = settings, base: Partial<PostOptions> = {} ): PostOptions {

	const options = { ...DEFAULT_POST_OPTIONS, ...base };
	for ( const name of ( params.get( 'post' ) ?? '' ).split( ',' ) ) Object.assign( options, ...( name === 'legacy' ? Object.values( LEGACY_SWITCHES ) : [ LEGACY_SWITCHES[ name ] ?? {} ] ) );
	return options;

}
