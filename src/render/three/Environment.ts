import type { PMREMGenerator, RenderTarget, Texture } from 'three/webgpu';
import { missing } from './Internals.ts';

// PMREMGenerator's own steps, driven one at a time so a bake's GGX filters spread over frames.
//
// PMREMGenerator's own steps (three r186), which a whole `fromEquirectangular` takes in one call
// (its `_fromTexture`), driven one at a time so a periodic bake's GGX filter chain spreads over
// frames (sky/SkyBake.ts `environmentSlices`). Taken in its order and with its arguments, the atlas
// comes out the same.

type Steps = {
	_lodMeshes: unknown[];
	_setSizeFromTexture( texture: Texture ): void;
	_init( target: RenderTarget ): void;
	_textureToCubeUV( texture: Texture, target: RenderTarget ): void;
	_applyGGXFilter( target: RenderTarget, lodIn: number, lodOut: number ): void;
	_setViewport( target: RenderTarget, x: number, y: number, width: number, height: number ): void;
};

/** A PMREM generator's bake of an environment into a cube-UV atlas it made before, as steps. */
export interface EnvironmentSteps {
	/** The source's conversion into the atlas's first level (its size taken from the source). */
	convert( source: Texture, target: RenderTarget ): void;
	/** The atlas's level count, as the generator's last set-up left it. */
	levels(): number;
	/** Level `to`'s GGX filter from level `from` (two render passes). */
	filter( target: RenderTarget, from: number, to: number ): void;
	/** The viewport the generator leaves on `target` after a whole bake. */
	resetViewport( target: RenderTarget ): void;
}

/** `pmrem`'s steps; throws, naming the first one missing, on a three without them. */
export function environmentSteps( pmrem: PMREMGenerator ): EnvironmentSteps {

	const p = pmrem as unknown as Steps;
	const members: [ string, boolean ][] = [
		[ 'PMREMGenerator._lodMeshes', Array.isArray( p._lodMeshes ) ],
		[ 'PMREMGenerator._setSizeFromTexture', typeof p._setSizeFromTexture === 'function' ],
		[ 'PMREMGenerator._init', typeof p._init === 'function' ],
		[ 'PMREMGenerator._textureToCubeUV', typeof p._textureToCubeUV === 'function' ],
		[ 'PMREMGenerator._applyGGXFilter', typeof p._applyGGXFilter === 'function' ],
		[ 'PMREMGenerator._setViewport', typeof p._setViewport === 'function' ],
	];
	for ( const [ entry, holds ] of members ) if ( ! holds ) throw missing( entry, 'environmentSteps (packages/render/three/Environment.ts)' );
	return {
		convert( source, target ) {

			p._setSizeFromTexture( source );
			p._init( target );
			p._textureToCubeUV( source, target );

		},
		levels: () => p._lodMeshes.length,
		filter: ( target, from, to ) => p._applyGGXFilter( target, from, to ),
		resetViewport: ( target ) => p._setViewport( target, 0, 0, target.width, target.height ),
	};

}
