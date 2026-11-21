import { Fn, clamp, float, max, min } from 'three/tsl';
import type * as THREE from 'three/webgpu';

type F = THREE.Node<'float'>;

// The vegetation's level of detail by distance: one rule every plant kind thins by.
//
// The vegetation's level of detail by distance: one rule every plant kind thins by (today the
// meadow, Grass.ts; the plant kinds to come follow it).
//
// Rules:
// - Beyond a reference distance a record keeps the share ( reference / distance )^2 of the parts it
//   keeps there, so its parts per pixel of screen hold: a tuft twice as far covers a quarter of the
//   pixels and draws a quarter of the blades. The kept parts stand in for the dropped ones (the
//   meadow widens and covers with each kept blade by the inverse of its share), so the density,
//   colour and brightness of a far field hold while the vertex work follows the screen instead of
//   the area planted. The meadow's cost at the island view was its vertex stage (hiding the meadow
//   saved the same time at 1080p as at 1440p), so fewer parts is what saves there.
// - It is a function of the view camera's distance to the record alone: no hysteresis and no
//   state, so a paused frame does not depend on the camera's path, and a grown island draws as a
//   fresh load of it does. Other cameras (the sun's map, the rain's bake) take every part.
// - Within the reference distance the share is exactly 1, so near records are untouched to the
//   bit: a kind multiplies its own near curve by it.
// - Two ways to land it (the meadow takes the first): drop parts and widen the kept ones (a draw
//   range or the cull's bands of the first parts: the parts are generated in a random order, so any
//   first ones are a fair subset), or scale a record to nothing by the same share.

/** The share of its parts a record keeps at `distance` (m) from the view camera, beyond
 * `reference` where it keeps them all (1 within it). */
export function screenShare( distance: number, reference: number ): number {

	const r = reference / Math.max( distance, 1e-3 );
	return Math.min( r * r, 1 );

}

/** `screenShare` as a node, for the vertex stage and the cull's kernels alike. */
export const screenShareNode = Fn( ( [ distance, reference ]: [ F, F ] ) => {

	const r = reference.div( max( distance, float( 1e-3 ) ) );
	return min( r.mul( r ), float( 1 ) );

} ).setLayout( { name: 'vegetationScreenShare', type: 'float', inputs: [
	{ name: 'distance', type: 'float' }, { name: 'reference', type: 'float' },
] } );

// A plant's two levels (PlantInstances.ts): the near one, the authored mesh, and a far one with
// fewer sides, rings and leaf cards. The switch starts where the plant's stem is `pixels` render
// pixels wide on screen, so it follows the screen as the share above does, not a distance in
// metres; over `band` of that distance on, both levels draw, the near one dropping out and the far
// one coming in by the same fade (`levelFade`), cut by the stipple (Coverage.ts `stippleThreshold`).
// The render height is high's baseline (1440 pixels: Quality.ts), not the frame's: the dynamic
// resolution's steps would otherwise move every switch, and a paused frame renders at the baseline.
export const LEVEL_SWITCH = { pixels: 3, band: 0.15, renderHeight: 1440 } as const;

/** Metres of distance per metre of stem width at which a stem is LEVEL_SWITCH.pixels wide, for a
 * camera whose projection's y scale ( 1 / tan( fov / 2 ) ) is `projectionY`. */
export function switchScale( projectionY: number ): number {

	return LEVEL_SWITCH.renderHeight * projectionY / ( 2 * LEVEL_SWITCH.pixels );

}

/** The far level's share in the switch's band, 0 at `start` (m) and 1 at ( 1 + band ) start, as
 * v = saturate( a d^2 + b ): the band's ends in the screen share's square measure (`screenShare`),
 * so it eases in as the plant's area on screen shrinks. */
export const levelFade = Fn( ( [ distance, start ]: [ F, F ] ) => {

	const end = ( 1 + LEVEL_SWITCH.band ) ** 2 - 1;
	const r = distance.div( max( start, float( 1e-3 ) ) );
	return clamp( r.mul( r ).sub( 1 ).div( end ), 0, 1 );

} ).setLayout( { name: 'vegetationLevelFade', type: 'float', inputs: [
	{ name: 'distance', type: 'float' }, { name: 'start', type: 'float' },
] } );
