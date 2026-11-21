import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import { tuneKeys, tuneObject } from '../debug/Tune.ts';
import { keyed, type DayKeys } from '../post/DayKeys.ts';

// What a tree's leaves pass on of the key, as a look keyed by the sun and the weather (post/DayKeys.ts).
//
// `through`: the share of the key a leaf in a crown's shade still takes, the light of the leaves
// over it (lighting/SunShadow.ts, a receiver marked `userData.keyThrough`): the sun's map takes every
// leaf as black, and a crown whose shaded leaves took only the ambient read dark and as blue as the
// sky. It stands at its share from 12 degrees up. It is 0 at a sun of 3 degrees, where a crown's
// shaded side stands dark against the low light, and under a closed sky and by night, whose light
// comes from all round and has no beam for a leaf to pass on.
//
// `back`: what a leaf sends on toward the viewer of the key on its far face, by that face's cosine
// (vegetation/VegetationLighting.ts): a crown seen against the sun glows rather than turning black. One
// share at every height of the sun, and 0 under a closed sky and by night for the same reason as
// `through`, and so a clump's light by night stays on its near face.
//
// `leafTint`: what a closed sky's light turns into in a crown, as materials/SwardTint.ts is in the
// sward. Under a closed sky the ambient lights a crown, and what reaches the eye of it has passed
// through and off several leaves, each keeping green and taking red and more of the blue: read
// plainly off the albedo, a crown under cloud stood greyer than under the sun. So a leaf's ambient
// diffuse (vegetation/VegetationLighting.ts) is multiplied by this tint, about one more scatter off a
// green leaf: blue falls with red, so the hue holds where red alone would turn it toward cyan, and
// green rises just enough to keep the lightness. It takes the ambient and the local lights' diffuse gathered with it
// (lighting/GroundBounce.ts), not the key, so it weakens where a gap lets the key in. Weighted by the cloud and by the sun's height: whole from 12 degrees up, gone
// at 3, under which a clouded crown takes the low warm light's colour (with the tint it stood
// emerald on a brown dusk), and so gone by night too. Exactly white under a clear or fair sky; the
// metering illuminance (lighting/MeterLight.ts) leaves it out, so it moves no exposure. The falling
// leaves (world/FallingLeaves.ts) take it too.
//
// `leafWarm`: the colour a translucent leaf's key light takes in the leaf (vegetation/
// VegetationLighting.ts, on the thin layer's share). What a sunlit leaf sends back has mostly
// travelled through its tissue, once or more, so it carries the leaf's own colour again: more red
// and much less blue than its albedo alone gives ((1.25, 1, 0.7) of red, green and blue). Without it a sunlit crown
// read as cyan as its sky-lit side. It takes the key and the local lights, not the ambient, and the
// metering leaves it out, so it moves no exposure.
//
// Read once a frame into uniforms (main.ts), so it is a function of the clock and the weather alone
// and a paused frame repeats. Tunable on debug pages (debug/Tune.ts):
// `lighting.vegetation.leaf.through@golden`, `@low`, `@high`, `@cloudy`, `@night`, and `back` alike;
// `lighting.vegetation.leaf.tint.r`, `g`, `b`; `lighting.vegetation.leaf.warm.r`, `g`, `b`.
const LEAF_LIGHT: DayKeys<{ through: number; back: number }> = tuneKeys( 'lighting.vegetation.leaf', {
	day: [
		{ at: 3, name: 'golden', values: { through: 0, back: 0.8 } },
		{ at: 12, name: 'low', values: { through: 0.18, back: 0.8 } },
		{ at: 40, name: 'high', values: { through: 0.18, back: 0.8 } },
	],
	cloudy: { through: 0, back: 0 },
	night: { through: 0, back: 0 },
}, { through: { min: 0, max: 0.5 }, back: { min: 0, max: 1.5 } } );

const TINT = tuneObject( { r: 0.5, g: 1.12, b: 0.2 }, { r: 'lighting.vegetation.leaf.tint.r', g: 'lighting.vegetation.leaf.tint.g', b: 'lighting.vegetation.leaf.tint.b' }, {
	r: { min: 0.2, max: 1.2 }, g: { min: 0.8, max: 1.4 }, b: { min: 0.1, max: 1.4 },
} );

const WARM = tuneObject( { r: 1.25, g: 1, b: 0.7 }, { r: 'lighting.vegetation.leaf.warm.r', g: 'lighting.vegetation.leaf.warm.g', b: 'lighting.vegetation.leaf.warm.b' }, {
	r: { min: 0.5, max: 1.5 }, g: { min: 0.5, max: 1.5 }, b: { min: 0.2, max: 1.5 },
} );

/** What a translucent leaf's key light is multiplied by (`WARM`). */
export const leafWarm = uniform( new THREE.Vector3( 1, 1, 1 ) );

/** The share of the key a shaded leaf takes through the leaves over it (`LEAF_LIGHT`). */
export const leafThrough = uniform( 0 );
/** The share of the key on a leaf's far face that comes through it (`LEAF_LIGHT`). */
export const leafBack = uniform( 0 );

/** What a leaf's ambient diffuse is multiplied by (`TINT`); white under a clear or fair sky. */
export const leafTint = uniform( new THREE.Vector3( 1, 1, 1 ) );

const keys = { through: 0, back: 0 };

/** The leaves' light at the sun's `elevation` (degrees), the night's weight `night` and the
 * weather's cloud `clouded`. */
export function updateLeafLight( elevation: number, night: number, clouded: number ): void {

	keyed( LEAF_LIGHT, elevation, night, keys, clouded );
	leafThrough.value = keys.through;
	leafBack.value = keys.back;
	const weight = clouded * THREE.MathUtils.smoothstep( elevation, 3, 12 );
	leafTint.value.set( 1 + ( TINT.r - 1 ) * weight, 1 + ( TINT.g - 1 ) * weight, 1 + ( TINT.b - 1 ) * weight );
	leafWarm.value.set( WARM.r, WARM.g, WARM.b );

}
