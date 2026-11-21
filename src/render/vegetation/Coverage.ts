import type * as THREE from 'three/webgpu';
import { Fn, float, max, fwidth, fract, diffuseColor, screenCoordinate, ivec2, int, vec2, dot } from 'three/tsl';
import { ALPHA_TO_COVERAGE, TEMPORAL } from '../post/Antialiasing.ts';
import { jitter } from '../post/Jitter.ts';
import { LEAF_ALPHA_TEST } from './LeafTexture.ts';

// How vegetation's cut-outs and blades cover a pixel by anti-aliasing mode, and its tag.
//
// How the vegetation's cut-outs and thin blades cover a pixel, by the anti-aliasing mode
// (post/Antialiasing.ts), and the tag that finds vegetation materials:
// - `ALPHA_TO_COVERAGE` picks the coverage path by the scene's samples; single-sampled, leaves take
//   a plain alpha test and blades a dither (`coverageDither`). Under the temporal modes the leaves'
//   test is a coverage band about the cutoff instead, dithered by the same threshold
//   (`coverageAlphaTest`, `useLeafCoverage`).
// - To find vegetation materials read `userData.vegetation` (`markVegetation`), never
//   `alphaToCoverage`: it is off without multisampling, and the depth prepass
//   (VegetationPrepass.ts) turns it off on the shaded draws. The terrain's meadow detail is flagged
//   `userData.meadowDetail` instead, so it never takes the vegetation's shadow filter.
// - A plant's two levels cross-fade by a stipple (`stippleThreshold`, PlantInstances.ts): each
//   pixel of a plant in the switch's band is drawn by one level or the other, never both and never
//   neither.

type F = THREE.Node<'float'>;

// Whether the vegetation's cut-outs and thin blades use alpha to coverage (the AA mode's plan,
// post/Antialiasing.ts: with 4 samples only). WebGPU multisamples with 4 samples or none (three
// maps any other count to one sample), and with one sample there is no coverage to give: the
// pipeline then ignores alpha, so the softened alpha-tested edges and the grass's lowered coverage
// would all draw solid. Without it leaves take a plain alpha test at the cutoff their
// coverage-preserving mips were built for (`leafAlphaTexture`), and grass blades an ordered 4x4
// dither (`coverageDither`); under the temporal modes the leaves take a dithered band about the
// cutoff (`coverageAlphaTest`).
export { ALPHA_TO_COVERAGE };

// Tags a material as vegetation (leaves and grass blades) for passes that treat it apart, such
// as a lighter shadow filter. `alphaToCoverage` cannot tell: it is off without multisampling.
// The tag lives in userData, so a cloned material keeps it.
export function markVegetation( material: THREE.Material ): void {

	material.userData.vegetation = true;

}

// The single-sample stand-in for alpha to coverage: a 4x4 Bayer threshold (16 levels, as many as
// the hardware's 4x pattern gives). Like that pattern it is the same for every blade and leaf over a
// pixel, so overlapping blades share their covered pixels rather than adding up: the soil and
// the flowers keep showing through a stack of widened blades as they do at 4x. A hash per blade
// (alphaHash) drew independent pixels for each and closed the meadow over them.
// Under the temporal modes the threshold steps through the pattern from frame to frame, the
// pixel's Bayer value plus 5 times the jitter's dither index (Jitter.ts), mod 16: over 16 frames
// every pixel takes every threshold once (5 is coprime to 16), so the resolve sees each blade's
// coverage rather than a fixed pattern of its pixels, while the blades over a pixel still share
// one threshold in each frame. A scene at rest takes the rest sequence's indices, so a paused
// frame does not depend on the frames before it.
export const coverageDither = Fn( () => {

	const p = ivec2( screenCoordinate.xy );
	const x = p.x.bitAnd( 3 ), y = p.y.bitAnd( 3 );
	const d = x.bitXor( y );
	const bayer = d.bitAnd( 1 ).shiftLeft( 3 ).bitOr( y.bitAnd( 1 ).shiftLeft( 2 ) ).bitOr( d.bitAnd( 2 ) ).bitOr( y.shiftRight( 1 ) );
	const index = TEMPORAL ? bayer.add( int( jitter.uniforms.ditherIndex ).mul( 5 ) ).bitAnd( 15 ) : bayer;
	return float( index ).add( 0.5 ).div( 16 );

} );

/** The temporal modes' leaf cut-out: a pixel keeps a share of an alpha-tested edge, as alpha to
 * coverage gives at 4x. The leaf's alpha crosses its cutoff over about one pixel (`fwidth`); the
 * share kept is ( alpha - cutoff ) / fwidth + 0.5, a band centred on the cutoff, and a pixel keeps
 * the leaf in a frame when the share reaches the frame's dither threshold (`coverageDither`), so the
 * resolve averages the edge's coverage instead of a hard step that crawls as the leaf moves across
 * pixels (a plain test at one sample a pixel). Centred on the cutoff, the band keeps the area the
 * coverage-preserving mips were built for (`leafAlphaTexture`). Returns the threshold for three's
 * alpha test, which discards alpha at or below it: alpha <= cutoff + ( threshold - 0.5 ) fwidth is
 * the band below the threshold. fwidth is taken at the top of the alpha test, before any branch of
 * three's (derivatives need uniform control flow). */
export function coverageAlphaTest( alpha: F, cutoff: F ): F {

	const width = max( fwidth( alpha ), 1e-4 );
	return cutoff.add( coverageDither().sub( 0.5 ).mul( width ) );

}

/** Under the temporal modes the leaf material's alpha test is the coverage band
 * (`coverageAlphaTest`), as its `alphaTestNode`: the depth prepass's copy takes it
 * (VegetationPrepass.ts) and the shaded draw leaves it out of its shader there, while the shadow
 * passes copy only the numeric `alphaTest` and keep their plain cut-out. 4x and one sample outside
 * the temporal modes keep theirs (alpha to coverage, the plain test). Set explicitly on every leaf
 * material: a leaf without a numeric `alphaTest` (`alphaTest: 'alphaOnly'` in Material.ts) would otherwise have no cut-out at all in its depth copy under the temporal modes. */
export function useLeafCoverage( material: THREE.NodeMaterial, cutoff = LEAF_ALPHA_TEST ): void {

	if ( TEMPORAL ) material.alphaTestNode = coverageAlphaTest( diffuseColor.a, float( cutoff ) );

}

/** The plants' level stipple (PlantInstances.ts): a threshold per pixel, 0..1, that the near level
 * keeps at or above the far level's share and the far level below it, so the two levels split each
 * pixel between them and their sum covers it once. Interleaved gradient noise (Jimenez, "Next
 * Generation Post Processing in Call of Duty: Advanced Warfare", SIGGRAPH 2014): its thresholds
 * spread evenly over every few pixels, so a half-faded plant reads as a fine grain rather than the
 * Bayer pattern's grid. Fixed per pixel in a paused page and at 4x (the test mode's image a function
 * of the view alone); under the temporal modes the pattern moves 5.588 pixels a frame by the
 * jitter's dither index, as Jimenez's does, so the resolve averages the fade (the rest sequence's
 * indices while the scene rests, as `coverageDither`). */
export const stippleThreshold = Fn( () => {

	const p = TEMPORAL ? screenCoordinate.xy.add( float( jitter.uniforms.ditherIndex ).mul( 5.588238 ) ) : screenCoordinate.xy;
	return fract( float( 52.9829189 ).mul( fract( dot( p, vec2( 0.06711056, 0.00583715 ) ) ) ) );

} );
