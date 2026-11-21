import { Matrix3, Vector3 } from 'three/webgpu';

// The display chain's colour spaces: the tone curve's working space and the white balance.
//
// The display chain's colour spaces (Pipeline.ts): the tone curve's working space and the white
// balance, as 3x3 matrices built here from chromaticities, so the white balance costs nothing per
// pixel beyond the two matrix products the curve takes anyway.
//
// The matrices are Stephen Hill's fit of ACES's RRT and ODT (three's acesFilmicToneMapping): it
// turns linear Rec.709 into AP1 (ACEScg) with the RRT's saturation, maps each channel through one
// curve (PipelineDisplay.ts toneCurve: at first Hill's, now an adjustable one fitted to it) and
// turns the result back through the ODT's saturation. The two saturations (about 0.89 together) are
// left at 1 now (CURVE_SATURATION): the curve no longer greys every colour alike, and the grade's
// hue bands (look/DisplayLook.ts `baseGrade`) set each hue's saturation instead. Per-channel tone
// mapping in the rendering space clips saturated colours toward cyan, magenta and yellow; AP1 rolls
// them off toward white, but bends reds and oranges toward yellow (a low sun's glints and sky).
// After Patry, "Real-Time Samurai Cinema" (SIGGRAPH 2021), the curve's space is AP1 with its red
// primary moved from x 0.713 to 0.75 (TONE_RED_X). Built with AP1's own primaries and Hill's
// saturations, the matrices are Hill's constants to their five decimals
// (`toneMatrices( { redX: AP1.r[ 0 ], ...HILL_SATURATION } )`).
//
// The white balance, after Patry too: the colour that white is to become (in linear Rec.709, as an
// artist picks it: bluer white for a bluer frame), reached by a von Kries adaptation in Bradford's
// cone space from D65 to that white's chromaticity, its luminance kept, and folded into the curve's
// input matrix. Bright colours still roll off to the display's white, since the output space stays
// Rec.709.

type XY = readonly [ number, number ];
interface Primaries { r: XY; g: XY; b: XY; white: XY }

const REC709: Primaries = { r: [ 0.64, 0.33 ], g: [ 0.30, 0.60 ], b: [ 0.15, 0.06 ], white: [ 0.3127, 0.3290 ] };
export const AP1: Primaries = { r: [ 0.713, 0.293 ], g: [ 0.165, 0.830 ], b: [ 0.128, 0.044 ], white: [ 0.32168, 0.33767 ] };
// The curve space's red primary (x; y stays AP1's): Patry's, which kept the low sun's reds from
// turning yellow without changing how yellows (a golden meadow) roll off.
export const TONE_RED_X = 0.75;
// Hill's fit: the RRT's and the ODT's saturation, in the curve's space.
export const HILL_SATURATION = { rrt: 0.96, odt: 0.93 };
// The curve's own: none.
const CURVE_SATURATION = { rrt: 1, odt: 1 };

// Bradford's cone response (Lam 1985), the space the white balance adapts in.
const BRADFORD = new Matrix3().set(
	0.8951, 0.2664, - 0.1614,
	- 0.7502, 1.7135, 0.0367,
	0.0389, - 0.0685, 1.0296,
);
const BRADFORD_INVERSE = BRADFORD.clone().invert();

const xyz = ( [ x, y ]: XY ) => new Vector3( x / y, 1, ( 1 - x - y ) / y );

// linear RGB of these primaries to XYZ, white at Y 1
function rgbToXYZ( p: Primaries ): Matrix3 {

	const r = xyz( p.r ), g = xyz( p.g ), b = xyz( p.b );
	const m = new Matrix3().set( r.x, g.x, b.x, r.y, g.y, b.y, r.z, g.z, b.z );
	const s = xyz( p.white ).applyMatrix3( m.clone().invert() );
	return m.multiply( new Matrix3().set( s.x, 0, 0, 0, s.y, 0, 0, 0, s.z ) );

}

// von Kries in Bradford's cone space: XYZ under one white to XYZ under another
function adaptation( from: Vector3, to: Vector3 ): Matrix3 {

	const a = from.clone().applyMatrix3( BRADFORD ), b = to.clone().applyMatrix3( BRADFORD );
	const scale = new Matrix3().set( b.x / a.x, 0, 0, 0, b.y / a.y, 0, 0, 0, b.z / a.z );
	return BRADFORD_INVERSE.clone().multiply( scale ).multiply( BRADFORD );

}

// a saturation matrix about the luminance of a space whose RGB to XYZ is `toXYZ`
function saturation( amount: number, toXYZ: Matrix3 ): Matrix3 {

	const e = toXYZ.elements, w = [ e[ 1 ], e[ 4 ], e[ 7 ] ], k = 1 - amount;
	return new Matrix3().set(
		k * w[ 0 ] + amount, k * w[ 1 ], k * w[ 2 ],
		k * w[ 0 ], k * w[ 1 ] + amount, k * w[ 2 ],
		k * w[ 0 ], k * w[ 1 ], k * w[ 2 ] + amount,
	);

}

const REC709_TO_XYZ = rgbToXYZ( REC709 );
const XYZ_TO_REC709 = REC709_TO_XYZ.clone().invert();

/** The tone curve's input matrix (linear Rec.709 to the curve's space, the RRT's saturation in it)
 * and output matrix (the ODT's saturation, back to linear Rec.709), for a curve space of AP1 with
 * its red primary at `redX`. */
export function toneMatrices( { redX = TONE_RED_X, rrt = CURVE_SATURATION.rrt, odt = CURVE_SATURATION.odt } = {} ): { input: Matrix3; output: Matrix3 } {

	const space: Primaries = { ...AP1, r: [ redX, AP1.r[ 1 ] ] };
	const toXYZ = rgbToXYZ( space ), fromXYZ = toXYZ.clone().invert();
	const toD60 = adaptation( xyz( REC709.white ), xyz( space.white ) );
	const toD65 = toD60.clone().invert();
	return {
		input: saturation( rrt, toXYZ ).multiply( fromXYZ ).multiply( toD60 ).multiply( REC709_TO_XYZ ),
		output: XYZ_TO_REC709.clone().multiply( toD65 ).multiply( toXYZ ).multiply( saturation( odt, toXYZ ) ),
	};

}

/** The white balance in linear Rec.709: what takes white (1, 1, 1) to `white`'s chromaticity at
 * luminance 1. */
export function whiteBalance( white: readonly [ number, number, number ] ): Matrix3 {

	const target = new Vector3( ...white ).applyMatrix3( REC709_TO_XYZ );
	target.divideScalar( target.y );
	const balance = adaptation( xyz( REC709.white ), target );
	return XYZ_TO_REC709.clone().multiply( balance ).multiply( REC709_TO_XYZ );

}
