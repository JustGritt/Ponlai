import * as THREE from 'three/webgpu';
import { float, texture3D } from 'three/tsl';

// The grade's look-up tables: two colour cubes after the tone curve, blended by a weight.
//
// The grade's look-up tables: colour cubes of LUT_SIZE^3 applied after the tone curve, two of them
// blended by a weight (Pipeline.ts `settings.lut`) so a look can move between two grades, by the
// time of day or by place. After Patry, "Real-Time Samurai Cinema" (SIGGRAPH 2021): 32^3 tables,
// used conservatively (a table that bends colours hard bands between its texels), the larger moves
// left to the white balance (and here the exposure). Their content is ours, tuned in our own scenes.
// - Encoding, our own choice: a table is indexed by q = sqrt( L ), L the curve's output (linear
//   display light, 0..1), and holds q too: a texel at q_in grades the linear colour q_in^2 and
//   stores the square root of the graded linear colour. Indexed by L itself, a table would spend
//   most of its cells on the highlights; the square root gives the darks more of them, spacing the
//   texels about as evenly in lightness as the sRGB transfer did, with no transfer of its own to
//   undo, so the output transfer (Pipeline.ts) is applied after the tables, once.
// - Half float, rounded to the nearest: the identity cube holds its values to 2^-12, so it returns
//   its input within a tenth of a display level (eight-bit texels would round to half a level).
// - Sampled at texel centres (the colour scaled by ( size - 1 ) / size and offset half a texel),
//   trilinear, at level 0: the rain layer grades inside branches, where an implicit level is not
//   allowed.
// - The page samples each table through one base node (`lutNodes`), so a table is swapped by
//   setting its `value` (a table of LUT_SIZE^3): new bind groups, no new pipeline; a table refilled
//   in place (`fillGradeLut`) is uploaded again.
//
// A table is generated from a grade's parameters (`GradeParams`) when the app loads, never stored
// as an image, so its parameters stay readable and later grades build on them (the day's, which the
// others build on: look/DisplayLook.ts `baseGrade`). On a display colour, in order, on the linear
// colour a texel stands for:
// - Its zone: shadows, midtones and highlights by its lightness (Oklab L of the linear colour, the
//   display's perceived lightness), the shadows fading out by `split[ 0 ]` and the highlights in from `split[ 1 ]` to
//   white, the midtones between (the shadows / midtones / highlights split of an editor's colour
//   wheels). Each zone brightens or darkens (`value`, stops), turns toward a tint (the colour its
//   white becomes, its luminance kept: as the white balance's, so two tints are a split toning) and
//   saturates, by its weight there.
// - A contrast curve on the lightness: a power curve each side of the pivot (the lightness the
//   exposed mid grey shows at), meeting with the slope `contrast` there and keeping black and white
//   where they are; the colour's chroma follows its lightness, so the curve leaves its saturation.
// - Saturation: the chroma times `saturation`, the zone's own, and the hue bands'.
// - Hue bands (grass, sky, sea, flowers, the warm hues, purple): a band about a centre hue (Oklab hue, degrees) of a
//   half-width, weighted by a raised cosine across it and faded out toward grey (low chroma, where
//   hue is noise) and, where it has one, above a chroma ceiling, scales the chroma, brightens or
//   darkens (stops) and turns the hue (degrees). The sea's colour is the water's own (Water.ts):
//   the grade only places it in the frame.
// Oklab (Björn Ottosson, "A perceptual color space for image processing", 2020): its lightness and
// hue keep a colour's hue as its chroma or lightness change, which linear RGB does not.

export const LUT_SIZE = 32;

type RGB = [ number, number, number ];

export interface GradeZone {
	// stops of brightness
	value: number;
	// the colour this zone's white becomes (linear Rec.709; its luminance kept)
	tint: RGB;
	// times the chroma
	saturation: number;
}

export interface GradeBand {
	// the band's centre and half-width (Oklab hue, degrees)
	hue: number; width: number;
	// times the chroma, stops of brightness, degrees of hue turned, at the band's centre
	saturation: number; value: number; shift: number;
	// the chroma (Oklab) the band fades out above, gone 0.03 higher (1: none, no display colour's
	// chroma reaches 0.33)
	ceiling: number;
}

export interface GradeParams {
	// the contrast curve's slope at the pivot
	contrast: number;
	// times the chroma
	saturation: number;
	// the lightness (Oklab L) where the shadows have faded out, and where the highlights begin
	split: [ number, number ];
	shadows: GradeZone; midtones: GradeZone; highlights: GradeZone;
	bands: { grass: GradeBand; sky: GradeBand; sea: GradeBand; flowers: GradeBand; warm: GradeBand; purple: GradeBand };
}

const zone = (): GradeZone => ( { value: 0, tint: [ 1, 1, 1 ], saturation: 1 } );
const band = ( hue: number, width: number, ceiling = 1 ): GradeBand => ( { hue, width, saturation: 1, value: 0, shift: 0, ceiling } );

/** The grade that changes nothing: its table is the identity. The bands sit where the kinds'
 * colours fall in the fair noon views (Oklab hue of their display colours, measured): the meadow
 * and the trees about 120 to 145 degrees, the sea 180 (the shallows) to 220 (the open sea toward
 * the horizon), the sky 230 (low) to 260 (high), the red flowers 20 to 30 when the meadow drew
 * them. The meadow draws no flowers now: the `flowers` band stays at their hue and grades whatever
 * red falls in it. Two bands are hues, not kinds: the warm ones from red to yellow (the low sun's
 * light, earth, bark), wide, and purple. A table grades colours, not things: the sky band's chroma
 * ceiling (between the noon sky's chroma, about 0.05 to 0.08, and the blue flowers' 0.12 to 0.14
 * when they were drawn) keeps a saturated blue that shares the sky's hue out of its band. */
export const NEUTRAL_GRADE: Readonly<GradeParams> = {
	contrast: 1, saturation: 1, split: [ 0.45, 0.7 ],
	shadows: zone(), midtones: zone(), highlights: zone(),
	bands: { grass: band( 132, 35 ), sky: band( 252, 30, 0.08 ), sea: band( 198, 28 ), flowers: band( 25, 35 ), warm: band( 70, 75 ), purple: band( 330, 35 ) },
};

/** A grade of `NEUTRAL_GRADE`'s parameters with `changes` over them (a deep copy). */
export function gradeOf( changes: GradeChanges, base: Readonly<GradeParams> = NEUTRAL_GRADE ): GradeParams {

	const g = structuredClone( base ) as GradeParams;
	mergeGrade( g, changes );
	return g;

}

/** Some of a grade's parameters, nested as the grade's own. */
export type GradeChanges = Partial<{ [ K in keyof GradeParams ]: GradeParams[ K ] extends readonly number[] ? GradeParams[ K ] : GradeParams[ K ] extends object ? Partial<{ [ L in keyof GradeParams[ K ] ]: GradeParams[ K ][ L ] extends readonly number[] ? GradeParams[ K ][ L ] : Partial<GradeParams[ K ][ L ]> }> : GradeParams[ K ] }>;

/** Sets `changes` over `grade` in place. */
export function mergeGrade( grade: GradeParams, changes: GradeChanges ): GradeParams {

	const into = ( target: Record<string, unknown>, from: Record<string, unknown> ) => {

		for ( const [ key, value ] of Object.entries( from ) ) {

			if ( value !== null && typeof value === 'object' && ! Array.isArray( value ) ) into( target[ key ] as Record<string, unknown>, value as Record<string, unknown> );
			else target[ key ] = Array.isArray( value ) ? [ ...value ] : value;

		}

	};
	into( grade as unknown as Record<string, unknown>, changes as Record<string, unknown> );
	return grade;

}

// a half float to the nearest (three's toHalfFloat drops the bits below the mantissa, up to a whole
// step low; the tables' values are all in 0..1, so the next code up is the next value)
const half = ( v: number ) => {

	const h = THREE.DataUtils.toHalfFloat( v );
	return Math.abs( THREE.DataUtils.fromHalfFloat( h + 1 ) - v ) < Math.abs( THREE.DataUtils.fromHalfFloat( h ) - v ) ? h + 1 : h;

};

/** The identity table: each texel holds its own colour. */
export function identityLut( size = LUT_SIZE ): THREE.Data3DTexture {

	const data = new Uint16Array( size * size * size * 4 );
	const one = THREE.DataUtils.toHalfFloat( 1 );
	for ( let b = 0, i = 0; b < size; b ++ ) for ( let g = 0; g < size; g ++ ) for ( let r = 0; r < size; r ++, i += 4 ) {

		data[ i ] = half( r / ( size - 1 ) );
		data[ i + 1 ] = half( g / ( size - 1 ) );
		data[ i + 2 ] = half( b / ( size - 1 ) );
		data[ i + 3 ] = one;

	}
	const lut = new THREE.Data3DTexture( data, size, size, size );
	lut.format = THREE.RGBAFormat;
	lut.type = THREE.HalfFloatType;
	lut.minFilter = lut.magFilter = THREE.LinearFilter;
	lut.wrapS = lut.wrapT = lut.wrapR = THREE.ClampToEdgeWrapping;
	lut.generateMipmaps = false;
	lut.colorSpace = THREE.NoColorSpace;
	lut.needsUpdate = true;
	return lut;

}

// the tables' encoding: the square root of linear display light, and back
const decode = ( q: number ) => q * q;
const encode = ( v: number ) => Math.sqrt( v );
const smoothstep = ( a: number, b: number, x: number ) => {

	const t = Math.min( Math.max( ( x - a ) / ( b - a ), 0 ), 1 );
	return t * t * ( 3 - 2 * t );

};
const luminance = ( c: RGB ) => 0.2126 * c[ 0 ] + 0.7152 * c[ 1 ] + 0.0722 * c[ 2 ];

// Oklab from and to linear Rec.709 (Ottosson's matrices)
function toOklab( r: number, g: number, b: number, out: RGB ): RGB {

	const l = Math.cbrt( 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b );
	const m = Math.cbrt( 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b );
	const s = Math.cbrt( 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b );
	out[ 0 ] = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
	out[ 1 ] = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
	out[ 2 ] = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
	return out;

}
function fromOklab( L: number, a: number, b: number, out: RGB ): RGB {

	const l = ( L + 0.3963377774 * a + 0.2158037573 * b ) ** 3;
	const m = ( L - 0.1055613458 * a - 0.0638541728 * b ) ** 3;
	const s = ( L - 0.0894841775 * a - 1.2914855480 * b ) ** 3;
	out[ 0 ] = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
	out[ 1 ] = - 1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
	out[ 2 ] = - 0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
	return out;

}

/** A table's colour ( 0..1, the square root of linear display light ) through the grade `p`, whose
 * contrast pivots at the lightness of the linear display value `grey` (the exposed mid grey
 * through the tone curve); returned in the same encoding. */
export function gradeColour( p: Readonly<GradeParams>, grey: number, c: Readonly<RGB>, out: RGB = [ 0, 0, 0 ] ): RGB {

	const lin: RGB = [ decode( c[ 0 ] ), decode( c[ 1 ] ), decode( c[ 2 ] ) ];
	const lab = toOklab( lin[ 0 ], lin[ 1 ], lin[ 2 ], [ 0, 0, 0 ] );
	// the zones' weights by the input's lightness
	const L0 = Math.max( lab[ 0 ], 0 );
	const ws = 1 - smoothstep( 0, p.split[ 0 ], L0 ), wh = smoothstep( p.split[ 1 ], 1, L0 ), wm = Math.max( 1 - ws - wh, 0 );
	const zones = [ p.shadows, p.midtones, p.highlights ], weights = [ ws, wm, wh ];
	let stops = 0, saturation = 0;
	const tint: RGB = [ 0, 0, 0 ];
	for ( let z = 0; z < 3; z ++ ) {

		const w = weights[ z ], Z = zones[ z ], k = 1 / luminance( Z.tint );
		stops += w * Z.value;
		saturation += w * Z.saturation;
		for ( let i = 0; i < 3; i ++ ) tint[ i ] += w * Z.tint[ i ] * k;

	}
	// (a colour no step changes is returned as it came: through Oklab and back it would move by a
	// rounding, and the neutral grade's table would not be the identity's)
	const near = ( x: number, to: number ) => Math.abs( x - to ) < 1e-9;
	const gain = 2 ** stops;
	let touched = ! near( gain, 1 ) || ! near( tint[ 0 ], 1 ) || ! near( tint[ 1 ], 1 ) || ! near( tint[ 2 ], 1 );
	if ( touched ) toOklab( lin[ 0 ] * gain * tint[ 0 ], lin[ 1 ] * gain * tint[ 1 ], lin[ 2 ] * gain * tint[ 2 ], lab );
	let [ L, a, b ] = lab;
	// the contrast curve on the lightness, the chroma following it
	if ( p.contrast !== 1 && L > 0 ) {

		touched = true;
		const q = Math.cbrt( grey ), k = p.contrast;
		const shown = Math.min( L < q ? q * ( L / q ) ** k : 1 - ( 1 - q ) * Math.max( ( 1 - L ) / ( 1 - q ), 0 ) ** k, 1 );
		a *= shown / L; b *= shown / L; L = shown;

	}
	// the hue bands
	const chroma = Math.hypot( a, b );
	let hue = Math.atan2( b, a ) * 180 / Math.PI, bandSaturation = 1, bandStops = 0, turn = 0;
	const coloured = smoothstep( 0.015, 0.05, chroma );
	for ( const B of Object.values( p.bands ) ) {

		const d = Math.abs( ( ( hue - B.hue ) % 360 + 540 ) % 360 - 180 );
		if ( d >= B.width ) continue;
		const w = ( 0.5 + 0.5 * Math.cos( Math.PI * d / B.width ) ) * coloured * ( 1 - smoothstep( B.ceiling, B.ceiling + 0.03, chroma ) );
		bandSaturation *= 1 + w * ( B.saturation - 1 );
		bandStops += w * B.value;
		turn += w * B.shift;

	}
	hue += turn;
	const scale = p.saturation * saturation * bandSaturation;
	if ( ! touched && near( scale, 1 ) && bandStops === 0 && turn === 0 ) {

		out[ 0 ] = c[ 0 ]; out[ 1 ] = c[ 1 ]; out[ 2 ] = c[ 2 ];
		return out;

	}
	const C = chroma * scale;
	if ( bandStops !== 0 ) L *= 2 ** ( bandStops / 3 );
	if ( turn !== 0 ) { a = C * Math.cos( hue * Math.PI / 180 ); b = C * Math.sin( hue * Math.PI / 180 ); } else if ( chroma > 0 ) { a *= C / chroma; b *= C / chroma; }
	fromOklab( L, a, b, out );
	for ( let i = 0; i < 3; i ++ ) out[ i ] = encode( Math.min( Math.max( out[ i ], 0 ), 1 ) );
	return out;

}

/** Fills `lut` (a table of LUT_SIZE^3, half float) with the grade `p` (see `gradeColour`) and marks
 * it for upload; a new table when none is given. */
export function fillGradeLut( p: Readonly<GradeParams>, grey: number, lut = identityLut() ): THREE.Data3DTexture {

	const n = LUT_SIZE, data = lut.image.data as Uint16Array;
	const c: RGB = [ 0, 0, 0 ], out: RGB = [ 0, 0, 0 ];
	for ( let bi = 0, i = 0; bi < n; bi ++ ) for ( let gi = 0; gi < n; gi ++ ) for ( let ri = 0; ri < n; ri ++, i += 4 ) {

		c[ 0 ] = ri / ( n - 1 ); c[ 1 ] = gi / ( n - 1 ); c[ 2 ] = bi / ( n - 1 );
		gradeColour( p, grey, c, out );
		data[ i ] = half( out[ 0 ] );
		data[ i + 1 ] = half( out[ 1 ] );
		data[ i + 2 ] = half( out[ 2 ] );

	}
	lut.needsUpdate = true;
	return lut;

}

/** The base nodes of the two tables, each the identity until a table is set as its `value`. */
export function lutNodes(): [ THREE.Texture3DNode, THREE.Texture3DNode ] {

	return [ texture3D( identityLut() ), texture3D( identityLut() ) ];

}

/** A table's colour q ( 0..1, the square root of linear display light: q = sqrt( L ) ) through a
 * table's base node, at texel centres and level 0; returned in the same encoding. */
export function sampleLut( lut: THREE.Texture3DNode, c: THREE.Node<'vec3'> ): THREE.Node<'vec3'> {

	const n = LUT_SIZE;
	return ( lut.sample( c.mul( ( n - 1 ) / n ).add( 0.5 / n ) ).level( float( 0 ) ) as unknown as THREE.Node<'vec4'> ).rgb;

}
