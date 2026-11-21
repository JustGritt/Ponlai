import * as THREE from 'three/webgpu';
import { Fn, vec3, dot, clamp } from 'three/tsl';
import { tuneUniform } from '../debug/Tune.ts';
import { markSkyRetune } from './SkyRetune.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

// The band-to-display matrix that turns per-band transmittances into display colours.
//
// The band-to-display matrix for transmittances computed at run time (the haze's, the sun disc's
// analytic one). Our transmittances are computed per band, one wavelength each for red, green and
// blue, and a band transmittance taken straight as a display colour reads too grey: a real
// spectrum's attenuation spreads further apart in a display's primaries than three samples of it
// do. The matrix pushes a chromatic transmittance away from grey and keeps grey grey:
// M = ( 1 + s ) I - s 1 w^T, w the Rec. 709 luminance weights (each row sums to 1, the diagonal
// above 1, the off-diagonals negative), so it keeps a transmittance's luminance and scales its
// chroma by 1 + s; the result is clamped to [0, 1]. A matrix derived from the CIE 1931 colour
// matching functions at the bands' wavelengths was tried and rejected: its red row is
// ill-conditioned (the blue band's wavelength has a positive red in linear sRGB) and it turned the
// haze's transmittance green and its veil magenta. One push `s` (`bandPush`) serves every reader.
// The haze's own keys (HazeKeys.ts sigma, desaturate) carry its colour, with the sky's and the
// haze's light set around them, so no push is left to give: s = 0 and the matrix is the identity,
// kept as a tune.

const LUMINANCE = [ 0.2126, 0.7152, 0.0722 ] as const;

/** The push from grey s of the band-to-display matrix, shared by every reader. */
export const bandPush = tuneUniform( 'sky.band.push', 0, { min: 0, max: 1.5, onChange: markSkyRetune } );

/** A band transmittance `t` as a display transmittance: clamp( M t, 0, 1 ) with push `s`. */
export const bandToDisplay = Fn( ( [ t, s ]: [ V3, F ] ) => {

	const y = dot( t, vec3( ...LUMINANCE ) );
	return clamp( t.mul( s.add( 1 ) ).sub( vec3( y.mul( s ) ) ), 0, 1 );

} ).setLayout( { name: 'bandToDisplay', type: 'vec3', inputs: [ { name: 't', type: 'vec3' }, { name: 's', type: 'float' } ] } );

/** `bandToDisplay` on the CPU, in place, with the shared push. */
export function bandToDisplayColor( t: THREE.Color, s = bandPush.value ): THREE.Color {

	const y = LUMINANCE[ 0 ] * t.r + LUMINANCE[ 1 ] * t.g + LUMINANCE[ 2 ] * t.b;
	const push = ( c: number ) => THREE.MathUtils.clamp( c * ( 1 + s ) - y * s, 0, 1 );
	return t.setRGB( push( t.r ), push( t.g ), push( t.b ) );

}
