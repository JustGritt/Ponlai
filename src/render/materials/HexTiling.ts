/*!
 * The triangle grid, the hex tiles' centres and the luminance-weighted blend are adapted from
 * hextile-demo by Morten S. Mikkelsen (https://github.com/mmikk/hextile-demo, hextiling.h),
 * the code of "Practical Real-Time Hex-Tiling", JCGT 11(3), 2022.
 * SPDX-FileCopyrightText: Copyright (c) 2022 mmikk
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { Fn, If, texture, float, vec2, vec3, vec4, floor, fract, step, sin, cos, pow, max, mix, smoothstep, clamp, sqrt, select } from 'three/tsl';
import { hash22 } from './ProceduralTextures.ts';

// Hex-tiling: a texture read without a regular repeat, its grain sharp inside each tile.
//
// Hex-tiling (Mikkelsen 2022, after Heitz and Neyret's histogram-preserving tiling, HPG 2018):
// the plane is cut into a triangle grid whose vertices are the centres of hexagonal tiles. Each
// tile reads the texture with its own random offset and a small random rotation, and a point
// blends the three tiles of its triangle, weighted by its barycentric coordinates raised to a
// high power (so most of a tile reads one sample) and by each sample's height, so the higher
// relief wins at the seams instead of the two cross-fading into a blur. The texture never repeats
// on a regular lattice, while its own grain stays sharp inside each tile.

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// Mikkelsen's grid scale: at rate 1 the tiles' centres lie 1 / ( 2 sqrt 3 ) of the texture's
// period apart.
const GRID = 2 * Math.sqrt( 3 );

// One tile's read: the sample position (rotated about the tile's centre and shifted by a hash of
// its vertex) and the rotation's cosine and sine, for the gradients and for turning a sampled
// slope back into the surface's frame. The hash is Hoskins's, as for the procedural textures, in
// place of the demo's sine hash and integer angle formula.
export const hexTap = Fn( ( [ st, vertex, rate, maxAngle ]: [ V2, V2, F, F ] ) => {

	const cen = vec2( vertex.x.add( vertex.y.mul( 0.5 ) ), vertex.y.mul( 1 / 1.15470054 ) ).div( rate.mul( GRID ) );
	const offset = hash22( vertex );
	const angle = hash22( vertex.yx.add( 71.31 ) ).x.mul( maxAngle );
	const cs = cos( angle ), si = sin( angle );
	const d = st.sub( cen );
	const uv = vec2( d.x.mul( cs ).add( d.y.mul( si ) ), d.y.mul( cs ).sub( d.x.mul( si ) ) ).add( cen ).add( offset );
	return vec4( uv, cs, si );

} ).setLayout( {
	name: 'hexTap', type: 'vec4',
	inputs: [ { name: 'st', type: 'vec2' }, { name: 'vertex', type: 'vec2' }, { name: 'rate', type: 'float' }, { name: 'maxAngle', type: 'float' } ],
} );

export interface HexGrid { w: V3; v1: V2; v2: V2; v3: V2 }

// The triangle of the skewed grid that holds st: its three vertices (the tiles) and the point's
// barycentric weights for them.
export function hexGrid( st: V2, rate: number ): HexGrid {

	const p = st.mul( rate * GRID );
	const skewed = vec2( p.x.sub( p.y.mul( 0.57735027 ) ), p.y.mul( 1.15470054 ) );
	const base = floor( skewed ).toVar();
	const f = fract( skewed ).toVar();
	const z = float( 1 ).sub( f.x ).sub( f.y );
	const s = step( 0, z.negate() ).toVar();
	const s2 = s.mul( 2 ).sub( 1 );
	const w = vec3( z.negate().mul( s2 ), s.sub( f.y.mul( s2 ) ), s.sub( f.x.mul( s2 ) ) ).toVar();
	return {
		w,
		v1: base.add( vec2( s, s ) ).toVar(),
		v2: base.add( vec2( s, float( 1 ).sub( s ) ) ).toVar(),
		v3: base.add( vec2( float( 1 ).sub( s ), s ) ).toVar(),
	};

}

// The blend's sharpness: the barycentric weights' exponent and how far the heights move the seam
// (Mikkelsen's g_exp and g_fallOffContrast).
export const HEX_EXPONENT = 7, HEX_CONTRAST = 0.6;

// A tile whose weight is below this share of the strongest one's is not read: with the weights
// raised to the seventh power three points in five then take one read and the rest mostly two
// (1.4 reads on average, against 1.7 for a cut at 0.4 %). The share ramps over a factor of two,
// so a tile fades out smoothly at the cut rather than popping; the seams between tiles only
// sharpen a little.
const SKIP_FROM = 0.03, SKIP_TO = 0.06;

// The three tiles' base weights, those negligible set to zero.
export function hexBaseWeights( w: V3 ): V3 {

	const b = pow( w, vec3( HEX_EXPONENT ) ).toVar();
	const top = max( b.x, max( b.y, b.z ) );
	return b.mul( smoothstep( SKIP_FROM, SKIP_TO, b.div( top ) ) ).toVar();

}

// The final weights from the base weights and each tile's height in [0, 1].
export function hexWeights( base: V3, height: V3 ): V3 {

	const W = base.mul( mix( vec3( 1 ), height, HEX_CONTRAST ) );
	return W.div( max( W.x.add( W.y ).add( W.z ), 1e-6 ) ).toVar();

}

// Turn a tile's gradient (or texture-space vector) into the rotated sample's frame...
export const rotateInto = ( g: V2, tap: V4 ): V2 => vec2( g.x.mul( tap.z ).add( g.y.mul( tap.w ) ), g.y.mul( tap.z ).sub( g.x.mul( tap.w ) ) );
// ...and a slope read from the rotated sample back into the surface's frame.
export const rotateBack = ( d: V2, tap: V4 ): V2 => vec2( d.x.mul( tap.z ).sub( d.y.mul( tap.w ) ), d.x.mul( tap.w ).add( d.y.mul( tap.z ) ) );

// A scan's colour and normal map read through the hex tiles (Mikkelsen's hex2colTex and
// bumphex2derivNMap): each tile reads both with its own offset and rotation and with the
// gradients turned to match, so the mip level and anisotropy follow the tile, and the tiles'
// colour, slope (turned back into the surface's frame) and roughness (the normal map's blue: it
// holds the normal's x and y only) are summed with one set of weights, from the colour's height. Call it inside the branch that
// needs the scan; the reads are explicit-gradient, so they may sit in non-uniform control flow.
export interface HexScan {
	// texture arrays and the scan's layer in them
	colour: THREE.Texture; normal: THREE.Texture; layer: number;
	// the texture's scale over st, the grid's rate and the tiles' largest turn in radians
	scale: number; rate: number; maxAngle: number;
	// what a tile left unread holds (its weight is zero)
	colourRest: V4; normalRest: V4;
	height: ( rgb: V3 ) => F; slope: ( normal: V4 ) => V2;
}
export function hexReadScan( st: V2, gx: V2, gy: V2, scan: HexScan, colour: V4, slope: V2, roughness: F ): void {

	const grid = hexGrid( st, scan.rate );
	const b = hexBaseWeights( grid.w );
	const taps = [ grid.v1, grid.v2, grid.v3 ].map( ( v ) => hexTap( st, v, float( scan.rate ), float( scan.maxAngle ) ).toVar() );
	const cs = [ scan.colourRest.toVar(), scan.colourRest.toVar(), scan.colourRest.toVar() ];
	const fs = [ scan.normalRest.toVar(), scan.normalRest.toVar(), scan.normalRest.toVar() ];
	taps.forEach( ( tap, i ) => {

		If( [ b.x, b.y, b.z ][ i ].greaterThan( 0 ), () => {

			const q = tap.xy.mul( scan.scale );
			const qgx = rotateInto( gx, tap ).mul( scan.scale ), qgy = rotateInto( gy, tap ).mul( scan.scale );
			cs[ i ].assign( texture( scan.colour, q ).grad( qgx, qgy ).depth( float( scan.layer ) ) );
			fs[ i ].assign( texture( scan.normal, q ).grad( qgx, qgy ).depth( float( scan.layer ) ) );

		} );

	} );
	const W = hexWeights( b, vec3( ...cs.map( ( ci ) => scan.height( ci.rgb ) ) as [ F, F, F ] ) );
	colour.assign( cs[ 0 ].mul( W.x ).add( cs[ 1 ].mul( W.y ) ).add( cs[ 2 ].mul( W.z ) ) );
	slope.assign( rotateBack( scan.slope( fs[ 0 ] ), taps[ 0 ] ).mul( W.x )
		.add( rotateBack( scan.slope( fs[ 1 ] ), taps[ 1 ] ).mul( W.y ) )
		.add( rotateBack( scan.slope( fs[ 2 ] ), taps[ 2 ] ).mul( W.z ) ) );
	roughness.assign( fs[ 0 ].b.mul( W.x ).add( fs[ 1 ].b.mul( W.y ) ).add( fs[ 2 ].b.mul( W.z ) ) );

}

// Height-based layer blending (as in Mishkinis, "Advanced Terrain Texture Splatting", 2013, and
// Unreal's height lerp): a layer's coverage mask no longer fades in as a smooth gradient over
// the surface below but fills that surface's low points first, so its edge follows the stone's
// relief within a transition `width` wide. The height must be a rank, spread evenly over [0, 1]
// (the share of the surface that lies lower). The mask sets the threshold through the inverse of
// the mean cover of that ramp, so over the surface the layer keeps the mask's mean cover, and a
// mask of 0 or 1 stays bare or covered everywhere: the edge only gathers into the low points.
export const heightBlend = Fn( ( [ mask, height, width ]: [ F, F, F ] ) => {

	// The ramp clamp( ( a - height ) / width ) covers a - width / 2 of the surface on average while
	// it lies wholly inside [0, 1], and a quadratic share of it at either end. Written from its
	// upper end a, a mask of exactly 0 or 1 gives exactly 0 or 1.
	const m = clamp( mask, 0, 1 );
	const halfWidth = width.mul( 0.5 );
	const low = sqrt( width.mul( 2 ).mul( m ) );
	const high = float( 1 ).add( width ).sub( sqrt( width.mul( 2 ).mul( float( 1 ).sub( m ) ) ) );
	const a = select( m.lessThan( halfWidth ), low, select( m.greaterThan( float( 1 ).sub( halfWidth ) ), high, m.add( halfWidth ) ) );
	return clamp( a.sub( height ).div( width ), 0, 1 );

} ).setLayout( {
	name: 'terrainHeightBlend', type: 'float',
	inputs: [ { name: 'mask', type: 'float' }, { name: 'height', type: 'float' }, { name: 'width', type: 'float' } ],
} );
