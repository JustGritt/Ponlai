/*!
 * Adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater), FoamTexture.js.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, float, vec2, vec4, int, ivec2, uint, instanceIndex, textureStore, fract, floor, sin, dot, min, max, length,
	smoothstep, mix, clamp,
} from 'three/tsl';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;

// Tileable procedural sea foam, generated once on the GPU (no external asset).
//
// Real foam comes in two forms. A fresh whitecap is an irregular bubbly mat: dense rafts with
// ragged edges, holes of every size and fine bubbles. As it ages the bubbles burst from the inside
// of the cells first, and what remains is a marbled lace: thin veins of foam along the borders of
// dark open-water cells, which the gale draws out into streaks. The water shader thresholds both
// against the local foam, so a crest's mat grows and tears, and a trail thins into lace and
// dissolves, instead of fading as a uniform white blot.
//   R: mat density (thresholded by fresh breaking)   G: lace, 1 on a vein (thresholded by old foam)
//   B: soft large-scale mottling                     A: streaks (stretched along the wind by the caller)
export function createFoamPattern( renderer: THREE.WebGPURenderer, size = 1024 ): THREE.StorageTexture {

	const tex = new THREE.StorageTexture( size, size );
	tex.type = THREE.HalfFloatType;
	tex.format = THREE.RGBAFormat;
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.magFilter = THREE.LinearFilter;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.generateMipmaps = true; // filled after the compute write (StorageTexture.mipmapsAutoUpdate)
	tex.anisotropy = 4;
	tex.name = 'foamPattern';

	const hash2 = ( p: V2 ) => fract( sin( vec2( dot( p, vec2( 127.1, 311.7 ) ), dot( p, vec2( 269.5, 183.3 ) ) ) ).mul( 43758.5453 ) );

	// periodic Worley: ( F1, F2 ), the distances to the nearest and second nearest feature point
	const worley2 = ( uv: V2, cells: number ): V2 => {

		const p = uv.mul( cells );
		const ip = floor( p ).toVar();
		const fp = fract( p ).toVar();
		const f1 = float( 8 ).toVar(), f2 = float( 8 ).toVar();
		for ( let j = - 1; j <= 1; j ++ ) for ( let i = - 1; i <= 1; i ++ ) {

			const o = vec2( i, j );
			const h = hash2( ip.add( o ).mod( cells ) );
			const d = length( o.add( h ).sub( fp ) ).toVar();
			f2.assign( max( min( f2, d ), f1 ) );
			f1.assign( min( f1, d ) );

		}
		return vec2( f1, f2 );

	};
	const worley = ( uv: V2, cells: number ): F => worley2( uv, cells ).x;

	const vnoise = ( uv: V2, cells: number ): F => {

		const p = uv.mul( cells );
		const i = floor( p ).toVar();
		const f = fract( p );
		const u = f.mul( f ).mul( float( 3 ).sub( f.mul( 2 ) ) ).toVar();
		const a = hash2( i.mod( cells ) ).x;
		const b = hash2( i.add( vec2( 1, 0 ) ).mod( cells ) ).x;
		const c = hash2( i.add( vec2( 0, 1 ) ).mod( cells ) ).x;
		const d = hash2( i.add( vec2( 1, 1 ) ).mod( cells ) ).x;
		return mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y );

	};

	const fbm = ( uv: V2, base: number, octaves: number ): F => {

		let s: F = float( 0 ), a = 0.5, n = 0;
		for ( let o = 0; o < octaves; o ++ ) {

			s = s.add( vnoise( uv, base * Math.pow( 2, o ) ).mul( a ) );
			n += a;
			a *= 0.5;

		}
		return s.div( n );

	};

	const kernel = Fn( () => {

		const n = uint( size );
		const px = ivec2( int( instanceIndex.mod( n ) ), int( instanceIndex.div( n ) ) );
		const uv = vec2( px ).add( 0.5 ).div( size ).toVar();

		// domain warp: organic, flowing shapes
		const w1 = vec2( fbm( uv, 3, 4 ), fbm( uv.add( 0.43 ), 3, 4 ) ).sub( 0.5 ).mul( 0.14 ).toVar();
		const wuv = uv.add( w1 ).toVar();

		// density: ragged rafts
		const dens = fbm( wuv, 4, 6 ).toVar();

		// holes of many sizes punched through the mat (Worley, radius varied by noise)
		const holeA = smoothstep( 0.28, 0.12, worley( wuv, 7 ).add( fbm( uv, 16, 3 ).sub( 0.5 ).mul( 0.25 ) ) );
		const holeB = smoothstep( 0.30, 0.16, worley( wuv.add( 0.17 ), 19 ).add( fbm( uv, 32, 2 ).sub( 0.5 ).mul( 0.3 ) ) );
		const holeC = smoothstep( 0.32, 0.18, worley( wuv.add( 0.61 ), 47 ) );
		const holes = clamp( holeA.mul( 0.9 ).add( holeB.mul( 0.7 ) ).add( holeC.mul( 0.45 ) ), 0, 1 );

		// fine bubbles: small bright dots
		const bubbles = smoothstep( 0.24, 0.08, worley( uv.add( 0.33 ), 140 ) ).mul( 0.8 )
			.add( smoothstep( 0.2, 0.05, worley( uv.add( 0.71 ), 260 ) ).mul( 0.5 ) ).toVar();

		// streaks, drawn out by the flow
		const streak = fbm( uv.add( w1.mul( 2 ) ), 12, 4 );

		const foam = clamp( dens.mul( 1.35 ).sub( holes.mul( 0.55 ) ).add( bubbles.mul( 0.08 ) ), 0, 1 );
		const mottle = fbm( uv, 2, 3 );

		// Lace: the borders between Worley cells (F2 - F1 is twice the distance to the nearest
		// border), two sizes of cell in a strongly warped domain so the veins meander like marbling.
		// The value falls off linearly from a vein, so a threshold sets the vein's width: young
		// foam keeps a wide web, the last of it a few hairlines. The veins' strength wanders along
		// them (and fine bubbles cling to them), so they break up rather than draw a closed net.
		const lw = vec2( fbm( uv.add( 0.11 ), 5, 4 ), fbm( uv.add( 0.67 ), 5, 4 ) ).sub( 0.5 ).mul( 0.2 );
		const luv = uv.add( lw ).add( w1 ).toVar();
		const cellsA = worley2( luv, 13 ).toVar(), cellsB = worley2( luv.add( 0.29 ), 31 ).toVar();
		const veinA = float( 1 ).sub( smoothstep( 0, 0.55, cellsA.y.sub( cellsA.x ) ) );
		const veinB = float( 1 ).sub( smoothstep( 0, 0.5, cellsB.y.sub( cellsB.x ) ) );
		const strength = fbm( uv.add( 0.53 ), 24, 3 ).toVar();
		const lace = clamp( max( veinA.mul( strength.mul( 0.5 ).add( 0.62 ) ), veinB.mul( strength.mul( 0.6 ).add( 0.42 ) ) )
			.add( bubbles.mul( 0.12 ) ), 0, 1 );
		textureStore( tex, px, vec4( foam, lace, mottle, streak ) );

	} )().compute( size * size, [ 64 ] ).setName( 'Ocean.FoamPattern' );
	renderer.compute( kernel );
	return tex;

}
