/*!
 * Adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater): its
 * volumetric clouds (`Clouds.js`, the view reconstruction) and its port of the Three.js Sky Pro
 * clouds (`SkyProClouds.js`, the density, lighting and march).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 *
 * The in-cloud lightning term (flashLight), the rain curtains and the rainbow drawn in them are
 * adapted from VolumetricCloudsCS1 by Anthony Ibrahim
 * (https://github.com/anthonytoumaibrahim/VolumetricCloudsCS1): `CloudRaymarch.shader`
 * (the point lights, MarchRain, BowColour) and `CloudMedia.cginc` / `CloudCommon.cginc`
 * (RainSigma, SampleRain).
 * SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Ibrahim
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { abs, dot, float, floor, length, max, mix, smoothstep, texture, vec2 } from 'three/tsl';
import type { Clouds } from './Clouds.ts';
import { sat, SHADOW_SIZE, SHADOW_SLOPE_FLOOR, SHADOW_SPAN } from './CloudModel.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

// The clouds' shadow as the lit surfaces and the air read it.
//
// The clouds' shadow as the lit surfaces and the air read it (Clouds.ts, whose header says how the
// map is cast and refreshed): the map's cubic lookups, the beam's sample and the fades toward lit,
// each a function of the clouds. A name in backticks with no file is this file's, Clouds.ts's or
// CloudModel.ts's.

/** Direct-light visibility under the clouds at a world position. */
export function shadow( clouds: Clouds, worldPosition: V3 ): F {

	return readShadow( clouds, worldPosition, 'r' );

}
/** The direct beam alone under the clouds at a world position, without the light the cloud
 * scatters on through (the map's blue channel), for the sea's glint: under a closed deck that
 * light is diffuse and makes no mirror image of the sun. Equal to `shadow` where no cloud lies
 * along the key. */
export function directShadow( clouds: Clouds, worldPosition: V3 ): F {

	return readShadow( clouds, worldPosition, 'b' );

}
export function readShadow( clouds: Clouds, worldPosition: V3, channel: 'r' | 'b' ): F {

	const q = worldPosition.xz.div( SHADOW_SPAN ).add( 0.5 ).toVar();
	// a cubic B-spline over the map's 47 m texels, from four bilinear taps (Sigg and Hadwiger,
	// "Fast Third-Order Texture Filtering", GPU Gems 2, ch. 20): the shadow's edges stay round
	// where a bilinear lookup draws straight segments and diamonds tens of metres long
	const c = q.mul( SHADOW_SIZE ).sub( 0.5 ).toVar();
	const i = floor( c ).toVar(), f = c.sub( i ).toVar();
	const f2 = f.mul( f ).toVar(), f3 = f2.mul( f ).toVar();
	const w0 = f.oneMinus().mul( f.oneMinus() ).mul( f.oneMinus() ).div( 6 );
	const w1 = f3.mul( 3 ).sub( f2.mul( 6 ) ).add( 4 ).div( 6 ).toVar();
	const w2 = f.mul( 3 ).add( f2.mul( 3 ) ).sub( f3.mul( 3 ) ).add( 1 ).div( 6 );
	const w3 = f3.div( 6 ).toVar();
	const g0 = w0.add( w1 ).toVar(), g1 = w2.add( w3 ).toVar();
	// the taps' texel centres, i + 0.5 moved by h0 = w1 / g0 - 1 and h1 = w3 / g1 + 1
	const p0 = i.add( 0.5 ).add( w1.div( g0 ).sub( 1 ) ).div( SHADOW_SIZE ).toVar();
	const p1 = i.add( 0.5 ).add( w3.div( max( g1, 1e-6 ) ).add( 1 ) ).div( SHADOW_SIZE ).toVar();
	const tap = ( x: F, y: F ): F => texture( clouds.shadowMap, vec2( x, y ) ).level( float( 0 ) )[ channel ];
	const s = mix(
		mix( tap( p0.x, p0.y ), tap( p1.x, p0.y ), g1.x ),
		mix( tap( p0.x, p1.y ), tap( p1.x, p1.y ), g1.x ),
		g1.y,
	);
	// The direct beam takes the full transmittance: the sky light already fills a cloud's shadow.
	// The map holds no data beyond its edge, so the shadow fades out toward it (shadowFadeAt).
	return mix( s, float( 1 ), shadowFadeAt( clouds, q, worldPosition ) );

}
/** The shadow map read for the point `p` of the air (the haze's shafts, VolumetricHaze.ts): the
 * map holds each ground point's flat beam toward `lightDir` (under a low light the beams about
 * that one, the air's channel narrower than the ground's from about 5 degrees up), its slope
 * clamped at SHADOW_SLOPE_FLOOR, so p is taken down that same beam to its ground point,
 * p.xz - L.xz p.y / max( L.y, floor ). Returns ( the green channel's bilinear sample, its
 * bicubic B-spline blur ), both faded toward lit as `shadow()`
 * fades them for p (the edge of the map, the distance and direction fades). The blur takes four
 * bilinear taps (Sigg and Hadwiger, "Fast Third-Order Texture Filtering", GPU Gems 2, 2005); the
 * B-spline smooths rather than interpolates (it does not pass through the texels' values), as a
 * blur for the unsharp mask should. */
export function shadowAlongBeam( clouds: Clouds, p: V3 ): V2 {

	const L = clouds.lightDir, shadowed = p;
	const ground = p.xz.sub( L.xz.mul( p.y.div( max( L.y, SHADOW_SLOPE_FLOOR ) ) ) );
	const q = ground.div( SHADOW_SPAN ).add( 0.5 ).toVar();
	const map = clouds.shadowMap;
	const point = texture( map, q ).level( float( 0 ) ).g;
	const coord = q.mul( SHADOW_SIZE ).sub( 0.5 ).toVar();
	const index = floor( coord ).toVar();
	const f = coord.sub( index ).toVar();
	const f2 = f.mul( f ), f3 = f2.mul( f );
	const w0 = f3.negate().add( f2.mul( 3 ) ).sub( f.mul( 3 ) ).add( 1 ).div( 6 );
	const w1 = f3.mul( 3 ).sub( f2.mul( 6 ) ).add( 4 ).div( 6 );
	const w2 = f3.mul( - 3 ).add( f2.mul( 3 ) ).add( f.mul( 3 ) ).add( 1 ).div( 6 );
	const w3 = f3.div( 6 );
	const g0 = w0.add( w1 ).toVar(), g1 = w2.add( w3 );
	const h0 = index.sub( 0.5 ).add( w1.div( g0 ) ).div( SHADOW_SIZE ).toVar();
	const h1 = index.add( 1.5 ).add( w3.div( g1 ) ).div( SHADOW_SIZE ).toVar();
	const tap = ( a: V2, b: V2 ) => texture( map, vec2( a.x, b.y ) ).level( float( 0 ) ).g;
	const blur = mix( mix( tap( h1, h1 ), tap( h0, h1 ), g0.x ), mix( tap( h1, h0 ), tap( h0, h0 ), g0.x ), g0.y );
	const lit = shadowFadeAt( clouds, q, shadowed );
	return mix( vec2( point, blur ), vec2( 1 ), lit );

}
// How far a shadow read at map coordinate q for the point `shadowed` fades toward lit: toward
// the map's edge, where it holds no data, and by the distance and direction fades (shadowFade).
export function shadowFadeAt( clouds: Clouds, q: V2, shadowed: V3 ): F {

	const edge = abs( q.sub( 0.5 ) );
	const outside = smoothstep( 0.42, 0.5, max( edge.x, edge.y ) );
	// Far and grazing shadows fade toward lit, so they do not dim the island wrongly: by the
	// squared horizontal distance from the island's centre, and by the squared distance of the
	// direction to the point (from the camera) from a chosen one (shadowFade, shadowFadeDirection).
	const f = clouds.shadowFade;
	const r = shadowed.xz.sub( clouds.shadowCentre );
	const toPoint = shadowed.sub( clouds.origin ).toVar();
	const d = toPoint.div( max( length( toPoint ), 1e-6 ) ).sub( clouds.shadowFadeDirection );
	const fade = max( sat( dot( r, r ).mul( f.x ).add( f.y ) ), sat( dot( d, d ).mul( f.z ).add( f.w ) ) );
	return max( outside, fade );

}
