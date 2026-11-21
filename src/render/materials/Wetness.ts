/*!
 * The rain runnels on steep rock are adapted from tidewater by Daniel Greenheck
 * (https://github.com/dgreenheck/tidewater): the rain-streak stains of `terrainRockSurface` in
 * `src/world/terrain/TerrainShading.js`, ported from WGSL to TSL.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 *
 * The falling flow inside those runnels is adapted from Godot_rain_shader by Lexpartizan
 * (https://github.com/Lexpartizan/Godot_rain_shader): the scrolling streak gradient of `rain.shader`.
 * SPDX-FileCopyrightText: Copyright (c) 2020 Lexpartizan
 * SPDX-License-Identifier: MIT
 */

import type * as THREE from 'three/webgpu';
import { Fn, texture, float, vec2, vec3, abs, pow, max, mix, clamp, smoothstep } from 'three/tsl';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

// Wet materials: pores darken, a water film glosses, standing water fills the hollows.
//
// Wet materials after Sébastien Lagarde, "Water drop 3a / 3b - Physically based wet surfaces"
// (2013): water first fills the pores of a rough, porous surface (it darkens and loses a little
// roughness), then a film of water covers it (water's own smooth specular and a flatter normal),
// and finally standing water fills the hollows (see ../weather/Puddles.ts).
// Rules: wetness darkens exposed rock and plants within small limits, glosses bare stone,
// persists after rain and leaves the tidal band alone; dry weather renders exactly as without it.
// Water runs down the cliffs in sparse darker runnels while they are wet.

// Water's hemispherical reflectance for diffuse light (n = 1.33), and the internal reflectance of
// the same interface seen from below: everything beyond the critical angle returns to the ground.
const R_EXTERNAL = 0.066;
const R_INTERNAL = 1 - ( 1 - R_EXTERNAL ) / ( 1.33 * 1.33 );
// The trapping model below ignores the grains' own interface, which Lekner and Dorf include; their
// wet/dry ratio bottoms out near 0.68 for stone (as Lagarde 3b reports), against about 0.5 here. Scale toward
// that, which also keeps the island's wet stone within the art direction's small limits.
const DARKENING = 0.62;
/** Roughness of a water film over stone; standing water is smoother still. */
export const FILM_ROUGHNESS = 0.1;
/** Specular reflectance at normal incidence of a water surface (n = 1.33) and of the dry materials. */
export const WATER_F0 = 0.02, DRY_F0 = 0.04;

/** Porosity from roughness, as Lagarde 3b suggests without a porosity map: smooth surfaces hold
 * no water in their pores, the roughest are fully porous. */
export const wetPorosity = ( rough: F ) => clamp( rough.sub( 0.5 ).div( 0.4 ), 0, 1 );

/** Albedo under water held in the pores or as a film, `weight` 0 (dry) to 1 (saturated).
 * Light scattered back by the ground is partly totally internally reflected at the water's upper
 * surface and returned to be absorbed again (Lekner and Dorf 1988, cited by Lagarde 3a): with r_e and r_i
 * the external and internal diffuse reflectance, a film over albedo a reflects
 * a (1 - r_e)(1 - r_i) / (1 - a r_i). Darker channels darken most, so wet colour also deepens. */
export const wetAlbedo = Fn( ( [ albedo, weight ]: [ V3, F ] ) => {

	const trapped = vec3( ( 1 - R_EXTERNAL ) * ( 1 - R_INTERNAL ) ).div( vec3( 1 ).sub( albedo.mul( R_INTERNAL ) ) );
	return albedo.mul( mix( vec3( 1 ), trapped, weight.mul( DARKENING ) ) );

} ).setLayout( { name: 'ponlaiWetAlbedo', type: 'vec3', inputs: [ { name: 'albedo', type: 'vec3' }, { name: 'weight', type: 'float' } ] } );

/** Roughness once the pores have filled: Lagarde 3b, written for roughness rather than gloss.
 * Water in the pores fills the finest relief, the more so the more porous the surface. */
export const wetRoughness = ( rough: F, porosity: F, soak: F ) =>
	rough.mul( mix( float( 1 ), mix( float( 1 ), float( 0.2 ), porosity ), soak.mul( 0.5 ) ) );

export interface RunnelInput {

	tex: THREE.TextureNode; // low-frequency variation in alpha (the detail textures' gravel layer)
	p: V3; n: V3; // world position and geometry normal
	dpx: V3; dpy: V3; // screen derivatives of p, taken in uniform control flow by the caller
	warp: F; // slow variation (0 to 1, tens of metres) that shifts the columns along the cliffs
	time: F; rain: F;

}

// Runnels: a few tens of centimetres wide, drawn out 9 times down the face (tidewater's 3.1 x 41 m
// projection, scaled to this island's smaller cliffs and narrower water paths). The pattern
// repeats every 1.6 m across a face, so a slow warp shifts the columns to hide the period.
const RUN_U = 1 / 1.6, RUN_V = 1 / 14, RUN_WARP = 2.7;
// The streaks inside them: the same projection 3.2 times finer across and 1.6 times down.
const FINE = vec2( 3.2, 1.6 );
// The flow pattern is four times finer again down the streaks (slugs of water around half a
// metre long) and falls at FLOW_RATE / ( FLOW_V * FINE.y * RUN_V ), about 0.9 m/s.
const FLOW_V = 4, FLOW_RATE = 0.4;

/** Water running down steep rock while it is wet: x = runnel (the paths the water takes, which
 * stay damp), y = water flowing in them now (only while it rains). Samples use explicit
 * gradients, so the caller may branch around it. */
export function runnels( { tex, p, n, dpx, dpy, warp, time, rain }: RunnelInput ): V2 {

	// After tidewater: the variation stretched vertically on the two vertical projection planes,
	// weighted by how squarely each one faces the surface.
	const shift = warp.mul( RUN_WARP );
	const uvA = vec2( p.z.mul( RUN_U ).add( shift ), p.y.mul( RUN_V ) );
	const uvB = vec2( p.x.mul( RUN_U ).add( 0.5 ).sub( shift ), p.y.mul( RUN_V ).add( 0.3 ) );
	const gA = [ vec2( dpx.z.mul( RUN_U ), dpx.y.mul( RUN_V ) ), vec2( dpy.z.mul( RUN_U ), dpy.y.mul( RUN_V ) ) ];
	const gB = [ vec2( dpx.x.mul( RUN_U ), dpx.y.mul( RUN_V ) ), vec2( dpy.x.mul( RUN_U ), dpy.y.mul( RUN_V ) ) ];
	const w4 = pow( abs( n.xz ), vec2( 4 ) ).toVar();
	const wSum = w4.x.add( w4.y ).add( 1e-5 );
	const stainS = texture( tex, uvA ).grad( gA[ 0 ], gA[ 1 ] ).a.mul( w4.x )
		.add( texture( tex, uvB ).grad( gB[ 0 ], gB[ 1 ] ).a.mul( w4.y ) ).div( wSum );
	const steep = float( 1 ).sub( smoothstep( 0.55, 0.85, n.y ) );
	// Where the water gathers on a face: broad, soft-edged zones. On their own they read as damp
	// patches rather than as water running down the rock.
	const gather = smoothstep( 0.52, 0.62, stainS ).mul( steep ).toVar();
	// Inside them the water keeps to a few narrow paths: the same variation, about three times
	// finer across the face and a little finer down it, cut at a threshold into streaks a hand
	// or two wide, so each zone splits into separate dark lines with crisp edges.
	const fA = [ uvA.mul( FINE ).add( vec2( 0.23, 0.61 ) ), gA[ 0 ].mul( FINE ), gA[ 1 ].mul( FINE ) ];
	const fB = [ uvB.mul( FINE ).add( vec2( 0.71, 0.13 ) ), gB[ 0 ].mul( FINE ), gB[ 1 ].mul( FINE ) ];
	const fineS = texture( tex, fA[ 0 ] ).grad( fA[ 1 ], fA[ 2 ] ).a.mul( w4.x )
		.add( texture( tex, fB[ 0 ] ).grad( fB[ 1 ], fB[ 2 ] ).a.mul( w4.y ) ).div( wSum );
	const streak = smoothstep( 0.47, 0.55, fineS );
	const runnel = gather.mul( mix( float( 0.25 ), float( 1 ), streak ) ).toVar();

	// After Godot_rain_shader: the streaks sampled again with a falling offset; where the
	// moving pattern clears a threshold, a slug of water is running down the runnel.
	const fall = time.mul( FLOW_RATE );
	const flowA = texture( tex, fA[ 0 ].mul( vec2( 1, FLOW_V ) ).add( vec2( 0.17, fall ) ) )
		.grad( fA[ 1 ].mul( vec2( 1, FLOW_V ) ), fA[ 2 ].mul( vec2( 1, FLOW_V ) ) ).a;
	const flowB = texture( tex, fB[ 0 ].mul( vec2( 1, FLOW_V ) ).add( vec2( 0.41, fall ) ) )
		.grad( fB[ 1 ].mul( vec2( 1, FLOW_V ) ), fB[ 2 ].mul( vec2( 1, FLOW_V ) ) ).a;
	const flowS = flowA.mul( w4.x ).add( flowB.mul( w4.y ) ).div( wSum );
	const flowing = smoothstep( 0, 0.12, max( flowS.sub( 0.5 ), 0 ) ).mul( gather ).mul( streak ).mul( clamp( rain.mul( 2 ), 0, 1 ) );
	return vec2( runnel, flowing );

}
