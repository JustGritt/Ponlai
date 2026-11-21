/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 *
 * Portions adapted from the AMD FidelityFX SDK 1.1.4, Stochastic Screen-Space Reflections
 * (https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, ffx_sssr_intersect.h).
 * SPDX-FileCopyrightText: Copyright (C) 2024 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * The sampling of Beckmann's visible normals adapted from pbrt-v3 (https://github.com/mmp/pbrt-v3,
 * src/core/microfacet.cpp, src/core/pbrt.h).
 * SPDX-FileCopyrightText: Copyright (c) 1998-2015, Matt Pharr, Greg Humphreys, and Wenzel Jakob
 * SPDX-License-Identifier: BSD-2-Clause
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Break, textureLoad, float, int, ivec2, vec2, vec3, vec4, mix, smoothstep, clamp, length, max, min, abs, select,
	sqrt, exp, log, exp2, floor, acos, sin, cos, normalize,
} from 'three/tsl';
import { cellIndex, NO_SURFACE, PYRAMID_LEVELS, SKY_Z, type DepthPyramid } from '../post/DepthPyramid.ts';

type F = THREE.Node<'float'>;
type I = THREE.Node<'int'>;
type IV2 = THREE.Node<'ivec2'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;
type M4 = THREE.Node<'mat4'>;

// One reflected ray of the sea, traced over the depth pyramid: what it meets of the opaque scene.
//
// What it reflects: the cliffs, the plants and the islanders, as the opaque pass drew them (its
// resolved colour and linear depth). The sea's reflection pass (SeaReflection.ts) samples one
// direction per pixel from the slopes the pixel does not resolve (`sampleBeckmannVisible`), traces
// it here, and denoises what the rays meet over the neighbourhood and the frames before.
//
// The trace (Uludag, "Hi-Z Screen-Space Cone-Traced Reflections", GPU Pro 5, 2014; the hierarchical
// march of AMD's FidelityFX Stochastic Screen-Space Reflections): the ray, from just off the water (a
// start that grows with the distance, tidewater's) to FARTHEST, in front of the camera and inside the
// box of what it can reflect, is projected to the screen and clipped to it, then walked cell by cell
// over the depth pyramid (post/DepthPyramid.ts), its view z at each cell's edges interpolated in
// perspective (McGuire & Mara, "Efficient GPU Screen-Space Ray Tracing", JCGT 3:4, 2014). A cell whose
// depth range (its nearest surface to its farthest one's back, THICKNESS behind it) the ray does not
// meet is stepped over and the trace climbs a level; a cell it meets, the trace moves up to where the
// ray enters that range and descends, down to the pixel the ray meets, which is the hit. Nothing is
// skipped, so a culm a few pixels wide is found, and a ray that passes behind a surface goes on past
// it. A ray that runs out of iterations short of its end is taken where it stopped, as FidelityFX
// takes it: the hit's check below keeps it only where it stopped on a surface.
//
// The hit: the opaque colour at the pixel the ray meets, one read at the colour's full resolution.
// One ray's colour is one direction's; the lobe of directions a pixel of a rippled sea reflects is
// the rays' spread over the pixels and the frames, which the denoiser gathers. (A read over the
// lobe's footprint on the screen, from one ray through the pixel's centre, mixed what lies at
// different depths inside the footprint: a hard line where a near cliff's edge crossed it, and a
// reflection that blurred more the higher it stood.) The hit's confidence is FidelityFX's check
// (`FFX_SSSR_ValidateHit`): none off the screen, on the sky, or within two pixels of where the ray
// started; the share of THICKNESS between the hit and the surface the depth holds there, squared, so
// that a ray passing behind a surface fades out with the depth it passes it by; and a fade over the
// screen's border (a twentieth of its height). On top of that, rays toward the camera fade out (what
// they meet is the back of things) and hits fade out from 120 m to FARTHEST.
//
// The plants (`plants`, the box SeaReflection.ts `setPlants` sets): the depth buffer holds only a
// grove's nearest culms and leaves, and a ray among them passes behind those by metres where it runs
// through the grove unseen, far more than a solid's THICKNESS. Taken as a miss, it fell back to the
// sky, so most of a grove's reflection was the sky's, the more so the steeper the ray. Inside the
// plants' box, what a ray crosses unseen is taken as a medium of the plants' mean free path: each ray
// draws how far it runs hidden before it meets a plant (`reach`, exponential of that mean, from the
// pixel's third blue-noise number), and a surface it passes behind by no more than that depth is met
// at full confidence, in the colour of the plant in front. Over the rays the denoiser gathers, the
// share that meets the plants after running a depth x hidden is the medium's, 1 - exp( -x / mean ).
// Outside the box, and where the drawn path is shorter, a surface is a solid's, THICKNESS deep.
//
// The box (Water.ts `reflectionBounds`), in two switches: cull, a ray that never meets the box within
// its stretch misses without tracing (most of the sea away from the island); clip, the trace covers
// only the stretch inside the box. The stretch before the box crosses the screen over the seabed and
// the shore, whose depths a ray can meet, and a hit there ends the ray on something it cannot reflect
// where the stretch in the box would find another: the image changes without it.
//
// The waterline: where the ray meets the cliff below the sea's reach, it could only have got there
// through the water it meets first. Falling back to the sky there (tidewater fades the lowest 0.4 m,
// sized for cliffs ten times taller) drew a bright band along every cliff foot. The sea at this
// island's cliffs is a near mirror in everyday weather, so the reflected rock continues down to the
// waterline, and rock just below it shows the rock where it leaves the water: the hit lifted to the
// waterline, if the depth buffer confirms rock there (a cliff face, not open seabed). A gale's swell
// heaves the surface at the rock by tens of centimetres; the reach grows with it and the lifted rock
// gives way to the sky, so a rough sea keeps its sky-lit facets. Hits deeper than 0.35 m under the
// sea level weigh nothing (Water.ts REFLECTION_FLOOR ends the box there).
//
// The camera is given (`ReflectionCamera`): the trace runs in a full-screen pass, whose own camera is
// not the scene's.

// The trace stops this far along the ray (m); its hits fade out from 120 m (tidewater's).
export const FARTHEST = 260;
// How far behind a surface the trace takes the surface to reach (m): a share of its distance, at
// least `least`. The depth buffer holds only the front of things; a ray that passes behind a surface
// by less than this meets it, by more goes on past it (among the plants, by their free path: the
// header, The plants). The share keeps a distant cliff, whose depth a pixel spans by metres, solid.
const THICKNESS = { least: 0.35, share: 0.02 } as const;
// FidelityFX's border fade: a twentieth of the screen's height on every side.
const BORDER = 0.05;

// The distances along a ray (origin, unit direction) at which it enters and leaves a box (slab
// method); enter > leave when it misses.
const rayBoxSpan = Fn( ( [ origin, direction, lo, hi ]: [ V3, V3, V3, V3 ] ) => {

	const safe = ( c: F ) => select( abs( c ).lessThan( 1e-6 ), select( c.lessThan( 0 ), float( - 1e-6 ), float( 1e-6 ) ), c );
	const inv = vec3( float( 1 ).div( safe( direction.x ) ), float( 1 ).div( safe( direction.y ) ), float( 1 ).div( safe( direction.z ) ) ).toVar();
	const a = lo.sub( origin ).mul( inv ).toVar(), b = hi.sub( origin ).mul( inv ).toVar();
	const enter = max( max( min( a.x, b.x ), min( a.y, b.y ) ), min( a.z, b.z ) );
	const leave = min( min( max( a.x, b.x ), max( a.y, b.y ) ), max( a.z, b.z ) );
	return vec2( enter, leave );

} ).setLayout( {
	name: 'rayBoxSpan', type: 'vec2',
	inputs: [ { name: 'origin', type: 'vec3' }, { name: 'direction', type: 'vec3' }, { name: 'lo', type: 'vec3' }, { name: 'hi', type: 'vec3' } ],
} );

// The error function and its inverse (pbrt-v3's approximations: Abramowitz & Stegun 7.1.26, and
// Giles, "Approximating the erfinv function", GPU Computing Gems Jade, 2011).
const erf = Fn( ( [ x ]: [ F ] ) => {

	const s = select( x.lessThan( 0 ), float( - 1 ), float( 1 ) );
	const a = abs( x ).toVar();
	const t = float( 1 ).div( a.mul( 0.3275911 ).add( 1 ) ).toVar();
	const poly = t.mul( 1.061405429 ).add( - 1.453152027 ).mul( t ).add( 1.421413741 ).mul( t ).add( - 0.284496736 ).mul( t ).add( 0.254829592 ).mul( t );
	return s.mul( float( 1 ).sub( poly.mul( exp( a.mul( a ).negate() ) ) ) );

} ).setLayout( { name: 'erfApprox', type: 'float', inputs: [ { name: 'x', type: 'float' } ] } );

const erfInv = Fn( ( [ x0 ]: [ F ] ) => {

	const x = clamp( x0, - 0.99999, 0.99999 ).toVar();
	const w = log( float( 1 ).sub( x ).mul( x.add( 1 ) ) ).negate().toVar();
	const p = float( 0 ).toVar();
	If( w.lessThan( 5 ), () => {

		const v = w.sub( 2.5 ).toVar();
		let q: F = float( 2.81022636e-08 );
		for ( const c of [ 3.43273939e-07, - 3.5233877e-06, - 4.39150654e-06, 0.00021858087, - 0.00125372503, - 0.00417768164, 0.246640727, 1.50140941 ] ) q = q.mul( v ).add( c );
		p.assign( q );

	} ).Else( () => {

		const v = sqrt( w ).sub( 3 ).toVar();
		let q: F = float( - 0.000200214257 );
		for ( const c of [ 0.000100950558, 0.00134934322, - 0.00367342844, 0.00573950773, - 0.0076224613, 0.00943887047, 1.00167406, 2.83297682 ] ) q = q.mul( v ).add( c );
		p.assign( q );

	} );
	return p.mul( x );

} ).setLayout( { name: 'erfInvApprox', type: 'float', inputs: [ { name: 'x0', type: 'float' } ] } );

/** A normal drawn from Beckmann's distribution of the visible normals (Heitz & d'Eon, "Importance
 * Sampling Microfacet-Based BSDFs using the Distribution of Visible Normals", EGSR 2014; pbrt-v3's
 * `BeckmannSample`), isotropic of roughness `alpha` (sqrt of the mean square slope over both axes),
 * seen from the unit direction `wi` in the surface's frame (its normal z), for the uniform pair `u`.
 * The sea's slopes are Gaussian (Cox & Munk), whose distribution is Beckmann's: GGX's long tail would
 * reflect far more of what lies well off the mirror direction. */
export const sampleBeckmannVisible = Fn( ( [ wi, alpha, u ]: [ V3, F, V2 ] ) => {

	// stretched to unit roughness
	const ws = normalize( vec3( wi.xy.mul( alpha ), max( wi.z, 1e-4 ) ) ).toVar();
	const cosTheta = ws.z.toVar();
	const sinTheta = sqrt( max( float( 1 ).sub( cosTheta.mul( cosTheta ) ), 0 ) ).toVar();
	const slope = vec2( 0 ).toVar();
	If( cosTheta.greaterThan( 0.9999 ), () => {

		// seen from straight above: the slopes' own Gaussian
		const r = sqrt( log( float( 1 ).sub( min( u.x, 0.999999 ) ) ).negate() );
		const phi = u.y.mul( 2 * Math.PI );
		slope.assign( vec2( cos( phi ), sin( phi ) ).mul( r ) );

	} ).Else( () => {

		// the slope along the view: the inverse of its cumulative distribution, by Newton's method
		// from a fitted guess, kept inside its bracket
		const tanTheta = sinTheta.div( cosTheta ).toVar();
		const cotTheta = float( 1 ).div( tanTheta ).toVar();
		const a = float( - 1 ).toVar();
		const c = erf( cotTheta ).toVar();
		const sx = max( u.x, 1e-6 ).toVar();
		const theta = acos( cosTheta ).toVar();
		const fit = theta.mul( theta.mul( theta.mul( - 0.0594 ).add( 0.4265 ) ).add( - 0.876 ) ).add( 1 );
		const b = c.sub( c.add( 1 ).mul( float( 1 ).sub( sx ).pow( fit ) ) ).toVar();
		const norm = float( 1 ).div( c.add( 1 ).add( tanTheta.mul( exp( cotTheta.mul( cotTheta ).negate() ) ).mul( 1 / Math.sqrt( Math.PI ) ) ) ).toVar();
		Loop( 9, () => {

			If( b.greaterThanEqual( a ).and( b.lessThanEqual( c ) ).not(), () => {

				b.assign( a.add( c ).mul( 0.5 ) );

			} );
			const inv = erfInv( b ).toVar();
			const value = norm.mul( b.add( 1 ).add( tanTheta.mul( exp( inv.mul( inv ).negate() ) ).mul( 1 / Math.sqrt( Math.PI ) ) ) ).sub( sx ).toVar();
			const derivative = norm.mul( float( 1 ).sub( inv.mul( tanTheta ) ) );
			If( abs( value ).lessThan( 1e-5 ), () => {

				Break();

			} );
			If( value.greaterThan( 0 ), () => {

				c.assign( b );

			} ).Else( () => {

				a.assign( b );

			} );
			b.subAssign( value.div( derivative ) );

		} );
		slope.assign( vec2( erfInv( b ), erfInv( max( u.y, 1e-6 ).mul( 2 ).sub( 1 ) ) ) );

	} );
	// turned to the view's azimuth, unstretched
	const cosPhi = select( sinTheta.lessThan( 1e-6 ), float( 1 ), clamp( ws.x.div( sinTheta ), - 1, 1 ) ).toVar();
	const sinPhi = select( sinTheta.lessThan( 1e-6 ), float( 0 ), clamp( ws.y.div( sinTheta ), - 1, 1 ) ).toVar();
	const turned = vec2( cosPhi.mul( slope.x ).sub( sinPhi.mul( slope.y ) ), sinPhi.mul( slope.x ).add( cosPhi.mul( slope.y ) ) ).mul( alpha );
	return normalize( vec3( turned.negate(), 1 ) );

} ).setLayout( { name: 'sampleBeckmannVisible', type: 'vec3', inputs: [ { name: 'wi', type: 'vec3' }, { name: 'alpha', type: 'float' }, { name: 'u', type: 'vec2' } ] } );

/** The scene camera, given to a full-screen pass: its view (world to view), world (view to world),
 * projection and inverse projection matrices. */
export interface ReflectionCamera {
	view: M4; world: M4; projection: M4; projectionInverse: M4;
}

export interface ScreenReflectionInputs {
	depth: THREE.TextureNode; // the opaque pass's linear depth: sample 0's view z (x)
	color: THREE.TextureNode; // its resolved colour, pre-exposed (read at level 0)
	pyramid: DepthPyramid; // the depth's pyramid
	iterations: number; // the most steps of the trace (WaterQuality `ssrIterations`), compiled in
	box: { min: V3; max: V3; cull: boolean; clip: boolean }; // what the sea can reflect (world space)
	plants: { min: V3; max: V3 }; // the plants' volume (world space; the header, The plants)
	waterLevel: F;
	gale: F; // the wind's gale share (shared.ts), which lowers the lifted rock at the waterline
	camera: ReflectionCamera;
	size: V2; // the render's size in pixels
}

/** What a ray meets: the colour (pre-exposed, as the opaque colour holds it), the confidence (a),
 * and the distance along the ray to where the trace ended (m). */
export interface ReflectionHit { colour: V4; length: F }

/** The trace, built into the calling shader: `trace( posV, Rv, reach )` for the view-space ray from
 * `posV` along unit `Rv`, which runs `reach` (m) hidden in the plants before it meets one (the
 * header, The plants). */
export function screenReflection( inputs: ScreenReflectionInputs ): ( posV: V3, Rv: V3, reach: F ) => ReflectionHit {

	const { depth, color, pyramid, iterations, box, plants, waterLevel, gale, camera, size } = inputs;
	const lastPixel = ivec2( size ).sub( int( 1 ) );
	const cells = pyramid.reader();
	const levels = pyramid.levels;
	// (a cell clamped to a level's: integer vectors, which three's typings of clamp do not take)
	const clampCell = ( cell: IV2, last: IV2 ) => clamp( cell as unknown as V2, ivec2( 0 ) as unknown as V2, last as unknown as V2 ) as unknown as IV2;
	const pixelOf = ( p: V2 ) => clampCell( ivec2( floor( p ) ), lastPixel );
	// sample 0's view z at a pixel
	const viewZAt = ( p: V2 ) => ( textureLoad( depth, pixelOf( p ) ) as unknown as V4 ).x;
	// the opaque colour at a pixel, at full resolution
	const colourAt = ( p: V2 ) => ( textureLoad( color, pixelOf( p ), int( 0 ) ) as unknown as V4 ).rgb;
	// a cell's depth range ( nearest, farthest ), level 0 the linear depth's sample 0
	const cellRange = Fn( ( [ level, cell ]: [ I, IV2 ] ) => {

		const range = vec2( 0 ).toVar();
		If( level.equal( int( 0 ) ), () => {

			const z = ( textureLoad( depth, clampCell( cell, lastPixel ) ) as unknown as V4 ).x.toVar();
			range.assign( vec2( z, select( z.lessThan( SKY_Z ), float( NO_SURFACE ), z ) ) );

		} ).Else( () => {

			const L = ( levels.element( level ) as unknown as V4 ).toVar();
			const c = clampCell( cell, ivec2( int( L.y ).sub( int( 1 ) ), int( L.z ).sub( int( 1 ) ) ) );
			range.assign( cells.element( cellIndex( L, c ) ) as unknown as V2 );

		} );
		return range;

	} ).setLayout( { name: 'cellRange', type: 'vec2', inputs: [ { name: 'level', type: 'int' }, { name: 'cell', type: 'ivec2' } ] } );
	// a clip-space point on the screen, in pixels (y down)
	const toPixels = ( h: V4, k: F ) => vec2( h.x.mul( k ).mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( h.y.mul( k ).mul( 0.5 ) ) ).mul( size );
	// a pixel's ray in view space, homogeneous (its direction is xyz, whatever w)
	const pixelRay = ( p: V2 ) => camera.projectionInverse.mul( vec4( p.x.div( size.x ).mul( 2 ).sub( 1 ), float( 1 ).sub( p.y.div( size.y ).mul( 2 ) ), 1, 1 ) );
	// how far behind a surface at view z `z` the trace takes it to reach
	const thicknessAt = ( z: F ) => max( abs( z ).mul( THICKNESS.share ), THICKNESS.least );

	// FidelityFX's check of a hit at pixel `p`, view-space point `hv`, from the ray's start at pixel
	// `start` (the header, The hit): the confidence the surface there is what the ray met; `reach`,
	// how far behind it the ray may run hidden among the plants and still meet them there
	const validate = ( p: V2, hv: V3, start: V2, reach: F ) => {

		const zs = viewZAt( p ).toVar();
		const r = pixelRay( p ).xyz.toVar();
		const distance = length( r.mul( zs.div( r.z ) ).sub( hv ) );
		const solid = float( 1 ).sub( smoothstep( 0, thicknessAt( zs ), distance ) ).toVar();
		const behind = zs.sub( hv.z ).toVar();
		const meets = max( solid.mul( solid ), select( behind.greaterThanEqual( 0 ).and( behind.lessThanEqual( reach ) ), float( 1 ), float( 0 ) ) );
		const q = p.div( size ).toVar();
		const fov = vec2( size.y.div( size.x ), 1 ).mul( BORDER ).toVar();
		const border = smoothstep( vec2( 0 ), fov, q ).mul( float( 1 ).sub( smoothstep( vec2( 1 ).sub( fov ), vec2( 1 ), q ) ) );
		const moved = abs( p.sub( start ) ).greaterThanEqual( vec2( 2 ) );
		const kept = moved.x.or( moved.y ).and( zs.greaterThanEqual( SKY_Z ) );
		return select( kept, meets.mul( border.x ).mul( border.y ), float( 0 ) );

	};

	return ( posV: V3, Rv: V3, reach: F ) => {

		const res = vec4( 0 ).toVar();
		const travelled = float( FARTHEST ).toVar();
		// (rays toward the camera fade out by this weight; those it zeroes are not traced)
		const directionWeight = smoothstep( 0.5, 0.1, Rv.z ).toVar();
		// the stretch of the ray traced (m along it): from just off the water, in front of the camera,
		// to FARTHEST
		const tIn = max( posV.z.negate().div( 60 ), 1 ).mul( 0.15 ).toVar();
		const tOut = float( FARTHEST ).toVar();
		If( Rv.z.greaterThan( 0 ), () => {

			tOut.assign( min( tOut, posV.z.negate().sub( 0.2 ).div( Rv.z ) ) );

		} );
		let traces = Rv.z.lessThanEqual( 0.6 ).and( directionWeight.greaterThan( 0 ) );
		// the ray in world space (the view is rigid, so distances along the ray are the same)
		const origin = camera.world.mul( vec4( posV, 1 ) ).xyz.toVar(), direction = camera.world.mul( vec4( Rv, 0 ) ).xyz.toVar();
		// where it runs among the plants, and how far it runs there hidden before it meets one (none
		// where it misses their volume)
		const amid = rayBoxSpan( origin, direction, plants.min, plants.max ).toVar();
		const hidden = select( amid.y.greaterThanEqual( amid.x ), reach, float( 0 ) ).toVar();
		const amidAt = ( t: F ) => select( t.greaterThanEqual( amid.x ).and( t.lessThanEqual( amid.y ) ), hidden, float( 0 ) );
		// where the ray is inside the box
		if ( box.cull || box.clip ) {

			const span = rayBoxSpan( origin, direction, box.min, box.max ).toVar();
			if ( box.cull ) traces = traces.and( span.y.greaterThanEqual( max( span.x, tIn ) ) ).and( span.x.lessThanEqual( tOut ) );
			if ( box.clip ) {

				tIn.assign( max( tIn, span.x ) );
				tOut.assign( min( tOut, span.y ) );

			}

		}
		If( traces.and( tOut.greaterThan( tIn ) ), () => {

			// the stretch's ends in view space, in clip space and on the screen
			const a = posV.add( Rv.mul( tIn ) ).toVar(), b = posV.add( Rv.mul( tOut ) ).toVar();
			const ha = camera.projection.mul( vec4( a, 1 ) ).toVar(), hb = camera.projection.mul( vec4( b, 1 ) ).toVar();
			const ka = float( 1 ).div( ha.w ).toVar(), kb = float( 1 ).div( hb.w ).toVar();
			const sa = toPixels( ha, ka ).toVar(), sb = toPixels( hb, kb ).toVar();
			const ho = camera.projection.mul( vec4( posV, 1 ) ).toVar();
			const start = toPixels( ho, float( 1 ).div( ho.w ) ).toVar();
			const d = sb.sub( sa ).toVar();
			// (a direction along a pixel axis divides by a tiny step, never by zero)
			const forward = vec2( select( d.x.greaterThanEqual( 0 ), float( 1 ), float( - 1 ) ), select( d.y.greaterThanEqual( 0 ), float( 1 ), float( - 1 ) ) ).toVar();
			const dSafe = max( abs( d ), vec2( 1e-5 ) ).mul( forward ).toVar();
			// the part of the stretch on the screen: s in [ s0, s1 ] along it
			const e0 = sa.negate().div( dSafe ).toVar(), e1 = size.sub( sa ).div( dSafe ).toVar();
			const s0 = max( max( min( e0.x, e1.x ), min( e0.y, e1.y ) ), 0 ).toVar();
			const s1 = min( min( max( e0.x, e1.x ), max( e0.y, e1.y ) ), 1 ).toVar();
			// the ray at screen parameter s: the share u of the stretch in view space (perspective-
			// correct); and the s at which it reaches view z `z`
			const along = ( s: F ) => s.mul( kb ).div( mix( ka, kb, s ) );
			const dz = b.z.sub( a.z ).toVar();
			const flat = abs( dz ).lessThan( 1e-6 ).toVar();
			const atZ = ( z: F ) => {

				const u = z.sub( a.z ).div( select( flat, float( 1 ), dz ) );
				return u.mul( ka ).div( kb.add( u.mul( ka.sub( kb ) ) ) );

			};
			const toward = vec2( max( forward, vec2( 0 ) ) ).toVar();
			const s = s0.toVar();
			const level = int( 0 ).toVar();
			const hit = float( 0 ).toVar();
			Loop( iterations, () => {

				If( s.greaterThanEqual( s1 ), () => {

					Break();

				} );
				const cellSize = exp2( float( level ) ).toVar();
				const cell = floor( sa.add( d.mul( s ) ).div( cellSize ) ).toVar();
				// where the ray leaves the cell (pushed a hundredth of a pixel into the next)
				const exits = cell.add( toward ).mul( cellSize ).add( forward.mul( 0.01 ) ).sub( sa ).div( dSafe );
				const sExit = min( min( exits.x, exits.y ), s1 ).toVar();
				const u0 = along( s ).toVar(), u1 = along( sExit ).toVar();
				const z0 = mix( a.z, b.z, u0 ).toVar(), z1 = mix( a.z, b.z, u1 ).toVar();
				const range = cellRange( level, ivec2( cell ) ).toVar();
				// (among the plants where the ray's stretch over the cell meets their span)
				const among = mix( tIn, tOut, u1 ).greaterThanEqual( amid.x ).and( mix( tIn, tOut, u0 ).lessThanEqual( amid.y ) );
				const back = range.y.sub( max( thicknessAt( range.y ), select( among, hidden, float( 0 ) ) ) ).toVar();
				const meets = min( z0, z1 ).lessThanEqual( range.x ).and( max( z0, z1 ).greaterThanEqual( back ) );
				If( meets.not(), () => {

					s.assign( sExit );
					level.assign( min( level.add( int( 1 ) ) as unknown as F, int( PYRAMID_LEVELS ) as unknown as F ) as unknown as I );

				} ).Else( () => {

					// on to where the ray enters the cell's depth range: its nearest surface going away
					// from the eye, the back of its farthest coming toward it
					const sIn = select( flat, s, atZ( select( z1.lessThan( z0 ), range.x, back ) ) );
					s.assign( clamp( sIn, s, sExit ) );
					If( level.equal( int( 0 ) ), () => {

						hit.assign( 1 );
						Break();

					} );
					level.subAssign( int( 1 ) );

				} );

			} );
			// A ray out of steps short of its end is taken where it stopped (the header, The trace).
			If( hit.lessThan( 0.5 ).and( s.lessThan( s1 ) ), () => {

				hit.assign( 1 );

			} );
			const u = along( s ).toVar();
			travelled.assign( mix( tIn, tOut, u ) );
			If( hit.greaterThan( 0.5 ), () => {

				const hv = mix( a, b, u ).toVar();
				const p = sa.add( d.mul( s ) ).toVar();
				const t = travelled;
				const hw = camera.world.mul( vec4( hv, 1 ) ).xyz.toVar();
				const base = directionWeight.mul( smoothstep( FARTHEST, 120, t ) ).toVar();
				const direct = base.mul( validate( p, hv, start, amidAt( t ) ) ).toVar();
				const seen = colourAt( p ).toVar();
				res.assign( vec4( seen, direct ) );
				// the waterline (the header): the hit lifted to the waterline, if rock is there
				const galeShare = clamp( gale, 0, 1 ).toVar();
				const waterline = waterLevel.add( mix( float( 0.03 ), float( 0.3 ), galeShare ) ).toVar();
				const above = smoothstep( waterLevel.sub( galeShare.mul( 0.1 ) ), waterline, hw.y ).toVar();
				If( above.lessThan( 1 ), () => {

					const lifted = camera.view.mul( vec4( hw.x, waterline, hw.z, 1 ) ).xyz.toVar();
					const hl = camera.projection.mul( vec4( lifted, 1 ) ).toVar();
					const pl = toPixels( hl, float( 1 ).div( hl.w ) ).toVar();
					const liftedTouch = float( 1 ).sub( smoothstep( 0.15, 0.6, abs( viewZAt( pl ).sub( lifted.z ) ) ) );
					const ql = pl.div( size );
					const onScreen = ql.x.greaterThanEqual( 0 ).and( ql.x.lessThanEqual( 1 ) ).and( ql.y.greaterThanEqual( 0 ) ).and( ql.y.lessThanEqual( 1 ) );
					// Deep hits (the seabed, rock seen past a downward-tilted facet) keep the sky.
					const shallow = smoothstep( waterLevel.sub( 0.35 ), waterLevel.sub( 0.15 ), hw.y ).mul( float( 1 ).sub( galeShare ) );
					// Weighted, so that a gale's sea (no lifted rock) blends as the direct hit alone.
					const kept = direct.mul( above ).toVar();
					const rock = select( onScreen, base.mul( liftedTouch ).mul( shallow ).mul( float( 1 ).sub( above ) ), float( 0 ) ).toVar();
					const weight = kept.add( rock ).toVar();
					res.assign( vec4( colourAt( pl ).mul( rock ).add( seen.mul( kept ) ).div( max( weight, 1e-4 ) ), weight ) );

				} );

			} );

		} );
		return { colour: res, length: travelled };

	};

}
