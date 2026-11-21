/*!
 * Adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { Fn, texture, uniform, float, vec2, vec4, dot, mix, smoothstep, clamp, max, abs, length } from 'three/tsl';
import { mulberry32 } from '../util/noise.ts';
import { REFERENCE_SPEED, gustAt } from '../weather/Wind.ts';
import type { Bounds } from '../RenderTypes.ts';
import type { Shared } from '../shared.ts';
import type { SeaLook } from '../look/Look.ts';

// Large-scale variation of the sea surface in world space: gusts and slicks.
//
// Large-scale variation of the sea surface in world space, which never repeats with the FFT's tiles
// (tidewater's SeaDetail, 4811ba4):
//  - gusts ("cat's paws"): patches of rougher water drifting downwind. Rough water reflects less of
//    the bright horizon sky, so from a low viewpoint gusts read as irregular dark patches, and the
//    sun's glitter path breaks into patches;
//  - slicks: long calm bands along the wind (surfactant films) where the capillary waves are damped,
//    mirror-like and bright, mostly in light to moderate wind;
//  - windrows: thin wavy lines along the wind in fresh wind (Langmuir circulation), mostly smooth
//    lanes with a trace of foam.
// `sample( xz )` gives ( rough, gust, slick, streak ): the multiplier of the short waves' slopes the
// pixel cannot resolve (and of the near-field ripples), and the three patterns in 0..1. Water.ts
// reads it at the surface's undisplaced point.
//
// A gust is one thing over the sea and the land: around the island the sea's gusts are the gusts
// that sweep the meadow (Wind.ts `gustAt`, a noise field swept with the wind's travel and reshaped
// in time), so a dark rough patch on the water upwind travels onto the island at the wind's speed,
// sweeps the grass at the same place and moment, and leaves the lee as a patch on the water again.
// The meadow's gust q (0..1) is mapped by LAND_LULL + LAND_SPAN x q, a mapping set for the
// earlier gust map (then 0 between gusts and 1 at a core, a third of the area in a gust: the same
// mean, 0.55, and spread, 0.26, as tidewater's, measured over both noise maps) and kept as it was.
// The gust field's repeat (Wind.ts GUST_TILE) would show toward the horizon, so it holds out to
// LAND_NEAR metres from the land's box and gives way to tidewater's two octaves by LAND_FAR
// (`setLand`, which Water.setLand calls, so it follows a grown island). The water reads the
// island's gust in its vertex stage (`landGust`: arithmetic, no texture) and hands it over in a
// varying; its grid's spacing grows to a few metres at LAND_FAR, finer than the gusts' 43 m
// cells. tidewater's octaves drift with the wind's integrated displacement (at the wind's speed;
// tidewater moved them at 0.7 times it), slower than the meadow's gusts (Wind.ts GUST_TRAVEL times
// the plants' speed), which run across the near water at the meadow's pace.
//
// Its inputs are the weather's wind (Wind.ts): the travel direction and the speed, the gust field's
// travel and time and the integrated displacement, so the pattern is a function of the wind's state
// and the scene clock and a paused frame is bit-identical. The noise is a tileable smooth fbm in four
// channels, made once on the CPU with fixed seeds (256 texels, half float, bilinear, repeating; low
// frequency, so no mipmaps). The gusts', slicks' and windrows' strengths are the look's
// (look/Look.ts `SeaLook` `detail`): the constructor takes them, as the render pipeline's stage
// hands them over (RenderStages.ts `createSeaAndPipeline`), so two made from two tables hold their
// own and this module holds none.

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;

const NOISE_SIZE = 256;
// the meadow's gust as the sea's (see the note at the top)
const LAND_LULL = 0.3;
const LAND_SPAN = 0.72;
/** tidewater's octaves drift with the wind's integrated displacement times this: at the wind's
 * speed (tidewater's own, 0.7, lagged behind it). */
export const OPEN_DRIFT = 1;
// metres from the land's box: the meadow's gusts alone within NEAR, tidewater's alone from FAR
const LAND_NEAR = 60;
const LAND_FAR = 220;

export class SeaDetail {

	readonly texture: THREE.DataTexture;
	readonly gustAmount: THREE.UniformNode<'float', number>;
	readonly slickAmount: THREE.UniformNode<'float', number>;
	readonly streakAmount: THREE.UniformNode<'float', number>;
	/** ( rough, gust, slick, streak ) at a world xz (explicit level: callable in branches), given
	 * the land's gust there (`landGust`; read here when left out). Inlined where it is called:
	 * as a layout function reading this module's uniforms and texture, r186 emitted it with the
	 * first material's binding names in the next one that called it. */
	readonly sample: ( xz: V2, landGust?: F ) => THREE.Node<'vec4'>;
	/** The meadow's gust at a world xz (Wind.ts `gustAt`): plain nodes, for a vertex stage. */
	readonly landGust: ( xz: V2 ) => F;
	/** The sea's gust at a world xz, 0..1, from the land's gust there, the wind's integrated
	 * displacement and the scene time (the wind probe evaluates it at other steps). */
	readonly gustOf: ( xz: V2, landGust: F, displacement: V2, time: F ) => F;
	// the land's box (setLand): its centre and half extents
	private landCentre: THREE.UniformNode<'vec2', THREE.Vector2>;
	private landHalf: THREE.UniformNode<'vec2', THREE.Vector2>;

	/** `detail`: the look's strengths of the gusts, the slicks and the windrows. */
	constructor( shared: Shared, detail: SeaLook[ 'detail' ] ) {

		this.gustAmount = uniform( detail.gust );
		this.slickAmount = uniform( detail.slick );
		this.streakAmount = uniform( detail.streak );
		this.landCentre = uniform( new THREE.Vector2() );
		this.landHalf = uniform( new THREE.Vector2() );
		this.texture = makeNoiseTexture( NOISE_SIZE );
		const noise = ( uv: V2 ) => texture( this.texture, uv ).level( float( 0 ) );
		const wind = shared.wind;
		const { gustAmount, slickAmount, streakAmount, landCentre, landHalf } = this;

		this.landGust = ( xz ) => gustAt( wind, xz );
		// tidewater's gusts before their contrast: two octaves (about 600 m and 230 m features), the
		// second slowly morphing; ( gust, the second octave )
		const openGust = ( xz: V2, displacement: V2, time: F ) => {

			const p = xz.sub( displacement.mul( OPEN_DRIFT ) ).toVar();
			const g1 = noise( p.div( 620 ) ).x;
			const g2 = noise( p.div( 230 ).add( vec2( time.mul( 0.0009 ), 0.37 ) ) ).y.toVar();
			return { gust: g1.mul( 0.62 ).add( g2.mul( 0.38 ) ).sub( 0.5 ).mul( 2.4 ).add( 0.5 ), g2 };

		};
		const blend = ( xz: V2, open: F, landGust: F ) => {

			const outside = length( max( abs( xz.sub( landCentre ) ).sub( landHalf ), 0 ) );
			const land = landGust.mul( LAND_SPAN ).add( LAND_LULL );
			const g = mix( open, land, smoothstep( LAND_FAR, LAND_NEAR, outside ) );
			return clamp( g.sub( 0.5 ).mul( gustAmount ).add( 0.5 ), 0, 1 );

		};
		this.gustOf = ( xz, landGust, displacement, time ) => blend( xz, openGust( xz, displacement, time ).gust, landGust );

		const sample = Fn( ( [ xz, landGust ]: [ V2, F ] ) => {

			const windSpeed = wind.breeze.mul( REFERENCE_SPEED );
			const open = openGust( xz, wind.displacement, shared.time );
			const g2 = open.g2;
			const gust = blend( xz, open.gust, landGust ).toVar();

			// a frame along the wind, lightly domain-warped so that the bands meander
			const w = wind.direction;
			const along = dot( xz, w ).toVar();
			const across = dot( xz, vec2( w.y.negate(), w.x ) ).add( g2.sub( 0.5 ).mul( 26 ) ).toVar();

			// slicks: long bands, strongest in light wind, torn apart by gusts
			const sl = noise( vec2( along.div( 1100 ), across.div( 70 ) ) ).z;
			const calmWind = smoothstep( 13, 4, windSpeed );
			const slick = smoothstep( 0.64, 0.76, sl ).mul( float( 1 ).sub( gust.mul( 0.8 ) ) ).mul( calmWind ).mul( slickAmount ).toVar();

			// windrows: thin lines about 10 m apart that come and go along their length
			const st = noise( vec2( along.div( 380 ), across.div( 11 ) ).add( vec2( 0.13, 0.71 ) ) ).w;
			const breakUp = noise( vec2( along.div( 140 ), across.div( 40 ) ).add( vec2( 0.51, 0.29 ) ) ).x;
			const freshWind = smoothstep( 6, 12, windSpeed );
			const streak = smoothstep( 0.68, 0.82, st ).mul( smoothstep( 0.4, 0.62, breakUp ) ).mul( freshWind ).mul( streakAmount ).toVar();

			// windrows show mostly as smooth lanes (surfactant and debris collect in the convergence
			// lines and damp the ripples), with only a trace of foam
			const rough = mix( float( 0.5 ), float( 1.5 ), gust ).mul( float( 1 ).sub( slick.mul( 0.8 ) ) )
				.mul( float( 1 ).sub( streak.div( max( streakAmount, 1e-3 ) ).mul( 0.45 ) ) );
			return vec4( rough, gust, slick, streak );

		} );
		this.sample = ( xz, landGust ) => sample( xz, landGust ?? this.landGust( xz ) );

	}

	/** The land's box (WorldShapes.landBounds), where the meadow's gusts hold. */
	setLand( b: Bounds ): void {

		this.landCentre.value.set( ( b.x0 + b.x1 ) / 2, ( b.z0 + b.z1 ) / 2 );
		this.landHalf.value.set( ( b.x1 - b.x0 ) / 2, ( b.z1 - b.z0 ) / 2 );

	}

}

// A tileable smooth fbm (periodic gradient noise, quintic fade) in four channels, each with its own
// seed, base frequency and octaves, normalised to 0..1.
function makeNoiseTexture( size: number ): THREE.DataTexture {

	const data = new Uint16Array( size * size * 4 );
	const channels = [
		{ seed: 11, freq: 4, oct: 4 },
		{ seed: 23, freq: 5, oct: 4 },
		{ seed: 37, freq: 4, oct: 3 },
		{ seed: 53, freq: 6, oct: 3 },
	];
	const values = new Float32Array( size * size );
	for ( let c = 0; c < 4; c ++ ) {

		const { seed, freq, oct } = channels[ c ];
		const rand = mulberry32( seed );
		// a gradient table per octave (a periodic lattice)
		const tables: { n: number; g: Float32Array }[] = [];
		for ( let o = 0; o < oct; o ++ ) {

			const n = freq << o;
			const g = new Float32Array( n * n * 2 );
			for ( let i = 0; i < n * n; i ++ ) {

				const a = rand() * Math.PI * 2;
				g[ i * 2 ] = Math.cos( a );
				g[ i * 2 + 1 ] = Math.sin( a );

			}
			tables.push( { n, g } );

		}

		let lo = Infinity, hi = - Infinity;
		for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

			let v = 0, amp = 1, norm = 0;
			for ( let o = 0; o < oct; o ++ ) {

				const { n, g } = tables[ o ];
				const fx = x / size * n, fy = y / size * n;
				const xi = Math.floor( fx ), yi = Math.floor( fy );
				const xf = fx - xi, yf = fy - yi;
				const grad = ( ix: number, iy: number, dx: number, dy: number ) => {

					const k = ( ( ( iy % n ) + n ) % n ) * n + ( ( ( ix % n ) + n ) % n );
					return g[ k * 2 ] * dx + g[ k * 2 + 1 ] * dy;

				};
				const u = xf * xf * xf * ( xf * ( xf * 6 - 15 ) + 10 );
				const w = yf * yf * yf * ( yf * ( yf * 6 - 15 ) + 10 );
				const a0 = grad( xi, yi, xf, yf ), a1 = grad( xi + 1, yi, xf - 1, yf );
				const b0 = grad( xi, yi + 1, xf, yf - 1 ), b1 = grad( xi + 1, yi + 1, xf - 1, yf - 1 );
				const top = a0 + ( a1 - a0 ) * u;
				v += ( top + ( ( b0 + ( b1 - b0 ) * u ) - top ) * w ) * amp;
				norm += amp;
				amp *= 0.5;

			}
			v /= norm;
			values[ y * size + x ] = v;
			lo = Math.min( lo, v );
			hi = Math.max( hi, v );

		}
		for ( let i = 0; i < size * size; i ++ ) data[ i * 4 + c ] = THREE.DataUtils.toHalfFloat( ( values[ i ] - lo ) / ( hi - lo ) );

	}

	const t = new THREE.DataTexture( data, size, size, THREE.RGBAFormat, THREE.HalfFloatType );
	t.wrapS = t.wrapT = THREE.RepeatWrapping;
	t.magFilter = t.minFilter = THREE.LinearFilter;
	t.generateMipmaps = false;
	t.name = 'seaDetailNoise';
	t.needsUpdate = true;
	return t;

}
