/*!
 * The strikes' flicker, return strokes, placement and in-cloud sources are adapted from
 * VolumetricCloudsCS1 by Anthony Ibrahim (https://github.com/anthonytoumaibrahim/VolumetricCloudsCS1:
 * VolumetricClouds/Sky/LightningFlicker.cs and CloudLightning.cs).
 * SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Ibrahim
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { attribute, cameraPosition, exp, float, length, max, min, mix, positionGeometry, smoothstep, step, uniform, vec4 } from 'three/tsl';
import { LightningStrike } from './LightningStrike.ts';
import { FLASH_SOURCES } from '../sky/CloudModel.ts';
import type { Clouds } from '../sky/Clouds.ts';
import { SUN_ILLUMINANCE } from '../sky/Atmosphere.ts';
import { SCENE_RADIANCE_SCALE } from '../lighting/Radiometry.ts';
import type { SceneLatents } from '../post/ExtraPass.ts';
import { reactiveUniforms, velocityOutput } from '../post/Motion.ts';

// Thunderstorm lightning over the sea: each strike's profile, light and flash.
//
// Thunderstorm lightning, kilometres out over the sea. Most strikes stay inside the cloud and
// light it from within (sheet lightning); the rest come down to the sea as a branching channel
// (LightningStrike.ts). Following VolumetricCloudsCS1, every strike has its own profile: how
// long it lasts, its return strokes, how nervously it flickers, its power, how far its light
// carries through the cloud and its tint. One intensity per strike drives the light inside the
// clouds (Clouds.flash), the channel and the flash on the island and the sea, so all three
// pulse as one.
//
// The storm is a function of the scene clock: time is cut into short slots, and a seeded hash
// decides whether a strike begins in each and everything about it. A paused frame therefore
// shows exactly the same strike every time, and `force` (`sky.lightning.force`) puts one in front
// of a review camera. A strike keeps the bearing it was given in the frame that first saw it (most
// come ahead of that frame's camera), so a scene jump forgets every strike (`reset`), as a new
// page knows none. Its light inside the clouds stays out of the view's accumulated history, the
// bake and the panorama (Clouds.ts).
//
// Under the temporal modes (Antialiasing.ts) a strike reaches the resolve twice. The channel draws
// in the opaque pass (the water's reflection reads the opaque copy) and overwrites the velocity
// with no motion and a reactive weight of 1 (Motion.ts), in both of its per-side draws (a
// transparent double-sided material: three draws the back faces, then the front, and the frame
// graph's warm-up compiles a clone per side, which keeps the material's MRT node): the resolve takes
// the channel from this frame alone. And the flash on the island, which changes the whole frame
// from one rendered frame to the next, sets the global `reactiveUniforms.flash` from how much the
// island's flash (`land`: the strikes' light as the fill weighs it, without its radiance scale)
// changed since the previous update: smoothstep( 0.02, 0.2, |land - previous| ). It is a function
// of the scene clock and the strikes like the rest, 0 on a paused frame after the first, and up only
// for the frames a strike flickers.

const SLOT = 0.5; // seconds
const MEAN_INTERVAL = 3.5; // seconds between strikes in a full storm
const MAX_DURATION = 1.8;
const SHEET_SHARE = 0.55; // most lightning stays in the cloud
const AHEAD_SHARE = 0.7; // and most of it where the viewer looks, as a storm worth watching would
const BOLTS = 2; // channels drawn at once
// Radiance of a channel's core, and of the light inside the cloud, against the sun's illuminance.
const CHANNEL_RADIANCE = 5;
const FLASH_RADIANCE = 0.6;
// Distance (m) over which the storm's rain and haze dim a channel by e.
const CHANNEL_AIR = 16000;
// A channel is at least this many pixels wide, its light spread over the width (no dashes).
const MIN_PIXELS = 1.6;
// The change of the island's flash from one update to the next over which the resolve's history
// weight falls from all to none (`reactiveUniforms.flash`).
const FLASH_REACTIVE = [ 0.02, 0.2 ];

export type StrikeKind = 'bolt' | 'sheet';

interface Source {
	offset: THREE.Vector3; // from the strike's in-cloud point
	reach: number;
	power: number;
	phase: number; // seconds this source lags the strike
}

interface Strike {
	id: number;
	start: number;
	duration: number;
	kind: StrikeKind;
	seed: number;
	ground: THREE.Vector3; // where a channel meets the sea
	top: THREE.Vector3; // where it leaves the cloud base
	center: THREE.Vector3; // the light inside the cloud
	sources: Source[];
	// Return strokes: times (s) and relative strength. A sheet's are smooth pulses of `width`.
	strokes: { at: number; strength: number }[];
	width: number;
	reseed: number;
	power: number;
	tint: THREE.Color;
}

export interface ForcedStrike {
	start: number; // scene seconds
	azimuth: number; // radians (atan2 of z and x) from the island
	distance: number; // metres
	kind?: StrikeKind;
	seed?: number;
}

// Hash of an integer pair to [0, 1).
function hash( a: number, b: number ): number {

	let t = Math.imul( a | 0, 0x9E3779B1 ) ^ Math.imul( ( b | 0 ) + 0x7F4A7C15, 0x85EBCA6B );
	t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
	t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
	return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

}

interface Bolt {
	mesh: THREE.Mesh;
	geometry: LightningStrike;
	strike: number; // id of the strike it holds, NaN when free
	age: THREE.UniformNode<'float', number>;
	channel: THREE.UniformNode<'float', number>;
	branches: THREE.UniformNode<'float', number>;
	leader: THREE.UniformNode<'float', number>;
	color: THREE.UniformNode<'color', THREE.Color>;
	cloudBase: THREE.UniformNode<'float', number>; // the base of the cloud its strike leaves (m)
}

export class Lightning {

	readonly group = new THREE.Group();
	/** The flash's share of the island's sky light this frame: radiance (scene units) of the lit
	 * cloud deck as seen from the island, 0 between strikes. */
	fill = 0;
	/** The strikes' light on the land this frame, as `fill` weighs it without its radiance (a
	 * strike of power 1 three kilometres off with a 2.5 km reach: 0.5); `reactiveUniforms.flash`
	 * follows its change. */
	land = 0;
	private landBefore: number | null = null;
	private forced: ForcedStrike | null = null;
	private readonly strikes = new Map<number, Strike>();
	private readonly bolts: Bolt[] = [];
	// A source keeps its slot in Clouds.flash for the life of its strike.
	private readonly slots: ( string | null )[] = new Array( FLASH_SOURCES ).fill( null );
	private readonly pixelAngle = uniform( 0.001 );
	private readonly radiance = SUN_ILLUMINANCE * SCENE_RADIANCE_SCALE;

	constructor() {

		this.group.name = 'lightning';
		for ( let i = 0; i < BOLTS; i ++ ) this.bolts.push( this.makeBolt() );

	}

	/** Review aid: a strike at a fixed time and place instead of the storm's own, until cleared
	 * with null. With the scene paused at `start` plus a fraction of a second, every frame shows
	 * the same moment of it. */
	force( strike: ForcedStrike | null ): void {

		this.forced = strike;
		this.strikes.delete( - 1 );

	}

	/** Forgets every strike, as a new page knows none (a scene jump, assemble/SceneJump.ts `establishScene`): a
	 * strike keeps the bearing it was given when first seen, ahead of the camera of that frame, so
	 * strikes seen from the scene before would come back where that scene's camera looked. Clears a
	 * forced strike too; the next update finds the storm's strikes again. */
	reset(): void {

		this.forced = null;
		this.strikes.clear();
		this.landBefore = null;
		this.slots.fill( null );
		this.fill = 0;
		this.land = 0;

	}

	/** Lights the strikes alight at scene time `time` (seconds) in a storm of strength `storm`
	 * (0 - 1), seen by `camera` on a drawing buffer `height` pixels tall. */
	update( time: number, storm: number, camera: THREE.PerspectiveCamera, height: number, clouds: Clouds ): void {

		this.pixelAngle.value = 2 * Math.tan( THREE.MathUtils.degToRad( camera.fov * 0.5 ) ) / ( camera.zoom || 1 ) / Math.max( height, 1 );
		camera.getWorldDirection( _forward );
		const heading = Math.atan2( _forward.z, _forward.x );
		const active: { strike: Strike; age: number }[] = [];
		if ( this.forced ) {

			const strike = this.strikes.get( - 1 ) ?? this.describe( - 1, this.forced.start, this.forced.kind, this.forced.azimuth, this.forced.distance, this.forced.seed ?? 1, clouds );
			this.strikes.set( - 1, strike );
			const age = time - strike.start;
			if ( age >= 0 && age < strike.duration ) active.push( { strike, age } );

		} else if ( storm > 0.05 ) {

			const chance = SLOT / MEAN_INTERVAL * Math.pow( storm, 1.5 );
			for ( let n = Math.floor( ( time - MAX_DURATION ) / SLOT ); n <= Math.floor( time / SLOT ); n ++ ) {

				let strike = this.strikes.get( n );
				if ( ! strike ) {

					if ( n < 0 || hash( n, 0 ) >= chance ) continue;
					const start = ( n + hash( n, 1 ) ) * SLOT;
					const kind = hash( n, 2 ) < SHEET_SHARE ? 'sheet' : 'bolt';
					const ahead = hash( n, 3 ) < AHEAD_SHARE;
					const azimuth = ahead ? heading + ( hash( n, 4 ) - 0.5 ) * 1.1 : hash( n, 4 ) * 2 * Math.PI;
					// Area-weighted over a ring kilometres out: nearer is rarer.
					const near = kind === 'bolt' ? 2500 : 3000, far = ( kind === 'bolt' ? 14000 : 18000 ) * ( 1.3 - 0.3 * storm );
					const distance = Math.sqrt( THREE.MathUtils.lerp( near * near, far * far, hash( n, 5 ) ) );
					strike = this.describe( n, start, kind, azimuth, distance, Math.floor( hash( n, 6 ) * 0x7fffffff ), clouds );
					// Only under the storm's closed sky: a front toward it brings its lightning as it closes.
					if ( clouds.coverage.value < 0.9 ) continue;
					this.strikes.set( n, strike );

				}
				const age = time - strike.start;
				if ( age >= 0 && age < strike.duration ) active.push( { strike, age } );

			}

		}
		for ( const [ id, strike ] of this.strikes ) if ( id >= 0 && ( strike.start > time + SLOT || strike.start + strike.duration < time - SLOT ) ) this.strikes.delete( id );
		this.light( active, clouds, camera.position );
		this.draw( active );
		// the flash's change since the last update, for the temporal resolve (none on the first)
		const change = Math.abs( this.land - ( this.landBefore ?? this.land ) );
		reactiveUniforms.flash.value = THREE.MathUtils.smoothstep( change, FLASH_REACTIVE[ 0 ], FLASH_REACTIVE[ 1 ] );
		this.landBefore = this.land;

	}

	// One strike's profile, all of it from its seed, in the clouds' shell (the weather's: Clouds.ts
	// `setShell`) as it stands when the strike is first described.
	private describe( id: number, start: number, kind: StrikeKind = 'bolt', azimuth: number, distance: number, seed: number, clouds: Clouds ): Strike {

		const r = ( k: number ) => hash( seed, k + 16 );
		const ground = new THREE.Vector3( Math.cos( azimuth ) * distance, - 1, Math.sin( azimuth ) * distance );
		// A channel slants a little on its way down.
		const lean = r( 0 ) * 2 * Math.PI, slant = 300 + 1200 * r( 1 );
		const bottom = clouds.shellBottom.value, thickness = clouds.shellThickness.value;
		const top = new THREE.Vector3( ground.x + Math.cos( lean ) * slant, bottom + 150, ground.z + Math.sin( lean ) * slant );
		// A stroke to the sea leaves the lower part of the cloud; sheet lightning can be anywhere
		// in it (CloudLightning.NewStrike).
		const depth = kind === 'bolt' ? THREE.MathUtils.lerp( 0.05, 0.25, r( 2 ) ) : THREE.MathUtils.lerp( 0.3, 0.7, r( 2 ) );
		const center = new THREE.Vector3( top.x + ( r( 3 ) - 0.5 ) * 500, bottom + thickness * depth, top.z + ( r( 4 ) - 0.5 ) * 500 );
		// Mostly blue-white, sometimes violet, now and then a warm white (Profile.RandomTint).
		const roll = r( 5 ), v = r( 6 );
		const tint = roll < 0.7 ? new THREE.Color( THREE.MathUtils.lerp( 0.72, 0.84, v ), THREE.MathUtils.lerp( 0.82, 0.9, v ), 1 )
			: roll < 0.9 ? new THREE.Color( THREE.MathUtils.lerp( 0.8, 0.9, v ), THREE.MathUtils.lerp( 0.72, 0.8, v ), 1 )
				: new THREE.Color( 1, THREE.MathUtils.lerp( 0.9, 0.96, v ), THREE.MathUtils.lerp( 0.78, 0.88, v ) );
		const p = r( 7 );
		const power = 0.55 + 0.95 * p * Math.sqrt( p );
		const reach = kind === 'bolt' ? THREE.MathUtils.lerp( 1600, 3200, r( 8 ) ) : THREE.MathUtils.lerp( 2200, 4500, r( 8 ) );
		const sources: Source[] = [ { offset: new THREE.Vector3(), reach, power: 1, phase: 0 } ];
		const strokes: { at: number; strength: number }[] = [];
		let duration: number, width = 0;
		if ( kind === 'bolt' ) {

			// The stepped leader comes down in about a tenth of a second; two to four return
			// strokes follow it up the same channel, each a little weaker.
			const count = 2 + Math.floor( r( 9 ) * 3 );
			let at = 0.06 + 0.08 * r( 10 );
			for ( let k = 0; k < count; k ++ ) {

				strokes.push( { at, strength: k === 0 ? 1 : 0.55 + 0.4 * r( 11 + k ) } );
				at += 0.04 + 0.11 * r( 15 + k );

			}
			duration = strokes[ count - 1 ].at + 0.45;

		} else {

			// Mostly short, now and then a long rolling one; one to four pulses, a long one always
			// more than one (Profile.Random). The light wanders through the cloud: a second source
			// a kilometre or two off flickers out of step with the first.
			const length = r( 9 );
			duration = ( 14 + 82 * length * length ) / 60;
			const roll2 = r( 10 );
			let count = roll2 < 0.45 ? 1 : roll2 < 0.75 ? 2 : roll2 < 0.92 ? 3 : 4;
			if ( duration > 55 / 60 && count < 2 ) count = 2;
			width = count === 1 ? 1 : 1.15 / count;
			for ( let k = 0; k < count; k ++ ) strokes.push( { at: ( k + r( 11 + k ) * 0.45 ) / count * ( 1 - width ) * duration, strength: 1 - 0.2 * k } );
			width *= duration;
			const angle = r( 15 ) * 2 * Math.PI, spread = 1000 + 2000 * r( 16 );
			sources.push( { offset: new THREE.Vector3( Math.cos( angle ) * spread, ( r( 17 ) - 0.5 ) * 1200, Math.sin( angle ) * spread ),
				reach: reach * 0.8, power: 0.4 + 0.4 * r( 18 ), phase: 0.04 + 0.12 * r( 19 ) } );

		}
		return { id, start, duration: Math.min( duration, MAX_DURATION ), kind, seed, ground, top, center, sources, strokes, width,
			reseed: 2 + Math.floor( r( 20 ) * 5 ), power, tint };

	}

	// Brightness of a strike `age` seconds in, 0 - 1, before its power. Each return stroke of a
	// channel flares within milliseconds and decays over tens of them; a sheet's pulses are
	// LightningFlicker's [6.75 t (1 - t)^2]^2. A flicker, reseeded every few 60 Hz frames, rides
	// on the envelope so the strike dies away rather than being cut off.
	private intensity( strike: Strike, age: number ): number {

		let envelope = 0;
		for ( const { at, strength } of strike.strokes ) {

			if ( strike.kind === 'bolt' ) {

				const t = age - at;
				if ( t < 0 ) continue;
				envelope = Math.max( envelope, strength * Math.min( t / 0.008, 1 ) * ( 0.7 * Math.exp( - t / 0.05 ) + 0.3 * Math.exp( - t / 0.22 ) ) );

			} else {

				const t = ( age - at ) / strike.width;
				if ( t <= 0 || t >= 1 ) continue;
				const pulse = 6.75 * t * ( t * ( t - 2 ) + 1 );
				envelope = Math.max( envelope, pulse * pulse * strength );

			}

		}
		// The leader glows faintly at the cloud base as it comes down.
		if ( strike.kind === 'bolt' ) envelope = Math.max( envelope, 0.08 * Math.min( age / strike.strokes[ 0 ].at, 1 ) );
		const frame = Math.floor( age * 60 );
		const level = hash( strike.seed, Math.floor( frame / strike.reseed ) + 1000 );
		return envelope * ( 0.45 + 0.55 * level );

	}

	// The light inside the clouds and on the island.
	private light( active: { strike: Strike; age: number }[], clouds: Clouds, viewer: THREE.Vector3 ): void {

		const lit = new Map<string, { position: THREE.Vector3; reach: number; color: THREE.Color }>();
		let fill = 0, land = 0;
		for ( const { strike, age } of active ) {

			strike.sources.forEach( ( source, k ) => {

				const level = this.intensity( strike, age - source.phase ) * source.power * strike.power;
				const position = strike.center.clone().add( source.offset );
				const color = strike.tint.clone().multiplyScalar( level * this.radiance * FLASH_RADIANCE );
				lit.set( `${ strike.id }:${ k }`, { position, reach: source.reach, color } );
				// The lit deck seen from the island: a patch a few kilometres across, smaller and
				// dimmer the farther off it is.
				const distance = Math.hypot( position.x - viewer.x, position.z - viewer.z );
				fill += level * this.radiance * FLASH_RADIANCE * 0.06 * source.reach / 2500 / ( 1 + ( distance / 3000 ) ** 2 );
				land += level * source.reach / 2500 / ( 1 + ( distance / 3000 ) ** 2 );

			} );

		}
		this.fill = fill;
		this.land = land;
		// Keep each source in its slot; a new one takes a free slot, and when more than four are
		// alight the dimmest go without.
		for ( let i = 0; i < FLASH_SOURCES; i ++ ) if ( this.slots[ i ] !== null && ! lit.has( this.slots[ i ]! ) ) this.slots[ i ] = null;
		const waiting = [ ...lit.keys() ].filter( ( key ) => ! this.slots.includes( key ) )
			.sort( ( a, b ) => lit.get( b )!.color.r - lit.get( a )!.color.r );
		for ( const key of waiting ) {

			const free = this.slots.indexOf( null );
			if ( free < 0 ) break;
			this.slots[ free ] = key;

		}
		let on = 0;
		for ( let i = 0; i < FLASH_SOURCES; i ++ ) {

			const target = clouds.flash[ i ];
			const source = this.slots[ i ] !== null ? lit.get( this.slots[ i ]! )! : null;
			if ( source ) {

				target.source.value.set( source.position.x, source.position.y, source.position.z, source.reach );
				target.color.value.copy( source.color );
				on = 1;

			} else {

				target.source.value.set( 0, - 1e6, 0, 1 );
				target.color.value.setRGB( 0, 0, 0 );

			}

		}
		clouds.flashOn.value = on;

	}

	// The channels of the strikes that come down.
	private draw( active: { strike: Strike; age: number }[] ): void {

		const bolts = active.filter( ( { strike } ) => strike.kind === 'bolt' ).slice( 0, BOLTS );
		for ( const bolt of this.bolts ) if ( ! bolts.some( ( { strike } ) => strike.id === bolt.strike ) ) {

			bolt.strike = NaN;
			bolt.mesh.visible = false;

		}
		for ( const { strike, age } of bolts ) {

			let bolt = this.bolts.find( ( b ) => b.strike === strike.id );
			if ( ! bolt ) {

				bolt = this.bolts.find( ( b ) => Number.isNaN( b.strike ) )!;
				bolt.strike = strike.id;
				this.grow( bolt.geometry, strike );

			}
			const first = strike.strokes[ 0 ].at;
			// The channel glows between its strokes a little longer than their flashes.
			let channel = 0;
			for ( const { at, strength } of strike.strokes ) {

				const t = age - at;
				if ( t >= 0 ) channel = Math.max( channel, strength * Math.min( t / 0.006, 1 ) * ( 0.6 * Math.exp( - t / 0.06 ) + 0.4 * Math.exp( - t / 0.3 ) ) );

			}
			const flicker = 0.75 + 0.25 * hash( strike.seed, Math.floor( age * 60 / strike.reseed ) + 2000 );
			bolt.age.value = age;
			bolt.channel.value = channel * flicker;
			// Branches light up with the leader and the first stroke; later strokes follow the
			// main channel only.
			const t0 = age - first;
			bolt.branches.value = t0 < 0 ? 0 : strike.strokes[ 0 ].strength * Math.min( t0 / 0.006, 1 ) * Math.exp( - t0 / 0.07 ) * flicker;
			bolt.leader.value = t0 < 0 ? 1 : 0;
			bolt.color.value.copy( strike.tint ).multiplyScalar( strike.power * this.radiance * CHANNEL_RADIANCE );
			bolt.cloudBase.value = strike.top.y - 150;
			bolt.mesh.visible = true;

		}

	}

	// Build a strike's channel: from just inside the cloud base down to the sea.
	private grow( geometry: LightningStrike, strike: Strike ): void {

		const down = new THREE.Vector3().subVectors( strike.ground, strike.top ).normalize();
		const up = new THREE.Vector3().crossVectors( down, new THREE.Vector3( 0, 0, 1 ) ).normalize();
		const last = strike.strokes[ strike.strokes.length - 1 ].at;
		geometry.build( {
			source: strike.top, destination: strike.ground, up,
			radius0: 2.5, radius1: 1.6, radius0Factor: 0.45, radius1Factor: 0.2, minRadius: 0.35,
			roughness: 0.93, straightness: 0.88,
			// It fades as a whole; only its last glow goes out from the cloud down.
			propagation: strike.strokes[ 0 ].at, vanishing: Math.max( last, strike.duration - 0.12 ), death: strike.duration,
			maxIterations: 8, ramification: 6, maxSubrayRecursion: 3, recursionProbability: 0.55,
			heightFactor: 0.45, sideWidthFactor: 1.1, floor: 120,
			seed: strike.seed,
		} );

	}

	private makeBolt(): Bolt {

		const geometry = new LightningStrike( 8, 6, 3 );
		const age = uniform( 0 ), channel = uniform( 0 ), branches = uniform( 0 ), leader = uniform( 0 );
		const color = uniform( new THREE.Color() ), cloudBase = uniform( 0 );
		const material = new THREE.MeshBasicNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide } );
		material.fog = false;
		material.toneMapped = false;
		const axis = attribute( 'axis', 'vec3' ), timing = attribute( 'timing', 'vec4' );
		// Widen the prisms to a pixel or so around their axis, kilometres out, and spread the
		// channel's light over the width (a thin branch stays dim, the main channel does not).
		const offset = positionGeometry.sub( axis );
		const distance = length( axis.sub( cameraPosition ) );
		const widen = max( float( 1 ), distance.mul( this.pixelAngle ).mul( MIN_PIXELS * 0.5 ).div( max( length( offset ), 1e-3 ) ) );
		material.positionNode = axis.add( offset.mul( widen ) );
		const thin = float( 1 ).div( widen ).pow( 0.6 ).toVarying( 'lightningThin' );
		const air = exp( distance.div( - CHANNEL_AIR ) ).toVarying( 'lightningAir' );
		const height = axis.y.toVarying( 'lightningHeight' );
		// The leader's front and the vanishing reach each point at its own times; between two
		// rings the front moves on smoothly.
		const arrive = timing.x, leave = timing.y, along = timing.z;
		const shown = step( arrive, age ).mul( step( age, leave ) );
		const trunk = float( 1 ).sub( step( 0.5, timing.w ) );
		// The stepped leader is faint, brightest at its tips.
		const tip = exp( age.sub( arrive ).div( - 0.015 ) ).mul( 0.1 ).add( 0.012 ).mul( leader );
		const strength = max( mix( branches.mul( float( 1 ).sub( along.mul( 0.6 ) ) ), channel, trunk ), tip );
		// The channel disappears into the cloud base.
		const base = float( 1 ).sub( smoothstep( cloudBase.sub( 600 ), cloudBase, height ) );
		material.colorNode = color.mul( min( strength.mul( shown ).mul( thin ).mul( base ).mul( air ), 4 ) );
		// Under the temporal modes the channel stays in the opaque pass (the water's reflection reads
		// the opaque copy, and must still show it) and writes no motion and a reactive weight of 1:
		// the resolve takes the bolt from this frame alone (Motion.ts).
		material.mrtNode = velocityOutput( vec4( 0, 0, 0, 1 ) );
		const mesh = new THREE.Mesh( geometry, material );
		mesh.visible = false;
		mesh.castShadow = mesh.receiveShadow = false;
		this.group.add( mesh );
		return { mesh, geometry, strike: NaN, age, channel, branches, leader, color, cloudBase };

	}

	/** Registers the channels with the frame graph's opaque pass (`group` is in the scene it
	 * draws), so the graph's warm-up compiles them before the first bolt: both passes of each
	 * transparent double-sided channel, through a copy of its material per side (Compile.ts), and
	 * their vertex buffers, so a bolt's first frame uploads only its own tree. The channels
	 * themselves are left alone, so a bolt may show while it runs. */
	registerLatent( graph: SceneLatents ): Promise<void> {

		return graph.latent( 'opaque', [ this.group ] );

	}

	dispose(): void {

		for ( const bolt of this.bolts ) {

			bolt.geometry.dispose();
			( bolt.mesh.material as THREE.Material ).dispose();

		}

	}

}

const _forward = new THREE.Vector3();
