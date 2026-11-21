/*!
 * Adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater), src/fx/Spray.js.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, Discard, uniform, float, uint, vec2, vec3, vec4, ivec2, instancedArray, instanceIndex, select,
	max, clamp, mix, smoothstep, length, normalize, dot, cross, exp, pow, sqrt, sin, cos, acos, fract,
	positionGeometry, positionView, positionWorld, cameraPosition, cameraProjectionMatrix, screenSize, screenCoordinate,
	textureLoad, texture, varyingProperty,
} from 'three/tsl';
import { outputStruct, overlayVelocity } from '../post/Motion.ts';
import { AFTER_RESOLVE } from '../post/Antialiasing.ts';
import { afterResolveCover } from '../post/AfterResolve.ts';

type F = THREE.Node<'float'>;
type U = THREE.Node<'uint'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

// GPU spray torn off breaking storm crests: drops, ligaments, dense spray and mist.
//
// GPU spray torn off breaking storm crests: drops, ligaments, dense spray and mist, after
// tidewater's Spray (its boat, terrain and surf-foam couplings are left out).
//
// The particles live in fixed groups: each emitter thread owns one group of slots and may only
// burst again once its last particle has died. There is no shared atomic ring head, so which
// slot a particle lands in never depends on thread scheduling and a paused frame stays
// bit-identical. Each step every idle emitter picks a random cell of the foam field and bursts with
// a probability that grows with the fresh breaking there (Poisson in time).
//
// Rendering (camera-facing sprites, premultiplied alpha, in the water target after the sea; under
// taa and taau after the resolve, at the canvas's resolution and unjittered, its occlusion from the
// nearest surface's texels around the pixel: post/AfterResolve.ts; under
// the ground truth in the overlay pass over the finished scene, Pipeline.ts, its coverage the
// reactive weight: Motion.ts):
//  * drops and ligaments are clear water: a sub-pixel drop is drawn as a motion streak whose
//    opacity conserves its cross-section, so its visual weight is the amount of water; they
//    show the sky they refract and light up when backlit (forward lobe)
//  * dense spray is white from every side (multiple scattering), with a forward lobe, torn into
//    strands and drop clusters as it ages
//  * mist is a thin, strongly forward-scattering, sky-tinted veil
// Everything is lit by the scene's key light under the clouds and by the sky; nothing glows.

export const SPRAY_KIND = { DROPLET: 0, MIST: 1, LIGAMENT: 2, SPRAY: 3 } as const;
const GRAVITY = 9.81;

// Per kind: droplet (a few mm, ballistic), mist (fine drops that follow the air), ligament (an
// elongated blob of ~1 cm torn off the crest), spray (a cloud of drops: the white of a crest).
const KIND = {
	tau: [ 3.0, 0.35, 5.0, 1.8 ], // drag time constant toward the air velocity (s): ~ drop size squared
	grav: [ GRAVITY, 0.3, GRAVITY, 8.5 ], // effective gravity (mist barely settles)
	turb: [ 0, 0.5, 0, 0.3 ], // turbulent wander (m/s^2, zero mean)
	grow: [ 0, 0.22, 0, 0.3 ], // size growth (1/s)
	dies: [ 1, 0, 1, 1 ], // killed when falling back into the water (mist skims over it)
	stretch: [ 1 / 40, 0, 1 / 40, 1 / 30 ], // motion blur (s of travel)
	alpha: [ 0.55, 0.06, 0.8, 0.7 ],
	fadeIn: [ 0.02, 0.2, 0.02, 0.03 ],
	fadeOut: [ 0.8, 0.45, 0.8, 0.7 ], // fraction of life when it starts to fade
};
const byKind = ( kind: F, values: number[] ): F => select( kind.lessThan( 0.5 ), float( values[ 0 ] ),
	select( kind.lessThan( 1.5 ), float( values[ 1 ] ), select( kind.lessThan( 2.5 ), float( values[ 2 ] ), float( values[ 3 ] ) ) ) );

// One burst: the slots of a group, with radius (m), base speeds (m/s) and life (s).
//   wind: share of the wind's speed; up: vertical launch speed; spread: random velocity
const BURST = [
	{ kind: SPRAY_KIND.SPRAY, radius: [ 0.3, 0.65 ], wind: [ 0.35, 0.6 ], up: [ 0.8, 2.2 ], spread: 0.9, life: [ 1.4, 2.2 ] },
	{ kind: SPRAY_KIND.SPRAY, radius: [ 0.2, 0.45 ], wind: [ 0.45, 0.75 ], up: [ 1.2, 2.8 ], spread: 1.1, life: [ 1.1, 1.8 ] },
	{ kind: SPRAY_KIND.SPRAY, radius: [ 0.15, 0.3 ], wind: [ 0.55, 0.85 ], up: [ 1.6, 3.2 ], spread: 1.2, life: [ 0.9, 1.5 ] },
	{ kind: SPRAY_KIND.MIST, radius: [ 0.7, 1.3 ], wind: [ 0.3, 0.55 ], up: [ 0.4, 1.0 ], spread: 0.6, life: [ 2.4, 3.4 ] },
	{ kind: SPRAY_KIND.MIST, radius: [ 0.5, 1.0 ], wind: [ 0.4, 0.65 ], up: [ 0.6, 1.4 ], spread: 0.8, life: [ 2.0, 3.0 ] },
	{ kind: SPRAY_KIND.LIGAMENT, radius: [ 0.006, 0.012 ], wind: [ 0.25, 0.45 ], up: [ 2.0, 3.6 ], spread: 1.0, life: [ 1.0, 1.4 ] },
	{ kind: SPRAY_KIND.DROPLET, radius: [ 0.002, 0.005 ], wind: [ 0.3, 0.6 ], up: [ 2.2, 4.2 ], spread: 1.4, life: [ 1.0, 1.5 ] },
	{ kind: SPRAY_KIND.DROPLET, radius: [ 0.002, 0.005 ], wind: [ 0.3, 0.6 ], up: [ 2.2, 4.2 ], spread: 1.4, life: [ 1.0, 1.5 ] },
];
const GROUP = BURST.length;
const GROUP_BUSY = 3.5; // s: longer than any life in a burst
// Under the ground truth (aa=gt, the spray before its resolve), the reactive weight per unit of
// coverage (at most 1), as the rain's drops (RainDrops.ts): at 1 the faint drops and mist kept most
// of their history and trailed.
const REACTIVE_GAIN = 4;

// PCG hash of a uint to [0, 1)
const hashU = Fn( ( [ seed ]: [ U ] ) => {

	const state = seed.mul( 747796405 ).add( 2891336453 );
	const word = state.shiftRight( state.shiftRight( 28 ).add( 4 ) ).bitXor( state ).mul( 277803737 );
	return float( word.shiftRight( 22 ).bitXor( word ) ).mul( 1 / 4294967296 );

} ).setLayout( { name: 'sprayHash', type: 'float', inputs: [ { name: 'seed', type: 'uint' } ] } );

// Henyey-Greenstein phase (1/sr)
const phaseHG = ( cosT: F, g: number ) => float( ( 1 - g * g ) / ( 4 * Math.PI ) )
	.div( pow( max( float( 1 + g * g ).sub( cosT.mul( 2 * g ) ), 1e-4 ), 1.5 ) );

export interface SprayField {
	/** Fresh breaking (0..1) at an undisplaced world xz, 0 outside the field; inside a compute Fn. */
	breakingAt( xz: V2 ): F;
	/** The displaced sea surface above an undisplaced xz: position offset (x, y, z). */
	surfaceOffset( xz: V2 ): V3;
	/** Orbital velocity of the surface (m/s, world axes). */
	surfaceVelocity( xz: V2 ): V3;
}

export interface SprayLighting {
	sunDir: V3;
	sunColor: V3;
	sky: V3;
	sunVisibility: ( p: V3 ) => F;
	wind: V2; // air velocity, m/s
	waterLevel: F;
	// resolved linear view z of the opaque pass (after the resolve: of the nearest surface, the sea's
	// included, which it then also hides the spray behind)
	opaqueDepth: THREE.Texture;
}

export class Spray {

	readonly mesh: THREE.Mesh;
	readonly groups: number;
	readonly count: number;
	readonly params = {
		rate: uniform( 20 ), // bursts per second per idle emitter at full breaking (only the steepest crests break)
		intensity: uniform( 1 ),
		gate: uniform( 0 ), // storm strength
		maxDistance: uniform( 160 ),
	};
	/** The camera position: compute kernels have no camera of their own. */
	readonly eye = uniform( new THREE.Vector3() );

	private renderer: THREE.WebGPURenderer;
	private dt = uniform( 0 );
	private seed = uniform( 0, 'uint' );
	private pos;
	private vel;
	private info;
	private busy;
	private emitKernel: THREE.ComputeNode;
	private updateKernel: THREE.ComputeNode;
	private clearKernel: THREE.ComputeNode;

	constructor( renderer: THREE.WebGPURenderer, field: SprayField, light: SprayLighting, groups = 2048 ) {

		this.renderer = renderer;
		this.groups = groups;
		const N = this.count = groups * GROUP;
		// pos: xyz, age (s) | vel: xyz, radius (m) | info: kind, life (s, 0 = dead), water height, seed
		this.pos = instancedArray( N, 'vec4' ).setName( 'sprayPos' );
		this.vel = instancedArray( N, 'vec4' ).setName( 'sprayVel' );
		this.info = instancedArray( N, 'vec4' ).setName( 'sprayInfo' );
		this.busy = instancedArray( groups, 'float' ).setName( 'sprayBusy' );
		const P = this.params, dt = this.dt;
		const rand = ( a: U, b: number ) => hashU( a.mul( 9781 ).add( uint( b * 6271 + 17 ) ).add( this.seed.mul( 2654435761 ) ) );
		const surfaceHeight = ( xz: V2 ) => light.waterLevel.add( field.surfaceOffset( xz ).y );

		this.clearKernel = Fn( () => {

			const i = instanceIndex;
			this.info.element( i ).assign( vec4( 0 ) );
			If( i.lessThan( uint( groups ) ), () => {

				this.busy.element( i ).assign( 0 );

			} );

		} )().compute( N, [ 64 ] ).setName( 'Spray.Clear' );

		// ---- emitters: one thread per group
		this.emitKernel = Fn( () => {

			const g = instanceIndex;
			const busy = this.busy.element( g );
			const left = busy.sub( dt ).toVar();
			If( left.lessThanEqual( 0 ), () => {

				left.assign( 0 );
				// A candidate spot, uniform over the disc of sea where spray can be seen.
				const radius = P.maxDistance.mul( rand( g, 1 ).sqrt() );
				const angle = rand( g, 2 ).mul( 2 * Math.PI );
				const xz = this.eye.xz.add( vec2( cos( angle ), sin( angle ) ).mul( radius ) ).toVar();
				const breaking = field.breakingAt( xz ).mul( P.gate ).toVar();
				const chance = float( 1 ).sub( exp( P.rate.mul( breaking ).mul( dt ).negate() ) );
				If( rand( g, 5 ).lessThan( chance ), () => {

					left.assign( GROUP_BUSY );
					const offset = field.surfaceOffset( xz ).toVar();
					const crest = vec3( xz.x.add( offset.x ), light.waterLevel.add( offset.y ), xz.y.add( offset.z ) ).toVar();
					const orbital = field.surfaceVelocity( xz ).toVar();
					const wind = vec3( light.wind.x, 0, light.wind.y ).toVar();
					const strength = breaking.sqrt().mul( 0.4 ).add( 0.6 ).toVar();
					BURST.forEach( ( b, k ) => {

						const slot = g.mul( GROUP ).add( k );
						const r = ( salt: number ) => rand( slot, 11 + salt );
						const radius = mix( float( b.radius[ 0 ] ), float( b.radius[ 1 ] ), r( 0 ) );
						const jitter = vec3( r( 1 ), r( 2 ), r( 3 ) ).sub( 0.5 );
						const velocity = wind.mul( mix( float( b.wind[ 0 ] ), float( b.wind[ 1 ] ), r( 4 ) ) ).add( orbital )
							.add( vec3( 0, mix( float( b.up[ 0 ] ), float( b.up[ 1 ] ), r( 5 ) ).mul( strength ), 0 ) )
							.add( jitter.mul( b.spread * 2 ) );
						const p = crest.add( jitter.mul( vec3( 0.9, 0.2, 0.9 ) ) ).add( vec3( 0, 0.05, 0 ) );
						const life = mix( float( b.life[ 0 ] ), float( b.life[ 1 ] ), r( 6 ) );
						this.pos.element( slot ).assign( vec4( p, 0 ) );
						this.vel.element( slot ).assign( vec4( velocity, radius ) );
						this.info.element( slot ).assign( vec4( b.kind, life, p.y, r( 7 ) ) );

					} );

				} );

			} );
			busy.assign( left );

		} )().compute( groups, [ 64 ] ).setName( 'Spray.Emit' );

		// ---- integrate live particles
		this.updateKernel = Fn( () => {

			const i = instanceIndex;
			const P4 = this.pos.element( i ), V4 = this.vel.element( i ), I4 = this.info.element( i );
			const info = I4.toVar();
			If( info.y.greaterThan( 0 ), () => {

				const p = P4.xyz.toVar();
				const v = V4.xyz.toVar();
				const age = P4.w.add( dt ).toVar();
				const kind = info.x;
				const seed = fract( info.w );

				// wind near the surface (about 70 % of the 10 m wind), gusty
				const gust = sin( age.mul( 0.7 ).add( p.x.mul( 0.05 ) ).add( seed.mul( 6.28 ) ) ).mul( 0.25 ).add( 0.85 );
				const wind = vec3( light.wind.x, 0, light.wind.y ).mul( gust.mul( 0.7 ) );
				// Mist first keeps moving with the air the crest pushed ahead of it, then joins the wind.
				const isMist = kind.greaterThan( 0.5 ).and( kind.lessThan( 1.5 ) );
				const tau = byKind( kind, KIND.tau ).mul( select( isMist, exp( age.mul( - 1.2 ) ).mul( 4 ).add( 1 ), float( 1 ) ) );
				const turb = vec3(
					sin( age.mul( 2.1 ).add( seed.mul( 40 ) ) ),
					sin( age.mul( 1.7 ).add( seed.mul( 17 ) ) ).mul( 0.5 ),
					cos( age.mul( 1.9 ).add( seed.mul( 29 ) ) ) ).mul( byKind( kind, KIND.turb ) );
				v.addAssign( wind.sub( v ).div( tau ).add( turb ).mul( dt ) );
				v.y.subAssign( byKind( kind, KIND.grav ).mul( dt ) );
				p.addAssign( v.mul( dt ) );

				const life = info.y.toVar();
				const water = surfaceHeight( p.xz ).toVar();
				If( p.y.lessThan( water ).and( v.y.lessThan( 0 ) ).and( age.greaterThan( 0.04 ) ), () => {

					If( byKind( kind, KIND.dies ).greaterThan( 0.5 ), () => {

						life.assign( 0 );

					} ).Else( () => {

						p.y.assign( water.add( 0.02 ) );
						v.y.assign( max( v.y, 0 ) );
						v.xz.mulAssign( 0.95 );

					} );

				} );
				If( age.greaterThan( life ), () => {

					life.assign( 0 );

				} );
				// mist and spray clouds grow as they dilute
				P4.assign( vec4( p, age ) );
				V4.assign( vec4( v, V4.w.mul( float( 1 ).add( dt.mul( byKind( kind, KIND.grow ) ) ) ) ) );
				I4.assign( vec4( info.x, life, water, info.w ) );

			} );

		} )().compute( N, [ 64 ] ).setName( 'Spray.Update' );

		this.mesh = this.buildMesh( light );
		renderer.compute( this.clearKernel );

	}

	/** The spray's compute kernels (emission, the step and the clear), for the frame graph's
	 * warm-up (StormSea.registerLatent). */
	get kernels(): THREE.ComputeNode[] {

		return [ this.emitKernel, this.updateKernel, this.clearKernel ];

	}

	/** Remove every particle (a new storm). */
	clear(): void {

		this.renderer.compute( this.clearKernel );

	}

	/** Advance by dt seconds of scene time; seed makes the step's random draws reproducible. */
	step( dt: number, seed: number ): void {

		if ( dt <= 0 ) return;
		this.dt.value = dt;
		this.seed.value = seed >>> 0;
		this.renderer.compute( [ this.emitKernel, this.updateKernel ] );

	}

	private buildMesh( light: SprayLighting ): THREE.Mesh {

		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( [ - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 1, 0 ] ), 3 ) );
		geometry.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
		geometry.boundingSphere = new THREE.Sphere( new THREE.Vector3(), 1e7 );

		const material = new THREE.NodeMaterial();
		material.name = 'Spray';
		material.transparent = true;
		material.depthWrite = false;
		// (after the resolve there is no depth to test: the cover below hides it)
		material.depthTest = ! AFTER_RESOLVE;
		material.side = THREE.DoubleSide;
		material.forceSinglePass = true;
		// premultiplied radiance, like the rain sharing this target
		material.blending = THREE.CustomBlending;
		material.blendEquation = THREE.AddEquation;
		material.blendSrc = THREE.OneFactor;
		material.blendDst = THREE.OneMinusSrcAlphaFactor;
		material.blendSrcAlpha = THREE.OneFactor;
		material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;

		const posA = this.pos.toAttribute(), velA = this.vel.toAttribute(), infoA = this.info.toAttribute();
		const vUV = varyingProperty( 'vec2', 'vSprayUV' );
		const vCol = varyingProperty( 'vec4', 'vSprayCol' ); // radiance, opacity
		const vMisc = varyingProperty( 'vec4', 'vSprayMisc' ); // kind + life fraction, water height, size, seed
		const vFwd = varyingProperty( 'vec3', 'vSprayFwd' ); // forward-scattered sun (thin parts glow with it)
		const P = this.params;
		const L = light.sunDir;

		material.positionNode = Fn( () => {

			const p = posA.xyz, age = posA.w, v = velA.xyz, r = velA.w;
			const kind = infoA.x, life = infoA.y;
			const isDrop = kind.lessThan( 0.5 );
			const isMist = kind.greaterThan( 0.5 ).and( kind.lessThan( 1.5 ) );
			const isLig = kind.greaterThan( 1.5 ).and( kind.lessThan( 2.5 ) );
			const isSpray = kind.greaterThan( 2.5 );
			const water = isDrop.or( isLig );
			const alive = life.greaterThan( 0 ).and( age.lessThan( life ) );

			const toCam = cameraPosition.sub( p );
			const dist = max( length( toCam ), 0.05 ).toVar();
			const Vd = toCam.div( dist ).toVar();
			// pixel footprint at this distance: drops are drawn at least ~1.3 px wide
			// ( 1, 1 ) of the projection is 1 / tan( fov / 2 )
			const p11 = ( cameraProjectionMatrix as unknown as { element( i: number ): { element( i: number ): F } } ).element( 1 ).element( 1 );
			const pixel = dist.mul( 2 ).div( p11.mul( screenSize.y ) );
			const size = max( r, pixel.mul( 1.3 ) ).toVar();

			// motion blur along the velocity projected on the view plane
			const vPerp = v.sub( Vd.mul( dot( v, Vd ) ) ).toVar();
			const speed = length( vPerp ).toVar();
			const stretchLen = speed.mul( byKind( kind, KIND.stretch ) );
			const elong = select( isLig, r.mul( 2 ), float( 0 ) );
			const up = select( speed.greaterThan( 1e-3 ), vPerp.div( max( speed, 1e-3 ) ), vec3( 0, 1, 0 ) ).toVar();
			// mist: a random rotation that slowly turns
			const seed = fract( infoA.w );
			const rot = seed.mul( 6.283 ).add( age.mul( seed.sub( 0.5 ) ) );
			const camRight = normalize( cross( vec3( 0, 1, 0 ), Vd ).add( vec3( 1e-5, 0, 0 ) ) ).toVar();
			const camUp = cross( Vd, camRight ).toVar();
			const mRight = camRight.mul( cos( rot ) ).add( camUp.mul( sin( rot ) ) );
			const mUp = camUp.mul( cos( rot ) ).sub( camRight.mul( sin( rot ) ) );
			// Dense spray is drawn along its motion too, tilted a little at random and stretched by
			// its speed: fibrous, streaked silhouettes instead of round puffs.
			const side = normalize( cross( Vd, up ) ).toVar();
			const tilt = fract( infoA.w.mul( 7.31 ) ).sub( 0.5 ).mul( 0.7 );
			const sUp = up.mul( cos( tilt ) ).add( side.mul( sin( tilt ) ) );
			const sSide = side.mul( cos( tilt ) ).sub( up.mul( sin( tilt ) ) );
			const axisY = select( water, up, select( isSpray, sUp, mUp ) );
			const axisX = select( water, side, select( isSpray, sSide, mRight ) );
			const sprayLength = select( isSpray, size.mul( clamp( speed.mul( 0.08 ), 0, 0.8 ) ), float( 0 ) );
			const halfY = size.add( stretchLen.mul( 0.5 ) ).add( elong ).add( sprayLength ).toVar();
			const halfX = select( isSpray, size.mul( 1.05 ), size ).toVar();
			const corner = positionGeometry.xy;
			const world = p.add( axisX.mul( corner.x.mul( halfX ) ) ).add( axisY.mul( corner.y.mul( halfY ) ) );

			// Coverage that conserves the water's cross-section: the drop's projected area (and the
			// time it spends on each pixel of its streak) spread over the drawn footprint. For the
			// Gaussian footprint exp( -k d^2 ) the peak is k r ( r + elong ) / ( halfX halfY ).
			const kShape = select( isLig, float( 4.5 ), float( 3.5 ) );
			const cover = select( water, r.mul( r.add( elong ) ).mul( kShape ).div( halfX.mul( halfY ) ).min( 1 ), clamp( r.div( size ), 0, 1 ) );

			// ---- lighting (per particle): the key light under the clouds, and the sky
			const cosT = dot( Vd.negate(), L ); // 1 = looking toward the sun through the particle
			const cosA = dot( Vd, L );
			const sun = light.sunColor.mul( light.sunVisibility( p ) ).toVar();
			const sky = light.sky;
			// clear water: the sky it refracts and reflects, a strong forward lobe when backlit
			const cWater = sky.mul( 0.9 ).add( sun.mul( phaseHG( cosT, 0.85 ).mul( 1.2 ).add( 0.12 ) ) );
			// dense spray: multiply scattered, white from any side (a diffuse sphere, its far side in
			// its own shadow), its forward lobe passed on separately
			const lambert = sqrt( max( float( 1 ).sub( cosA.mul( cosA ) ), 0 ) ).add( float( Math.PI ).sub( acos( cosA ) ).mul( cosA ) ).div( Math.PI );
			const cSpray = sun.mul( lambert.mul( 0.65 ).add( 0.35 ).div( Math.PI ) ).add( sky.mul( 1.0 ) );
			const fSpray = sun.mul( phaseHG( cosT, 0.6 ).mul( 0.9 ) );
			// mist: a thin veil of fine drops, strongly forward scattering, tinted by the sky
			const cMist = sun.mul( 0.25 / Math.PI ).add( sky.mul( 0.9 ) );
			const fMist = sun.mul( phaseHG( cosT, 0.75 ) );
			vFwd.assign( select( water, vec3( 0 ), select( isMist, fMist, fSpray ) ) );
			const color = select( water, cWater, select( isMist, cMist, cSpray ) );

			// opacity over the particle's life
			const t = age.div( max( life, 1e-3 ) ).toVar();
			const fadeIn = smoothstep( 0, byKind( kind, KIND.fadeIn ), t );
			const fadeOut = float( 1 ).sub( smoothstep( byKind( kind, KIND.fadeOut ), 1, t ) );
			// far away: fade out; very near the eye, puffs would fill the screen
			const distFade = smoothstep( P.maxDistance, P.maxDistance.mul( 0.55 ), dist ).mul( select( water, float( 1 ), smoothstep( 0.6, 3.0, dist ) ) );
			const a = byKind( kind, KIND.alpha ).mul( fadeIn ).mul( fadeOut ).mul( cover ).mul( distFade ).mul( P.intensity ).mul( P.gate.min( 1 ) );

			vUV.assign( corner );
			vCol.assign( vec4( color, a ) );
			// x: kind + 0.45 x life fraction (the kind tests below keep 0.5 of margin)
			vMisc.assign( vec4( kind.add( clamp( t, 0, 1 ).mul( 0.45 ) ), infoA.z, size, seed ) );
			// dead particles collapse off-screen
			return select( alive.and( a.greaterThan( 1e-4 ) ), world, vec3( 0, - 1e5, 0 ) );

		} )();

		const puff = makePuffTexture(), dots = makeDotsTexture();
		const depth = texture( light.opaqueDepth );
		material.fragmentNode = Fn( () => {

			const uv = vUV;
			const kind = vMisc.x;
			const isDrop = kind.lessThan( 0.5 );
			const isLig = kind.greaterThan( 1.5 ).and( kind.lessThan( 2.5 ) );
			const isSpray = kind.greaterThan( 2.5 );
			const t = clamp( fract( kind ).div( 0.45 ), 0, 1 ); // life fraction
			const r2 = dot( uv, uv );
			const sd = vec2( vMisc.w, vMisc.w.mul( 1.7 ) );
			// Filtered samples first, while control flow is uniform.
			const fib = texture( puff, vec2( uv.x.mul( 0.5 ), uv.y.mul( 0.26 ) ).add( sd ) ).x.toVar();
			const fine = texture( puff, vec2( uv.x.mul( 1.2 ), uv.y.mul( 0.6 ) ).add( sd.mul( 2.3 ) ) ).x.toVar();
			const dotted = texture( dots, uv.mul( vec2( 0.5, 0.32 ) ).add( sd.mul( 3.7 ) ) ).x.toVar();
			const m1 = texture( puff, uv.mul( 0.2 ).add( sd ) ).x.toVar();
			const m2 = texture( puff, uv.mul( 0.55 ).add( sd.mul( 3.1 ) ) ).x.toVar();
			// drop: a Gaussian streak; ligament: a slightly sharper, beaded blob
			const drop = exp( r2.mul( - 3.5 ) ).sub( 0.03 ).max( 0 );
			const lig = exp( r2.mul( - 4.5 ) ).mul( sin( uv.y.mul( 5.0 ).add( vMisc.w.mul( 40 ) ) ).mul( 0.2 ).add( 0.9 ) ).sub( 0.03 ).max( 0 );
			// dense spray: noise streaked along the motion, eroded from its edges inward and more as it
			// ages, so it tears into strands and drop clusters instead of shrinking
			const env = clamp( float( 1 ).sub( r2 ), 0, 1 );
			const field = fib.mul( 0.6 ).add( fine.mul( 0.4 ) ).add( env.sub( 0.55 ).mul( 0.75 ) ).sub( smoothstep( 0.8, 1.0, r2 ) );
			const erode = mix( float( 0.26 ), float( 0.7 ), t );
			const dens = clamp( field.sub( erode ).mul( 3 ), 0, 1 ).toVar();
			const torn = smoothstep( erode.sub( 0.04 ), erode.add( 0.2 ), field ).mul( dens.mul( 0.45 ).add( 0.55 ) );
			const swarm = dotted.mul( smoothstep( 0.25, 0.55, fib.add( env.mul( 0.45 ) ).sub( t.mul( 0.2 ) ) ) ).mul( float( 1 ).sub( t.mul( 0.5 ) ) );
			const sheet = max( torn.mul( float( 1 ).sub( smoothstep( 0.15, 0.6, t ) ) ), swarm );
			// mist: a soft veil of low-frequency noise that drifts and thins, never a disc
			const veil = smoothstep( 0.28, 0.8, m1.mul( 0.7 ).add( m2.mul( 0.3 ) ).add( env.mul( 0.3 ) ).sub( 0.12 ) ).mul( env.sqrt() ).mul( float( 1 ).sub( t.mul( 0.4 ) ) );
			const shape = select( isDrop, drop, select( isLig, lig, select( isSpray, sheet, veil ) ) );
			// self-shadowing inside thick spray; the forward-scattered sun lights up the thin parts
			const shade = select( isSpray, float( 1 ).sub( dens.mul( 0.15 ) ), float( 1 ) );
			const glow = select( isSpray, float( 1 ).sub( dens.mul( 0.6 ) ), float( 1 ) );

			// soft intersections with the opaque scene (resolved linear depth) and the sea under the particle
			const soft = vMisc.z.mul( 1.5 ).add( 0.03 );
			const fadeScene = AFTER_RESOLVE
				? afterResolveCover( light.opaqueDepth, ( _texel, z ) => clamp( positionView.z.sub( z ).div( soft ), 0, 1 ) )
				: clamp( positionView.z.sub( textureLoad( depth, ivec2( screenCoordinate.xy ) ).x ).div( soft ), 0, 1 );
			const fadeWater = clamp( positionWorld.y.sub( vMisc.y ).div( soft.mul( 0.6 ) ).add( 0.15 ), 0, 1 );
			const alpha = vCol.w.mul( shape ).mul( fadeScene ).mul( fadeWater ).min( 1 ).toVar();
			If( alpha.lessThan( 0.002 ), () => {

				Discard();

			} );
			return vec4( vCol.rgb.mul( shade ).add( vFwd.mul( glow ) ).mul( alpha ), alpha );

		} )();
		// under the ground truth it draws in the overlay pass, whose velocity it keeps (Motion.ts), with
		// its coverage times REACTIVE_GAIN as the reactive weight (the overlay MRT keeps the larger
		// where particles overlap): the resolve trusts its history less where spray covers the pixel.
		// After the resolve it writes its colour alone.
		const shaded = material.fragmentNode as THREE.Node<'vec4'>;
		if ( ! AFTER_RESOLVE ) material.fragmentNode = outputStruct( shaded, overlayVelocity( shaded.a.mul( REACTIVE_GAIN ).min( 1 ) ) );

		const mesh = new THREE.Mesh( geometry, material );
		( mesh as unknown as { count: number } ).count = this.count;
		mesh.frustumCulled = false;
		mesh.castShadow = mesh.receiveShadow = false;
		mesh.renderOrder = 20;
		mesh.name = 'stormSpray';
		mesh.visible = false;
		return mesh;

	}

	dispose(): void {

		this.emitKernel.dispose();
		this.updateKernel.dispose();
		this.clearKernel.dispose();
		this.mesh.geometry.dispose();
		( this.mesh.material as THREE.Material ).dispose();

	}

}

// Small tileable value-noise texture for spray and mist puffs (generated once on the CPU).
function makePuffTexture( size = 64 ): THREE.DataTexture {

	const data = new Uint8Array( size * size * 4 );
	const rnd = ( i: number, j: number, o: number ) => {

		const s = Math.sin( ( i % o ) * 127.1 + ( j % o ) * 311.7 + o * 17.3 ) * 43758.5453;
		return s - Math.floor( s );

	};
	const vnoise = ( x: number, y: number, o: number ) => {

		const i = Math.floor( x ), j = Math.floor( y );
		const fx = x - i, fy = y - j;
		const ux = fx * fx * ( 3 - 2 * fx ), uy = fy * fy * ( 3 - 2 * fy );
		const a = rnd( i, j, o ), b = rnd( i + 1, j, o ), c = rnd( i, j + 1, o ), d = rnd( i + 1, j + 1, o );
		return ( a * ( 1 - ux ) + b * ux ) * ( 1 - uy ) + ( c * ( 1 - ux ) + d * ux ) * uy;

	};
	for ( let j = 0; j < size; j ++ ) for ( let i = 0; i < size; i ++ ) {

		let s = 0, a = 0.5, n = 0;
		for ( let o = 4; o <= 32; o *= 2 ) {

			s += vnoise( i / size * o, j / size * o, o ) * a;
			n += a;
			a *= 0.55;

		}
		const v = Math.max( 0, Math.min( 1, ( s / n - 0.2 ) * 1.6 ) );
		const k = ( j * size + i ) * 4;
		data[ k ] = data[ k + 1 ] = data[ k + 2 ] = Math.round( v * 255 );
		data[ k + 3 ] = 255;

	}
	return finishTexture( new THREE.DataTexture( data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType ), 'sprayPuff' );

}

// Tileable texture of scattered drops (r: coverage) for clusters of drops. Heavy-tailed radii:
// many tiny drops, a few large ones.
function makeDotsTexture( size = 128, count = 150 ): THREE.DataTexture {

	const data = new Uint8Array( size * size * 4 );
	let seed = 12345;
	const rnd = () => ( ( seed = ( seed * 1664525 + 1013904223 ) >>> 0 ) / 4294967296 );
	const cover = new Float32Array( size * size );
	for ( let k = 0; k < count; k ++ ) {

		const cx = rnd() * size, cy = rnd() * size;
		const r = Math.min( 0.9 * Math.pow( 1 - rnd() * 0.97, - 0.55 ), 6 );
		const R = Math.ceil( r + 1.5 );
		for ( let dy = - R; dy <= R; dy ++ ) for ( let dx = - R; dx <= R; dx ++ ) {

			const x = ( ( Math.floor( cx ) + dx ) % size + size ) % size, y = ( ( Math.floor( cy ) + dy ) % size + size ) % size;
			const d = Math.hypot( Math.floor( cx ) + dx + 0.5 - cx, Math.floor( cy ) + dy + 0.5 - cy );
			cover[ y * size + x ] = Math.max( cover[ y * size + x ], Math.max( 0, Math.min( 1, r + 0.5 - d ) ) );

		}

	}
	for ( let i = 0; i < size * size; i ++ ) {

		data[ i * 4 ] = data[ i * 4 + 1 ] = data[ i * 4 + 2 ] = Math.round( cover[ i ] * 255 );
		data[ i * 4 + 3 ] = 255;

	}
	return finishTexture( new THREE.DataTexture( data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType ), 'sprayDots' );

}

function finishTexture( tex: THREE.DataTexture, name: string ): THREE.DataTexture {

	tex.name = name;
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.magFilter = THREE.LinearFilter;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.generateMipmaps = true;
	tex.needsUpdate = true;
	return tex;

}
