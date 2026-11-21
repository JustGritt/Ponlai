/*!
 * GPU rain with height-map collision, after "collision rain" in threejs-conference by
 * Anderson Mancini and Sunag (https://github.com/ektogamat/threejs-conference,
 * src/world/weather/createCollisionRain.js): drops in a camera-centred toroidal volume, a
 * compute pass that tests them against a top-down height map, and instanced billboards for
 * the streaks and their splashes.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Anderson Mancini and Sunag
 * SPDX-License-Identifier: MIT
 *
 * The same pattern is in the three.js example webgpu_compute_particles_rain.html.
 * SPDX-FileCopyrightText: Copyright (c) 2010-2026 three.js authors
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Break, instancedArray, instanceIndex, hash, uniform, varying, cameraPosition, positionGeometry, positionView,
	screenCoordinate, texture, textureLoad, textureSize, uv, ivec2, vec2, vec3, vec4, float, floor, fract, select, min, max, abs,
	length, normalize, cross, dot, smoothstep, clamp, exp, sin, sqrt, pow, mix, output,
} from 'three/tsl';
import type { RainCollision } from './RainCollision.ts';
import { overlayVelocity, velocityOutput } from '../post/Motion.ts';
import { AFTER_RESOLVE } from '../post/Antialiasing.ts';
import { afterResolveCover } from '../post/AfterResolve.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// The rain's GPU drops: streaks and splashes in camera volumes, collided against a height map.

// Under taa and taau (post/Antialiasing.ts AFTER_RESOLVE) the streaks, splashes and rings draw after
// the temporal resolve, at the canvas's resolution and unjittered (`afterGroup`,
// post/AfterResolve.ts): no depth buffer there, so they test the render texels around their pixel
// (the same tests as below, per texel), and their widths follow the canvas's pixel. (A splash is a glint of a few frames: kept before the resolve
// with its reactive weight, it still lost most of itself to the history.)

export const RAIN_MAX_DROPS = 49152;

// Drops fall at 7.2 - 10.2 m/s (the terminal speeds of 1.5 - 4 mm drops), in steps of 0.1 m/s:
// with whole tenths a cascade's fall repeats exactly every 10 H seconds, so the clock can be
// wrapped on the CPU and the shader never meets a large time.
const FALL_TENTHS_MIN = 72, FALL_TENTHS_RANGE = 31;
export const MEAN_FALL = ( FALL_TENTHS_MIN + ( FALL_TENTHS_RANGE - 1 ) / 2 ) / 10;
// Air carries the drops almost at its own speed; a little less keeps a gale's rain readable.
export const DRIFT = 0.8;
// The eye's exposure: a drop is seen as the streak it travels meanwhile.
const EXPOSURE = 0.032;
const DIAMETER = 0.003; // m, a little larger than real drops, for a stylised read
// Drops closer than this fade out (m): at arm's length a streak would cross much of the screen.
const NEAR_FADE = [ 0.8, 2 ];
// Lifetimes (s) of a splash on land, of the jet a drop throws up from the sea, and of its ring.
const SPRAY_LIFE = 0.26, SEA_SPRAY_LIFE = 0.16, RING_LIFE = 0.75;
const SPRAY_SIZE = 0.18, RING_SIZE = 0.5; // m
// Collision search: steps down the drop's path from the highest ground, then bisections.
const STEPS = 20, BISECTIONS = 5;
// Under the temporal modes, a splash's reactive weight per unit of its opacity (at most 1), and a
// streak's under the ground truth (aa=gt), which draws them before its resolve. A streak is
// sub-pixel and faint (opacities of 0.1 - 0.3), so at 1 the resolve kept most of its history there:
// the streaks lost a quarter of their energy against the ground truth in a storm and trailed. At 4
// they kept 0.83 - 0.87 of it, still broken into dashes, so under taa and taau they now draw after
// the resolve (post/AfterResolve.ts) and write no velocity.
const REACTIVE_GAIN = 4;

/** Nested volumes of drops, each a window onto a field that repeats every `size` metres, so
 * moving a window never moves a drop: it only trades drops at its faces. Near the eye a dense
 * small volume gives close streaks, a middle one and a wide one around the ground the view
 * looks at fill the depth; drops per volume fall as the volume grows (density ~ 1 / distance^3),
 * which keeps the screen coverage even across them. `share` is out of eight drops. A wide
 * window `floor`ed to the sea reaches 8 m below it, where sea splashes finish. The last
 * volume is a thin layer over the ground whose drops are not drawn: they stand for the rain
 * the drawn drops cannot resolve at a distance, and only splash. */
const CASCADES = [
	{ size: 10, height: 10, ahead: 4, share: 1, drawn: true },
	{ size: 30, height: 30, ahead: 12, share: 2, drawn: true },
	{ size: 84, height: 48, ahead: 36, share: 2, drawn: true, floor: - 8 },
	{ size: 36, height: 15, ahead: 36, share: 3, drawn: false, floor: - 8 },
];

// A drop is a small lens: it shows the scene behind it, inverted and squeezed, from a cone some
// 40 degrees around the line of sight, and reflects the sky off its rim. So rain has no light of
// its own: looking up it takes the sky's radiance, looking down the darker ground's and sea's, and
// only toward the key light (sun, moon) or in a flash does it glow.
const GROUND_ALBEDO = 0.14; // the island and sea under the rain, as seen through a drop
const SKY_SEEN = 0.9; // the sky inside the cone, a little under its cosine-weighted mean
// The sky reflected off the drop, lifted above Fresnel's few percent so that the streaks still
// read as faint lines against dark ground.
const REFLECTED = 0.25;
// Light refracted through a drop leaves within about 80 degrees of its path, mostly near it:
// a normalised lobe ( n + 1 ) / 2 pi cos^n, scaled by the share that is not reflected away.
const FORWARD_POWER = 4, FORWARD_SHARE = 0.6;

/** Radiance a falling drop sends along `view` (unit, from the eye), lit by the key light's
 * irradiance `key` from `keyDirection` and the sky: `sky` is its irradiance over pi lighting
 * the ground (a lightning flash brightens it as it does the island), `skySeen` the sky itself
 * as the drop shows and reflects it. `skyBias` adds to the sky's share of the cone behind. */
export const rainRadiance = Fn( ( [ view, sky, skySeen, keyDirection, key, skyBias ]: [ V3, V3, V3, V3, V3, F ] ) => {

	const share = clamp( view.y.mul( 0.6 ).add( 0.5 ).add( skyBias ), 0, 1 );
	const ground = sky.add( key.mul( max( keyDirection.y, 0 ).div( Math.PI ) ) ).mul( GROUND_ALBEDO );
	const behind = mix( ground, skySeen.mul( SKY_SEEN ), share );
	const forward = pow( max( dot( view, keyDirection ), 0 ), FORWARD_POWER ).mul( ( FORWARD_POWER + 1 ) / ( 2 * Math.PI ) * FORWARD_SHARE );
	return behind.add( skySeen.mul( REFLECTED ) ).add( key.mul( forward ) );

} ).setLayout( { name: 'ponlaiRainRadiance', type: 'vec3', inputs: [
	{ name: 'view', type: 'vec3' }, { name: 'sky', type: 'vec3' }, { name: 'skySeen', type: 'vec3' },
	{ name: 'keyDirection', type: 'vec3' }, { name: 'key', type: 'vec3' }, { name: 'skyBias', type: 'float' },
] } );

/** The light around the rain, in linear radiometric units (see `rainRadiance`). */
export interface RainLight {
	sky: V3;
	skySeen: V3;
	keyDirection: V3;
	key: V3;
}

export const lightRain = ( view: V3, light: RainLight, skyBias: F ): V3 =>
	rainRadiance( view, light.sky, light.skySeen, light.keyDirection, light.key, skyBias );

interface CascadeUniforms {
	centre: THREE.UniformNode<'vec3', THREE.Vector3>;
	// size, height, 1 / size, 1 / height
	shape: THREE.UniformNode<'vec4', THREE.Vector4>;
	// wind displacement / size (wrapped), fall clock / (10 height) (wrapped), drawn (1) or not (0)
	phase: THREE.UniformNode<'vec4', THREE.Vector4>;
}

export interface RainDropsOptions {

	collision: RainCollision;
	amount: THREE.UniformNode<'float', number>;
	wind: THREE.UniformNode<'vec2', THREE.Vector2>; // world x/z air velocity, m/s
	pixelAngle: THREE.UniformNode<'float', number>;
	seed?: number;

}

export class RainDrops {

	/** The drops' draws: the streaks and the splashes (empty after the resolve). */
	readonly group = new THREE.Group();
	/** After the resolve (Antialiasing.ts `AFTER_RESOLVE`), the streaks and the splashes, drawn in
	 * the after pass; else null. */
	readonly afterGroup: THREE.Group | null = null;
	readonly streaks: THREE.Mesh;
	readonly sprays: THREE.Mesh;
	readonly rings: THREE.Mesh;
	readonly compute: THREE.ComputeNode;
	// Per drop: the impact point and the splash's age (+10 at sea; < 0 none), for diagnostics.
	readonly splashes: THREE.StorageBufferNode<'vec4'>;
	private readonly cascades: CascadeUniforms[];
	private readonly eye = uniform( new THREE.Vector3() );
	private readonly displacement = new THREE.Vector2();
	private readonly materials: THREE.MeshBasicNodeMaterial[] = [];
	private readonly opacities: F[] = [];
	private readonly colours: ( () => V3 )[] = [];
	private light: RainLight = { sky: vec3( 0 ), skySeen: vec3( 0 ), keyDirection: vec3( 0, 1, 0 ), key: vec3( 0 ) };
	private readonly options: RainDropsOptions;
	private readonly forward = new THREE.Vector3();

	constructor( options: RainDropsOptions ) {

		this.options = options;
		const { collision, amount, wind, pixelAngle } = options;
		const base = ( ( options.seed ?? 8371 ) * 1000003 ) % 2147483647;
		this.cascades = CASCADES.map( ( c ) => ( {
			centre: uniform( new THREE.Vector3() ),
			shape: uniform( new THREE.Vector4( c.size, c.height, 1 / c.size, 1 / c.height ) ),
			phase: uniform( new THREE.Vector4( 0, 0, 0, c.drawn ? 1 : 0 ) ),
		} ) );
		const drops = instancedArray( RAIN_MAX_DROPS, 'vec4' ); // position, visibility
		const splashes = this.splashes = instancedArray( RAIN_MAX_DROPS, 'vec4' );
		const random = ( index: THREE.Node<'uint'>, k: number ) => hash( index.mul( 8 ).add( base + k ) ) as unknown as F;
		const fallTenths = ( index: THREE.Node<'uint'> ) => floor( random( index, 3 ).mul( FALL_TENTHS_RANGE ) ).add( FALL_TENTHS_MIN );

		// ---- simulation: every drop's state is a function of the clock, the wind's integrated
		// displacement and the camera, so a paused frame is bit-identical and no state drifts.
		const ground = collision.groundAt;
		const waterLevel = collision.waterLevel;
		const ceiling = float( collision.ceiling );
		this.compute = Fn( () => {

			const index = instanceIndex;
			// Interleaved, so a lower drop count thins every volume alike.
			const slot = index.mod( 8 ).toVar();
			const pick = <T extends THREE.Node>( values: T[] ) => {

				let result = values[ values.length - 1 ], end = 8;
				for ( let i = values.length - 2; i >= 0; i -- ) {

					end -= CASCADES[ i + 1 ].share;
					result = select( slot.lessThan( end ), values[ i ], result ) as unknown as T;

				}
				return result;

			};
			const centre = pick( this.cascades.map( ( c ) => c.centre as unknown as V3 ) ).toVar();
			const shape = pick( this.cascades.map( ( c ) => c.shape as unknown as V4 ) ).toVar();
			const phase = pick( this.cascades.map( ( c ) => c.phase as unknown as V4 ) ).toVar();
			const tenths = fallTenths( index ).toVar();
			const fall = tenths.mul( 0.1 ).toVar();
			const relative = vec3(
				fract( random( index, 0 ).add( phase.x ).sub( centre.x.mul( shape.z ) ).add( 0.5 ) ),
				fract( random( index, 1 ).sub( fract( tenths.mul( phase.z ) ) ).sub( centre.y.mul( shape.w ) ).add( 0.5 ) ),
				fract( random( index, 2 ).add( phase.y ).sub( centre.z.mul( shape.z ) ).add( 0.5 ) ),
			).sub( 0.5 ).toVar();
			const position = centre.add( relative.mul( shape.xyx ) ).toVar();
			const present = random( index, 4 ).lessThan( amount ).toVar();
			// Soft faces, so drops leaving a window fade rather than vanish; none right at the eye.
			const across = max( abs( relative.x ), abs( relative.z ) ).mul( 2 ).toVar();
			const face = max( across, abs( relative.y ).mul( 2 ) );
			const fade = float( 1 ).sub( smoothstep( 0.72, 1, face ) ).mul( smoothstep( NEAR_FADE[ 0 ], NEAR_FADE[ 1 ], length( position.sub( this.eye ) ) ) ).toVar();
			const visible = select( present, fade.mul( phase.w ), float( 0 ) ).toVar();
			const splash = vec4( 0, 0, 0, - 1 ).toVar();

			// Where did this drop meet the island or the sea? Walk its path back up to the highest
			// ground (or the window's top, where it entered) and search down for the first point
			// under the surface. Air carries it, so the path leans upwind with height.
			const lean = wind.mul( DRIFT ).div( fall ).toVar();
			const top = min( centre.y.add( shape.y.mul( 0.5 ) ), ceiling ).toVar();
			const along = ( y: F ) => position.xz.sub( lean.mul( y.sub( position.y ) ) ) as V2;
			If( position.y.lessThan( top ), () => {

				If( top.lessThan( ground( along( top ) ) ), () => {

					// The window opened inside the island: nothing fell here.
					visible.assign( 0 );

				} ).Else( () => {

					const span = top.sub( position.y ).toVar();
					const above = top.toVar();
					Loop( STEPS, ( { i } ) => {

						const y = top.sub( span.mul( float( i ).add( 1 ).div( STEPS ) ) ).toVar();
						If( y.lessThan( ground( along( y ) ) ), () => {

							const low = y.toVar(), high = above.toVar();
							for ( let b = 0; b < BISECTIONS; b ++ ) {

								const middle = low.add( high ).mul( 0.5 ).toVar();
								If( middle.lessThan( ground( along( middle ) ) ), () => {

									low.assign( middle );

								} ).Else( () => {

									high.assign( middle );

								} );

							}
							const hit = along( high ).toVar();
							const surface = ground( hit ).toVar();
							const age = high.sub( position.y ).div( fall ).toVar();
							const sea = surface.lessThan( waterLevel.add( 0.02 ) ).toVar();
							visible.assign( 0 );
							// Splashes thin out toward the window's faces instead of stopping at a line.
							const inside = across.lessThan( float( 1 ).sub( random( index, 5 ).mul( 0.4 ) ) );
							If( present.and( inside ).and( age.lessThan( select( sea, float( RING_LIFE ), float( SPRAY_LIFE ) ) ) ), () => {

								splash.assign( vec4( hit.x, surface, hit.y, age.add( select( sea, float( 10 ), float( 0 ) ) ) ) );

							} );
							Break();

						} );
						above.assign( y );

					} );

				} );

			} );
			drops.element( index ).assign( vec4( position, visible ) );
			splashes.element( index ).assign( splash );

		} )().compute( RAIN_MAX_DROPS, [ 64 ] ).setName( 'Rain.Drops' ) as unknown as THREE.ComputeNode;

		// ---- streaks: a drop's path during the exposure, a billboard turned about its velocity
		{

			const data = drops.toAttribute() as unknown as V4;
			const head = data.xyz;
			const velocity = vec3( wind.x.mul( DRIFT ), fallTenths( instanceIndex ).mul( - 0.1 ), wind.y.mul( DRIFT ) );
			const speed = length( velocity );
			const axis = velocity.div( speed );
			const toEye = cameraPosition.sub( head );
			const distance = length( toEye );
			const side = normalize( cross( axis, toEye ).add( vec3( 1e-6, 0, 0 ) ) );
			// Thinner than a pixel, a drop is drawn a pixel wide.
			const pixel = distance.mul( pixelAngle );
			const width = max( float( DIAMETER ), pixel.mul( 1.3 ) );
			// After the resolve a streak is drawn with one sample a pixel and nothing averages it: drawn
			// as it is, a line a pixel or so wide steps and breaks along its length where the multisampled
			// and supersampled images show it whole. It is prefiltered instead: widened to the width of
			// its profile convolved with a pixel (the variances add: a parabolic profile's is width^2 / 20,
			// a pixel's 1 / 12) and faded by the ratio, so its energy holds.
			const drawn = AFTER_RESOLVE ? sqrt( width.mul( width ).add( pixel.mul( pixel ).mul( 20 / 12 ) ) ) : width;
			const streak = speed.mul( EXPOSURE ).mul( random( instanceIndex, 5 ).mul( 0.5 ).add( 0.75 ) ).add( width );
			const shown = select( data.w.greaterThan( 0.004 ), float( 1 ), float( 0 ) );
			// (after the resolve there is no depth to test: bindOpaqueDepth's cover hides them)
			const material = this.makeMaterial( 'rainStreaks', ! AFTER_RESOLVE, AFTER_RESOLVE );
			material.positionNode = head.add( axis.mul( positionGeometry.y.sub( 0.5 ).mul( streak ) )
				.add( side.mul( positionGeometry.x.mul( drawn ) ) ).mul( shown ) );
			// Some drops catch the light, most are faint.
			const glint = random( instanceIndex, 6 ).pow( 2 ).mul( 1.3 ).add( 0.35 );
			// Each drawn drop stands for many unseen ones, so sub-pixel coverage is lifted (power < 1).
			const lift = pow( float( DIAMETER ).div( width ), 0.6 );
			const spread = AFTER_RESOLVE ? width.div( drawn ) : float( 1 );
			const strength = varying( data.w.mul( lift ).mul( spread ).mul( glint ) as F, 'vRainStreak' );
			const q = uv();
			const across = abs( q.x.sub( 0.5 ) ).mul( 2 );
			const profile = float( 1 ).sub( across.mul( across ) ).mul( smoothstep( 0, 0.2, q.y ) ).mul( smoothstep( 0, 0.06, float( 1 ).sub( q.y ) ) );
			// Heavier rain falls in bigger drops: brighter streaks in a storm.
			this.setOpacity( material, strength.mul( profile ).mul( amount.mul( 0.6 ).add( 0.5 ) ) );
			this.setColour( material, () => this.shade( head ) );
			this.streaks = this.makeMesh( material, new THREE.PlaneGeometry( 1, 1 ) );

		}

		// ---- splashes on land and from the sea: a crown and its droplets, seen as they fly
		{

			const data = splashes.toAttribute() as unknown as V4;
			const base = data.xyz;
			const sea = data.w.greaterThanEqual( 10 );
			const age = data.w.sub( select( sea, float( 10 ), float( 0 ) ) );
			const t = age.div( select( sea, float( SEA_SPRAY_LIFE ), float( SPRAY_LIFE ) ) );
			const live = data.w.greaterThanEqual( 0 ).and( t.lessThan( 1 ) );
			const toEye = cameraPosition.sub( base );
			const distance = length( toEye );
			const view = toEye.div( distance );
			// World up turned to face the eye: the sprite stands up from the ground and still faces
			// a camera looking down. e is how far the eye looks down on it.
			const right = normalize( cross( vec3( 0, 1, 0 ), view ).add( vec3( 1e-6, 0, 0 ) ) );
			const up = cross( view, right );
			const e = clamp( view.y, 0.05, 1 );
			const size = float( SPRAY_SIZE ).mul( select( sea, float( 0.55 ), float( 1 ) ) ).mul( random( instanceIndex, 6 ).mul( 0.6 ).add( 0.7 ) )
				.mul( amount.mul( 0.4 ).add( 0.7 ) );
			// Below a few pixels a splash keeps its energy spread over that many pixels instead.
			const drawn = max( size, distance.mul( pixelAngle ).mul( 7 ) );
			const energy = size.div( drawn ).pow( 2 );
			// The impact point sits low in the sprite seen from the side, central seen from above.
			const origin = e.mul( 0.38 ).add( - 0.42 );
			// Not depth-tested: a splash sinks a little way into the ground it lands on (see
			// bindOpaqueDepth), which the scene target's depth, holding the ground, would cut off.
			const material = this.makeMaterial( 'rainSprays', false, AFTER_RESOLVE );
			material.positionNode = base.add( right.mul( positionGeometry.x ).add( up.mul( positionGeometry.y.sub( origin ) ) )
				.mul( drawn ).mul( select( live, float( 1 ), float( 0 ) ) ) );
			const vT = varying( t, 'vSprayAge' ), vE = varying( e, 'vSprayTilt' ), vSeed = varying( random( instanceIndex, 7 ), 'vSpraySeed' );
			const vOrigin = varying( origin, 'vSprayOrigin' ), vEnergy = varying( energy, 'vSprayEnergy' );
			const vPixels = varying( drawn.div( distance.mul( pixelAngle ) ), 'vSprayPixels' );
			const shape = Fn( () => {

				// In units of the drawn sprite, centred on the impact.
				const q = uv();
				const p = vec2( q.x.sub( 0.5 ), q.y.sub( 0.5 ).sub( vOrigin ) ).toVar();
				// A few pixels across, the crown and droplets would only alias: a splash that small
				// reads as a brief glint over the impact, drawn at the enlarged size.
				const glint = exp( length( p.sub( vec2( 0, 0.07 ) ).div( vec2( 0.08, 0.14 ) ) ).pow( 2 ).negate() ).mul( pow( float( 1 ).sub( vT ), 1.5 ) );
				return mix( glint, sprayShape( p, vT, vE, vSeed ), smoothstep( 12, 40, vPixels ) );

			} )() as unknown as F;
			this.setOpacity( material, shape.mul( vEnergy ) );
			// The crown's walls and the droplets thrown up catch the sky above, and the churned water
			// scatters a little more than a single clear drop.
			this.setColour( material, () => this.shade( base, 0.6 ).mul( 1.3 ) );
			this.sprays = this.makeMesh( material, new THREE.PlaneGeometry( 1, 1 ) );

		}

		// ---- rings spreading on the water where a drop went in
		{

			const data = splashes.toAttribute() as unknown as V4;
			const t = data.w.sub( 10 ).div( RING_LIFE );
			const live = data.w.greaterThanEqual( 10 ).and( t.lessThan( 1 ) );
			const base = data.xyz.add( vec3( 0, 0.01, 0 ) );
			const distance = length( cameraPosition.sub( base ) );
			const size = float( RING_SIZE ).mul( random( instanceIndex, 6 ).mul( 0.7 ).add( 0.5 ) );
			const drawn = max( size, distance.mul( pixelAngle ).mul( 6 ) );
			const material = this.makeMaterial( 'rainRings', false, AFTER_RESOLVE );
			material.positionNode = base.add( vec3( positionGeometry.x, 0, positionGeometry.y.negate() ).mul( drawn ).mul( select( live, float( 1 ), float( 0 ) ) ) );
			const vT = varying( t, 'vRingAge' );
			const vEnergy = varying( size.div( drawn ).pow( 2 ), 'vRingEnergy' );
			const wave = ringWave( length( uv().sub( 0.5 ) ).mul( 2 ), vT );
			// Crests catch the sky, troughs face the darker sea: light and shade, not a white line.
			this.setOpacity( material, abs( wave ).mul( vEnergy ).mul( 0.16 ) );
			this.setColour( material, () => this.shade( base, 1 ).mul( select( wave.greaterThan( 0 ), float( 1.2 ), float( 0.35 ) ) ) );
			this.rings = this.makeMesh( material, new THREE.PlaneGeometry( 1, 1 ) );

		}
		this.group.name = 'weatherRain';
		// (after the resolve they draw in a pass of their own: `afterGroup`)
		if ( AFTER_RESOLVE ) {

			this.afterGroup = new THREE.Group();
			this.afterGroup.name = 'weatherRainAfter';
			this.afterGroup.add( this.rings, this.streaks, this.sprays );

		} else this.group.add( this.rings, this.streaks, this.sprays );

	}

	private makeMaterial( name: string, depthTest: boolean, after = false ): THREE.MeshBasicNodeMaterial {

		const material = new THREE.MeshBasicNodeMaterial( { transparent: true, depthTest, depthWrite: false, side: THREE.DoubleSide } );
		material.forceSinglePass = true;
		material.name = name;
		// under the temporal modes the splashes (and the streaks under aa=gt) draw in the overlay pass,
		// which keeps the velocity under them; only their reactive weight counts (Motion.ts): their
		// final opacity (`output`, set before the MRT's outputs are built) times REACTIVE_GAIN, which
		// the overlay MRT's max keeps where drops overlap, so the resolve trusts its history less where
		// a drop covers the pixel. After the resolve the streaks write their colour alone.
		if ( ! after ) material.mrtNode = velocityOutput( overlayVelocity( clamp( output.a.mul( REACTIVE_GAIN ), 0, 1 ) ) );
		this.materials.push( material );
		return material;

	}

	private setOpacity( material: THREE.MeshBasicNodeMaterial, opacity: F ): void {

		const index = this.materials.indexOf( material );
		this.opacities[ index ] = opacity;
		material.opacityNode = opacity;

	}

	private setColour( material: THREE.MeshBasicNodeMaterial, colour: () => V3 ): void {

		this.colours[ this.materials.indexOf( material ) ] = colour;
		material.colorNode = colour();

	}

	private makeMesh( material: THREE.Material, geometry: THREE.BufferGeometry ): THREE.Mesh {

		const mesh = new THREE.Mesh( geometry, material );
		mesh.count = RAIN_MAX_DROPS;
		mesh.frustumCulled = false;
		mesh.castShadow = mesh.receiveShadow = false;
		return mesh;

	}

	/** Light the water at `point` sends toward the eye (see `rainRadiance`); `skyBias` raises
	 * the sky's share for splashes and rings, whose curved walls face up. */
	private shade( point: V3, skyBias = 0 ): V3 {

		const view = normalize( point.sub( cameraPosition ) );
		return lightRain( view, this.light, float( skyBias ) );

	}

	/** The light around the drops (see `RainLight`). */
	setLight( light: RainLight ): void {

		this.light = light;
		this.materials.forEach( ( material, i ) => {

			material.colorNode = this.colours[ i ]();
			material.needsUpdate = true;

		} );

	}

	/** Reject drops and splashes behind the island and the figures, using the opaque pass's
	 * resolved linear view z. Splashes lie on surfaces, so they get a little slack. The sea hides
	 * the splashes behind it outright, as the depth test did (`surfaceDepth`: the nearest
	 * surface's view z, the sea's included); streaks keep the depth test. After the resolve there is
	 * no depth to test: the streaks test the nearest surface (the sea's included) and the splashes
	 * as here, at the four render texels around their pixel (AfterResolve.ts). */
	bindOpaqueDepth( linearDepth: THREE.Texture, surfaceDepth: THREE.Texture ): void {

		const depth = texture( linearDepth ), surface = texture( surfaceDepth );
		const size = vec2( textureSize( depth ) as unknown as THREE.Node<'ivec2'> );
		const pixel = ivec2( clamp( screenCoordinate.xy, vec2( 0 ), size.sub( 1 ) ) );
		const surfaceZ = textureLoad( depth, pixel ).r;
		const nearestZ = textureLoad( surface, pixel ).r;
		// View z is negative in front of the camera: a nearer fragment has the larger value.
		const ahead = positionView.z.sub( surfaceZ );
		const behindSea = nearestZ.greaterThan( surfaceZ ).and( positionView.z.lessThan( nearestZ ) );
		this.materials.forEach( ( material, i ) => {

			const slack = material === this.streaks.material ? 0 : 0.12;
			if ( AFTER_RESOLVE ) {

				const streak = material === this.streaks.material, spray = material === this.sprays.material;
				material.opacityNode = this.opacities[ i ].mul( afterResolveCover( surfaceDepth, ( texel, nearest ) => {

					if ( streak ) return smoothstep( 0, 0.06, positionView.z.sub( nearest ) );
					const opaque = textureLoad( depth, texel ).r;
					const shown = smoothstep( - slack, 0.06 - slack * 0.5, positionView.z.sub( opaque ) );
					return spray ? select( nearest.greaterThan( opaque ).and( positionView.z.lessThan( nearest ) ), float( 0 ), shown ) : shown;

				} ) );
				material.needsUpdate = true;
				return;

			}
			const visible = smoothstep( - slack, 0.06 - slack * 0.5, ahead );
			material.opacityNode = this.opacities[ i ].mul( material === this.sprays.material ? select( behindSea, float( 0 ), visible ) : visible );
			material.needsUpdate = true;

		} );

	}

	setCount( count: number ): void {

		const n = Math.max( 0, Math.min( RAIN_MAX_DROPS, Math.floor( count ) ) );
		this.streaks.count = this.sprays.count = this.rings.count = n;
		( this.compute as unknown as { count: number } ).count = Math.max( n, 1 );

	}

	get count(): number {

		return this.streaks.count;

	}

	/** Place the volumes for this camera and wrap the clocks. `displacement` is the wind's
	 * integrated travel (m), `time` the scene clock (s). */
	update( camera: THREE.Camera, time: number, displacement: THREE.Vector2 ): void {

		const eye = camera.position;
		this.eye.value.copy( eye );
		camera.getWorldDirection( this.forward );
		const forward = this.forward;
		const level = this.options.collision.waterLevel.value;
		this.displacement.copy( displacement ).multiplyScalar( DRIFT );
		CASCADES.forEach( ( c, i ) => {

			const u = this.cascades[ i ];
			if ( c.floor === undefined ) u.centre.value.copy( eye ).addScaledVector( forward, c.ahead );
			else {

				// Around the ground the camera looks at, or straight ahead toward the horizon.
				const flat = Math.hypot( forward.x, forward.z ) || 1;
				const reach = forward.y < - 0.02 ? Math.min( ( eye.y - level ) / - forward.y * flat, c.ahead ) : c.ahead;
				u.centre.value.set( eye.x + forward.x / flat * reach, level + c.floor + c.height / 2, eye.z + forward.z / flat * reach );

			}
			const wrap = ( v: number, period: number ) => ( ( v % period ) + period ) % period / period;
			u.phase.value.set( wrap( this.displacement.x, c.size ), wrap( this.displacement.y, c.size ), wrap( time, 10 * c.height ), c.drawn ? 1 : 0 );

		} );

	}

	dispose(): void {

		for ( const mesh of [ this.streaks, this.sprays, this.rings ] ) mesh.geometry.dispose();
		for ( const material of this.materials ) material.dispose();

	}

}

/** Opacity of a splash sprite at local point p (units of the splash size, the impact at the
 * origin, y up the screen), normalised age t, tilt e (sine of the eye's elevation) and seed:
 * a crown wall rising round the impact and droplets thrown out of it, each traced on its
 * ballistic arc and projected for the view. */
const sprayShape = ( p: V2, t: F, e: F, seed: F ): F => {

	const c = sqrt( float( 1 ).sub( e.mul( e ) ) );
	const fade = pow( float( 1 ).sub( t ), 1.2 );
	// The crown: a short wall of water whose ring widens as it rises and falls back.
	const radius = sqrt( t ).mul( 0.2 ).add( 0.05 ).toVar();
	const height = sin( min( t.mul( 1.4 ), 1 ).mul( Math.PI ) ).mul( 0.1 ).toVar();
	const around = clamp( p.x.div( radius ), - 1, 1 );
	const depth = sqrt( float( 1 ).sub( around.mul( around ) ) ).mul( radius ).mul( e ).toVar(); // ring's half-height on screen
	const inside = float( 1 ).sub( smoothstep( radius.mul( 0.85 ), radius, abs( p.x ) ) ).toVar();
	// Only the crown's rim catches the light, broken into the points it throws droplets from.
	const points = sin( around.mul( 7 ).add( seed.mul( 6.28 ) ) ).mul( 0.5 ).add( 0.5 ).pow( 2 ).mul( 0.7 ).add( 0.3 );
	const rim = ( y0: F ) => exp( p.y.sub( y0.add( height.mul( c ) ) ).div( 0.01 ).pow( 2 ).negate() );
	const front = rim( depth.negate() ).mul( points ).mul( 0.5 ).mul( inside );
	const back = rim( depth ).mul( points ).mul( 0.25 ).mul( inside );
	const crown = max( front, back ).mul( smoothstep( 0.005, 0.03, height ) );
	// Droplets: thrown out at different speeds and angles, they arc up and fall back.
	const droplets = float( 0 ).toVar();
	for ( let k = 0; k < 7; k ++ ) {

		const h1 = fract( seed.mul( 13.37 + k * 7.13 ).add( k * 0.618 ) );
		const h2 = fract( seed.mul( 29.71 + k * 3.31 ).add( k * 0.271 ) );
		const angle = h1.add( k / 7 ).mul( 2 * Math.PI );
		const out = h2.mul( 0.25 ).add( 0.2 ).mul( t );
		const rise = h2.mul( 0.5 ).add( 0.45 ).mul( t ).sub( t.mul( t ).mul( 0.85 ) );
		const at = vec2( out.mul( sin( angle.add( Math.PI / 2 ) ) ), out.mul( sin( angle ) ).mul( e ).negate().add( max( rise, 0 ).mul( c ) ) );
		const d = length( p.sub( at ) );
		const r = float( 0.022 ).mul( float( 1 ).sub( t.mul( 0.4 ) ) );
		droplets.addAssign( exp( d.div( r ).pow( 2 ).negate() ).mul( smoothstep( - 0.02, 0.01, rise ) ) );

	}
	// The impact itself: a brief flash.
	const spark = exp( length( p ).div( 0.03 ).pow( 2 ).negate() ).mul( float( 1 ).sub( smoothstep( 0, 0.3, t ) ) );
	return clamp( crown.add( droplets ).add( spark ), 0, 1.5 ).mul( fade ) as F;

};

/** A drop's ring at normalised radius r and age t: a front running out from the impact with
 * a wave and a half behind it, dying as it spreads. After Sébastien Lagarde, "Water drop 2b -
 * Dynamic rain and its effects" (2013): sin( clamp( 9 ( t - r ), 0, 3 ) pi ), scaled by
 * ( 1 - r ) and by the drop's remaining life. */
const ringWave = ( r: F, t: F ): F => {

	const front = pow( t, 0.75 );
	const phase = clamp( front.sub( r ).mul( 9 ), 0, 3 );
	const life = clamp( float( 1.15 ).sub( t ), 0, 1 );
	return sin( phase.mul( Math.PI ) ).mul( clamp( float( 1 ).sub( r ), 0, 1 ) ).mul( life ).mul( life ) as F;

};
