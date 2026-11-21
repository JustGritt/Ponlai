import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, uniform, vec2, vec3, vec4, float, floor, fract, sin, cos, dot, cross, normalize, length,
	abs, max, clamp, mix, smoothstep, exp, select, acos, atan,
} from 'three/tsl';
import { RainCollision } from './RainCollision.ts';
import { DRIFT, MEAN_FALL, RainDrops, lightRain, type RainLight } from './RainDrops.ts';
import { AFTER_RESOLVE } from '../post/Antialiasing.ts';
import type { SceneLatents } from '../post/ExtraPass.ts';

export { RAIN_MAX_DROPS } from './RainDrops.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// Rain: GPU streaks and splashes, a distant veil and ripples on the water.
//
// Rain: light streaks (GPU drops in nested camera volumes, RainDrops.ts; a distant veil of streaks
// in the post pass) plus analytic ripples on the water normal. Drops are a stateless function of
// the clock, the wind's integrated travel and the camera, so a paused frame is bit-identical; they
// stop on the first surface in the collision map (RainCollision.ts, the island's rest pose) and
// splash there (crown and droplets on land, rings on the sea). `registerLatent` registers its draws and kernels with the frame graph's `rain`
// pass, whose warm-up (post/graph/FrameGraph.ts) compiles them at startup, so a weather change
// builds no pipeline.
// Under taa and taau (Antialiasing.ts AFTER_RESOLVE) the drops, their splashes and the veil draw
// after the temporal resolve, at the canvas's resolution and unjittered (`afterGroup`, and the veil
// in the canvas pass: post/AfterResolve.ts), and their widths follow the canvas's pixel
// (`dropPixelAngle`). The ground truth (aa=gt) keeps them before its resolve: the drops in the
// overlay pass, which keeps the motion of the surface behind them and takes their opacity as the
// reactive weight (RainDrops.ts, Motion.ts), and the veil in the composite, which writes its
// opacity as the resolve's per-pixel `veilReactive` (Pipeline.ts), at the render's pixel
// (`pixelAngle`, which is the canvas's there).

// The veil: shells around the viewer at these distances (metres), each carrying a sheet of
// streaks. Mapped on the sphere about the rain's own direction, streaks converge on its
// vanishing points as in perspective, and a shell shows only where the scene lies beyond it.
const VEIL_SHELLS = [ 48, 80, 130 ];
// Fewer shells keep the nearest and the farthest, so the veil still spans the same depth.
const VEIL_SETS: Record<number, number[]> = { 1: [ 80 ], 2: [ 48, 130 ], 3: VEIL_SHELLS };
const DROP_WIDTH = 0.0016; // metres
const STREAK_LENGTH = 0.4; // metres travelled during the eye's exposure
const STREAK_SPACING = 0.22; // metres between columns of drops on a shell

const FALL_SPEED = MEAN_FALL;

export interface RainOptions {

	count?: number;
	seed?: number;
	/** the highest height a drop can meet (m): the rain's collision map holds nothing above it */
	ceiling: number;

}

/** Quality settings of the rain's shading, read when the water and post materials are built, so
 * a change takes a reload (or a rebuild of those materials): the veil's shells (1 - 3) and the
 * sea's ripple layers (1 - 2). The defaults are the high tier's. The drop count is separate
 * (`Rain.setQuality`) and changes at once. */
export const rainShading = { veilShells: 3, rippleLayers: 2 };

export function setRainShading( { veilShells = rainShading.veilShells, rippleLayers = rainShading.rippleLayers }: Partial<typeof rainShading> ): void {

	rainShading.veilShells = THREE.MathUtils.clamp( Math.round( veilShells ), 1, VEIL_SHELLS.length );
	rainShading.rippleLayers = THREE.MathUtils.clamp( Math.round( rippleLayers ), 1, 2 );

}

// Original analytic impact rings. A cell holds one short-lived ring whose entire support
// stays inside that cell, so two staggered grids need no nine-neighbour search. Ring centres
// change only between faded-out events; the result is deterministic for frozen captures.
const rainHash = Fn( ( [ p ]: [ V2 ] ) => {

	return fract( sin( dot( p, vec2( 71.73, 139.91 ) ) ).mul( 19687.317 ) );

} ).setLayout( { name: 'ponlaiRainHash', type: 'float', inputs: [ { name: 'p', type: 'vec2' } ] } );

/** World-space height gradient, in metres/metre. Add J-transpose times this gradient to a
 * displaced ocean's parameter-space height gradient before constructing its normal. The
 * caller supplies its existing world-space pixel footprint; no derivatives are taken here. */
export const rainRippleGradient = Fn( ( [ worldXZ, time, rain, footprint ]: [ V2, F, F, F ] ) => {

	const gradient = vec2( 0 ).toVar();
	const filter = float( 1 ).sub( smoothstep( 0.018, 0.075, footprint ) ).toVar();
	If( rain.greaterThan( 0.0001 ).and( filter.greaterThan( 0.001 ) ), () => {

		// Read when a material using this is built (see `rainShading`).
		Loop( rainShading.rippleLayers, ( { i } ) => {

			const layer = float( i ).toVar();
			const grid = worldXZ.mul( 0.625 ).add( vec2( 0.37, 0.61 ).mul( layer ) ).toVar();
			const cell = floor( grid ).toVar();
			const key = cell.add( vec2( 13.9, 7.3 ).mul( layer ) ).toVar();
			const phase = time.mul( mix( 0.83, 1.07, layer ) ).add( rainHash( key ) ).toVar();
			const age = fract( phase ).toVar();
			const event = floor( phase ).toVar();
			const center = vec2( rainHash( key.add( event.mul( 2.31 ) ) ), rainHash( key.add( vec2( 6.1, 3.7 ) ).add( event.mul( 1.73 ) ) ) )
				.mul( 0.24 ).add( 0.38 ).toVar();
			const offset = fract( grid ).sub( center ).mul( 1.6 ).toVar();
			const distance = max( length( offset ), 1e-4 ).toVar();
			const ring = distance.sub( age.mul( 0.39 ) ).toVar();
			const width = age.mul( 0.018 ).add( 0.026 ).toVar();
			const invWidth2 = float( 1 ).div( width.mul( width ) ).toVar();
			const envelope = smoothstep( 0.04, 0.16, age ).mul( float( 1 ).sub( age ).pow( 2 ) )
				.mul( float( 1 ).sub( smoothstep( 0.46, 0.53, distance ) ) ).toVar();
			const wave = ring.mul( 100 ).toVar();
			const slope = exp( ring.mul( ring ).mul( invWidth2 ).negate() )
				.mul( cos( wave ).mul( 100 ).sub( ring.mul( invWidth2 ).mul( 2 ).mul( sin( wave ) ) ) )
				.mul( envelope ).mul( 0.0003 );
			gradient.addAssign( offset.div( distance ).mul( slope ) );

		} );

	} );
	return gradient.mul( clamp( rain, 0, 1 ) ).mul( filter );

} ).setLayout( { name: 'ponlaiRainRippleGradient', type: 'vec2', inputs: [
	{ name: 'worldXZ', type: 'vec2' }, { name: 'time', type: 'float' },
	{ name: 'rain', type: 'float' }, { name: 'footprint', type: 'float' },
] } );

/** The rain: GPU drops that fall through the scene and splash where they land (RainDrops,
 * with collision against RainCollision's height map), a veil of distant streaks laid over
 * the composite, and the analytic impact rings on the water normal (`rainRippleGradient`). */
export class Rain {

	readonly uniforms = {
		time: uniform( 0 ),
		rain: uniform( 0 ),
		wind: uniform( new THREE.Vector2() ), // world-space x/z air velocity, m/s
		light: uniform( new THREE.Color( 0, 0, 0 ) ), // linear scene radiance, supplied by the shared sky/key
		waterLevel: uniform( 0 ),
	};
	// Angle subtended by one pixel, for the veil's and the drops' energy-preserving widths: the
	// render's pixel for the veil, the pixel the streaks are drawn at for the drops.
	readonly pixelAngle = uniform( 0.001 );
	readonly dropPixelAngle = uniform( 0.001 );
	readonly collision: RainCollision;
	readonly drops: RainDrops;
	/** The drops' streak draw; `group` holds it with the splashes (after the resolve, `afterGroup`
	 * holds it alone). */
	readonly mesh: THREE.Mesh;
	readonly group: THREE.Group;
	readonly afterGroup: THREE.Group | null;
	private light: RainLight;

	constructor( { count = 32768, seed = 8371, ceiling }: RainOptions ) {

		this.collision = new RainCollision( ceiling );
		if ( ! Number.isFinite( count ) || count < 0 ) throw new RangeError( 'Rain count must be finite and non-negative' );
		if ( ! Number.isFinite( seed ) ) throw new RangeError( 'Rain seed must be finite' );
		this.drops = new RainDrops( { collision: this.collision, amount: this.uniforms.rain, wind: this.uniforms.wind, pixelAngle: this.dropPixelAngle, seed } );
		this.mesh = this.drops.streaks;
		this.group = this.drops.group;
		this.afterGroup = this.drops.afterGroup;
		this.group.visible = false;
		this.setQuality( count );
		this.light = { sky: this.uniforms.light.rgb, skySeen: this.uniforms.light.rgb, keyDirection: vec3( 0, 1, 0 ), key: vec3( 0 ) };
		this.drops.setLight( this.light );

	}

	/** Draw the island's collision heights; call once the island and its plants exist. */
	bakeCollision( renderer: THREE.WebGPURenderer, roots: THREE.Object3D[] ): void {

		this.collision.bake( renderer, roots );

	}

	/** Bind the resolved linear view-Z textures of the opaque pass and of the nearest surface (the
	 * sea's included). The drops draw into the scene's target after the water; these tests reject
	 * drops hidden by island/character pixels with a little slack for the splashes on them. */
	bindOpaqueDepth( linearDepth: THREE.Texture, surfaceDepth: THREE.Texture ): void {

		this.drops.bindOpaqueDepth( linearDepth, surfaceDepth );

	}

	/** Light the rain from the displayed sky irradiance over pi (including the GPU interpolation,
	 * so it stays in step with sky transitions, and the lightning's fill) and the key light
	 * (direction toward it, linear irradiance), without an emissive floor. `skySeen` is the sky
	 * as the drops show it, if it differs (default `sky`). Null restores the `light` uniform.
	 * Drops take the light around them (`rainRadiance`): grey under a dark deck and at night,
	 * bright only toward the key or in a flash. */
	setLightNode( sky: V3 | null, keyDirection?: V3, keyColor?: V3, skySeen?: V3 ): void {

		const ambient = sky ?? this.uniforms.light.rgb;
		this.light = { sky: ambient, skySeen: skySeen ?? ambient, keyDirection: keyDirection ?? vec3( 0, 1, 0 ), key: keyColor ?? vec3( 0 ) };
		this.drops.setLight( this.light );

	}

	/** The `background` radiance along a view direction seen through the falling rain, for scene
	 * content at `distance` metres (very large for the sky), and in its alpha the veil's opacity
	 * there (0 without rain), which the composite hands to the temporal resolve as its per-pixel
	 * reactive weight (Pipeline.ts `veilReactive`). Thin drops are widened to a pixel at reduced
	 * coverage; each covers the background with the light it takes from around it, so the veil
	 * only lightens what is darker than the rain and dims what is brighter. */
	veil( direction: V3, distance: F, background: V3 ): V4 {

		const coverage = this.veilCoverage( direction, distance );
		const output = vec4( background, 0 ).toVar();
		If( this.uniforms.rain.greaterThan( 0.001 ), () => {

			output.assign( vec4( mix( background, this.veilLight( direction ), coverage ), coverage ) );

		} );
		return output;

	}

	/** The light a veil's streak sends along a view direction (its colour where it covers it). */
	veilLight( direction: V3 ): V3 {

		return lightRain( direction, this.light, float( 0 ) );

	}

	/** The veil's coverage along a view direction in front of content at `distance` (see `veil`):
	 * 0 without rain. */
	veilCoverage( direction: V3, distance: F ): F {

		const u = this.uniforms;
		const coverage = float( 0 ).toVar();
		const result = float( 0 ).toVar();
		const shells = VEIL_SETS[ rainShading.veilShells ];
		// (after the resolve, at every canvas pixel, nothing for what lies before the nearest shell, and
		// a shell only past it; before it, as it always was)
		const past = ( radius: number, work: () => void ) => AFTER_RESOLVE ? If( distance.greaterThan( radius * 0.9 ), work ) : work();
		If( u.rain.greaterThan( 0.001 ), () => {

			past( Math.min( ...shells ), () => {

				// Axis toward where the rain comes from, tilted by the wind; a frame around it.
				const source = normalize( vec3( u.wind.x.mul( - DRIFT ), FALL_SPEED, u.wind.y.mul( - DRIFT ) ) ).toVar();
				const side = normalize( cross( source, vec3( 0, 0, 1 ) ) ).toVar();
				const other = cross( source, side ).toVar();
				const cosine = clamp( dot( direction, source ), - 1, 1 ).toVar();
				const polar = acos( cosine ).toVar(); // 0 toward the source, pi along the fall
				const azimuth = atan( dot( direction, other ), dot( direction, side ) ).div( 2 * Math.PI ).add( 0.5 ).toVar();
				// Columns crowd toward both vanishing points; fade them out before they alias.
				const away = smoothstep( 0.12, 0.45, abs( sin( polar ) ) ).toVar();
				const amount = clamp( u.rain, 0, 1 ).toVar();
				for ( const radius of shells ) past( radius, () => {

					// Each shell keeps its own pattern whichever set it is drawn in.
					const index = VEIL_SHELLS.indexOf( radius );
					const columns = Math.round( 2 * Math.PI * radius / STREAK_SPACING );
					const column = azimuth.mul( columns ).toVar();
					const cell = floor( column ).toVar();
					const seed = vec2( cell, index * 17.13 ).toVar();
					// Arc length along the fall, scrolling at the fall speed with a per-column offset.
					const fall = polar.mul( radius ).sub( u.time.mul( FALL_SPEED * ( 0.9 + 0.04 * index ) ) ).sub( rainHash( seed ).mul( 37 ) ).toVar();
					const row = floor( fall.div( STREAK_LENGTH * 2.6 ) ).toVar();
					const along = fract( fall.div( STREAK_LENGTH * 2.6 ) ).mul( 2.6 ).toVar();
					const present = rainHash( vec2( cell.add( row.mul( 0.618 ) ), row.add( index * 3.7 ) ) ).lessThan( amount.mul( 0.8 ) );
					const shown = smoothstep( radius * 0.9, radius * 1.1, distance );
					// (after the resolve, at every canvas pixel, the rest only where a streak may show: past
					// its shell, present and along its length; before it, as it always was)
					const add = ( work: () => void ) => AFTER_RESOLVE ? If( present.and( along.lessThan( 1 ) ).and( shown.greaterThan( 0 ) ), work ) : work();
					add( () => {

						const centre = rainHash( seed.add( row ) ).mul( 0.6 ).add( 0.2 );
						// Angular half-width of the streak, at least half a pixel; energy kept by the ratio. After
						// the resolve (one sample a canvas pixel, nothing averaging it) at least 0.8 of a canvas
						// pixel: a streak half a pixel wide stepped and broke along its length.
						const drop = DROP_WIDTH / radius;
						const halfWidth = ( AFTER_RESOLVE ? max( float( drop ), this.dropPixelAngle.mul( 0.8 ) ) : max( float( drop ), this.pixelAngle.mul( 0.5 ) ) ).toVar();
						const across = abs( fract( column ).sub( centre ) ).mul( 2 * Math.PI / columns ).mul( sin( polar ) );
						const profile = float( 1 ).sub( smoothstep( halfWidth.mul( 0.4 ), halfWidth, across ) ).mul( float( drop ).div( halfWidth ) );
						// Each drop has its own brightness: some catch the light, most are faint.
						const glint = rainHash( seed.add( row.mul( 1.31 ) ).add( 5.1 ) ).pow( 2 ).mul( 1.6 ).add( 0.25 );
						const streak = smoothstep( 0, 0.15, along ).mul( float( 1 ).sub( smoothstep( 0.6, 1, along ) ) ).mul( glint );
						result.addAssign( AFTER_RESOLVE ? profile.mul( streak ).mul( shown ) : select( present, profile.mul( streak ).mul( shown ), float( 0 ) ) );

					} );

				} );
				coverage.assign( clamp( result.mul( away ).mul( amount.mul( 0.5 ).add( 0.5 ) ), 0, 1 ) );

			} );

		} );
		return coverage;

	}

	setQuality( count: number ): void {

		if ( ! Number.isFinite( count ) || count < 0 ) throw new RangeError( 'Rain count must be finite and non-negative' );
		this.drops.setCount( count );

	}

	/** Advance the drops to `time` for this camera and run their simulation. `displacement`
	 * is the wind's integrated travel (m), which carries the drops. `viewportHeight` is the
	 * render's height, `dropHeight` the height the streaks are drawn at (the canvas's after the
	 * resolve). */
	update( renderer: THREE.WebGPURenderer, camera: THREE.Camera, time: number, displacement: THREE.Vector2, viewportHeight = 0, dropHeight = viewportHeight ): void {

		if ( ! Number.isFinite( time ) ) throw new RangeError( 'Rain time must be finite' );
		this.uniforms.time.value = time;
		const fov = ( camera as THREE.PerspectiveCamera ).fov;
		const angle = ( height: number ) => 2 * Math.tan( fov * Math.PI / 360 ) / height;
		if ( fov && viewportHeight > 0 ) this.pixelAngle.value = angle( viewportHeight );
		if ( fov && dropHeight > 0 ) this.dropPixelAngle.value = angle( dropHeight );
		this.collision.waterLevel.value = this.uniforms.waterLevel.value;
		const visible = this.drops.count > 0 && this.uniforms.rain.value > 0.0001;
		this.group.visible = this.mesh.visible = visible;
		if ( ! visible ) return;
		this.drops.update( camera, time, displacement );
		renderer.compute( this.drops.compute );

	}

	/** Registers the drops and splashes with the frame graph's rain pass, and their simulation
	 * kernel, so the graph's warm-up compiles them before the first rain (after the resolve, the
	 * streaks with the after pass, `rainAfter`). Call it after `bindOpaqueDepth` and `setLightNode`,
	 * which change the materials, with `group` in the rain's scene. */
	registerLatent( graph: SceneLatents ): Promise<void> {

		const rain = graph.latent( 'rain', [ this.group ], [ this.drops.compute ] );
		return this.afterGroup ? Promise.all( [ rain, graph.latent( 'rainAfter', [ this.afterGroup ] ) ] ).then( () => undefined ) : rain;

	}

	dispose(): void {

		this.drops.dispose();
		this.collision.dispose();

	}

}
