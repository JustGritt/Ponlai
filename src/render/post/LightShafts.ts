import { tune, tuneUniform } from '../debug/Tune.ts';
import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, uniform, texture, textureLoad, uv, ivec2, vec2, vec4, float, min, clamp,
	dot, normalize, acos, exp, select, max, vec3,
} from 'three/tsl';
import { SUN_DISC_RADIUS } from '../lighting/SunDisc.ts';
import { MOON_ANGULAR_RADIUS } from '../sky/NightSky.ts';
import type { Sky } from '../sky/Sky.ts';
import type { FrameGraph, GraphResources, ResourceHandle, SizeClass } from './graph/FrameGraph.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

// Crepuscular rays and the flare of the sun or the moon, as passes on the frame graph.

const STEPS = 48;
const _grey = new THREE.Color();
// the passes' size: half the render, at least a texel
const HALF: SizeClass = { of: 'render', scale: 0.5, min: 1 };
// The sun's flare's visibility: the share of FLARE_TAPS depth taps in a disc of the drawn disk's
// angular radius (lighting/SunDisc.ts SUN_DISC_RADIUS) round the sun's place on screen that read the
// sky; 16 dim it as branches cross the disk without visible steps.
const FLARE_TAPS = 16;
// The sun's glow, the eye's and the lens's wide scattering of the disk: a Lorentzian in screen
// space, C / ( 1 + r^2 / w^2 ), r the distance from the sun's place on screen in frame heights, so
// its size follows the frame, not the field of view. Its half width w; C, its radiance at the sun
// over the key's irradiance, by the sun's elevation (degrees): about nine times the sky 7 - 15
// degrees off a high sun at 0.06 - 0.1 frame heights, and at a low sun, whose sky beside it is
// bright against its dimmed light, about 1.3 times that sky at 0.03 - 0.06; a little warmer than
// the disk. It reaches the high sun's level by 7 degrees: the glow's tail, which carries the halo
// 7 - 50 degrees from the sun, is the same at every height from there up. The glare's factor
// (`post.glare`, 0.0429) scales all of it: what shows is about two fifths of the sky 7 - 15 degrees
// off a high sun.
const GLOW_WIDTH = 0.045;
const GLOW = { low: 0.8, high: 1.9, from: 4, to: 7 };
const GLOW_TINT = vec3( 1.15, 1, 1 );
// The moon's flare, while the moon is the key: the same Lorentzian round the drawn disk, its
// width w (`width`, frame heights) its own, a little narrower than the sun's (0.035 against
// GLOW_WIDTH's 0.045); its C the sun's algorithm on the moon's elevation and its key irradiance
// (Sky.ts `keyColor`), times `gain`. Its colour is the key's moved `white` of the way to its
// luminance, near white (red over green about 0.99, blue over green 1.02): the eye's scatter of a
// small bright source, not the moonlight's blue. Tunable on debug pages (`post.moonFlare.*`).
const MOON_FLARE = {
	width: tuneUniform( 'post.moonFlare.width', 0.0347, { min: 0.02, max: 0.15, scale: 'log' } ),
	gain: tune( 'post.moonFlare.gain', 14.8, { min: 0.1, max: 1000, scale: 'log' } ),
	white: 0.95,
};

/** Crepuscular rays (the Tyndall effect in hazy air) and the flare of the sun, or of the moon.
 *
 * The rays are a screen-space stand-in, off by default (`strength` 0): the volumetric haze
 * (weather/VolumetricHaze.ts) shows the shafts every caster cuts out of the air over the land and
 * the sea, through the bamboo on a cloudy morning facing the sun as bands three to four times as
 * bright as the air between them. Facing the golden hour's sun the rays washed a glow over the sky
 * instead (a fifth brighter 4 - 15 degrees from the sun, its clouds lost) and smeared blocky bands
 * over the near ground; under clouds they added nothing measurable. The code stays for an A/B.
 *
 * Light scattered by haze shows the shadows clouds and the island cast through the air as
 * beams converging on the sun. In screen space they are the visible sky around the sun, with
 * cloud and island silhouettes, blurred radially toward the sun's position (after Mitchell,
 * "Volumetric Light Scattering as a Post-Process", GPU Gems 3, 2007).
 *
 * The flare is the eye's and the lens's scattering of the bright disk: a wide glow measured in frame
 * heights (GLOW_WIDTH, GLOW), near white at its core and falling off as the square of the distance,
 * past half the frame; no ring, streaks or ghosts. It is not added to the image: it goes into the
 * bloom's first level (PhysicalBloom.ts `extra`), so it shows only through the bloom, divided
 * by the bloom's strength there. It is scaled by its visibility (`occlusion`: the share of depth
 * taps over the disk that see the sky, times the clouds' mean transmittance over the disk, Sky.ts
 * `discCloudTransmittance`), by the share of the disk inside the frame (a sun outside the frame
 * casts none, and one crossing its edge fades with the share), and by the haze's transmittance
 * toward it. While the moon is the key (Sky.ts `keyIsMoon`) the flare is the moon's instead
 * (`setMoon`, MOON_FLARE): the same glow round the drawn moon (Sky.ts `moonDir`, the night key's
 * direction), its own width and a near-white colour, taken through the same visibility over the
 * moon's disk (its depth taps, the clouds' mean over it: the sky constants' second channel), share
 * in the frame and haze. The rays stay the sun's: the moon casts none. The atmosphere's own glow round the sun is
 * the sky's, the air's round the moon the haze's (VolumetricHaze.ts). All are camera-side effects
 * and leave the sky's radiometry unchanged. */
export class LightShafts {

	readonly strength = uniform( 0 );
	// (tunable on debug pages, debug/Tune.ts: `post.glare`)
	readonly glare = tuneUniform( 'post.glare', 0.0429, { min: 0, max: 0.2 } );
	private readonly sunUV = uniform( new THREE.Vector2( 0.5, 0.5 ) );
	private readonly sunDir = uniform( new THREE.Vector3( 0, 1, 0 ) );
	private readonly color = uniform( new THREE.Color( 0, 0, 0 ) );
	// 1 while the sun is in front of the camera, fading out as it leaves the view.
	private readonly facing = uniform( 0 );
	// The flare's body: the sun's, or the moon's while it is the key (`setMoon`). Its place on
	// screen, 1 for the moon, its colour (the sun's is `color`), the share of its drawn disk inside
	// the frame, and the glow's radiance over its irradiance (GLOW).
	private readonly flareUV = uniform( new THREE.Vector2( 0.5, 0.5 ) );
	private readonly moonFlare = uniform( 0 );
	private readonly flareColor = uniform( new THREE.Color( 0, 0, 0 ) );
	private readonly onScreen = uniform( 0 );
	private readonly glow = uniform( 0 );
	private readonly moon = { on: false, direction: new THREE.Vector3( 0, 1, 0 ), color: new THREE.Color() };
	private readonly haze = uniform( 0 );
	private readonly lowSun = uniform( 0 );
	// The disk's transmittance through fog between the camera and the sky.
	private readonly fogVisibility = uniform( 1 );
	private readonly projectionInverse = uniform( new THREE.Matrix4() );
	private readonly cameraWorld = uniform( new THREE.Matrix4() );
	private readonly clip = new THREE.Vector4();
	// the two half-resolution passes (the open sky near the sun, the rays marched over it), their
	// targets the frame graph's (see `passes`)
	private resources!: { source: ResourceHandle; rays: ResourceHandle };
	private sourceQuad!: THREE.QuadMesh;
	private raysQuad!: THREE.QuadMesh;
	// the flare's drawn disk's angular radius on screen, in uv per axis (see `occlusion`)
	private readonly discRadius = uniform( new THREE.Vector2() );

	/** The rays' radiance to add to the HDR image, from the resolved depth (the sea's included) and
	 * the sky. Its two half-resolution targets are declared on `graph`, which owns and sizes them;
	 * `passes` then adds the passes that draw them. */
	build( depth: THREE.Texture, sky: Sky, graph: GraphResources ): V3 {

		const sceneDepth = texture( depth );
		const resolution = graph.extent( { of: 'render' } );
		const direction = ( q: V2 ) => {

			const ndc = vec4( q.x.mul( 2 ).sub( 1 ), float( 1 ).sub( q.y ).mul( 2 ).sub( 1 ), 1, 1 );
			const view = this.projectionInverse.mul( ndc );
			return normalize( this.cameraWorld.mul( vec4( view.xyz.div( view.w ), 0 ) ).xyz );

		};

		// (half float, as three's rtt() made them)
		const source = graph.color( 'shafts.source', { size: HALF, format: THREE.RGBAFormat, type: THREE.HalfFloatType } );
		const rays = graph.color( 'shafts.rays', { size: HALF, format: THREE.RGBAFormat, type: THREE.HalfFloatType } );
		this.resources = { source, rays };
		// (read as their rtt() nodes were: with their uv, no uv transform)
		const tSource = texture( graph.texture( source ), uv() ), tRays = texture( graph.texture( rays ), uv() );

		// Half resolution: the open sky's brightness near the sun (r) and its plain visibility (g).
		this.sourceQuad = this.quad( 'Shafts.Source', Fn( () => {

			const q = uv().toVar();
			const px = ivec2( clamp( q.mul( resolution ), vec2( 0 ), resolution.sub( 1 ) ) );
			const z = ( textureLoad( sceneDepth, px ) as unknown as F ).toVar();
			const d = direction( q ).toVar();
			const open = float( 0 ).toVar();
			If( z.lessThan( - 9e4 ).and( d.y.greaterThan( 0 ) ), () => {

				open.assign( sky.clouds.sample( d ).a );

			} );
			const angle = acos( clamp( dot( d, this.sunDir ), - 1, 1 ) );
			// Beams carry the bright sky near the sun; a wide term alone would blur into a disc.
			const bright = exp( angle.div( - 0.05 ) ).mul( 1.5 ).add( exp( angle.div( - 0.15 ) ).mul( 0.15 ) );
			return vec4( open.mul( bright ), open, 0, 1 );

		} )() );

		const inside = ( p: V2 ) => select( p.x.greaterThanEqual( 0 ).and( p.x.lessThanEqual( 1 ) ).and( p.y.greaterThanEqual( 0 ) ).and( p.y.lessThanEqual( 1 ) ), float( 1 ), float( 0 ) );

		this.raysQuad = this.quad( 'Shafts.Rays', Fn( () => {

			const result = float( 0 ).toVar();
			If( this.facing.greaterThan( 0.001 ), () => {

				const q = uv().toVar();
				// March toward the sun; the reach covers the sun from anywhere on screen.
				const step = this.sunUV.sub( q ).mul( 0.9 / STEPS ).toVar();
				const p = q.toVar();
				const decay = float( 1 ).toVar();
				Loop( STEPS, () => {

					p.addAssign( step );
					result.addAssign( texture( tSource, p ).level( float( 0 ) ).r.mul( inside( p ) ).mul( decay ) );
					decay.mulAssign( 0.97 );

				} );
				result.mulAssign( 1 / STEPS );

			} );
			return vec4( result, 0, 0, 1 );

		} )() );

		return Fn( () => {

			const q = uv().toVar();
			// Beams need haze to scatter in, and a low sun's long path through it; clear air at
			// midday still shows them faintly. Weak enough that a clear sky only gains a glow.
			const air = min( this.haze.mul( 600 ).add( 0.5 ), 2 ).mul( this.lowSun.mul( 4 ).add( 1 ) );
			const beams = texture( tRays, q ).level( float( 0 ) ).r.mul( air ).mul( this.strength ).mul( 0.14 ).mul( this.facing );
			return this.color.rgb.mul( beams );

		} )();

	}

	/** The flare's visibility, the same for every pixel of a frame (PipelineDisplay.ts evaluates it
	 * once, in its constants pass): the share of FLARE_TAPS taps of `surfaceDepth` (the nearest
	 * surface's view z, the sky's below -9e4) in a disc of the drawn disk's radius round the
	 * flare's body's place on screen that read the sky, times the clouds' mean transmittance over
	 * its disk (`sunCloud`, or `moonCloud` for the moon's). The taps lie on a Vogel spiral, held
	 * inside the frame. */
	occlusion( surfaceDepth: THREE.Texture, size: V2, sunCloud: F, moonCloud: F ): F {

		const depth = texture( surfaceDepth );
		const golden = Math.PI * ( 3 - Math.sqrt( 5 ) );
		let open: F = float( 0 );
		for ( let i = 0; i < FLARE_TAPS; i ++ ) {

			const r = Math.sqrt( ( i + 0.5 ) / FLARE_TAPS ), a = i * golden;
			const at = this.flareUV.add( this.discRadius.mul( vec2( r * Math.cos( a ), r * Math.sin( a ) ) ) );
			const px = ivec2( clamp( at.mul( size ), vec2( 0 ), size.sub( 1 ) ) );
			open = open.add( select( ( textureLoad( depth, px ) as unknown as F ).lessThan( - 9e4 ), float( 1 ), float( 0 ) ) );

		}
		return open.mul( 1 / FLARE_TAPS ).mul( select( this.moonFlare.greaterThan( 0.5 ), moonCloud, sunCloud ) );

	}

	/** The flare's radiance at screen uv `q` for the bloom's input (PhysicalBloom.ts `extra`):
	 * C / ( 1 + |( dx a, dy )|^2 / w^2 ), a the aspect ratio, w GLOW_WIDTH (the moon's: MOON_FLARE's
	 * `width`), C the glow's radiance at the body times `visible` (`occlusion`, as a pass
	 * evaluated it), the disk's share in the frame and the haze's transmittance, divided by `gain`
	 * (the bloom's strength, which the bloom's result is added with). Below a gain of 0.001 the
	 * flare is left out: divided by a vanishing gain it would overflow the bloom's half-float levels
	 * to infinity and the frame to NaN. */
	flare( q: V2, aspect: F, visible: F, gain: F ): V3 {

		const offset = q.sub( this.flareUV ).mul( vec2( aspect, 1 ) );
		const r2 = dot( offset, offset ).toVar();
		// (both built before the choice, the sun's as it always was)
		const sunGlow = GLOW_TINT.mul( this.glow.div( r2.div( GLOW_WIDTH * GLOW_WIDTH ).add( 1 ) ) ).toVar();
		const moonGlow = vec3( this.glow.div( r2.div( MOON_FLARE.width.mul( MOON_FLARE.width ) ).add( 1 ) ) ).toVar();
		const glow = select( this.moonFlare.greaterThan( 0.5 ), moonGlow, sunGlow );
		const scale = visible.mul( this.onScreen ).mul( this.fogVisibility ).mul( this.glare ).div( max( gain, 1e-6 ) );
		return this.flareColor.rgb.mul( glow ).mul( select( gain.greaterThanEqual( 1e-3 ), scale, float( 0 ) ) );

	}

	private quad( name: string, fragment: THREE.Node<'vec4'> ): THREE.QuadMesh {

		const material = new THREE.NodeMaterial();
		material.name = name;
		material.fragmentNode = fragment;
		const quad = new THREE.QuadMesh( material );
		quad.name = name;
		return quad;

	}

	/** Adds the shafts' two passes to `graph`, after the pass that writes `surfaceDepth` (the
	 * nearest surface's linear view z) and before the composite: the open sky near the sun, then
	 * the rays marched over it toward the sun. Both run only while the sun faces the view (`facing`
	 * above 0): the composite's term is multiplied by it, so what they last drew, finite, adds
	 * exactly nothing otherwise (the composite reads them optionally). Returns the resources the
	 * composite reads. */
	passes<F>( graph: FrameGraph<F>, surfaceDepth: ResourceHandle ): ResourceHandle[] {

		const LABEL = 'light shafts';
		let { source, rays } = this.resources;
		const sourceQuad = this.sourceQuad, raysQuad = this.raysQuad;
		const when = () => this.facing.value > 0, whenLabel = 'facing > 0';
		graph.addPass( 'shaftsSource', 'quad', ( b ) => {

			b.read( surfaceDepth );
			source = b.write( source );
			return { label: LABEL, when, whenLabel, quad: sourceQuad };

		} );
		graph.addPass( 'shaftsRays', 'quad', ( b ) => {

			// (the same predicate as the source's)
			b.readOptional( source );
			rays = b.write( rays );
			return { label: LABEL, when, whenLabel, quad: raysQuad };

		} );
		return [ source, rays ];

	}

	/** The flare's body's direction: the moon's while it is the key (`setMoon`), else the sun's
	 * (Pipeline.ts takes the haze's transmittance toward it). */
	get flareDirection(): THREE.Vector3 { return this.moon.on ? this.moon.direction : this.sunDir.value; }

	/** The sun's direction and its direct irradiance, for the rays and the day's flare. */
	setSun( direction: THREE.Vector3, color: THREE.Color ): void {

		this.sunDir.value.copy( direction );
		this.color.value.copy( color );

	}

	/** The drawn moon's direction (Sky.ts `moonDir`), the key's irradiance and whether the moon is
	 * the key (Sky.ts `keyIsMoon`): while it is, the flare is the moon's (MOON_FLARE). */
	setMoon( direction: THREE.Vector3, color: THREE.Color, on: boolean ): void {

		this.moon.on = on;
		this.moon.direction.copy( direction );
		const luminance = color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
		this.moon.color.copy( color ).lerp( _grey.setScalar( luminance ), MOON_FLARE.white );

	}

	/** The frame's camera, haze and the haze's transmittance toward the sun. `projection`: the camera's
	 * projection without the temporal modes' jitter (Jitter.ts), which the sun's position on screen
	 * leaves out: a jittered sun would shake the rays' origin by a fraction of a pixel each frame
	 * (default: the camera's own, unjittered outside the graph's jittered span). */
	update( camera: THREE.PerspectiveCamera, haze: number, fogVisibility = 1, projection: THREE.Matrix4 = camera.projectionMatrix ): void {

		const key = this.sunDir.value;
		this.projectionInverse.value.copy( camera.projectionMatrixInverse );
		this.cameraWorld.value.copy( camera.matrixWorld );
		this.haze.value = haze;
		this.fogVisibility.value = fogVisibility;
		this.lowSun.value = 1 - THREE.MathUtils.smoothstep( key.y, 0.05, 0.35 );
		const c = this.clip.set( key.x, key.y, key.z, 0 ).applyMatrix4( camera.matrixWorldInverse ).applyMatrix4( projection );
		const forward = c.w > 1e-6;
		const x = forward ? c.x / c.w : 0, y = forward ? c.y / c.w : 0;
		this.sunUV.value.set( x * 0.5 + 0.5, 0.5 - y * 0.5 );
		// Fade as the sun leaves the view, and as it sets.
		const offscreen = Math.max( Math.abs( x ), Math.abs( y ) );
		const setting = THREE.MathUtils.smoothstep( key.y, - 0.02, 0.02 );
		this.facing.value = forward ? ( 1 - THREE.MathUtils.smoothstep( offscreen, 1.2, 2.2 ) ) * setting : 0;
		// The flare's body: the sun as above, or the moon while it is the key, placed the same way.
		const moon = this.moon.on;
		const body = moon ? this.moon.direction : key, radius = moon ? MOON_ANGULAR_RADIUS : SUN_DISC_RADIUS;
		this.moonFlare.value = moon ? 1 : 0;
		this.flareColor.value.copy( moon ? this.moon.color : this.color.value );
		let bx = x, by = y, bodyForward = forward, bodySetting = setting;
		if ( moon ) {

			const m = this.clip.set( body.x, body.y, body.z, 0 ).applyMatrix4( camera.matrixWorldInverse ).applyMatrix4( projection );
			bodyForward = m.w > 1e-6;
			bx = bodyForward ? m.x / m.w : 0;
			by = bodyForward ? m.y / m.w : 0;
			bodySetting = THREE.MathUtils.smoothstep( body.y, - 0.02, 0.02 );

		}
		this.flareUV.value.set( bx * 0.5 + 0.5, 0.5 - by * 0.5 );
		// the disk's share inside the frame, per axis: its radius in NDC from the projection's scale
		const e = projection.elements;
		const share = ( p: number, r: number ) => THREE.MathUtils.clamp( ( 1 - Math.abs( p ) + r ) / ( 2 * r ), 0, 1 );
		this.onScreen.value = bodyForward ? share( bx, radius * e[ 0 ] ) * share( by, radius * e[ 5 ] ) * bodySetting : 0;
		const elevation = THREE.MathUtils.radToDeg( Math.asin( THREE.MathUtils.clamp( body.y, - 1, 1 ) ) );
		this.glow.value = THREE.MathUtils.lerp( GLOW.low, GLOW.high, THREE.MathUtils.smoothstep( elevation, GLOW.from, GLOW.to ) ) * ( moon ? MOON_FLARE.gain() : 1 );
		const t = Math.tan( radius );
		this.discRadius.value.set( t * e[ 0 ] * 0.5, t * e[ 5 ] * 0.5 );

	}

	/** The inverse projection the rays' directions are reconstructed with (`update` takes the
	 * camera's): under the temporal modes the jittered one the depth they read was drawn with, set
	 * by Pipeline's jitter hook, so each pixel's ray meets its own depth sample. */
	setProjectionInverse( inverse: THREE.Matrix4 ): void {

		this.projectionInverse.value.copy( inverse );

	}

}
