/*!
 * Adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, texture, uniform, vec2, vec3, vec4, float, int, max, mix, clamp, smoothstep, exp, dot, length, select,
	screenCoordinate, positionWorld, normalWorld, positionViewDirection, normalView, normalViewGeometry, diffuseContribution,
	specularColor, specularColorBlended, specularF90, F_Schlick, roughness, luminance, diffuseColor, metalness,
} from 'three/tsl';
import { BODY_REFLECTANCE } from '../ocean/SeaOptics.ts';
import type { SunShadow } from './SunShadow.ts';
import type { Shared } from '../shared.ts';
import { useLightLayer } from '../debug/LightLayer.ts';
import { ambient, diffuseOcclusion, useAmbientGrade } from './AmbientGrade.ts';
import { shadeLocalLights } from './LocalLights.ts';
import { GROUND_BOUNCE_LAND, type AmbientReceiver } from './AmbientSample.ts';
import { liftKey } from './KeyLift.ts';
import { glossAlpha, specularGGX, fresnel90, keyGlint, glintDirection, glintNormalisation } from './SurfaceBRDF.ts';
import { environmentBRDF } from './EnvironmentBRDF.ts';
import { meterAmbient, meterKey, meterStart } from './MeterLight.ts';
import { onSceneDispose } from '../SceneDispose.ts';

// Sunlight bounced off the ground, one diffuse bounce, baked into a map over the island.
//
// Sunlight bounced off the ground, one diffuse bounce (tidewater's GroundBounce, 4811ba4): the
// sunlit plateau lights the islanders' undersides and the grass from below, the shallows light
// the cliff's walls over them.
//
// The bake: a map over the world fields' bounds (the land's box and 32 m round it: shoreXform) of
// the light the ground reflects per unit of the key's irradiance: its albedo (the land's; over the
// sea the floor's seen through the water column, down and back up, as tidewater's sea, and the
// water body's own reflectance under the surface, both through the surface at its diffuse
// transmittances; the land's and the floor's albedos and the land's height are the scene's, which
// describes its ground as a `BounceGround`: the island's in materials/BounceGround.ts) times
// the cosine of the key on the level ground times the key's visibility there (the sun's shadow map,
// a little above the ground so the plateau's relief stays under the lookup; tidewater takes its
// heightfield's shadow, which the island's own shadow map already holds here), minus what the
// environment's lower half already gives a face turned down (the distant sea's radiance, Sky's
// `seaRadiance`, where tidewater's environment has the atmosphere's dark ground), then blurred
// (a binomial 5x5 of 1 m steps, as 3x3 bilinear lookups). Alpha: the height of the bouncing
// surface (the land's, as the scene's ground gives it, or the sea level). It is drawn at 256 texels,
// blurred into the wave maps' array target (its last layer, WaveMaps `bounceLayer`), so the
// materials lit under the water take no texture more for it. It is drawn before the frame graph,
// after the sun's map. It is a function of the key's direction and colour, the sun's maps, the
// sea's radiance and the world fields. It is drawn with the sun's map's steps (`steps`: a step
// of the key's turn, SunShadow.ts, counted also while the swaying plants redraw that map every
// frame), from that map alone and the key as it stands then, and
// whenever the sea's radiance or the fields change (a growth swap and a scene jump through
// `invalidate`, since the fields change in place). Drawn every frame the clock ran, its two passes
// took a tenth of a millisecond or more; between steps the key turns by at most 0.2 degrees, which
// moves the blurred bounce by far less than a level, and a paused frame draws it as a fresh load
// does (a jump redraws the sun's map from scratch, and this map with it).
//
// The shading: irradiance = the key's colour x the cloud shadow (the land's key over the land's
// floor, Sky.ts `keyDirectVisibility`, which the sea in the map takes too, as the water drawn does:
// Sky.ts `seaKeyVisibility`) x the map x the lower hemisphere's
// view factor of the normal ( ( 1 - N.y ) / 2 ) x a fade with the height over the bouncing surface
// (none from a surface above the receiver: under the water the sea is above) x the self-shading of
// a face turned away from a low key (it sees the ground its own object shades) x SURROUND (objects
// round the receiver shade part of the ground it sees). The map is read in front of the face, REACH
// along its horizontal normal: tidewater reads under the point, which over its gentle shore and 4 m
// texels is the same ground, but a cliff's wall stands on the line between the plateau and the sea
// and sees the sea in front of it. One lookup at level 0, only for faces turned down or sideways
// above the sea.
//
// Every lit material adds it to its irradiance, so its diffuse albedo, its occlusion and its own
// ambient scale apply as to the sky's (tidewater adds it to the indirect diffuse light): a physical
// lighting model calls `addGroundBounce` at the start of its `indirect` (the terrain's, the
// vegetation's, `GroundBounceLightingModel` for the game's props), a model of its own reads
// `groundBounceLight` (the islanders'), never a bounce of its own. The water does not take it.
// The bounce takes the ambient grade's `sunBounce` where every material takes it
// (`groundBounceLight`). Beside the ambient volume (lighting/AmbientSample.ts), whose relight
// bounces the key off the land round each probe (AmbientRelight.ts), the bake keeps the sea's part
// alone (the land's albedo 0): the shallows that light the cliff's walls, which the volume's
// imaginary ground does not carry, so no material counts the land's bounce twice;
// `flags=groundBounceLand` keeps it whole beside the volume, and `flags=ambientVolume:0` (no
// volume) keeps it whole as before. The physical models extend `GroundBounceLightingModel`, which
// also reads the ambient volume (lighting/AmbientSample.ts), applies the ambient grade to the
// ambient light and the occlusion (lighting/AmbientGrade.ts) and the lighting layer's debug views
// (debug/LightLayer.ts: `flags=lightOnly`, `ambientOnly`); a model of its own calls
// `useLightLayer` in its constructor and reads the grade's uniforms itself.
// The model's direct light is its own: Lambert under 1 - lum( F ) and the shared GGX of
// lighting/SurfaceBRDF.ts (alpha the perceptual roughness squared, Schlick-GGX visibility) in place
// of three's BRDF_GGX; F90 follows F0 (`fresnel90`, set over the material's own F90 in `start`);
// the environment's specular takes the split sum integrated for that GGX (lighting/EnvironmentBRDF.ts) in place of three's DFG table, and its
// prefiltered environment is read at the roughness whose square is the GGX's alpha.
// The diffuse, direct and ambient, loses the Fresnel's luminance (1 - lum( F ), 1 - lum( F0 A +
// F90 B )), not a colour, and three's multiple scattering terms are gone. The whole direct term,
// diffuse and specular with its Fresnel, takes the key's shading direction (`shadeDirection`:
// lifted under a low key, lighting/KeyLift.ts), as the sun's shadows do; the specular takes its
// light from the disk's representative point round that direction (lighting/SurfaceBRDF.ts
// `keyGlint`). The diffuse of one light is `diffuseResponse` (Lambert under 1 - lum( F( V.H ) ), a
// subclass's own on top), which the key and every local light (lighting/LocalLights.ts, along their
// true direction, under the ambient diffuse's occlusion) take alike.
// Each term also adds its irradiance to the metering illuminance's sums (lighting/MeterLight.ts):
// the key's times its cosine in `direct`, the irradiance and the environment's (or the volume's)
// in the ambient diffuse; the occlusion multiplies them in the ambient grade's (AmbientGrade.ts).
// The lookup is inlined into each material: as a layout function reading this module's uniforms and
// texture, r186 emits it with the first material's binding names in the next one (SeaDetail.ts).

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

// the bake's texels per side (the blurred map takes the wave maps' target's)
const RAW = 256;
// objects round the receiver shade part of the ground it sees (their shadows are not in the map)
const SURROUND = 0.6;
// the blur's step (m): a binomial 5x5 of these (sigma one step), about the ground a receiver a metre
// or two above it integrates
const BLUR_STEP = 1;
// how far in front of a face the ground it sees is read (m)
const REACH = 1;
// how far above the ground the bake takes the key's visibility (m): above the plateau's relief
const SHADOW_LIFT = 0.2;
// the floor's light through the surface, down and back up: the sky's diffuse transmittance into the
// water (0.934) times the upwelling light's out of it, 1 - 0.475. Both are Fresnel's reflectance of
// water (n = 1.333) integrated over a uniform hemisphere, our own figures: 0.475 the mean internal
// reflectance for light from under the surface, most of it totally reflected past the critical
// angle, the mechanism Lekner & Dorf 1988 build their model on
const FLOOR_THROUGH = 0.934 * 0.525;
// the floor's light crosses the column twice: down as the sky's diffuse light, whose mean cosine
// under the surface is about 0.83 (1.2 times the depth, as Underwater.ts), and back up from a
// Lambertian floor, whose mean cosine is 0.5 (2 times the depth)
const FLOOR_PATH = 1.2 + 2.0;

/** The bake's texel as the scene's ground reads it, nodes of the bake's shader. */
export interface BounceTexel {
	/** The map's coordinate over the fields' bounds (shared.shoreXform). */
	at: THREE.Node<'vec2'>;
	/** The texel's world point. */
	xz: THREE.Node<'vec2'>;
	/** The shore's sea-level distance there (m, negative over the land). */
	shore: F;
}

/** The scene's ground as the bake reads it: each method is called once while the bake's shader is
 * built, takes the texel and returns nodes of that shader. */
export interface BounceGround {
	/** The land's linear albedo. */
	land( texel: BounceTexel ): V3;
	/** The sea floor's linear albedo, as seen through clear water. */
	floor( texel: BounceTexel ): V3;
	/** The height (m) of the land's surface. */
	height( texel: BounceTexel ): F;
}

export interface GroundBounceInputs {
	renderer: THREE.WebGPURenderer;
	shared: Shared;
	sunShadow: SunShadow;
	// the fields over the bounds of shared.shoreXform: the shore's sea-level distance and the
	// floor's height
	fields: { shore: THREE.TextureNode; floor: THREE.TextureNode };
	// the land's and the floor's albedos and the land's height
	ground: BounceGround;
	// the array target and layer the map is blurred into (WaveMaps `target`, `bounceLayer`)
	target: THREE.RenderTarget;
	layer: number;
	// the land's cloud shadow on the key at a world point, over the weather's floor (Sky.ts
	// `keyDirectVisibility`), the sea's too
	cloud: ( position: V3 ) => F;
}

export class GroundBounce {

	readonly strength = uniform( 1 );
	/** Draws of the map so far. */
	bakes = 0;
	private readonly ready = uniform( 0 );
	// the environment's lower half per unit of the key's irradiance (see the header)
	private readonly envGround = uniform( new THREE.Vector3() );
	private readonly raw: THREE.RenderTarget;
	private readonly rawQuad: THREE.QuadMesh;
	private readonly blurQuad: THREE.QuadMesh;
	private signature = '';

	constructor( private readonly inputs: GroundBounceInputs ) {

		const { shared, sunShadow, fields, ground: described, target } = inputs;
		this.raw = new THREE.RenderTarget( RAW, RAW, {
			type: THREE.HalfFloatType, depthBuffer: false, generateMipmaps: false,
			minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
		} );
		this.raw.texture.name = 'groundBounceRaw';

		const rawMat = new THREE.NodeMaterial();
		rawMat.name = 'GroundBounce.Raw';
		rawMat.fragmentNode = Fn( () => {

			// the texel's centre (screen row 0 is texture row 0) and its world xz
			const at = screenCoordinate.xy.div( RAW ).toVar();
			const xz = shared.shoreXform.xy.add( at.div( shared.shoreXform.zw ) ).toVar();
			const sd = texture( fields.shore, at ).level( float( 0 ) ).r.toVar();
			const floorY = texture( fields.floor, at ).level( float( 0 ) ).r;
			const L = shared.sunDir;
			const texel: BounceTexel = { at, xz, shore: sd };
			const land = described.land( texel );
			// the sea: the floor seen through the water column, down and back up (at a + bb, over
			// FLOOR_PATH times the depth), and the water body's own irradiance reflectance under its
			// surface, the deep water's R( 0- ) = BODY_REFLECTANCE bb / ( a + bb ) (ocean/SeaOptics.ts,
			// as the probes take it) times the share of it a column of this depth holds, 1 less the
			// floor's transmittance (the shallow-water form Water.ts takes from Lee et al. 1999), both
			// through the surface (FLOOR_THROUGH)
			const depth = max( shared.waterLevel.sub( floorY ), 0 );
			const kappa = shared.sigA.add( shared.bb );
			const body = shared.bb.mul( BODY_REFLECTANCE ).div( kappa );
			const floor = described.floor( texel );
			const column = exp( kappa.mul( depth ).mul( - FLOOR_PATH ) );
			const sea = mix( body, floor, column ).mul( FLOOR_THROUGH );
			const wet = smoothstep( - 0.25, 0.25, sd ).toVar();
			// (the land's bounce is the ambient volume's where materials read it: see the header)
			const albedo = GROUND_BOUNCE_LAND ? mix( land, sea, wet ) : sea.mul( wet );
			const ground = described.height( texel ).toVar();
			const top = mix( ground, shared.waterLevel, wet );
			const lift = select( wet.lessThan( 0.5 ), ground, shared.waterLevel ).add( SHADOW_LIFT );
			const visibility = sunShadow.currentPCFVisibility( sunShadow.coordinate.mul( vec4( xz.x, lift, xz.y, 1 ) ).xyz );
			const E = max( L.y, 0 ).mul( visibility ).mul( smoothstep( - 0.02, 0.05, L.y ) );
			return vec4( max( albedo.mul( E ).sub( this.envGround ), vec3( 0 ) ), top );

		} )();
		this.rawQuad = new THREE.QuadMesh( rawMat );
		this.rawQuad.name = 'GroundBounce.Raw';

		const blurMat = new THREE.NodeMaterial();
		blurMat.name = 'GroundBounce.Blur';
		blurMat.fragmentNode = Fn( () => {

			// the binomial 1 4 6 4 1 per axis as three bilinear taps (the outer pairs merged at their
			// weighted mean offset, 1.2 steps out): 9 lookups instead of 25
			const at = screenCoordinate.xy.div( target.width ).toVar();
			const step = shared.shoreXform.zw.mul( BLUR_STEP ).toVar();
			const taps = [ [ - 1.2, 5 / 16 ], [ 0, 6 / 16 ], [ 1.2, 5 / 16 ] ];
			let sum: THREE.Node<'vec4'> = vec4( 0 );
			for ( const [ y, wy ] of taps ) for ( const [ x, wx ] of taps ) {

				const tap = texture( this.raw.texture, at.add( step.mul( vec2( x, y ) ) ) ).level( float( 0 ) );
				sum = sum.add( tap.mul( wx * wy ) );

			}
			return sum;

		} )();
		this.blurQuad = new THREE.QuadMesh( blurMat );
		this.blurQuad.name = 'GroundBounce.Blur';

	}

	/** The bounce's irradiance at a world point and normal (none before the first bake). */
	irradiance( P: V3, N: V3 ): V3 {

		const { shared, target, layer } = this.inputs;
		const E = vec3( 0 ).toVar();
		const view = clamp( N.y.mul( - 0.5 ).add( 0.5 ), 0, 1 ).toVar();
		If( view.greaterThan( 0.03 ).and( P.y.greaterThan( shared.waterLevel.sub( 0.3 ) ) ).and( this.strength.greaterThan( 0 ) ).and( this.ready.greaterThan( 0 ) ), () => {

			const half = 0.5 / target.width;
			const at = clamp( P.xz.add( N.xz.mul( REACH ) ).sub( shared.shoreXform.xy ).mul( shared.shoreXform.zw ), half, 1 - half );
			const s = texture( target.texture, at ).depth( int( layer ) ).level( float( 0 ) ).toVar();
			const above = P.y.sub( s.w );
			// nothing from a surface above the receiver, a little less high up where more of the view
			// is taken by other objects
			const fade = smoothstep( - 0.3, 0.3, above ).mul( mix( float( 0.4 ), float( 1 ), smoothstep( 30, 4, above ) ) );
			// a face turned away from a low key looks at the ground its object shades (a long shadow)
			const L = shared.sunDir;
			const Lh = L.xz.div( max( length( L.xz ), 1e-3 ) );
			const selfShadow = float( 1 ).sub( clamp( dot( N.xz, Lh ).negate(), 0, 1 ).mul( clamp( L.y.oneMinus(), 0, 1 ) ).mul( 0.6 ) );
			E.assign( s.rgb.mul( shared.landKey ).mul( this.inputs.cloud( P ) ).mul( view.mul( fade ).mul( selfShadow ).mul( this.strength ).mul( SURROUND ) ) );

		} );
		return E;

	}

	/** Once a frame, after the sun's map: draws the map again at the sun's map's
	 * steps and when the sea or the fields changed (see the header). `sea`: the environment's lower
	 * half (Sky `seaRadiance`). */
	update( sea: THREE.Color ): void {

		const { renderer: r, shared, sunShadow, target, layer } = this.inputs;
		const fill = shared.skyFill.value, x = shared.shoreXform.value;
		const signature = [ sunShadow.steps, sunShadow.ready.value, sea.r, sea.g, sea.b, fill,
			x.x, x.y, x.z, x.w, shared.waterLevel.value ].join( ',' );
		if ( signature === this.signature ) return;
		this.signature = signature;
		// a face turned down receives pi times the lower half's radiance from the environment
		const key = shared.landKey.value;
		const per = ( radiance: number, k: number ) => Math.min( Math.PI * radiance * fill / Math.max( k, 1e-6 ), 1 );
		this.envGround.value.set( per( sea.r, key.x ), per( sea.g, key.y ), per( sea.b, key.z ) );
		const prev = r.getRenderTarget();
		r.setRenderTarget( this.raw );
		this.rawQuad.render( r );
		// (read at level 0 only; the target makes no mips here: WaveMaps.ts makes its chain after
		// the caustics alone)
		r.setRenderTarget( target, layer );
		this.blurQuad.render( r );
		r.setRenderTarget( prev );
		this.ready.value = 1;
		this.bakes ++;

	}

	/** Draws the map again on the next update: the world fields changed in place (a growth swap) or
	 * the scene jumped. */
	invalidate(): void {

		this.signature = '';

	}

}

let installed: GroundBounce | null = null;

/** Makes `bounce` the ground bounce every lit material adds, and returns it: before any of them
 * builds. */
export function installGroundBounce( bounce: GroundBounce ): GroundBounce {

	return installed = bounce;

}
onSceneDispose( 'ground bounce', () => { installed = null; } );

/** The ground bounce's irradiance at the surface being lit, at the ambient grade's `sunBounce`
 * (none where none is installed). */
export function groundBounceLight(): V3 {

	return installed ? installed.irradiance( positionWorld, normalWorld ).mul( ambient.sunBounce ) : vec3( 0 );

}

/** Adds the ground bounce to a physical lighting model's irradiance: called at the start of its
 * `indirect`, before three's indirect diffuse reads the irradiance. */
export function addGroundBounce( builder: THREE.NodeBuilder ): void {

	( builder.context as unknown as { irradiance: V3 } ).irradiance.addAssign( groundBounceLight() );

}

/** A light's diffuse cosine and the diffuse's Fresnel weight, ( saturate( N.L ), 1 - lum( F( V.H ) ) )
 * (view space; F0 and F90 the material's, Schlick's F): one shader function, which the key's direct
 * term and every local light call (`diffuseResponse`). */
export const diffuseFresnel = /*@__PURE__*/ Fn( ( [ N, V, L, f0, f90 ]: [ V3, V3, V3, V3, F ] ) => {

	const H = L.add( V ).normalize();
	const F = luminance( F_Schlick( { f0, f90, dotVH: clamp( dot( V, H ), 0, 1 ) } ) as unknown as V3 );
	return vec2( clamp( dot( N, L ), 0, 1 ), float( 1 ).sub( F ) );

} ).setLayout( { name: 'diffuseFresnel', type: 'vec2', inputs: [
	{ name: 'N', type: 'vec3' }, { name: 'V', type: 'vec3' }, { name: 'L', type: 'vec3' }, { name: 'f0', type: 'vec3' }, { name: 'f90', type: 'float' },
] } );

/** Three's physical lighting with the ground bounce: the game's props and the other lit materials
 * without a model of their own use it as it is, and the physical models (the terrain's and the
 * vegetation's, lighting/Underwater.ts and vegetation/VegetationLighting.ts, the feathers') extend it, so
 * the ambient volume's diffuse (lighting/AmbientSample.ts, `ambient`: degree 2 by default), the
 * ambient grade (lighting/AmbientGrade.ts), the local lights (lighting/LocalLights.ts), the low
 * key's lift (lighting/KeyLift.ts) and the lighting layer's debug views (debug/LightLayer.ts) reach
 * every one of them from here. */
export class GroundBounceLightingModel extends THREE.PhysicalLightingModel {

	/** How the model reads the ambient volume for its ambient diffuse (lighting/AmbientSample.ts;
	 * null: the environment's), read when the material builds. */
	ambient: AmbientReceiver | null;

	constructor( ambient: AmbientReceiver | null = { bands: 2 } ) {

		super();
		this.ambient = ambient;
		useAmbientGrade( this, () => this.ambient );
		useLightLayer( this );

	}

	/** The direction the key's whole direct term takes, for the key's view-space direction L: L
	 * lifted under a low key, the moon's fixed direction by night (lighting/KeyLift.ts).
	 * UnderwaterLightingModel keeps the refracted sun's own under the water. */
	shadeDirection( L: V3 ): V3 {

		return liftKey( L );

	}

	/** GGX's alpha from the material's gloss (lighting/SurfaceBRDF.ts), set in `start`. */
	protected alpha: F | null = null;
	/** The environment BRDF's ( A, B ) (lighting/EnvironmentBRDF.ts), set in `start`. */
	protected envBRDF: THREE.Node<'vec2'> | null = null;

	start( builder: THREE.NodeBuilder ): void {

		// the metering illuminance's sums (lighting/MeterLight.ts), filled by the terms below
		meterStart();
		const gloss = roughness.oneMinus().toVar( 'surfaceGloss' );
		this.alpha = glossAlpha( gloss ).toVar( 'surfaceAlpha' );
		// the split sum's A and B for the shared GGX (lighting/EnvironmentBRDF.ts), and the
		// environment's prefiltered lookup at the roughness whose square is the GGX's alpha
		this.envBRDF = environmentBRDF( gloss, clamp( dot( normalView, positionViewDirection ), 0, 1 ) ).toVar( 'envBRDF' );
		roughness.assign( this.alpha.sqrt() );
		// F90 from F0, whatever the material's setupSpecular set (lighting/SurfaceBRDF.ts)
		specularF90.assign( fresnel90( specularColor as unknown as V3 ) );
		// three's PhysicalLightingModel.start adds only iridescence and transmission (unused here)
		// and its DFG table, which this model replaces: binding that table beside ours took the
		// materials lit under the water past WebGPU's 16 sampled textures
		THREE.LightingModel.prototype.start.call( this, builder );

	}

	/** The split sum's single scattering F0 A + F90 B for a reflectance F0. */
	private envSpecular( f0: V3 ): V3 {

		const fab = this.envBRDF!;
		return f0.mul( fab.x ).add( vec3( specularF90 ).mul( fab.y ) );

	}

	/** The irradiance's diffuse (the ambient lights, the ground's bounce): Lambert under
	 * 1 - lum( F0 A + F90 B ), with no multiple scattering term. */
	indirectDiffuse( builder: THREE.NodeBuilder ): void {

		const { irradiance, reflectedLight } = builder.context as unknown as { irradiance: V3; reflectedLight: { indirectDiffuse: V3 } };
		const weight = float( 1 ).sub( luminance( this.envSpecular( specularColor as unknown as V3 ) ) );
		reflectedLight.indirectDiffuse.addAssign( irradiance.mul( diffuseContribution ).mul( weight ).mul( 1 / Math.PI ) );
		meterAmbient.addAssign( irradiance );

	}

	/** The environment's specular, radiance x ( F0 A + F90 B ), and its diffuse, Lambert under
	 * 1 - lum( F0 A + F90 B ) (a luminance, so a tinted F0 does not tint the diffuse its
	 * complement); no multiple scattering term (there are no metals here, where it shows). */
	indirectSpecular( builder: THREE.NodeBuilder ): void {

		const { radiance, iblIrradiance, reflectedLight } = builder.context as unknown as { radiance: V3; iblIrradiance: V3; reflectedLight: { indirectDiffuse: V3; indirectSpecular: V3 } };
		const dielectric = this.envSpecular( specularColor as unknown as V3 ).toVar();
		const single = mix( dielectric, this.envSpecular( diffuseColor.rgb ), metalness );
		reflectedLight.indirectSpecular.addAssign( radiance.mul( single ) );
		reflectedLight.indirectDiffuse.addAssign( iblIrradiance.mul( diffuseContribution ).mul( float( 1 ).sub( luminance( dielectric ) ) ).mul( 1 / Math.PI ) );
		meterAmbient.addAssign( iblIrradiance );

	}

	/** The diffuse a light gives, from the view-space direction to it L and its light without the
	 * cosine: what the surface reflects (on its albedo, before 1 / pi) and its metering illuminance
	 * (lighting/MeterLight.ts). Here Lambert under 1 - lum( F( V.H ) ) (`diffuseFresnel`);
	 * VegetationLighting's adds its wrap, light through and thin layer. The key's `direct` and every
	 * local light (lighting/LocalLights.ts) shade by it, so one light is the same on every kind. */
	diffuseResponse( L: V3, light: V3 ): { reflected: V3; meter: V3 } {

		const r = diffuseFresnel( normalView, positionViewDirection, L, specularColor as unknown as V3, specularF90 as unknown as F ).toVar();
		return { reflected: light.mul( diffuseContribution ).mul( r.x.mul( r.y ) ), meter: light.mul( r.x ) };

	}

	/** The physical direct light along `shadeDirection` (the key lifted under a low key: the key is
	 * the scene's one direct light, RenderStages.ts `sun`): Lambert under 1 - lum( F ) for the diffuse, and
	 * the shared GGX (lighting/SurfaceBRDF.ts) for the specular, from the disk's representative
	 * point round that direction. */
	direct( input: THREE.LightingModelDirectInput, builder: THREE.NodeBuilder ): void {

		void builder;
		const L = this.shadeDirection( input.lightDirection as V3 ).toVar(), V = positionViewDirection, N = normalView;
		const lightColor = input.lightColor as V3;
		const diffuse = this.diffuseResponse( L, lightColor );
		( input.reflectedLight.directDiffuse as V3 ).addAssign( diffuse.reflected.mul( 1 / Math.PI ) );
		meterKey.addAssign( diffuse.meter );
		// the specular from the disk's representative point (lighting/SurfaceBRDF.ts `keyGlint`)
		const R = keyGlint.radius;
		const Lg = glintDirection( L, N, V, normalViewGeometry, R.tan() ).toVar();
		const Hg = Lg.add( V ).normalize();
		const NL = clamp( dot( N, Lg ), 0, 1 ).toVar();
		const NV = clamp( dot( N, V ), 0, 1 );
		const NH = clamp( dot( N, Hg ), 0, 1 );
		const VHg = clamp( dot( V, Hg ), 0, 1 );
		const spec = specularGGX( NL, NV, NH, VHg, this.alpha!, specularColorBlended as unknown as V3, fresnel90( specularColorBlended as unknown as V3 ) )
			.mul( glintNormalisation( this.alpha!, R, keyGlint.spread ) );
		( input.reflectedLight.directSpecular as V3 ).addAssign( lightColor.mul( NL ).mul( spec ) );

	}

	/** What dims the local lights as the model dims its key (none here; the water column's
	 * attenuation in UnderwaterLightingModel). */
	localScale(): THREE.Node<'float'> | THREE.Node<'vec3'> | undefined {

		return undefined;

	}

	/** What dims the irradiance gathered before the local lights (the ambient lights and the
	 * ground's bounce): none here; VegetationLighting's occlusion from root to tip. */
	ambientScale(): THREE.Node<'float'> | undefined {

		return undefined;

	}

	indirect( builder: THREE.NodeBuilder ): void {

		addGroundBounce( builder );
		const scale = this.ambientScale();
		if ( scale ) ( builder.context as unknown as { irradiance: V3 } ).irradiance.mulAssign( scale );
		// the local lights' diffuse (lighting/LocalLights.ts), once per material: shaded as the key's
		// diffuse, under the ambient diffuse's occlusion and what dims the key
		const local = this.localScale();
		const occlusion = diffuseOcclusion( builder );
		shadeLocalLights( builder, ( L, light ) => this.diffuseResponse( L, light ), local ? occlusion.mul( local ).toVar() : occlusion );
		super.indirect( builder );

	}

}
