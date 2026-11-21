/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, float, vec3, vec4, positionWorld, normalViewGeometry, cameraViewMatrix,
	normalize, exp, dot, select, smoothstep, mix, max,
} from 'three/tsl';
import type { Shared } from '../shared.ts';
import type { WaveMaps } from '../ocean/WaveMaps.ts';
import { refractedSunDirectionNode, refractedSunBeamScaleNode } from './RefractedSun.ts';
import { GroundBounceLightingModel } from './GroundBounce.ts';
import { meterAmbient } from './MeterLight.ts';
import { ambient as grade } from './AmbientGrade.ts';
import type { AmbientReceiver } from './AmbientSample.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type B = THREE.Node<'bool'>;

// The light below the water's surface: one model for everything lit under it.
//
// The light below the water's surface, one model for everything lit under it (the terrain and the
// sea floor, TerrainMaterial.ts), as tidewater's UnderwaterLighting (4811ba4) lights it: the key
// light refracted at the mean surface and carried down the water column to the point under the long
// waves' surface (Beer-Lambert over the slant path, the share that enters the water and the beam's
// change of area), faded in over the first 8 cm, its light gathered and spread by the waves above
// into the caustics (WaveMaps.ts `causticsAt`, mean 1); and the ambient and specular light under
// the mean level, dimmer with the depth, faded in over the first 10 cm. Both paths attenuate by
// kappa = a + bb (shared.ts `sigA`, `bb`): the light scattered forward stays in the downwelling
// stream, so only absorption and backscattering take it away (Lee et al. 1999, Appl. Opt. 38:3831).
// The ambient light goes by Beer-Lambert over 1.2 times the depth, 1.2 being about 1 / mu_d, the
// inverse mean cosine of a uniform sky refracted into the water: mu_d = ( 1 + cos theta_c ) / 2 =
// 0.83, theta_c the critical angle, 48.6 degrees at n = 1.333. It enters the water at 0.934, the
// diffuse Fresnel transmittance of a uniform sky from air into water at n = 1.333 (the
// cosine-weighted mean of 1 - F over the hemisphere), which the environment's specular, the local
// lights and the ambient diffuse under the water all take. The long waves and the mean level come
// from the wave maps baked once a frame (WaveMaps.ts `wavesAt`), so the waterline on the rock rises
// and falls with the swell. The water shader adds the view path from the point back to the eye, so
// a material lit here must not. (tidewater leaves the key at its direction in the air with no share
// or beam factor: the same on a level floor; here the refracted direction lights steep submerged
// walls as the sun reaches them, and the share that enters the water follows the sun's height.)
//
// The ambient diffuse is the ambient volume's (lighting/AmbientSample.ts) above the water, as on
// every lit island material, and below it on the cliffs and the stones: the probes hold the island's
// shade and leave the water to the attenuation here, and a wall read by the environment's irradiance
// at its normal took the open sky's bright horizon, unshaded, as a pale band along its foot seen
// through the surface. A material that takes the volume by default weighs it out under the water,
// to the environment's light (`UnderwaterModelOptions.ambient`).
//
// Under overcast's closed deck, on its share, the ambient diffuse below the mean level takes the
// sea's sky gain in place of the land's (AmbientGrade.ts `seaSky`, on the attenuation's share):
// the light down there is the light going down through the sea, which the water drawn over it
// takes (the deck's gain, sky/Weather.ts DECK_SKY_GAIN, and the low sun's lift), not the land's
// (AmbientRelight.ts), so the floor seen through the shallows and the water over it take one
// gain. The specular and the local lights take neither gain. Every other sky keeps the land's (a
// gain of exactly 1: the values are those without it, but the shader's code differs, which moves a
// few pixels of a frame by one level).
//
// A stone under water reflects far less than in the air, the water's index standing close to its
// own: the material gives that share (`UnderwaterOptions.specular`), for the key and the sky alike,
// faded in over the first 10 cm. At the air's reflectance the sky lay over the sea floor and the
// walls as an additive pale veil.
//
// Under a deck the land's key keeps the weather's floor through the clouds (Sky.ts
// `keyDirectVisibility`, three's sun, which every receiver takes), and the sea's is the same (Sky.ts
// `seaKeyVisibility`): the sunlight a closed deck passes down and what a
// cumulus' shadow keeps of the key reach the sea's surface as they reach the land. A point under
// the long waves' surface takes three's sun with no share of its own: the floor's light enters
// ahead of the surface, as the land takes it along the key, so the share of the key that enters
// the water (`sunIntoWater`), its refracted direction, the beam's change of area and its path down
// the column apply to it as to the key's own beam, and the seabed under a deck is lit by the key
// the water drawn over it takes.
//
// A material calls `underwaterLight` with its own key above the water (the terrain's canopy shadowing;
// nothing for a material that never reaches the surface) and installs `UnderwaterLightingModel` as
// its lighting model; it also sets `userData.refractedSolarReceiver`,
// so RefractedSun.bindTerrain gives it the refracted sun's shadow map.
//

/** The light below the water at a point, for `UnderwaterLightingModel`. */
export interface UnderwaterLight {
	// whether the point lies below the long waves' surface
	under: B;
	// the water column above it to the mean level (m, negative above the water)
	depth: F;
	// the key light's colour multiplier: under water its path down the column, above it the
	// material's own `aboveSun`
	sun: V3;
	// the refracted key's direction in view space, which replaces the light's under water
	sunView: V3;
	// the ambient specular's and the local lights' multiplier: the attenuation down the column, 1 above
	// the water
	ambient: V3;
	// the specular's multiplier: the material's share below the mean level, 1 above the water
	specular: F;
	// the ambient diffuse's multiplier: `ambient` with the sea's sky gain over the land's on the
	// light carried down the column (AmbientGrade.ts `seaSky`), 1 above the water
	diffuse: V3;
}

export interface UnderwaterOptions {
	// the key's multiplier above the water (the terrain's canopy shadowing); 1 without one
	aboveSun?: V3;
	// the specular's share below the surface, the material's reflectance against water over its
	// reflectance against air; 1 without one
	specular?: number;
}

/** The underwater light at `positionWorld` (see the header). */
export function underwaterLight( shared: Shared, waveMaps: WaveMaps, { aboveSun, specular = 1 }: UnderwaterOptions = {} ): UnderwaterLight {

	// the water above this point: the long waves' surface for the key, the mean level for the
	// ambient (the wave maps; a level sea above the water's reach)
	const p = positionWorld;
	const sigT = shared.sigA.add( shared.bb ); // kappa = a + bb
	const waves = waveMaps.wavesAt( p ).toVar();
	const dSun = shared.waterLevel.add( waves.x ).sub( p.y ).toVar();
	const dW = shared.waterLevel.add( waves.w ).sub( p.y ).toVar();
	const under = dSun.greaterThan( 0 );
	// Snell's direction and the beam-area factor must change together. For a horizontal
	// floor, muWater * (muAir / muWater) preserves transmitted horizontal irradiance;
	// steep submerged walls now receive the correct refracted projection as well.
	const waterSun = refractedSunDirectionNode( shared.sunDir );
	const waterSunView = normalize( cameraViewMatrix.mul( vec4( waterSun, 0 ) ).xyz );
	// (`under` first builds the wave maps' variable, outside the branch, so that every later reader
	// finds it declared.) The caustics are read only below the water, with explicit gradients (the
	// pixel's footprint from the view ray), so their reads may sit inside it.
	const sun = Fn( () => {

		const att = ( aboveSun ?? vec3( float( 1 ) ) ).toVar();
		If( under, () => {

			const path = exp( sigT.negate().mul( dSun ).div( max( waterSun.y, 0.15 ) ) );
			const caustics = waveMaps.causticsAt( p, dSun, waves.yz );
			const beam = path.mul( shared.sunIntoWater ).mul( refractedSunBeamScaleNode( shared.sunDir ) ).mul( caustics );
			att.assign( mix( att, beam, smoothstep( 0, 0.08, dSun ) ) );

		} );
		return att;

	} )();
	const depth = dW.max( 0 );
	// (1.2 ~ 1 / mu_d, 0.934 the sky's diffuse transmittance into the water: the header)
	const ambAtt = exp( sigT.negate().mul( depth ).mul( 1.2 ) ).mul( 0.934 ).toVar();
	const below = smoothstep( 0, 0.1, depth ).toVar();
	const ambient = mix( vec3( 1 ), ambAtt, below );
	const diffuse = mix( vec3( 1 ), ambAtt.mul( grade.seaSky ), below );
	return { under, depth: dW, sun, sunView: waterSunView, ambient, diffuse, specular: mix( float( 1 ), float( specular ), below ) };

}

export interface UnderwaterModelOptions {
	// Fade the key out over the last few degrees before the geometric terminator (the terrain's
	// detail normals: past the terminator they still tilt some texels toward the light, which the
	// shadow map cannot shade). Off for a double-sided surface, whose geometric normal faces away
	// on its inner side.
	terminator?: boolean;
	// How the material reads the ambient volume (lighting/AmbientSample.ts), above and below the
	// water alike; null keeps the environment's irradiance everywhere. Left out, the volume at
	// degree 1 above the water, its share falling to nothing over the first 30 cm below the mean
	// level (`AMBIENT_DEPTH`), where the environment's light carried down the column takes over.
	ambient?: Omit<AmbientReceiver, 'weight'> | null;
}

// the depth (m) below the mean level over which the volume's share of the ambient falls to none
const AMBIENT_DEPTH = 0.3;

/** The physical lighting model with the key light and the ambient carried down the water column
 * (`underwaterLight`), on GroundBounceLightingModel (the ground's bounce, the low key's lift above
 * the water, the lighting layer's debug views). */
export class UnderwaterLightingModel extends GroundBounceLightingModel {

	light: UnderwaterLight;
	options: UnderwaterModelOptions;

	constructor( light: UnderwaterLight, options: UnderwaterModelOptions = {} ) {

		super();
		this.light = light;
		this.options = options;
		this.ambient = options.ambient === undefined ? { bands: 1, weight: smoothstep( 0, AMBIENT_DEPTH, light.depth ).oneMinus() } : options.ambient;

	}

	direct( input: THREE.LightingModelDirectInput, builder: THREE.NodeBuilder ): void {

		const lightDirection = select( this.light.under, this.light.sunView, input.lightDirection as V3 );
		// The bump terminator: past the geometric terminator the stone's detail normals still
		// tilt some texels toward the light, and the shadow map cannot shade them, since the
		// surface there occludes only itself (the bias clears it). Their sunlit specks twinkled
		// inside the cliff's own shadow along its creases. A surface facing away from the light
		// cannot see it; fade the key light out over the last few degrees before the terminator.
		// Above the water the terminator is the lifted key's (lighting/KeyLift.ts), as the sun's map
		// is drawn along it: a face turned from the true key but toward the lifted one takes the
		// lifted diffuse, lit or shaded by that map.
		let lightColor = ( input.lightColor as V3 ).mul( this.light.sun );
		if ( this.options.terminator !== false ) lightColor = lightColor.mul( smoothstep( 0, 0.06, dot( normalViewGeometry, this.shadeDirection( lightDirection ) ) ) );
		const color = lightColor.toVar();
		super.direct( { ...input, lightDirection, lightColor: color }, builder );
		( input.reflectedLight.directSpecular as V3 ).mulAssign( this.light.specular );

	}

	/** Under the water the refracted sun's own direction: the low key's lift is the air's
	 * (lighting/KeyLift.ts). */
	shadeDirection( L: V3 ): V3 {

		const lifted = super.shadeDirection( L ).toVar();
		return select( this.light.under, L, lifted );

	}

	/** The local lights reach a submerged point down the water column, as the ambient does. */
	localScale(): V3 {

		return this.light.ambient;

	}

	indirect( builder: THREE.NodeBuilder ): void {

		// the light the sunlit ground bounces up (above the water only: GroundBounce.ts), then three's
		super.indirect( builder );
		const reflected = ( builder.context as unknown as { reflectedLight: { indirectDiffuse: V3; indirectSpecular: V3 } } ).reflectedLight;
		// the ambient diffuse down the column, with the sea's sky gain in the land's (the specular
		// takes neither gain: AmbientGrade.ts)
		reflected.indirectDiffuse.mulAssign( this.light.diffuse );
		// the metering illuminance's ambient down the column, as the diffuse (lighting/MeterLight.ts)
		meterAmbient.mulAssign( this.light.diffuse );
		reflected.indirectSpecular.mulAssign( this.light.ambient.mul( this.light.specular ) );

	}

}
