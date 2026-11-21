/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, uv, positionWorld, cameraPosition, normalize, smoothstep, dot, vec2, vec3, vec4, float, sqrt,
	max, min, atan, asin, acos, clamp, cos, sin, mix, abs, select, Loop, cross, fwidth, length, step,
} from 'three/tsl';
import { SUN_LIMB_U1, SUN_LIMB_U2 } from '../lighting/Radiometry.ts';
import { SUN_DISC_RADIUS, SUN_DISC_SOLID_ANGLE, SUN_DISC_RADIANCE } from '../lighting/SunDisc.ts';
import { EARTH_RADIUS, MOON_DECK_GLOW } from './CloudModel.ts';
import { MOON_ANGULAR_RADIUS } from './NightSky.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { STORE_CEILING } from '../post/PreExposure.ts';
import type { Sky, SkyHaze } from './Sky.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// The sky's colour nodes: the dome's colour, the bakes' panoramas and the deck's glow.
//
// The sky's colour nodes (Sky.ts, whose header says what the sky draws and why): the dome's colour
// toward each pixel (`domeColour`), the bakes' toward each texel of their panoramas (`bakeColour`,
// the distant sea's body in `seaNode`), the closed deck's glow (`glowToward`) and the mean cloud
// transmittance over the discs (`discCloudTransmittance`), and the shader functions they share.
// Each takes the sky and is built when its material is; a name in backticks with no file is this
// file's or Sky.ts's.

// Directions the discs' cloud transmittance averages over each disc (discCloudTransmittance): the
// disc is about a degree wide, and 32 dim it without visible steps.
const DISC_SAMPLES = 32;
// The height over a low deck's base (metres) the bake fogs its clouds out to, where a closed deck's
// march gathers most of its light (the dome reads the march's own depth: `domeColour`).
const DECK_LIGHT_HEIGHT = 150;

// The atmosphere is the single source of sky radiance and solar extinction. The visible sky
// includes the sun disk; the water panorama and PMREM omit it because direct lighting already
// reflects the sun. This follows tidewater's separation of view and reflection radiance.
export const dirToEquirect = Fn( ( [ d ]: [ V3 ] ) => {

	return vec2( atan( d.z, d.x ).mul( 0.5 / Math.PI ).add( 0.5 ), asin( clamp( d.y, - 1, 1 ) ).div( Math.PI ).add( 0.5 ) );

} ).setLayout( { name: 'dirToEquirect', type: 'vec2', inputs: [ { name: 'd', type: 'vec3' } ] } );

export const equirectToDir = Fn( ( [ q ]: [ V2 ] ) => {

	const phi = q.x.sub( 0.5 ).mul( 2 * Math.PI ).toVar(), th = q.y.sub( 0.5 ).mul( Math.PI ).toVar();
	return vec3( cos( th ).mul( cos( phi ) ), sin( th ), cos( th ).mul( sin( phi ) ) );

} ).setLayout( { name: 'equirectToDir', type: 'vec3', inputs: [ { name: 'q', type: 'vec2' } ] } );

// The finite wave grid ends before the geometric horizon. Continue it with the reflection
// of a flat distant sea, approaching unit Fresnel at the horizon without a dark colour step.
const seaPanorama = Fn( ( [ d, clearSky, sea ]: [ V3, V3, V3 ] ) => {

	const c = clamp( abs( d.y ), 0, 1 ).toVar();
	const g = sqrt( c.mul( c ).add( 1.333 * 1.333 - 1 ) ).toVar();
	const a = g.sub( c ).div( g.add( c ) );
	const b = c.mul( g.add( c ) ).sub( 1 ).div( c.mul( g.sub( c ) ).add( 1 ) );
	const fr = a.mul( a ).mul( b.mul( b ).add( 1 ) ).mul( 0.5 );
	return mix( sea, clearSky, select( d.y.greaterThanEqual( 0 ), 1, fr ) );

} ).setLayout( { name: 'skySeaPanorama', type: 'vec3', inputs: [
	{ name: 'd', type: 'vec3' }, { name: 'clearSky', type: 'vec3' }, { name: 'sea', type: 'vec3' },
] } );

// A sky radiance or irradiance as lent to the land (`look.landSky`): drawn toward `tint.rgb` at its
// own luminance by `tint.w`, times `gain`. Linear in the light, so the graded probe's integral is
// the integral of the graded panorama.
export const lendLand = Fn( ( [ light, tint, gain ]: [ V3, V4, F ] ) => {

	return mix( light, tint.rgb.mul( dot( light, vec3( 0.2126, 0.7152, 0.0722 ) ) ), tint.w ).mul( gain );

} ).setLayout( { name: 'lendLand', type: 'vec3', inputs: [ { name: 'light', type: 'vec3' }, { name: 'tint', type: 'vec4' }, { name: 'gain', type: 'float' } ] } );

// The closed deck's glow about the key light toward `d` (CloudModel.ts DECK_GLOW), drawn where the sky is
// composed, not marched: smooth and round whatever the deck's depth and its rolls. The drawn deck
// takes it with both lobes; the lighting's sources (the bakes) take one smooth lobe of their own
// about the sun (Clouds `bakeGlowLobe`, `bakeGlowGain`) by the day's weight (`deckDay`,
// `glowToward`). The deck's mean radiance `deck` times the key's
// share of the light on its tops, two forward lobes toward the sun and the same two toward the moon
// mixed by the sun's share (each a Henyey-Greenstein lobe of mean 1 over the sphere, a narrow one
// for the peak and a broad one for the tail; the sun's tinted as the deck's tops take its light),
// the clouds' opacity toward `d`, the deck's share of the sky, and `pattern` to the power k: the
// base's local radiance over the deck's mean as its diffuse law gives it there (`deckPattern`), so
// the glow follows the base's own pattern, brightening its thin parts and leaving its dark masses
// dark (at k = 0 a smooth glow). `shares` = ( the key's share, the sun's share, the deck's share,
// k ), `form` = ( the narrow lobe's mean cosine, its weight, the broad lobe's mean cosine, its
// weight ).
const deckGlow = Fn( ( [ d, deck, opacity, sunDir, moonDir, tint, shares, form, pattern ]: [ V3, V3, F, V3, V3, V3, V4, V4, F ] ) => {

	const hg = ( g: F, nu: F ) => float( 1 ).sub( g.mul( g ) ).div( max( float( 1 ).add( g.mul( g ) ).sub( nu.mul( g ).mul( 2 ) ), 1e-4 ).pow( 1.5 ) );
	const lobes = ( nu: F ) => hg( form.x, nu ).mul( form.y ).add( hg( form.z, nu ).mul( form.w ) );
	const forward = mix( vec3( lobes( dot( d, moonDir ) ) ), tint.mul( lobes( dot( d, sunDir ) ) ), shares.y );
	return deck.mul( forward ).mul( shares.x.mul( shares.z ).mul( opacity ).mul( pattern.pow( shares.w ) ) );

} ).setLayout( { name: 'deckGlow', type: 'vec3', inputs: [
	{ name: 'd', type: 'vec3' }, { name: 'deck', type: 'vec3' }, { name: 'opacity', type: 'float' }, { name: 'sunDir', type: 'vec3' },
	{ name: 'moonDir', type: 'vec3' }, { name: 'tint', type: 'vec3' }, { name: 'shares', type: 'vec4' }, { name: 'form', type: 'vec4' },
	{ name: 'pattern', type: 'float' },
] } );

// The base's local radiance toward `d` over the deck's mean as the column's diffuse law gives it
// there, ( 3 / 7 )( 1 + 2 mu ) times the deck's mean radiance (CloudKernels.ts makeMarch: its mean over
// level ground is the deck's), moved toward a uniform deck by `flat` as the march moves it (Clouds
// `deckFlat`), so it has no trend with elevation and averages about 1 over the deck: what the glow
// follows (`deckGlow`). Held in [ 1e-6, 4 ], so its power is finite at any k
// and a lone bright seam cannot flare.
const deckPattern = Fn( ( [ d, local, deck, flat ]: [ V3, V3, V3, F ] ) => {

	const law = max( d.y, 0 ).mul( 6 / 7 ).add( 3 / 7 ).toVar();
	const mean = dot( deck, vec3( 0.2126, 0.7152, 0.0722 ) ).mul( law.add( float( 1 ).sub( law ).mul( flat ) ) );
	return clamp( dot( local, vec3( 0.2126, 0.7152, 0.0722 ) ).div( max( mean, 1e-20 ) ), 1e-6, 4 );

} ).setLayout( { name: 'deckPattern', type: 'float', inputs: [ { name: 'd', type: 'vec3' }, { name: 'local', type: 'vec3' }, { name: 'deck', type: 'vec3' }, { name: 'flat', type: 'float' } ] } );

/** The distant sea's body as the bake lights it (`look.seaKnee`): the light that reaches the sea, the
 * deck irradiance of this bake (drawn before its panoramas) and the sun's on the level water, over
 * the knee and at most 1, times the colour; `landSea` is its CPU twin. At the knee and above it is
 * `seaBody` itself, bit for bit. */
export function seaNode( sky: Sky ): V3 {

	const light = dot( sky.bakeDeckIrradianceNode, vec3( 0.2126, 0.7152, 0.0722 ) ).add( sky.seaSun );
	return sky.seaBody.mul( min( light.div( sky.seaKnee ), 1 ) ).add( sky.seaNight );

}

/** The environment's source toward each texel of the panorama: the sky over the island, its
 * clouds and their reflection in the distant sea, seen through `haze` when given (see
 * `fogEnvironment`). */
export function bakeColour( sky: Sky, haze?: SkyHaze ): V4 {

	return Fn( () => {

		const q = uv();
		const d = equirectToDir( q ).toVar();
		const cloud = sky.layerAt( vec2( q.x, abs( q.y.sub( 0.5 ) ).mul( 2 ) ) ).toVar();
		const up = vec3( d.x, abs( d.y ), d.z );
		// (the environment's source takes the light lent to the land: `look.landSky`; the haze as the
		// dome lays it, from the sea's level: see `fogEnvironment`)
		const clear = sky.gapSky( up ).toVar();
		// the closed deck's glow over the clouds, whole in the seen panorama, about the sun alone in
		// the lighting's sources (`bakeGlow`); its deck level without the weather's gain, which the
		// source takes below.
		const bakeDeck = sky.bakeDeckIrradianceNode.div( max( sky.ambientGain, 1e-6 ) ).toVar();
		const pattern = deckPattern( up, cloud.rgb, bakeDeck, sky.clouds.deckFlat ).toVar();
		const glow = glowToward( sky, up, bakeDeck, cloud.a, pattern ).mul( sky.bakeGlow );
		const sunGlow = glowToward( sky, up, bakeDeck, cloud.a, pattern, true ).mul( float( 1 ).sub( sky.bakeGlow ) );
		const lit = cloud.rgb.add( sunGlow.add( glow ) ).toVar();
		const air = clear.mul( cloud.a ).add( lit ).toVar();
		let fogged: V3 = air;
		if ( haze ) {

			// the clear sky through the haze along the whole way, the clouds through it out to the
			// shell's middle height (from the sea: the distance to that sphere, as the dome takes it),
			// under the low deck's share to a little above its base instead, where a deck's light comes
			// from (the dome takes the march's own depth there, which the panorama does not keep)
			const open = haze( up );
			const middle = mix( sky.clouds.shellBottom.add( sky.clouds.shellThickness.mul( 0.5 ) ), sky.clouds.shellBottom.add( DECK_LIGHT_HEIGHT ), sky.clouds.lowShare ).toVar();
			const inside = middle.mul( middle.add( EARTH_RADIUS * 2 ) ), b = up.y.mul( EARTH_RADIUS );
			const cloudHaze = haze( up, inside.div( sqrt( b.mul( b ).add( inside ) ).add( b ) ) );
			const seen = open.light.add( open.transmittance.mul( clear ) ).mul( cloud.a )
				.add( cloudHaze.light.mul( float( 1 ).sub( cloud.a ) ) ).add( cloudHaze.transmittance.mul( lit ) );
			// (the seen panorama only above the horizon, where the water reads it: what the sea
			// reflects is the sky the dome draws, and under cloud the haze's light is nearly all of
			// that; below the horizon the dome fogs it itself)
			fogged = mix( air, seen, max( sky.bakeHaze, step( 0, d.y ) ) );

		}
		const lent = lendLand( fogged.mul( sky.ambientGain ), sky.bakeLandTint, sky.bakeLandGain );
		// (no stars: starlight is a few hundredths of the moonlit sky's light on level ground, and the
		// sky the eye sees draws them itself)
		const radiance = seaPanorama( d, lent, seaNode( sky ) );
		return vec4( min( radiance.mul( sky.bakeStore ), vec3( STORE_CEILING ) ), mix( cloud.a, d.y, sky.irradiancePass ) );

	} )();

}

/** The dome's colour toward each pixel, seen through `haze` when given: a function of a view
 * direction and a distance along it (the sky's own by default) that returns the light the haze
 * adds and its transmittance (VolumetricHaze.ts `skyHaze`). The clouds are fogged at the
 * distance to the shell's middle height, under the low deck's share at the march's depth, and
 * under the cirrus's share of their opacity at the distance to the layer, the sky behind
 * them along the whole way. */
export function domeColour( sky: Sky, haze?: SkyHaze ): V4 {

	return Fn( () => {

		const d = normalize( positionWorld.sub( cameraPosition ) ).toVar();
		// The disc: cut where the cosine to the sun crosses the radius's, over one pixel's change
		// of that cosine (a clean edge at any size), darkened toward the limb by the quadratic law
		// (Radiometry.ts); the angle from the chord, which keeps its precision near the centre.
		const nu = dot( d, sky.displaySunDir ).toVar();
		const mask = clamp( nu.sub( Math.cos( SUN_DISC_RADIUS ) ).div( max( fwidth( nu ), 1e-7 ) ), 0, 1 );
		const r = asin( clamp( length( d.sub( sky.displaySunDir ) ).mul( 0.5 ), 0, 1 ) ).mul( 2 / SUN_DISC_RADIUS ).toVar();
		const m = float( 1 ).sub( sqrt( max( float( 1 ).sub( r.mul( r ) ), 0 ) ) ).toVar();
		const limb = float( 1 ).sub( m.mul( SUN_LIMB_U1 ) ).sub( m.mul( m ).mul( SUN_LIMB_U2 ) );
		// The integral of the visible disk is the key light's irradiance, times the displayed share
		// (lighting/SunDisc.ts SUN_DISC_RADIANCE). A shared scene unit
		// scale keeps the normalised disk finite in fp16 without dimming it independently. Its
		// colour is the air's analytic transmittance toward it (Atmosphere.ts discTransmittance).
		// The clouds hide it by their transmittance once, per pixel, in the composite below: an
		// edge crossing the disc dims it as smoothly as the march's opacity changes.
		const disk = sky.atmosphere.discTransmittance( d ).mul( sky.displaySolarIlluminance )
			.mul( mask ).mul( limb ).div( SUN_DISC_SOLID_ANGLE / SUN_DISC_RADIANCE ).mul( smoothstep( 0, 0.001, d.y ) );
		// Above the horizon the clear sky and the clouds are both live. Below, the baked distant-sea reflection continues the waves.
		const lower = sky.samplePanorama( d, float( 0 ) ).toVar();
		const cloud = sky.clouds.sampleView( d ).toVar();
		const above = d.y.greaterThanEqual( 0 ).toVar();
		const stars = sky.night.stars( d ).mul( sky.night.moonOcclusion( d ) );
		// What the clouds hide: the clear sky, the discs and the stars (below the horizon, the
		// reflection with its own clouds), seen through the haze along the whole way; the discs
		// take the clouds' transmittance once, here, and nowhere else.
		const behind = select( above, sky.gapSky( vec3( d.x, max( d.y, 0 ), d.z ), sky.domeGapGain ).mul( sky.displayAmbientGain ), lower.rgb )
			.add( disk.add( sky.night.moon( d ) ).add( stars ).mul( select( above, float( 1 ), lower.a ) ) );
		const through = select( above, cloud.a, float( 1 ) ).toVar();
		// the closed deck's glow over the clouds above the horizon (`deckGlow`), before the haze,
		// which hides it with the clouds by their distance
		// (the deck under the night's seen gain, `seenDeck`, after the glow's pattern)
		const pattern = deckPattern( d, cloud.rgb.mul( sky.displayAmbientGain ), sky.deckIrradianceNode, sky.clouds.deckFlat ).toVar();
		const glow = glowToward( sky, d, sky.deckIrradianceNode, cloud.a, pattern ).toVar();
		const clouds = select( above, cloud.rgb.mul( sky.seenDeck ).mul( sky.displayAmbientGain ), vec3( 0 ) ).add( glow.mul( step( 0, d.y ) ) ).toVar();
		if ( ! haze ) return vec4( behind.mul( through ).add( clouds ), 1 );
		const open = haze( d );
		// The clouds through the haze out to the shell's middle height: its light over their
		// opacity, their own (premultiplied) through its transmittance.
		// (from the march's own camera height, held under the base (Clouds `camY`): the depths
		// `sampleViewFog` returns are measured from there, and above the middle the root's
		// argument would turn negative)
		const middle = sky.clouds.shellBottom.add( sky.clouds.shellThickness.mul( 0.5 ) ).toVar();
		const camY = sky.clouds.camY;
		const r0 = camY.add( EARTH_RADIUS ).toVar(), b = r0.mul( max( d.y, 0 ) ).toVar();
		// (the distance to that sphere, written without a difference of near-equal terms)
		const inside = middle.sub( camY ).mul( r0.add( middle.add( EARTH_RADIUS ) ) ).toVar();
		const mid = inside.div( sqrt( b.mul( b ).add( inside ) ).add( b ) ).toVar();
		// Under the low deck's share, out to the march's own opacity-weighted depth instead
		// (Clouds `sampleViewFog`, its `depth`): a closed deck's light comes from its lowest
		// hundred metres, and at a low deck the middle lies some 2.5 times farther, which fogged
		// the deck at 5 - 12 degrees about twice as densely as its distance does.
		const fog = sky.clouds.sampleViewFog( d, mid );
		const reach = mix( mid, fog.depth, sky.clouds.lowShare ).toVar();
		// The cirrus lies above the cumulus' middle (the clear and fair skies' 6.6 km
		// against its 8.5), so farther along a ray: where its fibres make the opacity, out to
		// the layer instead, so the haze between fades them toward the horizon by their own
		// distance. One distance per pixel, mixed by the layer's share of the opacity there:
		// exact where no cumulus stands in front, and where one does it hides the layer. With
		// no layer (cloudy and overcast skies) the share is none and the reach the cumulus'
		// alone.
		const thinShare = clamp( fog.thin.div( max( float( 1 ).sub( through ), 1e-4 ) ), 0, 1 );
		const streakHeight = sky.clouds.streakHeight;
		const thinInside = max( streakHeight.sub( camY ).mul( r0.add( streakHeight.add( EARTH_RADIUS ) ) ), 0 ).toVar();
		const thin = thinInside.div( sqrt( b.mul( b ).add( thinInside ) ).add( b ) );
		const cloudHaze = haze( d, mix( reach, thin, thinShare ) );
		const fogged = cloudHaze.light.mul( float( 1 ).sub( through ) ).add( cloudHaze.transmittance.mul( clouds ) );
		const result = open.light.add( open.transmittance.mul( behind ) ).mul( through ).add( fogged ).toVar();
		if ( debugFlags.streakMask !== true ) return vec4( result, 1 );
		// (`flags=streakMask`: the cirrus's pixels where its opacity as seen passes 0.1 in a
		// red of the same luminance, those where it lies between 0.02 and 0.1 in such a green,
		// for `img.py stats --mask`)
		const luminance = dot( result, vec3( 0.2126, 0.7152, 0.0722 ) ).toVar();
		const edge = select( fog.thin.greaterThan( 0.02 ).and( above ), vec3( 0, luminance.div( 0.7152 ), 0 ), result );
		return vec4( select( fog.thin.greaterThan( 0.1 ).and( above ), vec3( luminance.div( 0.2126 ), 0, 0 ), edge ), 1 );

	} )();

}

/** The closed deck's glow toward `d` (`deckGlow`) over a deck of mean radiance `deck`, through
 * clouds of transmittance `transmittance` there, following the base's local radiance over its
 * mean `pattern` (`deckPattern`), with this frame's shares (Clouds `columnTop`); with `sunOnly`
 * the lighting's sources' glow: about the sun alone, one smooth lobe of its own (Clouds
 * `bakeGlowLobe`, `bakeGlowGain`), by the day's weight (`deckDay`, so exactly none by night). */
export function glowToward( sky: Sky, d: V3, deck: V3, transmittance: F, pattern: F, sunOnly = false ): V3 {

	const top = sky.clouds.columnTop();
	const c = sky.clouds;
	if ( sunOnly ) {

		// the lighting's sources: one lobe of their own (`bakeGlowLobe`, `bakeGlowGain`), smooth
		// (the pattern's power 0)
		return deckGlow( d, deck, float( 1 ).sub( transmittance ), sky.atmosphere.sunDir, sky.atmosphere.moonDir, c.lobeTint.rgb,
			vec4( top.keyShare, 1, c.broadShare, 0 ), vec4( c.bakeGlowLobe, c.bakeGlowGain, 0, 0 ), pattern ).mul( sky.deckDay );

	}

	// the day's form by `deckDay`, the moon's own by night (each exact at a weight of 1 or 0)
	const night = vec4( ...MOON_DECK_GLOW );
	const day = vec4( c.deckGlowLobe, c.deckGlow, c.deckGlowBroadLobe, c.deckGlowBroad );
	const form = night.mul( float( 1 ).sub( sky.deckDay ) ).add( day.mul( sky.deckDay ) );
	// the pattern's power: away from the sun the closed deck's by day (`deckGlowFar`, from
	// `deckGlowFrom` to `deckGlowTo` degrees), by overcast's share and the day's weight; the
	// moon's and the broken skies' DECK_GLOW_TEXTURE (exact at a weight of 0)
	const away = smoothstep( c.deckGlowFrom, c.deckGlowTo, acos( clamp( dot( d, sky.atmosphere.sunDir ), - 1, 1 ) ).mul( 180 / Math.PI ) );
	const k = mix( c.deckGlowTexture, mix( c.deckGlowTexture, c.deckGlowFar, away ), c.overcastShare.mul( sky.deckDay ) );
	return deckGlow( d, deck, float( 1 ).sub( transmittance ), sky.atmosphere.sunDir, sky.atmosphere.moonDir, c.lobeTint.rgb,
		vec4( top.keyShare, top.sunShare, c.broadShare, k ), form, pattern );

}

/** The mean cloud transmittance over the sun's disc (r) and over the moon's (g): DISC_SAMPLES
 * directions spread over each disc (a Vogel spiral, uniform by area) read from the clouds'
 * panorama, for the sky's constants texel (Pipeline.ts `skyConstants`), which the sun's flare and
 * the light shafts read; the dome's discs do not. One ray would switch the flare on and off as a
 * cloud's edge crosses the disc. */
export function discCloudTransmittance( sky: Sky ): V4 {

	const mean = ( centre: V3, radius: number ) => {

		const tangent = normalize( select( abs( centre.y ).greaterThan( 0.99 ), vec3( 1, 0, 0 ), vec3( centre.z, 0, centre.x.negate() ) ) ).toVar();
		const bitangent = cross( centre, tangent ).toVar();
		const sum = float( 0 ).toVar();
		Loop( DISC_SAMPLES, ( { i } ) => {

			const r = sqrt( float( i ).add( 0.5 ).div( DISC_SAMPLES ) ).mul( radius );
			const a = float( i ).mul( 2.399963 );
			sum.addAssign( sky.clouds.sample( normalize( centre.add( tangent.mul( cos( a ).mul( r ) ) ).add( bitangent.mul( sin( a ).mul( r ) ) ) ) ).a );

		} );
		return sum.div( DISC_SAMPLES );

	};
	return Fn( () => vec4( mean( sky.displaySunDir, SUN_DISC_RADIUS ), mean( sky.night.moonDir, MOON_ANGULAR_RADIUS ), 0, 1 ) )();

}
