/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater):
 * the haze's deep exponential height layer (its scale height) and its phase function,
 * from `AirHaze.js`.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, uniform, texture, texture3D, textureLoad, textureStore, instanceIndex, instancedArray, uv, int, ivec2, ivec3,
	vec2, vec3, vec4, float, max, min, clamp, exp2, log2, sqrt, mix, dot, normalize, select, fract, floor, ceil, varying, smoothstep, length,
} from 'three/tsl';
import type { Sky } from '../sky/Sky.ts';
import { EARTH_RADIUS, type Clouds } from '../sky/Clouds.ts';
import type { SunShadow } from '../lighting/SunShadow.ts';
import { ATMOSPHERE_RADIUS, MIE_G, PLANET_RADIUS, cornetteShanks, skyRayleighPhase, type Atmosphere } from '../sky/Atmosphere.ts';
import { bandPush, bandToDisplay, bandToDisplayColor } from '../sky/BandMatrix.ts';
import type { HazeKey } from './HazeKeys.ts';
import { AEROSOL_OPEN, type HazeMedium, expShare, hazeFormKeep, meanExtinction } from './HazeDepth.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { tune, tuneObject, tuneUniform } from '../debug/Tune.ts';
import { duskForce } from '../sky/DayCycle.ts';
import { markSkyRetune } from '../sky/SkyRetune.ts';
import type { FrameGraph, GraphResources, ResourceHandle, SizeClass } from '../post/graph/FrameGraph.ts';
import { TemporalRest, restEpoch } from '../post/TemporalRest.ts';
import { preExposed, sceneUnits, storePress } from '../post/PreExposure.ts';
import { shared } from '../shared.ts';
import { FOG_NOISE, fogDrift } from './FogNoise.ts';
import { DROPLET, dropletPhase } from './FogDroplets.ts';
import { halton } from '../post/Jitter.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// The atmospheric haze and the sea's fog: a froxel volume lit by the atmosphere, over the scene.
// The sine of the elevation the sun's rows are read at under a deck (see the header)
const DECK_ROW = 0.5;
// The elevation (17 degrees) under which the haze reads the earth's shadow as at it (see the header)
const SHADOW_ELEVATION_SIN = Math.sin( THREE.MathUtils.degToRad( 17 ) ), SHADOW_ELEVATION_COS = Math.cos( THREE.MathUtils.degToRad( 17 ) );
// The sea's reflectance of a diffuse sky: Fresnel's for water (n 1.33) averaged over the
// hemisphere with the cosine, about 0.066.
const SEA_DIFFUSE = 0.066;
// How far overcast's air turns the open sky's scatter colour it takes (see the header) toward its
// luminance, `haze.deckAirGrey` on a debug page.
const DECK_AIR_GREY = 0.93;
// Overcast's deck (see the header): the deck's diffuse law ( 3 / 7 )( 1 + 2 mu ) seen through the
// air's Cornette-Shanks phase, over an even deck's of the same irradiance, as a quartic in the
// view's sine of elevation whose coefficients are quadratics in the phase's anisotropy g (rows
// from mu^0 up, each from g^0 up): a least-squares fit to the quadrature over the anisotropy's
// tunable range, 0.5 to 0.9, within 0.004 at MIE_G and 0.037 at 0.9 (at MIE_G 0.71 looking
// level, 1.12 straight up). A single cubic fitted at MIE_G alone was 0.07 off at 0.76.
const DECK_DIFFUSE = [
	[ 0.8457, - 0.0508, - 0.2595 ],
	[ - 0.0331, 0.3896, - 0.2483 ],
	[ 0.3233, - 1.0249, 1.7246 ],
	[ 0.0476, - 0.2387, 0.327 ],
	[ - 0.3245, 1.3378, - 1.5456 ],
];
// Debug parameters for measuring the fog's light in the light kernel: the sun's visibility scaled
// (1 as it is, 0 the air all in shadow) and the island's and the clouds' shadow lifted (1 none).
const LIGHT_DEBUG = {
	sun: tuneUniform( 'haze.debug.sun', 1, { min: 0, max: 1 } ),
	noShadow: tuneUniform( 'haze.debug.noShadow', 0, { min: 0, max: 1 } ),
};
// the points along the sun's ray at which the banks' own shade is taken (metres from the froxel)
// and the stretch of the ray each stands for
const BANK_SHADE = [ [ 4, 8 ], [ 14, 12 ] ] as const;
// the ground layer alone, for HazeDepth.ts `meanExtinction`: the exponential layer empty
const GROUND_ONLY = vec3( 0, 1000, AEROSOL_OPEN );
// how far above the fog layer's top, in its widths, the drift is taken (the layer's density there
// is under 1/16 of its deep part's)
const DRIFT_REACH = 8;
// The distances (metres along the view) over which the air's in-scatter of the night sky goes from
// the light the sky sheds to the light the dome draws it at (shared.ts `seenNight`): the near air,
// which veils the land the camera looks at, keeps the sky's own light, and the far air, which meets
// the sky at the horizon, takes the seen sky's (as the far sea's reflection does, Water.ts
// SEEN_SKY). See the header.
const SEEN_AIR = { near: 200, far: 2000 } as const;

// The atmospheric haze and the sea's fog as one participating medium in a froxel volume, after Patry,
// "Real-Time Samurai Cinema", SIGGRAPH 2021 (its volumetric haze, built on Wronski, SIGGRAPH 2014,
// and Drobot, Digital Dragons 2017):
//   - the medium's density is analytic and integrable along any ray: a ground layer, the talk's
//     sigmoidal layer, dense below a soft top and thinning above it, and a deep exponential aerosol
//     layer. The ground layer is measured from the sea and the haze varies with height alone,
//     never with the camera: one medium for the volume, the sky, the probe's rays and the CPU's
//     readings, from any start. Its shape is the hour's key's (HazeKeys.ts: the layer's top over
//     the sea and its width, the aerosol's share and scale height). The talk's slide prints the
//     layer as lerp( d0, d1, 1/2 z^2 / ( z^2 + k_s^2 ) + 1/2 ), whose weight is even in z and no
//     step; this is the algebraic sigmoid we take it to stand for, rho_f ( 1 - u / sqrt( w^2 + u^2 ) ) with u the
//     height over the layer's top, whose mean over any span of heights has a closed form. The keys
//     lay it thin on the sea (a 14 m top, a step about 10 m wide), so the island's camera, 68 m up,
//     looks down through clear air and only the last metres of its rays cross the dense part,
//     while the far sea's, the horizon's and the low sky's rays run long through it and are hazed;
//     a camera lowered into the layer sees its near land through it too. Far above the top it thins
//     as the inverse square of the height, so the distant haze thins less with height than an
//     exponential layer would. The weather's haze sets rho_f (and the deep
//     layer's density, the key's share of it); its sea fog and the morning mist thicken it. Under a
//     low deck the aerosol layer ends at the deck's base (the key's `aerosolTop`: HazeKeys.ts), so
//     the air thins to nothing between the base and the eye's view of the deck's gaps; without a
//     deck it runs on over any height (HazeDepth.ts AEROSOL_OPEN) and every depth keeps its bits. An
//     optical depth is a ray's length times the mean density over the heights it spans: the closed
//     form along the ray, which holds its value on a level ray with no branch. The only local
//     variation in the density is the fog layer's banks (below, The fog layer). The closed form
//     lives in
//     HazeDepth.ts, which the clouds' march shares: `prepareMedium` writes the medium into the
//     clouds' copy too (`cloudMedium`), and the low deck's view march ends where it closes.
//   - the extinction is spectral: each band's is the density times the band's weight sigma, keyed
//     by the hour and the weather (HazeKeys.ts: by day red : green : blue = 0.38 : 0.53 : 0.655, at
//     the golden hour 0.45 : 0.53 : 1.19). The air loses its blue before its red (far land and sea
//     turn blue-grey and the low sun's light reddens), while cloud and fog droplets scatter grey
//     (the overcast's keys are still to be set toward it). The scattered light and the opacity are
//     integrated per channel, so the light volume holds three channels of L / alpha and, in its
//     fourth, the fog layer's drift (below), and the composite recomputes each channel's opacity. Every opacity the eye sees goes
//     through the band-to-display matrix (sky/BandMatrix.ts, the identity by default): the light
//     kernel's slice weights, the composite's opacity and the CPU `transmittance`.
//   - a froxel grid over the view in slices each deeper than the one before by a fixed share (the
//     talk's PS4 grid is 128 x 64 x 64; the high tier's 160 x 90 makes a froxel about 16 render pixels
//     across at 1440p). Its size is fixed, so no kernel follows the render's scale. The talk's slices
//     (10 cm to 100 km, each 20 % deeper) serve a camera a few metres from its subject; the island's
//     camera looks from 20 - 200 m, where they were 6 - 20 m deep and averaged the cliffs' and the
//     plants' shadows out of the air. From 10 m to 30 km, each 15 % deeper, 19 of the 64 slices lie
//     in that range instead of 12 (6.6 m deep at 50 m), at no cost; four times the froxels across
//     (the talk's PC density) sharpened the shadows less and cost about six times as much.
//   - the light is the atmosphere's (3b in the talk's terms): each froxel takes the sky's in-scatter
//     along its own view direction at its own altitude (Atmosphere.ts's fog table, without the
//     phase; a direction under the froxel's horizon reads the horizon's sky side), so distant haze
//     is the horizon sky's colour, the same in-scatter seen through more medium, over the far sea
//     as over the sky, and haze and sky meet without a seam at every hour. It comes in two shares, each
//     turned toward its luminance by the key's `desaturate` after its phase:
//       - Rayleigh: the table's Rayleigh share through the sky's Rayleigh phase with its forward
//         lean (Atmosphere.ts `skyRayleighPhase`) and without the sky's polariser (the hazy air's
//         light is not polarised as the clear sky's is), lit by the open sky (its colour times the
//         ambient visibility) and by the sun (its colour times the sun's visibility), both under
//         the earth's shadow over the seen sky (Atmosphere.ts `skyShadow`). The shadow is read with
//         the column held 17 degrees or more over the horizon (SHADOW_ELEVATION): the air low over
//         the horizon away from a set sun lies in front of the sky's own shadowed band, and takes
//         the shadow only once its edge has risen past 17 degrees, so no dark belt lies round the
//         horizon. The haze's own light per unit of its opacity (its light over one minus its
//         transmittance) falls from 60 degrees from the sun to 90 and rises again toward 170: the
//         sky's lean alone, the side away from the sun not lifted by a polariser of the haze's own.
//         By day that light stands under the bare sky's behind it, so the hazed low sky is a little
//         darker than the bare one on the sun's half (HazeKeys.ts, the weights); at a low sun, the
//         sky behind it darker, the haze is the lighter. This reaches everything the haze's sky
//         feeds, the sky's fog colour and the environment bakes, so the panorama, and the sky's
//         irradiance by about a percent at twilight. By day under a clear or fair sky the sun's colour here is
//         0: the air's blue is sky light, and sunlit blue haze would glow.
//       - aerosol: the table's Mie share through the Cornette-Shanks lobe, times the sun's colour
//         and visibility: the warm glow toward the sun. Like the Rayleigh share's, its rows hold the
//         earth's shadow already (the table's march takes the sun's light only where the planet
//         leaves the sun in view), so it takes no gate of its own at the froxel's horizon. Such a
//         gate (a smoothstep 3 degrees either side of the sun's rise over the froxel's horizon, the
//         horizon dipping by sqrt( 2 h / R )) took the shadow a second time, and with the haze all
//         but opaque just over the horizon the low sky toward a set sun fell into a dark band
//         under the sky over it.
//     The colours are the sky's illuminance colour (Atmosphere `sunIlluminance`), the open sky's
//     times the weather's diffuse gain and the sun's times its direct share, the sun's before the
//     weather's day sky (Sky.ts `sunBeamIlluminance`: the day sky lowers the sky's own light the
//     sun gives, not the beam through the air below it, which the land's key keeps too; under
//     overcast's closed deck it sets with the land's key, by the disk's visible share); the sun's Rayleigh
//     colour is the aerosol's times the key's `sunRayleigh`. Each share has one weight, keyed
//     (HazeKeys.ts), the weather's gain over both (2 under cloud) and `gain` (1, an A/B parameter) over
//     the sum; the rows, the phases and the colours being the sky's own, a weight is about the
//     share's light per unit of opacity over the sky's (at noon and an evening sun of 12 the
//     Rayleigh share's, at a weight of 1, about 1.0 - 1.3 of the sky's 60 - 90 degrees from the
//     sun). A gain over the keys' gains and weights (1.87 times 1.73 times 2.03 by day) had
//     stacked to 6.6 on it. The moon lights the haze
//     the same way, its two shares added to the sun's:
//     the moon's fog table (its shares under the moon's gains, Atmosphere.ts `moonRayleigh`,
//     `moonMie`, as the sky's are), its direction (the night key's: Sky.ts `track`), phases and
//     earth shadow, and its colours the sun's form on the moon's light (Atmosphere `moonIlluminance`,
//     the moon's sky, for its sky share; Sky.ts `moonBeamIlluminance`, the moon's beam before the
//     weather's day sky, for its direct share), except for its aerosol share, whose light and lobe are its own
//     (MOON_AEROSOL): a pale blue-green of the same luminance as the moon's sky light, times a gain,
//     through a Cornette-Shanks lobe of its own anisotropy, so the glow of the air around the moon is
//     tuned apart from the sun's while the day keeps the sky's lobe. Both bodies take the key light's
//     visibility below, which follows the moon at night, when the sun's shares are dark under the
//     earth's shadow and the horizon. The moon's table is read along the moon's direction at the
//     froxel's altitude, the sun's along the view. A row along the view is the in-scatter over that
//     ray's whole path and grows toward the horizon, so read along the view the haze's moonlight was
//     the phase's angle from the moon times the view's elevation: on the moon's side the far sky
//     brightened low down from the table (the moon's aerosol share alone about two thirds of the
//     sky 5 - 15 degrees up and more than 25 degrees from the moon), and the glow round the moon
//     came out flat in angle, its outer rings taking in the bright low rows. A haze lit by the moon
//     scatters it once, its angular shape set by the phase alone (Wronski, "Volumetric Fog",
//     SIGGRAPH 2014; Hillaire, "Physically Based and Unified Volumetric Rendering in Frostbite",
//     SIGGRAPH 2015): read along the moon, the moonlight a unit of haze scatters depends on the view
//     only through the phases, the view's elevation enters only through the haze's density and depth
//     along it, and toward the moon, where the view's elevation is the moon's, the level is the
//     one the view's row gave. Measured on the composite's sky at a moonlit tune (the moon 24
//     degrees up, the view centred on it): within 2 degrees of the moon unchanged (under 0.5 %); the ratio of
//     the sky 2 degrees from the moon to 17 degrees from it from 3.0 to 3.8; more than 25 degrees
//     from the moon, the sky 0 - 5 degrees up about 0.39 of what it was, 5 - 15 degrees up 0.73,
//     15 - 30 degrees 0.98, and 30 - 60 degrees up 7 % brighter (the rows along the moon are longer
//     than the high views'); facing away from the moon 0.77, 0.86, 0.99 and 1.01. The far sea
//     still meets the horizon sky without a step (both are the haze's there). By day the moon's
//     light is 0: a clear noon's and a golden hour's paused frames are unchanged byte for byte. The
//     earth's shadow stays the view's (held at 17 degrees): with the moon 10 to 40 degrees up
//     (KeyLift.ts) it is about 1. The level follows the moon's elevation through its row's
//     length (a lower moon's row is longer, so its haze brighter, where a real haze would dim as the
//     moonlight crosses more air); the moon keeps one direction all night (KeyLift.ts), so the haze
//     keeps one level from dusk to dawn, and the elevation is a tuning the haze's gain answers.
//     Under a clouded day's deck (HazeKeys.ts `deck`) the sun's rows are read at one elevation,
//     30 degrees (DECK_ROW), instead of along the view, by the deck's share: the air under a deck
//     is lit by the deck over it alike in every direction, not by a clear sky, so it no longer
//     brightens toward the horizon as a clear sky's long low rows do. Under cloud the sky the dome
//     shows is nearly all this air laid over the clouds (sky/SkyDome.ts `domeColour`: the clouds' own
//     undersides stand 4 - 8 stops under it), so along the view the deck's low band stood .7 - 1.6
//     stops over the sky's irradiance over pi, and read level .3 - .7. The bake, the reflection and
//     the froxels read the same rows, so the sky, its light on the land and the far haze move
//     together; with a level band and the irradiance about unchanged the deck shows half a stop
//     darker against the land.
//     Under the deck's share the open sky's light in the sun's rows, and in the moon's, gives way
//     to the deck's own (`scatterToward`): the air under a deck sees the deck over it and the sea
//     under it, nothing else of the sky. Its light is the sky's irradiance over pi without the haze over it (Sky.ts
//     `deckIrradianceNode`: the clear sky in the gaps times their transmittance plus the clouds'
//     own light, at the weather's ambient gain, baked and cross-faded with the sky), scattered
//     by the air (the deck's half in the deck's form through the air's phase, below): half of it from the deck, half from the sea's reflection of it under
//     (SEA_DIFFUSE), times the albedo and the open sky's visibility, with no gain, no grey and no
//     row: the deck and the air are one scale by construction. The sun's own light (its rows
//     through the clouds' shadow) stays as it was; the deck's share is cloudy's from a sun of 3
//     degrees and overcast's until the twilight (Weather.ts `deckLight`), so a low sun's air and a
//     fair or clear sky keep the open sky's light, and by night, once the moon's key has come in,
//     the whole clouded share (HazeKeys.ts `deck`): the deck is then lit by the island's moonlight
//     through its column (Clouds.ts `columnKey`), and the air under it by the deck, as by day.
//     Measured on the overcast night's horizon view (paused): the sky .38 / .35 to .41 / .40
//     (the boxes' L), its low band .38 / .34 to .41 / .40, the meadow .18 / .19 to .19 / .20. Measured on the horizon views (paused, fresh
//     loads): overcast's sky about as bright as with the rows (box median L .30 to .32, its spread
//     .08 to .07), cloudy's unchanged (.75), its chroma .022 to .010: the deck's light is mostly the
//     clouds' own, a warm grey, the gaps' blue a small share of it.
//     Overcast's deck is thick (HazeKeys.ts `overcastDeck`, its share), and its light leaves the
//     base in the column's angular form (Clouds.ts, the column light): a diffuse part
//     ( 3 / 7 )( 1 + 2 mu ), the standard overcast sky's luminance law ( 1 + 2 sin h ) / 3 (the CIE
//     standard overcast sky of Moon and Spencer, 1942, as given in ISO 15469:2004 /
//     CIE S 011/E:2003), which the drawn deck by day under overcast flattens toward a uniform deck
//     (Clouds.ts `deckFlat`) with the level irradiance kept, so the air, which takes the law
//     unflattened, takes the same level light; and over it, on the key's light alone, a forward
//     Henyey-Greenstein lobe about the key (Clouds.ts `forwardWeight`, `forwardLobe`), light added
//     to the deck's. By overcast's share the air takes it through its own phase: the diffuse part
//     as the deck's law weighted by the aerosol's phase over an even deck's of the same irradiance, a quartic in the
//     view's sine of elevation fitted in the anisotropy (DECK_DIFFUSE); the forward part as the
//     deck's own lobe about the key, on the key's share alone (the lobe's light over the deck's
//     level, `columnTop`'s key over its key and sky; under overcast's share the deck spreads its
//     forward light by the diffusion law, sky/CloudModel.ts OVERCAST_BROAD, so it comes in the diffuse part
//     and the lobe's share is the rest's alone), the diffuse part taking the rest of the deck's
//     irradiance. The lobe is the deck's, not the product of the deck's and the air's
//     anisotropies that one scattering of it would give: that product, a smaller mean cosine, would
//     dim the low air round the sun's place against the deck behind it. The open sky's light in the
//     sun's rows takes the same by that share, the law by the view's elevation h,
//     ( 3 + 6 sin h ) / 7, plus the forward part, so the glow round the sun's place is one form in
//     the deck and in the air under it. The sea's half stays even: it reaches the air from below,
//     where neither the deck's law nor its lobe points, and is SEA_DIFFUSE of the deck's light.
//     A cloudy deck's light takes the same form by the deck's whole share (HazeKeys.ts `deck`), the
//     open sky's light still overcast's alone. Scattered evenly, its forward lobe's light, on the
//     sun's side, would reach the views away from the sun as fully as those toward it, and the air
//     there would stand brighter than the deck behind it under a high sun. At night HazeKeys.ts
//     gives the whole clouded share to `deck` and none to `overcastDeck`, so a night's deck,
//     overcast's too, lights its air by this path: in the deck's form about the moon, overcast's
//     forward light spread by the clouds' share of it (sky/CloudModel.ts OVERCAST_BROAD), by the share of
//     its form each column keeps (below).
//     Along a long near-level ray the light the eye takes in has crossed a column of air several
//     optical depths deep, and light that crosses such a column is scattered on the way: our
//     approximation built on the similarity relation (van de Hulst, "Multiple Light Scattering",
//     1980; the multiple-scattering approximation of Hillaire, "A Scalable and Production Ready Sky
//     and Atmosphere Rendering Technique", EGSR 2020, takes what more than one scattering gives as
//     isotropic) takes the share that keeps its direction after an optical depth tau as
//     exp( - ( 1 - g ) tau ), g the aerosol's
//     anisotropy, and the rest is even. Over a column of depth T lit evenly along its length, the
//     light the eye takes from depth tau is weighed by exp( - tau ), so the share still in the deck's
//     form is the closed form ( 1 - e^-( 2 - g ) T ) / ( ( 2 - g ) ( 1 - e^-T ) )
//     (`hazeFormKeep`, T the column's whole depth to the grid's far end); the rest takes the even
//     light. It is 1 on a short column and 1 / ( 2 - g ) on a deep one: the eye sees about one
//     optical depth into the air, over which ( 1 - g ) of it is redirected; with the sky's
//     aerosol (g 0.76, Atmosphere.ts MIE_G) the share on a deep column is 0.81, so it moves the
//     low sky near the line and leaves the sky higher up, where the column is short. Overcast's
//     deck by day keeps its whole form (by overcast's share, HazeKeys.ts `overcastDeck`): its form
//     was tuned with the even light in it. Every other deck, and every deck at night, takes it.
//     It leaves out the light the air sends to a point from the level band round it (the layer is
//     thin straight up, so that light is a small part), which would pull a little further.
//     Overcast's air takes the deck's light at the deck's own scale, as the rule above has it: under
//     a deck the light the air scatters is the deck's, so no share of it is dropped. The low sky away
//     from the sun stays darker than the deck above it by the deck's own form and lobe; a smaller
//     share of the deck's light would darken it further and roughen the deck's blocks low in the
//     sky. Its colour is not the deck's warm grey but the open sky's scatter colour (the sun's
//     Rayleigh rows at the sun's light, before the key's grey) at the deck light's luminance, turned
//     toward grey by DECK_AIR_GREY (0.93), so the low deck keeps a faint cool cast; a lower value
//     takes more of the sky's blue. Cloudy, fair and clear take no share of it (overcast's share is
//     0 there, as at night), and the extinction is unchanged.
//   - the fog layer: under a clear or fair sky by day (HazeKeys.ts `fogLayer`: the day, the golden
//     hour and the morning) the ground layer is a fog layer, its top 12 m over the sea and its
//     edge 2 m wide (below the crowns), its density at the sea set by the key's visibility V (km): Koschmieder's 3.912 / V on green, less
//     the aerosol layer's share there, over the sigmoid's weight at the sea,
//     1 + top / sqrt( top^2 + width^2 ). Its droplets scatter grey: the key's `neutral` turns the
//     sigma's red and blue toward green. Its light is the droplets' own (`dropletToward`), not the
//     sky's rows: the sun's colour and visibility through a two-lobe Henyey-Greenstein phase
//     (forward g 0.8, backward -0.3 with a share of 0.3: the forward peak round the sun and the
//     faint back-scatter of water droplets) summed over four octaves of multiple scattering
//     (Wrenninge, "Oz: The Great and Volumetric", SIGGRAPH 2013 talk; Hillaire, "Physically Based
//     Sky, Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH 2016 course: octave k scaled by
//     a^k, its extinction by b^k and its anisotropy by c^k, a the key's `scatter`, b and c 0.5),
//     each octave attenuated by the layer's optical depth from the point to its top toward the sun
//     (over the sun's sine, held at 0.05 or more); and the open sky's light, the deck's irradiance
//     over pi (Sky.ts `deckIrradianceNode`) from above and the sea's reflection of it (SEA_DIFFUSE)
//     from below, scattered evenly, times the ambient visibility; both times the albedo. A froxel
//     takes the air's light and the droplets' by the fog's share of its density, so the sun's side
//     of a lit fog is white and the shadow under the island and the bamboo stays dark: the octaves
//     bring the lit fog's level up, the island's and the clouds' shadows still cut it. The dome,
//     the seen panorama and the probe's (`openSky`) take the same light by the ground layer's
//     share of the mean density along their rays, without the drift. The land's sky light (the
//     environment's source, Sky.ts `fogEnvironment`) takes the ground layer without the fog layer
//     and no droplets' light: the land's key loses nothing to the fog layer, so its sky light takes
//     nothing from it either.
//     The fog gathers into banks that drift (FogNoise.ts): the fog's part of each froxel's density,
//     up to DRIFT_REACH of the layer's widths over its top, is multiplied by a factor of mean 1 that
//     the wind carries and that slowly reshapes, so the fog thins and thickens across the sea and
//     over the island while the layer's height profile stays its mean. Its mean is 1 by
//     construction, everywhere and at every threshold, so the frame's mean opacity holds as the
//     banks pass and needs no measure of its own. A bank shades itself: the sun's optical depth
//     through the droplets (`dropletToward`) adds the banks' excess density, their factor less 1,
//     at BANK_SHADE's two points along the sun's ray, each over its stretch of the ray, so a
//     bank's sunward side is lit and its far side and the fog under it go grey, the way a cloud's
//     do. The light kernel stores in the volume's alpha the ratio of the drifted optical depth to
//     the undrifted one, each summed from the eye to the slice, and the composite multiplies the
//     exact depth it recomputes by it (`fogOver`'s `drifted`): the density anti-aliasing keeps its
//     smooth depth and shows the banks at the grid's resolution. The banks are a function of the
//     wind's integrated travel and the gusts' time, so a paused frame holds them. The clouded
//     share (HazeKeys.ts) turns the layer back into the hour's thin haze, except overcast's own sea
//     fog, and the night has none.
//   - the ambient visibility (the open sky's share that reaches a froxel) is the visibility volume's
//     second channel, sampled, filtered and held at rest with the key light's: 0 under the island's
//     cover and 1 above it, rising over 1.5 m either side of the cover's top (the rain's collision
//     map, RainCollision.ts: the highest surface seen from straight above, the bamboo's crowns, a
//     cliff's overhang; the sea off the island), times the ambient probes' sky visibility, never
//     below 0.024. Where the top is the sea's it is 1, under the surface and in the air just over
//     it alike: water hides nothing of the sky, and the slices under the surface show only through
//     the composite's tricubic filter, whose footprint reaches a slice or more past the sea's
//     depth, so their dark pulled the sea's pixels about 5 % under the air's own light. The
//     land's rise is unchanged (the slices inside the hills stay dark). That floor is fixed: the sky barely changes across its
//     range, so it waits for measures of the haze under cover. The probes' read is HazeCover `skyVisibility`,
//     which the pipeline fills with
//     lighting/AmbientSample.ts `ambientSkyVisibility` (1 on the open sea; the volume is installed
//     with the world, so the shadow kernels compile then, not in the warm-up: HazeCover `late`):
//     under the bamboo, the overhangs and in the island's hollows the haze darkens, the open haze
//     is unchanged. The cover's redraws (a Grow's) restart the rest.
//   - the shadow is the sun's stepped shadow maps (SunShadow.volumeVisibility) and the clouds'
//     shadow along the key light (`cloudShaft`: the clouds' map, its air's channel, read at the point's ground point
//     along the clouds' own light, the flat beam the map was built along, sharpened against its blur
//     so the gaps show as shafts, lifted above the shell's base and floored; the gain 10, no floor and no cut
//     toward the light until set), never below the floor a closed deck keeps (`keyFloor`, Sky.ts
//     `deckKeyFloor`: the light the deck passes down in place of the key it hides, so the air under
//     the deck's glow lights toward the sun by day and the moon by night through its own phase;
//     0 under every other sky), sampled at four points of each froxel whenever it is recomputed
//     (the frame's jittered point and three more the R3 sequence's steps from it, averaged, so the
//     history converges to the cell's mean) and filtered over frames, the history weighing 0.95
//     (the ambient visibility with it): what any caster or cloud cuts out of the lit air shows as shafts, whether or not the
//     sun is in view. The filter keeps the shadow's share (and the occlusion's) only, never the
//     light, so a change of the light shows at once and leaves no trail.
//   - a second kernel integrates each column of froxels front to back, every slice's optical depth
//     exact, and stores the in-scattered light divided by its opacity (L / alpha; the alpha channel
//     holds the fog layer's drift, above): a smooth
//     function of depth, unlike either alone. The composite recomputes the opacity at each pixel,
//     exactly, from the density functions (times the drift's ratio), and multiplies it back in: the thin layers over the
//     horizon and the sea do not band at the grid's depth steps (the talk's density
//     anti-aliasing). It samples the volume with a tricubic B-spline (eight trilinear taps; Sigg
//     and Hadwiger, "Fast Third-Order Texture Filtering", GPU Gems 2, 2005), which joins the shafts
//     to their openings.
//   - the haze covers the sky (`sky`), and the sky lays it on itself: the dome (Sky.ts `fogDome`,
//     `skyHaze` here) takes the atmosphere's colour through the haze along the whole way to the
//     atmosphere's top (or to the sea for a ray below the horizon), its optical depth taken along a
//     straight ray whose height climbs with the direction alone, never lifted by the planet's curve,
//     so the haze closes over the sky near the horizon (two parts, the far one from the height the
//     curve lifts the ray to, took too little air at 1 degree), plus the haze's light read at
//     that distance or the grid's far end, whichever is nearer; the clouds in front of it take the
//     same out to their shell's middle height, over their opacity; the composite leaves the sky's
//     pixels. So the haze's kernels run ahead of the scene passes, which declare the light volume
//     as a read (Pipeline.ts); they read the sun's maps, the clouds' shadow and the camera, never
//     the scene's depth, and keep the unjittered projection under the temporal modes. Both layers are laid over the atmosphere whole: its own aerosols
//     (Atmosphere.ts `mieScale`) are not taken out of them, the sky's fog being the volume's on top
//     of the atmosphere. The ground layer closes over the horizon, so the far sea and the horizon's
//     sky end in the same haze. The light the sky lends (Sky.ts's bakes, the probes) is the
//     atmosphere's alone: the haze is laid over the rendered image only.
//     By night the moon's open sky light and the deck's that the light volume's air scatters come to
//     the night sky as the dome draws it (`moonColourSeen`, `deckShareSeen`: shared.ts `seenNight`,
//     Weather.ts `seenNight`) by the froxel's distance: the near air keeps the sky's own light, the
//     far air from SEEN_AIR's far end on takes the seen gain whole, as the dome and the far sea's
//     reflection do (Water.ts SEEN_SKY), so the horizon's air meets the drawn sky, while what the
//     sky lights (the bakes, `openSky` among them, the probes, the land) keeps its own level. The
//     moon's direct single scatter keeps its light. Left whole, the haze's sky light stood over the
//     far sea at the horizon as a light band under a darker sky (cloudy, 0.1 in Oklab L over the sky
//     above it); taken at every distance, it took a further 0.1 - 0.25 stop off the land by night
//     (the low camera's cliffs, the horizon views' land), whose veil is the near air's.
//     The dome's own sky ray (the dome's call, `skyHaze`'s `dome`, with no distance) adds a thin
//     haze of its own to the medium's depth: uniform in height over the air's whole height, the
//     key's `skyDepth` (green's optical depth per kilometre of the ray; HazeKeys.ts). Over the
//     island's distances, a few kilometres, its share is negligible, while the sky's rays (about
//     100 km straight up, 280 km at 20 degrees, 480 km at 10) are long enough for it to close the
//     dome, so one depth separates the island's views from the sky by the length of their rays
//     alone. Its light is the volume's light per opacity times the whole opacity, as the medium's:
//     with a depth (a fog night's; every key keeps 0) the moon's disc fades behind it, the sky near
//     and below the moon turns to the haze's phase lobe over an even floor, and the stars and the
//     atmosphere show through only high up, where the rays are short. The clouds' haze, the probe's
//     sky rays, the CPU `transmittance` (the sun's glare, the flares' visibility) and `openSky`
//     (the light the sky lends) read the medium alone.
//   - The light over the land: the composite lays over the land only the key's `groundLight` share
//     of the haze's light, c ( 1 - a ) + L a groundLight, its transmittance the same everywhere
//     (a look's key sets it, look/Look.ts `HazeKey`; at 1 the result is the plain mix); the sea and
//     the sky keep the whole light at every hour, so the far sea and the horizon still end in the
//     haze. The land is told from the sea per pixel (`apply`): the sea is the nearest
//     surface lying in front of the opaque one (debug/RadianceAov.ts `isSea`); where the opaque
//     sample sees the sky and the nearest surface stands more than 2 m and a pixel's height there
//     over the sea (metres on the far sea, else a dark line along the horizon), the island's
//     skyline under the multisampled resolve, it is land. The depth reads and the scaling sit in a
//     branch on the uniform `groundLight` under 1, so the day draws none of it. The probe's surface
//     rays take the land's share unless they ask for the sea's (`probe`).
//
// Rules: the shadow's filter is a history-dependent effect (ARCHITECTURE.md, rule 4). While the scene runs, each
// froxel blends its new sample into its history, reprojected through the previous frame's camera,
// and takes the new sample alone where the history falls outside the grid, after a cut (the key
// light turns more than CUT in a frame, the camera jumps) and after `invalidate`. A scene at rest
// (the key TemporalRest tests: the camera, the clock, the sizes, the epoch's scene count, which a
// restart of the frame loop leaves, the app's signature and the haze's own inputs, the medium's
// among them) converges once, the way the temporal resolve's rest does: after its settle
// frames it averages a fixed sequence of jitters with exact weights, from nothing, then holds the
// result and skips the shadow kernel. A paused frame is then a function of the scene alone,
// however it was reached, and `pending` holds the page's settled signal until it has converged.
// The light kernel runs every frame. Local lights, when there
// are any, add to the light the integration takes per froxel; the talk filters theirs in a volume of
// their own. The light volume holds the light pre-exposed (PreExposure.ts: times the frame's 2^P,
// held at half float's largest), as the scene's colour does: unscaled, its light reaches near 2^0
// toward a low sun and lies under 2^-17 at a clear midnight (its first percentile), a span no fixed
// scale keeps at half float's full precision with room to spare. The kernel
// is a pass of the frame graph, after the frame's `beginFrame`, so its readers (`fogOver`: the
// composite, the dome, `probe`) divide out the same P. It keeps no history: the rest above is the
// shadow's, whose visibility no P scales.

// the shadow's sub-samples per froxel and recompute
const SUB_SAMPLES = 4;
// a key light that turns more than this in one frame is a cut: the shadow's history is dropped
const CUT = THREE.MathUtils.degToRad( 1 );
// a camera that moves more than this in one frame (metres, or radians of turn) is a cut
const JUMP = { distance: 8, angle: THREE.MathUtils.degToRad( 30 ) };
// view z of the sky (the composite's depth holds -1e5 there)
const SKY_Z = - 9e4;
// the sea's step in front of the opaque depth (m; debug/RadianceAov.ts `isSea` too), and the height
// over the sea above which a nearest surface over an opaque sample of the sky is a skyline (m): `apply`
export const SEA_STEP = 0.01;
const GROUND_SKYLINE = 2;
// the rays `probe` takes at once
const PROBE_RAYS = 64;
const _transmittance = new THREE.Color(), _reading = new THREE.Color();
// Rec. 709 luminance, toward which the light's colours turn (`desaturate`)
const LUMINANCE = vec3( 0.2126, 0.7152, 0.0722 );
const luminance = ( c: THREE.Color ): number => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;

// The moon's aerosol light in the haze: its colour as red and blue over green, scaled to the
// Rec. 709 luminance of the moon's beam there (Sky.ts `moonBeamIlluminance` times the weather's
// direct share), so a new colour moves the hue and not the level, times a gain; and the anisotropy of
// its Cornette-Shanks lobe. A pale blue-green under a lobe a little tighter than the sky's: the air
// around the moon glows cool rather than in the moon key's own blue, over a few degrees.
// Tunable on debug pages (debug/Tune.ts `haze.moon.*`).
const MOON_AEROSOL = {
	red: tune( 'haze.moon.red', 0.333, { min: 0.2, max: 1, onChange: markSkyRetune } ),
	blue: tune( 'haze.moon.blue', 0.669, { min: 0.5, max: 1.2, onChange: markSkyRetune } ),
	gain: tune( 'haze.moon.gain', 0.279, { min: 0.25, max: 4, scale: 'log', onChange: markSkyRetune } ),
	anisotropy: 0.797,
};
// the planet's radius and the atmosphere's top (m from its centre)
const ATMOSPHERE_BOTTOM = PLANET_RADIUS * 1e3, ATMOSPHERE_TOP = ATMOSPHERE_RADIUS * 1e3;

/** The froxel grid's size, fixed when the app loads (the quality tier's `post.hazeGrid`). */
export interface HazeGrid { width: number; height: number; depth: number }

export const DEFAULT_HAZE_GRID: Readonly<HazeGrid> = { width: 160, height: 90, depth: 64 };

/** The haze's parameters (the GUI's Haze folder, `__PONLAI__.pipeline.air.settings`). The shadow's
 * temporal filter and the composite's filters are the talk's, the slices fitted to the island's
 * camera (see the header; the medium's shape is the hour's key's, HazeKeys.ts), the phases from how
 * the air brightens toward the sun, the shares' weight and the gain by eye on a few points of the sky
 * and the island. Those the environment's bake lays over its source (`openSky`: BAKE_SETTINGS)
 * rebake it when they change, however they are set (`bakeStale`, RenderStages.ts `retuneSky`). */
export const HAZE_DEFAULTS = {
	// the slices: the grid's near and far view depth (metres) and each slice's depth over the last's
	near: 10, far: 30000, growth: 1.15,
	// the sea fog and the morning mist: the ground layer's deep green extinction per unit of the
	// weather's fog (1/m: one unit sees 1 km, the threshold of fog, Koschmieder's 3.912 / 1000)
	fogExtinction: 3.912 / 1000,
	// the light: the medium's single-scattering albedo, the aerosol share's phase (a Cornette-Shanks
	// lobe of the sky's aerosol anisotropy, Atmosphere.ts MIE_G, half its peak about 12 degrees off
	// the sun, 1/340 of it behind) and a gain over both shares (1: the keys' weights carry the level)
	albedo: 1, anisotropy: MIE_G, gain: 1,
	// the ambient visibility: its floor under cover, and the half-width (metres) of its rise
	// through the cover's top
	ambientFloor: 0.024, ambientSoftness: 1.5,
	// the shadow's filter: a running frame's weight (1 - the history's, 0.95), and the frames a rest
	// averages
	blend: 0.05, restFrames: 16,
	// A/B switches: the jittered shadow samples, the tricubic filter (else trilinear), the density
	// anti-aliasing (else the opacity the grid holds), the haze over the sky
	jitter: true, tricubic: true, densityAA: true, sky: true,
	// a debug view: 0 off, 1 the key light's visibility in the froxel at each pixel's depth, 2 the
	// haze's opacity (green's display opacity in every channel; the sky 2), 3 the haze's light alone
	// (the scene under it black)
	debug: 0,
};

export type HazeSettings = typeof HAZE_DEFAULTS;

// The settings `openSky` reads (the slices, the fog's extinction, the light, the switch over the
// sky): the environment's bake holds them as they stood when it ran.
const BAKE_SETTINGS = [ 'near', 'far', 'growth', 'fogExtinction', 'albedo', 'anisotropy', 'gain', 'sky' ] as const;

// expShare on each band
const expShare3 = ( x: V3 ): V3 => vec3( expShare( x.x ), expShare( x.y ), expShare( x.z ) );

// The medium's closed form (`meanExtinction`) is HazeDepth.ts's, which the clouds' march shares to
// end where the haze closes.

// Interleaved gradient noise (Jimenez, "Next Generation Post Processing in Call of Duty: Advanced
// Warfare", SIGGRAPH 2014): a fixed offset per froxel, so the frames' common jitter sequence lands
// on different points of neighbouring froxels.
const gradientNoise = Fn( ( [ p ]: [ V2 ] ) => fract( fract( dot( p, vec2( 0.06711056, 0.00583715 ) ) ).mul( 52.9829189 ) ) )
	.setLayout( { name: 'hazeGradientNoise', type: 'float', inputs: [ { name: 'p', type: 'vec2' } ] } );

/** What hides the haze from the open sky, for its ambient visibility (Pipeline.ts gives the rain's
 * collision map, RainCollision.ts, and the ambient probes' sky visibility): the island's highest surface seen from straight above, and
 * the ambient probes' sky visibility. */
export interface HazeCover {
	/** the height (m) of the highest surface under world ( x, z ), the sea's off the island; an
	 * inline node (no layout: the shadow kernel is not the only compute kernel that reads it) */
	topAt( xz: V2 ): F;
	/** the sea's height (m), the top off the island */
	readonly seaLevel: F;
	/** counts the top's redraws (a Grow's), so the shadow's rest starts over on each */
	readonly version: number;
	/** the probes' sky visibility at a world point, 1 off the island; left out, 1. Called when the
	 * shadow kernels build. */
	skyVisibility?: ( p: V3 ) => F;
	/** set when `skyVisibility` reads what exists only later (the ambient volume, installed with the
	 * world, after the graph's warm-up has started): the warm-up then leaves the shadow kernels,
	 * and the owner compiles them once it is ready (`shadowKernels`, Pipeline.ts
	 * `compileHazeShadow`), before the first frame */
	late?: boolean;
}

/** What the app tells the haze each frame (main.ts): the rest's signature (what changes the
 * frame without changing the camera or the clock: a forced strike, the sun maps' redraws) and
 * whether the scene may rest (no sky bake in flight, no transition running). */
export interface HazeFrame { time: number; signature: readonly number[]; stable: boolean }

/** The atmospheric haze and the sea's fog: a froxel volume lit by the atmosphere and shadowed by the sun's
 * maps and the clouds, composited over the scene and the sky (see the header). */
export class VolumetricHaze {

	// the weather's haze (the ground layer's deep extinction in the band of sigma 1, 2 rho_f, 1/m:
	// Weather.ts `haze`) and its sea fog or mist (0 none, 1 sees 1 km)
	readonly haze = uniform( 0 );
	readonly fog = uniform( 0 );
	// the hour's and the weather's key (HazeKeys.ts, `setLook`): each band's sigma, the light's
	// desaturation, the weather's gain over `gain`, the sun's share in the Rayleigh light and the two shares'
	// weights, the medium's shape (the ground layer's top over the sea and its width,
	// the aerosol layer's share, scale height and top, the dome's own depth per kilometre), the share
	// of its light the composite lays over the land (`apply`)
	// and the fog layer's (HazeKeys.ts, see the header, The fog layer): its share of the ground
	// layer, the droplets' share of its light, the extinction's neutral share, Wrenninge's a, the
	// visibility at the sea (km) and the banks' threshold at its top (FogNoise.ts)
	readonly look = { sigma: new THREE.Vector3( 1, 1, 1 ), desaturate: 0.26, gain: 1, sunRayleigh: 0, rayleighWeight: 1, mieWeight: 1,
		top: 10, width: 50, aerosolShare: 0.035, aerosolHeight: 1400, aerosolTop: Infinity, skyDepth: 0, groundLight: 1,
		fogLayer: 0, droplet: 0, neutral: 0, scatter: 0, visibility: 2, coverHigh: FOG_NOISE.high() };
	// the weather's share of the sun's direct light below the clouds and its diffuse gain
	// (Weather.ts `directVisibility`, `ambientGain`), which scale the light's colours; its day sky
	// comes with the open sky's light the atmosphere lends (sky/SkyTrack.ts `skyGain`), not with the sun's beam
	// (`sunBeam`)
	readonly weatherLight = { direct: 1, ambient: 1 };
	// The key light the air scatters: its direction and its irradiance below the clouds (a closed
	// deck leaves no beam to scatter).
	readonly keyDir = uniform( new THREE.Vector3( 0, 1, 0 ) );
	readonly keyColor = uniform( new THREE.Color( 0, 0, 0 ) );
	// The floor of the key's visibility through the clouds under a closed deck (Sky.ts
	// `deckKeyFloor`, set with the medium each frame and before each bake: RenderStages.ts `prepareHaze`):
	// the light the deck passes down in place of the key it hides, which the air takes through its
	// own phase toward the key, so the air under the deck's glow brightens with it. The dome's air and the seen panorama's take it, the lighting's sources
	// never (`openSky`, Sky.ts `bakeKeyFloor`).
	readonly keyFloor = uniform( 0 );
	// the moon's aerosol lobe (MOON_AEROSOL)
	private readonly moonAnisotropy = tuneUniform( 'haze.moon.mieG', MOON_AEROSOL.anisotropy, { min: 0.7, max: 0.82, onChange: markSkyRetune } );
	// the medium's and the light's tunable parameters (debug/Tune.ts; the sun's aerosol share takes
	// the sky's lobe, the moon's its own: `moonAnisotropy`, MOON_AEROSOL)
	readonly settings: HazeSettings = tuneObject( { ...HAZE_DEFAULTS },
		{ anisotropy: 'sky.atmosphere.mieG', ambientFloor: 'haze.ambient.floor' },
		{ anisotropy: { min: 0.5, max: 0.9 }, ambientFloor: { min: 0, max: 0.2 } } );
	readonly grid: Readonly<HazeGrid>;
	/** The shadow's rest (see the header); `pending` while it has not converged. */
	readonly rest = new TemporalRest();

	private readonly sunShadow: SunShadow;
	// the medium: the exponential layer ( density at sea level, scale height ) and the ground layer
	// ( rho_f, top, width ), from the weather and the settings, and each band's sigma
	private readonly layer = uniform( new THREE.Vector3( 0, 1400, AEROSOL_OPEN ) );
	private readonly ground = uniform( new THREE.Vector3( 0, 14, 9.431 ) );
	// the ground layer without the fog layer ( rho, top, width ): the density the air alone gives
	// it, in the layer's shape as it stands, for the land's sky light (`openSky` without `fogLayer`)
	private readonly groundClear = uniform( new THREE.Vector3( 0, 14, 9.431 ) );
	private readonly sigma = uniform( new THREE.Vector3( 1, 1, 1 ) );
	// the dome's own thin haze (the key's `skyDepth`), in the medium's depth per metre before each
	// band's sigma: green's optical depth per metre over green's sigma (`skyHaze` with `dome`)
	private readonly skyDepth = uniform( 0 );
	// the share of the air's light laid over the land (the key's `groundLight`, `apply`)
	private readonly groundLight = uniform( 1 );
	/** A copy of the medium a reader built before the haze holds (the clouds' `haze`, which ends
	 * the low deck's march where this medium closes: sky/CloudKernels.ts `makeMarch`), written with it. */
	cloudMedium: HazeMedium | null = null;
	// the light's settings: albedo, anisotropy, the gain
	private readonly lightSettings = uniform( new THREE.Vector3( 1, 0.76, 1 ) );
	// the light's colours toward their luminance (0 as given, 1 grey)
	private readonly desaturate = uniform( 0.26 );
	// the deck's share (HazeKeys.ts `deck`): the sun's rows read level under it (see the header)
	private readonly deckShare = uniform( 0 );
	// overcast's share of the deck (HazeKeys.ts `overcastDeck`): its sky's shape (see the header)
	private readonly overcastShare = uniform( 0 );
	private readonly deckAirGrey = tuneUniform( 'haze.deckAirGrey', DECK_AIR_GREY, { min: 0, max: 1, onChange: markSkyRetune } );
	// the light's colours per body (the sun's, the moon's: each one's illuminance the sky is drawn
	// with, scaled): the Rayleigh share's lit by the open sky and by the body, the aerosol share's;
	// and the shares' weights
	private readonly ambientColour = [ uniform( new THREE.Color( 0, 0, 0 ) ), uniform( new THREE.Color( 0, 0, 0 ) ) ] as const;
	// The moon's open sky colour and the deck's share on its light at the level the dome draws the
	// night sky (shared.ts `seenNight`, 1 by day, where these equal the ones above bit for bit), which
	// the light kernel's far air takes (SEEN_AIR), while the environment's bake (`openSky`), which
	// lights the land and the sea, keeps the sky's own light. The dome gains the moon's sky only
	// (Sky.ts `domeGapGain`): the sun's twilight sky keeps its light here too.
	private readonly moonColourSeen = uniform( new THREE.Color( 0, 0, 0 ) );
	private readonly deckShareSeen = uniform( 0 );
	private readonly sunRayleighColour = [ uniform( new THREE.Color( 0, 0, 0 ) ), uniform( new THREE.Color( 0, 0, 0 ) ) ] as const;
	private readonly mieColour = [ uniform( new THREE.Color( 0, 0, 0 ) ), uniform( new THREE.Color( 0, 0, 0 ) ) ] as const;
	private readonly weights = uniform( new THREE.Vector2( 1, 1 ) );
	// the fog layer's light (see the header, The fog layer): the droplets' share of the ground
	// layer's light, their lobes ( forward g, backward g, backward share ) and Wrenninge's octaves
	// ( a, b, c ); its banks (FogNoise.ts `fogDrift`): the detail's cell (m), the amplitude, the
	// cells' height over their width and the regions' cell (m); the threshold at the sea and at the
	// ramp's top, the ramp's height (m) and the regions' share of the variance; the detail's
	// roughness and its warp (cells)
	private readonly dropletShare = uniform( 0 );
	private readonly dropletLobes = uniform( new THREE.Vector3( 0.8, - 0.3, 0.3 ) );
	private readonly dropletOctaves = uniform( new THREE.Vector3( 0.3, 0.5, 0.5 ) );
	private readonly fogNoise = uniform( new THREE.Vector4( 100, 0, 1, 0 ) );
	private readonly fogCover = uniform( new THREE.Vector4( 0, 0, 1, 0.5 ) );
	private readonly fogShape = uniform( new THREE.Vector4( 0.6, 0.3, 0, 0 ) );
	private readonly atmosphere: Atmosphere;
	private readonly sunBeam: THREE.Color;
	private readonly moonBeam: THREE.Color;
	// the slices: near, the first slice's depth, the growth and its log2
	private readonly slices = uniform( new THREE.Vector4( 10, 0.6, 1.15, Math.log2( 1.15 ) ) );
	private readonly far = uniform( 100000 );
	// the grid's camera, unjittered (the volume does not shake with the temporal modes' jitter)
	private readonly gridInverse = uniform( new THREE.Matrix4() );
	private readonly gridProjection = uniform( new THREE.Matrix4() );
	private readonly cameraWorld = uniform( new THREE.Matrix4() );
	private readonly cameraInverse = uniform( new THREE.Matrix4() );
	private readonly origin = uniform( new THREE.Vector3() );
	// the previous frame's view-projection (unjittered), for the history's reprojection
	private readonly previousViewProjection = uniform( new THREE.Matrix4() );
	// the composite's rays: the projection the depth it reads was drawn with (jittered under the
	// temporal modes: `setProjectionInverse`)
	private readonly projectionInverse = uniform( new THREE.Matrix4() );
	// the shadow kernel's frame: the jitter's offset, the rest's index (-1 while running), the
	// running blend (1: no history)
	private readonly sequence = uniform( new THREE.Vector3( 0.5, 0.5, 0.5 ) );
	private readonly restIndex = uniform( - 1 );
	private readonly blend = uniform( 1 );
	// the clouds' shafts (cloudShaft): the unsharp mask's gain, the floor, the lift's cut toward the
	// key light
	readonly cloudShafts = uniform( new THREE.Vector3( 10, 0, 0 ) );
	// the ambient visibility's floor and the half-width of its rise (m)
	private readonly ambientSettings = uniform( new THREE.Vector2() );
	// BAKE_SETTINGS as the environment's last bake read them (`noteBake`)
	private baked: HazeSettings | null = null;
	private cover: HazeCover | null = null;
	// the composite's switches: the haze over the sky, tricubic, density anti-aliasing, the debug view
	private readonly switches = uniform( new THREE.Vector4( 1, 1, 1, 0 ) );
	// `probe`'s kernel and its buffers, built on its first call
	private probeKernel: { input: THREE.StorageBufferNode<'vec4'>; heights: THREE.StorageBufferNode<'vec2'>; out: THREE.StorageBufferNode<'vec4'>; kernel: THREE.ComputeNode } | null = null;
	// the render's size, the frame graph's (set in `apply`)
	private resolution!: THREE.UniformNode<'vec2', THREE.Vector2>;

	// The shadow's two volumes (r: the key light's visibility, g: the ambient visibility), written in
	// turn: `written` holds the newest. The light's volume (rgb: L / alpha, a: alpha at each
	// slice's far boundary).
	private readonly visibility: [ THREE.Storage3DTexture, THREE.Storage3DTexture ];
	private readonly light: THREE.Storage3DTexture;
	private visibilityKernels: THREE.ComputeNode[] = [];
	/** The shadow's kernels (both ping-pong sides), for an owner whose cover is `late` to compile. */
	get shadowKernels(): readonly THREE.ComputeNode[] {

		return this.visibilityKernels;

	}
	private lightKernels: THREE.ComputeNode[] = [];
	private resources: { visibility: ResourceHandle; light: ResourceHandle } | null = null;
	private written = 0;
	private historyValid = false;
	private shadowDue = true;
	private frame = 0;
	private readonly key: number[] = [];
	private readonly last = { valid: false, keyDir: new THREE.Vector3(), position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), slices: new THREE.Vector4(), viewProjection: new THREE.Matrix4() };
	private readonly viewProjection = new THREE.Matrix4();
	private readonly position = new THREE.Vector3();
	private readonly quaternion = new THREE.Quaternion();
	private readonly scale = new THREE.Vector3();

	// `sunBeam`, `moonBeam`: the sun's and the moon's illuminance the beam takes (Sky.ts
	// `sunBeamIlluminance`, `moonBeamIlluminance`), the sky's own (`atmosphere.sunIlluminance`,
	// `moonIlluminance`) where none is given (the WGSL build)
	constructor( sunShadow: SunShadow, atmosphere: Atmosphere, grid: Partial<HazeGrid> = {}, sunBeam: THREE.Color = atmosphere.sunIlluminance.value, moonBeam: THREE.Color = atmosphere.moonIlluminance.value ) {

		this.sunShadow = sunShadow;
		this.atmosphere = atmosphere;
		this.sunBeam = sunBeam;
		this.moonBeam = moonBeam;
		this.grid = { ...DEFAULT_HAZE_GRID, ...grid };
		const { width, height, depth } = this.grid;
		const volume = ( name: string ) => {

			const t = new THREE.Storage3DTexture( width, height, depth );
			t.name = name;
			t.type = THREE.HalfFloatType;
			t.format = THREE.RGBAFormat;
			t.generateMipmaps = false;
			t.minFilter = t.magFilter = THREE.LinearFilter;
			t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
			return t;

		};
		this.visibility = [ volume( 'haze.visibility0' ), volume( 'haze.visibility1' ) ];
		this.light = volume( 'haze.light' );
		this.rest.settle = 8;

	}

	// the view depth (metres) at slice coordinate c (slice k spans [ k, k + 1 ])
	private sliceDepth( c: F ): F {

		const s = this.slices;
		return s.x.add( s.y.mul( exp2( c.mul( s.w ) ).sub( 1 ) ).div( s.z.sub( 1 ) ) );

	}

	// the slice coordinate of a view depth
	private sliceCoordinate( z: F ): F {

		const s = this.slices;
		return log2( max( z.sub( s.x ), 0 ).mul( s.z.sub( 1 ) ).div( s.y ).add( 1 ) ).div( s.w );

	}

	// the grid's view ray through screen uv q: its unit view-space direction and its world direction
	private gridRay( q: V2 ): { view: V3; direction: V3 } {

		const ndc = vec4( q.x.mul( 2 ).sub( 1 ), float( 1 ).sub( q.y ).mul( 2 ).sub( 1 ), 1, 1 );
		const far = this.gridInverse.mul( ndc );
		const view = normalize( far.xyz.div( far.w ) ).toVar();
		const direction = normalize( this.cameraWorld.mul( vec4( view, 0 ) ).xyz ).toVar();
		return { view, direction };

	}

	// the world position at screen uv q and view depth z, on the grid's camera
	private gridPoint( q: V2, z: F ): V3 {

		const { view, direction } = this.gridRay( q );
		return this.origin.add( direction.mul( z.div( view.z.negate() ) ) );

	}

	/** The view ray through screen uv `q` of the camera `update` last took, with the inverse
	 * projection `projectionInverse`: in view space (`ray`, its z -1 deep, for a distance from a
	 * view z) and its world direction. For the rain veil drawn after the temporal resolve
	 * (Pipeline.ts), at the canvas's pixels and unjittered, in a full-screen pass: both are linear
	 * across the screen before the direction is normalised, so they are interpolated from the
	 * vertices. */
	viewRay( q: V2, projectionInverse: THREE.Node<'mat4'> ): { ray: V3; direction: V3 } {

		const ndc = vec3( q.x.mul( 2 ).sub( 1 ), float( 1 ).sub( q.y ).mul( 2 ).sub( 1 ), 1 );
		const viewFar = projectionInverse.mul( vec4( ndc, 1 ) );
		const ray = viewFar.xyz.div( viewFar.z.negate() );
		const direction = normalize( varying( this.cameraWorld.mul( vec4( ray, 0 ) ).xyz ) ).toVar();
		return { ray: varying( ray ).toVar(), direction };

	}

	// The key light's and the open sky's visibility at the point q (0 .. 1 on each axis) of the
	// froxel at `cell`.
	private visibilityAt( cell: V3, q: V3, size: V2, sky: Sky ): V2 {

		const p = this.gridPoint( cell.xy.add( q.xy ).div( size ), this.sliceDepth( cell.z.add( q.z ) ) ).toVar();
		// the sun's maps (lit outside them and before the first is drawn). The maps' depth ends
		// below the casters (at the sea's troughs), but a low sun throws their shadow through the
		// air far beyond that: a point past the far end lies behind every caster of its texel, so
		// it is compared at the far end.
		const coordinate = this.sunShadow.coordinate.mul( vec4( p, 1 ) ).xyz.toVar();
		const sun = this.sunShadow.volumeVisibility( vec3( coordinate.xy, min( coordinate.z, 0.99999 ) ) ).toVar();
		const key = mix( sun.mul( max( this.cloudShaft( p, normalize( p.sub( this.origin ) ), sky ), this.keyFloor ) ), float( 1 ), LIGHT_DEBUG.noShadow );
		// the open sky's: dark under the cover's top, lit above it, rising over ± the softness, open
		// wherever the top is the sea's, times the probes' sky visibility, never below the floor
		const cover = this.cover, A = this.ambientSettings;
		const top = cover ? cover.topAt( p.xz ).toVar() : float( 0 );
		const open = cover ? select( top.lessThan( cover.seaLevel.add( 0.02 ) ), float( 1 ), smoothstep( A.y.negate(), A.y, p.y.sub( top ) ) ) : float( 1 );
		const ambient = max( A.x, open.mul( cover?.skyVisibility ? cover.skyVisibility( p ) : float( 1 ) ) );
		return vec2( key, ambient );

	}

	// The clouds' visibility along the key light from the point p of the air. The clouds' shadow map
	// holds, for each ground point, the transmittance along a flat beam toward the clouds' light
	// (Clouds.ts `shadowAlongBeam`, the channel the air reads, narrower than the ground's from about
	// 5 degrees up), so the point is taken to its ground point along that same flat
	// beam and that light: a beam followed over the planet's curve instead, or along the haze's key
	// direction, meets the map a kilometre off at a low sun and leaves it below a sun of 0.08, where
	// the map clamps its own slope. The read is sharpened by an unsharp mask against its B-spline
	// blur, since the map is coarse and the shafts would blur into a general dimming without it;
	// lifted toward lit for a point above the shell's base (its height over the planet's curve,
	// a point d metres from the island's centre d^2 / 2R lower), less so the nearer `direction` (the
	// ray's that reaches p) lies to the key light (contrasty shafts against the sun), and floored.
	private cloudShaft( p: V3, direction: V3, sky: Sky ): F {

		const clouds = sky.clouds, s = this.keyDir, C = this.cloudShafts;
		const read = clouds.shadowAlongBeam( p ).toVar();
		const sharp = clamp( read.x.add( C.x.mul( read.x.sub( read.y ) ) ), 0, 1 );
		// the height above the sphere, from the shell's base (0) to its top (1)
		const r = p.xz.sub( clouds.shadowCentre );
		const height = p.y.add( dot( r, r ).div( 2 * EARTH_RADIUS ) );
		const lift = clamp( height.sub( clouds.shellBottom ).div( clouds.shellThickness ), 0, 1 ).mul( float( 1 ).sub( sharp ) );
		const toward = clamp( dot( direction, s ), 0, 1 );
		return max( C.y, sharp.add( lift ).sub( lift.mul( C.z ).mul( toward ) ) );

	}

	// The shadow kernel writing volume `w` (reading the other one's history): one thread per froxel.
	private buildVisibilityKernel( w: number, sky: Sky ): THREE.ComputeNode {

		const { width: W, height: H, depth: D } = this.grid;
		const target = this.visibility[ w ];
		const history = texture3D( this.visibility[ 1 - w ] );
		const before = texture3D( this.visibility[ 1 - w ] );
		const size = vec2( W, H );
		return Fn( () => {

			If( instanceIndex.lessThan( W * H * D ), () => {

				const x = int( instanceIndex.mod( W ) ).toVar(), y = int( instanceIndex.div( W ).mod( H ) ).toVar(), z = int( instanceIndex.div( W * H ) ).toVar();
				const cell = vec3( float( x ), float( y ), float( z ) ).toVar();
				// the jittered point inside the froxel: the frame's offset shifted by the froxel's own
				const n = vec3( gradientNoise( cell.xy.add( cell.z.mul( 5.588238 ) ) ), gradientNoise( cell.yx.add( cell.z.mul( 7.123 ) ).add( 17.3 ) ),
					gradientNoise( vec2( cell.z.mul( 3.17 ).add( cell.x ), cell.y.mul( 1.91 ).add( 41.7 ) ) ) );
				const o = fract( this.sequence.add( n ) ).toVar();
				// N sub-samples about it, the R3 sequence's steps apart (Roberts, "The Unreasonable
				// Effectiveness of Quasirandom Sequences", 2018), averaged: a recomputed froxel takes its
				// cell's mean, so its history converges to that rather than to whichever point it took
				const visible = vec2( 0 ).toVar();
				Loop( SUB_SAMPLES, ( { i } ) => {

					const q = fract( o.add( vec3( 0.8191725, 0.6710436, 0.5497005 ).mul( float( i ) ) ) );
					visible.addAssign( this.visibilityAt( cell, q, size, sky ) );

				} );
				visible.divAssign( SUB_SAMPLES );
				const out = visible.toVar();
				If( this.restIndex.greaterThanEqual( 0 ), () => {

					// at rest: the exact mean of the sequence so far (index 0 starts it over)
					const kept = before.load( ivec3( x, y, z ) ).xy;
					out.assign( kept.mul( this.restIndex ).add( visible ).div( this.restIndex.add( 1 ) ) );

				} ).ElseIf( this.blend.lessThan( 1 ), () => {

					// running: the history at the froxel's centre as the previous frame saw it
					const centre = this.gridPoint( cell.xy.add( 0.5 ).div( size ), this.sliceDepth( cell.z.add( 0.5 ) ) );
					const clip = this.previousViewProjection.mul( vec4( centre, 1 ) ).toVar();
					const ndc = clip.xy.div( clip.w );
					const uvw = vec3( ndc.x.mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( ndc.y.mul( 0.5 ) ), this.sliceCoordinate( clip.w ).div( D ) ).toVar();
					const inside = clip.w.greaterThan( 0 ).and( uvw.x.greaterThanEqual( 0 ) ).and( uvw.x.lessThanEqual( 1 ) )
						.and( uvw.y.greaterThanEqual( 0 ) ).and( uvw.y.lessThanEqual( 1 ) ).and( uvw.z.lessThanEqual( 1 ) );
					If( inside, () => {

						out.assign( mix( history.sample( uvw ).level( float( 0 ) ).xy, visible, this.blend ) );

					} );

				} );
				textureStore( target, ivec3( x, y, z ), vec4( out, 0, 1 ) );

			} );

		} )().compute( W * H * D, [ 64 ] ).setName( `Haze.Shadow${ w }` );

	}

	// The light kernel reading shadow volume `r`: one thread per column, front to back, each channel
	// through its own optical depth.
	// The light the air scatters toward the eye along `direction` (the light kernel's, per slice:
	// see the header): a function of a point's height (metres) and of the sun's and the open sky's
	// visibility there, its terms per body built once for the direction.
	// `deck` is the sky's irradiance over pi without its haze (Sky.ts `deckIrradianceNode`), which
	// lights the air under a deck.
	// `keep` is the share of a deck's light (all but overcast's by day) that keeps the deck's form
	// along this column (`hazeFormKeep`, see the header).
	// `seen`: the sky's light as the view sees it (the light kernel's: the closure's `far`, the
	// froxel's share of the seen gain by its distance, blends `moonColourSeen` and `deckShareSeen`
	// in, SEEN_AIR), else as it lights (the bake's); the closure takes `far` exactly when `seen`.
	private scatterToward( direction: V3, deck: V3, clouds: Clouds, keep: F ): ( y: F, sun: F, ambient: F ) => V3;
	private scatterToward( direction: V3, deck: V3, clouds: Clouds, keep: F, seen: true ): ( y: F, sun: F, ambient: F, far: F ) => V3;
	private scatterToward( direction: V3, deck: V3, clouds: Clouds, keep: F, seen = false ): ( y: F, sun: F, ambient: F, far?: F ) => V3 {

		const [ albedo, g, gain ] = [ this.lightSettings.x, this.lightSettings.y, this.lightSettings.z ];
		// (overcast's deck: its light's form through the air's phase, see the header) the forward
		// part, the deck's own Henyey-Greenstein lobe about the key over its mean over the sphere;
		// the diffuse part, DECK_DIFFUSE at the air's anisotropy. Their shares follow the deck's top
		// (Clouds.ts `columnTop`, worked out once a frame there): the lobe's light over an even
		// deck's of the deck's irradiance and the diffuse part's share of that irradiance, so the
		// lobe is the key's alone and leaves with it as the key sets.
		const nu = dot( direction, clouds.lightDir ).toVar();
		const lobeOf = ( k: F ) => float( 1 ).sub( k.mul( k ) ).div( max( float( 1 ).add( k.mul( k ) ).sub( nu.mul( k ).mul( 2 ) ), 1e-4 ).pow( 1.5 ) );
		// (under overcast's share the deck's forward light is spread by the diffusion law, Clouds.ts
		// OVERCAST_BROAD: it reaches the air in the diffuse part, and the lobe's share is the rest's,
		// `columnTop`)
		const forward = lobeOf( clouds.lobe );
		const mu = clamp( direction.y, - 1, 1 );
		const inG = ( row: number[] ) => g.mul( row[ 2 ] ).add( row[ 1 ] ).mul( g ).add( row[ 0 ] );
		let diffuse = inG( DECK_DIFFUSE[ 4 ] );
		for ( let i = 3; i >= 0; i -- ) diffuse = diffuse.mul( mu ).add( inG( DECK_DIFFUSE[ i ] ) );
		const top = clouds.columnTop();
		const lobeShare = top.lobeShare.toVar(), diffuseShare = top.diffuseShare.toVar();
		const formOf = ( law: F, share: F ) => mix( float( 1 ), law.mul( diffuseShare ).add( forward.mul( lobeShare ) ), share );
		// (the deck's own light by the whole deck's share: overcast's by day whole, any other deck's,
		// and any deck's at night, by the share of it the column keeps in that form)
		const deckForm = formOf( diffuse, mix( this.deckShare.mul( keep ), this.deckShare, this.overcastShare ) ).toVar();
		// the sky's in-scatter along the column's direction at each froxel's altitude (the fog
		// table, whose rows hold the earth's shadow), through the phases at use: the Rayleigh share
		// through the sky's phase without its polariser, lit by the open sky and by the sun under
		// the earth's shadow over the seen sky, the aerosol share by the sun; each share's colour
		// turned toward its luminance first
		const atmosphere = this.atmosphere;
		const grey = ( colour: V3 ): V3 => mix( colour, vec3( dot( colour, LUMINANCE ) ), this.desaturate );
		// per body (0 the sun, 1 the moon), what the column shares: its direction and the phases
		const bodies = ( [ 0, 1 ] as const ).map( ( b ) => {

			const light = b === 0 ? atmosphere.sunDir : atmosphere.moonDir;
			const nu = dot( direction, light ).toVar();
			// (the earth's shadow over the seen sky, read with the column held SHADOW_ELEVATION or
			// more over the horizon: see the header)
			const level = vec2( direction.x, direction.z ).toVar();
			const across = level.div( max( length( level ), 1e-6 ) ).mul( SHADOW_ELEVATION_COS ).toVar();
			const held = vec3( across.x, SHADOW_ELEVATION_SIN, across.y ).toVar();
			const raised = select( direction.y.lessThan( SHADOW_ELEVATION_SIN ), held, direction );
			const shadow = atmosphere.skyShadowOf( raised, light ).toVar();
			// (overcast's open sky: by the view's elevation and its angle to the key, see the header)
			const overcast = b === 0 ? formOf( max( direction.y, 0 ).mul( 6 ).add( 3 ).div( 7 ), this.overcastShare ).toVar() : float( 1 );
			// (under a deck each body's open sky gives way to the deck's light, by its share)
			const open = float( 1 ).sub( this.deckShare ).toVar();
			return {
				b, light, overcast, open, miePhase: cornetteShanks( nu, b === 0 ? g : this.moonAnisotropy ).toVar(),
				rayleighPhase: skyRayleighPhase( nu, atmosphere.rayleighForward ).toVar(), shadow,
				direct: this.sunRayleighColour[ b ],
			};

		} );
		// the deck's light on the air under it: the deck over it and the sea's reflection of the deck
		// under it, scattered evenly (half the sphere each), overcast's deck in its form through the
		// air's phase (see the header), the sea's even (it reaches the air from below, where neither
		// the deck's law nor its lobe points)
		const deckBase = deck.mul( deckForm.add( SEA_DIFFUSE ).mul( 0.5 ) );
		const deckLight = seen ? deckBase.toVar() : deckBase.mul( this.deckShare ).toVar();
		// (a + ( b - a ) s: exactly a where the seen value equals the physical one, by day)
		// (the colour uniforms as vec3 nodes: three's types keep 'color' apart from 'vec3', the shader
		// does not; a vec3() around them would add a node to the emitted code)
		const asV3 = ( colour: THREE.UniformNode<'color', THREE.Color> ) => colour as unknown as V3;
		const sunSky = asV3( this.ambientColour[ 0 ] ), moonSky = asV3( this.ambientColour[ 1 ] ), moonSeen = asV3( this.moonColourSeen );
		return ( y, sun, ambient, far ) => {

			const ambientColour: V3[] = seen && far ? [ sunSky, moonSky.add( moonSeen.sub( moonSky ).mul( far ) ) ] : [ sunSky, moonSky ];
			const deckAir = seen && far ? deckLight.mul( this.deckShare.add( this.deckShareSeen.sub( this.deckShare ).mul( far ) ) ) : deckLight;

			// (the froxel's altitude, km)
			const height = max( y, 0 ).div( 1000 ).toVar();
			const scattered = vec3( 0 ).toVar();
			// (the open sky's scatter colour: the sun's Rayleigh rows at their colour of light, before the
			// key's grey)
			let skyColour: V3 = vec3( 1 );
			for ( const body of bodies ) {

				// (the sun's rows along the view, level under a deck, the moon's along the moon: see the header)
				const rows = atmosphere.fogRows( body.b === 0 ? mix( direction.y, float( DECK_ROW ), this.deckShare ) : body.light.y, height, body.b );
				const rayleighRows = rows.rayleigh.mul( body.rayleighPhase ).toVar();
				if ( body.b === 0 ) skyColour = rayleighRows.mul( ambientColour[ 0 ] );
				const rayleigh = grey( rayleighRows )
					.mul( ambientColour[ body.b ].mul( ambient ).mul( body.overcast ).mul( body.open ).add( body.direct.mul( sun ) ) ).mul( body.shadow );
				const mie = grey( rows.mie.mul( body.miePhase ) ).mul( this.mieColour[ body.b ] ).mul( sun );
				scattered.addAssign( rayleigh.mul( this.weights.x ).add( mie.mul( this.weights.y ) ) );

			}
			// overcast's share of the deck's light on the air at the deck light's luminance in the open
			// sky's scatter colour, turned toward grey by DECK_AIR_GREY (see the header)
			const hue = skyColour.div( max( dot( skyColour, LUMINANCE ), 1e-12 ) );
			const tinted = mix( hue, vec3( 1 ), this.deckAirGrey ).mul( dot( deckAir, LUMINANCE ) );
			const deckAirLight = mix( deckAir, tinted, this.overcastShare );
			return scattered.mul( gain.mul( albedo ) ).add( deckAirLight.mul( ambient ).mul( albedo ) );

		};

	}

	// The fog layer's droplets' light toward the eye along `direction` (see the header, The fog
	// layer), a function of a point's height (metres) and of the sun's and the open sky's visibility
	// there: the sun's beam through the droplets' phase with their multiple scattering, over the fog
	// layer's depth above the point toward the sun, and the open sky's light (`deck`, its irradiance
	// over pi) scattered evenly, the sky over the upper half of the sphere and the sea's reflection
	// of it under the lower, as the deck's light on the air is taken (`scatterToward`).
	private dropletToward( direction: V3, deck: V3, banks = true ): ( p: V3, sun: F, ambient: F ) => V3 {

		const sunDir = this.atmosphere.sunDir;
		const nu = dot( direction, sunDir ).toVar();
		const rise = max( sunDir.y, 0.05 ).toVar();
		// (the layer's top: its density there 1/800 of the deep part's)
		const top = this.ground.y.add( this.ground.z.mul( 20 ) ).toVar();
		const sky = deck.mul( ( 1 + SEA_DIFFUSE ) * 0.5 ).toVar();
		// (the banks' own shade toward the sun: the banks' factor less its mean at BANK_SHADE's
		// points along the sun's ray, each over its stretch of the ray; taken where the light kernel
		// takes the banks, under the drift's reach while they are on)
		const lift = sunDir.toVar();
		const reach = this.ground.y.add( this.ground.z.mul( DRIFT_REACH ) ).toVar();
		return ( p, sun, ambient ) => {

			const y = p.y;
			const above = max( top.sub( y ), 1 );
			const shade = float( 0 ).toVar();
			if ( banks ) If( this.fogNoise.y.greaterThan( 0 ).and( y.lessThan( reach ) ), () => {

				for ( const [ at, stretch ] of BANK_SHADE ) {

					const q = p.add( lift.mul( at ) ).toVar();
					const factor = fogDrift( q, float( 0 ), this.fogNoise, this.fogCover, this.fogShape, shared.wind );
					shade.addAssign( meanExtinction( q.y, q.y, GROUND_ONLY, this.ground ).mul( factor.sub( 1 ) ).mul( stretch ) );

				}

			} );
			const tau = max( this.sigma.y.mul( above ).mul( meanExtinction( y, top, GROUND_ONLY, this.ground ) ).div( rise ).add( this.sigma.y.mul( shade ) ), 0 );
			const phase = dropletPhase( nu, tau, this.dropletLobes, this.dropletOctaves );
			return this.mieColour[ 0 ].mul( sun.mul( phase ) ).add( sky.mul( ambient ) ).mul( this.lightSettings.x );

		};

	}

	private buildLightKernel( r: number, deck: V3, clouds: Clouds ): THREE.ComputeNode {

		const { width: W, height: H, depth: D } = this.grid;
		const shade = texture3D( this.visibility[ r ] );
		const target = this.light;
		return Fn( () => {

			If( instanceIndex.lessThan( W * H ), () => {

				const x = int( instanceIndex.mod( W ) ).toVar(), y = int( instanceIndex.div( W ) ).toVar();
				const { view, direction } = this.gridRay( vec2( float( x ).add( 0.5 ), float( y ).add( 0.5 ) ).div( vec2( W, H ) ) );
				const forward = view.z.negate().toVar();
				// (the column's whole depth to the grid's far end, for the share of the deck's form it keeps)
				const tEnd = this.sliceDepth( float( D ) ).div( forward ).toVar();
				const keep = hazeFormKeep( this.sigma.y.mul( tEnd.mul( meanExtinction( this.origin.y, this.origin.y.add( direction.y.mul( tEnd ) ), this.layer, this.ground ) ) ), this.lightSettings.y );
				const scatter = this.scatterToward( direction, deck, clouds, keep, true );
				const droplet = this.dropletToward( direction, deck );
				const reach = this.ground.y.add( this.ground.z.mul( DRIFT_REACH ) ).toVar();
				// (alpha: the band opacity in front; shown: the display opacity, through the band-to-display
				// matrix, which keeps 1 and so acts on an opacity as on a transmittance; the medium's depth
				// in front, its mean and drifted, for the composite's drift)
				const sum = vec3( 0 ).toVar(), alpha = vec3( 0 ).toVar(), shown = vec3( 0 ).toVar();
				const meanDepth = float( 0 ).toVar(), driftDepth = float( 0 ).toVar();
				const t0 = this.slices.x.div( forward ).toVar();
				const y0 = this.origin.y.add( direction.y.mul( t0 ) ).toVar();
				Loop( D, ( { i } ) => {

					const t1 = this.sliceDepth( float( i ).add( 1 ) ).div( forward ).toVar();
					const y1 = this.origin.y.add( direction.y.mul( t1 ) ).toVar();
					const span = t1.sub( t0 ).toVar();
					// the medium's mean density over the slice, and the fog layer's within it, which the
					// drift's factor (its mean over the view 1) scales
					const mean = meanExtinction( y0, y1, this.layer, this.ground ).toVar();
					const fog = meanExtinction( y0, y1, GROUND_ONLY, this.ground ).toVar();
					const factor = float( 1 ).toVar();
					const middle = this.origin.add( direction.mul( t0.add( t1 ).mul( 0.5 ) ) ).toVar();
					If( this.fogNoise.y.greaterThan( 0 ).and( min( y0, y1 ).lessThan( reach ) ), () => {

						factor.assign( fogDrift( middle, span, this.fogNoise, this.fogCover, this.fogShape, shared.wind ) );

					} );
					const density = max( mean.add( fog.mul( factor.sub( 1 ) ) ), 0 ).toVar();
					meanDepth.addAssign( span.mul( mean ) );
					driftDepth.addAssign( span.mul( density ) );
					const tau = this.sigma.mul( span.mul( density ) ).toVar();
					// the slice's share of the light that reaches the eye, per channel: what it scatters
					// times what lies in front of it
					const w = vec3( 1 ).sub( alpha ).mul( tau.mul( expShare3( tau ) ) ).toVar();
					const fraction = shade.load( ivec3( x, y, i ) ).toVar();
					const sun = fraction.r.mul( LIGHT_DEBUG.sun ).toVar();
					// the air's light and the fog layer's droplets', by the fog layer's share of the slice's
					// density times the droplets' share of its light
					const height = y0.add( y1 ).mul( 0.5 ).toVar();
					// (clamped: high over the layers both means are rounding noise, under a deck's top the
					// aerosol layer's none)
					const share = this.dropletShare.mul( clamp( fog.mul( factor ).div( max( density, 1e-30 ) ), 0, 1 ) );
					// (the seen gain's share by the slice's distance, SEEN_AIR)
					const far = smoothstep( SEEN_AIR.near, SEEN_AIR.far, t0.add( t1 ).mul( 0.5 ) );
					const scattered = mix( scatter( height, sun, fraction.g, far ), droplet( middle, sun, fraction.g ), share ).toVar();
					alpha.addAssign( w );
					const next = bandToDisplay( alpha, bandPush ).toVar();
					sum.addAssign( scattered.mul( next.sub( shown ) ) );
					shown.assign( next );
					// L / alpha: the mean light of the haze in front, its limit where there is none yet
					const light = select( shown.y.greaterThan( 1e-20 ), sum.div( max( shown, vec3( 1e-20 ) ) ), scattered );
					// (the debug view: the visibility as a grey card in the key light, so the display's
					// exposure shows 1 as mid grey)
					const out = select( this.switches.w.equal( 1 ), vec3( fraction.r.mul( dot( this.keyColor.rgb, vec3( 0.2126, 0.7152, 0.0722 ) ) ).mul( 0.18 / Math.PI ) ), light );
					// (a: the drift's share of the medium's depth in front, its drifted over its mean)
					const ratio = select( meanDepth.greaterThan( 1e-30 ), driftDepth.div( max( meanDepth, 1e-30 ) ), float( 1 ) );
					textureStore( target, ivec3( x, y, i ), vec4( preExposed( out.mul( storePress.store ) ), ratio ) );
					t0.assign( t1 );
					y0.assign( y1 );

				} );

			} );

		} )().compute( W * H, [ 64 ] ).setName( `Haze.Light${ r }` );

	}

	// The light volume at uvw, through a tricubic B-spline (eight trilinear taps) or trilinear.
	private sampleLight( uvw: V3 ): V4 {

		const { width: W, height: H, depth: D } = this.grid;
		const volume = texture3D( this.light );
		const size = vec3( W, H, D );
		const trilinear = volume.sample( uvw ).level( float( 0 ) );
		const coord = uvw.mul( size ).sub( 0.5 ).toVar();
		const index = floor( coord ).toVar();
		const f = coord.sub( index ).toVar();
		const f2 = f.mul( f ), f3 = f2.mul( f );
		const w0 = f3.negate().add( f2.mul( 3 ) ).sub( f.mul( 3 ) ).add( 1 ).div( 6 );
		const w1 = f3.mul( 3 ).sub( f2.mul( 6 ) ).add( 4 ).div( 6 );
		const w2 = f3.mul( - 3 ).add( f2.mul( 3 ) ).add( f.mul( 3 ) ).add( 1 ).div( 6 );
		const w3 = f3.div( 6 );
		const g0 = w0.add( w1 ).toVar(), g1 = w2.add( w3 ).toVar();
		const h0 = index.sub( 0.5 ).add( w1.div( g0 ) ).div( size ).toVar();
		const h1 = index.add( 1.5 ).add( w3.div( g1 ) ).div( size ).toVar();
		const tap = ( a: V3, b: V3, c: V3 ) => volume.sample( vec3( a.x, b.y, c.z ) ).level( float( 0 ) );
		const z0 = mix( mix( tap( h1, h1, h0 ), tap( h0, h1, h0 ), g0.x ), mix( tap( h1, h0, h0 ), tap( h0, h0, h0 ), g0.x ), g0.y );
		const z1 = mix( mix( tap( h1, h1, h1 ), tap( h0, h1, h1 ), g0.x ), mix( tap( h1, h0, h1 ), tap( h0, h0, h1 ), g0.x ), g0.y );
		const tricubic = mix( z1, z0, g0.z );
		return select( this.switches.y.greaterThan( 0.5 ), tricubic, trilinear );

	}

	// `colour` seen through the haze over `reach` metres along the grid's ray through screen uv q,
	// `forward` its view depth per metre: the light volume sampled at reach (at most the grid's far
	// end), the opacity exact from the medium's optical depth `depth` (before each band's sigma; or
	// the grid's own with the density anti-aliasing off), each channel through its sigma and the
	// band-to-display matrix. `ground`, given, is the share of the light laid over `colour` (1 keeps
	// it whole), built only inside a branch the key's `groundLight` under 1 takes, a uniform branch
	// (see `apply`): at 1 the result is the plain mix's.
	private fogOver( colour: V3, q: V2, forward: F, reach: F, depth: F, drifted: boolean, ground?: () => F ): { result: V3; opacity: V3 } {

		const { depth: D } = this.grid;
		const at = min( reach.mul( forward ), this.far ).toVar();
		const uvw = vec3( q, this.sliceCoordinate( at ).sub( 0.5 ).div( D ) );
		const light = this.sampleLight( uvw ).toVar();
		// (the volume holds the light pre-exposed, under the debug press PreExposure.ts `storePress`:
		// see the header)
		const lit = sceneUnits( light.rgb ).mul( storePress.read ).toVar();
		// (a drifted reader's depth takes the fog layer's drift, the volume's alpha: see the header,
		// The fog layer)
		const along = drifted ? depth.mul( light.a ).toVar() : depth;
		const opacityOf = ( d: F ) => {

			const tau = this.sigma.mul( d ).toVar();
			return bandToDisplay( tau.mul( expShare3( tau ) ), bandPush );

		};
		const exact = opacityOf( along ).toVar();
		// (without the density anti-aliasing, the depth out to the far boundary of the slice the
		// surface lies in, as the grid's slices hold the medium)
		const held = opacityOf( along.mul( this.sliceDepth( ceil( this.sliceCoordinate( at ) ) ).div( max( at, 1e-6 ) ) ) ).toVar();
		const opacity = select( this.switches.z.greaterThan( 0.5 ), exact, held ).toVar();
		// (the debug views: the visibility card alone, or the haze's light over black)
		const under = select( this.switches.w.equal( 3 ), vec3( 0 ), colour );
		let over = mix( under, lit, opacity );
		if ( ground ) {

			// (the light alone scaled, the transmittance kept: c ( 1 - a ) + lit a ground)
			const scaled = over.toVar();
			If( this.groundLight.lessThan( 1 ), () => {

				scaled.assign( under.mul( vec3( 1 ).sub( opacity ) ).add( lit.mul( opacity ).mul( ground() ) ) );

			} );
			over = scaled;

		}
		const result = select( this.switches.w.equal( 1 ), lit, over );
		return { result, opacity };

	}

	/** The haze toward the world direction `d`, for the sky dome (Sky.ts `fogDome`), out to
	 * `distance` (a cloud's) or by default the sky's own (to the atmosphere's top, or to the sea for a
	 * ray below the horizon): the light it adds (the light volume sampled there or at the grid's far
	 * end, whichever is nearer, times the opacity) and its transmittance, the opacity taken along the
	 * whole distance; none with `sky` off. The ray's screen position is the grid's own (unjittered)
	 * projection of `d`, so the dome reads the volume where the grid holds it whether or not it draws
	 * jittered. `from` stands the ray's start at another height (metres; the camera's by default),
	 * which changes the opacity only: the light is the volume's, read on the camera's grid. `dome`
	 * (the dome's own call, Pipeline.ts) adds, on the sky's own ray only (no `distance`, no `from`),
	 * the dome's thin haze (the key's `skyDepth`, see the header) to the medium's depth; every other
	 * caller (the clouds' haze, `probe`) reads the medium alone. */
	skyHaze( d: V3, distance?: F, from?: F, dome = false ): { light: V3; transmittance: V3 } {

		const view = this.cameraInverse.mul( vec4( d, 0 ) ).xyz.toVar();
		const clip = this.gridProjection.mul( vec4( view, 1 ) ).toVar();
		const q = vec2( clip.x.div( clip.w ).mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( clip.y.div( clip.w ).mul( 0.5 ) ) );
		const forward = max( view.z.negate(), 1e-4 ).toVar();
		// the distance to the atmosphere's top on the sphere (a difference of squares, which keeps
		// its digits at the planet's radius), or to the sea, flat, as the haze's surfaces take it
		const y = () => from ?? this.origin.y;
		const r = y().add( ATMOSPHERE_BOTTOM ).toVar();
		const top = r.mul( d.y ).negate().add( sqrt( r.mul( r ).mul( d.y.mul( d.y ) ).add( float( ATMOSPHERE_TOP ).sub( r ).mul( r.add( ATMOSPHERE_TOP ) ) ) ) );
		const down = d.y.lessThan( 0 ).toVar();
		const whole = select( down, y().div( max( d.y.negate(), 1e-6 ) ), top ).toVar();
		const length = ( distance ? min( distance, whole ) : whole ).toVar();
		// the optical depth along a straight ray, as the haze's surfaces take it: the length times the
		// medium's mean density over the heights it spans (`meanExtinction`), the height climbing with
		// d.y alone, not lifted by the planet's curve. So the haze closes over the sky near the horizon:
		// at 1 degree the ray holds the ground layer and the aerosols' lower air for tens of kilometres.
		// (Two parts, the far one from the height the curve lifts the ray to, took about half a degree's
		// worth of air too little there: 1 - T in green 0.873 against 0.913 straight, a bent ray
		// integrated in steps 0.889, in the same medium; above 3 degrees the three differ by 0.02 or less.)
		const medium = length.mul( meanExtinction( y(), y().add( d.y.mul( length ) ), this.layer, this.ground ) );
		// the dome's own thin haze, uniform in height over its whole length (below the horizon too,
		// to the sea), in the medium's units: green's transmittance gains exp( - skyDepth L_km ), the
		// other bands follow through sigma (0 by default adds exactly nothing)
		const depth = dome && ! distance && ! from ? medium.add( length.mul( this.skyDepth ) ) : medium;
		const fog = this.fogOver( vec3( 0 ), q, forward, length, depth, false );
		const on = this.switches.x.greaterThan( 0.5 );
		return { light: select( on, fog.result, vec3( 0 ) ), transmittance: select( on, vec3( 1 ).sub( fog.opacity ), vec3( 1 ) ) };

	}

	/** The haze toward the world direction `d` from `height` metres over the sea, out to `distance`
	 * (a cloud's) or by default the sky's whole way (as `skyHaze` takes it), shadowed by the clouds alone: the light it adds and its
	 * transmittance, for the environment's bake (Sky.ts `fogEnvironment`), which takes every
	 * direction, so it marches the light kernel's slices along `d` itself instead of reading the
	 * camera's grid (the same light, the sun and the open sky seen whole at every slice). The light
	 * is the mean over the slices out to the sky's distance or the grid's far end, whichever is
	 * nearer, times the opacity taken along the whole distance; none with `sky` off. Without
	 * `fogLayer` the ground layer is the air's alone (its density without the fog layer's, in the
	 * layer's shape) and the droplets add no light: the land's sky light (Sky.ts `fogEnvironment`),
	 * whose key takes no loss through the fog layer either. With no fog layer the two are the same
	 * light. */
	openSky( d: V3, height: number, clouds: Sky, distance?: F, fogLayer = true ): { light: V3; transmittance: V3 } {

		const { depth: D } = this.grid;
		const y = float( height );
		const ground = fogLayer ? this.ground : this.groundClear;
		const r = y.add( ATMOSPHERE_BOTTOM ).toVar();
		const top = r.mul( d.y ).negate().add( sqrt( r.mul( r ).mul( d.y.mul( d.y ) ).add( float( ATMOSPHERE_TOP ).sub( r ).mul( r.add( ATMOSPHERE_TOP ) ) ) ) );
		const sky = select( d.y.lessThan( 0 ), y.div( max( d.y.negate(), 1e-6 ) ), top );
		const whole = ( distance ? min( distance, sky ) : sky ).toVar();
		const reach = min( whole, this.far ).toVar();
		const keep = hazeFormKeep( this.sigma.y.mul( reach.mul( meanExtinction( y, y.add( d.y.mul( reach ) ), this.layer, ground ) ) ), this.lightSettings.y );
		const air = this.scatterToward( d, clouds.bakeDeckIrradianceNode, clouds.clouds, keep );
		const droplet = fogLayer ? this.dropletToward( d, clouds.bakeDeckIrradianceNode, false ) : null;
		// (the air's light and the fog layer's droplets', by the fog layer's share of the mean density
		// over the heights from a to b, as the light kernel takes them without the drift; the air's
		// alone without the fog layer)
		const scatter = ( a: F, b: F, sun: F ): V3 => {

			if ( ! droplet ) return air( a.add( b ).mul( 0.5 ).toVar(), sun, float( 1 ) );
			const share = this.dropletShare.mul( clamp( meanExtinction( a, b, GROUND_ONLY, ground ).div( max( meanExtinction( a, b, this.layer, ground ), 1e-30 ) ), 0, 1 ) );
			const height = a.add( b ).mul( 0.5 ).toVar();
			return mix( air( height, sun, float( 1 ) ), droplet( vec3( 0, height, 0 ), sun, float( 1 ) ), share );

		};
		// the light kernel's march along d (its view depth per metre 1), each slice cut at reach
		const sum = vec3( 0 ).toVar(), alpha = vec3( 0 ).toVar(), shown = vec3( 0 ).toVar();
		const t0 = min( this.slices.x, reach ).toVar();
		const y0 = y.add( d.y.mul( t0 ) ).toVar();
		// the sun's visibility at a point t metres along d: the clouds' shadow, as the shadow kernel
		// takes it (no caster of the island's: its maps reach no farther than the island)
		// (along d from its own start, never the camera's: the bake holds no camera; the floor
		// under a closed deck, `keyFloor`, in the seen panorama only, as the dome's air takes it)
		const floor = this.keyFloor.mul( clouds.bakeKeyFloor ).toVar();
		const shaft = ( t: F ): F => max( this.cloudShaft( vec3( 0, y, 0 ).add( d.mul( t ) ), d, clouds ), floor );
		const first = scatter( y, y0, shaft( t0.mul( 0.5 ) ) ).toVar();
		Loop( D, ( { i } ) => {

			const t1 = min( this.sliceDepth( float( i ).add( 1 ) ), reach ).toVar();
			const y1 = y.add( d.y.mul( t1 ) ).toVar();
			const tau = this.sigma.mul( t1.sub( t0 ).mul( meanExtinction( y0, y1, this.layer, ground ) ) ).toVar();
			const w = vec3( 1 ).sub( alpha ).mul( tau.mul( expShare3( tau ) ) ).toVar();
			alpha.addAssign( w );
			const next = bandToDisplay( alpha, bandPush ).toVar();
			sum.addAssign( scatter( y0, y1, shaft( t0.add( t1 ).mul( 0.5 ) ) ).mul( next.sub( shown ) ) );
			shown.assign( next );
			t0.assign( t1 );
			y0.assign( y1 );

		} );
		const mean = select( shown.y.greaterThan( 1e-20 ), sum.div( max( shown, vec3( 1e-20 ) ) ), first );
		// the opacity along the whole distance, as `skyHaze` takes it
		const depth = whole.mul( meanExtinction( y, y.add( d.y.mul( whole ) ), this.layer, ground ) );
		const tau = this.sigma.mul( depth ).toVar();
		const opacity = bandToDisplay( tau.mul( expShare3( tau ) ), bandPush ).toVar();
		const on = this.switches.x.greaterThan( 0.5 );
		return { light: select( on, mean.mul( opacity ), vec3( 0 ) ), transmittance: select( on, vec3( 1 ).sub( opacity ), vec3( 1 ) ) };

	}

	/** The haze the frame lays over each of `rays` (world directions from the camera; at most
	 * PROBE_RAYS): over a surface at `distance` metres, as the composite fogs it, or with no
	 * distance over the sky, as the dome fogs itself (`skyHaze`, to the atmosphere's top). Each
	 * reading is the opacity 1 - T per display channel and the light it lays at full opacity (the
	 * haze's added colour over the opacity), evaluated on the GPU by the composite's own nodes
	 * (`fogOver`) on the last frame's camera, medium and light volume, in a kernel built on the
	 * first call (none exists on a page that never calls it). A ray's `height` (metres over the sea)
	 * starts it there instead of at the camera: the opacity is the medium's own along that ray, as
	 * a camera there would see it, while the light stays the camera's volume's. A sky ray reads the
	 * medium alone, without the dome's own thin haze (`skyHaze`'s `dome`, the key's `skyDepth`). A
	 * surface ray's light is the land's unless `sea` is set: the key's `groundLight` scales it as the
	 * composite scales the light over the land (`apply`). For a script that reads cells. */
	async probe( renderer: THREE.WebGPURenderer, rays: readonly { direction: THREE.Vector3Like; distance?: number; height?: number; sea?: boolean }[] ): Promise<{ opacity: number[]; light: number[] }[]> {

		if ( rays.length > PROBE_RAYS ) throw new Error( `VolumetricHaze.probe: at most ${ PROBE_RAYS } rays` );
		if ( ! this.probeKernel ) {

			const input = instancedArray( PROBE_RAYS, 'vec4' ), heights = instancedArray( PROBE_RAYS, 'vec2' ), out = instancedArray( PROBE_RAYS * 2, 'vec4' );
			const kernel = Fn( () => {

				const ray = input.element( instanceIndex ).toVar();
				const d = normalize( ray.xyz ).toVar();
				// (x: the start's height; y: 1 over the sea, 0 over the land)
				const start = heights.element( instanceIndex ).toVar();
				const from = start.x;
				// (the medium alone: not the dome's own haze, so the readings are the medium's opacity)
				const sky = this.skyHaze( d, undefined, from, false );
				// (a surface: the composite's ray, through the grid's own projection of d)
				const view = this.cameraInverse.mul( vec4( d, 0 ) ).xyz.toVar();
				const clip = this.gridProjection.mul( vec4( view, 1 ) ).toVar();
				const q = vec2( clip.x.div( clip.w ).mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( clip.y.div( clip.w ).mul( 0.5 ) ) );
				const distance = max( ray.w, 0 ).toVar();
				const ground = () => float( 1 ).add( this.groundLight.sub( 1 ).mul( float( 1 ).sub( start.y ) ) );
				const surface = this.fogOver( vec3( 0 ), q, view.z.negate(), distance, distance.mul( meanExtinction( from, from.add( d.y.mul( distance ) ), this.layer, this.ground ) ), true, ground );
				const onSky = ray.w.lessThan( 0 ).toVar();
				const opacity = select( onSky, vec3( 1 ).sub( sky.transmittance ), surface.opacity ).toVar();
				const added = select( onSky, sky.light, surface.result ).toVar();
				out.element( instanceIndex.mul( 2 ) ).assign( vec4( opacity, 0 ) );
				out.element( instanceIndex.mul( 2 ).add( 1 ) ).assign( vec4( added, 0 ) );

			} )().compute( PROBE_RAYS, [ 64 ] ).setName( 'Haze.Probe' );
			this.probeKernel = { input, heights, out, kernel };

		}
		const { input, heights, out, kernel } = this.probeKernel;
		const data = ( input.value as THREE.StorageBufferAttribute ).array as Float32Array;
		const from = ( heights.value as THREE.StorageBufferAttribute ).array as Float32Array;
		data.fill( 0 );
		rays.forEach( ( ray, i ) => {

			data.set( [ ray.direction.x, ray.direction.y, ray.direction.z, ray.distance ?? - 1 ], i * 4 );
			from[ i * 2 ] = ray.height ?? this.origin.value.y;
			from[ i * 2 + 1 ] = ray.sea ? 1 : 0;

		} );
		( input.value as THREE.StorageBufferAttribute ).needsUpdate = true;
		( heights.value as THREE.StorageBufferAttribute ).needsUpdate = true;
		await renderer.computeAsync( kernel );
		const buffer = new Float32Array( await renderer.getArrayBufferAsync( out.value as THREE.StorageBufferAttribute ) );
		return rays.map( ( _r, i ) => {

			const opacity = Array.from( buffer.subarray( i * 8, i * 8 + 3 ) ), added = buffer.subarray( i * 8 + 4, i * 8 + 7 );
			return { opacity, light: opacity.map( ( o, c ) => o > 0 ? added[ c ] / o : 0 ) };

		} );

	}

	/** `depth` is the linear view z of the nearest surface, the sea's included, and `opaqueDepth` the
	 * opaque scene's (its r: sample 0's), which tell the land from the sea for the key's
	 * `groundLight`: the haze's light over the land is scaled by it, over the sea and the sky never
	 * (see the header, The light over the land). `veil` returns the
	 * colour `background` seen through the rain along a view direction, in front of content at a
	 * given distance, and the veil's opacity there in its alpha, which the result keeps (0 without
	 * a veil); it is also given the haze's transmittance to that content (`background` is `color`
	 * times it plus the haze's light), and may return that in the alpha instead (the composite
	 * after the resolve, Pipeline.ts, where the veil is laid later along `viewRay`). The volumes are
	 * imported into `graph`; `passes` then adds the kernels' passes. Called once, as the frame graph
	 * is built. */
	apply( color: V3, depth: THREE.Texture, opaqueDepth: THREE.Texture, sky: Sky, graph: GraphResources, veil?: ( direction: V3, distance: F, background: V3, transmittance: F ) => V4 ): V4 {

		const { width: W, height: H } = this.grid;
		this.resolution = graph.extent( { of: 'render' } );
		const fixed: SizeClass = { of: 'fixed', width: W, height: H };
		this.resources = {
			visibility: graph.import( 'haze.visibility', this.visibility[ 0 ], fixed ),
			light: graph.import( 'haze.light', this.light, fixed ),
		};
		this.visibilityKernels = [ this.buildVisibilityKernel( 0, sky ), this.buildVisibilityKernel( 1, sky ) ];
		this.lightKernels = [ this.buildLightKernel( 0, sky.deckIrradianceNode, sky.clouds ), this.buildLightKernel( 1, sky.deckIrradianceNode, sky.clouds ) ];
		const sceneDepth = texture( depth ), opaque = texture( opaqueDepth );
		return Fn( () => {

			const q = uv().toVar();
			const px = ivec2( clamp( q.mul( this.resolution ), 0, this.resolution.sub( 1 ) ) );
			const z = ( textureLoad( sceneDepth, px ) as unknown as F ).toVar();
			const ndc = vec3( q.x.mul( 2 ).sub( 1 ), float( 1 ).sub( q.y ).mul( 2 ).sub( 1 ), 1 );
			const viewFar = this.projectionInverse.mul( vec4( ndc, 1 ) ).toVar();
			const ray = viewFar.xyz.div( viewFar.w ).toVar();
			const view = normalize( ray ).toVar();
			const direction = normalize( this.cameraWorld.mul( vec4( view, 0 ) ).xyz ).toVar();
			const forward = view.z.negate().toVar();
			// (the sky fogs itself, `skyHaze`: its pixels are left as they are)
			const sky_ = z.lessThan( SKY_Z ).toVar();
			// (the veil's distance: the surface's, 1e5 for the sky)
			const distance = select( sky_, float( 1e5 ), z.div( view.z ) ).toVar();
			// The share of the light here while the key's `groundLight` is under 1 (the header, The light
			// over the land): the sea's 1, the land's `groundLight`. The sea lies in front of the opaque
			// surface (debug/RadianceAov.ts `isSea`), unless sample 0 sees the sky and the nearest sample
			// stands over GROUND_SKYLINE plus the pixel's height there (`slack`): the island's skyline.
			const ground = () => {

				const opaqueZ = ( textureLoad( opaque, px ) as unknown as V4 ).r;
				const sea = z.greaterThan( opaqueZ.add( SEA_STEP ) );
				const height = this.origin.y.add( direction.y.mul( distance ) );
				const slack = distance.mul( length( this.projectionInverse.mul( vec4( 0, float( 2 ).div( this.resolution.y ), 0, 0 ) ).xyz.div( viewFar.w ) ).div( length( ray ) ) );
				const skyline = opaqueZ.lessThan( SKY_Z ).and( height.greaterThan( slack.add( GROUND_SKYLINE ) ) );
				const land = select( sea.and( skyline.not() ), float( 0 ), float( 1 ) );
				return float( 1 ).add( this.groundLight.sub( 1 ).mul( land ) );

			};
			const fogged = this.fogOver( color, q, forward, distance, distance.mul( meanExtinction( this.origin.y, this.origin.y.add( direction.y.mul( distance ) ), this.layer, this.ground ) ), true, ground );
			// (the opacity view: the sky 2)
			const result = select( this.switches.w.equal( 2 ), select( sky_, vec3( 2 ), fogged.opacity ), select( sky_, color, fogged.result ) );
			// (the veil's: green's)
			const transmittance = float( 1 ).sub( select( sky_, float( 0 ), fogged.opacity.y ) );
			return veil ? veil( direction, distance, result, transmittance ) : vec4( result, 0 );

		} )();

	}

	/** Adds the haze's two kernels to `graph`, before the composite: the shadow's (skipped while a
	 * converged rest holds it) and the light's. Returns the light volume, which the composite reads. */
	passes<Frame>( graph: FrameGraph<Frame> ): ResourceHandle[] {

		const LABEL = 'volumetric haze';
		let { visibility, light } = this.resources!;
		const { width, height } = this.grid;
		const size: SizeClass = { of: 'fixed', width, height };
		graph.addPass( 'hazeShadow', 'compute', ( b ) => {

			visibility = b.storage( visibility );
			return {
				label: LABEL, size, when: () => this.shadowDue && debugFlags.hazeKernels !== false, whenLabel: 'not held by a rest', compute: this.cover?.late ? undefined : this.visibilityKernels,
				run: ( _f, g ) => {

					this.written = 1 - this.written;
					g.renderer.compute( this.visibilityKernels[ this.written ] );

				},
			};

		} );
		graph.addPass( 'hazeLight', 'compute', ( b ) => {

			b.readOptional( visibility );
			light = b.storage( light );
			return {
				label: LABEL, size, compute: this.lightKernels, run: ( _f, g ) => {

					// (`flags.hazeKernels = false` skips the kernels, for timing them: the composite
					// then reads the volumes as they last were)
					if ( debugFlags.hazeKernels === false ) return;
					g.renderer.compute( this.lightKernels[ this.written ] );

				},
			};

		} );
		return [ light ];

	}

	/** Records the settings the environment's bake reads (BAKE_SETTINGS), as it bakes (RenderStages.ts
	 * `bakeEnvironment`). */
	noteBake(): void {

		const S = this.settings, baked = this.baked ??= { ...S };
		for ( const k of BAKE_SETTINGS ) ( baked as Record<string, number | boolean> )[ k ] = S[ k ];

	}

	/** Whether a setting the environment's bake reads has changed since it last baked (false before
	 * the first `noteBake`): RenderStages.ts `retuneSky` rebakes it then, as a fresh load with the settings
	 * would bake. */
	get bakeStale(): boolean {

		const S = this.settings, baked = this.baked;
		if ( ! baked ) return false;
		for ( const k of BAKE_SETTINGS ) if ( baked[ k ] !== S[ k ] ) return true;
		return false;

	}

	/** Sets the medium and its light from the look, the amounts and the weather's light (none of
	 * it from the camera): what `openSky` reads, so
	 * the environment's bake calls it before it bakes (RenderStages.ts `prepareHaze`), and `update` every
	 * frame. */
	prepareMedium(): void {

		const S = this.settings;
		// the medium
		// (the haze is the deep part's extinction, 2 rho_f, in the band of sigma 1; the fog's unit is
		// the deep part's green extinction)
		const look = this.look;
		// (the fog layer's extinction turned toward neutral, green's kept: one sigma for both layers,
		// so the aerosol layer's turns with it, see the header, The fog layer)
		const green = Math.max( look.sigma.y, 1e-6 ), neutral = look.neutral;
		this.sigma.value.set( look.sigma.x + ( green - look.sigma.x ) * neutral, look.sigma.y, look.sigma.z + ( green - look.sigma.z ) * neutral );
		const rho = this.haze.value / 2 + this.fog.value * S.fogExtinction / ( 2 * green );
		// (the fog layer's rho_f from the visibility at the sea, Koschmieder's 3.912 / V in green: the
		// ground layer's weight at the sea, 1 + top / sqrt( top^2 + width^2 ), less the aerosol
		// layer's share there, which the weather's haze keeps)
		const atSea = 1 + look.top / Math.max( Math.hypot( look.top, look.width ), 1e-6 );
		const fogRho = Math.max( 3.912 / ( look.visibility * 1000 ) / green - rho * look.aerosolShare, 0 ) / atSea;
		const ground = rho + ( fogRho - rho ) * look.fogLayer;
		this.ground.value.set( ground, look.top, look.width );
		this.groundClear.value.set( rho, look.top, look.width );
		this.layer.value.set( rho * look.aerosolShare, look.aerosolHeight, Math.min( look.aerosolTop, AEROSOL_OPEN ) );
		// the fog layer's light and its drift
		this.dropletShare.value = look.droplet;
		this.dropletLobes.value.set( DROPLET.forward(), DROPLET.backward(), DROPLET.backShare() );
		this.dropletOctaves.value.set( look.scatter, DROPLET.octaveExtinction(), DROPLET.octaveAnisotropy() );
		this.fogNoise.value.set( FOG_NOISE.scale(), FOG_NOISE.amplitude() * look.fogLayer, FOG_NOISE.squash(), FOG_NOISE.region() );
		this.fogCover.value.set( FOG_NOISE.low(), look.coverHigh, look.top + look.width, FOG_NOISE.regionShare() );
		this.fogShape.value.set( FOG_NOISE.rough(), FOG_NOISE.warp(), 0, 0 );
		this.skyDepth.value = look.skyDepth / 1000 / Math.max( look.sigma.y, 1e-6 );
		this.groundLight.value = look.groundLight;
		const copy = this.cloudMedium;
		if ( copy ) {

			copy.layer.value.copy( this.layer.value );
			copy.ground.value.copy( this.ground.value );
			// the band that closes last; none without the haze over the sky, which leaves the clouds
			// unfogged (nor with debugFlags.cloudHazeEnd = false, the whole march's A/B)
			const sigma = this.sigma.value;
			copy.sigma.value = S.sky && debugFlags.cloudHazeEnd !== false ? Math.min( sigma.x, sigma.y, sigma.z ) : 0;

		}
		this.desaturate.value = look.desaturate;
		this.lightSettings.value.set( S.albedo, S.anisotropy, S.gain * look.gain );
		// the light's colours per body: its illuminance the sky is drawn with (Atmosphere
		// `sunIlluminance`, `moonIlluminance`: the moon key), the open sky's by the weather's diffuse
		// gain, the body's by its direct share (each body's before the weather's day sky, `sunBeam`,
		// `moonBeam`), and
		// its Rayleigh light the key's share of the aerosol's; the moon's aerosol light then takes its
		// own colour at the same luminance, times its gain
		const weather = this.weatherLight;
		// (a debug page may weigh the sun's, sky/DayCycle.ts `duskForce`)
		const sunSky = duskForce( 'hazeSky', 1 ), sunLight = duskForce( 'hazeSun', 1 );
		[ this.atmosphere.sunIlluminance.value, this.atmosphere.moonIlluminance.value ].forEach( ( illuminance, b ) => {

			this.ambientColour[ b ].value.copy( illuminance ).multiplyScalar( b === 0 ? weather.ambient * sunSky : weather.ambient );
			if ( b === 1 ) this.moonColourSeen.value.copy( this.ambientColour[ 1 ].value ).multiplyScalar( shared.seenNight.value );
			this.mieColour[ b ].value.copy( b === 0 ? this.sunBeam : this.moonBeam ).multiplyScalar( b === 0 ? weather.direct * sunLight : weather.direct );
			this.sunRayleighColour[ b ].value.copy( this.mieColour[ b ].value ).multiplyScalar( look.sunRayleigh );

		} );
		const moonMie = this.mieColour[ 1 ].value;
		const red = MOON_AEROSOL.red(), blue = MOON_AEROSOL.blue();
		const level = luminance( moonMie ) * MOON_AEROSOL.gain() / ( 0.2126 * red + 0.7152 + 0.0722 * blue );
		moonMie.setRGB( red * level, level, blue * level );
		this.weights.value.set( look.rayleighWeight, look.mieWeight );
		this.ambientSettings.value.set( S.ambientFloor, S.ambientSoftness );
		this.deckShareSeen.value = this.deckShare.value * shared.seenNight.value;
		this.switches.value.set( S.sky ? 1 : 0, S.tricubic ? 1 : 0, S.densityAA ? 1 : 0, S.debug );
		// the slices
		const { depth } = this.grid;
		const g = S.growth, first = ( S.far - S.near ) * ( g - 1 ) / ( Math.pow( g, depth ) - 1 );
		this.slices.value.set( S.near, first, g, Math.log2( g ) );
		this.far.value = S.far;

	}

	/** Once per frame, before the graph runs: the camera (its view, and its projection without the
	 * temporal modes' jitter), the render's and the output's sizes, and the app's frame. Sets the
	 * medium and the light from the weather and the settings, and the shadow's filter for the
	 * frame: at rest, running, or held. */
	update( camera: THREE.Camera, projection: THREE.Matrix4, width: number, height: number, outputWidth: number, outputHeight: number, frame: HazeFrame ): void {

		const S = this.settings;
		this.projectionInverse.value.copy( ( camera as THREE.PerspectiveCamera ).projectionMatrixInverse );
		this.gridInverse.value.copy( projection ).invert();
		this.gridProjection.value.copy( projection );
		this.cameraInverse.value.copy( camera.matrixWorldInverse );
		this.cameraWorld.value.copy( camera.matrixWorld );
		this.origin.value.setFromMatrixPosition( camera.matrixWorld );
		this.prepareMedium();

		// the shadow's filter
		this.viewProjection.multiplyMatrices( projection, camera.matrixWorldInverse );
		// (the temporal rest's key, restKey, with the epoch's scene count: see restEpoch)
		const key = this.key;
		key.length = 0;
		key.push( ...camera.matrixWorld.elements, ...projection.elements, frame.time, width, height, outputWidth, outputHeight, restEpoch.scene,
			...frame.signature, ...this.keyDir.value.toArray(), this.keyColor.value.r, this.keyColor.value.g, this.keyColor.value.b, this.keyFloor.value,
			...this.ground.value.toArray(), ...this.layer.value.toArray(), ...this.sigma.value.toArray(),
			this.dropletShare.value, ...this.dropletLobes.value.toArray(), ...this.dropletOctaves.value.toArray(), ...this.fogNoise.value.toArray(), ...this.fogCover.value.toArray(), ...this.fogShape.value.toArray(),
			LIGHT_DEBUG.sun.value, LIGHT_DEBUG.noShadow.value,
			S.ambientFloor, S.ambientSoftness, this.cover?.version ?? 0,
			S.near, S.far, S.growth, S.restFrames, S.jitter ? 1 : 0 );
		this.rest.sequence = S.restFrames;
		this.rest.observe( key, frame.stable );
		camera.matrixWorld.decompose( this.position, this.quaternion, this.scale );
		const last = this.last;
		const cut = ! last.valid || last.keyDir.angleTo( this.keyDir.value ) > CUT || ! last.slices.equals( this.slices.value )
			|| last.position.distanceTo( this.position ) > JUMP.distance || last.quaternion.angleTo( this.quaternion ) > JUMP.angle;
		if ( cut ) this.historyValid = false;
		const index = this.rest.index;
		this.shadowDue = ! this.rest.converged || index !== null;
		const jitter = ( i: number ) => S.jitter ? this.sequence.value.set( halton( i, 2 ), halton( i, 3 ), halton( i, 5 ) ) : this.sequence.value.setScalar( 0.5 );
		if ( index !== null ) {

			this.restIndex.value = index;
			jitter( index + 1 );

		} else {

			this.restIndex.value = - 1;
			this.frame = ( this.frame + 1 ) % Math.max( S.restFrames, 1 );
			jitter( this.frame + 1 );

		}
		this.blend.value = this.historyValid ? S.blend : 1;
		this.previousViewProjection.value.copy( last.viewProjection );
		// what the next frame reprojects through
		last.valid = true;
		last.keyDir.copy( this.keyDir.value );
		last.slices.copy( this.slices.value );
		last.position.copy( this.position );
		last.quaternion.copy( this.quaternion );
		last.viewProjection.copy( this.viewProjection );
		if ( this.shadowDue ) this.historyValid = true;

	}

	/** The inverse projection the composite's rays are reconstructed with (`update` takes the
	 * camera's): under the temporal modes the jittered one the depth they read was drawn with, set
	 * by Pipeline's jitter hook, so each pixel's ray meets its own depth sample. */
	setProjectionInverse( inverse: THREE.Matrix4 ): void {

		this.projectionInverse.value.copy( inverse );

	}

	/** The shadow's rest has not converged (main.ts holds the settled signal for it). */
	get pending(): boolean {

		return ! this.rest.converged;

	}

	/** Takes the hour's and the weather's key (HazeKeys.ts): each band's sigma, the desaturation, the
	 * gain, the sun's share in the Rayleigh light, the shares' weights and the medium's shape. The amounts themselves go
	 * in through `haze` and `fog`. */
	setLook( key: Readonly<HazeKey> ): void {

		const look = this.look;
		look.sigma.set( key.sigmaRed, key.sigmaGreen, key.sigmaBlue );
		this.deckShare.value = key.deck;
		this.overcastShare.value = key.overcastDeck;
		look.desaturate = key.desaturate;
		look.gain = key.gain;
		look.sunRayleigh = key.sunRayleigh;
		look.rayleighWeight = key.rayleighWeight;
		look.mieWeight = key.mieWeight;
		look.top = key.top;
		look.width = key.width;
		look.aerosolShare = key.aerosolShare;
		look.aerosolHeight = key.aerosolHeight;
		look.aerosolTop = key.aerosolTop;
		look.skyDepth = key.skyDepth;
		look.groundLight = key.groundLight;
		// (a key without a fog layer holds none)
		look.fogLayer = key.fogLayer ?? 0;
		look.droplet = key.droplet ?? 0;
		look.neutral = key.neutral ?? 0;
		look.scatter = key.scatter ?? 0;
		look.visibility = key.visibility ?? 2;
		look.coverHigh = key.fogCoverHigh ?? FOG_NOISE.high();

	}

	/** Takes what hides the haze from the open sky (see HazeCover); before the frame graph is built
	 * (`apply`), whose shadow kernel reads it. Without one the ambient visibility is 1. */
	setCover( cover: HazeCover ): void {

		this.cover = cover;

	}

	/** Drops the shadow's history and its rest (a scene jump: `establishScene`). */
	invalidate(): void {

		this.historyValid = false;
		this.last.valid = false;
		this.rest.restart();

	}

	/** The medium's transmittance from a point at `height` toward `direction`, to the atmosphere's top
	 * (for the sun's glare, a sky pixel, as the dome takes it, `skyHaze`: 1 with `sky` off):
	 * each channel's through its sigma and the band-to-display matrix, in `rgb` if given, and their
	 * luminance returned (which the matrix keeps, short of its clamp). The medium's alone: the dome's
	 * own thin haze (`skyHaze`'s `dome`) is left out, so the glare and the flares it dims (the moon's
	 * among them) do not fade with the dome. */
	transmittance( height: number, direction: THREE.Vector3, rgb = _transmittance ): number {

		if ( ! this.settings.sky ) return 1;
		// (as `skyHaze` takes it: along the whole way to the atmosphere's top, on a straight ray)
		const dy = Math.max( direction.y, 1e-4 ), r = height + ATMOSPHERE_BOTTOM;
		const length = - r * dy + Math.sqrt( r * r * dy * dy + ( ATMOSPHERE_TOP - r ) * ( ATMOSPHERE_TOP + r ) );
		const tau = length * this.meanExtinctionAt( height, height + dy * length );
		const sigma = this.sigma.value;
		bandToDisplayColor( rgb.setRGB( Math.exp( - tau * sigma.x ), Math.exp( - tau * sigma.y ), Math.exp( - tau * sigma.z ) ) );
		return 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;

	}

	/** Green's extinction at sea level (1/m), as the last `update` set it, which the screen-space
	 * rays (LightShafts.ts, off by default) read as how hazy the air is: under a clear or fair sky
	 * about five times what they read when the air near the camera was thinned (more under cloud).
	 * It is the sea's, not the camera's: on a clear day the air at the island camera's 68 m holds
	 * about 1/27 of it. */
	get seaLevelExtinction(): number {

		return this.sigma.value.y * this.meanExtinctionAt( 0, 0 );

	}

	/** Readings of the medium as the last `update` set it, for the probes (tools/kit/probe.mjs): green's
	 * extinction at `height` (1/m) and each band's sigma; a horizontal view's transmittance at that height
	 * over 1 km and 3 km (RGB) and the distance at which its luminance falls to half (m); and, given
	 * two points, the RGB and luminance transmittance between them (an eye and the island's centre).
	 * The transmittances are a straight ray's from the eye. */
	readings( height: number, from?: THREE.Vector3Like, to?: THREE.Vector3Like ): {
		extinction: number; sigma: number[]; km1: number[]; km3: number[]; half: number; between?: { rgb: number[]; Y: number; distance: number };
	} {

		const sigma = this.sigma.value.toArray();
		const density = this.meanExtinctionAt( height, height );
		const display = ( c: number[] ) => bandToDisplayColor( _reading.fromArray( c ) ).toArray();
		const across = ( d: number ) => display( sigma.map( ( t ) => Math.exp( - t * density * d ) ) );
		const luminance = ( c: number[] ) => 0.2126 * c[ 0 ] + 0.7152 * c[ 1 ] + 0.0722 * c[ 2 ];
		// the luminance falls monotonically with distance: halve the bracket
		let lo = 0, hi = 1e6;
		for ( let i = 0; i < 60; i ++ ) {

			const mid = ( lo + hi ) / 2;
			if ( luminance( across( mid ) ) > 0.5 ) lo = mid; else hi = mid;

		}
		const round = ( x: number ) => Number( x.toPrecision( 4 ) );
		const out: ReturnType<VolumetricHaze[ 'readings' ]> = { extinction: round( density * sigma[ 1 ] ), sigma, km1: across( 1000 ).map( round ), km3: across( 3000 ).map( round ), half: round( lo ) };
		if ( from && to ) {

			const distance = Math.hypot( to.x - from.x, to.y - from.y, to.z - from.z );
			const tau = distance * this.meanExtinctionAt( from.y, to.y );
			const rgb = display( sigma.map( ( t ) => Math.exp( - t * tau ) ) );
			out.between = { rgb: rgb.map( round ), Y: round( luminance( rgb ) ), distance: round( distance ) };

		}
		return out;

	}

	// meanExtinction on the CPU (the density, over heights a to b), from the layers `update` set
	private meanExtinctionAt( a0: number, b0: number ): number {

		const a = Math.max( Math.min( a0, b0 ), 0 ), b = Math.max( a0, b0, 0 ), span = b - a;
		const share = ( x: number ) => x < 1e-3 ? 1 - x / 2 + x * x / 6 : ( 1 - Math.exp( - x ) ) / x;
		const layer = this.layer.value, ground = this.ground.value;
		const low = Math.min( a, layer.z ), under = Math.min( b, layer.z ) - low;
		const density = b > layer.z ? layer.x * under / Math.max( span, 1e-30 ) : layer.x;
		const ua = a - ground.y, ub = b - ground.y, w2 = ground.z * ground.z;
		const r = Math.sqrt( ua * ua + w2 ) + Math.sqrt( ub * ub + w2 );
		return density * Math.exp( - low / layer.y ) * share( under / layer.y ) + ground.x * ( 1 - ( ua + ub ) / Math.max( r, 1e-6 ) );

	}

}
