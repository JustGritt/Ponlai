/*!
 * Adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater): its
 * volumetric clouds (`Clouds.js`, the view reconstruction) and its port of the Three.js Sky Pro
 * clouds (`SkyProClouds.js`, the density, lighting and march).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 *
 * The in-cloud lightning term (flashLight), the rain curtains and the rainbow drawn in them are
 * adapted from VolumetricCloudsCS1 by Anthony Ibrahim
 * (https://github.com/anthonytoumaibrahim/VolumetricCloudsCS1): `CloudRaymarch.shader`
 * (the point lights, MarchRain, BowColour) and `CloudMedia.cginc` / `CloudCommon.cginc`
 * (RainSigma, SampleRain).
 * SPDX-FileCopyrightText: Copyright (c) 2026 Anthony Ibrahim
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	cos, exp, float, Fn, If, instancedArray, instanceIndex, int, ivec2, length, Loop, max, min, mix,
	normalize, sin, smoothstep, storage, texture, textureLoad, textureStore, uint, uniform, uv, vec2,
	vec3, vec4,
} from 'three/tsl';
import { bakeSynoptic, loadCloudShape } from '../materials/CloudNoise.ts';
import { bakeStreaks } from './Streaks.ts';
import { RAINBOW_SIZE, RAINBOW_TABLE } from './RainbowTable.data.ts';
import type { Atmosphere } from './Atmosphere.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { tuneUniform } from '../debug/Tune.ts';
import { markSkyRetune } from './SkyRetune.ts';
import { HazeMedium } from '../weather/HazeDepth.ts';
import { preExposure, storePress } from '../post/PreExposure.ts';
import {
	BAKE_GLOW_GAIN, BAKE_GLOW_LOBE, BASE_SCALE, BOUNDS_RES, COLUMN_DIFFUSION, DECK_GLOW,
	DECK_GLOW_BROAD, DECK_GLOW_BROAD_LOBE, DECK_GLOW_FAR, DECK_GLOW_FROM, DECK_GLOW_LOBE,
	DECK_GLOW_TEXTURE, DECK_GLOW_TO, DECK_GRADATION, DECK_MEAN_RAYS, DENSITY, directionToDome,
	ditherOrder, DOME_DEFAULTS, driftOffset, EROSION_SCALE, EVOLVE_PERIOD, FIBRE_BROAD, FIBRE_FINE,
	FLASH_SOURCES, FORWARD_LOBE, FORWARD_WEIGHT, forwardLevel, HORIZON_BAND, HORIZON_RING,
	layerDistance, LIGHT_GROWTH, LOW_DRIFT, LOW_VEER, luminance, MARCH_DEFAULTS, MAX_LIGHT_TAPS,
	ORDER, OVERCAST_BROAD, OVERCAST_WEIGHT, REBUILD_SLOTS, sat, SETTLE_CYCLES, SHADOW_ORDER,
	SHADOW_PHASES, SHADOW_SIZE, SHADOW_SPAN, SHELL_BOTTOM, SHELL_MIN, SHELL_THICKNESS, STRATUS_CAP,
	STRATUS_RELIEF, STREAK_AMBIENT, STREAK_HEIGHT, STREAK_LIT, STREAK_MU_FLOOR, STREAK_PATCHES,
	STREAK_PEAK, SYNOPTIC_SIZE, TILE, VIEW_HEADROOM, WEATHER_COVERAGE, WEATHER_DETAIL, WEATHER_LARGE,
	WEATHER_MASS, WEATHER_RES, weatherFbm, type CloudMarch, type CloudViewScale, type View,
	type ViewTextures,
} from './CloudModel.ts';
import { buildKernels } from './CloudKernels.ts';
import { buildView, sampleView, sampleViewFog } from './CloudView.ts';
import { directShadow, shadow, shadowAlongBeam } from './CloudShadow.ts';

// The haze reads the planet's radius beside the clouds' type.
export { EARTH_RADIUS } from './CloudModel.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// Volumetric clouds, one model for every sky, its parameters set by the weather.
//
// Volumetric clouds, one model for every sky. A kind of sky is a row of the model's parameters
// (look/SkyLook.ts `cloudSkies`: the shell, the cloud type and its spread, the far fade; Weather.ts
// WEATHER_PRESETS: the coverage and the density), and a change of weather moves every parameter
// from one row to the next, so no sky has code of its own and any two blend. The framework is
// tidewater's port of the Three.js Sky Pro clouds ("Partly cloudy"): the density's columns, the
// march, the panorama, the shadow map and the view's accumulation.
//   - density: a cloud shell; a procedural weather map sets where cloud may stand, the weather's
//     coverage how much of it does (the weather plus the coverage less one, a column's surplus),
//     and a 64^3 Perlin-Worley shape noise dilates the tops and erodes the margins, more strongly
//     higher up. The coverage read through a weather map and a cloud type that runs from stratus to
//     cumulus are Schneider's (Schneider, "Real-Time Volumetric Cloudscapes", GPU Pro 7 / SIGGRAPH
//     2015; "Nubis: Authoring Real-Time Volumetric Cloudscapes with the Decima Engine", SIGGRAPH
//     2017). The type (`cloudType`: 0 a stratus, 1 a cumulus) sets a column's form, each term
//     mixed linearly between the two ends (STRATUS, CUMULUS, `formOf`): a cumulus heaps as high
//     as its surplus, its margin crisp and eroded hard at the top, its bases standing apart on the
//     richer weather; a stratus stands as a slab wherever it has any surplus, its margin soft and
//     little eroded, with one base, and thinner where the weather is poor, so a closed layer keeps
//     masses and seams (`topOf`). The weather map's second channel, a field of cells 14 km across,
//     moves the type from place to place by the sky's spread (`typeSpread`, `typeAt`), so one sky
//     holds flatter and taller clouds side by side. The shell is the sky's too (`setShell`):
//     clear's and fair's cumulus from 4 to 9.2 km, cloudy's stratocumulus from 1.5 to 4.5 km,
//     overcast's stratus from 0.6 to 3 km, as low layers stand, so a ray to a low layer crosses far
//     less of the haze below it. A point's height in the noise is its height over the shell's base,
//     so the clouds keep their shapes while a change of weather moves the shell. A fair sky's heaps
//     thin into the air before the horizon, where they would pile into bands (`farFade`); a layer
//     runs to it. The parked weathers (rain, storm, fog) count as overcast in the skies' shares, so
//     they take its row and the haze's end of the march: the rain's curtains hang from the layer's
//     base, the march ends storm lightning past the haze's reach, and a bolt keeps the base it was
//     described under through a change of weather. Every reader of the shell follows its uniforms:
//     the march, the shadow, the rain, the dome's haze over the clouds (Sky.ts), the haze's shafts
//     (VolumetricHaze.ts) and the lightning (Lightning.ts).
//     The march's bounds (a sample's largest top, the weather grid's empty cells) hold for every
//     type the sky has: each term of the form is linear in the type and a top rises with its
//     surplus at every type, so the bounds are taken at the two ends of the sky's types.
//   - lighting: each step's light is weighed by the opacity the step itself adds,
//     T ( 1 - exp( -sigma dt ) ), which keeps the energy whatever the step's length (Hillaire,
//     "Physically Based and Unified Volumetric Rendering in Frostbite", SIGGRAPH 2015). The
//     march sums five factors, each per unit of its light's colour, and colours them once it ends
//     (rain, the cirrus and the view's clamp share the targets as colours, so the factors are
//     coloured in the march, not where they are shown):
//     the key's beam. A light march toward the key light in segments that grow geometrically and
//     span the shell's depth, run however faint the light, so the cloud's own shadow moves without
//     a jump while no direct light reaches it (SkyTrack.ts trackClouds); the sample's own depth is taken
//     over the finest step's length whatever the march's cone, so the view and the coarser
//     panorama light a cloud alike. Multiple scattering is three orders, each with half the
//     optical depth, half the energy and half the asymmetry of the one before (Wrenninge, "Oz: The
//     Great and Volumetric", SIGGRAPH 2013), on two Henyey-Greenstein lobes (forward 0.8, back
//     -0.2, 80 / 20: tidewater's droplet phase's first order). A powder term dims the thin margins
//     seen away from the key and the bases shade toward the shell's floor (tidewater's terms). Its
//     colour is the key's irradiance at the shell (`lightColor`: the sun's with its low-sun beam,
//     SkyTrack.ts trackClouds).
//     The sky's light, in two parts: the sky over the tops and the sky round the horizon, each by
//     what the cloud straight above a sample leaves of it (two reads up the column, never less than
//     a fifth: the light that comes round the cloud), the tops' share growing with the height in
//     the shell and the horizon's largest at its floor. The sky over the tops is the open sky's
//     mean radiance over the upper hemisphere, with the clear day's gain on each body's share
//     (`columnTopOf`, look/SkyLook.ts `clearSky`): a cloud under a layer is lit by the sky above it, not by
//     the sky along the view it is seen against. The horizon's is the mean of the whole ring round
//     the cloud over a band 3 to 24 degrees up (`horizonMean`), not the side toward the sun.
//     The sea's light back up, on the bases: the light on the tops times the sea's albedo.
//     The column's light: the key's light that the column above a sample passes down by diffusion,
//     a conservative slab's transmission in the diffusion approximation, 1 / ( 1 + 0.75 ( 1 - g )
//     tau ) (the Eddington form; the theory of thick layers is van de Hulst's, "Multiple Light
//     Scattering", 1980), tau the column's optical depth from the sample to its top less the
//     erosion's mean depth there (measured from the shape noise at build: `erosionMean`), where the
//     orders above no longer reach. Whole under a layer and a part under a heap, whose light leaves
//     by its sides (the form's `column`), the part rising to the whole as the sky closes and the
//     heaps join (COLUMN_CLOSE). A heap's part is the sun's alone (`columnSunShare`): by the moon
//     only a layer's column is lit. Thick masses read dark and thin seams lighter, near grey, the
//     level set by the cloud's own density and depth. It takes the irradiance on a level top over
//     pi (`columnKey`: by day the sun's above the cloud, by night the island's own moon key, each
//     under the weather's day sky at its own height, so the base and the land under it are lit by
//     one key). Under a closed sky the weather states the level its sky stands at (Weather.ts
//     `daySky`, already on that key), so by overcast's share the layer's own mean transmission is
//     divided out (`columnLevel`, `layerTransmission`): the level is the weather's once, and the
//     slab's law leaves the layer its relief. The dome draws the layer at that radiance, the one
//     that lights the land and the sea.
//     The column's light leaves the base with the diffusion law ( 3 / 7 )( 1 + 2 mu ), mu the
//     view's cosine to the zenith: the escape function's diffusion approximation for a thick
//     conservative layer (van de Hulst 1980; Kokhanovsky, Earth-Science Reviews 64, 2004), the CIE
//     standard overcast sky's law, three times as bright overhead as at the horizon, which gives a
//     level surface the column's irradiance. Under overcast's share by day it is moved toward a
//     uniform base by DECK_GRADATION (`deckFlat`), the level surface's irradiance kept: the CIE
//     general skies grade the overcast sky's zenith over its horizon as steep, moderate or uniform
//     (ISO 15469:2004 / CIE S 011/E:2003).
//     The light on the tops and round the horizon is a function of uniforms and the atmosphere
//     alone, so one thread works it out once a frame (`Clouds.ColumnTop`, a `renderer.compute`
//     ahead of the other kernels in `update`, outside the frame graph as they are) into a buffer
//     the march, the panorama, the dome and the haze read, dispatched under every sky.
//     The march draws no forward lobe of its own under a layer: what a thin part passes toward the
//     key is its beam's light. A closed layer's round glow about the key's place is drawn where the
//     sky is composed (SkyDome.ts `deckGlow`, DECK_GLOW: a narrow lobe for its peak and a broad one
//     for its tail, following the base's own pattern), on the shares `Clouds.ColumnTop` works out,
//     and the lighting's sources take it about the sun as one smooth lobe (BAKE_GLOW_LOBE,
//     BAKE_GLOW_GAIN): an overcast sky somewhat brighter toward the sun is the CIE general skies'
//     overcast types 2 and 4, and how much brighter, drawn and lit, is our choice. The forward
//     lobe's weight and mean cosine (FORWARD_WEIGHT, FORWARD_LOBE, and under overcast's share the
//     spread light's OVERCAST_WEIGHT, OVERCAST_BROAD) remain as the angular form of the light under
//     a low layer for the air below it (VolumetricHaze.ts `scatterToward`), by the shares the same
//     kernel stores, and the glow's colour (`lobeTint`).
//     The night's faint ambient (`nightAmbient`) joins the sky's two parts. The march adds no air
//     of its own, nor does the cirrus: the dome lays the haze over both, the clouds at their
//     distance and the layer at its own (SkyDome.ts domeColour).
//   - the cirrus (`streaksAt`): patches of fibres combed along the drift (Streaks.ts) on a
//     sheet `streakHeight` up (8.5 km), near the top of the clear and fair skies' cumulus shell
//     (4 to 9.2 km), where few cumulus reach, so drawn behind them in the composite (sky, thin
//     layer, cumulus, rain), its optical depth the weather's amount
//     (Weather.ts `streaks`) times `streakPeak` times its patches, fibres and silk, its opacity
//     along the slant path. Lit from the atmosphere at its own height (see `streaksAt`), its direct
//     share the sun's light, colour and level, through the air from each of its points at the
//     sun's elevation there (`streakSunColour`, `streakLitRadius`), so the side away from the sun
//     goes out first and the side toward it reddens last, each
//     share under an overall colour by the sun's elevation (`keyDirect`, `keyAmbient`: Sky.ts
//     `look.streakDirect`, `look.streakAmbient`: under a low sun the direct share warm and the ambient cool,
//     so the layer reads rose on both sides of the sky rather than the yellow-white of the sun's
//     light alone, both turning back to white between 8 and 30 degrees; white under a higher sun
//     and by night), the night's faint ambient (`nightAmbient`, SkyTrack.ts trackClouds) taken with the
//     sky's light under the same gains. The clouds' march takes the
//     ambient as its sky light too.
//   - march: coarse steps until cloud is found, then fine ones; whole cells of the weather map
//     that cannot reach the ray's height are skipped, the march going on past them at the pixel's
//     own offset into a step (resumed at the cell's edge, the steps fell in phase across the
//     pixels and drew rows of bands on a sharp base). Under a whole low deck the view's march ends
//     where the haze lets 1/500 through (`haze`, the haze's medium: weather/HazeDepth.ts
//     `hazeReach`), since the dome fogs that deck at the march's own depth: through a broken
//     deck's gaps toward the horizon the rest of the shell lies tens of kilometres deep in haze.
//     Ended there, the view's march at the cloudy noon horizon took about 4 % less GPU time (most
//     rays close inside the deck first) and the frame moved one level at most
//     (`debugFlags.cloudHazeEnd = false` marches the whole shell). The panorama keeps the whole
//     shell: its clouds are fogged at one distance from the sea, and below the horizon read unfogged.
// The view is ray marched in screen space, one pixel of every 4 x 4 block per frame, accumulated
// with reprojection (camera motion and wind drift), clamped to this frame's neighbourhood and
// upsampled bicubically. Beside the colour the view accumulates the cumulus's opacity-weighted
// depth and, third, the cirrus's opacity as seen through the cumulus in front of it
// (`sampleViewFog`), by the same reprojection and factors but unclamped: the dome fogs a low deck
// at the depth and the layer's share of the opacity at the layer's distance (Sky.ts
// `domeColour`). A trace's own depth is a sixteenth of the pixels at a
// new jitter each frame, which read directly would move the fog over the deck from frame to frame,
// and at rest would print the last trace's 4 x 4 blocks. A panorama of the upper sky serves
// reflections, the environment and the sun's visibility; a shadow map holds the layer's transmittance along the key light, at the
// weather's share of the layer's optical depth (`shadowDensity`, Weather.ts), read with a
// cubic B-spline built from four bilinear taps (`shadow`), with fades toward lit by the distance
// from the island's centre and by the view's angle from a chosen direction (inert until set);
// a low light's map takes rays over the height of its disc (`shadowDisc`), as a sun takes
// seconds to pass a cloud's edge: a single ray's edge is sharp, a grazing ray's depth along tens
// of kilometres of shell closing it within a sliver of the deck's fringe, and the shadow slides
// kilometres per degree; under a light whose whole disc is below the rays' floor
// (SHADOW_SLOPE_FLOOR) one ray is left, held still; a light on the horizon casts none, the map
// fading to lit as the light sets (`shadowShare`), or under a closed deck, on overcast's share, to the
// deck's mean transmittance along the light (`deckShadowShare`): the haze under the deck, lit
// along the sun the deck still hides, lifted the sky's light as the sun set;
// the map's red channel is the whole disc, which the ground reads; its green channel is what the
// air reads, the mean of the two rays nearest the disc's centre, since the whole disc leaves the
// haze no shaft at 6 degrees, and under about 5 degrees the whole disc again (`shadowAirNarrow`,
// SkyTrack.ts SHADOW_AIR_NARROW), since the sky's dome meters the air's light and the two rays alone
// put half a stop per 0.02 degree back into the exposure looking toward the sun; with one ray the
// channels are equal, so above 11 degrees the map holds what one ray cast; its blue channel is the
// whole disc's direct beam alone, without the light the cloud scatters on through (`directShadow`),
// which only the sea's glint reads: that light is diffuse and makes no mirror image of the sun;
// the haze reads the green channel at each point's ground point along the same flat beam, with
// its B-spline blur for sharpening (`shadowAlongBeam`).
// This project keeps its own weather on top: a change of weather moves the whole sky's parameters
// together, in time alone (Weather.ts), rain hangs in curtains under raining cloud, and a thin
// layer of fibres lies high behind the clouds, lit from the atmosphere there (`streaksAt`).
// Both drift downwind with the wind aloft, a multiple of the weather's surface wind veered to its
// right (setDrift), while the cumulus' structure rises through them in place (`evolve`).
// Everything returns vec4( in-scattered radiance, transmittance ).
//
// Rules: the view accumulates over frames, and a scene at rest (paused clock, still camera)
// settles its clouds once and keeps them, so a paused frame is bit-identical. The march
// assumes a camera below the cloud base: a free camera above the low deck sees it from 50 m
// under its base, and the dome fogs it from that same height (`camY`). On high the view's resolution is a share of the display
// (0.45 of its ratio up to 1.5: 1152x648 at 2K output, under half the render's linear size there,
// and 1296x729 at 4K output), not of the render, so the
// dynamic render scale does not soften the clouds again. 0.45 is the knee of a sweep against
// 0.58: it differed from it no more than a finer view did, and 0.40 stepped up; the view's
// march costs little where the island camera looks, much where the sky fills the frame. A new view size (a resized window; on the tiers below high,
// whose view follows the render, a new render scale) swaps the view's textures under the same
// kernels, which index by uniforms: a kernel built per size built its pipelines mid-play.
// Lightning's light inside the
// cloud stays out of the view's accumulated history, the bake and the panorama. The panorama and
// the shadow map refresh in part each frame and whole on a jump and when a paused scene settles.
// The view's traces, their range boxes and its history hold the radiance pre-exposed, times the
// scale its kernels were dispatched at (PreExposure.ts `preExposure.scale`, held at half float's
// largest) and VIEW_HEADROOM, so they keep the scene's range from noon to a moonless night. The
// clouds update ahead of the frame's `beginFrame`, so a frame whose P changes draws the
// clouds at the last one's: each history buffer keeps the scale it was written at
// (`historyScale`), `sampleView` divides by the shown one's, and the next resolve takes the
// reprojected history to its own scale by their ratio (a power of two: the bits of what lay in
// range stay). A view kept at rest holds the scale it settled at: a new P settles it again, so a
// paused frame takes the clouds at its own P, never at the 0 a jump starts P from.
// Every explicit bake (the load's, a scene jump's, a seek's, a retune's: RenderStages.ts `bakeEnvironment`,
// through Sky.reset) restarts the clouds as they stand before a fresh load's first frame (`restart`),
// then sets their drift, evolution and axes to the scene clock, for the bake that follows.
// Where the parts are: the model's constants and shared helpers in CloudModel.ts; the kernels (the
// density, the march `makeMarch`, the panorama, the shadow map, the rain `rainAt`, the cirrus
// `streaksAt`, the deck's mean) in CloudKernels.ts; the view's buffers, kernels and reads
// (`sampleView`, `sampleViewFog`) in CloudView.ts; the shadow's reads in CloudShadow.ts.

export class Clouds {

	// Coverage and density over the island; they drive the lighting decisions for the deck.
	readonly coverage = uniform( 0.28 );
	readonly density = uniform( 1 );
	// the share of the layer's optical depth its shadow map is cast with (the weather's)
	readonly shadowDensity = uniform( 1 );
	// the sine of the angle over which the map's rays spread across the light's disc, and how many
	// rays that takes (SkyTrack.ts SHADOW_DISC): one, the disc's centre, under a light higher than that,
	// and one, held still, under a light whose whole disc is below SHADOW_SLOPE_FLOOR
	readonly shadowDisc = uniform( 0 );
	readonly shadowRays = uniform( 1, 'int' );
	// the share of the two centre rays in the channel the air reads, the rest the whole disc's
	// (SkyTrack.ts SHADOW_AIR_NARROW)
	readonly shadowAirNarrow = uniform( 0 );
	// The weather's cloud type (0 a stratus, 1 a cumulus: STRATUS, CUMULUS) and how far the weather
	// map's second channel moves it from place to place, either way (`typeAt`).
	readonly cloudType = uniform( 1 );
	readonly typeSpread = uniform( 0 );
	// How far the cloud thins with distance, over FAR_FADE along the ray: 1 leaves none past its
	// end, so a fair sky's heaps melt into the air before the horizon; 0 keeps a layer whole to it.
	readonly farFade = uniform( 0 );
	// The weather's rain (0 to 1), which the curtains under the cloud fall by (`rainAt`).
	readonly rain = uniform( 0 );
	// The level of the light a column passes down, over the slab's own transmission: 1 leaves the
	// slab's; under a closed sky whose weather states the level its sky stands at (Weather.ts
	// `daySky`, already on the key the layer takes) it is one over the layer's mean transmission
	// (`layerTransmission`), so the level is the weather's once and the slab's law gives the
	// layer its relief, thick parts darker than thin.
	readonly columnLevel = uniform( 1 );
	// Rain that fell recently (WeatherState.showers): it lingers in showers under the thicker cells.
	readonly showers = uniform( 0 );
	// Horizontal metres a drop drifts downwind per metre it falls.
	readonly rainSlant = uniform( 0.4 );
	// Strength of the rainbows in sunlit rain (0 to 1), set by the sky's rules for when they show.
	readonly rainbow = uniform( 0 );
	// How far the heaps of a broken sky may shower (0 to 1): none in fog, whose cover is a layer of
	// mist and stratus rather than rain cells.
	readonly heapShowers = uniform( 1 );
	// The cirrus (`streaksAt`): the weather's amount of it, its height (m) and its fibres'
	// optical depth, the light's share on its unlit side and the start (m) of its density's fade
	// with the panorama's footprint; and the unit direction of the surface wind, which its fibres
	// lie along veered by LOW_VEER, and the rain's slant and the tops' lean follow.
	readonly streaks = uniform( 0 );
	readonly streakHeight = tuneUniform( 'sky.streaks.height', STREAK_HEIGHT, { min: 6000, max: 11000, onChange: markSkyRetune } );
	readonly streakPeak = tuneUniform( 'sky.streaks.peak', STREAK_PEAK, { min: 0.2, max: 4, onChange: markSkyRetune } );
	readonly streakMuFloor = tuneUniform( 'sky.streaks.muFloor', STREAK_MU_FLOOR, { min: 0.1, max: 1, onChange: markSkyRetune } );
	readonly streakAmbient = tuneUniform( 'sky.streaks.ambient', STREAK_AMBIENT, { min: 0, max: 4, onChange: markSkyRetune } );
	// The synoptic field's values over which the layer's patches rise from none to whole.
	readonly streakPatchLow = tuneUniform( 'sky.streaks.patchLow', STREAK_PATCHES.low, { min: 0, max: 1, onChange: markSkyRetune } );
	readonly streakPatchHigh = tuneUniform( 'sky.streaks.patchHigh', STREAK_PATCHES.high, { min: 0, max: 1, onChange: markSkyRetune } );
	// The silk between the fibres (the tufts' share of the depth), the share of the depth left where
	// the fine fibres' mips average them, and how far the fibres eat into a patch's edge
	// (`streaksAt`).
	readonly streakFloor = tuneUniform( 'sky.streaks.floor', 0.4, { min: 0, max: 1, onChange: markSkyRetune } );
	readonly streakThin = tuneUniform( 'sky.streaks.thin', 0.5, { min: 0, max: 1, onChange: markSkyRetune } );
	readonly streakErode = tuneUniform( 'sky.streaks.erode', 0.5, { min: 0, max: 0.95, onChange: markSkyRetune } );
	readonly streakFade = tuneUniform( 'sky.streaks.aaStart', 1500, { min: 100, max: 20000, scale: 'log', onChange: markSkyRetune } );
	// The cap on the forward lobe, as the cosine to the sun where the lobe's value is taken, so a
	// streak against the sun's disc is no brighter than the glare round it.
	readonly streakForwardCap = tuneUniform( 'sky.streaks.forwardCap', 0.99, { min: 0.9, max: 0.9999, onChange: markSkyRetune } );
	// The sun's light before the air, lifted with the low sun's beam (SkyTrack.ts trackClouds), which
	// each point of the layer takes through the air toward the sun at the sun's elevation there
	// (`streaksAt`); the radius (m) over which a point's lead toward the sun raises it, and the
	// width (degrees) of the fade as its disc sinks under the layer's horizon; and the overall
	// colour on each of the layer's two shares (look/SkyLook.ts `streakDirect`, `streakAmbient`).
	readonly streakSunColour = uniform( new THREE.Color( 1, 1, 1 ) );
	readonly streakLitRadius = tuneUniform( 'sky.streaks.litRadius', STREAK_LIT.radius, { min: 1e5, max: 2e7, scale: 'log', onChange: markSkyRetune } );
	readonly streakLitWidth = tuneUniform( 'sky.streaks.litWidth', STREAK_LIT.width, { min: 0.1, max: 6, onChange: markSkyRetune } );
	readonly keyDirect = uniform( new THREE.Color( 1, 1, 1 ) );
	readonly keyAmbient = uniform( new THREE.Color( 1, 1, 1 ) );
	readonly windAxis = uniform( new THREE.Vector2( 1, 0 ) );
	// The cumulus' drift: the field is sampled at p + windOffset. Set through setDrift, which keeps
	// it within one noise period.
	readonly windOffset = uniform( new THREE.Vector2() );
	// The cirrus's drift in its frame (`streaksAt`'s toWind), wrapped on the period of each
	// texture it samples: the synoptic field's and each fibre scale's (setDrift).
	readonly streakShift = { synoptic: uniform( new THREE.Vector2() ), fine: uniform( new THREE.Vector2() ), broad: uniform( new THREE.Vector2() ) };
	readonly lightDir = uniform( new THREE.Vector3( 0.75, 0.53, - 0.12 ).normalize() );
	// The share of the clouds' shadow the map holds, none with the light on the horizon: Sky.ts
	// fades it, since a step there showed on the meter through the haze's light even with no direct
	// light on the island.
	readonly shadowShare = uniform( 1 );
	// Overcast's share of the sky (Sky.ts): as the map fades, on this share it fades to the closed
	// deck's mean transmittance along the light (`Clouds.DeckMean`), not to lit.
	readonly deckShadowShare = uniform( 0 );
	// Irradiance arriving at the cloud layer from the key light (scene units).
	readonly lightColor = uniform( new THREE.Color( 0.5, 0.45, 0.38 ) );
	// The key's irradiance on the clouds' tops, which a column passes down to its base (scene
	// units, SkyTrack.ts trackClouds): by day the sun's above the cloud, as `lightColor`; by night the
	// island's own moon key, not the clouds' moonlight, so the base and the land under it take one moon;
	// each times the weather's day sky at its own height (Sky.ts `daySkyMoon`).
	readonly columnKey = uniform( new THREE.Color( 0, 0, 0 ) );
	// The sun's share of the column's key (SkyTrack.ts trackClouds: 1 by day, 0 once the sun's light has
	// left it). A heap's part of the column's light goes by it, so that part is the sun's alone: lit
	// by the moon too, a fair sky's heaps read brighter in late twilight than the night's level is
	// set with and the meter takes the scene down. A layer's column takes the whole key.
	readonly columnSunShare = uniform( 1 );
	// The gain the open sky over the clouds' tops takes back (look/SkyLook.ts `clearSky`, `gapSky`): the
	// clear day's sky gain over the clouded sky's, on each body's share of the sky at its own height
	// (Sky.ts `gapGain`), read by the column's kernel alone (`columnTopOf`).
	readonly openSkyGain = { sun: uniform( 1 ), moon: uniform( 1 ) };
	// The night's faint ambient (Sky.ts `displayNightAmbient`), added to the atmosphere's own sky
	// light wherever the clouds take it.
	readonly nightAmbient = uniform( new THREE.Color( 0, 0, 0 ) );
	// Art balance of beam to sky light inside the clouds. A low sun's beam is already reddened
	// by the atmosphere; favouring it keeps sunset clouds glowing instead of muddy.
	readonly directGain = uniform( 1 );
	readonly ambientGain = uniform( 1 );
	// How far the cirrus's direct light (`streaksAt`) is desaturated toward its luminance: the
	// haze's clear-day value when it was set (weather/HazeKeys.ts), not linked to it; the haze's has
	// since been set higher, and the layer's keeps its own until it is set.
	readonly lightDesaturation = uniform( 0.26 );
	// The cirrus's colours (`streaksAt`): the moon's direct light gain (SkyTrack.ts trackClouds) and
	// the shaded side's desaturation.
	readonly moonLight = uniform( 0 );
	readonly ambientDesaturation = uniform( 0.26 );
	// The cloud shadow's fades toward lit (shadow): clamp( a_d r^2 + b_d ), r the horizontal distance
	// (m) from shadowCentre, and clamp( a_a |d - d0|^2 + b_a ), d the direction from the camera to the
	// point and d0 shadowFadeDirection, as ( a_d, b_d, a_a, b_a ). They start inert: the distance's
	// reaches 1 only past the map's half span (6 km), where the map's own edge fade has already
	// lifted the shadow, and the angle's is 0.
	readonly shadowFade = uniform( new THREE.Vector4( 10 / ( SHADOW_SPAN / 2 ) ** 2, - 10, 0, 0 ) );
	readonly shadowCentre = uniform( new THREE.Vector2( 0, 0 ) );
	readonly shadowFadeDirection = uniform( new THREE.Vector3( 0, - 1, 0 ) );
	readonly evolve = uniform( 0 );
	// The shell the clouds stand in, its base and thickness (m), and the share of the sky whose
	// layer stands low (cloudy's and overcast's), each blended by the sky's shares (`setShell`).
	readonly shellBottom = uniform( SHELL_BOTTOM );
	readonly shellThickness = uniform( SHELL_THICKNESS );
	readonly lowShare = uniform( 0 );
	// Overcast's share of the four skies (SkyTrack.ts trackClouds), which the dome's composite of a
	// closed layer reads (SkyDome.ts `glowToward`). Not `deckShare` (VolumetricHaze.ts),
	// `broadShare` or `deckShadowShare`, which carry the same share to the haze, the forward
	// light's spread and the shadow's fade, each set apart.
	readonly overcastShare = uniform( 0 );
	// The angular form of the light under a low layer for the air below it (FORWARD_WEIGHT): the
	// forward lobe's weight on the key's light and its mean cosine, and under overcast's share the
	// amount of forward light the closed layer spreads by the diffusion law, as a lobe's weight and
	// mean cosine whose light on level ground it takes (OVERCAST_BROAD). A change rebakes the
	// panorama and the environment.
	readonly forwardWeight = tuneUniform( 'sky.clouds.forwardWeight', FORWARD_WEIGHT, { min: 0, max: 3, onChange: markSkyRetune } );
	readonly forwardLobe = tuneUniform( 'sky.clouds.forwardLobe', FORWARD_LOBE, { min: 0, max: 0.95, onChange: markSkyRetune } );
	readonly overcastBroad = tuneUniform( 'sky.clouds.overcastBroad', OVERCAST_BROAD, { min: 0, max: 0.95, onChange: markSkyRetune } );
	readonly overcastWeight = tuneUniform( 'sky.clouds.overcastWeight', OVERCAST_WEIGHT, { min: 0, max: 3, onChange: markSkyRetune } );
	// The closed deck's glow about the key light (DECK_GLOW, SkyDome.ts `deckGlow`): the narrow
	// lobe's weight and mean cosine, the broad lobe's, and the power of the base's own pattern it
	// follows. The seen panorama draws it too, so a change rebakes.
	readonly deckGlow = tuneUniform( 'sky.clouds.deckGlow', DECK_GLOW, { min: 0, max: 3, onChange: markSkyRetune } );
	readonly deckGlowLobe = tuneUniform( 'sky.clouds.deckGlowLobe', DECK_GLOW_LOBE, { min: 0.5, max: 0.95, onChange: markSkyRetune } );
	readonly deckGlowBroad = tuneUniform( 'sky.clouds.deckGlowBroad', DECK_GLOW_BROAD, { min: 0, max: 3, onChange: markSkyRetune } );
	readonly deckGlowBroadLobe = tuneUniform( 'sky.clouds.deckGlowBroadLobe', DECK_GLOW_BROAD_LOBE, { min: 0, max: 0.9, onChange: markSkyRetune } );
	readonly deckGlowTexture = tuneUniform( 'sky.clouds.deckGlowTexture', DECK_GLOW_TEXTURE, { min: 0, max: 1.5, onChange: markSkyRetune } );
	// the closed deck's k away from the key by day, and the angles (degrees) of its rise
	readonly deckGlowFar = tuneUniform( 'sky.clouds.deckGlowFar', DECK_GLOW_FAR, { min: 0.2, max: 1.2, onChange: markSkyRetune } );
	readonly deckGlowFrom = tuneUniform( 'sky.clouds.deckGlowFrom', DECK_GLOW_FROM, { min: 10, max: 30, onChange: markSkyRetune } );
	readonly deckGlowTo = tuneUniform( 'sky.clouds.deckGlowTo', DECK_GLOW_TO, { min: 31, max: 60, onChange: markSkyRetune } );
	// The closed deck's glow as the lighting's sources take it (SkyDome.ts `glowToward`): one smooth
	// lobe about the sun, its mean cosine and weight; the dome and the seen panorama draw the two
	// above.
	readonly bakeGlowLobe = tuneUniform( 'sky.clouds.bakeGlowLobe', BAKE_GLOW_LOBE, { min: 0.7, max: 0.85, onChange: markSkyRetune } );
	readonly bakeGlowGain = tuneUniform( 'sky.clouds.bakeGlowGain', BAKE_GLOW_GAIN, { min: 0, max: 3, onChange: markSkyRetune } );
	// The closed deck's gradation from the zenith to the horizon (DECK_GRADATION): 1 the diffusion
	// law's 3 : 1, 0 a uniform deck, by overcast's share and the day's weight (`deckFlat`, set by
	// SkyTrack.ts trackClouds).
	readonly deckGradation = tuneUniform( 'sky.clouds.deckGradation', DECK_GRADATION, { min: 0, max: 1, onChange: markSkyRetune } );
	readonly deckFlat = uniform( 0 );
	// The share of the forward light spread by the diffusion law (overcast's share of the sky) and
	// the weight in use, `forwardWeight` moved to `overcastWeight` by that share (SkyTrack.ts trackClouds).
	readonly broadShare = uniform( 0 );
	readonly weight = uniform( FORWARD_WEIGHT );
	// The narrow lobe's mean cosine (`forwardLobe`), what the air under the deck reads.
	readonly lobe = uniform( FORWARD_LOBE );
	// The lobe's colour over the column key's: the sun's colour above the tops over the colour the
	// closed deck leaves the key (SkyTrack.ts trackClouds), white without overcast's share or a sun.
	readonly lobeTint = uniform( new THREE.Color( 1, 1, 1 ) );
	// The forward light on level ground at the key's height (Clouds.ts forwardLevel), set by Sky.ts
	// trackClouds: the narrow lobe's (`narrowLevel`) and it and the spread light's (the lobe of mean
	// cosine `overcastBroad`) by overcast's share (`forwardLevel`, the deck's level).
	readonly forwardLevel = uniform( forwardLevel( 0.5, FORWARD_LOBE ) );
	readonly narrowLevel = uniform( forwardLevel( 0.5, FORWARD_LOBE ) );
	// The haze's medium, which the haze writes (VolumetricHaze.ts `prepareMedium`, through
	// Pipeline.ts): where it closes, the low deck's view march ends (`makeMarch`).
	readonly haze = new HazeMedium();
	// The shell's mean height (km), where the clouds read the atmosphere's in-scatter table.
	private readonly cloudHeightKm = uniform( ( SHELL_BOTTOM + SHELL_THICKNESS / 2 ) / 1000 );
	readonly origin = uniform( new THREE.Vector3( 0, 15, 0 ) );
	// The panorama; replaced when its size changes (setDome), so readers go through domeNode.
	dome: THREE.StorageTexture;
	// Where a change of the panorama's power of two takes its texels to the new one and back (a
	// half-float texture cannot be read and written in one dispatch), at the panorama's size.
	domeScratch: THREE.StorageTexture;
	readonly shadowMap: THREE.StorageTexture;
	// Lightning (weather/Lightning.ts): each source's world position (m) and reach (m) in xyzw,
	// and its radiance this frame; `flashOn` is 1 while any of them is lit.
	readonly flash = Array.from( { length: FLASH_SOURCES }, () => ( { source: uniform( new THREE.Vector4( 0, - 1e6, 0, 1 ) ), color: uniform( new THREE.Color( 0, 0, 0 ) ) } ) );
	readonly flashOn = uniform( 0 );

	// Camera-relative frame: the camera sits on the planet axis at `camY`, its height held between a
	// metre and 50 m under the base (`update`), as every closed form under the shell assumes. The
	// march and the dome that fogs it at the march's depth (SkyDome.ts `domeColour`) take this one
	// height, so from a free camera inside or above the deck its depth and its fog agree.
	readonly camY = uniform( 15 );
	readonly camXZ = uniform( new THREE.Vector2() );
	readonly windN = uniform( new THREE.Vector2( 1, 0 ) );
	readonly phase = uniform( new THREE.Vector2() );
	readonly shadowRow = uniform( 0 ); // which of every SHADOW_PHASES shadow rows to refresh
	// Light march taps (offset, segment length) and their mip levels (shape, erosion).
	readonly lightTaps = Array.from( { length: MAX_LIGHT_TAPS }, () => uniform( new THREE.Vector4() ) );
	readonly lightLods = Array.from( { length: MAX_LIGHT_TAPS }, () => uniform( new THREE.Vector2() ) );
	// Tier settings (setDome, setMarch, setShadowRefresh, setViewScale).
	domeSize: { width: number; height: number; strideX: number; strideY: number } = { ...DOME_DEFAULTS };
	private domeOrder = ditherOrder( DOME_DEFAULTS.strideX, DOME_DEFAULTS.strideY );
	private readonly domeNode: THREE.TextureNode;
	// Half a panorama texel in uv: the tent filter's tap offsets.
	private readonly domeTexel = uniform( new THREE.Vector2( 0.5 / DOME_DEFAULTS.width, 0.5 / DOME_DEFAULTS.height ) );
	march: CloudMarch = { ...MARCH_DEFAULTS };
	private shadowRowsPerFrame = 1 / SHADOW_PHASES;
	private viewScale: CloudViewScale = { render: 0.5 };
	// The display the view's budget refers to: CSS pixels and the canvas's pixel ratio.
	private readonly display = { width: 1, height: 1, ratio: 1 };
	// The view: camera basis and frustum tangents now, at the previous trace, and as last traced.
	readonly cam = { right: uniform( new THREE.Vector3( 1, 0, 0 ) ), up: uniform( new THREE.Vector3( 0, 1, 0 ) ), fwd: uniform( new THREE.Vector3( 0, 0, - 1 ) ), tan: uniform( new THREE.Vector2( 1, 1 ) ) };
	readonly prev = { right: uniform( new THREE.Vector3( 1, 0, 0 ) ), up: uniform( new THREE.Vector3( 0, 1, 0 ) ), fwd: uniform( new THREE.Vector3( 0, 0, - 1 ) ), tan: uniform( new THREE.Vector2( 1, 1 ) ) };
	readonly viewCam = { right: uniform( new THREE.Vector3( 1, 0, 0 ) ), up: uniform( new THREE.Vector3( 0, 1, 0 ) ), fwd: uniform( new THREE.Vector3( 0, 0, - 1 ) ), tan: uniform( new THREE.Vector2( 1, 1 ) ) };
	readonly viewSize = uniform( new THREE.Vector2( 4, 4 ) );
	readonly traceSize = uniform( new THREE.Vector2( 1, 1 ) );
	// the trace's and the view's widths, which the kernels index by (uniforms: see buildView)
	readonly traceWidth = uniform( 1, 'uint' );
	readonly viewWidth = uniform( 1, 'uint' );
	readonly displayHeight = uniform( 4 );
	readonly viewValid = uniform( 0 );
	readonly slot = uniform( new THREE.Vector2() );
	readonly subPixel = uniform( new THREE.Vector2() );
	readonly frameNoise = uniform( 0 );
	readonly rebuildK = uniform( - 1 );
	readonly minAlpha = uniform( 0.12 );
	readonly historyValid = uniform( 0 );
	// 0 while a scene at rest settles: nothing moves, so the history needs no clamp to stay clean.
	readonly clampHistory = uniform( 1 );
	// The pre-exposure's scale each history buffer was written at, the shown one's (whose inverse
	// `sampleView` takes) and the ratio the resolve takes the reprojected history by: this
	// dispatch's scale over the scale of the history it reads (see the rules in the header).
	private readonly historyScale = [ 1, 1 ];
	/** The luminance of the light the clouds take, for the stores' powers of two, and the
	 * panorama's power of two for that light and the weather's skies (SkyTrack.ts trackClouds;
	 * PreExposure.ts `storeStops`). */
	storeLight = 0;
	storeStops = 18;
	// The panorama's power of two (stops) and its scale, written and read back (`sample`).
	private domeStops = Number.NaN;
	readonly domeStore = uniform( 1 );
	private readonly domeRead = uniform( 1 );
	private shownScale = 1;
	readonly viewUnscale = uniform( 1 );
	readonly historyGain = uniform( 1 );
	// The scale the view's kernels write at: the pre-exposure's times VIEW_HEADROOM (and the debug
	// press, PreExposure.ts `storePress`), set before they are dispatched.
	readonly writeScale = uniform( 1 );
	readonly camDelta = uniform( new THREE.Vector3() );

	shape: THREE.Data3DTexture | null = null;
	// The erosion noise's mean over the shape texture (its weighted channels: erodedDensity), per
	// unit of the erosion's strength and coverage: where the column light's slab ends under a top.
	erosionMean = 0.173 / 2;
	synoptic: THREE.StorageTexture | null = null;
	weather: THREE.StorageTexture | null = null;
	bounds: THREE.StorageTexture | null = null;
	streakCover: THREE.RenderTarget | null = null;
	private loading: Promise<void> | null = null;
	panoramaKernel: THREE.ComputeNode | null = null;
	// The panorama's texels times `domeRescale` into the scratch, and back (`update`).
	rescaleKernels: [ THREE.ComputeNode, THREE.ComputeNode ] | null = null;
	readonly domeRescale = uniform( 1 );
	shadowKernel: THREE.ComputeNode | null = null;
	// The light on the layer's tops and round its horizon, and the shares of the air's light under
	// a closed layer it sets (`columnTop`): uniforms and the atmosphere alone, so one thread works
	// them out once a frame (`Clouds.ColumnTop`, ahead of every kernel here: `update`) into four
	// texels the march, the panorama, the haze's light and the environment's haze read, instead of
	// each pixel and froxel working out its sky samples. ( key, lobe share ), ( sky, diffuse share ),
	// ( the key's share of the tops' light, the sun's share of the glow's lobe, 0, 0 ) for the
	// closed layer's glow (SkyDome.ts `deckGlow`), ( the horizon's ring, 0 ).
	private readonly columnStore = instancedArray( 4, 'vec4' ).setName( 'cloudColumnTop' );
	private readonly columnRead = storage( this.columnStore.value, 'vec4', 4 ).toReadOnly().setName( 'cloudColumnTopRead' );
	private columnKernel: THREE.ComputeNode | null = null;
	// The closed deck's mean transmittance along the light, ( with the light scattered on through,
	// the direct beam alone ): DECK_MEAN_RAYS rays (`Clouds.DeckMean`), then their mean in the last
	// element (`Clouds.DeckMeanSum`), dispatched only while the shadow map fades under a deck.
	readonly deckMeanStore = instancedArray( DECK_MEAN_RAYS + 1, 'vec2' ).setName( 'cloudDeckMean' );
	readonly deckMeanRead = storage( this.deckMeanStore.value, 'vec2', DECK_MEAN_RAYS + 1 ).toReadOnly().setName( 'cloudDeckMeanRead' );
	deckMeanKernel: THREE.ComputeNode | null = null;
	deckMeanSumKernel: THREE.ComputeNode | null = null;
	private previousFlash = '';
	view: View | null = null;
	readonly viewNode = texture( new THREE.Texture() );
	readonly viewDepthNode = texture( new THREE.Texture() );
	marchView: ( ( rd: V3, jitter: F ) => { L: V3; F: V3; T: F; depth: F } ) | null = null;
	// This frame's lightning in the view, traced at a quarter of its resolution.
	readonly flashNode = texture( new THREE.Texture() );
	streaksAt: ( ( rd: V3, pxAngle: F ) => V4 ) | null = null;
	rainAt: ( ( rd: V3, pxAngle: F, jitter: F, withFlash?: boolean ) => { color: V4; flash: V3 } ) | null = null;
	rainbowTable: THREE.DataTexture | null = null;
	private frame = 0;
	needsFull = true;
	// View accumulation state.
	private pingPong = 0;
	private traces = 0;
	private rebuild = 0;
	private since = 0;
	private hasPrevious = false;
	private readonly previousCamera = new THREE.Vector3();
	private readonly previousKey = new THREE.Vector3();
	// The surface wind's displacement the drift was last set from, and the one the previous frame's
	// view saw (the reprojection's drift is their difference, unwrapped).
	private readonly displacement = new THREE.Vector2();
	private readonly previousDisplacement = new THREE.Vector2();
	private previousTime = 0;
	private settled = false;
	private turned = 0;

	constructor( readonly atmosphere: Atmosphere ) {

		this.dome = makeDome( DOME_DEFAULTS.width, DOME_DEFAULTS.height );
		this.domeNode = texture( this.dome );
		this.domeScratch = makeDome( DOME_DEFAULTS.width, DOME_DEFAULTS.height, 'cloudDomeRescale' );
		this.shadowMap = new THREE.StorageTexture( SHADOW_SIZE, SHADOW_SIZE );
		this.shadowMap.type = THREE.UnsignedByteType;
		this.shadowMap.minFilter = this.shadowMap.magFilter = THREE.LinearFilter;
		this.shadowMap.generateMipmaps = false;
		this.shadowMap.name = 'cloudShadow';

	}

	get ready(): boolean { return this.panoramaKernel !== null; }

	/** Loads the shape noise, bakes the weather and the cirrus's fibres, and builds the panorama and
	 * shadow kernels. */
	init( renderer: THREE.WebGPURenderer ): Promise<void> {

		this.loading ??= loadCloudShape( renderer ).then( ( shape ) => this.build( renderer, shape ) );
		return this.loading;

	}

	private build( renderer: THREE.WebGPURenderer, shape: THREE.Data3DTexture ): void {

		this.shape = shape;
		const texels = shape.image.data as Uint8Array;
		let sum = 0;
		for ( let i = 0; i < texels.length; i += 4 ) sum += 0.113 * texels[ i ] + 0.04 * texels[ i + 1 ] + 0.02 * texels[ i + 2 ];
		this.erosionMean = sum / ( texels.length / 4 ) / 255;
		this.synoptic = bakeSynoptic( renderer );
		this.streakCover = bakeStreaks( renderer );
		const storage = ( size: number, filter: THREE.MagnificationTextureFilter ) => {

			const t = new THREE.StorageTexture( size, size );
			t.type = THREE.UnsignedByteType;
			t.wrapS = t.wrapT = THREE.RepeatWrapping;
			t.minFilter = t.magFilter = filter;
			t.generateMipmaps = false;
			return t;

		};
		const weather = this.weather = storage( WEATHER_RES, THREE.LinearFilter );
		const bounds = this.bounds = storage( BOUNDS_RES, THREE.NearestFilter );
		const weatherKernel = Fn( () => {

			const n = uint( WEATHER_RES );
			const c = ivec2( int( instanceIndex.mod( n ) ), int( instanceIndex.div( n ) ) );
			const q = vec2( c ).div( WEATHER_RES ).toVar();
			const mass = sat( weatherFbm( q, WEATHER_MASS ).sub( 0.5 ).mul( WEATHER_MASS[ 3 ] ).add( 0.5 ) );
			const detail = weatherFbm( q, WEATHER_DETAIL ).mul( 2 ).sub( 1 ).mul( WEATHER_DETAIL[ 3 ] );
			// the second channel: the large-scale field (WEATHER_LARGE)
			textureStore( weather, c, vec4( sat( mass.add( detail ).add( WEATHER_COVERAGE - 0.5 ) ), sat( weatherFbm( q, WEATHER_LARGE ) ), 0, 1 ) );

		} )().compute( WEATHER_RES * WEATHER_RES, [ 64 ] ).setName( 'Clouds.Weather' );
		// The most cloud-capable weather in each cell of a coarse grid, including the texels its
		// bilinear lookups reach across the cell's edges (and the wrap).
		const weatherLoad = texture( weather );
		const boundsKernel = Fn( () => {

			const n = uint( BOUNDS_RES );
			const c = ivec2( int( instanceIndex.mod( n ) ), int( instanceIndex.div( n ) ) );
			const k = WEATHER_RES / BOUNDS_RES;
			const lo = ivec2( vec2( c ).mul( k ).sub( 1 ) );
			const highest = float( 0 ).toVar();
			Loop( { start: 0, end: k + 2, type: 'int' }, { start: 0, end: k + 2, type: 'int' }, ( { i, j } ) => {

				const p = lo.add( ivec2( int( i ), int( j ) ) ).add( WEATHER_RES ).mod( WEATHER_RES );
				highest.assign( max( highest, textureLoad( weatherLoad, p ).r ) );

			} );
			textureStore( bounds, c, vec4( min( highest.add( 1 / 255 ), 1 ), 0, 0, 1 ) );

		} )().compute( BOUNDS_RES * BOUNDS_RES, [ 64 ] ).setName( 'Clouds.WeatherBounds' );
		renderer.compute( weatherKernel );
		renderer.compute( boundsKernel );
		this.rainbowTable = makeRainbowTexture();
		this.columnKernel = Fn( () => {

			// the shares of the air's light under overcast's deck (VolumetricHaze.ts `scatterToward`):
			// the narrow lobe's light over an even deck's of the deck's irradiance (twice the lobe's
			// radiance over the irradiance over pi, since an even deck fills half the sphere), on the
			// key's light alone and by the share of it still a lobe ( 1 - `broadShare` ), and the
			// diffuse part's share of that irradiance, the spread forward light's included (the deck's
			// level keeps it: `forwardLevel`)
			const top = this.columnTopOf();
			const keyTop = luminance( top.key ).toVar();
			const level = keyTop.mul( this.weight.mul( this.forwardLevel ).add( 1 ) ).add( luminance( top.sky ) );
			const lobeShare = keyTop.mul( this.weight ).mul( 2 ).div( max( level, 1e-20 ) ).mul( float( 1 ).sub( this.broadShare ) ).toVar();
			const diffuseShare = float( 1 ).sub( lobeShare.mul( this.narrowLevel ).mul( 0.5 ) );
			// the deck's glow (SkyDome.ts `deckGlow`): the key's share of the light on the tops,
			// its illuminance along its own direction (not its level ground's, which goes at the
			// horizon while the light still reaches the deck) against the sky's over them; and the
			// sun's share of the lobe, as the march mixes its lobes, toward the zenith
			const keyAlong = luminance( this.columnKey.rgb ).mul( 1 / Math.PI ).toVar();
			const keyShare = keyAlong.div( max( keyAlong.add( luminance( top.sky ) ), 1e-20 ) );
			const a = this.atmosphere;
			const sunY = luminance( a.cloudRows( float( 1 ), this.cloudHeightKm, 0 ).mie ).mul( luminance( a.sunIlluminance.rgb ) );
			const moonY = luminance( a.cloudRows( float( 1 ), this.cloudHeightKm, 1 ).mie ).mul( luminance( a.moonIlluminance.rgb ) );
			const sunShare = smoothstep( 0, 1, sunY.div( max( sunY.add( moonY ), 1e-20 ) ) );
			this.columnStore.element( 0 ).assign( vec4( top.key, lobeShare ) );
			this.columnStore.element( 1 ).assign( vec4( top.sky, diffuseShare ) );
			this.columnStore.element( 2 ).assign( vec4( keyShare, sunShare, 0, 0 ) );
			this.columnStore.element( 3 ).assign( vec4( this.horizonMean(), 0 ) );

		} )().compute( 1, [ 1 ] ).setName( 'Clouds.ColumnTop' );
		buildKernels( this );

	}

	/** The horizontal unit vector toward the sun's azimuth, from which the rings of sky samples that
	 * turn with the sun are laid (the horizon's and the deck top's, makeMarch and `columnTop`): the
	 * offset keeps a sun at the zenith off the zero vector. */
	private sunAzimuth(): V2 {

		return normalize( vec2( this.atmosphere.sunDir.x.add( 1e-5 ), this.atmosphere.sunDir.z ) ).toVar();

	}

	/** The light on the layer's tops over pi (the key's on a level top and the sky's mean over the
	 * upper hemisphere) and the horizon's ring, which the march colours its factors by (makeMarch),
	 * the shares of the air's light under a low layer they set (VolumetricHaze.ts `scatterToward`)
	 * and the closed layer's glow's (SkyDome.ts `deckGlow`: the key's share of the tops' light, the
	 * sun's share of the lobe), as this frame's `Clouds.ColumnTop` worked them out (`columnTopOf`). */
	columnTop(): { key: V3; sky: V3; horizon: V3; lobeShare: F; diffuseShare: F; keyShare: F; sunShare: F } {

		const a = this.columnRead.element( 0 ), b = this.columnRead.element( 1 ), c = this.columnRead.element( 2 );
		return { key: a.xyz, sky: b.xyz, horizon: this.columnRead.element( 3 ).xyz, lobeShare: a.w, diffuseShare: b.w, keyShare: c.x, sunShare: c.y };

	}

	/** The horizon sky a cloud's base sees: the whole ring round it, not the side toward the sun. A
	 * low sun's orange glow fills a short arc of that ring, so the ring's average is far paler, and
	 * the sun's own colour reaches the clouds through their direct light. The sky's radiance is
	 * symmetric about the sun's azimuth, so a half ring from the sun round to its opposite side
	 * holds the whole ring's average; its azimuths are relative to the sun, so the average moves
	 * smoothly with it. The ring is a band, not a line (HORIZON_BAND, from 3 to 24 degrees up): a
	 * cloud's side takes the low sky's light nearly evenly over that band, and the twilight's glow
	 * lies in its lowest few degrees, so a ring read at the horizon alone lit the clouds orange and
	 * brighter than the sky behind them with the sun 9 degrees under it. */
	private horizonMean(): V3 {

		const toward = this.sunAzimuth();
		const sum = vec3( 0 ).toVar();
		Loop( HORIZON_RING, ( { i } ) => {

			const a = float( i ).add( 0.5 ).mul( Math.PI / HORIZON_RING ).toVar();
			const c = cos( a ), n = sin( a );
			for ( const up of HORIZON_BAND ) {

				const out = Math.sqrt( 1 - up * up );
				sum.addAssign( this.atmosphere.skyRadiance( vec3( toward.x.mul( c ).sub( toward.y.mul( n ) ).mul( out ), up, toward.x.mul( n ).add( toward.y.mul( c ) ).mul( out ) ), false ) );

			}

		} );
		return sum.mul( 1 / ( HORIZON_RING * HORIZON_BAND.length ) );

	}

	/** The light on the clouds' tops over pi: the key's above the tops (`columnKey`) times its
	 * height's cosine, and the drawn sky's over them (not the clouds' own night glow, which the
	 * drawn sky has none of), a coarse cosine-weighted mean of its radiance over the upper
	 * hemisphere (exact for a uniform sky): two rings, 30 and 60 degrees from the zenith, of four
	 * azimuths from the sun's, weighted alike. The sky over the tops is the open sky, the clear
	 * day's, so it takes the clear day's gain back (`openSkyGain`). */
	private columnTopOf(): { key: V3; sky: V3 } {

		const toward = this.sunAzimuth();
		const sky = vec3( 0 ).toVar();
		for ( const zenith of [ 30, 60 ] ) {

			const up = Math.cos( zenith * Math.PI / 180 ), out = Math.sin( zenith * Math.PI / 180 );
			for ( const [ c, n ] of [ [ 1, 0 ], [ 0, 1 ], [ - 1, 0 ], [ 0, - 1 ] ] ) {

				sky.addAssign( this.atmosphere.skyRadiance( vec3( toward.x.mul( c ).sub( toward.y.mul( n ) ).mul( out ), up, toward.x.mul( n ).add( toward.y.mul( c ) ).mul( out ) ), false, this.openSkyGain ) );

			}

		}
		return { key: this.columnKey.rgb.mul( max( this.lightDir.y, 0 ) ).mul( 1 / Math.PI ), sky: sky.mul( 1 / 8 ) };

	}

	/** Advance the clouds for a frame: the panorama, the shadow map, the distance field after a
	 * coverage change, and the view as seen by `camera` (its drawing buffer `width` by `height`).
	 * `time` is the scene clock: a still camera over a paused clock stops tracing once the view
	 * has converged, so a paused frame is bit-identical. */
	update( renderer: THREE.WebGPURenderer, camera: THREE.PerspectiveCamera | null, width: number, height: number, time: number ): void {

		if ( ! this.panoramaKernel || ! this.shadowKernel ) return;
		// debugFlags.cloudShadowRows = 4 restores the old rate (a quarter of the rows per frame).
		const rowsFlag = debugFlags.cloudShadowRows;
		const shadowRows = typeof rowsFlag === 'number' && Number.isFinite( rowsFlag ) && rowsFlag > 0 ? 1 / rowsFlag : this.shadowRowsPerFrame;
		// Before the first frame there is no view yet, only the panorama for the first bake.
		const position = camera ? camera.position : this.origin.value;
		this.origin.value.copy( position );
		// The march assumes a camera below the cloud base.
		this.camY.value = THREE.MathUtils.clamp( position.y, 1, this.shellBottom.value - 50 );
		this.camXZ.value.set( position.x, position.z );
		this.windN.value.copy( this.windAxis.value );
		this.updateLightTaps();
		// The light on the layer's tops and round its horizon, ahead of every reader.
		renderer.compute( this.columnKernel! );
		// The deck's mean transmittance, which the shadow map's rows take while it fades under a deck.
		if ( this.deckShadowShare.value > 0 && this.shadowShare.value < 1 ) {

			renderer.compute( this.deckMeanKernel! );
			renderer.compute( this.deckMeanSumKernel! );

		}
		// The panorama's power of two for the clouds' light now: a new one takes the texels it holds
		// to it in this frame, ahead of this frame's phase and of every reader (the view and the
		// shadow map keep their own), so its texels and its readers always share one. Only the
		// first, with nothing held yet, writes every phase.
		const stops = this.storeStops;
		const restore = stops !== this.domeStops;
		const rescale = restore && Number.isFinite( this.domeStops );
		if ( restore ) {

			this.domeRescale.value = rescale ? 2 ** ( stops - this.domeStops ) : 1;
			this.domeStops = stops;
			this.domeStore.value = 2 ** stops;
			this.domeRead.value = 2 ** - stops;

		}
		// Panorama: an ordered pattern spreads each frame's texels over the map. A full refresh runs
		// every phase through the same kernel, so a refreshed map matches a progressive one.
		const full = this.needsFull;
		this.needsFull = false;
		if ( rescale && ! full ) for ( const kernel of this.rescaleKernels! ) renderer.compute( kernel );
		this.refreshMaps( renderer, full, shadowRows, full || ( restore && ! rescale ) );

		if ( ! camera ) return;
		// ---- view
		const [ w, h ] = this.viewDimensions( width, height );
		if ( ! this.view || this.view.width !== w || this.view.height !== h ) buildView( this, w, h );
		this.displayHeight.value = Math.max( 4, height );
		const e = camera.matrixWorld.elements;
		const tanY = Math.tan( THREE.MathUtils.degToRad( camera.fov * 0.5 ) ) / ( camera.zoom || 1 );
		const C = this.cam, P = this.prev;
		P.right.value.copy( C.right.value ); P.up.value.copy( C.up.value ); P.fwd.value.copy( C.fwd.value ); P.tan.value.copy( C.tan.value );
		C.right.value.set( e[ 0 ], e[ 1 ], e[ 2 ] ).normalize();
		C.up.value.set( e[ 4 ], e[ 5 ], e[ 6 ] ).normalize();
		C.fwd.value.set( - e[ 8 ], - e[ 9 ], - e[ 10 ] ).normalize();
		C.tan.value.set( tanY * camera.aspect, tanY );
		// Camera cuts (teleports, big turns, jumps of the key light, zoom) restart the history.
		const moved = this.hasPrevious ? position.distanceTo( this.previousCamera ) : Infinity;
		this.turned = C.fwd.value.angleTo( P.fwd.value );
		const key = this.lightDir.value;
		if ( full || moved > 8 || this.turned > 0.5 || key.angleTo( this.previousKey ) > 0.05 || Math.abs( C.tan.value.y - P.tan.value.y ) > 1e-3 * C.tan.value.y ) this.resetHistory();
		this.previousKey.copy( key );
		// Camera motion since the last frame minus the clouds' drift. The field is sampled at
		// position + offset, so the clouds move by minus the offset's change.
		const step = this.hasPrevious ? this.displacement.clone().sub( this.previousDisplacement ) : new THREE.Vector2();
		const low = driftOffset( step, LOW_DRIFT, LOW_VEER );
		const dx = this.hasPrevious ? position.x - this.previousCamera.x : 0, dy = this.hasPrevious ? position.y - this.previousCamera.y : 0;
		const dz = this.hasPrevious ? position.z - this.previousCamera.z : 0;
		this.camDelta.value.set( dx + low.x, dy, dz + low.y );
		// At rest: a paused clock and a still camera (a jump in weather or time while paused still
		// counts, and settles at once).
		// A change of lightning while paused (a review strike) retraces the view, so its flash is current.
		const flash = this.flash.map( ( { source, color } ) => `${ source.value.toArray() },${ color.value.toArray() }` ).join( ';' );
		const resting = this.hasPrevious && moved <= 1e-4 && this.turned <= 1e-6 && time === this.previousTime && flash === this.previousFlash;
		this.previousFlash = flash;
		this.previousCamera.copy( position );
		this.previousDisplacement.copy( this.displacement );
		this.previousTime = time;
		this.hasPrevious = true;

		// (a view kept at rest under another P than now: see the rules in the header)
		const rescaled = this.settled && this.shownScale !== this.storeScale();
		if ( resting && ( ! this.settled || full || rescaled ) ) {

			// The scene has come to rest: settle the view once, from scratch, then keep it (or settle it
			// again at a new P). Depending only on the scene and P, a paused frame is
			// bit-identical. The panorama and the shadow map (camera-relative, refreshed a share per
			// frame) are brought up to date with it, so a paused frame does not wait out their refresh
			// cycles after a camera move.
			if ( ! full ) this.refreshMaps( renderer, true, 1 );
			this.settle( renderer );

		} else if ( this.rebuild < 16 ) {

			// After a cut: several slots per frame, until every pixel has a sample of its own.
			for ( let k = 0; k < REBUILD_SLOTS && this.rebuild < 16; k ++ ) {

				if ( k > 0 ) this.holdCamera();
				this.rebuildK.value = this.rebuild;
				this.trace( renderer, ORDER[ this.rebuild ++ ] );

			}

		} else if ( ! resting ) {

			this.settled = false;
			const slots = this.turned > 0.003 ? 2 : 1;
			this.rebuildK.value = - 1;
			// A running average while a pixel has few samples, then an exponential one; the count
			// advances once per frame, however many slots the frame traces.
			const n = 1 + Math.floor( this.since ++ / 16 );
			this.minAlpha.value = Math.max( 0.12, 1 / ( n + 1 ) );
			for ( let k = 0; k < slots; k ++ ) {

				if ( k > 0 ) this.holdCamera();
				// A turning camera traces a second slot per frame, so the reprojected history stays sharp.
				this.trace( renderer, ORDER[ ( this.frame + k * 8 ) % 16 ] );

			}

		}
		this.frame ++;

	}

	// Refresh the panorama and the cloud shadow map: whole, or this frame's share (one phase of the
	// panorama's pattern, `shadowRows` of the shadow map's rows); `dome` refreshes the whole panorama
	// alone.
	private refreshMaps( renderer: THREE.WebGPURenderer, full: boolean, shadowRows: number, dome = full ): void {

		const order = this.domeOrder;
		for ( const [ x, y ] of dome ? order : [ order[ this.frame % order.length ] ] ) {

			this.phase.value.set( x, y );
			renderer.compute( this.panoramaKernel! );

		}
		const passes = full ? SHADOW_PHASES : Math.min( SHADOW_PHASES, Math.max( 1, Math.round( shadowRows * SHADOW_PHASES ) ) );
		const first = full ? 0 : ( this.frame * passes ) % SHADOW_PHASES;
		for ( let k = 0; k < passes; k ++ ) {

			this.shadowRow.value = SHADOW_ORDER[ ( first + k ) % SHADOW_PHASES ];
			renderer.compute( this.shadowKernel! );

		}

	}

	// The scale the view's kernels write at now (`writeScale`).
	private storeScale(): number {

		return preExposure.scale.value * VIEW_HEADROOM * storePress.store.value;

	}

	// Trace one slot of every 4 x 4 block and resolve it into the history.
	private trace( renderer: THREE.WebGPURenderer, order: number ): void {

		const view = this.view!;
		this.slot.value.set( order % 4, Math.floor( order / 4 ) );
		// R2 sequence over the traces (one offset per 16-frame cycle, so each pixel sees them all),
		// and a golden-ratio march offset per cycle: every trace of a pixel gets a new offset.
		const n = Math.floor( this.traces / 16 ) + 1;
		this.subPixel.value.set( ( ( 0.5 + n * 0.7548776662 ) % 1 ) - 0.5, ( ( 0.5 + n * 0.5698402910 ) % 1 ) - 0.5 ).multiplyScalar( 0.25 );
		this.frameNoise.value = ( n * 0.6180339887 ) % 1;
		this.traces ++;
		// (the kernels run at the pre-exposure's scale as it stands now; the resolve reads the other
		// history buffer and writes this one)
		const scale = this.storeScale();
		this.writeScale.value = scale;
		renderer.compute( view.kernels.trace );
		renderer.compute( view.kernels.high );
		renderer.compute( view.kernels.box );
		this.historyGain.value = scale / this.historyScale[ 1 - this.pingPong ];
		renderer.compute( view.kernels.resolve[ this.pingPong ] );
		this.historyScale[ this.pingPong ] = scale;
		this.shownScale = scale;
		this.viewUnscale.value = 1 / scale;
		this.viewNode.value = view.history[ this.pingPong ];
		this.viewDepthNode.value = view.depthHistory[ this.pingPong ];
		this.pingPong = 1 - this.pingPong;
		this.historyValid.value = 1;
		const C = this.cam, V = this.viewCam;
		V.right.value.copy( C.right.value ); V.up.value.copy( C.up.value ); V.fwd.value.copy( C.fwd.value ); V.tan.value.copy( C.tan.value );
		this.viewValid.value = 1;

	}

	// Rebuild the view and average SETTLE_CYCLES samples per pixel in one go, with the trace
	// sequence restarted, so the result depends only on the scene.
	private settle( renderer: THREE.WebGPURenderer ): void {

		this.resetHistory();
		this.traces = 0;
		this.holdCamera();
		this.clampHistory.value = 0;
		for ( let k = 0; k < 16; k ++ ) {

			this.rebuildK.value = k;
			this.trace( renderer, ORDER[ k ] );

		}
		this.rebuildK.value = - 1;
		for ( let k = 16; k < 16 * SETTLE_CYCLES; k ++ ) {

			this.minAlpha.value = 1 / ( Math.floor( k / 16 ) + 1 );
			this.trace( renderer, ORDER[ k % 16 ] );

		}
		this.clampHistory.value = 1;
		this.rebuild = 16;
		this.settled = true;

	}

	// No camera motion between the traces of one frame.
	private holdCamera(): void {

		const C = this.cam, P = this.prev;
		P.right.value.copy( C.right.value ); P.up.value.copy( C.up.value ); P.fwd.value.copy( C.fwd.value ); P.tan.value.copy( C.tan.value );
		this.camDelta.value.set( 0, 0, 0 );

	}

	// The light march toward the key light: segments growing by LIGHT_GROWTH, the first sized so
	// that together they span the shell's depth, spread over a narrow cone (golden-angle spiral) and
	// read at mip levels matching their spacing.
	private updateLightTaps(): void {

		const L = this.lightDir.value;
		const tangent = Math.abs( L.y ) > 0.99 ? new THREE.Vector3( 0, - L.z, L.y ) : new THREE.Vector3( L.z, 0, - L.x );
		tangent.normalize();
		const bitangent = new THREE.Vector3().crossVectors( L, tangent );
		const taps = this.march.lightTaps;
		const first = this.shellThickness.value * ( LIGHT_GROWTH - 1 ) / ( LIGHT_GROWTH ** taps - 1 );
		for ( let k = 0; k < taps; k ++ ) {

			const length = first * LIGHT_GROWTH ** k, mid = first * ( LIGHT_GROWTH ** k - 1 ) / ( LIGHT_GROWTH - 1 ) + length * 0.5;
			const angle = ( k + 1 ) * 2.399963, radius = Math.sqrt( ( k + 1.5 ) / ( taps + 1 ) ) * 0.05 * mid;
			const u = Math.cos( angle ) * radius, v = Math.sin( angle ) * radius;
			this.lightTaps[ k ].value.set(
				L.x * mid + tangent.x * u + bitangent.x * v, L.y * mid + tangent.y * u + bitangent.y * v,
				L.z * mid + tangent.z * u + bitangent.z * v, length );
			const footprint = mid * 0.1;
			const lod = ( scale: number ) => Math.max( 0, Math.log2( footprint * 64 / scale ) );
			this.lightLods[ k ].value.set( lod( BASE_SCALE ), lod( EROSION_SCALE ) );

		}

	}

	/** Restart the view accumulation, rebuilding it at full rate over the next frames. */
	resetHistory(): void {

		this.rebuild = 0;
		this.since = 0;
		this.settled = false;
		this.historyValid.value = 0;

	}

	/** The next update recomputes the whole panorama and restarts the view, e.g. after a jump in
	 * time or weather. */
	invalidate(): void {

		this.needsFull = true;
		this.resetHistory();

	}

	/** Back to the state before a fresh load's first frame (every explicit bake, through Sky.reset, which
	 * then sets the wind's offset, the evolution and the axes to the scene clock): the wind's offset,
	 * the evolution, the wind's axis and the viewer as constructed, no view history, the
	 * frame count at zero and a whole refresh due. */
	restart(): void {

		this.evolve.value = 0;
		this.setDrift( new THREE.Vector2() );
		this.windAxis.value.set( 1, 0 );
		this.origin.value.set( 0, 15, 0 );
		for ( const { source, color } of this.flash ) {

			source.value.set( 0, - 1e6, 0, 1 );
			color.value.setRGB( 0, 0, 0 );

		}
		this.flashOn.value = 0;
		this.frame = 0;
		this.pingPong = 0;
		this.historyScale.fill( 1 );
		this.shownScale = 1;
		this.viewUnscale.value = 1;
		this.historyGain.value = 1;
		this.traces = 0;
		this.turned = 0;
		this.hasPrevious = false;
		this.previousCamera.set( 0, 0, 0 );
		this.previousKey.set( 0, 0, 0 );
		this.previousDisplacement.set( 0, 0 );
		this.previousTime = 0;
		this.previousFlash = '';
		if ( this.view ) {

			this.viewNode.value = this.view.history[ 0 ];
			this.viewDepthNode.value = this.view.depthHistory[ 0 ];

		}
		this.viewValid.value = 0;
		this.invalidate();

	}

	/** In-scattered radiance (rgb) and transmittance (a) of the clouds in a direction, from the
	 * panorama: reflections, the environment and the sun's visibility. A small tent filter over
	 * four bilinear taps hides the per-texel march offsets. */
	sample( direction: V3 ): V4 {

		const q = directionToDome( direction ).toVar();
		const texel = this.domeTexel;
		const tap = ( x: number, y: number ) => this.domeNode.sample( q.add( texel.mul( vec2( x, y ) ) ) ).level( float( 0 ) );
		const s = tap( - 1, - 1 ).add( tap( 1, - 1 ) ).add( tap( - 1, 1 ) ).add( tap( 1, 1 ) ).mul( 0.25 ).toVar();
		// The map holds the upper sky only: below the horizon the clouds fade out instead of
		// repeating the horizon's row.
		const below = smoothstep( - 0.04, 0, direction.y );
		return vec4( s.rgb.mul( this.domeRead ).mul( storePress.read ).mul( below ), mix( float( 1 ), s.a, below ) );

	}

	// Light from the lightning sources at a camera-relative point: an inverse-square core,
	// softened so a source never blows out, times an exponential reach standing in for the cloud
	// the light crosses (VolumetricCloudsCS1's Lightning()). There is no march toward the source:
	// added inside the front-to-back march, the cloud between the glow and the viewer dims it, so
	// a flash reads as coming from within the cloud.
	flashLight( p: V3 ): V3 {

		const world = vec3( p.x.add( this.camXZ.x ), p.y, p.z.add( this.camXZ.y ) ).toVar();
		let sum: V3 = vec3( 0 );
		for ( const { source, color } of this.flash ) {

			const r = length( world.sub( source.xyz ) );
			const core = source.w.mul( 0.2 );
			sum = sum.add( color.mul( exp( r.div( source.w ).negate() ).div( r.mul( r ).div( core.mul( core ) ).add( 1 ) ) ) );

		}
		return sum;

	}

	/** Light of the lightning inside the clouds in a direction, for reflections and directions
	 * outside the view: the march's light where the ray enters the cloud deck, times the cloud's
	 * `opacity` there (the dome fogs it with the clouds). Without texture lookups it costs the
	 * water shader no sampler. */
	flashAt( direction: V3, opacity: F ): V3 {

		const result = vec3( 0 ).toVar();
		If( this.flashOn.greaterThan( 0.5 ).and( direction.y.greaterThan( 0 ) ), () => {

			// A little above the base, where the march gathers most of a thick deck's light.
			const t = layerDistance( this.camY, direction.y, this.shellBottom.add( 150 ) ).toVar();
			const p = vec3( direction.x.mul( t ), this.camY.add( direction.y.mul( t ) ), direction.z.mul( t ) );
			result.assign( this.flashLight( p ).mul( opacity ).mul( smoothstep( 0, 0.04, direction.y ) ) );

		} );
		return result;

	}

	/** The clouds as seen directly: the view, upsampled bicubically and looked up with the camera
	 * it was traced with; directions outside it, or a view without data yet, take the panorama. */
	sampleView( direction: V3 ): V4 {

		return sampleView( this, direction );

	}

	/** What the dome fogs the clouds by toward `direction`, as the view holds it beside its colour
	 * (the resolve's depth history): `depth`, the distance (metres) to the cumulus's light, the
	 * march's opacity-weighted depth (`makeMarch`), `fallback` outside the view, before it has data,
	 * or where it holds no cumulus; and `thin`, the cirrus's opacity as seen through what lies
	 * in front of it, 0 outside the view. */
	sampleViewFog( direction: V3, fallback: F ): { depth: F; thin: F } {

		return sampleViewFog( this, direction, fallback );

	}

	/** Review aid: the cosine-weighted fraction of the upper sky behind cloud (transmittance
	 * under one half) in the panorama. */
	async skyCoverage( renderer: THREE.WebGPURenderer ): Promise<number> {

		const width = 256, height = 64;
		const target = new THREE.RenderTarget( width, height, { type: THREE.FloatType, depthBuffer: false } );
		const material = new THREE.NodeMaterial();
		material.depthTest = material.depthWrite = false;
		material.toneMapped = false;
		material.fragmentNode = vec4( this.domeNode.sample( uv() ).level( float( 0 ) ).a );
		const quad = new THREE.QuadMesh( material );
		const previous = renderer.getRenderTarget();
		renderer.setRenderTarget( target );
		quad.render( renderer );
		renderer.setRenderTarget( previous );
		const pixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height ) as Float32Array;
		let covered = 0, total = 0;
		for ( let y = 0; y < height; y ++ ) {

			// Rows may come back flipped; the weight is symmetric in neither, so use the row's own uv.
			const v = ( y + 0.5 ) / height, lat = v * v * Math.PI / 2;
			const weight = Math.cos( lat ) * Math.sin( lat ) * v;
			for ( let x = 0; x < width; x ++ ) {

				total += weight;
				if ( pixels[ ( y * width + x ) * 4 ] < 0.5 ) covered += weight;

			}

		}
		target.dispose();
		material.dispose();
		return covered / total;

	}

	/** Direct-light visibility under the clouds at a world position. */
	shadow( worldPosition: V3 ): F {

		return shadow( this, worldPosition );

	}

	/** The direct beam alone under the clouds at a world position, without the light the cloud
	 * scatters on through (the map's blue channel), for the sea's glint: under a closed deck that
	 * light is diffuse and makes no mirror image of the sun. Equal to `shadow` where no cloud lies
	 * along the key. */
	directShadow( worldPosition: V3 ): F {

		return directShadow( this, worldPosition );

	}

	/** The shadow map read for the point `p` of the air (the haze's shafts, VolumetricHaze.ts): the
	 * map holds each ground point's flat beam toward `lightDir` (under a low light the beams about
	 * that one, the air's channel narrower than the ground's from about 5 degrees up), its slope
	 * clamped at SHADOW_SLOPE_FLOOR, so p is taken down that same beam to its ground point,
	 * p.xz - L.xz p.y / max( L.y, floor ). Returns ( the green channel's bilinear sample, its
	 * bicubic B-spline blur ), both faded toward lit as `shadow()`
	 * fades them for p (the edge of the map, the distance and direction fades). The blur takes four
	 * bilinear taps (Sigg and Hadwiger, "Fast Third-Order Texture Filtering", GPU Gems 2, 2005); the
	 * B-spline smooths rather than interpolates (it does not pass through the texels' values), as a
	 * blur for the unsharp mask should. */
	shadowAlongBeam( p: V3 ): V2 {

		return shadowAlongBeam( this, p );

	}

	/** The shell the march reads: its base and thickness (m) and the share of the sky whose layer
	 * stands low. Sky.ts blends the weathers' shells by their shares; the light taps follow at the
	 * next update. */
	setShell( bottom: number, thickness: number, lowShare: number ): void {

		// (a shell is never thinner than SHELL_MIN: the march divides by its depth, and the tunes
		// can set a base over its top)
		thickness = Math.max( thickness, SHELL_MIN );
		this.shellBottom.value = bottom;
		this.shellThickness.value = thickness;
		this.lowShare.value = lowShare;
		this.cloudHeightKm.value = ( bottom + thickness / 2 ) / 1000;

	}

	/** The mean diffuse transmission of a closed layer of the shell and density as set: a stratus
	 * slab's (the diffusion approximation's, as the march's column takes it) over its relief,
	 * every depth from its thinnest to its cap alike; 1 for a layer of no depth. */
	layerTransmission(): number {

		const k = COLUMN_DIFFUSION * this.shellThickness.value * STRATUS_CAP * this.density.value * DENSITY;
		if ( ! ( k > 1e-6 ) ) return 1;
		return Math.log( ( 1 + k ) / ( 1 + k * ( 1 - STRATUS_RELIEF ) ) ) / ( k * STRATUS_RELIEF );

	}

	/** The shape noise's rise at the scene clock `time` (s). */
	setEvolution( time: number ): void {

		this.evolve.value = ( time * 2.5 ) % EVOLVE_PERIOD;

	}

	/** Drift the clouds by the surface wind's integrated displacement (metres downwind; Wind.ts
	 * `displacement`): a function of the wind's state alone, so a paused frame and a seek draw the
	 * same sky. Set `windAxis` first: the cirrus's shift is kept in its frame. */
	/** Each layer's sampling offset before wrapping, from the displacement setDrift was last given:
	 * the layer moves by minus its change (the wind probe's W5 reads the drifts off these). The
	 * streaks drift with the cumulus. */
	layerOffsets(): { cumulus: THREE.Vector2; streaks: THREE.Vector2 } {

		const low = driftOffset( this.displacement, LOW_DRIFT, LOW_VEER );
		return { cumulus: low, streaks: low.clone() };

	}

	setDrift( displacement: THREE.Vector2 ): void {

		this.displacement.copy( displacement );
		const wrap = ( v: number, period: number ) => v - period * Math.round( v / period );
		const low = driftOffset( displacement, LOW_DRIFT, LOW_VEER );
		this.windOffset.value.set( wrap( low.x, TILE ), wrap( low.y, TILE ) );
		// The streaks drift with the cumulus, in the frame their shader samples in (toWind), wrapped
		// per texture: the periods do not nest, and an offset wrapped on one would jump the other.
		const w = this.windAxis.value.clone().normalize().rotateAround( new THREE.Vector2(), LOW_VEER );
		const qx = low.x * w.x + low.y * w.y, qy = low.y * w.x - low.x * w.y;
		const { synoptic, fine, broad } = this.streakShift;
		synoptic.value.set( wrap( qx, SYNOPTIC_SIZE ), wrap( qy, SYNOPTIC_SIZE ) );
		const fibres = ( shift: THREE.Vector2, { tile, turn }: { tile: number; turn: number } ) => {

			// wrapped on the tile's lattice, which sampleStreaks turns by `turn`
			const c = Math.cos( turn ), s = Math.sin( turn );
			const u = wrap( qx * c + qy * s, tile ), v = wrap( qy * c - qx * s, tile );
			shift.set( u * c - v * s, u * s + v * c );

		};
		fibres( fine.value, FIBRE_FINE );
		fibres( broad.value, FIBRE_BROAD );

	}

	dispose(): void {

		this.dome.dispose();
		this.domeScratch.dispose();
		this.shadowMap.dispose();
		for ( const t of [ this.shape, this.synoptic, this.weather, this.bounds, this.rainbowTable ] ) t?.dispose();
		this.streakCover?.dispose();
		for ( const kernel of [ this.panoramaKernel, ...this.rescaleKernels ?? [], this.shadowKernel ] ) kernel?.dispose();
		this.deckMeanKernel?.dispose();
		this.deckMeanSumKernel?.dispose();
		this.columnKernel?.dispose();
		this.disposeView();

	}

	disposeView(): void {

		const view = this.view;
		if ( ! view ) return;
		this.disposeViewTextures( view );
		for ( const kernel of [ view.kernels.trace, view.kernels.high, view.kernels.box, ...view.kernels.resolve ] ) kernel.dispose();
		this.view = null;
		this.viewValid.value = 0;

	}

	disposeViewTextures( view: ViewTextures ): void {

		for ( const t of [ view.trace, view.depth, view.high, view.motion, view.flash, view.boxMin, view.boxMax, ...view.history, ...view.depthHistory ] ) t.dispose();

	}

	// The view's size for a render of width x height pixels (see CloudViewScale).
	private viewDimensions( width: number, height: number ): [ number, number ] {

		const override = debugFlags.cloudViewScale;
		const scale = typeof override === 'number' ? { render: override } : this.viewScale;
		let w: number, h: number;
		if ( 'render' in scale ) {

			w = width * scale.render;
			h = height * scale.render;

		} else {

			// A share (at most 1) of the render high draws at its nominal ratio. The size follows the
			// display alone, not the render: the high tier's dynamic resolution moves the render's scale, and a
			// view that followed it would be rebuilt, with its kernels, at every step.
			const nominal = Math.min( this.display.ratio, 1.5 ) * scale.display;
			w = this.display.width * nominal;
			h = this.display.height * nominal;

		}
		return [ Math.max( 8, Math.round( w ) ), Math.max( 8, Math.round( h ) ) ];

	}

	/** The display the view's budget refers to (CSS pixels and the canvas's pixel ratio). */
	setDisplay( width: number, height: number, ratio: number ): void {

		this.display.width = width;
		this.display.height = height;
		this.display.ratio = ratio;

	}

	/** The view's resolution, as a fraction of the render or of the display (CloudViewScale). A
	 * new size rebuilds the view on the next update. */
	setViewScale( scale: CloudViewScale ): void {

		const value = 'render' in scale ? scale.render : scale.display;
		if ( ! ( value > 0 && value <= 1 ) ) throw new RangeError( 'The cloud view scale must lie in (0, 1]' );
		this.viewScale = { ...scale };

	}

	/** The panorama's size and refresh pattern: one texel of every strideX x strideY (powers of
	 * two dividing the size) per frame. A change rebuilds the panorama and refreshes it fully. */
	setDome( width: number, height: number, strideX: number, strideY: number ): void {

		const power = ( n: number ) => Number.isInteger( n ) && n > 0 && ( n & ( n - 1 ) ) === 0;
		if ( ! power( strideX ) || ! power( strideY ) || width % strideX !== 0 || height % strideY !== 0 ) throw new RangeError( 'Panorama strides must be powers of two dividing its size' );
		const d = this.domeSize;
		if ( d.width === width && d.height === height && d.strideX === strideX && d.strideY === strideY ) return;
		this.domeSize = { width, height, strideX, strideY };
		this.domeOrder = ditherOrder( strideX, strideY );
		if ( d.width !== width || d.height !== height ) {

			const previous = this.dome;
			this.dome = makeDome( width, height );
			this.domeNode.value = this.dome;
			this.domeTexel.value.set( 0.5 / width, 0.5 / height );
			previous.dispose();
			this.domeScratch.dispose();
			this.domeScratch = makeDome( width, height, 'cloudDomeRescale' );

		}
		if ( this.panoramaKernel ) buildKernels( this );
		this.invalidate();

	}

	/** The march's step caps and taps (CloudMarch); a change rebuilds the kernels once. */
	setMarch( march: Partial<CloudMarch> ): void {

		const next = { ...this.march, ...march };
		if ( ! ( next.lightTaps >= 1 && next.lightTaps <= MAX_LIGHT_TAPS ) ) throw new RangeError( `Light taps must lie in 1 .. ${ MAX_LIGHT_TAPS }` );
		for ( const key of [ 'viewSteps', 'panoramaSteps', 'rainSteps' ] as const ) if ( ! ( Number.isInteger( next[ key ] ) && next[ key ] > 0 ) ) throw new RangeError( `${ key } must be a positive integer` );
		if ( ( Object.keys( next ) as ( keyof CloudMarch )[] ).every( ( key ) => next[ key ] === this.march[ key ] ) ) return;
		this.march = next;
		if ( this.panoramaKernel ) buildKernels( this );
		this.invalidate();

	}

	/** The share of the cloud shadow map's rows refreshed per frame: 1/8 (the default), 1/4,
	 * 1/2 or 1. A jump still refreshes all of it at once. */
	setShadowRefresh( rowsPerFrame: number ): void {

		const passes = rowsPerFrame * SHADOW_PHASES;
		if ( ! ( Number.isInteger( passes ) && passes >= 1 && passes <= SHADOW_PHASES ) ) throw new RangeError( `Shadow rows per frame must be a multiple of 1/${ SHADOW_PHASES }` );
		this.shadowRowsPerFrame = rowsPerFrame;

	}

	/** Compile the panorama, shadow and view kernels ahead of their first use. The view's kernels
	 * are built here at the drawing buffer's size if no frame has built them yet: they keep across
	 * sizes (`buildView`). */
	async warm( renderer: THREE.WebGPURenderer ): Promise<void> {

		await this.init( renderer );
		if ( ! this.view ) {

			const size = renderer.getDrawingBufferSize( new THREE.Vector2() );
			const [ w, h ] = this.viewDimensions( size.x, size.y );
			buildView( this, w, h );

		}
		const view = this.view!;
		const kernels = [ this.columnKernel!, this.panoramaKernel!, ...this.rescaleKernels!, this.shadowKernel!, this.deckMeanKernel!, this.deckMeanSumKernel! ];
		kernels.push( view.kernels.trace, view.kernels.high, view.kernels.box, ...view.kernels.resolve );
		await renderer.compileComputeAsync( kernels );

	}

}

function makeDome( width: number, height: number, name = 'cloudDome' ): THREE.StorageTexture {

	const dome = new THREE.StorageTexture( width, height );
	dome.type = THREE.HalfFloatType;
	dome.wrapS = THREE.RepeatWrapping;
	dome.wrapT = THREE.ClampToEdgeWrapping;
	dome.minFilter = dome.magFilter = THREE.LinearFilter;
	dome.generateMipmaps = false;
	dome.name = name;
	return dome;

}

// The rainbow's colours by angle from the antisolar point (RainbowTable.data.ts), as a half-float
// strip, linearly filtered between its 0.125 degree entries.
function makeRainbowTexture(): THREE.DataTexture {

	const data = new Uint16Array( RAINBOW_SIZE * 4 );
	for ( let i = 0; i < RAINBOW_SIZE; i ++ ) {

		for ( let c = 0; c < 3; c ++ ) data[ i * 4 + c ] = THREE.DataUtils.toHalfFloat( RAINBOW_TABLE[ i * 3 + c ] );
		data[ i * 4 + 3 ] = THREE.DataUtils.toHalfFloat( 1 );

	}
	const table = new THREE.DataTexture( data, RAINBOW_SIZE, 1, THREE.RGBAFormat, THREE.HalfFloatType );
	table.minFilter = table.magFilter = THREE.LinearFilter;
	table.wrapS = table.wrapT = THREE.ClampToEdgeWrapping;
	table.generateMipmaps = false;
	table.name = 'rainbowTable';
	table.needsUpdate = true;
	return table;

}

