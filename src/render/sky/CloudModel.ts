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
	asin, atan, clamp, cos, dot, float, floor, Fn, fract, log2, max, mix, select, sin, sqrt, uint,
	uvec3, vec2, vec3, vec4,
} from 'three/tsl';
import { tune } from '../debug/Tune.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// The clouds' model: the shell's, density's, light's and march's constants and shared helpers.
//
// The clouds' model (Clouds.ts, whose header says what it is and why): the constants of the shell, the
// density, the light and the march, the dome's and the shadow map's layouts, and the helpers every
// kernel shares (the density's weather noise, the phase, the dome's mapping, the deck's forward
// level). A name in backticks with no file is this file's or Clouds.ts's.

export const EARTH_RADIUS = 6_360_000;
// The high shell's base and thickness (m), the clear and fair skies' cumulus. The shell the march
// reads is `shellBottom` and `shellThickness`, blended by the sky's shares (look/SkyLook.ts `cloudSkies`), so
// cloudy's and overcast's layers stand lower (`setShell`).
export const SHELL_BOTTOM = 4000, SHELL_THICKNESS = 5200;
// What the panorama stores is the radiance times a power of two that follows the clouds' light
// and the weather's skies (`storeStops`, set by SkyTrack.ts trackClouds: 2^11 at noon, 2^16 to 2^20 on
// a moonlit night, by how far the weather's clouds spread the night's range; PreExposure.ts
// `storeStops`, look/SkyLook.ts `storeRange`): unscaled, a clear midnight's panorama has its first
// percentile near 2^-21 (2^-16 over the texels at least a tenth covered), under half float's 2^-14, where its steps are wide enough to show as dark flecks
// once exposed, and no one fixed scale keeps a cloudy noon's brightest and a darker night's
// thinnest both in range. A power of two, so what lay in range keeps its bits. The panorama is
// refreshed a share per frame and kept for the frames between, so its scale is the light's and not
// the frame's pre-exposure (the view takes that: see the rules above). When the light's power of
// two changes (`domeStops`: twelve to seventeen times through a dusk, by the weather), the texels
// the panorama holds are taken to the new one by the ratio of the two in that frame, ahead of
// the frame's phase and of every reader (`rescaleKernels`: through a scratch of the panorama's
// size, since a half-float texture cannot be read and written in one dispatch): marching every
// phase again there doubled a cloudy frame's GPU time. A power of two scales a half float exactly
// where it lies in range, and the power keeps the texels a margin clear of the subnormals and of
// the ceiling (look/SkyLook.ts `storeRange`); a texel outside it carries its old rounding until its phase
// comes round. The progressive refresh already keeps texels written frames apart, so this adds
// no history of its own: once every phase has come round after a change, the panorama equals the
// whole rewrite's bit for bit (measured through a cloudy dusk), a jump writes it whole
// (`invalidate`), and a paused page settles it whole. Like the march, the rescale runs in
// `update`, ahead of the frame graph: it must fall between the texels it scales and this frame's
// phase. Its writes are held at STORE_CEILING. Lightning's light is stored apart and
// bare (its reach is far above this range's top).
// What the view stores over the pre-exposed radiance: by day P runs 1 to 3, and at that scale the
// faint edges of thin cloud low over the sea fell under half float's 2^-14 (a low view across
// the water lost a few dozen pixels a level or more, up to 54). Seven stops keep them at the 2^8
// the view held under a fixed scale, a night (P 5 to 9) at 2^12 or more, and the brightest a
// dusk shows (P 13) about three stops under STORE_CEILING.
export const VIEW_HEADROOM = 2 ** 7;
// The preset's scales, rounded so they nest: the wrapped wind offset (TILE) holds a whole
// number of periods of the weather map, the shape noise and the erosion noise.
export const WEATHER_SCALE = 28672, BASE_SCALE = 7168, EROSION_SCALE = BASE_SCALE / 8;
export const TILE = WEATHER_SCALE;
// The shape noise rises through the cloud layer (its evolution in place, `evolve`: metres) and wraps
// on its vertical period. It drifts with the weather map: shifted horizontally it slid through the
// weather on its own and, faster than the drift, carried the visible heaps upwind.
export const EVOLVE_PERIOD = BASE_SCALE;
export const WEATHER_RES = 1024, BOUNDS_RES = 64;
export const DENSITY = 0.019 / 0.9; // extinction (1/m) at a density of one; fair weather's 0.9 is the preset
export const BASE_STRENGTH = 0.69; // dilation of the tops by the shape noise
export const EDGE = 0.095; // soft margin, as a fraction of the shell
const EROSION_BASE = 0.24, EROSION_PEAK = 2.15;
export const BASE_WEATHER = 0.54, BASE_WEATHER_END = 0.13; // bases need richer weather
// The two ends of the cloud's type (`cloudType`: 0 a stratus, 1 a cumulus), which every type
// between mixes linearly, term by term (`formOf`): the soft margin of the tops and bases (a
// fraction of the shell), the erosion's strength at the base and at the top, how much richer
// weather a base needs than its body (a cumulus' bases stand apart; a stratus has one base), and
// the share of the key's light its column passes down by diffusion (whole under a layer, a part
// under a heap, whose light leaves by its sides). Set by eye on the whole frame.
export const STRATUS = { edge: 0.2, erosionBase: 0.6, erosionPeak: 1, baseWeather: 0, column: 1 };
export const CUMULUS = { edge: EDGE, erosionBase: EROSION_BASE, erosionPeak: EROSION_PEAK, baseWeather: BASE_WEATHER, column: 0.3 };
// A stratus' column (`topOf`): a slab STRATUS_CAP of the shell deep, risen over STRATUS_RISE of
// the weather's surplus at its margins and thinner by up to STRATUS_RELIEF where the weather is
// poor, so a closed layer keeps masses and seams.
// The coverage over which a heap's column passes as a layer's does (`column`): apart, a heap
// loses the light in it by its sides; joined in a closing sky, there are none to lose it by.
export const COLUMN_CLOSE = [ 0.4, 0.8 ] as const;
// The least depth of a shell (m).
export const SHELL_MIN = 100;
export const STRATUS_CAP = 0.6, STRATUS_RISE = 0.15, STRATUS_RELIEF = 0.92, STRATUS_SPAN = [ 0.45, 0.95 ] as const;
export const SKEW = 1750; // metres the tops lean downwind
// More cloud far out, from HORIZON_START over HORIZON_RAMP metres along the ray.
export const HORIZON_START = 20000, HORIZON_RAMP = 45000, HORIZON_AMOUNT = 0.12;
// The distances along the ray (m) over which the far fade thins the cloud (`farFade`): tidewater's
// melt of the preset's cumulus into the sky.
export const FAR_FADE = [ 25000, 45000 ] as const;
export const AMBIENT = 0.7 * ( 1 + 0.99 );
// The march's lighting terms (makeMarch), tidewater's Sky Pro values: the powder's weight looking
// away from the key; the shadowing of the bases, whole at the shell's floor and gone BASE_SHADOW_HEIGHT
// of the shell up; the horizon's share of the sky's light at the floor; the sea's albedo for the
// light it sends back up; and the two reads of the cloud straight above a sample that stand for the
// sky's way down to it, each a rise (m) and the stretch of column (m) it stands for.
export const POWDER = 0.7, BASE_SHADOW = 0.88, BASE_SHADOW_HEIGHT = 0.13, HORIZON_SHARE = 0.55;
export const SEA_BOUNCE = [ 0.0091, 0.0152, 0.0185 ] as const;
export const SKY_PROBES = [ [ 125, 250 ], [ 600, 700 ] ] as const;
export const HORIZON_RING = 16; // azimuths in the half ring of horizon sky a cloud's base sees (see horizonMean)
export const HORIZON_BAND = [ 0.05, 0.2, 0.4 ] as const; // the sines of the elevations that ring is read at
export const STEP = 25; // metres: the finest step and the first light segment
// The cloud's two phase lobes (see makeMarch, `phaseOrders`): tidewater's droplet phase's first
// order, two Henyey-Greenstein lobes, `forward` and `back` asymmetries, the back one's share
// `backShare` of the light at every order of the scattering. Tunable at load
// (`sky.clouds.lobeForward`, `lobeBack`, `lobeBackShare`).
export const LOBE = {
	forward: tune( 'sky.clouds.lobeForward', 0.8, { min: 0.3, max: 0.95, load: true } )(),
	back: tune( 'sky.clouds.lobeBack', - 0.2, { min: - 0.6, max: 0.2, load: true } )(),
	backShare: tune( 'sky.clouds.lobeBackShare', 0.2, { min: 0, max: 0.6, load: true } )(),
};
// The column's diffusion coefficient: a conservative slab passes 1 / ( 1 + COLUMN_DIFFUSION tau )
// of the light on it, the diffusion (Eddington) approximation's 0.75 ( 1 - g ) with the forward
// lobe's asymmetry for g, not the two lobes' mean: the layers' levels are set with it.
export const COLUMN_DIFFUSION = 0.75 * ( 1 - LOBE.forward );
// The light march's segments grow by this factor, the first sized so that they span the shell's
// depth together (updateLightTaps).
export const LIGHT_GROWTH = 2.25;
// The most light march taps a tier may ask for (see setMarch); the uniforms hold this many.
export const MAX_LIGHT_TAPS = 6, FULL_LIGHTING_ALPHA = 0.5, LIGHT_REUSE_STEPS = 3;
export const AP_DIST = 30000; // aerial perspective scale toward the horizon
export const FAR = 250000;
// The clouds drift with the wind aloft, derived from the surface wind's integrated displacement
// (setDrift), so they follow the weather's wind: the cumulus and the cirrus at LOW_DRIFT times
// the surface wind, turned LOW_VEER to its right (veering with height, as the wind aloft does north
// of the equator), the layer's fibres combed along that drift. Steering winds at 4 to 9 km
// run two to three times the 10 m wind; at 0.21 times it the cumulus all but stood still.
export const LOW_DRIFT = 2.25, LOW_VEER = 12 * Math.PI / 180;
export const SYNOPTIC_SIZE = 409600;
// The cirrus (`streaksAt`): its height (m, `sky.streaks.height`), the vertical optical depth
// of its densest fibres per unit of the weather's amount (`sky.streaks.peak`), the floor of the
// vertical cosine its slant path divides by (`sky.streaks.muFloor`), and the ambient light's share
// per unit of opacity (`sky.streaks.ambient`). Set on the clear and fair noon's horizon and sky
// views and on the low sun's horizon views, against the sky round the layer (`img.py stats --mask`
// over a `flags=streakMask` capture). From 7 to 10 km up the views drew alike. At a floor of 0.25
// a low layer's streaks closed to eight tenths opaque from about 12 degrees down and stood 0.17
// above the sky in Oklab L, white bars; at 0.8 a fibre seen low is at most a quarter more opaque
// than overhead (a fibre about as deep as it is wide, whose slant path the width bounds). At these
// values a third of the clear noon's sky on its horizon view lies under the layer past 0.02 opaque
// and 7 % past 0.1, its cores 0.019 above the sky round them in Oklab L (10 to 90 %: -0.007 to
// 0.052), white and a little brighter than the sky (the earlier streaks stood 0.054); overhead
// 14 % of the sky is past 0.1. On the low sun's horizon view the band from 1 to 5 degrees up stays
// bare (under 0.02 opaque in all but 0.2 % of it), and 16 % of the sky within 15 degrees of the sun
// is covered. The direct share sets most of its light (away from the sun, its back lobe).
export const STREAK_HEIGHT = 8500, STREAK_PEAK = 0.9, STREAK_MU_FLOOR = 0.8, STREAK_AMBIENT = 0.7 * AMBIENT;
// The radius (m) over which a point of the layer's lead toward the sun raises the sun's elevation
// there (`sky.streaks.litRadius`), and the width (degrees) of the fade as the setting disc sinks
// under the layer's horizon (`sky.streaks.litWidth`). At the Earth's radius the layer went out on
// the side away from the sun about half a degree of the sun's elevation before the side toward
// it, the whole layer between -1.7 and -2.2 degrees; at half of it the away side goes out at
// about -1.5 and the toward side at -2.9, at a third at -1.2 and -3.2. From the noon down to the
// sun on the horizon the layer reads as at the Earth's radius within 0.01 in Oklab L; with the
// sun 2 degrees under it, the side toward the sun, still lit, is 0.015 lighter.
export const STREAK_LIT = { radius: EARTH_RADIUS / 2, width: 2 };
// The patches (the synoptic field's values over which they rise from none to whole: the horizon
// views hold eight or nine of more than a square degree, blurred over half a degree, the largest
// about a sixth of the sky) and the fibres' two scales (metres a tile, turn in radians: Streaks.ts
// `sampleStreaks`).
export const STREAK_PATCHES = { low: 0.2, high: 0.8 };
export const FIBRE_FINE = { tile: 40000, turn: 0 }, FIBRE_BROAD = { tile: 92000, turn: 0.5 };
// Where there is no cirrus (cloudy and overcast skies), the view's history behind thin cumulus is
// reprojected from this height (m), above every shell's top: the sky, far beyond the clouds.
export const BEYOND_SHELLS = 10000;
// The panorama (see setDome): its size and refresh pattern, one texel of every strideX x strideY
// per frame. Its main consumer, the bake's 512 x 128 cloud layer, averages four taps of it, and
// the light shafts and the sky outside the view read it directly; a full refresh every 32 frames
// is far quicker than the clouds drift.
export const DOME_DEFAULTS = { width: 1024, height: 256, strideX: 4, strideY: 8 } as const;
export const SHADOW_SIZE = 256, SHADOW_SPAN = 12000;
// The cloud shadow map refreshes one of every SHADOW_PHASES rows per pass (see setShadowRefresh);
// cloud shadows drift across its 47 m texels far slower than a full refresh, but under a low sun,
// whose motion slides them a few texels in one.
export const SHADOW_PHASES = 8;
// The least slope of the rays the map is cast along and read back by: under it a ray would run
// farther through the shell than its steps can follow, about 4.6 degrees.
export const SHADOW_SLOPE_FLOOR = 0.08;
export const SHADOW_ORDER = [ 0, 4, 2, 6, 1, 5, 3, 7 ];
// The closed deck's mean transmittance along the light, which the map fades to under overcast's
// share as the light sets (`deckShadowShare`): rays from a world-anchored grid of DECK_MEAN_GRID
// squared points over the middle DECK_MEAN_SPAN metres of the map, the island's.
export const DECK_MEAN_GRID = 8, DECK_MEAN_SPAN = 6000, DECK_MEAN_RAYS = DECK_MEAN_GRID * DECK_MEAN_GRID;
// Lightning inside the cloud: point sources lighting the march.
export const FLASH_SOURCES = 4;
// Rain curtains: marched from RAIN_START (nearer rain is the streaks of weather/Rain.ts) to the
// cloud base or RAIN_FAR, in segments that lengthen with distance (CloudMarch.rainSteps).
export const RAIN_START = 700, RAIN_FAR = 42000;
export const RAIN_SIGMA = 0.0004; // extinction (1/m) of a full shower: about ten kilometres of visibility
// The curtains' pattern: lumps RAIN_LUMPS metres across and streaks RAIN_STREAK metres across
// (both dividing TILE, so the wrapped wind offset shows no seam),
// stretched along the fall (vertical period RAIN_PERIOD) and sliding down RAIN_FALL metres per unit
// of `evolve` (2.5 per second: about 8 m/s). The fall over one evolve period is a whole number of
// vertical periods, so the wrap of `evolve` never shows.
export const RAIN_STREAK = 512, RAIN_LUMPS = 2048, RAIN_FALL = 3.2;
export const RAIN_PERIOD = EVOLVE_PERIOD * RAIN_FALL / 6;
// Share of the rain's extinction the bows scatter: the table is per geometric cross-section,
// and large drops extinguish twice that (the other half is diffracted straight ahead). Doubled
// for the art: the stylised curtains are thinner than real rain, and a bow should still read.
export const BOW_SHARE = 0.5 * 2;
// The secondary bow is lifted further (to about a quarter of the primary instead of a seventh):
// in curtains this thin it would otherwise not read at all.
export const SECONDARY_BOW_GAIN = 1.8;
// Share of a strike's light the rain returns, against the cloud's: drops scatter less of it back.
export const RAIN_FLASH = 0.5;
// 4 x 4 ordered dither: one pixel of every block per frame.
export const ORDER = [ 0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12 ];
export const REBUILD_SLOTS = 4; // slots traced per frame right after a camera cut
export const SETTLE_CYCLES = 12; // samples per pixel a scene at rest averages before its view is kept

// The march's step caps and taps per tier (setMarch): today's high values by default. Each is a
// compile-time constant of the kernels, so a change rebuilds them once.
export interface CloudMarch {
	viewSteps: number; // the view's march
	panoramaSteps: number; // the panorama's march
	lightTaps: number; // taps of the light march toward the key light (1 to MAX_LIGHT_TAPS)
	rainSteps: number; // segments of the rain curtains' march
}
export const MARCH_DEFAULTS: CloudMarch = { viewSteps: 256, panoramaSteps: 128, lightTaps: 6, rainSteps: 16 };

// How the view's resolution is set: a fraction of the render, or of the display (the render high
// draws at its nominal ratio: the display's, up to 1.5), so a lower render scale does not soften
// the clouds a second time.
export type CloudViewScale = { render: number } | { display: number };

// The sampling offset of a layer drifting `rate` times the surface wind's displacement `d`, turned
// `veer` radians to its right (toward +z from +x): the clouds move by minus the offset.
export function driftOffset( d: THREE.Vector2, rate: number, veer: number ): THREE.Vector2 {

	return d.clone().rotateAround( new THREE.Vector2(), veer ).multiplyScalar( - rate );

}

// The panorama's refresh order over its sx x sy pattern (powers of two): consecutive frames land
// far apart, as an ordered dither does, so a partly refreshed map shows no sweep. A larger axis
// first alternates its coarsest offset alone; then 2 x 2 Bayer steps, coarse to fine. For 4 x 4
// this is the view's ORDER.
export function ditherOrder( sx: number, sy: number ): [ number, number ][] {

	const ax = Math.log2( sx ), ay = Math.log2( sy ), pairs = Math.min( ax, ay );
	const order: [ number, number ][] = [];
	for ( let n = 0; n < sx * sy; n ++ ) {

		let r = n, x = 0, y = 0;
		for ( let level = Math.max( ax, ay ) - 1; level >= pairs; level -- ) {

			if ( r & 1 ) {

				if ( ax > ay ) x += 1 << level;
				else y += 1 << level;

			}
			r >>= 1;

		}
		for ( let level = pairs - 1; level >= 0; level -- ) {

			const v = r & 3;
			r >>= 2;
			x += [ 0, 1, 1, 0 ][ v ] << level;
			y += [ 0, 1, 0, 1 ][ v ] << level;

		}
		order.push( [ x, y ] );

	}
	return order;

}

// The Sky Pro coverage for this project's weather coverage: fair weather (0.32) is the preset's
// partly cloudy sky (0.49), a cloudy sky (0.67) stays broken into big heaps, and overcast and
// rain (0.94 - 1) close the shell.
export const shellCoverage = ( c: F ) => select( c.lessThan( 0.32 ), c.mul( 0.49 / 0.32 ),
	select( c.lessThan( 0.8 ), c.sub( 0.32 ).mul( 0.17 / 0.48 ).add( 0.49 ), c.sub( 0.8 ).mul( 0.34 / 0.2 ).add( 0.66 ) ) );

// Upper-hemisphere map: longitude across, sqrt(latitude) up, denser near the horizon.
export const directionToDome = Fn( ( [ d ]: [ V3 ] ) => {

	const lat = asin( clamp( d.y, 0, 1 ) ).mul( 2 / Math.PI );
	return vec2( atan( d.z, d.x ).mul( 0.5 / Math.PI ).add( 0.5 ), sqrt( max( lat, 0 ) ) );

} ).setLayout( { name: 'cloudDirectionToDome', type: 'vec2', inputs: [ { name: 'd', type: 'vec3' } ] } );

export const domeToDirection = Fn( ( [ q ]: [ V2 ] ) => {

	const lat = q.y.mul( q.y ).mul( Math.PI / 2 ).toVar(), phi = q.x.sub( 0.5 ).mul( 2 * Math.PI ).toVar();
	return vec3( cos( lat ).mul( cos( phi ) ), sin( lat ), cos( lat ).mul( sin( phi ) ) );

} ).setLayout( { name: 'cloudDomeToDirection', type: 'vec3', inputs: [ { name: 'q', type: 'vec2' } ] } );

// Distance along a ray from height h (metres) with vertical cosine mu to the sphere `a` metres
// above sea level, for a viewer below it. Written without R^2 terms to keep float precision.
export const layerDistance = Fn( ( [ h, mu, a ]: [ F, F, F ] ) => {

	const rho = h.add( EARTH_RADIUS ).toVar();
	const b = rho.mul( mu ).toVar();
	const inside = a.sub( h ).mul( h.add( a ).add( 2 * EARTH_RADIUS ) ).toVar(); // r^2 - rho^2 > 0
	const root = sqrt( max( b.mul( b ).add( inside ), 0 ) ).toVar();
	return select( b.greaterThan( 0 ), inside.div( root.add( b ) ), root.sub( b ) );

} ).setLayout( { name: 'cloudLayerDistance', type: 'float', inputs: [
	{ name: 'h', type: 'float' }, { name: 'mu', type: 'float' }, { name: 'a', type: 'float' },
] } );

// Normalised Henyey-Greenstein phase.
export const henyeyGreenstein = ( cosTheta: F, g: number ) =>
	float( 1 - g * g ).div( max( float( 1 + g * g ).sub( cosTheta.mul( 2 * g ) ), 1e-3 ).pow( 1.5 ).mul( 4 * Math.PI ) );

// The angular form of the light under a low layer, as the air below it takes it (VolumetricHaze.ts
// `scatterToward`, by the shares `Clouds.ColumnTop` stores): a diffuse part, the escape function's
// diffusion law ( 3 / 7 )( 1 + 2 mu ), over the key's light and the sky's, and a forward part added
// over it on the key's light alone, a Henyey-Greenstein lobe toward the key whose radiance is
// FORWARD_WEIGHT times the key's irradiance over pi times the lobe over its mean over the sphere
// (about 45 at the key for a mean cosine of 0.8, 0.06 opposite it). Its weight and mean cosine are
// the form's own, not derived from the layer's depth (a forward share derived from it dies past an
// optical depth of 2). The march draws no such lobe on the layer itself: what a thin part passes
// toward the key is its beam's light (makeMarch).
export const FORWARD_WEIGHT = 0.6, FORWARD_LOBE = 0.8;
// The forward light under overcast's share, spread over the layer by the diffusion law: its
// amount on level ground is a lobe's of mean cosine OVERCAST_BROAD weighted OVERCAST_WEIGHT
// (`forwardLevel`). Multiple scattering makes the sky under a thick cloud evenly bright: a
// droplet's phase is narrow (mean cosine about 0.85), so after n scatterings the light keeps a mean
// cosine of about 0.85^n, and under a layer several optical depths thick the forward light has
// lost its direction. The two set the lobe's share of the air's light under a closed layer and the
// diffuse part's, not the layer's own level, which is the weather's (`columnLevel`).
export const OVERCAST_BROAD = 0.5, OVERCAST_WEIGHT = 2.4;
// The closed deck's glow about the key light, drawn where the sky is composed (SkyDome.ts
// `deckGlow`): two Henyey-Greenstein lobes, each of mean 1 over the sphere, times the deck's mean
// radiance: a narrow one (mean cosine DECK_GLOW_LOBE, weight DECK_GLOW) that makes the glow's peak
// and sets its half-power radius, and a broad one (DECK_GLOW_BROAD_LOBE, weight DECK_GLOW_BROAD)
// that makes its long tail, 35 - 90 degrees out. A single lobe could not do both: one narrow enough
// for the half-power radius left no tail past 35 degrees, and one weighted for the tail stood too
// high at the light's place. Raise the narrow
// weight for a brighter peak, its mean cosine for a tighter one; the broad weight for a brighter
// tail, which also lifts the deck far from the light (a weight of 1 is about 1.2 times the deck's
// own level there). The glow follows the deck's own pattern (DECK_GLOW_TEXTURE, SkyDome.ts
// `deckGlow`): it is multiplied by m^k, m the base's local radiance over the deck's mean (its
// diffuse law divided out), so it brightens the thin parts of the base and leaves its dark masses
// showing rather than laying a flat sheet of light over them. k is small: near the light m holds
// the thin parts' own forward light, and at k = 1 they stood out as hot spots twice the glow round
// them. Away from the key, where no thin part holds its forward light, the closed deck's glow by
// day follows the pattern more (`deckGlowFar`): k rises from DECK_GLOW_TEXTURE at DECK_GLOW_FROM
// degrees from the key to DECK_GLOW_FAR at DECK_GLOW_TO, so the glow, at half or more of the
// drawn deck's radiance there, no longer lays a flat sheet over the base's rolls; the moon's
// glow and the broken skies keep DECK_GLOW_TEXTURE everywhere (SkyDome.ts `glowToward`).
// At these values, overcast under a sun of 30 degrees, the drawn sky about the sun's place stands
// about two and a half stops over the sky on the far side. These are the day's: the glow about the moon keeps the
// form the night was set with (MOON_DECK_GLOW: mean cosine and weight of the narrow lobe, then of
// the broad one), the two passing over the civil twilight (Sky.ts `deckDay`, `glowToward`).
export const MOON_DECK_GLOW = [ 0.85, 0.45, 0.3, 0.48 ] as const;
export const DECK_GLOW = 0.22, DECK_GLOW_LOBE = 0.8, DECK_GLOW_BROAD = 0.1, DECK_GLOW_BROAD_LOBE = 0.4, DECK_GLOW_TEXTURE = 0.2;
export const DECK_GLOW_FAR = 1.2, DECK_GLOW_FROM = 30, DECK_GLOW_TO = 45;
// The closed deck's gradation under overcast's share by day (DECK_GRADATION, `deckGradation`: 1
// the diffusion law's 3 : 1, 0 uniform, between the CIE gradations' steep and uniform), and the
// glow the lighting's sources take about the sun, one smooth lobe's mean cosine and weight
// (BAKE_GLOW_LOBE, BAKE_GLOW_GAIN, SkyDome.ts `glowToward`). The dome draws the deck at the radiance
// the march gives, the one that lights the land and the sea: a deck drawn under it stood a stop
// and more under the meadow it lit, away from the sun.
export const DECK_GRADATION = 0.3, BAKE_GLOW_LOBE = 0.85, BAKE_GLOW_GAIN = 2;

/** The forward lobe's light on level ground: the integral over the upper hemisphere of a
 * Henyey-Greenstein lobe of mean cosine g about a key of height mu0 (its direction's y), each
 * direction weighted by its cosine to the zenith, times 4: the irradiance over pi that the lobe
 * over its mean over the sphere gives a level surface (the diffuse part's is 1), which the air
 * under the deck weighs the lobe's light by (VolumetricHaze.ts `scatterToward`). Exact as a
 * series: the lobe's Legendre moments are g^l and the clamped cosine's are known in closed form
 * (Ramamoorthi and Hanrahan, "An Efficient Representation for Irradiance Environment Maps",
 * SIGGRAPH 2001), so by the addition theorem the integral is sum_l a_l g^l P_l( mu0 ). 1 at
 * g = 0. */
export function forwardLevel( mu0: number, g: number ): number {

	const x = THREE.MathUtils.clamp( mu0, - 1, 1 );
	// a_0 = 1/4, a_1 = 1/2; odd l > 1 vanish; even l: ( 2l + 1 ) / 2 ( -1 )^( l/2 - 1 ) / ( ( l + 2 )( l - 1 ) )
	// times binomial( l, l/2 ) / 2^l
	let sum = 0.25 + 0.5 * g * x;
	let previous = 1, current = x, power = g, central = 1;
	for ( let l = 1; l < 160; l ++ ) {

		const next = ( ( 2 * l + 1 ) * x * current - l * previous ) / ( l + 1 );
		previous = current;
		current = next;
		power *= g;
		const n = l + 1;
		if ( n % 2 === 0 ) {

			central *= ( n - 1 ) / n;
			const a = ( 2 * n + 1 ) / 2 * ( ( n / 2 ) % 2 === 1 ? 1 : - 1 ) / ( ( n + 2 ) * ( n - 1 ) ) * central;
			sum += a * power * current;

		}

	}
	return 4 * Math.max( sum, 0 );

}
export const sat = ( x: F ) => clamp( x, 0, 1 );
export const luminance = ( c: V3 ) => dot( c, vec3( 0.2126, 0.7152, 0.0722 ) );
export const desaturate = ( c: V3, k: F ) => mix( c, vec3( luminance( c ) ), k );
export const over = ( a: V4, b: V4 ) => vec4( a.rgb.add( b.rgb.mul( a.a ) ), a.a.mul( b.a ) );
// Mip level of a noise of period `scale` for a footprint (m).
export const coneLod = ( footprint: F, scale: number ) => max( log2( max( footprint.mul( 64 / scale ), 1 ) ), 0 );

// The weather map: Sky Pro's fBm of hashed gradient noise, tileable.
export const WEATHER_MASS = [ 4, 5, 0, 1.32 ], WEATHER_DETAIL = [ 6, 6, 1, 0.13 ], WEATHER_COVERAGE = 0.26;
// The weather map's second channel: a large-scale field that moves the cloud type from place to
// place (`typeAt`). The fBm's lowest frequency, octaves and seed: 2 cells a tile (14.3 km cells,
// the field repeating with the weather map), one octave. One cell a tile is degenerate: the period-1 lattice
// gives its four corners one gradient. Its seed puts the lattice's z cells apart from the first
// channel's.
export const WEATHER_LARGE = [ 2, 1, 2 ];
const weatherGradient = ( cell: THREE.Node<'uvec3'>, offset: V3 ) => {

	const seed = ( v: THREE.Node<'uint'> ) => v.mul( uint( 1664525 ) ).add( uint( 1013904223 ) ).toVar();
	const x = seed( cell.x ), y = seed( cell.y ), z = seed( cell.z );
	const mixUp = () => {

		x.addAssign( y.mul( z ) ); y.addAssign( z.mul( x ) ); z.addAssign( x.mul( y ) );

	};
	mixUp();
	x.assign( x.bitXor( x.shiftRight( uint( 16 ) ) ) ); y.assign( y.bitXor( y.shiftRight( uint( 16 ) ) ) ); z.assign( z.bitXor( z.shiftRight( uint( 16 ) ) ) );
	mixUp();
	const h = x.shiftRight( uint( 24 ) ).bitAnd( uint( 15 ) ).toVar();
	const u = select( h.lessThan( uint( 8 ) ), offset.x, offset.y );
	const v = select( h.lessThan( uint( 4 ) ), offset.y, select( h.equal( uint( 12 ) ).or( h.equal( uint( 14 ) ) ), offset.x, offset.z ) );
	return select( h.bitAnd( uint( 1 ) ).equal( uint( 0 ) ), u, u.negate() ).add( select( h.bitAnd( uint( 2 ) ).equal( uint( 0 ) ), v, v.negate() ) );

};
const weatherPerlin = ( p: V3, period: number ) => {

	const cell = floor( p ).toVar(), f = fract( p ).toVar();
	const w = f.mul( f ).mul( f ).mul( f.mul( f.mul( 6 ).sub( 15 ) ).add( 10 ) ).toVar();
	const g = ( x: number, y: number, z: number ) => weatherGradient( uvec3( cell.add( vec3( x, y, z ) ).mod( period ) ), f.sub( vec3( x, y, z ) ) );
	return mix( mix( mix( g( 0, 0, 0 ), g( 1, 0, 0 ), w.x ), mix( g( 0, 1, 0 ), g( 1, 1, 0 ), w.x ), w.y ),
		mix( mix( g( 0, 0, 1 ), g( 1, 0, 1 ), w.x ), mix( g( 0, 1, 1 ), g( 1, 1, 1 ), w.x ), w.y ), w.z );

};
export const weatherFbm = ( q: V2, profile: number[] ) => {

	let frequency = Math.max( 1, Math.round( profile[ 0 ] ) ), weight = 0.5, weights = 0;
	let sum: F = float( 0 );
	for ( let i = 0; i < profile[ 1 ]; i ++ ) {

		sum = sum.add( weatherPerlin( vec3( q, profile[ 2 ] * 13.37 + 0.5 ).mul( frequency ), frequency ).mul( weight ) );
		weights += weight; weight *= 0.5; frequency *= 2;

	}
	return sum.div( weights ).mul( 0.5 ).add( 0.5 );

};

export interface ViewTextures {
	trace: THREE.StorageTexture; depth: THREE.StorageTexture; high: THREE.StorageTexture; motion: THREE.StorageTexture; flash: THREE.StorageTexture;
	boxMin: THREE.StorageTexture; boxMax: THREE.StorageTexture; history: THREE.StorageTexture[];
	// the cumulus's depth and the cirrus's opacity beside the history, accumulated with it
	// (`sampleViewFog`)
	depthHistory: THREE.StorageTexture[];
}
// a texture node of the view's kernels, and which of the view's textures it holds
export type ViewSlot = [ { value: THREE.Texture }, ( t: ViewTextures ) => THREE.StorageTexture ];
export interface View extends ViewTextures {
	width: number; height: number;
	kernels: { trace: THREE.ComputeNode; high: THREE.ComputeNode; box: THREE.ComputeNode; resolve: THREE.ComputeNode[] };
	slots: ViewSlot[];
}

// Bicubic Catmull-Rom in five bilinear taps (corners dropped).
export function catmullRom( node: THREE.TextureNode, uv: V2, size: V2 ): V4 {

	const sp = uv.mul( size );
	const tp1 = floor( sp.sub( 0.5 ) ).add( 0.5 ).toVar();
	const f = sp.sub( tp1 ).toVar();
	const w0 = f.mul( f.mul( f.mul( - 0.5 ).add( 1 ) ).sub( 0.5 ) );
	const w1 = f.mul( f ).mul( f.mul( 1.5 ).sub( 2.5 ) ).add( 1 );
	const w2 = f.mul( f.mul( f.mul( - 1.5 ).add( 2 ) ).add( 0.5 ) ).toVar();
	const w3 = f.mul( f ).mul( f.mul( 0.5 ).sub( 0.5 ) );
	const w12 = w1.add( w2 ).toVar();
	const tc0 = tp1.sub( 1 ).div( size ), tc3 = tp1.add( 2 ).div( size ), tc12 = tp1.add( w2.div( w12 ) ).div( size ).toVar();
	const a = w12.x.mul( w0.y ), b = w0.x.mul( w12.y ), cc = w12.x.mul( w12.y ), d = w3.x.mul( w12.y ), e = w12.x.mul( w3.y );
	const s = ( u: V2 ) => node.sample( u ).level( float( 0 ) );
	return s( vec2( tc12.x, tc0.y ) ).mul( a ).add( s( vec2( tc0.x, tc12.y ) ).mul( b ) ).add( s( tc12 ).mul( cc ) )
		.add( s( vec2( tc3.x, tc12.y ) ).mul( d ) ).add( s( vec2( tc12.x, tc3.y ) ).mul( e ) ).div( a.add( b ).add( cc ).add( d ).add( e ) );

}
