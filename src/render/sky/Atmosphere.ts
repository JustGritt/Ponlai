/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, uv, uniform, texture, vec2, vec3, vec4, float, int,
	sqrt, max, min, pow, exp, log, length, abs, acos, clamp, cos, sin, dot, normalize, mix, select, floor, step, sign,
} from 'three/tsl';
import { SCENE_RADIANCE_SCALE, SUN_ANGULAR_RADIUS, SUN_SETTING_RADIUS } from '../lighting/Radiometry.ts';
import { bandPush, bandToDisplay } from './BandMatrix.ts';
import { tuneApply, tuneUniform } from '../debug/Tune.ts';
import { markSkyRetune } from './SkyRetune.ts';

// The atmosphere's clear-sky scattering: transmittance and multiple-scattering LUTs in TSL.
//
// Hillaire 2020, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique":
// tidewater's transmittance and multiple-scattering LUTs, ported to TSL fragment passes, the
// latter's sun axis denser near the horizon (`multipleScattering`). Distances are kilometres. This
// module contains clear-sky scattering only; the visible sun, environment lighting and scene lights
// are composed by Sky from the same atmosphere.
//
// The planet: ground at 6360 km and the atmosphere's top 100 km above it (6460 km). The shell's
// thickness sets how fast the sky darkens toward the zenith and where the twilight wedge sits. The
// air over 60 km is thin (its Rayleigh share e^-7.5 of the sea's and less), but from the sun 8
// degrees under the horizon down it is the only air over the viewer still in sunlight (the earth's
// shadow over the zenith stands near 62 km at -8 degrees, 98 km at -10). With the top at 60 km the
// deep twilight's sky was the multiple scattering alone: 20 degrees up it stood 1.7 stops under
// this sky at -8, 4.1 at -10, 4.5 at -10.5 and 2.4 at -14 (.34 at -6, nothing by day), and the
// zenith fell by about 3 stops a degree down to -10.25, then by under 1 from -10.75, a kink; now
// it falls by 2.2 stops a degree at -8 and 1.5 at -11 to -12, smoothly. The march's steps hold at
// this height (twice the table's or the multiple-scattering bake's steps change the sky by .03
// stop or less). The in-scatter table's 21 r slices keep its lowest three near the heights over the
// sea that 16 gave a 60 km shell (0.25, 1.01 and 2.27 km, then 0.27, 1.07 and 2.41), where the
// clouds and the haze read it.
// The aerosol scatters through one Cornette-Shanks lobe of anisotropy MIE_G, nothing else: no
// isotropic base and no extra gain along long paths. The soft wide glow round the sun is that lobe
// seen through the haze in front of the sky (weather/VolumetricHaze.ts), whose forward lobe takes
// the same anisotropy.
// The aerosol's amount is `mieScale` (our clear sea air) times a density ρ_M (`aerosol`, a
// tunable): one value for all weathers, applied wherever the medium is (the bakes and the sun's
// transmittance), so a change rebakes the tables and the environment. The air aloft is clean, 1.0,
// and the sun's aureole is carried by the lobe's anisotropy, 0.76, and by the haze's aerosol share
// near the sea (HazeKeys.ts), not by the amount aloft. With seven times as much aloft (6.89 at an
// anisotropy of 0.63) the aerosol's extinction along the low sky's long rays darkened the sky's last
// few degrees under the sky above them: on the horizon views the band over the horizon stood about
// a quarter of the sky's lightness (Oklab L) under the sky 10 degrees up from sun -2 to 0, a dark
// ribbon, and 0.06 - 0.08 over it at noon. At 1.75 the ribbon was gone, the low sky brightening to
// the horizon, and the sun at the golden hour less red, but the glow over the horizon toward a low
// sun stood too far over the band away from it; the amount is tuned with the phases below. Over
// the sea the band is mostly the haze's light (VolumetricHaze.ts), lit by these rows: its weights
// set how it stands against the sky above.
//
// The sky's in-scatter is one table without azimuth or phase (Elek 2009, "Rendering
// parametrizable planetary atmospheres with multiple scattering in real-time"): single plus
// multiple scattering along a view ray, per unit of the sun's illuminance, over the sun's
// elevation (cosine muS), the ray's (mu) and the viewer's radius r, Rayleigh RGB (the multiple
// scattering, near isotropic, folded into it, times 4 pi: `tableBake`) and Mie RGB apart. Under the
// seen sky's polariser the folded term's mean over the directions is 0.86 of the isotropic term's
// (1 under the Rayleigh phase alone). The phase is applied where the
// table is read, so the Mie peak round the sun needs no texels and each reader takes its own
// phase; with the phase out, what is left depends only weakly on the azimuth between the view and
// the sun, so the table holds its mean over the azimuth. It is 89 (muS) x 96 (mu) x 21 (r), baked
// once per change of the medium (not when the sun moves: its elevation is an axis), as a 2D atlas
// of the r slices side by side, 32-bit floats (as every table here: `makeLUT`): RGBA Rayleigh RGB
// and Mie R, RG Mie GB. Its axes are
// non-linear (Bruneton and Neyret 2008, Yusov 2013): r by the square root of the height's share
// of the shell (the camera stays within about 200 m of the sea), mu split at the horizon of the
// radius it is read at, each half by the square root of its distance from the horizon (more
// texels near it), muS by Bruneton and Neyret's exponential, finer toward sunset (the day's
// sunsets are long), from the sun 18 degrees under the horizon (`muSUnit`). A ray under the
// horizon of its radius ends on the ground, its length in closed form (none from the ground
// itself). Each frame that the sun or the moon moves, two slices are
// cut from it for the sun's and for the moon's muS: a sky row at the viewer's height over mu (512
// texels, which a cubic B-spline makes smooth on a 4K sky) and a fog table over mu and altitude
// (128 x 16, the altitude spaced quadratically from 2 m up to `fogReach`, finer low down: the fog grid is
// coarser than either axis), which the haze reads per froxel (`fogRows`). The haze reads the fog
// table's sky half only: a view ray under its froxel's horizon is read as the horizon's sky-side
// ray at that radius. The ground half holds the in-scatter of the short way down to the sea, a
// few hundred metres, so read there the haze over the far sea came out about a fifth of the
// horizon sky and a dark band lay between the sea and the sky; read on the sky side, the far sea
// and land fade into the horizon sky's colour, while near the eye the haze is too thin for the
// change to show, and no ray above the horizon reads anything else. The slices are cut at a viewer height of 2 m (`viewHeight`), not the
// camera's: the management camera stands 22 - 190 m up, where the sky row differs from the 2 m one
// by little. Every read, of the table and of the slices, is a cubic B-spline built
// from bilinear taps (Sigg and Hadwiger, "Fast Third-Order Texture Filtering", GPU Gems 2, 2005),
// each tap held inside its half of mu (and its r slice in the atlas), so the ground-hitting and
// the sky-hitting halves never blend. `skyRadiance` is the sky row by the view's elevation times
// the phases and the earth's shadow, applied per pixel where it is read, with no other axis of the
// azimuth from the sun: the Rayleigh share by half Rayleigh's phase and half a linear forward lobe
// (`skyRayleighPhase`, k = SKY_LEAN), darkened by a polariser (`polariser`, p = SKY_POLARISER: the
// band 90 degrees from the sun that the sky's partial polarisation deepens, the aerosol and the
// multiple scattering depolarising the rest), the aerosol's by its Cornette-Shanks lobe (MIE_G);
// the light the sky sheds on the clouds is read without the polariser. The earth's shadow over the
// seen sky (`skyShadow`) is a factor of its own that only takes light away: on the side away from
// the light, by the cosine of the angle past 90 degrees from it (none at 90), falling off with the
// elevation over the shadow's edge (the plane through the viewer that holds the light's direction,
// square to the planet's radius toward it: the edge rises opposite a set sun and sinks under the
// horizon as the sun climbs), so it lies on the low sky opposite a low sun and is gone by day.
// The polariser, the lean, the aerosol aloft (`aerosol`) and the rising sun's haze aerosol weight
// (the haze look's mist key) are tuned together by least squares against the clear sky as the eye
// sees it (the haze over it, no clouds) at sun 0, 2, 12 and noon, with the noon sea's level held;
// the haze's weights at a low sun were tuned again later with these rows held beside the haze's own
// (HazeKeys.ts). The sky
// 130 degrees from the sun stands at .85-.91 of the sky 60 degrees from it 5 - 20 degrees up from
// sun 0 to 12, the sky opposite a sun at 0 at 1.29-1.43 times the sky 90 degrees from it 5 - 15
// degrees up (1.23 at 2; the haze before the dawn, its weight set by the band over the horizon,
// flattens the lowest of these from about 1.34), and on the horizon view at a rising sun of 2 -
// 2.7 degrees the band over the horizon 15 - 25 degrees from the sun 1.61-1.62 times (display,
// linear) the band 40 degrees from it, 2.5 before that tuning (polariser 0.35, lean 0.35, aerosol
// 1.75, the mist's weight 0.76). With the multiple scattering whole, the bare sky's
// first two relations moved by under 1 % and the third fell by 7 % with the aerosol's share; a
// second tuning of the lean, the polariser and the sky's shadow on the bare sky could not take it back
// (their rms over the relations 0.018 to 0.017 in log), so they stay, and the mist's weights
// were tuned again with the dawn's rows (HazeKeys.ts). With the polariser
// near full strength (0.835, as it was) the first two stood at 1.0-1.24 and about 7: the 90 degree
// band was a deep trench. A march per azimuth (the table's at each of 32 azimuths, the planet's
// shadow over each point's horizon) was
// tried in place of the rows and left the side away from a low sun too bright.
// The haze takes `skyShadow` too, read 17 degrees or more over the horizon (VolumetricHaze.ts).
// The clouds' shell reads the table at its own height, whose shadow is the ratio form
// (`earthShadow`): the view ray's lit share toward the view (the share of its length in
// the air outside the planet's shadow, a cylinder behind it along the light, its edge softened by
// the sun's angular radius) over that share's mean over the azimuth (`shadowMean`, a small table
// cut with the slices), 1 averaged over the azimuth and by day. The moon lights the sky the
// same way, with its own row, direction and illuminance (`moonIlluminance`), added to the sun's: a
// moonlit night's faint blue sky, brightest round the moon. Its Rayleigh rows and its aerosol's
// rows each take a gain (`moonRayleigh`, `moonMie`: `sky.night.moonRayleigh`,
// `sky.night.moonMie`, 1 the atmosphere's own; their ratio the night sky's structure and one
// level over both its light, MOON_SKY_LEVEL), applied where the rows are read
// (`bodyGains`, in `skyRows`, `fogRows` and `cloudRows`), so every reader follows: the seen sky,
// the environment's bakes and the light they lend, the haze's light and the far sea it fades
// into the horizon's colour, and the clouds' sky light and march, which keep agreeing with the
// sky behind them. The moon's in-scatter grows toward the horizon with the path's length, so the
// night sky away from the moon brightens low down (the haze reads the moon's rows along the moon,
// not the view, so it does not: weather/VolumetricHaze.ts); the Rayleigh share's gain under the
// aerosol's presses the dome down to a dark floor there, the aerosol's rows also feeding the glow
// round the moon (the haze's moon lobe).
// The level both moon row gains take over their ratio (`moonRayleigh`, `moonMie`). Its definition:
// the whole moonlit sky (the dome, the haze and the clouds) gives level ground the cosine-weighted
// irradiance (Sky.ts `skyIrradiance`, from the environment's harmonics) it gives with both gains at
// 1, on the clear midnight with the moon at its default height. It is a measured value, so it is
// measured again whenever what reads the moon's rows changes: the 2.4 stops measured first had come
// to give 0.58 stop over the gains at 1, and the land's sky light as much over the light the sun's
// sky lends it at the same height. 1.65 stops gives that irradiance within 0.1 %; the
// irradiance is linear in the level, its part outside the rows (the clouds' direct moonlight, the
// stars, the night's ambient) a constant.
const MOON_SKY_LEVEL = 2 ** 1.65;
const RG = 6360, RT = 6460;
/** The planet's radius and the atmosphere's top, km from the planet's centre. */
export const PLANET_RADIUS = RG, ATMOSPHERE_RADIUS = RT;
// the transmittance LUT; the multiple-scattering LUT, MS_SUN nodes over the sun's cosine c by MS_R
// over the height (linear), its sun axis u = 0.5 + 0.5 sign( c ) sqrt( |c| ), dense near the
// horizon (`multipleScattering`)
const T_W = 256, T_H = 64, MS_SUN = 192, MS_R = 32;
// the in-scatter table (muS, mu, r) and its atlas, the sky row, the fog table (mu, altitude); the
// azimuths the table averages over, and its march's steps
const TABLE_MUS = 89, TABLE_MU = 96, TABLE_R = 21, ATLAS_W = TABLE_MUS * TABLE_R;
const SKY_ROW = 512, FOG_MU = 128, FOG_H = 16;
// the fog table's lowest altitude (km): 2 m over the sea, as the sky row's viewer, since a ray from
// the ground itself has no length below the horizon
const FOG_LOW = 0.002;
const TABLE_AZIMUTHS = 8, TABLE_STEPS = 32;
// the earth's shadow's lit share averaged over the azimuth (`shadowMean`): 256 (mu) x 32
// (altitude) per body, each texel the mean of 16 azimuths over half a turn, the altitude by the
// fourth power of its axis from 2 m up to `fogReach` (a grazing ray's share near the sea changes
// fastest with it); and the floor added to both sides of the shadow's ratio (`earthShadow`)
const SHADOW_MU = 256, SHADOW_H = 32, SHADOW_AZIMUTHS = 16, SHADOW_FLOOR = 0.15;
const H = Math.sqrt( RT * RT - RG * RG );
const RAYLEIGH = [ 5.802e-3, 13.558e-3, 33.1e-3 ] as const;
// The ozone's absorption per band (per km at the tent's peak) is integrated over each display
// channel's band, not sampled at one wavelength per band as the Rayleigh coefficients are. Its
// Chappuis band is broad and peaks near 600 nm, between the red and the green samples, so point
// samples near 680, 550 and 440 nm (the values it had, 0.65, 1.88, 0.085 e-3) gave the red about a
// quarter of the absorption its channel takes over its band, and green more than its own: the
// twilight sky, lit through hundreds of kilometres of the ozone layer, kept its red and lost its
// green, a magenta red from 10 degrees up with the sun 5.6 under the horizon (red over green about
// 2.1, blue over green 1.5 at 20 degrees). Each channel's coefficient here is the cross section
// averaged over the channel's linear Rec. 709 response (from the CIE 1931 2-degree colour matching
// functions, its negative lobes included, the linearisation of a display channel's absorption)
// weighted by a 5778 K black body, at 223 K, for a column of 300 Dobson units over the tent's 15 km
// equivalent width: 2.59, 1.66 and -0.08 e-3, the blue's slight negative taken as none (our
// computation; the cross sections are Serdyuchenko, Gorshelev et al. 2014, "High spectral
// resolution ozone absorption cross-sections - Part 2: Temperature dependence", Atmospheric
// Measurement Techniques 7). Over grazing columns of 10 - 60 vertical ones the effective red rises
// to 2.6 - 2.85 and the green falls to 1.65 - 1.58; a black body of 6504 K and a temperature of
// 293 K move them less. The column (OZONE_COLUMN, `ozoneScale`, in units of 300 DU) sets the
// twilight sky's hue by elevation, here under the twilight's near-neutral warm sky light (Sky.ts
// `look.skyLight`, `dusk`): at a column of 0.6 the bare sky with the sun 5.6 under the horizon goes
// from orange at the glow (Oklab hue 59 at 1 degree) through yellow to near grey from 20 degrees
// up (chroma under 0.007 to the zenith); at 1 it turned blue from 12 degrees up (chroma 0.031 at
// 20 degrees), at 0.5 a yellow grey (0.010). The day's sky hardly moves (red over green 0.02 - 0.04
// lower at noon); the low sun's transmittance loses red (red over green 3.06 to 2.34 at
// 2.9 degrees, 1.45 to 1.29 at 12), and the golden hour's bare sky away from the sun turns from
// orange toward yellow.
const OZONE = [ 2.59e-3, 1.66e-3, 0 ] as const;
const OZONE_COLUMN = 0.6;
const _traced = new THREE.Vector3();

export const SUN_ILLUMINANCE = 11;
/** The aerosol's Cornette-Shanks anisotropy, the sky's and the haze's alike. */
export const MIE_G = 0.76;
/** The sky's Rayleigh phase's forward lean (`skyRayleighPhase`'s k), the sky's and the haze's alike. */
export const SKY_LEAN = 0.32;
/** The polariser's strength on the seen sky's Rayleigh share (`polariser`'s p). */
export const SKY_POLARISER = 0.26;
/** The earth's shadow over the seen sky (`skyShadow`): the share it takes at most, straight away
 * from the light on its edge, and the angle (degrees) over the edge in which it falls by e. */
const SKY_SHADOW_DEPTH = 0.55, SKY_SHADOW_SOFTNESS = 3.2;
export { SUN_ANGULAR_RADIUS } from '../lighting/Radiometry.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// -1 means no positive intersection. The same helper also detects planet occlusion of sunlight.
const raySphereNearest = Fn( ( [ ro, rd, radius ]: [ V3, V3, F ] ) => {

	const b = dot( ro, rd ).toVar();
	// ( |ro| - radius ) ( |ro| + radius ): |ro|^2 - radius^2 loses the difference to rounding for a
	// point near the sphere (a planet's radius squared)
	const r = length( ro ).toVar();
	const c = r.sub( radius ).mul( r.add( radius ) ).toVar();
	const disc = b.mul( b ).sub( c ).toVar();
	const sq = sqrt( max( disc, 0 ) );
	const t0 = b.negate().sub( sq ).toVar(), t1 = b.negate().add( sq ).toVar();
	return select( disc.lessThan( 0 ), - 1,
		select( t0.greaterThan( 0 ), t0, select( t1.greaterThan( 0 ), t1, - 1 ) ) );

} ).setLayout( { name: 'atmoRaySphereNearest', type: 'float', inputs: [
	{ name: 'ro', type: 'vec3' }, { name: 'rd', type: 'vec3' }, { name: 'radius', type: 'float' },
] } );

// A march step's in-scatter weight per unit of its length, ( 1 - exp( -x ) ) / x for the step's
// optical depth x per channel, by its series under x = 0.01: in 32-bit floats 1 - exp( -x ) keeps
// no digits under x of about 1e-7, a step's depth in the air some 80 km up. (The bakes divided
// 1 - exp( -x ) by the extinction floored at 1e-6 per km, which lost most of the light that the air
// above 69 km scatters in red, 76 in green and 83 in blue.)
const stepShare = Fn( ( [ x ]: [ V3 ] ) => {

	const series = vec3( 1 ).sub( x.mul( 0.5 ) ).add( x.mul( x ).div( 6 ) );
	const exact = vec3( 1 ).sub( exp( x.negate() ) ).div( max( x, vec3( 1e-30 ) ) );
	return mix( series, exact, step( vec3( 0.01 ), x ) );

} ).setLayout( { name: 'atmoStepShare', type: 'vec3', inputs: [ { name: 'x', type: 'vec3' } ] } );

// Only pure parameters enter a shared layout function. A closed-over UniformNode or TextureNode
// would retain the first material's generated binding names when Three reuses the function.

// The table's muS axis, Bruneton and Neyret 2008's exponential 1 - exp( -3 muS - a ), finer low
// down, 1 at the zenith and 0 at the sun 18 degrees under the horizon (MUS_LOW, the night's start),
// where theirs, a = 0.6, ends at muS = -0.2 (11.5 degrees under): below its end a table's sky stops
// falling with the sun and the deep twilight hangs at its last level. The offset only shifts the
// exponential, so on TABLE_MUS = 89 texels the axis keeps the texels per unit of muS that 64 gave
// theirs, within 0.07 % everywhere.
const MUS_LOW = Math.sin( - 18 * Math.PI / 180 ), MUS_A = - 3 * MUS_LOW;
const muSUnit = Fn( ( [ muS ]: [ F ] ) => {

	return clamp( float( 1 ).sub( exp( muS.mul( - 3 ).sub( MUS_A ) ) ).div( 1 - Math.exp( - 3 - MUS_A ) ), 0, 1 );

} ).setLayout( { name: 'atmoMuSUnit', type: 'float', inputs: [ { name: 'muS', type: 'float' } ] } );

const unitMuS = Fn( ( [ u ]: [ F ] ) => {

	return log( float( 1 ).sub( u.mul( 1 - Math.exp( - 3 - MUS_A ) ) ) ).negate().sub( MUS_A ).div( 3 );

} ).setLayout( { name: 'atmoUnitMuS', type: 'float', inputs: [ { name: 'u', type: 'float' } ] } );

// The mu axis of `n` texels at radius r: the texel coordinate (centres at i + 0.5) of mu, the lower
// half below the horizon (its last texel just under it), the upper above (its first just over it),
// each by the square root of the distance from the horizon; and the coordinates its half spans.
const muTexel = Fn( ( [ mu, r, n ]: [ F, F, F ] ) => {

	const horizon = sqrt( max( float( 1 ).sub( float( RG * RG ).div( r.mul( r ) ) ), 0 ) ).negate().toVar();
	const half = n.mul( 0.5 ).toVar();
	const up = sqrt( max( mu.sub( horizon ), 0 ).div( float( 1 ).sub( horizon ) ) );
	const down = sqrt( max( horizon.sub( mu ), 0 ).div( horizon.add( 1 ) ) );
	const above = mu.greaterThan( horizon );
	return vec3(
		select( above, half.add( 0.5 ).add( up.mul( half.sub( 1 ) ) ), half.sub( 0.5 ).sub( down.mul( half.sub( 1 ) ) ) ),
		select( above, half.add( 0.5 ), float( 0.5 ) ),
		select( above, n.sub( 0.5 ), half.sub( 0.5 ) ) );

} ).setLayout( { name: 'atmoMuTexel', type: 'vec3', inputs: [
	{ name: 'mu', type: 'float' }, { name: 'r', type: 'float' }, { name: 'n', type: 'float' },
] } );

// muTexel's inverse at a texel centre t; a ray never lies exactly on the horizon.
const texelMu = Fn( ( [ t, r, n ]: [ F, F, F ] ) => {

	const horizon = sqrt( max( float( 1 ).sub( float( RG * RG ).div( r.mul( r ) ) ), 0 ) ).negate().toVar();
	const half = n.mul( 0.5 ).toVar();
	const j = t.sub( 0.5 ).toVar();
	const up = j.sub( half ).div( half.sub( 1 ) ), down = half.sub( 1 ).sub( j ).div( half.sub( 1 ) );
	return select( j.greaterThanEqual( half ),
		horizon.add( max( up.mul( up ), 1e-6 ).mul( float( 1 ).sub( horizon ) ) ),
		horizon.sub( max( down.mul( down ), 1e-6 ).mul( horizon.add( 1 ) ) ) );

} ).setLayout( { name: 'atmoTexelMu', type: 'float', inputs: [
	{ name: 't', type: 'float' }, { name: 'r', type: 'float' }, { name: 'n', type: 'float' },
] } );

// A cubic B-spline along one axis as two linear taps (Sigg and Hadwiger 2005): at texel coordinate
// c (centres at i + 0.5), the taps' coordinates and the first's weight.
const bsplineTaps = Fn( ( [ c ]: [ F ] ) => {

	const x = c.sub( 0.5 ).toVar();
	const i = floor( x ).toVar();
	const f = x.sub( i ).toVar();
	const f2 = f.mul( f ), f3 = f2.mul( f );
	const w0 = f3.negate().add( f2.mul( 3 ) ).sub( f.mul( 3 ) ).add( 1 ).div( 6 );
	const w1 = f3.mul( 3 ).sub( f2.mul( 6 ) ).add( 4 ).div( 6 );
	const w3 = f3.div( 6 );
	const g0 = w0.add( w1 ).toVar();
	const g1 = float( 1 ).sub( g0 );
	return vec3( i.sub( 0.5 ).add( w1.div( g0 ) ), i.add( 1.5 ).add( w3.div( max( g1, 1e-6 ) ) ), g0 );

} ).setLayout( { name: 'atmoBsplineTaps', type: 'vec3', inputs: [ { name: 'c', type: 'float' } ] } );

// The cubic B-spline's four weights at fraction f.
const bsplineWeights = Fn( ( [ f ]: [ F ] ) => {

	const f2 = f.mul( f ), f3 = f2.mul( f );
	return vec4( f3.negate().add( f2.mul( 3 ) ).sub( f.mul( 3 ) ).add( 1 ).div( 6 ), f3.mul( 3 ).sub( f2.mul( 6 ) ).add( 4 ).div( 6 ),
		f3.mul( - 3 ).add( f2.mul( 3 ) ).add( f.mul( 3 ) ).add( 1 ).div( 6 ), f3.div( 6 ) );

} ).setLayout( { name: 'atmoBsplineWeights', type: 'vec4', inputs: [ { name: 'f', type: 'float' } ] } );

/** Rayleigh's phase function (1/sr). */
export const rayleighPhase = Fn( ( [ nu ]: [ F ] ) => {

	return nu.mul( nu ).add( 1 ).mul( 3 / ( 16 * Math.PI ) );

} ).setLayout( { name: 'atmoRayleighPhase', type: 'float', inputs: [ { name: 'nu', type: 'float' } ] } );

/** The sky's Rayleigh phase (1/sr): half Rayleigh's and half a linear forward lobe
 * (1 + k nu) / (4 pi), whose k leans the share round the sun brighter than the share away from it.
 * The multiple scattering folded into the table's Rayleigh part takes it too (baked times 4 pi,
 * this phase's mean being 1 / ( 4 pi )). */
export const skyRayleighPhase = Fn( ( [ nu, k ]: [ F, F ] ) => {

	return nu.mul( nu ).add( 1 ).mul( 3 / ( 32 * Math.PI ) ).add( k.mul( nu ).add( 1 ).mul( 1 / ( 8 * Math.PI ) ) );

} ).setLayout( { name: 'atmoSkyRayleighPhase', type: 'float', inputs: [ { name: 'nu', type: 'float' }, { name: 'k', type: 'float' } ] } );

/** A polariser's darkening of the sky's Rayleigh share: 1 - p (1 - nu^2) / (1 + nu^2), deepest at
 * 90 degrees from the light, where single Rayleigh scattering is polarised the most, and none
 * toward and away from it. */
export const polariser = Fn( ( [ nu, p ]: [ F, F ] ) => {

	const nu2 = nu.mul( nu ).toVar();
	return float( 1 ).sub( p.mul( float( 1 ).sub( nu2 ) ).div( nu2.add( 1 ) ) );

} ).setLayout( { name: 'atmoPolariser', type: 'float', inputs: [ { name: 'nu', type: 'float' }, { name: 'p', type: 'float' } ] } );

/** The earth's shadow over the seen sky toward `dir` for a light toward `light` (the module's
 * header): 1 - depth max( 0, -nu ) exp( -e / softness ), nu the cosine between the two and e the
 * sine of `dir`'s elevation over the shadow's edge (the plane that holds the light's direction,
 * square to the vertical's part across it), held at 0 under the edge; `softness` a sine too. */
export const skyShadow = Fn( ( [ dir, light, depth, softness ]: [ V3, V3, F, F ] ) => {

	const across = vec3( 0, 1, 0 ).sub( light.mul( light.y ) ).toVar();
	const edge = across.div( max( length( across ), 1e-6 ) );
	const over = max( dot( dir, edge ), 0 );
	const away = max( dot( dir, light ).negate(), 0 );
	return float( 1 ).sub( depth.mul( away ).mul( exp( over.div( softness ).negate() ) ) );

} ).setLayout( { name: 'atmoSkyShadow', type: 'float', inputs: [
	{ name: 'dir', type: 'vec3' }, { name: 'light', type: 'vec3' }, { name: 'depth', type: 'float' }, { name: 'softness', type: 'float' },
] } );

/** The length of a view ray from height `h` (km above the ground) with elevation sine `mu`, out to
 * `T` km, that lies in the planet's shadow for a light whose elevation sine is `muL` and whose
 * cosine with the ray is `dL` (litShare). The shadow is the cylinder of the planet's radius
 * about the axis through its centre along the light, on the far side of the plane through the
 * centre square to the light. A point p + t d of the ray lies in the cylinder where
 * | p + t d |^2 - ( ( p + t d ) . L )^2 < RG^2, that is A t^2 + 2 B t + C < 0 with A = 1 - dL^2,
 * B = r ( mu - muL dL ) and C = r^2 ( 1 - muL^2 ) - RG^2 ( r = RG + h, the viewer straight up the
 * axis of its own frame, and r^2 - RG^2 written as h ( r + RG ) to keep its digits); and beyond the
 * plane where r muL + t dL < 0, t past or short of -r muL / dL by the sign of dL. */
const shadowChord = Fn( ( [ h, mu, muL, dL, T ]: [ F, F, F, F, F ] ) => {

	const r = h.add( RG ).toVar();
	const A = max( float( 1 ).sub( dL.mul( dL ) ), 1e-6 ).toVar();
	const B = r.mul( mu.sub( muL.mul( dL ) ) ).toVar();
	const rL = r.mul( muL ).toVar();
	const C = h.mul( r.add( RG ) ).sub( rL.mul( rL ) ).toVar();
	const disc = B.mul( B ).sub( A.mul( C ) ).toVar();
	const root = sqrt( max( disc, 0 ) ).toVar();
	const lo = max( B.negate().sub( root ).div( A ), 0 ).toVar();
	const hi = min( B.negate().add( root ).div( A ), T ).toVar();
	const toward = step( 0, dL ).toVar();
	const plane = rL.negate().div( mix( min( dL, - 1e-6 ), max( dL, 1e-6 ), toward ) ).toVar();
	const before = min( hi, plane ).sub( lo ).toVar(), past = hi.sub( max( lo, plane ) ).toVar();
	return max( mix( past, before, toward ), 0 ).mul( step( 0, disc ) );

} ).setLayout( { name: 'atmoShadowChord', type: 'float', inputs: [
	{ name: 'h', type: 'float' }, { name: 'mu', type: 'float' }, { name: 'muL', type: 'float' },
	{ name: 'dL', type: 'float' }, { name: 'T', type: 'float' },
] } );

/** The sun's disc in four strips stacked along its elevation, [ offset, share ]: a disc of radius 1
 * cut at heights -1, -0.5, 0, 0.5 and 1, each strip's centroid height and its share of the disc's
 * area (the outer strips 0.196, the inner 0.304). */
const SUN_STRIPS: ReadonlyArray<readonly [ number, number ]> = [ [ - 0.705, 0.1955 ], [ - 0.244, 0.3045 ], [ 0.244, 0.3045 ], [ 0.705, 0.1955 ] ];

/** The lit share of the air seen from `h` km above the ground along `dir`, lit from `light`: 1
 * less the share of the view ray's length in the atmosphere (to its top, or to the ground for a
 * ray that meets it) that lies in the planet's shadow (`shadowChord`). The penumbra comes from the
 * sun's angular radius: the shadow is taken for each of four strips of the sun's disc
 * (SUN_STRIPS), the light turned up or down by the strip's offset times SUN_ANGULAR_RADIUS in its
 * vertical plane, the way the disc's edge sinks under the horizon, and the strips' shares added.
 * By day no part of a ray reaches the shadow, and the share is 1. */
const litShare = Fn( ( [ h, dir, light ]: [ F, V3, V3 ] ) => {
	const r = h.add( RG ).toVar();
	const mu = dir.y.toVar();
	const b = r.mul( mu ).toVar();
	const top = b.negate().add( sqrt( max( b.mul( b ).add( float( RT ).sub( r ).mul( r.add( RT ) ) ), 0 ) ) ).toVar();
	// the ground, where the ray meets it: ( r^2 - RG^2 ) / ( -b + sqrt( b^2 - r^2 + RG^2 ) ), the near
	// root written without the difference of two near numbers
	const lift = h.mul( r.add( RG ) ).toVar();
	const graze = b.mul( b ).sub( lift ).toVar();
	const ground = lift.div( max( b.negate().add( sqrt( max( graze, 0 ) ) ), 1e-9 ) ).toVar();
	const meets = step( 0, graze ).mul( step( b, 0 ) ).toVar();
	const T = max( mix( top, min( top, ground ), meets ), 1e-6 ).toVar();
	const raise = vec3( 0, 1, 0 ).sub( light.mul( light.y ) ).toVar();
	const lifted = raise.div( max( length( raise ), 1e-6 ) ).mul( SUN_ANGULAR_RADIUS ).toVar();
	const shaded = float( 0 ).toVar();
	for ( const [ offset, share ] of SUN_STRIPS ) {

		const strip = normalize( light.add( lifted.mul( offset ) ) );
		shaded.addAssign( shadowChord( h, mu, strip.y, dot( dir, strip ), T ).mul( share ) );

	}
	return clamp( float( 1 ).sub( shaded.div( T ) ), 0, 1 );

} ).setLayout( { name: 'atmoLitShare', type: 'float', inputs: [
	{ name: 'h', type: 'float' }, { name: 'dir', type: 'vec3' }, { name: 'light', type: 'vec3' },
] } );

/** Schuler's approximation of the Chapman grazing-incidence function (C. Schuler, "An
 * Approximation to the Chapman Grazing-Incidence Function for Atmospheric Scattering", GPU Pro 3,
 * 2012): the air mass of an exponential layer of scale height H along a ray of zenith cosine
 * `cosChi` from height h, with X = R / H the planet's radius and h the height in scale heights. */
const chapman = Fn( ( [ X, h, cosChi ]: [ F, F, F ] ) => {

	const c = sqrt( X.add( h ) ).toVar();
	const above = c.div( c.mul( cosChi ).add( 1 ) ).mul( exp( h.negate() ) ).toVar();
	const x0 = sqrt( max( float( 1 ).sub( cosChi.mul( cosChi ) ), 0 ) ).mul( X.add( h ) ).toVar();
	const below = sqrt( x0 ).mul( 2 ).mul( exp( clamp( X.sub( x0 ), - 80, 80 ) ) )
		.sub( c.div( max( float( 1 ).sub( c.mul( cosChi ) ), 1e-4 ) ).mul( exp( h.negate() ) ) ).toVar();
	// (never below 0: the grazing branch is a difference that float error can take under it)
	return max( select( cosChi.greaterThanEqual( 0 ), above, below ), 0 );

} ).setLayout( { name: 'atmoChapman', type: 'float', inputs: [
	{ name: 'X', type: 'float' }, { name: 'h', type: 'float' }, { name: 'cosChi', type: 'float' },
] } );

/** The Cornette-Shanks phase function of anisotropy g (1/sr). */
export const cornetteShanks = Fn( ( [ nu, g ]: [ F, F ] ) => {

	const g2 = g.mul( g ).toVar();
	return float( 1 ).sub( g2 ).mul( nu.mul( nu ).add( 1 ) ).mul( 3 / ( 8 * Math.PI ) )
		.div( g2.add( 2 ).mul( pow( max( g2.add( 1 ).sub( g.mul( nu ).mul( 2 ) ), 1e-4 ), 1.5 ) ) );

} ).setLayout( { name: 'atmoCornetteShanks', type: 'float', inputs: [ { name: 'nu', type: 'float' }, { name: 'g', type: 'float' } ] } );

const transmittanceUV = Fn( ( [ r, mu ]: [ F, F ] ) => {

	const rho = sqrt( max( r.mul( r ).sub( RG * RG ), 0 ) ).toVar();
	const disc = r.mul( r ).mul( mu.mul( mu ).sub( 1 ) ).add( RT * RT );
	const d = max( r.mul( mu ).negate().add( sqrt( max( disc, 0 ) ) ), 0 );
	const dMin = float( RT ).sub( r ), dMax = rho.add( H );
	const xMu = d.sub( dMin ).div( max( dMax.sub( dMin ), 1e-6 ) );
	// Match the texel-centre parameterization used to construct the LUT.
	return vec2( xMu.mul( ( T_W - 1 ) / T_W ).add( 0.5 / T_W ),
		rho.div( H ).mul( ( T_H - 1 ) / T_H ).add( 0.5 / T_H ) );

} ).setLayout( { name: 'atmoTransmittanceUV', type: 'vec2', inputs: [
	{ name: 'r', type: 'float' }, { name: 'mu', type: 'float' },
] } );

// The tables are 32-bit floats, filtered (WebGPU's float32-filterable, which three requests when
// the adapter has it). Per unit of the sun's illuminance the twilight's in-scatter spans some 30
// stops from noon to the sun 18 degrees down, more than half floats hold: in them (steps of
// 6e-8 under 6e-5) the sky 20 degrees up and at the zenith read 0 from the sun at -10 degrees,
// and at -9 its channels stood up to a quarter off, while at noon and -6 the same sky agreed with
// a double-precision march of this model within 1 %.
function makeLUT( width: number, height: number, name: string, format: THREE.PixelFormat = THREE.RGBAFormat ): THREE.RenderTarget {

	const target = new THREE.RenderTarget( width, height, {
		type: THREE.FloatType, format, depthBuffer: false, generateMipmaps: false,
		minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
		wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping,
	} );
	target.texture.name = name;
	return target;

}

function makeQuad( fragmentNode: THREE.Node ): THREE.QuadMesh {

	const material = new THREE.NodeMaterial();
	material.fragmentNode = fragmentNode;
	material.depthTest = false;
	material.depthWrite = false;
	material.toneMapped = false;
	return new THREE.QuadMesh( material );

}

export class Atmosphere {

	readonly sunDir = uniform( new THREE.Vector3( 0.3, 0.6, - 0.7 ).normalize() );
	readonly sunIlluminance = uniform( new THREE.Color().setScalar( SUN_ILLUMINANCE * SCENE_RADIANCE_SCALE ) );
	readonly rayleighScale = uniform( 1 );
	// Aerosol: two and a half times Hillaire's clear air. A daytime sea sky is hazy: the side toward
	// the sun takes a soft cream glow, where the clear air left a cyan horizon with no direction to
	// its light.
	readonly mieScale = uniform( 2.5 );
	/** the aerosol's density ρ_M over `mieScale`: it scales the Mie scattering and extinction alike
	 * (the absorption's share kept) in the bakes and the sun's transmittance, never as a gain on the
	 * Mie row where it is read, which would leave out the extinction and over-brighten the horizon at
	 * large scales */
	readonly aerosol = tuneUniform( 'sky.atmosphere.aerosol', 1.0, { min: 0.5, max: 40, scale: 'log', onChange: markSkyRetune } );
	private readonly mieDensity = this.mieScale.mul( this.aerosol );
	readonly mieG = tuneUniform( 'sky.atmosphere.mieG', MIE_G, { min: 0.55, max: 0.9, onChange: markSkyRetune } );
	/** the forward lean k of the sky's Rayleigh phase (`skyRayleighPhase`) */
	readonly rayleighForward = tuneUniform( 'sky.atmosphere.rayleighForward', SKY_LEAN, { min: 0, max: 1, onChange: markSkyRetune } );
	/** the polariser's strength p on the sky's Rayleigh share (`polariser`) */
	readonly polarisation = tuneUniform( 'sky.atmosphere.polarisation', SKY_POLARISER, { min: 0, max: 1, onChange: markSkyRetune } );
	/** the earth's shadow over the seen sky (`skyShadow`): its depth and its softness (degrees) */
	readonly shadowDepth = tuneUniform( 'sky.atmosphere.shadowDepth', SKY_SHADOW_DEPTH, { min: 0, max: 1, onChange: markSkyRetune } );
	readonly shadowSoftness = tuneUniform( 'sky.atmosphere.shadowSoftness', SKY_SHADOW_SOFTNESS, { min: 0.5, max: 15, onChange: markSkyRetune } );
	/** the floor added to both sides of the clouds' shadow ratio (`earthShadow`) */
	readonly shadowFloor = tuneUniform( 'sky.atmosphere.shadowFloor', SHADOW_FLOOR, { min: 0, max: 1, onChange: markSkyRetune } );
	/** the ozone's absorption per band (per km at its density's peak, OZONE) and its column's scale:
	 * tunable on debug pages (`sky.atmosphere.ozoneR`, `ozoneG`, `ozoneB`, `sky.atmosphere.ozone`),
	 * each a change of the medium that rebakes the tables and the environment */
	readonly ozone = uniform( new THREE.Vector3( ...OZONE ) );
	readonly ozoneScale = tuneUniform( 'sky.atmosphere.ozone', OZONE_COLUMN, { min: 0, max: 4, onChange: markSkyRetune } );
	readonly groundAlbedo = uniform( new THREE.Color( 0.06, 0.08, 0.1 ) );
	readonly viewHeight = uniform( RG + 0.002 );
	// the moon's direction, for the moon's slices of the in-scatter table
	readonly moonDir = uniform( new THREE.Vector3( 0, - 1, 0 ) );
	// the moon's illuminance the moonlit sky is drawn with (Sky.ts keys it)
	readonly moonIlluminance = uniform( new THREE.Color( 0, 0, 0 ) );
	/** gains on the moon's Rayleigh rows and on its aerosol's rows, as every reader takes them
	 * (`bodyGains`: the sky's, the fog table's, the clouds'): 1, the atmosphere's own. Their ratio is
	 * the clear night sky's structure, a dark dome under the haze's moonlit light: the Rayleigh's
	 * 0.175 against the aerosol's 1.3, set by least squares against the clear night's sky rows with the
	 * multiple scattering whole. Both then take one level, MOON_SKY_LEVEL, so the whole moonlit sky
	 * (the dome, the haze and the clouds, which read these rows alike) gives level ground the
	 * cosine-weighted irradiance it gives at gains of 1, the atmosphere's own: the land takes the
	 * moonlit sky's light as the sun's sky lends it, and the sky's look keeps its ratio. */
	readonly moonRayleigh = tuneUniform( 'sky.night.moonRayleigh', 0.175 * MOON_SKY_LEVEL, { min: 0, max: 8, onChange: markSkyRetune } );
	readonly moonMie = tuneUniform( 'sky.night.moonMie', 1.3 * MOON_SKY_LEVEL, { min: 0, max: 16, onChange: markSkyRetune } );
	// the highest altitude the fog table holds (km): the haze grid's far end
	readonly fogReach = uniform( 30 );
	readonly transmittanceLUT = makeLUT( T_W, T_H, 'atmoTransmittance' );
	readonly multiScatLUT = makeLUT( MS_SUN, MS_R, 'atmoMultiScat' );
	// The in-scatter table's atlas (Rayleigh RGB and Mie R; Mie GB), the sky row and the fog table cut
	// from it, each in the same two parts: the sun's slice in the first row(s), the moon's after.
	readonly table = [ makeLUT( ATLAS_W, TABLE_MU, 'atmoInScatter', THREE.RGBAFormat ), makeLUT( ATLAS_W, TABLE_MU, 'atmoInScatterMie', THREE.RGFormat ) ] as const;
	readonly skyRow = [ makeLUT( SKY_ROW, 2, 'atmoSkyRow', THREE.RGBAFormat ), makeLUT( SKY_ROW, 2, 'atmoSkyRowMie', THREE.RGFormat ) ] as const;
	readonly fogTable = [ makeLUT( FOG_MU, 2 * FOG_H, 'atmoFogTable', THREE.RGBAFormat ), makeLUT( FOG_MU, 2 * FOG_H, 'atmoFogTableMie', THREE.RGFormat ) ] as const;
	// the lit share's mean over the azimuth, cut with the slices: the sun's rows, then the moon's
	readonly shadowMean = makeLUT( SHADOW_MU, 2 * SHADOW_H, 'atmoShadowMean', THREE.RedFormat );

	private readonly quads: THREE.QuadMesh[];
	private staticKey = '';
	private cutKey = '';

	constructor( readonly renderer: THREE.WebGPURenderer ) {

		for ( const [ c, band ] of [ 'R', 'G', 'B' ].entries() ) tuneApply( `sky.atmosphere.ozone${ band }`, OZONE[ c ], { min: 0, max: 0.01 }, ( value ) => {

			if ( this.ozone.value.getComponent( c ) === value ) return;
			this.ozone.value.setComponent( c, value );
			markSkyRetune();

		} );

		// RGB is molecular scattering, A is aerosol scattering. Packing allows one function call
		// per march sample without a custom shader struct or shader-language source strings.
		const mediumScattering = Fn( ( [ h, rayleighScale, mieScale ]: [ F, F, F ] ) => {

			const ray = vec3( ...RAYLEIGH ).mul( exp( h.negate().div( 8 ) ) ).mul( rayleighScale );
			const mie = exp( h.negate().div( 1.2 ) ).mul( 3.996e-3 ).mul( mieScale );
			return vec4( ray, mie );

		} ).setLayout( { name: 'atmoMediumScattering', type: 'vec4', inputs: [
			{ name: 'h', type: 'float' }, { name: 'rayleighScale', type: 'float' }, { name: 'mieScale', type: 'float' },
		] } );

		const extinction = Fn( ( [ h, ray, mie, ozoneBands, ozoneScale ]: [ F, V3, F, V3, F ] ) => {

			const ozone = max( float( 1 ).sub( abs( h.sub( 25 ) ).div( 15 ) ), 0 );
			return ray.add( mie.mul( 4.440 / 3.996 ) ).add( ozoneBands.mul( ozone ).mul( ozoneScale ) );

		} ).setLayout( { name: 'atmoExtinction', type: 'vec3', inputs: [
			{ name: 'h', type: 'float' }, { name: 'ray', type: 'vec3' }, { name: 'mie', type: 'float' },
			{ name: 'ozoneBands', type: 'vec3' }, { name: 'ozoneScale', type: 'float' },
		] } );

		// These JavaScript node factories inline texture accesses in the current material. The
		// parameterization itself remains a shared, pure shader function with an explicit layout.
		const sampleTransmittance = ( r: F, mu: F ): V3 => {

			return texture( this.transmittanceLUT.texture, transmittanceUV( r, mu ) ).level( float( 0 ) ).rgb;

		};

		const sampleMultiScat = ( r: F, cosSun: F ): V3 => this.multipleScattering( r, cosSun );

		const transmittanceBake = Fn( () => {

			const q = uv().toVar();
			const xMu = q.x.sub( 0.5 / T_W ).mul( T_W / ( T_W - 1 ) );
			const xR = q.y.sub( 0.5 / T_H ).mul( T_H / ( T_H - 1 ) );
			const rho = xR.mul( H ).toVar();
			const r = sqrt( rho.mul( rho ).add( RG * RG ) ).toVar();
			const dMin = float( RT ).sub( r ), dMax = rho.add( H );
			const d = mix( dMin, dMax, xMu ).toVar();
			const mu = clamp( select( d.equal( 0 ), 1,
				float( H * H ).sub( rho.mul( rho ) ).sub( d.mul( d ) ).div( r.mul( d ).mul( 2 ).max( 1e-6 ) ) ), - 1, 1 ).toVar();
			const ro = vec3( 0, r, 0 ).toVar();
			const rd = vec3( sqrt( max( float( 1 ).sub( mu.mul( mu ) ), 0 ) ), mu, 0 ).toVar();
			// The outermost outward texel is already in space: its optical depth is zero.
			const dt = max( raySphereNearest( ro, rd, float( RT ) ), 0 ).div( 40 ).toVar();
			const od = vec3( 0 ).toVar();
			Loop( 40, ( { i } ) => {

				const p = ro.add( rd.mul( float( i ).add( 0.5 ).mul( dt ) ) );
				const h = length( p ).sub( RG ).toVar();
				const m = mediumScattering( h, this.rayleighScale, this.mieDensity ).toVar();
				od.addAssign( extinction( h, m.rgb, m.a, this.ozone, this.ozoneScale ).mul( dt ) );

			} );
			return vec4( exp( od.negate() ), 1 );

		} )();

		const multiScatBake = Fn( () => {

			// the texel's node: the sun's cosine from the axis' square root (`multipleScattering`)
			const q = uv().sub( vec2( 0.5 / MS_SUN, 0.5 / MS_R ) ).mul( vec2( MS_SUN / ( MS_SUN - 1 ), MS_R / ( MS_R - 1 ) ) ).toVar();
			const s = q.x.mul( 2 ).sub( 1 ).toVar();
			const cosSun = s.mul( abs( s ) ).toVar();
			const r = clamp( q.y, 0.001, 0.999 ).mul( RT - RG ).add( RG );
			const sun = vec3( 0, cosSun, sqrt( max( float( 1 ).sub( cosSun.mul( cosSun ) ), 0 ) ).negate() ).toVar();
			const ro = vec3( 0, r, 0 ).toVar(), Lsum = vec3( 0 ).toVar(), fmsSum = vec3( 0 ).toVar();
			const throughput = vec3( 1 ).toVar(), L = vec3( 0 ).toVar(), fms = vec3( 0 ).toVar();
			// One Loop with both ranges gives the direction and distance counters distinct names.
			// The integrator stays in this one material, where its uniforms and texture bindings belong.
			Loop( { start: int( 0 ), end: int( 64 ), type: 'int' }, { start: int( 0 ), end: int( 20 ), type: 'int' }, ( { i, j } ) => {

				const a = float( i.mod( int( 8 ) ) ).add( 0.5 ).div( 8 );
				const b = float( i.div( int( 8 ) ) ).add( 0.5 ).div( 8 );
				const theta = a.mul( 2 * Math.PI ).toVar(), phi = acos( float( 1 ).sub( b.mul( 2 ) ) ).toVar();
				const rd = vec3( cos( theta ).mul( sin( phi ) ), cos( phi ), sin( theta ).mul( sin( phi ) ) ).toVar();
				const bottom = raySphereNearest( ro, rd, float( RG ) ).toVar();
				const top = raySphereNearest( ro, rd, float( RT ) );
				const tMax = select( bottom.greaterThan( 0 ), bottom, top ).toVar();
				const dt = tMax.div( 20 ).toVar();
				If( j.equal( 0 ), () => {

					throughput.assign( vec3( 1 ) );
					L.assign( vec3( 0 ) );
					fms.assign( vec3( 0 ) );

				} );
				const p = ro.add( rd.mul( float( j ).add( 0.3 ).mul( dt ) ) ).toVar();
				const pr = length( p ).toVar(), h = pr.sub( RG ).toVar();
				const m = mediumScattering( h, this.rayleighScale, this.mieDensity ).toVar();
				const ext = extinction( h, m.rgb, m.a, this.ozone, this.ozoneScale ).toVar();
				const scattering = m.rgb.add( m.a ).toVar();
				const cosSunP = dot( p.div( pr ), sun ).toVar();
				const TSun = sampleTransmittance( pr, cosSunP ).toVar();
				const shadow = select( raySphereNearest( p, sun, float( RG ) ).greaterThan( 0 ), 0, 1 );
				const S = TSun.mul( shadow ).mul( scattering ).mul( 1 / ( 4 * Math.PI ) ).toVar();
				const TStep = exp( ext.mul( dt ).negate() ).toVar();
				const integral = throughput.mul( dt ).mul( stepShare( ext.mul( dt ) ) ).toVar();
				L.addAssign( S.mul( integral ) );
				fms.addAssign( scattering.mul( integral ) );
				throughput.mulAssign( TStep );
				If( j.equal( 19 ), () => {

					If( bottom.greaterThan( 0 ), () => {

						const up = normalize( ro.add( rd.mul( tMax ) ) );
						const cosSunGround = dot( up, sun ).toVar();
						const TGround = sampleTransmittance( float( RG ), cosSunGround );
						L.addAssign( TGround.mul( throughput ).mul( max( cosSunGround, 0 ) ).mul( this.groundAlbedo ).div( Math.PI ) );

					} );
					Lsum.addAssign( L );
					fmsSum.addAssign( fms );

				} );

			} );
			const Lin = Lsum.div( 64 ), fmsAvg = fmsSum.div( 64 );
			return vec4( Lin.div( max( vec3( 1 ).sub( fmsAvg ), vec3( 1e-4 ) ) ), 1 );

		} )();

		// The in-scatter table: per texel of the atlas, the light scattered toward the viewer along the
		// view ray (to the ground or the atmosphere's top), per unit of the sun's illuminance and
		// without the phase, its mean over TABLE_AZIMUTHS azimuths of the sun about the vertical:
		// Rayleigh RGB with the multiple scattering of both media (Hillaire 2020's Psi_ms times their
		// scattering, isotropic) and Mie RGB. Psi_ms is added with no phase (the paper's equation 11:
		// the single scattering takes the phase, Psi_ms, in 1/sr, the scattering alone); folded into
		// the Rayleigh slot it takes the Rayleigh phase where the slot is read (`skyRayleighPhase`,
		// whose mean over the sphere is 1 / ( 4 pi )), so it is baked times 4 pi: averaged over the
		// directions it then reaches the sky as the isotropic term does. Without the factor the
		// twilight under the sun at -8 degrees and below, nearly all multiple scattering, stood 3.65
		// stops under that (noon 0.35 - 0.43).
		const tableBake = ( mie: boolean ) => Fn( () => {

			const pixel = floor( uv().mul( vec2( ATLAS_W, TABLE_MU ) ) ).toVar();
			const slice = floor( pixel.x.div( TABLE_MUS ) ).toVar();
			const muS = unitMuS( pixel.x.sub( slice.mul( TABLE_MUS ) ).div( TABLE_MUS - 1 ) ).toVar();
			const u = slice.div( TABLE_R - 1 );
			const r = u.mul( u ).mul( RT - RG ).add( RG ).toVar();
			const mu = texelMu( pixel.y.add( 0.5 ), r, float( TABLE_MU ) ).toVar();
			const sinS = sqrt( max( float( 1 ).sub( muS.mul( muS ) ), 0 ) ).toVar();
			const ro = vec3( 0, r, 0 ).toVar();
			const rd = vec3( sqrt( max( float( 1 ).sub( mu.mul( mu ) ), 0 ) ), mu, 0 ).toVar();
			// (a ray under the horizon of its radius ends on the ground, in closed form, so one from the
			// ground itself has no length rather than a chord through the planet)
			const horizon = sqrt( max( float( 1 ).sub( float( RG * RG ).div( r.mul( r ) ) ), 0 ) ).negate();
			const ground = max( r.mul( mu ).negate().sub( sqrt( max( r.mul( r ).mul( mu.mul( mu ).sub( 1 ) ).add( RG * RG ), 0 ) ) ), 0 );
			const tMax = select( mu.lessThan( horizon ), ground, max( raySphereNearest( ro, rd, float( RT ) ), 0 ) ).toVar();
			const throughput = vec3( 1 ).toVar(), rayleigh = vec3( 0 ).toVar(), aerosol = vec3( 0 ).toVar();
			Loop( { start: int( 0 ), end: int( TABLE_STEPS ), type: 'int' }, { start: int( 0 ), end: int( TABLE_AZIMUTHS ), type: 'int' }, ( { i, j } ) => {

				const t0 = float( i ).div( TABLE_STEPS ), t1 = float( i ).add( 1 ).div( TABLE_STEPS );
				const ta = t0.mul( t0 ).mul( tMax ), tb = t1.mul( t1 ).mul( tMax );
				const dt = tb.sub( ta ).toVar();
				const p = ro.add( rd.mul( mix( ta, tb, 0.3 ) ) ).toVar();
				const pr = length( p ).toVar(), h = max( pr.sub( RG ), 0 ).toVar();
				const m = mediumScattering( h, this.rayleighScale, this.mieDensity ).toVar();
				const ext = extinction( h, m.rgb, m.a, this.ozone, this.ozoneScale ).toVar();
				// the azimuths over half a turn: the view lies in a vertical plane, so the other half
				// mirrors them
				const phi = float( j ).add( 0.5 ).mul( Math.PI / TABLE_AZIMUTHS );
				const sun = vec3( sinS.mul( cos( phi ) ), muS, sinS.mul( sin( phi ) ) ).toVar();
				const cosSun = dot( p.div( pr ), sun ).toVar();
				const TSun = sampleTransmittance( pr, cosSun ).mul( select( raySphereNearest( p, sun, float( RG ) ).greaterThan( 0 ), 0, 1 ) ).toVar();
				const ms = sampleMultiScat( pr, cosSun ).mul( 4 * Math.PI );
				const TStep = exp( ext.mul( dt ).negate() ).toVar();
				const integral = throughput.mul( dt ).mul( stepShare( ext.mul( dt ) ) ).mul( 1 / TABLE_AZIMUTHS ).toVar();
				rayleigh.addAssign( TSun.mul( m.rgb ).add( ms.mul( m.rgb.add( m.a ) ) ).mul( integral ) );
				aerosol.addAssign( TSun.mul( m.a ).mul( integral ) );
				If( j.equal( TABLE_AZIMUTHS - 1 ), () => {

					throughput.mulAssign( TStep );

				} );

			} );
			return mie ? vec4( aerosol.g, aerosol.b, 0, 1 ) : vec4( rayleigh, aerosol.r );

		} )();

		// a slice cut from the table: per texel, its body (0 the sun, 1 the moon), the ray's mu and the
		// radius, read through the table's B-spline
		const cut = ( part: number, place: () => { body: F; mu: F; r: F } ) => Fn( () => {

			const { body, mu, r } = place();
			const muS = select( body.lessThan( 0.5 ), this.sunDir.y, this.moonDir.y );
			return this.tableSample( part, muS, mu, r );

		} )();
		const skyRowPlace = () => {

			const pixel = floor( uv().mul( vec2( SKY_ROW, 2 ) ) ).toVar();
			return { body: pixel.y, mu: texelMu( pixel.x.add( 0.5 ), this.viewHeight, float( SKY_ROW ) ), r: this.viewHeight };

		};
		const fogPlace = () => {

			const pixel = floor( uv().mul( vec2( FOG_MU, 2 * FOG_H ) ) ).toVar();
			const body = floor( pixel.y.div( FOG_H ) ).toVar();
			const a = pixel.y.sub( body.mul( FOG_H ) ).div( FOG_H - 1 );
			const r = a.mul( a ).mul( this.fogReach.sub( FOG_LOW ) ).add( RG + FOG_LOW ).toVar();
			return { body, mu: texelMu( pixel.x.add( 0.5 ), r, float( FOG_MU ) ), r };

		};

		// the lit share's mean over the azimuth of the light about the vertical, for the body's
		// elevation, the ray's mu and the altitude: SHADOW_AZIMUTHS midpoints over half a turn (the
		// share is even in the azimuth)
		const shadowBake = Fn( () => {

			const pixel = floor( uv().mul( vec2( SHADOW_MU, 2 * SHADOW_H ) ) ).toVar();
			const body = floor( pixel.y.div( SHADOW_H ) ).toVar();
			const a = pixel.y.sub( body.mul( SHADOW_H ) ).div( SHADOW_H - 1 ).toVar();
			const h = a.mul( a ).mul( a.mul( a ) ).mul( this.fogReach.sub( FOG_LOW ) ).add( FOG_LOW ).toVar();
			const mu = texelMu( pixel.x.add( 0.5 ), h.add( RG ), float( SHADOW_MU ) ).toVar();
			const dir = vec3( sqrt( max( float( 1 ).sub( mu.mul( mu ) ), 0 ) ), mu, 0 ).toVar();
			const muL = select( body.lessThan( 0.5 ), this.sunDir.y, this.moonDir.y ).toVar();
			const across = sqrt( max( float( 1 ).sub( muL.mul( muL ) ), 0 ) ).toVar();
			const sum = float( 0 ).toVar();
			Loop( SHADOW_AZIMUTHS, ( { i } ) => {

				const phi = float( i ).add( 0.5 ).mul( Math.PI / SHADOW_AZIMUTHS ).toVar();
				sum.addAssign( litShare( h, dir, vec3( across.mul( cos( phi ) ), muL, across.mul( sin( phi ) ) ) ) );

			} );
			return vec4( sum.div( SHADOW_AZIMUTHS ), 0, 0, 1 );

		} )();

		this.quads = [ makeQuad( transmittanceBake ), makeQuad( multiScatBake ), makeQuad( tableBake( false ) ), makeQuad( tableBake( true ) ),
			makeQuad( cut( 0, skyRowPlace ) ), makeQuad( cut( 1, skyRowPlace ) ), makeQuad( cut( 0, fogPlace ) ), makeQuad( cut( 1, fogPlace ) ),
			makeQuad( shadowBake ) ];

	}

	// The in-scatter table's part (0: Rayleigh RGB and Mie R, 1: Mie GB) at ( muS, mu, r ), through
	// a cubic B-spline in all three axes: two bilinear taps per axis in muS and mu, held inside the
	// r slice and the half of mu, over the four nearest r slices.
	private tableSample( part: number, muS: F, mu: F, r: F ): V4 {

		const atlas = this.table[ part ].texture;
		const s = bsplineTaps( muSUnit( muS ).mul( TABLE_MUS - 1 ).add( 0.5 ) ).toVar();
		const m = muTexel( mu, r, float( TABLE_MU ) ).toVar();
		const v = bsplineTaps( m.x ).toVar();
		const x0 = clamp( s.x, 0.5, TABLE_MUS - 0.5 ).toVar(), x1 = clamp( s.y, 0.5, TABLE_MUS - 0.5 ).toVar();
		const y0 = clamp( v.x, m.y, m.z ).div( TABLE_MU ).toVar(), y1 = clamp( v.y, m.y, m.z ).div( TABLE_MU ).toVar();
		const u = sqrt( clamp( r.sub( RG ).div( RT - RG ), 0, 1 ) ).mul( TABLE_R - 1 ).toVar();
		const k = floor( u ).toVar();
		const w = bsplineWeights( u.sub( k ) ).toVar();
		const sum = vec4( 0 ).toVar();
		for ( let d = - 1; d <= 2; d ++ ) {

			const offset = clamp( k.add( d ), 0, TABLE_R - 1 ).mul( TABLE_MUS ).toVar();
			const tap = ( x: F, y: F ) => texture( atlas, vec2( offset.add( x ).div( ATLAS_W ), y ) ).level( float( 0 ) );
			const bicubic = mix( mix( tap( x1, y1 ), tap( x0, y1 ), s.z ), mix( tap( x1, y0 ), tap( x0, y0 ), s.z ), v.z );
			sum.addAssign( bicubic.mul( [ w.x, w.y, w.z, w.w ][ d + 1 ] ) );

		}
		return sum;

	}

	/** The sky row at the camera's height for the view direction's `mu`: the in-scatter per unit of
	 * the illuminance of `body` (0 the sun, 1 the moon), without the phase: Rayleigh (with the
	 * multiple scattering) and Mie, through a cubic B-spline held inside its half of mu. */
	skyRows( mu: F, body: 0 | 1 ): { rayleigh: V3; mie: V3 } {

		const m = muTexel( mu, this.viewHeight, float( SKY_ROW ) ).toVar();
		const b = bsplineTaps( m.x ).toVar();
		const x0 = clamp( b.x, m.y, m.z ).div( SKY_ROW ).toVar(), x1 = clamp( b.y, m.y, m.z ).div( SKY_ROW ).toVar();
		const y = ( body + 0.5 ) / 2;
		const read = ( part: number ) => mix( texture( this.skyRow[ part ].texture, vec2( x1, y ) ).level( float( 0 ) ),
			texture( this.skyRow[ part ].texture, vec2( x0, y ) ).level( float( 0 ) ), b.z );
		const a = read( 0 ).toVar(), c = read( 1 ).toVar();
		return this.bodyGains( { rayleigh: a.rgb, mie: vec3( a.a, c.r, c.g ) }, body );

	}

	/** The fog table at `height` (km above the sea) for a direction's `mu` (the haze reads the sun's
	 * along the view, the moon's along the moon): the in-scatter along that ray from that altitude
	 * per unit of the illuminance of `body` (0 the sun, 1 the moon), without the phase, for the
	 * haze's light (weather/VolumetricHaze.ts): a cubic B-spline in
	 * mu (held inside its half, at the radius read) and in the altitude's quadratic axis. A ray under
	 * the horizon of that radius reads the horizon's sky side (the module's header says why). */
	fogRows( mu: F, height: F, body: 0 | 1 ): { rayleigh: V3; mie: V3 } {

		const r = max( height, FOG_LOW ).add( RG ).toVar();
		const horizon = sqrt( max( float( 1 ).sub( float( RG * RG ).div( r.mul( r ) ) ), 0 ) ).negate();
		const m = muTexel( max( mu, horizon.add( 1e-6 ) ), r, float( FOG_MU ) ).toVar();
		const b = bsplineTaps( m.x ).toVar();
		const x0 = clamp( b.x, m.y, m.z ).div( FOG_MU ).toVar(), x1 = clamp( b.y, m.y, m.z ).div( FOG_MU ).toVar();
		const c = bsplineTaps( sqrt( clamp( height.sub( FOG_LOW ).div( this.fogReach.sub( FOG_LOW ) ), 0, 1 ) ).mul( FOG_H - 1 ).add( 0.5 ) ).toVar();
		const row = ( y: F ) => clamp( y, 0.5, FOG_H - 0.5 ).add( body * FOG_H ).div( 2 * FOG_H );
		const y0 = row( c.x ).toVar(), y1 = row( c.y ).toVar();
		const read = ( part: number ) => {

			const tap = ( x: F, y: F ) => texture( this.fogTable[ part ].texture, vec2( x, y ) ).level( float( 0 ) );
			return mix( mix( tap( x1, y1 ), tap( x0, y1 ), b.z ), mix( tap( x1, y0 ), tap( x0, y0 ), b.z ), c.z );

		};
		const a = read( 0 ).toVar(), d = read( 1 ).toVar();
		return this.bodyGains( { rayleigh: a.rgb, mie: vec3( a.a, d.r, d.g ) }, body );

	}

	/** The in-scatter table itself at `height` (km above the sea) for the view direction's `mu`, for
	 * the sun's (0) or the moon's (1) muS: the clouds' rows at their shell's mean height (Clouds.ts),
	 * which no slice holds, and which follows the weather (a node). */
	cloudRows( mu: F, height: F | number, body: 0 | 1 ): { rayleigh: V3; mie: V3 } {

		const muS = ( body === 0 ? this.sunDir : this.moonDir ).y;
		const r = typeof height === 'number' ? float( RG + height ) : float( RG ).add( height );
		const a = this.tableSample( 0, muS, mu, r ).toVar(), c = this.tableSample( 1, muS, mu, r ).toVar();
		return this.bodyGains( { rayleigh: a.rgb, mie: vec3( a.a, c.r, c.g ) }, body );

	}

	/** A body's rows as its readers take them: the sun's as the tables hold them, the moon's
	 * Rayleigh row times `moonRayleigh` and its aerosol's row times `moonMie`. */
	private bodyGains( rows: { rayleigh: V3; mie: V3 }, body: 0 | 1 ): { rayleigh: V3; mie: V3 } {

		if ( body === 0 ) return rows;
		return { rayleigh: rows.rayleigh.mul( this.moonRayleigh ), mie: rows.mie.mul( this.moonMie ) };

	}

	/** The clear sky's radiance toward `dir`: the sun's sky row times the phases at use (the sky's
	 * Rayleigh phase under the polariser, the aerosol's Cornette-Shanks lobe), times the earth's
	 * shadow over the seen sky (`skyShadow`), times the sun's illuminance the sky is drawn with;
	 * plus the same for the moon, with its
	 * row, its direction and its illuminance. `polarised` false leaves the polariser out, for the sky
	 * light that lights the clouds (a polariser darkens what the eye sees of the sky, not the
	 * light the sky sheds on them). `gains` scales each body's share on its own (the open sky's gain
	 * behind the clouds, Sky.ts `gapSky`, which each body takes at its own height). */
	skyRadiance( dir: V3, polarised = true, gains?: { sun: F; moon: F } ): V3 {

		const shares = this.skyShares( dir, polarised, gains );
		return shares.sun.add( shares.moon );

	}

	/** The sun's and the moon's shares of the sky toward `dir` apart (`skyRadiance`'s two terms,
	 * each times its gain when `gains` is given): the exposure's meter weighs the night sky's seen
	 * gain on the moon's share by them (post/Metering.ts). */
	skyShares( dir: V3, polarised = true, gains?: { sun: F; moon: F } ): { sun: V3; moon: V3 } {

		const sun = this.bodySky( dir, 0, this.sunDir, this.sunIlluminance, polarised );
		const moon = this.bodySky( dir, 1, this.moonDir, this.moonIlluminance, polarised );
		return gains ? { sun: sun.mul( gains.sun ), moon: moon.mul( gains.moon ) } : { sun, moon };

	}

	/** One body's share of the sky toward `dir`: its sky row times the phases at use and the earth's
	 * shadow, times its illuminance. */
	private bodySky( dir: V3, body: 0 | 1, light: V3, illuminance: THREE.UniformNode<'color', THREE.Color>, polarised: boolean ): V3 {

		const row = this.skyRows( dir.y, body );
		const nu = dot( dir, light ).toVar();
		let rayleigh: F = skyRayleighPhase( nu, this.rayleighForward );
		if ( polarised ) rayleigh = rayleigh.mul( polariser( nu, this.polarisation ) );
		return row.rayleigh.mul( rayleigh ).add( row.mie.mul( cornetteShanks( nu, this.mieG ) ) )
			.mul( this.skyShadowOf( dir, light ) ).mul( illuminance );

	}

	/** The earth's shadow over the seen sky toward `dir` for a light toward `light` (`skyShadow`,
	 * at this atmosphere's depth and softness). */
	skyShadowOf( dir: V3, light: V3 ): F {

		return skyShadow( dir, light, this.shadowDepth, this.shadowSoftness.mul( Math.PI / 180 ) );

	}

	/** The lit share's mean over the azimuth (`shadowMean`) at `height` km above the sea for a ray
	 * of elevation sine `mu` and the body `body` (0 the sun, 1 the moon): bilinear, held inside its
	 * half of mu and its body's rows. */
	private shadowMeanAt( height: F, mu: F, body: 0 | 1 ): F {

		const m = muTexel( mu, height.add( RG ), float( SHADOW_MU ) ).toVar();
		const x = clamp( m.x, m.y, m.z ).div( SHADOW_MU );
		const a = sqrt( sqrt( clamp( height.sub( FOG_LOW ).div( this.fogReach.sub( FOG_LOW ) ), 0, 1 ) ) );
		const y = clamp( a.mul( SHADOW_H - 1 ).add( 0.5 ), 0.5, SHADOW_H - 0.5 ).add( body * SHADOW_H ).div( 2 * SHADOW_H );
		return texture( this.shadowMean.texture, vec2( x, y ) ).level( float( 0 ) ).r;

	}

	/** The earth's shadow on the in-scatter read from the table (or a slice of it) at `height` km
	 * above the sea toward `dir`, for the body `body` (0 the sun, 1 the moon): the lit share toward
	 * `dir` (`litShare`) over its mean over the azimuth (`shadowMeanAt`), each raised by
	 * SHADOW_FLOOR. The table holds the in-scatter averaged over the azimuth with the planet's
	 * shadow in it, so its shadow is the mean one already; the ratio takes that mean out and puts
	 * this azimuth's in. Averaged over the azimuth the factor is 1 (the twilight's total is the
	 * table's, not dimmed a second time); toward the light it rises above 1, opposite it falls
	 * below, and the shadow rises opposite the setting sun as a band from the horizon. By day both
	 * shares are 1 and so is the factor. The floor stands for the light the table holds that no
	 * shadow takes (the multiple scattering), taken as a share of a fully lit ray's single
	 * scattering: 0.15, the multiple scattering's share of the in-scatter at the clouds' shell (6.6 km
	 * up) with the sun 0 - 6 degrees up, 0.12 - 0.18 in luminance (it was 0.1 when the multiple
	 * scattering stood at 1 / 4 pi of its level, and its share there 0.015 - 0.022). Where the whole ray lies in
	 * the shadow at every azimuth the factor goes to 1 (the table's value kept), and it never
	 * exceeds ( 1 + floor ) / floor, about 7.7. Tunable: `sky.atmosphere.shadowFloor`. The divisor is
	 * held at 1e-6 or more, a no-op at any floor above that: at a floor of 0 a ray in the shadow at
	 * every azimuth (any at night) would divide 0 by 0, and a NaN in the clouds' sky light spreads
	 * through the bloom and the meter over the whole frame; it takes 0 instead. */
	private earthShadow( height: F, dir: V3, body: 0 | 1 ): F {

		const light = body === 0 ? this.sunDir : this.moonDir;
		return litShare( height, dir, light ).add( this.shadowFloor ).div( max( this.shadowMeanAt( height, dir.y, body ).add( this.shadowFloor ), 1e-6 ) );

	}

	/** The earth's shadow toward `dir` from a point `height` km above the sea, for the body `body`
	 * (the haze's froxels, the clouds' shell). */
	earthShadowAt( height: F, dir: V3, body: 0 | 1 ): F {

		return this.earthShadow( max( height, 0 ), dir, body );

	}

	/** The multiple-scattering LUT at radius `r` (km from the planet's centre; `viewHeight` for the
	 * camera's) for a sun of cosine `cosSun`: the light scattered more than once that a unit of
	 * scattering takes there, per unit of the sun's illuminance (isotropic, 1/sr: Hillaire 2020's
	 * Psi_ms). The paper's sun axis is linear in the cosine; here it is linear in the cosine's
	 * signed square root, its nodes dense near the horizon, where Psi_ms falls by orders of
	 * magnitude over a few degrees of the sun's descent and a bilinear read between nodes far apart
	 * follows the upper one. On 32 nodes linear in the cosine (3.7 degrees apart near the horizon)
	 * the sky stood .15 stop over a read on nodes half a degree apart under the sun at -6 degrees
	 * and .75 - 3.4 stops over it from -8 to -11.5; on these 192 (half a degree apart at -10, closer
	 * toward the horizon) it stands within .07 stop of it. */
	multipleScattering( r: F, cosSun: F ): V3 {

		const q = vec2( sign( cosSun ).mul( sqrt( abs( cosSun ) ) ).mul( 0.5 ).add( 0.5 ), r.sub( RG ).div( RT - RG ) );
		return texture( this.multiScatLUT.texture, q.mul( vec2( ( MS_SUN - 1 ) / MS_SUN, ( MS_R - 1 ) / MS_R ) ).add( vec2( 0.5 / MS_SUN, 0.5 / MS_R ) ) ).level( float( 0 ) ).rgb;

	}

	/** The analytic transmittance from the camera to space along `dir`, for the sun's disc: Rayleigh
	 * and aerosol optical depths as each one's extinction at the ground times its scale height
	 * (8 km, 1.2 km) times Schuler's Chapman air mass, without the ozone, through the band-to-display
	 * matrix (BandMatrix.ts). */
	discTransmittance( dir: V3 ): V3 {

		const height = this.viewHeight.sub( RG ).toVar();
		const rayleigh = vec3( ...RAYLEIGH ).mul( this.rayleighScale ).mul( 8 ).mul( chapman( float( RG / 8 ), height.div( 8 ), dir.y ) );
		const mie = this.mieDensity.mul( 4.44e-3 * 1.2 ).mul( chapman( float( RG / 1.2 ), height.div( 1.2 ), dir.y ) );
		return bandToDisplay( exp( rayleigh.add( mie ).negate() ), bandPush );

	}

	transmittanceToSpace( dir: V3 ): V3 {

		const q = transmittanceUV( this.viewHeight, dir.y );
		return texture( this.transmittanceLUT.texture, q ).level( float( 0 ) ).rgb;

	}

	/** The transmittance from `heightKm` above the sea toward a direction of elevation sine `mu`,
	 * read from the LUT, the ray held just above the ground's horizon as the CPU twin traces it
	 * (`sunTransmittance`): the cirrus's sunlight at each of its points (Clouds.ts
	 * `streaksAt`). */
	transmittanceFrom( heightKm: F, mu: F ): V3 {

		const r = float( RG ).add( heightKm ).toVar();
		const horizon = sqrt( max( float( 1 ).sub( float( RG * RG ).div( r.mul( r ) ) ), 0 ) ).negate();
		return texture( this.transmittanceLUT.texture, transmittanceUV( r, max( mu, horizon.add( 1e-4 ) ) ) ).level( float( 0 ) ).rgb;

	}

	// CPU twin of the extinction march, used for the DirectionalLight's irradiance. More samples
	// than the LUT bake stabilize the very low sun without a GPU readback or a separate light model.
	// The disk's visible share is kept apart (sunVisibleShareToSun) so the key's hue can be art-directed
	// from the air's extinction alone: folded in, the fading disk would read as extra air mass.
	transmittanceToSun(): THREE.Color {

		return this.sunTransmittance( this.sunDir.value, this.viewHeight.value - RG );

	}

	/** The moon's light transmittance from the view height, by its elevation alone (`sunTransmittance`
	 * toward `moonDir`): the key's extinction on the way down, which the moonlit sky, the haze and
	 * the clouds integrate themselves (Sky.ts `track`). */
	transmittanceToMoon(): THREE.Color {

		const y = this.moonDir.value.y, height = this.viewHeight.value;
		if ( y !== this.moonTransmittanceKey[ 0 ] || height !== this.moonTransmittanceKey[ 1 ] ) {

			this.moonTransmittance.copy( this.sunTransmittance( this.moonDir.value, height - RG ) );
			this.moonTransmittanceKey[ 0 ] = y;
			this.moonTransmittanceKey[ 1 ] = height;

		}
		return this.moonTransmittance;

	}
	private readonly moonTransmittance = new THREE.Color();
	private readonly moonTransmittanceKey = [ NaN, NaN ];

	/** The share above the horizon of the setting sun's fading disk (SUN_SETTING_RADIUS) seen from the
	 * view height. */
	sunVisibleShareToSun(): number {

		return this.sunVisibleShare( this.sunDir.value, this.viewHeight.value - RG );

	}

	/** Share of a sun disk of angular radius `radius` (radians) centred on `dir` that stands
	 * above the horizon seen from `heightKm`. */
	sunVisibleShare( dir: THREE.Vector3, heightKm: number, radius = SUN_SETTING_RADIUS ): number {

		const horizon = - Math.acos( RG / ( RG + heightKm ) );
		const elevation = Math.asin( THREE.MathUtils.clamp( dir.y, - 1, 1 ) );
		return THREE.MathUtils.smoothstep( elevation, horizon - radius, horizon + radius );

	}

	/** Sunlight transmittance from `heightKm` toward the sun centred on `dir`, before its disk's
	 * visible share. The ray to the disk's centre meets the ground while its upper limb still
	 * shines: trace the ray just above the horizon instead, so the light fades with the setting
	 * disk rather than cutting out at once. */
	sunTransmittance( dir: THREE.Vector3, heightKm: number ): THREE.Color {

		const horizon = - Math.acos( RG / ( RG + heightKm ) );
		const elevation = Math.asin( THREE.MathUtils.clamp( dir.y, - 1, 1 ) );
		// Transmittance depends only on the ray's elevation.
		const traced = Math.max( elevation, horizon + 1e-4 );
		_traced.set( Math.cos( traced ), Math.sin( traced ), 0 );
		return this.transmittanceAt( _traced, heightKm );

	}

	/** Sunlight from `heightKm` toward a sun disk of angular radius `radius` (radians) centred on
	 * `dir`: its transmittance weighted by the share of the disk above the horizon. */
	sunlightAt( dir: THREE.Vector3, heightKm: number, radius = SUN_SETTING_RADIUS ): THREE.Color {

		const visible = this.sunVisibleShare( dir, heightKm, radius );
		if ( visible === 0 ) return new THREE.Color( 0, 0, 0 );
		return this.sunTransmittance( dir, heightKm ).multiplyScalar( visible );

	}

	/** Sunlight transmittance from `heightKm` above sea level toward a direction. */
	transmittanceAt( dir: THREE.Vector3, heightKm: number ): THREE.Color {

		const r = RG + heightKm;
		const b = r * dir.y, groundDisc = b * b - ( r * r - RG * RG );
		if ( groundDisc >= 0 && - b - Math.sqrt( groundDisc ) > 0 ) return new THREE.Color( 0, 0, 0 );
		const distance = - b + Math.sqrt( b * b - ( r * r - RT * RT ) );
		const dt = distance / 256, opticalDepth = [ 0, 0, 0 ];
		for ( let i = 0; i < 256; i ++ ) {

			const t = ( i + 0.5 ) * dt;
			const h = Math.sqrt( t * t + 2 * r * dir.y * t + r * r ) - RG;
			const ray = Math.exp( - h / 8 ) * this.rayleighScale.value;
			const mie = Math.exp( - h / 1.2 ) * this.mieScale.value * this.aerosol.value * 4.440e-3;
			const ozone = Math.max( 0, 1 - Math.abs( h - 25 ) / 15 ) * this.ozoneScale.value;
			for ( let c = 0; c < 3; c ++ ) opticalDepth[ c ] += ( RAYLEIGH[ c ] * ray + mie + this.ozone.value.getComponent( c ) * ozone ) * dt;

		}
		return new THREE.Color( Math.exp( - opticalDepth[ 0 ] ), Math.exp( - opticalDepth[ 1 ] ), Math.exp( - opticalDepth[ 2 ] ) );

	}

	async update(): Promise<void> {

		await this.renderer.init();
		this.updateSync();

	}

	// Runtime sky updates run inside the frame after renderer initialisation, without an
	// asynchronous boundary that would suspend or interleave the frame's GPU submissions.
	updateSync(): void {

		const albedo = this.groundAlbedo.value;
		const staticKey = [ this.rayleighScale.value, this.mieScale.value, this.aerosol.value, this.ozoneScale.value,
			...this.ozone.value.toArray(), albedo.r, albedo.g, albedo.b ].join( ',' );
		const cutKey = [ staticKey, this.sunDir.value.y, this.moonDir.value.y, this.viewHeight.value, this.fogReach.value ].join( ',' );
		if ( cutKey === this.cutKey ) return;

		const previous = this.renderer.getRenderTarget();
		const render = ( target: THREE.RenderTarget, quad: number ) => {

			this.renderer.setRenderTarget( target );
			this.quads[ quad ].render( this.renderer );

		};
		try {

			if ( staticKey !== this.staticKey ) {

				// the medium changed: the LUTs and the in-scatter table (the sun's motion leaves them)
				render( this.transmittanceLUT, 0 );
				render( this.multiScatLUT, 1 );
				render( this.table[ 0 ], 2 );
				render( this.table[ 1 ], 3 );
				this.staticKey = staticKey;

			}
			render( this.skyRow[ 0 ], 4 );
			render( this.skyRow[ 1 ], 5 );
			render( this.fogTable[ 0 ], 6 );
			render( this.fogTable[ 1 ], 7 );
			render( this.shadowMean, 8 );
			this.cutKey = cutKey;

		} finally {

			this.renderer.setRenderTarget( previous );

		}

	}

	dispose(): void {

		for ( const quad of this.quads ) ( quad.material as THREE.NodeMaterial ).dispose();
		this.transmittanceLUT.dispose();
		this.multiScatLUT.dispose();
		for ( const target of [ ...this.table, ...this.skyRow, ...this.fogTable, this.shadowMean ] ) target.dispose();

	}

}
