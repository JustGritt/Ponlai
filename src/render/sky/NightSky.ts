/*!
 * The lunar disk is adapted from three-geospatial by Shota Matsuda
 * (https://github.com/takram-design-engineering/three-geospatial).
 * SPDX-FileCopyrightText: Copyright (c) 2024 Shota Matsuda
 * SPDX-License-Identifier: MIT
 *
 * Earthshine is adapted from a-starry-sky by David Evans (https://github.com/Dante83/a-starry-sky).
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 David Evans
 * SPDX-License-Identifier: MIT
 *
 * The moon's colour and heights come from NASA's Scientific Visualization Studio CGI Moon Kit
 * (public domain), processed by tools/island/build-moon-textures.py.
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, uniform, vec2, vec3, float, int, uint, floor, fract, sin, tan, log, dot, cross, atan, asin,
	normalize, length, fwidth, dFdx, dFdy, abs, clamp, max, min, mix, pow, smoothstep, exp, sqrt, select, acos,
	sign, texture,
} from 'three/tsl';
import { SCENE_RADIANCE_SCALE, MOON_GLINT_RADIUS } from '../lighting/Radiometry.ts';
import { tuneUniform } from '../debug/Tune.ts';
import { markSkyRetune } from './SkyRetune.ts';
import { assetUrl } from '../FetchAsset.ts';
import { STAR_CELLS, CELL_SLOTS, CELL_COUNT, MILKY_WAY_WIDTH, MILKY_WAY_HEIGHT, skyWords, starRecords, loadNightSkyData } from './StarCatalog.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

// The night sky: a real sky's stars and Milky Way, fixed, and the moon.
//
// The night sky: stars, Milky Way and moon. The stars and Milky Way are a real sky, the one seen
// from 38 degrees south at a local sidereal time of 14h (DayCycle.ts STARS), fixed: they neither
// turn through the night nor drift over the year; procedural grain and faint stars only add detail
// below the catalogue and the map. Stars are points with a natural spread of brightness, kept below
// white except the brightest few; only those near the horizon scintillate. The night is
// art-directed (docs/design/night.md, The night sky): the sky's colours are drawn as authored, and
// any night-vision shift is the post chain's (Purkinje.ts, at the look's strength: on at night in
// the island's look, off in the render pipeline's neutral look), over the whole frame.

// The drawn disk's angular radius: five times the true moon's (Radiometry.ts MOON_GLINT_RADIUS,
// 0.0045 rad), about 2.6 degrees across, a look choice so the disk reads as the night's light
// source in the island's views (about 75 pixels across at 2K under a 49 degree field of view,
// where the colour map's near side has 512 texels and the height map's 256). Only the disk grows:
// its radiance is per unit solid angle and stays as it was, and the moonlight's level is the key's
// own (look/SkyLook.ts `moonKey`), never this disk's solid angle. What follows the radius: the ray/sphere
// normal, the limb's chord (MOON_CHORD), the map footprint's cap (per radius), the stars it hides
// (`moonOcclusion`), the clouds' mean over the disk for the flare (Sky.ts `discCloudTransmittance`)
// and the fade below the horizon (`lunarDisk`).
export const MOON_ANGULAR_RADIUS = 5 * MOON_GLINT_RADIUS;

const nightHash = Fn( ( [ p ]: [ V2 ] ) => {

	return fract( sin( dot( p, vec2( 153.71, 61.39 ) ) ).mul( 39241.173 ) );

} ).setLayout( { name: 'ponlaiNightHash', type: 'float', inputs: [ { name: 'p', type: 'vec2' } ] } );

const nightNoise = Fn( ( [ p ]: [ V2 ] ) => {

	const cell = floor( p ).toVar();
	const f = fract( p ).toVar();
	const w = f.mul( f ).mul( float( 3 ).sub( f.mul( 2 ) ) ).toVar();
	return mix( mix( nightHash( cell ), nightHash( cell.add( vec2( 1, 0 ) ) ), w.x ),
		mix( nightHash( cell.add( vec2( 0, 1 ) ) ), nightHash( cell.add( vec2( 1, 1 ) ) ), w.x ), w.y );

} ).setLayout( { name: 'ponlaiNightNoise', type: 'float', inputs: [ { name: 'p', type: 'vec2' } ] } );

const hash3 = Fn( ( [ p ]: [ V3 ] ) => {

	return fract( sin( dot( p, vec3( 127.1, 311.7, 74.7 ) ) ).mul( 43758.5453 ) );

} ).setLayout( { name: 'ponlaiNightHash3', type: 'float', inputs: [ { name: 'p', type: 'vec3' } ] } );

// Value noise on the celestial sphere, for the Milky Way's clouds and dust lanes.
const noise3 = Fn( ( [ p ]: [ V3 ] ) => {

	const i = floor( p ).toVar();
	const f = fract( p ).toVar();
	const w = f.mul( f ).mul( float( 3 ).sub( f.mul( 2 ) ) ).toVar();
	const at = ( x: number, y: number, z: number ) => hash3( i.add( vec3( x, y, z ) ) );
	return mix(
		mix( mix( at( 0, 0, 0 ), at( 1, 0, 0 ), w.x ), mix( at( 0, 1, 0 ), at( 1, 1, 0 ), w.x ), w.y ),
		mix( mix( at( 0, 0, 1 ), at( 1, 0, 1 ), w.x ), mix( at( 0, 1, 1 ), at( 1, 1, 1 ), w.x ), w.y ), w.z );

} ).setLayout( { name: 'ponlaiNightNoise3', type: 'float', inputs: [ { name: 'p', type: 'vec3' } ] } );

// The Milky Way's floor: the mask's darkest open sky (about its median) holds the unresolved
// light of faint stars everywhere; the stylised night keeps that sky dark.
const MILKY_WAY_FLOOR = 0.03;

// Brightness of the Milky Way (0 to about 2) in the equatorial direction c, from its mask value.
// Its shape, the bulge toward Sagittarius, the Great Rift and the dark clouds come from the
// NASA/Gaia mask; procedural grain breaks it up finer than the mask's third of a degree, so it
// reads as unresolved stars rather than a blurred photograph. `fine` is 0 for blurred lookups
// (reflections, bakes), which skip the small-scale structure.
const milkyWay = Fn( ( [ c, mask, fine ]: [ V3, F, F ] ) => {

	const light = max( mask.mul( mask ).sub( MILKY_WAY_FLOOR ), 0 ).div( 1 - MILKY_WAY_FLOOR );
	// A gentle lift keeps the faint outer band visible beside the bright core.
	const result = pow( light, 0.75 ).mul( 2.2 ).toVar();
	// The grain costs four noise lookups; most of the sky lies outside the band.
	If( result.greaterThan( 0.01 ).and( fine.greaterThan( 0.5 ) ), () => {

		const clouds = noise3( c.mul( 61 ).add( 7.7 ) ).sub( 0.5 ).mul( 0.2 ).add( 1 );
		// Unresolved stars: a fine mottling over the clouds.
		const grain = noise3( c.mul( 170 ).add( 4.4 ) ).mul( 0.55 ).add( noise3( c.mul( 390 ).add( 8.8 ) ).mul( 0.45 ) );
		result.mulAssign( clouds.mul( grain.mul( 0.5 ).add( 0.75 ) ) );

	} );
	return result;

} ).setLayout( { name: 'ponlaiMilkyWay', type: 'float', inputs: [ { name: 'c', type: 'vec3' }, { name: 'mask', type: 'float' }, { name: 'fine', type: 'float' } ] } );

// One byte of the Milky Way mask, as 0 to 1.
const maskTexel = ( x: THREE.Node<'int'>, y: THREE.Node<'int'> ) => {

	const index = y.mul( MILKY_WAY_WIDTH ).add( x ).toVar();
	const word = skyWords.element( index.shiftRight( 2 ).add( CELL_COUNT ) );
	return float( word.shiftRight( uint( index.bitAnd( 3 ).mul( 8 ) ) ).bitAnd( uint( 255 ) ) ).div( 255 );

};

// The mask bilinearly filtered in the equatorial direction c (its layout: right ascension 0h at
// the centre, increasing to the left; declination +90 on the first row), wrapping in right
// ascension. Read here rather than in the shader function above, which cannot bind buffers.
const milkyWayMaskAt = ( c: V3 ) => {

	const x = fract( float( 0.5 ).sub( atan( c.y, c.x ).div( 2 * Math.PI ) ) ).mul( MILKY_WAY_WIDTH ).sub( 0.5 ).toVar();
	const y = acos( clamp( c.z, - 1, 1 ) ).div( Math.PI ).mul( MILKY_WAY_HEIGHT ).sub( 0.5 ).toVar();
	const x0 = floor( x ).toVar(), y0 = floor( y ).toVar();
	const fx = x.sub( x0 ), fy = y.sub( y0 );
	const left = int( x0 ).bitAnd( MILKY_WAY_WIDTH - 1 ).toVar();
	const right = left.add( 1 ).bitAnd( MILKY_WAY_WIDTH - 1 ).toVar();
	const top = int( clamp( y0, 0, MILKY_WAY_HEIGHT - 1 ) ).toVar();
	const bottom = int( clamp( y0.add( 1 ), 0, MILKY_WAY_HEIGHT - 1 ) ).toVar();
	return mix( mix( maskTexel( left, top ), maskTexel( right, top ), fx ),
		mix( maskTexel( left, bottom ), maskTexel( right, bottom ), fx ), fy );

};

// Footprint (radians across a pixel's derivative) at which a baked star carries the energy of a
// star drawn on screen. On screen a star stays a fixed fraction of a pixel wide and keeps its
// brightness at any resolution or zoom, as the eye sees a point; bakes conserve its energy.
const STAR_REFERENCE_FOOTPRINT = 0.0015;
// Star image width in footprints: a little under a pixel, wide enough that a star drifting with
// the sky's rotation keeps its brightness as it crosses pixels instead of shimmering.
const STAR_WIDTH = 0.5;
// Stars brighter than about first magnitude (in the catalogue's brightness units) get a coloured
// halo BRIGHT_STAR_HALO_WIDTH times the star's width, of a fixed faint peak BRIGHT_STAR_HALO: it
// shows the colour the clipped core has lost, without swelling the brightest into blobs.
const BRIGHT_STAR_HALO_START = 0.4, BRIGHT_STAR_HALO_WIDTH = 4, BRIGHT_STAR_HALO = 0.05;

// The face of a cube around the celestial sphere that c points through, and c's cell in a grid of
// cells x cells on that face. StarCatalog.ts buckets the catalogue with the same selection.
const cubeCell = ( c: V3, cells: number ) => {

	const a = abs( c ).toVar();
	const xMajor = a.x.greaterThanEqual( a.y ).and( a.x.greaterThanEqual( a.z ) );
	const yMajor = a.y.greaterThanEqual( a.z );
	// Face index and in-face coordinates in [-1, 1].
	const face = select( xMajor, select( c.x.greaterThan( 0 ), float( 0 ), float( 1 ) ),
		select( yMajor, select( c.y.greaterThan( 0 ), float( 2 ), float( 3 ) ), select( c.z.greaterThan( 0 ), float( 4 ), float( 5 ) ) ) ).toVar();
	const major = select( xMajor, a.x, select( yMajor, a.y, a.z ) );
	const uv = select( xMajor, c.zy, select( yMajor, c.xz, c.xy ) ).div( major ).toVar();
	const cell = min( floor( uv.mul( 0.5 ).add( 0.5 ).mul( cells ) ), cells - 1 ).toVar();
	return { xMajor, yMajor, face, cell };

};

// A star image: a Gaussian a little under a pixel wide around its direction. Bakes widen it to
// their own footprint and keep its energy.
const starProfile = ( c: V3, starDir: V3, footprint: F, conserve: boolean ) => {

	const variance = footprint.mul( footprint ).mul( STAR_WIDTH * STAR_WIDTH ).toVar();
	const separation = c.sub( starDir ).toVar();
	const profile = exp( dot( separation, separation ).div( variance ).mul( - 0.5 ) ).toVar();
	if ( conserve ) profile.mulAssign( float( STAR_REFERENCE_FOOTPRINT * STAR_REFERENCE_FOOTPRINT * STAR_WIDTH * STAR_WIDTH ).div( variance ) );
	return profile;

};

// Only low stars scintillate, seen through the thick air near the horizon: an irregular flicker,
// not a steady pulse; high stars burn steadily. seed and phase are two per-star constants.
const scintillation = ( time: F, elevation: F, seed: F, phase: F ) => {

	const flicker = nightNoise( vec2( time.mul( 2.3 ).add( seed.mul( 57 ) ), phase.mul( 91 ) ) ).sub( 0.5 );
	return flicker.mul( float( 1 ).sub( smoothstep( 0.02, 0.2, elevation ) ) ).mul( 0.4 ).add( 1 );

};

// The real stars, from the Yale Bright Star Catalogue: the stars of the cell c falls in
// (StarCatalog.ts lists each star in every cell its image reaches).
const catalogueStars = ( c: V3, footprint: F, conserve: boolean, time: F, elevation: F ) => {

	const radiance = vec3( 0 ).toVar();
	const { face, cell } = cubeCell( c, STAR_CELLS );
	const packed = skyWords.element( int( cell.y ).mul( 6 ).add( int( face ) ).mul( STAR_CELLS ).add( int( cell.x ) ) ).toVar();
	const start = int( packed.shiftRight( uint( Math.log2( CELL_SLOTS ) ) ) ).toVar();
	const count = int( packed.bitAnd( uint( CELL_SLOTS - 1 ) ) ).toVar();
	Loop( { start: int( 0 ), end: count, type: 'int', condition: '<' }, ( { i } ) => {

		const record = start.add( i ).mul( 2 ).toVar();
		const star = starRecords.element( record ).toVar();
		const profile = starProfile( c, star.xyz, footprint, conserve );
		If( profile.greaterThan( 1e-4 ), () => {

			const look = starRecords.element( record.add( 1 ) ).toVar();
			radiance.addAssign( look.rgb.mul( star.w ).mul( profile ).mul( scintillation( time, elevation, look.w, star.w ) ) );

		} );
		// The brightest stars' cores clip to white on screen and lose their colour; a faint halo a
		// few pixels wide, in their colour deepened, lets Antares read orange and Rigel blue.
		if ( ! conserve ) If( star.w.greaterThan( BRIGHT_STAR_HALO_START ), () => {

			const look = starRecords.element( record.add( 1 ) ).toVar();
			const halo = starProfile( c, star.xyz, footprint.mul( BRIGHT_STAR_HALO_WIDTH ), false );
			const hue = pow( look.rgb.div( max( look.rgb.x, max( look.rgb.y, look.rgb.z ) ) ), vec3( 2.5 ) );
			const strength = smoothstep( BRIGHT_STAR_HALO_START, BRIGHT_STAR_HALO_START * 3, star.w ).mul( BRIGHT_STAR_HALO );
			radiance.addAssign( hue.mul( halo ).mul( strength ) );

		} );

	} );
	return radiance;

};

// A procedural layer of stars too faint for the catalogue: at most one star per cube cell, kept
// away from the cell's edges, so a direction only ever needs its own cell. Brightness follows the
// sky's own statistics: every magnitude brighter holds about a third as many stars. `faintest`
// is the dimmest star's peak and `range` how many times brighter the brightest may be.
const starLayer = ( c: V3, footprint: F, conserve: boolean, cells: number, occupancy: F, seed: number, faintest: number, range: number, time: F, elevation: F ) => {

	const radiance = vec3( 0 ).toVar();
	const { xMajor, yMajor, face, cell } = cubeCell( c, cells );
	const key = cell.add( vec2( face.mul( 1031 ).add( seed ), face.mul( 557 ).sub( seed ) ) ).toVar();
	If( nightHash( key ).lessThan( occupancy ), () => {

		const jitter = vec2( nightHash( key.add( vec2( 29.3, 71.7 ) ) ), nightHash( key.add( vec2( 83.1, 17.6 ) ) ) ).mul( 0.6 ).add( 0.2 );
		const st = cell.add( jitter ).div( cells ).mul( 2 ).sub( 1 ).toVar();
		const onFace = select( xMajor, vec3( select( c.x.greaterThan( 0 ), float( 1 ), float( - 1 ) ), st.y, st.x ),
			select( yMajor, vec3( st.x, select( c.y.greaterThan( 0 ), float( 1 ), float( - 1 ) ), st.y ),
				vec3( st.x, st.y, select( c.z.greaterThan( 0 ), float( 1 ), float( - 1 ) ) ) ) );
		// Counts grow by about 2.8 per magnitude, so the brightness of a uniform draw u is u^-0.9.
		const draw = max( nightHash( key.add( vec2( 41.7, 113.9 ) ) ), 1e-4 ).toVar();
		const brightness = min( pow( draw, - 0.9 ), range ).mul( faintest );
		const profile = starProfile( c, normalize( onFace ), footprint, conserve );
		const temperature = nightHash( key.add( vec2( 197.1, 7.3 ) ) ).toVar();
		const color = mix( mix( vec3( 1, 0.72, 0.5 ), vec3( 1, 0.95, 0.88 ), smoothstep( 0, 0.35, temperature ) ),
			vec3( 0.7, 0.82, 1 ), smoothstep( 0.55, 1, temperature ) );
		radiance.assign( color.mul( brightness ).mul( profile ).mul( scintillation( time, elevation, temperature, draw ) ) );

	} );
	return radiance;

};

// The lunar disk follows takram's MoonNode (three-geospatial): the view ray meets a sphere for
// the surface normal, which looks up NASA's colour map and is bent by the LOLA heights through
// a screen-space cotangent frame, and an Oren-Nayar surface of roughness 1 is lit by the sun.

// The kit's mean lunar radius in metres: heights over it give slopes on the unit sphere.
const LUNAR_RADIUS = 1737400;
// Mean linear luminance of the colour map's near side, weighted by projected area on the disk
// (printed by tools/island/build-moon-textures.py). Dividing by it keeps the disk's brightness in
// moonColor, so the map only redistributes it into maria, highlands and craters.
const NEAR_SIDE_LUMINANCE = 0.397;
// Oren-Nayar (roughness 1) radiance at normal incidence and view: the full moon's centre.
const OREN_NAYAR_A = ( 1 / Math.PI ) * ( 1 - 0.5 * ( 1 / 1.33 ) + 0.17 * ( 1 / 1.13 ) );
const OREN_NAYAR_B = ( 1 / Math.PI ) * ( 0.45 * ( 1 / 1.09 ) );
// Scales the textured disk so a full moon sends as much light as the painted disk it replaced
// (within 1 % over the disk): the moon's brightness and the night grade were tuned against it.
const MOON_BRIGHTNESS = 0.72;
// Past first quarter the sunlit side turns away and the brightest point of the disk dims, so a
// physically shaded crescent reads as a grey sliver. The eye, adapted to the little crescent,
// sees it bright: this share of that loss is given back (an art direction; 0 is physical).
const CRESCENT_LIFT = 0.7;
// The colour map exaggerates the moon's tint (warm highlands, bluish maria); the eye sees a
// near-neutral silver grey, so only this share of the map's colour is kept.
const MOON_SATURATION = 0.4;
// Near full, the disk alone is drawn this much dimmer than the light it sends (the key light and
// the glow keep their level): at full brightness the curve clipped the highlands to white
// and the maria lost their contrast. Quarters and crescents keep their brightness.
const FULL_MOON_EXPOSURE = 0.65;
// Earthshine, after a-starry-sky: the faint, bluish light of the sunlit Earth on the lunar
// night side. FULL_EARTHSHINE is a-starry-sky's strength under a full Earth (at new moon, half
// of 0.19) against its sunlit surface, which is about twice the albedo; the colour is its
// blue-biased Earth albedo.
const FULL_EARTHSHINE = 0.5 * 0.19 * 0.5;
const EARTHSHINE_COLOR = new THREE.Vector3( 0.18, 0.30, 0.55 );
// Steepest slope the relief may give the normal (about 22 degrees); the height map's own slopes
// rarely pass 0.17.
const MAX_MOON_SLOPE = 0.4;
// Largest map footprint per pixel over the footprint at the disk's centre.
const MOON_FOOTPRINT_CAP = 3;
// Squared chord between the view ray and the moon's centre at the disk's edge.
const MOON_CHORD = 2 * ( 1 - Math.cos( MOON_ANGULAR_RADIUS ) );

// Unit normal of the near side of a sphere of the given angular radius where the view ray
// meets it. Off the disk it continues smoothly (unnormalised), so screen derivatives stay
// meaningful across the rim.
const raySphereNormal = Fn( ( [ ray, center, angularRadius ]: [ V3, V3, F ] ) => {

	const cosRay = dot( center, ray );
	// From the centre to its projection on the ray, and the half chord along the ray.
	const p = ray.mul( cosRay ).sub( center ).toVar();
	const s = sqrt( max( angularRadius.mul( angularRadius ).sub( dot( p, p ) ), 0 ) );
	return p.sub( ray.mul( s ) ).div( angularRadius );

} ).setLayout( { name: 'ponlaiMoonNormal', type: 'vec3', inputs: [
	{ name: 'ray', type: 'vec3' }, { name: 'center', type: 'vec3' }, { name: 'angularRadius', type: 'float' },
] } );

// Oren-Nayar diffuse with roughness 1 and albedo 1 (Yasuhiro Fujii's improved form,
// https://mimosa-pudica.net/improved-oren-nayar.html): the dusty regolith keeps a full moon
// nearly flat to its limb instead of the darkening of a Lambertian ball.
const orenNayar = Fn( ( [ light, view, normal ]: [ V3, V3, V3 ] ) => {

	const cosLight = dot( normal, light ).toVar();
	const cosView = dot( normal, view ).toVar();
	const s = dot( light, view ).sub( cosLight.mul( cosView ) ).toVar();
	// Avoids a seam at the edge, where max( cosLight, cosView ) goes to zero.
	const t = mix( float( 1 ), max( max( cosLight, cosView ), 0.1 ), smoothstep( 0, 0.1, s ) );
	return max( cosLight, 0 ).mul( s.div( t ).mul( OREN_NAYAR_B ).add( OREN_NAYAR_A ) );

} ).setLayout( { name: 'ponlaiMoonOrenNayar', type: 'float', inputs: [
	{ name: 'light', type: 'vec3' }, { name: 'view', type: 'vec3' }, { name: 'normal', type: 'vec3' },
] } );

// How much sunlit Earth the moon sees: the Lommel-Seeliger phase law from a-starry-sky's
// earthshine, taken at Earth's phase angle as seen from the moon, which is the moon's
// elongation from the sun (zero, a full Earth, at new moon).
const earthPhaseLaw = Fn( ( [ elongation ]: [ F ] ) => {

	const x = clamp( elongation.mul( 0.5 ), 1e-3, Math.PI / 2 - 1e-3 ).toVar();
	return clamp( float( 1 ).sub( sin( x ).mul( tan( x ) ).mul( log( float( 1 ).div( tan( x.mul( 0.5 ) ) ) ) ) ), 0, 1 );

} ).setLayout( { name: 'ponlaiEarthPhaseLaw', type: 'float', inputs: [ { name: 'elongation', type: 'float' } ] } );

// Forward scattering by haze around the moon: a tight aureole and a wide, faint halo, scaled
// by how much of the disk is lit. Painted on the dome, so it stays off by default (`glow` 0): the
// air's haze scatters the moonlight around the disk itself (VolumetricHaze.ts, the moon's aerosol
// share), and this is kept for a touch of aureole within a degree or two of the disk that the
// haze's grid cannot resolve.
const nightMoonGlow = Fn( ( [ direction, sunDir, moonDir, moonColor ]: [ V3, V3, V3, V3 ] ) => {

	const angle = acos( clamp( dot( direction, moonDir ), - 1, 1 ) ).toVar();
	const lit = float( 1 ).sub( dot( sunDir, moonDir ) ).mul( 0.5 );
	const glow = exp( angle.div( - 0.035 ) ).mul( 0.12 ).add( exp( angle.div( - 0.3 ) ).mul( 0.015 ) );
	return moonColor.mul( glow ).mul( lit ).mul( smoothstep( - 0.05, 0.05, moonDir.y ) );

} ).setLayout( { name: 'ponlaiNightMoonGlow', type: 'vec3', inputs: [
	{ name: 'direction', type: 'vec3' }, { name: 'sunDir', type: 'vec3' },
	{ name: 'moonDir', type: 'vec3' }, { name: 'moonColor', type: 'vec3' },
] } );

export class NightSky {

	// the sunlight on the drawn moon: the direction its phase lights it from (DayCycle
	// `moonSunDir`), so the disk shows the phase whatever the hour
	readonly sunDir = uniform( new THREE.Vector3( 1, 0, 0 ) );
	readonly moonDir = uniform( new THREE.Vector3( - 1, 0, 0 ) );
	// Full illuminated disk radiance in scene units. Phase comes only from the directions.
	readonly moonColor = uniform( new THREE.Color( 0.035, 0.04, 0.0475 ) );
	readonly starVisibility = uniform( 0 );
	// World to celestial frame: fixed, the pole STARS.latitude up on the side away from noon
	// (sky/DayCycle.ts STARS).
	readonly frame = uniform( new THREE.Matrix3() );
	readonly time = uniform( 0 );
	// A bright moon's scattered light washes out the Milky Way and the faintest stars.
	readonly moonWash = uniform( 0 );
	// Strength of the painted glow around the moon (`nightMoonGlow`; 0, the default, leaves the
	// disk and the haze's own glow).
	readonly glow = tuneUniform( 'sky.night.moonGlow', 0.00068, { min: 0, max: 1, onChange: markSkyRetune } );
	// Scale of the lunar relief over its true height (an art control; 0 leaves a smooth ball).
	readonly relief = uniform( 1.4 );
	// Strength of the earthshine over a-starry-sky's (an art control).
	readonly earthshine = uniform( 1 );
	// The near side's colour and LOLA heights (see tools/island/build-moon-textures.py). Loading
	// starts with the sky; the first bake waits for it, so no frame shows an untextured moon.
	readonly colorMap: THREE.Texture;
	readonly heightMap: THREE.Texture;
	private readonly loading: Promise<void>;

	constructor() {

		const loader = new THREE.TextureLoader();
		const base = assetUrl( 'sky/moon/' );
		const load = ( name: string, colorSpace: string ): [ THREE.Texture, Promise<void> ] => {

			let map!: THREE.Texture;
			const loaded = new Promise<void>( ( resolve, reject ) => {

				map = loader.load( base + name, () => resolve(), undefined, () => reject( new Error( `Moon texture ${ name } failed to load` ) ) );

			} );
			map.colorSpace = colorSpace;
			// Longitude wraps; latitude ends at the poles.
			map.wrapS = THREE.RepeatWrapping;
			// The map stretches toward its poles, and the disk toward its limb.
			map.anisotropy = 4;
			return [ map, loaded ];

		};
		const [ colorMap, colorLoaded ] = load( 'moon_color.jpg', THREE.SRGBColorSpace );
		const [ heightMap, heightLoaded ] = load( 'moon_height.png', THREE.NoColorSpace );
		this.colorMap = colorMap;
		this.heightMap = heightMap;
		this.loading = Promise.all( [ colorLoaded, heightLoaded ] ).then( () => undefined );

	}

	get ready(): boolean { return this.loaded; }
	private loaded = false;

	/** Load the star catalogue and the Milky Way mask; the sky waits for them before its first bake. */
	async init(): Promise<void> {

		await loadNightSkyData();
		this.loaded = true;

	}

	stars( direction: V3, angularFootprint?: F, visibility: F = this.starVisibility ): V3 {

		// The dome derives its pixel footprint before varying control flow. Reflection and
		// bake callers may supply a roughness/texel footprint explicitly, avoiding derivatives
		// in their branches. A separate bake visibility keeps future endpoints off the dome.
		const footprint = max( angularFootprint ?? length( fwidth( direction ) ), 1e-5 ).toVar();
		const conserve = angularFootprint !== undefined;
		const radiance = vec3( 0 ).toVar();
		// (the visibility carries the night sky's level, which scales inversely to the night's
		// exposure, so a deep night's stars draw at a visibility of about 1e-4: only a sky without
		// stars, the day's, skips them)
		const v = clamp( visibility, 0, 1 ).toVar();
		If( v.greaterThan( 0 ), () => {

			const c = normalize( this.frame.mul( direction ) ).toVar();
			const band = milkyWay( c, milkyWayMaskAt( c ), float( conserve ? 0 : 1 ) ).toVar();
			const elevation = direction.y;
			// The catalogue's nine thousand real stars, down to the edge of naked-eye visibility;
			// below them a dense procedural field thickening along the Milky Way; and the band's
			// own unresolved glow.
			const bright = catalogueStars( c, footprint, conserve, this.time, elevation );
			const wash = clamp( this.moonWash, 0, 1 );
			// Two procedural layers below the catalogue: a field everywhere, and a dense dust of
			// barely resolved stars that only the Milky Way's star clouds hold, so the band reads
			// as stars rather than mist.
			const faint = starLayer( c, footprint, conserve, 180, band.mul( 0.4 ).add( 0.05 ), 17, 0.004, 3, this.time, elevation )
				.add( starLayer( c, footprint, conserve, 400, band.mul( 0.6 ), 29, 0.004, 3, this.time, elevation ) )
				.mul( float( 1 ).sub( wash.mul( 0.7 ) ) );
			// Blue-white star clouds, warming only in the bulge toward Sagittarius.
			const glow = mix( vec3( 0.66, 0.75, 1 ), vec3( 1, 0.88, 0.74 ), smoothstep( 0.9, 2, band ) ).mul( band ).mul( 0.03 )
				.mul( float( 1 ).sub( wash.mul( 0.88 ) ) );
			// Thick air near the horizon dims the stars and the band.
			const horizon = smoothstep( - 0.01, 0.12, elevation );
			radiance.assign( bright.add( faint ).add( glow ).mul( v ).mul( horizon ).mul( SCENE_RADIANCE_SCALE ) );

		} );
		return radiance;

	}

	/** 1 except over the moon's disk, which hides the stars behind it. */
	moonOcclusion( direction: V3 ): F {

		const facing = dot( direction, this.moonDir );
		const radius = length( direction.sub( this.moonDir.mul( facing ) ) ).div( MOON_ANGULAR_RADIUS );
		return select( facing.greaterThan( 0 ), smoothstep( 0.98, 1.06, radius ), float( 1 ) );

	}

	/** Resolves once the moon's colour and height maps have loaded. */
	load(): Promise<void> {

		return this.loading;

	}

	moon( direction: V3 ): V3 {

		return this.lunarDisk( direction )
			.add( nightMoonGlow( direction, this.sunDir, this.moonDir, this.moonColor.rgb ).mul( this.glow ) );

	}

	// After takram's MoonNode: the disk's normal from a ray/sphere intersection, NASA's colour,
	// LOLA relief through a screen-space cotangent frame, Oren-Nayar shading and a limb
	// antialiased by the chord's screen derivative. The phase still comes only from the sun and
	// moon directions, and lunar north stays up as before.
	private lunarDisk( direction: V3 ): V3 {

		return Fn( () => {

			const moonDir = this.moonDir, sunDir = this.sunDir;
			// Squared chord to the moon's centre: its screen derivative gives the limb a one-pixel
			// edge at any field of view.
			const chord = direction.sub( moonDir ).toVar();
			const chordLength = dot( chord, chord ).toVar();
			const filterWidth = fwidth( chordLength ).toVar();
			// The lunar frame: x toward the observer (0 degrees longitude), y lunar east (the
			// observer's right), z lunar north. The normal, its map coordinates and their screen
			// derivatives are all taken here in uniform control flow; past the rim the normal
			// continues smoothly, so the derivatives hold up to the edge.
			const up = select( abs( moonDir.y ).greaterThan( 0.98 ), vec3( 0, 0, 1 ), vec3( 0, 1, 0 ) );
			const east = normalize( cross( moonDir, up ) ).toVar();
			const north = cross( east, moonDir ).toVar();
			const normal = raySphereNormal( direction, moonDir, float( MOON_ANGULAR_RADIUS ) ).toVar();
			const local = vec3( dot( normal, moonDir ).negate(), dot( normal, east ), dot( normal, north ) ).toVar();
			const uv = vec2( atan( local.y, local.x ).mul( 0.5 / Math.PI ).add( 0.5 ),
				asin( clamp( local.z, - 1, 1 ) ).mul( 1 / Math.PI ).add( 0.5 ) ).toVar();
			// The map's footprint grows without bound toward the limb and the poles. Capped at a few
			// times the footprint at the disk's centre, the limb keeps its own albedo instead of the
			// whole map's average from the smallest mips.
			const cap = length( fwidth( direction ) ).mul( MOON_FOOTPRINT_CAP / ( Math.PI * MOON_ANGULAR_RADIUS ) ).toVar();
			const footprint = ( d: V2 ) => d.mul( min( float( 1 ), cap.div( max( length( d ), 1e-9 ) ) ) );
			const uvdx = footprint( dFdx( uv ) ).toVar(), uvdy = footprint( dFdy( uv ) ).toVar();
			const ndx = dFdx( local ).toVar(), ndy = dFdy( local ).toVar();
			const radiance = vec3( 0 ).toVar();
			If( chordLength.lessThan( MOON_CHORD ), () => {

				// Heights in lunar radii, from the map's two filtered bytes (see the build script).
				const height = ( offset: V2 ) => {

					const h = texture( this.heightMap, uv.add( offset ) ).grad( uvdx, uvdy );
					return h.r.mul( 65280 ).add( h.g.mul( 255 ) ).mul( 0.5 / LUNAR_RADIUS );

				};
				const hdx = height( uvdx ).sub( height( uvdx.negate() ) ).mul( 0.5 );
				const hdy = height( uvdy ).sub( height( uvdy.negate() ) ).mul( 0.5 );
				// Cotangent frame: the surface gradient from the screen derivatives of the position
				// on the unit sphere, which is its normal.
				const r1 = cross( ndy, local ).toVar();
				const r2 = cross( local, ndx ).toVar();
				const det = dot( ndx, r1 ).toVar();
				const gradient = r1.mul( hdx ).add( r2.mul( hdy ) ).mul( sign( det ) );
				// The tilt is the gradient over |det|. A small disk's rim pixels share their quad with
				// sky pixels, where the frame can collapse (det near 0) and throw the normal toward the
				// sun; the true slopes at this map's scale stay far below the cap.
				const tilt = gradient.mul( this.relief ).div( max( abs( det ), 1e-20 ) ).toVar();
				const bumped = normalize( local.sub( tilt.mul( min( float( 1 ), float( MAX_MOON_SLOPE ).div( max( length( tilt ), 1e-9 ) ) ) ) ) );
				// Fade the relief toward the limb, where the derivatives grow without bound.
				const n = mix( local, bumped, smoothstep( 0, 0.3, dot( normal, direction ).negate() ) ).toVar();
				const surface = moonDir.mul( n.x.negate() ).add( east.mul( n.y ) ).add( north.mul( n.z ) ).toVar();
				// The colour map's luminance carries the albedo's pattern; moonColor its level.
				const map = texture( this.colorMap, uv ).grad( uvdx, uvdy ).rgb.toVar();
				const grey = dot( map, vec3( 0.2126, 0.7152, 0.0722 ) );
				const albedo = mix( vec3( grey ), map, MOON_SATURATION ).mul( MOON_BRIGHTNESS / NEAR_SIDE_LUMINANCE ).toVar();
				// The disk's brightest point at this phase: the subsolar point while it is in view,
				// then the bright limb, where the Oren-Nayar term above reduces to this.
				const cosPhase = dot( sunDir, moonDir ).negate().toVar();
				const peak = select( cosPhase.greaterThan( 0 ), float( 1 ),
					sqrt( max( float( 1 ).sub( cosPhase.mul( cosPhase ) ), 0 ) ).mul( cosPhase.mul( OREN_NAYAR_B / OREN_NAYAR_A ).add( 1 ) ) );
				const lift = float( 1 ).div( mix( float( 1 ), max( peak, 0.05 ), CRESCENT_LIFT ) );
				const sunlit = orenNayar( sunDir, direction.negate(), surface ).mul( lift.mul( 1 / OREN_NAYAR_A ) );
				// Earthshine on the night side (a-starry-sky): strongest under a full Earth at new
				// moon, gone on the sunlit side and in the sun's glare close to conjunction.
				const elongation = acos( clamp( cosPhase.negate(), - 1, 1 ) ).toVar();
				const night = float( 1 ).sub( smoothstep( 0, 0.1, dot( surface, sunDir ) ) );
				const earthshine = vec3( EARTHSHINE_COLOR.x, EARTHSHINE_COLOR.y, EARTHSHINE_COLOR.z )
					.mul( earthPhaseLaw( elongation ).mul( night ).mul( smoothstep( 0.1, 0.35, elongation ) ).mul( this.earthshine ).mul( FULL_EARTHSHINE ) );
				// (Reversed edges, as takram writes it, are undefined for smoothstep in WGSL.)
				const limb = float( 1 ).sub( smoothstep( float( MOON_CHORD ).sub( filterWidth ), MOON_CHORD, chordLength ) );
				const horizon = smoothstep( - MOON_ANGULAR_RADIUS, 0.015, direction.y );
				// At exact conjunction, roundoff in the tangent plane can leave a faint positive
				// incidence at the rim. Suppress only this vanishing phase interval.
				const phase = smoothstep( 0, 0.0002, float( 1 ).sub( dot( sunDir, moonDir ) ).mul( 0.5 ) );
				const exposure = mix( float( 1 ), float( FULL_MOON_EXPOSURE ), smoothstep( 0, 0.9, cosPhase ) );
				radiance.assign( this.moonColor.rgb.mul( albedo ).mul( exposure ).mul( sunlit.mul( phase ).add( earthshine ) ).mul( limb ).mul( horizon ) );

			} );
			return radiance;

		} )();

	}

}
