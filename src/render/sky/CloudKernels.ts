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
	abs, acos, asin, Break, clamp, Continue, dot, exp, float, floor, Fn, fract, If, instanceIndex,
	int, ivec2, length, Loop, max, min, mix, normalize, not, select, sin, smoothstep, sqrt, texture,
	texture3D, textureLoad, textureStore, uint, vec2, vec3, vec4,
} from 'three/tsl';
import { sampleStreaks } from './Streaks.ts';
import { RAINBOW_MAX_ANGLE } from './RainbowTable.data.ts';
import { skyRayleighPhase } from './Atmosphere.ts';
import { SUN_SETTING_RADIUS } from '../lighting/Radiometry.ts';
import { STORE_CEILING, storePress } from '../post/PreExposure.ts';
import type { Clouds } from './Clouds.ts';
import {
	AMBIENT, AP_DIST, BASE_SCALE, BASE_SHADOW, BASE_SHADOW_HEIGHT, BASE_STRENGTH, BASE_WEATHER_END,
	BOUNDS_RES, BOW_SHARE, COLUMN_CLOSE, COLUMN_DIFFUSION, coneLod, CUMULUS, DECK_MEAN_GRID,
	DECK_MEAN_RAYS, DECK_MEAN_SPAN, DENSITY, desaturate, domeToDirection, EARTH_RADIUS, EDGE,
	EROSION_SCALE, FAR, FAR_FADE, FIBRE_BROAD, FIBRE_FINE, FULL_LIGHTING_ALPHA, henyeyGreenstein,
	HORIZON_AMOUNT, HORIZON_RAMP, HORIZON_SHARE, HORIZON_START, layerDistance, LIGHT_REUSE_STEPS,
	LOBE, LOW_VEER, luminance, over, POWDER, RAIN_FALL, RAIN_FAR, RAIN_FLASH, RAIN_LUMPS, RAIN_PERIOD,
	RAIN_SIGMA, RAIN_START, RAIN_STREAK, sat, SEA_BOUNCE, SECONDARY_BOW_GAIN, SHADOW_PHASES,
	SHADOW_SIZE, SHADOW_SLOPE_FLOOR, SHADOW_SPAN, shellCoverage, SKEW, SKY_PROBES, STEP, STRATUS,
	STRATUS_CAP, STRATUS_RELIEF, STRATUS_RISE, STRATUS_SPAN, SYNOPTIC_SIZE, WEATHER_SCALE,
} from './CloudModel.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

// The clouds' kernels: density, march, panorama, shadow map, rain, cirrus and deck mean.
//
// The clouds' kernels (Clouds.ts, whose header says what they draw): the density, the march, the
// panorama, the shadow map, the rain, the cirrus and the deck's mean, built for the current tier
// settings as one function of the clouds. A name in backticks with no file is this file's, Clouds.ts's
// or CloudModel.ts's.

/** Build the march and the panorama and shadow kernels for the current tier settings; the view's
 * kernels follow on its next update. */
export function buildKernels( clouds: Clouds ): void {

	const shape = clouds.shape!, weather = clouds.weather!, bounds = clouds.bounds!, synoptic = clouds.synoptic!;
	const streakCover = clouds.streakCover!.texture;
	const erosionMean = clouds.erosionMean;
	const { viewSteps, panoramaSteps, lightTaps, rainSteps } = clouds.march;
	const { width: domeWidth, height: domeHeight, strideX, strideY } = clouds.domeSize;
	const domeTexelAngle = 2 * Math.PI / domeWidth;
	for ( const kernel of [ clouds.panoramaKernel, ...clouds.rescaleKernels ?? [], clouds.shadowKernel ] ) kernel?.dispose();
	clouds.deckMeanKernel?.dispose();
	clouds.deckMeanSumKernel?.dispose();
	const shapeNode = texture3D( shape ), weatherNode = texture( weather ), boundsNode = texture( bounds ), synopticNode = texture( synoptic );

	// More cloud toward the horizon, by distance along the ray.
	const horizonCoverage = ( t: F ) => smoothstep( HORIZON_START, HORIZON_START + HORIZON_RAMP, t ).mul( HORIZON_AMOUNT );
	// Height in the shell (0 at the base, 1 at the top) of a camera-relative point; the parabolic
	// curvature term avoids subtracting two planet radii in float.
	const shellHeight = ( p: V3 ) => p.y.add( dot( p.xz, p.xz ).div( 2 * EARTH_RADIUS ) ).sub( clouds.shellBottom ).div( clouds.shellThickness );
	const weatherUV = ( pxz: V2 ) => pxz.add( clouds.camXZ ).add( clouds.windOffset ).div( WEATHER_SCALE );
	const weatherAt = ( pxz: V2 ) => weatherNode.sample( weatherUV( pxz ) ).level( float( 0 ) ).r;
	// The cloud type at a place: the weather's (`cloudType`) moved by the weather map's second
	// channel over the weather's spread, so one sky holds flatter and taller clouds side by side.
	const typeAt = ( field: F ) => sat( clouds.cloudType.add( field.sub( 0.5 ).mul( clouds.typeSpread ) ) );
	// A term of the cloud's form at a type: stratus' at 0, cumulus' at 1, linear between, so its
	// largest and smallest over a range of types lie at the range's ends (the march's bounds).
	const formOf = ( key: keyof typeof STRATUS, type: F ): F => mix( float( STRATUS[ key ] ), float( CUMULUS[ key ] ), type );
	// The top of a column, as a height in the shell, from the weather's surplus there (the
	// weather plus the coverage less one, dilated by the shape noise): a cumulus heaps as high
	// as its surplus; a stratus stands as a slab STRATUS_CAP of the shell deep wherever it has
	// any, rounded off over STRATUS_RISE at its margins and thinner by STRATUS_RELIEF where
	// the weather is poor; the types between mix the two. It rises with the surplus at every
	// type, so the largest surplus of a cell bounds its tops.
	const topOf = ( surplus: F, type: F ): F => {

		const slab = smoothstep( 0, STRATUS_RISE, surplus ).mul( smoothstep( STRATUS_SPAN[ 0 ], STRATUS_SPAN[ 1 ], surplus ).mul( STRATUS_RELIEF ).add( 1 - STRATUS_RELIEF ) ).mul( STRATUS_CAP );
		return mix( slab, surplus, type );

	};

	// Cloud presence at p without the erosion: a smooth bound of the cloud and what the eroded
	// density needs. `lods`: the shape's and the erosion's mip levels. Its vars and the shell
	// height are declared where the caller stands (`candidateAt`) and filled in a branch
	// (`fillCandidate`), so a sample that skips the fill still reads them.
	const candidateAt = ( p: V3, coverage: F, lods: V2 ) => {

		const h = shellHeight( p ).toVar();
		return { conservative: float( 0 ).toVar(), top: float( 0 ).toVar(), floorMask: float( 0 ).toVar(), type: float( 0 ).toVar(), edge: float( EDGE ).toVar(), h, coverage, lods, position: vec3( 0 ).toVar() };

	};
	type Candidate = ReturnType<typeof candidateAt>;
	// A camera-relative point in the noise's coordinates: drifted, leaning downwind with its
	// height in the shell and evolving. Its height is the point's over the shell's base, so the
	// clouds keep their shapes while a change of weather moves the shell.
	const positionOf = ( p: V3, h: F ) => {

		const world = p.xz.add( clouds.camXZ ).add( clouds.windOffset );
		const lean = clouds.windN.mul( max( h, 0 ).mul( SKEW ) );
		return vec3( world.x.sub( lean.x ), h.mul( clouds.shellThickness ).sub( clouds.evolve ), world.y.sub( lean.y ) );

	};
	const fillCandidate = ( c: Candidate, p: V3 ) => {

		const { h, coverage, lods } = c;
		If( h.greaterThan( - 0.1 ).and( h.lessThan( 1.3 ) ).and( coverage.greaterThan( 0 ) ), () => {

			const texel = weatherNode.sample( weatherUV( p.xz ) ).level( float( 0 ) ).rg.toVar();
			const w = texel.x;
			c.type.assign( typeAt( texel.y ) );
			c.edge.assign( formOf( 'edge', c.type ) );
			// An all-one noise sample bounds the dilation: a failed bound saves the volume fetch.
			const maximumTop = topOf( w.add( coverage ).sub( 1 ).add( coverage.mul( 1.34 * BASE_STRENGTH ) ), c.type );
			// a cumulus' base needs richer weather than its body, so the bases stand apart
			const required = float( 1 ).sub( smoothstep( 0, BASE_WEATHER_END, h ) ).mul( formOf( 'baseWeather', c.type ) ).toVar();
			c.floorMask.assign( smoothstep( required.sub( 0.1 ), required, w ) );
			If( maximumTop.greaterThanEqual( h.sub( c.edge ) ).and( c.floorMask.greaterThan( 0 ) ), () => {

				c.position.assign( positionOf( p, h ) );
				const base = shapeNode.sample( c.position.div( BASE_SCALE ) ).level( lods.x ).rgb;
				const dilation = dot( base, vec3( 0.7, 0.41, 0.23 ) ).mul( BASE_STRENGTH );
				c.top.assign( topOf( w.add( coverage ).sub( 1 ).add( dilation.mul( coverage ) ), c.type ) );
				c.conservative.assign( smoothstep( c.edge.negate(), c.edge, c.top.sub( h ) ).mul( smoothstep( c.edge.negate(), c.edge, h ) ).mul( c.floorMask ) );

			} );

		} );

	};
	// The cloud: its margins eroded by the shape noise at a finer scale, the more so higher up
	// and the more the taller its type.
	const erodedDensity = ( c: Candidate ) => {

		const density = float( 0 ).toVar();
		If( c.conservative.greaterThan( 0 ), () => {

			const strength = mix( formOf( 'erosionBase', c.type ), formOf( 'erosionPeak', c.type ), sat( c.h.div( max( c.top, 0.001 ) ) ) ).toVar();
			const maximumErosion = strength.mul( c.coverage ).mul( 0.173 );
			density.assign( c.conservative );
			If( min( c.top.sub( c.h ), c.h ).lessThan( maximumErosion.add( c.edge ) ), () => {

				const erosion = dot( shapeNode.sample( c.position.div( EROSION_SCALE ) ).level( c.lods.y ).rgb, vec3( 0.113, 0.04, 0.02 ) ).mul( strength ).mul( c.coverage ).toVar();
				density.assign( smoothstep( c.edge.negate(), c.edge, c.top.sub( erosion ).sub( c.h ) ).mul( smoothstep( c.edge.negate(), c.edge, c.h.sub( erosion ) ) ).mul( c.floorMask ) );

			} );

		} );
		return density;

	};
	// The density at p: the eroded cloud, or under `cheap` its smooth bound alone (the light's
	// far taps and the sky's probes, which need no margins). A plain function, not a shared
	// helper: one read from several kernels can be emitted with another kernel's uniform names
	// (r186).
	const cloudDensity = ( p: V3, coverage: F, lods: V2, cheap: THREE.Node<'bool'> | boolean ) => {

		const c = candidateAt( p, coverage, lods );
		fillCandidate( c, p );
		if ( cheap === true ) return c.conservative;
		if ( cheap === false ) return erodedDensity( c );
		const density = c.conservative.toVar();
		If( not( cheap ), () => {

			density.assign( erodedDensity( c ) );

		} );
		return density;

	};
	// The mip levels of the shape and erosion noise for a footprint (m).
	const lodsFor = ( footprint: F ) => vec2( coneLod( footprint, BASE_SCALE ), coneLod( footprint, EROSION_SCALE ) );
	// The cloud's phase function at each order of its multiple scattering: two Henyey-Greenstein
	// lobes, forward and back, whose asymmetries halve from one order to the next (Wrenninge et
	// al. 2013; the lobes and their halving as tidewater's Sky Pro clouds take them).
	const phaseOrders = ( nu: F ) => vec3( henyeyGreenstein( nu, LOBE.forward ), henyeyGreenstein( nu, LOBE.forward / 2 ), henyeyGreenstein( nu, LOBE.forward / 4 ) ).mul( 1 - LOBE.backShare )
		.add( vec3( henyeyGreenstein( nu, LOBE.back ), henyeyGreenstein( nu, LOBE.back / 2 ), henyeyGreenstein( nu, LOBE.back / 4 ) ).mul( LOBE.backShare ) );

	const skyAt = ( rd: V3 ) => clouds.atmosphere.skyRadiance( rd, false ).add( clouds.nightAmbient.rgb );
	const heightAlongRay = ( t: F, rd: V3 ) => clouds.camY.add( rd.y.mul( t ) ).add( dot( rd.xz, rd.xz ).mul( t ).mul( t ).div( 2 * EARTH_RADIUS ) ).sub( clouds.shellBottom ).div( clouds.shellThickness );

	// Ray march the shell along rd (camera relative); `flash` adds the lightning inside the
	// cloud; `hazeEnd` ends a low layer's march where the haze closes.
	const makeMarch = ( maxSteps: number, pixelCone: F, stepCone: F, flash = false, hazeEnd = false ) => ( rd: V3, jitter: F ) => {

		const start = layerDistance( clouds.camY, rd.y, clouds.shellBottom ).toVar();
		const end = min( layerDistance( clouds.camY, rd.y, clouds.shellBottom.add( clouds.shellThickness ) ), FAR ).toVar();
		// Under a whole low layer the dome fogs the view's clouds at the march's own depth (Sky.ts
		// `domeColour`), so past the distance at which the haze lets 1/500 through, 8-bit's floor,
		// nothing the march would find shows: it ends there. Through a broken layer's gaps near
		// the horizon that spares most of the march. A high layer is fogged at the shell's middle
		// instead, which can lie nearer than its clouds, and the panorama at its own distance from
		// the sea (SkyDome.ts `bakeColour`): those march the whole shell.
		if ( hazeEnd ) If( clouds.lowShare.greaterThan( 0.999 ), () => {

			end.assign( min( end, clouds.haze.reach( rd.y, clouds.camY, max( start, 100 ), end ) ) );

		} );
		// the far fade (`farFade`): whole, it leaves nothing past its end
		If( clouds.farFade.greaterThanEqual( 1 ), () => {

			end.assign( min( end, FAR_FADE[ 1 ] ) );

		} );
		const color = vec3( 0 ).toVar(), flashColor = vec3( 0 ).toVar();
		// The march's light factors, each per unit of its light's colour and summed over the
		// march premultiplied by the opacity: the key's beam after the cloud toward it (every
		// order of its scattering, the phase included), the sky over the tops and round the
		// horizon by what the cloud above leaves of it, the light the sea sends back up and the
		// key's light the column above passes down by diffusion.
		const directLight = float( 0 ).toVar(), skyLight = float( 0 ).toVar(), horizonLight = float( 0 ).toVar();
		const bounceLight = float( 0 ).toVar(), columnLight = float( 0 ).toVar();
		const transmission = float( 1 ).toVar(), weightedDepth = float( 0 ).toVar();
		const coverageBound = shellCoverage( clouds.coverage ).toVar();
		If( rd.y.greaterThan( - 0.001 ).and( end.greaterThan( start ) ).and( coverageBound.greaterThan( 0 ) ), () => {

			const L = clouds.lightDir;
			const nu = dot( rd, L ).toVar();
			const phases = phaseOrders( nu ).toVar();
			// the powder's weight: whole looking away from the key, none toward it
			const powderWeight = float( 1 ).sub( smoothstep( 0.2, 0.95, nu ) ).mul( POWDER ).toVar();
			// the types the sky holds, for the bounds of its cells
			const typeLow = typeAt( float( 0 ) ).toVar(), typeHigh = typeAt( float( 1 ) ).toVar();
			const edgeBound = max( formOf( 'edge', typeLow ), formOf( 'edge', typeHigh ) ).toVar();
			const baseBound = min( formOf( 'baseWeather', typeLow ), formOf( 'baseWeather', typeHigh ) ).toVar();
			const sigmaScale = clouds.density.mul( DENSITY ).toVar();
			// (a heap's share of its column's light rises to a layer's as the sky closes and the
			// heaps join: COLUMN_CLOSE)
			const columnClosed = smoothstep( COLUMN_CLOSE[ 0 ], COLUMN_CLOSE[ 1 ], clouds.coverage ).toVar();
			const t = start.add( jitter.mul( STEP ) ).toVar();
			const coarse = int( 1 ).toVar(), emptyRun = int( 0 ).toVar();
			const accumulatedDepth = float( 0 ).toVar(), refineEnd = float( - 1 ).toVar();
			const cellStart = float( - 1 ).toVar(), cellEnd = float( - 1 ).toVar(), maximumWeather = float( 1 ).toVar();
			const shadowAt = float( - FAR ).toVar(), shadowDensity = float( - 1 ).toVar(), shadowCheap = int( - 1 ).toVar();
			const shadowTau = float( 0 ).toVar(), skyTau = float( 0 ).toVar();
			Loop( maxSteps, () => {

				If( t.greaterThanEqual( end ).or( transmission.lessThan( 0.003 ) ), () => {

					Break();

				} );
				If( coarse.equal( 1 ), () => {

					// Skip whole cells of the weather grid that cannot hold cloud at the ray's height.
					If( t.greaterThanEqual( cellEnd ).or( t.lessThan( cellStart ) ), () => {

						const grid = weatherUV( rd.xz.mul( t ) ).mul( BOUNDS_RES ).toVar();
						maximumWeather.assign( textureLoad( boundsNode, ivec2( floor( grid ) ).mod( BOUNDS_RES ).add( BOUNDS_RES ).mod( BOUNDS_RES ) ).r );
						const f = fract( grid ).toVar();
						const ahead = vec2( select( rd.x.greaterThanEqual( 0 ), float( 1 ).sub( f.x ), f.x ), select( rd.z.greaterThanEqual( 0 ), float( 1 ).sub( f.y ), f.y ) )
							.div( max( abs( rd.xz ).mul( BOUNDS_RES / WEATHER_SCALE ), vec2( 1e-7 ) ) );
						cellStart.assign( t );
						cellEnd.assign( min( end, t.add( max( min( ahead.x, ahead.y ), 0.05 ) ) ) );

					} );
					const turning = rd.y.negate().mul( EARTH_RADIUS ).div( max( dot( rd.xz, rd.xz ), 1e-6 ) );
					const lowest = heightAlongRay( clamp( turning, t, cellEnd ), rd );
					const highest = max( heightAlongRay( t, rd ), heightAlongRay( cellEnd, rd ) );
					const coverage = coverageBound.add( horizonCoverage( cellEnd ) );
					// the cell's highest top, at whichever end of the sky's types gives the higher
					const surplus = maximumWeather.add( coverage ).sub( 1 ).add( coverage.mul( 1.34 * BASE_STRENGTH ) ).toVar();
					const maximumTop = max( topOf( surplus, typeLow ), topOf( surplus, typeHigh ) );
					const required = float( 1 ).sub( smoothstep( 0, BASE_WEATHER_END, highest ) ).mul( baseBound );
					If( maximumTop.lessThan( lowest.sub( edgeBound ) ).or( maximumWeather.lessThanEqual( required.sub( 0.1 ) ) ), () => {

						// The march goes on past the cell at the pixel's own offset into a step:
						// resumed at the cell's edge itself, every ray through the cell restarted on
						// the same plane, their steps fell in phase across the pixels, and a sharp
						// base showed the steps as rows of bands.
						t.assign( cellEnd.add( max( float( STEP ), cellEnd.mul( stepCone ).mul( 1.5 ) ).mul( jitter ).add( 0.05 ) ) );
						Continue();

					} );

				} );
				const footprint = t.mul( pixelCone );
				const fineStep = max( float( STEP ), t.mul( stepCone ).mul( 1.5 ) ).toVar();
				const p = vec3( rd.x.mul( t ), clouds.camY.add( rd.y.mul( t ) ), rd.z.mul( t ) ).toVar();
				const coverage = coverageBound.add( horizonCoverage( t ) ).toVar();
				const lods = lodsFor( footprint ).toVar();
				const c = candidateAt( p, coverage, lods );
				fillCandidate( c, p );
				const density = erodedDensity( c ).toVar();
				If( coarse.equal( 1 ), () => {

					// Coarse steps until cloud is found, then back up and refine.
					If( density.greaterThan( 0 ), () => {

						refineEnd.assign( t );
						t.assign( min( max( start, t.sub( fineStep.mul( 4 ) ) ).add( fineStep ), refineEnd ) );
						coarse.assign( 0 );
						emptyRun.assign( 0 );

					} ).Else( () => {

						t.addAssign( fineStep.mul( 4 ) );

					} );
					Continue();

				} );
				If( density.lessThanEqual( 0 ), () => {

					shadowAt.assign( - FAR );
					emptyRun.addAssign( 1 );
					If( t.lessThan( refineEnd ), () => {

						t.assign( min( t.add( fineStep ), refineEnd ) );

					} ).Else( () => {

						If( emptyRun.greaterThanEqual( 4 ), () => {

							coarse.assign( 1 );

						} );
						t.addAssign( select( coarse.equal( 1 ), fineStep.mul( 4 ), fineStep ) );

					} );
					Continue();

				} );
				emptyRun.assign( 0 );
				refineEnd.assign( - 1 );
				// (thinned toward the far fade's end: the extinction alone, so a thinned cloud
				// keeps its light)
				const sigmaT = density.mul( sigmaScale ).mul( float( 1 ).sub( smoothstep( FAR_FADE[ 0 ], FAR_FADE[ 1 ], t ).mul( clouds.farFade ) ) ).toVar();
				const surfaceStep = clamp( float( 0.5 ).div( max( sigmaT, 1e-6 ) ), fineStep.mul( 0.15 ), fineStep );
				const stepLength = min( mix( surfaceStep, fineStep.mul( 3 ), smoothstep( 1, 3, accumulatedDepth ) ), end.sub( t ) ).toVar();
				const height = sat( c.h ).toVar();
				const stepT = exp( sigmaT.mul( stepLength ).negate() ).toVar();
				const cheap = float( 1 ).sub( transmission ).greaterThanEqual( FULL_LIGHTING_ALPHA ).toVar();
				const reuse = min( fineStep.mul( LIGHT_REUSE_STEPS ), Math.max( 20, EROSION_SCALE * 0.03 ) );
				If( abs( t.sub( shadowAt ) ).greaterThanEqual( reuse ).or( abs( density.sub( shadowDensity ) ).greaterThan( 0.08 ) ).or( select( cheap, int( 1 ), int( 0 ) ).notEqual( shadowCheap ) ), () => {

					// The light march: segments growing along a narrow cone toward the key light,
					// spanning the shell's depth. It runs however faint the light, so the cloud's
					// own shadow moves without a jump while no direct light reaches it.
					const tau = float( 0 ).toVar();
					for ( let i = 0; i < lightTaps; i ++ ) {

						If( tau.lessThan( 32 ), () => {

							const tap = clouds.lightTaps[ i ];
							tau.addAssign( cloudDensity( p.add( tap.xyz ), coverage, max( lods, clouds.lightLods[ i ] ), cheap ).mul( tap.w ).mul( sigmaScale ) );

						} );

					}
					shadowTau.assign( tau );
					// The sky's way down to the sample: two reads of the cloud straight above it,
					// each standing for a stretch of the column (SKY_PROBES), a level coarser.
					const above = float( 0 ).toVar();
					for ( const [ rise, reach ] of SKY_PROBES ) above.addAssign( cloudDensity( p.add( vec3( 0, rise, 0 ) ), coverage, lods.add( 1 ), true ).mul( reach ) );
					skyTau.assign( above.mul( sigmaScale ) );
					shadowAt.assign( t );
					shadowDensity.assign( density );
					shadowCheap.assign( select( cheap, int( 1 ), int( 0 ) ) );

				} );
				// The key's light at the sample: each order of scattering takes half the optical
				// depth of the one before and half its energy, at its own phase (`phaseOrders`),
				// over the light's path and the sample's own first stretch. The powder term dims
				// the thin margins seen away from the key, which have scattered little light back
				// yet; the bases' own shade deepens toward the shell's floor.
				const local = sigmaT.mul( STEP );
				const quarter = exp( shadowTau.add( local ).mul( - 0.25 ) ).toVar(), half = quarter.mul( quarter ).toVar();
				const energy = dot( vec3( half.mul( half ), half.mul( 0.5 ), quarter.mul( 0.25 ) ), phases );
				const powder = mix( float( 1 ), float( 1 ).sub( exp( density.mul( - 2 ) ) ), powderWeight );
				const baseShadow = mix( float( 1 ), mix( float( 0.15 ), float( 1 ), smoothstep( 0, BASE_SHADOW_HEIGHT, height ) ), float( BASE_SHADOW ) );
				const alpha = transmission.mul( float( 1 ).sub( stepT ) ).toVar();
				directLight.addAssign( alpha.mul( energy ).mul( powder ).mul( baseShadow ) );
				// The sky's light: what the cloud above leaves of it, never less than a fifth (the
				// light that comes round the cloud); the sky over the tops high in the cloud, more
				// of the horizon's low in it. The sea's light back up reaches the bases.
				const visibility = float( 0.8 ).div( skyTau.add( local ).mul( 0.35 ).add( 1 ) ).add( 0.2 ).mul( alpha ).toVar();
				const rise = sqrt( height ).toVar();
				skyLight.addAssign( visibility.mul( rise.mul( HORIZON_SHARE ).add( 1 - HORIZON_SHARE ) ) );
				horizonLight.addAssign( visibility.mul( float( 1 ).sub( rise ).mul( HORIZON_SHARE ) ) );
				bounceLight.addAssign( alpha.mul( float( 1 ).sub( height ) ) );
				// The key's light the column above passes down: a conservative slab's diffuse
				// transmission (the diffusion approximation's, COLUMN_DIFFUSION) through the
				// column up to the sample's top (`candidate`'s, less the erosion's mean there:
				// the density's own form), where the orders above no longer reach (the last
				// order's share gone). Whole under a layer, a part under a heap, whose light
				// leaves by its sides (the type's `column`), and that part the sun's alone
				// (`columnSunShare`). Thin parts of a layer pass more
				// than thick ones, so its masses read darker than its seams, and its level
				// follows its own optical depth.
				const columnTau = max( min( c.top, 1 ).sub( c.coverage.mul( formOf( 'erosionPeak', c.type ) ).mul( erosionMean ) ).sub( c.h ), 0 ).mul( clouds.shellThickness ).mul( sigmaScale );
				const columnT = float( 1 ).div( columnTau.mul( COLUMN_DIFFUSION ).add( 1 ) );
				columnLight.addAssign( alpha.mul( float( 1 ).sub( quarter ) ).mul( columnT ).mul( mix( formOf( 'column', c.type ).mul( mix( float( 1 ), clouds.columnSunShare, c.type ) ), float( 1 ), columnClosed ) ).mul( clouds.columnLevel ) );
				if ( flash ) {

					If( clouds.flashOn.greaterThan( 0.5 ), () => {

						flashColor.addAssign( clouds.flashLight( p ).mul( alpha ) );

					} );

				}
				weightedDepth.addAssign( alpha.mul( t ) );
				transmission.mulAssign( stepT );
				accumulatedDepth.addAssign( sigmaT.mul( stepLength ) );
				t.addAssign( stepLength );

			} );

		} );
		const alpha = float( 1 ).sub( transmission ).toVar();
		// The factors' colours: the key's irradiance at the shell (`lightColor`); the light on the
		// layer's tops (`columnTop`: the key's on a level top and the sky's mean radiance over
		// the upper hemisphere, each over pi, and the horizon's ring) for the sky's light, the
		// sea's and the column's, which leaves the base by the diffusion law, ( 3/7 )( 1 + 2 mu ).
		If( alpha.greaterThan( 0 ), () => {

			const top = clouds.columnTop();
			const night = clouds.nightAmbient.rgb;
			const ambient = top.sky.add( night ).mul( skyLight ).add( top.horizon.add( night ).mul( horizonLight ) )
				.add( top.key.add( top.sky ).mul( vec3( ...SEA_BOUNCE ) ).mul( bounceLight ) );
			// (moved toward a uniform base under a closed sky by day, the level ground's light
			// kept: `deckFlat`)
			const slope = max( rd.y, 0 ).mul( 6 / 7 ).add( 3 / 7 ).toVar();
			const law = slope.add( float( 1 ).sub( slope ).mul( clouds.deckFlat ) );
			color.assign( clouds.lightColor.rgb.mul( directLight.mul( clouds.directGain ) )
				.add( ambient.mul( clouds.ambientGain.mul( AMBIENT ) ) )
				.add( top.key.mul( columnLight.mul( law ) ) ) );

		} );
		const depth = select( alpha.greaterThan( 1e-4 ), weightedDepth.div( max( alpha, 1e-4 ) ), start.add( 4000 ) ).toVar();
		// No air of its own: the dome fogs the clouds at their distance (Sky.ts fogDome).
		const L = color;
		// The lightning inside the cloud, kept apart (see buildView).
		const F = flashColor;
		// The march stops once 0.3 % of the light gets through: the rest counts as opaque.
		return { L, F, T: sat( transmission.sub( 0.003 ).div( 0.997 ) ), depth };

	};

	// Distant rain below the cloud base, after VolumetricCloudsCS1's rain march: grey curtains
	// marched along the view ray under the cloud that is raining. A drop at some height left
	// the base upwind of where it is now, so the cloud above is read where the drop came from
	// and the curtains lean with the wind; their streaky texture is the shape noise stretched
	// along the fall and sliding down with it. A raining deck rains everywhere, harder under
	// its richest cells; the tall heaps of a broken sky shower; and after the rain has passed
	// showers linger under the thicker cells. The curtains are lit by the light under the
	// cloud (dark under a thick deck) and by the sun where it reaches them under the clouds,
	// bright from the side; in sunlight opposite the sun they carry the rainbows. They fade
	// into the haze with distance as the clouds do.
	const rainbowNode = texture( clouds.rainbowTable! );
	// Cloud standing at the shell height h above a camera-relative point, from the weather map
	// alone (the shape noise at its mean, the weather's own type): enough to say whether the sun
	// gets through to the rain, or how thick the cloud a drop fell from is.
	const cloudAbove = ( pxz: V2, height: number ) => {

		const h = float( height );
		const w = weatherAt( pxz );
		const coverage = shellCoverage( clouds.coverage );
		const top = topOf( w.add( coverage ).sub( 1 ).add( coverage.mul( 0.45 ) ), clouds.cloudType );
		const required = float( 1 ).sub( smoothstep( 0, BASE_WEATHER_END, h ) ).mul( formOf( 'baseWeather', clouds.cloudType ) );
		const edge = formOf( 'edge', clouds.cloudType );
		return { w, top, coverage, presence: smoothstep( edge.negate(), edge, top.sub( h ) ).mul( smoothstep( required.sub( 0.1 ), required, w ) ) };

	};
	const rainAt = ( rd: V3, pxAngle: F, jitter: F, withFlash = false ) => {

		const radiance = vec3( 0 ).toVar(), flash = vec3( 0 ).toVar();
		const through = float( 1 ).toVar();
		const end = min( layerDistance( clouds.camY, rd.y, clouds.shellBottom ), RAIN_FAR ).toVar();
		// Nothing to march without rain, lingering showers or a sky broken into heaps.
		const raining = max( clouds.rain, max( clouds.showers, clouds.coverage.sub( 0.5 ).mul( clouds.heapShowers ) ) );
		If( end.greaterThan( RAIN_START * 2 ).and( rd.y.greaterThan( 0 ) ).and( raining.greaterThan( 0 ) ), () => {

			const L = clouds.lightDir;
			const cosine = dot( rd, L ).toVar();
			const sun = clouds.lightColor.rgb.mul( clouds.directGain ).toVar();
			const hasSun = L.y.greaterThan( 0.01 ).and( max( sun.x, max( sun.y, sun.z ) ).greaterThan( 1e-5 ) ).toVar();
			// Drops scatter forward far more than cloud does, and a little back by their outer
			// reflection; opposite the sun the bows add the colours of the table (relative to
			// isotropic, so the phase is that over 4 pi).
			const bowAt = acos( clamp( cosine.negate(), - 1, 1 ) ).mul( 180 / Math.PI / RAINBOW_MAX_ANGLE ).toVar();
			const secondary = smoothstep( 46 / RAINBOW_MAX_ANGLE, 48 / RAINBOW_MAX_ANGLE, bowAt ).mul( SECONDARY_BOW_GAIN - 1 ).add( 1 );
			const bow = rainbowNode.sample( vec2( min( bowAt, 1 ), 0.5 ) ).level( float( 0 ) ).rgb.mul( secondary ).mul( select( bowAt.lessThan( 1 ), clouds.rainbow, float( 0 ) ) );
			const grey = min( henyeyGreenstein( cosine, 0.75 ).mul( 4 * Math.PI ), 8 ).mul( 0.85 ).add( 0.15 );
			const phase = bow.mul( BOW_SHARE ).add( grey ).mul( 1 / ( 4 * Math.PI ) ).toVar();
			// The rain has no colour of its own: it takes the light where it falls. From above, the
			// base of the cloud it falls from, lit as the clouds light their own bases (dimmer still
			// under a closed deck); from the sides, the sky toward the horizon in the view
			// direction; from below the dark sea gives back almost nothing. Where the sun reaches
			// the drops they add its light, brightest looking toward it.
			const zenith = skyAt( vec3( 0, 1, 0 ) ).toVar();
			const horizon = clouds.columnTop().horizon.add( clouds.nightAmbient.rgb ).toVar();
			const closed = smoothstep( 0.7, 0.95, shellCoverage( clouds.coverage ) ).toVar();
			const deckLight = mix( float( 1 ), float( 0.3 ), closed ).toVar();
			const cloudBase = mix( zenith, horizon, 0.55 ).mul( AMBIENT * 0.25 ).mul( deckLight ).toVar();
			const sideLight = skyAt( normalize( vec3( rd.x.add( 1e-5 ), 0.03, rd.z ) ) ).mul( deckLight ).toVar();
			// Distant curtains fade into the air ahead: toward the sky behind them where it is
			// open, toward the deck's base where a closed deck hides it; under a deck the haze is
			// thinner too (as for the clouds themselves).
			const sky = mix( skyAt( rd ), cloudBase, closed ).toVar();
			const airScale = float( - 1 / AP_DIST ).div( closed.mul( 5 ).add( 1 ) ).toVar();
			const span = end.sub( RAIN_START ).toVar();
			const lean = clouds.windN.mul( clouds.rainSlant ).toVar();
			const fall = clouds.evolve.mul( RAIN_FALL ).toVar();
			const ly = max( L.y, 0.03 ).toVar();
			Loop( rainSteps, ( { i } ) => {

				If( through.lessThan( 0.01 ), () => {

					Break();

				} );
				// Segments lengthen with distance: a shower's edge a kilometre off needs precision,
				// the far end of a 40 km ray does not.
				const t0 = float( i ).div( rainSteps ).pow( 1.8 ).mul( span ).add( RAIN_START ).toVar();
				const dt = float( i ).add( 1 ).div( rainSteps ).pow( 1.8 ).mul( span ).add( RAIN_START ).sub( t0 ).toVar();
				const t = t0.add( dt.mul( jitter ) ).toVar();
				const altitude = clouds.camY.add( rd.y.mul( t ) ).add( dot( rd.xz, rd.xz ).mul( t ).mul( t ).div( 2 * EARTH_RADIUS ) ).toVar();
				const below = max( clouds.shellBottom.sub( altitude ), 0 ).toVar();
				const pxz = rd.xz.mul( t ).toVar();
				// Where the drop left the cloud base, and how much it rains there.
				const source = pxz.sub( lean.mul( below ) ).toVar();
				// The base sits anywhere in the lowest part of the shell, lower under richer weather.
				const above = cloudAbove( source, BASE_WEATHER_END );
				const rain = clouds.rain;
				// A raining deck rains everywhere, hardest under the richest weather of its cells.
				const core = smoothstep( 0.25, 0.45, above.w );
				const deck = rain.mul( core.mul( core ).mul( 0.75 ).add( 0.25 ) ).mul( above.presence );
				const broken = smoothstep( 0.55, 0.62, above.coverage ).mul( float( 1 ).sub( smoothstep( 0.7, 0.85, above.coverage ) ) );
				const heaps = smoothstep( 0.2, 0.35, above.top ).mul( above.presence ).mul( broken ).mul( clouds.heapShowers ).mul( 0.4 );
				const linger = max( clouds.showers.sub( rain ), 0 ).mul( smoothstep( 0.12, 0.3, above.top ) ).mul( above.presence ).mul( 0.5 );
				const amount = max( deck, max( heaps, linger ) ).toVar();
				If( amount.greaterThan( 0.003 ), () => {

					// Uneven curtains: broad lumps, and streaks stretched along the fall, both
					// following the drops' slanted paths and sliding down as they fall.
					const world = source.add( clouds.camXZ ).add( clouds.windOffset ).toVar();
					const v = altitude.add( fall ).div( RAIN_PERIOD ).toVar();
					const footprint = t.mul( pxAngle );
					const lumps = shapeNode.sample( vec3( world.x.div( RAIN_LUMPS ), v.div( 6 ), world.y.div( RAIN_LUMPS ) ) ).level( coneLod( footprint, RAIN_LUMPS ) ).r;
					const streaks = shapeNode.sample( vec3( world.x.div( RAIN_STREAK ), v, world.y.div( RAIN_STREAK ) ) ).level( coneLod( footprint, RAIN_STREAK ) ).g;
					const pattern = sat( lumps.mul( 3 ).sub( 1.2 ) ).mul( streaks.mul( streaks ).mul( 2.6 ).add( 0.15 ) );
					// Light showers evaporate on the way down; rain starts at the base, and the
					// nearest is left to the streaks around the camera.
					const reach = float( 1 ).sub( sat( amount.mul( 2.5 ) ) ).mul( 2600 ).sub( 1500 );
					const shape = smoothstep( reach, reach.add( 1500 ), altitude ).mul( smoothstep( 0, 300, below ) ).mul( smoothstep( RAIN_START, RAIN_START * 2.5, t ) );
					const sigma = amount.mul( pattern ).mul( shape ).mul( RAIN_SIGMA ).toVar();
					// The sun reaches the drops where no cloud stands on its way: the base and
					// two heights in the shell, where the beam from clouds point crosses them.
					const lit = float( 0 ).toVar();
					If( hasSun, () => {

						const reachShell = ( h: number ) => pxz.add( L.xz.mul( below.add( clouds.shellThickness.mul( h ) ).div( ly ) ) );
						const tau = cloudAbove( reachShell( 0.02 ), 0.02 ).presence.add( cloudAbove( reachShell( 0.18 ), 0.18 ).presence )
							.add( cloudAbove( reachShell( 0.4 ), 0.4 ).presence ).mul( 3 );
						lit.assign( exp( tau.negate() ) );

					} );
					// Under the cloud it falls from the rain sees the dark base above (about as bright
					// as the clouds' own base light) and the darker sea below; only a broken sky lets
					// the sky's own light in from the sides.
					// Rain always falls from a cloud: overhead it sees that cloud's base.
					const ambient = cloudBase.mul( 0.4 ).add( sideLight.mul( 0.2 ) ).mul( clouds.ambientGain ).add( clouds.nightAmbient.rgb.mul( 0.3 ) );
					const light = sun.mul( phase ).mul( lit ).add( ambient );
					const air = exp( t.mul( airScale ) );
					const alpha = float( 1 ).sub( exp( sigma.mul( dt ).negate() ) ).toVar();
					radiance.addAssign( mix( sky, light, air ).mul( through ).mul( alpha ) );
					// A strike lights the rain around it: added apart from the rest, like the clouds' own
					// flash, so it stays out of the accumulated view.
					if ( withFlash ) If( clouds.flashOn.greaterThan( 0.5 ), () => {

						const p = vec3( pxz.x, altitude, pxz.y );
						flash.addAssign( clouds.flashLight( p ).mul( RAIN_FLASH ).mul( air ).mul( through ).mul( alpha ) );

					} );
					through.mulAssign( float( 1 ).sub( alpha ) );

				} );

			} );

		} );
		return { color: vec4( radiance, through ), flash };

	};

	// The cirrus: a sheet of fibres `streakHeight` up on the curved Earth, so they close up
	// toward the horizon in true perspective, combed along the drift (Streaks.ts) and gathered in
	// patches a few to twenty kilometres across (the synoptic field), filtered by each pixel's
	// footprint on the sheet. Within a patch the depth is the fibres' with a silk between them (the
	// tufts' field at `streakFloor`), its edge eaten by the fibres (`streakErode`), and it thins
	// where the fine fibres' mips average them into a veil (`streakThin`), so far off the layer
	// keeps its gaps rather than merging into a sheet. Its density fades where one texel of the panorama spans more
	// of the sheet than `streakFade` along the view (the texel's angle times t / mu, metres), so
	// toward the horizon the panorama's slow refresh and coarse texels show an even veil, not
	// sparkle (after Patry 2021, SIGGRAPH Advances, p. 94-96: lower a low-resolution cloud map's
	// density where the cloud's position moves fast with its texture coordinate). The view takes
	// the same fade, so the dome and the panorama it reflects in agree. Its light is the
	// atmosphere's at the layer's height: the sun's direct share the sun's light
	// through the air from each point at the sun's elevation there, colour and level both
	// (`streakSunColour`), so at any moment the side away from the sun is the redder and goes
	// out first, and the side toward it reddens last; the moon's in the colour of the aerosol's
	// in-scatter there toward the ray at its gain (`moonLight`); each desaturated, under two
	// Henyey-Greenstein lobes weighted by the light through the streak toward the body times
	// the light through it along the ray, forward (0.85) where both pass, back (-0.15, times
	// 2.16 for the light scattered more than once) where little does (after Patry 2021, p. 97;
	// the smoothstep on the weight is ours), the forward lobe held at its value
	// `streakForwardCap` from the sun; the sky's share the Rayleigh in-scatter there under its
	// phase, desaturated, times the earth's shadow at the layer and each body's illuminance, at
	// `streakAmbient` per unit of opacity, the night's ambient with it. Each
	// share takes an overall colour (`keyDirect`, `keyAmbient`). No air of its own here: the
	// dome fogs it at its own distance (SkyDome.ts domeColour, `sampleViewFog`).
	const streaksAt = ( rd: V3, pxAngle: F ) => {

		const result = vec4( 0, 0, 0, 1 ).toVar();
		If( clouds.streaks.greaterThan( 0.01 ).and( rd.y.greaterThan( 0 ) ), () => {

			const t = layerDistance( clouds.camY, rd.y, clouds.streakHeight ).toVar();
			const sheet = rd.xz.mul( t ).toVar();
			const mu = max( rd.y.add( dot( rd.xz, sheet ).div( EARTH_RADIUS ) ), 0.02 ).toVar();
			// The footprint on the sheet: across the view, and along it stretched by 1 / mu.
			const radial = normalize( rd.xz.add( vec2( 1e-6, 0 ) ) ).toVar();
			const fa = vec2( radial.y.negate(), radial.x ).mul( t.mul( pxAngle ) );
			const fb = radial.mul( t.mul( pxAngle ).div( mu ) );
			// The drift's frame: x along the surface wind veered LOW_VEER to its right.
			const a = clouds.windAxis, c = Math.cos( LOW_VEER ), sn = Math.sin( LOW_VEER );
			const w = vec2( a.x.mul( c ).sub( a.y.mul( sn ) ), a.y.mul( c ).add( a.x.mul( sn ) ) ).toVar();
			const toWind = ( v: V2 ) => vec2( dot( v, w ), dot( v, vec2( w.y.negate(), w.x ) ) );
			const q = toWind( sheet.add( clouds.camXZ ) ).toVar();
			const syn = synopticNode.sample( q.add( clouds.streakShift.synoptic ).div( SYNOPTIC_SIZE ) ).level( float( 0 ) );
			const patches = smoothstep( clouds.streakPatchLow, clouds.streakPatchHigh, syn.x );
			// Fibres at two scales hide the texture's tiling.
			const f1 = sampleStreaks( streakCover, q.add( clouds.streakShift.fine ), toWind( fa ), toWind( fb ), FIBRE_FINE.tile, FIBRE_FINE.turn ).toVar();
			const f2 = sampleStreaks( streakCover, q.add( clouds.streakShift.broad ), toWind( fa ), toWind( fb ), FIBRE_BROAD.tile, FIBRE_BROAD.turn ).toVar();
			const fibre = f1.x.mul( 0.6 ).add( f2.x.mul( 0.4 ) );
			const tufts = f1.y.mul( 0.6 ).add( f2.y.mul( 0.4 ) );
			const silk = tufts.mul( clouds.streakFloor ).add( fibre.mul( float( 1 ).sub( clouds.streakFloor ) ) );
			// a patch's edge eaten by the fibres: toward it only the densest reach out
			const reach = clamp( patches.sub( float( 1 ).sub( fibre ).mul( clouds.streakErode ) ).div( float( 1 ).sub( clouds.streakErode ) ), 0, 1 );
			// where the fine fibres' mips average them, the layer thins
			const thin = mix( 1, clouds.streakThin, smoothstep( 1, 4, f1.z ) );
			const spread = t.mul( domeTexelAngle ).div( mu );
			const fade = clouds.streakFade.div( max( spread, clouds.streakFade ) );
			const tau = clouds.streaks.mul( clouds.streakPeak ).mul( reach ).mul( silk ).mul( thin ).mul( fade ).toVar();
			const own = exp( tau.negate().div( max( mu, clouds.streakMuFloor ) ) ).toVar();
			const A = clouds.atmosphere;
			const heightKm = clouds.streakHeight.div( 1000 ).toVar();
			const sunRows = A.cloudRows( rd.y, heightKm, 0 ), moonRows = A.cloudRows( rd.y, heightKm, 1 );
			const chroma = ( v: V3 ) => v.div( max( luminance( v ), 1e-20 ) );
			const capped = henyeyGreenstein( clouds.streakForwardCap, 0.85 ).toVar();
			// a body's phase times the light through the streak toward it
			const lobes = ( toward: V3 ) => {

				const through = exp( tau.negate().div( max( toward.y, 0.1 ) ).mul( 0.5 ) ).toVar();
				const nu = dot( rd, toward ).toVar();
				const forward = min( henyeyGreenstein( nu, 0.85 ), capped );
				return mix( henyeyGreenstein( nu, - 0.15 ).mul( 2.16 ), forward, smoothstep( 0, 1, through.mul( own ) ) ).mul( through );

			};
			// The sun at clouds point of the sheet: its elevation there, the sun's raised by the point's
			// lead toward it over `streakLitRadius`; its light through the air from there, which
			// gives the direct share its colour as well as its level; and its
			// fade, none once its setting disc is under the layer's horizon (the dip at its height).
			const flat = A.sunDir.xz.div( max( length( A.sunDir.xz ), 1e-4 ) );
			const local = asin( clamp( A.sunDir.y, - 1, 1 ) ).add( dot( sheet, flat ).div( clouds.streakLitRadius ) ).toVar();
			const dip = acos( float( EARTH_RADIUS ).div( clouds.streakHeight.add( EARTH_RADIUS ) ) );
			const litFade = smoothstep( 0, clouds.streakLitWidth.mul( Math.PI / 180 ), local.add( dip ).add( SUN_SETTING_RADIUS ) );
			const sunLight = A.transmittanceFrom( heightKm, sin( clamp( local, - Math.PI / 2, Math.PI / 2 ) ) ).mul( clouds.streakSunColour ).mul( litFade );
			const direct = desaturate( sunLight, clouds.lightDesaturation ).mul( lobes( A.sunDir ) )
				.add( chroma( desaturate( moonRows.mie, clouds.lightDesaturation ) ).mul( clouds.moonLight.mul( lobes( A.moonDir ) ) ) );
			const skyOf = ( rows: { rayleigh: V3 }, body: 0 | 1, light: V3, illuminance: V3 ) => desaturate(
				rows.rayleigh.mul( skyRayleighPhase( dot( rd, light ), A.rayleighForward ) ), clouds.ambientDesaturation )
				.mul( A.earthShadowAt( heightKm, rd, body ) ).mul( illuminance );
			const sky = skyOf( sunRows, 0, A.sunDir, A.sunIlluminance.rgb ).add( skyOf( moonRows, 1, A.moonDir, A.moonIlluminance.rgb ) )
				.add( clouds.nightAmbient.rgb ).mul( clouds.streakAmbient.mul( clouds.ambientGain ) );
			const lit = direct.mul( clouds.keyDirect ).add( sky.mul( clouds.keyAmbient ) );
			result.assign( vec4( lit.mul( float( 1 ).sub( own ) ), own ) );

		} );
		return result;

	};

	const domeCone = float( domeTexelAngle );
	const marchPano = makeMarch( panoramaSteps, domeCone, domeCone );
	const pixelCone = clouds.cam.tan.y.mul( 2 ).div( clouds.displayHeight );
	// The panorama refreshes a fraction of its texels per frame and feeds bakes that cross-fade
	// over seconds, too slow for a flash: only the view carries the lightning in its march;
	// reflections take it from `flashAt`.
	const viewCone = clouds.cam.tan.y.mul( 2 ).div( clouds.viewSize.y );
	clouds.marchView = makeMarch( viewSteps, pixelCone, viewCone, true, true );
	clouds.streaksAt = streaksAt;
	clouds.rainAt = rainAt;

	// ---- panorama: a fixed per-texel offset breaks step banding; the tent filter averages it.
	const columns = domeWidth / strideX;
	const dome = clouds.dome;
	clouds.panoramaKernel = Fn( () => {

		const x = instanceIndex.mod( uint( columns ) ).mul( uint( strideX ) ).add( uint( clouds.phase.x ) );
		const y = instanceIndex.div( uint( columns ) ).mul( uint( strideY ) ).add( uint( clouds.phase.y ) );
		const q = vec2( float( x ).add( 0.5 ).div( domeWidth ), float( y ).add( 0.5 ).div( domeHeight ) );
		const jitter = fract( sin( dot( vec2( float( x ), float( y ) ), vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) );
		const rd = domeToDirection( q ).toVar();
		const m = marchPano( rd, jitter );
		const result = over( rainAt( rd, float( domeTexelAngle ), jitter ).color, over( vec4( m.L, m.T ), streaksAt( rd, float( domeTexelAngle ) ) ) );
		textureStore( dome, ivec2( int( x ), int( y ) ), vec4( min( result.rgb.mul( clouds.domeStore ).mul( storePress.store ), vec3( STORE_CEILING ) ), result.a ) );

	} )().compute( domeWidth * domeHeight / ( strideX * strideY ), [ 64 ] ).setName( 'Clouds.Panorama' );
	// A new power of two: every texel times the ratio of the new power to the old into the
	// scratch, held at the march's ceiling, then copied back.
	const scratch = clouds.domeScratch, domeIn = texture( dome ), scratchIn = texture( scratch );
	const texelAt = () => ivec2( int( instanceIndex.mod( uint( domeWidth ) ) ), int( instanceIndex.div( uint( domeWidth ) ) ) );
	clouds.rescaleKernels = [
		Fn( () => {

			const p = texelAt().toVar();
			const s = textureLoad( domeIn, p ).toVar();
			textureStore( scratch, p, vec4( min( s.rgb.mul( clouds.domeRescale ), vec3( STORE_CEILING ) ), s.a ) );

		} )().compute( domeWidth * domeHeight, [ 64 ] ).setName( 'Clouds.PanoramaRescale' ),
		Fn( () => {

			const p = texelAt().toVar();
			textureStore( dome, p, textureLoad( scratchIn, p ) );

		} )().compute( domeWidth * domeHeight, [ 64 ] ).setName( 'Clouds.PanoramaRescale' ),
	];

	// ---- cloud shadow: transmittance of the shell along the key light around the island, one
	// of every SHADOW_PHASES rows per pass.
	// What gets through the shell to a ground point along a ray climbing `ly` per unit of `L.xz`:
	// ( the direct beam, the direct beam with the light scattered on through ). One function per
	// kernel: a helper reading uniforms can be emitted with another kernel's uniform names (r186).
	const makeCastRay = ( name: string ) => Fn( ( [ ground, ly ]: [ V2, F ] ) => {

		const ray = vec3( clouds.lightDir.x, ly, clouds.lightDir.z ).toVar();
		const t0 = clouds.shellBottom.div( ly ).toVar(), dt = clouds.shellThickness.div( 16 ).div( ly ).toVar();
		// (the optical depth at the weather's share of the layer's: `shadowDensity`)
		const tau = float( 0 ).toVar();
		const coverage = shellCoverage( clouds.coverage ).toVar();
		Loop( 16, ( { i } ) => {

			const p = vec3( ground.x, 0, ground.y ).add( ray.mul( t0.add( float( i ).add( 0.5 ).mul( dt ) ) ) ).toVar();
			tau.addAssign( cloudDensity( p, coverage, vec2( 2, 2 ), false ).mul( clouds.density ).mul( DENSITY ).mul( clouds.shadowDensity ).mul( dt ) );
			If( tau.greaterThan( 12 ), () => {

				Break();

			} );

		} );
		// Multiple scattering lets a little light through even thick cloud.
		const direct = exp( tau.negate() ).toVar();
		return vec2( direct, direct.mul( 0.8 ).add( exp( tau.mul( - 0.25 ) ).mul( 0.2 ) ) );

	} ).setLayout( { name, type: 'vec2', inputs: [ { name: 'ground', type: 'vec2' }, { name: 'ly', type: 'float' } ] } );
	const castRay = makeCastRay( 'cloudCastRay' ), castMeanRay = makeCastRay( 'cloudDeckMeanRay' );
	const meanCells = uint( DECK_MEAN_GRID );
	clouds.deckMeanKernel = Fn( () => {

		const cell = vec2( float( instanceIndex.mod( meanCells ) ), float( instanceIndex.div( meanCells ) ) );
		const ground = cell.add( 0.5 ).div( DECK_MEAN_GRID ).sub( 0.5 ).mul( DECK_MEAN_SPAN ).sub( clouds.camXZ ).toVar();
		const beam = castMeanRay( ground, max( clouds.lightDir.y, SHADOW_SLOPE_FLOOR ) );
		clouds.deckMeanStore.element( instanceIndex ).assign( vec2( beam.y, beam.x ) );

	} )().compute( DECK_MEAN_RAYS, [ 64 ] ).setName( 'Clouds.DeckMean' );
	clouds.deckMeanSumKernel = Fn( () => {

		const sum = vec2( 0 ).toVar();
		Loop( DECK_MEAN_RAYS, ( { i } ) => {

			sum.addAssign( clouds.deckMeanStore.element( i ) );

		} );
		clouds.deckMeanStore.element( DECK_MEAN_RAYS ).assign( sum.div( DECK_MEAN_RAYS ) );

	} )().compute( 1, [ 1 ] ).setName( 'Clouds.DeckMeanSum' );
	clouds.shadowKernel = Fn( () => {

		const n = uint( SHADOW_SIZE );
		const x = instanceIndex.mod( n ), y = instanceIndex.div( n ).mul( uint( SHADOW_PHASES ) ).add( uint( clouds.shadowRow ) );
		const ground = vec2( float( x ), float( y ) ).add( 0.5 ).div( SHADOW_SIZE ).sub( 0.5 ).mul( SHADOW_SPAN ).sub( clouds.camXZ ).toVar();
		const visibility = float( 1 ).toVar(), sharp = float( 1 ).toVar(), direct = float( 1 ).toVar();
		If( clouds.shadowShare.greaterThan( 0 ), () => {

			// Rays spaced evenly over the height of the light's disc, each weighted by the disc's
			// width there: its footprint on the ground runs kilometres along the light under a
			// low sun, far beyond what the read's filter or a cloud's edge soften, and it takes
			// the disc seconds to pass an edge. The heights alone: the footprint across the light
			// is hundreds of metres, and the shadow slides along it.
			const rays = float( clouds.shadowRays ).toVar();
			const cast = float( 0 ).toVar(), castDirect = float( 0 ).toVar(), total = float( 0 ).toVar();
			const centre = float( 0 ).toVar(), centres = float( 0 ).toVar();
			Loop( clouds.shadowRays, ( { i } ) => {

				const across = float( i ).add( 0.5 ).div( rays ).mul( 2 ).sub( 1 ).toVar();
				const width = sqrt( float( 1 ).sub( across.mul( across ) ) ).toVar();
				const beam = castRay( ground, max( clouds.lightDir.y.add( across.mul( clouds.shadowDisc ) ), SHADOW_SLOPE_FLOOR ) ).toVar();
				const ray = beam.y;
				cast.addAssign( ray.mul( width ) );
				castDirect.addAssign( beam.x.mul( width ) );
				total.addAssign( width );
				// the ray or two nearest the disc's centre: one ray stands for itself
				If( abs( across ).mul( rays ).lessThan( 2 ), () => {

					centre.addAssign( ray );
					centres.addAssign( 1 );

				} );

			} );
			const disc = cast.div( total ).toVar();
			visibility.assign( mix( float( 1 ), disc, clouds.shadowShare ) );
			sharp.assign( mix( float( 1 ), mix( disc, centre.div( centres ), clouds.shadowAirNarrow ), clouds.shadowShare ) );
			direct.assign( mix( float( 1 ), castDirect.div( total ), clouds.shadowShare ) );

		} );
		// Under a closed deck the fade goes to the deck's mean, not to lit: mix( mix( 1, mean,
		// deck ), map, share ), an exact no-op off overcast and with the map whole.
		const toMean = float( 1 ).sub( clouds.shadowShare ).mul( clouds.deckShadowShare ).toVar();
		const deckMean = clouds.deckMeanRead.element( DECK_MEAN_RAYS ).toVar();
		visibility.addAssign( toMean.mul( deckMean.x.sub( 1 ) ) );
		sharp.addAssign( toMean.mul( deckMean.x.sub( 1 ) ) );
		direct.addAssign( toMean.mul( deckMean.y.sub( 1 ) ) );
		textureStore( clouds.shadowMap, ivec2( int( x ), int( y ) ), vec4( visibility, sharp, direct, 1 ) );

	} )().compute( SHADOW_SIZE * SHADOW_SIZE / SHADOW_PHASES, [ 64 ] ).setName( 'Clouds.Shadow' );
	// The view's kernels close over the march: rebuild them on the next update.
	clouds.disposeView();
	clouds.needsFull = true;

}
