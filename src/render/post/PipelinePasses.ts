/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	mrt, output,
} from 'three/tsl';
import { TEMPORAL } from './Antialiasing.ts';
import { overlayMRT, sceneMRT } from './Motion.ts';
import { preExposedRGBA } from './PreExposure.ts';
import { afterMRT } from './AfterResolve.ts';
import { SS_SHADOW, screenShadowView } from './MeadowShadows.ts';
import { GTAO, gtaoView } from './GTAO.ts';
import { FrameGraph, type ResourceHandle } from './graph/FrameGraph.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { displacementField } from '../vegetation/Displacement.ts';
import { declareExtraPasses } from './ExtraPass.ts';
import { waveMapsPass } from '../ocean/WaveMaps.ts';
import { localLightsPass } from '../lighting/LocalLights.ts';
import type { PassBuilder, Pipeline, PipelineFrame } from './Pipeline.ts';

// The frame's passes on the frame graph, declared in order.
//
// The frame's passes on the frame graph, in order (`declarePasses`): the order and the rules are
// Pipeline.ts's header (Frame graph). With them the temporal modes' scene half
// (`declareTemporalScene`) and what the water refracts (`declareRefraction`). The pipeline's
// constructor declares them once its materials are built; each takes the pipeline and reads its
// targets and quads.

// what the water pass draws in a frame without water: nothing
const NO_WATER = new THREE.Scene();
// (the depth pyramid is built for the water alone)
const hasWater = ( f: PipelineFrame ) => f.water !== null;
// the sea's reflection's passes meet the frame: they run where there is water, drawn by the frame's
// camera after the water's own setup (Water.ts `prepare`, which sets the box the trace keeps to)
const SEA_HOOKS = { active: hasWater, camera: ( f: PipelineFrame ) => f.camera, before: ( f: PipelineFrame ) => f.water?.prepare( f.sun ) };
// before the water draws: its setup, and its read of the reflection bound to the image written last
const waterBefore = ( p: Pipeline, f: PipelineFrame ) => {

	f.water?.prepare( f.sun );
	p.seaReflection.bind();

};

// The refraction pass's only attachment, cleared to zero coverage (alpha 0: nothing drawn there),
// its colour pre-exposed as the scene target's (PreExposure.ts).
const REFRACTION_MRT: THREE.MRTNode = mrt( { refraction: preExposedRGBA( output ) } ).setClearColor( 'refraction', new THREE.Color( 0, 0, 0 ), 0 );

// The blades read the displacement field in every pass that draws the meadow, and the frame
// before's for their previous position under the temporal modes.
function readDisplacement( p: Pipeline, b: PassBuilder ): void {

	if ( ! p.displacement ) return;
	b.read( p.displacement.field );
	if ( p.displacement.previous ) b.read( p.displacement.previous );

}

// The frame's passes, in order. The scene half draws into the scene's multisampled target and
// resolves what the water, the rain and the post passes read. Each sample keeps the opaque
// surface or the sea, whichever is nearer, so silhouettes against the sea are antialiased like
// the rest of the image. (Drawn into a target of their own, single-sampled, the water had to
// decide per pixel between the sea and the opaque colour: edges stepped, and the opaque colour
// at an edge pixel held the sky the sea was not there to cover, a pale outline round everything
// over the sea.) Rain does not write depth, so surface focus and overlapping translucent streaks
// are preserved; its splashes compare the resolved depths themselves. The post passes follow,
// each into a target of the graph's or the canvas. Ahead of the scene, after the haze and the
// sky's constants, the displacement field (vegetation/Displacement.ts): its sprites at a
// fixed size, read by the meadow's passes; then the scene's extra passes (the island's falling
// leaves' step), the wave maps and the caustics
// (ocean/WaveMaps.ts), which the materials lit under the water read, and the local lights'
// clusters (lighting/LocalLights.ts), by night.
export function declarePasses( p: Pipeline ): void {

	const g = p.graph;
	// the haze first, ahead of the scene, whose sky dome reads its light volume (Sky.ts
	// `fogDome`): its shadow (skipped while a rest holds it) and its light volume; they read the
	// sun's maps, the clouds' shadow and the camera, never the scene's depth
	const hazeReads = p.air.passes( g );
	// then the sky's constants, which the flare's visibility reads (Sky.ts `discCloudTransmittance`)
	let skyConstants = p.skyConstants.output;
	const skyConstantsQuad = p.skyConstants.quad;
	g.addPass( 'skyConstants', 'quad', ( b ) => {

		skyConstants = b.write( skyConstants );
		return { label: 'sky constants', quad: skyConstantsQuad };

	} );
	// the displacement field, which the passes that draw the meadow read
	// (vegetation/Displacement.ts)
	p.displacement = displacementField ? displacementField.declare( g ) : null;
	// the scene's extra passes (the falling leaves' step, whose buffers the opaque scene draws)
	declareExtraPasses( g, p.extraPasses );
	// the wave maps and the caustics, which the materials lit under the water read
	waveMapsPass( g, () => p.waveMaps );
	// the local lights' clusters, which every lit material of the scene passes reads
	// (lighting/LocalLights.ts)
	localLightsPass( g );
	let { sceneColor, sceneDepth, linearDepth, opaqueColor, underwater, surfaceDepth, velocity, meterLight } = p.resources;
	if ( TEMPORAL ) [ sceneColor, surfaceDepth, velocity, opaqueColor, meterLight, linearDepth ] = declareTemporalScene( p, hazeReads );
	else {

		// the vegetation's depth first, the opaque scene's copies (post/OpaquePrepass.ts), the
		// screen-space shadows over both (MeadowShadows.ts), then the opaque scene over the vegetation's depth
		let { meadowCoverage, meadowDepth, meadowShadows: shade } = p.resources;
		let { gtaoOpaque, gtaoOpaqueDepth, gtaoDepth, gtaoRaw, gtaoBlur, gtao } = p.resources;
		const depthQuad = p.meadowDepthQuad, shadeQuad = p.meadowShadowQuad;
		const gtaoOn = () => GTAO && debugFlags.gtao !== false;
		const ssOn = () => SS_SHADOW && debugFlags.ssShadow !== false;
		if ( meadowCoverage && meadowDepth && shade && depthQuad && shadeQuad ) {

			g.addPass( 'meadowDepth', 'scene', ( b ) => {

				readDisplacement( p, b );
				meadowCoverage = b.write( meadowCoverage!, { clear: true } );
				sceneDepth = b.depth( sceneDepth, { write: true, clear: true } );
				return { label: 'vegetation depth', scene: ( f ) => f.prepassScene!, camera: ( f ) => f.camera };

			} );
			// the opaque scene's distance at half resolution, for the ambient occlusion and the
			// screen-space shadows over the whole scene
			if ( gtaoOpaque && gtaoOpaqueDepth ) g.addPass( 'opaqueDepth', 'scene', ( b ) => {

				gtaoOpaque = b.write( gtaoOpaque!, { clear: true } );
				gtaoOpaqueDepth = b.depth( gtaoOpaqueDepth!, { write: true, clear: true } );
				return { label: 'opaque depth', when: () => gtaoOn() || ssOn(), whenLabel: 'gtao|ssShadow', scene: ( f ) => f.opaquePrepassScene!, camera: ( f ) => f.camera };

			} );
			g.addPass( 'meadowDepthResolve', 'quad', ( b ) => {

				b.read( sceneDepth, 'samples' );
				if ( SS_SHADOW && gtaoOpaque ) b.readOptional( gtaoOpaque );
				meadowDepth = b.write( meadowDepth! );
				return { label: 'meadow shadows', quad: depthQuad };

			} );
			g.addPass( 'meadowShadows', 'quad', ( b ) => {

				b.read( meadowDepth! );
				shade = b.write( shade! );
				return { label: 'meadow shadows', quad: shadeQuad };

			} );

		}
		// the ambient occlusion (GTAO.ts): the nearer of the opaque scene's distance and the
		// vegetation's depth as view z, the slices and their blur, which the opaque scene's
		// receivers load
		const quads = p.gtaoQuads;
		if ( meadowCoverage && gtaoOpaque && gtaoDepth && gtaoRaw && gtaoBlur && gtao && quads ) {

			g.addPass( 'gtaoDepth', 'quad', ( b ) => {

				b.readOptional( sceneDepth, 'samples' );
				b.readOptional( gtaoOpaque! );
				gtaoDepth = b.write( gtaoDepth! );
				return { label: 'gtao', when: gtaoOn, whenLabel: 'gtao', quad: quads.depth };

			} );
			g.addPass( 'gtao', 'quad', ( b ) => {

				b.readOptional( gtaoDepth! );
				gtaoRaw = b.write( gtaoRaw! );
				return { label: 'gtao', when: gtaoOn, whenLabel: 'gtao', quad: quads.slices };

			} );
			// the blur, separable: along x, then along y (the view z for its edges' planes)
			g.addPass( 'gtaoBlurX', 'quad', ( b ) => {

				b.readOptional( gtaoRaw! );
				b.readOptional( gtaoDepth! );
				gtaoBlur = b.write( gtaoBlur! );
				return { label: 'gtao', when: gtaoOn, whenLabel: 'gtao', quad: quads.blurX };

			} );
			g.addPass( 'gtaoBlurY', 'quad', ( b ) => {

				b.readOptional( gtaoBlur! );
				b.readOptional( gtaoDepth! );
				gtao = b.write( gtao! );
				return { label: 'gtao', when: gtaoOn, whenLabel: 'gtao', quad: quads.blurY };

			} );
		}
		// the receivers read the occlusion and the whole scene's shadows in the opaque pass alone
		// (the refraction's draws of the same materials read 1)
		if ( gtao || SS_SHADOW ) {

			const end = g.hooks.onPassEnd;
			g.hooks.onPassEnd = ( pass, f ) => {

				if ( pass.name === 'opaque' ) gtaoView.on.value = screenShadowView.on.value = 0;
				end?.( pass, f );

			};

		}
		const occlusion = gtao;
		g.addPass( 'opaque', 'scene', ( b ) => {

			// the blades read their shadows, the ground marches over the resolved depth, the
			// receivers load the ambient occlusion and the whole scene's shadows
			if ( shade ) b.read( shade );
			if ( meadowDepth && ! SS_SHADOW ) b.read( meadowDepth );
			if ( occlusion ) b.readOptional( occlusion );
			for ( const h of hazeReads ) b.read( h );
			readDisplacement( p, b );
			sceneColor = b.write( sceneColor, { clear: true } );
			meterLight = b.write( meterLight, { clear: true } );
			sceneDepth = b.depth( sceneDepth, { write: true, clear: ! shade } );
			return {
				label: 'opaque scene', mrt: sceneMRT, scene: ( f ) => f.scene, camera: ( f ) => f.camera,
				before: occlusion || SS_SHADOW ? () => {

					gtaoView.on.value = gtaoOn() ? 1 : 0;
					screenShadowView.on.value = ssOn() ? 1 : 0;

				} : undefined,
			};

		} );
		let refraction: ResourceHandle, refractionDepth: ResourceHandle;
		[ underwater, refraction, refractionDepth ] = declareRefraction( p, sceneColor, sceneDepth );
		// linear depth, and a copy of the opaque colour for the water to refract and reflect (a pass
		// cannot sample the target it draws into)
		g.addPass( 'depthResolve', 'quad', ( b ) => {

			b.read( sceneDepth, 'samples' );
			linearDepth = b.write( linearDepth );
			return { label: 'depth & copy resolves', quad: p.depthQuad };

		} );
		// its pyramid, which the water's reflection traces over (DepthPyramid.ts)
		p.depthPyramid.pass( g, linearDepth, false, hasWater );
		const copyQuad = p.copyQuad!;
		g.addPass( 'opaqueCopy', 'quad', ( b ) => {

			b.read( sceneColor );
			opaqueColor = b.write( opaqueColor );
			return { label: 'depth & copy resolves', quad: copyQuad };

		} );
		// the sea's reflection of the opaque image, traced and denoised (ocean/SeaReflection.ts)
		const seaReflected = p.seaReflection.passes( g, { linearDepth, opaqueColor, pyramid: p.depthPyramid, iterations: p.seaIterations }, false, SEA_HOOKS );
		g.addPass( 'water', 'scene', ( b ) => {

			b.read( opaqueColor );
			b.read( linearDepth );
			b.read( underwater );
			b.readOptional( refraction );
			b.readOptional( refractionDepth );
			b.readOptional( seaReflected );
			sceneColor = b.write( sceneColor );
			meterLight = b.write( meterLight );
			sceneDepth = b.depth( sceneDepth, { write: true } );
			return {
				label: 'water', mrt: sceneMRT, before: ( f ) => waterBefore( p, f ),
				scene: ( f ) => f.water ? f.waterScene : NO_WATER, camera: ( f ) => f.camera,
			};

		} );
		g.addPass( 'surfaceResolve', 'quad', ( b ) => {

			b.read( sceneDepth, 'samples' );
			surfaceDepth = b.write( surfaceDepth );
			return { label: 'depth & copy resolves', quad: p.surfaceQuad };

		} );
		// (the last pass into the scene target: its samples are resolved, then never read)
		g.addPass( 'rain', 'scene', ( b ) => {

			b.read( linearDepth );
			b.read( surfaceDepth );
			sceneColor = b.write( sceneColor );
			meterLight = b.write( meterLight, { blend: 'keep' } );
			b.depth( sceneDepth, { write: false } );
			return {
				label: 'rain draw', mrt: overlayMRT, when: ( f ) => f.rainScene !== undefined, whenLabel: 'rainVisible',
				scene: ( f ) => f.rainScene!, camera: ( f ) => f.camera,
			};

		} );

	}
	let { hdr, sharp, display, easu, canvas, veilReactive, rainAir } = p.resources;
	// the light shafts: the open sky near the sun, and the rays marched over it (while the sun is
	// toward the view: the composite weighs them by `facing`, 0 otherwise)
	const shaftReads = p.shafts.passes( g, surfaceDepth );
	// the post passes' per-frame constants, into a texel: the flare's visibility (the
	// nearest surface's depth over the disk, the sky constants' cloud transmittance over it)
	// and the rods' colour's retinal scale
	let constants = p.constants.output;
	const constantsQuad = p.constants.quad;
	g.addPass( 'postConstants', 'quad', ( b ) => {

		b.read( surfaceDepth );
		b.read( skyConstants );
		constants = b.write( constants );
		return { label: 'post constants (flare visibility, rods\' scale)', quad: constantsQuad };

	} );
	// what the composite reads: the scene (after the rain, which may be skipped), the nearest
	// surface's depth, the opaque scene's (which tells the land from the sea for the haze's light
	// over the land), the haze's light, the light shafts and the constants (the flare's
	// visibility, the rods' colour's scale)
	const compositeReads = ( b: PassBuilder ) => {

		b.readOptional( sceneColor );
		b.read( surfaceDepth );
		b.read( linearDepth );
		for ( const h of hazeReads ) b.read( h );
		for ( const h of shaftReads ) b.readOptional( h );
		b.read( constants );

	};
	// the metering histograms' clear, before the composite (Metering.ts)
	const metering = p.metering;
	const { exposureStatePrev } = p.resources;
	if ( hdr ) metering.clearPass( g );
	// the composite of the haze, the veil and the shafts, rendered once
	const compositeQuad = p.compositeQuad;
	if ( hdr && compositeQuad ) g.addPass( 'composite', 'quad', ( b ) => {

		compositeReads( b );
		hdr = b.write( hdr! );
		// (the attachments in this order: the struct's; the MRT only declares both to the graph,
		// unblended, as the resolve's does)
		if ( veilReactive ) veilReactive = b.write( veilReactive );
		if ( rainAir ) rainAir = b.write( rainAir );
		return { label: 'composite (air, fog, veil)', quad: compositeQuad, mrt: veilReactive || rainAir ? mrt( {} ) : null };

	} );
	// the metering histograms of hdr and meterLight, after the composite and before the bloom,
	// which reads this frame's exposure (Metering.ts)
	if ( hdr ) metering.histogramPass( g, hdr, meterLight, surfaceDepth );
	// the meter: the exposure state from the histograms, which every pass after it reads (and the
	// next frame, from its copy)
	const exposureState = metering.meterPass( g, p.resources.exposureState, exposureStatePrev );
	g.copyToPersistent( exposureState, exposureStatePrev );
	const readExposure = ( b: PassBuilder ) => b.read( exposureState );
	// FSR 3 on hdr (Fsr3Upscaler.ts), whose output (or, while the rest's sequence runs, the held copy
	// of the last running frame's) the passes below read in place of hdr
	const fsr3 = p.fsr3 ? p.fsr3.declare( g as unknown as FrameGraph<unknown>, { hdr: hdr!, surfaceDepth, velocity: velocity!, veilReactive, sceneColor, opaqueColor, exposure: exposureState } ) : null;
	const readHdr = ( b: PassBuilder ) => {

		if ( fsr3 ) {

			b.readOptional( fsr3.output );
			b.readOptional( fsr3.held );

		} else b.read( hdr! );

	};
	// the bloom (PhysicalBloom.ts, sixteen passes): the sky's mask from the surface's depth, then
	// the pyramid from hdr, or the scene's colour without the composite
	const bloomResult = p.bloom.passes( g, 'bloom', ( b ) => {

		if ( hdr ) readHdr( b );
		else b.readOptional( sceneColor );
		readExposure( b );

	}, ( b ) => b.read( surfaceDepth ), ( b ) => b.read( constants ) );
	// the sharp image: the composite (inline, without the composite pass: the profiler's
	// composite then) and the blooms
	const sharpQuad = p.sharpQuad;
	g.addPass( 'sharp', 'quad', ( b ) => {

		if ( hdr ) readHdr( b );
		else compositeReads( b );
		b.read( bloomResult );
		sharp = b.write( sharp );
		return { label: hdr ? 'sharp (composite + blooms)' : 'composite (air, fog, veil)', quad: sharpQuad };

	} );
	// the local tone mapping's grid (LocalTone.ts), then the grade
	const toneGrid = p.localTone ? p.localTone.passes( g, ( b ) => {

		b.read( sharp );
		readExposure( b );

	} ) : null;
	const displayQuad = p.displayQuad;
	g.addPass( 'display', 'quad', ( b ) => {

		b.read( sharp );
		if ( toneGrid ) b.read( toneGrid );
		readExposure( b );
		display = b.write( display );
		return { label: 'grade', quad: displayQuad };

	} );
	// the temporal resolve and its canvas pass (TemporalAA.ts)
	if ( p.temporal ) {

		const { surfaceDepthPrev } = p.resources;
		let rainOverlay = p.resources.rainOverlay;
		// After the resolve (AfterResolve.ts): the rain's streaks and the storm's spray, unjittered,
		// into the overlay at the canvas's size, which the canvas pass lays over the resolved image
		// (with the composite at the pixel, and the nearest surface for the grade's sky test).
		const after = rainOverlay ? () => {

			g.addPass( 'rainAfter', 'scene', ( b ) => {

				b.read( surfaceDepth );
				rainOverlay = b.write( rainOverlay!, { clear: true } );
				return {
					label: 'rain after resolve', mrt: afterMRT, when: ( f ) => f.afterScene !== undefined && ! p.afterHeld(), whenLabel: 'streaks || spray, not held',
					scene: ( f ) => f.afterScene!, camera: ( f ) => f.camera,
				};

			} );
			// what the layer reads at a render texel (`rainInput`, `rainBehind`), while it runs
			let rainInput = p.resources.rainInput!, rainBehind = p.resources.rainBehind;
			const inputQuad = p.rainInputQuad!;
			g.addPass( 'rainInput', 'quad', ( b ) => {

				// (the grade's input: the sharp image)
				b.read( sharp );
				b.read( surfaceDepth );
				if ( rainAir ) {

					b.read( rainAir );
					// (after the overlay pass, which does not run after the resolve)
					b.readOptional( sceneColor );

				}
				// (the attachments in the struct's order; the MRT only declares them)
				rainInput = b.write( rainInput );
				if ( rainBehind ) rainBehind = b.write( rainBehind );
				return { label: 'rain layer', when: ( f ) => p.layerRuns( f ) && ! p.afterHeld(), whenLabel: 'rain || spray, not held', quad: inputQuad, mrt: rainBehind ? mrt( {} ) : null };

			} );
			// the grade's table, for the canvas pass that draws the layer
			let rainGrade = p.resources.rainGrade!;
			const gradeQuad = p.rainGradeQuad!;
			const temporal = p.temporal!;
			g.addPass( 'rainGrade', 'quad', ( b ) => {

				readExposure( b );
				rainGrade = b.write( rainGrade );
				return { label: 'rain layer', when: () => temporal.drawsLayer, whenLabel: 'layer in the canvas pass', quad: gradeQuad };

			} );
			// The change the rain makes to the graded image (`layerAt`), while it rains or the
			// overlay holds something: drawn here into `rainLayer` for the frame that completes the
			// rest (held after it) and for TAA's upscaled canvas pass, and by the canvas pass itself
			// in every other frame (TemporalAA.ts `layerInline`; one canvas-size target less to
			// write and read).
			let rainLayer = p.resources.rainLayer!;
			const layerQuad = p.layerQuad!;
			g.addPass( 'rainLayer', 'quad', ( b ) => {

				b.read( surfaceDepth );
				b.read( display );
				if ( toneGrid ) b.read( toneGrid );
				readExposure( b );
				b.readOptional( rainInput );
				if ( rainBehind ) b.readOptional( rainBehind );
				b.readOptional( rainOverlay! );
				rainLayer = b.write( rainLayer );
				return { label: 'rain layer', when: ( f ) => p.layerRuns( f ) && ! p.afterHeld() && ! temporal.drawsLayer, whenLabel: 'rain || spray, not held, not inline', quad: layerQuad };

			} );
			return { optional: [ rainLayer, rainOverlay!, rainInput, ...( rainBehind ? [ rainBehind ] : [] ), rainGrade ], required: [ surfaceDepth, display, exposureState, ...( toneGrid ? [ toneGrid ] : [] ) ] };

		} : undefined;
		p.temporal.declare( { display, velocity: velocity!, surfaceDepth, surfaceDepthPrev: surfaceDepthPrev!, veilReactive, canvas, after } );
		return;

	}
	// to the canvas: copied at the display's resolution, upscaled below it
	g.addPass( 'output', 'chain', ( b ) => {

		b.read( display );
		canvas = b.write( canvas );
		return { label: 'output', when: () => ! p.upscaling, whenLabel: '!upscaling', pipeline: p.output! };

	} );
	const fsr = p.fsr, upscaledOutput = p.upscaledOutput, upscaledOutputThree = p.upscaledOutputThree;
	if ( easu && fsr && upscaledOutput ) {

		g.addPass( 'easu', 'quad', ( b ) => {

			b.read( display );
			easu = b.write( easu! );
			return { label: 'output & FSR', when: () => p.upscaling, whenLabel: 'upscaling', quad: fsr.quad };

		} );
		g.addPass( 'upscaledOutput', 'chain', ( b ) => {

			b.readOptional( easu! );
			canvas = b.write( canvas );
			return { label: 'output & FSR', when: () => p.upscaling, whenLabel: 'upscaling', pipeline: upscaledOutput };

		} );

	} else if ( upscaledOutputThree ) {

		// three's FSR1Node: its passes run inside the chain, at the canvas's size
		g.addPass( 'upscaledOutputThree', 'chain', ( b ) => {

			b.read( display );
			canvas = b.write( canvas );
			return { label: 'output & FSR', when: () => p.upscaling, whenLabel: 'upscaling', pipeline: upscaledOutputThree };

		} );

	}

}

// The scene half under the temporal modes: one sample, jittered (the passes from the opaque scene
// to the overlay are the graph's jittered span), the motion vectors written beside the colour.
// The opaque and water passes draw with the scene MRT (sceneMRT), the rain and the storm's spray
// in the overlay pass with the overlay MRT, which keeps the motion of the surfaces under them
// (Motion.ts). Returns the scene colour's, the nearest surface's, the velocity's, the opaque
// copy's, the metering illuminance's and the opaque depth's last versions.
function declareTemporalScene( p: Pipeline, hazeReads: ResourceHandle[] ): [ ResourceHandle, ResourceHandle, ResourceHandle, ResourceHandle, ResourceHandle, ResourceHandle ] {

	const g = p.graph;
	let { sceneColor, sceneDepth, linearDepth, opaqueColor, surfaceDepth, meterLight } = p.resources;
	let velocity = p.resources.velocity!;
	g.addPass( 'opaque', 'scene', ( b ) => {

		// (the sky dome reads the haze's light volume)
		for ( const h of hazeReads ) b.read( h );
		readDisplacement( p, b );
		sceneColor = b.write( sceneColor, { clear: true } );
		velocity = b.write( velocity, { clear: true } );
		meterLight = b.write( meterLight, { clear: true } );
		sceneDepth = b.depth( sceneDepth, { write: true, clear: true } );
		return { label: 'opaque scene', mrt: sceneMRT, jittered: true, scene: ( f ) => f.scene, camera: ( f ) => f.camera };

	} );
	const [ underwater, refraction, refractionDepth ] = declareRefraction( p, sceneColor, sceneDepth );
	// (one sample: the farthest of the samples is sample 0)
	g.addPass( 'depthResolve', 'quad', ( b ) => {

		b.read( sceneDepth, 'samples' );
		linearDepth = b.write( linearDepth );
		return { label: 'depth & copy resolves', jittered: true, quad: p.depthQuad };

	} );
	p.depthPyramid.pass( g, linearDepth, true, hasWater );
	// single-sampled, the opaque colour is copied as a texture (no draw: it replaces every texel)
	const from = g.texture( sceneColor ), to = g.texture( opaqueColor );
	g.addPass( 'opaqueCopy', 'callback', ( b ) => {

		b.read( sceneColor );
		opaqueColor = b.write( opaqueColor, { clear: true } );
		return { label: 'depth & copy resolves', jittered: true, copy: true, run: () => p.renderer.copyTextureToTexture( from, to ) };

	} );
	const seaReflected = p.seaReflection.passes( g, { linearDepth, opaqueColor, pyramid: p.depthPyramid, iterations: p.seaIterations }, true, SEA_HOOKS );
	g.addPass( 'water', 'scene', ( b ) => {

		b.read( opaqueColor );
		b.read( linearDepth );
		b.read( underwater );
		b.readOptional( refraction );
		b.readOptional( refractionDepth );
		b.readOptional( seaReflected );
		sceneColor = b.write( sceneColor );
		velocity = b.write( velocity );
		meterLight = b.write( meterLight );
		sceneDepth = b.depth( sceneDepth, { write: true } );
		return {
			label: 'water', mrt: sceneMRT, jittered: true, before: ( f ) => waterBefore( p, f ),
			scene: ( f ) => f.water ? f.waterScene : NO_WATER, camera: ( f ) => f.camera,
		};

	} );
	g.addPass( 'surfaceResolve', 'quad', ( b ) => {

		b.read( sceneDepth, 'samples' );
		surfaceDepth = b.write( surfaceDepth );
		return { label: 'depth & copy resolves', jittered: true, quad: p.surfaceQuad };

	} );
	// The rain and the spray over the finished scene: their colour blends as each material says,
	// the velocity under them is kept and its reactive weight takes the larger value. One scene
	// draws both, sorted by renderOrder: the spray (20) over the rain, where msaa draws the spray
	// in the water's pass and the rain over it, so where both show the image differs a little.
	g.addPass( 'overlay', 'scene', ( b ) => {

		b.read( linearDepth );
		b.read( surfaceDepth );
		sceneColor = b.write( sceneColor, { blend: 'material' } );
		velocity = b.write( velocity, { blend: 'keep-max' } );
		meterLight = b.write( meterLight, { blend: 'keep' } );
		b.depth( sceneDepth, { write: false } );
		return {
			label: 'overlay', mrt: overlayMRT, jittered: true, when: ( f ) => f.overlay === true && f.rainScene !== undefined, whenLabel: 'rain || spray',
			scene: ( f ) => f.rainScene!, camera: ( f ) => f.camera,
		};

	} );
	// the nearest surface's depth, kept for the next frame (the resolve's; FSR keeps its own)
	if ( p.resources.surfaceDepthPrev ) g.copyToPersistent( surfaceDepth, p.resources.surfaceDepthPrev );
	return [ sceneColor, surfaceDepth, velocity, opaqueColor, meterLight, linearDepth ];

}

// What the water refracts, right after the opaque pass, with the scene's jitter under the temporal
// modes: the underwater pyramid from the opaque pass's colour and samples, then the refraction
// source (ocean/Refraction.ts), the world under the sea through the wider view, shaded only
// outside the mask drawn first into its depth (see refractionMaskQuad). The refraction and its
// mask are off with `flags.refraction = false` (the water then refracts the opaque image alone:
// Water.ts). Returns the pyramid's, the refraction colour's and its depth's versions.
function declareRefraction( p: Pipeline, sceneColor: ResourceHandle, sceneDepth: ResourceHandle ): [ ResourceHandle, ResourceHandle, ResourceHandle ] {

	const g = p.graph;
	let { underwater, refractionReach, refraction, refractionDepth } = p.resources;
	g.addPass( 'underwaterPyramid', 'quad', ( b ) => {

		b.read( sceneColor );
		b.read( sceneDepth, 'samples' );
		underwater = b.write( underwater );
		return { label: 'underwater pyramid', jittered: TEMPORAL, quad: p.underwaterQuad };

	} );
	const on = () => debugFlags.refraction !== false;
	g.addPass( 'refractionReach', 'quad', ( b ) => {

		b.read( sceneDepth, 'samples' );
		b.read( underwater );
		refractionReach = b.write( refractionReach );
		return { label: 'refraction', jittered: TEMPORAL, when: on, whenLabel: 'refraction', quad: p.refractionReachQuad };

	} );
	g.addPass( 'refractionMask', 'quad', ( b ) => {

		b.read( underwater );
		// (the reach runs whenever the mask does)
		b.readOptional( refractionReach );
		refraction = b.write( refraction, { clear: true } );
		refractionDepth = b.depth( refractionDepth, { write: true, clear: true } );
		return { label: 'refraction', mrt: REFRACTION_MRT, jittered: TEMPORAL, when: on, whenLabel: 'refraction', quad: p.refractionMaskQuad };

	} );
	g.addPass( 'refraction', 'scene', ( b ) => {

		refraction = b.write( refraction );
		refractionDepth = b.depth( refractionDepth, { write: true } );
		return {
			label: 'refraction', mrt: REFRACTION_MRT, jittered: TEMPORAL, when: on, whenLabel: 'refraction',
			before: ( f ) => p.refraction.update( f.camera ), scene: ( f ) => f.scene, camera: () => p.refraction.camera,
		};

	} );
	return [ underwater, refraction, refractionDepth ];

}
