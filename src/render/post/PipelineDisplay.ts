/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, texture, textureLoad, uv, screenCoordinate, ivec2, vec2, vec3, vec4, float, mix,
	smoothstep, clamp, max, min, length, sin, log2, exp2, floor, luminance, sRGBTransferEOTF, sqrt,
	pow, outputStruct, mat4, atan, cos, abs, select, mod, step, uniform,
} from 'three/tsl';
import { fsr1 } from 'three/addons/tsl/display/FSR1Node.js';
import { FSR } from './FSR.ts';
import { PhysicalBloom } from './PhysicalBloom.ts';
import { Metering } from './Metering.ts';
import { purkinjeScale, purkinjeShift } from './Purkinje.ts';
import { AA_MODE, AFTER_RESOLVE, FSR3, RESOLVE, TEMPORAL } from './Antialiasing.ts';
import { onTaaCut } from './Motion.ts';
import { preExposed, preExposure, sceneUnits } from './PreExposure.ts';
import { onSceneCut } from './SceneEvents.ts';
import { jitter } from './Jitter.ts';
import { TemporalAA, type TemporalMode } from './TemporalAA.ts';
import { Fsr3Upscaler } from './fsr3/Fsr3Upscaler.ts';
import { QUALITY, type TemporalSettings } from '../Quality.ts';
import { FrameGraph } from './graph/FrameGraph.ts';
import { shared } from '../shared.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { toneMatrices } from './ToneSpace.ts';
import { BLUE_NOISE, blueNoiseTexture } from './BlueNoise.ts';
import { sampleLut } from './GradeLut.ts';
import { LocalTone } from './LocalTone.ts';
import { tuneUniformOf } from '../debug/Tune.ts';
import { settings } from '../Settings.ts';
import { hasGpuFeature } from '../three/Backend.ts';
import type { DisplayTables } from '../look/DisplayLook.ts';
import type { Pipeline, PipelineFrame } from './Pipeline.ts';
type Extent = ReturnType<FrameGraph<PipelineFrame>[ 'extent' ]>;
type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// The display chain's materials: composite, bloom, tone mapping, grade, output and resolve.
//
// The display chain's materials (`buildDisplay`), which the pipeline's constructor builds in its
// place among its steps: the composite of the haze, the shafts and the rods' colour over the scene,
// the metering, the bloom, the sharp image, the local tone mapping, the grade and the display pass,
// the output and the upscale, the rain's layer after the resolve and the temporal resolve. Its
// numbers chosen for the look are the display's tables the pipeline was made with
// (look/DisplayLook.ts `createDisplayTables`, Pipeline.ts `look`); the pipeline sets its uniforms
// each frame (Pipeline.ts `updateDisplay`). The cool share's edges and overcast's highlights are
// uniforms built as this module loads, so they keep their place among the nodes, and set from the
// tables (their tunables registered) as the chain builds: the display chains of one page share
// them. The rules of the frame and its passes are Pipeline.ts's header.
//
// tidewater's display chain (PostFX.js, 4811ba4), as it has become: its physical bloom
// (PhysicalBloom.ts) added to the image; its grade in linear light (the vignette; its saturation
// and contrast left for the grade table's hue bands, look/DisplayLook.ts `baseGrade`, and the
// shade's contrast, `shadowContrast`); our tone curve (ToneCurve.ts) on the exposed colour, in the
// curve space of ToneSpace.ts with the white balance folded into its input matrix; the grade's
// tables; overcast's highlights toward grey (`highlightGrey`); the BT.709 OETF shown as sRGB, and a
// blue-noise dither (BlueNoise.ts; its luminance-weighted grain is gone). tidewater meters its
// frame's luminance and scales it by an auto exposure before the grade; here the meter reads the
// light on the scene (Metering.ts, Exposure.ts).

// The rain veil's reactive weight: its opacity at the pixel times `gain`, at most `max` (a dense veil
// lowers the history's weight there to 40 % of its frames; a single faint streak barely moves it).
const VEIL_REACTIVE = { gain: 1.5, max: 0.6 };
// After the resolve (AfterResolve.ts), the share of a pixel the rain covers in one sample, where its
// change to the graded colour is taken before it is spread over the pixel: a supersampled image
// grades each sample, which a streak covers fully or not at all, and averages the graded samples,
// so over the dark meadow and sea, in the curve's toe, its thin rain shows brighter than the same
// coverage spread before the grade. Thin streaks drawn at their least width cover about two thirds
// of the pixels their prefiltered quads touch (RainDrops.ts).
const AFTER_COVERAGE = 0.65;
// The rain layer's grade table (`gradeTable`): colours per channel, and the range they span (stops of
// the scene colour times the exposure, the display's white level at 0).
export const GRADE_TABLE = { size: 64, low: - 14, high: 7 };

// The tone curve, each channel through one curve in the curve's space: `input` takes the exposed
// linear Rec.709 there (with the white balance), `output` brings it back (ToneSpace.ts
// toneMatrices, without saturation of their own). The curve is a rational function of the third
// degree over the third (ToneCurve.ts), its coefficients solved on the CPU from the slope at black,
// what the exposed mid grey shows and the input that shows white (`settings.curve`).
export const TONE_SPACE = toneMatrices();
const toneCurve = Fn( ( [ c, input, output, num, den ]: [ V3, THREE.Node<'mat3'>, THREE.Node<'mat3'>, V3, V3 ] ) => {

	const x = max( input.mul( c ) as unknown as V3, vec3( 0 ) ).toVar();
	// ( a x^3 + b x^2 + c x ) / ( d x^3 + e x^2 + f x + 1 ), by Horner
	const n = x.mul( num.x ).add( num.y ).mul( x ).add( num.z ).mul( x );
	const d = x.mul( den.x ).add( den.y ).mul( x ).add( den.z ).mul( x ).add( 1 );
	return clamp( output.mul( clamp( n.div( d ), 0, 1 ) ) as unknown as V3, 0, 1 );

} ).setLayout( { name: 'toneCurve', type: 'vec3', inputs: [ { name: 'c', type: 'vec3' }, { name: 'input', type: 'mat3' }, { name: 'output', type: 'mat3' }, { name: 'num', type: 'vec3' }, { name: 'den', type: 'vec3' } ] } );
// The BT.709 OETF (ITU-R BT.709): 4.5 l under 0.018, 1.099 l^0.45 - 0.099 over it. Its code shown
// as sRGB is darker in the shadows than the linear value it encodes: the output's contrast.
const bt709 = Fn( ( [ l ]: [ V3 ] ) => {

	// (per channel: a 0/1 weight rather than a select of vectors)
	return mix( l.mul( 4.5 ), pow( max( l, vec3( 0.018 ) ), vec3( 0.45 ) ).mul( 1.099 ).sub( 0.099 ), step( vec3( 0.018 ), l ) );

} ).setLayout( { name: 'bt709', type: 'vec3', inputs: [ { name: 'l', type: 'vec3' } ] } );
// The local tone mapping (LocalTone.ts), on unless `flags=localTone:false` at load.
const LOCAL_TONE_ON = debugFlags.localTone !== false;
/** Where a colour counts as cool for the desaturation's `cool` share (look/DisplayLook.ts
 * `desaturation`): its blue less its red, as a share of its spread (max less min), from `COOL_FROM`
 * (none of it cool) to `COOL_TO` (all of it), read through the white balance (`balance`). Set from
 * the tables' `coolShare` as the chain builds (`buildDisplay`). */
const COOL_FROM = uniform( 0 );
const COOL_TO = uniform( 0 );
// overcast's highlights toward grey (look/DisplayLook.ts `highlightGrey`): the amount and the edges,
// set from the tables as the chain builds
const HIGHLIGHT_GREY = { amount: uniform( 0 ), from: uniform( 0 ), to: uniform( 0 ) };
/** How cool the colour `c` (white balanced) is, 0 to 1 (`coolFrom`, `coolTo`: COOL_FROM, COOL_TO). The
 * edges come in as inputs: a laid-out helper that reads uniforms in its body can be emitted with
 * another material's uniform names (r186), which broke the rain's materials under taa. */
const coolShare = Fn( ( [ c, coolFrom, coolTo ]: [ V3, F, F ] ) => {

	const spread = max( max( c.r, c.g ), c.b ).sub( min( min( c.r, c.g ), c.b ) );
	return smoothstep( coolFrom, coolTo, c.b.sub( c.r ).div( max( spread, 1e-6 ) ) );

} ).setLayout( { name: 'coolShare', type: 'float', inputs: [ { name: 'c', type: 'vec3' }, { name: 'coolFrom', type: 'float' }, { name: 'coolTo', type: 'float' } ] } );

// The night's grade (art direction: a film's night, day for night): the meadow's greens turned
// toward blue-green, the frame a touch silver-blue, the bloom stronger so the moon, its clouds and
// its path on the sea carry a small cool halo. The turn is a rotation in the opponent plane at the
// centre of the green band (105 degrees, shading off over 55 degrees either side), weighted by
// saturation, so grey stone, sand, skin and the navy sea and sky keep their colours; it keeps the
// channels' mean, so the meadow keeps its brightness. Moonlight and the night sky are blue, but the
// meadow's yellow-green albedo kept it a dark olive under them, and a light blue enough to turn it
// turned the stone, sand and skin blue too. 0.4 radians takes the meadow to about 143 degrees, a
// cool green toward blue-green (0.65 read as teal). The tint is drawn toward at `tint` and keeps
// the luminance. The shadows' readability comes from the sky light (look/SkyLook.ts `clearSkyFill`),
// not a lift. Off by default with the clear-sky fill light (`clearSkyFill`), as the night keeps the moon's own light (`flags=filmNight:true`
// at load for A/Bs).
export const NIGHT = debugFlags.filmNight === true ? { greenTurn: 0.4, tint: [ 0.84, 0.96, 1.16 ] as const, amount: 0.3, bloom: 1.5 } : { greenTurn: 0, tint: [ 0.84, 0.96, 1.16 ] as const, amount: 0, bloom: 0 };
const nightGreenTurn = Fn( ( [ c0, turn ]: [ V3, F ] ) => {

	const c = c0.toVar();
	const ca = c.r.sub( c.g.add( c.b ).mul( 0.5 ) ).toVar();
	const cb = c.g.sub( c.b ).mul( Math.sqrt( 3 ) / 2 ).toVar();
	// (atan2 ( 0, 0 ) is undefined on some backends; a grey's hue is keyed out below anyway)
	const hue = atan( cb, select( abs( ca ).add( abs( cb ) ).greaterThan( 1e-7 ), ca, float( 1 ) ) );
	const peak = max( c.r, max( c.g, c.b ) );
	const sat = peak.sub( min( c.r, min( c.g, c.b ) ) ).div( max( peak, 1e-5 ) );
	const band = float( 1 ).sub( smoothstep( 0, 55 * Math.PI / 180, abs( mod( hue.sub( 105 * Math.PI / 180 ).add( Math.PI ), 2 * Math.PI ).sub( Math.PI ) ) ) );
	const t = band.mul( smoothstep( 0.04, 0.3, sat ) ).mul( turn ).toVar();
	const ra = ca.mul( cos( t ) ).sub( cb.mul( sin( t ) ) ).toVar();
	const rb = ca.mul( sin( t ) ).add( cb.mul( cos( t ) ) ).toVar();
	const m = c.r.add( c.g ).add( c.b ).div( 3 ).toVar();
	const k = 1 / Math.sqrt( 3 );
	return max( vec3( m.add( ra.mul( 2 / 3 ) ), m.sub( ra.div( 3 ) ).add( rb.mul( k ) ), m.sub( ra.div( 3 ) ).sub( rb.mul( k ) ) ), vec3( 0 ) );

} ).setLayout( { name: 'nightGreenTurn', type: 'vec3', inputs: [ { name: 'c', type: 'vec3' }, { name: 'turn', type: 'float' } ] } );
const NIGHT_TINT = ( () => {

	const [ r, g, b ] = NIGHT.tint, l = r * 0.2126 + g * 0.7152 + b * 0.0722;
	return new THREE.Vector3( r / l, g / l, b / l );

} )();

/** The vignette's factor at screen uv `at` for its `strength`, of the shape `shape`
 * (look/DisplayLook.ts `vignette`). */
export function vignetteFactor( shape: DisplayTables[ 'vignette' ], at: V2, strength: F ): F {

	const fall = smoothstep( shape.inner, shape.outer, length( at.sub( 0.5 ) ) ).mul( shape.depth );
	return mix( float( 1 ), fall.oneMinus(), strength );

}

/** What the display chain takes from the pipeline's constructor besides the pipeline: the tier's
 * resolve settings, the graph's render and canvas extents, the exposure's read (a new node at each
 * call) and one read of it, whether the display holds linear light, and the graph's dev checks. */
export interface DisplayBuild {
	temporal: TemporalSettings | undefined;
	renderSize: Extent; canvasSize: Extent;
	exposureRead: () => F; exposure: F;
	linearDisplay: boolean; dev: boolean;
}

/** Builds the display chain's materials into the pipeline `p`, in the constructor's place for
 * them (after the scene half's resolves, before the passes are declared). */
export function buildDisplay( p: Pipeline, c: DisplayBuild ): void {

	const renderer = p.renderer, sky = p.sky, rain = p.rain, g = p.graph, u = p.u, look = p.look;
	// the cool share's edges and overcast's highlights, from the tables (tunable on debug pages,
	// debug/Tune.ts: `post.desaturate.coolFrom`, `coolTo`, `post.highlight.amount`, `from`, `to`)
	tuneUniformOf( COOL_FROM, 'post.desaturate.coolFrom', look.coolShare.from, { min: - 1, max: 1 } );
	tuneUniformOf( COOL_TO, 'post.desaturate.coolTo', look.coolShare.to, { min: - 1, max: 1 } );
	for ( const k of [ 'amount', 'from', 'to' ] as const ) tuneUniformOf( HIGHLIGHT_GREY[ k ], `post.highlight.${ k }`, look.highlightGrey[ k ], { min: 0, max: 1 } );
	const { sceneColor, surfaceDepth, velocity, veilReactive, opaqueColor, meterLight, exposureStatePrev, rainAir, rainOverlay, rainLayer, rainInput, rainBehind, rainGrade, hdr, sharp, display, easu } = p.resources;
	const { temporal, renderSize, canvasSize, exposureRead, exposure, linearDisplay: LINEAR_DISPLAY, dev: DEV } = c;
	// ---- composite: the scene target holds the opaque image with the sea and the rain drawn over it
	const sceneTex = texture( g.texture( sceneColor ) );
	const tSurfaceDepth = p.textures.surfaceDepth;
	// (the air's alpha: the rain veil's opacity, for the resolve's veilReactive; after the resolve,
	// where the veil is drawn, the air's transmittance)
	const veil = rain && ! AFTER_RESOLVE ? ( d: V3, distance: F, background: V3 ) => rain.veil( d, distance, background ) : undefined;
	// What the post passes take that is the same for every pixel of a frame, evaluated once into
	// a texel by a pass of its own and loaded there: the flare's visibility (a: LightShafts.ts
	// `occlusion`, the depth taps over its body's disk times the sky constants' cloud
	// transmittance over it, the sun's disk's or the moon's) and the rods' colour's retinal scale
	// (r: Purkinje.ts `purkinjeScale`). In full float, so the readers get exactly the value it computed itself.
	const constants = p.constants.output = g.color( 'postConstants', { size: { of: 'fixed', width: 1, height: 1 }, format: THREE.RGBAFormat, type: THREE.FloatType } );
	const constant = textureLoad( texture( g.texture( constants ) ), ivec2( 0 ) ) as unknown as THREE.Node<'vec4'>;
	// (in scene units: the haze, the shafts and the rods' colour are; hdr is pre-exposed again)
	// (the opaque scene's depth tells the land from the sea for the air's light over the land)
	const aired = p.air.apply( sceneUnits( sceneTex.rgb ), tSurfaceDepth, p.textures.linearDepth, sky, g, rainAir ? ( _d, _distance, background, transmittance ) => vec4( background, transmittance ) : veil );
	const shafts = p.shafts.build( tSurfaceDepth, sky, g );
	// The sky's own per-frame constants, which the flare's visibility reads (the post constants
	// below): the mean cloud transmittance over the sun's disc and over the moon's (Sky.ts
	// `discCloudTransmittance`). The dome's discs take the clouds' transmittance per pixel instead.
	p.skyConstants.output = g.color( 'skyConstants', { size: { of: 'fixed', width: 1, height: 1 }, format: THREE.RGBAFormat, type: THREE.FloatType } );
	const skyConstantsMat = new THREE.NodeMaterial();
	skyConstantsMat.name = 'Sky.Constants';
	skyConstantsMat.fragmentNode = sky.discCloudTransmittance();
	p.skyConstants.quad = new THREE.QuadMesh( skyConstantsMat );
	p.skyConstants.quad.name = 'Sky.Constants';
	const constantsMat = new THREE.NodeMaterial();
	constantsMat.name = 'Post.Constants';
	const discCloud = textureLoad( texture( g.texture( p.skyConstants.output ) ), ivec2( 0 ) ) as unknown as THREE.Node<'vec4'>;
	// (r: the rods' colour's retinal scale, from the light that falls on open level ground: the
	// key's share at its lifted cosine and the sky's irradiance lent to the land, at its fill;
	// Purkinje.ts `purkinjeScale`)
	const openLight = u.purkinjeKey.add( luminance( sky.landIrradianceNode ).mul( shared.skyFill ) );
	constantsMat.fragmentNode = vec4( purkinjeScale( u.purkinjeRetina, openLight ), 0, 0, p.shafts.occlusion( tSurfaceDepth, g.extent( { of: 'render' } ), discCloud.r, discCloud.g ) );
	p.constants.quad = new THREE.QuadMesh( constantsMat );
	p.constants.quad.name = 'Post.Constants';
	// with the rods' colour by night (Purkinje.ts, off: its strength 0; its retinal scale the
	// constants' r), so the bloom, the local tone's grid and the grade all read it
	const composited = preExposed( purkinjeShift( aired.rgb.add( shafts ), constant.r, u.purkinje ) );
	// Rendered once (the composite pass, into hdr), the composite feeds the sharp image and both
	// blooms' high passes; as an expression each of them evaluated it again, the haze's
	// tricubic lookups included. The high passes run at half resolution and sample it bilinearly, as they sampled
	// the scene colour before.
	let tHdr: THREE.TextureNode | null = null;
	if ( hdr ) {

		const compositeMat = new THREE.NodeMaterial();
		compositeMat.name = 'Post.Composite';
		// (under the resolve also the veil's reactive weight, the second target: VEIL_REACTIVE)
		compositeMat.fragmentNode = veilReactive
			? outputStruct( vec4( composited, 1 ), vec4( min( aired.a.mul( VEIL_REACTIVE.gain ), VEIL_REACTIVE.max ) ) )
			: vec4( composited, 1 );
		// After the resolve the air's transmittance instead, the second target: the composite is
		// affine in the scene colour at a pixel, hdr = A scene + B (A the transmittance, B the light
		// the haze and the shafts add), so a streak drawn after the resolve takes the air of what
		// lies behind it as it would have here (AfterResolve.ts).
		if ( rainAir ) compositeMat.fragmentNode = outputStruct( vec4( composited, 1 ), vec4( aired.a ) );
		p.compositeQuad = new THREE.QuadMesh( compositeMat );
		p.compositeQuad.name = 'Post.Composite';
		// (with the uv rtt() gave its texture: a texture() without one transforms it)
		// Under fsr3 what follows the composite reads FSR's output instead (Fsr3Upscaler.ts: its
		// passes take hdr, declared after the composite pass).
		if ( FSR3 ) ( p as { fsr3: Fsr3Upscaler | null } ).fsr3 = new Fsr3Upscaler( renderer, g as unknown as FrameGraph<unknown>, {
			hdr, surfaceDepth, velocity: velocity!, veilReactive, sceneColor, opaqueColor,
		}, { settings: temporal ?? QUALITY.high.taa, exposure: exposureRead, debugView: DEV && settings.has( 'fsrDebug' ) } );
		tHdr = p.fsr3 ? p.fsr3.source( uv() ) : texture( g.texture( hdr ), uv() );

	}
	const hdrColor: V3 = tHdr ? tHdr.rgb : composited;
	// the metering histograms of hdr (without the composite pass they are never taken, and the
	// meter reads none) and the meter, from the sky's geometric-mean luminance (Exposure.ts; the
	// log the dome's metering illuminance is made of, set by main.ts before the pipeline)
	const metering = p.metering = new Metering( g.texture( hdr ?? sceneColor ), g.texture( meterLight ), tSurfaceDepth, renderSize, { projInv: p.u.projInv, cameraWorld: p.u.cameraWorld, atmosphere: p.sky.atmosphere, gains: p.sky.gapGain } );
	metering.buildMeter( shared.skyMeterLog2, g.texture( exposureStatePrev ) );
	// the adaptation starts over at every cut, in every mode (SceneEvents.ts; the temporal modes'
	// camera jumps that no event covers through Motion.ts)
	onSceneCut( ( reason ) => {

		p.exposureReset = true;
		// a scene jump takes a fresh load's pre-exposure; a loop restart or the page shown again
		// keep the scene (SceneEvents.ts `SAME_SCENE`)
		if ( reason !== 'loop start' && reason !== 'visible' ) preExposure.reset();

	} );
	if ( TEMPORAL ) onTaaCut( () => { p.exposureReset = true; } );

	// ---- tidewater's physical bloom (PhysicalBloom.ts): from hdr (under fsr3 FSR's output), or the
	// scene's colour where the composite is evaluated inline (`post=inline-composite`), added to
	// the image at its strength
	// (the sky: its surface past the far plane, view z below -9e4, at the render texel)
	const isSky = ( at: V2 ): F => select( ( textureLoad( tSurfaceDepth, ivec2( clamp( at.mul( renderSize ), vec2( 0 ), renderSize.sub( 1 ) ) ) ) as unknown as F ).lessThan( - 9e4 ), float( 1 ), float( 0 ) );
	// The flare (the sun's, or the moon's while it is the key) joins the first level
	// (LightShafts.ts `flare`), divided by the strength the result comes out with (the levels'
	// weights' sum, times the factor it is added with), so it shows only through the bloom.
	const frame = g.extent( { of: 'render' } );
	// (in hdr's units: pre-exposed)
	const flare = ( q: V2, strength: F ) => p.shafts.flare( q, frame.x.div( frame.y ), constant.a, strength.mul( u.bloom ) ).mul( preExposure.scale );
	const bloom = p.bloom = new PhysicalBloom( look, g, tHdr ?? sceneTex, exposure, isSky, flare );
	const withBloom = hdrColor.add( bloom.read( uv() ).mul( u.bloom ) );

	// ---- the sharp image: the composite and the blooms (the composite evaluated here, inline,
	// without the composite pass)
	const sharpMat = new THREE.NodeMaterial();
	sharpMat.name = 'Post.Sharp';
	sharpMat.fragmentNode = vec4( withBloom, 1 );
	p.sharpQuad = new THREE.QuadMesh( sharpMat );
	p.sharpQuad.name = 'Post.Sharp';
	// (the sharp image as its rtt() node was: with its uv, no uv transform)
	const tSharp = texture( g.texture( sharp ), uv() );

	// (at the pixel drawn, or at render uv `q`: the rain's layer, AfterResolve.ts)
	const sharpAt = ( q?: V2 ): V3 => ( q ? tSharp.sample( q ) : tSharp ).rgb;
	// the local tone mapping's grid of the sharp image (LocalTone.ts)
	const localTone = ( p as { localTone: LocalTone | null } ).localTone = LOCAL_TONE_ON ? new LocalTone( g, tSharp, exposure, renderSize ) : null;
	// scene radiance at screen uv `q` with the local tone mapping's change there (for the grade's
	// table, which holds the rest of the grade)
	const localAt = ( lin: V3, q: V2 ): V3 => localTone ? localTone.apply( lin.mul( exposure ), q ).div( exposure ) : lin;

	// ---- tidewater's grade, to display values: the scene at the exposure, the vignette (the
	// resolve leaves it to the canvas pass, after the accumulation), the tone curve with the
	// white balance, the grade's tables, overcast's highlights toward grey (`highlightGrey`) and the
	// BT.709 OETF (its saturation and contrast are the
	// table's, GradeLut.ts) (`encode` false: the output code's sRGB decode). `at`: the
	// screen uv whose local tone mapping it takes (none for the grade's table).
	// the vignette's factor at the pixel being drawn (`vignette`): round in the frame's own uv, the
	// same into an image of any size
	const vignetteAt = ( _size: THREE.UniformNode<'vec2', THREE.Vector2> ) => {

		return vignetteFactor( look.vignette, uv(), u.vignette );

	};
	const grade = ( lin: V3, vignette: boolean, encode: boolean, at?: V2 ): V3 => {

		const c = lin.mul( exposure ).toVar();
		// the desaturation (`desaturation`), before the local gain
		c.addAssign( vec3( luminance( c ) ).sub( c ).mul( u.desaturate.mul( mix( float( 1 ), coolShare( u.balance.mul( c ), COOL_FROM, COOL_TO ), u.desaturateCool ) ) ) );
		if ( localTone && at ) c.assign( localTone.apply( c, at ) );
		// the shadows' contrast (`shadowContrast`)
		c.mulAssign( exp2( min( log2( max( luminance( c ), 1e-6 ) ).sub( log2( u.shadePivot ) ), 0 ).mul( u.shadeSlope ) ) );
		// the night's grade (NIGHT): the greens turned toward blue-green, a silver-blue tint
		If( u.night.greaterThan( 0.001 ), () => {

			c.assign( nightGreenTurn( c, u.night.mul( NIGHT.greenTurn ) ) );
			c.mulAssign( mix( vec3( 1 ), vec3( NIGHT_TINT ), u.night.mul( NIGHT.amount ) ) );

		} );
		// the tone (TONE): highlights a touch golden and shadows a touch teal, the contrast round mid
		// grey softer
		const lights = smoothstep( 0.04, 0.9, luminance( c ) );
		c.mulAssign( mix( vec3( 0.95, 1.0, 1.04 ), vec3( 1.06, 1.0, 0.9 ), lights ).sub( 1 ).mul( u.tone ).add( 1 ) );
		c.assign( pow( max( c, vec3( 0 ) ).div( 0.18 ), vec3( float( 1 ).sub( u.tone.mul( 0.2 ) ) ) ).mul( 0.18 ) );
		if ( vignette ) c.mulAssign( vignetteAt( FSR3 ? canvasSize : renderSize ) );
		const t = ( toneCurve( c, u.toneIn, u.toneOut, u.toneNum, u.toneDen ) as V3 ).toVar();
		// the grade's tables, indexed by the square root of the curve's linear output (GradeLut.ts)
		const q = sqrt( max( t, vec3( 0 ) ) ).toVar();
		const looked = mix( sampleLut( p.luts[ 0 ], q ), sampleLut( p.luts[ 1 ], q ), u.lut ).toVar();
		// the output transfer: the tables' linear output (the square of what they hold) encoded by
		// the BT.709 OETF, a code the canvas shows as sRGB (`encode` false: that code's sRGB decode,
		// so the temporal canvas pass's sRGB encode gives the code back)
		const graded = looked.mul( looked ).toVar();
		// overcast's highlights toward grey (`highlightGrey`): a weight of 0 adds nothing
		const y = luminance( graded );
		graded.addAssign( vec3( y ).sub( graded ).mul( HIGHLIGHT_GREY.amount.mul( smoothstep( HIGHLIGHT_GREY.from, HIGHLIGHT_GREY.to, y ) ).mul( u.highlightShare ) ) );
		const v = ( bt709( clamp( graded, 0, 1 ) as V3 ) as V3 ).toVar();
		return encode ? v : sRGBTransferEOTF( v ) as V3;

	};
	// the blue-noise dither (BlueNoise.ts), after the output transfer: ( n - 0.5 ) / 255 on each
	// channel, n the tile's threshold at the pixel drawn (modulo its size), static; once per
	// canvas pixel, in whichever pass draws the canvas (`on` 0 leaves it out: a display pass
	// whose image is upscaled after)
	const noise = blueNoiseTexture();
	const finish = ( s: V3, _size: THREE.UniformNode<'vec2', THREE.Vector2>, on: F ) => {

		If( on.greaterThan( 0.5 ), () => {

			const p = ivec2( floor( screenCoordinate.xy ) ).toVar();
			const n = ( textureLoad( noise, ivec2( p.x.bitAnd( BLUE_NOISE - 1 ), p.y.bitAnd( BLUE_NOISE - 1 ) ) ) as unknown as V4 ).r;
			s.addAssign( n.sub( 0.5 ).div( 255 ) );

		} );
		return vec4( s, 1 );

	};

	// The graded image, at the render's size, always into one float target (the display pass),
	// which the canvas pass copies or upscales: one build of the grade, whatever the output.
	// Float, so the copy below hands the canvas exactly the values a direct draw would. The
	// temporal resolve accumulates it before the vignette and dither, which its canvas
	// pass adds.
	const displayMat = new THREE.NodeMaterial();
	displayMat.name = 'Post.Display';
	displayMat.fragmentNode = Fn( () => finish( grade( sharpAt(), ! RESOLVE, ! LINEAR_DISPLAY, uv() ), FSR3 ? canvasSize : renderSize, RESOLVE ? float( 0 ) : u.ditherOn ) )();
	p.displayQuad = new THREE.QuadMesh( displayMat );
	p.displayQuad.name = 'Post.Display';

	// At the display's resolution it is copied to the canvas, dither and all.
	const tDisplay = g.texture( display );
	if ( ! RESOLVE ) {

		p.output = new THREE.RenderPipeline( renderer, textureLoad( tDisplay, ivec2( screenCoordinate.xy ) ) );
		p.output.outputColorTransform = false;

	}

	// Below it, FSR 1 (AMD FidelityFX Super Resolution 1, FSR.ts) upscales the image (it expects
	// display values) before the dither, so the dither stays one display pixel fine and is not
	// sharpened. The browser's stretch of a small canvas was a plain bilinear filter. The canvas
	// pass sharpens EASU's result itself; three's FSR1Node, kept for A/B (post=three-fsr), took
	// a canvas-sized pass and target more.
	if ( easu ) {

		const fsr = p.fsr = new FSR( tDisplay, g.texture( easu ), hasGpuFeature( renderer, 'float32-filterable' ), g );
		p.upscaledOutput = new THREE.RenderPipeline( renderer, Fn( () => finish( fsr.rcas( p.sharpness ).toVar(), canvasSize, float( 1 ) ) )() );
		p.upscaledOutput.outputColorTransform = false;

	} else if ( ! RESOLVE && ! FSR3 ) {

		const upscaled = fsr1( texture( tDisplay ), p.sharpness );
		p.upscaledOutputThree = new THREE.RenderPipeline( renderer, Fn( () => finish( upscaled.rgb.toVar(), canvasSize, float( 1 ) ) )() );
		p.upscaledOutputThree.outputColorTransform = false;

	}
	// What the rain's layer pass draws at the canvas's size, for the canvas pass to add to the
	// resolved image after the resolve (AfterResolve.ts): the change the rain makes to the graded
	// colour at the pixel (without the vignette, the canvas pass's; in linear light under
	// `linear`), and in alpha 1 where it makes one. The rain is laid over the grade's input at the
	// render texel nearest the point the pixel shows (the display pass's own: the sharp image), in
	// scene radiance as it was before the grade, and that is graded: the change is the difference from what the display
	// pass graded at the texel. (Without `linear` the display holds encoded values: both are
	// graded here.) Blended after the grade instead, faint rain over the dark sea and meadow, in
	// the curve's toe, came out brighter than over the same pixels before it.
	// - The veil at the canvas pixel's unjittered ray, in front of what lies at that texel, as the
	//   composite laid it over the air (Rain.veil; the shafts the composite added over it now lie
	//   under it).
	// - Then the overlay of streaks, splashes and spray through the composite at that texel, which
	//   is A scene + B: A the air's transmittance (rainAir), B the grade's input less A scene.
	// Nothing while it is dry and the after pass did not draw (the overlay then holds an earlier
	// frame's: `afterOn`).
	const tAfterDepth = texture( tSurfaceDepth );
	const tAfterDisplay = texture( tDisplay );
	// What the layer reads at a render texel, drawn by a pass of its own at the render's size while
	// it runs (`rainInput`): the grade's input there (as the display pass has it: `diffused`) and
	// the air's transmittance (A), in `rainInput`; and the light the haze and the shafts add (B,
	// the grade's input less A times the scene), in `rainBehind`. Evaluated per canvas pixel instead
	// (six or more of them read each texel), they took a fifth of the layer's time.
	const tRainInput = rainInput ? texture( g.texture( rainInput ) ) : null;
	const tRainBehind = rainBehind ? texture( g.texture( rainBehind ) ) : null;
	// (one body for both attachments: an output struct is the material's own node, so the two are
	// a matrix's columns)
	const inputs = rainInput ? Fn( () => {

		const texel = ivec2( screenCoordinate.xy ).toVar();
		const before = sharpAt( vec2( texel ).add( 0.5 ).div( renderSize ) ).toVar();
		if ( ! rainAir ) return mat4( vec4( before, 1 ), vec4( 0 ), vec4( 0 ), vec4( 0 ) );
		const keep = ( textureLoad( texture( g.texture( rainAir ) ), texel ) as unknown as F ).toVar();
		const scene = ( textureLoad( sceneTex, texel ) as unknown as THREE.Node<'vec4'> ).rgb;
		return mat4( vec4( before, keep ), vec4( before.sub( scene.mul( keep ) ), 1 ), vec4( 0 ), vec4( 0 ) );

	} )() as unknown as { element( i: number ): THREE.Node<'vec4'> } : null;
	const inputNode = inputs ? ( rainAir ? outputStruct( inputs.element( 0 ), inputs.element( 1 ) ) : inputs.element( 0 ) ) as unknown as THREE.Node<'vec4'> : null;
	if ( inputNode ) {

		const inputMat = new THREE.NodeMaterial();
		inputMat.name = 'Post.RainInput';
		inputMat.fragmentNode = inputNode;
		p.rainInputQuad = new THREE.QuadMesh( inputMat );
		p.rainInputQuad.name = 'Post.RainInput';

	}
	// The grade as a table, for the canvas pass of a running frame (`layerAt( true )`): the grade of
	// GRADE_TABLE.size^3 colours, logarithmically spaced per channel (the scene colour times the
	// exposure from 2^low to 2^high: below, within two levels of black in the curve's toe; above,
	// burnt to white), baked by a pass of its own while the layer runs (`rainGrade`) and read trilinearly
	// (two bilinear taps). The change is the difference of two
	// readings (with the rain and without), so the table's error cancels to first order; graded in
	// full at every canvas pixel, the grade took a third of the layer's time. The frame that
	// completes the rest grades in full (its layer stands for every paused frame).
	const T = GRADE_TABLE;
	const gradeTableNode = rainGrade ? Fn( () => {

		const p = floor( screenCoordinate.xy ).toVar();
		const b = floor( p.x.div( T.size ) );
		const index = vec3( p.x.sub( b.mul( T.size ) ), p.y, b );
		const lin = exp2( index.div( T.size - 1 ).mul( T.high - T.low ).add( T.low ) ).div( exposure );
		return vec4( grade( lin, false, ! LINEAR_DISPLAY ), 1 );

	} )() as unknown as THREE.Node<'vec4'> : null;
	if ( gradeTableNode ) {

		const gradeMat = new THREE.NodeMaterial();
		gradeMat.name = 'Post.RainGrade';
		gradeMat.fragmentNode = gradeTableNode;
		p.rainGradeQuad = new THREE.QuadMesh( gradeMat );
		p.rainGradeQuad.name = 'Post.RainGrade';

	}
	const gradeTable = ( lin: V3 ): V3 => {

		const table = g.texture( rainGrade! );
		const c = clamp( log2( max( lin.mul( exposure ), vec3( 2 ** T.low ) ) ).sub( T.low ).div( T.high - T.low ), 0, 1 ).mul( T.size - 1 ).toVar();
		const b0 = floor( c.z ).toVar();
		const v = c.y.add( 0.5 ).div( T.size ).toVar();
		const slice = ( b: F ) => ( texture( table, vec2( c.x.add( 0.5 ).add( b.mul( T.size ) ).div( T.size * T.size ), v ) ).level( float( 0 ) ) as unknown as THREE.Node<'vec4'> ).rgb;
		return mix( slice( b0 ), slice( min( b0.add( 1 ), T.size - 1 ) ), c.z.sub( b0 ) );

	};
	// (`table`: graded by the table)
	const layerAt = ( table = false ) => Fn( () => {

		const out = vec4( 0 ).toVar();
		// the render texel nearest the point the pixel shows (its centre scaled to the render, less
		// this frame's jitter)
		const texel = ivec2( clamp( screenCoordinate.xy.mul( renderSize.div( canvasSize ) ).sub( jitter.uniforms.px ).sub( 0.5 ).round(), vec2( 0 ), renderSize.sub( 1 ) ) ).toVar();
		// what covers the pixel: the veil (at the pixel's unjittered ray, in front of what lies at the
		// texel) and the overlay, while either shows
		const veil = vec4( 0 ).toVar();
		if ( rain ) If( rain.uniforms.rain.greaterThan( 0.001 ), () => {

			// (the sky's -1e5 and anything past the farthest shell alike: they all lie beyond it)
			const { ray, direction } = p.air.viewRay( uv(), jitter.uniforms.unjitteredProjectionInverse );
			const distance = min( ( textureLoad( tAfterDepth, texel ) as unknown as F ).negate().mul( length( ray ) ), 1000 );
			veil.a.assign( rain.veilCoverage( direction, distance ) );

		} );
		const o = vec4( 0 ).toVar();
		If( u.afterOn.greaterThan( 0 ), () => {

			o.assign( textureLoad( texture( g.texture( rainOverlay! ) ), ivec2( screenCoordinate.xy ) ) );

		} );
		// (below a 256th the change shows in no level)
		If( max( veil.a, o.a ).greaterThan( 1 / 256 ), () => {

			if ( rain ) veil.rgb.assign( rain.veilLight( p.air.viewRay( uv(), jitter.uniforms.unjitteredProjectionInverse ).direction ).mul( veil.a ).mul( preExposure.scale ) );
			const input = ( textureLoad( tRainInput!, texel ) as unknown as THREE.Node<'vec4'> ).toVar();
			const before = input.rgb;
			// the veil over the grade's input
			const after = before.mul( float( 1 ).sub( veil.a ) ).add( veil.rgb ).toVar();
			If( o.a.greaterThan( 1 / 1024 ), () => {

				// then the overlay's radiance (premultiplied) through the composite, A scene + B (the
				// legacy inline composite writes no transmittance: the streaks then take no air)
				const a = min( o.a, 1 ).toVar();
				if ( tRainBehind ) after.assign( after.mul( float( 1 ).sub( a ) ).add( o.rgb.mul( input.a ) ).add( ( textureLoad( tRainBehind, texel ) as unknown as THREE.Node<'vec4'> ).rgb.mul( a ) ) );
				else after.assign( after.mul( float( 1 ).sub( a ) ).add( o.rgb ) );

			} );
			// (the change at the rain's own coverage of a sample, AFTER_COVERAGE, spread over the pixel)
			const full = max( before.add( after.sub( before ).div( AFTER_COVERAGE ) ), vec3( 0 ) );
			const at = vec2( texel ).add( 0.5 ).div( renderSize ).toVar();
			if ( table ) out.assign( vec4( gradeTable( localAt( full, at ) ).sub( gradeTable( localAt( before, at ) ) ).mul( AFTER_COVERAGE ), 1 ) );
			else {

				const graded = LINEAR_DISPLAY ? ( textureLoad( tAfterDisplay, texel ) as unknown as THREE.Node<'vec4'> ).rgb : grade( before, false, ! LINEAR_DISPLAY, at );
				out.assign( vec4( grade( full, false, ! LINEAR_DISPLAY, at ).sub( graded ).mul( AFTER_COVERAGE ), 1 ) );

			}

		} );
		return out;

	} )() as unknown as THREE.Node<'vec4'>;
	if ( rainOverlay ) {

		const layerMat = new THREE.NodeMaterial();
		layerMat.name = 'Post.RainLayer';
		layerMat.fragmentNode = layerAt();
		p.layerQuad = new THREE.QuadMesh( layerMat );
		p.layerQuad.name = 'Post.RainLayer';

	}
	// the temporal resolve, its passes declared after the display pass
	if ( RESOLVE ) ( p as { temporal: TemporalAA | null } ).temporal = new TemporalAA( g as unknown as FrameGraph<unknown>, renderer, {
		mode: AA_MODE as TemporalMode, settings: temporal ?? QUALITY.high.taa,
		vignette: vignetteAt, finish,
		// (the rain's layer, drawn by a pass of its own: in the canvas pass it held too many registers)
		// (read only while the pass runs: in fair weather the canvas pass reads nothing more)
		layer: rainLayer ? () => Fn( () => {

			const l = vec4( 0 ).toVar();
			If( u.layerOn.greaterThan( 0 ), () => {

				l.assign( textureLoad( texture( g.texture( rainLayer ) ), ivec2( screenCoordinate.xy ) ) );

			} );
			return l;

		} )() as unknown as THREE.Node<'vec4'> : undefined,
		// (and the layer itself, for the canvas pass of a running frame while it shows)
		layerInline: rainLayer ? { node: () => layerAt( true ), on: () => p.u.layerOn.value > 0 } : undefined,
	} );

}
