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
	abs, clamp, dot, float, floor, Fn, fract, If, instanceIndex, int, ivec2, length, max, min, mix,
	normalize, select, sin, smoothstep, storageTexture, texture, textureLoad, textureStore, vec2,
	vec3, vec4,
} from 'three/tsl';
import { HALF_MAX } from '../post/PreExposure.ts';
import type { Clouds } from './Clouds.ts';
import {
	BEYOND_SHELLS, catmullRom, layerDistance, over, type View, type ViewSlot, type ViewTextures,
} from './CloudModel.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// The clouds' view: its buffers for a view size and the kernels that trace and resolve it.
//
// The clouds' view (Clouds.ts, whose header says how it is traced, reprojected and resolved): its
// buffers for a view size and the kernels that trace and resolve it, built once and kept across
// sizes, each a function of the clouds. A name in backticks with no file is this file's,
// Clouds.ts's or CloudModel.ts's.

/** Build the view buffers for a view size, and the kernels for the first one. Later sizes keep
 * the kernels and their pipelines: their textures are swapped under them and the sizes they
 * index by are uniforms, so a resized window (and, on the tiers whose view follows the render
 * scale, a new scale) builds no compute pipeline mid-play (a kernel rebuilt per size built five). */
export function buildView( clouds: Clouds, width: number, height: number ): void {

	const make = ( w: number, h: number, name: string ) => {

		const t = new THREE.StorageTexture( w, h );
		t.type = THREE.HalfFloatType;
		t.minFilter = t.magFilter = THREE.LinearFilter;
		t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
		t.generateMipmaps = false;
		t.name = name;
		return t;

	};
	const kept = clouds.view;
	if ( kept ) clouds.disposeViewTextures( kept );
	const tw = Math.ceil( width / 4 ), th = Math.ceil( height / 4 );
	const trace = make( tw, th, 'cloudTrace' ), depth = make( tw, th, 'cloudTraceDepth' ), high = make( tw, th, 'cloudHighTrace' );
	// A flash lasts a fraction of a second, a pixel's history seconds: averaged in, a flash
	// would be printed in a 4 x 4 pattern and linger as a ghost (the history's clamp cannot
	// catch it in a dark sky). The lightning is kept out of the history and shown from each
	// frame's own trace instead, upsampled; its glow is soft.
	const flashTrace = make( tw, th, 'cloudFlashTrace' );
	const motion = make( tw, th, 'cloudMotion' ), boxMin = make( tw, th, 'cloudBoxMin' ), boxMax = make( tw, th, 'cloudBoxMax' );
	const history = [ make( width, height, 'cloudViewA' ), make( width, height, 'cloudViewB' ) ];
	const depthHistory = [ make( width, height, 'cloudViewDepthA' ), make( width, height, 'cloudViewDepthB' ) ];
	clouds.viewSize.value.set( width, height );
	clouds.traceSize.value.set( tw, th );
	clouds.traceWidth.value = tw;
	clouds.viewWidth.value = width;
	const textures: ViewTextures = { trace, depth, high, motion, flash: flashTrace, boxMin, boxMax, history, depthHistory };
	if ( kept ) {

		for ( const [ node, pick ] of kept.slots ) node.value = pick( textures );
		for ( const kernel of [ kept.kernels.trace, kept.kernels.high, kept.kernels.box ] ) kernel.count = tw * th;
		for ( const kernel of kept.kernels.resolve ) kernel.count = width * height;
		showView( clouds, { width, height, ...textures, kernels: kept.kernels, slots: kept.slots } );
		return;

	}
	// the kernels' texture nodes, kept to swap their textures at the next size
	const slots: ViewSlot[] = [];
	const stored = ( pick: ViewSlot[ 1 ] ) => {

		const node = storageTexture( pick( textures ) );
		slots.push( [ node, pick ] );
		return node;

	};
	const read = ( pick: ViewSlot[ 1 ] ) => {

		const node = texture( pick( textures ) );
		slots.push( [ node, pick ] );
		return node;

	};
	const marchView = clouds.marchView!, streaksAt = clouds.streaksAt!, rainAt = clouds.rainAt!;
	const C = clouds.cam, P = clouds.prev;

	const viewDir = ( uv: V2 ) => {

		const ndc = vec2( uv.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uv.y.mul( 2 ) ) ).mul( C.tan );
		return normalize( C.fwd.add( C.right.mul( ndc.x ) ).add( C.up.mul( ndc.y ) ) );

	};
	const tracePixel = () => ivec2( int( instanceIndex.mod( clouds.traceWidth ) ), int( instanceIndex.div( clouds.traceWidth ) ) );
	const tracedAt = ( tp: THREE.Node<'ivec2'> ) => vec2( tp ).mul( 4 ).add( clouds.slot ).add( 0.5 ).add( clouds.subPixel );
	const hash = ( px: V2 ) => fract( sin( dot( px, vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) );
	const project = ( d: V3, right: V3, upv: V3, fwd: V3, tanv: V2 ) => {

		const z = max( dot( d, fwd ), 1e-4 );
		const ndc = vec2( dot( d, right ), dot( d, upv ) ).div( tanv.mul( z ) );
		return vec2( ndc.x.mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( ndc.y.mul( 0.5 ) ) );

	};

	const traceOut = stored( ( t ) => t.trace ), flashOut = stored( ( t ) => t.flash ), depthOut = stored( ( t ) => t.depth );
	const traceKernel = Fn( () => {

		const tp = tracePixel().toVar();
		const rd = viewDir( tracedAt( tp ).div( clouds.viewSize ) ).toVar();
		// A random offset per pixel, stepped by the golden ratio over its traces: shared by a block,
		// the pixels' errors would correlate and leave the blocks visible once averaged, and a
		// structured pattern (interleaved gradient noise) leaves diagonal hatching.
		const jitter = fract( hash( vec2( tp ).mul( 4 ).add( clouds.slot ) ).add( clouds.frameNoise ) );
		const m = marchView( rd, jitter );
		const rainSample = rainAt( rd, C.tan.y.mul( 2 ).div( clouds.displayHeight ), jitter, true );
		const rain = rainSample.color.toVar();
		const traced = over( rain, vec4( m.L, m.T ) ).toVar();
		textureStore( traceOut, tp, vec4( min( traced.rgb.mul( clouds.writeScale ), vec3( HALF_MAX ) ), traced.a ) );
		textureStore( flashOut, tp, vec4( m.F.mul( rain.a ).add( rainSample.flash ), 1 ) );
		// The depth (m) under half float's largest value: an empty ray's depth, past the shell, passes
		// 65 km toward the horizon, where the haze in front has closed, and an inf stored there turned
		// the motion and the filters around it to NaN. Beside it the opacity and their product (the
		// depth in km), which the resolve filters, so a cloud's edge averages its depth by opacity
		// and not toward the empty texel's depth past it.
		const depth = min( m.depth, 6.5e4 ).toVar(), opacity = float( 1 ).sub( m.T ).toVar();
		textureStore( depthOut, tp, vec4( depth, opacity, depth.mul( opacity ).div( 1000 ), 1 ) );

	} )().compute( tw * th, [ 64 ] ).setName( 'Clouds.ViewTrace' );

	const depthNode = read( ( t ) => t.depth );
	const highOut = stored( ( t ) => t.high ), motionOut = stored( ( t ) => t.motion );
	const highKernel = Fn( () => {

		const tp = tracePixel().toVar();
		const rd = viewDir( tracedAt( tp ).div( clouds.viewSize ) ).toVar();
		const hi = vec4( 0, 0, 0, 1 ).toVar();
		If( rd.y.greaterThan( - 0.01 ), () => {

			hi.assign( streaksAt( rd, C.tan.y.mul( 2 ).div( clouds.viewSize.y ) ) );

		} );
		textureStore( highOut, tp, vec4( min( hi.rgb.mul( clouds.writeScale ), vec3( HALF_MAX ) ), hi.a ) );
		// Motion for the resolve: the closest cumulus sample around clouds block (like the closest
		// depth dilation of TAA), so a cloud's edges move with it and the foreground wins.
		const best = vec3( 1e9, 0, 0 ).toVar();
		const tmax = clouds.traceSize.sub( 1 );
		for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

			const dz = textureLoad( depthNode, ivec2( clamp( vec2( tp ).add( vec2( x, y ) ), vec2( 0 ), tmax ) ) ).toVar();
			const key = select( dz.y.greaterThan( 0.15 ), dz.x, float( 1e9 ) );
			If( key.lessThan( best.x ), () => {

				best.assign( vec3( key, dz.x, dz.y ) );

			} );

		}
		If( best.x.greaterThan( 1e8 ), () => {

			const own = textureLoad( depthNode, tp );
			best.assign( vec3( 0, own.x, own.y ) );

		} );
		textureStore( motionOut, tp, vec4( best.y, best.z, 0, 1 ) );

	} )().compute( tw * th, [ 64 ] ).setName( 'Clouds.ViewHigh' );

	const traceNode = read( ( t ) => t.trace ), highNode = read( ( t ) => t.high );
	const boxMinOut = stored( ( t ) => t.boxMin ), boxMaxOut = stored( ( t ) => t.boxMax );
	// The range of clouds frame's samples is taken over the clouds composited onto the sky: a
	// per-channel range of radiance and transmittance would pair the dark radiance of clear
	// samples with the low transmittance of cloudy ones, an opaque black cloud. The sky is the
	// one the dome lays the clouds over (Sky.ts `clearSky`: the atmosphere's, polarised, with
	// no night ambient, which is the clouds' own light): a history clamped over another sky
	// trades that sky's light for its own as its transmittance moves, and where the two skies
	// part (most before sunrise, when the night ambient is at its full) it printed dark and
	// light blocks that grew while the clock ran.
	const boxKernel = Fn( () => {

		const tp = tracePixel().toVar();
		const sky = clouds.atmosphere.skyRadiance( viewDir( tracedAt( tp ).div( clouds.viewSize ) ) ).mul( clouds.writeScale ).toVar();
		const tmax = clouds.traceSize.sub( 1 );
		// (the range starts at the first sample, so it holds at any scale)
		const lo = vec4( 0 ).toVar(), hi = vec4( 0 ).toVar();
		for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

			const q = ivec2( clamp( vec2( tp ).add( vec2( x, y ) ), vec2( 0 ), tmax ) );
			const c = over( textureLoad( traceNode, q ), textureLoad( highNode, q ) ).toVar();
			const composite = vec4( c.rgb.add( sky.mul( c.a ) ), c.a );
			lo.assign( x === - 1 && y === - 1 ? composite : min( lo, composite ) );
			hi.assign( x === - 1 && y === - 1 ? composite : max( hi, composite ) );

		}
		textureStore( boxMinOut, tp, vec4( min( lo.rgb, vec3( HALF_MAX ) ), lo.a ) );
		textureStore( boxMaxOut, tp, vec4( min( hi.rgb, vec3( HALF_MAX ) ), hi.a ) );

	} )().compute( tw * th, [ 64 ] ).setName( 'Clouds.ViewBox' );

	const motionNode = read( ( t ) => t.motion ), boxMinNode = read( ( t ) => t.boxMin ), boxMaxNode = read( ( t ) => t.boxMax );
	// from history[ src ] into history[ dst ]
	const resolve = ( src: number, dst: number ) => {

		const srcNode = read( ( t ) => t.history[ src ] ), dstOut = stored( ( t ) => t.history[ dst ] );
		const srcDepth = read( ( t ) => t.depthHistory[ src ] ), dstDepth = stored( ( t ) => t.depthHistory[ dst ] );
		// A trace's depth as the depth history holds it: the cumulus's opacity times its depth (km),
		// premultiplied in the trace, and that opacity, so the history and the upsampling's bilinear
		// filter average the depth by opacity as they average the colour; and beside them the thin
		// layer's opacity as seen through the cumulus and the rain in front of it (its trace's and
		// the march's transmittances), which the dome fogs at the layer's distance (Sky.ts
		// domeColour, `sampleViewFog`).
		const weighted = ( dz: V4, trace: V4, streaks: V4 ) => vec3( dz.z, dz.y, clamp( trace.a.mul( float( 1 ).sub( streaks.a ) ), 0, 1 ) );
		return Fn( () => {

			const p = ivec2( int( instanceIndex.mod( clouds.viewWidth ) ), int( instanceIndex.div( clouds.viewWidth ) ) ).toVar();
			const uv = vec2( p ).add( 0.5 ).div( clouds.viewSize ).toVar();
			const rd = viewDir( uv ).toVar();
			const out = vec4( 0, 0, 0, 1 ).toVar();
			const outDepth = vec3( 0 ).toVar();
			If( rd.y.greaterThan( - 0.035 ), () => {

				const tp = p.div( 4 ).toVar();
				const fresh = p.x.mod( 4 ).equal( int( clouds.slot.x ) ).and( p.y.mod( 4 ).equal( int( clouds.slot.y ) ) );
				const dz = textureLoad( motionNode, tp ).toVar();
				// Where clouds cloud point was last frame (camera motion and wind drift): the cumulus,
				// or the cirrus, which drifts with it, where no cumulus is in front (with no layer,
				// the sky beyond the shells).
				const pdC = normalize( rd.mul( dz.x ).add( clouds.camDelta ) );
				const farHeight = select( clouds.streaks.greaterThan( 0.01 ), clouds.streakHeight, float( BEYOND_SHELLS ) );
				const pdH = normalize( rd.mul( layerDistance( clouds.camY, rd.y, farHeight ) ).add( clouds.camDelta ) ).toVar();
				const pd = select( dz.y.greaterThan( 0.3 ), pdC, pdH );
				const puv = mix( project( pdH, P.right, P.up, P.fwd, P.tan ), project( pdC, P.right, P.up, P.fwd, P.tan ), smoothstep( 0.1, 0.5, dz.y ) ).toVar();
				const valid = clouds.historyValid.greaterThan( 0.5 ).and( dot( pd, P.fwd ).greaterThan( 0.01 ) ).and( puv.x.greaterThan( 0 ) ).and( puv.x.lessThan( 1 ) )
					.and( puv.y.greaterThan( 0 ) ).and( puv.y.lessThan( 1 ) ).and( pd.y.greaterThan( - 0.05 ) ).toVar();
				// This frame's samples upsampled (they sit at the slot of each block).
				const suv = vec2( p ).sub( clouds.slot ).div( 4 ).add( 0.5 ).div( clouds.traceSize ).toVar();
				const upTrace = traceNode.sample( suv ).level( float( 0 ) ).toVar(), upHigh = highNode.sample( suv ).level( float( 0 ) ).toVar();
				const upsampled = over( upTrace, upHigh ).toVar();
				const upsampledDepth = weighted( depthNode.sample( suv ).level( float( 0 ) ), upTrace, upHigh ).toVar();
				If( valid, () => {

					// Reprojected history clamped to the range of clouds frame's samples around it (with
					// a margin): real changes cannot leave ghosts, so samples average over many frames.
					// (taken to clouds dispatch's scale: `historyGain`)
					const reprojected = max( catmullRom( srcNode, puv, clouds.viewSize ), vec4( 0 ) ).toVar();
					const h = vec4( reprojected.rgb.mul( clouds.historyGain ), reprojected.a ).toVar();
					const lo = textureLoad( boxMinNode, tp ), hi = textureLoad( boxMaxNode, tp );
					const pad = hi.sub( lo ).mul( 0.25 ).add( vec4( vec3( 0.01 ).mul( clouds.writeScale ), 0.01 ) );
					const sky = clouds.atmosphere.skyRadiance( rd ).mul( clouds.writeScale ).toVar();
					const clamped = clamp( vec4( h.rgb.add( sky.mul( h.a ) ), h.a ), lo.sub( pad ), hi.add( pad ) ).toVar();
					const kept = vec4( max( clamped.rgb.sub( sky.mul( clamped.a ) ), vec3( 0 ) ), clamped.a );
					out.assign( select( clouds.rebuildK.lessThan( 0 ).and( clouds.clampHistory.greaterThan( 0.5 ) ), kept, h ) );
					// (the depth reprojected bilinearly, unclamped: no range of it shows)
					outDepth.assign( srcDepth.sample( puv ).level( float( 0 ) ).xyz );

				} ).Else( () => {

					out.assign( upsampled );
					outDepth.assign( upsampledDepth );

				} );
				If( fresh, () => {

					const curTrace = textureLoad( traceNode, tp ).toVar(), curHigh = textureLoad( highNode, tp ).toVar();
					const cur = over( curTrace, curHigh );
					// Average the jittered samples over time; a little shorter where clouds move fast on
					// screen. Rebuilding after a cut, each pixel takes its own first sample as is.
					const motionPx = length( puv.sub( uv ).mul( clouds.viewSize ) );
					const a = clamp( motionPx.mul( 0.1 ).add( clouds.minAlpha ), clouds.minAlpha, 0.35 );
					out.assign( select( valid.and( clouds.rebuildK.lessThan( 0 ) ), mix( out, cur, a ), cur ) );
					const curDepth = weighted( textureLoad( depthNode, tp ), curTrace, curHigh );
					outDepth.assign( select( valid.and( clouds.rebuildK.lessThan( 0 ) ), mix( outDepth, curDepth, a ), curDepth ) );

				} ).ElseIf( clouds.rebuildK.greaterThanEqual( 0 ).and( valid ), () => {

					// Pixels without their own sample since the cut average the upsampled samples of
					// every slot traced so far (their refresh rank is the 4 x 4 Bayer index).
					const x = float( p.x.mod( 4 ) ), y = float( p.y.mod( 4 ) );
					const b2 = ( a: F, b: F ) => abs( a.sub( b ) ).mul( 2 ).add( b );
					const rank = b2( x.mod( 2 ), y.mod( 2 ) ).mul( 4 ).add( b2( floor( x.div( 2 ) ), floor( y.div( 2 ) ) ) );
					If( rank.greaterThan( clouds.rebuildK ), () => {

						out.assign( mix( out, upsampled, float( 1 ).div( clouds.rebuildK.add( 1 ) ) ) );
						outDepth.assign( mix( outDepth, upsampledDepth, float( 1 ).div( clouds.rebuildK.add( 1 ) ) ) );

					} );

				} );

			} );
			textureStore( dstOut, p, vec4( min( out.rgb, vec3( HALF_MAX ) ), clamp( out.a, 0, 1 ) ) );
			textureStore( dstDepth, p, vec4( outDepth, 1 ) );

		} )().compute( width * height, [ 64 ] ).setName( 'Clouds.ViewResolve' );

	};
	showView( clouds, {
		width, height, ...textures, slots,
		kernels: { trace: traceKernel, high: highKernel, box: boxKernel, resolve: [ resolve( 1, 0 ), resolve( 0, 1 ) ] },
	} );

}
// the view's new buffers in use, its history started over
export function showView( clouds: Clouds, view: View ): void {

	clouds.view = view;
	clouds.viewNode.value = view.history[ 0 ];
	clouds.viewDepthNode.value = view.depthHistory[ 0 ];
	clouds.flashNode.value = view.flash;
	clouds.viewValid.value = 0;
	clouds.resetHistory();

}

/** The clouds as seen directly: the view, upsampled bicubically and looked up with the camera
 * it was traced with; directions outside it, or a view without data yet, take the panorama. */
export function sampleView( clouds: Clouds, direction: V3 ): V4 {

	const V = clouds.viewCam;
	const result = vec4( 0 ).toVar();
	const z = dot( direction, V.fwd ).toVar();
	const ndc = vec2( dot( direction, V.right ), dot( direction, V.up ) ).div( V.tan.mul( max( z, 1e-4 ) ) );
	const uv = vec2( ndc.x.mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( ndc.y.mul( 0.5 ) ) ).toVar();
	If( clouds.viewValid.greaterThan( 0.5 ).and( z.greaterThan( 0.01 ) ).and( uv.x.greaterThanEqual( 0 ) ).and( uv.x.lessThanEqual( 1 ) )
		.and( uv.y.greaterThanEqual( 0 ) ).and( uv.y.lessThanEqual( 1 ) ), () => {

		const v = max( catmullRom( clouds.viewNode, uv, clouds.viewSize ), vec4( 0 ) );
		// The last trace's samples sit at its slot and sub-pixel offset in each 4 x 4 block.
		const fuv = uv.mul( clouds.viewSize ).sub( clouds.slot ).sub( clouds.subPixel ).sub( 0.5 ).div( 4 ).add( 0.5 ).div( clouds.traceSize );
		const flash = clouds.flashNode.sample( fuv ).level( float( 0 ) ).rgb.mul( clouds.flashOn );
		const above = smoothstep( - 0.05, - 0.03, direction.y );
		result.assign( vec4( v.rgb.mul( clouds.viewUnscale ).add( flash ).mul( above ), mix( float( 1 ), min( v.a, 1 ), above ) ) );

	} ).Else( () => {

		// Only outside the view (or before it has data) does the panorama show, so only there
		// are its taps and the lightning's light evaluated.
		const panorama = clouds.sample( direction ).toVar();
		result.assign( vec4( panorama.rgb.add( clouds.flashAt( direction, float( 1 ).sub( panorama.a ) ) ), panorama.a ) );

	} );
	return result;

}
/** What the dome fogs the clouds by toward `direction`, as the view holds it beside its colour
 * (the resolve's depth history): `depth`, the distance (metres) to the cumulus's light, the
 * march's opacity-weighted depth (`makeMarch`), `fallback` outside the view, before it has data,
 * or where it holds no cumulus; and `thin`, the cirrus's opacity as seen through what lies
 * in front of it, 0 outside the view. */
export function sampleViewFog( clouds: Clouds, direction: V3, fallback: F ): { depth: F; thin: F } {

	const V = clouds.viewCam;
	const depth = fallback.toVar(), thin = float( 0 ).toVar();
	const z = dot( direction, V.fwd ).toVar();
	const ndc = vec2( dot( direction, V.right ), dot( direction, V.up ) ).div( V.tan.mul( max( z, 1e-4 ) ) );
	const uv = vec2( ndc.x.mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( ndc.y.mul( 0.5 ) ) ).toVar();
	If( clouds.viewValid.greaterThan( 0.5 ).and( z.greaterThan( 0.01 ) ).and( uv.x.greaterThanEqual( 0 ) ).and( uv.x.lessThanEqual( 1 ) )
		.and( uv.y.greaterThanEqual( 0 ) ).and( uv.y.lessThanEqual( 1 ) ), () => {

		const held = clouds.viewDepthNode.sample( uv ).level( float( 0 ) ).toVar();
		If( held.y.greaterThan( 1e-3 ), () => {

			depth.assign( held.x.div( held.y ).mul( 1000 ) );

		} );
		thin.assign( held.z );

	} );
	return { depth, thin };

}
