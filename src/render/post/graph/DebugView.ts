import * as THREE from 'three/webgpu';
import {
	Fn, abs, atan, clamp, float, floor, fract, int, ivec2, length, log2, max, min, neutralToneMapping, round, screenCoordinate,
	sRGBTransferOETF, select, texture, textureLoad, uniform, vec2, vec3, vec4,
} from 'three/tsl';
import { FrameGraph, type GraphDescription } from './FrameGraph.ts';
import type { ResourceDecl, ResourceHandle } from './GraphCompiler.ts';
import { debugFlags } from '../../debug/DebugFlags.ts';
import { settings, type SettingsQuery } from '../../Settings.ts';

// The frame graph's debug view: one of the graph's resources shown in place of the frame.
//
// The frame graph's debug view (debug and dev builds): `fgview=<resource>[.<output>]` shows one of
// the graph's resources on the canvas in place of the frame. Only when the URL asks for it, it
// appends a pass to the graph before the graph compiles (`FrameGraph.beforeCompile`): the pass
// reads the resource (marked as the debug view, so the graph keeps it live and stored) and draws it
// over the whole canvas after the canvas passes, so what it shows is exactly what the frame's own
// passes left there, the canvas output's input (`display`) and the upscale's (`FSR.easu`)
// included. Its shader is built only then; without the option nothing is declared, compiled or
// drawn, and the frame is the same to the byte. The graph compiles once, so another choice (the
// Debug folder's dropdown, filled from `describe()`) reloads the page with it.
//
// How a resource is shown follows what it holds:
//   hdr       scene radiance (the scene colour, the opaque copy, the fog's in-scatter): the
//             grade's exposure and PBR Neutral's curve, then the sRGB encoding
//   display   display values (the graded image, the upscale's EASU image): as they are
//   linear    the graded image in linear light (`display` under the temporal resolve, half float:
//             Pipeline.ts): the sRGB encoding
//   depth     a depth buffer, its sample 0 when multisampled (`.sample<n>` picks another): turned
//             into view z with the frame's camera, then
//   viewZ     a linear view z or distance (`linearDepth`, `surfaceDepth`, the fog's nearest depth,
//             the blur's distances): grey on a log scale from 0.5 m (black) to 2 km (white; the
//             sky and the far sea too)
//   velocity  a motion vector (NDC delta): its direction as a hue, its length (in render pixels,
//             full at 16) as the brightness
//   single    one channel (a red format, or the `.r` / `.g` / `.b` / `.a` output): grey, clamped
//   history   a persistent or ping-pong colour (`.previous`: the ping-pong's previous texture): as
//             colour, clamped
//   vector    the displacement field (vegetation/Displacement.ts), its `.previous` too: its
//             world direction (x, z) as a hue, its size as the brightness (full at 10, about a
//             full push of the islanders' legs)
// Every texel is loaded, not filtered, at the resource's own size stretched over the canvas. The
// legacy attachment ops (`fgOps=legacy`) discard the scene target's samples in the rain pass
// whatever reads them, so a view of `sceneDepth` while it rains needs the derived ops.
//
// The temporal modes' graph (a `velocity` resource) has views of its motion vectors besides
// `velocity` itself:
//   class           the velocity's b, a colour per class (Motion.ts MOTION_CLASS): opaque grey,
//                   vegetation green, water blue, character orange, sky pale blue
//   reactive        the velocity's a, grey
//   reproject       the previous frame's `display` (in linear light under the resolve) moved along
//                   this frame's motion vectors
//                   (Catmull-Rom, no clip, no blend): over a moving camera it should match the frame
//                   (the motion vectors' unit checks, MV-UNIT); the view keeps a copy of `display`
//                   (`displayPrev`, made at the frame's end) for it
//   reproject.diff  for measurements: r the largest channel's difference between `display` and its
//                   reprojection, g 1 on static opaque geometry (class opaque, not the sky), b 1
//                   where the point is disoccluded or was off screen (the previous frame's nearest
//                   surface there lay more than 2 % nearer or farther than the point did, the point
//                   moved by the camera alone: for static geometry)
// and, with the resolve (TemporalAA.ts), views of its decisions, each the resolve's own shader with
// one of its values written instead of the colour (into `taaDebug`, at the history's size):
//   weight        the history's share of the resolved pixel (black: this frame alone, white: all
//                 history), grey
//   disocclusion  red where the previous frame's nearest surface shows the point was hidden, green
//                 where it was off screen, blue where the history was reset (a cut, a resize)
//   clip          how far the history lay outside the neighbourhood's clip box (0 inside, white at
//                 twice its width), grey

export type DebugViewKind = 'hdr' | 'display' | 'linear' | 'depth' | 'viewZ' | 'velocity' | 'vector' | 'single' | 'history' | 'colour';

export interface DebugViewRequest { resource: string; output: string | null }

/** What the resolve's debug views ask of it (TemporalAA.ts `declareDebugView`). */
export interface TemporalDebugSource {
	declareDebugView( view: 'weight' | 'disocclusion' | 'clip', inputs: { display: ResourceHandle; velocity: ResourceHandle; surfaceDepth: ResourceHandle; surfaceDepthPrev: ResourceHandle; veilReactive: ResourceHandle | null } ): ResourceHandle;
}

export interface DebugViewOptions {
	// the grade's exposure on scene radiance (the HDR view's)
	exposure(): number;
	// the temporal resolve, for its views (null without one); asked while the graph compiles
	temporal?: () => TemporalDebugSource | null;
	// the temporal modes' motion vectors (Motion.ts, Jitter.ts), for their views: the previous
	// rendered frame's unjittered view-projection, the unjittered inverse projection, the jitter's
	// NDC shift, and the velocity's b below which a surface is opaque (no class of its own)
	motion?: {
		previousViewProjection: THREE.UniformNode<'mat4', THREE.Matrix4>;
		projectionInverse: THREE.UniformNode<'mat4', THREE.Matrix4>;
		jitterNdc: THREE.UniformNode<'vec2', THREE.Vector2>;
		opaqueBelow: number;
	};
}

// resources that hold display values rather than scene radiance
const DISPLAY_REFERRED = new Set( [ 'display', 'FSR.easu' ] );
const CHANNELS = [ 'r', 'g', 'b', 'a' ];
// the log scale of the depth views, in metres (the island and its sea span about 1 m to 1 km)
const NEAR = 0.5, FAR = 2000;
// velocity at full brightness, in render pixels
const VELOCITY_FULL = 16;
// the displacement field's size at full brightness
const VECTOR_FULL = 10;

/** `fgview` from the app's options or `params` (null: none). */
export function debugViewFromURL( params: SettingsQuery = settings ): string | null {

	const value = params.get( 'fgview' );
	return value && value !== 'off' ? value : null;

}

/** Splits `fgview`'s value into a resource of `names` and an output: the whole value when it names
 * a resource (names may hold dots, `FSR.easu`), else the part before the last dot. */
export function parseDebugView( value: string, names: readonly string[] ): DebugViewRequest | null {

	if ( names.includes( value ) ) return { resource: value, output: null };
	const dot = value.lastIndexOf( '.' );
	if ( dot > 0 && names.includes( value.slice( 0, dot ) ) ) return { resource: value.slice( 0, dot ), output: value.slice( dot + 1 ) };
	return null;

}

function kindOf( r: ResourceDecl, output: string | null ): DebugViewKind {

	const viewZ = /depth|distance/i.test( r.name ) && r.kind !== 'depth';
	if ( output && CHANNELS.includes( output ) ) return viewZ ? 'viewZ' : 'single';
	if ( r.kind === 'depth' ) return 'depth';
	if ( /velocity/i.test( r.name ) ) return 'velocity';
	if ( r.name === 'displacement' ) return 'vector';
	if ( DISPLAY_REFERRED.has( r.name ) ) return r.name === 'display' && r.desc.type === THREE.HalfFloatType ? 'linear' : 'display';
	if ( viewZ ) return 'viewZ';
	if ( r.desc.format === THREE.RedFormat ) return 'single';
	if ( r.persistent || r.pingPong ) return 'history';
	if ( r.desc.type === THREE.HalfFloatType || r.desc.type === THREE.FloatType ) return 'hdr';
	return 'colour';

}

// the temporal modes' views of the motion vectors, and of the resolve (see the note at the top)
const MOTION_VIEWS = [ 'class', 'reactive', 'reproject', 'reproject.diff' ];
const RESOLVE_VIEWS = [ 'weight', 'disocclusion', 'clip' ] as const;

/** The choices the Debug folder offers for a graph: every resource but the canvas, plus the
 * channels worth a look of their own (a two-channel resource's second channel, an HDR colour's
 * alpha), and the motion vectors' views where the graph has them. */
export function debugViewChoices( d: GraphDescription ): string[] {

	const choices: string[] = d.resources.some( ( r ) => r.name === 'velocity' ) ? [ ...MOTION_VIEWS ] : [];
	if ( d.resources.some( ( r ) => r.name === 'taaHistory' ) ) choices.push( ...RESOLVE_VIEWS );
	for ( const r of d.resources ) {

		if ( r.canvas ) continue;
		choices.push( r.name );
		if ( r.kind !== 'color' || DISPLAY_REFERRED.has( r.name ) ) continue;
		if ( r.format === 'RG' ) choices.push( `${ r.name }.g` );
		else if ( r.format === 'RGBA' && ( r.type === 'HalfFloat' || r.type === 'Float' ) ) choices.push( `${ r.name }.a` );

	}
	return choices;

}

/** Appends the debug view of `value` (`<resource>[.<output>]`) to the next graph compiled. */
export function installDebugView( value: string, options: DebugViewOptions ): void {

	const extend = ( graph: FrameGraph<unknown> ) => {

		FrameGraph.beforeCompile = FrameGraph.beforeCompile.filter( ( e ) => e !== extend );
		try {

			declareDebugView( graph, value, options );

		} catch ( error ) {

			console.warn( `fgview=${ value }: ${ ( error as Error ).message }` );

		}

	};
	FrameGraph.beforeCompile.push( extend );

}

function declareDebugView( graph: FrameGraph<unknown>, value: string, options: DebugViewOptions ): void {

	const decl = graph.decl;
	if ( MOTION_VIEWS.includes( value ) ) {

		if ( ! options.motion ) throw new Error( 'no motion vectors were given to the debug view' );
		declareMotionView( graph, value, options.motion );
		return;

	}
	if ( ( RESOLVE_VIEWS as readonly string[] ).includes( value ) ) {

		declareResolveView( graph, value as typeof RESOLVE_VIEWS[ number ], options );
		return;

	}
	const names = decl.resources.filter( ( r ) => ! r.canvas ).map( ( r ) => r.name );
	const request = parseDebugView( value, names );
	if ( ! request ) throw new Error( `no such resource (the graph has ${ names.join( ', ' ) })` );
	const r = decl.resources.find( ( x ) => x.name === request.resource )!;
	const canvas = decl.resources.find( ( x ) => x.canvas );
	if ( ! canvas ) throw new Error( 'the graph draws no canvas' );
	const output = request.output;
	const sampleIndex = output && /^sample\d+$/.test( output ) ? Number( output.slice( 6 ) ) : 0;
	if ( output && ! CHANNELS.includes( output ) && output !== 'previous' && ! ( r.kind === 'depth' && /^sample\d+$/.test( output ) ) ) {

		throw new Error( `unknown output '${ output }' (r, g, b, a, previous, or sample<n> of a depth)` );

	}
	if ( output === 'previous' && ! r.pingPong ) throw new Error( `'${ r.name }' keeps no previous texture` );
	const handle: ResourceHandle = output === 'previous'
		? { resource: r.id, version: - 1, name: r.name, previous: true }
		: { resource: r.id, version: r.latest, name: r.name };
	const kind = kindOf( r, output );

	// the texel under each canvas pixel, at the resource's own size
	const size = graph.extent( r.desc.size );
	const canvasSize = graph.extent( { of: 'canvas' } );
	const node = texture( graph.texture( handle ) );
	const coord = ivec2( clamp( floor( screenCoordinate.xy.mul( size ).div( canvasSize ) ), vec2( 0 ), size.sub( 1 ) ) );
	const exposure = uniform( 1 );
	const projInv = uniform( new THREE.Matrix4() );
	const toViewZ = ( d: THREE.Node<'float'> ) => {

		const v = projInv.mul( vec4( 0, 0, d, 1 ) );
		return select( d.lessThan( 0.999999 ), v.z.div( v.w ), float( - 1e5 ) );

	};
	const logDepth = ( z: THREE.Node<'float'> ) => clamp( log2( max( abs( z ), NEAR ).div( NEAR ) ).div( Math.log2( FAR / NEAR ) ), 0, 1 );
	const channel = output && CHANNELS.includes( output ) ? output : 'r';

	// hue from a direction (an HSV wheel at full saturation)
	const wheel = ( v: THREE.Node<'vec2'> ) => {

		const h = fract( atan( v.y, v.x ).div( 2 * Math.PI ) );
		return clamp( abs( fract( vec3( h ).add( vec3( 1, 2 / 3, 1 / 3 ) ) ).mul( 6 ).sub( 3 ) ).sub( 1 ), 0, 1 );

	};
	const shown = Fn( () => {

		if ( kind === 'depth' ) {

			const d = ( textureLoad( node, coord, int( sampleIndex ) ) as unknown as THREE.Node<'vec4'> ).x;
			return vec3( logDepth( toViewZ( d ) ) );

		}
		const t = textureLoad( node, coord ) as unknown as THREE.Node<'vec4'>;
		const c = ( t as unknown as Record<string, THREE.Node<'float'>> )[ channel ];
		switch ( kind ) {

			case 'viewZ': return vec3( logDepth( c ) );
			case 'single': return vec3( clamp( c, 0, 1 ) );
			case 'display': return t.rgb;
			case 'linear': return sRGBTransferOETF( clamp( t.rgb, 0, 1 ) );
			case 'hdr': return sRGBTransferOETF( clamp( neutralToneMapping( t.rgb, exposure ) as THREE.Node<'vec3'>, 0, 1 ) );
			case 'velocity': {

				// hue from the direction, brightness from the length
				const px = t.xy.mul( size ).mul( 0.5 );
				return wheel( px ).mul( min( length( px ).div( VELOCITY_FULL ), 1 ) );

			}
			case 'vector': return wheel( t.xy ).mul( min( length( t.xy ).div( VECTOR_FULL ), 1 ) );
			default: return clamp( t.rgb, 0, 1 );

		}

	} );
	const pipeline = new THREE.RenderPipeline( graph.renderer, vec4( shown() as THREE.Node<'vec3'>, 1 ) );
	pipeline.outputColorTransform = false;

	decl.setDebugView( handle );
	graph.addPass( 'debugView', 'chain', ( b ) => {

		b.readOptional( handle, kind === 'depth' ? 'samples' : 'sample' );
		b.write( { resource: canvas.id, version: canvas.latest, name: canvas.name } );
		return {
			label: 'debug view',
			pipeline,
			run: ( f, g ) => {

				exposure.value = options.exposure();
				const camera = ( f as { camera?: THREE.Camera } | undefined )?.camera;
				if ( camera ) projInv.value.copy( camera.projectionMatrixInverse );
				// a ping-pong resource's current (or previous) texture moves every frame
				if ( r.pingPong ) node.value = g.texture( handle );
				pipeline.render();

			},
		};

	} );

}

// The temporal modes' views of the motion vectors: `class`, `reactive`, `reproject` and
// `reproject.diff` (see the note at the top).
function declareMotionView( graph: FrameGraph<unknown>, value: string, motion: NonNullable<DebugViewOptions[ 'motion' ]> ): void {

	const decl = graph.decl;
	const find = ( name: string ) => decl.resources.find( ( r ) => r.name === name );
	const velocity = find( 'velocity' ), display = find( 'display' ), canvas = find( 'canvas' );
	if ( ! velocity || ! display || ! canvas ) throw new Error( 'the graph has no motion vectors (aa=taa or taau)' );
	const latest = ( r: ResourceDecl ): ResourceHandle => ( { resource: r.id, version: r.latest, name: r.name } );
	const vHandle = latest( velocity );
	const size = graph.extent( velocity.desc.size );
	const canvasSize = graph.extent( { of: 'canvas' } );
	const coord = ivec2( clamp( floor( screenCoordinate.xy.mul( size ).div( canvasSize ) ), vec2( 0 ), size.sub( 1 ) ) );
	const tVelocity = texture( graph.texture( vHandle ) );
	const reproject = value.startsWith( 'reproject' );
	// the previous frame's display, kept by a copy at each frame's end
	let displayPrev: ResourceHandle | null = null, surface: ResourceDecl | undefined, surfacePrev: ResourceDecl | undefined;
	if ( reproject ) {

		surface = find( 'surfaceDepth' );
		surfacePrev = find( 'surfaceDepthPrev' );
		if ( ! surface || ! surfacePrev ) throw new Error( 'the graph keeps no previous surface depth' );
		displayPrev = graph.color( 'displayPrev', { ...display.desc }, true );
		graph.copyToPersistent( latest( display ), displayPrev );

	}
	const cameraWorld = uniform( new THREE.Matrix4() );
	// (`flags.reprojectScale`: the motion scaled, to show the residual is least at its own length)
	const reprojectScale = uniform( 1 );
	type V3 = THREE.Node<'vec3'>;
	const shown = Fn( (): V3 => {

		const v = textureLoad( tVelocity, coord ) as unknown as THREE.Node<'vec4'>;
		if ( value === 'class' ) {

			const k = round( clamp( v.b, 0, 1 ).mul( 4 ) );
			const colours: V3[] = [ vec3( 0.5 ), vec3( 0.2, 0.75, 0.2 ), vec3( 0.15, 0.35, 0.9 ), vec3( 0.95, 0.55, 0.15 ), vec3( 0.65, 0.8, 1 ) ];
			let c: V3 = colours[ 4 ];
			for ( let i = 3; i >= 0; i -- ) c = select( k.lessThan( i + 0.5 ), colours[ i ], c );
			return c;

		}
		if ( value === 'reactive' ) return vec3( clamp( v.a, 0, 1 ) );
		// the previous display where this pixel's point was, by a Catmull-Rom of 4 x 4 loads (a float
		// texture may not be filterable), as the resolve fetches its history: bilinear blurs a
		// pixel-wide blade by a quarter of its contrast at half a pixel, which no motion vector undoes
		const tPrev = texture( graph.texture( displayPrev! ) );
		const q = vec2( coord ).add( 0.5 ).toVar();
		const qPrev = q.sub( v.xy.mul( vec2( 0.5, - 0.5 ) ).mul( size ).mul( reprojectScale ) ).sub( 0.5 ).toVar();
		const base = floor( qPrev ).toVar();
		const f = qPrev.sub( base ).toVar();
		const weights = ( t: THREE.Node<'float'> ) => {

			const t2 = t.mul( t ), t3 = t2.mul( t );
			return [ t3.mul( - 0.5 ).add( t2 ).sub( t.mul( 0.5 ) ), t3.mul( 1.5 ).sub( t2.mul( 2.5 ) ).add( 1 ), t3.mul( - 1.5 ).add( t2.mul( 2 ) ).add( t.mul( 0.5 ) ), t3.mul( 0.5 ).sub( t2.mul( 0.5 ) ) ];

		};
		const wx = weights( f.x ), wy = weights( f.y );
		const previous = vec3( 0 ).toVar();
		for ( let j = 0; j < 4; j ++ ) for ( let i = 0; i < 4; i ++ ) {

			const texel = textureLoad( tPrev, ivec2( clamp( base.add( vec2( i - 1, j - 1 ) ), vec2( 0 ), size.sub( 1 ) ) ) ) as unknown as THREE.Node<'vec4'>;
			previous.addAssign( texel.rgb.mul( wx[ i ].mul( wy[ j ] ) ) );

		}
		if ( value === 'reproject' ) return previous as unknown as V3;
		const now = ( textureLoad( texture( graph.texture( latest( display ) ) ), coord ) as unknown as THREE.Node<'vec4'> ).rgb;
		const d = abs( now.sub( previous ) );
		const diff = max( d.r, max( d.g, d.b ) );
		// static opaque geometry: class opaque, not the sky
		const z = ( textureLoad( texture( graph.texture( latest( surface! ) ) ), coord ) as unknown as THREE.Node<'vec4'> ).r.toVar();
		const opaque = v.b.lessThan( motion.opaqueBelow ).and( z.greaterThan( - 9e4 ) );
		// where the point was, moved by the camera alone, and the previous nearest surface there
		const ndc = vec2( q.x.div( size.x ).mul( 2 ).sub( 1 ), float( 1 ).sub( q.y.div( size.y ).mul( 2 ) ) ).add( motion.jitterNdc );
		const ray = motion.projectionInverse.mul( vec4( ndc, 1, 1 ) ).toVar();
		const dir = ray.xyz.div( ray.w ).toVar();
		const world = cameraWorld.mul( vec4( dir.mul( z.div( dir.z ) ), 1 ) );
		const clip = motion.previousViewProjection.mul( world ).toVar();
		const expected = clip.w.negate();
		const uvPrev = qPrev.add( 0.5 ).div( size );
		const off = uvPrev.x.lessThan( 0 ).or( uvPrev.y.lessThan( 0 ) ).or( uvPrev.x.greaterThan( 1 ) ).or( uvPrev.y.greaterThan( 1 ) );
		const was = ( textureLoad( texture( graph.texture( { resource: surfacePrev!.id, version: 0, name: surfacePrev!.name } ) ), ivec2( clamp( floor( uvPrev.mul( size ) ), vec2( 0 ), size.sub( 1 ) ) ) ) as unknown as THREE.Node<'vec4'> ).r;
		const disoccluded = off.or( abs( was.sub( expected ) ).greaterThan( abs( expected ).mul( 0.02 ) ) );
		return vec3( diff, select( opaque, float( 1 ), float( 0 ) ), select( disoccluded, float( 1 ), float( 0 ) ) );

	} );
	const pipeline = new THREE.RenderPipeline( graph.renderer, vec4( shown() as THREE.Node<'vec3'>, 1 ) );
	pipeline.outputColorTransform = false;
	decl.setDebugView( vHandle );
	graph.addPass( 'debugView', 'chain', ( b ) => {

		b.readOptional( vHandle );
		if ( reproject ) {

			b.read( latest( display ) );
			b.read( displayPrev! );
			b.read( latest( surface! ) );
			b.read( { resource: surfacePrev!.id, version: 0, name: surfacePrev!.name } );

		}
		b.write( latest( canvas ) );
		return {
			label: 'debug view',
			pipeline,
			run: ( f ) => {

				const camera = ( f as { camera?: THREE.Camera } | undefined )?.camera;
				if ( camera ) cameraWorld.value.copy( camera.matrixWorld );
				reprojectScale.value = typeof debugFlags.reprojectScale === 'number' ? debugFlags.reprojectScale : 1;
				pipeline.render();

			},
		};

	} );

}

// The resolve's views (`weight`, `disocclusion`, `clip`; see the note at the top): the resolve declares
// a pass of its own shader writing that value into `taaDebug`, shown here stretched over the canvas.
function declareResolveView( graph: FrameGraph<unknown>, value: typeof RESOLVE_VIEWS[ number ], options: DebugViewOptions ): void {

	const decl = graph.decl;
	const resolve = options.temporal?.() ?? null;
	if ( ! resolve ) throw new Error( 'the graph has no temporal resolve (aa=taa, taau or gt)' );
	const find = ( name: string ) => {

		const r = decl.resources.find( ( x ) => x.name === name );
		if ( ! r ) throw new Error( `the graph has no '${ name }'` );
		return { resource: r.id, version: r.latest, name: r.name } as ResourceHandle;

	};
	const canvas = decl.resources.find( ( x ) => x.canvas );
	if ( ! canvas ) throw new Error( 'the graph draws no canvas' );
	const target = resolve.declareDebugView( value, {
		display: find( 'display' ), velocity: find( 'velocity' ), surfaceDepth: find( 'surfaceDepth' ),
		// (no veil's reactive weight when the veil is drawn after the resolve)
		surfaceDepthPrev: find( 'surfaceDepthPrev' ), veilReactive: decl.resources.some( ( x ) => x.name === 'veilReactive' ) ? find( 'veilReactive' ) : null,
	} );
	const r = decl.get( target );
	const size = graph.extent( r.desc.size );
	const canvasSize = graph.extent( { of: 'canvas' } );
	const coord = ivec2( clamp( floor( screenCoordinate.xy.mul( size ).div( canvasSize ) ), vec2( 0 ), size.sub( 1 ) ) );
	const shown = ( textureLoad( texture( graph.texture( target ) ), coord ) as unknown as THREE.Node<'vec4'> ).rgb;
	const pipeline = new THREE.RenderPipeline( graph.renderer, vec4( clamp( shown, 0, 1 ), 1 ) );
	pipeline.outputColorTransform = false;
	decl.setDebugView( target );
	graph.addPass( 'debugView', 'chain', ( b ) => {

		b.read( target );
		b.write( { resource: canvas.id, version: canvas.latest, name: canvas.name } );
		return { label: 'debug view', pipeline };

	} );

}
