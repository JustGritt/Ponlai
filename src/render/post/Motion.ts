import * as THREE from 'three/webgpu';
import {
	float, mrt, output, outputStruct as threeOutputStruct, positionLocal, positionPrevious, renderGroup, screenUV,
	select, uniform, vec2, vec4,
} from 'three/tsl';
import { TEMPORAL } from './Antialiasing.ts';
import { onSceneDispose } from '../SceneDispose.ts';
import { jitter } from './Jitter.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { preExposedRGBA, preExposedScalar } from './PreExposure.ts';

// Motion vectors for the temporal modes, written with the scene's colour through an MRT.
//
// Motion vectors for the temporal modes (Antialiasing.ts: aa=taa|taau): where each surface point
// drawn this frame was in the previous rendered frame, as NDC of the unjittered views. They are
// written with the scene's colour into the frame graph's `sceneColor` and `velocity` (Pipeline.ts)
// through an MRT, which finds its outputs and its blend modes by the textures' names, the graph's
// resource names. The velocity is RGBA16F, cleared to ( 0, 0, 1, 0 ):
//   rg  ndcNow - ndcPrev, unjittered (the jitter is no motion)
//   b   the surface's class (MOTION_CLASS: how the resolve treats its history)
//   a   its reactive weight: 1 asks the resolve to trust the history less (a bolt, foam, spray)
//
// In three r186 the blend of each attachment comes only from the renderer's context MRT (the
// pass's `mrt`: WebGPUPipelineUtils reads `context.mrt.getBlendMode( texture.name )`, NoBlending
// unless set), whatever a material's own MRT node says, which changes its shader's outputs only.
// The scene MRT draws the colour with each material's own blending and overwrites the velocity, so
// whatever is drawn last at a pixel, transparent or not, leaves its motion there. The overlay MRT,
// for what the overlay pass draws over the finished scene (under the ground truth the rain and the
// storm's spray, which taa and taau draw after the resolve: AfterResolve.ts), blends
// the colour as the material says and keeps the velocity's rgb (Zero / One), while its alpha, the
// reactive weight, takes the larger of the two (One / One, Max): a streak keeps the motion of the
// surface behind it. tools/render/verify-frame-graph-gpu.mjs checks both blends on the GPU.
//
// Each vertex is moved back to where it was a rendered frame ago and projected there (one varying
// per material); the current NDC needs no varying: under one sample a fragment is shaded at its
// pixel's centre, whose NDC plus the jitter's shift is the unjittered NDC of the surface point it
// shows. The previous position, in the material's local space:
// - three's `positionPrevious` for a material without a positionNode: its geometry, or the
//   skinned or instanced vertex of the previous frame (three's skinning and instancing write it
//   whenever the context MRT has a `velocity`). A skinned mesh in the attached bind mode gets its
//   previous vertex in its CURRENT space, so its object's motion would count twice: its material
//   rebases it (CharacterMaterial.ts, the islanders);
// - a material whose positionNode animates (the wind's sway, the sea) assigns `positionPrevious`
//   in its position stage and says so with `userData.previousPosition = true`;
// - any other positionNode is taken as not moving: its displaced position now.
// The object's own motion comes from its world matrix in the previous frame (per rendered frame, not
// per draw: an object drawn twice in a frame keeps its motion), the camera's from the previous
// view-projection. A material can write a velocity of its own through `velocityOutput` (its MRT
// node, merged over the context's), a fragmentNode material through `outputStruct`.
// A material with a fragmentNode ignores the MRT: under the temporal modes it must write both
// outputs (`outputStruct`), or its pipeline has an output too few; a material whose positionNode
// animates without `positionPrevious` gets the camera's and the object's motion only. A material's
// MRT node replaces its output when the pass has no MRT: set one only under the temporal modes
// (`velocityOutput` returns null otherwise). Drawing a scene material into a second target layout or
// MRT in one page builds it twice, which r186 may emit with the first build's uniform names in a
// layout function (the sea's `sceneViewZ` then fails to compile): time the velocity's cost by
// comparing pages (`--compare` of `aa=off` against a temporal mode), not by switching a page's
// target.
//
// The metering illuminance (`meterLight`, Pipeline.ts) is the scene target's attachment beside the
// colour in every mode: after `velocity` under the temporal modes, second under msaa and off. Every
// pass that draws into the scene target therefore has an MRT (three gives a pass without one a
// single output, which a pipeline of two attachments does not validate), and every fragmentNode
// material drawn there writes it through `outputStruct`. The scene MRT writes 0 (no reading: the
// meter skips it), unblended: the attachment has one channel, so three outputs it as a scalar and
// WebGPU refuses any blend that reads the source's alpha, which every transparent blending of three
// does. A transparent draw in the opaque or the water pass (the lightning's bolt, the spray in msaa's
// water pass) therefore writes 0 over the M under it, and its pixels drop out of the meter. The
// overlay MRT and msaa's rain MRT keep it (src x 0 + dst x 1, no alpha read). A lit material writes
// its own M through `meterOutput` (its MRT node, merged over the pass's); its MRT node would replace
// its output in a pass without an MRT, so such a material is drawn only in passes that have one (the
// scene's and the refraction's, whose MRT drops the names its target lacks).
// Both are pre-exposed here (PreExposure.ts): the scene MRT's and the overlay MRT's colour, an
// `outputStruct`'s colour and M, and a `meterOutput`'s M are given in scene units and stored times
// 2^P, so a material never multiplies by it itself.
//
// A cut (`taaCut`: a seek, a jump of the camera, a new scene) makes the previous frame the current
// one for the camera, the objects and whatever else registers with `onTaaCut` (the wind's previous
// state, the histories), so nothing moves across it. The app's sites ask for it through their scene
// events (SceneEvents.ts, whose table says which cut). The camera's own jumps that no event covers
// are found here: in one rendered frame more than 8 m, more than 30 degrees, or a field of view
// changed by more than 2 %.

/** The classes a surface writes in the velocity's b. */
export const MOTION_CLASS = { opaque: 0, vegetation: 0.25, water: 0.5, character: 0.75, sky: 1 } as const;
export type MotionClass = keyof typeof MOTION_CLASS;

/** The velocity's clear value: no motion, class sky (nothing drawn there), not reactive. */
export const VELOCITY_CLEAR = [ 0, 0, 1, 0 ] as const;

/** The unjittered view-projection of the frame being rendered and of the previous rendered frame,
 * and the camera's position then. */
export const motionUniforms = {
	viewProjection: uniform( new THREE.Matrix4() ).setGroup( renderGroup ),
	previousViewProjection: uniform( new THREE.Matrix4() ).setGroup( renderGroup ),
	previousCameraPosition: uniform( new THREE.Vector3() ).setGroup( renderGroup ),
};

/** Reactive weights that are not per pixel: `flash`, a lightning strike's light over the whole frame
 * (the resolve trusts its history less while it changes; set by Lightning.update). And the sea's
 * glints' reactive weight (Water.ts): smoothstep( glintFrom, glintTo, the glint's luminance over a
 * white diffuse surface's ) times glintMax (TemporalSettings.glintReactive, set by the resolve). */
export const reactiveUniforms = { flash: uniform( 0 ), glintFrom: uniform( 2 ), glintTo: uniform( 8 ), glintMax: uniform( 1 ) };

// ---- the frame's state

/** Rendered frames (the motion's clock), the cuts applied and the last one's reason, and whether the
 * frame being rendered is one. */
export const motionState = { frame: 0, historyResets: 0, lastCut: null as string | null, cut: false };
let cutPending: string | null = 'start';
const cutListeners: ( ( reason: string ) => void )[] = [];
const camera = { position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), fov: 0, valid: false };
// a disposed scene's frames, cuts and listeners go with it: the next one starts with a cut, as a load does
onSceneDispose( 'motion', () => {

	Object.assign( motionState, { frame: 0, historyResets: 0, lastCut: null, cut: false } );
	cutPending = 'start';
	cutListeners.length = 0;
	camera.valid = false;

} );
const _position = new THREE.Vector3(), _quaternion = new THREE.Quaternion(), _scale = new THREE.Vector3(), _viewProjection = new THREE.Matrix4();

/** Asks for a cut: the next frame rendered has no motion from the one before (the camera's, the
 * objects', the wind's: see `onTaaCut`) and the histories start over. */
export function taaCut( reason: string ): void {

	cutPending ??= reason;

}

/** Called with the reason when a cut is applied, at the start of the frame it applies to. */
export function onTaaCut( listener: ( reason: string ) => void ): void {

	cutListeners.push( listener );

}

// the vertical field of view of a projection, in radians
const fovOf = ( p: THREE.Matrix4 ) => 2 * Math.atan( 1 / p.elements[ 5 ] );

/** At the start of a rendered frame, with its camera unjittered: the view-projections, the camera's
 * jumps, and any cut asked for since the last frame. */
export function beginMotionFrame( view: THREE.Camera ): void {

	motionState.frame ++;
	_viewProjection.multiplyMatrices( view.projectionMatrix, view.matrixWorldInverse );
	view.matrixWorld.decompose( _position, _quaternion, _scale );
	const fov = fovOf( view.projectionMatrix );
	if ( camera.valid && cutPending === null ) {

		const moved = _position.distanceTo( camera.position );
		const turned = 2 * Math.acos( Math.min( Math.abs( _quaternion.dot( camera.quaternion ) ), 1 ) );
		const zoomed = Math.abs( fov - camera.fov ) / Math.max( camera.fov, 1e-6 );
		if ( moved > 8 ) taaCut( 'camera moved' );
		else if ( turned > THREE.MathUtils.degToRad( 30 ) ) taaCut( 'camera turned' );
		else if ( zoomed > 0.02 ) taaCut( 'camera zoomed' );

	}
	const u = motionUniforms;
	const cut = cutPending;
	motionState.cut = cut !== null;
	if ( cut !== null ) {

		cutPending = null;
		motionState.historyResets ++;
		motionState.lastCut = cut;

	}
	// the camera's motion (left out under `flags.mv.camera = false`)
	const still = cut !== null || ! camera.valid || debugFlags.mv.camera === false;
	u.previousViewProjection.value.copy( still ? _viewProjection : u.viewProjection.value );
	u.previousCameraPosition.value.copy( still ? _position : camera.position );
	u.viewProjection.value.copy( _viewProjection );
	camera.position.copy( _position );
	camera.quaternion.copy( _quaternion );
	camera.fov = fov;
	camera.valid = true;
	if ( cut !== null ) for ( const listener of cutListeners ) listener( cut );

}

// ---- per object: its world matrix in the frame being rendered and in the previous rendered frame

interface Track { frame: number; current: THREE.Matrix4; previous: THREE.Matrix4 }
const tracks = new WeakMap<THREE.Object3D, Track>();

// The object's world matrix a rendered frame ago, taken at its first draw in a frame: an object not
// drawn in the previous frame (hidden, out of view) or across a cut has none and takes its matrix now.
function previousModelOf( object: THREE.Object3D ): THREE.Matrix4 {

	const frame = motionState.frame;
	let t = tracks.get( object );
	if ( t === undefined ) {

		t = { frame, current: object.matrixWorld.clone(), previous: object.matrixWorld.clone() };
		tracks.set( object, t );
		return t.previous;

	}
	if ( t.frame !== frame ) {

		const continuous = t.frame === frame - 1 && ! motionState.cut && debugFlags.mv.object !== false;
		t.previous.copy( continuous ? t.current : object.matrixWorld );
		t.current.copy( object.matrixWorld );
		t.frame = frame;

	}
	return t.previous;

}

// (an object uniform: set for each draw before its bindings are uploaded)
const previousModel = uniform( new THREE.Matrix4() ).onObjectUpdate( ( { object } ) => object ? previousModelOf( object ) : undefined );

/** A surface's motion vector, class and reactive weight (the velocity's rgba): see the note at the
 * top for the previous position it takes. */
class MotionNode extends THREE.TempNode<'vec4'> {

	static get type(): string {

		return 'PonlaiMotionNode';

	}

	private readonly cls: MotionClass;
	private readonly reactive: THREE.Node<'float'>;

	constructor( cls: MotionClass, reactive: THREE.Node<'float'> ) {

		super( 'vec4' );
		this.cls = cls;
		this.reactive = reactive;

	}

	override setup( builder: THREE.NodeBuilder ): THREE.Node<'vec4'> {

		const material = ( builder as unknown as { material: THREE.Material | null } ).material;
		const own = material?.userData.previousPosition === true;
		const positioned = ( material as THREE.NodeMaterial | null )?.positionNode != null;
		const local = positioned && ! own ? positionLocal : positionPrevious;
		const previous = motionUniforms.previousViewProjection.mul( previousModel.mul( vec4( local, 1 ) ) ).toVarying( 'vMotionPrevious' );
		// the unjittered NDC of the point this pixel's centre shows (screen uv's y runs down)
		const now = vec2( screenUV.x.mul( 2 ).sub( 1 ), float( 1 ).sub( screenUV.y.mul( 2 ) ) ).add( jitter.uniforms.ndc );
		// (a point behind the previous camera has no previous NDC: no motion)
		const motion = select( previous.w.greaterThan( 1e-6 ), now.sub( previous.xy.div( previous.w ) ), vec2( 0 ) );
		return vec4( motion, MOTION_CLASS[ this.cls ], this.reactive );

	}

}

/** The velocity of a surface of class `cls` (camera and object motion, and the material's own
 * through `positionPrevious`), with a reactive weight. */
export function motionNode( cls: MotionClass, reactive: THREE.Node<'float'> | number = 0 ): THREE.Node<'vec4'> {

	return new MotionNode( cls, typeof reactive === 'number' ? float( reactive ) : reactive ) as unknown as THREE.Node<'vec4'>;

}

/** What the overlay pass's materials write into the velocity: only the reactive weight counts (the
 * overlay MRT keeps the rgb under them). */
export function overlayVelocity( reactive: THREE.Node<'float'> | number = 0 ): THREE.Node<'vec4'> {

	return vec4( 0, 0, 0, reactive );

}

/** The metering illuminance's attachment, the frame graph's resource of that name (Pipeline.ts). */
export const METER_LIGHT = 'meterLight';

// M as the attachment's output (one channel: the MRT outputs it as a scalar)
const NO_METER = float( 0 );

/** A fragmentNode material's outputs in the scene target's attachment order: its colour (an RGBA
 * in scene units), under the temporal modes its velocity, and the metering illuminance `meter` (M,
 * scene units; 0, no reading, when not given: see the note at the top); colour and M stored
 * pre-exposed. */
export function outputStruct( color: THREE.Node, velocity: THREE.Node<'vec4'>, meter?: THREE.Node<'float'> ): THREE.Node {

	const m = meter ? preExposedScalar( meter ) : NO_METER;
	const c = preExposedRGBA( color as THREE.Node<'vec4'> );
	return ( TEMPORAL ? threeOutputStruct( c, velocity, m ) : threeOutputStruct( c, m ) ) as unknown as THREE.Node;

}

/** A material's own velocity, as its `mrtNode` (merged over the pass's MRT) under the temporal modes;
 * null otherwise, where the scene passes keep their own. */
export function velocityOutput( velocity: THREE.Node<'vec4'> ): THREE.MRTNode | null {

	return TEMPORAL ? mrt( { velocity } ) : null;

}

/** A lit material's metering illuminance M (scene units of luminance: Pipeline.ts `meterLight`), and
 * under the temporal modes its own velocity if given, as its `mrtNode` (merged over the pass's MRT).
 * Only for materials drawn in passes with an MRT (see the note at the top). */
export function meterOutput( meter: THREE.Node<'float'>, velocity?: THREE.Node<'vec4'> ): THREE.MRTNode {

	const m = preExposedScalar( meter );
	return mrt( TEMPORAL && velocity ? { [ METER_LIGHT ]: m, velocity } : { [ METER_LIGHT ]: m } );

}

// ---- the passes' MRTs

const materialBlending = () => new THREE.BlendMode( THREE.MaterialBlending );

// the overlay's velocity blend: rgb kept, alpha the larger
function keepMaxBlending(): THREE.BlendMode {

	const blend = new THREE.BlendMode( THREE.CustomBlending );
	blend.blendSrc = THREE.ZeroFactor;
	blend.blendDst = THREE.OneFactor;
	blend.blendEquation = THREE.AddEquation;
	blend.blendSrcAlpha = THREE.OneFactor;
	blend.blendDstAlpha = THREE.OneFactor;
	blend.blendEquationAlpha = THREE.MaxEquation;
	return blend;

}

// the metering illuminance kept under a draw: src x 0 + dst x 1 (no factor reads an alpha)
function keepBlending(): THREE.BlendMode {

	const blend = new THREE.BlendMode( THREE.CustomBlending );
	blend.blendSrc = blend.blendSrcAlpha = THREE.ZeroFactor;
	blend.blendDst = blend.blendDstAlpha = THREE.OneFactor;
	blend.blendEquation = blend.blendEquationAlpha = THREE.AddEquation;
	return blend;

}

/** The opaque and water passes' MRT: every material's colour as it blends, under the temporal modes
 * its motion as an ordinary surface's (class opaque), overwriting the velocity, and the metering
 * illuminance, overwritten, no reading (0) unless the material writes one (cleared to 0). */
export const sceneMRT: THREE.MRTNode = mrt( TEMPORAL
	? { sceneColor: preExposedRGBA( output ), velocity: motionNode( 'opaque' ), [ METER_LIGHT ]: NO_METER }
	: { sceneColor: preExposedRGBA( output ), [ METER_LIGHT ]: NO_METER } )
	.setBlendMode( 'sceneColor', materialBlending() )
	.setClearColor( METER_LIGHT, new THREE.Color( 0, 0, 0 ), 0 )
	.setClearColor( 'velocity', new THREE.Color( VELOCITY_CLEAR[ 0 ], VELOCITY_CLEAR[ 1 ], VELOCITY_CLEAR[ 2 ] ), VELOCITY_CLEAR[ 3 ] );

/** The overlay pass's MRT (the rain, the storm's spray): the velocity kept, its reactive weight the
 * larger, and the metering illuminance kept. Under msaa and off the rain pass's (no velocity). */
export const overlayMRT: THREE.MRTNode = mrt( TEMPORAL
	? { sceneColor: preExposedRGBA( output ), velocity: overlayVelocity( 0 ), [ METER_LIGHT ]: NO_METER }
	: { sceneColor: preExposedRGBA( output ), [ METER_LIGHT ]: NO_METER } )
	.setBlendMode( 'sceneColor', materialBlending() )
	.setBlendMode( 'velocity', keepMaxBlending() )
	.setBlendMode( METER_LIGHT, keepBlending() );

// ---- the debug check

/** One material the debug check lists: where it draws, what it lacks. */
export interface MotionIssue { pass: string; object: string; material: string; issue: string }

/** Debug builds: the materials of the scenes the temporal passes draw (`passes`: a pass's name, the
 * scene it draws, and whether its MRT keeps the velocity, the overlay's) that would leave a wrong or
 * missing velocity:
 * - a fragmentNode without a velocity output (`outputStruct`), or a vertexNode without a velocity of
 *   its own (`velocityOutput`, or a fragmentNode's `outputStruct` that writes it, as the water's
 *   does): under an MRT with a velocity its pipeline has an output too few, or its motion is not
 *   where its vertices are;
 * - an animated positionNode (`userData.motion = 'animated'`, or a wind displacement) without its
 *   previous position (`userData.previousPosition`): its own motion is missing, camera and object
 *   motion only;
 * - a positionNode of unknown motion (no `userData.motion`): taken as not moving.
 * The overlay keeps the velocity under its draws, so only their outputs count there. */
export function motionCheck( passes: { pass: string; scene: THREE.Object3D; keeps: boolean }[] ): MotionIssue[] {

	const issues: MotionIssue[] = [];
	for ( const { pass, scene, keeps } of passes ) {

		scene.traverse( ( object ) => {

			const mesh = object as THREE.Mesh;
			if ( ! mesh.isMesh && ! ( object as THREE.Points ).isPoints && ! ( object as THREE.Line ).isLine ) return;
			const list = Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ];
			for ( const m of list as THREE.NodeMaterial[] ) {

				if ( ! m || ! m.isNodeMaterial ) continue;
				const add = ( issue: string ) => issues.push( { pass, object: object.name || object.type, material: m.name || m.type, issue } );
				const own = m.mrtNode?.has( 'velocity' ) === true;
				const fragment = m.fragmentNode as ( THREE.Node & { isOutputStructNode?: boolean; members?: unknown[] } ) | null;
				const writes = fragment?.isOutputStructNode === true && ( fragment.members?.length ?? 0 ) >= 2;
				if ( fragment && ! writes ) add( 'fragmentNode without a velocity output' );
				if ( keeps ) continue;
				if ( m.vertexNode && ! own && ! writes ) add( 'vertexNode without a velocity of its own' );
				if ( own || m.positionNode === null || m.userData.previousPosition === true ) continue;
				const animated = m.userData.motion === 'animated' || m.userData.windDisplacement !== undefined;
				if ( animated ) add( 'animated without a previous position: camera and object motion only' );
				else if ( m.userData.motion !== 'static' ) add( 'positionNode of unknown motion (userData.motion): taken as not moving' );

			}

		} );

	}
	return issues;

}
