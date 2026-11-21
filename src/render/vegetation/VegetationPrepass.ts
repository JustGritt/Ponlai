import * as THREE from 'three/webgpu';
import { materialColor, vec4 } from 'three/tsl';
import { debugFlags } from '../debug/DebugFlags.ts';
import { ALPHA_TO_COVERAGE, TEMPORAL } from '../post/Antialiasing.ts';
import { VELOCITY_CLEAR, velocityOutput } from '../post/Motion.ts';
import { MEADOW_SHADOWS } from '../post/MeadowShadows.ts';
import { followPlantDraw } from './PlantInstances.ts';
import { settings } from '../Settings.ts';
import { onSceneDispose } from '../SceneDispose.ts';

// A depth prepass for the alpha-to-coverage vegetation (today the meadow's blades).
//
// Profiling at 4x MSAA showed that nearly all of the vegetation's cost is its fragment shading:
// the same draws with a flat shader were 4.4 ms (render scale 1) to 5.9 ms (1.5) cheaper over the
// island, almost as much as hiding them. Coverage written from the shader keeps the GPU from
// rejecting hidden fragments before they are shaded, so every blade and leaf behind the front one
// is shaded too, and a coverage pipeline shades more finely than a plain one besides.
//
// So every coverage mesh is drawn twice:
// - a depth-only copy first (`-depth`, drawn two steps earlier): the same vertex stage, alpha and
//   coverage, with a shader that computes nothing else and writes no colour;
// - then the shaded mesh, testing depth for equality without writing it, and without coverage or
//   an alpha test of its own: equal depth already limits it to the samples the copy covered, the
//   copy's coverage. It is an ordinary opaque draw, so hidden fragments are rejected before they
//   are shaded.
// The samples each blade and leaf covers stay exactly as before; only the shading of the visible
// ones changes a little (same brightness, slightly crisper detail inside the canopy), since a
// coverage pipeline shaded them more finely.
//
// Under the temporal modes (Antialiasing.ts: aa=taa|taau) the scene has one sample and no
// coverage: the vegetation is alpha-tested instead (leaves by a coverage band about their cutoff,
// grass blades by their coverage, both against the per-frame dither: `coverageAlphaTest` and
// `coverageDither` in vegetation/Coverage.ts), and the same pair is drawn. The depth copy then discards what
// the alpha test discards, and the shaded draw again keeps neither coverage nor an alpha test:
// equal depth limits it to the pixels the copy kept, and an opaque draw rejects the hidden blades
// and leaves before they are shaded (an alpha-tested one cannot: its discard is decided by the
// shader). Without multisampling and outside the temporal modes (aa=off) nothing changes.
//
// debugFlags (A/B in one page with tools/render/profile-frame.mjs --toggle):
// - vegPrepass = false: the old single coverage pass.
// - vegPrepassOpaque = false: the shaded draw keeps its coverage and alpha test after the
//   prepass (under the temporal modes its alpha test alone). The image is then the old one, but
//   it saves far less.
// URL options, read at load: vegPrepass=0, vegPrepass=coverage (vegPrepassOpaque = false).
//
// With the meadow's screen-space shadows (post/MeadowShadows.ts: at 4x outside the temporal modes) the depth copies
// are not their meshes' children but live in a scene of their own (`prepassScene`), drawn in a pass
// of its own before the opaque scene, into the same depth, which the shadows' march reads and the
// opaque pass loads: the equal test holds as before. Before each of its renders the scene takes its
// copies' transforms and visibility from their meshes, ancestors included, as children would.
//
// Rules for anything that touches it:
// - At 4x the vegetation's coverage is decided by the prepass alone: the shaded draw is plain
//   opaque geometry limited by equal depth to the prepass's samples.
// - A depth copy follows its mesh's per-draw choices (blade count, `grassLod`, the culled list and
//   its bands, which are the shared geometry's indirect offsets), so the equal test holds for
//   every batch; a per-object uniform (`onObjectUpdate`) reads the drawn object's own userData, so
//   each copy carries the fields its shader reads.
// - Keep `material.alphaTest` itself at its value (the shadow passes copy it) and leave it out of
//   the shaded draw's shader build only.
// - A new alpha-to-coverage vegetation kind registers with `prepass: true` (vegetation/Registry.ts:
//   main.ts hands the registered roots to `installVegetationPrepass`). The pairs follow the meshes
//   under the roots, walked before each render (a grown chunk's bamboo: Bamboo.ts), a new mesh on a
//   patched material paired with the depth copy's material its neighbours drew with, the copy of a
//   mesh that left dropped and disposed a few walks later. Anything that finds
//   vegetation after the first frame reads `userData.vegetation` (this turns `alphaToCoverage` off).
// - A depth copy's material is made once per shaded material (the grass's three batches share one),
//   so its node graph and pipelines are built once.

// Equal depth needs the prepass and the shaded draw to place every vertex bit for bit alike. The
// two programs differ (the shaded one computes much more), and the shader compiler then reorders
// or contracts the position's arithmetic differently: a few samples fail the test and show the
// clear colour behind. WGSL's `@invariant` on the clip position forbids that. three does not emit
// it, so both materials patch the declarations of the builder that compiles them (the opaque
// prepass's pairs too: OpaquePrepass.ts).
export function invariantPosition( material: THREE.NodeMaterial ): void {

	const setup = material.setup;
	material.setup = function ( this: THREE.NodeMaterial, builder: THREE.NodeBuilder ) {

		const b = builder as unknown as { getBuiltins?: ( stage: string ) => string };
		const builtins = b.getBuiltins;
		if ( builtins && ! Object.prototype.hasOwnProperty.call( b, 'getBuiltins' ) ) {

			b.getBuiltins = function ( stage: string ) {

				return builtins.call( this, stage ).replace( '@builtin( position )', '@invariant @builtin( position )' );

			};

		}
		return setup.call( this, builder );

	};

}

// The depth-only copy of a coverage material: its vertex stage, alpha (the leaf map's and the
// alpha map's, or the blades' coverage) and alpha test, nothing else. Under the temporal modes it
// takes the alpha test without coverage (one sample: three would draw a coverage material's alpha
// above zero solid; the test is the material's `alphaTestNode`, the leaves' coverage band or the
// blades' dither, vegetation/Coverage.ts), and a constant velocity of its own: `colorWrite = false` masks every
// attachment, so the prepass writes no motion, and the copy need not compute the motion the
// scene's MRT would ask of it (nor carry the varying).
function depthMaterial( source: THREE.MeshStandardNodeMaterial ): THREE.MeshBasicNodeMaterial {

	const m = new THREE.MeshBasicNodeMaterial( { side: source.side, alphaToCoverage: ! TEMPORAL } );
	m.positionNode = source.positionNode;
	m.opacityNode = source.opacityNode;
	m.alphaTestNode = source.alphaTestNode;
	m.alphaTest = source.alphaTest;
	m.alphaMap = source.alphaMap;
	if ( source.map ) {

		m.map = source.map;
		m.colorNode = vec4( 1, 1, 1, ( materialColor as unknown as THREE.Node<'vec4'> ).w );

	} else m.colorNode = vec4( 1 );
	m.colorWrite = false;
	if ( TEMPORAL ) m.mrtNode = velocityOutput( vec4( ...VELOCITY_CLEAR ) );
	m.name = ( source.name || 'vegetation' ) + '-depth';
	invariantPosition( m );
	return m;

}

/** The depth copies' scene with the meadow's screen-space shadows (post/MeadowShadows.ts): drawn in a pass of its
 * own before the opaque scene, into the same depth (Pipeline.ts). */
export const prepassScene = new THREE.Scene();
prepassScene.name = 'vegetation prepass';
// a disposed scene's copies go with it (its renderer freed what they held)
onSceneDispose( 'vegetation prepass', () => {

	prepassScene.clear();
	prepassScene.onBeforeRender = THREE.Object3D.prototype.onBeforeRender;

} );

// the shaded mesh, its material and its depth copy
type Pair = { mesh: THREE.Mesh; material: THREE.MeshStandardNodeMaterial; depth: THREE.Mesh };

/** What `installVegetationPrepass` installed (kept on `scene.userData.vegetationPrepass` for the
 * diagnostics): its pairs of a depth copy and a shaded mesh, the shaded materials patched and their
 * depth copies' materials. */
export interface VegetationPrepassInfo { pairs: number; materials: number; depthMaterials: number }

/** Give every vegetation mesh under `roots` (`markVegetation`) its depth prepass: at 4x, where its
 * cut-outs take alpha to coverage, and under the temporal modes, where they are alpha-tested. With
 * one sample outside the temporal modes there is no coverage material, and nothing changes. */
export function installVegetationPrepass( scene: THREE.Scene, roots: THREE.Object3D[] ): VegetationPrepassInfo {

	const option = settings.get( 'vegPrepass' );
	if ( option === '0' ) debugFlags.vegPrepass = false;
	if ( option === 'coverage' ) debugFlags.vegPrepassOpaque = false;

	const pairs: Pair[] = [];
	const patched = new Set<THREE.Material>();
	// one depth copy per shaded material, whatever number of meshes draw with it (the grass's
	// batches share theirs): the same shader, so the same pipelines
	const depthCopies = new Map<THREE.Material, THREE.MeshBasicNodeMaterial>();
	// whether a mesh takes a pair: a vegetation mesh (`markVegetation`) when its cut-outs take
	// coverage (4x) or an alpha test that the prepass can take over (the temporal modes). Found by
	// the tag, as `alphaToCoverage` is off under the temporal modes and once the prepass applies.
	const eligible = ( object: THREE.Object3D ): object is THREE.Mesh => {

		const mesh = object as THREE.Mesh;
		// (a mesh off the view's layer, a caster of the sun's map alone, draws no depth there)
		if ( ! mesh.isMesh || mesh.userData.prepassCopy || Array.isArray( mesh.material ) || ! mesh.layers.isEnabled( 0 ) ) return false;
		const material = mesh.material as THREE.Material;
		return ( TEMPORAL || ALPHA_TO_COVERAGE ) && material.userData.vegetation === true;

	};
	const pair = ( mesh: THREE.Mesh ) => {

		const material = mesh.material as THREE.MeshStandardNodeMaterial;
		let depthCopy = depthCopies.get( material );
		if ( ! depthCopy ) depthCopies.set( material, depthCopy = depthMaterial( material ) );
		let depth: THREE.Mesh;
		if ( ( mesh as THREE.InstancedMesh ).isInstancedMesh ) {

			const source = mesh as THREE.InstancedMesh;
			const copy = new THREE.InstancedMesh( source.geometry, depthCopy, source.instanceMatrix.count );
			copy.instanceMatrix = source.instanceMatrix;
			copy.count = source.count;
			depth = copy;

		} else depth = new THREE.Mesh( mesh.geometry, depthCopy );
		depth.name = mesh.name + '-depth';
		// (the rain's collision bake, RainCollision.ts, leaves the copies out: they print the same shape)
		depth.userData.prepassCopy = true;
		depth.frustumCulled = mesh.frustumCulled;
		depth.castShadow = false;
		depth.receiveShadow = false;
		depth.renderOrder = mesh.renderOrder - 2;
		depth.visible = false;
		// The grass and the plants pick their instances as they are drawn (Grass.ts,
		// vegetation/PlantInstances.ts); the copy draws first, so it makes the same choice for the
		// same camera. It reads the culled list only
		// when the mesh does (`grassCulled`): otherwise it would draw the first tufts in slot order
		// and the shaded draw's equal test would fail over most of the meadow.
		const before = mesh.onBeforeRender;
		depth.onBeforeRender = ( renderer, s, camera, geometry, mat, group ) => {

			before.call( mesh, renderer, s, camera, geometry, mat, group );
			if ( ( depth as THREE.InstancedMesh ).isInstancedMesh ) ( depth as THREE.InstancedMesh ).count = ( mesh as THREE.InstancedMesh ).count;
			depth.userData.grassLod = mesh.userData.grassLod;
			depth.userData.grassCulled = mesh.userData.grassCulled;
			// a plant's draw (vegetation/PlantInstances.ts): its draw and instance count
			followPlantDraw( depth, mesh );

		};
		// a child: it follows the mesh's transform and visibility; with the meadow's screen-space shadows, in the
		// scene of its own pass, following them there (`prepassScene`)
		if ( MEADOW_SHADOWS ) {

			depth.matrixAutoUpdate = false;
			depth.matrixWorldAutoUpdate = false;
			prepassScene.add( depth );

		} else mesh.add( depth );
		if ( ! patched.has( material ) ) {

			patched.add( material );
			invariantPosition( material );
			// The opaque shaded draw leaves the alpha test out of its shader only: the shadow
			// passes copy `alphaTest` from the material as they draw, and keep the leaf shapes. The
			// grass's dither (its `alphaTestNode`, single-sampled) goes the same way.
			const setup = material.setupDiffuseColor;
			( material as unknown as { setupDiffuseColor: ( b: THREE.NodeBuilder ) => void } ).setupDiffuseColor = function ( this: THREE.MeshStandardNodeMaterial, builder: THREE.NodeBuilder ) {

				const test = this.alphaTest, testNode = this.alphaTestNode;
				if ( this.userData.prepassOpaque ) {

					this.alphaTest = 0;
					this.alphaTestNode = null;

				}
				setup.call( this, builder );
				this.alphaTest = test;
				this.alphaTestNode = testNode;

			};
			// Under the temporal modes coverage stays off either way, and three's cache key leaves
			// out userData: the shader with the alpha test and the one without need keys of their own.
			if ( TEMPORAL ) {

				const key = material.customProgramCacheKey;
				material.customProgramCacheKey = function ( this: THREE.MeshStandardNodeMaterial ) {

					return key.call( this ) + ( this.userData.prepassOpaque ? '|prepassOpaque' : '' );

				};

			}

		}
		pairs.push( { mesh, material, depth } );

	};
	// The pairs follow the meshes under the roots (a grown chunk's: its material is its neighbours',
	// so its copy builds no pipeline), walked before each render: a new mesh is paired, and the copy
	// of one that left is dropped, disposed as an object a few walks later (three keeps a render
	// object, and its uniform buffers, per object until then).
	const paired = new Map<THREE.Mesh, Pair>();
	const retired: { depth: THREE.Mesh; at: number }[] = [];
	let walks = 0;
	const sync = () => {

		walks ++;
		while ( retired.length > 0 && retired[ 0 ].at <= walks ) retired.shift()!.depth.dispose();
		const seen = new Set<THREE.Mesh>();
		for ( const root of roots ) root.traverse( ( object ) => { if ( eligible( object ) ) seen.add( object ); } );
		let changed = false;
		for ( const mesh of seen ) if ( ! paired.has( mesh ) ) {

			pair( mesh );
			paired.set( mesh, pairs[ pairs.length - 1 ] );
			changed = true;

		}
		for ( const [ mesh, p ] of paired ) if ( ! seen.has( mesh ) ) {

			p.depth.removeFromParent();
			pairs.splice( pairs.indexOf( p ), 1 );
			paired.delete( mesh );
			retired.push( { depth: p.depth, at: walks + 6 } );
			changed = true;

		}
		// (a new pair takes the flags' state)
		if ( changed ) {

			state = '';
			info.pairs = pairs.length;

		}

	};

	let state = '';
	const info: VegetationPrepassInfo = { pairs: 0, materials: 0, depthMaterials: 0 };
	sync();

	// Applied before each render of the scene, so the debug flags switch it at run time.
	const apply = () => {

		const on = debugFlags.vegPrepass !== false;
		const opaque = on && debugFlags.vegPrepassOpaque !== false;
		const key = `${ on } ${ opaque }`;
		if ( key === state ) return;
		state = key;
		for ( const p of pairs ) {

			p.depth.visible = on;
			const m = p.material;
			if ( m.depthWrite === ! on && m.userData.prepassOpaque === opaque && m.userData.prepassOn === on ) continue;
			m.depthWrite = ! on;
			m.depthFunc = on ? THREE.EqualDepth : THREE.LessEqualDepth;
			m.alphaToCoverage = ! opaque && ! TEMPORAL;
			m.userData.prepassOpaque = opaque;
			m.userData.prepassOn = on;
			m.needsUpdate = true;

		}

	};
	const previous = scene.onBeforeRender;
	scene.onBeforeRender = ( ...args ) => {

		sync();
		apply();
		previous.apply( scene, args );

	};
	// (the copies in a scene of their own draw before the scene: they take their meshes' transform
	// and visibility, ancestors included, as children would)
	if ( MEADOW_SHADOWS ) prepassScene.onBeforeRender = () => {

		sync();
		apply();
		for ( const p of pairs ) {

			p.mesh.updateWorldMatrix( true, false );
			p.depth.matrixWorld.copy( p.mesh.matrixWorld );
			let shown = debugFlags.vegPrepass !== false;
			for ( let o: THREE.Object3D | null = p.mesh; o && shown; o = o.parent ) shown = o.visible;
			p.depth.visible = shown;

		}

	};
	info.pairs = pairs.length;
	info.materials = patched.size;
	info.depthMaterials = depthCopies.size;
	scene.userData.vegetationPrepass = info;
	return info;

}
