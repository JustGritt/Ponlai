import * as THREE from 'three/webgpu';
import { positionView, vec4 } from 'three/tsl';
import { followPlantDraw } from '../vegetation/PlantInstances.ts';
import { onSceneDispose } from '../SceneDispose.ts';

// The opaque scene's depth prepass, for the screen-space occlusion and the screen-space shadows.
//
// The opaque scene's depth prepass, for the screen-space ambient occlusion (post/GTAO.ts) and the
// screen-space shadows over the whole scene (post/MeadowShadows.ts `SS_SHADOW`): the island's chunks,
// the submerged stones, the sea floor, the bamboo's culms and the yunpeng drawn before the opaque
// scene, at the occlusion's half resolution into a target of their own, so that both read the whole
// scene's depth (theirs beside the vegetation's, VegetationPrepass.ts) before the materials that
// take them are shaded. It is built with either (GTAO.ts `GTAO`: the high tier's `ao` setting, `flags=gtao`;
// `SS_SHADOW`: unless `flags=ssShadow:false`), and drawn while either runs: otherwise nothing here
// is made.
//
// - A copy draws with a material that shares its mesh's position node and side and writes the
//   distance along the view (-z in view space; 0, the clear, where none drew) into a single-sampled
//   half-resolution target with a depth of its own; meshes whose materials agree on these share
//   one, so the copies build a few pipelines in all. With screen-space shadows over the whole scene the
//   target holds a second channel, the depth a surface runs behind its front where a material marks
//   it thinner than the march assumes (`userData.screenThickness`, post/MeadowShadows.ts
//   `screenThickness`: the bamboo's culms), 0 elsewhere; a thickness is part of the copy's key.
//   Drawn into the scene's 4x depth at full resolution instead (so that the opaque scene could test
//   equal against it) it cost about as much as the occlusion's own passes; at half resolution, single-sampled, a fraction of that. The shaded
//   meshes keep their programs and draws as they are without the occlusion.
// - The copies live in a scene of their own (`opaquePrepassScene`), as the vegetation's do. Before
//   each of its renders the scene walks its roots, makes a copy for a mesh it has not met (a grown
//   chunk: its material and layout are its neighbours', so it builds no pipeline), drops the copy
//   of one that left, and gives every copy its mesh's geometry (a growth swaps the stones' and the
//   floor's), transform and visibility, ancestors included. A dropped copy is disposed as an object
//   three renders later, which frees the render objects three keeps for it.
// - Left out: what has its own prepass (the vegetation's coverage meshes, `userData.vegetation`),
//   skinned meshes (the islanders: their skinning would have to repeat bit for bit), instanced
//   meshes (none outside the vegetation; a copy would have to follow its instances and bounds),
//   what writes no depth or blends, and what lies off the view's layer.
// - A plant's draws (the bamboo's culms: vegetation/PlantInstances.ts) are plain meshes whose
//   instances the GPU cull picks as they draw: a copy runs its mesh's `onBeforeRender` and takes its
//   draw and count. It draws the culms whole, without their level stipple, so within the switch's
//   band both levels write their depth here (the nearer wins: the culms differ by their sides).

/** The opaque copies' scene: drawn in a pass of its own after the vegetation's depth (Pipeline.ts). */
export const opaquePrepassScene = new THREE.Scene();
opaquePrepassScene.name = 'opaque prepass';
// a disposed scene's copies go with it (its renderer freed what they held)
onSceneDispose( 'opaque prepass', () => {

	opaquePrepassScene.clear();
	opaquePrepassScene.onBeforeRender = THREE.Object3D.prototype.onBeforeRender;

} );

// whether a mesh takes a copy
function eligible( object: THREE.Object3D ): object is THREE.Mesh {

	const mesh = object as THREE.Mesh;
	if ( ! mesh.isMesh || ( mesh as THREE.SkinnedMesh ).isSkinnedMesh || ( mesh as THREE.InstancedMesh ).isInstancedMesh || mesh.userData.prepassCopy || Array.isArray( mesh.material ) ) return false;
	if ( ! mesh.layers.isEnabled( 0 ) ) return false;
	const m = mesh.material as THREE.Material;
	// eslint-disable-next-line ponlai/vegetation-tag -- not finding vegetation: no cut-out under alpha to coverage, a plant's or an islander's, can draw an opaque depth copy; vegetation is left out by its tag beside it
	return ( m as THREE.NodeMaterial ).isNodeMaterial === true && m.depthWrite && ! m.transparent && ! m.alphaToCoverage && m.userData.vegetation !== true && m.colorWrite;

}

/** Installs the opaque prepass for the meshes under `roots`, now and as they change. Returns the
 * number of copies made at install. */
export function installOpaquePrepass( roots: THREE.Object3D[] ): number {

	const depthMaterials = new Map<string, THREE.MeshBasicNodeMaterial>();
	const ids = new Map<THREE.Node | null, number>();
	const depthMaterial = ( source: THREE.NodeMaterial ) => {

		const position = source.positionNode ?? null;
		const thickness = ( source.userData.screenThickness as number | undefined ) ?? 0;
		if ( ! ids.has( position ) ) ids.set( position, ids.size );
		const key = `${ ids.get( position ) } ${ source.side } ${ thickness }`;
		let m = depthMaterials.get( key );
		if ( ! m ) {

			m = new THREE.MeshBasicNodeMaterial( { side: source.side } );
			m.positionNode = position;
			m.colorNode = vec4( positionView.z.negate(), thickness, 0, 1 );
			m.toneMapped = false;
			m.name = `opaque-depth ${ depthMaterials.size }`;
			depthMaterials.set( key, m );

		}
		return m;

	};

	const copies = new Map<THREE.Mesh, THREE.Mesh>();
	const make = ( mesh: THREE.Mesh ) => {

		const material = depthMaterial( mesh.material as THREE.NodeMaterial );
		const copy = new THREE.Mesh( mesh.geometry, material );
		copy.name = mesh.name + '-depth';
		copy.userData.prepassCopy = true;
		copy.matrixAutoUpdate = false;
		copy.matrixWorldAutoUpdate = false;
		copy.castShadow = false;
		copy.receiveShadow = false;
		// a plant's draw picks its instances as it draws (vegetation/PlantInstances.ts): the copy
		// asks it for the same view and takes its draw and count
		if ( mesh.userData.plantDraw !== undefined ) copy.onBeforeRender = ( renderer, scene, camera, geometry, m, group ) => {

			mesh.onBeforeRender( renderer, scene, camera, geometry, m, group );
			followPlantDraw( copy, mesh );

		};
		copies.set( mesh, copy );
		opaquePrepassScene.add( copy );

	};
	const seen = new Set<THREE.Mesh>();
	// copies of meshes that left, disposed a few renders later: three keeps a render object, and
	// its uniform buffers, per object until the object's dispose, and a command buffer still held
	// may draw the copy (Growth.ts disposes the meshes it replaces the same way)
	let renders = 0;
	const retired: { copy: THREE.Mesh; at: number }[] = [];
	const walk = () => {

		renders ++;
		while ( retired.length > 0 && retired[ 0 ].at <= renders ) retired.shift()!.copy.dispose();
		seen.clear();
		for ( const root of roots ) root.traverse( ( object ) => {

			if ( eligible( object ) ) seen.add( object );

		} );
		for ( const mesh of seen ) if ( ! copies.has( mesh ) ) make( mesh );
		for ( const [ mesh, copy ] of copies ) if ( ! seen.has( mesh ) ) {

			opaquePrepassScene.remove( copy );
			copies.delete( mesh );
			retired.push( { copy, at: renders + 3 } );

		}

	};
	walk();

	opaquePrepassScene.onBeforeRender = () => {

		walk();
		for ( const [ mesh, copy ] of copies ) {

			mesh.updateWorldMatrix( true, false );
			copy.matrixWorld.copy( mesh.matrixWorld );
			copy.geometry = mesh.geometry;
			copy.renderOrder = mesh.renderOrder;
			copy.frustumCulled = mesh.frustumCulled;
			let shown = true;
			for ( let o: THREE.Object3D | null = mesh; o && shown; o = o.parent ) shown = o.visible;
			copy.visible = shown;

		}

	};
	return copies.size;

}
