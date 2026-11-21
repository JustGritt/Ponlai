/*!
 * The top-down collision height pass follows "collision rain" in threejs-conference by
 * Anderson Mancini and Sunag (https://github.com/ektogamat/threejs-conference,
 * src/world/weather/createCollisionHeight.js).
 * SPDX-FileCopyrightText: Copyright (c) 2026 Anderson Mancini and Sunag
 * SPDX-License-Identifier: MIT
 *
 * The same pass is in the three.js example webgpu_compute_particles_rain.html.
 * SPDX-FileCopyrightText: Copyright (c) 2010-2026 three.js authors
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { Fn, ivec2, max, positionWorld, select, texture, textureLoad, uniform, vec4 } from 'three/tsl';
import type { Bounds } from '../RenderTypes.ts';
import { compileLatentInto } from '../post/graph/Compile.ts';
import { followPlantDraw } from '../vegetation/PlantInstances.ts';
import { shared } from '../shared.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;

// The rain's collision map: the island's highest surface seen from straight above.

// Texels per metre: a drop finds the meadow, a plant's crown or a hanging strand to within
// a few centimetres, finer than a splash.
const TEXELS_PER_METRE = 14;
// Rim foliage hangs and the cliff toe flares beyond the land's footprint.
const PAD = 6;

/** The highest surface under every point of the island, seen from straight above: an
 * orthographic camera draws world height into a small half-float target over the land's box
 * (`setArea`). Beyond it, and wherever nothing was drawn, the sea is the ground. The island
 * never moves, so the map is drawn once (with the plants at rest), not every frame, and again on
 * a Grow, whatever the weather: besides the rain's drops, the haze reads it for the cover that
 * hides it from the open sky (VolumetricHaze.ts HazeCover, `topAt` and `version`).
 * A bake draws into a second map (`setArea` sizes it, `bakeSteps` may spread over frames) while
 * the readers keep the map of the bake before; the finished map is swapped in whole, with its
 * placement, through the one texture node every reader samples by (its base's value: no new
 * material or pipeline), so no frame reads a map half drawn. The map swapped out is released. */
export class RainCollision {

	// the map the readers sample (`front`) and the one a bake draws into (`back`)
	private front: THREE.RenderTarget;
	private back: THREE.RenderTarget;
	// the base node every reader's load clones, whose value is the front map's texture
	private readonly map: THREE.TextureNode;
	readonly camera: THREE.OrthographicCamera;
	// x0, z0 of the map and 1 / its width and depth (m), and its texels: the front map's; the back
	// map's wait in `area` until its bake finishes
	private readonly xform = uniform( new THREE.Vector4() );
	private readonly size = uniform( new THREE.Vector2() );
	private readonly area = { xform: new THREE.Vector4(), size: new THREE.Vector2( 1, 1 ) };
	readonly waterLevel = uniform( 0 );
	// Highest height in the map: above it a drop cannot have hit anything yet.
	readonly ceiling: number;
	/** Height of the first surface a drop meets falling onto world (x, z): the island's top
	 * wherever it was drawn, else the sea. */
	readonly groundAt: ( xz: V2 ) => F;
	/** `groundAt` built inline, for a kernel of its own (a layout helper reading uniforms can be
	 * emitted with another compute kernel's uniform names: docs/reference/three-behaviour.md, Shader functions) */
	readonly topAt: ( xz: V2 ) => F;
	/** counts the bakes finished, each one's map whole (the haze's rest starts over on a new one) */
	version = 0;

	// `ceiling`: the highest height in the map (m), over everything a bake draws
	constructor( ceiling: number ) {

		// sized to the land by setArea; until then an empty map (the sea everywhere)
		const map = () => {

			const target = new THREE.RenderTarget( 1, 1, { type: THREE.HalfFloatType, depthBuffer: true } );
			target.texture.name = 'rainCollisionHeight';
			// Nearest texels: a filtered lookup would blend the cliff top with the sea below it.
			target.texture.minFilter = target.texture.magFilter = THREE.NearestFilter;
			target.texture.generateMipmaps = false;
			return target;

		};
		this.front = map();
		this.back = map();
		this.map = texture( this.front.texture );
		this.size.value.set( 1, 1 );
		this.camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 80 );
		// Looking straight down with -z up the image: x runs right, z down the rows from z0.
		this.camera.up.set( 0, 0, - 1 );
		this.ceiling = ceiling;
		const heights = this.map, xform = this.xform, size = this.size, waterLevel = this.waterLevel;
		const top = ( xz: V2 ): F => {

			const uv = xz.sub( xform.xy ).mul( xform.zw ).toVar();
			const inside = uv.x.greaterThanEqual( 0 ).and( uv.x.lessThan( 1 ) ).and( uv.y.greaterThanEqual( 0 ) ).and( uv.y.lessThan( 1 ) );
			const sample = textureLoad( heights, ivec2( uv.clamp( 0, 0.9999 ).mul( size ) ) ).toVar();
			return max( select( inside.and( sample.a.greaterThan( 0.5 ) ), sample.r, waterLevel ), waterLevel );

		};
		this.topAt = top;
		this.groundAt = Fn( ( [ xz ]: [ V2 ] ) => top( xz ) ).setLayout( { name: 'ponlaiRainGround', type: 'float', inputs: [ { name: 'xz', type: 'vec2' } ] } ) as unknown as ( xz: V2 ) => F;

	}

	/** The map the readers sample (the last finished bake's). */
	get target(): THREE.RenderTarget { return this.front; }

	/** Covers the land's box `bounds` (WorldShapes.landBounds) and PAD round it, for the next bake
	 * (the readers keep the last bake's map and placement until it finishes); bake after. */
	setArea( bounds: Bounds ): void {

		const x0 = bounds.x0 - PAD, z0 = bounds.z0 - PAD;
		const width = bounds.x1 - bounds.x0 + PAD * 2, depth = bounds.z1 - bounds.z0 + PAD * 2;
		const w = Math.ceil( width * TEXELS_PER_METRE ), h = Math.ceil( depth * TEXELS_PER_METRE );
		this.back.setSize( w, h );
		this.area.xform.set( x0, z0, 1 / width, 1 / depth );
		this.area.size.set( w, h );
		const camera = this.camera;
		camera.left = - width / 2; camera.right = width / 2; camera.top = depth / 2; camera.bottom = - depth / 2;
		camera.updateProjectionMatrix();
		camera.position.set( x0 + width / 2, 40, z0 + depth / 2 );
		camera.lookAt( x0 + width / 2, 0, z0 + depth / 2 );
		camera.updateMatrixWorld();

	}

	/** Draw the height of every mesh under `roots` (skinned figures excepted: they move). Each
	 * material's rest-pose position and alpha test are kept, so blades and leaf cards print
	 * their own shapes rather than their bounding quads. A closed solid's cast position is the sun
	 * shadow's push away from the light (SunShadow.castFrontFaces), not a pose: it follows the sun,
	 * so a map drawn with it would depend on the hour it was drawn at (a grown page's against a
	 * fresh load's, where the push is still 0); such a material prints its own surface. The ground
	 * is dry: standing water sinks the blades (Grass.ts, `shared.wetness`), which the weather sets
	 * every frame, so the draws hold it at 0, as at the load's bake before the first frame. Again
	 * whenever the island grows: the height materials and the meshes' proxies are kept from bake to
	 * bake (a proxy per mesh, a material per source material), so a bake after the first builds no
	 * pipeline, and every mesh is drawn whole (all its instances, its whole draw range) as at the
	 * first. */
	bake( renderer: THREE.WebGPURenderer, roots: THREE.Object3D[] ): void {

		const steps = this.bakeSteps( renderer, roots );
		while ( ! steps.next().done );

	}

	/** `bake` in steps of `per` meshes a frame (a generator: Growth.ts takes a step a frame), drawn
	 * into the map in turn without clearing it between them: the map is the same as one draw of them
	 * all (each texel the highest surface's height). */
	* bakeSteps( renderer: THREE.WebGPURenderer, roots: THREE.Object3D[], per = Infinity ): Generator<void, void> {

		const scene = this.scene;
		const seen = new Set<THREE.Mesh>();
		for ( const root of roots ) root.traverse( ( object ) => {

			const mesh = object as THREE.Mesh;
			// (a vegetation mesh's depth prepass copy, VegetationPrepass.ts, prints the same shape; a
			// mesh marked `rainBake: false` leaves its shape to another: a plant's far level)
			if ( ! mesh.isMesh || ( mesh as THREE.SkinnedMesh ).isSkinnedMesh || ! object.visible || mesh.userData.prepassCopy || mesh.userData.rainBake === false ) return;
			seen.add( mesh );
			const instanced = mesh as THREE.InstancedMesh;
			const proxy = this.proxyOf( mesh );
			proxy.geometry = mesh.geometry;
			// (the grass picks its count per draw, Grass.ts: its whole count is `bakeCount`)
			if ( instanced.isInstancedMesh ) ( proxy as THREE.InstancedMesh ).count = mesh.userData.bakeCount ?? instanced.count;
			mesh.updateWorldMatrix( true, false );
			proxy.matrix.copy( mesh.matrixWorld );

		} );
		const dropped: THREE.Mesh[] = [];
		for ( const [ mesh, proxy ] of this.proxies ) {

			if ( seen.has( mesh ) ) continue;
			scene.remove( proxy );
			this.proxies.delete( mesh );
			dropped.push( proxy );

		}
		scene.updateMatrixWorld( true );
		const list = [ ...this.proxies.values() ];
		for ( let first = 0; first < list.length; first += per ) {

			const part = list.slice( first, first + per );
			for ( const p of list ) p.visible = false;
			for ( const p of part ) p.visible = true;
			// every geometry drawn whole (the grass sets its blades' range per view draw)
			const ranges = part.map( ( p ) => [ p.geometry.drawRange.start, p.geometry.drawRange.count ] as const );
			for ( const p of part ) p.geometry.setDrawRange( 0, Infinity );
			const target = renderer.getRenderTarget(), autoClear = renderer.autoClear;
			const clear = renderer.getClearColor( new THREE.Color() ), alpha = renderer.getClearAlpha();
			renderer.setRenderTarget( this.back );
			renderer.setClearColor( 0x000000, 0 );
			if ( first === 0 ) renderer.clear();
			renderer.autoClear = false;
			// the ground dry (see `bake`), the weather's wetness back for the frame's own draws
			const wetness = shared.wetness.value;
			shared.wetness.value = 0;
			renderer.render( scene, this.camera );
			shared.wetness.value = wetness;
			renderer.autoClear = autoClear;
			renderer.setRenderTarget( target );
			renderer.setClearColor( clear, alpha );
			part.forEach( ( p, k ) => p.geometry.setDrawRange( ranges[ k ][ 0 ], ranges[ k ][ 1 ] ) );
			for ( const p of list ) p.visible = true;
			if ( first + per < list.length ) yield;

		}
		// the meshes gone since the last bake: their proxies' draws released (three keeps a render
		// object, and its uniform buffers, per object until it is disposed), once the new proxies
		// have drawn, so a pipeline they shared is never left without a user
		for ( const proxy of dropped ) proxy.dispose();
		// the finished map swapped in whole, with its placement; the one swapped out released
		[ this.front, this.back ] = [ this.back, this.front ];
		this.map.value = this.front.texture;
		this.xform.value.copy( this.area.xform );
		this.size.value.copy( this.area.size );
		this.back.dispose();
		this.version ++;

	}

	/** Makes the proxies of meshes a later bake will draw (a grown island's new chunks: Growth.ts)
	 * and compiles them into the map, so that bake builds nothing. */
	prepare( renderer: THREE.WebGPURenderer, meshes: THREE.Mesh[] ): Promise<void> {

		const proxies = meshes.map( ( mesh ) => {

			const proxy = this.proxyOf( mesh );
			mesh.updateWorldMatrix( true, false );
			proxy.matrix.copy( mesh.matrixWorld );
			proxy.updateMatrixWorld();
			return proxy;

		} );
		return compileLatentInto( renderer, proxies, this.scene, this.camera, this.back );

	}

	// The mesh's proxy, made on first need: its height material, its instances.
	private proxyOf( mesh: THREE.Mesh ): THREE.Mesh {

		let proxy = this.proxies.get( mesh );
		if ( proxy ) return proxy;
		const source = Array.isArray( mesh.material ) ? mesh.material[ 0 ] : mesh.material;
		const instanced = mesh as THREE.InstancedMesh;
		if ( instanced.isInstancedMesh ) {

			const copy = new THREE.InstancedMesh( mesh.geometry, this.heightOf( source ), instanced.count );
			copy.instanceMatrix = instanced.instanceMatrix;
			proxy = copy;

		} else proxy = new THREE.Mesh( mesh.geometry, this.heightOf( source ) );
		// a plant's draw takes its arguments as it draws (vegetation/PlantInstances.ts): every
		// plant at its near level for this camera
		if ( mesh.userData.plantDraw !== undefined ) {

			const plant = proxy;
			plant.onBeforeRender = ( renderer, scene, camera, geometry, material, group ) => {

				mesh.onBeforeRender( renderer, scene, camera, geometry, material, group );
				followPlantDraw( plant, mesh );

			};
			plant.onAfterRender = ( renderer, scene, camera, geometry, material, group ) => mesh.onAfterRender( renderer, scene, camera, geometry, material, group );

		}
		proxy.matrixAutoUpdate = false;
		proxy.frustumCulled = false;
		this.proxies.set( mesh, proxy );
		this.scene.add( proxy );
		return proxy;

	}

	private readonly scene = new THREE.Scene();
	private readonly proxies = new Map<THREE.Mesh, THREE.Mesh>();
	private readonly materials = new Map<THREE.Material, THREE.MeshBasicNodeMaterial>();

	private heightOf( source: THREE.Material ): THREE.MeshBasicNodeMaterial {

		let material = this.materials.get( source );
		if ( material ) return material;
		const node = source as THREE.MeshStandardNodeMaterial;
		material = new THREE.MeshBasicNodeMaterial( { side: THREE.DoubleSide } );
		// the rest pose a vegetation material casts with (VegetationShadow.ts `castRest`, whatever
		// `flags.shadowWind` says), never the shadow's push
		const rest = node.userData.shadowPush ? null : node.userData.castRest ?? node.castShadowPositionNode;
		material.positionNode = rest ?? node.positionNode ?? null;
		if ( node.alphaMap ) {

			material.alphaMap = node.alphaMap;
			material.alphaTest = node.alphaTest;

		}
		material.outputNode = vec4( positionWorld.y, 0, 0, 1 );
		this.materials.set( source, material );
		return material;

	}

	dispose(): void {

		this.front.dispose();
		this.back.dispose();
		for ( const material of this.materials.values() ) material.dispose();

	}

}
