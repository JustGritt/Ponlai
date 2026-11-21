import * as THREE from 'three/webgpu';
import {
	Fn, Discard, uniform, textureLoad, float, vec2, vec3, vec4, ivec2, uv, positionWorld, normalWorldGeometry,
	max, sqrt, normalize, select, clamp, refract,
} from 'three/tsl';
import type { Shared } from '../shared.ts';
import { solarShadowVisibility, solarVolumeVisibility } from './SolarShadow.ts';
import { NORMAL_OFFSET } from './SunShadow.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { LEVEL_MARGIN, type SubmergedSubset } from './CasterFormat.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type M4 = THREE.Node<'mat4'>;

// The key light's shadow below the water: a map drawn along the sun refracted at the surface.
//
// The key light's shadow below the water: a map drawn along the sun refracted at the mean surface,
// read by submerged receivers (and the sea floor's stones, which cast only here). It redraws when
// the key light has turned by its angle gate (0.03 degrees on high; the quality tier sets it), and
// exactly whenever the light is at rest (the same direction on two consecutive updates), so a
// paused frame is the same however it was reached: after a paused seek it redraws on the second
// update. The map grows once from its constructed size to 2048 when a light's fit needs it; a scene
// jump (`reset`, then `update` at once) returns it to the constructed size and draws it, as a new
// page fits and draws that light before its first frame. Its proxies draw only the casters'
// triangles that reach below the water: cut here from the vertices, or for the chunks' meshes in
// the world's workers (`setCasters`, ChunkDerived.ts).
// A submerged receiver reads the air's map (the island's shadow) where its light entered the water:
// up the sun ray refracted through the long waves' slope to their surface (`waveEntryNode`, from the
// wave maps: WaveMaps.ts), as tidewater's hookShadowPosition finds it (`bindTerrain`).

const WATER_IOR = 1.333;
// Turn of the key light below which a running clock keeps the drawn map (see RefractedSun.update).
const ANGLE_GATE = THREE.MathUtils.degToRad( 0.03 );
// Caster triangles reaching no lower than LEVEL_MARGIN above the water level cannot write the map
// (its material discards everything above the level), so the proxies leave them out
// (ChunkDerived.ts, which cuts the chunks' meshes alike in the world's workers).

// Snell's law for the mean horizontal surface. Both vectors point toward the source.
// The source beam's irradiance changes by cos(thetaAir) / cos(thetaWater): a horizontal
// area intercepts the same transmitted power on either side of the interface.
export const refractedSunDirectionNode = Fn( ( [ airDirection ]: [ V3 ] ) => {

	const air = normalize( airDirection ).toVar();
	const mu = sqrt( max( float( 1 ).sub( float( 1 ).sub( air.y.mul( air.y ) ).div( WATER_IOR * WATER_IOR ) ), 0 ) );
	return vec3( air.x.div( WATER_IOR ), mu, air.z.div( WATER_IOR ) );

} ).setLayout( { name: 'refractedSolarDirection', type: 'vec3', inputs: [ { name: 'airDirection', type: 'vec3' } ] } );

export const refractedSunBeamScaleNode = Fn( ( [ airDirection ]: [ V3 ] ) => {

	const air = normalize( airDirection ).toVar();
	return max( air.y, 0 ).div( max( refractedSunDirectionNode( air ).y, 1e-5 ) );

} ).setLayout( { name: 'refractedSolarBeamScale', type: 'float', inputs: [ { name: 'airDirection', type: 'vec3' } ] } );

export const refractedSurfaceEntryNode = Fn( ( [ point, airDirection, waterLevel ]: [ V3, V3, F ] ) => {

	const direction = refractedSunDirectionNode( airDirection ).toVar();
	return point.add( direction.mul( max( waterLevel.sub( point.y ), 0 ).div( direction.y ) ) );

} ).setLayout( { name: 'refractedSolarSurfaceEntry', type: 'vec3', inputs: [
	{ name: 'point', type: 'vec3' }, { name: 'airDirection', type: 'vec3' }, { name: 'waterLevel', type: 'float' },
] } );

const shadowCoordinate = Fn( ( [ point, matrix, bias ]: [ V3, M4, F ] ) => {

	const clip = matrix.mul( vec4( point, 1 ) ).toVar();
	const ndc = clip.xyz.div( clip.w ).toVar();
	return vec3( ndc.x.mul( 0.5 ).add( 0.5 ), ndc.y.mul( - 0.5 ).add( 0.5 ), ndc.z.sub( bias ) );

} ).setLayout( { name: 'refractedSolarShadowCoordinate', type: 'vec3', inputs: [
	{ name: 'point', type: 'vec3' }, { name: 'matrix', type: 'mat4' }, { name: 'bias', type: 'float' },
] } );

export function refractedSunDirection( airDirection: THREE.Vector3, target = new THREE.Vector3() ): THREE.Vector3 {

	target.copy( airDirection ).normalize();
	const mu = Math.sqrt( Math.max( 1 - ( 1 - target.y * target.y ) / ( WATER_IOR * WATER_IOR ), 0 ) );
	return target.set( target.x / WATER_IOR, mu, target.z / WATER_IOR );

}

function refractedSurfaceEntry( point: THREE.Vector3, airDirection: THREE.Vector3, waterLevel: number, target = new THREE.Vector3() ): THREE.Vector3 {

	const direction = refractedSunDirection( airDirection );
	return target.copy( point ).addScaledVector( direction, Math.max( waterLevel - point.y, 0 ) / direction.y );

}

function makeDepthTarget( size: number ): THREE.RenderTarget {

	const depth = new THREE.DepthTexture( size, size, THREE.FloatType );
	depth.compareFunction = THREE.LessEqualCompare;
	depth.minFilter = depth.magFilter = THREE.LinearFilter;
	depth.name = 'submerged-solar-depth';
	const target = new THREE.RenderTarget( size, size, { depthTexture: depth } );
	target.texture.name = 'submerged-solar-unused-color';
	return target;

}

function fitCamera( camera: THREE.OrthographicCamera, bounds: THREE.Box3, direction: THREE.Vector3, receiverFloor?: number ): void {

	const points: THREE.Vector3[] = [];
	const receiverBounds = bounds.clone();
	for ( const x of [ bounds.min.x, bounds.max.x ] ) for ( const y of [ bounds.min.y, bounds.max.y ] ) for ( const z of [ bounds.min.z, bounds.max.z ] ) {

		const point = new THREE.Vector3( x, y, z );
		points.push( point );
		if ( receiverFloor !== undefined ) {

			const end = point.clone().addScaledVector( direction, - Math.max( y - receiverFloor, 0 ) / direction.y );
			points.push( end );
			receiverBounds.expandByPoint( end );

		}

	}
	const centre = receiverBounds.getCenter( new THREE.Vector3() );
	const distance = receiverBounds.getSize( new THREE.Vector3() ).length() + 10;
	camera.position.copy( centre ).addScaledVector( direction, distance );
	camera.up.set( 0, 1, 0 );
	if ( Math.abs( direction.y ) > 0.99 ) camera.up.set( 0, 0, 1 );
	camera.lookAt( centre );
	camera.updateMatrixWorld( true );
	const lightBounds = new THREE.Box3();
	for ( const point of points ) {

		lightBounds.expandByPoint( point.applyMatrix4( camera.matrixWorldInverse ) );

	}
	// Fit the actual refracted rays, not an axis-aligned box around their swept volume.
	// Extending a ray to the deepest floor preserves its light-space XY and only expands
	// the depth interval. The 420 m horizon mesh therefore cannot dilute local resolution.
	const padding = 2;
	camera.left = lightBounds.min.x - padding;
	camera.right = lightBounds.max.x + padding;
	camera.bottom = lightBounds.min.y - padding;
	camera.top = lightBounds.max.y + padding;
	camera.near = Math.max( 0.1, - lightBounds.max.z - padding );
	camera.far = Math.max( camera.near + 1, - lightBounds.min.z + padding );
	camera.updateProjectionMatrix();

}

interface RefractedSunOptions { mapSize?: number; contactBias?: number }
// `empty`: its subset holds no triangle (a part of the island wholly above the water), so its proxy
// is not drawn: WebGPU warns about a draw of no indices.
// `geometry`: the source's geometry when the proxy was made; `cut`: its subset, cut at `cutAt`.
interface Caster { source: THREE.Mesh; proxy: THREE.Mesh; floor: boolean; subset?: SubmergedSubset; empty?: boolean; geometry: THREE.BufferGeometry; cut: THREE.BufferGeometry | null; cutAt: number }

// The source's triangles that reach below `level` in world space, sharing its position (all of
// them when every triangle does: allTriangles).
function submergedTriangles( source: THREE.Mesh, level: number ): THREE.BufferGeometry | null {

	const geometry = source.geometry, position = geometry.getAttribute( 'position' );
	if ( ! position ) return null;
	const index = geometry.getIndex();
	const count = index ? index.count : position.count;
	const e = source.matrixWorld.elements;
	const worldY = ( v: number ) => e[ 1 ] * position.getX( v ) + e[ 5 ] * position.getY( v ) + e[ 9 ] * position.getZ( v ) + e[ 13 ];
	const below = new Uint8Array( position.count );
	for ( let v = 0; v < position.count; v ++ ) below[ v ] = worldY( v ) < level ? 1 : 0;
	const kept: number[] = [];
	for ( let t = 0; t + 2 < count; t += 3 ) {

		const a = index ? index.getX( t ) : t, b = index ? index.getX( t + 1 ) : t + 1, c = index ? index.getX( t + 2 ) : t + 2;
		if ( below[ a ] || below[ b ] || below[ c ] ) kept.push( a, b, c );

	}
	if ( kept.length === count ) return allTriangles( source );
	// the map's material reads the position alone
	const subset = new THREE.BufferGeometry();
	subset.setAttribute( 'position', position );
	subset.setIndex( position.count > 65535 ? new THREE.Uint32BufferAttribute( kept, 1 ) : new THREE.Uint16BufferAttribute( kept, 1 ) );
	return sourceBounds( subset, source );

}

// A subset cut in a worker (ChunkDerived.ts), sharing the source's position as above.
function givenTriangles( source: THREE.Mesh, index: Uint32Array | null ): THREE.BufferGeometry | null {

	if ( ! index ) return allTriangles( source );
	const subset = new THREE.BufferGeometry();
	subset.setAttribute( 'position', source.geometry.getAttribute( 'position' ) );
	subset.setIndex( new THREE.BufferAttribute( index, 1 ) );
	return sourceBounds( subset, source );

}

// Every triangle of the source, as a subset (its position and index shared): the proxies then all
// draw a position-only geometry, so a caster whose triangles all reach below the water (a grown
// island's chunk under it) finds the pipeline the others built, where its own geometry's vertex
// layout would build another. Null for a mesh without an index (drawn as it is).
function allTriangles( source: THREE.Mesh ): THREE.BufferGeometry | null {

	const index = source.geometry.getIndex();
	if ( ! index ) return null;
	const subset = new THREE.BufferGeometry();
	subset.setAttribute( 'position', source.geometry.getAttribute( 'position' ) );
	subset.setIndex( index );
	subset.userData.sharedIndex = true;
	return sourceBounds( subset, source );

}

// A subset culled by its source's bounds, which hold its triangles: found by the frustum test at
// its first draw, they would read every vertex of the source (the whole sea floor's: tens of
// milliseconds in the frame a grown island swaps its floor in, Growth.ts). Culling by either is
// conservative, so the map is the same.
function sourceBounds( subset: THREE.BufferGeometry, source: THREE.Mesh ): THREE.BufferGeometry {

	const geometry = source.geometry;
	if ( geometry.boundingSphere ) subset.boundingSphere = geometry.boundingSphere.clone();
	if ( geometry.boundingBox ) subset.boundingBox = geometry.boundingBox.clone();
	return subset;

}

// Disposing a geometry releases every attribute's GPU buffer: detach the shared position (and a
// shared index) first, so only the subset's own index goes.
function disposeSubset( subset: THREE.BufferGeometry ): void {

	subset.deleteAttribute( 'position' );
	if ( subset.userData.sharedIndex ) subset.setIndex( null );
	subset.dispose();

}
interface Receiver {
	material: THREE.NodeMaterial;
	position: THREE.Node | null;
	shadow: THREE.NodeMaterial['receivedShadowNode'];
}

export class RefractedSun {

	readonly target: THREE.RenderTarget;
	readonly camera = new THREE.OrthographicCamera();
	readonly matrix = uniform( new THREE.Matrix4() );
	readonly worldSpan = uniform( new THREE.Vector2( 1, 1 ) );
	readonly depthRange = uniform( 1 );
	readonly mapSize: THREE.Node<'vec2'>;
	readonly contactBias: ReturnType<typeof uniform<'float', number>>;
	readonly directionNode: V3;
	readonly beamScaleNode: F;
	private readonly scene = new THREE.Scene();
	private readonly material = new THREE.NodeMaterial();
	private readonly casters: Caster[] = [];
	private readonly receivers: Receiver[] = [];
	private readonly lastSun = new THREE.Vector3( NaN, NaN, NaN );
	// the key direction at the previous update call, drawn or not
	private readonly seenSun = new THREE.Vector3( NaN, NaN, NaN );
	private readonly mapResolution = uniform( new THREE.Vector2() );
	private lastLevel = NaN;
	// the level the proxies' triangle subsets were cut at
	private subsetLevel = NaN;
	private dirty = true;
	/** Maps drawn since the page loaded (the settled signal's signature, post/Settled.ts; the dynamic
	 * resolution leaves frames with periodic work out of its quantiles). */
	draws = 0;
	// the map's size as constructed (it may grow once to 2048: see update)
	private readonly initialSize: number;
	// Turn of the key light (radians) below which a moving light keeps the drawn map: a quality
	// tier's setting. A light at rest is always drawn exactly.
	angleGate = ANGLE_GATE;

	constructor( private readonly renderer: THREE.WebGPURenderer, private readonly shared: Pick<Shared, 'waterLevel' | 'sunDir'>, { mapSize = 1024, contactBias = 0.02 }: RefractedSunOptions = {} ) {

		const size = Math.min( 2048, Math.max( 256, mapSize ) );
		this.initialSize = size;
		this.target = makeDepthTarget( size );
		this.mapResolution.value.set( size, size );
		this.mapSize = this.mapResolution;
		this.contactBias = uniform( contactBias );
		this.directionNode = refractedSunDirectionNode( shared.sunDir );
		this.beamScaleNode = refractedSunBeamScaleNode( shared.sunDir );
		this.camera.coordinateSystem = renderer.coordinateSystem;
		this.material.fragmentNode = Fn( () => {

			// An above-water blocker belongs to the air map. Keeping it in both maps would
			// project a false, unrefracted submerged shadow from its above-water portion.
			// The interface itself belongs to air as well: retaining a box's bottom face at
			// exactly sea level would duplicate its air shadow along the underwater ray.
			Discard( positionWorld.y.greaterThanEqual( shared.waterLevel ) );
			return vec4( 0, 0, 0, 1 );

		} )();
		this.material.side = THREE.DoubleSide;
		this.material.depthTest = true;
		this.material.depthWrite = true;
		this.material.blending = THREE.NoBlending;
		this.material.toneMapped = false;
		this.material.fog = false;

	}

	surfaceEntryNode( point: V3 ): V3 {

		return refractedSurfaceEntryNode( point, this.shared.sunDir, this.shared.waterLevel );

	}

	submergedVisibilityNode( point: V3, geometricNormal?: V3, active?: THREE.Node<'bool'> ): F {

		const normal = geometricNormal ? normalize( geometricNormal ) : undefined;
		// Geometry normals describe the caster plane; bump normals would move the receiver
		// independently of the depth geometry and turn detail textures into shadow artefacts.
		// The shared filter compares a footprint's four texels at one depth, so the receiver looks
		// its shadow up the sun's normal offset away in this map's texels (SunShadow NORMAL_OFFSET:
		// what clears a lit plane's rise inside a footprint); nearer contacts are not resolved.
		const texel = max( this.worldSpan.x.div( this.mapSize.x ), this.worldSpan.y.div( this.mapSize.y ) );
		const receiver = normal ? point.add( normal.mul( texel.mul( NORMAL_OFFSET.near ) ) ) : point;
		const coordinate = shadowCoordinate( receiver, this.matrix, this.contactBias.div( this.depthRange ) );
		// the air's disk filter (SolarShadow's POISSON_RADIUS, in this map's texels: on its coarser
		// texels the same disk draws a wider penumbra)
		return solarShadowVisibility( this.target.depthTexture!, coordinate, { mapSize: this.mapSize, active } );

	}

	submergedVolumeCoordinateNode( point: V3 ): V3 {

		return shadowCoordinate( point, this.matrix, this.contactBias.div( this.depthRange ) );

	}

	submergedVolumeVisibilityAtCoordinateNode( coordinate: V3 ): F {

		return solarVolumeVisibility( this.target.depthTexture!, coordinate, this.mapSize );

	}

	submergedVolumeVisibilityNode( point: V3 ): F {

		// A point in water has no surface normal to offset. Retain only the small ray
		// clearance used by terrain; points above the interface have no submerged ray leg.
		const coordinate = this.submergedVolumeCoordinateNode( point );
		const visibility = this.submergedVolumeVisibilityAtCoordinateNode( coordinate );
		return select( point.y.lessThan( this.shared.waterLevel ), visibility, float( 1 ) );

	}

	/** The air map's lookup point for a submerged receiver, as tidewater's hookShadowPosition finds
	 * it: back up the sun ray refracted through the long waves' slope above (`waves`, the wave
	 * maps: height over the sea level, slope x and z) to their surface, where the light entered the
	 * water. The shadows of the cliffs then land where refraction puts them and sway with the swell.
	 * (tidewater also jitters the point over a disk that grows with the depth, for the light's
	 * forward scattering, and lets its temporal resolve average it; the sun shadow's filter here
	 * takes the receiver plane from the point's derivatives, which a jitter would break, so the
	 * point is left unjittered.) The point itself where it lies above the long waves. */
	waveEntryNode( point: V3, waves: THREE.Node<'vec4'> ): V3 {

		return Fn( () => {

			const w = waves.toVar();
			const depth = this.shared.waterLevel.add( w.x ).sub( point.y ).toVar();
			const up = refract( this.shared.sunDir.negate(), normalize( vec3( w.y.negate(), 1, w.z.negate() ) ), 1 / WATER_IOR ).negate().toVar();
			const entry = point.add( up.mul( depth.div( max( up.y, 0.15 ) ) ) ).toVar();
			return select( depth.greaterThan( 0 ), entry, point );

		} )();

	}

	/** Gives every refracted-sun receiver under `root` its submerged shadows: the air map read where
	 * the light entered the water (the long waves' surface from `waves`, the wave maps: WaveMaps.ts;
	 * the mean level without), times this map. */
	bindTerrain( root: THREE.Object3D, waves?: { wavesAt( point: V3 ): THREE.Node<'vec4'> } ): void {

		const seen = new Set<THREE.NodeMaterial>();
		root.traverse( ( object ) => {

			if ( ! ( object as THREE.Mesh ).isMesh ) return;
			const mesh = object as THREE.Mesh;
			for ( const material of Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ] ) {

				if ( ! material.userData.refractedSolarReceiver || seen.has( material as THREE.NodeMaterial ) ) continue;
				const receiver = material as THREE.NodeMaterial;
				seen.add( receiver );
				if ( this.receivers.some( ( entry ) => entry.material === receiver ) ) continue;
				this.receivers.push( { material: receiver, position: receiver.receivedShadowPositionNode, shadow: receiver.receivedShadowNode } );
				const underwater = positionWorld.y.lessThan( this.shared.waterLevel );
				receiver.receivedShadowPositionNode = waves ? this.waveEntryNode( positionWorld, waves.wavesAt( positionWorld ) )
					: select( underwater, this.surfaceEntryNode( positionWorld ), positionWorld );
				// The runtime passes the existing shadow node; @types/three omits that argument.
				// The filter takes derivatives, so it runs for every pixel (in uniform control flow, not
				// inside the underwater branch, where a quad straddling the waterline would read
				// undefined ones); above the water it skips its map reads and returns lit.
				receiver.receivedShadowNode = Fn( ( [ airVisibility ]: [ V3 ] ) => {

					const submerged = this.submergedVisibilityNode( positionWorld, normalWorldGeometry, underwater ).toVar();
					return airVisibility.mul( select( underwater, submerged, float( 1 ) ) );

				} ) as unknown as THREE.NodeMaterial['receivedShadowNode'];
				receiver.needsUpdate = true;

			}

		} );

	}

	/** Draw the static meshes under `root` whose materials receive this map. `subsets` gives
	 * meshes' triangle subsets cut off the main thread (the chunks', ChunkDerived.ts: world space,
	 * for meshes drawn with an identity world matrix), used while the water level stays within
	 * half the margin of the level they were cut at; every other mesh is cut here. Again when the
	 * casters change (a grown island, Growth.ts): a mesh with the same geometry and subset as
	 * before keeps its proxy and its cut, so only the new ones are made. Returns the disposal of the
	 * proxies dropped, for once the new ones have drawn. */
	setCasters( root: THREE.Object3D, subsets?: ReadonlyMap<THREE.Object3D, SubmergedSubset> ): () => void {

		const before = new Map( this.casters.map( ( c ) => [ c.source, c ] ) );
		this.scene.clear();
		this.casters.length = 0;
		root.updateWorldMatrix( true, true );
		root.traverse( ( object ) => {

			const source = object as THREE.Mesh;
			if ( ! source.isMesh || ( source as THREE.InstancedMesh ).isInstancedMesh || ( source as THREE.SkinnedMesh ).isSkinnedMesh ) return;
			// World terrain is static. Animated grass/ivy/foliage must stay in the normal air
			// shadow pass; their undisplaced geometry would be an invalid submerged blocker.
			const materials = Array.isArray( source.material ) ? source.material : [ source.material ];
			if ( ! materials.some( ( material ) => material.userData.refractedSolarReceiver ) ) return;
			const subset = subsets?.get( source ), kept = before.get( source );
			if ( kept && kept.geometry === source.geometry && kept.subset === subset ) {

				before.delete( source );
				this.casters.push( kept );
				this.scene.add( kept.proxy );
				return;

			}
			const proxy = new THREE.Mesh( source.geometry, this.material );
			proxy.matrixAutoUpdate = false;
			proxy.matrix.copy( source.matrixWorld );
			proxy.visible = source.visible;
			proxy.name = `submerged-${ source.name }`;
			this.scene.add( proxy );
			const floor = source.userData.refractedSolarFloor === true || materials.every( ( material ) => material.userData.refractedSolarFloor === true );
			this.casters.push( { source, proxy, floor, subset, geometry: source.geometry, cut: null, cutAt: NaN } );

		} );
		for ( const dropped of before.values() ) if ( dropped.cut ) disposeSubset( dropped.cut );
		this.dirty = true;
		// (three keeps a dropped proxy's draws, and their uniform buffers, until it is disposed)
		const dropped = [ ...before.values() ].map( ( caster ) => caster.proxy );
		return () => { for ( const proxy of dropped ) proxy.dispose(); };

	}

	invalidate(): void { this.dirty = true; }

	/** Whether the drawn map is not the current light's (or level's): the next update due redraws
	 * it (the settled signal waits for that: post/Settled.ts). */
	get pending(): boolean {

		return this.dirty || this.lastLevel !== this.shared.waterLevel.value || ! this.lastSun.equals( this.shared.sunDir.value );

	}

	/** The map as constructed, redrawn at the next update (a scene jump, assemble/SceneJump.ts `establishScene`,
	 * which calls `update` right after, as `prepare` draws a new page's map before its first frame):
	 * a map grown to 2048 for one light keeps that size, so a light that a new page fits at the
	 * constructed size would be drawn at another resolution. */
	reset(): void {

		if ( this.target.width !== this.initialSize ) {

			this.target.setSize( this.initialSize, this.initialSize );
			this.mapResolution.value.set( this.initialSize, this.initialSize );

		}
		this.lastSun.set( NaN, NaN, NaN );
		this.seenSun.set( NaN, NaN, NaN );
		this.dirty = true;

	}

	// A static map gets no second frame to recover from a not-yet-ready first-use pipeline.
	// Call during scene setup, before the application starts its animation loop. Compile with
	// this exact target/depth format, then redraw after all caster pipelines are available.
	async prepare(): Promise<void> {

		this.update();
		const state = THREE.RendererUtils.resetRendererState( this.renderer, {} as THREE.RendererUtils.RendererState );
		try {

			this.renderer.setScissorTest( false );
			this.renderer.setRenderTarget( this.target );
			await this.renderer.compileAsync( this.scene, this.camera );

		} finally { THREE.RendererUtils.restoreRendererState( this.renderer, state ); }
		this.invalidate();
		this.update();

	}

	// Leave out the caster triangles that lie wholly above the water (plus a margin): the map's
	// material discards their fragments anyway, so the map is unchanged and their vertices are
	// not transformed at every redraw. Every caster when the level moved, else those not cut yet.
	private cutSubsets( level: number, all: boolean ): void {

		for ( const caster of this.casters ) {

			if ( ! all && caster.cutAt === level ) continue;
			const { source, proxy, subset: given } = caster;
			if ( caster.cut ) disposeSubset( caster.cut );
			source.updateWorldMatrix( true, false );
			const subset = debugFlags.refractedSunSubset === false ? null
				: given && Math.abs( level - given.level ) <= LEVEL_MARGIN * 0.5 ? givenTriangles( source, given.index )
					: submergedTriangles( source, level + LEVEL_MARGIN );
			proxy.geometry = subset ?? source.geometry;
			caster.empty = subset !== null && subset.getIndex()!.count === 0;
			caster.cut = subset;
			caster.cutAt = level;

		}
		this.subsetLevel = level;

	}

	update(): boolean {

		const sun = this.shared.sunDir.value, level = this.shared.waterLevel.value;
		// A light that has not moved since the previous call is at rest (a paused clock, or a seek
		// on one): draw it exactly, so a paused frame is the same however it was reached. A moving
		// light keeps the drawn map until it has turned by the gate; the map then trails the light
		// by at most that turn (debugFlags.refractedSunGate = false: redraw at any turn, as before).
		const resting = this.seenSun.equals( sun );
		this.seenSun.copy( sun );
		if ( ! this.dirty && this.lastLevel === level ) {

			if ( this.lastSun.equals( sun ) ) return false;
			if ( ! resting && debugFlags.refractedSunGate !== false && this.lastSun.angleTo( sun ) < this.angleGate ) return false;

		}
		const moved = ! ( Math.abs( level - this.subsetLevel ) <= LEVEL_MARGIN * 0.5 );
		if ( moved || this.casters.some( ( c ) => c.cutAt !== this.subsetLevel ) ) this.cutSubsets( moved ? level : this.subsetLevel, moved );
		const localBounds = new THREE.Box3();
		let floorY = Infinity;
		for ( const { source, proxy, floor, empty } of this.casters ) {

			source.updateWorldMatrix( true, false );
			proxy.matrix.copy( source.matrixWorld );
			proxy.matrixWorldNeedsUpdate = true;
			// (its source still fits the map: the fit is that of the whole caster set)
			proxy.visible = source.visible && ! empty;
			if ( ! source.visible ) continue;
			if ( ! source.geometry.boundingBox ) source.geometry.computeBoundingBox();
			const bounds = source.geometry.boundingBox!.clone().applyMatrix4( source.matrixWorld );
			if ( floor ) floorY = Math.min( floorY, bounds.min.y );
			else localBounds.union( bounds );

		}
		if ( localBounds.isEmpty() ) localBounds.set( new THREE.Vector3( - 1, level - 1, - 1 ), new THREE.Vector3( 1, level, 1 ) );
		// Above-water geometry is clipped out of this map and must not broaden its fit.
		localBounds.max.y = Math.max( localBounds.min.y, Math.min( localBounds.max.y, level ) );
		if ( ! Number.isFinite( floorY ) ) floorY = localBounds.min.y;
		fitCamera( this.camera, localBounds, refractedSunDirection( sun ), floorY );
		this.matrix.value.multiplyMatrices( this.camera.projectionMatrix, this.camera.matrixWorldInverse );
		this.worldSpan.value.set( this.camera.right - this.camera.left, this.camera.top - this.camera.bottom );
		this.depthRange.value = this.camera.far - this.camera.near;
		// Keep local contact detail without unbounded GPU allocations. The static map can
		// grow once to 2048; sun movement never shrinks/reallocates it afterward (a scene jump does: reset).
		if ( Math.max( this.worldSpan.value.x, this.worldSpan.value.y ) / this.target.width > 0.08 && this.target.width < 2048 ) {

			this.target.setSize( 2048, 2048 );
			this.mapResolution.value.set( 2048, 2048 );

		}
		const state = THREE.RendererUtils.resetRendererState( this.renderer, {} as THREE.RendererUtils.RendererState );
		try {

			this.renderer.setScissorTest( false );
			this.renderer.setRenderTarget( this.target );
			this.renderer.render( this.scene, this.camera );

		} finally { THREE.RendererUtils.restoreRendererState( this.renderer, state ); }
		this.lastSun.copy( sun );
		this.lastLevel = level;
		this.dirty = false;
		this.draws ++;
		return true;

	}

	dispose(): void {

		for ( const { material, position, shadow } of this.receivers ) {

			material.receivedShadowPositionNode = position;
			material.receivedShadowNode = shadow;
			material.needsUpdate = true;

		}
		this.receivers.length = 0;
		for ( const caster of this.casters ) if ( caster.cut ) disposeSubset( caster.cut );
		this.casters.length = 0;
		this.scene.clear();
		this.material.dispose();
		this.target.dispose();

	}

}

interface RefractedSunProbeResult {
	passed: boolean;
	pixels: number;
	air: { expectedSpan: number[]; measuredSpan: number[]; maximumEdgeError: number; mismatchedInteriorPixels: number };
	submerged: { expectedSpan: number[]; measuredSpan: number[]; maximumEdgeError: number; mismatchedInteriorPixels: number };
	horizontalIrradiance: { expected: number; measured: number; relativeError: number };
	refractionShortening: number;
	fixture: { role: string; expectedDepth: number; measuredDepth: number; absoluteError: number }[];
	grazingPlane: { normalLightCosine: number; minimumVisibility: number; maximumDarkening: number; unoffsetMinimumVisibility: number; sampledPixels: number };
}

// An isolated GPU regression probe, invoked through the dev harness after renderer.init().
// The red channel tests the refracted entry into an above-water box shadow; green tests a
// completely submerged box; alpha tests the beam-area/NdotL cancellation on a flat floor.
// A zero-radius source makes edge locations deterministic while retaining the production
// projection, clipping, depth comparison and receiver-plane code paths.
export async function probeRefractedSun( renderer: THREE.WebGPURenderer ): Promise<RefractedSunProbeResult> {

	const pixels = 256, floorY = - 4, extent = 15;
	const elevation = 32 * Math.PI / 180;
	const sun = new THREE.Vector3( Math.cos( elevation ), Math.sin( elevation ), 0 );
	const waterSun = refractedSunDirection( sun );
	const uniforms = { waterLevel: uniform( 0 ), sunDir: uniform( sun ) };
	const refracted = new RefractedSun( renderer, uniforms );
	const group = new THREE.Group();
	const placeholder = new THREE.NodeMaterial();
	placeholder.userData.refractedSolarReceiver = true;
	const floor = new THREE.Mesh( new THREE.PlaneGeometry( extent * 2, 8 ).rotateX( - Math.PI / 2 ), placeholder );
	floor.userData.refractedSolarFloor = true;
	floor.position.y = floorY;
	const airBox = new THREE.Mesh( new THREE.BoxGeometry( 2, 3, 2 ), placeholder );
	airBox.position.y = 1.5;
	const wetBox = new THREE.Mesh( new THREE.BoxGeometry( 2, 1, 2 ), placeholder );
	wetBox.position.set( 7, - 1.5, 0 );
	group.add( floor, airBox, wetBox );
	group.updateMatrixWorld( true );
	refracted.setCasters( group );
	const airScene = new THREE.Scene();
	const airMaterial = new THREE.NodeMaterial();
	airMaterial.side = THREE.DoubleSide;
	airMaterial.toneMapped = false;
	airMaterial.blending = THREE.NoBlending;
	airMaterial.depthTest = true;
	airMaterial.depthWrite = true;
	airMaterial.fragmentNode = Fn( () => {

		Discard( positionWorld.y.lessThan( uniforms.waterLevel ) );
		return vec4( 0, 0, 0, 1 );

	} )();
	for ( const source of [ floor, airBox, wetBox ] ) {

		const proxy = new THREE.Mesh( source.geometry, airMaterial );
		proxy.matrixAutoUpdate = false;
		proxy.matrix.copy( source.matrixWorld );
		airScene.add( proxy );

	}
	const airCamera = new THREE.OrthographicCamera();
	airCamera.coordinateSystem = renderer.coordinateSystem;
	fitCamera( airCamera, new THREE.Box3().setFromObject( group ), sun );
	const airMatrix = uniform( new THREE.Matrix4().multiplyMatrices( airCamera.projectionMatrix, airCamera.matrixWorldInverse ) );
	const airTarget = makeDepthTarget( 1024 );
	const airRange = airCamera.far - airCamera.near;
	const output = new THREE.RenderTarget( pixels, 1, { type: THREE.FloatType, depthBuffer: false } );
	const material = new THREE.NodeMaterial();
	material.toneMapped = false;
	material.depthTest = material.depthWrite = false;
	material.blending = THREE.NoBlending;
	material.fragmentNode = Fn( () => {

		const point = vec3( uv().x.mul( 2 * extent ).sub( extent ), floorY, 0 );
		const entry = refracted.surfaceEntryNode( point );
		const airVisibility = solarShadowVisibility( airTarget.depthTexture!, shadowCoordinate( entry, airMatrix, float( 0.02 / airRange ) ), { mapSize: vec2( 1024 ) } );
		const wetVisibility = refracted.submergedVisibilityNode( point, vec3( 0, 1, 0 ) );
		return vec4( airVisibility, wetVisibility, airVisibility.mul( wetVisibility ), refracted.directionNode.y.mul( refracted.beamScaleNode ) );

	} )();
	const quad = new THREE.QuadMesh( material );
	const diagnosticTarget = new THREE.RenderTarget( 4, 1, { type: THREE.FloatType, depthBuffer: false } );
	const diagnosticMaterial = new THREE.NodeMaterial();
	diagnosticMaterial.toneMapped = false;
	diagnosticMaterial.depthTest = diagnosticMaterial.depthWrite = false;
	diagnosticMaterial.blending = THREE.NoBlending;
	diagnosticMaterial.fragmentNode = Fn( () => {

		const x = select( uv().x.lessThan( 0.25 ), float( - 6 ),
			select( uv().x.lessThan( 0.5 ), float( - 12 ), select( uv().x.lessThan( 0.75 ), float( 5 ), float( 0 ) ) ) );
		const point = vec3( x, floorY, 0 );
		const airCoordinate = shadowCoordinate( refracted.surfaceEntryNode( point ), airMatrix, float( 0 ) );
		const wetCoordinate = shadowCoordinate( point, refracted.matrix, float( 0 ) );
		// Raw depth reads deliberately use another material. Current Three cannot safely
		// register an ordinary sampler before a comparison sampler for the same depth UUID.
		const airDepth = textureLoad( airTarget.depthTexture!, ivec2( clamp( airCoordinate.xy.mul( 1024 ), vec2( 0 ), vec2( 1023 ) ) ) ) as unknown as F;
		const wetDepth = textureLoad( refracted.target.depthTexture!, ivec2( clamp( wetCoordinate.xy.mul( refracted.mapSize ), vec2( 0 ), refracted.mapSize.sub( 1 ) ) ) ) as unknown as F;
		return vec4( airDepth, wetDepth, airCoordinate.z, wetCoordinate.z );

	} )();
	const diagnosticQuad = new THREE.QuadMesh( diagnosticMaterial );
	const grazingMap = new RefractedSun( renderer, uniforms );
	const grazingGroup = new THREE.Group();
	const grazingPlane = new THREE.Mesh( new THREE.PlaneGeometry( 20, 8 ).rotateX( - Math.PI / 2 ), placeholder );
	grazingPlane.position.y = - 12;
	grazingPlane.rotation.z = 43 * Math.PI / 180;
	grazingGroup.add( grazingPlane );
	grazingGroup.updateMatrixWorld( true );
	grazingMap.setCasters( grazingGroup );
	const grazingNormal = new THREE.Vector3( 0, 1, 0 ).applyQuaternion( grazingPlane.quaternion );
	const grazingTransform = uniform( grazingPlane.matrixWorld.clone() );
	const grazingTarget = new THREE.RenderTarget( 128, 16, { type: THREE.FloatType, depthBuffer: false } );
	const grazingMaterial = new THREE.NodeMaterial();
	grazingMaterial.toneMapped = false;
	grazingMaterial.depthTest = grazingMaterial.depthWrite = false;
	grazingMaterial.blending = THREE.NoBlending;
	grazingMaterial.fragmentNode = Fn( () => {

		const local = vec3( uv().x.mul( 12 ).sub( 6 ), 0, uv().y.mul( 4 ).sub( 2 ) );
		const point = grazingTransform.mul( vec4( local, 1 ) ).xyz;
		const normal = vec3( ...grazingNormal.toArray() );
		const offset = grazingMap.submergedVisibilityNode( point, normal );
		const unoffset = grazingMap.submergedVisibilityNode( point );
		return vec4( offset, unoffset, 1, 1 );

	} )();
	const grazingQuad = new THREE.QuadMesh( grazingMaterial );
	const animation = renderer.getAnimationLoop();
	await renderer.setAnimationLoop( null );
	const state = THREE.RendererUtils.resetRendererState( renderer, {} as THREE.RendererUtils.RendererState );
	// QuadMesh installs its full-screen vertex node only during draw. Match that exact key
	// when warming the probe, then submit the measurement after compileAsync completes.
	const renderReady = async ( mesh: THREE.QuadMesh ) => {

		const nodeMaterial = mesh.material as THREE.NodeMaterial;
		const before = mesh.onBeforeRender, originalVertex = nodeMaterial.vertexNode;
		let fullscreenVertex = originalVertex;
		mesh.onBeforeRender = ( ...args ) => {

			fullscreenVertex = nodeMaterial.vertexNode;
			before.apply( mesh, args );

		};
		try {

			mesh.render( renderer );
			nodeMaterial.vertexNode = fullscreenVertex;
			await renderer.compileAsync( mesh, mesh.camera );
			nodeMaterial.vertexNode = originalVertex;
			mesh.render( renderer );

		} finally { mesh.onBeforeRender = before; nodeMaterial.vertexNode = originalVertex; }

	};
	let data: THREE.TypedArray, diagnosticData: THREE.TypedArray, grazingData: THREE.TypedArray;
	try {

		try {

			renderer.setScissorTest( false );
			await refracted.prepare();
			await grazingMap.prepare();
			renderer.setRenderTarget( airTarget );
			renderer.render( airScene, airCamera );
			await renderer.compileAsync( airScene, airCamera );
			renderer.render( airScene, airCamera );
			renderer.setRenderTarget( output );
			await renderReady( quad );
			renderer.setRenderTarget( diagnosticTarget );
			await renderReady( diagnosticQuad );
			renderer.setRenderTarget( grazingTarget );
			await renderReady( grazingQuad );

		} finally { THREE.RendererUtils.restoreRendererState( renderer, state ); }
		// Release renderer state before waiting: the application's animation loop may
		// submit another frame while this asynchronous readback is pending.
		data = await renderer.readRenderTargetPixelsAsync( output, 0, 0, pixels, 1 );
		diagnosticData = await renderer.readRenderTargetPixelsAsync( diagnosticTarget, 0, 0, 4, 1 );
		grazingData = await renderer.readRenderTargetPixelsAsync( grazingTarget, 0, 0, 128, 16 );

	} finally {

		refracted.dispose();
		grazingMap.dispose();
		airTarget.dispose();
		output.dispose();
		diagnosticTarget.dispose();
		grazingTarget.dispose();
		material.dispose();
		diagnosticMaterial.dispose();
		grazingMaterial.dispose();
		airMaterial.dispose();
		placeholder.dispose();
		floor.geometry.dispose();
		airBox.geometry.dispose();
		wetBox.geometry.dispose();
		grazingPlane.geometry.dispose();
		await renderer.setAnimationLoop( animation );

	}
	const waterSlope = waterSun.x / waterSun.y, airSlope = sun.x / sun.y;
	const expectedAir = [ - 1 - 3 * airSlope + floorY * waterSlope, 1 + floorY * waterSlope ];
	const expectedWet = [ 6 - ( - 1 - floorY ) * waterSlope, 8 - ( - 2 - floorY ) * waterSlope ];
	const step = 2 * extent / pixels;
	const inspect = ( channel: number, expected: number[] ) => {

		const covered: number[] = [];
		let mismatchedInteriorPixels = 0;
		for ( let i = 0; i < pixels; i ++ ) {

			const x = - extent + ( i + 0.5 ) * step;
			const blocked = data[ i * 4 + channel ] < 0.5;
			if ( blocked ) covered.push( x );
			const expectedBlocked = x > expected[ 0 ] && x < expected[ 1 ];
			if ( blocked !== expectedBlocked && Math.min( Math.abs( x - expected[ 0 ] ), Math.abs( x - expected[ 1 ] ) ) > step ) mismatchedInteriorPixels ++;

		}
		const measured = covered.length ? [ covered[ 0 ] - step / 2, covered[ covered.length - 1 ] + step / 2 ] : [ NaN, NaN ];
		return { expectedSpan: expected, measuredSpan: measured,
			maximumEdgeError: Math.max( Math.abs( measured[ 0 ] - expected[ 0 ] ), Math.abs( measured[ 1 ] - expected[ 1 ] ) ), mismatchedInteriorPixels };

	};
	const air = inspect( 0, expectedAir ), submerged = inspect( 1, expectedWet );
	const measured = data[ 3 ], relativeError = Math.abs( measured - sun.y ) / sun.y;
	const boxDepth = ( point: THREE.Vector3, direction: THREE.Vector3, box: THREE.Box3, matrix: THREE.Matrix4 ) => {

		const ray = new THREE.Ray( point.clone().addScaledVector( direction, 1000 ), direction.clone().negate() );
		const hit = ray.intersectBox( box, new THREE.Vector3() );
		return hit ? hit.applyMatrix4( matrix ).z : 1;

	};
	const airBounds = new THREE.Box3().setFromObject( airBox ), wetBounds = new THREE.Box3().setFromObject( wetBox );
	const fixtureInputs = [
		{ role: 'air-blocked', expectedDepth: boxDepth( refractedSurfaceEntry( new THREE.Vector3( - 6, floorY, 0 ), sun, 0 ), sun, airBounds, airMatrix.value ), measuredDepth: diagnosticData[ 0 ] },
		{ role: 'air-lit', expectedDepth: 1, measuredDepth: diagnosticData[ 4 ] },
		{ role: 'submerged-blocked', expectedDepth: boxDepth( new THREE.Vector3( 5, floorY, 0 ), waterSun, wetBounds, refracted.matrix.value ), measuredDepth: diagnosticData[ 9 ] },
		{ role: 'submerged-lit', expectedDepth: new THREE.Vector3( 0, floorY, 0 ).applyMatrix4( refracted.matrix.value ).z, measuredDepth: diagnosticData[ 13 ] },
	];
	const fixture = fixtureInputs.map( ( sample ) => ( { ...sample, absoluteError: Math.abs( sample.measuredDepth - sample.expectedDepth ) } ) );
	let minimumVisibility = 1, unoffsetMinimumVisibility = 1;
	for ( let i = 0; i < 128 * 16; i ++ ) {

		minimumVisibility = Math.min( minimumVisibility, grazingData[ i * 4 ] );
		unoffsetMinimumVisibility = Math.min( unoffsetMinimumVisibility, grazingData[ i * 4 + 1 ] );

	}
	const grazing = { normalLightCosine: grazingNormal.dot( waterSun ), minimumVisibility,
		maximumDarkening: 1 - minimumVisibility, unoffsetMinimumVisibility, sampledPixels: 128 * 16 };
	return {
		passed: air.mismatchedInteriorPixels === 0 && submerged.mismatchedInteriorPixels === 0
			&& air.maximumEdgeError < step * 1.5 && submerged.maximumEdgeError < step * 1.5 && relativeError < 1e-5
			&& fixture.every( ( sample ) => Number.isFinite( sample.measuredDepth ) && sample.absoluteError < 0.004 )
			&& Number.isFinite( minimumVisibility ) && minimumVisibility > 0.999
			// without the normal offset the hardware comparisons shade the grazing plane: the
			// fixture exercises the footprint's rise that the offset clears
			&& Number.isFinite( unoffsetMinimumVisibility ) && unoffsetMinimumVisibility < 0.95,
		pixels, air, submerged, horizontalIrradiance: { expected: sun.y, measured, relativeError },
		refractionShortening: - floorY * ( airSlope - waterSlope ), fixture, grazingPlane: grazing,
	};

}
