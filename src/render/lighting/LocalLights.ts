import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, int, uint, float, uniform, storage, instanceIndex, vec3, vec4, dot, min, max, clamp, floor, exp, log, select,
	inverseSqrt, positionWorld, cameraViewMatrix,
} from 'three/tsl';
import type { FrameGraph } from '../post/graph/FrameGraph.ts';
import { GRID, CLUSTERS, PER_CLUSTER, CLUSTER_WORDS, GUARD, slicesFor, sortInView } from './LocalLightClusters.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { meterKey } from './MeterLight.ts';
import type { Look } from '../look/Look.ts';
import { checkDeclarations, type Declarations } from '../Extensions.ts';
import { onSceneDispose } from '../SceneDispose.ts';

// Local lights: small light sources on the island, every one in view, by clustered shading.
//
// Local lights: small light sources on the island (by night the yunpeng's clumps, Yunpeng.ts) that
// light the ground and the plants round them, every source in view at once, by clustered forward
// shading (Olsson, Billeter and Assarsson, "Clustered Deferred and Forward Shading", HPG 2012; the
// mechanism as three r186's ClusteredLightsNode has it: a CPU sort by view depth with per-slice
// ranges, one thread per cluster testing spheres against the cluster's box, no atomics; written
// anew here, none of its code). No light object per source: three would unroll every light into
// every lit material, multiplying the shaders and their compile time, and a source coming or going
// would build pipelines. No pipeline depends on the sources, their count or the view. (The sources
// as the project's own list, evaluated by every lit material in a loop, inverse-square with the
// window below and no shadows: the technique of tidewater's `src/materials/LocalLights.js`,
// https://github.com/dgreenheck/tidewater, used without its code.)
//
// The light of one source at a receiving point, d from it, on a surface of normal N:
//   E = I max( N.L, 0 ) ( 1 - ( d / r )^4 )^2 / ( d^2 + SOFT )
// with I its intensity (rgb, the illuminance at 1 m), r its range: the inverse-square law, a window
// that brings it smoothly to zero at r, and SOFT (1 m^2) standing for the source's size, so a point
// close to a clump of glowing stalks is not lit as if by a point (the window and the + 1 of Karis,
// "Real Shading in Unreal Engine 4", SIGGRAPH 2013). Diffuse only, no shadows, no specular: the
// sources are dim and broad, a highlight of one would read as a point. Each source's light is
// shaded as the model shades its key's diffuse: the loop hands the model's `shade` the light's true
// direction L (view space; a low key's lift is the sun's and the moon's alone) and its light
// without the cosine, I W / ( d^2 + SOFT ), and the model returns what the surface reflects (on its
// albedo) and its metering illuminance. The physical models take Lambert under 1 - lum( F( V.H ) )
// (lighting/GroundBounce.ts `diffuseFresnel`), a plant its wrap, its light through and its thin
// layer (vegetation/VegetationLighting.ts), an islander its plain Lambert. The sum takes the occlusion
// the model's ambient diffuse takes (lighting/AmbientGrade.ts `diffuseOcclusion`: the material's,
// times the screen's where it receives it; an islander's own), so a pool shows the contact shadow in
// cracks, at blade roots and culm feet, and whatever dims the model's key (the water column's
// attenuation: lighting/Underwater.ts); the vegetation's occlusion from root to tip
// (VegetationLighting `scale`) dims only the ambient terms, not these lights.
//
// The clusters (their constants and the CPU sort in LocalLightClusters.ts, which node checks:
// tools/render/verify-local-lights.mjs). The main camera's view is cut into GRID.x x GRID.y screen tiles
// (TILE px at the 2K baseline, 2560 x 1440) times GRID.z depth slices; the grid is fixed whatever the render's
// size, so the dynamic resolution never rebuilds the kernel or changes its dispatch. A fragment
// finds its cluster from its world position, through the main camera's view and unjittered
// projection (uniforms set before the scene passes, outside the temporal modes' jittered span),
// never from its screen coordinate: the scene draws at the render scale, and the refraction pass
// (ocean/Refraction.ts) draws through a wider view whose fragments beyond the screen fall in the
// edge tiles, which the kernel widens by the refraction's guard band so their boxes hold them. The
// slices: the first from the camera to FIRST_SLICE, the others exponential from there to the
// farthest light's view depth plus its range (a uniform, per frame: the camera's far is 60 km, and
// slices spread to it would leave half of them past 200 m); a fragment beyond it is lit by no
// source and skips the lookup.
//
// The tables. The sources live in one storage buffer (position and range, intensity), written
// only when the source table changes (a load, a growth, a test flag): checked each frame against
// what was written, it costs nothing otherwise; a table that outgrows it takes a new, larger
// buffer (a storage buffer cannot grow in place), the nodes' `value` swapped, so no pipeline is
// built. Each frame the CPU sorts the sources in view (their spheres against the main view widened
// by the guard band) by view depth, then index (a deterministic order), and writes that order and
// each slice's range in it into a second buffer: the kernel tests only a slice's range. The
// kernel, one compute pass on the frame graph ahead of the scene passes (`localLightsPass`), runs
// one thread per cluster: its box in view space, the slice's range in order, each source's sphere
// against the box, the first PER_CLUSTER that touch it written as 16-bit source indices, two to a
// word, after the count of all that touch it (so an overflow, truncated in depth order, shows in
// `info()`). Storage buffers, not textures: the terrain's fragment stage already binds all of
// WebGPU's 16 sampled textures; the light table and the clusters add two storage buffers to every
// lit material's fragment stage. With no source in view (by day, the strength 0) the pass does not
// run and the count uniform is 0, so the lit materials skip the lookup: a day frame costs what it
// did before the lights.
//
// Receivers: every lit material through its lighting model root (lighting/GroundBounce.ts
// GroundBounceLightingModel: the terrain and the cliffs above and under the water, the stones, every
// vegetation kind with the meadow, the bamboo and the yunpeng, the game's props and the islanders'
// figures). A yunpeng clump takes its own light too: the stalks of a glowing clump do light
// each other, and the near field's SOFT keeps that below the glow's own radiance. The water, the sky,
// the clouds and the haze take none (the haze could later, in its froxels' light).
//
// The result is a function of the view and the source table alone, with no history: a scene jump
// clears nothing. `gain` is the art gain on top of every source's own intensity (the look's, which
// the registry takes where it is made, `createLocalLights`; the GUI's Local lights folder); `flags.localLights = false` (or 0, at load or at run time) turns them off;
// `flags=lightTest:N` (at load) adds N white test sources over the meadow (`testSources`), for cost
// tests only; debug builds read the state as `__PONLAI__.localLights.info()` (a promise: the
// sources, those in view, the fullest cluster and the clusters that overflowed, read back from the
// last pass that ran).
//
// The registry. The sources come from kinds of light (`LocalLightKind`: the yunpeng's clumps, the
// test sources) registered on the render pipeline's `LocalLights` (`createLocalLights`, which the
// render pipeline's stages call once; `add`), read in the order registered and never by name. A
// kind gives its sources now (`sources()`), read when it is registered and again at `refresh()`,
// which whoever moves them calls (the island's growth, after a swap). A kind declares the three
// things every registered definition does (Extensions.ts), or `add` refuses it. A source without an
// intensity or a range of its own is the default, a white point light (DEFAULT_SOURCE). The
// registry is an object of the render pipeline's; the buffers, the uniforms and the kernel above
// stay the module's, since every lit material's shader names them: one light table per page.

/** The light table's first capacity; a larger table takes the next power of two, up to 16 bits. */
const CAPACITY = 1024;
const MAX_SOURCES = 1 << 16;
// the near field's softening (m^2): the size of a source
const SOFT = 1;
// (The art gain over the sources' own intensity is the look's (look/Look.ts `LocalLightsLook`): the
// registry takes its table as an argument where the render pipeline's stage makes it
// (RenderStages.ts `createLight`), so two registries made from two tables hold their own gain
// (tools/render/verify-look-params.mjs), and this module holds none.)

type V3 = THREE.Node<'vec3'>;
type F = THREE.Node<'float'>;
type U = THREE.Node<'uint'>;

// the buffers: the sources ( x, y, z, range ), ( r, g, b, 0 ) per source; the order (each slice's
// range, then the sources in view sorted); the clusters
let capacity = CAPACITY;
const tableAttribute = ( n: number ) => new THREE.StorageBufferAttribute( new Float32Array( n * 8 ), 4 );
const orderAttribute = ( n: number ) => new THREE.StorageBufferAttribute( new Uint32Array( 2 * GRID.z + n ), 1 );
let table = tableAttribute( capacity );
let order = orderAttribute( capacity );
const clusters = new THREE.StorageBufferAttribute( new Uint32Array( CLUSTERS * CLUSTER_WORDS ), 1 );
// the nodes: the kernel's and the materials' (read only), whose `value` a new buffer replaces
const tableRead = storage( table, 'vec4', capacity * 2 ).toReadOnly().setName( 'localLightTable' );
const orderRead = storage( order, 'uint', 2 * GRID.z + capacity ).toReadOnly().setName( 'localLightOrder' );
const clustersWrite = storage( clusters, 'uint', CLUSTERS * CLUSTER_WORDS ).setName( 'localLightClusters' );
const clustersRead = storage( clusters, 'uint', CLUSTERS * CLUSTER_WORDS ).toReadOnly().setName( 'localLightClustersRead' );

// the frame's uniforms, shared by the kernel and the materials
const u = {
	// the sources in view (0: no lookup)
	count: uniform( 0, 'int' ),
	// the strength times the gain, over every source's intensity
	scale: uniform( 0 ),
	// the main camera's view, and its unjittered projection's ( P00, P11, P20, P21 )
	view: uniform( new THREE.Matrix4() ),
	lens: uniform( new THREE.Vector4( 1, 1, 0, 0 ) ),
	// the slices: the first's far end, the last's, and ( GRID.z - 1 ) / ln( far / first )
	slices: uniform( new THREE.Vector3( ...slicesFor( 0 ) ) ),
};

/** The slice of a view depth: 0 up to the first's end, then exponential to the last's. */
function sliceOf( depth: F ): THREE.Node<'int'> {

	const k = floor( log( max( depth, 1e-6 ).div( u.slices.x ) ).mul( u.slices.z ) ).add( 1 );
	return int( select( depth.lessThan( u.slices.x ), float( 0 ), clamp( k, 1, GRID.z - 1 ) ) );

}

/** The kernel: one thread per cluster ( x fastest, then y, then the slice ), its box in view space,
 * its slice's range of the sorted sources tested in order. */
export const localLightsKernel = Fn( () => {

	const i = instanceIndex;
	const cx = i.mod( uint( GRID.x ) ), cy = i.div( uint( GRID.x ) ).mod( uint( GRID.y ) ), cz = i.div( uint( GRID.x * GRID.y ) );
	const fx = float( cx ), fy = float( cy ), fz = float( cz );
	// the tile in NDC, y up (tile row 0 the top), the edge tiles out to the guard band
	const x0 = select( cx.equal( uint( 0 ) ), float( - 1 - GUARD.left ), fx.mul( 2 / GRID.x ).sub( 1 ) );
	const x1 = select( cx.equal( uint( GRID.x - 1 ) ), float( 1 + GUARD.right ), fx.add( 1 ).mul( 2 / GRID.x ).sub( 1 ) );
	const y1 = select( cy.equal( uint( 0 ) ), float( 1 + GUARD.top ), float( 1 ).sub( fy.mul( 2 / GRID.y ) ) );
	const y0 = select( cy.equal( uint( GRID.y - 1 ) ), float( - 1 - GUARD.bottom ), float( 1 ).sub( fy.add( 1 ).mul( 2 / GRID.y ) ) );
	// the slice's depths: slice 0 from the camera, slice s >= 1 from first * ( far / first )^( ( s - 1 ) / ( GRID.z - 1 ) )
	const at = ( s: F ) => u.slices.x.mul( exp( s.sub( 1 ).div( u.slices.z ) ) );
	const near = select( cz.equal( uint( 0 ) ), float( 0 ), at( fz ) );
	const far = at( fz.add( 1 ) );
	// a view-space x at depth d through NDC n: d ( n + P20 ) / P00 (y alike); the box over both depths
	const ax0 = x0.add( u.lens.z ).div( u.lens.x ), ax1 = x1.add( u.lens.z ).div( u.lens.x );
	const ay0 = y0.add( u.lens.w ).div( u.lens.y ), ay1 = y1.add( u.lens.w ).div( u.lens.y );
	const lo = vec3( min( ax0.mul( near ), ax0.mul( far ) ), min( ay0.mul( near ), ay0.mul( far ) ), far.negate() ).toVar();
	const hi = vec3( max( ax1.mul( near ), ax1.mul( far ) ), max( ay1.mul( near ), ay1.mul( far ) ), near.negate() ).toVar();
	const base = i.mul( uint( CLUSTER_WORDS ) ).toVar();
	const n = uint( 0 ).toVar();
	const held = uint( 0 ).toVar();
	const start = orderRead.element( cz.mul( uint( 2 ) ) ).toVar(), end = orderRead.element( cz.mul( uint( 2 ) ).add( uint( 1 ) ) ).toVar();
	Loop( { start, end, type: 'uint', condition: '<' }, ( { i: k } ) => {

		const source = orderRead.element( uint( 2 * GRID.z ).add( k as unknown as U ) ).toVar();
		const light = tableRead.element( source.mul( uint( 2 ) ) ).toVar();
		const p = u.view.mul( vec4( light.xyz, 1 ) ).xyz.toVar();
		const off = p.sub( clamp( p, lo, hi ) );
		If( dot( off, off ).lessThanEqual( light.w.mul( light.w ) ), () => {

			If( n.lessThan( uint( PER_CLUSTER ) ), () => {

				// two to a word: the even one held, written with the odd one
				If( n.bitAnd( uint( 1 ) ).equal( uint( 0 ) ), () => {

					held.assign( source );

				} ).Else( () => {

					clustersWrite.element( base.add( uint( 1 ) ).add( n.shiftRight( uint( 1 ) ) ) ).assign( held.bitOr( source.shiftLeft( uint( 16 ) ) ) );

				} );

			} );
			n.addAssign( uint( 1 ) );

		} );

	} );
	// a last even one alone (PER_CLUSTER is even, so a full list has none)
	If( n.lessThan( uint( PER_CLUSTER ) ).and( n.bitAnd( uint( 1 ) ).equal( uint( 1 ) ) ), () => {

		clustersWrite.element( base.add( uint( 1 ) ).add( n.shiftRight( uint( 1 ) ) ) ).assign( held );

	} );
	clustersWrite.element( base ).assign( n );

} )().compute( CLUSTERS, [ 64 ] ).setName( 'local light clusters' );

/** How a lighting model shades one light: from the view-space direction to the light, L (unit),
 * and the light without the cosine (rgb), what the surface reflects (on its albedo, before 1 / pi)
 * and the light's metering illuminance (lighting/MeterLight.ts). */
export type LocalShade = ( L: V3, light: V3 ) => { reflected: V3; meter: V3 };

/** The lights of the fragment's cluster shaded by the model's `shade`, added to its direct diffuse
 * and the metering illuminance's key, both times `weight` (the occlusion and what dims the key):
 * called from the model's `indirect` (three has run the direct lights by then), once per material.
 * Inline, not a `setLayout` function: three r186 emits a laid-out function once and reuses its code
 * in later materials with the uniform names of the first one's build (the count read as another
 * material's `object.nodeUniform0`, which failed to compile); it runs in the fragment stage only.
 * `weight` is built before the loop (a `toVar`), so nothing it samples is taken inside the branch. */
export function shadeLocalLights( builder: THREE.NodeBuilder, shade: LocalShade, weight?: F | V3 ): void {

	const reflected = ( builder.context as unknown as { reflectedLight: { directDiffuse: V3 } } ).reflectedLight;
	const sum = Fn( () => {

		const total = vec3( 0 ).toVar();
		// (no texture is sampled inside, so the branches may diverge)
		If( u.count.greaterThan( 0 ), () => {

			const P = positionWorld;
			const v = u.view.mul( vec4( P, 1 ) ).xyz.toVar();
			const depth = v.z.negate().toVar();
			If( depth.lessThan( u.slices.y ), () => {

				// the tile through the unjittered projection, clamped to the grid (a fragment beyond
				// the screen, the refraction's guard band, falls in an edge tile, whose box holds it)
				const d = max( depth, 1e-6 );
				const nx = v.x.mul( u.lens.x ).div( d ).sub( u.lens.z ), ny = v.y.mul( u.lens.y ).div( d ).sub( u.lens.w );
				const tx = int( clamp( floor( nx.mul( 0.5 ).add( 0.5 ).mul( GRID.x ) ), 0, GRID.x - 1 ) );
				const ty = int( clamp( floor( ny.mul( - 0.5 ).add( 0.5 ).mul( GRID.y ) ), 0, GRID.y - 1 ) );
				const cluster = tx.add( ty.mul( GRID.x ) ).add( sliceOf( depth ).mul( GRID.x * GRID.y ) );
				const base = cluster.mul( CLUSTER_WORDS ).toVar();
				const touching = clustersRead.element( base ).toVar();
				const n = int( select( touching.lessThan( uint( PER_CLUSTER ) ), touching, uint( PER_CLUSTER ) ) ).toVar();
				Loop( { start: int( 0 ), end: n, type: 'int', condition: '<' }, ( { i } ) => {

					const word = clustersRead.element( base.add( 1 ).add( i.shiftRight( 1 ) ) );
					const source = int( word.shiftRight( uint( i.bitAnd( 1 ).mul( 16 ) ) ).bitAnd( uint( 0xffff ) ) ).toVar();
					const light = tableRead.element( source.mul( 2 ) );
					const colour = tableRead.element( source.mul( 2 ).add( 1 ) );
					const d = light.xyz.sub( P ).toVar();
					const d2 = dot( d, d ).toVar();
					const x = d2.div( light.w.mul( light.w ) );
					const window = max( x.mul( x ).oneMinus(), 0 );
					const L = cameraViewMatrix.mul( vec4( d.mul( inverseSqrt( max( d2, 1e-6 ) ) ), 0 ) ).xyz.toVar();
					const s = shade( L, colour.xyz.mul( u.scale ).mul( window.mul( window ).div( d2.add( SOFT ) ) ) );
					total.addAssign( s.reflected );
					// their illuminance joins the key's in the metering illuminance (lighting/MeterLight.ts)
					meterKey.addAssign( weight ? s.meter.mul( weight ) : s.meter );

				} );

			} );

		} );
		return total;

	} )() as V3;
	reflected.directDiffuse.addAssign( ( weight ? sum.mul( weight ) : sum ).mul( 1 / Math.PI ) );

}

export interface LocalSource {
	/** world position (m) */
	position: THREE.Vector3;
	/** linear rgb intensity (the illuminance at 1 m) at full strength, before the gain */
	intensity: THREE.Vector3;
	/** where its light ends (m) */
	range: number;
}

/** A source as a kind gives it: without an intensity or a range it takes the default's. */
export type LocalSourceSpec = Pick<LocalSource, 'position'> & Partial<LocalSource>;

/** The default source, a white point light: its intensity (each of r, g, b) and its range (m). */
export const DEFAULT_SOURCE = { intensity: 1, range: 6.5 };

/** A kind of local light: a name, the three declarations and its sources now. */
export interface LocalLightKind extends Declarations {
	/** 'yunpeng', ... */
	name: string;
	/** the kind's sources as they stand; read when the kind is registered and at every `refresh` */
	sources(): readonly LocalSourceSpec[];
}

const off = () => debugFlags.localLights === false || debugFlags.localLights === 0;
const _p = new THREE.Vector3();
// the sources' view-space positions and ranges, each frame
let view = new Float64Array( CAPACITY * 4 );
// what the table holds (the sources it was written from: count, then 7 numbers each)
let written: number[] = [ 0 ];

/** The CPU side: the kinds of light and their sources, the tables written from them and the
 * frame's sort. */
export class LocalLights {

	/** the kinds, in the order registered */
	readonly kinds: LocalLightKind[] = [];
	/** every kind's sources, in the kinds' order (`refresh`) */
	sources: LocalSource[] = [];
	/** the art gain (the look's at first), the GUI's */
	gain: number;
	/** off: no light (the GUI's; `flags.localLights = false` too) */
	enabled = true;
	/** the sources' share of their light now (0 by day) */
	strength = 0;
	/** the renderer, for `info()` (the pass sets it) */
	renderer: THREE.WebGPURenderer | null = null;
	/** `look`: the look's tables the registry takes its gain from. */
	constructor( look: LocalLightsTables ) {

		this.gain = look.localLights.gain;

	}
	/** Registers a kind of light and takes its sources; refused without its three declarations. */
	add( kind: LocalLightKind ): void {

		checkDeclarations( 'LocalLights', kind.name, kind );
		this.kinds.push( kind );
		this.refresh();

	}
	/** Reads every kind's sources again, in the kinds' order: called by whoever moved them. */
	refresh(): void {

		this.sources = this.kinds.flatMap( ( kind ) => kind.sources().map( ( s ) => s.intensity && s.range !== undefined ? s as LocalSource : {
			position: s.position,
			intensity: s.intensity ?? new THREE.Vector3( DEFAULT_SOURCE.intensity, DEFAULT_SOURCE.intensity, DEFAULT_SOURCE.intensity ),
			range: s.range ?? DEFAULT_SOURCE.range,
		} ) );

	}
	/** The pass runs this frame: lit, on and with sources. */
	get active(): boolean {

		return this.strength > 0 && this.enabled && ! off() && this.sources.length > 0;

	}
	/** Once a frame, before it renders: the sources' share of their light (0 by day). The table is
	 * rewritten when the sources changed; the sort waits for the pass (`cluster`). */
	update( strength: number ): void {

		this.strength = strength;
		u.scale.value = strength * this.gain;
		u.count.value = 0;
		if ( this.active ) this.writeTable();

	}
	/** The source table into its buffer, if it differs from what the buffer holds. */
	writeTable(): void {

		const sources = this.sources.length > MAX_SOURCES ? this.sources.slice( 0, MAX_SOURCES ) : this.sources;
		let same = written[ 0 ] === sources.length;
		for ( let i = 0; same && i < sources.length; i ++ ) {

			const s = sources[ i ], k = 1 + i * 7;
			same = written[ k ] === s.position.x && written[ k + 1 ] === s.position.y && written[ k + 2 ] === s.position.z && written[ k + 3 ] === s.range
				&& written[ k + 4 ] === s.intensity.x && written[ k + 5 ] === s.intensity.y && written[ k + 6 ] === s.intensity.z;

		}
		if ( same ) return;
		if ( sources.length > capacity ) {

			// a new, larger pair of buffers (a storage buffer cannot grow in place): new bind groups, no pipeline
			while ( capacity < sources.length ) capacity *= 2;
			table = tableAttribute( capacity );
			order = orderAttribute( capacity );
			tableRead.value = table;
			orderRead.value = order;

		}
		const data = table.array as Float32Array;
		written = [ sources.length ];
		sources.forEach( ( s, i ) => {

			data.set( [ s.position.x, s.position.y, s.position.z, s.range, s.intensity.x, s.intensity.y, s.intensity.z, 0 ], i * 8 );
			written.push( s.position.x, s.position.y, s.position.z, s.range, s.intensity.x, s.intensity.y, s.intensity.z );

		} );
		table.clearUpdateRanges();
		table.addUpdateRange( 0, Math.max( sources.length, 1 ) * 8 );
		table.needsUpdate = true;

	}
	/** Before the scene passes, outside the jittered span: the camera's view and projection, the
	 * sources in view sorted by depth and each slice's range in that order. Returns how many are in
	 * view (0: nothing to dispatch). */
	cluster( camera: THREE.PerspectiveCamera ): number {

		camera.updateMatrixWorld();
		const V = camera.matrixWorldInverse, e = camera.projectionMatrix.elements;
		u.view.value.copy( V );
		u.lens.value.set( e[ 0 ], e[ 5 ], e[ 8 ], e[ 9 ] );
		const n = written[ 0 ];
		if ( view.length < n * 4 ) view = new Float64Array( capacity * 4 );
		for ( let i = 0; i < n; i ++ ) {

			const k = 1 + i * 7;
			_p.set( written[ k ], written[ k + 1 ], written[ k + 2 ] ).applyMatrix4( V );
			view[ i * 4 ] = _p.x; view[ i * 4 + 1 ] = _p.y; view[ i * 4 + 2 ] = _p.z; view[ i * 4 + 3 ] = written[ k + 3 ];

		}
		const { order: sorted, ranges, slices } = sortInView( view, n, e, camera.near );
		u.slices.value.fromArray( slices );
		const words = order.array as Uint32Array;
		words.set( ranges, 0 );
		words.set( sorted, 2 * GRID.z );
		order.clearUpdateRanges();
		order.addUpdateRange( 0, 2 * GRID.z + sorted.length );
		order.needsUpdate = true;
		u.count.value = sorted.length;
		return sorted.length;

	}
	/** (debug) The sources, those in view this frame, and from the clusters as the last pass left
	 * them (read back) the fullest cluster's count and how many clusters overflowed PER_CLUSTER. */
	async info(): Promise<{ sources: number; inView: number; fullest: number; overflowed: number; perCluster: number; clusterBytes: number; tableBytes: number }> {

		let fullest = 0, overflowed = 0;
		if ( this.renderer && u.count.value > 0 ) {

			const words = new Uint32Array( await this.renderer.getArrayBufferAsync( clusters ) );
			for ( let c = 0; c < CLUSTERS; c ++ ) {

				const n = words[ c * CLUSTER_WORDS ];
				fullest = Math.max( fullest, n );
				if ( n > PER_CLUSTER ) overflowed ++;

			}

		}
		return {
			sources: this.sources.length, inView: u.count.value, fullest, overflowed, perCluster: PER_CLUSTER,
			clusterBytes: clusters.array.byteLength, tableBytes: table.array.byteLength + order.array.byteLength,
		};

	}

}

// the app's registry, which the pass below reads (the pipeline declares the pass without it)
let installed: LocalLights | null = null;

/** The look's tables the local lights take (`LocalLights`, `createLocalLights`). */
export type LocalLightsTables = Pick<Look, 'localLights'>;

/** The render pipeline's local lights, made once a scene from `look`'s tables: the registry of the kinds
 * of light, whose table the module's buffers hold and whose pass (`localLightsPass`) the pipeline
 * declares. */
export function createLocalLights( look: LocalLightsTables ): LocalLights {

	if ( installed ) throw new Error( 'LocalLights: a scene has one light table, and it is already made' );
	installed = new LocalLights( look );
	return installed;

}
onSceneDispose( 'local lights', () => {

	// the registry gone, the buffers as the page loaded them: their first size, empty
	installed = null;
	if ( capacity !== CAPACITY ) {

		capacity = CAPACITY;
		table = tableAttribute( capacity );
		order = orderAttribute( capacity );
		tableRead.value = table;
		orderRead.value = order;

	}
	( table.array as Float32Array ).fill( 0 );
	( order.array as Uint32Array ).fill( 0 );
	( clusters.array as Uint32Array ).fill( 0 );
	written = [ 0 ];

} );

/** `flags=lightTest:N`: N white sources over the meadow for cost tests, at the cells of a 4 m
 * world grid with the lowest hashes among those whose point and its four neighbours 3 m off lie on
 * the land (`inside`), each at a hashed point of its cell, `lift` over the ground (`height`), with
 * the range and the mean luminance of `like` (stand-in yunpeng clumps' sources, Yunpeng.ts
 * `sampleLights`; a range of 6.5 m and an intensity of 1 without them). Loaded once: a growth does not move them. */
export function testSources( n: number, bounds: { x0: number; z0: number; x1: number; z1: number }, inside: ( x: number, z: number ) => boolean, height: ( x: number, z: number ) => number, like: readonly LocalSource[], lift = 0.8 ): LocalSource[] {

	const CELL = 4, MARGIN = 3;
	const hash = ( i: number, j: number, salt: number ) => {

		let h = Math.imul( i, 0x27d4eb2d ) ^ Math.imul( j, 0x165667b1 ) ^ Math.imul( salt, 0x9e3779b9 );
		h = Math.imul( h ^ ( h >>> 15 ), 0x85ebca6b );
		h = Math.imul( h ^ ( h >>> 13 ), 0xc2b2ae35 );
		return ( ( h ^ ( h >>> 16 ) ) >>> 0 ) / 4294967296;

	};
	const cells: { h: number; x: number; z: number }[] = [];
	for ( let j = Math.floor( bounds.z0 / CELL ); j * CELL < bounds.z1; j ++ ) for ( let i = Math.floor( bounds.x0 / CELL ); i * CELL < bounds.x1; i ++ ) {

		const x = ( i + hash( i, j, 1 ) ) * CELL, z = ( j + hash( i, j, 2 ) ) * CELL;
		if ( inside( x, z ) && inside( x + MARGIN, z ) && inside( x - MARGIN, z ) && inside( x, z + MARGIN ) && inside( x, z - MARGIN ) ) cells.push( { h: hash( i, j, 0 ), x, z } );

	}
	cells.sort( ( a, b ) => a.h - b.h || a.x - b.x || a.z - b.z );
	const lum = like.length ? like.reduce( ( t, s ) => t + 0.2126 * s.intensity.x + 0.7152 * s.intensity.y + 0.0722 * s.intensity.z, 0 ) / like.length : 1;
	const range = like.length ? like[ 0 ].range : 6.5;
	return cells.slice( 0, n ).map( ( c ) => ( { position: new THREE.Vector3( c.x, height( c.x, c.z ) + lift, c.z ), intensity: new THREE.Vector3( lum, lum, lum ), range } ) );

}

/** The clusters' compute pass, declared ahead of every scene pass (the light table and the
 * clusters are storage buffers, which the graph does not track as resources: the pass is ordered
 * by its declaration and never culled, as the falling leaves' are). Fixed in size (the grid at
 * the 2K baseline), so the dynamic resolution builds nothing; it runs only while the lights are
 * active, and dispatches nothing when no source is in view. */
export function localLightsPass<Frame extends { camera: THREE.Camera }>( g: FrameGraph<Frame> ): void {

	g.addPass( 'localLights', 'compute', () => ( {
		label: 'local lights', size: { of: 'fixed', width: CLUSTERS, height: 1 }, neverCull: true,
		when: () => installed?.active === true, whenLabel: 'local lights',
		compute: localLightsKernel,
		run: ( f, graph ) => {

			const lights = installed!;
			lights.renderer = graph.renderer;
			if ( lights.cluster( f.camera as THREE.PerspectiveCamera ) > 0 ) graph.renderer.compute( localLightsKernel );

		},
	} ) );

}
