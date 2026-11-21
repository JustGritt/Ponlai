import * as THREE from 'three/webgpu';
import { Fn, If, Return, float, globalId, int, ivec2, max, min, select, storage, texture, textureLoad, uniform, uniformArray, vec2 } from 'three/tsl';
import type { FrameGraph, ResourceHandle } from './graph/FrameGraph.ts';
import type { PassBuilder } from './graph/GraphCompiler.ts';

type I = THREE.Node<'int'>;
type IV2 = THREE.Node<'ivec2'>;
type V2 = THREE.Node<'vec2'>;
type V4 = THREE.Node<'vec4'>;

// The opaque scene's depth as a pyramid of nearest and farthest depths.
//
// The sea's reflection is traced over it (ocean/ScreenReflection.ts).
//
// A hierarchical depth buffer (Hi-Z: Uludag, "Hi-Z Screen-Space Cone-Traced Reflections", GPU Pro 5,
// 2014): level 0 is the linear depth itself (Pipeline.ts `linearDepth`, its x: sample 0's view z),
// and each level above holds, per cell, the nearest and the farthest view z of the four cells under
// it, so a cell of level l spans 2^l pixels on a side and holds the depth range of everything drawn
// in it. A ray traced over it steps a whole cell at a time where it passes in front of the cell's
// nearest surface or behind its farthest one (by more than a thickness the trace sets), climbs a
// level after each such step and descends where it may meet something, down to level 0: no pixel
// is skipped, so a culm a few pixels wide is found, and a ray that passes behind a surface goes on
// past it. Both depths are kept (a min-max pyramid) so that a ray behind a figure does not crawl
// through every pixel of it: with the nearest alone, every cell holding the figure looks hit.
//
// - The sky (a view z under -9e4: the resolve writes -1e5) holds nothing: it takes no part in a
//   cell's farthest depth, which is NO_SURFACE (positive, in front of the camera) in a cell of sky
//   alone, and its nearest stays -1e5, so no ray ever meets such a cell.
// - Levels: PYRAMID_LEVELS above level 0, each half the one under it rounded up, so cell x of level
//   l covers pixels [ x 2^l, ( x + 1 ) 2^l ), the last one in part, and a cell whose last child lies
//   past its parent level's edge reads the edge child again. At 4K's render (3840 x 2160) the top
//   level's cells are 1024 pixels wide, 4 x 3 of them.
// - Storage: one buffer of vec2 ( nearest, farthest ) per cell, the levels one after the other, a
//   storage buffer rather than a texture, which the sea reflection's trace
//   (ocean/ScreenReflection.ts) reads through a read-only storage node (`reader`). `levels` holds
//   each level's ( offset in cells, width, height, 0 ), level 0's offset unused, set on the CPU
//   from the render's size.
// - The kernels: one per level, each reading the level under it (level 1 the linear depth), in
//   8 x 8 workgroups dispatched by the level's size at run time (`renderer.compute( kernel, [ x, y,
//   1 ] )`), so a step of the high tier's dynamic scale changes the dispatch, not the pipeline (three-behaviour,
//   Compute kernels). Successive dispatches see each other's writes (WebGPU orders storage writes
//   between dispatches).
// - The buffer is sized for the render's size; a larger render takes a new buffer (a storage buffer
//   cannot grow in place: three-behaviour, Storage buffers), which every node over it takes as its
//   value, with new bind groups and no new pipeline.
// - On the graph: a compute pass after the depth resolve that reads `linearDepth`, never culled, and
//   run in its declared order before the water's pass, which reads the buffer (a buffer, not a
//   graph resource, as Metering.ts's).
// Cost: about a third of the render's pixels written once, the first level reading four depth
// texels per cell.

/** The levels above the linear depth's own. */
export const PYRAMID_LEVELS = 10;
/** A cell's farthest depth where it holds only the sky: in front of the camera, so no ray reaches
 * behind it. */
export const NO_SURFACE = 1e30;
/** View z under which the linear depth holds the sky (the resolve writes -1e5). */
export const SKY_Z = - 9e4;
const TILE = 8;

export class DepthPyramid {

	/** Per level 0..PYRAMID_LEVELS: ( offset in cells, width, height, 0 ). */
	readonly levels: THREE.UniformArrayNode<'vec4'>;
	private readonly table: THREE.Vector4[] = [];
	private readonly kernels: THREE.ComputeNode[] = [];
	private readonly nodes: THREE.StorageBufferNode<'vec2'>[] = [];
	private attribute: THREE.StorageBufferAttribute;
	private capacity = 0;
	private size = { width: 0, height: 0 };

	constructor( linearDepth: THREE.Texture ) {

		for ( let l = 0; l <= PYRAMID_LEVELS; l ++ ) this.table.push( new THREE.Vector4() );
		this.levels = uniformArray( this.table, 'vec4' ) as unknown as THREE.UniformArrayNode<'vec4'>;
		this.attribute = this.allocate( 1 );
		const tDepth = texture( linearDepth );
		const cells = this.node();
		for ( let l = 1; l <= PYRAMID_LEVELS; l ++ ) {

			// (plain uniforms over the table's own vectors: a kernel's values are those of its dispatch)
			const level = uniform( this.table[ l ] ), parent = uniform( this.table[ l - 1 ] );
			const kernel = Fn( () => {

				const cell = ( ivec2( globalId.xy as unknown as THREE.Node<'uvec2'> ) as unknown as IV2 ).toVar();
				If( cell.x.greaterThanEqual( int( level.y ) ).or( cell.y.greaterThanEqual( int( level.z ) ) ), () => {

					Return();

				} );
				const last = ivec2( int( parent.y ).sub( int( 1 ) ), int( parent.z ).sub( int( 1 ) ) ).toVar();
				const nearest = float( - NO_SURFACE ).toVar(), farthest = float( NO_SURFACE ).toVar();
				for ( const [ dx, dy ] of [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ] ) {

					// (min over integer vectors: three's typings take float ones only)
					const child = ( min( cell.mul( int( 2 ) ).add( ivec2( dx, dy ) ) as unknown as V2, last as unknown as V2 ) as unknown as IV2 ).toVar();
					if ( l === 1 ) {

						const z = ( textureLoad( tDepth, child ) as unknown as V4 ).x.toVar();
						nearest.assign( max( nearest, z ) );
						farthest.assign( min( farthest, select( z.lessThan( SKY_Z ), float( NO_SURFACE ), z ) ) );

					} else {

						const range = cells.element( cellIndex( parent, child ) ) as unknown as V2;
						nearest.assign( max( nearest, range.x ) );
						farthest.assign( min( farthest, range.y ) );

					}

				}
				cells.element( cellIndex( level, cell ) ).assign( vec2( nearest, farthest ) );

			} )().computeKernel( [ TILE, TILE, 1 ] ).setName( `DepthPyramid.level${ l }` ) as THREE.ComputeNode;
			this.kernels.push( kernel );

		}

	}

	/** The cells, read-only, for a shader that traces over them (with `levels`). */
	reader(): THREE.StorageBufferNode<'vec2'> {

		return this.node().toReadOnly() as unknown as THREE.StorageBufferNode<'vec2'>;

	}

	/** Sets the levels for a render of `width` x `height` pixels, growing the buffer if it is short. */
	resize( width: number, height: number ): void {

		if ( width === this.size.width && height === this.size.height ) return;
		this.size = { width, height };
		let w = width, h = height, offset = 0;
		this.table[ 0 ].set( 0, w, h, 0 );
		for ( let l = 1; l <= PYRAMID_LEVELS; l ++ ) {

			w = Math.ceil( w / 2 );
			h = Math.ceil( h / 2 );
			this.table[ l ].set( offset, w, h, 0 );
			offset += w * h;

		}
		if ( offset > this.capacity ) {

			// (a quarter more, so that the dynamic scale's steps up do not each take a new buffer)
			this.attribute = this.allocate( Math.ceil( offset * 1.25 ) );
			for ( const node of this.nodes ) node.value = this.attribute;

		}

	}

	/** The pyramid's pass, after the depth resolve and before the water, reading `linearDepth`
	 * (`jittered` with the scene half under the temporal modes). */
	pass<Frame>( g: FrameGraph<Frame>, linearDepth: ResourceHandle, jittered: boolean, when?: ( f: Frame ) => boolean ): void {

		const kernels = this.kernels;
		g.onResize( { of: 'render' }, ( width, height ) => this.resize( width, height ) );
		g.addPass( 'depthPyramid', 'compute', ( b: PassBuilder ) => {

			b.read( linearDepth );
			return {
				label: 'depth pyramid', size: { of: 'render' }, neverCull: true, jittered, when, whenLabel: when ? 'water' : undefined, compute: kernels,
				run: ( _f, graph ) => {

					for ( let l = 1; l <= PYRAMID_LEVELS; l ++ ) {

						const v = this.table[ l ];
						graph.renderer.compute( kernels[ l - 1 ], [ Math.ceil( v.y / TILE ), Math.ceil( v.z / TILE ), 1 ] );

					}

				},
			};

		} );

	}

	private allocate( cells: number ): THREE.StorageBufferAttribute {

		this.capacity = cells;
		return new THREE.StorageBufferAttribute( new Float32Array( cells * 2 ), 2 );

	}

	private node(): THREE.StorageBufferNode<'vec2'> {

		const node = storage( this.attribute, 'vec2', this.capacity ) as unknown as THREE.StorageBufferNode<'vec2'>;
		this.nodes.push( node );
		return node;

	}

}

/** The index in the buffer of `cell` in the level whose ( offset, width, height ) `level` holds. */
export function cellIndex( level: V4 | THREE.Node<'vec4'>, cell: IV2 ): I {

	return int( level.x ).add( cell.y.mul( int( level.y ) ) ).add( cell.x );

}
