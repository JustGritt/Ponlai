import * as THREE from 'three/webgpu';
import { Fn, If, Loop, texture, textureLoad, vec2, vec4, float, int, ivec2, floor, exp, exp2, log2, max, min, clamp, select, sqrt, screenCoordinate, luminance, mix, uniform } from 'three/tsl';
import type { FrameGraph, GraphResources, ResourceHandle } from './graph/FrameGraph.ts';
import type { PassBuilder } from './graph/GraphCompiler.ts';
import { keyed, type DayKeys } from './DayKeys.ts';
import { tuneKeys } from '../debug/Tune.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// Local tone mapping: a gain on the frame's large-scale brightness that leaves detail alone.
//
// Local tone mapping: a gain on the frame's large-scale brightness that leaves its detail alone,
// after Patry, "Real-Time Samurai Cinema" (SIGGRAPH 2021), on the bilateral grid of Chen, Paris and
// Durand, "Real-time Edge-Aware Image Processing with the Bilateral Grid" (SIGGRAPH 2007). Large
// regions are drawn toward each other (the sky and the sunlit sand against the shadows) while every
// local texture keeps its full contrast. In stops of the base B (log2 of the exposed luminance's
// large-scale level over mid grey, 0.18, below) over a hinge h (stops over mid grey), b = B - h,
// the pixel's colour is multiplied by 2^G( b ), a smooth hinge on a hyperbola:
//   G( b ) = s_lo b + ( s_hi - s_lo ) ( b + sqrt( b^2 + w^2 ) ) / 2
// a slope s_lo below the hinge, s_hi above it and a smooth knee about w stops wide between
// ( ( b + sqrt( b^2 + w^2 ) ) / 2 tends to b above the hinge and to 0 below it, and is w / 2 at the
// hinge itself). The gain is a function of the base
// alone: a bright pixel inside a dark region keeps its offset from its base, and nothing local
// loses contrast, as a global curve's would. In every keyframe (LOCAL_TONE, DayKeys.ts) s_hi is a
// negative slope, small but for the night's (at its bound, -0.6), so everything brighter than the
// hinge (mid grey itself) is drawn down a little per stop; s_lo is keyed either way: negative under
// the low and the high sun, where the shadows are lifted toward the hinge (a negative slope times a
// negative b), positive in the golden hour, under cloud and at night, where they are pressed down.
// s_hi = s_lo = 0 turns it off, which any keyframe may do, and is where the uniforms start. The
// shadows' contrast (look/DisplayLook.ts `shadowContrast`) reads its pivot after this gain.
// The base is 40 % the bilateral grid's value at the pixel and 60 % a wide Gaussian blur of the
// frame (B-spline upsampled), the blend of the talk above: the grid alone rings on smooth gradients
// (clouds, soft shadows, highlights), the blur alone haloes at edges.
// The grid holds log2 of the exposed luminance over mid grey (stops), as homogeneous sums
// ( sum w V, sum w ), each normalised by a cell's full weight so that a full cell's sums are about
// ( V, 1 ).
// Four passes, every target of a fixed size, so the dynamic resolution never rebuilds them:
// - `luma`: the exposed image's log2 luminance (over mid grey) at LUMA (eight by eight samples a grid
//   cell), a bilinear tap a sample, a 2x2 block of them in a texel (PACK);
// - `grid`: the grid, GRID cells over the frame by GRID.depth slices of log luminance (RANGE, the
//   ends clamped), in an atlas of the slices stacked in rows, with one slice more, the cells' plain
//   means (no luminance axis), for the wide blur, and one row more for the frame's mean. Each
//   sample splats with trilinear weights: across the frame into the 2x2 cells whose centres lie
//   within a cell of it, each by its tent in x times its tent in y (so a sample moving across a cell
//   border hands its weight over smoothly, not at once), and along the luminance axis into the two
//   slices nearest its value (a Gaussian splat along the axis smeared a bright pixel's value into
//   slices well below it); a [ 1, 2, 1 ] / 4 blur along the luminance axis follows, the smallest
//   that removes the slices' aliasing. All of it is linear, so the gather does it at once: a cell
//   weights each sample of the 2x2 tiles about its centre by the two spatial tents and the
//   luminance tent convolved with [ 1, 2, 1 ] / 4, with nothing beyond the range's ends (the
//   samples clamped into it: one at an end keeps three quarters of its weight, the blur's share
//   past the end dropped) or the frame's edges;
// - `gridX`, `gridY`: the Gaussian blur across the cells, a separable pair within each slice. The
//   grid's is isotropic in pixels, so a halo about an edge is round on screen: sigma 1.5 cells
//   across, the same width in pixels down (BLUR.grid times a cell's width over its height, from
//   the render size: cells are taller than wide at 16:9). The wide slice takes a radius of 13
//   cells, the talk's (BLUR.wide, in cells on both axes). On the mean's row `gridX` takes each
//   column's mean of the unblurred wide slice and `gridY` the mean of those: the frame's;
// - and in the grade (Pipeline.ts), `apply`: the grid read at the pixel and its own luminance,
//   the eight cells about it (four in each of the two slices on either side, eight loads) each
//   normalised on its own and the values interpolated, and the wide slice read by a cubic
//   B-spline (four bilinear taps).
// The weights are kept near one (a sample's sum to one over the cells and the slices, 0.75 for one
// in the end slices; the blurs' to one), so the half-float sums hold their precision. Each of the
// eight cells is normalised before the slicing interpolates (its sum over its weight). An empty
// one (its weight well under EMPTY) takes the mean of those of the eight that hold values instead,
// and where none does, the frame's mean, each blended in by the cell's weight against EMPTY, so the
// base has no step where a cell empties: the pixel's own luminance never enters its base, so an
// isolated pixel is compressed with the cells about it, not made its own base.
// It reads nothing from an earlier frame: a function of the frame, so a paused frame is bit-identical.

const GRID = { width: 64, height: 32, depth: 64 };
const TILE = 8;
const LUMA = { width: GRID.width * TILE, height: GRID.height * TILE };
// the luma target packs a 2x2 block of samples in a texel's channels (x, y, z, w: the block's
// top left, top right, bottom left, bottom right), so a cell takes its tile's 64 samples in 16
// loads and the luma pass reads neighbouring pixels together
const PACK = 2;
// the atlas: the slices, then the wide slice, each GRID.height rows, then the frame's mean's row
const ATLAS = { width: GRID.width, height: GRID.height * ( GRID.depth + 1 ) + 1 };
const MEAN_ROW = ATLAS.height - 1;
// mid grey of the exposed image (the grade's and the tone curve's input): the grid's
// values are log2 luminance over it
const MID = Math.log2( 0.18 );
// the slices' range, in stops about mid grey, and their spacing
const RANGE = { low: - 10, high: 6 };
const SLICE = ( RANGE.high - RANGE.low ) / ( GRID.depth - 1 );
// the blurs across the cells (sigma in cells, the grid's across, its sigma down following the cells'
// aspect; taps out to `radius` either side)
const BLUR = { grid: 1.5, wide: 13 / 3, radius: 13 };
// the bilateral share of the base
const BILATERAL = 0.4;
// a cell whose weights sum to well under this takes its neighbours' mean (see the header)
const EMPTY = 1e-3;
// the share, in full cells, under which the neighbours' mean gives way to the frame's
const NONE = 1e-3;

/** The smooth hinge's keys by the sun's elevation and at night (DayKeys.ts): the hinge h (stops
 * over the exposed mid grey, 0.18: 0 is mid grey itself), the slopes above it (`high`, s_hi) and
 * below it (`low`, s_lo), in stops of gain per stop of the base, and the knee's width (`width`,
 * w, stops). The width starts at half a stop in every key. The slopes are set per key, and the
 * golden hour's, the low sun's and the night's widths too (the hinge stays where it started): the
 * golden hour's knee is more than a stop wide, the low sun's at its bound, a sharp turn. The
 * night's `high` and `width` sit at their bounds: a knee two stops wide that presses the moonlit
 * highlights down. */
export interface LocalToneKey { [ key: string ]: number; hinge: number; high: number; low: number; width: number }
export const LOCAL_TONE: { keys: DayKeys<LocalToneKey> } = {
	// the exposure's keys (look/ExposureLook.ts `keys`); tunable on debug pages
	// (debug/Tune.ts): `post.localTone.<value>@<key>`
	keys: tuneKeys( 'post.localTone', {
		day: [
			{ at: 3, name: 'golden', values: { hinge: 0, high: - 0.201, low: 0.183, width: 1.04 } },
			{ at: 12, name: 'low', values: { hinge: 0, high: - 0.224, low: - 0.206, width: 0.05 } },
			{ at: 40, name: 'high', values: { hinge: 0, high: - 0.192, low: - 0.0355, width: 0.5 } },
		],
		cloudy: { hinge: 0, high: - 0.23, low: 0.216, width: 0.5 },
		night: { hinge: 0, high: - 0.6, low: 0.168, width: 2 },
	}, { hinge: { min: - 3, max: 2 }, high: { min: - 0.6, max: 0.2 }, low: { min: - 0.4, max: 0.4 }, width: { min: 0.05, max: 2 } } ),
};

export class LocalTone {

	// (neutral until the first update: both slopes 0)
	readonly uniforms = { hinge: uniform( 0 ), high: uniform( 0 ), low: uniform( 0 ), width: uniform( 0.5 ) };
	private readonly keyedValues: LocalToneKey = { hinge: 0, high: 0, low: 0, width: 0.5 };
	private luma: ResourceHandle;
	private grid: ResourceHandle;
	private gridX: ResourceHandle;
	private gridXY: ResourceHandle;
	private readonly quads: THREE.QuadMesh[];
	private readonly tGrid: THREE.TextureNode;
	private readonly gridTexture: THREE.Texture;

	/** `source`: the image the grade takes (scene radiance); `exposure` scales it to the exposed
	 * image; `size` is its size in pixels (the cells' aspect). The targets are declared on `graph`. */
	constructor( graph: GraphResources, source: THREE.TextureNode, exposure: F, size: THREE.UniformNode<'vec2', THREE.Vector2> ) {

		const target = ( name: string, width: number, height: number, format: THREE.PixelFormat ) =>
			graph.color( name, { size: { of: 'fixed', width, height }, format, type: THREE.HalfFloatType } );
		this.luma = target( 'localTone.luma', LUMA.width / PACK, LUMA.height / PACK, THREE.RGBAFormat );
		this.grid = target( 'localTone.grid', ATLAS.width, ATLAS.height, THREE.RGFormat );
		this.gridX = target( 'localTone.gridX', ATLAS.width, ATLAS.height, THREE.RGFormat );
		this.gridXY = target( 'localTone.gridXY', ATLAS.width, ATLAS.height, THREE.RGFormat );
		this.gridTexture = graph.texture( this.gridXY );
		this.tGrid = texture( this.gridTexture );
		this.quads = [
			quad( 'LocalTone.Luma', lumaPass( source, exposure ) ),
			quad( 'LocalTone.Grid', gridPass( graph.texture( this.luma ) ) ),
			quad( 'LocalTone.GridX', blurPass( graph.texture( this.grid ), 0, float( BLUR.grid ) ) ),
			// (a cell's width over its height, in pixels: ( W / GRID.width ) / ( H / GRID.height ))
			quad( 'LocalTone.GridY', blurPass( graph.texture( this.gridX ), 1, size.x.div( size.y ).mul( BLUR.grid * GRID.height / GRID.width ) ) ),
		];

	}

	/** Adds the passes to `graph` (the first one's reads of the source declared by `reads`). Returns
	 * the grid's version the grade reads. */
	passes<Frame>( graph: FrameGraph<Frame>, reads: ( b: PassBuilder ) => void ): ResourceHandle {

		const [ lumaQuad, gridQuad, xQuad, yQuad ] = this.quads;
		graph.addPass( 'localToneLuma', 'quad', ( b ) => {

			reads( b );
			this.luma = b.write( this.luma );
			return { label: 'local tone luma', quad: lumaQuad };

		} );
		graph.addPass( 'localToneGrid', 'quad', ( b ) => {

			b.read( this.luma );
			this.grid = b.write( this.grid );
			return { label: 'local tone grid', quad: gridQuad };

		} );
		graph.addPass( 'localToneGridX', 'quad', ( b ) => {

			b.read( this.grid );
			this.gridX = b.write( this.gridX );
			return { label: 'local tone blur', quad: xQuad };

		} );
		graph.addPass( 'localToneGridY', 'quad', ( b ) => {

			b.read( this.gridX );
			this.gridXY = b.write( this.gridXY );
			return { label: 'local tone blur', quad: yQuad };

		} );
		return this.gridXY;

	}

	/** Sets the hinge's keys for the sun's `elevation` (degrees), the night's weight and the
	 * weather's cloud (LOCAL_TONE, DayKeys.ts). */
	update( elevation: number, night: number, cloud: number ): void {

		const k = keyed( LOCAL_TONE.keys, elevation, night, this.keyedValues, cloud );
		this.uniforms.hinge.value = k.hinge;
		this.uniforms.high.value = k.high;
		this.uniforms.low.value = k.low;
		this.uniforms.width.value = k.width;

	}

	/** The exposed colour `c` at screen uv `at`, locally tone mapped. Explicit levels only, so it may
	 * run inside branches. */
	apply( c: V3, at: V2 ): V3 {

		const u = this.uniforms;
		const v = log2( max( luminance( c ), 2 ** - 24 ) ).sub( MID ).toVar();
		// the grid at the pixel and its own luminance: the eight cells about it (two slices, four
		// cells in each), each normalised on its own, then interpolated (trilinear in the values)
		const zf = clamp( v, RANGE.low, RANGE.high ).sub( RANGE.low ).div( SLICE ).toVar();
		const z0 = min( floor( zf ), GRID.depth - 2 ).toVar();
		const st = clamp( at.mul( vec2( GRID.width, GRID.height ) ).sub( 0.5 ), vec2( 0 ), vec2( GRID.width - 1, GRID.height - 1 ) ).toVar();
		const s0 = floor( st ).toVar();
		const i0 = ivec2( s0 ).toVar();
		const i1 = ivec2( min( s0.add( 1 ), vec2( GRID.width - 1, GRID.height - 1 ) ) ).toVar();
		const f = st.sub( s0 ).toVar();
		// each cell's sum over its weight padded by EMPTY ( a ) and its fullness w / ( w + EMPTY ) ( v ):
		// a full cell's a is its value and its v 1, an empty one's both 0
		const cells: { a: F; v: F }[] = [];
		for ( const dz of [ 0, 1 ] ) for ( const y of [ i0.y, i1.y ] ) for ( const x of [ i0.x, i1.x ] ) {

			const g = ( textureLoad( this.gridTexture, ivec2( x, y.add( int( z0.add( dz ) ).mul( int( GRID.height ) ) ) ) ) as unknown as V4 ).xy.toVar();
			const pad = g.y.add( EMPTY ).toVar();
			cells.push( { a: g.x.div( pad ).toVar(), v: g.y.div( pad ).toVar() } );

		}
		// the full cells' mean, the frame's mean ( MEAN_ROW ) weighing as NONE of a full cell, and each
		// cell's value: its own, blended into that mean as it empties, so the base has no step there
		const mean = ( textureLoad( this.gridTexture, ivec2( 0, MEAN_ROW ) ) as unknown as V4 ).xy.toVar();
		const fill = cells.reduce<F>( ( sum, c ) => sum.add( c.a ), mean.x.div( max( mean.y, 1e-6 ) ).mul( NONE ) )
			.div( cells.reduce<F>( ( sum, c ) => sum.add( c.v ), float( NONE ) ) ).toVar();
		const value = cells.map( ( c ) => c.a.add( c.v.oneMinus().mul( fill ) ) );
		const slice = ( k: number ): F => mix( mix( value[ k ], value[ k + 1 ], f.x ), mix( value[ k + 2 ], value[ k + 3 ], f.x ), f.y );
		const bilateral = mix( slice( 0 ), slice( 4 ), zf.sub( z0 ) );
		const wide = this.wide( at );
		const base = mix( wide, bilateral, BILATERAL ).toVar();
		// the base's stops over the hinge, and the knee: the hyperbola ( b + sqrt( b^2 + w^2 ) ) / 2,
		// whose asymptotes are b above the hinge and 0 below it
		const b = base.sub( u.hinge ).toVar();
		const knee = b.add( sqrt( b.mul( b ).add( u.width.mul( u.width ) ) ) ).mul( 0.5 );
		const gain = u.low.mul( b ).add( u.high.sub( u.low ).mul( knee ) );
		return c.mul( exp2( gain ) );

	}

	// the wide slice at screen uv `at`: a cubic B-spline over its cells, four bilinear taps (rows
	// kept inside the slice)
	private wide( at: V2 ): F {

		const size = vec2( GRID.width, GRID.height );
		const st = at.mul( size ).sub( 0.5 ).toVar();
		const i = floor( st ).toVar();
		const f = st.sub( i ).toVar();
		const f2 = f.mul( f ), f3 = f2.mul( f );
		const w0 = float( 1 ).sub( f ).pow( 3 ).div( 6 );
		const w1 = f3.mul( 3 ).sub( f2.mul( 6 ) ).add( 4 ).div( 6 );
		const w2 = f3.mul( - 3 ).add( f2.mul( 3 ) ).add( f.mul( 3 ) ).add( 1 ).div( 6 );
		const w3 = f3.div( 6 );
		const g0 = w0.add( w1 ).toVar(), g1 = w2.add( w3 ).toVar();
		// the taps' positions, in cells (centres at + 0.5)
		const p0 = i.sub( 0.5 ).add( w1.div( g0 ) ).toVar();
		const p1 = i.add( 1.5 ).add( w3.div( g1 ) ).toVar();
		const top = GRID.depth * GRID.height;
		const tap = ( x: F, y: F ): V2 => ( this.tGrid.sample( vec2(
			clamp( x, 0.5, GRID.width - 0.5 ).div( GRID.width ),
			clamp( y, 0.5, GRID.height - 0.5 ).add( top ).div( ATLAS.height ),
		) ).level( float( 0 ) ) as unknown as V4 ).xy;
		const s = tap( p0.x, p0.y ).mul( g0.x.mul( g0.y ) )
			.add( tap( p1.x, p0.y ).mul( g1.x.mul( g0.y ) ) )
			.add( tap( p0.x, p1.y ).mul( g0.x.mul( g1.y ) ) )
			.add( tap( p1.x, p1.y ).mul( g1.x.mul( g1.y ) ) ).toVar();
		return s.x.div( max( s.y, 1e-6 ) );

	}

}

function quad( name: string, fragment: V4 ): THREE.QuadMesh {

	const material = new THREE.NodeMaterial();
	material.name = name;
	material.fragmentNode = fragment;
	const mesh = new THREE.QuadMesh( material );
	mesh.name = name;
	return mesh;

}

// the exposed log2 luminance over mid grey, at LUMA, a 2x2 block a texel (PACK): one bilinear tap
// a sample (a cell's 64 samples are averaged and blurred, so a sparser footprint shows nowhere)
function lumaPass( source: THREE.TextureNode, exposure: F ): V4 {

	return Fn( () => {

		const p = floor( screenCoordinate.xy ).toVar();
		const sample = ( x: number, y: number ): F => {

			const at = vec2( p.x.mul( PACK ).add( x + 0.5 ).div( LUMA.width ), p.y.mul( PACK ).add( y + 0.5 ).div( LUMA.height ) );
			const l = luminance( source.sample( at ).rgb ).mul( exposure );
			return log2( max( l, 2 ** - 24 ) ).sub( MID );

		};
		return vec4( sample( 0, 0 ), sample( 1, 0 ), sample( 0, 1 ), sample( 1, 1 ) );

	} )() as unknown as V4;

}

// a cell of the atlas from the samples of the 2x2 tiles about its centre (see the header)
function gridPass( luma: THREE.Texture ): V4 {

	// the texels a cell reads across and down (two tiles), and the luma target's size in texels
	const SPAN = 2 * TILE / PACK;
	const TEXELS = [ LUMA.width / PACK, LUMA.height / PACK ];
	return Fn( () => {

		const p = ivec2( screenCoordinate.xy ).toVar();
		const z = p.y.div( int( GRID.height ) ).toVar();
		// (the first texel: the cell's own tile's, less half a tile)
		const first = ivec2( p.x, p.y.sub( z.mul( int( GRID.height ) ) ) ).mul( int( TILE / PACK ) ).sub( int( TILE / PACK / 2 ) ).toVar();
		const wide = z.equal( int( GRID.depth ) ).toVar();
		const sum = vec2( 0 ).toVar();
		const tent = ( d: F ): F => max( float( 1 ).sub( d.abs() ), 0 );
		Loop( { start: int( 0 ), end: int( SPAN ), type: 'int' }, { start: int( 0 ), end: int( SPAN ), type: 'int' }, ( { i, j } ) => {

			const t = first.add( ivec2( j, i ) ).toVar();
			const inside = t.x.greaterThanEqual( int( 0 ) ).and( t.y.greaterThanEqual( int( 0 ) ) )
				.and( t.x.lessThan( int( TEXELS[ 0 ] ) ) ).and( t.y.lessThan( int( TEXELS[ 1 ] ) ) );
			const on = select( inside, float( 1 ), float( 0 ) ).toVar();
			const at = ivec2( clamp( vec2( t ), vec2( 0 ), vec2( TEXELS[ 0 ] - 1, TEXELS[ 1 ] - 1 ) ) );
			const four = ( textureLoad( luma, at ) as unknown as V4 ).toVar();
			// (the block's offset from the cell's centre, in samples: PACK j + 0.5 - TILE, and alike down)
			const jx = float( j ).mul( PACK ).add( 0.5 - TILE ).toVar();
			const iy = float( i ).mul( PACK ).add( 0.5 - TILE ).toVar();
			for ( const [ c, ox, oy ] of [ [ 'x', 0, 0 ], [ 'y', 1, 0 ], [ 'z', 0, 1 ], [ 'w', 1, 1 ] ] as const ) {

				const v = four[ c ];
				const across = tent( jx.add( ox ).div( TILE ) ).mul( tent( iy.add( oy ).div( TILE ) ) ).mul( on ).toVar();
				// (the sample's distance from this slice, in slices)
				const d = clamp( v, RANGE.low, RANGE.high ).sub( RANGE.low ).div( SLICE ).sub( float( z ) ).toVar();
				const along = tent( d.add( 1 ) ).add( tent( d ).mul( 2 ) ).add( tent( d.sub( 1 ) ) ).mul( 0.25 );
				const w = select( wide, across, across.mul( along ) );
				sum.addAssign( vec2( v.mul( w ), w ) );

			}

		} );
		// (a sample's weights over the cells and the slices sum to one, three quarters at the range's
		// ends; a full cell's spatial weights sum to TILE^2)
		return vec4( sum.div( TILE * TILE ), 0, 1 );

	} )() as unknown as V4;

}

// the Gaussian across the cells along one axis (0: x, 1: y), within each slice: the taps unrolled,
// the wide slice's weights constant, the grid's of sigma `sigma` (cells) and normalised by their sum
function blurPass( source: THREE.Texture, axis: 0 | 1, sigma: F ): V4 {

	const wide = Array.from( { length: 2 * BLUR.radius + 1 }, ( _, i ) => Math.exp( - 0.5 * ( ( i - BLUR.radius ) / BLUR.wide ) ** 2 ) );
	const wideSum = wide.reduce( ( a, b ) => a + b );
	return Fn( () => {

		const p = ivec2( screenCoordinate.xy ).toVar();
		// (the mean's row blurs as the wide slice's last row, then is replaced below)
		const row = select( p.y.equal( int( MEAN_ROW ) ), int( MEAN_ROW - 1 ), p.y ).toVar();
		const z = row.div( int( GRID.height ) ).toVar();
		const y = row.sub( z.mul( int( GRID.height ) ) ).toVar();
		const isWide = z.equal( int( GRID.depth ) ).toVar();
		const along = ( axis === 0 ? p.x : y ).toVar();
		const length = axis === 0 ? GRID.width : GRID.height;
		const sum = vec2( 0 ).toVar(), total = float( 0 ).toVar();
		const falloff = float( - 0.5 ).div( sigma.mul( sigma ) ).toVar();
		for ( let i = - BLUR.radius; i <= BLUR.radius; i ++ ) {

			const k = i + BLUR.radius;
			// (taps outside the slice hold nothing: the homogeneous sums renormalise themselves)
			const q = along.add( int( i ) );
			const inside = q.greaterThanEqual( int( 0 ) ).and( q.lessThan( int( length ) ) );
			// (clamp on an int: TSL's typings take floats only)
			const c = clamp( q as unknown as F, int( 0 ) as unknown as F, int( length - 1 ) as unknown as F ) as unknown as THREE.Node<'int'>;
			const at = axis === 0 ? ivec2( c, row ) : ivec2( p.x, c.add( z.mul( int( GRID.height ) ) ) );
			const w = select( isWide, float( wide[ k ] / wideSum ), exp( falloff.mul( i * i ) ) ).toVar();
			total.addAssign( w );
			sum.addAssign( ( textureLoad( source, at ) as unknown as V4 ).xy.mul( select( inside, w, float( 0 ) ) ) );

		}
		const out = sum.div( total ).toVar();
		// the mean's row: along x each column's mean of the unblurred wide slice, along y the mean of
		// those columns, so the frame's mean ( V, 1 ) (the apply reads its first texel)
		const count = axis === 0 ? GRID.height : GRID.width;
		If( p.y.equal( int( MEAN_ROW ) ), () => {

			const mean = vec2( 0 ).toVar();
			Loop( { start: int( 0 ), end: int( count ), type: 'int' }, ( { i } ) => {

				const at = axis === 0 ? ivec2( p.x, i.add( int( GRID.depth * GRID.height ) ) ) : ivec2( i, int( MEAN_ROW ) );
				mean.addAssign( ( textureLoad( source, at ) as unknown as V4 ).xy );

			} );
			out.assign( mean.div( count ) );

		} );
		return vec4( out, 0, 1 );

	} )() as unknown as V4;

}
