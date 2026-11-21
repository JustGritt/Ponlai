/*!
 * Adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater): the cirrus
 * fibres of its `Clouds.js` (sparse seeds drawn out along a meandering flow by line integral
 * convolution, hooked tails downwind of tufts) and their footprint-filtered lookup.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, Loop, instanceIndex, textureStore, texture, uv, int, uint, float, vec2, vec3, vec4, ivec2,
	floor, fract, dot, mod, exp, max, min, sin, cos, atan, normalize, clamp, length, log2, select,
} from 'three/tsl';
import { hash33, perlin3 } from '../materials/CloudNoise.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

// The thin cloud layer's fibres, baked once into a tileable texture by line integral convolution.
//
// The thin cloud layer's fibres (Clouds.ts `streaksAt`), baked once into a mipmapped, tileable
// texture, x along the drift. Sparse seeds, gathered in tufts by a fractal noise of 5 cells a tile,
// are drawn out by line integral convolution (Cabral and Leedom 1993, "Imaging Vector Fields Using
// Line Integral Convolution", SIGGRAPH) along a smooth flow that meanders across the drift
// (gradient noise of 3 cells, up to about half a radian off it, plus the curl of a noise of 3, 7
// and 15 cells): each strand fades in and out along its length, so it ends frayed rather than cut.
// A second convolution from the tufts runs downwind of them and sags sideways, drawing hooked
// tails. A fractal noise of 24 cells a tile breaks the strands into pieces of uneven density. The
// sky samples the texture at two scales turned against each other (CloudModel.ts FIBRE_FINE and
// FIBRE_BROAD), which hides its tiling and keeps the fibres from all running parallel.
// Red holds the fibres' density, green the tufts' field, which sets the silk between the fibres
// (`streaksAt`). At 44 steps and seeds of 0.18 cells the strands read overhead as short dashes;
// 72 steps and finer seeds draw them out into fibres.

// The tufts: the fractal noise's gain and offset before it is clamped to [0, 1]. At 2 and 0.75 the
// tufts stood apart and the fibres read as flecks; at these they fill most of a tile, as silk.
const CLUSTER = { gain: 1.6, offset: 0.3 };
// Texels a side, and the convolution's length in texels (a strand runs about this long).
export const FIBRE_RES = 1024;
const LIC_STEPS = 72;

// Tileable gradient noise in [-1, 1] on the unit square, `period` cells across.
const noise2 = ( p: V2, period: number, seed: number ) => perlin3( vec3( p.mul( period ), seed ), float( period ) );
// Fractal noise about [0, 1], with the contrast the clustering and breakup thresholds expect.
const fbm2 = ( p: V2, period: number, seed: number ) => noise2( p, period, seed ).mul( 0.55 )
	.add( noise2( p, period * 2, seed + 1.7 ).mul( 0.3 ) ).add( noise2( p, period * 4, seed + 3.1 ).mul( 0.15 ) )
	.mul( 1.6 ).add( 0.5 );
const sat = ( x: F ) => clamp( x, 0, 1 );
// Where the seeds gather: tufts of fibres with bare sky between them.
const clusterAt = ( p: V2 ) => sat( fbm2( p, 5, 0.9 ).mul( CLUSTER.gain ).sub( CLUSTER.offset ) );

// Sparse jittered points, `n` cells across the texture: each cell holds one with probability
// `cluster * prob`, a Gaussian of `sigma` cells.
const points = ( p: V2, cluster: F, n: number, prob: number, sigma: number, seed: number ) => {

	const q = p.mul( n );
	const ip = floor( q ).toVar(), fp = fract( q ).toVar();
	const v = float( 0 ).toVar();
	for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

		const o = vec2( x, y );
		const h = hash33( vec3( mod( ip.add( o ), float( n ) ), seed ) );
		const on = select( h.z.lessThan( cluster.mul( prob ) ), float( 1 ), float( 0 ) );
		const d = o.add( h.xy.mul( 0.8 ).add( 0.1 ) ).sub( fp );
		const strength = hash33( vec3( mod( ip.add( o ), float( n ) ), seed + 7.1 ) ).x.mul( 0.7 ).add( 0.3 );
		v.assign( max( v, exp( dot( d, d ).mul( - 0.5 / ( sigma * sigma ) ) ).mul( on ).mul( strength ) ) );

	}
	return v;

};

/** Bake the fibres: red their density, green the tufts' field. Needs an initialised renderer. */
export function bakeStreaks( renderer: THREE.WebGPURenderer ): THREE.RenderTarget {

	// The flow: r = seeds, g = the flow's angle, b = tufts, a = the sideways sag just downwind of them.
	const flow = new THREE.StorageTexture( FIBRE_RES, FIBRE_RES );
	flow.type = THREE.HalfFloatType;
	flow.wrapS = flow.wrapT = THREE.RepeatWrapping;
	flow.minFilter = flow.magFilter = THREE.LinearFilter;
	flow.generateMipmaps = false;
	const flowKernel = Fn( () => {

		const n = uint( FIBRE_RES );
		const c = ivec2( int( instanceIndex.mod( n ) ), int( instanceIndex.div( n ) ) );
		const p = vec2( c ).add( 0.5 ).div( FIBRE_RES ).toVar();
		const psi = ( q: V2 ) => noise2( q, 3, 0.1 ).mul( 0.5 ).add( noise2( q.add( 0.37 ), 7, 0.2 ).mul( 0.25 ) ).add( noise2( q.add( 0.71 ), 15, 0.3 ).mul( 0.1 ) );
		const e = 1 / FIBRE_RES;
		const curl = vec2( psi( p.add( vec2( 0, e ) ) ).sub( psi( p.sub( vec2( 0, e ) ) ) ), psi( p.sub( vec2( e, 0 ) ) ).sub( psi( p.add( vec2( e, 0 ) ) ) ) ).div( 2 * e );
		const dir = normalize( vec2( 1, 0 ).add( vec2( 0, noise2( p.add( 0.13 ), 3, 0.4 ).mul( 0.5 ) ) ).add( curl.mul( 0.05 ) ) );
		const cluster = clusterAt( p ).toVar();
		const seeds = max( points( p, cluster, 110, 0.3, 0.12, 0 ), points( p, cluster, 44, 0.3, 0.18, 3.3 ).mul( 0.8 ) );
		const tuft = points( p, cluster, 40, 0.4, 0.1, 6.1 );
		const droop = points( p, cluster, 40, 0.4, 0.3, 6.1 );
		textureStore( flow, c, vec4( seeds, atan( dir.y, dir.x ), tuft, droop ) );

	} )().compute( FIBRE_RES * FIBRE_RES, [ 64 ] ).setName( 'Clouds.FibreFlow' );
	renderer.compute( flowKernel );

	const target = new THREE.RenderTarget( FIBRE_RES, FIBRE_RES, { depthBuffer: false, generateMipmaps: true, format: THREE.RGFormat, type: THREE.UnsignedByteType } );
	target.texture.wrapS = target.texture.wrapT = THREE.RepeatWrapping;
	target.texture.minFilter = THREE.LinearMipmapLinearFilter;
	target.texture.magFilter = THREE.LinearFilter;
	target.texture.name = 'streakFibres';
	const flowNode = texture( flow );
	const material = new THREE.NodeMaterial();
	material.depthTest = material.depthWrite = false;
	material.toneMapped = false;
	material.fragmentNode = Fn( () => {

		const p = uv().toVar();
		const x1 = p.toVar(), x2 = p.toVar();
		const fib = float( 0 ).toVar(), head = float( 0 ).toVar();
		Loop( LIC_STEPS, ( { i } ) => {

			const k = float( i );
			// Short strands: each fades in and out along its length.
			const a = flowNode.sample( x1 ).level( float( 0 ) );
			fib.addAssign( a.x.mul( sin( k.add( 0.5 ).mul( Math.PI / LIC_STEPS ) ) ) );
			x1.subAssign( vec2( cos( a.y ), sin( a.y ) ).div( FIBRE_RES ) );
			// Hooked tails: from a tuft downwind, sagging sideways.
			const b = flowNode.sample( x2 ).level( float( 0 ) );
			head.addAssign( b.z.mul( exp( k.mul( - 1 / 12 ) ) ) );
			x2.subAssign( normalize( vec2( cos( b.y ), sin( b.y ).add( b.w.mul( 1.2 ) ) ) ).div( FIBRE_RES ) );

		} );
		const breakup = sat( fbm2( p, 24, 0.71 ).mul( 1.6 ).sub( 0.25 ) );
		const wisps = float( 1 ).sub( exp( fib.mul( - 35 / LIC_STEPS ) ) ).mul( breakup );
		const hooks = float( 1 ).sub( exp( head.mul( - 0.5 ) ) );
		return vec4( sat( max( wisps, hooks ) ), clusterAt( p ), 0, 1 );

	} )();
	const quad = new THREE.QuadMesh( material );
	const previous = renderer.getRenderTarget();
	renderer.setRenderTarget( target );
	quad.render( renderer );
	renderer.setRenderTarget( previous );
	material.dispose();
	flow.dispose();
	return target;

}

// Taps along a footprint's long axis (`sampleStreaks`).
const TAPS = 3;

/** The fibres (density, tufts) and the mip level taken, at layer position `q` (metres, x along the drift), averaged
 * over a pixel's footprint on the layer, whose axes `fa` and `fb` are in the same frame, on a tile
 * of `tile` metres turned by `turn` radians: TAPS taps spread along the long axis, each at the mip
 * level of the short axis or of the long axis' share between taps, whichever is larger. Near the
 * horizon the footprint runs tens of times longer along the view than across it, and a level for
 * the whole of it would blur the fibres across as well. (The compute kernels that draw the layer
 * take no sample with gradients, which would filter anisotropically.) */
export function sampleStreaks( fibres: THREE.Texture, q: V2, fa: V2, fb: V2, tile: number, turn: number ): V3 {

	const c = Math.cos( turn ), s = Math.sin( turn );
	const rotate = ( v: V2 ) => vec2( v.x.mul( c ).add( v.y.mul( s ) ), v.y.mul( c ).sub( v.x.mul( s ) ) ).div( tile );
	const at = rotate( q ).toVar(), a = rotate( fa ).toVar(), b = rotate( fb ).toVar();
	const la = length( a ).mul( FIBRE_RES ), lb = length( b ).mul( FIBRE_RES );
	const major = select( la.greaterThan( lb ), a, b ).toVar();
	const lod = max( log2( max( min( la, lb ), max( la, lb ).div( TAPS ) ) ), 0 ).toVar();
	const node = texture( fibres );
	let sum: V2 = vec2( 0 );
	for ( let i = 0; i < TAPS; i ++ ) sum = sum.add( node.sample( at.add( major.mul( ( i + 0.5 ) / TAPS - 0.5 ) ) ).level( lod ).xy );
	return vec3( sum.div( TAPS ), lod );

}
