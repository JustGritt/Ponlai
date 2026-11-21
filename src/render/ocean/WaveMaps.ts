/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, If, attribute, texture, uniform, varying, vec2, vec3, vec4, float, int, normalize, refract, max, abs,
	clamp, mix, smoothstep, exp, length, screenCoordinate, cameraPosition, cameraProjectionMatrix,
	screenSize,
} from 'three/tsl';
import { FFT_SIZE, type OceanFFT } from './OceanFFT.ts';
import { waveDamping, type WaveDampingInputs } from './WaveDamping.ts';
import { shared } from '../shared.ts';
import { compileInto } from '../post/graph/Compile.ts';
import type { FrameGraph } from '../post/graph/FrameGraph.ts';

// What the light under the water needs from the sea's surface: the wave maps and the caustics.
//
// What the light under the water needs from the sea's surface, baked once a frame into one array
// target, on the frame graph ahead of the scene passes (`waveMapsPass`), after the sea's cascades
// (computed in the tick): the wave maps and the caustics.
//
// The wave maps (tidewater's UnderwaterLighting bake, 4811ba4): what the light under the water needs
// at a point depends on its xz alone (the long waves' height and slope, the mean level), so it is
// baked into two maps centred on the camera (0.25 m texels over 128 m, 2 m over 1 km, snapped to
// their texels so they do not crawl) instead of every fragment that can be under the water reading
// the FFT and the shore waves. Beyond the far map: a level sea. The long waves are the three
// coarsest cascades' height and the two coarsest's slope, damped as the sea's surface is
// (WaveDamping.ts), and the rock's reflected wave; none over dry land. The mean level is the two
// longest cascades' height, read coarser. They are read at level 0 within half a texel of their
// edges, so neither the repeat wrap nor the mip chain reaches them.
//
// The caustics, by rasterised photon splatting (Evan Wallace's "WebGL Water", as tidewater's
// Caustics.js, 4811ba4). A fine grid over one FFT tile is drawn into the caustics' layer; each
// vertex is a point on the wave surface, the sun ray is refracted through the surface normal there
// and followed down to a plane D metres below, and the vertex is placed at that landing point. Each
// triangle writes its area ratio (surface patch / its image on the floor) additively, which is the
// light's concentration: focusing folds form the bright networks. The ratio is computed from the
// triangle's corners in the vertex stage, never from screen derivatives: the quads those come from
// form differently from one draw to the next on a busy GPU, and at a fold the ratio turns their
// last bit into a visible change, so the same inputs drew different layers
// (`docs/reference/three-behaviour.md`, Split draws). The grid covers the tile plus a margin on
// every side (the FFT tiles, so the margin is the neighbouring tiles' surface), so light refracted
// in from across the tile's edge lands inside it and the result tiles seamlessly. Two
// layers, each with two focal planes (shallow, deep) blended by the real depth at lookup: fine
// networks from the finest cascade (its slopes read one level down, so ripples under about 11 cm,
// which defocus at once, are left out) in R and G, and broad focusing from the next cascade, whose
// tile is of another size, so the product never repeats, in B and A, each over the whole layer in
// its own tile's coordinates. The pattern is a function of the cascades and the key's direction
// alone, so of the scene clock: it keeps no history, a paused frame is bit-identical, and the layer is drawn
// again only when the scene time, the spectrum or the key's direction changed. The splats conserve
// the light, so the layer's mean is 1 (but for the few triangles folded past a cap).
// The lookup (`causticsAt`): the light that lands at a floor point entered the water up-sun along
// the refracted sun ray, through the slope of the long waves above; it is blurred with depth (the
// sun's disk and forward scattering) through the layer's mips, filtered over the pixel's footprint
// on a level floor, found from the view ray (anisotropically: the floor is mostly seen at grazing
// angles), takes the broad layer as
// mix( 1, broad, 0.6 ), and fades in over the first half metre and out with depth, times
// `strength`: the pattern moves the light on the floor without changing its level. While the key is
// the moon or gives no light the splats are not drawn, no mips are made and every lookup returns 1
// (`lit`), with no fetch.
//
// The target's layers: the two wave maps, the caustics, then the ground bounce's map (drawn by
// lighting/GroundBounce.ts outside the graph, at level 0 only), so the materials lit under the
// water, at WebGPU's 16 sampled textures, take one texture for all of them. It keeps a short mip
// chain (MIPS levels), made only after the caustics are drawn (the target's `generateMipmaps` is
// off but for that last draw): three makes it for every layer, but only the caustics read it (the
// full chain of nine levels below the base a layer cost up to about a quarter of a millisecond of
// GPU time a frame at 4K).

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

const ETA = 1 / 1.333;

// the shared target's size, and its mip levels: the caustics' lookup blurs through three below the
// base (the wave maps and the ground bounce read level 0 only)
const RES = 512;
const MIPS = 4;

// the target's layers: a layer per wave map ( height - sea level, slope x, slope z, mean level - sea
// level ), the caustics ( fine shallow, fine deep, broad shallow, broad deep ), then the ground
// bounce's map
const WAVE_EXTENTS = [ 128, 1024 ];
const CAUSTICS_LAYER = WAVE_EXTENTS.length;
const BOUNCE_LAYER = CAUSTICS_LAYER + 1;
const LAYERS = BOUNCE_LAYER + 1;
// Above this height over the sea level no water reaches (m): the lookups return a level sea there.
const WAVE_REACH = 2.5;

/** What the wave maps' bake reads (WaveMaps.attachWaves). */
type WaveMapInputs = WaveDampingInputs;

/** The cascades the caustics refract the sun through. */
type CausticCascades = Pick<OceanFFT, 'cascades' | 'sizes' | 'derivatives'>;

interface LayerOptions { grid: number; depths: [ number, number ]; channel: 0 | 2; slopeLevel: number; margin: number }

class CausticLayer {

	readonly tile: number;
	readonly channel: 0 | 2;

	constructor( fft: CausticCascades, cascade: number, private target: THREE.RenderTarget, scene: THREE.Scene,
		{ grid, depths, channel, slopeLevel, margin }: LayerOptions ) {

		this.tile = fft.sizes[ cascade ];
		this.channel = channel;

		// The grid over [-margin, 1 + margin]^2 (in tile units) at the tile's vertex density, its cells
		// split in two triangles, drawn as two meshes (each cell's lower-left and upper-right
		// triangle) so that every vertex starts at most one triangle of each: a triangle's first
		// vertex (WebGPU's provoking vertex) computes the triangle's area ratio from its own corner and
		// the two others it is given, and hands it on flat (no screen derivatives: see the note at the
		// top). Each vertex is shaded once a mesh instead of once per triangle corner.
		const g = Math.ceil( grid * ( 1 + 2 * margin ) );
		const at = ( i: number, j: number ) => [ i / g * ( 1 + 2 * margin ) - margin, j / g * ( 1 + 2 * margin ) - margin ];
		const V = ( g + 1 ) * ( g + 1 );
		const pos = new Float32Array( V * 3 );
		// each vertex's two other corners: of the cell's lower-left triangle it starts ( right, up ),
		// of the upper-right triangle it starts as the cell's upper-right corner ( left, down )
		const lower = new Float32Array( V * 4 ), upper = new Float32Array( V * 4 );
		for ( let j = 0; j <= g; j ++ ) for ( let i = 0; i <= g; i ++ ) {

			const v = j * ( g + 1 ) + i;
			pos.set( at( i, j ), v * 3 );
			lower.set( [ ...at( Math.min( i + 1, g ), j ), ...at( i, Math.min( j + 1, g ) ) ], v * 4 );
			upper.set( [ ...at( Math.max( i - 1, 0 ), j ), ...at( i, Math.max( j - 1, 0 ) ) ], v * 4 );

		}

		const lowerIdx = new Uint32Array( g * g * 3 ), upperIdx = new Uint32Array( g * g * 3 );
		for ( let j = 0, n = 0; j < g; j ++ ) for ( let i = 0; i < g; i ++, n += 3 ) {

			const a = j * ( g + 1 ) + i;
			lowerIdx.set( [ a, a + 1, a + g + 1 ], n );
			upperIdx.set( [ a + g + 2, a + g + 1, a + 1 ], n );

		}

		const position = new THREE.BufferAttribute( pos, 3 );
		const halves = ( [ [ lower, lowerIdx ], [ upper, upperIdx ] ] as const ).map( ( [ corners, idx ] ) => {

			const geo = new THREE.BufferGeometry();
			geo.setAttribute( 'position', position );
			geo.setAttribute( 'corners', new THREE.BufferAttribute( corners, 4 ) );
			geo.setIndex( new THREE.BufferAttribute( idx, 1 ) );
			return geo;

		} );

		const tile = this.tile;
		depths.forEach( ( D, plane ) => {

			// a surface point (tile units) and where its refracted sun ray lands on the plane (m)
			const land = ( q0: V2 ) => {

				const d = texture( fft.derivatives, q0 ).depth( int( cascade ) ).level( float( slopeLevel ) );
				const s = vec2( d.x.div( max( d.z.add( 1 ), 0.3 ) ), d.y.div( max( d.w.add( 1 ), 0.3 ) ) );
				const n = normalize( vec3( s.x.negate(), 1, s.y.negate() ) );
				const T = refract( shared.sunDir.negate(), n, ETA );
				const T0 = refract( shared.sunDir.negate(), vec3( 0, 1, 0 ), ETA );
				// the flat surface's offset is removed so the pattern stays registered with the entry
				// point (the lookup applies it again with the real depth)
				const off = T.xz.div( max( T.y.negate(), 0.15 ) ).sub( T0.xz.div( max( T0.y.negate(), 0.15 ) ) ).mul( D );
				return q0.mul( tile ).add( off );

			};
			const area = ( a: V2, b: V2, c: V2 ) => {

				const e = b.sub( a ), f = c.sub( a );
				return abs( e.x.mul( f.y ).sub( e.y.mul( f.x ) ) );

			};

			for ( const geo of halves ) {

				const q0 = attribute<'vec3'>( 'position', 'vec3' ).xy;
				const corners = attribute<'vec4'>( 'corners', 'vec4' );
				const q = land( q0 );
				// the light is conserved: each triangle writes its own area's worth over its image, so
				// the layer's mean is 1 (the margins' light that leaves the tile comes back in from the
				// other side), with no reading of it at lookup. Only triangles folded to under 1/64 of
				// their area are capped, so the half floats' sums stay finite; their peaks are spread by
				// the lookup's blur.
				const ao = area( q0, corners.xy, corners.zw ).mul( tile * tile );
				const an = area( q, land( corners.xy ), land( corners.zw ) );
				const I = varying( ao.div( max( an, ao.mul( 1 / 64 ) ).max( 1e-12 ) ) )
					.setInterpolation( THREE.InterpolationSamplingType.FLAT, THREE.InterpolationSamplingMode.FIRST );

				const mat = new THREE.NodeMaterial();
				mat.name = 'caustics splat';
				// texture v runs down the target in WebGPU: row 0 at the top (clip y = +1)
				const uvq = q.div( tile );
				mat.vertexNode = vec4( uvq.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uvq.y.mul( 2 ) ), 0, 1 );
				const out = [ 0, 0, 0, 0 ].map( ( _, k ) => k === channel + plane ? I : float( 0 ) );
				mat.fragmentNode = vec4( out[ 0 ], out[ 1 ], out[ 2 ], out[ 3 ] );
				mat.blending = THREE.CustomBlending;
				mat.blendEquation = THREE.AddEquation;
				mat.blendSrc = THREE.OneFactor;
				mat.blendDst = THREE.OneFactor;
				mat.blendSrcAlpha = THREE.OneFactor;
				mat.blendDstAlpha = THREE.OneFactor;
				mat.depthTest = false;
				mat.depthWrite = false;
				mat.side = THREE.DoubleSide;
				const mesh = new THREE.Mesh( geo, mat );
				mesh.frustumCulled = false;
				scene.add( mesh );

			}

		} );

	}

	// One lookup of this layer's two planes ( shallow, deep ): over the pixel's footprint stretched to
	// at least the blur level's texel.
	fetch( uv: V2, level: F, footprint: { dx: V2; dy: V2 } ): V2 {

		const t = texture( this.target.texture, uv ).depth( int( CAUSTICS_LAYER ) );
		const planes = ( v: THREE.Node<'vec4'> ) => this.channel === 0 ? v.xy : v.zw;
		const minLength = level.exp2().div( RES );
		const stretch = ( g: V2 ) => g.div( this.tile ).toVar().mul( max( minLength.div( max( length( g.div( this.tile ) ), 1e-9 ) ), 1 ) );
		return planes( t.grad( stretch( footprint.dx ), stretch( footprint.dy ) ) );

	}

}

export class WaveMaps {

	renderer: THREE.WebGPURenderer;
	readonly target: THREE.RenderTarget;
	/** The target's layer that holds the ground bounce's map (lighting/GroundBounce.ts draws it). */
	readonly bounceLayer = BOUNCE_LAYER;
	/** The caustics' contrast (0: none). Their mean stays 1 whatever it is. */
	readonly strength = uniform( 0.75 );

	// the wave maps: each one's world origin (its texel ( 0, 0 )'s corner) and its bake, once attached
	private readonly origins = WAVE_EXTENTS.map( () => uniform( new THREE.Vector2() ) );
	private readonly wavesReady = uniform( 0 );
	private bakes: { quad: THREE.QuadMesh; layer: number }[] = [];
	// the caustics: their splats (which refract the key's direction, shared.sunDir) and whether
	// they are drawn
	/** The caustics' splats, drawn into their layer through `splatCamera`. */
	readonly splats = new THREE.Scene();
	readonly splatCamera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 1 );
	private readonly fine: CausticLayer;
	private readonly broad: CausticLayer;
	private readonly lit = uniform( 0 );
	private readonly clear = new THREE.Color();
	// what the caustics' layer was last drawn from ( scene time, the spectrum's revision, the key's
	// direction ), and what this frame's would be: the same inputs draw the same layer
	private readonly drawnFrom = [ NaN, NaN, NaN, NaN, NaN ];
	private readonly drawFrom = [ 0, 0, 0, 0, 0 ];

	constructor( renderer: THREE.WebGPURenderer, fft: CausticCascades ) {

		this.renderer = renderer;
		this.target = new THREE.RenderTarget( RES, RES, {
			depth: LAYERS, type: THREE.HalfFloatType, generateMipmaps: false,
			minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter,
			wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping, depthBuffer: false, anisotropy: 8,
		} );
		this.target.texture.name = 'waveMaps';
		// three takes a render target's level count from its texture's mipmaps when there are any
		// (r186 Textures.getMipLevels; nothing is uploaded from them): a short chain, three
		// downsamples per layer instead of nine
		this.target.texture.mipmaps = Array.from( { length: MIPS }, () => ( { data: new Uint8Array( 0 ), width: 0, height: 0 } ) );
		const fine = fft.cascades - 1;
		this.fine = new CausticLayer( fft, fine, this.target, this.splats, { grid: 256, depths: [ 1.2, 4.0 ], channel: 0, slopeLevel: 1, margin: 0.35 } );
		this.broad = new CausticLayer( fft, fine - 1, this.target, this.splats, { grid: 128, depths: [ 3.0, 9.0 ], channel: 2, slopeLevel: 0.5, margin: 0.35 } );

	}

	/** Compiles the splats' draws for their layer ahead of their first frame: while the key gives no
	 * light that frame comes only with the dawn. */
	warm(): Promise<void> {

		return compileInto( this.renderer, this.splats, this.splatCamera, this.target );

	}

	/** Bakes the wave maps from now on (the world's fields and the shore waves exist): until then
	 * the lookups read a level sea. */
	attachWaves( inputs: WaveMapInputs ): void {

		const { fft } = inputs;
		WAVE_EXTENTS.forEach( ( extent, level ) => {

			const texel = extent / RES, origin = this.origins[ level ];
			// the world xz at this texel's centre (screen row 0 is texture row 0)
			const xz = screenCoordinate.xy.mul( texel ).add( origin ).toVar();
			const { weights, seaDepth } = waveDamping( inputs, xz );
			const w = [ weights.x, weights.y, weights.z ];
			const at = ( field: THREE.Texture, c: number, lvl: number ) => texture( field, xz.div( fft.sizes[ c ] ) ).depth( int( c ) ).level( float( lvl ) );
			// the cascades read no finer than the bake's texel (and at least two levels down)
			const lvl = ( c: number ) => Math.max( 2, Math.log2( texel * FFT_SIZE / fft.sizes[ c ] ) );
			const waves = new THREE.NodeMaterial();
			waves.fragmentNode = Fn( () => {

				let h: F = float( 0 ), mean: F = float( 0 ), slope: V2 = vec2( 0 );
				for ( let c = 0; c < Math.min( 3, fft.cascades ); c ++ ) {

					h = h.add( at( fft.displacement, c, lvl( c ) ).y.mul( w[ c ] ) );
					if ( c < 2 ) {

						slope = slope.add( at( fft.derivatives, c, lvl( c ) ).xy.mul( w[ c ] ) );
						mean = mean.add( at( fft.displacement, c, 3 ).y.mul( w[ c ] ) );

					}

				}
				const shore = inputs.shoreWaves.surfaceAt( xz, shared.time ).toVar();
				// waves only over water: none over dry land (a deep trough there, so nothing above it
				// counts as under the water)
				const height = h.add( shore.x ).mul( smoothstep( 0, 1, seaDepth ) ).sub( smoothstep( 0, - 0.4, seaDepth ).mul( 10 ) );
				return vec4( height, slope.add( shore.yz ), mean.mul( smoothstep( 0, 3, seaDepth ) ) );

			} )();
			this.bakes.push( { quad: new THREE.QuadMesh( waves ), layer: level } );

		} );

	}

	/** This frame's state, set in the tick (no rendering): the maps' origins round the camera,
	 * whether the key lights the caustics (`lit`: the sun, not the moon, and giving light) and what
	 * they would be drawn from (the splats read the key's direction from shared.sunDir). */
	prepare( camera: THREE.Camera, keyDir: THREE.Vector3, lit: boolean, time: number, spectrumRevision: number ): void {

		// the maps' origins, snapped to whole texels so their texel centres stay put in the world
		WAVE_EXTENTS.forEach( ( extent, level ) => {

			const texel = extent / RES;
			this.origins[ level ].value.set( Math.floor( camera.position.x / texel ) * texel - extent / 2, Math.floor( camera.position.z / texel ) * texel - extent / 2 );

		} );
		this.lit.value = lit ? 1 : 0;
		const from = this.drawFrom;
		from[ 0 ] = time; from[ 1 ] = spectrumRevision; from[ 2 ] = keyDir.x; from[ 3 ] = keyDir.y; from[ 4 ] = keyDir.z;

	}

	/** The bake (the graph's pass, `waveMapsPass`): the wave maps, then the caustics while lit and
	 * when their inputs changed, with the mip chain after them (only the caustics read it). */
	render(): void {

		const r = this.renderer;
		const tex = this.target.texture;
		const draws: ( () => void )[] = [];
		for ( const { quad, layer } of this.bakes ) draws.push( () => {

			r.setRenderTarget( this.target, layer );
			quad.render( r );

		} );
		const caustics = this.lit.value > 0 && this.drawFrom.some( ( v, i ) => v !== this.drawnFrom[ i ] );
		if ( caustics ) draws.push( () => {

			r.getClearColor( this.clear );
			const alpha = r.getClearAlpha(), autoClear = r.autoClear, autoClearColor = r.autoClearColor;
			r.setClearColor( 0x000000, 0 );
			r.autoClear = true;
			r.autoClearColor = true;
			r.setRenderTarget( this.target, CAUSTICS_LAYER );
			r.render( this.splats, this.splatCamera );
			r.setClearColor( this.clear, alpha );
			r.autoClear = autoClear;
			r.autoClearColor = autoClearColor;
			this.drawFrom.forEach( ( v, i ) => this.drawnFrom[ i ] = v );

		} );
		if ( draws.length === 0 ) return;
		const prev = r.getRenderTarget();
		// the mip chain only after the last draw of a frame that drew the caustics
		draws.forEach( ( draw, i ) => {

			tex.generateMipmaps = caustics && i === draws.length - 1;
			draw();

		} );
		tex.generateMipmaps = false;
		r.setRenderTarget( prev );
		if ( this.bakes.length > 0 ) this.wavesReady.value = 1;

	}

	/** The wave maps at a world point: ( height of the long waves' surface over the sea level, their
	 * slope x and z, mean level over the sea level ); a level sea above WAVE_REACH, beyond the far map
	 * and before the first bake. Explicit levels: any branch. */
	wavesAt( P: V3 ): THREE.Node<'vec4'> {

		return Fn( () => {

			const out = vec4( 0 ).toVar();
			const inside = ( st: V2 ) => st.x.greaterThan( 0.5 ).and( st.y.greaterThan( 0.5 ) ).and( st.x.lessThan( RES - 0.5 ) ).and( st.y.lessThan( RES - 0.5 ) );
			const st = WAVE_EXTENTS.map( ( extent, level ) => P.xz.sub( this.origins[ level ] ).div( extent / RES ).toVar() );
			const read = ( level: number ) => () => {

				out.assign( texture( this.target.texture, st[ level ].div( RES ) ).depth( int( level ) ).level( float( 0 ) ) );

			};
			If( this.wavesReady.greaterThan( 0.5 ).and( P.y.lessThan( shared.waterLevel.add( WAVE_REACH ) ) ), () => {

				If( inside( st[ 0 ] ), read( 0 ) ).ElseIf( inside( st[ 1 ] ), read( 1 ) );

			} );
			return out;

		} )();

	}

	/** The caustics' light factor (mean 1) at a world point `depth` metres under the
	 * long waves' surface, whose slope above is `slope`. Explicit gradients: any branch. */
	causticsAt( P: V3, depth: F, slope: V2 ): V3 {

		const F = this.fine, B = this.broad;
		// none right at the surface, strongest in the first metres, fading with depth; no lookups
		// where it is 1 (the moon's or no key, or no water above)
		const k = smoothstep( 0.03, 0.5, depth ).mul( exp( depth.mul( - 0.06 ) ) ).mul( this.strength ).mul( this.lit ).toVar();
		const caustic = float( 1 ).toVar();
		If( k.greaterThan( 0 ), () => {

			// the light reaching this point entered the water up-sun along the refracted sun ray
			const Ls = refract( shared.sunDir.negate(), normalize( vec3( slope.x.negate(), 1, slope.y.negate() ) ), ETA );
			const entry = P.xz.sub( Ls.xz.mul( depth.div( max( Ls.y.negate(), 0.15 ) ) ) ).toVar();
			// deeper: softer (finite sun disk and forward scattering)
			const blur = clamp( depth.mul( 0.4 ).sub( 0.2 ), 0, MIPS - 1 );
			// The pixel's footprint on a level floor, from the view ray (no screen derivatives: taken
			// before the branch, they cost a tenth of a millisecond or more of the opaque pass at 4K):
			// a pixel spans t * 2 / ( P11 * height ) across the ray, stretched by 1 / |v.y| along
			// it, P11 = 1 / tan( fov / 2 ) the pass's own projection's [ 1 ][ 1 ] (the jitter moves
			// only its third column) and height the pass's own target's.
			const ray = P.sub( cameraPosition ).toVar();
			const t = length( ray );
			const p11 = ( cameraProjectionMatrix as unknown as { element( i: number ): { element( i: number ): F } } ).element( 1 ).element( 1 );
			const across = t.mul( 2 ).div( p11.mul( screenSize.y ) );
			const h = ray.xz.div( max( length( ray.xz ), 1e-6 ) ).toVar();
			const footprint = {
				dx: vec2( h.y.negate(), h.x ).mul( across ),
				dy: h.mul( across.mul( t ).div( max( abs( ray.y ), t.mul( 0.02 ) ) ) ),
			};
			const fine = F.fetch( entry.div( F.tile ), blur, footprint );
			const broad = B.fetch( entry.div( B.tile ), float( 2.5 ), footprint );
			const c = mix( fine.x, fine.y, clamp( depth.sub( 1.2 ).div( 2.8 ), 0, 1 ) ); // between the fine layer's planes
			const br = mix( broad.x, broad.y, clamp( depth.sub( 3 ).div( 6 ), 0, 1 ) ); // between the broad layer's planes
			caustic.assign( mix( float( 1 ), c.mul( mix( float( 1 ), br, 0.6 ) ), k ) );

		} );
		return vec3( caustic );

	}

}

/** The wave maps' bake on the frame graph, ahead of the scene passes that read them (the
 * terrain, the floor and the fish under the water). */
export function waveMapsPass<Frame>( g: FrameGraph<Frame>, maps: () => WaveMaps | null ): void {

	g.addPass( 'waveMaps', 'callback', () => ( {
		label: 'wave maps', size: { of: 'fixed', width: RES, height: RES }, neverCull: true,
		run: () => maps()?.render(),
	} ) );

}
