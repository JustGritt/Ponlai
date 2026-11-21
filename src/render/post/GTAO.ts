import * as THREE from 'three/webgpu';
import {
	Fn, If, abs, acos, clamp, cos, cross, dot, float, floor, int, ivec2, length, max, min, mix, mod, normalize, positionView,
	round, screenCoordinate, select, sign, sin, texture, textureLoad, uniform, vec2, vec3, vec4,
} from 'three/tsl';
import { debugFlags } from '../debug/DebugFlags.ts';
import { tuneUniform } from '../debug/Tune.ts';
import { shared } from '../shared.ts';
import { MEADOW_SHADOWS } from './MeadowShadows.ts';
import { loadTier, qualityProfile } from '../Quality.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// Screen-space ambient occlusion by GTAO: the metre-scale occlusion the probes are too coarse for.
//
// Screen-space ambient occlusion: a soft, metre-scale loss of the ambient light where the grass,
// the culms and the ground's relief close the sky over a point, the occlusion within the meadow's
// blades, a grove's leaves and culms and the rock's relief, which the ambient volume's 2 m probes
// and the terrain's vertex occlusion are too coarse to hold. Ground-truth ambient occlusion
// (GTAO: Jimenez, Wu, Pesce and Jarabo, "Practical Realtime Strategies for Accurate Indirect
// Occlusion", Activision, 2016): per pixel a few slices of the view-aligned hemisphere, in each the
// highest horizon on either side found over the depth, and the cosine-weighted visible arc between
// them integrated in closed form against the slice's projected normal. The slices start a little off
// the surface along its normal (START_OFFSET of the view distance), so the depth's precision does not
// leave a faint self-occlusion on flat and gently sloping ground. Its practical shape (the low
// horizons the integration starts from, the projected normal's length as the slice's weight)
// follows Intel's XeGTAO (MIT) as a technique; the code is
// ours.
//
// The occlusion is a broad term, never a contact line: in shadow, the meadow about a sixth
// darker than without it at eye height and a fourteenth from the management camera's distance,
// the leaves under a tenth, a culm's sides a few per cent, lit surfaces a few per cent, the grass
// at a culm's foot no darker than the shaded grass around it, and the darkest tenth of the shadowed
// pixels about a quarter darker. The ground on the turf takes the stronger of this and its roots' share of
// the sky, the smaller of the two factors (materials/TerrainMaterial.ts `cliffSurface`), and the
// roots' share is set by the turf's cover alone, whatever stands on it: the soil among the blades,
// a grove's litter and a culm's foot take none of this wherever the sky a slice leaves is above
// about a third, where the roots' is the stronger. The ground off the turf (cliff, rock, beach,
// stones) takes it whole. A horizon's weight falls off with how much nearer the eye than the point
// it stands and over a side's last steps (Thin occluders, below). The
// kernel's size follows the view distance continuously: within `fullDistance` (22 m, the
// management camera's nearest, scaled by the surface's cosine to the view) a slice reaches
// `maxTexels` (128 half-resolution texels) with four evenly spaced steps a side, and beyond it the
// reach on screen and the step count (rounded, at least one) fall as 1 / z, so near views see small
// crevices, far ones the large forms, and no tier of steps shows
// between them. The result is linear in the visibility v over a floor, `base` + ( 1 - `base` ) v,
// with no curve on it, so the deepest crevice keeps a little sky. Its setting: `base` 0.09, the
// land's mean albedo, as a direction that sees no sky sees an occluder, which under an even sky
// sends back its albedo times the sky's radiance (one bounce; the albedo as
// lighting/AmbientSample.ts CONE_OCCLUSION measures it); `cutoff` 0.7 m (The cutoff, below);
// `recede` 0.1 (Thin occluders, below).
// Measured against the frame without it (at run time), in the lighting layer under an 18-degree
// sun, on a 256 x 144 grid's cells whose direct share of the light is under a quarter (in shadow)
// or over 0.6 (lit), over seven views at eye height 4-40 m off in the frame's lower half: the
// meadow in shadow 17 % darker at the median, its darkest tenth 26 %, the leaves 8 % (the two views
// that show shaded leaves), the rock 3 % (four views), the culms 2 % (six), the ground beside the
// grass 1 % (six), lit cells 3 %. From the management camera's distance (two views 45 m off,
// 30 degrees down over a grove-filled meadow) under a 50-degree sun: the meadow in shadow 7 %, the
// ground beside it 2 %, lit cells under 1 %; under the 18-degree one 5 %, 2 % and 1 %. Shaded
// grass one cell beside a culm is 11 % darker, less than the shaded grass three or more cells off
// any culm (16 %), over the three views with shaded grass beside culms, and from the management
// camera's distance under the 50-degree sun 6 % against 9 % (the two views): no halo.
// Where it and a surface's own occlusion multiply, the product keeps a floor (the look's, which the
// shared uniforms take where they are made: shared.ts `aoFloor`): the deepest crevice or contact
// keeps a share of the ambient light and never turns black.
//
// Every material that reads the ambient light takes it into its own occlusion: the blades multiply
// it into their own from root to tip, the other receivers multiply it in whole, and the ground on
// the turf alone takes the smaller of it and its roots' share of the sky (world/Grass.ts
// `GRASS_ROOTS`, materials/TerrainMaterial.ts `cliffSurface`; above). The depth it reads holds the
// blades themselves, so on a blade it counts a second time the occlusion the blade's own term already
// holds; the cutoff (below) keeps that count small.
//
// It applies to the ambient light only: the materials that take it multiply it into
// their lighting model's ambient occlusion (lighting/AmbientGrade.ts `useAmbientGrade`), so it
// shades the sky's light, the ground's bounce and, through three's specular occlusion, the sky's
// reflection, and the ambient grade's `aoStrength` scales it with the rest; the sun's direct light
// is never touched, and nothing darkens the image after the fact. The receivers mark themselves
// (`receiveScreenAO`): the sea floor, the bamboo's culms and leaves, the yunpeng and the meadow's
// blades; the island's ground and the stones take it in their own occlusion (`aoNode`, on the turf
// the smaller of it and the roots' share), which reaches the same ambient occlusion. The islanders
// and the game's props do not take it: they are not in the depth it reads (post/OpaquePrepass.ts).
//
// The frame (Pipeline.ts), at 4x outside the temporal modes (with the meadow's screen-space shadows, whose
// vegetation depth pass it reads):
//   opaqueDepth  the opaque scene's copies (post/OpaquePrepass.ts) at half resolution, single
//                sampled, their distance along the view into a target of their own (drawn before
//                the meadow's screen-space shadows, which read it too over the whole scene: MeadowShadows.ts)
//   gtaoDepth    the view z of each half-resolution texel: the nearer of the vegetation's depth
//                (sample 0 of its full pixel's top left) and the opaque copies' distance
//   gtao         the slices over that depth, (occlusion, view z) at half resolution
//   gtaoBlurX    a separable blur of it along x, then along y (gtaoBlurY), five binomial taps
//   gtaoBlurY    each, every tap weighed by how near its view z lies to the plane of the
//                centre's surface (its normal from the depth, as the slices take it), so the
//                occlusion runs on along a slope and does not bleed between a culm and the ground
//                behind it; (occlusion, view z)
//   opaque       the receivers load it at their pixel: a depth-aware bilinear upsample of the four
//                half-resolution texels around it (texels whose view z differs from the
//                fragment's are dropped; with none left, the nearest in depth)
// Half resolution: the occlusion is a soft, low-frequency term, and the slices' cost is in
// their depth loads, a quarter as many there; the upsample's depth test keeps a culm's occlusion
// off the ground behind it and the ground's off the culm. Two slices of four steps a side: against
// three, the frames in the lighting layer moved by under 1 % on average and a few per cent in the
// noisiest cells (measured with squared step spacing, a fixed count and a 5x5 depth-only blur), the
// relations above not at all, and the blur still averaged the pattern's phases (the floor at a
// grove's edge a touch blotchier when zoomed in).
//
// Cost (tools/render/profile-frame.mjs --compare, production builds on the high tier at the baseline, four of the kept
// scenes, with the earlier 5x5 blur in one pass): the opaque depth and the three AO passes together take about 3 % of the frame's GPU time
// (3.8 % on the meadow close-up), half what they took with three slices over the copies drawn into
// the scene's 4x depth (5.5-6.5 %); the opaque pass, which had tested against those copies, takes
// 1-3 % more, so the frame gains 1-3 %.
//
// No temporal filter: the slices' rotation and the steps' offsets follow a fixed 4x4 pattern of
// the texel's position (a Bayer matrix), and the blur's 5x5 footprint averages it out in space
// alone (its binomial weights do not weigh the pattern's phases exactly alike). A paused
// frame is a function of its depth: it is bit-identical, the occlusion keeps no history, a scene
// jump has nothing to clear, and the `settled` signal (Settled.ts) does not wait on it.
//
// Thin occluders: a depth buffer cannot tell a 10 cm culm from a wall, so the ground behind a culm
// would take the culm as a horizon and a dark halo would follow its outline. Two things keep a
// sample from closing the sky alone. Its weight falls with how much nearer the eye than the point
// it stands, to none at `cutoff` (0.7 m), and over the far half of a side's steps; a sample's
// horizon is drawn back toward the lowest one by that loss of weight, so a blade or culm before the
// distant ground leaves it open. And the horizon recedes: a sample below the horizon so far does
// not leave it standing, but lets it sink `recede` / steps of the way toward it, so one culm that
// raised it gives way to the open ground behind it, and only a wall that holds it up step after
// step keeps the sky closed.
//
// The cutoff: 0.7 m, so of what stands nearer the eye than a point, only what is nearer by less
// than a stride closes its sky (a sample no nearer than the point keeps its whole weight): the
// blades round a blade, a culm beside the grass, the ground's own relief. At 3.3 m the blades
// several metres in front of a sunlit blade counted as its occluders. Measured on the ambient light
// at eye height under an 18-degree sun, over seven views of the meadow (cells 4-40 m off in the
// frame's lower half whose nine texels show one kind), from 3.3 m to 0.7 m: a sunlit blade's loss
// 0.50 stop to 0.26 (the same under a 14.5- and a 21-degree sun), the leaves' 0.19 to 0.08 (six
// views), the culms' 0.11 to 0.03, the rock's 0.07 to 0.03 (five views), the bare ground's none at
// either (the turf's roots, above). The meadow in shadow then reads 0.99 stop over the shaded bare
// ground among it (0.78 at 3.3 m: six views of a grove's edge, 5.5 m up and 27.5 degrees down,
// read in the lighting layer, without the albedo); a sunlit blade's direct share of its light
// under a closed sky falls from 0.82 to 0.79 and in clear from 0.86 to 0.84, as its ambient light
// rises.
//
// On by default where it can run: the high tier's `ao` setting (Quality.ts) builds it at load, at
// 4x outside the temporal modes with the meadow's screen-space shadows. `flags=gtao:false` builds without it (nothing here is
// declared, compiled or drawn, and the frame is the one without it to the byte), `flags=gtao` with
// it on a tier whose setting is off. At run time `flags.gtao = false` skips its passes and the receivers
// read 1, so tools/render/profile-frame.mjs --toggle "s.flags.gtao = on" times it. Its parameters are
// `__PONLAI__.gtao` in debug builds; the AO map is shown with `fgview=gtao.r` (and the undenoised
// one with `fgview=gtaoRaw.r`; graph/DebugView.ts).

/** Built at load: at 4x outside the temporal modes with the meadow's screen-space shadows, where
 * the tier's `ao` setting or `flags=gtao` asks for it. */
export const GTAO = MEADOW_SHADOWS && ( debugFlags.gtao === undefined ? qualityProfile( loadTier() ).ao : debugFlags.gtao !== false );

/** The slices per pixel and the depth steps on each side of each. */
const SLICES = 2, STEPS = 4;
/** View z beyond which nothing is occluded (the sky, -1e5 in the depth's resolve). */
const FAR = 1e4;
/** The half float's range, which the occlusion's view z is clamped to. */
const HALF_MAX = 6e4;
/** The slices start this share of the view distance off the surface along its normal: about the
 * half-resolution view z's precision, so a flat or gently sloping surface does not occlude itself. */
const START_OFFSET = 1e-3;

/** The occlusion's parameters: `cutoff` (m) how far in front of the point a sample stands when it no
 * longer counts, `recede` the share of the way toward a lower sample the horizon sinks over a
 * slice's side, `base` the occlusion where the slices see no sky (the result is linear in the
 * visibility above it), `edge` the blur's edge sharpness (a tap off the centre's plane by
 * 1 / edge of the view distance counts not at all), `maxTexels` the reach on screen of the full
 * kernel (half-resolution texels), `fullDistance` (m) the view distance within which a surface
 * facing the eye takes the full kernel (the management camera's nearest: 22 m), the reach and the
 * steps falling as 1 / z beyond it. `cutoff`, `recede` and `base` are tunable on debug pages
 * (debug/Tune.ts `post.gtao.<value>`). */
export const gtaoUniforms = {
	cutoff: tuneUniform( 'post.gtao.cutoff', 0.7, { min: 0.25, max: 4, scale: 'log' } ),
	recede: tuneUniform( 'post.gtao.recede', 0.1, { min: 0, max: 0.5 } ),
	base: tuneUniform( 'post.gtao.base', 0.09, { min: 0, max: 0.4 } ),
	edge: uniform( 50 ), maxTexels: uniform( 128 ), fullDistance: uniform( 22 ),
};

/** A surface's ambient occlusion `ao` (its own times the screen's) held at the floor or above
 * (shared.ts `aoFloor`: the least a surface keeps, the look's), a constant in the shader. */
export function floorOcclusion( ao: F ): F {

	return max( ao, shared.aoFloor );

}

/** What the passes and the receivers share (Pipeline.ts sets them): the half-resolution size and
 * the render's, the projection's ( P00, P11, P20, P21 ), and 1 while the opaque pass draws with the
 * occlusion on (else the receivers read 1). */
export const gtaoView = {
	size: uniform( new THREE.Vector2( 1, 1 ) ), render: uniform( new THREE.Vector2( 1, 1 ) ),
	proj: uniform( new THREE.Vector4( 1, 1, 0, 0 ) ), on: uniform( 0 ),
};

/** The base node the receivers load the occlusion through (Pipeline.ts sets the graph's texture). */
export const gtaoMap = texture( new THREE.DataTexture( new Float32Array( [ 1, - HALF_MAX, 0, 0 ] ), 1, 1, THREE.RGBAFormat, THREE.FloatType ) );

// A texel's load: three r186 gives each load's clone the texture's uv matrix (a mat3 uniform per
// load site, the identity here), so the loads leave it out.
type Loaded = { setUpdateMatrix( value: boolean ): unknown };
function load( map: THREE.TextureNode, q: V2 ): V4 {

	return ( textureLoad( map, ivec2( q ) ) as unknown as Loaded ).setUpdateMatrix( false ) as V4;

}

/** Marks `material` as a receiver (with the flag; else nothing). */
export function receiveScreenAO( material: THREE.Material ): void {

	if ( GTAO ) material.userData.screenAO = true;

}

/** The projection's ( P00, P11, P20, P21 ) of `camera`, for `gtaoView.proj`. */
export function gtaoProjection( camera: THREE.Camera, out: THREE.Vector4 ): void {

	const e = camera.projectionMatrix.elements;
	out.set( e[ 0 ], e[ 5 ], e[ 8 ], e[ 9 ] );

}

/** The occlusion at the fragment's pixel (1 open), for a receiver's lighting model: the
 * depth-aware bilinear upsample. Inlined into each material (loads only: no uniformity rule). */
export function screenAOAt(): F {

	// the full pixel ( 2i, 2j ) a half texel holds: its centre lies at half-texel coordinate i
	const h = screenCoordinate.xy.sub( 0.5 ).mul( 0.5 ).toVar();
	const i0 = floor( h ).toVar();
	const f = h.sub( i0 ).toVar();
	const z = positionView.z.toVar();
	const tol = abs( z ).mul( 0.03 ).add( 0.05 ).toVar();
	const last = gtaoView.size.sub( 1 );
	const sum = float( 0 ).toVar(), weight = float( 0 ).toVar();
	const nearest = float( 1 ).toVar(), nearestDz = float( 1e9 ).toVar();
	for ( const [ ox, oy ] of [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ] ) {

		const t = load( gtaoMap, clamp( i0.add( vec2( ox, oy ) ), vec2( 0 ), last ) ).toVar();
		const dz = abs( t.y.sub( z ) ).toVar();
		const bilinear = ( ox ? f.x : f.x.oneMinus() ).mul( oy ? f.y : f.y.oneMinus() );
		const w = bilinear.mul( clamp( dz.div( tol ).oneMinus(), 0, 1 ) ).toVar();
		sum.addAssign( t.x.mul( w ) );
		weight.addAssign( w );
		const closer = dz.lessThan( nearestDz );
		nearest.assign( select( closer, t.x, nearest ) );
		nearestDz.assign( select( closer, dz, nearestDz ) );

	}
	const ao = select( weight.greaterThan( 1e-3 ), sum.div( max( weight, 1e-3 ) ), nearest );
	return mix( float( 1 ), ao, gtaoView.on );

}

/** The view z at each half-resolution texel: the nearer of sample 0 of its full pixel ( 2i, 2j ) in
 * the multisampled `depth` (the vegetation's) and the opaque copies' distance in `opaque` (-1e5
 * where nothing was drawn). */
export function gtaoDepthMaterial( depth: THREE.Texture, opaque: THREE.Texture, projInv: THREE.Node<'mat4'> ): THREE.NodeMaterial {

	const ms = texture( depth ), copies = texture( opaque );
	const m = new THREE.NodeMaterial();
	m.fragmentNode = Fn( () => {

		const half = floor( screenCoordinate.xy ).toVar();
		const full = half.mul( 2 ).toVar();
		const d = ( ( textureLoad( ms, ivec2( full ), int( 0 ) ) as unknown as Loaded ).setUpdateMatrix( false ) as F ).toVar();
		const z = float( - 1e5 ).toVar();
		If( d.lessThan( 0.999999 ), () => {

			const q = full.add( 0.5 ).div( gtaoView.render );
			const v = projInv.mul( vec4( q.x.mul( 2 ).sub( 1 ), float( 1 ).sub( q.y ).mul( 2 ).sub( 1 ), d, 1 ) ).toVar();
			z.assign( v.z.div( v.w ) );

		} );
		// the opaque copies' distance along the view (0 where none drew), the nearer of the two
		const distance = ( ( textureLoad( copies, ivec2( half ), int( 0 ) ) as unknown as Loaded ).setUpdateMatrix( false ) as unknown as V4 ).x.toVar();
		If( distance.greaterThan( 0 ), () => {

			z.assign( max( z, distance.negate() ) );

		} );
		return vec4( z, 0, 0, 0 );

	} )();
	m.name = 'gtaoDepth';
	return m;

}

// The 4x4 Bayer matrix's value at a texel (0..15): from the 2x2 one, 2 ( x xor y ) + y, at both
// scales (the xor of two bits as their sum mod 2).
function bayer4( p: V2 ): F {

	const b2 = ( x: F, y: F ) => mod( x.add( y ), 2 ).mul( 2 ).add( y );
	const lo = b2( mod( p.x, 2 ), mod( p.y, 2 ) );
	const hi = b2( mod( floor( p.x.mul( 0.5 ) ), 2 ), mod( floor( p.y.mul( 0.5 ) ), 2 ) );
	return lo.mul( 4 ).add( hi );

}

/** Reading the half-resolution view z `zMap`: a texel's view z, its view-space point, and the
 * surface's normal there from the depth. */
function depthReader( zMap: THREE.TextureNode ) {

	const { size, render, proj } = gtaoView;
	const last = size.sub( 1 );
	const zAt = ( q: V2 ): F => load( zMap, clamp( q, vec2( 0 ), last ) ).x;
	// a texel's view-space point from its view z (the full pixel ( 2i, 2j )'s centre)
	const viewAt = ( q: V2, z: F ): V3 => {

		const uv = q.mul( 2 ).add( 0.5 ).div( render );
		const ndc = vec2( uv.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uv.y.mul( 2 ) ) );
		return vec3( ndc.add( proj.zw ).mul( z.negate() ).div( proj.xy ), z );

	};
	// the normal at `here` (view z `z`, point `P`, toward the eye `V`), on each axis from the
	// neighbour nearer in depth (an edge's other side is the farther one), turned toward the eye
	const normalAt = ( here: V2, z: F, P: V3, V: V3 ): V3 => {

		const side = ( dx: number, dy: number ) => {

			const q = here.add( vec2( dx, dy ) );
			const zq = zAt( q ).toVar();
			return { z: zq, p: viewAt( q, zq ).toVar() };

		};
		const l = side( - 1, 0 ), r = side( 1, 0 ), u = side( 0, - 1 ), d = side( 0, 1 );
		const toR = r.p.sub( P ).toVar(), fromL = P.sub( l.p ).toVar();
		const toD = d.p.sub( P ).toVar(), fromU = P.sub( u.p ).toVar();
		const ddx = select( abs( r.z.sub( z ) ).lessThan( abs( z.sub( l.z ) ) ), toR, fromL ).toVar();
		const ddy = select( abs( d.z.sub( z ) ).lessThan( abs( z.sub( u.z ) ) ), toD, fromU ).toVar();
		// (screen y runs down: ddy points down the view, and dy x dx toward the eye)
		const n0 = normalize( cross( ddy, ddx ) ).toVar();
		return select( dot( n0, V ).lessThan( 0 ), n0.negate(), n0 );

	};
	return { zAt, viewAt, normalAt };

}

/** The slices over the half-resolution view z `depth`: ( occlusion, view z ). */
export function gtaoMaterial( depth: THREE.Texture ): THREE.NodeMaterial {

	const { cutoff, recede, base, maxTexels, fullDistance } = gtaoUniforms;
	const { zAt, viewAt, normalAt } = depthReader( texture( depth ) );
	const m = new THREE.NodeMaterial();
	m.fragmentNode = Fn( () => {

		const here = floor( screenCoordinate.xy ).toVar();
		const z = zAt( here ).toVar();
		const ao = float( 1 ).toVar();
		If( z.greaterThan( - FAR ), () => {

			const P = viewAt( here, z ).toVar();
			const V = normalize( P.negate() ).toVar();
			const N = normalAt( here, z, P, V ).toVar();
			// the slices' origin, lifted off the surface along its normal (START_OFFSET)
			const O = P.add( N.mul( z.negate().mul( START_OFFSET ) ) ).toVar();
			// the share of the full kernel: whole within `fullDistance` (scaled by the surface's cosine
			// to the view), falling as 1 / z beyond; the reach on screen (half-resolution texels) and
			// the steps a side follow it continuously, and under a texel the point is left open
			const share = min( fullDistance.mul( clamp( dot( N, V ), 0, 1 ) ).div( z.negate() ), 1 ).toVar();
			const steps = clamp( round( share.mul( STEPS ) ), 1, STEPS ).toVar();
			const texels = maxTexels.mul( share ).toVar();
			const fade = clamp( texels.sub( 1 ), 0, 1 ).toVar();
			// the fixed pattern: the slices' rotation and the steps' offset
			const k = bayer4( here ).toVar();
			const turn = k.add( 0.5 ).div( 16 ).toVar();
			const offset = mod( k, 4 ).add( 0.5 ).div( 4 ).toVar();
			const visible = float( 0 ).toVar();
			for ( let s = 0; s < SLICES; s ++ ) {

				const phi = float( s ).add( turn ).mul( Math.PI / SLICES ).toVar();
				const c = cos( phi ).toVar(), sn = sin( phi ).toVar();
				// the slice's direction in view space, and on screen (y down)
				const dirV = vec3( c, sn, 0 );
				const dirS = vec2( c, sn.negate() );
				const ortho = dirV.sub( V.mul( dot( dirV, V ) ) ).toVar();
				const axis = normalize( cross( ortho, V ) ).toVar();
				const projN = N.sub( axis.mul( dot( N, axis ) ) ).toVar();
				const projLen = length( projN ).toVar();
				const cosN = clamp( dot( projN, V ).div( max( projLen, 1e-4 ) ), - 1, 1 ).toVar();
				const n = sign( dot( ortho, projN ) ).mul( acos( cosN ) ).toVar();
				// the horizons' cosines, from the lowest the hemisphere about the normal allows
				const low = [ cos( n.add( Math.PI / 2 ) ).toVar(), cos( n.sub( Math.PI / 2 ) ).toVar() ];
				const horizon = [ low[ 0 ].toVar(), low[ 1 ].toVar() ];
				const half = steps.mul( 0.5 ).toVar();
				const sink = recede.div( steps ).toVar();
				for ( let i = 0; i < STEPS; i ++ ) If( float( i ).lessThan( steps ), () => {

					// evenly spaced along the reach, near to far
					const t = float( i ).add( offset ).div( steps );
					const along = dirS.mul( t.mul( texels ).add( 1 ) ).toVar();
					for ( let e = 0; e < 2; e ++ ) {

						const q = floor( here.add( e === 0 ? along : along.negate() ).add( 0.5 ) ).toVar();
						const d3 = viewAt( q, zAt( q ) ).sub( O ).toVar();
						const len = max( length( d3 ), 1e-4 );
						// the sample's weight falls with how far it stands in front of the point (a
						// culm before the ground counts not at all from `cutoff` on) and over the
						// far half of the steps; a falling one is drawn back to the lowest horizon
						const ahead = clamp( d3.z.div( cutoff ), 0, 1 );
						const late = clamp( float( i ).sub( half ).div( half ), 0, 1 );
						const cs = mix( dot( d3, V ).div( len ), low[ e ], max( late, ahead ) ).toVar();
						// a higher sample raises the horizon; a lower one lets it sink a little toward it
						horizon[ e ].assign( select( cs.greaterThan( horizon[ e ] ), cs, mix( horizon[ e ], cs, sink ) ) );

					}

				} );
				// the horizon angles on either side, within the hemisphere about the normal, and the
				// visible arc between them, cosine-weighted (the paper's closed form)
				const h1 = acos( clamp( horizon[ 0 ], - 1, 1 ) );
				const h0 = acos( clamp( horizon[ 1 ], - 1, 1 ) ).negate();
				const H0 = n.add( clamp( h0.sub( n ), - Math.PI / 2, Math.PI / 2 ) );
				const H1 = n.add( clamp( h1.sub( n ), - Math.PI / 2, Math.PI / 2 ) );
				const sinN = sin( n );
				const arc = ( hh: F ) => cosN.add( hh.mul( 2 ).mul( sinN ) ).sub( cos( hh.mul( 2 ).sub( n ) ) ).mul( 0.25 );
				visible.addAssign( projLen.mul( arc( H0 ).add( arc( H1 ) ) ) );

			}
			// linear in the visibility, over a floor: the deepest crevice keeps `base` of the sky
			ao.assign( mix( float( 1 ), mix( base, float( 1 ), clamp( visible.div( SLICES ), 0, 1 ) ), fade ) );

		} );
		return vec4( ao, max( z, - HALF_MAX ), 0, 0 );

	} )();
	m.name = 'gtao';
	return m;

}

/** One axis of the blur of the slices' ( occlusion, view z ) `raw` over the half-resolution view z
 * `depth` (`dx`, `dy` the axis): five taps with binomial weights ( 1, 4, 6, 4, 1 ) / 16, each
 * weighed by how near its view z lies to the plane of the centre's surface there (`edge`: none from
 * a relative difference of 1 / edge). Writes ( occlusion, view z ). */
export function gtaoBlurMaterial( raw: THREE.Texture, depth: THREE.Texture, dx: number, dy: number ): THREE.NodeMaterial {

	const map = texture( raw );
	const { zAt, viewAt, normalAt } = depthReader( texture( depth ) );
	const last = gtaoView.size.sub( 1 );
	const m = new THREE.NodeMaterial();
	m.fragmentNode = Fn( () => {

		const here = floor( screenCoordinate.xy ).toVar();
		const centre = load( map, here ).toVar();
		const out = centre.x.toVar();
		const z = zAt( here ).toVar();
		If( z.greaterThan( - FAR ), () => {

			const P = viewAt( here, z ).toVar();
			const N = normalAt( here, z, P, normalize( P.negate() ) ).toVar();
			const NP = dot( N, P ).toVar();
			const scale = gtaoUniforms.edge.div( z.negate() ).toVar();
			const sum = float( 0 ).toVar(), weight = float( 0 ).toVar();
			const binomial = [ 1, 4, 6, 4, 1 ];
			for ( let t = - 2; t <= 2; t ++ ) {

				const q = clamp( here.add( vec2( dx * t, dy * t ) ), vec2( 0 ), last ).toVar();
				const ao = t === 0 ? centre.x : load( map, q ).x;
				// the view z the centre's plane has along the tap's ray ( viewAt at z = -1 )
				const ray = viewAt( q, float( - 1 ) ).toVar();
				const den = dot( N, ray ).toVar();
				const predicted = select( abs( den ).greaterThan( 1e-4 ), NP.div( den ).negate(), z );
				const e = t === 0 ? float( 1 ) : clamp( abs( predicted.sub( zAt( q ) ) ).mul( scale ), 0, 1 ).oneMinus();
				const w = e.mul( binomial[ t + 2 ] / 16 ).toVar();
				sum.addAssign( ao.mul( w ) );
				weight.addAssign( w );

			}
			out.assign( sum.div( max( weight, 1e-4 ) ) );

		} );
		return vec4( out, centre.y, 0, 0 );

	} )();
	m.name = dx ? 'gtaoBlurX' : 'gtaoBlurY';
	return m;

}
