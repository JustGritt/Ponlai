import * as THREE from 'three/webgpu';
import {
	Fn, If, uniform, texture, textureLoad, reference, float, int, vec2, vec3, vec4, ivec2, uv,
	min, max, abs, clamp, cos, sin, dot, select, dFdx, dFdy, floor, fract, mix, smoothstep, step, positionWorld,
	interleavedGradientNoise, vogelDiskSample, screenCoordinate,
} from 'three/tsl';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type B = THREE.Node<'bool'>;

// The sun's soft shadow: a rotated Poisson disk of comparison samples with a fixed penumbra.
//
// The sun's soft shadow: a Poisson disk of POISSON_POINTS points of POISSON_RADIUS map texels' radius,
// turned per screen pixel by a fixed hash (the same every frame, so a paused frame is bit-identical), each
// point one hardware comparison sample (a 2x2 bilinear comparison) whose comparison depth is the
// receiver's moved along its plane to the point (the depth slope from the coordinate's screen
// derivatives). The penumbra is the same width whatever the caster's distance: a contact-hardening
// search (PCSS) varied it with the blocker depth, at the cost of six more loads per receiver and
// penumbrae that hardened and softened as blockers came and went.
// The hardware compares the four texels of a footprint at one depth, so the plane's change inside
// a footprint (up to a texel's worth of slope) is left to the receivers' normal offset
// (SunShadow `offsetCoordinate`): a blocker closer to its receiver than that offset clears is not
// resolved by the map, and the screen-space shadow draws such contacts. Comparing at each texel's
// centre instead (a gather compared by hand) resolved them but cost several milliseconds a frame.
// textureSampleCompare runs only in uniform control flow: the disk is sampled unconditionally and
// a receiver outside the map or not `active` takes 1 by a select afterwards; only a uniform
// condition (the stepped sun's weight) may branch around it.
// three picks one sampler type per depth texture per material, so every sampled read of these maps
// is a comparison: the volume lookups gather with a comparison (`solarVolumeVisibility`); raw
// depths are read with textureLoad, which takes no sampler.
// Filters that take `dFdx` / `dFdy` stay out of `select()` / `If` branches (`installOpenShadow`).

// The disk: POISSON_POINTS points in the unit disk whose least distance apart is the largest
// found, from a fixed seed (`poissonDisk`), scaled to POISSON_RADIUS texels. The disk with its
// points' bilinear footprints spans 2 ( POISSON_RADIUS + 1 ) texels, about 3.5 (some 10 cm on a
// 2048 map of today's island, whose texels are about 3 cm across the sun at 15-22 degrees); its
// 10-90 % edge, the bilinear comparison's one-texel ramp included, is about 1.5 texels, some
// 4-5 cm, under a bamboo culm's width (5-17 cm), so a culm's shadow keeps a dark core metres from
// its foot. At a radius of 2 that edge was about 3.4 texels, some 10 cm. Measured on a lone culm's
// shadow over bare ground, 0.6-3.2 m from its foot, per pixel at 4K, averaged along the shadow:
// with a radius of 2 the key's least visibility across it was 0.45-0.46 (lit / shadow 2.2) and
// its 10-90 % edge 0.22 and 0.12 m at 22 and 15 degrees; with 1, 0.36 at 22 degrees; with 0.75,
// 0.31 and 0.20 (lit / shadow 3.2 and 4.9), edges 0.13 and 0.07 m. In stepped wind clips of the
// island's view the frames' temporal high-pass rose 2-10 % with the regions' contrast, its ratio
// to their temporal deviation unchanged. A wider disk shows its screen-fixed pattern.
const POISSON_POINTS = 8, POISSON_RADIUS = 0.75;

function poissonDisk( count: number, seed: number ): [ number, number ][] {

	// mulberry32: a small seeded stream, so the disk is the same on every load
	let state = seed >>> 0;
	const random = () => {

		state = ( state + 0x6d2b79f5 ) >>> 0;
		let t = state;
		t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
		t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};
	const inDisk = (): [ number, number ] => {

		for ( ;; ) {

			const x = random() * 2 - 1, y = random() * 2 - 1;
			if ( x * x + y * y <= 1 ) return [ x, y ];

		}

	};
	const least = ( points: [ number, number ][] ) => {

		let d = Infinity;
		for ( let i = 0; i < points.length; i ++ ) for ( let j = i + 1; j < points.length; j ++ ) {

			d = Math.min( d, Math.hypot( points[ i ][ 0 ] - points[ j ][ 0 ], points[ i ][ 1 ] - points[ j ][ 1 ] ) );

		}
		return d;

	};
	// the best of many random sets, then moves of single points kept while the least distance grows
	let best = Array.from( { length: count }, inDisk ), bestLeast = least( best );
	for ( let trial = 0; trial < 4096; trial ++ ) {

		const points = Array.from( { length: count }, inDisk ), d = least( points );
		if ( d > bestLeast ) { best = points; bestLeast = d; }

	}
	for ( let move = 0; move < 8192; move ++ ) {

		const step = 0.1 * ( 1 - move / 8192 );
		const i = Math.floor( random() * count );
		const [ x, y ] = [ best[ i ][ 0 ] + ( random() * 2 - 1 ) * step, best[ i ][ 1 ] + ( random() * 2 - 1 ) * step ];
		if ( x * x + y * y > 1 ) continue;
		const moved = best.slice();
		moved[ i ] = [ x, y ];
		const d = least( moved );
		if ( d > bestLeast ) { best = moved; bestLeast = d; }

	}
	return best;

}

const POISSON_DISK = poissonDisk( POISSON_POINTS, 0x50ad15c );

/** The receiver plane's depth slope over the map's coordinate, from the coordinate's screen
 * derivatives (take them in uniform control flow). */
export const receiverGradient = Fn( ( [ dx, dy ]: [ V3, V3 ] ) => {

	const determinant = dx.x.mul( dy.y ).sub( dx.y.mul( dy.x ) ).toVar();
	// Judge the projected plane's conditioning relative to its derivative lengths. An
	// absolute determinant threshold changes with viewport size and camera distance.
	const scale = max( dot( dx.xy, dx.xy ).mul( dot( dy.xy, dy.xy ) ), 1e-30 );
	const condition = determinant.mul( determinant ).div( scale ).toVar();
	const valid = condition.greaterThan( 1e-6 ).and( abs( determinant ).greaterThan( 1e-20 ) );
	const denominator = select( valid, determinant, float( 1 ) );
	const gradient = vec2( dx.z.mul( dy.y ).sub( dy.z.mul( dx.y ) ),
		dy.z.mul( dx.x ).sub( dx.z.mul( dy.x ) ) ).div( denominator );
	return select( valid, clamp( gradient, vec2( - 16 ), vec2( 16 ) ), vec2( 0 ) )
		.mul( smoothstep( 1e-6, 1e-4, condition ) );

} ).setLayout( { name: 'solarShadowReceiverGradient', type: 'vec2', inputs: [
	{ name: 'dx', type: 'vec3' }, { name: 'dy', type: 'vec3' },
] } );
export interface SolarShadowParameters {
	mapSize: V2;
	depthKind?: 'depth' | 'color';
	// false skips the raw depth reads (the result is then fully lit): a map that is not in use
	active?: B;
	// the layer of an array depth texture
	layer?: number;
}

// One map of a filtered lookup: its coordinate and layer.
export interface SolarShadowLayer {
	coordinate: V3;
	active?: B;
	layer?: number;
}

interface SolarShadowSharedParameters {
	mapSize: V2;
	depthKind?: 'depth' | 'color';
}

// Address one layer of an array depth texture; a plain texture passes through.
function inLayer( node: THREE.TextureNode, layer?: number ): THREE.TextureNode {

	return layer === undefined ? node : node.depth( int( layer ) ) as unknown as THREE.TextureNode;

}

// An orthographic ray stays inside its endpoints' bounds. Reject only disjoint
// bounds so a ray crossing a shadow map between two outside endpoints is retained.
export const volumeShadowRayMayIntersect = Fn( ( [ start, end ]: [ V3, V3 ] ) => {

	const low = min( start, end ).toVar(), high = max( start, end ).toVar();
	return low.x.lessThanEqual( 1 ).and( low.y.lessThanEqual( 1 ) ).and( low.z.lessThanEqual( 1 ) )
		.and( high.x.greaterThanEqual( 0 ) ).and( high.y.greaterThanEqual( 0 ) ).and( high.z.greaterThanEqual( 0 ) );

} ).setLayout( { name: 'volumeShadowRayMayIntersect', type: 'bool', inputs: [
	{ name: 'start', type: 'vec3' }, { name: 'end', type: 'vec3' },
] } );

// Volume samples have no receiver plane. Gather the four base-level comparisons without
// derivatives (textureGatherCompare), so depth integration can run in nonuniform control flow, and
// weigh them bilinearly: the hardware comparison's bilinear PCF, through the comparison sampler
// the surface filter binds the map with.
// Keep texture bindings local to the material, as for the surface-shadow factory below.
export function solarVolumeVisibility( depth: THREE.Texture | THREE.TextureNode, coordinate: V3, mapSize: V2, active?: B, layer?: number ): F {

	return Fn( () => {

		const sc = coordinate.toVar();
		let inside = sc.x.greaterThanEqual( 0 ).and( sc.y.greaterThanEqual( 0 ) )
			.and( sc.x.lessThanEqual( 1 ) ).and( sc.y.lessThanEqual( 1 ) )
			.and( sc.z.greaterThanEqual( 0 ) ).and( sc.z.lessThanEqual( 1 ) );
		if ( active ) inside = inside.and( active );
		const visibility = float( 1 ).toVar();
		If( inside, () => {

			const size = max( mapSize, vec2( 1 ) ).toVar();
			const weight = fract( sc.xy.mul( size ).sub( 0.5 ) ).toVar();
			// textureGatherCompare has no implicit LOD or uniform-flow restriction; WGSL returns the
			// texels (0,1), (1,1), (1,0), (0,0) as x, y, z, w, 1 where z <= d (LessEqual)
			const taps = ( inLayer( texture( depth, sc.xy ), layer ).gather().compare( sc.z ) as unknown as THREE.Node<'vec4'> ).toVar();
			visibility.assign( mix( mix( taps.w, taps.z, weight.x ), mix( taps.x, taps.y, weight.x ), weight.y ) );

		} );
		return visibility;

	} )();

}

// three's PCFShadowFilter: five bilinear comparisons on a Vogel disk of one texel's radius, the disk
// turned per pixel by interleaved gradient noise (fixed per pixel, so a paused frame is bit-identical). Each
// comparison gathers its footprint's comparisons (solarVolumeVisibility), so it runs in any control
// flow.
export function solarPCFVisibility( depth: THREE.Texture | THREE.TextureNode, coordinate: V3, mapSize: V2, active?: B, layer?: number ): F {

	return Fn( () => {

		const sc = coordinate.toVar();
		const phi = interleavedGradientNoise( screenCoordinate.xy ).mul( 2 * Math.PI ).toVar();
		const texel = vec2( 1 ).div( max( mapSize, vec2( 1 ) ) ).toVar();
		let sum: F = float( 0 );
		for ( let i = 0; i < 5; i ++ ) {

			const at = vec3( sc.xy.add( ( vogelDiskSample( int( i ), int( 5 ), phi ) as unknown as V2 ).mul( texel ) ), sc.z );
			sum = sum.add( solarVolumeVisibility( depth, at, mapSize, active, layer ) );

		}
		return sum.div( 5 );

	} )();

}

// A soft read over a disc of the receiver's surface, metres wide, for the vertex stage: DISC_TAPS
// gathers on a Vogel spiral at fixed angles, so every vertex takes the same disc and a still scene
// is bit-identical, each gather's four comparisons meaned (at this width a bilinear weight is nothing). A
// point outside the map or the map's depth, or before the first map (`ready` 0), is lit: the far sea
// lies beyond the map's depth range, where the comparison would read shadow and move the meter.
// The bounds are the disc's, not each tap's: the map's rim is empty, so a tap past it reads lit.
// Sixteen gathers, each some 8 cm wide, about 1.3 m apart: at eight, 1.9 m apart, the taps landed on
// the leaves' dapple as noise, and the sea's long thin grid triangles combed it into streaks (the
// streaks' peak to peak .025 of L at eight, .011 at sixteen and at thirty-two).
// `across` and `along` are the disc's radii in the map's coordinate. Plain nodes on purpose, with
// step() and max() for the bounds: an Fn or an If here, in the water's vertex path, comes out of the
// r186 builder with unresolved uniform names (the shadow material, the vertex module).
const DISC_TAPS = 16, GOLDEN_ANGLE = 2.39996323;
const DISC = Array.from( { length: DISC_TAPS }, ( _, i ) => {

	const radius = Math.sqrt( ( i + 0.5 ) / DISC_TAPS ), angle = i * GOLDEN_ANGLE;
	return [ radius * Math.cos( angle ), radius * Math.sin( angle ) ];

} );

export function solarDiscVisibility( depth: THREE.Texture | THREE.TextureNode, coordinate: V3, across: V3, along: V3, ready: F, layer?: number ): F {

	const sc = coordinate.toVar();
	const within = ( x: F ) => step( 0, x ).mul( step( x, 1 ) );
	const inside = within( sc.x ).mul( within( sc.y ) ).mul( within( sc.z ) ).mul( step( 0.5, ready ) );
	let sum: F = float( 0 );
	for ( const [ u, v ] of DISC ) {

		const at = sc.add( across.mul( u ) ).add( along.mul( v ) ).toVar();
		const taps = inLayer( texture( depth, at.xy ), layer ).gather().compare( at.z ) as unknown as THREE.Node<'vec4'>;
		sum = sum.add( dot( taps, vec4( 0.25 ) ) );

	}
	return max( sum.div( DISC_TAPS ), float( 1 ).sub( inside ) );

}

// The disk filter for one map (or one layer of an array) of a depth texture.
// Call inside a Fn: it returns the visibility variable.
function solarShadowLayer( depth: THREE.Texture | THREE.TextureNode, { coordinate, active, layer }: SolarShadowLayer,
	{ mapSize, depthKind = 'depth' }: SolarShadowSharedParameters ): F {

	const sc = coordinate.toVar();
	// Derivatives all execute in uniform control flow. The depth slope carries the comparison
	// depth to each disk point along the receiver's plane, so the disk's width adds no
	// self-shadowing and needs no slope bias (within a point's footprint the normal offset covers it). A receiver plane too flat
	// to matter compares the plain depth; a NaN gradient fails the test too and takes the same path
	// (comparing along it would shadow the pixel outright); a huge but finite one, where the
	// derivatives straddle a jump such as the refracted entry point at the waterline, passes,
	// already clamped by receiverGradient.
	const gradient = receiverGradient( dFdx( sc ), dFdy( sc ) ).toVar();
	const plane = select( max( abs( gradient.x ), abs( gradient.y ) ).greaterThan( 1e-7 ), gradient, vec2( 0 ) ).toVar();
	const size = max( mapSize, vec2( 1 ) ).toVar();
	// the disk's turn: a hash of the screen pixel, fixed per pixel
	const angle = interleavedGradientNoise( screenCoordinate.xy ).mul( 2 * Math.PI ).toVar();
	const turn = vec2( cos( angle ), sin( angle ) ).mul( POISSON_RADIUS ).div( size ).toVar();
	const loadDepth = ( pixel: THREE.Node<'ivec2'> ): F => {

		const loaded = inLayer( textureLoad( depth, pixel ), layer );
		return depthKind === 'color' ? loaded.x : loaded as unknown as F;

	};
	const compareSample = ( q: V2, compareDepth: F ): F => {

		// one hardware comparison: the four texels of the footprint at one comparison depth
		if ( depthKind === 'depth' ) return ( inLayer( texture( depth, q ), layer ).compare( compareDepth ) as unknown as F ).toVar();
		// a colour render target holding depth (the probe's reference): the same semantics in four loads
		const texel = q.mul( size ).sub( 0.5 ).toVar(), base = floor( texel ).toVar();
		const weight = fract( texel ).toVar();
		const compare = ( x: number, y: number ) => {

			const pixel = clamp( base.add( vec2( x, y ) ), vec2( 0 ), size.sub( 1 ) ).toVar();
			return select( compareDepth.lessThanEqual( loadDepth( ivec2( pixel ) ) ), float( 1 ), float( 0 ) );

		};
		return mix( mix( compare( 0, 0 ), compare( 1, 0 ), weight.x ),
			mix( compare( 0, 1 ), compare( 1, 1 ), weight.x ), weight.y ).toVar();

	};
	let inside = sc.x.greaterThanEqual( 0 ).and( sc.y.greaterThanEqual( 0 ) )
		.and( sc.x.lessThanEqual( 1 ) ).and( sc.y.lessThanEqual( 1 ) )
		.and( sc.z.greaterThanEqual( 0 ) ).and( sc.z.lessThanEqual( 1 ) );
	if ( active ) inside = inside.and( active );
	// sampled in uniform control flow whatever the receiver (textureSampleCompare), then selected
	let sum: F = float( 0 );
	for ( const [ px, py ] of POISSON_DISK ) {

		// the point turned and scaled to the map: ( px, py ) rotated by the pixel's angle
		const offset = vec2( turn.x.mul( px ).sub( turn.y.mul( py ) ), turn.y.mul( px ).add( turn.x.mul( py ) ) ).toVar();
		const q = sc.xy.add( offset ).toVar();
		const within = q.x.greaterThanEqual( 0 ).and( q.y.greaterThanEqual( 0 ) )
			.and( q.x.lessThanEqual( 1 ) ).and( q.y.lessThanEqual( 1 ) );
		const sample = compareSample( q, sc.z.add( dot( plane, offset ) ) );
		sum = sum.add( select( within, sample, float( 1 ) ) );

	}
	return select( inside, sum.div( POISSON_POINTS ), float( 1 ) ).toVar();

}

// Texture access stays in a material-local node factory. A shared layout function must never
// close over a texture or uniform binding generated for a different material.
export function solarShadowVisibility( depth: THREE.Texture | THREE.TextureNode, coordinate: V3,
	{ mapSize, depthKind = 'depth', active, layer }: SolarShadowParameters ): F {

	return Fn( () => solarShadowLayer( depth, { coordinate, active, layer }, { mapSize, depthKind } ) )();

}

// Two maps of one depth texture blended by weight (the stepped sun: previous, current), each
// filtered on its own: a map's result never depends on the other, so the cross-fade stays
// continuous across a redraw, when the current map becomes the previous one.
export function solarSteppedShadowVisibility( depth: THREE.Texture | THREE.TextureNode, previous: SolarShadowLayer, current: SolarShadowLayer, weight: F,
	{ mapSize }: { mapSize: V2 } ): F {

	return Fn( () => {

		// textureSampleCompare needs uniform control flow: branch on the uniform weight only
		const before = float( 1 ).toVar();
		If( weight.lessThan( 1 ), () => {

			before.assign( solarShadowLayer( depth, previous, { mapSize } ) );

		} );
		return mix( before, solarShadowLayer( depth, current, { mapSize } ), weight );

	} )();

}

// The filter as a light's own filterNode (the probes' standard receiver).
export function installSolarShadowFilter( sun: THREE.DirectionalLight ): void {

	const mapSize = reference( 'mapSize', 'vec2', sun.shadow );
	const filter = Fn( ( { depthTexture, shadowCoord }: { depthTexture: THREE.DepthTexture; shadowCoord: V3 } ) =>
		solarShadowVisibility( depthTexture, shadowCoord, { mapSize } ) );
	// The runtime exposes filterNode; @types/three has not declared this WebGPU hook yet.
	( sun.shadow as unknown as { filterNode: typeof filter } ).filterNode = filter;

}

type ShadowFilterInputs = { filterFn: ( inputs: object ) => F; shadowCoord: V3; depthTexture: unknown; shadow: unknown; depthLayer: unknown };
type FrustumShadowFilter = ( builder: THREE.NodeBuilder, inputs: ShadowFilterInputs ) => F;

// three.js runs a light's shadow filter only inside its shadow camera's frustum and leaves every
// receiver outside it fully lit. A visibility that covers the whole scene (the clouds') would stop
// at the map's edge if it lived in the filter, so it multiplies the frustum-tested result instead:
// once per receiver, inside the frustum or beyond it.
// three also emits the filter inside the frustum test's branch, where the receiver plane's
// derivatives are undefined for a quad that straddles the test (the refracted entry points at the
// waterline jump in and out of the map): such pixels read whatever the neighbouring lanes left and
// changed with any edit to the shader. The filter is evaluated first, in uniform control flow (it
// returns lit outside the map itself), and only its result is selected. `strength` is the weather's
// share of the map's shadow (Weather.ts `castShadow`, 1 in every sky now); how deep the cast shadows
// read under cloud is the key's visibility's (under cloud the land's floor, look/SkyLook.ts `keyThroughCloud`),
// which multiplies the result.
export function installOpenShadow( sun: THREE.DirectionalLight, visibility: ( position: V3 ) => F, strength: F ): void {

	const node = new THREE.ShadowNode( sun, null ) as unknown as { setupShadowFilter: FrustumShadowFilter };
	node.setupShadowFilter = ( _builder, { filterFn, depthTexture, shadowCoord, shadow, depthLayer } ) => Fn( () => {

		const filtered = filterFn( { depthTexture, shadowCoord, shadow, depthLayer } ).toVar();
		const frustum = shadowCoord.x.greaterThanEqual( 0 ).and( shadowCoord.x.lessThanEqual( 1 ) )
			.and( shadowCoord.y.greaterThanEqual( 0 ) ).and( shadowCoord.y.lessThanEqual( 1 ) )
			.and( shadowCoord.z.lessThanEqual( 1 ) );
		return mix( float( 1 ), select( frustum, filtered, float( 1 ) ), strength ).mul( visibility( positionWorld ) );

	} )();
	sun.shadow.shadowNode = node as unknown as THREE.Node;

}

// QuadMesh temporarily installs its own full-screen vertex node during render. Capture that
// node for compileAsync so the precompiled cache key is identical to the subsequent draw.
async function renderSolarProbeQuad( renderer: THREE.WebGPURenderer, quad: THREE.QuadMesh ): Promise<void> {

	const material = quad.material as THREE.NodeMaterial;
	const before = quad.onBeforeRender, originalVertex = material.vertexNode;
	let fullscreenVertex = originalVertex;
	quad.onBeforeRender = ( ...args ) => {

		fullscreenVertex = material.vertexNode;
		before.apply( quad, args );

	};
	try {

		quad.render( renderer );
		material.vertexNode = fullscreenVertex;
		await renderer.compileAsync( quad, quad.camera );
		material.vertexNode = originalVertex;
		quad.render( renderer );

	} finally {

		quad.onBeforeRender = before;
		material.vertexNode = originalVertex;

	}

}

// Isolated dev-server regression probe. This export is unused by the app and tree-shaken from
// production: it renders a known depth discontinuity without island geometry or post effects.
export async function verifySolarShadow( renderer: THREE.WebGPURenderer ) {

	const width = 512, height = 64, span = 2, depthRange = 16;
	const depthTexture = new THREE.DepthTexture( width, width, THREE.FloatType );
	depthTexture.compareFunction = THREE.LessEqualCompare;
	depthTexture.minFilter = THREE.LinearFilter;
	depthTexture.magFilter = THREE.LinearFilter;
	const field = new THREE.RenderTarget( width, width, { type: THREE.FloatType, depthTexture, depthBuffer: true } );
	const output = new THREE.RenderTarget( width, height, { type: THREE.FloatType, depthBuffer: false } );
	const fieldMaterial = new THREE.NodeMaterial(), outputMaterial = new THREE.NodeMaterial();
	const fieldDepth = select( uv().x.lessThan( 0.5 ), float( 0.25 ), float( 0.95 ) );
	fieldMaterial.fragmentNode = vec4( fieldDepth, fieldDepth, fieldDepth, float( 1 ) );
	fieldMaterial.depthNode = fieldDepth;
	fieldMaterial.toneMapped = false;
	fieldMaterial.depthTest = false;
	fieldMaterial.depthWrite = true;
	fieldMaterial.blending = THREE.NoBlending;
	outputMaterial.depthTest = false;
	outputMaterial.depthWrite = false;
	outputMaterial.toneMapped = false;
	outputMaterial.blending = THREE.NoBlending;
	const sun = new THREE.DirectionalLight();
	sun.shadow.camera.left = - span / 2; sun.shadow.camera.right = span / 2;
	sun.shadow.camera.top = span / 2; sun.shadow.camera.bottom = - span / 2;
	sun.shadow.camera.near = 0; sun.shadow.camera.far = depthRange;
	sun.shadow.mapSize.set( width, width );
	installSolarShadowFilter( sun );
	const standardFilter = ( sun.shadow as unknown as { filterNode: ( input: { depthTexture: THREE.DepthTexture; shadowCoord: V3 } ) => F } ).filterNode;
	const parameters = { mapSize: vec2( width ) };
	outputMaterial.fragmentNode = Fn( () => {

		const q = uv();
		const far = vec3( q, 0.875 ), near = vec3( q, 0.250625 );
		const standard = standardFilter( { depthTexture, shadowCoord: far } );
		const color = solarShadowVisibility( field.texture, far, { ...parameters, depthKind: 'color' } );
		const contact = solarShadowVisibility( depthTexture, near, parameters );
		const bilinear = solarVolumeVisibility( depthTexture, far, vec2( width ) );
		return vec4( standard, color, contact, bilinear );

	} )();
	const fieldQuad = new THREE.QuadMesh( fieldMaterial ), outputQuad = new THREE.QuadMesh( outputMaterial );
	const diagnosticTarget = new THREE.RenderTarget( 4, 1, { type: THREE.FloatType, depthBuffer: false } );
	const diagnosticMaterial = new THREE.NodeMaterial();
	diagnosticMaterial.depthTest = diagnosticMaterial.depthWrite = false;
	diagnosticMaterial.toneMapped = false;
	diagnosticMaterial.blending = THREE.NoBlending;
	diagnosticMaterial.fragmentNode = Fn( () => {

		const q = vec2( uv().x, 0.5 );
		const comparison = ( texture( depthTexture, q ).compare( float( 0.875 ) ) as unknown as F ).toVar();
		const rawDepth = textureLoad( depthTexture, ivec2( q.mul( width ) ) ) as unknown as F;
		const rawColor = texture( field.texture, q ).level( float( 0 ) ).x;
		return vec4( rawDepth, rawColor, comparison, 1 );

	} )();
	const diagnosticQuad = new THREE.QuadMesh( diagnosticMaterial );
	const animation = renderer.getAnimationLoop();
	await renderer.setAnimationLoop( null );
	const state = THREE.RendererUtils.resetRendererState( renderer, {} as THREE.RendererUtils.RendererState );
	const renderReady = ( quad: THREE.QuadMesh ) => renderSolarProbeQuad( renderer, quad );
	try {

		renderer.setScissorTest( false );
		renderer.autoClear = true;
		renderer.setRenderTarget( field );
		await renderReady( fieldQuad );
		renderer.setRenderTarget( diagnosticTarget );
		await renderReady( diagnosticQuad );
		renderer.setRenderTarget( output );
		await renderReady( outputQuad );
		const raw = await renderer.readRenderTargetPixelsAsync( diagnosticTarget, 0, 0, 4, 1 ) as Float32Array;
		const fieldChecks = Array.from( { length: 4 }, ( _, i ) => ( {
			uvX: ( i + 0.5 ) / 4, depth: raw[ i * 4 ], color: raw[ i * 4 + 1 ],
			comparison: raw[ i * 4 + 2 ], drawMarker: raw[ i * 4 + 3 ],
		} ) );
		const pixels = await renderer.readRenderTargetPixelsAsync( output, 0, 0, width, height ) as Float32Array;
		const profiles = Array.from( { length: 4 }, () => new Array<number>( width ).fill( 0 ) );
		let maximumDifference = 0;
		for ( let k = 0; k < width * height; k ++ ) {

			for ( let c = 0; c < 4; c ++ ) {

				const value = pixels[ k * 4 + c ];
				if ( ! Number.isFinite( value ) || value < 0 || value > 1.00001 ) throw new Error( 'Solar shadow probe returned an invalid visibility' );
				const row = Math.floor( k / width );
				if ( row >= height / 4 && row < height * 3 / 4 ) profiles[ c ][ k % width ] += value / ( height / 2 );

			}
			maximumDifference = Math.max( maximumDifference, Math.abs( pixels[ k * 4 ] - pixels[ k * 4 + 1 ] ) );

		}
		// the 10-90 % width of each profile's edge in texels (one sample per texel), from the crossings
		// interpolated between samples, over the interior only: a disk point beyond the map's border
		// reads lit, which is no penumbra
		const crossing = ( p: number[], level: number ) => {

			for ( let x = width / 4 + 1; x < width * 3 / 4; x ++ ) {

				if ( p[ x ] >= level ) return x - 1 + ( level - p[ x - 1 ] ) / Math.max( p[ x ] - p[ x - 1 ], 1e-9 );

			}
			return NaN;

		};
		const widths = profiles.map( ( p ) => crossing( p, 0.9 ) - crossing( p, 0.1 ) );
		const profileChecks = profiles.map( ( p ) => ( {
			minimum: Math.min( ...p ), maximum: Math.max( ...p ), first: p[ 0 ], last: p[ width - 1 ],
			center: p.slice( width / 2 - 4, width / 2 + 4 ),
		} ) );
		const diagnostics = { widths, fieldChecks, profileChecks, maximumDifference };
		for ( let i = 0; i < 4; i ++ ) {

			const expectedDepth = i < 2 ? 0.25 : 0.95, expectedComparison = i < 2 ? 0 : 1;
			const check = fieldChecks[ i ];
			if ( Math.abs( check.depth - expectedDepth ) > 1e-4 || Math.abs( check.color - expectedDepth ) > 1e-4 ||
				Math.abs( check.comparison - expectedComparison ) > 1e-4 || check.drawMarker < 0.99999 ) {

				throw new Error( `Solar shadow test field was not rendered correctly: ${ JSON.stringify( diagnostics ) }` );

			}

		}
		// the disk's penumbra: as wide for a caster at the contact as for one 10 m away (within a tenth
		// of a texel, so a contact edge hardened to half fails), no wider than the disk and its
		// bilinear ramp ( 1 + 2 POISSON_RADIUS texels), and at least a quarter texel wider than the
		// bilinear comparison alone, whose ramp is at most a texel
		if ( ! widths.every( Number.isFinite ) || widths[ 3 ] > 1 || widths[ 0 ] < widths[ 3 ] + 0.25 ||
			widths[ 0 ] > 1 + 2 * POISSON_RADIUS || Math.abs( widths[ 2 ] - widths[ 0 ] ) > 0.1 ) {

			throw new Error( `Solar disk filter width failed: ${ JSON.stringify( diagnostics ) }` );

		}
		if ( maximumDifference > 0.01 ) throw new Error( `Colour-depth and standard shadow samplers disagree by ${ maximumDifference }` );
		for ( const profile of profiles ) {

			// Samples outside the map are intentionally lit. Test interior regions beyond the
			// filter support, so the map border cannot contaminate the umbra assertion.
			if ( profile.slice( width / 4 - 50, width / 4 + 50 ).some( ( v ) => v > 1e-5 ) ||
				profile.slice( width * 3 / 4 - 50, width * 3 / 4 + 50 ).some( ( v ) => v < 0.99999 ) ) {

				throw new Error( `Solar shadow probe lost its interior umbra or lit region: ${ JSON.stringify( diagnostics ) }` );

			}

		}
		return { ...diagnostics, diskRadiusTexels: POISSON_RADIUS,
			farReceiverGapMetres: 10, contactReceiverGapMetres: 0.01, width, height, profiles };

	} finally {

		THREE.RendererUtils.restoreRendererState( renderer, state );
		field.dispose(); output.dispose(); diagnosticTarget.dispose();
		fieldMaterial.dispose(); outputMaterial.dispose(); diagnosticMaterial.dispose();
		// QuadMesh owns shared geometry; the test must leave it for other render passes.
		await renderer.setAnimationLoop( animation );

	}

}

// A clear tilted depth field, looked up a normal offset away as the game's receivers are, must stay
// fully lit at every subtexel phase. The reference native PCF, without the offset, proves that the
// fixture exercises the receiver plane's rise inside a footprint. It is the hardware comparison
// sampler alone, drawn by a material of its own; the gathered volume lookup (solarVolumeVisibility)
// must match it.
// The hardware compares a footprint's four texels at one depth, so a blocker nearer its receiver
// than the normal offset clears plus one footprint's rise is not resolved by the map: such contacts
// (a 3 cm blocker on this plane) are the screen-space shadow's. The blocker here stands 3 cm beyond
// that reach and must cast a full umbra with its edge within a texel of where the offset lookup
// puts it.
// Filtering hardware may hold the bilinear weights to 8 bits per axis: up to 1/256 on each. The
// same bounds the surface filter's comparisons against the colour copy's hand-weighted ones.
const VOLUME_REFERENCE_TOLERANCE = 2 / 255;
export async function verifySolarShadowRamp( renderer: THREE.WebGPURenderer ) {

	const mapSize = 256, width = 512, height = 32, span = 8, depthRange = 219;
	const depthTexture = new THREE.DepthTexture( mapSize, mapSize, THREE.FloatType );
	depthTexture.compareFunction = THREE.LessEqualCompare;
	depthTexture.minFilter = depthTexture.magFilter = THREE.LinearFilter;
	const field = new THREE.RenderTarget( mapSize, mapSize, { type: THREE.FloatType, depthTexture } );
	const output = new THREE.RenderTarget( width, height, { type: THREE.FloatType, depthBuffer: false } );
	const reference = new THREE.RenderTarget( width, height, { type: THREE.FloatType, depthBuffer: false } );
	const fieldMaterial = new THREE.NodeMaterial(), outputMaterial = new THREE.NodeMaterial(), referenceMaterial = new THREE.NodeMaterial();
	for ( const material of [ fieldMaterial, outputMaterial, referenceMaterial ] ) {

		material.depthTest = false;
		material.depthWrite = material === fieldMaterial;
		material.toneMapped = false;
		material.blending = THREE.NoBlending;

	}
	const gradient = uniform( new THREE.Vector2() ), phase = uniform( new THREE.Vector2() ), gap = uniform( 0 );
	// the receivers' normal offset as the game applies it (SunShadow offsetCoordinate), in this
	// map's coordinate: the plane's normal toward the light, NORMAL_OFFSET.near texels long (the
	// eye's end, the smallest), loaded here so this module keeps no import of SunShadow
	const { NORMAL_OFFSET } = await import( './SunShadow.ts' );
	const shift = uniform( new THREE.Vector3() );
	const planeDepth = ( q: V2 ) => float( 0.5 ).add( dot( gradient, q.sub( 0.5 ) ) );
	const fieldDepth = planeDepth( uv() ).sub( select( uv().x.lessThan( 0.5 ), gap, float( 0 ) ) );
	fieldMaterial.fragmentNode = vec4( fieldDepth, fieldDepth, fieldDepth, float( 1 ) );
	fieldMaterial.depthNode = fieldDepth;
	const sun = new THREE.DirectionalLight();
	sun.shadow.camera.left = - span / 2; sun.shadow.camera.right = span / 2;
	sun.shadow.camera.top = span / 2; sun.shadow.camera.bottom = - span / 2;
	sun.shadow.camera.near = 0; sun.shadow.camera.far = depthRange;
	sun.shadow.mapSize.set( mapSize, mapSize );
	installSolarShadowFilter( sun );
	const standardFilter = ( sun.shadow as unknown as { filterNode: ( input: { depthTexture: THREE.DepthTexture; shadowCoord: V3 } ) => F } ).filterNode;
	const parameters = { mapSize: vec2( mapSize ) };
	const probeCoordinate = () => uv().mul( vec2( 0.4, 0.2 ) ).add( vec2( 0.3, 0.4 ) ).add( phase.div( mapSize ) ).toVar();
	// Only 1 cm of physical ray clearance besides the normal offset.
	const probeDepth = ( q: V2 ) => planeDepth( q ).sub( 0.01 / depthRange ).toVar();
	// Hardware PCF semantics: one comparison depth for the four texels, from the comparison
	// sampler. One material cannot bind the depth texture with two sampler types, so this one
	// holds only the comparison.
	referenceMaterial.fragmentNode = Fn( () => {

		const q = probeCoordinate();
		return vec4( texture( depthTexture, q ).compare( probeDepth( q ) ) as unknown as F, 0, 0, 1 );

	} )();
	outputMaterial.fragmentNode = Fn( () => {

		const q = probeCoordinate();
		const receiverDepth = probeDepth( q );
		// The same semantics as the volume lookup reads them (gathered and compared by hand),
		// checked against the reference above.
		const native = solarVolumeVisibility( depthTexture, vec3( q, receiverDepth ), vec2( mapSize ) ).toVar();
		const coordinate = vec3( q, receiverDepth ).add( shift );
		const standard = standardFilter( { depthTexture, shadowCoord: coordinate } );
		const color = solarShadowVisibility( field.texture, coordinate, { ...parameters, depthKind: 'color' } );
		return vec4( standard, color, native, float( 1 ) );

	} )();
	const fieldQuad = new THREE.QuadMesh( fieldMaterial ), outputQuad = new THREE.QuadMesh( outputMaterial );
	const referenceQuad = new THREE.QuadMesh( referenceMaterial );
	const animation = renderer.getAnimationLoop();
	await renderer.setAnimationLoop( null );
	const state = THREE.RendererUtils.resetRendererState( renderer, {} as THREE.RendererUtils.RendererState );
	const cases: { elevation: number; slopeDegrees: number; phase: number[]; clearMinimum: number[];
		nativeClearMinimum: number; volumeClearMinimum: number; volumeReferenceDifference: number;
		contactMaximumUmbra: number[]; contactMinimumLit: number[];
		contactWidthMetres: number[]; contactEdgeErrorMetres: number[]; maximumDifference: number; blockerGapMetres: number }[] = [];
	const failures: typeof cases = [];
	let ready = false;
	try {

		renderer.setScissorTest( false );
		renderer.autoClear = true;
		for ( const elevation of [ 32, 10 ] ) for ( const slopeDegrees of [ 0, 37 ] ) for ( const subpixel of [
			[ 0.125, 0.375 ], [ 0.375, 0.625 ], [ 0.625, 0.875 ], [ 0.875, 0.125 ],
		] ) {

			const slope = span / ( depthRange * Math.tan( THREE.MathUtils.degToRad( elevation ) ) );
			const angle = THREE.MathUtils.degToRad( slopeDegrees );
			gradient.value.set( Math.cos( angle ) * slope, Math.sin( angle ) * slope );
			// the plane rises toward the light along -gradient: its unit normal toward the light is
			// ( cos e * dir, -sin e ) in metres across and along the map
			const sunward = THREE.MathUtils.degToRad( elevation ), offsetMetres = NORMAL_OFFSET.near * span / mapSize;
			// the blocker: 3 cm beyond the clearance the offset gives ( d / sin e ) plus one footprint's
			// rise ( cot e * ( |cos a| + |sin a| ) texels ), the depth the map leaves to the screen
			const reachMetres = offsetMetres / Math.sin( sunward ) +
				span / mapSize / Math.tan( sunward ) * ( Math.abs( Math.cos( angle ) ) + Math.abs( Math.sin( angle ) ) );
			const blockerMetres = reachMetres + 0.03;
			shift.value.set( Math.cos( angle ) * Math.cos( sunward ) * offsetMetres / span,
				Math.sin( angle ) * Math.cos( sunward ) * offsetMetres / span, - Math.sin( sunward ) * offsetMetres / depthRange );
			phase.value.set( subpixel[ 0 ], subpixel[ 1 ] );
			const clearMinimum = [ 1, 1 ], maximumUmbra = [ 0, 0 ], minimumLit = [ 1, 1 ];
			let nativeMinimum = 1, volumeMinimum = 1, volumeDifference = 0, maximumDifference = 0;
			const profiles = [ new Array<number>( width ).fill( 0 ), new Array<number>( width ).fill( 0 ) ];
			for ( const contact of [ false, true ] ) {

				gap.value = contact ? blockerMetres / depthRange : 0;
				renderer.setRenderTarget( field );
				if ( ! ready ) await renderSolarProbeQuad( renderer, fieldQuad );
				else fieldQuad.render( renderer );
				renderer.setRenderTarget( output );
				if ( ! ready ) await renderSolarProbeQuad( renderer, outputQuad );
				else outputQuad.render( renderer );
				renderer.setRenderTarget( reference );
				if ( ! ready ) await renderSolarProbeQuad( renderer, referenceQuad );
				else referenceQuad.render( renderer );
				ready = true;
				const pixels = await renderer.readRenderTargetPixelsAsync( output, 0, 0, width, height ) as Float32Array;
				const hardware = await renderer.readRenderTargetPixelsAsync( reference, 0, 0, width, height ) as Float32Array;
				for ( let k = 0; k < width * height; k ++ ) {

					const x = k % width, y = Math.floor( k / width );
					const qx = ( x + 0.5 ) / width * 0.4 + 0.3 + subpixel[ 0 ] / mapSize;
					if ( pixels[ k * 4 + 3 ] < 0.99999 || hardware[ k * 4 + 3 ] < 0.99999 ) throw new Error( 'Solar slope fixture did not draw' );
					const native = hardware[ k * 4 ];
					if ( ! Number.isFinite( native ) || native < 0 || native > 1.00001 ) throw new Error( 'Solar slope reference contains an invalid visibility' );
					if ( ! contact ) nativeMinimum = Math.min( nativeMinimum, native );
					volumeDifference = Math.max( volumeDifference, Math.abs( pixels[ k * 4 + 2 ] - native ) );
					for ( let c = 0; c < 3; c ++ ) {

						const value = pixels[ k * 4 + c ];
						if ( ! Number.isFinite( value ) || value < 0 || value > 1.00001 ) throw new Error( 'Solar slope fixture contains an invalid visibility' );
						if ( ! contact ) {

							if ( c < 2 ) clearMinimum[ c ] = Math.min( clearMinimum[ c ], value );
							else volumeMinimum = Math.min( volumeMinimum, value );

						} else if ( c < 2 ) {

							if ( qx < 0.48 ) maximumUmbra[ c ] = Math.max( maximumUmbra[ c ], value );
							if ( qx > 0.52 ) minimumLit[ c ] = Math.min( minimumLit[ c ], value );
							if ( y >= height / 4 && y < height * 3 / 4 ) profiles[ c ][ x ] += value / ( height / 2 );

						}

					}
					maximumDifference = Math.max( maximumDifference, Math.abs( pixels[ k * 4 ] - pixels[ k * 4 + 1 ] ) );

				}

			}
			const contactWidthMetres = profiles.map( ( p ) => p.filter( ( v ) => v > 0.1 && v < 0.9 ).length * span * 0.4 / width );
			const contactEdgeErrorMetres = profiles.map( ( p ) => {

				let nearest = 0;
				for ( let x = 1; x < width; x ++ ) if ( Math.abs( p[ x ] - 0.5 ) < Math.abs( p[ nearest ] - 0.5 ) ) nearest = x;
				const q = ( nearest + 0.5 ) / width * 0.4 + 0.3 + subpixel[ 0 ] / mapSize;
				// the offset lookup reads the map shift.x further along: the edge is expected there
				return Math.abs( q + shift.value.x - 0.5 ) * span;

			} );
			const result = { elevation, slopeDegrees, phase: subpixel, clearMinimum, nativeClearMinimum: nativeMinimum,
				volumeClearMinimum: volumeMinimum, volumeReferenceDifference: volumeDifference, contactMaximumUmbra: maximumUmbra,
				contactMinimumLit: minimumLit, contactWidthMetres, contactEdgeErrorMetres, maximumDifference, blockerGapMetres: blockerMetres };
			cases.push( result );
			// The gathered volume lookup must be the hardware comparison's bilinear PCF, to within the
			// filtering hardware's sub-texel weight precision; a misplaced texel differs by a quarter
			// or more somewhere along the step.
			if ( clearMinimum.some( ( v ) => v < 0.99999 ) || nativeMinimum > 0.95 || volumeDifference > VOLUME_REFERENCE_TOLERANCE ||
				maximumUmbra.some( ( v ) => v > 1e-5 ) || minimumLit.some( ( v ) => v < 0.99999 ) ||
				contactWidthMetres.some( ( v ) => v > ( 2 * POISSON_RADIUS + 1.5 ) * span / mapSize ) ||
				contactEdgeErrorMetres.some( ( v ) => v > span / mapSize ) || maximumDifference > VOLUME_REFERENCE_TOLERANCE ) {

				failures.push( result );

			}

		}
		if ( failures.length ) throw new Error( `Solar receiver-plane interpolation failed: ${ JSON.stringify( failures ) }` );
		// depth fetches per receiver: one hardware comparison (a 2x2 footprint) per disk point
		return { mapSize, width, height, texelMetres: span / mapSize, physicalClearanceMetres: 0.01,
			normalOffsetTexels: NORMAL_OFFSET.near, contactBlockerBeyondReachMetres: 0.03, diskPoints: POISSON_POINTS, diskRadiusTexels: POISSON_RADIUS,
			depthFetches: POISSON_POINTS, cases };

	} finally {

		THREE.RendererUtils.restoreRendererState( renderer, state );
		field.dispose(); output.dispose(); reference.dispose();
		fieldMaterial.dispose(); outputMaterial.dispose(); referenceMaterial.dispose();
		await renderer.setAnimationLoop( animation );

	}

}
