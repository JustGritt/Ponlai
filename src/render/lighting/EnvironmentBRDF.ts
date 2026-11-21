import * as THREE from 'three/webgpu';
import { texture, vec2, clamp, float } from 'three/tsl';

type F = THREE.Node<'float'>;

// The environment's specular BRDF for the shared GGX, by Karis's split sum, as a table.
//
// The environment's specular BRDF for the shared GGX (lighting/SurfaceBRDF.ts), by Karis's split
// sum (Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013): environment specular = the
// prefiltered environment x ( F0 A + F90 B ), A and B integrated over the same D and visibility the
// direct light takes, so the two highlights agree in roughness: alpha = ( 1 - g ) squared and the
// Schlick-GGX visibility. Three's DFG table takes the same alpha but its own Smith visibility.
// The table: SIZE x SIZE, RG half floats, gloss g along u, N.V along v over [ NV_MIN, 1 ] (an edge
// at N.V = 0 would hold nothing and blacken a ring at grazing angles). A and B are smooth, and
// bilinear lookups of 32 x 32 texels stay within a percent of them. It is integrated on the CPU at
// load (SAMPLES Hammersley points per texel, GGX importance sampled, a few milliseconds): it
// depends on the code alone, so it needs no asset and no GPU pass.

const SIZE = 32;
const NV_MIN = 1 / 64;
const SAMPLES = 512;

function radicalInverse( i: number ): number {

	let bits = i;
	bits = ( ( bits << 16 ) | ( bits >>> 16 ) ) >>> 0;
	bits = ( ( ( bits & 0x55555555 ) << 1 ) | ( ( bits & 0xAAAAAAAA ) >>> 1 ) ) >>> 0;
	bits = ( ( ( bits & 0x33333333 ) << 2 ) | ( ( bits & 0xCCCCCCCC ) >>> 2 ) ) >>> 0;
	bits = ( ( ( bits & 0x0F0F0F0F ) << 4 ) | ( ( bits & 0xF0F0F0F0 ) >>> 4 ) ) >>> 0;
	bits = ( ( ( bits & 0x00FF00FF ) << 8 ) | ( ( bits & 0xFF00FF00 ) >>> 8 ) ) >>> 0;
	return bits / 4294967296;

}

/** A and B for a gloss and N.V (V in the xz plane, N = +z). */
function integrate( g: number, NV: number ): [ number, number ] {

	const r = Math.max( 1 - g, 1e-3 );
	const alpha = r * r, a2 = alpha * alpha, k = alpha / 2;
	const Vx = Math.sqrt( 1 - NV * NV ), Vz = NV;
	let A = 0, B = 0;
	for ( let i = 0; i < SAMPLES; i ++ ) {

		const u = ( i + 0.5 ) / SAMPLES, v = radicalInverse( i );
		const phi = 2 * Math.PI * u;
		const cosT = Math.sqrt( ( 1 - v ) / ( 1 + ( a2 - 1 ) * v ) ), sinT = Math.sqrt( 1 - cosT * cosT );
		const Hx = sinT * Math.cos( phi ), Hz = cosT;
		const VH = Vx * Hx + Vz * Hz;
		const Lz = 2 * VH * Hz - Vz;
		if ( Lz <= 0 || VH <= 0 ) continue;
		const NL = Lz, NH = Hz;
		// f N.L / pdf, pdf = D N.H / ( 4 V.H ), f = F D vis: the visibility's 1 / 4 cancels the pdf's 4
		const vis = 1 / ( ( NL * ( 1 - k ) + k ) * ( NV * ( 1 - k ) + k ) );
		const w = vis * NL * VH / NH;
		const Fc = Math.pow( 1 - VH, 5 );
		A += ( 1 - Fc ) * w;
		B += Fc * w;

	}
	return [ A / SAMPLES, B / SAMPLES ];

}

let table: THREE.DataTexture | null = null;

function bake(): THREE.DataTexture {

	const data = new Uint16Array( SIZE * SIZE * 2 );
	for ( let j = 0; j < SIZE; j ++ ) for ( let i = 0; i < SIZE; i ++ ) {

		const [ A, B ] = integrate( i / ( SIZE - 1 ), NV_MIN + ( 1 - NV_MIN ) * j / ( SIZE - 1 ) );
		data[ ( j * SIZE + i ) * 2 ] = THREE.DataUtils.toHalfFloat( A );
		data[ ( j * SIZE + i ) * 2 + 1 ] = THREE.DataUtils.toHalfFloat( B );

	}
	const t = new THREE.DataTexture( data, SIZE, SIZE, THREE.RGFormat, THREE.HalfFloatType );
	t.name = 'EnvironmentBRDF';
	t.minFilter = THREE.LinearFilter;
	t.magFilter = THREE.LinearFilter;
	t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
	t.generateMipmaps = false;
	t.needsUpdate = true;
	return t;

}

/** ( A, B ) of the split sum for gloss g and N.V: the environment's specular is
 * prefiltered x ( F0 A + F90 B ). */
export function environmentBRDF( g: F, NV: F ): THREE.Node<'vec2'> {

	table ??= bake();
	const v = clamp( NV, NV_MIN, 1 ).sub( NV_MIN ).div( 1 - NV_MIN );
	const uv = vec2( clamp( g, 0, 1 ), v ).mul( ( SIZE - 1 ) / SIZE ).add( 0.5 / SIZE );
	return texture( table, uv ).level( float( 0 ) ).rg;

}
