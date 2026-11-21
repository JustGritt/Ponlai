// Deterministic JS noise for the world generator (runs in the mesh worker).
//   noise3( x, y, z )      improved Perlin, ~[-1, 1]
//   fbm2( x, y, oct )      fractal sum, ~[-1, 1]
//   hash2i / hash3i        integer hashes to [0, 1)
//   mulberry32( seed )     small seeded RNG

export type Rng = () => number;

export function mulberry32( seed: number ): Rng {

	let a = seed >>> 0;
	return function () {

		a = ( a + 0x6D2B79F5 ) >>> 0;
		let t = a;
		t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
		t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

const P = new Uint8Array( 512 );
{

	const rng = mulberry32( 0x5eed );
	const p = Array.from( { length: 256 }, ( _, i ) => i );
	for ( let i = 255; i > 0; i -- ) {

		const j = Math.floor( rng() * ( i + 1 ) );
		[ p[ i ], p[ j ] ] = [ p[ j ], p[ i ] ];

	}

	for ( let i = 0; i < 512; i ++ ) P[ i ] = p[ i & 255 ];

}

const fade = ( t: number ) => t * t * t * ( t * ( t * 6 - 15 ) + 10 );

function grad( h: number, x: number, y: number, z: number ): number {

	const u = h < 8 ? x : y;
	const v = h < 4 ? y : ( h === 12 || h === 14 ? x : z );
	return ( ( h & 1 ) ? - u : u ) + ( ( h & 2 ) ? - v : v );

}

export function noise3( x: number, y: number, z: number ): number {

	const xi = Math.floor( x ), yi = Math.floor( y ), zi = Math.floor( z );
	const X = xi & 255, Y = yi & 255, Z = zi & 255;
	x -= xi; y -= yi; z -= zi;
	const u = fade( x ), v = fade( y ), w = fade( z );
	const A = P[ X ] + Y, AA = P[ A ] + Z, AB = P[ A + 1 ] + Z;
	const B = P[ X + 1 ] + Y, BA = P[ B ] + Z, BB = P[ B + 1 ] + Z;
	const x1 = x - 1, y1 = y - 1, z1 = z - 1;
	// Each corner's gradient once (the lerps below used to evaluate every one of them twice); the
	// arithmetic is unchanged, so the field stays bit-identical.
	const gAA = grad( P[ AA ] & 15, x, y, z ), gBA = grad( P[ BA ] & 15, x1, y, z );
	const gAB = grad( P[ AB ] & 15, x, y1, z ), gBB = grad( P[ BB ] & 15, x1, y1, z );
	const gAA1 = grad( P[ AA + 1 ] & 15, x, y, z1 ), gBA1 = grad( P[ BA + 1 ] & 15, x1, y, z1 );
	const gAB1 = grad( P[ AB + 1 ] & 15, x, y1, z1 ), gBB1 = grad( P[ BB + 1 ] & 15, x1, y1, z1 );
	const l1 = gAA + u * ( gBA - gAA );
	const l2 = gAB + u * ( gBB - gAB );
	const l3 = gAA1 + u * ( gBA1 - gAA1 );
	const l4 = gAB1 + u * ( gBB1 - gAB1 );
	const m1 = l1 + v * ( l2 - l1 );
	const m2 = l3 + v * ( l4 - l3 );
	return m1 + w * ( m2 - m1 );

}

function fbm3( x: number, y: number, z: number, oct = 4 ): number {

	let s = 0, a = 0.5, n = 0;
	for ( let i = 0; i < oct; i ++ ) {

		s += a * noise3( x, y, z );
		n += a;
		x = x * 2.03 + 17.1; y = y * 2.03 + 3.7; z = z * 2.03 + 9.2;
		a *= 0.5;

	}

	return s / n;

}

export function fbm2( x: number, y: number, oct = 4 ): number {

	return fbm3( x, y, 0.37, oct );

}

function hash2i( i: number, j: number ): number {

	let h = Math.imul( i | 0, 374761393 ) + Math.imul( j | 0, 668265263 );
	h = Math.imul( h ^ ( h >>> 13 ), 1274126177 );
	return ( ( h ^ ( h >>> 16 ) ) >>> 0 ) / 4294967296;

}

export function hash3i( i: number, j: number, k: number ): number {

	return hash2i( i + Math.imul( k | 0, 1013 ), j - Math.imul( k | 0, 7919 ) );

}

export const smoothstep = ( a: number, b: number, x: number ): number => {

	const t = Math.min( Math.max( ( x - a ) / ( b - a ), 0 ), 1 );
	return t * t * ( 3 - 2 * t );

};

// polynomial smooth max
export function smax( a: number, b: number, k: number ): number {

	const h = Math.max( k - Math.abs( a - b ), 0 ) / k;
	return Math.max( a, b ) + h * h * k * 0.25;

}
