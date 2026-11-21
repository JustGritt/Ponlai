import * as THREE from 'three/webgpu';
import { storage } from 'three/tsl';
import { fetchAsset } from '../FetchAsset.ts';

// The real night sky's data: the star catalogue and the Milky Way map, loaded once.
//
// The real night sky's data, loaded once before the first sky bake:
//   stars.bin     the Yale Bright Star Catalogue (9096 stars to about magnitude 6.5), packed by
//                 tools/island/build-star-catalog.mjs
//   milkyway.png  the Milky Way's broad light and dust from NASA SVS's Deep Star Maps 2020
//                 (Gaia DR2), reduced by tools/island/build-milky-way.py
// Both are in the equatorial frame of DayCycle's starFrame: x toward right ascension 0h, y toward
// 6h, z toward the north celestial pole.

// The stars are bucketed by direction into cells on the faces of a cube, so a sky pixel only
// tests the few stars of its own cell. A star near a cell's edge is listed in every cell its
// image can reach (MARGIN radians), so no pixel needs to look at its neighbours.
export const STAR_CELLS = 64;
const MARGIN = 0.006;
// Each cell's word holds start * CELL_SLOTS + count; stars in a cell beyond CELL_SLOTS - 1 (the
// faintest, as the list is sorted by brightness) are dropped.
export const CELL_SLOTS = 64;
// A listed star takes two vec4: direction and peak radiance, then colour and a flicker seed.
// Each star is listed about 2.2 times.
const RECORD_CAPACITY = 24576;

// Brightness of a star image's peak. Magnitudes follow Pogson's law (each is 10^0.4 times
// fainter), here at RANGE of its rate: a magnitude 6.5 star, at the edge of naked-eye visibility,
// gets FAINTEST, and the sky's brightest (Sirius) about 350 times that rather than 1500, so only
// the brightest few dozen reach white and the rest keep their colour.
const FAINTEST = 0.014;
const RANGE = 0.8;
// Star colours as black bodies show them; an art control, 1 for the true chromaticity.
const SATURATION = 1;

function cubeCell( x: number, y: number, z: number ): number {

	// The same face and cell selection as the shader (NightSky.ts cubeCell).
	const ax = Math.abs( x ), ay = Math.abs( y ), az = Math.abs( z );
	let face: number, u: number, v: number, major: number;
	if ( ax >= ay && ax >= az ) { face = x > 0 ? 0 : 1; u = z; v = y; major = ax; }
	else if ( ay >= az ) { face = y > 0 ? 2 : 3; u = x; v = z; major = ay; }
	else { face = z > 0 ? 4 : 5; u = x; v = y; major = az; }
	const cx = Math.min( STAR_CELLS - 1, Math.floor( ( u / major * 0.5 + 0.5 ) * STAR_CELLS ) );
	const cy = Math.min( STAR_CELLS - 1, Math.floor( ( v / major * 0.5 + 0.5 ) * STAR_CELLS ) );
	return ( cy * 6 + face ) * STAR_CELLS + cx;

}

export interface StarGrid { cells: Uint32Array; records: Float32Array; stars: number; entries: number; dropped: number; }

/** Bucket the packed catalogue into the cell and record buffers' data. */
export function buildStarGrid( buffer: ArrayBuffer ): StarGrid {

	const view = new DataView( buffer );
	if ( view.getUint32( 0, true ) !== 0x35435342 ) throw new Error( 'Star catalogue: bad header' );
	const stars = view.getUint32( 4, true );
	const lists: number[][] = Array.from( { length: 6 * STAR_CELLS * STAR_CELLS }, () => [] );
	const directions = new Float32Array( stars * 3 );
	const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), d = new THREE.Vector3(), p = new THREE.Vector3();
	for ( let i = 0; i < stars; i ++ ) {

		const o = 8 + i * 8;
		const ra = view.getUint16( o, true ) / 65536 * 2 * Math.PI;
		const dec = view.getInt16( o + 2, true ) / 32767 * Math.PI / 2;
		d.set( Math.cos( dec ) * Math.cos( ra ), Math.cos( dec ) * Math.sin( ra ), Math.sin( dec ) );
		directions.set( [ d.x, d.y, d.z ], i * 3 );
		// The star's own cell and those of 16 points around it at MARGIN.
		e1.set( - Math.sin( ra ), Math.cos( ra ), 0 );
		e2.crossVectors( d, e1 );
		const cells = new Set( [ cubeCell( d.x, d.y, d.z ) ] );
		for ( let k = 0; k < 16; k ++ ) {

			const a = k / 16 * 2 * Math.PI;
			p.copy( d ).addScaledVector( e1, Math.cos( a ) * MARGIN ).addScaledVector( e2, Math.sin( a ) * MARGIN );
			cells.add( cubeCell( p.x, p.y, p.z ) );

		}
		for ( const cell of cells ) lists[ cell ].push( i );

	}
	const cells = new Uint32Array( lists.length );
	const records = new Float32Array( RECORD_CAPACITY * 8 );
	const capacity = RECORD_CAPACITY;
	let entries = 0, dropped = 0;
	const color = new THREE.Color(), hue = new THREE.Color();
	lists.forEach( ( list, cell ) => {

		const count = Math.min( list.length, CELL_SLOTS - 1, capacity - entries );
		dropped += list.length - count;
		cells[ cell ] = entries * CELL_SLOTS + count;
		for ( let k = 0; k < count; k ++ ) {

			const i = list[ k ], o = 8 + i * 8, r = entries * 8;
			const magnitude = view.getUint8( o + 4 ) / 20 - 2;
			color.setRGB( view.getUint8( o + 5 ) / 255, view.getUint8( o + 6 ) / 255, view.getUint8( o + 7 ) / 255 );
			// Desaturate toward the colour's own luminance, then normalise to unit luminance so
			// the magnitude alone sets how bright the star looks.
			const luminance = color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
			hue.copy( color );
			color.setRGB( luminance, luminance, luminance ).lerp( hue, SATURATION ).multiplyScalar( 1 / luminance );
			records.set( [ directions[ i * 3 ], directions[ i * 3 + 1 ], directions[ i * 3 + 2 ],
				FAINTEST * 10 ** ( 0.4 * RANGE * ( 6.5 - magnitude ) ),
				color.r, color.g, color.b, ( i * 0.6180339887 ) % 1 ], r );
			entries ++;

		}

	} );
	return { cells, records, stars, entries, dropped };

}

// The Milky Way mask: equirectangular, right ascension 0h at the centre and increasing to the
// left, declination +90 on the first row; each byte holds the square root of the band's luminance.
export const MILKY_WAY_WIDTH = 1024;
export const MILKY_WAY_HEIGHT = 512;
export const CELL_COUNT = 6 * STAR_CELLS * STAR_CELLS;

// Storage buffers rather than textures: the water samples the sky with as many textures as a
// fragment shader may bind. `skyWords` holds the cells' start * CELL_SLOTS + count, then the
// mask's bytes four to a word; `starRecords` two vec4 per listed star.
const words = new THREE.StorageBufferAttribute( new Uint32Array( CELL_COUNT + MILKY_WAY_WIDTH * MILKY_WAY_HEIGHT / 4 ), 1 );
const records = new THREE.StorageBufferAttribute( new Float32Array( RECORD_CAPACITY * 8 ), 4 );
export const skyWords = storage( words, 'uint', words.count ).toReadOnly();
export const starRecords = storage( records, 'vec4', records.count ).toReadOnly();

async function loadStars(): Promise<void> {

	const response = await fetchAsset( 'sky/stars.bin' );
	if ( ! response.ok ) throw new Error( `Star catalogue: HTTP ${ response.status }` );
	const grid = buildStarGrid( await response.arrayBuffer() );
	( words.array as Uint32Array ).set( grid.cells );
	( records.array as Float32Array ).set( grid.records );
	words.needsUpdate = records.needsUpdate = true;

}

async function loadMilkyWay(): Promise<void> {

	const response = await fetchAsset( 'sky/milkyway.png' );
	if ( ! response.ok ) throw new Error( `Milky Way mask: HTTP ${ response.status }` );
	// Raw greyscale values: no colour management or premultiplication on the way.
	const bitmap = await createImageBitmap( await response.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' } );
	const context = new OffscreenCanvas( MILKY_WAY_WIDTH, MILKY_WAY_HEIGHT ).getContext( '2d' )!;
	context.drawImage( bitmap, 0, 0 );
	bitmap.close();
	const rgba = context.getImageData( 0, 0, MILKY_WAY_WIDTH, MILKY_WAY_HEIGHT ).data;
	const mask = new Uint8Array( words.array.buffer, CELL_COUNT * 4, MILKY_WAY_WIDTH * MILKY_WAY_HEIGHT );
	for ( let i = 0; i < mask.length; i ++ ) mask[ i ] = rgba[ i * 4 ];
	words.needsUpdate = true;

}

let loading: Promise<void> | null = null;

/** Fill the star and Milky Way buffers; later calls share the first load. */
export function loadNightSkyData(): Promise<void> {

	loading ??= Promise.all( [ loadStars(), loadMilkyWay() ] ).then( () => undefined );
	return loading;

}
