import * as THREE from 'three/webgpu';
import type { Pipeline } from '../post/Pipeline.ts';
import { preExposure } from '../post/PreExposure.ts';
import { FIT_CLASSES } from './Classes.ts';
import { SEA_STEP } from '../weather/VolumetricHaze.ts';

// What each cell of a radiance grid shows besides its radiance: depth and the other channels.
//
// What each cell of a grid shows besides its radiance (debug builds, through debug/Radiance.ts:
// `__PONLAI__.radiance( { grid: [ nx, ny ], cell: n, aov } )`), for a script that reads cells,
// which reads every page it compares on one grid of cells. A cell is the n by n texels at its centre (the
// render's texels; n = 0 the whole cell), the same texels Radiance.ts's grid averages with `cell`.
// Per cell, `values` holds `k` numbers, row by row from the top-left:
//   depth   ( view depth in metres of the nearest surface, the sea's included, averaged over the
//           cell's texels that are not sky (0 if all are), the share of its texels that are sky,
//           the share that are sea ): the sky is the nearest surface's view z beyond -9e4
//           (Pipeline.ts `surfaceDepth`), the sea a nearest surface in front of the opaque pass's
//           (`linearDepth`, its sample 0) by more than 1 cm.
//   normal  ( the world normal x, y, z ): the mean of the texels' normals, each the cross product
//           of the view positions to its neighbours along x and y (on each axis the neighbour with
//           the smaller step in depth, so an edge takes the surface it lies on), turned to face the
//           camera, normalised; ( 0, 0, 0 ) where every texel is sky. The camera's matrices are
//           read when called (the frame drawn last must have been drawn with them: a paused page).
//   ao      ( the screen-space occlusion's visibility, 1 open ): post/GTAO.ts's output at half
//           size, sampled at the cell's texels' halves; 1 where the app builds no GTAO.
//   class   ( the kind's index in debug/Classes.ts FIT_CLASSES, or -1 ): on a page with
//           `flags=classes`, the kind most of the cell's texels decode to, -1 when no kind holds
//           more than half of them; the sky and the sea by depth (as `depth` above), the rest from
//           the opaque colour's codes. `texels: true` (cell n > 0) also returns `texels`, each of
//           the cells' texels' own kind, cell by cell, n by n per cell in rows.
// `rings: { sun, step, count, sky }` (readRings below) returns instead the luminance around the sun
// in rings of angle, for the bloom's and the glare's profile; `sector` and `elevation` keep only
// the texels in a sector about the light or a range of elevation, and `elevationStep` adds the
// sky's sums per ring and elevation band, from which a script takes the excess over a background
// that varies with elevation (a closed deck's, brighter overhead) texel by texel. Any light's
// direction serves as `sun` (the moon's by night).

export type Aov = 'depth' | 'normal' | 'ao' | 'class';
export interface AovResult { aov: Aov; nx: number; ny: number; k: number; size: [ number, number ]; values: number[]; texels?: number[] }

const round = ( x: number, digits = 6 ): number => Number( x.toPrecision( digits ) );

/** The texels of each cell's centre: calls `visit( cell, x, y )` for the n by n texels at the
 * centre of each of the nx by ny cells of a width by height image (n = 0: every texel). */
export function forCellTexels( nx: number, ny: number, width: number, height: number, n: number, visit: ( cell: number, x: number, y: number ) => void ): void {

	if ( n <= 0 ) {

		for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) visit( Math.floor( y * ny / height ) * nx + Math.floor( x * nx / width ), x, y );
		return;

	}
	const lo = Math.floor( ( n - 1 ) / 2 );
	for ( let j = 0; j < ny; j ++ ) for ( let i = 0; i < nx; i ++ ) {

		const cx = Math.floor( ( i + 0.5 ) * width / nx ) - lo, cy = Math.floor( ( j + 0.5 ) * height / ny ) - lo;
		for ( let dy = 0; dy < n; dy ++ ) for ( let dx = 0; dx < n; dx ++ ) {

			const x = Math.min( width - 1, Math.max( 0, cx + dx ) ), y = Math.min( height - 1, Math.max( 0, cy + dy ) );
			visit( j * nx + i, x, y );

		}

	}

}

/** A render target's texture read back whole: ( x, y, channel ) to its stored value. */
export async function readTexture( renderer: THREE.WebGPURenderer, target: THREE.RenderTarget, index: number ): Promise<{ width: number; height: number; at: ( x: number, y: number, c?: number ) => number }> {

	const texture = target.textures[ index ];
	const width = target.width, height = target.height;
	const raw = await renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height, index );
	const half = texture.type === THREE.HalfFloatType;
	const channels = texture.format === THREE.RedFormat ? 1 : texture.format === THREE.RGFormat ? 2 : 4, bytes = half ? 2 : 4;
	// rows come back padded to 256 bytes (three's copyTextureToBuffer)
	const stride = Math.ceil( width * channels * bytes / 256 ) * 256 / bytes;
	const at = half
		? ( x: number, y: number, c = 0 ) => THREE.DataUtils.fromHalfFloat( ( raw as Uint16Array )[ y * stride + x * channels + c ] )
		: ( x: number, y: number, c = 0 ) => ( raw as Float32Array )[ y * stride + x * channels + c ];
	return { width, height, at };

}

const SKY_Z = - 9e4;

/** Reads one of the auxiliary values over a grid of cells (see the header). */
export async function readAov( renderer: THREE.WebGPURenderer, pipeline: Pipeline, camera: THREE.Camera | undefined, options: { aov: Aov; grid?: [ number, number ]; cell?: number; texels?: boolean } ): Promise<AovResult> {

	const { aov } = options;
	const [ nx, ny ] = options.grid ?? [ 16, 9 ];
	const n = options.cell ?? 0;
	const cells = nx * ny;
	const surfaceAt = pipeline.radianceSource( 'surfaceDepth' )!;
	const surface = await readTexture( renderer, surfaceAt.target, surfaceAt.index );
	const { width, height } = surface;
	const zAt = ( x: number, y: number ) => surface.at( x, y );
	const isSky = ( x: number, y: number ) => zAt( x, y ) < SKY_Z;
	const result = ( k: number, values: number[] ): AovResult => ( { aov, nx, ny, k, size: [ width, height ], values } );

	if ( aov === 'depth' || aov === 'class' ) {

		const opaque = await readTexture( renderer, pipeline.depthRT, 0 );
		const isSea = ( x: number, y: number ) => ! isSky( x, y ) && zAt( x, y ) > opaque.at( x, y, 0 ) + SEA_STEP;
		if ( aov === 'depth' ) {

			const sum = new Float64Array( cells * 4 );
			forCellTexels( nx, ny, width, height, n, ( c, x, y ) => {

				sum[ c * 4 + 3 ] ++;
				if ( isSky( x, y ) ) {

					sum[ c * 4 + 1 ] ++;
					return;

				}
				sum[ c * 4 ] += - zAt( x, y );
				if ( isSea( x, y ) ) sum[ c * 4 + 2 ] ++;

			} );
			const values: number[] = [];
			for ( let c = 0; c < cells; c ++ ) {

				const total = sum[ c * 4 + 3 ], sky = sum[ c * 4 + 1 ];
				values.push( total > sky ? round( sum[ c * 4 ] / ( total - sky ), 5 ) : 0, round( sky / total, 4 ), round( sum[ c * 4 + 2 ] / total, 4 ) );

			}
			return result( 3, values );

		}
		const colourAt = pipeline.radianceSource( 'opaque' )!;
		const colour = await readTexture( renderer, colourAt.target, colourAt.index );
		const sky = FIT_CLASSES.indexOf( 'sky' ), sea = FIT_CLASSES.indexOf( 'sea' );
		const decode = ( x: number, y: number ): number => {

			if ( isSky( x, y ) ) return sky;
			if ( isSea( x, y ) ) return sea;
			const r = colour.at( x, y, 0 );
			if ( ! ( r > 0 ) ) return - 1;
			const g = colour.at( x, y, 1 ) / r, b = colour.at( x, y, 2 ) / r;
			const k = Math.round( g * 10 ) - 1;
			if ( k < 0 || k >= FIT_CLASSES.length ) return - 1;
			const t = ( k + 1 ) / 10;
			return Math.abs( g - t ) < 0.02 && Math.abs( b - t * t ) < 0.03 ? k : - 1;

		};
		const votes = new Float64Array( cells * ( FIT_CLASSES.length + 1 ) );
		const stride = FIT_CLASSES.length + 1;
		const texels: number[] = [];
		forCellTexels( nx, ny, width, height, n, ( c, x, y ) => {

			const kind = decode( x, y );
			votes[ c * stride + kind + 1 ] ++;
			if ( options.texels ) texels.push( kind );

		} );
		const values: number[] = [];
		for ( let c = 0; c < cells; c ++ ) {

			let total = 0, best = 0, kind = - 1;
			for ( let k = 0; k < stride; k ++ ) total += votes[ c * stride + k ];
			for ( let k = 1; k < stride; k ++ ) if ( votes[ c * stride + k ] > best ) {

				best = votes[ c * stride + k ];
				kind = k - 1;

			}
			values.push( best * 2 > total ? kind : - 1 );

		}
		return { ...result( 1, values ), ...( options.texels ? { texels } : {} ) };

	}
	if ( aov === 'ao' ) {

		const resources = ( pipeline as unknown as { resources: { gtao: unknown }; graph: { target( h: unknown ): THREE.RenderTarget; texture( h: unknown ): THREE.Texture } } );
		const h = resources.resources.gtao;
		const values = new Array<number>( cells ).fill( 1 );
		if ( ! h ) return result( 1, values );
		const target = resources.graph.target( h );
		const ao = await readTexture( renderer, target, target.textures.indexOf( resources.graph.texture( h ) ) );
		const sum = new Float64Array( cells * 2 );
		forCellTexels( nx, ny, width, height, n, ( c, x, y ) => {

			const hx = Math.min( ao.width - 1, Math.floor( x * ao.width / width ) ), hy = Math.min( ao.height - 1, Math.floor( y * ao.height / height ) );
			sum[ c * 2 ] += ao.at( hx, hy, 0 );
			sum[ c * 2 + 1 ] ++;

		} );
		for ( let c = 0; c < cells; c ++ ) values[ c ] = round( sum[ c * 2 ] / Math.max( 1, sum[ c * 2 + 1 ] ), 4 );
		return result( 1, values );

	}
	// normal
	if ( ! camera ) throw new Error( 'radiance: the normal needs the camera' );
	camera.updateMatrixWorld();
	const projInv = ( camera as THREE.PerspectiveCamera ).projectionMatrixInverse;
	const v = new THREE.Vector3();
	const view = ( x: number, y: number, out: THREE.Vector3 ) => {

		v.set( ( x + 0.5 ) / width * 2 - 1, 1 - ( y + 0.5 ) / height * 2, 1 ).applyMatrix4( projInv );
		return out.copy( v ).multiplyScalar( zAt( x, y ) / v.z );

	};
	const p = new THREE.Vector3(), a = new THREE.Vector3(), b = new THREE.Vector3(), dx = new THREE.Vector3(), dy = new THREE.Vector3(), nrm = new THREE.Vector3();
	const step = ( x: number, y: number, ox: number, oy: number, out: THREE.Vector3 ) => {

		// the neighbour on the side with the smaller step in depth (inside the image)
		const x0 = x - ox, y0 = y - oy, x1 = x + ox, y1 = y + oy;
		const z = zAt( x, y );
		const in0 = x0 >= 0 && y0 >= 0 && ! isSky( x0, y0 ), in1 = x1 < width && y1 < height && ! isSky( x1, y1 );
		const use1 = in1 && ( ! in0 || Math.abs( zAt( x1, y1 ) - z ) <= Math.abs( zAt( x0, y0 ) - z ) );
		if ( ! use1 && ! in0 ) return false;
		if ( use1 ) out.subVectors( view( x1, y1, a ), p );
		else out.subVectors( p, view( x0, y0, b ) );
		return true;

	};
	const sum = new Float64Array( cells * 3 );
	forCellTexels( nx, ny, width, height, n, ( c, x, y ) => {

		if ( isSky( x, y ) ) return;
		view( x, y, p );
		if ( ! step( x, y, 1, 0, dx ) || ! step( x, y, 0, 1, dy ) ) return;
		nrm.crossVectors( dx, dy );
		if ( nrm.dot( p ) > 0 ) nrm.negate();
		nrm.normalize();
		sum[ c * 3 ] += nrm.x; sum[ c * 3 + 1 ] += nrm.y; sum[ c * 3 + 2 ] += nrm.z;

	} );
	const toWorld = new THREE.Matrix3().setFromMatrix4( camera.matrixWorld );
	const values: number[] = [];
	for ( let c = 0; c < cells; c ++ ) {

		nrm.set( sum[ c * 3 ], sum[ c * 3 + 1 ], sum[ c * 3 + 2 ] );
		if ( nrm.lengthSq() > 0 ) nrm.applyMatrix3( toWorld ).normalize();
		values.push( round( nrm.x, 4 ), round( nrm.y, 4 ), round( nrm.z, 4 ) );

	}
	return result( 3, values );

}

export interface RingsOptions {
	/** the direction toward the sun (world) */
	sun: [ number, number, number ];
	/** the rings' width in degrees from the sun (default 0.5) and their count (default 120) */
	step?: number; count?: number;
	/** the sky's band the profile is divided by, degrees from the sun (default [ 30, 60 ]) */
	sky?: [ number, number ];
	/** a band of angle from the sun, degrees, whose texels' median luminance is returned too */
	band?: [ number, number ];
	/** bands of angle from the sun, degrees, each read as the median over its sky texels of the
	 * luminance the bloom adds (sharp minus hdr) */
	halo?: [ number, number ][];
	/** only the texels in this sector about the sun, degrees: 0 straight below it, 90 to its right
	 * as seen facing it, 180 above, 270 to its left; a sector whose start exceeds its end wraps
	 * through 0. Applies to the rings, the sky's band, `band`, `halo` and `table`. */
	sector?: [ number, number ];
	/** only the texels whose ray's elevation (degrees above the horizontal) lies in this range;
	 * applies as `sector` does */
	elevation?: [ number, number ];
	/** the width in degrees of the elevation bands of `table` (none without it) */
	elevationStep?: number;
}
export interface RingsResult {
	step: number; count: number; size: [ number, number ]; preExposure: number;
	/** per source, each ring's mean luminance (scene units), null where no texel falls */
	rings: { sharp: ( number | null )[]; hdr: ( number | null )[] };
	/** the same over each ring's sky texels only, null where none falls */
	skyRings: { sharp: ( number | null )[]; hdr: ( number | null )[] };
	/** per source, the mean luminance of the sky's texels in the band `sky` */
	sky: { sharp: number | null; hdr: number | null };
	/** per source, the median luminance of every texel in the band `band` (null without one) */
	band: { sharp: number | null; hdr: number | null } | null;
	/** per band of `halo`, the median of sharp minus hdr over its sky texels and their count */
	halo: { median: number | null; n: number }[] | null;
	/** each ring's count of sky texels (every second texel on each axis) */
	skyCounts: number[];
	/** with `elevationStep`: the composite's (hdr) luminance summed over the sky's texels of each
	 * ring and elevation band, as [ ring, band, sum, count ] for every pair that holds any (band b
	 * spans b * step - 90 to ( b + 1 ) * step - 90 degrees), so a script can take each texel's
	 * excess over a background that varies with elevation (the sum less the count times the
	 * band's background) without the texels */
	table: { elevationStep: number; cells: [ number, number, number, number ][] } | null;
}

/** The luminance around the sun in rings of angle (degrees between a texel's ray and the sun),
 * over every texel and over the sky's texels alone, of the sharp image (the composite with the bloom, before the local tone and the
 * tone curve: Pipeline.ts) and of the composite before the bloom (`hdr`), and both over the sky's
 * texels in a band of angle from the sun (the sky by `surfaceDepth`, as `depth` above), and with
 * `band` the median over every texel in another band (the sun's disk, whose mean the bright limb
 * and the edge's blur would sway), and with `halo` the median over each band's sky texels of what
 * the bloom adds. Every second
 * texel on each axis; scene units (the pre-exposure taken out). The camera's matrices are read when
 * called (a paused page). */
export async function readRings( renderer: THREE.WebGPURenderer, pipeline: Pipeline, camera: THREE.Camera | undefined, options: RingsOptions ): Promise<RingsResult> {

	if ( ! camera ) throw new Error( 'radiance: the rings need the camera' );
	const step = options.step ?? 0.5, count = options.count ?? 120;
	const [ skyLo, skyHi ] = options.sky ?? [ 30, 60 ];
	const stops = preExposure.stops, unscale = 2 ** - stops;
	const surfaceAt = pipeline.radianceSource( 'surfaceDepth' )!;
	const sharpAt = pipeline.radianceSource( 'sharp' ), hdrAt = pipeline.radianceSource( 'hdr' );
	if ( ! sharpAt || ! hdrAt ) throw new Error( 'radiance: the rings need the sharp image and the composite (hdr)' );
	const [ surface, sharp, hdr ] = await Promise.all( [ surfaceAt, sharpAt, hdrAt ].map( ( at ) => readTexture( renderer, at.target, at.index ) ) );
	camera.updateMatrixWorld();
	const projInv = ( camera as THREE.PerspectiveCamera ).projectionMatrixInverse;
	const sunView = new THREE.Vector3( ...options.sun ).transformDirection( camera.matrixWorldInverse );
	const v = new THREE.Vector3();
	const DEG = 180 / Math.PI;
	const sums = { sharp: new Float64Array( count ), hdr: new Float64Array( count ) }, n = new Float64Array( count );
	const skySums = { sharp: new Float64Array( count ), hdr: new Float64Array( count ) }, skyCount = new Float64Array( count );
	const skySum = { sharp: 0, hdr: 0 };
	let skyN = 0;
	const [ bandLo, bandHi ] = options.band ?? [ Infinity, - Infinity ];
	const bandOf = { sharp: [] as number[], hdr: [] as number[] };
	const haloBands = options.halo ?? [];
	const haloOf = haloBands.map( () => [] as number[] );
	const haloLo = Math.min( Infinity, ...haloBands.map( ( b ) => b[ 0 ] ) ), haloHi = Math.max( - Infinity, ...haloBands.map( ( b ) => b[ 1 ] ) );
	const lum = ( t: typeof sharp, x: number, y: number ) => ( 0.2126 * t.at( x, y, 0 ) + 0.7152 * t.at( x, y, 1 ) + 0.0722 * t.at( x, y, 2 ) ) * unscale;
	const { width, height } = sharp;
	// the sector's axes about the sun (world): straight below it, and to its right as seen facing it
	const sunWorld = new THREE.Vector3( ...options.sun ).normalize();
	const below = new THREE.Vector3( 0, - 1, 0 ).addScaledVector( sunWorld, sunWorld.y ).normalize();
	const right = new THREE.Vector3().crossVectors( sunWorld, new THREE.Vector3( 0, 1, 0 ) ).normalize();
	const [ sectorFrom, sectorTo ] = options.sector ?? [ 0, 360 ];
	const [ elevationLo, elevationHi ] = options.elevation ?? [ - 90, 90 ];
	const tableStep = options.elevationStep ?? 0;
	const filtered = options.sector !== undefined || options.elevation !== undefined || tableStep > 0;
	const table = new Map<number, [ number, number, number, number ]>();
	const w = new THREE.Vector3();
	for ( let y = 0; y < height; y += 2 ) for ( let x = 0; x < width; x += 2 ) {

		v.set( ( x + 0.5 ) / width * 2 - 1, 1 - ( y + 0.5 ) / height * 2, 1 ).applyMatrix4( projInv ).normalize();
		const angle = Math.acos( Math.max( - 1, Math.min( 1, v.dot( sunView ) ) ) ) * DEG;
		let elevation = 0;
		if ( filtered ) {

			w.copy( v ).transformDirection( camera.matrixWorld );
			elevation = Math.asin( Math.max( - 1, Math.min( 1, w.y ) ) ) * DEG;
			if ( elevation < elevationLo || elevation >= elevationHi ) continue;
			if ( options.sector ) {

				const around = ( Math.atan2( w.dot( right ), w.dot( below ) ) * DEG + 360 ) % 360;
				const inside = sectorFrom <= sectorTo ? around >= sectorFrom && around < sectorTo : around >= sectorFrom || around < sectorTo;
				if ( ! inside ) continue;

			}

		}
		const k = Math.floor( angle / step );
		const inRing = k < count, inSky = angle >= skyLo && angle < skyHi, inBand = angle >= bandLo && angle < bandHi;
		const inHalo = angle >= haloLo && angle < haloHi;
		if ( ! inRing && ! inSky && ! inBand && ! inHalo ) continue;
		const ys = lum( sharp, x, y );
		const hx = Math.min( hdr.width - 1, Math.floor( x * hdr.width / width ) ), hy = Math.min( hdr.height - 1, Math.floor( y * hdr.height / height ) );
		const yh = lum( hdr, hx, hy );
		const sx = Math.min( surface.width - 1, Math.floor( x * surface.width / width ) ), sy = Math.min( surface.height - 1, Math.floor( y * surface.height / height ) );
		const isSky = surface.at( sx, sy ) < SKY_Z;
		if ( inRing ) {

			sums.sharp[ k ] += ys;
			sums.hdr[ k ] += yh;
			n[ k ] ++;
			if ( isSky ) {

				skySums.sharp[ k ] += ys;
				skySums.hdr[ k ] += yh;
				skyCount[ k ] ++;
				if ( tableStep > 0 ) {

					const band = Math.floor( ( elevation + 90 ) / tableStep ), key = k * 10000 + band;
					const cell = table.get( key ) ?? [ k, band, 0, 0 ];
					cell[ 2 ] += yh;
					cell[ 3 ] ++;
					table.set( key, cell );

				}

			}

		}
		if ( inBand ) {

			bandOf.sharp.push( ys );
			bandOf.hdr.push( yh );

		}
		if ( inHalo && isSky ) haloBands.forEach( ( b, i ) => {

			if ( angle >= b[ 0 ] && angle < b[ 1 ] ) haloOf[ i ].push( ys - yh );

		} );
		if ( inSky ) {

			if ( isSky ) {

				skySum.sharp += ys;
				skySum.hdr += yh;
				skyN ++;

			}

		}

	}
	const ring = ( s: Float64Array, of = n ) => Array.from( s, ( t, k ) => of[ k ] ? round( t / of[ k ] ) : null );
	const median = ( a: number[] ) => {

		if ( ! a.length ) return null;
		a.sort( ( p, q ) => p - q );
		const h = a.length >> 1;
		return round( a.length % 2 ? a[ h ] : ( a[ h - 1 ] + a[ h ] ) / 2 );

	};
	return {
		step, count, size: [ width, height ], preExposure: stops,
		rings: { sharp: ring( sums.sharp ), hdr: ring( sums.hdr ) },
		skyRings: { sharp: ring( skySums.sharp, skyCount ), hdr: ring( skySums.hdr, skyCount ) },
		sky: { sharp: skyN ? round( skySum.sharp / skyN ) : null, hdr: skyN ? round( skySum.hdr / skyN ) : null },
		band: options.band ? { sharp: median( bandOf.sharp ), hdr: median( bandOf.hdr ) } : null,
		halo: options.halo ? haloOf.map( ( a ) => ( { median: median( a ), n: a.length } ) ) : null,
		skyCounts: Array.from( skyCount ),
		table: tableStep > 0 ? { elevationStep: tableStep, cells: [ ...table.values() ].map( ( c ) => [ c[ 0 ], c[ 1 ], round( c[ 2 ] ), c[ 3 ] ] ) } : null,
	};

}
