import * as THREE from 'three/webgpu';
import type { Pipeline } from '../post/Pipeline.ts';
import { preExposure } from '../post/PreExposure.ts';
import { type Aov, type AovResult, type RingsOptions, type RingsResult, forCellTexels, readAov, readRings } from './RadianceAov.ts';

// The scene's linear radiance over screen boxes, as numbers, measured before the exposure.
//
// The scene's linear radiance over screen boxes, as numbers (debug builds:
// `__PONLAI__.radiance( { boxes, source, image } )`, e.g. from tools/kit/probe.mjs or a capture's
// `--eval`): the lighting layer measured before the exposure, the tone curve, the grade and the
// bloom, which a capture's PNG has all been through. Under lightOnly (debug/LightLayer.ts) the
// radiance of a grey card of albedo 0.5 is the light at that point times 0.5 / pi.
//
// The source is the frame graph's (Pipeline.ts `radianceSource`): `opaque` (the default) is the
// resolved opaque colour (at 4x the box resolve of the samples), the lit surfaces and the sky
// before the sea, the haze and the post chain; `hdr` is the composite's output, with the sea and
// the air over them. Both are at the render's size, which a box's fractions follow whatever the
// render scale. The texture is read back once per call (the frame drawn last; on a paused page,
// wait for `settled` first) and every box is reduced on the CPU.
//
// boxes: { name: [ x, y, w, h ] } in fractions of the frame from its top-left corner, or in pixels of
// an image of `image: [ width, height ]` (a capture's PNG, for the boxes `img.py stats` takes);
// without boxes, the whole frame. Per box: `px` (texels averaged), `rgb` (mean linear Rec. 709),
// `Y` (the mean's luminance), `Yp` (the luminance's 10th, 50th and 90th percentiles over the box),
// and `oklab` of the mean ( L, a, b, C and the hue h in degrees). `exposure` is the exposure state
// the last frame left (Pipeline.readExposure: EV, its target, and `exposure`, the factor the display
// chain applies), so radiance times its `exposure` is the exposed value.
// source `meterLight` reads the metering illuminance M instead (Pipeline.ts: one channel, scene units
// of luminance; its `rgb` is ( M, 0, 0 ) and its `Y` and `Yp` are M's). Under msaa its resolve is kept by its
// reader, the metering histograms (Metering.ts). `nonSky: true` leaves out the sky's
// texels (the nearest surface's view z below -9e4, Pipeline.ts `surfaceDepth`), the sea's kept. Every
// box also reports `range`, the half float's limits over its texels' luminance as stored: the share
// below its smallest normal value 2^-14 (`below`, zeros included: `zero` counts them alone) and at
// or above 65504 (`above`), with the least and greatest stored value (the meter's half-float
// check, Pipeline.ts). The targets hold radiance pre-exposed, times 2^P (post/PreExposure.ts):
// `range` is in the stored values, everything else in scene units (divided by 2^P, P the last
// frame's, `preExposure` in stops).
// grid: [ nx, ny ] also returns the frame's mean linear colour over an nx by ny grid of cells (row
// by row from the top-left, `grid.rgb` flat, three per cell): a map of the light to compare two
// views of one frame cell by cell (lightOnly against lightOnly with ambientOnly gives each cell's
// share of direct light, so a box can be chosen by what light falls in it). `cell: n` averages only
// the n by n texels at each cell's centre (debug/RadianceAov.ts `forCellTexels`), and `boxes: {}`
// skips the boxes (the whole frame's percentiles sort every texel). `texels: true` with `cell` also
// returns `grid.texels`, each of those texels' own colour (three per texel, cell by cell, n by n per
// cell in rows; NaN for the sky's under `nonSky` and for a texel not finite), for measures
// that take a pixel's light with its own normal.
// aov: 'depth' | 'normal' | 'ao' | 'class' returns, instead of the radiance, what each cell of the
// grid shows (debug/RadianceAov.ts: view depth and the sky's and sea's shares, the world normal, the
// screen-space occlusion, the kind under `flags=classes`). `rings: { sun, step, count, sky }` returns
// the luminance around the sun in rings of angle, before and after the bloom, in a sector about it
// or a range of elevation with `sector`, `elevation`, and per elevation band with `elevationStep`
// (RadianceAov.ts `readRings`). `source: 'sharp'` reads the composite with the bloom added.

type Box = [ number, number, number, number ];
export interface RadianceOptions { boxes?: Record<string, Box>; source?: 'opaque' | 'hdr' | 'sharp' | 'meterLight'; image?: [ number, number ]; grid?: [ number, number ]; nonSky?: boolean; cell?: number; texels?: boolean; aov?: Aov; rings?: RingsOptions }
interface BoxRadiance {
	px: number; rgb: [ number, number, number ]; Y: number; Yp: [ number, number, number ];
	oklab: { L: number; a: number; b: number; C: number; h: number };
	range: { below: number; zero: number; above: number; min: number; max: number };
}

// half float's smallest normal value and its largest
const HALF_NORMAL = 2 ** - 14, HALF_MAX = 65504;

const round = ( x: number, digits = 6 ): number => Number( x.toPrecision( digits ) );

/** Oklab of a linear Rec. 709 colour (Ottosson's matrices). */
function oklab( r: number, g: number, b: number ): { L: number; a: number; b: number; C: number; h: number } {

	const l = Math.cbrt( 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b );
	const m = Math.cbrt( 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b );
	const s = Math.cbrt( 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b );
	const L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
	const A = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
	const B = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
	const h = ( Math.atan2( B, A ) * 180 / Math.PI + 360 ) % 360;
	return { L: round( L ), a: round( A ), b: round( B ), C: round( Math.hypot( A, B ) ), h: round( h, 5 ) };

}

/** The mean linear radiance over each box (see the header). */
export async function readRadiance( renderer: THREE.WebGPURenderer, pipeline: Pipeline, options: RadianceOptions = {}, camera?: THREE.Camera ): Promise<AovResult | RingsResult | {
	source: string; size: [ number, number ]; preExposure: number; exposure: Awaited<ReturnType<Pipeline[ 'readExposure' ]>>; boxes: Record<string, BoxRadiance | { error: string }>;
	grid?: { nx: number; ny: number; rgb: number[]; texels?: number[] };
}> {

	if ( options.rings ) return readRings( renderer, pipeline, camera, options.rings );
	if ( options.aov ) return readAov( renderer, pipeline, camera, { aov: options.aov, grid: options.grid, cell: options.cell, texels: options.texels } );
	const source = options.source ?? 'opaque';
	const at = pipeline.radianceSource( source );
	if ( ! at ) throw new Error( `radiance: no '${ source }' in this graph` );
	const { target, index } = at;
	const texture = target.textures[ index ];
	const width = target.width, height = target.height;
	// (the P the frame drawn last stored at: read before the await, which a new frame may follow)
	const stops = preExposure.stops, unscale = 2 ** - stops;
	const raw = await renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height, index );
	// rows come back padded to 256 bytes (three's copyTextureToBuffer)
	const half = texture.type === THREE.HalfFloatType;
	const channels = texture.format === THREE.RedFormat ? 1 : 4, bytes = half ? 2 : 4;
	const stride = Math.ceil( width * channels * bytes / 256 ) * 256 / bytes;
	// (stored values; `rgbAt` below gives scene units)
	const value = half ? ( i: number ) => THREE.DataUtils.fromHalfFloat( ( raw as Uint16Array )[ i ] ) : ( i: number ) => ( raw as Float32Array )[ i ];
	// a texel's ( r, g, b ) in scene units (one channel: green and blue 0)
	const rgbAt = ( x: number, y: number ): [ number, number, number ] => {

		const i = y * stride + x * channels;
		return channels === 1 ? [ value( i ) * unscale, 0, 0 ] : [ value( i ) * unscale, value( i + 1 ) * unscale, value( i + 2 ) * unscale ];

	};
	// the sky's texels, left out under `nonSky`
	let sky: ( x: number, y: number ) => boolean = () => false;
	if ( options.nonSky ) {

		const at = pipeline.radianceSource( 'surfaceDepth' )!;
		const z = await renderer.readRenderTargetPixelsAsync( at.target, 0, 0, width, height, at.index ) as Float32Array;
		const zStride = Math.ceil( width * 4 / 256 ) * 256 / 4;
		sky = ( x, y ) => z[ y * zStride + x ] < - 9e4;

	}
	const boxes = options.boxes ?? { frame: [ 0, 0, 1, 1 ] as Box };
	const [ iw, ih ] = options.image ?? [ 1, 1 ];
	const out: Record<string, BoxRadiance | { error: string }> = {};
	for ( const [ name, [ bx, by, bw, bh ] ] of Object.entries( boxes ) ) {

		const x0 = Math.max( 0, Math.floor( bx / iw * width ) ), y0 = Math.max( 0, Math.floor( by / ih * height ) );
		const x1 = Math.min( width, Math.ceil( ( bx + bw ) / iw * width ) ), y1 = Math.min( height, Math.ceil( ( by + bh ) / ih * height ) );
		if ( x1 <= x0 || y1 <= y0 ) {

			out[ name ] = { error: 'empty box' };
			continue;

		}
		let r = 0, g = 0, b = 0, n = 0, below = 0, zero = 0, above = 0;
		const lum: number[] = [];
		for ( let y = y0; y < y1; y ++ ) for ( let x = x0; x < x1; x ++ ) {

			if ( sky( x, y ) ) continue;
			const [ R, G, B ] = rgbAt( x, y );
			if ( ! Number.isFinite( R + G + B ) ) continue;
			r += R; g += G; b += B; n ++;
			const Y = channels === 1 ? R : 0.2126 * R + 0.7152 * G + 0.0722 * B;
			lum.push( Y );
			// (the half float's limits on the stored value)
			const stored = Y / unscale;
			if ( stored < HALF_NORMAL ) below ++;
			if ( Y === 0 ) zero ++;
			if ( stored >= HALF_MAX ) above ++;

		}
		if ( n === 0 ) {

			out[ name ] = { error: 'no finite texel' };
			continue;

		}
		r /= n; g /= n; b /= n;
		lum.sort( ( p, q ) => p - q );
		const pct = ( q: number ) => round( lum[ Math.min( n - 1, Math.floor( q * n ) ) ] );
		out[ name ] = {
			px: n, rgb: [ round( r ), round( g ), round( b ) ], Y: round( channels === 1 ? r : 0.2126 * r + 0.7152 * g + 0.0722 * b ),
			Yp: [ pct( 0.1 ), pct( 0.5 ), pct( 0.9 ) ], oklab: oklab( r, g, b ),
			range: { below: round( below / n, 4 ), zero: round( zero / n, 4 ), above: round( above / n, 4 ), min: round( lum[ 0 ] / unscale ), max: round( lum[ n - 1 ] / unscale ) },
		};

	}
	let grid: { nx: number; ny: number; rgb: number[]; texels?: number[] } | undefined;
	if ( options.grid ) {

		const [ nx, ny ] = options.grid;
		const sum = new Float64Array( nx * ny * 4 );
		const texels: number[] = [];
		forCellTexels( nx, ny, width, height, options.cell ?? 0, ( cell, x, y ) => {

			const [ R, G, B ] = sky( x, y ) ? [ NaN, NaN, NaN ] : rgbAt( x, y );
			if ( options.texels ) texels.push( ...( Number.isFinite( R + G + B ) ? [ round( R, 5 ), round( G, 5 ), round( B, 5 ) ] : [ NaN, NaN, NaN ] ) );
			if ( ! Number.isFinite( R + G + B ) ) return;
			const c = cell * 4;
			sum[ c ] += R; sum[ c + 1 ] += G; sum[ c + 2 ] += B; sum[ c + 3 ] ++;

		} );
		const rgb: number[] = [];
		for ( let c = 0; c < nx * ny; c ++ ) for ( let k = 0; k < 3; k ++ ) rgb.push( sum[ c * 4 + 3 ] > 0 ? round( sum[ c * 4 + k ] / sum[ c * 4 + 3 ], 4 ) : 0 );
		grid = { nx, ny, rgb, ...( options.texels ? { texels } : {} ) };

	}
	return { source, size: [ width, height ], preExposure: stops, exposure: await pipeline.readExposure(), boxes: out, ...( grid ? { grid } : {} ) };

}
