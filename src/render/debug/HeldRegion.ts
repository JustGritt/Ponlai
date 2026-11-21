import * as THREE from 'three/webgpu';
import type { Pipeline } from '../post/Pipeline.ts';
import { readTexture } from './RadianceAov.ts';

// Reads where a paused frame is compared within the held tolerance.
//
// The region is the cells of the frame the sea's reflection reaches.
// The sea's reflection (ocean/SeaReflection.ts) averages a fixed sequence of frames of the opaque
// image and then holds its result. On a paused page under MSAA a few of those frames can differ in
// their last bits from load to load (the browser's or the driver's doing, not the scene's), and the
// held result keeps what they held. So the reflection's pixels of a paused frame are compared within
// a tolerance (tools/kit/tolerance.py, `HELD`), and the rest of the frame byte for byte. The water
// reads the reflection at its own pixel (Water.ts), and the reflection is 0 wherever its prepass
// traced nothing, so the region is the texels the reflection last traced from
// (SeaReflection.ts `surfaceHistory`: view distance over 0, roughness not negative).
//
// `__PONLAI__.heldRegion( cell )` on debug pages (the capture tools record it with each shot of a
// set: tools/kit/capture.mjs, tools/island/verify-scene-jump.mjs) reads that surface back once, as
// a grid of cells of `cell` by `cell` texels of the render (4): a cell is set when any texel in it
// traces. Returns the render's size, the cell's size, the grid and the cells as a hexadecimal
// string, four cells a digit, row by row from the top-left, the first cell in a digit's lowest bit;
// texel ( x, y ) lies in cell ( floor( x / cell ), floor( y / cell ) ). The image the tools compare
// is the render scaled to the output, so a tool maps its pixels to texels by the two sizes and
// grows the region to cover the upscale's and the sharpening's reach (tolerance.py `held_region`).

export interface HeldRegion { size: [ number, number ]; cell: number; grid: [ number, number ]; cells: string }

/** The cells the held reflection reaches in the frame drawn last (see the header). */
export async function readHeldRegion( renderer: THREE.WebGPURenderer, pipeline: Pipeline, cell = 4 ): Promise<HeldRegion> {

	const g = pipeline.graph, h = pipeline.seaReflection.surfaceHistory;
	const target = g.target( h );
	const surface = await readTexture( renderer, target, target.textures.indexOf( g.texture( h ) ) );
	const { width, height } = surface;
	const nx = Math.ceil( width / cell ), ny = Math.ceil( height / cell );
	const cells = new Uint8Array( nx * ny );
	for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) {

		if ( surface.at( x, y, 3 ) > 0 && surface.at( x, y, 2 ) >= 0 ) cells[ Math.floor( y / cell ) * nx + Math.floor( x / cell ) ] = 1;

	}
	const digits: string[] = [];
	for ( let c = 0; c < cells.length; c += 4 ) digits.push( ( cells[ c ] | ( cells[ c + 1 ] ?? 0 ) << 1 | ( cells[ c + 2 ] ?? 0 ) << 2 | ( cells[ c + 3 ] ?? 0 ) << 3 ).toString( 16 ) );
	return { size: [ width, height ], cell, grid: [ nx, ny ], cells: digits.join( '' ) };

}
