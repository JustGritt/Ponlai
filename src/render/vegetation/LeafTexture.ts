import * as THREE from 'three/webgpu';

// The folded leaf: a colour map and an alpha map drawn from one silhouette.
//
// The folded leaf every leafy plant shares: a colour map and an alpha map drawn from one silhouette, the alpha map's
// mips built on the CPU to keep the leaf's coverage at the cutoff (`LEAF_ALPHA_TEST`). Shadow passes
// keep `alphaTest` and `alphaMap` but not `map`, so alpha-tested foliage sets both: the leaf `map` is
// opaque and the silhouette lives in `alphaMap` alone (box-filtered mips would fill the card). Alpha
// to coverage turns alpha short of cutoff + `fwidth` into partial coverage, so leaf interiors stay at
// 1 rather than scaled down.
// The folded leaf is kept for the plant kinds to come; no kind uses it today (the leafy plants it
// was drawn for, the rim bushes, the rim's hanging trails and the plateau's shrubs and small trees,
// are removed; the material factory's 'leaf' family, vegetation/Material.ts, takes its alpha map by
// default). Its coverage-preserving mips (`coverageAlphaTexture`) serve any silhouette: the old
// trees' leaf sprays take them (vegetation/TreeSurfaces.ts LeafPainter).

/** A square canvas of `size` texels to draw a texture on: the vegetation's one use of the
 * page's document (tools/repo/verify-layers.mjs, the browser-globals check), which every texture drawn on a canvas
 * takes from here. */
export function textureCanvas( size: number ): HTMLCanvasElement {

	const c = document.createElement( 'canvas' );
	c.width = c.height = size;
	return c;

}

// The leaf as a transparent silhouette; the colour map and the coverage map are both made
// from it, so they always agree.
function drawLeaf( size: number ): HTMLCanvasElement {

	const c = textureCanvas( size );
	const g = c.getContext( '2d' )!;
	g.clearRect( 0, 0, size, size );
	g.save();
	g.scale( size, size );
	const shade = g.createLinearGradient( 0.06, 0.6, 0.96, 0.25 );
	shade.addColorStop( 0, '#b5b5b5' );
	shade.addColorStop( 0.47, '#eeeeee' );
	shade.addColorStop( 0.55, '#cbcbcb' );
	shade.addColorStop( 1, '#f7f7f7' );
	g.fillStyle = shade;
	g.beginPath();
	g.moveTo( 0.5, 0.99 );
	g.bezierCurveTo( 0.44, 0.8, 0.08, 0.63, 0.065, 0.5 );
	g.bezierCurveTo( 0.09, 0.3, 0.4, 0.16, 0.5, 0.015 );
	g.bezierCurveTo( 0.65, 0.21, 0.92, 0.32, 0.935, 0.5 );
	g.bezierCurveTo( 0.9, 0.67, 0.57, 0.81, 0.5, 0.99 );
	g.fill();
	g.strokeStyle = 'rgba(255,255,235,0.44)';
	g.lineWidth = 0.009;
	g.beginPath();
	g.moveTo( 0.5, 0.95 );
	g.quadraticCurveTo( 0.47, 0.53, 0.5, 0.06 );
	g.stroke();
	g.lineWidth = 0.004;
	g.strokeStyle = 'rgba(110,120,90,0.18)';
	for ( let i = 0; i < 4; i ++ ) {

		const y = 0.36 + i * 0.12;
		const w = Math.sin( y * Math.PI ) * 0.31;
		g.beginPath();
		g.moveTo( 0.5, y + 0.09 );
		g.quadraticCurveTo( 0.5 - w * 0.6, y + 0.015, 0.5 - w, y - 0.08 );
		g.moveTo( 0.5, y + 0.08 );
		g.quadraticCurveTo( 0.5 + w * 0.6, y, 0.5 + w, y - 0.09 );
		g.stroke();

	}
	g.restore();
	return c;

}

// The leaf's colour, opaque. Its silhouette lives only in leafAlphaTexture, whose mips keep
// the leaf's coverage: an alpha here would be box-filtered by the GPU and, multiplied into the
// alpha map, thin distant leaves again. Shadow passes read the alpha map alone, so both passes
// now cut the same silhouette. The card around the leaf stays black, as the transparent texels
// were, so the colour mips match the previous ones.
export function leafTexture( size = 128 ): THREE.CanvasTexture {

	const c = textureCanvas( size );
	const g = c.getContext( '2d' )!;
	g.fillStyle = '#000';
	g.fillRect( 0, 0, size, size );
	g.drawImage( drawLeaf( size ), 0, 0 );
	const tex = new THREE.CanvasTexture( c );
	tex.colorSpace = THREE.SRGBColorSpace;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.anisotropy = 4;
	return tex;

}

export const LEAF_ALPHA_TEST = 0.3;

// Share of a w x w alpha level above the cutoff once bilinearly filtered (clamped at the
// edges, as the texture is), sampled on a grid at least 64 samples across.
function filteredCoverage( alpha: Float32Array, w: number, cutoff: number ): number {

	const steps = Math.max( 1, Math.ceil( 64 / w ) ), n = w * steps;
	const at = ( x: number, y: number ) => alpha[ Math.min( w - 1, Math.max( 0, y ) ) * w + Math.min( w - 1, Math.max( 0, x ) ) ];
	let covered = 0;
	for ( let j = 0; j < n; j ++ ) for ( let i = 0; i < n; i ++ ) {

		const u = ( i + 0.5 ) / steps - 0.5, v = ( j + 0.5 ) / steps - 0.5;
		const x = Math.floor( u ), y = Math.floor( v ), fx = u - x, fy = v - y;
		const top = at( x, y ) + ( at( x + 1, y ) - at( x, y ) ) * fx;
		const bottom = at( x, y + 1 ) + ( at( x + 1, y + 1 ) - at( x, y + 1 ) ) * fx;
		if ( top + ( bottom - top ) * fy > cutoff ) covered ++;

	}
	return covered / ( n * n );

}

// Coverage-preserving alpha mips, after Ignacio Castano, "Computing Alpha Mipmaps" (2010). The
// leaf covers under half its card, yet its mean alpha (~0.46) is above the 0.3 cutoff, so plain
// box mips fill the card: distant leaves turned into squares. Each level is box-filtered from
// the unremapped level above, then remapped so the share of its bilinearly filtered surface
// above the cutoff matches the full-size leaf. Castano scales alpha down; alpha-to-coverage turns
// anything short of cutoff + fwidth into partial coverage, so a scaled-down interior would make
// whole leaves see-through. The remap is a steep ramp through the cutoff instead: leaves stay
// solid, with a filtered edge. Built once on the CPU; the GPU does not regenerate them. The
// levels are opaque grey canvases (alpha maps read colour, not alpha), uploaded with the same
// flip as the colour map.
export function leafAlphaTexture( size = 128, cutoff = LEAF_ALPHA_TEST ): THREE.CanvasTexture {

	const leaf = drawLeaf( size ).getContext( '2d' )!.getImageData( 0, 0, size, size ).data;
	const alpha = new Float32Array( size * size );
	for ( let i = 0; i < alpha.length; i ++ ) alpha[ i ] = leaf[ i * 4 + 3 ] / 255;
	return coverageAlphaTexture( alpha, size, cutoff );

}

/** An alpha map from a square silhouette's alpha (0..1, row by row), its mips keeping the
 * full-size silhouette's coverage at `cutoff` (see leafAlphaTexture). */
export function coverageAlphaTexture( alpha: Float32Array, size: number, cutoff = LEAF_ALPHA_TEST ): THREE.CanvasTexture {

	let level = alpha;
	let covered = 0;
	for ( const a of level ) if ( a > cutoff ) covered ++;
	const coverage = covered / level.length;

	const mips: HTMLCanvasElement[] = [];
	for ( let w = size; ; w >>= 1 ) {

		let remapped = level;
		if ( w < size ) {

			let peak = 0;
			for ( const a of level ) peak = Math.max( peak, a );
			const ramp = ( 1 - cutoff ) / ( 0.1 * peak );
			const remap = ( threshold: number ) => level.map( ( a ) => Math.min( 1, Math.max( 0, cutoff + ( a - threshold ) * ramp ) ) );
			// Coverage falls as the threshold rises: bisect for the full-size leaf's share,
			// but keep the densest texel solid so a far bush never vanishes outright.
			let lo = 0, hi = peak * 0.9;
			for ( let i = 0; i < 20; i ++ ) {

				const mid = ( lo + hi ) * 0.5;
				if ( filteredCoverage( remap( mid ), w, cutoff ) > coverage ) lo = mid; else hi = mid;

			}
			remapped = remap( ( lo + hi ) * 0.5 );

		}
		const c = textureCanvas( w );
		const g = c.getContext( '2d' )!;
		const image = g.createImageData( w, w );
		for ( let i = 0; i < remapped.length; i ++ ) {

			const v = Math.round( remapped[ i ] * 255 );
			image.data[ i * 4 ] = image.data[ i * 4 + 1 ] = image.data[ i * 4 + 2 ] = v;
			image.data[ i * 4 + 3 ] = 255;

		}
		g.putImageData( image, 0, 0 );
		mips.push( c );
		if ( w === 1 ) break;

		const half = w >> 1, next = new Float32Array( half * half );
		for ( let y = 0; y < half; y ++ ) for ( let x = 0; x < half; x ++ ) {

			const i = 2 * y * w + 2 * x;
			next[ y * half + x ] = ( level[ i ] + level[ i + 1 ] + level[ i + w ] + level[ i + w + 1 ] ) * 0.25;

		}
		level = next;

	}

	const tex = new THREE.CanvasTexture( mips[ 0 ] );
	tex.mipmaps = mips as unknown as THREE.Texture[ 'mipmaps' ];
	tex.generateMipmaps = false;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.anisotropy = 4;
	return tex;

}
