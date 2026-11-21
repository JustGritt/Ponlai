import * as THREE from 'three/webgpu';
import { renderGroup, uniform } from 'three/tsl';
import { debugFlags } from '../debug/DebugFlags.ts';
import { ffxFsr3UpscalerGetJitterPhaseCount } from './fsr3/Host.ts';

// The temporal modes' sub-pixel jitter: a Halton (2, 3) shift of the camera's projection.
//
// The temporal modes' sub-pixel jitter (Antialiasing.ts: aa=taa|taau): each frame the scene half
// of the frame graph (its jittered span, from the opaque scene to the overlay) renders with the
// camera's projection shifted by a fraction of a render pixel, so successive frames sample each
// pixel at different points and the resolve can accumulate them. After Karis, "High Quality
// Temporal Supersampling" (SIGGRAPH 2014): a Halton (2, 3) sequence, whose points fill the pixel
// evenly whatever the length a frame cuts it at.
//
// Frame k of the running sequence takes Halton index (k mod L) + 1, the offset (h2 - 0.5, h3 - 0.5)
// in render pixels. TAA keeps L = 16; TAAU, whose output pixels are smaller than the render's,
// needs about eight samples per output pixel, so L = clamp( ceil( 8 r^2 ), 16, 64 ) for an output
// r times the render per axis. A scene at rest runs its own sequence (`restIndex` 1..`restLength`,
// set by the rest convergence: REST_SEQUENCE for the temporal modes, the ground truth's 64), so a
// paused frame is the same however it was reached.
//
// The rest's offsets are centred on their sequence's mean, (h2 - m2, h3 - m3) with m the mean of
// the sequence's Halton values: a prefix of Halton is not centred on the pixel (indices 1..32 average
// ( -0.015, -0.019 ) px from 0.5, 1..64 ( -0.008, -0.015 )), and the ground truth's box filter
// averages its samples with equal weights, so its image moved by that mean against the scene (a
// shift every temporal mode was then measured with against it). The temporal modes' rest weights
// its samples by their distance from their output pixel, which already cancels most of the mean
// (centring moves its image by under 0.01 px), but takes the same centring so both average about
// the pixel's centre. The running sequence keeps the plain offsets: its resolve weights samples the
// same way, and its history is no average of whole sequences.
//
// The offset moves the sampled point, x right and y down in render pixels: each render pixel shows
// the scene at its centre plus `px`. The projection is shifted the other way in NDC (by `ndc`),
// applied to the camera by `begin` and undone exactly by `end` (the graph's jitter hooks, around
// the jittered passes that run). Everything outside the span sees the camera unjittered: the sky
// and grass cull keys, the haze's froxel grid, the light shafts' sun position, the motion vectors (which are the motion
// of the unjittered views: the jitter is no motion). `debugFlags.jitter = false` renders without
// it (the motion vectors' unit checks).
//
// Under fsr3 (Antialiasing.ts) the sequence is FSR 3's own (post/fsr3/Host.ts): its length is
// `ffxFsr3UpscalerGetJitterPhaseCount( render width, canvas width )`, int( 8 r^2 ) without a clamp
// (18 at 4K output rendered at 1440p, the high tier's baseline there), and frame k takes Halton index
// ( k mod L ) + 1 as above, but negated: `px = -( h - 0.5 )`. FSR's offset J places pixel i's
// sample at i + 0.5 - J, where `px` places it at the centre plus `px`, so J = -px is upstream's own offset and the shifted projection
// equals upstream's bit for bit. Its rest runs the same plain offsets, from index 0 and repeating
// every L frames ( ( restIndex - 1 ) mod L ), with no centring: FSR weighs its samples by its own
// accumulation, not by a box over the sequence; the dither then follows the rest index itself.

/** The radical inverse of `index` in `base`: the Halton sequence's value. */
export function halton( index: number, base: number ): number {

	let f = 1, r = 0;
	for ( let i = index; i > 0; i = Math.floor( i / base ) ) {

		f /= base;
		r += f * ( i % base );

	}
	return r;

}

const means = new Map<number, THREE.Vector2>();

/** The mean of Halton ( 2, 3 ) over indices 1..n: where a sequence of n offsets is centred. */
export function haltonMean( n: number ): THREE.Vector2 {

	let m = means.get( n );
	if ( m === undefined ) {

		m = new THREE.Vector2();
		for ( let i = 1; i <= n; i ++ ) {

			m.x += halton( i, 2 );
			m.y += halton( i, 3 );

		}
		m.divideScalar( Math.max( n, 1 ) );
		means.set( n, m );

	}
	return m;

}

/** Whose sequence the jitter runs (Pipeline sets it for the app's mode). */
export type JitterMode = 'taa' | 'taau' | 'fsr3';

/** The temporal modes' rest sequence length (its indices are 1..REST_SEQUENCE). */
export const REST_SEQUENCE = 32;

/** The running sequence's length: 16 for TAA, more for TAAU's smaller output pixels (`ratio`: the
 * output's size over the render's, per axis). */
export function sequenceLength( mode: JitterMode, ratio: number, renderWidth = 1, outputWidth = renderWidth * ratio ): number {

	if ( mode === 'fsr3' ) return Math.max( ffxFsr3UpscalerGetJitterPhaseCount( renderWidth, Math.round( outputWidth ) ), 1 );
	return mode === 'taa' ? 16 : THREE.MathUtils.clamp( Math.ceil( 8 * ratio * ratio ), 16, 64 );

}

export class Jitter {

	/** This frame's offset of the sampled point from each render pixel's centre (render px; x right,
	 * y down). */
	readonly px = new THREE.Vector2();
	/** This frame's place in its sequence (0-based; the Halton index is k + 1), and the length. */
	k = 0;
	length = 16;
	/** Rendered frames so far (the running sequence's clock: frames the pacer skips do not count). */
	frame = 0;
	/** 1..restLength while a scene at rest runs the rest sequence (TemporalRest), else null, and the
	 * rest sequence's length (its offsets are centred on its mean). */
	restIndex: number | null = null;
	restLength = REST_SEQUENCE;
	readonly uniforms = {
		// the vegetation's per-frame dither index (a threshold of the pixel's 4x4 Bayer value plus 5
		// times it, mod 16; `coverageDither` in vegetation/Coverage.ts): the rendered frame mod 16 in the
		// running sequence, k mod 16 at rest (see `prepare`)
		ditherIndex: uniform( 0 ),
		// the offset in render pixels (`px`)
		px: uniform( new THREE.Vector2() ),
		// what to add to a jittered fragment's NDC for the unjittered view's: ( 2 px.x / w, -2 px.y / h )
		ndc: uniform( new THREE.Vector2() ).setGroup( renderGroup ),
		// the camera's projection without the jitter (the sky, which must not jitter, and the culls)
		unjitteredProjection: uniform( new THREE.Matrix4() ).setGroup( renderGroup ),
		unjitteredProjectionInverse: uniform( new THREE.Matrix4() ).setGroup( renderGroup ),
	};
	/** TAA's sequence, TAAU's or FSR 3's (Pipeline sets it for the app's mode). */
	mode: JitterMode = 'taa';
	private readonly projection = new THREE.Matrix4();
	private readonly projectionInverse = new THREE.Matrix4();
	private readonly shift = new THREE.Matrix4();
	private applied: THREE.Camera | null = null;

	/** Before a frame renders (the camera unjittered): this frame's offset, from the render's size and
	 * the output's ratio to it (and under fsr3 the output's width), and the unjittered projection. */
	prepare( camera: THREE.Camera, width: number, height: number, ratio: number, outputWidth = width * ratio ): void {

		const fsr3 = this.mode === 'fsr3';
		this.length = sequenceLength( this.mode, ratio, width, outputWidth );
		this.k = this.restIndex !== null ? ( fsr3 ? ( this.restIndex - 1 ) % this.length : this.restIndex - 1 ) : this.frame % this.length;
		// The dither's index runs on its own 16-frame cycle in the running sequence, so any 16
		// consecutive frames give each pixel every threshold once whatever L is (with k mod 16, TAAU's
		// L = 18 at a ratio of 1.5 repeated two thresholds and skipped two in each 16 frames across its
		// wrap); at rest it follows the rest index, so a paused frame stays path independent.
		const dither = this.restIndex !== null ? this.restIndex - 1 : this.frame;
		this.frame ++;
		if ( debugFlags.jitter === false ) this.px.set( 0, 0 );
		else if ( fsr3 ) this.px.set( 0.5 - halton( this.k + 1, 2 ), 0.5 - halton( this.k + 1, 3 ) );
		else if ( this.restIndex !== null ) {

			const m = haltonMean( this.restLength );
			this.px.set( halton( this.k + 1, 2 ) - m.x, halton( this.k + 1, 3 ) - m.y );

		} else this.px.set( halton( this.k + 1, 2 ) - 0.5, halton( this.k + 1, 3 ) - 0.5 );
		const u = this.uniforms;
		u.ditherIndex.value = dither % 16;
		u.px.value.copy( this.px );
		u.ndc.value.set( 2 * this.px.x / Math.max( width, 1 ), - 2 * this.px.y / Math.max( height, 1 ) );
		this.projection.copy( camera.projectionMatrix );
		this.projectionInverse.copy( camera.projectionMatrixInverse );
		u.unjitteredProjection.value.copy( this.projection );
		u.unjitteredProjectionInverse.value.copy( this.projectionInverse );

	}

	/** Shifts the camera's projection by this frame's offset (the scene point at a pixel's centre plus
	 * `px` lands on the centre). */
	begin( camera: THREE.Camera ): void {

		const d = this.uniforms.ndc.value;
		this.shift.makeTranslation( - d.x, - d.y, 0 );
		camera.projectionMatrix.copy( this.projection ).premultiply( this.shift );
		camera.projectionMatrixInverse.copy( camera.projectionMatrix ).invert();
		this.applied = camera;

	}

	/** Restores the unjittered projection, bit for bit. */
	end( camera: THREE.Camera ): void {

		if ( this.applied !== camera ) return;
		camera.projectionMatrix.copy( this.projection );
		camera.projectionMatrixInverse.copy( this.projectionInverse );
		this.applied = null;

	}

}

/** The app's jitter: one, since the materials that read its uniforms are built once. */
export const jitter = new Jitter();
