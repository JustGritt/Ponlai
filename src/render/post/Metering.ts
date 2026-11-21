import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, atomicAdd, atomicLoad, atomicStore, clamp, float, floor, int, ivec2, localId, log2, luminance, max, min, normalize, round, select, storage, textureLoad, texture,
	uint, uniform, vec2, vec3, vec4, workgroupArray, workgroupBarrier, workgroupId, instanceIndex,
} from 'three/tsl';
import { exposureAdapt, exposureTarget } from './Exposure.ts';
import { preExposure } from './PreExposure.ts';
import { shared } from '../shared.ts';
import type { Atmosphere } from '../sky/Atmosphere.ts';
import type { FrameGraph, ResourceHandle } from './graph/FrameGraph.ts';
import type { PassBuilder } from './graph/GraphCompiler.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type U = THREE.Node<'uint'>;
type Uniform2 = THREE.UniformNode<'vec2', THREE.Vector2>;

// The metering histograms: the illuminance's and the luminance's, read by the exposure.
//
// The metering histograms: two histograms a frame, H_M of log2 of the metering illuminance
// (`meterLight`, M = Y( E ) / pi: Pipeline.ts) and H_Y of log2 of the luminance of `hdr`, the
// composite's output, before the bloom (the bloom reads this frame's exposure, so a histogram
// after it would close a loop). The exposure's meter reads them.
// - Bins: 128 over 32 stops, a quarter stop each, of log2 of the values as the targets store them,
//   pre-exposed by 2^P (PreExposure.ts), from 2^-16 to 2^16: half float's own range, its largest
//   value (65504, where the pre-exposed writers hold, the sun's disc among them) in the top bin and
//   two stops of its subnormals at the bottom (which still keep 8 significant bits there, finer
//   than a bin). Bin i's centre is 2^( LOW + ( i + 0.5 ) / 4 ) as stored and
//   2^( LOW - P + ( i + 0.5 ) / 4 ) in scene units (which carry SCENE_RADIANCE_SCALE): P is a whole
//   number of stops, so in scene units the bins lie on the same quarter-stop grid at every P, and
//   only the range moves with it. The meter takes P back out of the trimmed means (PreExposure.ts
//   `log2Scale`), so the exposure works in scene units (Exposure.ts). The range follows the
//   pre-exposure, and so the exposure: what the targets hold at full precision the histograms
//   never clip. A range fixed in scene units (it was 2^-20 to 2^12) did: the island view of a
//   clear evening at -7 degrees, the darkest the twilight's land gets before the night's light
//   comes in (P 14), had 89 % of hdr's samples and 2.5 % of meterLight's in its lowest bin, M's
//   10th percentile 0.85 stop over its floor; now none, that percentile 11 stops over the floor,
//   and every target measured stays as it was (the trimmed means' percentiles lay above that
//   floor). At noon (P 2) the range spans 2^-18 to 2^14 in scene units, at night (P 8) 2^-24 to
//   2^8: a clear night's darkest non-sky values (about 2^-18.6 in hdr, 2^-15.8 in M) lie 5.4 and
//   8.2 stops over its floor.
// - Splat: each sample adds to the two bins whose centres are nearest its value, weighted
//   linearly (a tent), in fixed point: weight x 256, the two weights summing to exactly 256, so a
//   histogram's total is 256 x the samples it took. A value past either end's centre goes whole
//   into that end bin. Samples with M <= 0 or Y <= 0 add nothing (a pixel no metered material
//   drew, or NaN): only an exact 0 is no reading, the half-float subnormals count.
// - The sky's pixels (`surfaceDepth` past -9e4, as the composite tells them apart) enter H_Y at the
//   sky's own light while the night sky is drawn below the light it sheds (shared.ts `seenNight`,
//   Weather.ts `seenNight`, under 1): the dome takes that gain g on the moon's share of the sky
//   only (Sky.ts `domeGapGain`), so a sky pixel is divided by ( S + M g ) / ( S + M ), S and M the
//   sun's and the moon's shares of the open sky toward the pixel as the dome draws them, each
//   times its gap gain (Atmosphere.ts `skyShares`, Sky.ts `gapGain`, at the dome's own direction,
//   held above the horizon). The exposure is so set by the light, as the
//   land and the sea take it, and the darker drawn sky does not loosen the highlight guard: with it
//   in, a clear night's lowest camera rose about a stop in EV. Divided by g whole, the twilight sky,
//   which is the sun's and keeps its light, was raised instead and the clear dusk's EV fell more
//   than a stop. The clouds, drawn at g whole, are read at the clear sky's ratio: at night that is g,
//   at dusk under it. The stars and the moon's disc, which keep their light, are a few samples, far
//   under the guard's 3 %. By day (g = 1) the pixel is read as it is, so the day's histograms and
//   bytes are as they were.
// - Sampling, fixed in uv: 32 x 16 tiles of 16 x 16 samples, one sample at the centre of each
//   cell of a 512 x 256 grid over the frame, at the render texel it falls in (textureLoad: the
//   value as rendered, no filtering). 131072 samples a frame whatever the render's size, so the
//   dispatch never changes with the dynamic resolution (a kernel dispatched by a target's size is
//   rebuilt at every step) and the meter reads the same scene at any scale.
// - Accumulation: one workgroup per tile adds its samples into workgroup memory with integer
//   atomics, then each of its 256 threads adds one bin into the global buffer of 2 x 128 uint
//   (H_M at 0..127, H_Y at 128..255). Integer sums do not depend on the order the threads and
//   workgroups ran in, so the histograms are deterministic. A small kernel clears the buffer
//   earlier in the frame (its own pass, before the composite).
// - On the graph: the histogram pass reads `hdr` and `meterLight`, which keeps meterLight's MSAA
//   resolve alive and stored (the derived attachment ops drop a resolve no pass reads). The
//   buffer is not a graph resource: the passes are never culled and run in their declared order,
//   the clear before the composite, the histograms after it and before the bloom; the exposure's
//   meter reads the buffer after them.
// - The meter: a one-texel render pass after the histograms (and before FSR 3 and the bloom, which
//   read its result) that reads the buffer and evaluates the exposure target (Exposure.ts) into
//   the exposure state, ( EV, v, EV_t, moved ) in one RGBA32F texel (`state`), adapted from
//   the previous frame's (a persistent copy the graph makes at the frame's end: `previous`;
//   Exposure.ts `exposureAdapt`). The trimmed
//   means walk each histogram twice: its total, then the weight between the two percentiles, a bin
//   that straddles one counted in part, times its centre.
// - Without the composite pass (`post=inline-composite`) no histogram is taken: the buffer stays
//   empty and the meter reads none (Exposure.ts, No reading).
// - `read()` copies the buffer back for probes (a promise of the 256 counts, binned as stored: a
//   probe takes P, the diagnostics' `preExposure.stops`, back out).

/** Bins per histogram. */
export const METER_BINS = 128;
/** log2 of the lowest bin's lower edge, in stored units (scene units times 2^P). */
export const METER_LOW = - 16;
/** Bins per stop. */
export const METER_BINS_PER_STOP = 4;
/** Fixed-point weight of one sample. */
export const METER_ONE = 256;
/** The sampling grid: tiles across and down, samples per tile side. */
export const METER_TILES = [ 32, 16 ] as const;
export const METER_TILE = 16;

export class Metering {

	/** H_M then H_Y, METER_BINS counts each. */
	readonly buffer: THREE.StorageBufferAttribute;
	/** The exposure's keys (look/ExposureLook.ts `keysAt`, set once a frame), the GUI's offset
	 * (stops), the adaptation's step (the scene clock's, seconds) and its time constants rising and falling
	 * (seconds), and the reset (1: the state takes the target). */
	readonly uniforms = { key: uniform( 0 ), highlight: uniform( 0 ), skyShare: uniform( 0 ), offset: uniform( 0 ), dt: uniform( 0 ), up: uniform( 0.05 ), down: uniform( 0.05 ), snap: uniform( 1 ) };
	private readonly clearKernel: THREE.ComputeNode;
	private meterQuad: THREE.QuadMesh | null = null;
	private readonly histogramKernel: THREE.ComputeNode;

	/** `hdr` and `meterLight` are the graph's textures (resolved), and `surfaceDepth` the
	 * composite's surface depth (single-sampled, past -9e4 on the sky's pixels), all three at
	 * `size`'s pixel size, the render size; `view` the camera's inverse projection and world
	 * matrix, the atmosphere and the open sky's gains the dome draws the sky with (Sky.ts
	 * `gapGain`), for the sky's direction and shares (see the header). */
	constructor( hdr: THREE.Texture, meterLight: THREE.Texture, surfaceDepth: THREE.Texture, size: Uniform2,
		view: { projInv: THREE.Node<'mat4'>; cameraWorld: THREE.Node<'mat4'>; atmosphere: Atmosphere; gains: { sun: F; moon: F } } ) {

		const count = 2 * METER_BINS;
		this.buffer = new THREE.StorageBufferAttribute( new Uint32Array( count ), 1 );
		const bins = storage( this.buffer, 'uint', count ).toAtomic();

		this.clearKernel = Fn( () => {

			atomicStore( bins.element( instanceIndex ), uint( 0 ) );

		} )().compute( count, [ count ] ).setName( 'Metering.Clear' ) as THREE.ComputeNode;

		const tHdr = texture( hdr );
		const tMeter = texture( meterLight );
		const tSurface = texture( surfaceDepth );
		const grid = vec2( METER_TILES[ 0 ] * METER_TILE, METER_TILES[ 1 ] * METER_TILE );
		this.histogramKernel = Fn( () => {

			const local = workgroupArray( 'uint', count ).toAtomic() as unknown as { element: ( i: U ) => U };
			// (WebGPU zero-initialises workgroup memory)
			const cell = workgroupId.xy.mul( METER_TILE ).add( localId.xy );
			const at = vec2( cell ).add( 0.5 ).div( grid );
			const texel = ivec2( clamp( floor( at.mul( size ) ), vec2( 0 ), size.sub( 1 ) ) ).toVar();
			// (both as stored, pre-exposed: the meter takes P out of the means)
			// (the sky's at its own light: see the header)
			const sky = ( textureLoad( tSurface, texel ) as unknown as F ).lessThan( - 9e4 ).and( shared.seenNight.lessThan( 1 ) );
			const seen = luminance( ( textureLoad( tHdr, texel ) as unknown as THREE.Node<'vec4'> ).rgb ).toVar();
			// (the pixel's direction: NDC y up, uv's down)
			const v = view.projInv.mul( vec4( at.x.mul( 2 ).sub( 1 ), float( 1 ).sub( at.y.mul( 2 ) ), 0.5, 1 ) ).toVar();
			const d = normalize( view.cameraWorld.mul( vec4( v.xyz.div( v.w ), 0 ) ).xyz ).toVar();
			const shares = view.atmosphere.skyShares( vec3( d.x, max( d.y, 0 ), d.z ), true, view.gains );
			const sun = luminance( shares.sun ).toVar(), moon = luminance( shares.moon ).toVar();
			const level = sun.add( moon.mul( shared.seenNight ) ).div( max( sun.add( moon ), 1e-30 ) ).toVar();
			const y = select( sky, seen.div( max( level, 1e-6 ) ), seen ).toVar();
			const m = ( textureLoad( tMeter, texel ) as unknown as THREE.Node<'vec4'> ).r.toVar();
			// the tent: v the position in bin units, bin i's centre at v = i
			const splat = ( value: F, base: number ) => {

				If( value.greaterThan( 0 ), () => {

					const v = clamp( log2( value ).sub( METER_LOW ).mul( METER_BINS_PER_STOP ).sub( 0.5 ), 0, METER_BINS - 1 ).toVar();
					const i0 = clamp( floor( v ), 0, METER_BINS - 2 ).toVar();
					const upper = uint( round( v.sub( i0 ).mul( METER_ONE ) ) ).toVar();
					const index = uint( int( i0 ) ).add( base ).toVar();
					atomicAdd( local.element( index ), uint( METER_ONE ).sub( upper ) ).toStack();
					atomicAdd( local.element( index.add( 1 ) ), upper ).toStack();

				} );

			};
			splat( m, 0 );
			splat( y, METER_BINS );
			workgroupBarrier();
			const lane = localId.y.mul( METER_TILE ).add( localId.x ).toVar();
			const sum = ( atomicLoad( local.element( lane ) ) as unknown as U ).toVar();
			If( sum.greaterThan( 0 ), () => {

				atomicAdd( bins.element( lane ), sum ).toStack();

			} );

		} )().computeKernel( [ METER_TILE, METER_TILE, 1 ] ).setName( 'Metering.Histograms' ) as THREE.ComputeNode;

	}

	/** Builds the meter's one-texel pass: `sky` is log2 of the sky's geometric-mean luminance (scene units: shared.ts `skyMeterLog2`),
	 * `previous` the texture holding the previous frame's state. */
	buildMeter( sky: F, previous: THREE.Texture ): void {

		const counts = storage( this.buffer, 'uint', 2 * METER_BINS ).toReadOnly();
		// ( the trimmed mean of log2 between the fractions `low` and `high` of the total weight, the
		// total ) of the histogram from bin `base`
		const trimmed = Fn( ( [ base, low, high ]: [ THREE.Node<'uint'>, F, F ] ) => {

			const total = float( 0 ).toVar();
			Loop( METER_BINS, ( { i } ) => {

				total.addAssign( float( counts.element( base.add( uint( i ) ) ) ) );

			} );
			const from = total.mul( low ).toVar(), to = total.mul( high ).toVar();
			const below = float( 0 ).toVar(), weight = float( 0 ).toVar(), sum = float( 0 ).toVar();
			Loop( METER_BINS, ( { i } ) => {

				const w = float( counts.element( base.add( uint( i ) ) ) ).toVar();
				const share = max( min( below.add( w ), to ).sub( max( below, from ) ), 0 ).toVar();
				weight.addAssign( share );
				sum.addAssign( share.mul( float( i ).add( 0.5 ).div( METER_BINS_PER_STOP ).add( METER_LOW ) ) );
				below.addAssign( w );

			} );
			return vec2( sum.div( max( weight, 1e-6 ) ), total );

		} ).setLayout( { name: 'meterTrimmed', type: 'vec2', inputs: [ { name: 'base', type: 'uint' }, { name: 'low', type: 'float' }, { name: 'high', type: 'float' } ] } );
		const u = this.uniforms;
		const material = new THREE.NodeMaterial();
		material.name = 'Metering.Meter';
		material.fragmentNode = Fn( () => {

			const m = ( trimmed( uint( 0 ), float( 0.1 ), float( 0.9 ) ) as unknown as V2 ).toVar();
			const y = ( trimmed( uint( METER_BINS ), float( 0.9 ), float( 0.97 ) ) as unknown as V2 ).toVar();
			const one = float( METER_ONE );
			// (the means as stored, back to scene units: P is the frame's, the histograms' too)
			const mScene = m.x.sub( preExposure.log2Scale ), yScene = y.x.sub( preExposure.log2Scale );
			const t = exposureTarget( mScene, select( m.y.greaterThanEqual( one ), float( 1 ), float( 0 ) ), yScene, select( y.y.greaterThanEqual( one ), float( 1 ), float( 0 ) ), sky, u.key, u.highlight, u.skyShare, u.offset ) as unknown as THREE.Node<'vec3'>;
			const before = textureLoad( texture( previous ), ivec2( 0 ) ) as unknown as THREE.Node<'vec4'>;
			return exposureAdapt( t, before, u.dt, u.up, u.down, u.snap );

		} )();
		this.meterQuad = new THREE.QuadMesh( material );
		this.meterQuad.name = 'Metering.Meter';

	}

	/** The clear, declared before the composite. */
	clearPass<Frame>( g: FrameGraph<Frame> ): void {

		const kernel = this.clearKernel;
		g.addPass( 'meteringClear', 'compute', () => ( {
			label: 'metering', size: { of: 'fixed', width: 2 * METER_BINS, height: 1 }, neverCull: true, compute: [ kernel ],
			run: ( _f, graph ) => graph.renderer.compute( kernel ),
		} ) );

	}

	/** The histograms, declared after the composite and before the bloom, reading `hdr`,
	 * `meterLight` and `surfaceDepth` (the versions the composite and the scene passes left). */
	histogramPass<Frame>( g: FrameGraph<Frame>, hdr: ResourceHandle, meterLight: ResourceHandle, surfaceDepth: ResourceHandle ): void {

		const kernel = this.histogramKernel;
		g.addPass( 'meteringHistograms', 'compute', ( b: PassBuilder ) => {

			b.read( hdr );
			b.read( surfaceDepth );
			// (optional: the rain, its last writer under msaa, may be skipped)
			b.readOptional( meterLight );
			return {
				label: 'metering', size: { of: 'fixed', width: METER_TILES[ 0 ] * METER_TILE, height: METER_TILES[ 1 ] * METER_TILE }, neverCull: true, compute: [ kernel ],
				run: ( _f, graph ) => graph.renderer.compute( kernel, [ METER_TILES[ 0 ], METER_TILES[ 1 ], 1 ] ),
			};

		} );

	}

	/** The meter, declared after the histograms and before FSR 3 and the bloom: writes `state`;
	 * returns its new version. */
	meterPass<Frame>( g: FrameGraph<Frame>, state: ResourceHandle, previous: ResourceHandle ): ResourceHandle {

		const quad = this.meterQuad!;
		g.addPass( 'meter', 'quad', ( b: PassBuilder ) => {

			b.read( previous );
			state = b.write( state );
			return { label: 'metering', quad };

		} );
		return state;

	}

	/** The two histograms as last written: H_M at 0..127, H_Y at 128..255. */
	async read( renderer: THREE.WebGPURenderer ): Promise<Uint32Array> {

		return new Uint32Array( await renderer.getArrayBufferAsync( this.buffer ) );

	}

}
