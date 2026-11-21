import * as THREE from 'three/webgpu';
import { Fn, If, clamp, exp, float, max, mix, mod, smoothstep, vec2, vec3 } from 'three/tsl';
import { perlin3 } from '../materials/CloudNoise.ts';
import { CLOCK_PERIOD, evolutionTime, type WindUniforms } from './Wind.ts';
import { tune } from '../debug/Tune.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// The fog layer's banks: a noise factor that gathers the fog into banks the wind carries.
//
// The fog layer's banks (VolumetricHaze.ts, the light kernel): the fog layer's density times a
// factor of mean 1 that gathers the fog into banks with clear air between them, which the wind
// carries over the island and which slowly change their shape. The layer's height profile is the
// banks' mean; the factor only moves its fog about.
//
// - The field: the sum of two fields of Perlin noise (Perlin, "Improving Noise", SIGGRAPH 2002;
//   CloudNoise.ts `perlin3`), each scaled to a spread of about 1 (one octave's standard deviation
//   is about 0.22, OCTAVE_SPREAD), weighted so the sum's spread stays 1:
//   - the regions: one octave in cells `region` metres across, `regionShare` of the variance, so
//     whole stretches of the sea hold more fog than others;
//   - the detail: four octaves, each at twice the last's frequency and `rough` times its weight,
//     in cells `scale` metres across and `squash` times that high (banks wider than they are
//     tall), the rest of the variance. Its coordinates are first displaced across by two more
//     octaves at the detail's own scale, `warp` cells per unit of their spread (domain warping:
//     Quilez, "Domain warping", iquilezles.org), which draws the banks out into the tongues and
//     curls of drifting fog rather than round blobs.
// - The banks: the factor is the field over a threshold, max( n - c, 0 ), divided by that ramp's
//   mean over a unit normal field, phi( c ) - c ( 1 - Phi( c ) ), so its mean is 1 at every
//   threshold. The threshold rises with height, from `low` at the sea to `high` at `ramp` metres
//   (the layer's top and its edge's width): low down the fog closes into a sheet over the water,
//   higher up it breaks into banks, and the banks' tops stand at different heights. Where the
//   threshold is high the banks are few and dense, the clear air between them wide. At -1 and
//   0.5 the fog is a near-continuous veil whose density varies by a factor of a few, the way a
//   deep sea fog thins and thickens as it drifts; raised past 1, it breaks into separate banks.
// - Its motion: the field moves with the air (Wind.ts `displacement`, the weather's integrated
//   travel, never time * a speed) and evolves along its height axis at FOG_EVOLUTION cells a
//   second, an integrated time the wind gives (Wind.ts `evolutionTime`, the gusts' time axis
//   rescaled, which wraps every FOG_EVOLUTION x CLOCK_PERIOD cells), so both are functions of the
//   scene clock and the weather, held when the wind is frozen. The field repeats every FOG_PERIOD
//   cells on every axis, which is where the evolution's time wraps, and the travel is taken modulo
//   FOG_PERIOD cells, so neither wrap shows and the travel keeps its digits.
// - Far slices are many cells deep: one sample there is noise of its own, which flickers as the
//   field moves. Each field's weight fades to 0 as a slice's length grows from half its cell to
//   one and a half (the detail first, then the regions), and with the regions' the factor fades to
//   its mean, 1, so the far fog keeps the layer's profile.

/** The field's evolution, cells a second (a bank reshapes within about 1 / FOG_EVOLUTION s). */
export const FOG_EVOLUTION = 0.02;
/** The field's period in cells on every axis, where the evolution's time wraps. */
export const FOG_PERIOD = FOG_EVOLUTION * CLOCK_PERIOD;

/** The parameters: the cell's size (metres) and its height over its width, the amplitude, the threshold
 * at the sea and at the ramp's top, the regions' cell (metres) and share of the field's variance,
 * the detail's roughness and its warp (cells)
 * (`haze.fog.<noiseScale|squash|noiseAmp|coverLow|coverHigh|regionScale|regionShare|rough|warp>`). */
export const FOG_NOISE = {
	scale: tune( 'haze.fog.noiseScale', 40, { min: 10, max: 150, scale: 'log' } ),
	squash: tune( 'haze.fog.squash', 0.45, { min: 0.15, max: 1 } ),
	amplitude: tune( 'haze.fog.noiseAmp', 1, { min: 0, max: 1 } ),
	low: tune( 'haze.fog.coverLow', - 1, { min: - 2, max: 1.5 } ),
	high: tune( 'haze.fog.coverHigh', 0.5, { min: - 1, max: 2.5 } ),
	region: tune( 'haze.fog.regionScale', 300, { min: 80, max: 1500, scale: 'log' } ),
	regionShare: tune( 'haze.fog.regionShare', 0.3, { min: 0.05, max: 0.95 } ),
	rough: tune( 'haze.fog.rough', 0.6, { min: 0.3, max: 0.8 } ),
	warp: tune( 'haze.fog.warp', 0.3, { min: 0, max: 1.5 } ),
};

// one octave's spread (see above)
const OCTAVE_SPREAD = 0.22;

// the detail field at p (cells), of spread about 1, repeating every `period` cells: four octaves,
// each `rough` times the last's weight
const fogField = Fn( ( [ p, period, rough ]: [ V3, F, F ] ) => {

	const r2 = rough.mul( rough ), r3 = r2.mul( rough );
	const a = perlin3( p, period );
	const b = perlin3( p.mul( 2 ).add( vec3( 17, 5, 9 ) ), period.mul( 2 ) );
	const c = perlin3( p.mul( 4 ).add( vec3( 3, 11, 23 ) ), period.mul( 4 ) );
	const d = perlin3( p.mul( 8 ).add( vec3( 13, 19, 2 ) ), period.mul( 8 ) );
	const sum = a.add( b.mul( rough ) ).add( c.mul( r2 ) ).add( d.mul( r3 ) );
	return sum.div( r2.add( r2.mul( r2 ) ).add( r3.mul( r3 ) ).add( 1 ).sqrt().mul( OCTAVE_SPREAD ) );

} ).setLayout( { name: 'fogField', type: 'float', inputs: [ { name: 'p', type: 'vec3' }, { name: 'period', type: 'float' }, { name: 'rough', type: 'float' } ] } );

// the mean of max( n - c, 0 ) over a unit normal n: phi( c ) - c ( 1 - Phi( c ) ), Phi by its
// logistic fit
const rampMean = ( c: F ): F => exp( c.mul( c ).mul( - 0.5 ) ).mul( 0.3989 ).sub( c.div( exp( c.mul( 1.702 ) ).add( 1 ) ) );

/** The banks' factor (mean 1, never below 0) at the world point `p`, for a slice `length` metres
 * deep; `noise` holds the cell's size (m), the amplitude, the cells' height over their width and the
 * regions' cell (m), `cover` the threshold at the sea and at `cover.z` metres up and the regions'
 * share of the field's variance, `wind` the weather's wind. */
export function fogDrift( p: V3, length: F, noise: V4, cover: V4, shape: V4, wind: WindUniforms ): F {

	const time = evolutionTime( wind, FOG_EVOLUTION );
	// p in cells `size` metres across, carried by the wind
	const cells = ( size: F ): V3 => {

		const travel = mod( wind.displacement, vec2( FOG_PERIOD ).mul( size ) );
		return vec3( p.x.sub( travel.x ).div( size ), p.y.div( size.mul( noise.z ) ).add( time ), p.z.sub( travel.y ).div( size ) );

	};
	// each field's weight, its share of the variance faded as the slice outgrows its cell
	const regionWeight = clamp( float( 1.5 ).sub( length.div( noise.w ) ), 0, 1 ).toVar();
	const detailWeight = clamp( float( 1.5 ).sub( length.div( noise.x ) ), 0, 1 ).mul( float( 1 ).sub( cover.w ).sqrt() ).toVar();
	const field = float( 0 ).toVar();
	If( regionWeight.greaterThan( 0 ), () => {

		field.assign( perlin3( cells( noise.w ).add( vec3( 29, 13, 7 ) ), float( FOG_PERIOD ) ).div( OCTAVE_SPREAD ).mul( cover.w.sqrt() ) );
		If( detailWeight.greaterThan( 0 ), () => {

			const q = cells( noise.x ).toVar();
			If( shape.y.greaterThan( 0 ), () => {

				const wx = perlin3( q.add( vec3( 5, 31, 17 ) ), float( FOG_PERIOD ) );
				const wz = perlin3( q.add( vec3( 23, 3, 41 ) ), float( FOG_PERIOD ) );
				q.addAssign( vec3( wx, 0, wz ).mul( shape.y.div( OCTAVE_SPREAD ) ) );

			} );
			field.addAssign( fogField( q, float( FOG_PERIOD ), shape.x ).mul( detailWeight ) );

		} );

	} );
	// (rescaled to a unit spread whatever the detail's weight)
	const n = field.div( cover.w.add( detailWeight.mul( detailWeight ) ).sqrt() );
	const c = mix( cover.x, cover.y, smoothstep( 0, 1, clamp( p.y.div( cover.z ), 0, 1 ) ) ).toVar();
	const banks = max( n.sub( c ), 0 ).div( rampMean( c ) );
	return mix( float( 1 ), banks, noise.y.mul( regionWeight ) );

}
