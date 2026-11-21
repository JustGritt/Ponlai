/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 * Based on the spectrum helpers in src/ocean/OceanFFT.js and the sea states in src/ui/AppUI.js at
 * 4811ba48d795197de5621985f404e765c0b7c0ef.
 */

// The ocean's wave spectrum on the CPU: the sea states, the wave systems and slope variance.

const GRAVITY = 9.81;

export interface WaveSpectrumSettings {
	scale?: number; // multiplies the whole spectrum (tidewater's WaveSystem.scale)
	windSpeed: number;
	windDirection: number; // degrees in the Fourier convention; the positive IFFT travels oppositely
	fetch: number; // metres, unlike Tidewater's kilometre-valued UI
	spread: number; // blend from broad cosine-squared spreading to frequency-dependent spreading
	swell: number; // extra directional concentration, independently of spectral height
	peakEnhancement: number;
	shortWavesFade: number;
	peakPeriod?: number;
}

// tidewater's distant swell: a second wave system of its own wind and fetch, at a heading of its
// own (tidewater's 5 degrees against the local sea's 25), scaled by the sea state. Its values are
// the look's (look/Look.ts `SeaLook`), which the render pipeline's stage hands the FFT ocean as it
// makes it (RenderStages.ts `createSeaAndPipeline`).
export interface SwellParams {
	scale: number;
	windSpeed: number;
	fetch: number; // m
	directionOffset: number; // degrees from the local sea's heading
	spread: number;
	swell: number;
	peakEnhancement: number;
	shortWavesFade: number;
}

// tidewater's sea states (AppUI.js `SEA`), keyed by the wind 10 m up: the local sea's fetch, the
// choppiness, the swell's scale and the whitecaps (0..1). The wind levels of weather/Wind.ts fall on them: gentle (3.8 m/s)
// next to Calm, fresh (7) on Breezy, tidewater's default, strong (11) next to Choppy.
const SEA_STATES = [
	{ wind: 3.5, fetch: 40e3, choppiness: 0.75, swell: 0.28, whitecaps: 0.2 }, // Calm
	{ wind: 7, fetch: 120e3, choppiness: 0.9, swell: 0.48, whitecaps: 0.5 }, // Breezy
	{ wind: 12, fetch: 300e3, choppiness: 1.05, swell: 0.68, whitecaps: 0.75 }, // Choppy
	{ wind: 20, fetch: 900e3, choppiness: 1.2, swell: 1.0, whitecaps: 1 }, // Storm
];

/** The sea state at a wind speed, interpolated between tidewater's (held at the first and last),
 * with its whitecaps' threshold on the surface's compression (AppUI.js `whitecaps()`). */
export function seaState( windSpeed: number ): { fetch: number; choppiness: number; swell: number; foamBias: number } {

	const s = SEA_STATES;
	let i = 0;
	while ( i < s.length - 2 && windSpeed > s[ i + 1 ].wind ) i ++;
	const t = Math.min( Math.max( ( windSpeed - s[ i ].wind ) / ( s[ i + 1 ].wind - s[ i ].wind ), 0 ), 1 );
	const mix = ( a: number, b: number ) => a + ( b - a ) * t;
	const whitecaps = mix( s[ i ].whitecaps, s[ i + 1 ].whitecaps );
	return { fetch: mix( s[ i ].fetch, s[ i + 1 ].fetch ), choppiness: mix( s[ i ].choppiness, s[ i + 1 ].choppiness ), swell: mix( s[ i ].swell, s[ i + 1 ].swell ),
		foamBias: 0.5 + 0.16 * whitecaps + 0.01 * Math.min( Math.max( windSpeed - 7, - 5 ), 12 ) };

}

function dispersion( k: number, depth: number ): number {

	return Math.sqrt( GRAVITY * k * Math.tanh( Math.min( k * depth, 20 ) ) );

}

function dispersionDerivative( k: number, depth: number ): number {

	if ( k < 1e-8 ) return Math.sqrt( GRAVITY * depth );
	const kd = Math.min( k * depth, 20 ), th = Math.tanh( kd );
	return GRAVITY * ( th + k * depth * ( 1 - th * th ) ) / ( 2 * dispersion( k, depth ) );

}

function tmaCorrection( omega: number, depth: number ): number {

	const omegaH = omega * Math.sqrt( depth / GRAVITY );
	if ( omegaH <= 1 ) return omegaH * omegaH * 0.5;
	if ( omegaH < 2 ) return 1 - ( 2 - omegaH ) ** 2 * 0.5;
	return 1;

}

function normalisationFactor( s: number ): number {

	// Tidewater's fit keeps cos^(2s) direction distributions comparable in energy as
	// their width changes with frequency; spread must not double as an amplitude dial.
	const s2 = s * s, s3 = s2 * s, s4 = s3 * s;
	return s < 5 ? - 0.000564 * s4 + 0.00776 * s3 - 0.044 * s2 + 0.192 * s + 0.163
		: - 4.80e-8 * s4 + 1.07e-5 * s3 - 9.53e-4 * s2 + 0.059 * s + 0.393;

}

/** A wave system's spectral terms (tidewater's `updateSpectrumUniforms`): the JONSWAP scale
 * (times the system's own scale) and peak frequency, the heading in radians, the spreading blend,
 * the directional concentration, the peak enhancement and the short waves' squared fade length. */
export interface WaveSystemTerms { alpha: number; peakOmega: number; angle: number; spread: number; concentration: number; gamma: number; fade2: number }

export function waveSystemTerms( settings: WaveSpectrumSettings ): WaveSystemTerms {

	if ( settings.peakPeriod !== undefined && ( ! Number.isFinite( settings.peakPeriod ) || settings.peakPeriod <= 0 ) ) throw new RangeError( 'Swell period must be positive and finite.' );
	const U = Math.max( 0.1, settings.windSpeed ), fetch = Math.max( 1, settings.fetch );
	return {
		alpha: 0.076 * Math.pow( U * U / ( fetch * GRAVITY ), 0.22 ) * ( settings.scale ?? 1 ),
		peakOmega: settings.peakPeriod === undefined ? 22 * Math.pow( GRAVITY * GRAVITY / ( U * fetch ), 0.33 ) : 2 * Math.PI / settings.peakPeriod,
		angle: settings.windDirection * Math.PI / 180,
		spread: Math.max( 0, Math.min( 1, settings.spread ) ),
		concentration: Math.max( 0, Math.min( 1, settings.swell ) ),
		gamma: Math.max( 1, settings.peakEnhancement ),
		fade2: settings.shortWavesFade ** 2,
	};

}

// The TMA-corrected JONSWAP energy per unit wavenumber at k, summed over the directions (both
// spreads integrate to 1 over the circle), with the short waves' fade.
function radialEnergy( k: number, depth: number, { alpha, peakOmega, gamma, fade2 }: WaveSystemTerms ): number {

	const omega = dispersion( k, depth );
	const sigma = omega <= peakOmega ? 0.07 : 0.09;
	const r = Math.exp( - ( ( omega - peakOmega ) ** 2 ) / ( 2 * sigma * sigma * peakOmega * peakOmega ) );
	const jonswap = tmaCorrection( omega, depth ) * alpha * GRAVITY * GRAVITY / omega ** 5
		* Math.exp( - 1.25 * ( peakOmega / omega ) ** 4 ) * gamma ** r;
	return jonswap * Math.exp( - fade2 * k * k ) * dispersionDerivative( k, depth );

}

/** Returns the two-dimensional k-space energy density, including the omega-to-k Jacobian (the
 * GPU's is OceanFFT's `waveEnergy`, the same formulas). */
export function createWaveSpectrum( settings: WaveSpectrumSettings, depth: number ): ( kx: number, kz: number ) => number {

	if ( ! Number.isFinite( depth ) || depth <= 0 ) throw new RangeError( 'Ocean depth must be positive and finite.' );
	const terms = waveSystemTerms( settings );
	const { peakOmega, angle, spread, concentration } = terms;
	return ( kx, kz ) => {

		const k = Math.hypot( kx, kz );
		if ( k < 1e-6 ) return 0;
		const ratio = dispersion( k, depth ) / peakOmega;
		const spreadPower = ( ratio > 1 ? 9.77 * ratio ** - 2.5 : 6.97 * ratio ** 5 )
			+ Math.tanh( Math.min( ratio, 20 ) ) * 16 * concentration * concentration;
		const theta = Math.atan2( kz, kx ) - angle, c = Math.cos( theta );
		const broad = c > 0 ? c * c * 2 / Math.PI : 0;
		const narrow = normalisationFactor( spreadPower ) * Math.abs( Math.cos( theta * 0.5 ) ) ** ( 2 * spreadPower );
		const direction = broad + ( narrow - broad ) * spread;
		return radialEnergy( k, depth, terms ) * direction / k;

	};

}

/** Cox and Munk's mean square slope of a clean sea surface (1954: sun glitter photographed from the
 * air), both axes together, at a wind of U m/s: 0.003 + 0.00512 U. It counts every wave down to the
 * capillaries. */
export const COX_MUNK = { base: 0.003, perWind: 0.00512 } as const;

/** The share of the circle of radius k inside the square of half width k1 (the directions taken as
 * even there): 1 within the square's inscribed circle, falling to 0 at its corners. */
export function squareShare( k: number, k1: number ): number {

	return 1 - 4 / Math.PI * Math.acos( Math.min( k1 / k, 1 ) );

}

/** A wave system's slope spectrum: the mean square slope per unit wavenumber at k (rad/m), its energy
 * times k squared over the ring, every direction summed. */
export function slopeDensity( settings: WaveSpectrumSettings, depth: number ): ( k: number ) => number {

	const terms = waveSystemTerms( settings );
	return ( k ) => radialEnergy( k, depth, terms ) * k * k;

}

/** The mean square slope a wave system holds between the wavenumbers k0 and k1 (rad/m): its energy
 * times k squared over that ring. With `square`, k1 is the half width of a square of modes (a
 * cascade's last mode before its Nyquist) and the square's corners are counted too, each ring of
 * them by the share of its circle inside the square (`squareShare`). */
export function slopeVariance( settings: WaveSpectrumSettings, depth: number, k0: number, k1: number, square = false ): number {

	const density = slopeDensity( settings, depth );
	const integrate = ( a: number, b: number, share: ( k: number ) => number ) => {

		// over log k, where the slope spectrum is nearly flat (Simpson's rule, 96 intervals)
		const n = 96, span = Math.log( b / a );
		let sum = 0;
		for ( let i = 0; i <= n; i ++ ) {

			const k = a * Math.exp( span * i / n );
			const w = i === 0 || i === n ? 1 : i % 2 ? 4 : 2;
			sum += w * density( k ) * k * share( k );

		}
		return sum * span / ( 3 * n );

	};
	const ring = integrate( k0, k1, () => 1 );
	return square ? ring + integrate( k1, k1 * Math.SQRT2, ( k ) => squareShare( k, k1 ) ) : ring;

}
