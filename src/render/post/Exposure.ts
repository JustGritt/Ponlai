import * as THREE from 'three/webgpu';
import { Fn, If, abs, clamp, exp, exp2, float, log2, max, min, mix, select, vec3, vec4 } from 'three/tsl';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// The frame's exposure target, in stops, from the metering histograms and the sky.
//
// The frame's exposure target, in stops (EV = log2 of the factor that scales scene radiance into
// the exposed image), from the metering histograms (Metering.ts) and the sky. The meter reads the
// light falling on the scene, not the light leaving it, so the albedo stays out of it: a dark
// meadow stays dark against a pale beach (after Patry, "Real-Time Samurai Cinema", SIGGRAPH 2021:
// a primarily illuminance-based exposure).
// - m_M: the mean of log2 M (the metering illuminance) over its histogram between the 10th and the
//   90th percentile of the total weight, bins at their centres, a bin that straddles a percentile
//   counted in part: a trimmed mean that ignores crevices and glints. The histograms bin the values
//   as stored, pre-exposed, so their range follows the exposure; the meter takes P back out of both
//   means, which reach the target in scene units (Metering.ts, PreExposure.ts).
// - m_H: the mean of log2 Y (hdr's luminance) between its 90th and 97th percentile: the
//   brightness of the frame's brightest few percent. The sky's pixels enter it at the sky's own
//   light, before the night sky's seen gain darkens the drawn sky (Metering.ts, shared.ts
//   `seenNight`), so the exposure follows the light the scene takes, not how dark the night sky is
//   drawn over it.
// - m_S: log2 of the sky's geometric-mean luminance exp( mean ln Y ) over the upper hemisphere
//   (shared.ts `skyMeterLog2`: Sky.ts reduces it from each environment bake, and the dome writes
//   the same luminance as its metering illuminance), so a sky-filled view is metered by the sky's
//   overall brightness, not by the bright sky about the sun.
// - The keys (look/ExposureLook.ts `keys`, DayKeys.ts): the key κ (a white diffuser under
//   the frame's typical light lands at 2^κ), the highlight key κ_H (the bright band lands at 2^κ_H
//   at most) and the sky's share β.
// - The target, in three steps:
//   - The scene's exposure: e_M = κ - m_M, held down by the highlight guard e_H = κ_H - m_H as
//     their soft minimum, e_G = -k log2( 2^( -e_M / k ) + 2^( -e_H / k ) ) with k = SOFT_MIN (a
//     quarter of a stop): the lower of the two where they lie apart, eased where they meet, so the
//     exposure does not turn a corner as a view's highlight guard takes over from its meter (at most
//     k under the lower, where they are equal; under a tenth of a stop once they lie half a stop
//     apart).
//   - The blend: the sky's own exposure e_S = κ - m_S and the scene's blend in stops by the sky's
//     share, e = ( 1 - β ) e_G + β e_S: a weighted metering pattern, the sky weighing into the meter
//     as a region of the frame does in a camera's weighted metering. The guard holds down the
//     scene's share alone, never the sky's: the bright band about a low sun (the glitter on the sea,
//     the glow) darkens what the land's light asks for, while the sky's part still follows the
//     sky's own light, so the sky weighs the same in every view under one sky.
//   - The bound: EV_t = clamp( e, e_S - 4, e_S + 5 ) plus the GUI's offset (stops), both bounds
//     offset too (EXPOSURE_UNDER, EXPOSURE_OVER): a window about the sky's exposure that the
//     adaptation holds EV in (`exposureAdapt`). Across the kept weathers' views e lies from about
//     2.6 stops under e_S to 1.6 over it (the land at the blue hour and before sunrise): the land's
//     illuminance, the sun's included, lies over the sky's own luminance by day and by night. The
//     lowest is a free camera low over the sea looking toward a low sun (guard-held: 2.6 stops
//     under at 12 degrees, 1.2 to 1.7 at 1 to 3 degrees). The window keeps a frame of glare or of
//     black from carrying the exposure past what the sky allows, and leaves every kept view free,
//     the nearest 1.4 stops inside it.
// - No reading: where M's histogram holds less than a sample's weight (no metered material in
//   view, or no material writing M at all), e_M is the sky's exposure e_S, so the scene's share is
//   the sky's too, still held down by the highlight guard; where Y's holds none, the guard drops
//   out and the scene's share is e_M alone.
// - The night's darkness is the night key's job: the meter reads night light as it reads the day's.
//   The night key comes in by the exposure's own night weight (sky/DayCycle.ts `exposureNight`),
//   not the display's: where the meter hands over to it is set by the light, apart from the grade.
// - Adaptation (`exposureAdapt`): the exposure EV follows its target EV_t as a critically damped
//   second-order filter on EV, EV'' = -2 ω EV' - ω^2 ( EV - EV_t ), ω = 1 / τ (Lowe, "Critically
//   Damped Ease-In/Ease-Out Smoothing", Game Programming Gems 4, 2004): it starts with no kink;
//   from rest a held target is reached without overshoot; a target that moves while the EV is
//   moving can be passed by a little. A target that turns back mid-way bends the motion rather
//   than restarting it. Each frame takes the filter's exact step over the scene clock's step dt
//   for the target held over it, with y = EV - EV_t and v = EV':
//   y' = ( y + ( v + ω y ) dt ) e^( -ω dt ), v' = ( v - ω ( v + ω y ) dt ) e^( -ω dt ), stable at
//   any dt and the same curve at any frame rate. τ is keyed by the way the exposure moves:
//   `adaptUp` while it rises (the view grew darker), `adaptDown` while it falls (the view grew
//   brighter). Both are 0.05 s in every key, the eye's light adaptation: Durand and Dorsey model
//   the cones' recovery of sensitivity after a rise or a small fall of the light as a first-order
//   filter of time constant 0.1 s ("Interactive Tone Mapping", Eurographics Workshop on
//   Rendering, 2000), and the scene's changes of light by day (a pan from the sea to the land, a
//   cloud's shadow) are of that kind, not the minutes of dark adaptation after a large fall. Two
//   such filters in a row, as this one is, lag an input that moves at a steady rate (a pan) by
//   2 τ, a first-order filter by its own time constant, so τ = 0.05 s lags a pan by the eye's
//   0.1 s. From rest the share of a held step still to go is y( t ) / y( 0 ) = ( 1 + ω t )
//   e^( -ω t ), so half the step is covered in about 0.084 s and 90 % of it in about 0.19 s.
//   Held at a bound of the target's range, EV stops there and its
//   velocity is cleared, so it leaves the bound from rest.
//   The state, one RGBA32F texel: ( EV, v, EV_t, moved ), stops, stops per second, stops, and 1
//   where this frame's EV differs from the previous frame's (as floats, exactly), else 0. A reset
//   (the app's first frame, a scene event that cuts), a paused clock (dt 0) and so the test mode,
//   a paused page, put EV = EV_t and v = 0: a paused frame equals a fresh load's. `moved` is a
//   diagnostic (the app's exposure readout): after a snap it is 1 for one frame, until EV stands
//   still. Nothing in the frame reads the previous frame's exposure before the meter.
// - A stepped clip (capture-clip.mjs, `window.__PONLAI_CLIP__` set) snaps the same way on every
//   frame: a temporal rest converged on a stopped clock jumps to its target while running frames
//   still lag behind theirs, so a clip compared with its rest or with MSAA would differ by the lag
//   alone. Freezing the adaptation on every side compares the images, not the adaptation.
// - EV stays absolute (scene units to the exposed image). `hdr` is stored pre-exposed by 2^P
//   (PreExposure.ts), so the display chain reads the exposure as 2^( EV - P ) (Pipeline.ts
//   `exposureRead`), and P follows this EV read back a frame or two late.

/** The window about the sky's exposure e_S the target is held in, stops under and over it (see the
 * header). */
export const EXPOSURE_UNDER = 4;
export const EXPOSURE_OVER = 5;
/** The soft minimum's width k, in stops (see the header). */
export const SOFT_MIN = 0.25;

/** The target: ( EV_t, its lower bound, its upper bound ), stops, from the trimmed means `meter`
 * ( m_M ) and `high` ( m_H ), whether each histogram had a reading (`hasMeter`, `hasHigh`: 0 or
 * 1), the sky's log2 luminance `sky` ( m_S ), the keys and the GUI's `offset` (stops). */
export const exposureTarget = Fn( ( [ meter, hasMeter, high, hasHigh, sky, key, highlight, skyShare, offset ]: F[] ) => {

	// the sky's exposure and the scene's (the sky's where the scene has no reading)
	const eS = key.sub( sky ).toVar();
	const eM = select( hasMeter.greaterThan( 0.5 ), key.sub( meter ), eS ).toVar();
	// the highlight guard on the scene's branch alone: the soft minimum, as the lower one less its
	// easing (no power of a large value to overflow)
	const eH = highlight.sub( high ).toVar();
	const guarded = min( eM, eH ).sub( log2( exp2( abs( eM.sub( eH ) ).div( - SOFT_MIN ) ).add( 1 ) ).mul( SOFT_MIN ) ).toVar();
	const scene = select( hasHigh.greaterThan( 0.5 ), guarded, eM );
	// the two blended in stops by the sky's share, held in the window about the sky's exposure
	const e = mix( scene, eS, skyShare );
	const lo = eS.sub( EXPOSURE_UNDER ).add( offset ), hi = eS.add( EXPOSURE_OVER ).add( offset );
	return vec3( clamp( e.add( offset ), lo, hi ), lo, hi );

} ).setLayout( { name: 'exposureTarget', type: 'vec3', inputs: [
	{ name: 'meter', type: 'float' }, { name: 'hasMeter', type: 'float' }, { name: 'high', type: 'float' }, { name: 'hasHigh', type: 'float' },
	{ name: 'sky', type: 'float' }, { name: 'key', type: 'float' }, { name: 'highlight', type: 'float' }, { name: 'skyShare', type: 'float' }, { name: 'offset', type: 'float' },
] } );

/** The exposure state's step: from the target `aim` ( EV_t, its bounds: `exposureTarget` ), the
 * previous state ( EV, v, EV_t, moved ), the scene clock's step `dt` (seconds), the time
 * constants `up` and `down` (seconds) and `snap` (1: reset to the target), the new state. */
export const exposureAdapt = Fn( ( [ aim, previous, dt, up, down, snap ]: [ V3, V4, F, F, F, F ] ) => {

	const t = aim.x.toVar(), lo = aim.y.toVar(), hi = aim.z.toVar();
	const y0 = previous.x.sub( t ).toVar(), v0 = previous.y.toVar();
	// (rising while the target lies over the exposure)
	const tau = select( y0.lessThan( 0 ), up, down ).toVar();
	const w = float( 1 ).div( max( tau, 1e-4 ) ).toVar();
	const decay = exp( w.mul( dt ).negate() ).toVar();
	const c = v0.add( w.mul( y0 ) ).mul( dt ).toVar();
	const ev = t.add( y0.add( c ).mul( decay ) ).toVar();
	const v = v0.sub( w.mul( c ) ).mul( decay ).toVar();
	If( ev.lessThan( lo ).or( ev.greaterThan( hi ) ), () => {

		ev.assign( clamp( ev, lo, hi ) );
		v.assign( 0 );

	} );
	If( snap.greaterThan( 0.5 ), () => {

		ev.assign( t );
		v.assign( 0 );

	} );
	return vec4( ev, v, t, select( ev.notEqual( previous.x ), float( 1 ), float( 0 ) ) );

} ).setLayout( { name: 'exposureAdapt', type: 'vec4', inputs: [
	{ name: 'aim', type: 'vec3' }, { name: 'previous', type: 'vec4' }, { name: 'dt', type: 'float' },
	{ name: 'up', type: 'float' }, { name: 'down', type: 'float' }, { name: 'snap', type: 'float' },
] } );
