import * as THREE from 'three/webgpu';
import { min, uniform, vec3, vec4 } from 'three/tsl';
import { restEpoch } from './TemporalRest.ts';
import { tuneApply } from '../debug/Tune.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

// Pre-exposure: one scale on the half-float targets that keeps their values in range.
//
// Pre-exposure: every writer of the scene's half-float colour (`sceneColor`, and so `hdr`) and of
// the metering illuminance (`meterLight`) multiplies its value by 2^P, one uniform for the whole
// frame, so what the targets hold sits in half float's range whatever the light. Scene units (what
// the lighting computes, SCENE_RADIANCE_SCALE) span about 2^-19 to 2^12 between a clear night's
// shade and noon's glints; half float keeps 2^-14 to 65504 at full precision. Unscaled, 62 % of a
// clear night's non-sky hdr pixels in the island view (45 % in the horizon view) and 2.3 % of its
// meterLight were subnormal, and stored a little dark: pre-exposed, the night's and the dusk's
// images come out up to about 0.8 of an 8-bit level brighter (0.3 to 0.6 over the frame), noon's
// unchanged but for a few pixels.
// - P = round( EV ) - PRE_EXPOSURE_BIAS, clamped to PRE_EXPOSURE_RANGE, from the exposure EV of the
//   last frame read back (Exposure.ts: EV is log2 of the factor that brings scene radiance to the
//   exposed image). Raw = scene x 2^P = exposed x 2^( P - EV ), so the exposed image's mid-grey
//   (about 0.18) is stored near 2^-5.5 and its brightest glints (noon's sun on the sea, about 2^17
//   exposed) near 2^14, both well inside the range: at the bias of 3 a clear night's darkest non-sky
//   value is stored about 3.5 stops above 2^-14 (7 in meterLight) and noon's brightest about 2.5
//   stops below 65504. With no bias (P = EV, the exposed image itself) those glints overflowed.
// - Whole stops, not the exact EV: a power of two scales a half float exactly (only the exponent
//   changes), so wherever no value under- or overflowed the image is bit for bit what it was
//   unscaled, and every reader that divides P back out gets the same value it got before. An exact
//   EV would round every stored value differently each frame and make each frame's image depend on
//   the exposure's path. No hysteresis either: P is a function of the EV read back, so a paused page
//   (whose exposure snaps to its target) reaches the same P whatever came before.
// - The previous frame's: the exposure is computed on the GPU after the scene, so P comes from a
//   readback (Pipeline.ts, one in flight, a frame or two late). The scale only has to keep values in
//   range, so a P a few frames old is harmless; P moves when the EV crosses a half stop.
// - A scene event that cuts (but a loop restart or the page shown again, which keep the scene)
//   resets P to the fresh load's, 0: a scene jump then takes the path of a fresh load. A new P bumps
//   the rest's epoch (TemporalRest.ts), not counted as a change of the scene (the haze's rest holds
//   the sun's visibility, which no P scales, and keeps going), so the temporal rests restart on
//   frames of one P.
// - `pending` holds the settled signal until the P a frame used is the one its own exposure asks
//   for, read from a frame of the settled view: one drawn at least the tier's settle frames
//   (`restSettle`, Settled.ts) into a run of frames that share the view (`viewFrame`: the rest's
//   key without the pre-exposure's own epoch bumps) while the page waited for nothing but the
//   pre-exposure (`quiet`: stable, nothing else in assemble/SceneFrame.ts's `pending`, no pipeline compiling). Any
//   other reading still sets P but never ends `pending`, and a new key or a frame that is not quiet
//   makes it pending again at once, cut or not (a retune, a camera moved on a paused page): the
//   readings still in flight are of the earlier frames. A readback can lag many frames (more on a
//   shared GPU), and readings come one at a time, so the reading that ends `pending` may be of a
//   frame drawn long before. Two kinds of frame drawn before the view settled ask for a slightly
//   other EV than the settled view: a frame of an earlier view and a frame of this view drawn
//   while what settles without changing the key still moved (the haze's shadow while its rest
//   averages its sequence, which holds the settled signal itself; the clouds' view until its second
//   still frame). Where the settled view's EV lay near a half stop, such a reading ended `pending`
//   with the other P; the settled signal then held after its settle frames while no reading of a
//   settled frame had come back yet, and the reading that came back next changed P under a page
//   already reported settled: two runs of one view caught different P (the metering a sample apart,
//   the EV in its last digits), or the frame of the change. A frame of the settled view is the
//   frame the page draws from then on, so under MSAA (and aa=off) every later reading asks for the
//   P that ended `pending`, and P does not change under a settled page. Under the temporal modes the
//   meter reads the jittered image before the resolve, whose EV moves a little from frame to frame
//   once the rest holds and the jitter runs on: an EV nearer a half stop than that can still move P
//   (the epoch bump then restarts the rest, and the signal turns false).
// - The settled P is then a function of the view as long as the EV a frame meters does not depend
//   on the P it was drawn at. It does only at half float's edges (a value that is subnormal or
//   flushed to 0 at one P and not the other, the HALF_MAX clamp; the metering histograms' ends lie
//   at those edges too, Metering.ts): the bias keeps every non-sky value of the views measured
//   above off them at the neighbouring P too, and the sun's disc falls in the meter's top bin at
//   any P. A view with values there would
//   meter neighbouring P apart by those values alone, and an EV within that distance of a half stop
//   then has no single P. At the subnormal edge a higher P keeps more of the dark values, meters
//   brighter and asks for a lower P: the two P ask for each other, P alternates and the page never
//   settles. At the HALF_MAX clamp a higher P clips more of the bright values, meters darker and
//   asks for a higher P: each P asks for itself, and the page settles on whichever it reached
//   first.
// - Without the settled signal's inputs (production builds) no frame counts as quiet: P follows the
//   readings as above and `pending` stays true, which nothing reads there.
// - Readers that work in scene units divide it out (`inverse`): the water's reads of the opaque
//   image, the underwater pyramid and the refraction image (Water.ts), the composite (Pipeline.ts:
//   the haze, the shafts and the rods' colour work in scene units, its output is scaled again),
//   FSR 3's reactive generator (Fsr3Upscaler.ts), the haze's light volume (VolumetricHaze.ts
//   `fogOver`), and the clouds' view, by the scale it was written at (Clouds.ts `sampleView`: the
//   clouds update before `beginFrame`, so a frame whose P changes draws them at the last frame's).
//   The metering histograms instead bin the values as stored, so their range follows P as the
//   stored values do, and the meter takes P back out of
//   their means in stops (`log2Scale`, Metering.ts): with a range fixed in scene units, a dark
//   twilight's values fell under its floor while the targets still held them at full precision. The
//   display chain's exposure is relative, 2^( EV - P ), so the bloom, the local tone and the grade
//   see the exposed image they saw before; FSR 3 takes 2^P as its pre-exposure (its history
//   follows a new P by the ratio).
// - The writers: the scene passes' MRTs (Motion.ts `sceneMRT`, `overlayMRT`), a fragmentNode's
//   `outputStruct` and a lit material's `meterOutput` (Motion.ts), the refraction pass's MRT, the
//   composite and what the post chain adds in hdr's units (the sun's flare, the rain veil:
//   Pipeline.ts), the rain overlay after the resolve (AfterResolve.ts `afterMRT`), the clouds'
//   view kernels (Clouds.ts: its traces, range boxes and history, whose history follows a new P by
//   the ratio), the haze's light kernel (VolumetricHaze.ts). Each also holds its value at or below
//   half float's largest (`preExposed`), so the sun's disc stays finite.

/** Stops below the exposure the scene is stored at (see the header). */
export const PRE_EXPOSURE_BIAS = 3;
/** The stops P may take, a guard against a broken reading. The darkest scene asks for 18: a clear
 * evening's island view with the sun near -7 degrees, before the night's layers come in. Clamped at
 * 16 there, its darkest values went subnormal and a few thousand pixels came out a level dark. */
export const PRE_EXPOSURE_RANGE = [ - 8, 22 ] as const;
/** Half float's largest value: what a pre-exposed writer stores at most. */
export const HALF_MAX = 65504;

/** What a writer at a power of two of its own stores at most (the sky's bakes and the clouds'
 * panorama, at `storeStops`): a stop under half float's largest, so a value a little over its
 * scale's range stays finite and keeps the stop below it. */
export const STORE_CEILING = 2 ** 15;

/** Where the sun's light on the clouds is stored (stops). */
const STORE_LEVEL = 10;
/** A store's power of two without a light on the clouds (after sunset, while only the afterglow
 * lights them, until the moon's light comes in), and the most it takes under a light (as the
 * moon's comes in, its light is still far under a night's). The afterglow keeps its brightest
 * under 2^13 at either: under the first as the sun's light leaves the clouds, under the second as
 * the moon's reaches them, the afterglow by then faded a few stops more. */
const STORE_STOPS_DARK = 18, STORE_STOPS_MAX = 20;

/** The power of two (stops) a store of the sky's bakes or the clouds' panorama holds its radiance
 * at (sky/SkyTrack.ts trackClouds), for `light`, the luminance of the light the clouds take (the sun's above
 * their tops while it lights them, and their moonlight with its gains), `moonShare`, the moon's
 * share of it, and `night`, where that store's range sits best under the moon (stops over the
 * light: look/SkyLook.ts `storeNight`, from the weather's clouds). By day the light itself is stored near
 * 2^STORE_LEVEL, which keeps the brightest a day holds (a cloudy noon's cumulus) under 2^13; under
 * the moon the store's range is centred in what a night must keep, by the moon's share between the
 * two. Over the kept weathers and every hour a store's brightest then stays under 2^13 and its
 * first percentile over 2^-8, and at night (the fronts between the kept skies' nights too),
 * pressed 12 stops darker, over 2^-13, a stop over half float's 2^-14. A function of the scene's state alone: never of the exposure, or of what a store
 * held before. */
export function storeStops( light: number, moonShare: number, night: number ): number {

	if ( ! ( light > 0 ) ) return STORE_STOPS_DARK;
	const level = STORE_LEVEL + ( night - STORE_LEVEL ) * moonShare;
	return Math.min( STORE_STOPS_MAX, Math.round( level - Math.log2( light ) ) );

}

/** A debug press of the stored skies (debug/Tune.ts `post.store.press`, in stops, 0 by default):
 * the sky's bakes, the clouds' panorama and view and the haze's light volume store their radiance
 * times `store` (2^-press) and their readers multiply by `read` (2^press), so the page draws the
 * image it drew while those buffers hold values as many stops darker: how a darker night would sit
 * in half float. Negative, they hold values that many stops brighter: a frame that comes out the
 * same at -2, 0 and +2 holds nothing at half float's edges. A load-time value (`?tune=`), so
 * every bake and history starts pressed. At 0 both are 1 and change nothing. */
export const storePress = { store: uniform( 1 ), read: uniform( 1 ) };
tuneApply( 'post.store.press', 0, { min: - 8, max: 16, load: true }, ( stops ) => {

	storePress.store.value = 2 ** - stops;
	storePress.read.value = 2 ** stops;

} );

/** P for an exposure of `ev` stops. */
export function preExposureStops( ev: number ): number {

	if ( ! Number.isFinite( ev ) ) return 0;
	return THREE.MathUtils.clamp( Math.round( ev ) - PRE_EXPOSURE_BIAS, PRE_EXPOSURE_RANGE[ 0 ], PRE_EXPOSURE_RANGE[ 1 ] );

}

interface PreExposureTicket { stops: number; generation: number; view: number; still: number }

/** What main.ts lists in the settled signal's `pending` while the pre-exposure waits (Settled.ts
 * leaves it out of the quiet a reading must be taken in). */
export const PRE_EXPOSURE_PENDING = 'pre-exposure';

class PreExposure {

	/** 2^P and 2^-P for the frame being rendered. */
	readonly scale = uniform( 1 );
	readonly inverse = uniform( 1 );
	/** P itself, log2 of `scale`, for the frame being rendered: for readers that work in stops. */
	readonly log2Scale = uniform( 0 );
	/** P of the frame being rendered. */
	stops = 0;
	/** The P the last reading asks for, applied at the next frame's start. */
	private next = 0;
	// readings taken before the last reset are dropped
	private generation = 0;
	// whether the last reading's frame used the P its own exposure asks for, and was a frame of the
	// settled view now drawn
	private fixed = false;
	// counts the changes of what a frame draws from and the frames that were not quiet (`viewFrame`),
	// the last frame's key, the quiet frames of the view so far and the frames that settle it
	private view = 0;
	private readonly viewKey: number[] = [];
	private still = 0;
	private settle = 8;
	/** Readings applied (diagnostics). */
	readings = 0;
	/** Changes of P (diagnostics). */
	changes = 0;
	/** The EV of the last reading applied (diagnostics). */
	ev = NaN;

	/** Back to the fresh load's P (a scene event that cuts: PipelineDisplay.ts listens). */
	reset(): void {

		this.next = 0;
		this.fixed = false;
		this.generation ++;

	}

	/** At the frame's start, before anything reads the rest's epoch: applies the P the last reading
	 * asked for. */
	beginFrame(): void {

		if ( this.next === this.stops ) return;
		this.stops = this.next;
		this.scale.value = 2 ** this.stops;
		this.inverse.value = 2 ** - this.stops;
		this.log2Scale.value = this.stops;
		this.changes ++;
		restEpoch.bump( 'pre-exposure', false );

	}

	/** Every rendered frame, before its ticket: what it draws from (TemporalRest.ts `restKey`, with
	 * the app's signature, its epoch without the pre-exposure's own bumps: `changes`), whether the
	 * page waits for nothing but the pre-exposure (`quiet`: false without the settled signal's
	 * inputs) and the frames a quiet view takes to settle (`settle`, the tier's `restSettle`). A new
	 * key, or a frame that is not quiet, makes the pre-exposure pending at once and leaves the
	 * readings of earlier frames unable to end it; so does a frame drawn before the view's settle
	 * frames. */
	viewFrame( key: readonly number[], quiet: boolean, settle: number ): void {

		this.settle = Math.max( 1, settle );
		let same = quiet && key.length === this.viewKey.length;
		for ( let i = 0; same && i < key.length; i ++ ) same = key[ i ] === this.viewKey[ i ];
		if ( same ) {

			this.still ++;
			return;

		}
		this.view ++;
		this.still = quiet ? 1 : 0;
		// (a confirmation was the earlier view's: the readings in flight may be of that view still)
		this.fixed = false;
		this.viewKey.length = 0;
		this.viewKey.push( ...key );

	}

	/** What a reading of this frame's exposure needs to be taken into account: its P, generation,
	 * view and the quiet frames of the view so far, this one included. */
	ticket(): PreExposureTicket {

		return { stops: this.stops, generation: this.generation, view: this.view, still: this.still };

	}

	/** The exposure EV a frame rendered at `ticket` ended with (read back). */
	observe( ticket: PreExposureTicket, ev: number ): void {

		if ( ticket.generation !== this.generation ) return;
		this.ev = ev;
		this.next = preExposureStops( ev );
		this.fixed = ticket.stops === this.next && ticket.view === this.view && ticket.still >= this.settle;
		this.readings ++;

	}

	/** Whether P may still change on a still scene (the settled signal waits). */
	get pending(): boolean {

		return ! this.fixed || this.next !== this.stops;

	}

	/** The state, live (debug builds: the diagnostics' `preExposure`). */
	diagnostics(): Record<string, number | boolean> {

		return { stops: this.stops, next: this.next, pending: this.pending, ev: this.ev, changes: this.changes, readings: this.readings, view: this.view, still: this.still, settle: this.settle };

	}

}

/** The app's pre-exposure (see the header). */
export const preExposure = new PreExposure();

/** A scene-unit colour as a pre-exposed writer stores it. */
export function preExposed( color: V3 ): V3 {

	return min( color.mul( preExposure.scale ), vec3( HALF_MAX ) );

}

/** A scene-unit RGBA (alpha untouched) as a pre-exposed writer stores it. */
export function preExposedRGBA( color: THREE.Node<'vec4'> ): THREE.Node<'vec4'> {

	return vec4( preExposed( color.rgb ), color.a );

}

/** A scene-unit scalar (the metering illuminance) as a pre-exposed writer stores it. */
export function preExposedScalar( value: F ): F {

	return min( value.mul( preExposure.scale ), HALF_MAX );

}

/** A pre-exposed colour read back in scene units. */
export function sceneUnits( color: V3 ): V3 {

	return color.mul( preExposure.inverse );

}
