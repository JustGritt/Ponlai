import { Vector2 } from 'three/webgpu';
import type { Node } from 'three/webgpu';
import { Fn, clamp as clampNode, dot, floor, fract, ivec3, mix, select, sin, uint, uniform, vec2, vec3, vec4 } from 'three/tsl';
import { debugFlags } from '../debug/DebugFlags.ts';
import { tune } from '../debug/Tune.ts';
import { PREVAILING_HEADING, type WeatherState } from '../sky/Weather.ts';
import type { WindLook } from '../look/Look.ts';

// The weather's one wind, integrated on the CPU, and the wind field all vegetation moves in.
//
// The weather's one wind, integrated on the CPU, what every part of the scene takes from it, and
// the one wind field all vegetation moves in.
//
// The weather's state gives three numbers (sky/Weather.ts, its wind a dimension of its own beside
// the sky: four levels and a heading that turns about the prevailing one with the weather):
// `windDirection`, a heading in the sea spectrum's convention (the wind comes from it and travels
// toward heading + 180 degrees), `windSpeed`, the mean wind 10 m up (U10), and `gustiness`, its
// gust factor (a 3 s gust over the mean). `Wind` is their one reader: it integrates them and
// publishes in `shared.wind` every quantity a part reads, so no part keeps a speed or direction of
// its own. U_ref is REFERENCE_SPEED, the gentle level's 3.8 m/s (fair's load), at which the sea was
// tuned:
// - vegetation: `direction`; the plants' speed U_p (`plantSpeed`, m/s), a piecewise-linear map of
//   U10 (the look's `plantWind`, `plantSpeedOf`), from which the wind's strength on the plants is taken
//   (`windStrength`: the grass's gusted wind, the limbs' `plantLoad`) and the grass's swing opens;
//   the gust field's along-wind travel (`gustScroll`, at GUST_TRAVEL U_p, shared with the sea) and
//   its own time (`gustTime`), which together sweep and reshape the gusts; the plants' phase clock
//   (`plantClock`, the plants' speed integrated: their own travel, m), on which the grass and the
//   plants sway, and `clock`, scene seconds, which does not follow the wind: the leaves' flutter
//   (by `breeze`, U / U_ref). The plants' main bend alone takes the drag of U10 itself (`breeze`:
//   `plantWind`). Everything else reads U_p, not U10: a meadow and a grove move as much under a
//   gentle 10 m wind as a lively one wants, which the true U10 near the ground would leave nearly
//   still, while the sea, the clouds, the rain and the haze keep U10, the sea's state the true one.
//   The grass and the plants sway faster as the wind rises, and a stronger wind bends them further:
//   how hard it blows reads at a glance. The gust factor enters nothing today: the gusts' share of
//   the strength is the look's `gustGain`.
// - the sea: the FFT's spectrum follows `speed` and `heading` (OceanFFT.setWind: tidewater's sea
//   state at that wind), so the sea is the same function of the weather however the app reached
//   it: the wind sea turns with the weather's wind, while the swell keeps the prevailing heading
//   (`spectrumWind`) and its own offset from it (OceanFFT `swellHeading`,
//   `swell.directionOffset`), and the shore waves follow the swell. `gale`, the wind beyond
//   1.5 U_ref, heaves the sea at the rock (Water.ts).
//   Around the island the sea's gusts (its cat's paws) are the meadow's: `gustAt`, the gust field
//   below, read by SeaDetail.ts, so a gust crosses from the water onto the meadow and off again
//   as one patch, at the meadow's pace; farther out tidewater's octaves drift with `displacement`.
// - the clouds: `displacement`, which Clouds.setDrift turns into each layer's drift aloft;
//   `direction` for the fronts, the cirrus's streaks and the tops' lean (sky/Sky.ts, Clouds.ts).
// - the air: `displacement` (the clouds' drift), `velocity` (rain, the storm's spray),
//   `direction` (the storm sea's foam).
// - the falling leaves (FallingLeaves.ts), which take several fixed steps between two of the
//   wind's updates: `windBefore`, the frame's wind moved back by a node's seconds, so the gusts'
//   scroll and time are still read here alone.
// - the fog's banks (FogNoise.ts): `displacement` for their drift and `evolutionTime`, the gusts'
//   time axis rescaled to the banks' own rate, for their change of shape.
// The wind probe's W5 (`__PONLAI__.windProbe`) reports where each of them travels.
//
// The gust field (`gustAt`): one octave of periodic 3D gradient noise, its two horizontal axes the
// world's xz in the wind's frame less the wind's integrated along-wind travel (`gustScroll`), over
// GUST_SCALE, its third an integrated time (`gustTime`, GUST_EVOLUTION cells a second). The travel
// sweeps the pattern downwind; the time axis reshapes it as it goes, so a gust breaks up and
// another rises elsewhere (travel alone would slide a fixed picture past, time alone would make the
// gusts blink in place). The pattern travels faster than the plants' wind, GUST_TRAVEL times their
// speed U_p (their travel integrated, so a change of wind sweeps on without a jump): a gust runs
// across the meadow and the grove as a wave, a patch crossing its own size in a second or two on a
// fair day, and a stronger wind sends them faster. Much faster and, seen from the island camera's
// height, the gusts would flicker rather than run. The gust q is the noise mapped to 0..1. The wind's strength on the plants
// is linear in their speed, w = U_p ( 1 + gustGain q ) (`windStrength`), so a light breeze shows its gusts and a
// gale does not press everything flat at once. (Gusts as a noise field scrolled with the wind: Eric
// Wohllaib, "Procedural Grass in 'Ghost of Tsushima'", GDC 2021, 2D Perlin noise scrolled in the
// wind's direction; Bill Rockenbeck, "Blowing from the West: Simulating Wind in Ghost of
// Tsushima", GDC 2021, the wind's strength varied by time-varying Perlin noise.)
// `windField` gives the gust and the limbs' load from it (`plantLoad`); still air is the rest pose.
// The meadow's blades read the gust at their root moved by a small offset of their own
// (`gustJitter`), so a gust's edge frays across neighbouring blades, and take the wind gusted there
// as the plants' main bend does (Grass.ts SWING).
// Plants answer it in Crysis's layers (Sousa, GPU Gems 3 ch. 16): the whole plant bends from its
// root, every vertex keeping its distance from it (vegetation/Deform.ts `mainBend`). The main bend
// follows the drag on the plant: the gusted wind 10 m up over fair's (`plantWind`), raised to its
// kind's drag exponent and scaled by the plant's own load in fair's mean wind, set by its height,
// its crown's area and its stiffness (Deform.ts, Bamboo.ts, Yunpeng.ts). A limb on a plant
// (Deform.ts `branchBend`, `sprayBend`) takes a load linear in the wind's strength at the root,
// u_p = sat( w / plantSpeed ) (`plantLoad`, `windField`'s z). A plant's swing about its bend
// runs on the plants' travel (`plantClock`) at the kind's rate in swings a metre, so its frequency
// is the rate times U_p and a stronger wind sways the plants faster; each plant swings a little
// quicker by a band of its own (up to `plantRateSpread` in PLANT_RATE_BANDS steps) and on a phase
// of its own (`plantWave`, from a hash of its root), so neighbours sway apart. Leaf edges, sprays
// and fronds flutter on their own phases at fixed rates (`clock`), a small motion on top.
//
// Everything is a function of the wind's integrated clocks and travel (`clock`, `plantClock`,
// `gustScroll`, `gustTime`, wrapped where every rate completes whole cycles or the noise repeats), never time * a changing speed,
// so a paused frame and a seek are reproducible; shadows and the rain's collision map keep the rest
// pose. For motion vectors (TAA) every displacement is a function of the wind's state alone, its
// vertex inputs bound inside: each vegetation material records it with `setWindDisplacement`,
// evaluated in the material's position stage (its rest inputs read `positionLocal` before three
// assigns the swayed position) with `shared.wind` for the frame drawn or `shared.wind.previous`
// (the last rendered frame's, equal to the current one while paused and after a seek, so a jump
// leaves no motion); the grass's widening keeps the current camera's.
//
// `previous` is kept for motion vectors: `update` copies the current state into it before it
// integrates, once per rendered frame (main.ts calls it once per tick, and every tick renders),
// `reset` (a seek) and `cut` (Motion.ts's `taaCut`, through `onTaaCut`) make it the current state.
// The wrapped clocks may differ by a wrap between the two: every shader reads them only through
// functions periodic in their wrap (`wave` at whole cycles per CLOCK_PERIOD, of `clock` and `plantClock` alike, `flutterWave` of `clock`, `plantWave`
// of `plantClock` at rates rounded to whole cycles per CLOCK_PERIOD, the gust noise's period over GUST_TILE and GUST_TIME_PERIOD), so the difference of the two poses is the frame's motion alone
// (tools/render/verify-wind.mjs checks the sources).
// The URL option `windView` shows this field and every plant's answer to it, and
// `__PONLAI__.windProbe` measures them (packages/render/debug/WindView.ts, WindProbe.ts; docs/reference/url-options.md).
// `debugFlags.windFreeze = true` holds the vegetation's wind (its clocks and gust scroll, previous
// equal to current) while the scene clock runs, for clips that want only the shadows or the jitter
// to move; the clouds and the sea keep their wind.

/** Fair's mean wind (m/s): the sea's calm design was tuned at it. */
export const REFERENCE_SPEED = 3.8;
/** Fair's gust factor. */
export const REFERENCE_GUSTINESS = 1.4;
const HEADING = PREVAILING_HEADING; // for a wind built without one
const clamp = ( value: number, low: number, high: number ) => Math.max( low, Math.min( high, value ) );
// The vegetation's clocks (seconds, or metres of the plants' travel) wrap here. Shaders take
// fract( clock * f ) only at rates f with at most three decimals (Hz, or swings a metre), so every
// one of them completes whole cycles in a period.
export const CLOCK_PERIOD = 1000;

// The gust field (see `gustAt`): patches about GUST_SCALE across (the noise's cell, metres), the
// same along and across the wind. The noise repeats every GUST_CELLS cells in both horizontal axes,
// GUST_TILE metres, where the travel wraps, and every GUST_TIME_PERIOD cells in time, which
// GUST_EVOLUTION cells a second crosses in CLOCK_PERIOD seconds, where `gustTime` wraps: neither
// wrap shows. A gust reshapes within about 1 / GUST_EVOLUTION seconds. GUST_SCALE sets the gusts'
// correlation length (the lag at which the field's autocorrelation falls under 1/e, the wind
// probe's `correlationLength`), 0.51 GUST_SCALE along and across the wind alike, as the length
// scale of the wind's turbulence near the ground. EN 1991-1-4 Annex B gives the along-wind
// integral length L( z ) = 300 m ( z / 200 m )^( 0.67 + 0.05 ln z0 ) (as given in C. Katz,
// "Computational Mechanics: Enhanced Civil Engineering Applications (Wind)", TU Munich lecture
// slides), 40 m for terrain category 0 (open sea and the coast exposed to it, z0 = 0.003 m) at its
// z_min of 1 m, about the meadow's height (Z. Markovic, J. Dobric and N. Gluhovic, "Comparison of
// wind actions according to Eurocode and previous regulations", Scientific Journal of Civil
// Engineering 9 (1), 2020, Table 1), the island's wind coming off the sea; the across-wind length
// of the along-wind gusts is about 0.3 of it (0.30 and 0.33 in the two forms Katz gives). One
// length for both axes is their geometric mean, 40 m x sqrt( 0.3 ) = 22 m, which GUST_SCALE 43
// gives (measured 22 m). Tunable at load (`wind.gust.scale`).
export const GUST_SCALE = tune( 'wind.gust.scale', 43, { min: 8, max: 80, scale: 'log', load: true } )();
const GUST_CELLS = 32;
export const GUST_TILE = GUST_SCALE * GUST_CELLS;
export const GUST_EVOLUTION = 0.1;
export const GUST_TIME_PERIOD = GUST_EVOLUTION * CLOCK_PERIOD;
// The gust pattern's along-wind speed over the plants' speed U_p (`gustScroll`): the large eddies
// over a plant canopy travel at about 1.8 times the mean wind at its top (Raupach, Finnigan and
// Brunet 1996, as given in Tse, Poindexter and Variano 2016; a forest gives 1.75, Christen et al.
// 2008), U_p taken as that wind.
export const GUST_TRAVEL = 1.8;
// The gusts' share of the wind's strength (`gustGain`, `windStrength`: w = S ( 1 + gustGain q )), a
// blade's largest offset for its gust sample (`gustJitter`), the limbs' reference speed
// (`plantSpeed`, `plantLoad`), the plants' speed by the wind's (`plantWind`, `plantSpeedOf`) and the
// spread of their sway rates (`plantRateSpread`, `plantWave`) are the look's (look/Look.ts
// `WindLook`). The wind takes them where it is created, not as this module loads: `windResponse`
// makes them from a look's table, `createWindUniforms` carries them in both of the vegetation's
// wind states (`response`), and every function here reads them from the state or the response it
// is given, so two winds made from two tables hold their own (tools/render/verify-look-params.mjs). The
// numbers reach the shaders as literals, as before.
// The gust q's mean (`gustAt`): the gust noise is symmetric about 0, so q about 1/2.
export const GUST_MEAN = 0.5;
// The air's density (kg/m^3), the one Koizumi et al.'s drag coefficients were computed with
// (Deform.ts CROWN_DRAG). The island's roughness length z0 (m): EN 1991-1-4's terrain category 0,
// open sea and the coast exposed to it (as given in Markovic, Dobric and Gluhovic, see GUST_SCALE).
// The mean wind at a height z (m) is U10 ln( z / z0 ) / ln( 10 / z0 ), the log law (`logLaw`).
export const AIR_DENSITY = 1.2;
export const ROUGHNESS = 0.003;
/** The mean wind at `z` metres over U10's, by the log law over ROUGHNESS (0 at and below z0). */
export const logLaw = ( z: number ): number => Math.max( 0, Math.log( z / ROUGHNESS ) / Math.log( 10 / ROUGHNESS ) );
// Each plant's sway rate is its kind's times 1 + up to the look's `plantRateSpread`, in PLANT_RATE_BANDS
// even steps from 0 (`plantWave`).
export const PLANT_RATE_BANDS = 4;
const TAU = Math.PI * 2;

/** The wind's look (look/Look.ts `WindLook`) as the wind reads it (`windResponse`): the table's
 * numbers, the limbs' reference speed `plantSpeed` tunable at load (`wind.plants.speed`). Plain
 * numbers, which the shaders take as literals. */
export type WindResponse = Readonly<WindLook>;

/** The wind's look made from a look's table: registers the limbs' reference speed as a tunable
 * (debug/Tune.ts) under the name it always had. The tune table is one per process: on a debug page
 * a second response reads the first one's `wind.plants.speed`. */
export function windResponse( look: WindLook ): WindResponse {

	return {
		gustGain: look.gustGain,
		gustJitter: look.gustJitter,
		plantWind: look.plantWind,
		plantSpeed: tune( 'wind.plants.speed', look.plantSpeed, { min: 10, max: 100, scale: 'log', load: true } )(),
		plantRateSpread: look.plantRateSpread,
	};

}

/** The plants' speed U_p (m/s) for a mean wind 10 m up `u10` (m/s): a line through the response's
 * `plantWind` points, and past the last one on along its last piece's slope. */
export function plantSpeedOf( response: Pick<WindResponse, 'plantWind'>, u10: number ): number {

	const keys = response.plantWind;
	let i = 1;
	while ( i < keys.length - 1 && u10 > keys[ i ][ 0 ] ) i ++;
	const [ x0, y0 ] = keys[ i - 1 ], [ x1, y1 ] = keys[ i ];
	return Math.max( 0, y0 + ( y1 - y0 ) * ( u10 - x0 ) / ( x1 - x0 ) );

}

// The wind's state as the vegetation's displacement reads it (see `windField` and
// `setWindDisplacement`); every value integrated or set from the weather, never time * a changing
// speed, so a paused frame and a seek reproduce the same pose. `response` is no uniform: the
// look's numbers the functions below read from the state, the same object in both states.
function vegetationWind( direction: Vector2, response: WindResponse ) {

	return {
		direction: uniform( direction.clone() ), // world-space travel direction; unit x/z
		breeze: uniform( 1 ), // speed / fair speed: the leaves' flutter and the plants' drag (`plantWind`)
		plantSpeed: uniform( plantSpeedOf( response, REFERENCE_SPEED ) ), // the plants' speed U_p, m/s (`plantWind`)
		clock: uniform( 0 ), // scene seconds modulo CLOCK_PERIOD: the leaves' flutter
		gustScroll: uniform( 0 ), // the gusts' along-wind travel (GUST_TRAVEL x the plants') modulo GUST_TILE, m: their sweep
		gustTime: uniform( 0 ), // the gust noise's time axis, GUST_EVOLUTION cells a second, modulo GUST_TIME_PERIOD
		plantClock: uniform( 0 ), // the plants' phase clock: their speed integrated modulo CLOCK_PERIOD, m: the grass's and the plants' sway
		response,
	};

}

/** The inputs of every vegetation wind displacement besides the vertex's own (its rest position,
 * root and amplitudes). `createWindUniforms` holds two: the frame being rendered (the uniforms
 * themselves) and the frame rendered before it (`previous`), for motion vectors. */
export type WindState = ReturnType<typeof vegetationWind>;

/** A wind state's values as two vec4s, ( direction, breeze, clock ) and ( gust scroll, gust time,
 * plant clock, plants' speed ), and the state read back from two such nodes with the wind's
 * `response`: a debug kernel's per-step wind (packages/island/debug/WindProbe.ts), so the scrolls
 * stay read here alone (tools/render/verify-wind.mjs). */
export const WIND_STATE_VEC4S = 2;
export function packWindState( state: WindState ): number[] {

	return [ state.direction.value.x, state.direction.value.y, state.breeze.value, state.clock.value,
		state.gustScroll.value, state.gustTime.value, state.plantClock.value, state.plantSpeed.value ];

}

export function windStateOf( a: Node<'vec4'>, b: Node<'vec4'>, response: WindResponse ): WindState {

	return { direction: a.xy, breeze: a.z, clock: a.w, gustScroll: b.x, gustTime: b.y, plantClock: b.z, plantSpeed: b.w, response } as unknown as WindState;

}

/** Add this object to shared before constructing materials, then give that same object to
 * Wind. Every consumer holds the same node identities throughout weather transitions. `response`:
 * the wind's look (`windResponse`), which both states carry. */
export function createWindUniforms( response: WindResponse ) {

	const angle = HEADING * Math.PI / 180;
	const direction = new Vector2( - Math.cos( angle ), - Math.sin( angle ) );
	return {
		...vegetationWind( direction, response ),
		velocity: uniform( direction.clone().multiplyScalar( REFERENCE_SPEED ) ), // air velocity, m/s
		displacement: uniform( new Vector2() ), // integrated air displacement, metres
		gale: uniform( 0 ), // the wind beyond 1.5 x the fair-weather wind, U / U_ref - 1.5 in [ 0, 1.6 ]
		// The vegetation's wind as it was for the previous rendered frame: equal to the current
		// one while paused and after a seek or reset (no motion across a jump).
		previous: vegetationWind( direction, response ),
	};

}

export type WindUniforms = ReturnType<typeof createWindUniforms>;

export interface WindSettings {

	speed?: number; // mean wind 10 m up, m/s
	heading?: number; // OceanSpectrum Fourier heading, degrees; IFFT travel is opposite
	gustiness?: number; // gust factor, a 3 s gust over the mean (1 for none)
	time?: number;

}

/** A weather state's wind as Wind's settings. */
export const weatherWind = ( state: Pick<WeatherState, 'windSpeed' | 'windDirection' | 'gustiness'> ): Required<Omit<WindSettings, 'time'>> =>
	( { speed: state.windSpeed, heading: state.windDirection, gustiness: state.gustiness } );

function validate( speed: number, heading: number, gustiness: number, time: number ): void {

	if ( ! Number.isFinite( speed ) || speed < 0 ) throw new RangeError( 'Wind speed must be finite and non-negative' );
	if ( ! Number.isFinite( heading ) ) throw new RangeError( 'Wind heading must be finite' );
	if ( ! Number.isFinite( gustiness ) || gustiness < 1 ) throw new RangeError( 'Wind gustiness must be finite and at least 1' );
	if ( ! Number.isFinite( time ) || time < 0 ) throw new RangeError( 'Wind time must be finite and non-negative' );

}

// PCG's output permutation on one word (Jarzynski and Olano, "Hash Functions for GPU Rendering",
// JCGT 2020), for the gust noise's lattice.
function pcg( n: Node<'uint'> ): Node<'uint'> {

	const state = n.mul( uint( 747796405 ) ).add( uint( 2891336453 ) );
	const word = state.shiftRight( state.shiftRight( uint( 28 ) ).add( uint( 4 ) ) ).bitXor( state ).mul( uint( 277803737 ) );
	return word.shiftRight( uint( 22 ) ).bitXor( word );

}

// Ken Perlin's improved noise (SIGGRAPH 2002) in 3D, its lattice wrapped every GUST_CELLS cells in
// x and y and GUST_TIME_PERIOD in z, about -1..1. Its gradients are Perlin's twelve edge vectors,
// picked by a PCG hash of the wrapped lattice point; fixed seed, so the field is the same on every
// load. A function of its argument alone (no uniform), so one WGSL function serves every material
// and kernel that calls it.
const gustNoise = Fn( ( [ p ]: [ Node<'vec3'> ] ) => {

	const cell = floor( p ).toVar();
	const f = p.sub( cell ).toVar();
	const u = f.mul( f ).mul( f ).mul( f.mul( f.mul( 6 ).sub( 15 ) ).add( 10 ) ).toVar();
	// the lattice point wrapped in integers, so a cell on the period's edge cannot land on the
	// period itself through a float division's rounding
	const period = ivec3( GUST_CELLS, GUST_CELLS, GUST_TIME_PERIOD );
	const lattice = ivec3( cell ).toVar();
	const corner = ( x: number, y: number, z: number ) => {

		const o = vec3( x, y, z );
		const w = lattice.add( ivec3( x, y, z ) ).mod( period ).add( period ).mod( period );
		const h = pcg( uint( w.x ).bitXor( pcg( uint( w.y ).bitXor( pcg( uint( w.z ).add( uint( 0x2545f491 ) ) ) ) ) ) ).bitAnd( uint( 15 ) );
		const d = f.sub( o );
		const a = select( h.lessThan( uint( 8 ) ), d.x, d.y );
		const b = select( h.lessThan( uint( 4 ) ), d.y, select( h.equal( uint( 12 ) ).or( h.equal( uint( 14 ) ) ), d.x, d.z ) );
		return select( h.bitAnd( uint( 1 ) ).equal( uint( 0 ) ), a, a.negate() ).add( select( h.bitAnd( uint( 2 ) ).equal( uint( 0 ) ), b, b.negate() ) );

	};
	const x00 = mix( corner( 0, 0, 0 ), corner( 1, 0, 0 ), u.x );
	const x10 = mix( corner( 0, 1, 0 ), corner( 1, 1, 0 ), u.x );
	const x01 = mix( corner( 0, 0, 1 ), corner( 1, 0, 1 ), u.x );
	const x11 = mix( corner( 0, 1, 1 ), corner( 1, 1, 1 ), u.x );
	return mix( mix( x00, x10, u.y ), mix( x01, x11, u.y ), u.z );

} ).setLayout( { name: 'windGustNoise', type: 'float', inputs: [ { name: 'p', type: 'vec3' } ] } );

/** The gust q at a point (world x, z), 0..1: the gust noise (see the note at the top) at the
 * point in the wind's frame less the wind's travel, over GUST_SCALE, and at `gustTime`, mapped from
 * -1..1 to 0..1. The one gust everything reads: the plants (`windField`, `windStrength`), the
 * meadow's blades and the sea's cat's paws (SeaDetail.ts). Callable in any stage, and in a vertex
 * stage whose result reaches a varying (the noise is a layout function of its argument alone). */
export function gustAt( wind: WindState, position: Node<'vec2'> ): Node<'float'> {

	const direction = wind.direction;
	const along = dot( position, direction ).sub( wind.gustScroll );
	const across = dot( position, vec2( direction.y.negate(), direction.x ) );
	const n = gustNoise( vec3( along.div( GUST_SCALE ), across.div( GUST_SCALE ), wind.gustTime ) );
	return clampNode( n.mul( 0.5 ).add( 0.5 ), 0, 1 );

}

/** The wind `back` seconds (a node, 0 or more) before the frame's, for a kernel that takes several
 * fixed steps between two of the wind's updates (FallingLeaves.ts): the frame's state with the
 * gusts' scroll moved back along the wind at the rate it travels (GUST_TRAVEL times the plants'
 * speed, `plantSpeed`), their time at GUST_EVOLUTION cells a second, and the air's displacement at
 * its velocity. The rates and the heading are the frame's: a weather change over the last `back`
 * seconds is not replayed. Read its gusts with `gustAt`: the scroll and time may fall below 0,
 * where the gust noise repeats over GUST_TILE and GUST_TIME_PERIOD as it does at their wraps. */
export function windBefore( wind: WindUniforms, back: Node<'float'> ): WindState & { displacement: Node<'vec2'> } {

	return {
		...wind,
		gustScroll: wind.gustScroll.sub( wind.plantSpeed.mul( GUST_TRAVEL ).mul( back ) ),
		gustTime: wind.gustTime.sub( back.mul( GUST_EVOLUTION ) ),
		displacement: wind.displacement.sub( wind.velocity.mul( back ) ),
	} as unknown as WindState & { displacement: Node<'vec2'> };

}

/** An evolution time of `rate` cells a second for a field of its own that the wind reshapes (the
 * fog's banks, FogNoise.ts): the gusts' time axis rescaled, so it is integrated with the wind's
 * clocks (a function of the scene clock, held with them) and wraps where they wrap, every
 * `rate` x CLOCK_PERIOD cells. The reader's field must repeat over that period, as the gust noise
 * repeats over GUST_TIME_PERIOD. */
export function evolutionTime( wind: WindState, rate: number ): Node<'float'> {

	return wind.gustTime.mul( rate / GUST_EVOLUTION );

}

/** The wind's strength on the plants (m/s) at the gust `q` (`gustAt`): linear in the plants' speed
 * S = U_p (`plantSpeed`), S ( 1 + gustGain q ). */
export function windStrength( wind: WindState, q: Node<'float'> ): Node<'float'> {

	return wind.plantSpeed.mul( q.mul( wind.response.gustGain ).add( 1 ) );

}

/** A blade's offset (m, x and z within +-the wind's `gustJitter`) for its gust sample, from a hash
 * of its root (world x, z): added to the root it reads `gustAt` at. */
export function gustJitter( wind: WindState, root: Node<'vec2'> ): Node<'vec2'> {

	const jitter = wind.response.gustJitter;
	const x = fract( sin( dot( root, vec2( 63.7264, 10.873 ) ) ).mul( 42319.7231 ) );
	return vec2( x, rootHash( root ) ).mul( 2 * jitter ).sub( jitter );

}

/** The wind at a point on the island (world x, z), for the vertex stage of vegetation: gusts are a
 * noise field swept with the wind's integrated travel and reshaped in time, so they visibly cross
 * the meadow. Returns
 * ( direction x, direction z, load, gust ): the wind's unit direction, the limbs' load there
 * (`plantLoad`) and the gust q (`gustAt`). One noise evaluation. */
export function windField( wind: WindState, position: Node<'vec2'> ): Node<'vec4'> {

	return Fn( () => windAt( wind, position ) )() as Node<'vec4'>;

}

/** `windField` as plain nodes, no function: callable in a vertex stage whose result reaches a
 * varying. */
export function windAt( wind: WindState, position: Node<'vec2'> ): Node<'vec4'> {

	const gust = gustAt( wind, position );
	return vec4( wind.direction, plantLoad( wind, gust ), gust );

}

/** The limbs' load at the gust `q` (`gustAt`): the wind's strength there over the limbs'
 * reference speed, u_p = sat( S ( 1 + gustGain q ) / plantSpeed ) (`windStrength`), 0..1. */
export function plantLoad( wind: WindState, q: Node<'float'> ): Node<'float'> {

	return clampNode( windStrength( wind, q ).div( wind.response.plantSpeed ), 0, 1 );

}

/** The plants' wind at the gust `q` (`gustAt`): the gusted wind 10 m up over fair's mean,
 * u = ( U10 / U_ref ) ( 1 + gustGain q ) / ( 1 + gustGain GUST_MEAN ) (`breeze`), its mean the
 * level's U10 over U_ref. The drag on a plant follows it (Deform.ts `mainBend`). */
export function plantWind( wind: WindState, q: Node<'float'> ): Node<'float'> {

	const gain = wind.response.gustGain;
	return wind.breeze.mul( q.mul( gain ).add( 1 ) ).div( 1 + gain * GUST_MEAN );

}

/** A plant's own sway rate (swings a metre of the plants' travel) for its kind's `rate` (whole
 * cycles per CLOCK_PERIOD) and its hash `xi` (0..1), under the wind's `response`:
 * rate ( 1 + plantRateSpread k / ( B - 1 ) ), k its band, floor( B fract( 16 xi ) ) of
 * B = PLANT_RATE_BANDS (the hash's lower digits, so the band does not follow the phase the plant
 * takes from `xi`), rounded to three decimals so it completes whole cycles per CLOCK_PERIOD. The
 * CPU twin of `plantWave`'s rate. */
export function plantRate( response: Pick<WindResponse, 'plantRateSpread'>, rate: number, xi: number ): number {

	const band = Math.floor( ( xi * 16 % 1 ) * PLANT_RATE_BANDS );
	return Math.floor( rate * CLOCK_PERIOD * ( 1 + response.plantRateSpread * band / ( PLANT_RATE_BANDS - 1 ) ) + 0.5 ) / CLOCK_PERIOD;

}

/** A plant's sway, sin( 2 pi f_p phi + phase ), phi the plants' clock (`plantClock`, metres of
 * their travel), at its own rate f_p in swings a metre (`plantRate`: its kind's `rate` and a band
 * of its own from `xi`, its hash, 0..1), so it swings f_p U_p times a second and faster as the wind
 * rises. The rate is rounded to whole cycles per CLOCK_PERIOD, so the sway stays seamless where the
 * clock wraps; `rate` itself must be one (it refuses any other when the shader is built). */
export function plantWave( wind: WindState, rate: number, xi: Node<'float'>, phase: Node<'float'> ): Node<'float'> {

	if ( Math.abs( rate * CLOCK_PERIOD - Math.round( rate * CLOCK_PERIOD ) ) > 1e-6 ) throw new RangeError( `plantWave: ${ rate } a metre is not whole cycles per ${ CLOCK_PERIOD } m` );
	const band = floor( fract( xi.mul( 16 ) ).mul( PLANT_RATE_BANDS ) );
	const cycles = floor( band.mul( rate * CLOCK_PERIOD * wind.response.plantRateSpread / ( PLANT_RATE_BANDS - 1 ) ).add( rate * CLOCK_PERIOD + 0.5 ) );
	return sin( fract( wind.plantClock.mul( cycles.div( CLOCK_PERIOD ) ) ).mul( TAU ).add( phase ) );

}

/** A leaf's flutter beat, -1..1: the smooth triangle wave of the scene clock at `rate` Hz (whole
 * cycles per CLOCK_PERIOD, refused otherwise when the shader is built) plus `phase` (cycles): the
 * triangle wave t = | fract( x + 0.5 ) 2 - 1 | eased as t^2 ( 3 - 2 t ) (Sousa's
 * SmoothTriangleWave, GPU Gems 3 ch. 16), taken about its middle. Its rounded crests and steep
 * crossings read as a leaf's irregular flick rather than a sine's even swing. */
export function flutterWave( wind: WindState, rate: number, phase: Node<'float'> ): Node<'float'> {

	if ( Math.abs( rate * CLOCK_PERIOD - Math.round( rate * CLOCK_PERIOD ) ) > 1e-6 ) throw new RangeError( `flutterWave: ${ rate } Hz is not whole cycles per ${ CLOCK_PERIOD } s` );
	const t = fract( fract( wind.clock.mul( rate ) ).add( phase ).add( 0.5 ) ).mul( 2 ).sub( 1 ).abs();
	return t.mul( t ).mul( t.mul( - 2 ).add( 3 ) ).mul( 2 ).sub( 1 );

}

/** sin( 2 pi rate time + phase ) for a vegetation clock that wraps at CLOCK_PERIOD: `clock` (scene
 * seconds, `rate` in Hz) or `plantClock` (metres of the plants' travel, `rate` in swings a metre,
 * so the swing's frequency is rate x U_p). `rate` with at most three decimals, so it stays seamless where the clock wraps; it refuses any other
 * rate (whole cycles per CLOCK_PERIOD) when the shader is built. */
export function wave( time: Node<'float'>, rate: number, phase: Node<'float'> | number = 0 ): Node<'float'> {

	if ( Math.abs( rate * CLOCK_PERIOD - Math.round( rate * CLOCK_PERIOD ) ) > 1e-6 ) throw new RangeError( `wave: ${ rate } Hz is not whole cycles per ${ CLOCK_PERIOD } s` );
	return sin( fract( time.mul( rate ) ).mul( Math.PI * 2 ).add( phase ) );

}

/** A vegetation material's wind: its swayed vertex position (in the space of its
 * `positionNode`, world space for every vegetation mesh here) as a function of the wind's state
 * alone, the vertex's own inputs (rest position, root, amplitudes) bound inside. The material is
 * drawn with `( shared.wind )`; evaluated with `( shared.wind.previous )` in the same vertex stage
 * it gives where the vertex was in the previous frame, for motion vectors. */
export type WindDisplacement = ( wind: WindState ) => Node<'vec3'>;

/** Record `displacement` on a material (and return its pose for the current frame). */
export function setWindDisplacement( material: { userData: Record<string, unknown> }, displacement: WindDisplacement, wind: WindState ): Node<'vec3'> {

	material.userData.windDisplacement = displacement;
	return displacement( wind );

}

/** The wind displacement a vegetation material was built with, if any. */
export function windDisplacement( material: { userData: Record<string, unknown> } ): WindDisplacement | undefined {

	return material.userData.windDisplacement as WindDisplacement | undefined;

}

/** A hash of a plant's root (world x, z), 0..1. */
export function rootHash( root: Node<'vec2'> ): Node<'float'> {

	return fract( sin( dot( root, vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) );

}

/** A hash of a plant's root (world x, z) to a phase, 0..2 pi. */
export function rootPhase( root: Node<'vec2'> ): Node<'float'> {

	return rootHash( root ).mul( Math.PI * 2 );

}

/** CPU integration of weather wind, independent of accelerated celestial time. Weather
 * already eases its speed/heading; this class integrates that changing velocity rather than
 * multiplying it by the elapsed scene time, which would teleport clouds and plant phases. */
export class Wind {

	readonly uniforms: WindUniforms;
	private currentSpeed = REFERENCE_SPEED;
	private currentHeading = HEADING;
	private currentGustiness = REFERENCE_GUSTINESS;
	private elapsed = 0;
	private plantTravel = 0; // the plants' speed integrated, m (float64 here; wrapped for the shaders)
	private gustAge = 0; // the gust noise's time axis, cells (float64 here; wrapped for the shaders)
	private readonly previousVelocity = new Vector2();

	constructor( uniforms: WindUniforms, settings: WindSettings = {} ) {

		this.uniforms = uniforms;
		this.reset( settings );

	}

	get speed(): number { return this.currentSpeed; }
	get heading(): number { return this.currentHeading; }
	get gustiness(): number { return this.currentGustiness; }
	get time(): number { return this.elapsed; }

	/** The wind the sea's spectrum starts from (OceanFFT's `windSpeed`, `windDirection` and
	 * `swellHeading`): REFERENCE_SPEED and the prevailing heading, which the weather's own speed and
	 * heading replace (OceanFFT.setWind), so a page that changed its weather draws the sea a page
	 * loaded in it draws; the swell keeps the prevailing heading. */
	get spectrumWind(): { windSpeed: number; windDirection: number; swellHeading: number } {

		return { windSpeed: REFERENCE_SPEED, windDirection: PREVAILING_HEADING, swellHeading: PREVAILING_HEADING };

	}

	// Explicit seek/startup only. Reconstructing a fixed preset at time t is deterministic;
	// a live weather change must call update so its existing travel history is preserved.
	reset( { speed = REFERENCE_SPEED, heading = HEADING, gustiness = REFERENCE_GUSTINESS, time = 0 }: WindSettings = {} ): void {

		validate( speed, heading, gustiness, time );
		this.publish( speed, heading, gustiness );
		this.elapsed = time;
		this.uniforms.displacement.value.copy( this.uniforms.velocity.value ).multiplyScalar( time );
		this.plantTravel = time * plantSpeedOf( this.uniforms.response, speed );
		this.gustAge = time * GUST_EVOLUTION;
		this.publishVegetation();
		this.keepPrevious();

	}

	update( dt: number, speed = this.currentSpeed, heading = this.currentHeading, gustiness = this.currentGustiness ): void {

		validate( speed, heading, gustiness, dt );
		this.keepPrevious();
		this.integrate( dt, speed, heading, gustiness );
		// (debug) the vegetation's wind held: its clocks back to the last frame's, no motion
		if ( debugFlags.windFreeze === true ) this.hold();

	}

	/** A cut of the temporal modes' history (Motion.ts `taaCut`): the previous frame's wind becomes
	 * the current one, so nothing sways across the cut. */
	cut(): void {

		this.keepPrevious();

	}

	private integrate( dt: number, speed: number, heading: number, gustiness: number ): void {

		const previousSpeed = this.currentSpeed;
		this.previousVelocity.copy( this.uniforms.velocity.value );
		this.publish( speed, heading, gustiness );
		if ( dt === 0 ) return;
		this.elapsed += dt;
		// Trapezoidal integration follows the smoothly transitioning CPU weather state and
		// does not change when a rendering-quality setting changes its update cadence.
		this.uniforms.displacement.value.addScaledVector( this.previousVelocity.add( this.uniforms.velocity.value ), dt * 0.5 );
		const response = this.uniforms.response;
		this.plantTravel += dt * ( plantSpeedOf( response, previousSpeed ) + plantSpeedOf( response, speed ) ) * 0.5;
		this.gustAge += dt * GUST_EVOLUTION;
		this.publishVegetation();

	}

	// windFreeze: the integrated clocks keep the values `previous` holds, and `previous` takes the
	// rest of the current state (the weather's direction and breeze), so both stay equal.
	private hold(): void {

		const u = this.uniforms, p = u.previous;
		u.clock.value = p.clock.value;
		u.gustScroll.value = p.gustScroll.value;
		u.gustTime.value = p.gustTime.value;
		u.plantClock.value = p.plantClock.value;
		this.keepPrevious();

	}

	// Called once per rendered frame (update) before the new state: `previous` then holds what the
	// last frame was drawn with.
	private keepPrevious(): void {

		const u = this.uniforms, p = u.previous;
		p.direction.value.copy( u.direction.value );
		p.breeze.value = u.breeze.value;
		p.plantSpeed.value = u.plantSpeed.value;
		p.clock.value = u.clock.value;
		p.gustScroll.value = u.gustScroll.value;
		p.gustTime.value = u.gustTime.value;
		p.plantClock.value = u.plantClock.value;

	}

	private publishVegetation(): void {

		const u = this.uniforms;
		u.clock.value = this.elapsed % CLOCK_PERIOD;
		u.gustScroll.value = GUST_TRAVEL * this.plantTravel % GUST_TILE;
		u.gustTime.value = this.gustAge % GUST_TIME_PERIOD;
		u.plantClock.value = this.plantTravel % CLOCK_PERIOD;

	}

	private publish( speed: number, heading: number, gustiness: number ): void {

		const angle = ( ( heading % 360 ) + 360 ) % 360 * Math.PI / 180;
		this.currentSpeed = speed;
		this.currentHeading = heading;
		this.currentGustiness = gustiness;
		// Match ShoreWaves.setDirection: the positive inverse FFT propagates opposite k.
		this.uniforms.direction.value.set( - Math.cos( angle ), - Math.sin( angle ) );
		this.uniforms.velocity.value.copy( this.uniforms.direction.value ).multiplyScalar( speed );
		const ratio = speed / REFERENCE_SPEED;
		this.uniforms.breeze.value = ratio;
		this.uniforms.plantSpeed.value = plantSpeedOf( this.uniforms.response, speed );
		this.uniforms.gale.value = clamp( ratio - 1.5, 0, 1.6 );

	}

}
