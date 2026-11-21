import type * as THREE from 'three/webgpu';
import { vec3, max, abs, dot, cos, sin, normalize, length, pow, sqrt, tan } from 'three/tsl';
import { flutterWave, plantWave, plantWind, rootHash, windField, type WindState } from '../weather/Wind.ts';

// How the vegetation moves: the main bend, branch and spray bends, and the leaves' flutter.
//
// How the vegetation moves: the plants' main bend (`mainBend`: the yunpeng's stalks, the bamboo's
// culms and leaves), a branch's own bend after it (`branchBend`: the bamboo's branches and the
// cards on them), a sub-branch's after that (`sprayBend`: the bamboo's sprays) and their leaves'
// flutter (`leafFlutter`), in layers: a main bend of the whole plant with detail bends added on
// top of it (the layering Sousa describes in "Vegetation Procedural Animation and Shading in
// Crysis", GPU Gems 3, ch. 16). Each kind's wind numbers are constants beside its material; the
// bends' and the flutter's shared ones are here (PLANT_BEND, BRANCH_BEND, SPRAY_BEND,
// LEAF_FLUTTER, CROWN_DRAG). A plant's main bend follows the drag of the wind on it, as a
// cantilever's under its crown's load (PLANT_BEND), saturating as the crown tilts out of the wind,
// and swings at its kind's rate on the plants' travel, each plant on a phase and a band of the rate
// of its own (Wind.ts `plantWave`): a stronger wind bends a plant further, widens its swing and
// quickens it, as it quickens the grass's. A limb feels the part of the wind square to it, on the
// wind's strength at the root (Wind.ts `plantLoad`), and its bend saturates with that load. The
// meadow's blades move in Grass.ts (`bladeVertex`), on the same wind field and clock. Each function
// returns nodes.

const TAU = Math.PI * 2;

type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type F = THREE.Node<'float'>;
type V4 = THREE.Node<'vec4'>;

/** A leaved crown's drag coefficient, C_D = `c` U^`v` (U the wind at the crown, m/s): the mean of
 * the power laws fitted to three leaved broadleaf (poplar) crowns measured in the field,
 * 1.77 U^-0.911, 1.14 U^-0.824 and 1.79 U^-0.714, on the crown taken as an ellipse loaded at its
 * middle (A. Koizumi, J. Motoyama, K. Sawata, Y. Sasaki and T. Hirai, "Evaluation of drag
 * coefficients of poplar-tree crowns by a field test method", Journal of Wood Science 56 (3),
 * 2010). Their coefficients' and exponents' means stay within 3 % of the three laws' mean from 1 to
 * 12 m/s; none of the three is better founded than the others. The drag 0.5 rho C_D A U^2 then
 * grows as U^( 2 + v ), U^1.18: the crown streamlines as the wind rises (its Vogel exponent v). */
export const CROWN_DRAG = { c: 1.57, v: - 0.816 } as const;

/** The plants' main bend (`mainBend`): the top's lean theta( lambda ) = `most` lambda / ( `half` +
 * lambda ) under the load lambda, the drag on the crown over the culm's stiffness, lambda =
 * F0 H^2 / EI0 (F0 the drag before the crown tilts, H the height, EI0 the flexural rigidity at the
 * foot). The bamboo's culm bent as an elastica (a hollow tube tapering as its mesh does,
 * Bamboo.ts `radiusAt`, loaded at 0.775 of its height by a crown whose area falls as the cosine of
 * its tilt there) leans its tip, seen from its foot, by a function of lambda alone; `most` and
 * `half` are its least-squares fit for lambda up to 5 (a lean of 38 degrees), within 0.45 degrees
 * of it there and 3 degrees at lambda 50 (the fitted `most`, 60 degrees, is the curve's, not a
 * lean the culm reaches: the elastica creeps on toward 90 degrees). The crown's tilt saturates the
 * lean, so the load needs no clamp: the grown culm (Bamboo.ts `culmLoad`) leans 2.7, 7.4, 13.4 and
 * 19.8 degrees in the four levels' mean winds, the elastica 2.4, 6.9, 13.3 and 20.0. `swing`, the
 * swing's amplitude in the wind a plant feels, A sqrt( U ) with U the level's mean wind (both over
 * fair's mean wind: Wind.ts `plantWind`); `lag` (f_h), how far the swing's phase turns from the root
 * to the top (rad over the height share). The swing and the gusts together move the wind a plant
 * feels about its mean as a turbulent wind does: its spread over its mean is the wind's turbulence
 * intensity I. I comes from each wind level's gust factor G (a 3 s gust over the mean, sky/Weather.ts
 * WIND_LEVELS)
 * by EN 1991-1-4's peak pressure ( 1 + 7 I ) times the mean's, G^2 = 1 + 7 I (as given in
 * Z. Markovic, J. Dobric and N. Gluhovic, "Comparison of wind actions according to Eurocode and
 * previous regulations", Scientific Journal of Civil Engineering 9 (1), 2020): 0.18, 0.14, 0.12 and
 * 0.10 from the light level to the strong. The gusts alone move the wind by about 0.08 of its mean
 * (the wind probe with no swing: the grown culm's top by 0.066, 0.084, 0.075 and 0.076 of its mean
 * lean over 40 s, the wind's spread times the drag's exponent and the bend's slope). So the swing's
 * amplitude is sqrt( 2 ( I^2 - 0.08^2 ) ) of the mean wind: 0.09, 0.16, 0.23 and 0.25 of fair's, a
 * turbulence that grows slower than the wind, and `swing` is the least-squares fit of A sqrt( U )
 * to it over the four levels (the exponent 1/2 the best of 0.3 to 0.7), which brings the wind's
 * spread to 0.19, 0.14, 0.11 and 0.10. The probe then measures the grown culm's top moving by
 * 0.21, 0.14, 0.11 and 0.10 of its mean lean: the swing widens with the wind in metres while the
 * bend's saturation narrows it as a share. */
export const PLANT_BEND = { most: 1.043, half: 2.84, swing: 0.155, lag: 1 } as const;

/** A plant kind's wind in its main bend (`mainBend`): its sway `rate` (swings a metre of the
 * plants' travel) and `drag`, the exponent of the wind in the drag on it (2 + CROWN_DRAG.v for a
 * leaved crown, nearer 2 for a bare stalk). */
export interface PlantWind { rate: number; drag: number }

/** A plant's top's lean (rad), theta( lambda ) (PLANT_BEND), in the wind `u` over fair's for its
 * `load` lambda_0 and `kind`'s drag exponent (`mainBend`). */
export function plantLean( u: F, load: F, kind: PlantWind ): F {

	const lambda = load.mul( pow( max( u, 1e-4 ), kind.drag ) ).toVar();
	return lambda.mul( PLANT_BEND.most ).div( lambda.add( PLANT_BEND.half ) );

}

/** A plant's main bend: the whole plant turns from its root about the level axis square to the
 * wind, by alpha = give theta( lambda ), theta( lambda ) = theta_max lambda / ( lambda + L_h )
 * (PLANT_BEND `most`, `half`), the load lambda = lambda_0 u^n: lambda_0 the plant's load in fair's
 * mean wind (`load`: its height, crown and stiffness, Bamboo.ts, Yunpeng.ts), n its kind's `drag`,
 * u = u_w + A sqrt( U ) s the wind it feels over fair's: u_w the gusted wind at its root (Wind.ts
 * `plantWind`), U the mean's (`breeze`), A `swing`. The swing rides on the wind, so it narrows as
 * the bend saturates, as a stiffening stalk's does. `give` is the vertex's ( 1 - stiffness ) eta^2,
 * eta its share of the plant's height ( 0 at the root, 1 at the top), so the plant arcs over from
 * the root and its top leans by theta. s = sin( 2 pi f_p t + 2 pi xi - f_h eta ) is the plant's
 * swing, t the plants' clock (Wind.ts `plantWave`: its kind's `rate`, swings a metre of the plants'
 * travel, a band of its own and its phase xi from a hash of its root; f_h PLANT_BEND.lag, so the
 * top lags the foot a little). Each
 * vertex keeps its distance from the root (the turn's horizontal offset, eta's height times
 * tan alpha, renormalised to the vertex's distance). `wind` is the frame's wind state
 * (`shared.wind`, or `.previous`). Returns the bent position, the wind field ( direction x,
 * direction z, the limbs' load, gust ) at the root and the plant's sway: its top's horizontal
 * offset over its height (world x, z; for the wind view and probe, packages/render/debug/WindView.ts; a shader
 * that does not read it compiles as before). */
export function mainBend( wind: WindState, position: V3, root: V3, give: F, eta: F, load: F, kind: PlantWind ): { position: V3; field: THREE.Node<'vec4'>; sway: V2 } {

	const field = windField( wind, root.xz ).toVar();
	const xi = rootHash( root.xz );
	const swing = plantWave( wind, kind.rate, xi, xi.mul( TAU ).sub( eta.mul( PLANT_BEND.lag ) ) );
	const u = plantWind( wind, field.w ).add( swing.mul( sqrt( wind.breeze ) ).mul( PLANT_BEND.swing ) );
	const top = plantLean( u, load, kind ).toVar();
	const along = vec3( field.x, 0, field.y );
	const rel = position.sub( root ).toVar();
	const offset = along.mul( tan( top.mul( give ) ).mul( max( rel.y, 0 ) ) );
	return { position: normalize( rel.add( offset ).add( vec3( 0, 1e-5, 0 ) ) ).mul( length( rel ) ).add( root ), field, sway: along.xz.mul( tan( top ) ) };

}

/** A branch's wind (`branchBend`): `most`, the angle (rad) its bend tends to at a yield of 1;
 * `swing`, its swing's amplitude A sqrt( y ) (rad), y the limb's load's yield (`limbYield`), which
 * grows slower than the bend as the plants' swing does (PLANT_BEND); `along` (f_s), how far the
 * swing's phase turns along it (rad a metre); its rate (swings a metre of the plants' travel, each
 * branch a band of its own quicker as a plant is: Wind.ts `plantWave`). Kept at the lean and swing
 * the branches had before the wind's reference speed moved: the bamboo's branches, which rise a
 * little from level, feel 0.7 of the wind on average over a turn of it, so at a fair day's mean
 * gust they lean 10 degrees and swing by 2.7, before their own stiffness. */
export const BRANCH_BEND = { most: 0.64, swing: 0.09, along: 1, rate: 0.16 } as const;

/** A sub-branch's wind (`sprayBend`: the bamboo's leaf sprays on their branch), as BRANCH_BEND's:
 * pressed half as far, quicker than the branch, swinging as much, so a spray tosses on the branch
 * that carries it. */
export const SPRAY_BEND = { most: 0.32, swing: 0.09, along: 1, rate: 0.21 } as const;

type LimbWind = { most: number; swing: number; along: number; rate: number };

// How a limb's bend stiffens: its yield is half its load at this load, past the 1 a limb can take
// (Wind.ts `plantLoad`), so the yield of a load of 1 is 0.75.
const LIMB_HALF = 3;

/** A limb's load's yield: u_p / ( 1 + |u_p| / LIMB_HALF ), linear for a light load and
 * saturating. */
const limbYield = ( load: F ): F => load.div( abs( load ).div( LIMB_HALF ).add( 1 ) );

// A limb's bend (`branchBend`, `sprayBend`): the offset of `position` as the limb from `from` to
// `to` leans over from `from`, on its hash `xi`, its yield `give` and the numbers `P`.
function limbBend( wind: WindState, position: V3, xi: F, from: V3, to: V3, give: F, field: V4, P: LimbWind ): V3 {

	const span = to.sub( from ).toVar();
	const b = span.div( max( length( span ), 1e-4 ) ).toVar();
	const w = vec3( field.x, 0, field.y );
	const across = w.sub( b.mul( dot( b, w ) ) ).toVar();
	const felt = max( length( across ), 1e-4 ).toVar();
	const along = max( dot( position.sub( from ), b ), 0 ).toVar();
	const y = limbYield( field.z.mul( felt ) ).toVar();
	const swing = plantWave( wind, P.rate, xi, xi.mul( TAU ).sub( along.mul( P.along ) ) );
	const lean = y.mul( P.most ).add( sqrt( y ).mul( swing ).mul( P.swing ) ).mul( give ).toVar();
	return across.div( felt ).mul( sin( lean ) ).add( b.mul( cos( lean ).sub( 1 ) ) ).mul( along );

}

/** A limb's hash, from where its end stands (`to`) moved by its plant's root (`root`), so each
 * limb of each plant swings on its own. */
const limbHash = ( root: V3, to: V4 ): F => rootHash( root.xz.add( to.xz ) );

/** A branch's own bend, an offset (m) added after the plant's main bend (`mainBend`, whose `field`
 * it takes: the wind's direction w and the limbs' load u_p at the root). A slender limb feels the
 * part of the wind square to it (the cross-flow principle), n = w - ( w . b ) b, b its direction;
 * a limb's load is linear in the wind (Wind.ts `plantLoad`), so the branch from B0 to B1 (`from`,
 * `to`: TreeMesh.ts BRANCH_ATTRIBUTES, its yield 1 - k_b in `to.w`) takes the load u_p |n| and leans
 * in the plane of b and n by
 * phi = ( 1 - k_b ) ( y theta_b + A sqrt( y ) sin( 2 pi f_b t + 2 pi xi - f_s l ) ),
 * y = u_p |n| / ( 1 + u_p |n| / L_h ) the load's yield (`limbYield`, L_h LIMB_HALF), theta_b and A
 * BRANCH_BEND's `most` and `swing`, l the vertex's distance along it from B0, t the plants' clock,
 * xi its hash (`limbHash`), f_b BRANCH_BEND.rate with a band of the branch's own. Each vertex moves
 * as the point of the limb's line at its l does, l ( n / |n| sin( phi ) + b ( cos( phi ) - 1 ) ):
 * a branch across the wind is pressed downwind, one along it hardly moves, and each swings on its
 * own. A vertex on no branch has l 0 and stays. The bend is taken in the rest pose and added to the
 * bent position: the main bend's few degrees do not turn the branch's line. */
export function branchBend( wind: WindState, position: V3, root: V3, from: V4, to: V4, field: V4 ): V3 {

	return limbBend( wind, position, limbHash( root, to ), from.xyz, to.xyz, to.w, field, BRANCH_BEND );

}

/** A sub-branch's own bend (a leaf spray on its branch), added after the branch's: the same form
 * as `branchBend` from C0 to C1 (`sprayFrom`, `sprayTo`: TreeMesh.ts BRANCH_ATTRIBUTES, its yield
 * in `sprayTo.w`) by SPRAY_BEND. */
export function sprayBend( wind: WindState, position: V3, root: V3, sprayFrom: V4, sprayTo: V4, field: V4 ): V3 {

	return limbBend( wind, position, limbHash( root, sprayTo ), sprayFrom.xyz, sprayTo.xyz, sprayTo.w, field, SPRAY_BEND );

}

/** A leaf's flutter (a plant's leaves, cards or ribbons, about its main bend): `calm` of it between
 * the gusts, all of it in a gust's core; `twist` of it across the wind (a leaf turning over in the
 * gusts), the rest along its normal; `spread`, the phase it takes per metre of its rest position
 * along each axis (cycles; a few coprime decimals, so neighbouring leaves of one card or ribbon
 * flick apart). */
export const LEAF_FLUTTER = { calm: 0.25, twist: 0.65, spread: [ 0.37, 0.53, 0.71 ] } as const;

/** A leaf's flutter (an offset, m) by `amount` (A_l w^2, m at a fair day's gust core, w the
 * vertex's weight: 0 where the leaf holds to its stalk, 1 at its tip, so a leaf swings about its
 * stalk rather than moving whole), on its own `phase` (0..1, cycles) and its rest `position`'s
 * (LEAF_FLUTTER.spread) at `rate` Hz, as the smooth triangle wave (Wind.ts `flutterWave`), under
 * the wind field `field` at the plant's root (`mainBend`'s): it flutters as the gusts pass and
 * nearly rests between them, across the wind more than along its `normal`. `wind` is the frame's
 * wind state (`shared.wind`, or `.previous`). */
export function leafFlutter( wind: WindState, field: V4, normal: V3, amount: F, phase: F, rate: number, position: V3 ): V3 {

	const P = LEAF_FLUTTER;
	const strength = wind.breeze.mul( field.w.mul( 1 - P.calm ).add( P.calm ) );
	const beat = flutterWave( wind, rate, phase.add( dot( position, vec3( ...P.spread ) ) ) );
	const across = vec3( field.y.negate(), 0, field.x );
	return across.mul( P.twist ).add( normal.mul( 1 - P.twist ) ).mul( beat.mul( amount ).mul( strength ) );

}
