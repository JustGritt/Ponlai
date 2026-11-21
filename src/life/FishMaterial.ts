/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater), src/world/fish/FishMaterial.js.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	Fn, attribute, float, vec2, vec3, color, sin, cos, mix, smoothstep, floor, fract, abs, sign, select, length,
	normalize, cross, positionGeometry, normalGeometry, normalLocal, positionPrevious,
} from 'three/tsl';
import { type Shared, type WaveMaps, underwaterLight, UnderwaterLightingModel, TEMPORAL, writeMeter } from '../render/index.ts';
import { PART, type Anatomy } from './FishGeometry.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// The material of a fish drawn from instances (Wenyao.ts), its look by its skin table.
//
// Copied from Ponlai's island (packages/island/life/FishMaterial.ts at d5ad84b6) for the waterfall's
// salmon (src/falls/salmon.ts); two changes: a skin can take its head's colour over the back alone
// (`Skin.headOver`), and can show less of the flank's silver (`Skin.silver`), as a spawning
// salmon's red flank shows little.
//
// The vertex stage poses the model (tidewater's swimVertex: a travelling wave down the body that
// grows toward the tail, and the turning bend) and folds its wings: each ray of a paired fin turns
// about the fin's root, back toward the tail, to `FOLD` of its spread angle, so a swimming fish
// carries its wings closed along its flank and a gliding one spreads them. Then it scales, turns
// and places the fish by its instance (no instance matrix: the pose is local), and adds the sway it
// is given (where the eye sees it through the waves). Under the temporal modes the same pose at the
// instance's previous position and phase, with the sway as it is now, is the previous position.
//
// The fragment stage paints the skin from the vertex's place on the fish (aData, FishGeometry.ts):
// the back's colour with its markings over the silvery flank and the pale belly, the head, the
// snout, a painted eye, and the fins darker toward their edge with their rays. The silvery flank
// is a metal-like specular layer (guanine). It is lit as everything under the water is
// (Underwater.ts: the key down the water column, the ambient dimmer
// with depth); above the water, as it flies, as in the air. Its ambient diffuse is the ambient
// volume's (lighting/AmbientSample.ts) through the model's default, degree 1 weighted by depth: in
// flight the volume's light where the fish is, under the water the environment's carried down the
// column.

/** The colours of a species' skin (sRGB hex). */
export interface Skin {
	back: number; marks: number; flank: number; belly: number; head: number; snout: number;
	fin: number; finEdge: number; iris: number;
	// where the head ends and the snout's colour stops (fractions of the total length)
	headEnd: number; snoutEnd: number;
	// where the head's colour lies: over all of the head (the default), or over its back alone
	headOver?: 'all' | 'back';
	// how much of the flank's guanine silver it shows (1 the default; a spawning salmon's red flank
	// little)
	silver?: number;
	// the eye: its place along the fish and height (fractions of the total length) and radius
	eye: { u: number; y: number; r: number };
}

/** A wing's rays turn back to this share of their spread angle when folded. */
const FOLD = 0.1;

// rotates v by the unit quaternion q
const rotateQ = ( q: V4, v: V3 ): V3 => v.add( cross( q.xyz, cross( q.xyz, v ).add( v.mul( q.w ) ) ).mul( 2 ) );

// turns v about the vertical by angle a
const turnY = ( v: V3, a: F ) => vec3( v.x.mul( cos( a ) ).add( v.z.mul( sin( a ) ) ), v.y, v.z.mul( cos( a ) ).sub( v.x.mul( sin( a ) ) ) );

/** The fish material for a mesh whose instances carry iPose ( x, y, z, length ), iTurn (quaternion),
 * iMotion ( phase, amplitude, bend, fold ) and iBefore ( the previous x, y, z, phase ). `sway`: an
 * offset added to each placed vertex (the water's swimmerSway: where the eye sees it through the
 * waves). */
export function makeFishMaterial( S: Anatomy, skin: Skin, shared: Shared, waveMaps: WaveMaps, roots: { pectoral: number; pelvic: number }, sway?: ( at: V3 ) => V3 ): THREE.MeshStandardNodeMaterial {

	const mat = new THREE.MeshStandardNodeMaterial();
	mat.name = 'fish';
	const aData = attribute( 'aData', 'vec4' );
	const pose = attribute( 'iPose', 'vec4' ), turn = attribute( 'iTurn', 'vec4' );
	const motion = attribute( 'iMotion', 'vec4' ), before = attribute( 'iBefore', 'vec4' );
	const part = floor( aData.y.add( 0.01 ) );

	// the model at a phase of its swimming wave: the wings folded, the body bent
	const posed = ( phase: F ) => {

		const p = positionGeometry.toVar(), n = normalGeometry.toVar();
		const fold = motion.w;
		for ( const [ id, F, rootX ] of [ [ PART.PECTORAL, S.pectoral, roots.pectoral ], [ PART.PELVIC, S.pelvic, roots.pelvic ] ] as const ) {

			const t = aData.w.div( Math.max( F.rays - 1, 1 ) );
			const spread = mix( float( F.lead ), float( F.last ), t );
			const side = sign( p.x );
			const angle = spread.mul( 1 - FOLD ).mul( fold ).mul( side ).mul( select( part.equal( id ), 1, 0 ) );
			const pivot = vec3( side.mul( rootX ), F.y, 0.5 - F.u * S.body );
			p.assign( turnY( p.sub( pivot ), angle ).add( pivot ) );
			n.assign( turnY( n, angle ) );

		}
		// the travelling wave, larger toward the tail, and the turn's bend
		const u = aData.x;
		const env = u.mul( u ).mul( 0.85 ).add( 0.08 ).mul( smoothstep( 0, 0.25, u ).mul( 0.7 ).add( 0.3 ) );
		const wave = phase.sub( u.mul( 5.6 ) );
		const lateral = sin( wave ).mul( env ).mul( motion.y ).add( motion.z.mul( u.mul( u ) ) );
		// the wave's slope along the body turns the normal about the vertical
		const slope = cos( wave ).mul( - 5.6 ).mul( env ).mul( motion.y ).add( motion.z.mul( u ).mul( 2 ) );
		return { p: p.add( vec3( lateral, 0, 0 ) ), n: normalize( turnY( n, slope.negate().atan() ) ) };

	};
	const place = ( p: V3, at: V3 ) => {

		const placed = rotateQ( turn, p.mul( pose.w ) ).add( at ).toVar();
		return sway ? placed.add( sway( placed ) ) : placed;

	};

	mat.positionNode = Fn( () => {

		const now = posed( motion.x );
		normalLocal.assign( rotateQ( turn, now.n ) );
		if ( TEMPORAL ) positionPrevious.assign( place( posed( before.w ).p, before.xyz ) );
		return place( now.p, pose.xyz );

	} )();
	if ( TEMPORAL ) mat.userData.previousPosition = true;

	// ---- the skin
	const c = ( hex: number ) => color( hex );
	const u = aData.x, h = aData.w;
	const body = part.equal( PART.BODY );
	// counter-shading: the pale belly, the silvery flank, the dark back
	const flank = mix( c( skin.belly ), c( skin.flank ), smoothstep( - 0.7, - 0.05, h ) );
	// the back's colour over the upper half, as a fish seen from above is dark
	const backShare = smoothstep( - 0.1, 0.3, h );
	// the back's markings: broken dark bars across it, behind the head
	const bars = sin( u.mul( 46 ).add( sin( aData.z.mul( 31 ).add( u.mul( 9 ) ) ).mul( 1.4 ) ) ).mul( 0.5 ).add( 0.5 );
	const marks = smoothstep( 0.62, 0.8, bars ).mul( smoothstep( 0.1, 0.4, h ) ).mul( smoothstep( skin.headEnd, skin.headEnd + 0.04, u ) );
	const back = mix( c( skin.back ), c( skin.marks ), marks );
	let skinColor: V3 = mix( flank, back, backShare );
	// the head (all of it, or its back), and the snout's colour at its tip
	const head = float( 1 ).sub( smoothstep( skin.headEnd - 0.025, skin.headEnd + 0.01, u ) );
	skinColor = mix( skinColor, c( skin.head ), skin.headOver === 'back' ? head.mul( backShare ) : head );
	skinColor = mix( skinColor, c( skin.snout ), float( 1 ).sub( smoothstep( skin.snoutEnd * 0.6, skin.snoutEnd, u ) ) );
	// the painted eye on each side of the head
	const eye = length( positionGeometry.zy.sub( vec2( 0.5 - skin.eye.u, skin.eye.y ) ) ).div( skin.eye.r );
	skinColor = mix( skinColor, c( skin.iris ), float( 1 ).sub( smoothstep( 0.85, 1, eye ) ) );
	skinColor = mix( skinColor, c( 0x07090b ), float( 1 ).sub( smoothstep( 0.42, 0.55, eye ) ) );
	// the fins: darker toward the edge, the rays a shade darker than the membrane between them
	const ray = abs( fract( aData.w.add( 0.5 ) ).sub( 0.5 ) );
	const finColor = mix( c( skin.fin ), c( skin.finEdge ), smoothstep( 0.45, 1, aData.z ) ).mul( mix( 0.78, 1, smoothstep( 0.05, 0.22, ray ) ) );
	mat.colorNode = select( body, skinColor, finColor );
	// the guanine's silver on the flank, little on the back and none on the fins
	const silver = smoothstep( - 0.8, - 0.1, h ).mul( float( 1 ).sub( backShare.mul( 0.75 ) ) ).mul( float( 1 ).sub( head.mul( 0.6 ) ) ).mul( skin.silver ?? 1 );
	mat.metalnessNode = select( body, silver.mul( 0.55 ), float( 0 ) );
	mat.roughnessNode = select( body, mix( float( 0.42 ), float( 0.26 ), silver ), float( 0.5 ) );

	const light = underwaterLight( shared, waveMaps );
	( mat as unknown as { setupLightingModel: () => THREE.LightingModel } ).setupLightingModel = () => new UnderwaterLightingModel( light, { terminator: false } );
	writeMeter( mat );
	return mat;

}
