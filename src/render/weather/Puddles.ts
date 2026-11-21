import type * as THREE from 'three/webgpu';
import { Fn, If, Loop, PI, float, vec2, vec3, floor, fract, sin, dot, length, max, clamp, mix, smoothstep, mx_noise_float } from 'three/tsl';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;

// Puddles: standing water that fills level hollows as the ground soaks and shrinks as it dries.
//
// Standing water fills fixed hollows on level ground as the ground soaks, drowning the meadow
// blades but for a few tips, and shrinks inside its own damp stain as it dries. Puddles mirror the
// sky through an art-directed reflectance (stronger than water's 2 %, still following Fresnel), so
// they read as water from above.

/** Standing water on level ground: x = cover (0 to 1), y = depth (0 at the waterline, 1 where the
 * hollow is deep enough to hide the ground's relief), z = the soaked ground around each puddle.
 * Puddles fill the hollows of a fixed field as the ground soaks, grow while rain keeps it wet and
 * shrink as it dries, so they always reappear in the same places. The soaked margin is ground the
 * water has only just left: it stays dark a little longer than the puddle itself, so a drying
 * puddle leaves a dark rim and then a dark stain. `wetness` is the weather's persistent surface moisture;
 * `relief` (0 to 1, 0.5 neutral) lets the waterline creep into the ground's cracks first. */
export const puddleField = Fn( ( [ xz, wetness, relief ]: [ V2, F, F ] ) => {

	const hollows = mx_noise_float( vec3( xz.div( 3.2 ), 0.37 ) ).mul( 0.7 )
		.add( mx_noise_float( vec3( xz.div( 1.1 ), 4.1 ) ).mul( 0.3 ) ).mul( 0.5 ).add( 0.5 )
		.add( relief.sub( 0.5 ).mul( 0.05 ) ).toVar();
	// No standing water until the ground is soaked; then the waterline rises through the field.
	const level = mix( float( 0.8 ), float( 0.63 ), smoothstep( 0.3, 0.85, wetness ) ).toVar();
	// A crisp waterline: a soft one reads as a stain on the ground rather than as water.
	const cover = smoothstep( level, level.add( 0.025 ), hollows ).mul( smoothstep( PUDDLE_WET, 0.45, wetness ) );
	const depth = smoothstep( level.add( 0.01 ), level.add( 0.12 ), hollows );
	// The damp footprint follows the waterline of a wetter ground, so a drying puddle shrinks inside
	// its own dark stain, which then fades as the soil dries.
	const dampLevel = mix( float( 0.8 ), float( 0.63 ), smoothstep( 0.12, 0.6, wetness ) );
	const margin = smoothstep( dampLevel, dampLevel.add( 0.03 ), hollows ).mul( smoothstep( 0.05, 0.2, wetness ) );
	return vec3( cover, depth, margin );

} ).setLayout( { name: 'ponlaiPuddleField', type: 'vec3', inputs: [ { name: 'xz', type: 'vec2' }, { name: 'wetness', type: 'float' }, { name: 'relief', type: 'float' } ] } );

/** The wetness up to which no water stands (`puddleField`'s cover is exactly 0 there). */
export const PUDDLE_WET = 0.3;

/** Standing water, 0 to 1 (the cover of `puddleField` on neutral relief). */
export const puddleCover = ( xz: V2, wetness: F ) => puddleField( xz, wetness, float( 0.5 ) ).x;

const rippleHash = Fn( ( [ p ]: [ V2 ] ) => {

	return fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ).mul( 43758.5453 ) );

} ).setLayout( { name: 'ponlaiPuddleRippleHash', type: 'float', inputs: [ { name: 'p', type: 'vec2' } ] } );

// One drop per cell and cycle; four staggered layers.
const RIPPLE_CELL = 0.55; // metres
const RIPPLE_RADIUS = 0.32; // of a cell, so a ring never leaves its cell
const RIPPLE_RATE = 0.9; // drops per cell and second, per layer

/** Rain rings on standing water, after Sébastien Lagarde, "Water drop 2b - Dynamic rain and its
 * effects" (2013). Lagarde reads a ripple texture (radial falloff, ring direction, time offset);
 * here each cell computes the same terms analytically: a drop at a random point, its rings
 * travelling outward as it ages (sin( clamp( ( age - 1 + falloff ) * 9, 0, 3 ) * pi ), three
 * half-waves behind the front), heavier rain enabling more of the four time-offset layers.
 * Returns the water's slope in xy; rings finer than a pixel fade out, and z is the slope lost to
 * that filter, which the caller can treat as roughness. */
export const puddleRipples = Fn( ( [ xz, time, rain, footprint ]: [ V2, F, F, F ] ) => {

	const result = vec3( 0 ).toVar();
	// A ring is 1.5 waves in a third of its radius: about 4 cm per wave, lost at two pixels a wave.
	const resolved = float( 1 ).sub( smoothstep( 0.01, 0.025, footprint ) ).toVar();
	If( rain.greaterThan( 0.0001 ), () => {

		const gradient = vec2( 0 ).toVar();
		Loop( 4, ( { i } ) => {

			const layer = float( i ).toVar();
			// Lagarde: the layers switch on one by one as the rain grows heavier.
			const weight = clamp( rain.mul( 4 ).sub( layer ), 0, 1 ).toVar();
			const grid = xz.div( RIPPLE_CELL ).add( vec2( 0.37, 0.71 ).mul( layer ) ).toVar();
			const cell = floor( grid ).toVar();
			const key = cell.add( vec2( 17.3, 9.1 ).mul( layer ) ).toVar();
			const phase = time.mul( RIPPLE_RATE ).add( rippleHash( key ) ).add( layer.mul( 0.25 ) ).toVar();
			const age = fract( phase ).toVar();
			const event = floor( phase ).toVar();
			const centre = vec2( rippleHash( key.add( event.mul( 1.37 ) ) ), rippleHash( key.add( vec2( 5.3, 2.9 ) ).add( event.mul( 2.11 ) ) ) )
				.mul( 1 - RIPPLE_RADIUS * 2 ).add( RIPPLE_RADIUS ).toVar();
			const offset = fract( grid ).sub( centre ).toVar();
			const distance = max( length( offset ), 1e-4 ).toVar();
			const falloff = clamp( float( 1 ).sub( distance.div( RIPPLE_RADIUS ) ), 0, 1 );
			const front = age.sub( 1 ).add( falloff );
			const fade = clamp( weight.mul( 0.8 ).add( 0.2 ).sub( age ), 0, 1 );
			const ring = sin( clamp( front.mul( 9 ), 0, 3 ).mul( PI ) ).mul( falloff ).mul( fade );
			gradient.addAssign( offset.div( distance ).mul( ring ) );

		} );
		// Lagarde scales the slope by 0.35; steeper rings tilt the reflection further across the
		// sky's gradient and the Fresnel curve, so they still read under a flat overcast and over
		// the puddles' art-directed reflectance, which flattens Fresnel's contrast between a
		// ring's crest and trough.
		const strength = clamp( rain.mul( 1.5 ), 0, 1 ).mul( 0.9 );
		result.assign( vec3( gradient.mul( strength ).mul( resolved ), float( 1 ).sub( resolved ).mul( strength ) ) );

	} );
	return result;

} ).setLayout( { name: 'ponlaiPuddleRipples', type: 'vec3', inputs: [
	{ name: 'xz', type: 'vec2' }, { name: 'time', type: 'float' },
	{ name: 'rain', type: 'float' }, { name: 'footprint', type: 'float' },
] } );
