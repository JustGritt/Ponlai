// The Purkinje shift: the colour the rods lend low light, turning dark colours blue.
//
// The Purkinje shift: the colour the rods lend low light. As the light falls the rods take over
// from the cones and, sharing their pathways, add to the cones' signals, so a moonlit scene is not a
// darker day: its dark colours come up and turn blue. Patry presented the model and its shader in
// "Real-Time Samurai Cinema: Lighting, Atmosphere, and Tonemapping in Ghost of Tsushima" (SIGGRAPH
// 2021 Advances in Real-Time Rendering, p.160-168), after Cao, Pokorny, Smith and Zele 2008 ("Rod
// contributions to color perception: linear with rod contrast") and Kirk and O'Brien 2011
// ("Perceptually based tone mapping for low-light conditions"); the RGB colours' spectra are Smits
// 2000's method ("An RGB to spectrum conversion for reflectances").
// - A scene colour c (linear Rec.709) gives the cones' and the rods' responses q = M c, M_ij the
//   integral of receptor i's response curve, the D65 illuminant and primary j's reflectance
//   spectrum over 380 to 720 nm. The curves: Stockman and Sharpe's 2 degree cone fundamentals and
//   the CIE 1951 scotopic V'(lambda), each normalised to a peak of 1 (Kirk and O'Brien's form);
//   D65 normalised so that white has a luminance of 1. The spectra: the smoothest three, each in
//   [ 0, 1 ] and the three summing to 1, that give the three primaries under D65 (5 nm steps).
// - The cones' gains g_i = ( 1 + 0.33 ( q_i + k_i q_R ) / m_i )^-1/2 (m the cones' maximal
//   sensitivities 0.63721, 0.39242, 1.6064; the rods' input k 0.2, 0.2, 0.29), and the rods' change
//   in the opponent space (red-green, blue-yellow, luminance) is
//   K / S [ -( k3 + rw ), 1 + k3 rw, 0; p k3, ( 1 - p ) k3, 1; p S, ( 1 - p ) S, 0 ] diag( k ) diag( m )^-1 g q_R
//   (K 45, S 10, k3 0.6, rw 0.139, p 0.6189), taken back to RGB through the opponent matrix's and
//   M's cone rows' inverses. The constants are the ones Patry gives. Folded, the shader is the
//   talk's: lmsr = LMSR_FROM_RGB c, c + RGB_FROM_GAIN rsqrt( 1 + lmsr.xyz ) lmsr.w.
// - The model reads the light at the eye, and our night is lit far above a real one (a full moon
//   gives the island about a sunset's light). So the colour is taken to a retinal illuminance
//   first, and the change scaled back: white on open level ground under the night's light is taken
//   to `retina` trolands. That value sets where the lift starts: the cones' gains fall as the light
//   at the eye rises, so lit white keeps little of the rods' change, and the darker a colour is
//   against it the more the rods lift it and the bluer it turns. A look sets it (its display
//   table's `purkinje`). Ours, 250, is far above a real moonlit night's few trolands, tuned to the
//   shape of the dark tones on screen: before the tone curve the lit meadow gains about a fifth, a
//   colour four stops below it about 1.35 times as much.
//   The light on that ground (`purkinjeScale`'s input, Pipeline.ts) is a defined reference, the
//   luminance of a white Lambertian card in the open, not what the meter reads
//   (lighting/MeterLight.ts also takes the shadows, the sky's occlusion and the ground's bounce):
//   the key's luminance at the lifted cosine level ground takes (lighting/KeyLift.ts
//   `liftedKeyHeight`; the sun's at dusk, the moon's at night, under the weather's dimming but not
//   the shadow of the cloud cover) plus the sky's irradiance lent to the land (Sky.ts
//   `landIrradianceNode`) at its fill. Each pixel still turns as blue as its own light is dark
//   against that card. Under a cloud cover the card stays brighter than the ground, so a cloudy or
//   an overcast night turns bluer than a clear one at the same strength, as a darker night should.
//   The post constants' texel holds the scale (Pipeline.ts), once a frame. Only the darkest tones'
//   gains approach 1, where the rods' change is several times the colour itself, a strong blue;
//   `strength` takes it back to the look, and the night's weight fades it in (`night`,
//   Pipeline's). The rods' light brightens the frame too.
// - `strength` (`post.purkinje.strength@night`, 0 to 0.4) and `retina`
//   (`post.purkinje.retina@night`, 1 to 2000) are the look's (look/DisplayLook.ts `purkinje`), read
//   every frame; at strength 0 the shift adds nothing and the night's colour is the grade's and the
//   white balance's.
// - The test mode: nothing the shift reads comes from the meter or from the frame's exposure (the
//   key and the fill are functions of the clock, the weather and the tuning; the sky's irradiance
//   is the environment's, which on a running page cross-fades between bakes as it does for every
//   material it lights), so the shift is a function of the frame's light and colour. The meter
//   reads the shifted frame and settles on it as on any other, a paused frame is bit-identical
//   at any strength and a scene jump draws a fresh load's; on a running page the shift follows the
//   light as smoothly as the light changes, never the exposure's adaptation or the pre-exposure's
//   whole stops. The metered exposure is not the scale: the shift moves the meter's
//   target, so a scale read from the previous frame's exposure closes a loop through the meter, and
//   a paused page with the shift on was seen never to reach an EV that repeated.
// - The composite takes it (Pipeline.ts), so everything after reads the shifted light: the bloom
//   (a moonlit glint glows in the night's colour), the local tone's grid (its base and the pixel
//   agree) and the grade. Under fsr3 the upscaler resolves the shifted image; the rain drawn after
//   the resolve (taa, taau) leaves it out.
import { Fn, If, dot, inverseSqrt, max, vec3 } from 'three/tsl';
import type * as THREE from 'three/webgpu';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

/** The colour's scale to `retina`'s retinal illuminance (`purkinjeShift`'s `scale`), from
 * `retina`, the retinal illuminance of white on open level ground under the night's light (the
 * display's tables, look/DisplayLook.ts `purkinje`), and `light`, the luminance of that white in
 * scene units (see the header). */
export const purkinjeScale = ( retina: F, light: F ) => retina.div( max( light, 1e-12 ) );

// the receptors' responses from linear Rec.709: the cones' gain arguments 0.33 ( q_i + k_i q_R ) / m_i, then q_R
// (computed offline from the published tables named above)
const GAIN_L = vec3( 0.152563, 0.423367, 0.0806355 );
const GAIN_M = vec3( 0.0798717, 0.696033, 0.161123 );
const GAIN_S = vec3( 0.0031211, 0.0459169, 0.130669 );
const ROD = vec3( 0.0150385, 0.589431, 0.38577 );
// the rods' change in RGB from the cones' gains (per unit q_R): rows R, G, B
const FROM_GAIN_R = vec3( 8.42712, - 9.02804, 0.268173 );
const FROM_GAIN_G = vec3( 1.53333, 6.29916, - 0.288388 );
const FROM_GAIN_B = vec3( 17.3479, 17.2285, 1.5731 );

/** The colour `c` (linear scene units; `scale` takes it to the retinal illuminance `retina` sets:
 * `purkinjeScale`) with the rods' change added, times `amount` (`c` itself while `amount` is 0,
 * by day). */
export const purkinjeShift = Fn( ( [ c, scale, amount ]: [ V3, F, F ] ) => {

	const out = c.toVar();
	If( amount.greaterThan( 0 ), () => {

		const x = c.mul( scale ).toVar();
		const gain = inverseSqrt( vec3( dot( GAIN_L, x ), dot( GAIN_M, x ), dot( GAIN_S, x ) ).add( 1 ) ).toVar();
		const rods = dot( ROD, x ).div( scale );
		out.addAssign( vec3( dot( FROM_GAIN_R, gain ), dot( FROM_GAIN_G, gain ), dot( FROM_GAIN_B, gain ) ).mul( rods.mul( amount ) ) );

	} );
	return out;

} ).setLayout( { name: 'purkinjeShift', type: 'vec3', inputs: [ { name: 'c', type: 'vec3' }, { name: 'scale', type: 'float' }, { name: 'amount', type: 'float' } ] } );
