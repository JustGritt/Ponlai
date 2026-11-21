import { tune } from '../debug/Tune.ts';
import { SUN_DISPLAY_RADIUS, SUN_DISPLAY_SOLID_ANGLE, SUN_GLINT_RADIUS, projectedSolidAngle } from './Radiometry.ts';

// The sun's disk as the app draws it, the sky's disk and the glints' disk sized together.
//
// The sun's disk as the app draws it, sized by one tunable (debug/Tune.ts
// `lighting.sun.discScale`, read as the app loads): the sky's disk (Radiometry.ts
// SUN_DISPLAY_RADIUS, sky/Sky.ts, post/LightShafts.ts) and the disk the glints take their
// representative point on (SUN_GLINT_RADIUS, lighting/SurfaceBRDF.ts `keyGlint`) scale together, so
// the highlight on wet stone and the disk in the sky stay one size relative to each other. The
// displayed disk's radiance is a share of the radiance that would give it the key's irradiance
// (`lighting.sun.discRadiance`, 0.0837, its solid angle worked out again for its size): the drawn disk, and
// so what the bloom draws from it, take that share; the key light, the glints and the glare do not,
// and the environment's bakes leave the disk out (sky/Sky.ts). The glints' disk is tunable at load
// on its own (`lighting.sun.glintRadius`).
// Kept out of Radiometry.ts, which node tools import without the app's modules.

const scale = tune( 'lighting.sun.discScale', 1, { min: 0.5, max: 2, scale: 'log', load: true } )();
/** The sky's disk's angular radius (radians). */
export const SUN_DISC_RADIUS = SUN_DISPLAY_RADIUS * scale;
/** Its projected solid angle, the limb-darkened profile's (Radiometry.ts). */
export const SUN_DISC_SOLID_ANGLE = scale === 1 ? SUN_DISPLAY_SOLID_ANGLE : projectedSolidAngle( SUN_DISC_RADIUS );
/** The displayed disk's radiance as a share of the one whose integral is the key's irradiance. */
export const SUN_DISC_RADIANCE = tune( 'lighting.sun.discRadiance', 0.0837, { min: 0.02, max: 1, scale: 'log', load: true } )();
/** The glints' disk's angular radius (radians): Radiometry.ts SUN_GLINT_RADIUS, tunable at load
 * (`lighting.sun.glintRadius`), times the disk's scale. */
export const SUN_GLINT_DISC = tune( 'lighting.sun.glintRadius', SUN_GLINT_RADIUS, { min: 0.005, max: 0.08, scale: 'log', load: true } )() * scale;
