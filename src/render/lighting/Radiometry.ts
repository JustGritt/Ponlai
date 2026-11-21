// The scene's radiometric units and the sun's sizes: the radiance scale and the disk's radii.

// All scene-linear radiance and irradiance use the same unit scale. Keeping sunlight below
// fp16's finite range must not change its brightness relative to the sky or its reflections.
// Exposure and display-space bloom thresholds undo this scale in Pipeline.
export const SCENE_RADIANCE_SCALE = 1 / 16;
// The physical sun (0.27 degrees), for its solid angle. The sky draws its disk at
// SUN_DISPLAY_RADIUS, 1.1 degrees (about 4 times the true one), as games commonly draw it larger:
// a true-size disk reads as a point at the island camera's field of view. Its edge is cut where
// the view's cosine to the sun crosses the radius's, over one pixel's change of that cosine, so it
// stays clean at any size; inside, it darkens toward the limb by the quadratic law
// 1 - u1 (1 - m) - u2 (1 - m)^2, m = sqrt( 1 - (r / R)^2 ), with visible-light coefficients
// u1 = 0.47, u2 = 0.23. The glow round it is the sun's flare, which goes through the bloom
// (post/LightShafts.ts `flare`, post/PhysicalBloom.ts). The setting sun's light fades as a
// disk of SUN_SETTING_RADIUS (twice the true one) sinks into the sea, which the key's handoff to
// the moon is timed to (DayCycle.ts), and the moon is drawn at that size.
export const SUN_ANGULAR_RADIUS = 0.004675;
export const SUN_DISPLAY_RADIUS = 1.1 * Math.PI / 180;
export const SUN_LIMB_U1 = 0.47, SUN_LIMB_U2 = 0.23;
export const SUN_SETTING_RADIUS = 2 * SUN_ANGULAR_RADIUS;
// The disk the glints take their representative point on (lighting/SurfaceBRDF.ts `keyGlint`): the
// sun as the sky draws it, so a smooth surface mirrors the disk the sky shows, and a highlight on
// wet stone or a culm's node reads as a small full disk rather than a pin; tunable at load in
// SunDisc.ts (`lighting.sun.glintRadius`). The moon's is its physical angular radius.
export const SUN_GLINT_RADIUS = SUN_DISPLAY_RADIUS;
export const MOON_GLINT_RADIUS = 0.0045;

function solarDiskProfile( radius: number ): number {

	if ( radius >= 1 ) return 0;
	const m = 1 - Math.sqrt( 1 - radius * radius );
	return 1 - SUN_LIMB_U1 * m - SUN_LIMB_U2 * m * m;

}

// E = integral( L * cos(theta) dOmega ). The same radial limb profile is used in Sky (its edge
// there is a pixel wide, which the integral leaves out).
// Midpoint quadrature normalises the profile rather than treating a softened disk as uniform.
// Radiometric integral: https://pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Working_with_Radiometric_Integrals
const SAMPLES = 4096;
export function projectedSolidAngle( radius: number ): number {

	let sum = 0;
	for ( let i = 0; i < SAMPLES; i ++ ) {

		const theta = ( i + 0.5 ) * radius / SAMPLES;
		sum += solarDiskProfile( theta / radius ) * Math.sin( theta ) * Math.cos( theta ) * 2 * Math.PI * radius / SAMPLES;

	}
	return sum;

}
export const SUN_PROJECTED_SOLID_ANGLE = projectedSolidAngle( SUN_ANGULAR_RADIUS );
// The displayed disk spreads the same irradiance over its larger area.
export const SUN_DISPLAY_SOLID_ANGLE = projectedSolidAngle( SUN_DISPLAY_RADIUS );
