// The sea's inherent optical properties per channel, as plain numbers shared by all readers.
//
// The sea's inherent optical properties, per channel (1 / m), as plain numbers: shared.ts makes the
// page's uniforms of them, and the ambient probes (world/AmbientProbes.ts), baked in the chunk
// workers and in node, read them here, so both take the water from one place.
//
// Open Case 1 water with 0.5 mg m^-3 of chlorophyll: its absorption ABSORPTION and its
// backscattering coefficient BACKSCATTER. The water is our choice for the temperate open sea round a
// small island; calling it Jerlov's type II is our own reading of that concentration. Worked from
// spectra into linear sRGB by a fit of white light's transmittance through the water (the colour
// matching functions in Wyman, Sloan & Shirley's 2013 analytic form, JCGT 2(2)): pure water's
// absorption from Pope & Fry 1997 (Appl. Opt. 36:8710) and Smith & Baker 1981 (Appl. Opt. 20:177),
// the particles' by chlorophyll from Bricaud et al. 1998 (JGR 103:31033); water's scattering after
// Morel 1974, the particles' after Gordon & Morel 1983 with the backscattering ratio of Ulloa et al.
// 1994. The coloured dissolved matter is added in proportion to the chlorophyll, our assumption from
// what Case 1 water is (water whose optics follow its phytoplankton), not a relation taken from those
// papers. Red is absorbed over some 3 m, green and blue over some 17 and 25 m. The attenuation of the
// light going down and coming back is a + bb (Lee et al. 1999, Appl. Opt. 38:3831).
export const ABSORPTION = [ 0.30, 0.060, 0.040 ] as const;
export const BACKSCATTER = [ 0.0022, 0.0028, 0.0046 ] as const;

// The same water's total scattering coefficient b, which the floor seen through the water reads
// (ocean/Water.ts): pure seawater's b_w = 0.00288 ( lambda / 500 )^-4.32 (Morel 1974) and the
// particles' b_p = 0.30 Chl^0.62 ( 550 / lambda ) (Gordon & Morel 1983), at 600, 545 and 465 nm for
// the three channels. With the backscattering ratio above it gives back BACKSCATTER within a few per
// cent, so it is the same water. The beam attenuation a + b, some 0.48, 0.26 and 0.28 / m, is four to
// six times the light's diffuse attenuation a + bb in green and blue: most of the light from the floor
// that still arrives a few metres away has been scattered on its way, a little off its line.
export const SCATTER = [ 0.180, 0.199, 0.235 ] as const;

// The particles' forward scattering, which blurs the floor's image: the share of b_p within some 20
// degrees of the beam (0.9 of it at that backscattering ratio, Fournier & Forand 1994's phase function
// set by its backscattering ratio as in Mobley, Sundman & Boss 2002, Appl. Opt. 41:1035), as a
// coefficient, and the rms of its angle along one axis (radians, about 4 degrees). The wider share
// leaves the image for the water body's light. One coefficient (green's) serves every channel: the
// spread follows the particles, which scatter alike across the visible.
export const FORWARD_SCATTER = 0.177, FORWARD_ANGLE = 0.070;

// The deep water's irradiance reflectance just under its surface, R( 0- ) = 0.33 bb / ( a + bb ):
// the common first-order form of the relation Gordon, Brown & Jacobs (1975, Appl. Opt. 14:417)
// fitted to Monte Carlo runs of a homogeneous ocean (the coefficient as usually quoted, not read from
// that paper). For the water above, about ( 0.0024, 0.0147, 0.034 ).
export const BODY_REFLECTANCE = 0.33;

export function bodyReflectance( a: readonly number[] = ABSORPTION, bb: readonly number[] = BACKSCATTER ): number[] {

	return a.map( ( ai, i ) => BODY_REFLECTANCE * bb[ i ] / ( ai + bb[ i ] ) );

}
