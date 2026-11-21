// Runtime switches for timing a change inside one page, exposed in debug builds.
//
// Runtime switches for timing a change inside one page (debug builds expose them as
// window.__PONLAI__.flags). An optimisation under development keeps its old path and reads a flag
// where it chooses between the two, every frame:
//
//   import { debugFlags } from './DebugFlags.ts';
//   const mesh = debugFlags.lowPolyGrass ? newGrass : oldGrass;   // or skip a pass, pick a material
//
// and tools/render/profile-frame.mjs flips it every few frames with --toggle "s.flags.lowPolyGrass = on",
// after warming both paths, to A/B it in seconds without a reload. A shader change keeps both
// materials (or both nodes) built and swaps them on the flag, rather than branching inside one
// shader, so each side runs its own pipeline. A missing flag reads as undefined: the old path
// while a change is under development. Once an optimisation ships, a missing flag must mean the
// shipped path, so its flag either selects the old path only on an explicit `false`
// (`flags.seaReflection = false`) or is named for the old path (`flags.skyWholeBake = true`); the code
// that reads the flag says which. --toggle sets the flag explicitly on both sides either way.
//
// `mv` holds the temporal modes' motion vector sources (packages/render/post/Motion.ts), each on unless set to
// false: `flags.mv.camera = false` leaves the camera's motion out of every surface's velocity, as
// MV-UNIT compares a source on against off. `camera` and `object` (an object's world matrix) are
// read by Motion.ts; `skin`, `wind`, `sea` and `foam` by the movers that write those motions
// (`wind`: the vegetation's materials, through WindMotion.ts, rebuild without the previous
// sway, so a --toggle of it also times that sway).
//
// Under aa=fsr3: `fsrMipBias = false` holds the terrain's mip bias at none (post/fsr3/MipBias.ts,
// read every frame; on unless false); `fsrAutoReactive = true` builds and runs FSR's reactive-mask
// generator (post/fsr3/Fsr3Upscaler.ts, read at load; off unless true); `fsrJitterSign` and
// `fsrMotionScale` (numbers, 1 unless set) multiply the jitter offset and the motion vector scale
// FSR is given, every frame (the conventions' checks: -1 tells FSR the wrong jitter sign).
//
// `shadowWind = false` casts the plants' shadows in their rest pose (vegetation/VegetationShadow.ts) and
// brings back the sun's map's stepped redraws (lighting/SunShadow.ts); on unless false, the swayed
// shadows redrawn every frame. Read every frame; its cost: node tools/render/profile-frame.mjs <url>
// --toggle "s.flags.shadowWind = on".
//
// `windFreeze = true` holds the vegetation's wind while the scene clock runs (Wind.ts): the plants
// stand in one pose with no motion, for clips where only the shadows or the jitter should move.
//
// `trample = false` leaves the islanders' trail through the meadow (world/Trample.ts) out
// when set at load (`flags=trample:false`: no displacement field, no sprites, the blades read
// none), and empties its walkers when set at run time (--toggle "s.flags.trample = !on" times the
// sprites' draw).
// `clearSkyFill` (a number, or `true` for 1; off unless set) scales the sky light the island takes
// for its clear-sky fill light (look/SkyLook.ts `clearSkyFill`), read whenever the sky tracks the clock (an
// environment bake later, `refreshEnvironment()` while paused).
// `skyGain = false` leaves the sun's in-scattered sky light as the atmosphere gives it (sky/Sky.ts
// `look.skyLight`), read whenever the sky tracks the clock (an environment bake later,
// `refreshEnvironment()` while paused).
// `streaks` (a number, 1 unless set) scales the weather's amount of the thin cloud layer
// (sky/Clouds.ts `streaksAt`; 0 turns it off), read whenever the sky tracks the clock;
// `streakMask = true` (at load) paints the dome's pixels where the layer's opacity as seen passes
// 0.1 in a red of their own luminance and those where it lies between 0.02 and 0.1 in a green, for
// `img.py stats --mask` (sky/SkyDome.ts domeColour).
// `lowKey = false` leaves the low sun's key as its transmittance gives it (look/SkyLook.ts `lowKey`), read
// whenever the sky tracks the clock.
// `clearSky = false` leaves the clear day's sky light without its gain against the key (sky/Sky.ts
// `look.clearSky`), read whenever the sky tracks the clock (an environment bake later).
// `lowSun` (a number, or `true` for 1; off unless set) scales the low sun's warm light and
// blue-violet shadows (sky/SkyTrack.ts lowSun), read every frame (the shadows' tint at the next environment bake).
// `keyLift = false` keeps the direct light and the sun's shadows on the low key's own direction
// (lighting/KeyLift.ts), read at each publish of the lighting.
// Read at load: `tone` (a number, 0 unless set) and `filmNight = true` bring back the grade's tone
// and the film's night (post/Pipeline.ts TONE, post/PipelineDisplay.ts NIGHT); `localTone = false`
// builds the display chain without its local tone mapping (post/LocalTone.ts).
// The meadow's light, each on unless false: `meadowShadows` (its screen-space shadows and the
// vegetation's depth pass, post/MeadowShadows.ts), `groundCast` (the crowns' shadow on the turf,
// materials/TerrainMaterial.ts), `groundShadow` (the blades' screen-space shadow on the turf, the
// same), `grassClumps` (the blades' clumps, world/Grass.ts CLUMP);
// `grassShadow = false` (at load) turns off the shadowing by the meadow's layer on its blades and the
// ground under and beside it (world/Grass.ts GRASS_SHADOW); `grassSunMap = false` (at load) keeps
// the sun's shadow map off the blades, their own shades kept (world/Grass.ts);
// `grassLod = true` (at load) turns on the meadow's LOD, off unless true: from 20 m along the
// ground, the blades thinned a quarter kept per doubling and the kept ones widened (world/Grass.ts
// LOD);
// `grassDensity` and `grassWidth` (at load, 1 unless set) scale the blades per m^2 and their width;
// `grassMask = true` (at load) paints the blades flat red, for the cover of `img.py stats --cover`;
// `grassMask = 'exact'` draws them unlit in a red far above the scene's radiance, each pixel's
// cover exact in the opaque colour, `grassMask = 'albedo'` unlit in their diffuse colour and
// `grassMask = 'lean'` as 'exact' with each leaf's lean in green and blue (tools/island/meadow-light.mjs).
// `bareGround = true` (at load) hides the blades, keeps their cast out of the sun's map and lights
// the turf as bare ground (world/Grass.ts BARE_GROUND): with `lightOnly`, the light bare ground
// takes beside the sward, at the same pixels as a frame with the blades.
// The lighting layer's debug views, read at load (debug/LightLayer.ts): `lightOnly` (1 or true:
// every lit surface's diffuse albedo 0.5 grey, or `grey:<albedo>`'s; 2: the specular off too) and `ambientOnly = true`
// (the key light out of the lit materials: only the sky's light and the ground's bounce), which
// combine, and `albedoOnly = true` (every lit surface unlit in its diffuse albedo, over the other
// two), `opaqueVegetation = true` (the vegetation lit as opaque surfaces, no light through it,
// combining with the others), `normalOnly = true` (every lit surface unlit in its shading normal, ( n + 1 ) / 2), `occlusionOnly = true` (every lit surface unlit in the occlusion its ambient diffuse takes,
// over those three), and `classes = true` (each lit surface in a flat colour naming its kind,
// debug/Classes.ts, over all of them); `__PONLAI__.radiance( boxes )` (debug/Radiance.ts) reads the scene's linear radiance.
// `ambientRelight = false` (or 0) skips the ambient light's relight (lighting/AmbientRelight.ts),
// read every frame, so --toggle "s.flags.ambientRelight = on" times its kernel (the output keeps the
// last frame's; the first relight runs whatever the flag, so off at load keeps the first relight's).
// Read at load (lighting/AmbientSample.ts): `ambientVolume = false` (or 0) builds the lit materials
// on the environment's irradiance as before the ambient volume, the ground bounce whole;
// `groundBounceLand = true` (or 1) keeps the ground bounce's land beside the volume;
// `canopyShadow = true` (or 1) keeps the bamboo canopy's baked shadowing beside it (the chunks' sky
// visibility and the culms'), which the volume carries; `ambientVolumeOff = 'terrain/meadow'`
// sends the receivers named back to the environment's irradiance (the A/B of each one's cost; for
// timing only: the bakes and the ground bounce's land stay the volume's, and `ambientVolume = false`
// gives the old path).
// `gtao = false` (at load) builds without the screen-space ambient occlusion and the opaque
// scene's depth prepass it reads (post/GTAO.ts, post/OpaquePrepass.ts), which the high tier's `ao`
// setting builds; `gtao = true` builds them on a tier whose setting is off. At run time `gtao = false` skips
// their passes and the receivers read 1, so --toggle "s.flags.gtao = on" times it.
// `ssShadow = false` (at load) builds the meadow's screen-space shadows over the vegetation's depth
// alone, as before they reached the whole scene (post/MeadowShadows.ts `SS_SHADOW`; on unless
// false); at run time it keeps the opaque scene's depth out of the shadows, its prepass skipped unless the
// occlusion runs, and the receivers other than the meadow read 1, so
// --toggle "s.flags.ssShadow = on" times it.
// `localLights = false` (or 0; at load or at run time) turns the local lights off, their clusters'
// pass skipped and the lit materials' lookup with it (lighting/LocalLights.ts), so
// --toggle "s.flags.localLights = on" times them by night.
// `lightTest = N` (at load) adds N white test sources over the meadow to the local lights, with the
// yunpeng's range (lighting/LocalLights.ts `testSources`), for their cost tests only.
// `hazeKernels = false` skips the volumetric haze's two kernels (weather/VolumetricHaze.ts), read
// every frame, so --toggle "s.flags.hazeKernels = on" times them (the composite then reads the
// volumes as they last were).
// `fallingLeaves = false` skips the falling leaves' pass and hides them (world/FallingLeaves.ts),
// read every frame, so --toggle "s.flags.fallingLeaves = on" times them.
// `cloudHazeEnd = false` marches the low deck's view through its whole shell instead of ending where
// the haze closes (sky/CloudKernels.ts `makeMarch`; the haze's copy of its medium, weather/VolumetricHaze.ts
// `prepareMedium`), read every frame, so --toggle "s.flags.cloudHazeEnd = !on" times the end.

import { settings } from '../Settings.ts';

export interface MotionSourceFlags { camera?: boolean; object?: boolean; skin?: boolean; wind?: boolean; sea?: boolean; foam?: boolean }
export const debugFlags: Record<string, boolean | number | string | MotionSourceFlags | undefined> & { mv: MotionSourceFlags } = { mv: {} };

/** Sets a flag from code of another layer (the island's wind probe, a URL option that stands for a
 * flag): the flags' one operation for a writer that is not the renderer's own. */
export function setDebugFlag( name: string, value: boolean | number | string | MotionSourceFlags | undefined ): void {

	debugFlags[ name ] = value;

}

// `flags=<name>:<value>,...` in the URL (debug builds: the dev server, ?debug, ?profile) sets flags
// as this module loads, before any module reads them, so a capture or a clip A/Bs a flag by its
// query. A value reads as true / false, a number, or else the string itself (a bare name, without
// `:value`, as true); `mv.<source>` sets a motion source. (Under node, as the plain verify tools run
// modules, no settings are installed: no flags.)
const query = settings;
if ( query.has( 'flags' ) && ( import.meta.env?.DEV || query.has( 'debug' ) || query.has( 'profile' ) ) ) {

	for ( const entry of ( query.get( 'flags' ) ?? '' ).split( ',' ) ) {

		// a bare name reads as `name:true`
		const at = entry.indexOf( ':' );
		if ( at === 0 || entry === '' ) continue;
		const name = at < 0 ? entry : entry.slice( 0, at ), text = at < 0 ? 'true' : entry.slice( at + 1 );
		const value = text === 'true' ? true : text === 'false' ? false : text !== '' && Number.isFinite( Number( text ) ) ? Number( text ) : text;
		if ( name.startsWith( 'mv.' ) ) ( debugFlags.mv as Record<string, unknown> )[ name.slice( 3 ) ] = value;
		else debugFlags[ name ] = value;

	}

}
