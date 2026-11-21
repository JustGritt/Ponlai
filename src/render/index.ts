// The render pipeline's main entry point: the only names other packages import from the render pipeline.
//
// The render pipeline's main entry point: the names the island, the game and the app layer import
// from the render pipeline, and no others. Another package imports the render pipeline through this
// file (or through render/data.ts, for its plain code, or render/setup.ts, for what the app installs
// before the render pipeline loads), never one of its modules by its path
// (tools/repo/verify-layers.mjs). It holds re-exports only, so loading it loads the whole renderer,
// whose modules declare and register as they load and do nothing else: no top-level await and no
// browser global read at load (tools/repo/verify-entry-points.mjs). The debug modules the app loads
// on demand are not here: they are lazy entry points (tools/repo/layers.json), imported dynamically
// or for their types only.
export { placeVegetation, vegetationCasters, bakeRainCollision } from './RenderStages.ts';
export type { RenderParts } from './RenderStages.ts';
export { startScene } from './assemble/StartScene.ts';
export type { SceneContent, SceneContext, SceneFrameSpec } from './assemble/StartScene.ts';
export type { SceneFrame } from './assemble/SceneFrame.ts';
export type { SceneJumpContent } from './assemble/SceneJump.ts';
export type { SceneHandlesContent } from './assemble/SceneHandles.ts';
export { assetUrl } from './FetchAsset.ts';
export { OPEN_SEA_SHORE, neutralBounceGround, neutralFields, neutralLattice, neutralSeaLattice } from './NeutralInputs.ts';
export { classifyBy } from './debug/Classes.ts';
export type { ClassRule } from './debug/Classes.ts';
export { debugFlags, setDebugFlag } from './debug/DebugFlags.ts';
export { FrameStats } from './debug/FrameStats.ts';
export { TEST_MODE } from './debug/TestMode.ts';
export { tune, tuneApply, tuneObject, tuneUniform } from './debug/Tune.ts';
export { GAIN, WIND_VIEW } from './debug/WindView.ts';
export { CANOPY_SHADE } from './lighting/AmbientSample.ts';
export type { AmbientReceiver } from './lighting/AmbientSample.ts';
export type { AmbientPrepared, AmbientView, AmbientVolume } from './lighting/AmbientVolume.ts';
export { GroundBounceLightingModel } from './lighting/GroundBounce.ts';
export type { BounceGround } from './lighting/GroundBounce.ts';
export type { UserDataKey, UserDataName } from './UserData.ts';
export { liftKeyWorld } from './lighting/KeyLift.ts';
export { testSources } from './lighting/LocalLights.ts';
export type { LocalLightKind, LocalSource } from './lighting/LocalLights.ts';
export { CANOPY_TOP, TURF_CANOPY_DEPTH, meadowCanopyLight, meadowGroundLight } from './lighting/MeadowCanopy.ts';
export { meterKey, writeMeter } from './lighting/MeterLight.ts';
export type { RefractedSun } from './lighting/RefractedSun.ts';
export { solarVolumeVisibility } from './lighting/SolarShadow.ts';
export { UnderwaterLightingModel, underwaterLight } from './lighting/Underwater.ts';
export type { UnderwaterLight, UnderwaterModelOptions } from './lighting/Underwater.ts';
export { AMBIENT_SETTINGS } from './look/AmbientLook.ts';
export { heightBlend, hexReadScan } from './materials/HexTiling.ts';
export type { HexScan } from './materials/HexTiling.ts';
export { DETAIL_SIZE, bakeProceduralTextures } from './materials/ProceduralTextures.ts';
export type { DetailTextures } from './materials/ProceduralTextures.ts';
export { DRY_F0, FILM_ROUGHNESS, WATER_F0, runnels, wetAlbedo, wetPorosity, wetRoughness } from './materials/Wetness.ts';
export type { OceanFFT } from './ocean/OceanFFT.ts';
export { markSwimmer, refractedGroup, showUnderwater } from './ocean/Refraction.ts';
export { OPEN_DRIFT } from './ocean/SeaDetail.ts';
export type { SeaDetail } from './ocean/SeaDetail.ts';
export type { ShoreFields, ShoreWaves } from './ocean/ShoreWaves.ts';
export { swimmerSway } from './ocean/Water.ts';
export type { WaveMaps } from './ocean/WaveMaps.ts';
export { AA_MODE, ALPHA_TO_COVERAGE, FSR3, TEMPORAL } from './post/Antialiasing.ts';
export type { AAMode } from './post/Antialiasing.ts';
export type { DayKey } from './post/DayKeys.ts';
export type { ExtraPass } from './post/ExtraPass.ts';
export { nextFrame } from './post/FramePacer.ts';
export { GTAO, floorOcclusion, receiveScreenAO, screenAOAt } from './post/GTAO.ts';
export { LOCAL_TONE } from './post/LocalTone.ts';
export { MEADOW_SHADOWS, SS_SHADOW, meadowGroundShadow, meadowShadowAt, meadowShadowFloor, receiveScreenShadow, screenShadowAt, screenThickness } from './post/MeadowShadows.ts';
export { motionNode, motionState, velocityOutput } from './post/Motion.ts';
export type { PipelineSettings } from './post/Pipeline.ts';
export { declareSceneEvents, renderEvent } from './post/SceneEvents.ts';
export { fsr3MipBiasScale } from './post/fsr3/MipBias.ts';
export { compileLatentInto } from './post/graph/Compile.ts';
export { debugViewChoices } from './post/graph/DebugView.ts';
export { shared } from './shared.ts';
export type { Shared } from './shared.ts';
export { DAY_PRESETS, duskForce, nightGlowShare } from './sky/DayCycle.ts';
export type { DayCycle, DayPreset } from './sky/DayCycle.ts';
export { markSkyRetune } from './sky/SkyRetune.ts';
export { WEATHER_PRESETS, WIND_LEVELS, isParkedWeather } from './sky/Weather.ts';
export type { Weather, WeatherName, WindLevel } from './sky/Weather.ts';
export { gpuDevice, gpuTexture } from './three/Backend.ts';
export { puddleField, puddleRipples } from './weather/Puddles.ts';
export type { Rain } from './weather/Rain.ts';
export { AIR_DENSITY, GUST_MEAN, GUST_TRAVEL, PLANT_RATE_BANDS, REFERENCE_SPEED, WIND_STATE_VEC4S, Wind, createWindUniforms, gustAt, gustJitter, logLaw, packWindState, plantSpeedOf, plantWind, rootHash, setWindDisplacement, wave, windBefore, windDisplacement, windField, windStateOf, windStrength } from './weather/Wind.ts';
export type { WindResponse, WindState, WindUniforms } from './weather/Wind.ts';
export { boundsSteps, runSteps } from './util/Steps.ts';
export { geometrySteps, textureSteps } from './util/Upload.ts';
export { previousSway, useRestPoseShadows } from './vegetation/VegetationShadow.ts';
export { followWindMotionFlag } from './vegetation/WindMotion.ts';
export { useLeafCoverage } from './vegetation/Coverage.ts';
export { defaultPlant } from './vegetation/DefaultPlant.ts';
export { CROWN_DRAG, branchBend, leafFlutter, mainBend, plantLean } from './vegetation/Deform.ts';
export type { PlantWind } from './vegetation/Deform.ts';
export { RESPONSE, createDisplacementField, displacementAt, displacementField } from './vegetation/Displacement.ts';
export type { DisplacementField } from './vegetation/Displacement.ts';
export { transmission, useSpecular, useVegetationMotion, vegetationMaterial } from './vegetation/Material.ts';
export { PLANT_PART, PlantInstances, levelDraws, placeVertex, plantPoint } from './vegetation/PlantInstances.ts';
export type { PlantDraw, PlantInstance, PlantPool } from './vegetation/PlantInstances.ts';
export { Vegetation } from './vegetation/Registry.ts';
export type { GrowthInput, GrowthSteps, KindGrowth, PlantSite, VegetationKind } from './vegetation/Registry.ts';
export { BRANCH_ATTRIBUTES, MeshBuilder, UP, branchValues, leafLayout, smooth, sprayValues, streamAt } from './vegetation/TreeMesh.ts';
export { IDENTITY_COLOUR, LeafPainter, leafMaterial, plantTint } from './vegetation/TreeSurfaces.ts';
export type { LeafAtlas, PlantTint } from './vegetation/TreeSurfaces.ts';
export { unjitteredPadded } from './vegetation/VegetationCull.ts';
