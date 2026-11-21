// The `userData` keys the render pipeline knows: where each sits, who sets it and who reads it.
//
// The `userData` keys the render pipeline knows. three's `userData` is an open bag on every object,
// material, geometry and texture; the render pipeline and what is built on it leave marks there for
// each other (a material says what it takes part in, a pass finds its copies again). The bag has no
// type, so a mark set under one spelling and read under another fails without a word. This is the
// list of every key a file of the render pipeline sets or reads: where the mark sits (`on`), who
// sets it (`sets`), who reads it (`reads`) and what it means. tools/repo/verify-layers.mjs fails
// when a file names a key that is not here (or, above the render pipeline, on its own layer's
// list), or when the list holds a key no file names; a key reached through an alias of the bag
// (`const data = material.userData`) is not seen. The render pipeline's entry point offers the names
// (`UserDataName`) and the shape of an entry (`UserDataKey`) to the layers above.
//
// A mark the island's or the game's code keeps to itself (set and read above the render pipeline)
// is not the render pipeline's: it is listed on that layer's own list (tools/repo/layers.json,
// `userData`; the island's in world/IslandUserData.ts). A new mark the render pipeline reads is
// declared here with the change that adds it.
//
// This module is data only and imports nothing, so the CPU-only tools load it under plain node.

export interface UserDataKey {
	/** what carries the mark */
	on: 'material' | 'object' | 'geometry' | 'texture' | 'scene';
	/** who sets it */
	sets: string;
	/** who reads it */
	reads: string;
	/** what it means */
	means: string;
}

export const USER_DATA = {
	vegetation: {
		on: 'material', sets: 'vegetation/Coverage.ts markVegetation',
		reads: 'SunShadow.ts, VegetationPrepass.ts, OpaquePrepass.ts, vegetation/Registry.ts, the rain\'s bake, the kinds\' view, the island\'s wind probe and terrain',
		means: 'a cut-out vegetation material: it takes the vegetation\'s shadow filter and the depth prepass, and is found by this mark, never by alphaToCoverage',
	},
	meadowDetail: {
		on: 'material', sets: 'the island\'s terrain material', reads: 'vegetation/Coverage.ts (documented there)',
		means: 'a ground material that draws the meadow\'s far detail: vegetation in look, but it keeps the ordinary shadow filter',
	},
	prepassCopy: {
		on: 'object', sets: 'VegetationPrepass.ts, OpaquePrepass.ts',
		reads: 'the two prepasses, vegetation/Registry.ts, RainCollision.ts',
		means: 'a depth-only copy of a mesh, which no list of the scene\'s own meshes takes',
	},
	prepassOpaque: {
		on: 'material', sets: 'VegetationPrepass.ts', reads: 'VegetationPrepass.ts',
		means: 'the material draws over its depth prepass without its cut-out test (part of its cache key)',
	},
	prepassOn: {
		on: 'material', sets: 'VegetationPrepass.ts', reads: 'VegetationPrepass.ts',
		means: 'whether the depth prepass is on for the material, as last applied',
	},
	vegetationPrepass: {
		on: 'scene', sets: 'VegetationPrepass.ts', reads: 'VegetationPrepass.ts (the diagnostics)',
		means: 'what the vegetation\'s depth prepass installed in the scene',
	},
	grassLod: {
		on: 'object', sets: 'the meadow\'s meshes (Grass.ts)', reads: 'VegetationPrepass.ts',
		means: 'the blade mesh\'s level-of-detail choice per draw, which its depth copy follows',
	},
	grassCulled: {
		on: 'object', sets: 'the meadow\'s meshes (Grass.ts), VegetationPrepass.ts for the copy', reads: 'VegetationPrepass.ts',
		means: 'the mesh draws the culled list of its instances, and so does its depth copy',
	},
	plantDraw: {
		on: 'object', sets: 'vegetation/PlantInstances.ts', reads: 'vegetation/PlantInstances.ts, OpaquePrepass.ts, RainCollision.ts',
		means: 'which draw of an instanced plant kind the mesh is (the index of its range in the cull\'s list)',
	},
	rainBake: {
		on: 'object', sets: 'vegetation/PlantInstances.ts', reads: 'RainCollision.ts',
		means: 'false on a mesh that leaves its shape in the rain\'s collision bake to another (a plant\'s far level)',
	},
	bakeCount: {
		on: 'object', sets: 'an instanced mesh that picks its count per draw', reads: 'RainCollision.ts',
		means: 'the mesh\'s whole instance count, for the rain\'s collision bake',
	},
	castRest: {
		on: 'material', sets: 'VegetationShadow.ts', reads: 'VegetationShadow.ts, RainCollision.ts',
		means: 'the rest pose the material casts its shadow with, for the passes that keep it whatever the debug flag',
	},
	shadowPush: {
		on: 'material', sets: 'SunShadow.ts', reads: 'RainCollision.ts',
		means: 'the sun\'s map wrapped the material\'s cast position with its push along the light, which the rain\'s bake must not take',
	},
	sunOcclusion: {
		on: 'material', sets: 'the island\'s terrain and meadow materials', reads: 'SunShadow.ts',
		means: 'a node of the material\'s own shadowing of the key light, taken instead of the screen-space shadow',
	},
	sunVisibility: {
		on: 'material', sets: 'the meadow\'s material (a debug flag)', reads: 'SunShadow.ts',
		means: 'a node that replaces the sun\'s map for the material',
	},
	screenShadow: {
		on: 'material', sets: 'MeadowShadows.ts receiveScreenShadow', reads: 'SunShadow.ts',
		means: 'the material takes the key light\'s screen-space shadow',
	},
	screenThickness: {
		on: 'material', sets: 'MeadowShadows.ts screenThickness', reads: 'OpaquePrepass.ts',
		means: 'the thickness (m) the screen-space shadows give the material\'s depth copy',
	},
	screenAO: {
		on: 'material', sets: 'GTAO.ts receiveScreenAO', reads: 'AmbientGrade.ts',
		means: 'the material\'s ambient light takes the screen-space ambient occlusion',
	},
	keyThrough: {
		on: 'material', sets: 'a tree\'s leaf material (the bamboo\'s)', reads: 'SunShadow.ts, vegetation/LeafLight.ts',
		means: 'a shaded leaf takes the key\'s light through the leaves over it',
	},
	refractedSolarReceiver: {
		on: 'material', sets: 'RefractedSun.ts, the island\'s terrain material', reads: 'RefractedSun.ts, Underwater.ts',
		means: 'the material is lit by the refracted sun under water, and its meshes are drawn into the underwater map',
	},
	refractedSolarFloor: {
		on: 'material', sets: 'RefractedSun.ts, the island\'s terrain material', reads: 'RefractedSun.ts',
		means: 'the material is the sea floor, which only receives the underwater map',
	},
	sharedIndex: {
		on: 'geometry', sets: 'RefractedSun.ts', reads: 'RefractedSun.ts',
		means: 'a subset geometry that borrows its source\'s index, which its disposal must not free',
	},
	shoreWaveShelter: {
		on: 'texture', sets: 'ShoreWaves.ts', reads: 'the debug tools, through the page',
		means: 'the shore field\'s measurements, kept on its texture',
	},
	previousPosition: {
		on: 'material', sets: 'Motion.ts, VegetationShadow.ts, Water.ts, the fish\'s material', reads: 'Motion.ts, VegetationShadow.ts, WindMotion.ts',
		means: 'the material writes its own previous position in its position stage, for its velocity',
	},
	motion: {
		on: 'material', sets: 'VegetationShadow.ts, Water.ts, any material with a positionNode', reads: 'Motion.ts',
		means: 'how the material\'s positionNode moves (\'animated\' or \'static\'), for the motion check',
	},
	windDisplacement: {
		on: 'material', sets: 'weather/Wind.ts', reads: 'weather/Wind.ts windDisplacement, Motion.ts',
		means: 'the material\'s wind displacement (its current and previous sway)',
	},
	windMotion: {
		on: 'material', sets: 'WindMotion.ts, VegetationShadow.ts', reads: 'WindMotion.ts, VegetationShadow.ts',
		means: 'false when the material builds without the wind\'s motion in its velocity (a debug flag; part of its cache key)',
	},
} as const satisfies Record<string, UserDataKey>;

export type UserDataName = keyof typeof USER_DATA;
