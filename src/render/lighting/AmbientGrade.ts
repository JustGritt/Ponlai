import * as THREE from 'three/webgpu';
import { negateOnBackSide, normalView, normalViewGeometry, positionViewDirection, roughness, uniform, vec3 } from 'three/tsl';
import { floorOcclusion, screenAOAt } from '../post/GTAO.ts';
import { ambientEnvironment, ambientVolumeOn, specularOcclusion, type AmbientReceiver } from './AmbientSample.ts';
import { meterAmbient } from './MeterLight.ts';
import { AMBIENT_SETTINGS, DEFAULT_AMBIENT_SETTINGS as D, type AmbientSettings, type AmbientTables } from '../look/AmbientLook.ts';
import { overcastDeckLight, withMoon } from '../sky/Weather.ts';

// The ambient grade: one table of parameters on the ambient light, per sun elevation and weather.
//
// The ambient grade: one table of parameters on the island's ambient light, per sun elevation and per
// weather, so the art direction can depart from the physically lit result by the hour and the sky
// (a style grade on a physically based base, the approach Patry describes: lighting
// parameters, among them the ambient diffuse and specular balance and the sky's brightness, adjusted
// globally at the granularity of the weather and time-of-day states; "Real-Time Samurai Cinema",
// SIGGRAPH 2021, p. 10, and "Samurai Shading in Ghost of Tsushima", SIGGRAPH 2020, p. 9). The
// table multiplies on top of the sky's own grading (RenderStages.ts `skyFill`, look/SkyLook.ts `landSky`), which
// it leaves as it is.
//
// Every cell starts from the same values (`D`): identity but for the volume's directionality,
// 0.25 at every hour and in every weather, so that a surface in shadow turned toward the open sky
// keeps a little more of it than one turned toward the grove or the cliff (a small change: a culm's
// two sides in a grove's shadow move by a few per cent apart). The value is Patry's ("Real-Time
// Samurai Cinema", p. 43: a 25 % boost toward the direction of the harmonics' linear maximum, used
// throughout the game), kept after an A/B in our scenes. The other parameters are set by the hour
// and the sky as the scene's light is tuned. The sun's bounce is graded above the physically lit
// result from the middle to the low sun: the lit ground's own bounced light is carried further into
// the shadow beside it, a choice of style; no parameter adds sky light to the shadows.
//
// Set so far: the sun's bounce. The sky's diffuse light takes no parameter: the volume's sky term and
// the sky it bounces follow the sky itself at gain 1 (AmbientRelight.ts) at every hour and in every
// weather, so the evening and the night take the light their own sky gives, and the metering
// illuminance takes the sky's light as it stands (it is the incident light: Pipeline.ts
// `meterLight`). The sky's reflection takes half from a sun at 5.5 degrees to one at -1.3
// (`skySpecular`), under clear and fair skies alike, so a low sun's shadows stand dark against the
// lit rock and does not rise back through the sunset; the night's row returns it by -2.6 degrees.
// A clear evening keeps the day's exposure to -7 degrees (DayCycle.ts `displayNight`), but its
// reflection there is of a warm or a deep sky: on the island view the rock's mean C is 0.022 to
// 0.028 from -7 to -10 degrees, under a clear morning's 0.035 to 0.042 at -8 and -9 with the
// reflection held at 0.1. A clear morning takes the day's exposure long before the sun
// (DayCycle.ts `displayNight`, look/ExposureLook.ts `dawn`), and under it the night's full
// reflection of a bright, saturated sky lit the rock's ledges and the leaves cyan (at -8 degrees
// the cliff's brightest tenth L 0.33, C 0.11 with it, 0.19, 0.03 without), so the rising sun's rows
// (the table's `gradeRising`) hold the clear and the fair sky's reflection at 0.1 from -2.6 down to
// -12, where the display's night is whole again (DayCycle.ts `displayNight`), and give it back to
// the night's by -13. A return from -7 still lit the rock's edges from -9 to -7 degrees (on the
// island view their brightest tenth C 0.083 to 0.085 at -9 and -8, 0.066 or less held). What blue
// they keep (C about a fifth of their L at -6.5 degrees, whether the reflection is 0.1 or none) is
// the dawn's white balance over a grey surface (look/DisplayLook.ts `whiteBalance`), not this light.
//
// The sun's bounce under the clear and the fair sky is 1 (no grade) at the 60 degree row, about 1.3
// at the sun's noon of 52 degrees, 2 at 30, 2.3 from 20 down to 9, 1.8 at 7 and 1.4 at 5.5, and 1
// again from 4 degrees to its last light at -1.3, in the morning as in the evening (the rising
// sun's rows keep these). Raised so, the key's bounce off the lit meadow is a flat, warm green
// light in the shadow beside it, which does not follow the sun's shadow: the grass in a wide shadow
// beside the lit meadow stays green and turns only a little blue, while the lit grass moves by a
// few per cent (the readings: docs/reference/look-values.md, Lighting and shadows). The bounce at
// 1 is set with the probes' sky gain (AmbientRelight.ts PROBE_SKY) by the lit ground's share of
// the key light's direct light and the shadows' depth under it (there). Under the cloudy sky every row takes about half of
// the bounce at no grade (8 / 15), under overcast a third (0.33 from 2 degrees down). The moon's
// key bounces as the sun's does: its part takes the rows at the moon's height (the evening's), by
// its share of the key (Sky.ts `moonKeyShare`, `update`), so the night's bounce is the day's at
// that height (2.3 under the clear and the fair sky with the moon at 20 degrees, 8 / 15 under
// cloudy, a third under overcast). The night's rows are the sun's last, for the sun's part. The
// volume bounces the key as an even lobe over the whole bounced visibility, weighed by level
// ground's cosine of the key (AmbientRelight.ts), so it falls with the sun as the light on the
// ground does and reaches a face as far as bounced light reaches it from any side: it lights what
// faces lit ground or a lit crown (a blade's underside, a culm's sides, the leaves, the
// islanders). Read at degree 2, the island's receivers' read, with `sunBounce` at 1 (no grade)
// under the clear sky the bounces (the sky's and the key's) make 16 % of open level ground's light
// in shade at 62 degrees (the noon of the sun's earlier, higher path), 13 % at 50, 9 % at 30, 6 %
// at 20, 4.5 % at 13 and 10 and 3 % at 4. It also scales the sea's
// bounce onto the cliffs (GroundBounce.ts) and the islanders' bounce.
//
// Overcast is a dark grey sky with the sun still shining (Weather.ts): its sun's bounce takes a
// third of the bounce at no grade, times the sun's part of the weather's land gain under its deck
// (the tables' `overcastLand`, which its sky light takes too: AmbientRelight.ts); the moon's key
// takes a part of its own (`overcastMoonLand`) at its own height, so the night's bounce under the
// deck takes that whole.
// Under a cloud's shadow the open ground beside a grove reads 1.2 to 1.8 times its shade inside the
// grove's shadow; under overcast a grove's floor reads about 0.6 to 0.7 of the open ground.
//
// The table: rows by the sun's elevation (degrees, descending), interpolated linearly, the end rows
// held beyond them (look/SkyLook.ts `landSky`'s form), from the high sun down to the night; columns the kept
// weathers (clear, fair, cloudy, overcast), weighted by the weather's own blend of them
// (Weather.ts `skyMix`: a front moves the weights as it moves the sky; a parked weather counts as
// overcast). Evaluated on the CPU once a frame from the scene clock's sun and the weather (`update`)
// into the uniforms below, so it is a function of the clock and the weather alone, and a change of
// hour or weather writes uniforms only, never building a pipeline.
//
// The parameters and what reads them:
//   skySpecular     the environment's specular light (with three's multiple-scattering share of
//                   the ambient), and the islanders' sky rim.
//   skyBounce       the ambient volume's bounced sky on top of the sky's light (AmbientRelight.ts): a
//                   style parameter, 1 in every row.
//   sunBounce       the sunlight the ground bounces up (GroundBounce.ts `groundBounceLight`, where
//                   every material takes it: by default its sea alone, the volume carrying the
//                   land's), and the ambient volume's bounce of the key (AmbientRelight.ts).
//   directionality  the ambient volume's directionality boost (AmbientRelight.ts; 0 none).
//   aoStrength      how much of the ambient occlusion applies, mix( 1, ao, k ): the physical models'
//                   occlusion (`aoNode`: the terrain's vertex AO, the bamboo's canopy over the
//                   ground and the meadow, `groveSky`, where the volume does not carry it, the
//                   plants' own), times the screen-space
//                   occlusion on the materials that take it (post/GTAO.ts `receiveScreenAO`; the
//                   island's ground takes it in its own `aoNode`), the product held at
//                   the look's floor (post/GTAO.ts `floorOcclusion`) where the two multiply, and the islanders' baked
//                   occlusion in their fill. The local lights take the same occlusion
//                   (`diffuseOcclusion`, lighting/LocalLights.ts).
// The water, the sky, the clouds, the haze and the rain are not the island's ambient and do not
// read it.
//
// The GUI's Ambient folder shows the values and, with `override`, replaces the table's output with
// its sliders for live tuning; debug builds expose the controller as `__PONLAI__.ambient` and its
// values in the diagnostics.
// The table, its parameters and its gains under the deck are a look's, made into the grade's tables
// where the render pipeline's light stage creates the controller (look/AmbientLook.ts
// `createAmbientTables`, RenderStages.ts `createLight`), so the controller reads the tables it was
// given; this file keeps the uniforms, the controller that writes them and the lighting models' use
// of them.

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

/** The weight of the weather's land gain under overcast (look/AmbientLook.ts `overcastLand`) on
 * the land's sky light (AmbientRelight.ts) at a light's `elevation` (degrees), of which overcast's share is `overcast`
 * (sky/Weather.ts `skyMix`, a parked sky counting as overcast): the share times its deck's light
 * (sky/Weather.ts `overcastDeckLight`: at every sun by day, gone by -6 degrees), taken at the sun's
 * height and at the moon's for their parts of the sky's light. The key's bounce takes the same at
 * the key's two heights, each with its own part (`update`, `overcastMoonLand`). */
export function overcastLandWeight( elevation: number, overcast: number ): number {

	return overcast * overcastDeckLight( elevation );

}

/** The grade's uniforms, read by the lighting models. */
export const ambient = {
	skySpecular: uniform( 1 ),
	skyBounce: uniform( 1 ),
	sunBounce: uniform( 1 ),
	directionality: uniform( D.directionality ),
	aoStrength: uniform( 1 ),
	/** The land's sky gains under a deck over the sky's bake (sky/Weather.ts `deckSkyGain`, the land's
	 * own under overcast: AmbientRelight.ts OVERCAST_LAND_UNIT times the tables' `overcastLand`), which the volume
	 * carries: the environment's diffuse takes them where it stands in for the volume (a material
	 * built without a receiver, which by default is the sea floor: materials/TerrainMaterial.ts; and
	 * the volume's share under the water), so the ground under the water keeps the land's light (its
	 * ambient diffuse below the water under overcast takes the sea's, `seaSky`). 1 under a clear or
	 * fair sky. Written by AmbientRelight.ts. */
	landSky: uniform( 1 ),
	/** The sea's sky gain over the land's under overcast's deck, on its share (AmbientRelight.ts
	 * `seaScale` over `skyScale`: the water's deck gain, sky/Weather.ts DECK_SKY_GAIN, and its low
	 * sun's lift, over the land's gain), which the ambient diffuse under the water takes
	 * (lighting/Underwater.ts): the light down there is the light going down through the sea. 1
	 * under every other sky. Written by AmbientRelight.ts. */
	seaSky: uniform( 1 ),
};

const _moonRow = { ...D } as AmbientSettings;

/** The grade's controller: the values it wrote last, and the GUI's override. */
export interface AmbientGrade {
	/** The tables it reads (look/AmbientLook.ts `createAmbientTables`). */
	readonly tables: AmbientTables;
	/** The values in the uniforms. */
	readonly current: AmbientSettings;
	/** While on, `manual` replaces the table's output. */
	override: boolean;
	readonly manual: AmbientSettings;
	/** The table's output (whatever the override) at the last update. */
	readonly table: AmbientSettings;
	/** Once a frame: the sun's elevation (degrees), the weather's column weights (Weather `skyMix`),
	 * whether the sun is rising, and the moon's elevation (degrees) and its share of the key (Sky.ts
	 * `moonElevation`, `moonKeyShare`). */
	update( elevation: number, weights: readonly number[], rising: boolean, moonElevation: number, moonShare: number ): void;
}

/** The grade's controller on the grade's `tables`, made by the render pipeline's light stage
 * (RenderStages.ts `createLight`): it writes the uniforms above, which every lighting model reads. */
export function createAmbientGrade( tables: AmbientTables ): AmbientGrade {

	const { evaluate, sunBounceGain, overcastLand, overcastMoonLand } = tables;
	return {
		tables,
		current: { ...D },
		override: false,
		manual: { ...D },
		table: { ...D },
		update( elevation, weights, rising, moonElevation, moonShare ) {

			evaluate( elevation, weights, this.table, rising );
			// the key's bounce: the moon's part takes the rows at the moon's height, on the evening's side
			this.table.sunBounce = withMoon( this.table.sunBounce, evaluate( moonElevation, weights, _moonRow ).sunBounce, moonShare );
			Object.assign( this.current, this.override ? this.manual : this.table );
			for ( const key of AMBIENT_SETTINGS ) ambient[ key ].value = this.current[ key ];
			ambient.sunBounce.value *= sunBounceGain();
			// the weather's land gain under overcast's deck (`overcastLand`), the override's value too:
			// the moon's part of the key at the moon's height, as the bounce's rows above
			// (the sun's part by its own gain over the moon's, `overcastMoonLand`: exactly the moon's
			// term where the sun's deck light is gone)
			// (A look whose moon's part is no gain, 1, leaves no ratio to take. This is the expression
			// below in the limit of the moon's part going to 1: the moon's term goes to 0 and the sun's
			// keeps its own gain, so the two paths meet there with no step.)
			if ( overcastMoonLand === 1 ) {

				ambient.sunBounce.value *= 1 + ( overcastLand() - 1 ) * ( ( weights[ 3 ] ?? 0 ) * withMoon( overcastDeckLight( elevation ), 0, moonShare ) );
				return;

			}
			const sunPart = ( overcastLand() - 1 ) / ( overcastMoonLand - 1 );
			ambient.sunBounce.value *= 1 + ( overcastMoonLand - 1 ) * ( ( weights[ 3 ] ?? 0 ) * withMoon( overcastDeckLight( elevation ) * sunPart, overcastDeckLight( moonElevation ), moonShare ) );

		},
	};

}

/** The ambient occlusion `ao` at the grade's strength, mix( 1, ao, k ), written so that k = 1 gives
 * ao exactly (a lerp's ao - 1 rounds). */
export function gradeOcclusion( ao: F ): F {

	return ao.mul( ambient.aoStrength ).add( ambient.aoStrength.oneMinus() );

}

type Reflected = { directDiffuse: V3; directSpecular: V3; indirectDiffuse: V3; indirectSpecular: V3 };
type Context = { reflectedLight: Reflected; ambientOcclusion: F; iblIrradiance: V3; radiance: V3; thinIrradiance?: V3 };

const occlusions = new WeakMap<THREE.NodeBuilder, F>();

/** The occlusion the ambient diffuse takes, graded: the material's, times the screen's where the
 * material receives it (post/GTAO.ts). Built once per build, where it is first asked for (a model's
 * local lights read it before the ambient's occlusion: lighting/LocalLights.ts), and shared. */
export function diffuseOcclusion( builder: THREE.NodeBuilder ): F {

	let occlusion = occlusions.get( builder );
	if ( ! occlusion ) {

		const ao = ( builder.context as unknown as Context ).ambientOcclusion;
		const screen = ( builder as unknown as { material: THREE.Material | null } ).material?.userData.screenAO === true;
		occlusion = gradeOcclusion( screen ? floorOcclusion( ao.mul( screenAOAt() ) ) : ao ).toVar();
		occlusions.set( builder, occlusion );

	}
	return occlusion;

}

/** Applies the grade to a physical lighting model (three's PhysicalLightingModel or a subclass):
 * called in the constructor of the model root, GroundBounceLightingModel. Its environment terms
 * (three adds the environment's diffuse and specular in `indirectSpecular`, and nothing else
 * there) are gathered apart and added back, the specular scaled by skySpecular; its `ambientOcclusion`
 * takes the occlusion at aoStrength. With the ambient volume on and a `receiver` (the model's way
 * of reading it, lighting/AmbientSample.ts), the volume's irradiance (graded already) replaces the
 * environment's there, and the environment's specular takes the volume's occlusion. Its
 * `ambientOcclusion` is replaced: the occlusion multiplies the ambient diffuse, and the occlusion
 * times the reflection cone against the horizon (AmbientSample.ts `specularOcclusion`) the
 * environment's specular, on every receiver. A thin layer's light (a receiver's `thin`:
 * AmbientSample.ts) is left on the context as `thinIrradiance` for its model (the environment's
 * stand-in without the volume: its irradiance, a uniform sky's). With every parameter at 1 the sums and products are exactly those
 * without the grade; the compiled shaders still differ, which moved a few dozen pixels of a 4K
 * frame by one level against the frame without it. */
export function useAmbientGrade( model: THREE.LightingModel, receiver: () => AmbientReceiver | null = () => null ): void {

	const m = model as unknown as {
		indirectSpecular( builder: THREE.NodeBuilder ): void;
		ambientOcclusion( builder: THREE.NodeBuilder ): void;
	};
	const indirectSpecular = m.indirectSpecular.bind( model );
	m.indirectSpecular = ( builder ) => {

		const context = builder.context as unknown as Context;
		const r = context.reflectedLight;
		const diffuse = r.indirectDiffuse, specular = r.indirectSpecular;
		r.indirectDiffuse = vec3( 0 ).toVar();
		r.indirectSpecular = vec3( 0 ).toVar();
		const wanted = receiver();
		const name = ( builder as unknown as { material: THREE.Material | null } ).material?.name;
		const reads = wanted && ambientVolumeOn( wanted.kind ?? name ) ? wanted : null;
		if ( reads ) {

			// the volume's irradiance in the environment's place (graded by the relight already),
			// the environment's specular occluded by it (lighting/AmbientSample.ts)
			const iblIrradiance = context.iblIrradiance, radiance = context.radiance;
			const volume = ambientEnvironment( iblIrradiance.mul( ambient.landSky ), reads );
			context.iblIrradiance = volume.irradiance;
			// a thin layer's light (AmbientSample.ts `thin`), for its model to read
			context.thinIrradiance = volume.thin;
			context.radiance = radiance.mul( volume.occlusion );
			indirectSpecular( builder );
			context.iblIrradiance = iblIrradiance;
			context.radiance = radiance;
			diffuse.addAssign( r.indirectDiffuse );

		} else {

			// the environment in the volume's place, its diffuse with the land's sky gains the volume
			// carries; its specular as in the volume's branch, without them
			const iblIrradiance = context.iblIrradiance;
			context.iblIrradiance = iblIrradiance.mul( ambient.landSky );
			if ( wanted?.thin ) context.thinIrradiance = context.iblIrradiance;
			indirectSpecular( builder );
			context.iblIrradiance = iblIrradiance;
			diffuse.addAssign( r.indirectDiffuse );

		}
		specular.addAssign( r.indirectSpecular.mul( ambient.skySpecular ) );
		r.indirectDiffuse = diffuse;
		r.indirectSpecular = specular;

	};
	// three's occlusion replaced: the ambient diffuse takes the occlusion, the environment's
	// specular the occlusion times the reflection cone against the horizon (AmbientSample.ts
	// `specularOcclusion`), in place of three's estimate from the occlusion and N.V; alpha is the
	// model's GGX alpha (GroundBounceLightingModel sets roughness to its square root)
	m.ambientOcclusion = ( builder ) => {

		const context = builder.context as unknown as Context;
		const occlusion = diffuseOcclusion( builder );
		const Ng = negateOnBackSide( normalViewGeometry ) as unknown as V3;
		const cone = specularOcclusion( normalView, Ng, positionViewDirection, roughness.mul( roughness ) );
		context.reflectedLight.indirectDiffuse.mulAssign( occlusion );
		// the metering illuminance's ambient takes the ambient diffuse's occlusion (MeterLight.ts)
		meterAmbient.mulAssign( occlusion );
		context.reflectedLight.indirectSpecular.mulAssign( cone.mul( occlusion ) );

	};

}
