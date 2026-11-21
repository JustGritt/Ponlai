import type { Look } from '../../render/setup.ts';
import { SEA } from './Sea.ts';
import { DAY } from './Day.ts';
import { AMBIENT } from './Ambient.ts';
import { EXPOSURE } from './Exposure.ts';
import { DISPLAY } from './Display.ts';
import { HAZE } from './Haze.ts';
import { SKY } from './Sky.ts';
import { OCCLUSION } from './Occlusion.ts';
import { LOCAL_LIGHTS } from './LocalLights.ts';
import { MEADOW_CANOPY } from './MeadowCanopy.ts';
import { WIND } from './Wind.ts';

// The island's look: its numbers for every table of the render pipeline's look, gathered in one object.
//
// The island's look: its numbers for every table of the render pipeline's look (look/Look.ts), one
// file per system beside this one, each holding numbers and pure functions of numbers and importing
// only the render pipeline's types. The app's entry picks it (PageSettings.ts) and the app hands
// it to the render pipeline's stages (main.ts); a tool hands it to what it creates. The comments
// beside the numbers say what each does on screen and how it was set; the reader modules
// (packages/render/look/) say what each table is.
export const ISLAND_LOOK: Look = { sea: SEA, day: DAY, ambient: AMBIENT, exposure: EXPOSURE, display: DISPLAY, haze: HAZE, sky: SKY, occlusion: OCCLUSION, localLights: LOCAL_LIGHTS, meadowCanopy: MEADOW_CANOPY, wind: WIND };
