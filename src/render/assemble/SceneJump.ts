// A running scene's jump: moves it to another scene as a fresh load of its query holds it.
//
// A running scene's jump (StartScene.ts): `establishScene` moves the page to another scene as a
// fresh load of its query holds it, `setTime` moves the (paused) scene clock, and SCENE_OPTIONS
// names the query options that make a scene on any page; an app adds its own (`sceneOptions`).
// A whole scene at once, as a page loads it (the capture tools' one page per set; the profiler's
// scene switches). `establishScene( query )` (a string or URLSearchParams) leaves the page as a
// fresh load of `?<query>` would draw it, from any scene before: once the caller has rendered a
// frame at the jump's camera and then run what it runs after a load (a view's eval), the paused
// frame equals the fresh load's, to the byte wherever the fresh load is itself reproducible
// (tools/island/verify-scene-jump.mjs checks every parity view from several scenes before).
// - The scene's options come from the query, each one left out at its load default
//   (RenderStages.ts `sceneOptions`, with the app's defaults): the weather established (wet in
//   rain; auto weather without `weather`), the day clock (hour or preset, moon phase, day length,
//   paused), the scene clock (`t`, `paused`) and the wind integrated to it, the cloud override. The
//   app's own options then (`jump`: its camera, say).
// - What the scenes before left behind is cleared, in the order the histories were registered
//   (post/SceneHistory.ts): the render pipeline's, the app's own (`histories`), the render
//   pipeline's light, the frame pacer's. The sky bakes as a fresh load's first bake (Sky.reset: both
//   endpoints and atlases, the clouds as before the first frame), the lightning forgets its
//   strikes, the storm sea its foam window, the refracted sun its map (back to its size and redrawn
//   at once after the bake, as the load's `prepare` draws it), the fog its bank; the sun's map
//   redraws from scratch and the temporal modes cut their history ('establish scene').
// - Load-time options the query names (quality, aa, dpr, the debug switches: read once when
//   materials and the frame graph are built) must equal the page's own, or it throws: reload
//   instead. A page may read one of them otherwise (`pageValue`: the island's `grow`, the land step
//   shown). Those it leaves out are the page's. GUI edits and console changes are not undone.
// The frame after the jump must render at the jump's camera before a view's eval moves it, as a
// fresh load's first frames do before a capture's eval runs: the storm's strikes take their
// bearings ahead of the camera of the frame that first sees them (Lightning.ts), the storm sea
// places its window there (StormSea.ts).
// The options form `{ weather?, preset?, hour?, time? }` (tools/render/profile-frame.mjs) sets what it
// names, keeps the rest and the camera, and clears the leftovers all the same.
// `__PONLAI__.sceneJump` (1) marks a page that keeps this contract: tools/kit/parity/capture-set.mjs
// captures a set in one page only where it is set (older builds' establishScene takes only the
// options form and leaves state behind). Raise it if the contract changes in a way callers see.
import { registerLightHistories, registerSceneHistories, sceneOptions, type RenderParts, type SceneDefaults } from '../RenderStages.ts';
import { parseSettings, settings as params, type SettingName, type SettingsQuery } from '../Settings.ts';
import { renderEvent } from '../post/SceneEvents.ts';
import type { DayPreset } from '../sky/DayCycle.ts';
import { type WeatherName } from '../sky/Weather.ts';
import { weatherWind } from '../weather/Wind.ts';
import type { FramePacer } from '../post/FramePacer.ts';

/** The query options that make a scene on any page: a jump may change them. */
export const SCENE_OPTIONS = [ 'weather', 'wind', 'windDir', 'preset', 'hour', 'moon', 't', 'clouds', 'dayLength', 'paused', 'clock' ];
export const SCENE_JUMP_VERSION = 1;

export type SceneRequest = string | { toString(): string; getAll( name: string ): string[] } | { weather?: WeatherName; preset?: DayPreset; hour?: number; time?: number };

export interface SceneJumpContent {
	/** every option a jump may change on this page (SCENE_OPTIONS by default) */
	sceneOptions?: readonly string[];
	/** the app's value of a load-time option, where it is not the URL's */
	pageValue?( key: string ): string[] | null;
	/** the app's part of a jump to a query (read by the app's own names), after the scene's options are set: its camera, say */
	jump?( query: SettingsQuery<string> ): Promise<void> | void;
	/** registers the app's histories, between the render pipeline's and its light's */
	histories?(): void;
}

export interface SceneJump {
	readonly sceneOptions: readonly string[];
	establishScene( scene: SceneRequest ): Promise<void>;
	setTime( t: number ): Promise<void>;
}

/** The scene jump of a running scene. */
export function createSceneJump( render: RenderParts, defaults: SceneDefaults, pacer: FramePacer, content: SceneJumpContent ): SceneJump {

	const { dayCycle, weather, wind, bakeExplicitly, refreshEnvironment, history } = render;
	const options = content.sceneOptions ?? SCENE_OPTIONS;
	// What a jump clears of the scenes before, registered layer by layer in the order it is cleared
	// (post/SceneHistory.ts): the render pipeline's, the page's, the render pipeline's light, the
	// pacer's.
	registerSceneHistories( render );
	content.histories?.();
	registerLightHistories( render );
	// a fresh load starts at the held frame rate, not at a fallback the last scene fell to
	history.onJump( 'frame pacer', () => pacer.reset() );
	const changeScene = async ( scene: SceneRequest ): Promise<void> => {

		const query = typeof scene === 'string' ? parseSettings( scene ) : 'getAll' in scene && typeof scene.getAll === 'function' ? parseSettings( scene.toString() ) : null;
		if ( query ) {

			const pageValues = ( key: string ) => content.pageValue?.( key ) ?? params.getAll( key as SettingName );
			const keys = [ ...new Set( query.entries.map( ( [ key ] ) => key ) ) ];
			const fixed = keys.filter( ( key ) => ! options.includes( key ) && query.getAll( key as SettingName ).join( '&' ) !== pageValues( key ).join( '&' ) );
			if ( fixed.length ) throw new Error( `establishScene: ${ fixed.join( ', ' ) } differ from the page's and are read once at load: reload the page with them` );
			const next = sceneOptions( query, defaults );
			render.cloudOverride = next.clouds;
			weather.establish( next.weather, next.wind );
			weather.auto = next.autoWeather;
			dayCycle.reset( next.day );
			if ( next.preset ) dayCycle.setPreset( next.preset );
			render.paused = next.paused;
			render.simTime = next.time;
			await content.jump?.( query );

		} else {

			const request = scene as { weather?: WeatherName; preset?: DayPreset; hour?: number; time?: number };
			render.cloudOverride = null;
			if ( request.weather ) {

				weather.auto = false;
				weather.establish( request.weather );

			}
			if ( request.preset ) dayCycle.setPreset( request.preset );
			else if ( request.hour !== undefined ) dayCycle.setHour( request.hour );
			if ( request.time !== undefined ) render.simTime = request.time;

		}
		history.jump( 'before bake' );
		renderEvent( 'establish scene' );
		render.manualSun = false;
		await bakeExplicitly();
		history.jump( 'after bake' );

	};
	return {
		sceneOptions: options,
		// (the loop draws no frame from its first change until its bake has published: `bakeExplicitly`)
		async establishScene( scene: SceneRequest ): Promise<void> {

			render.jumpsInFlight ++;
			try {

				await changeScene( scene );

			} finally {

				render.jumpsInFlight --;

			}

		},
		// moves the (paused) clock, so captures can step through an animation in one page load
		setTime( t: number ): Promise<void> {

			if ( ! Number.isFinite( t ) || t < 0 ) throw new RangeError( 'Scene time must be finite and non-negative' );
			render.simTime = t;
			wind.reset( { ...weatherWind( weather.state ), time: t } );
			renderEvent( 'seek' );
			return refreshEnvironment();

		},
	};

}
