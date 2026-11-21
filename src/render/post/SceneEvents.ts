import { TEMPORAL } from './Antialiasing.ts';
import { motionState, taaCut } from './Motion.ts';
import { restEpoch } from './TemporalRest.ts';
import { onSceneDispose } from '../SceneDispose.ts';

// The app's scene events for the temporal modes: what changes the scene says so in one call.
//
// The app's scene events for the temporal modes (Antialiasing.ts: aa=taa|taau). A temporal frame
// is a function of the frames before it, so whatever changes the scene in a way the frames
// themselves do not show must say so, at the site that does it, with one call: `renderEvent( reason
// )` for the render pipeline's own reasons, the function of the declaration that names it for the
// others. Every event bumps the rest's epoch (TemporalRest.ts: a paused scene converges again from
// scratch); a cut also drops the motion across it and starts the histories over (Motion.ts
// `taaCut`: the camera's, the objects' and the wind's previous state become the current one), for a
// change the history must not carry over (a jump of the camera or of the clock, a new scene,
// another canvas). An event that only retunes what is drawn (a GUI control, a front sweeping in
// over seconds) keeps the history: the running resolve follows it as it follows a moving scene.
//
// The sites call it with their reason. A reason is declared before it is raised, with whether it
// cuts (`declareSceneEvents`): the render pipeline declares its own below (`renderEvent`), and what
// is built on it declares its own where it raises them (the island's growth, the game), so this
// module names none of theirs. A declaration returns the function that raises its reasons and no
// others, so a misspelt reason, or one of another declaration, does not compile; a reason declared
// twice is refused. A declaration says only whether a reason cuts, so the reasons only the app
// raises (the loop's start, a resize, a view jump) are declared among the render pipeline's, where
// the rest of what they mean is known. Generic sites only (the app's setters and handlers), never
// the camera controls' internals: a teleport of the camera that no event covers (a direct
// assignment from the console, a control's own jump) is caught by Motion.ts's camera-delta detector
// (more than 8 m, 30 degrees or 2 % of field of view in one rendered frame). Under MSAA the epoch
// bumps, which the page's settled signal (Settled.ts) reads in every mode, and the cuts reach
// `onSceneCut`'s listeners, which hear them in every mode (the exposure's adaptation: Pipeline.ts);
// the temporal histories listen through Motion.ts `onTaaCut`.
//
// The diagnostics' `taa.lastEvent` (TemporalAA.ts) holds the last event: its reason, whether it cut,
// and the rendered frame it followed (`frame`: the motion's frame count when it was raised; it
// applies to the next frame rendered).

// every declared reason, and whether it cuts
const declared = new Map<string, boolean>();

/** Declares reasons of scene events, each with whether it cuts the history, and returns the
 * function that raises them (and only them). A reason already declared throws. */
export function declareSceneEvents<const T extends Record<string, boolean>>( reasons: T ): ( reason: keyof T & string ) => void {

	for ( const [ reason, cut ] of Object.entries( reasons ) ) {

		if ( declared.has( reason ) ) throw new Error( `SceneEvents: the reason '${ reason }' is declared twice` );
		declared.set( reason, cut );

	}
	return raise;

}

/** The render pipeline's own reasons, and whether each cuts. */
const RENDER_EVENTS = {
	// the frame loop starts (the app's first frame) or restarts (`loop.start()` after a stop)
	'loop start': true,
	// the page shows again (visibilitychange to visible): its last frame may be minutes old
	'visible': true,
	// the scene clock is set (`setTime`): the sea, the wind and the sky jump
	'seek': true,
	// a whole scene at once (`establishScene`: weather, time and clock)
	'establish scene': true,
	// a weather or its wind set at once (`setWeather( name, 0 )`, `setWind( choice, 0 )`), and one
	// that sweeps in over seconds
	'weather jump': true,
	'weather front': false,
	// the time of day or a time preset set at once (`setHour`, `setPreset`)
	'set hour': true,
	'set preset': true,
	// the tier's runtime settings applied (`applyQuality`: the clouds' view and march, the cadences)
	'apply quality': true,
	// the canvas changed size (a window resize); a render-scale step is not an event: the histories
	// are carried across it (TemporalAA.ts)
	'resize': true,
	// a view jump (keys R/1/2/3, the GUI's view, `setCameraView`) and a switch of the camera's mode
	'view jump': true,
	'camera mode': true,
	// any control of the GUI (bubbling to its root): it retunes the image in place
	'gui': false,
	// a look's parameter set from a script (`setGrade`: the grade's table filled anew; `setLawnGain`):
	// the same, without the GUI
	'retune': false,
	// the app's keyboard toggles (Space: pause, H: the GUI)
	'key toggle': false,
	// `__PONLAI__.taa.restart()`
	'restart': true,
} as const;

// the events after which a still frame draws what it drew before (the loop restarted, the page
// shown again, a key toggled the pause or the GUI): they leave the epoch's `scene` count
const SAME_SCENE: ReadonlySet<string> = new Set<keyof typeof RENDER_EVENTS>( [ 'loop start', 'visible', 'key toggle' ] );

/** The last event raised (the diagnostics' `taa.lastEvent`), and how many. */
export const sceneEvents = {
	last: null as { reason: string; cut: boolean; frame: number } | null,
	count: 0,
};

const cutListeners: ( ( reason: string ) => void )[] = [];
// a disposed scene's listeners and events go with it, and the rest's epoch starts over
onSceneDispose( 'scene events', () => {

	cutListeners.length = 0;
	sceneEvents.last = null;
	sceneEvents.count = 0;
	Object.assign( restEpoch, { value: 0, scene: 0, reason: null } );

} );

/** Called with the reason whenever an event that cuts is raised, in every anti-aliasing mode. */
export function onSceneCut( listener: ( reason: string ) => void ): void {

	cutListeners.push( listener );

}

// A change of the scene the temporal modes must know of: bumps the rest's epoch, and cuts the
// history where the reason's declaration says so.
function raise( reason: string ): void {

	restEpoch.bump( reason, ! SAME_SCENE.has( reason ) );
	const cut = declared.get( reason ) === true;
	if ( cut ) for ( const listener of cutListeners ) listener( reason );
	if ( ! TEMPORAL ) return;
	if ( cut ) taaCut( reason );
	sceneEvents.last = { reason, cut, frame: motionState.frame };
	sceneEvents.count ++;

}

/** Raises one of the render pipeline's own reasons. */
export const renderEvent = declareSceneEvents( RENDER_EVENTS );
