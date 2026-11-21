import { TEMPORAL } from './Antialiasing.ts';
import { onTaaCut } from './Motion.ts';

// The scene's histories as one table: whatever keeps state registers how to clear it.
//
// The scene's histories, as one table of the render pipeline's: whatever keeps state from the
// frames or the scenes before registers how to clear it, and the two events that clear walk the
// table instead of naming each owner. A new effect with a history joins by one registration beside
// its creation; an owner the app does not have is not in the table.
// - A cut of the temporal modes' history (Motion.ts `taaCut`: a seek, a jump of the camera, a new
//   scene): `onCut` entries, whose previous state becomes the current one (the wind's, the
//   displacement field's). They are registered where their owners are made, since a cut can come
//   as soon as frames render. The table listens to the motion's cuts once, from its first entry,
//   under the temporal modes only: its place among the cut's other listeners (the pipeline's, the
//   resolve's) is that first entry's.
// - A scene jump (`establishScene`: the app moves to another scene as a fresh load of its query
//   holds it): `onJump` entries, which clear what the scenes before left behind. Most run before
//   the jump's bake of the sky; an entry that needs the light the bake publishes runs after it
//   (`'after bake'`: the refracted sun's map).
// Each list runs in the order registered, which is part of the contract: the entries are
// registered in the order the app cleared them when it named each one. The scene events
// (SceneEvents.ts) are the other half of a jump: its 'establish scene' cuts the histories that
// listen to the motion's cuts themselves.

type Entry = { name: string; clear: () => void };

export class SceneHistory {

	private readonly cuts: Entry[] = [];
	private readonly jumps = { 'before bake': [] as Entry[], 'after bake': [] as Entry[] };

	/** Registers what a cut of the temporal modes' history clears (`name` for the debug list). */
	onCut( name: string, clear: () => void ): void {

		if ( this.cuts.length === 0 && TEMPORAL ) onTaaCut( () => { for ( const entry of this.cuts ) entry.clear(); } );
		this.cuts.push( { name, clear } );

	}

	/** Registers what a scene jump clears, before the jump's bake of the sky or after it. */
	onJump( name: string, clear: () => void, when: 'before bake' | 'after bake' = 'before bake' ): void {

		this.jumps[ when ].push( { name, clear } );

	}

	/** A scene jump: clears every `onJump` entry of `when`, in the order registered. */
	jump( when: 'before bake' | 'after bake' ): void {

		for ( const entry of this.jumps[ when ] ) entry.clear();

	}

	/** The entries' names, in order (debug, and tools/render/verify-extensions.mjs). */
	names(): { cut: string[]; jump: string[]; afterBake: string[] } {

		const of = ( list: Entry[] ) => list.map( ( entry ) => entry.name );
		return { cut: of( this.cuts ), jump: of( this.jumps[ 'before bake' ] ), afterBake: of( this.jumps[ 'after bake' ] ) };

	}

}
