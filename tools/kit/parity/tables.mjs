// The project's views and clips tables, loaded from where the configuration names them
// (tools/project.mjs `tables`), so the capture-and-compare tools beside this file name no view, clip
// or table of their own. Each table is imported once: a view that adhoc.mjs registers into VIEWS is
// seen by every reader of the table, the table's own module included.
//
//   import { viewTable, clipTable } from './tables.mjs';
//   const { VIEWS, viewEval } = viewTable;
//
// What a views table exports:
//   VIEWS                  { name: { q, eval?, crops? } }, a plain object the tools add ad-hoc views
//                          to. `q`: the view's query, added after the set's own; `eval`: page code (a
//                          string, compared as text) that sets the view up; `crops`: the names of the
//                          view's crop anchors.
//   viewNames( group? )    the names of a group of views. Without a group: the default set, what a run
//                          without --views takes. `special`: the views a run leaves out unless they
//                          are named or the group is asked for.
//   expandViews( spec )    a --views value (names and groups, comma-separated) as view names.
//   viewEval( name, crops? )  the page code a capture evaluates once the page's handle
//                          (`page.handle`) exists: the view's eval, then the crop anchors `crops`
//                          (default the view's own), which the page reports through the `page.crops`
//                          global as { anchor: { x, y, w, h, anchor, world, distance } | { missing } }.
//   ANCHOR_NAMES           every crop anchor's name.
// Run as `node <table> list [group]`, it prints a group's names, one per line (tools/kit/gates.py).
//
// What a clips table exports:
//   CLIPS                  { name: { view, clock, frames, path, q?, crops?, scale?, lead?, gtDense? } }:
//                          the clip's view (a views table's name), 'paused' or 'running', its frame
//                          count, its camera path, extra query, crop anchors other than the view's, a
//                          render scale schedule, lead frames and dense ground-truth runs. A clip named
//                          `rest` is one paused frame of any view at its own pose.
//   CLIP_GROUPS            the clip groups' names; clipNames( group? ) the clips of one (default all).
//   gtSampling( clip, frames? )  the ground truth's frames of a clip: { every, dense: [ [ a, b ] ] }.
//   clipEval( name, { frames?, dt?, setup? } )  page code, run after the view's eval, that sets the
//                          `page.clip` global to { name, frames, lead, clock, dt, scale, camera( i ) }
//                          (`scale`: absolute render ratios, one per frame, or null) and `page.clipNotes`
//                          to a list of notes.
//   BASELINE_PLAN          { tier: { rest: [ view ], full: [ view ], clips: [ [ clip, view | null ] ] } }:
//                          baseline.mjs's entries per quality tier (rest images, those also taken whole,
//                          clips on their own view or another).
//   CARRY_SPOT             { clips: [ clip ], gtViews: [ view ] }: carry.mjs's spot checks.
// Run as `node <table> list [group]`, it prints a group's names, one per line.
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import PROJECT from '../../project.mjs';
import { ROOT } from '../root.mjs';

// a table the configuration names, or, for a project without it, a stand-in whose every read fails
// with that message (so a tool that needs only the other table still runs)
const load = async ( kind ) => {

	const file = PROJECT.tables?.[ kind ];
	if ( file ) return import( pathToFileURL( path.join( ROOT, file ) ).href );
	return new Proxy( {}, { get: ( _, key ) => {

		if ( key === 'then' ) return undefined; // not a thenable
		throw new Error( `this project has no ${ kind } table (tools/project.mjs: tables.${ kind } is null)` );

	} } );

};

export const viewTable = await load( 'views' );
export const clipTable = await load( 'clips' );
