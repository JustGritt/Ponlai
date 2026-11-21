// node:assert/strict with a count of the assertions made, so a check's summary says how many it
// ran: a check that silently runs fewer (a table skipped, a file not found once it moved) shows in
// the count, which a plain "passed" hides.
//   import { assert, assertions } from './assert-count.mjs';
//   ...
//   console.log( `... passed (${ assertions() } assertions)` );
// Every function of node:assert/strict counts one call (assert( x ) itself too); its classes
// (AssertionError) pass through as they are.
import strict from 'node:assert/strict';

let count = 0;

export const assert = new Proxy( strict, {

	apply( target, self, args ) {

		count ++;
		return Reflect.apply( target, self, args );

	},

	get( target, key ) {

		const value = target[ key ];
		if ( typeof value !== 'function' || ! /^[a-z]/.test( String( key ) ) ) return value;
		return ( ...args ) => {

			count ++;
			return value.apply( target, args );

		};

	},

} );

/** How many assertions have been made so far. */
export const assertions = () => count;
