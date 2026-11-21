import type { QualityProfile } from './Quality.ts';

// What every registered definition declares: its cost setting, test-mode behaviour and growth.
//
// What every registered definition declares. The render pipeline is extended at named points, each
// a registry its owner reads in the order registered and never by name: a kind of plant
// (vegetation/Registry.ts), a kind of local light (lighting/LocalLights.ts), a pass of the frame
// (post/ExtraPass.ts). A definition at any of them states three things. New effects forgot them
// more than once, and the omission showed only later (a nondeterministic capture, a frame time
// that crept up, a grown page that differed from a fresh load): the quality setting that governs
// its cost, its test-mode behaviour, and how it follows the land's growth. They are required
// fields, so leaving one out is a type error, and a registry refuses a definition without them
// (`checkDeclarations`: a definition from plain JavaScript, or one cast past the types).
//
// The declarations are statements, not hooks: the registries do not act on them. They say where
// to look: the setting to change in a cost test, the switch the test mode uses to hold the effect,
// and the follower that carries it through a growth. tools/render/verify-extensions.mjs registers
// definitions of its own at every point and checks that one without them is refused.
//
// This module imports types only, so the CPU-only tools can load it under plain node.

/** The quality setting that governs a definition's cost: a field of the quality table (Quality.ts,
 * which is closed: a new setting is a new field there), or 'none' when no tier changes it. */
export type CostSetting = keyof QualityProfile | 'none';

/** A definition's test-mode behaviour (debug/TestMode.ts: a paused page whose frame is
 * bit-identical from frame to frame): 'stateless' when what it draws is a function of the scene alone, with no history;
 * otherwise whether the test mode freezes it or turns it off, by what (`by`: the switch or the
 * site), and when it is settled (`settled`: what the page's settled signal waits for, or that it
 * waits for nothing). */
export type TestModeBehaviour = 'stateless' | { held: 'frozen' | 'off'; by: string; settled: string };

/** How a definition follows the land's growth: 'independent' when it does not depend on the land,
 * otherwise the follower that carries it through a growth (`follower`: what prepares its next
 * version and swaps it in, in the swap's frame). */
export type LandDependence = 'independent' | { follower: string };

/** The three declarations of a registered definition. */
export interface Declarations {
	cost: CostSetting;
	testMode: TestModeBehaviour;
	land: LandDependence;
}

const text = ( value: unknown ): value is string => typeof value === 'string' && value !== '';

/** Refuses a definition that lacks one of the three declarations: throws, naming the registry
 * (`point`), the definition and what is missing. */
export function checkDeclarations( point: string, name: string, definition: Partial<Declarations> ): void {

	const missing: string[] = [];
	if ( ! text( definition.cost ) ) missing.push( 'cost (a quality setting, or \'none\')' );
	const t = definition.testMode;
	if ( t !== 'stateless' && ! ( typeof t === 'object' && t !== null && ( t.held === 'frozen' || t.held === 'off' ) && text( t.by ) && text( t.settled ) ) ) missing.push( 'testMode (\'stateless\', or { held, by, settled })' );
	const l = definition.land;
	if ( l !== 'independent' && ! ( typeof l === 'object' && l !== null && text( l.follower ) ) ) missing.push( 'land (\'independent\', or { follower })' );
	if ( missing.length ) throw new Error( `${ point }: '${ name }' is refused, it does not declare its ${ missing.join( ', its ' ) }` );

}
