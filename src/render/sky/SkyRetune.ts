// Sky retuning: a tunable that changes the sky or the air marks the environment for a rebake.
//
// Tunables that change the sky or the air (debug/Tune.ts) mark it here through their `onChange`:
// the frame reads most of them at once, but the environment's bakes (the reflection, the ambient
// light and the irradiance the island takes from the sky) hold what the sky looked like when they
// ran, so RenderStages.ts's `retuneSky` rebakes the environment after any value set marks it, as an
// explicit bake from the state before a fresh load's first (Sky.ts `reset`), so the page equals a fresh
// load with that value. The `tune` handle's `set` runs it at once; the frame loop runs it before
// each frame for a value set any other way (the GUI, `tune.reset`, a script writing a settings
// field). The haze's settings that the bake lays over its source are found by comparison instead
// (VolumetricHaze.ts `bakeStale`), whatever sets them. Every explicit bake takes the values as they
// stand and clears the mark. A bake in flight holds the settled signal, so a tool that waits for it
// reads the rebaked frame.

export const skyRetune = { dirty: false };

/** An `onChange` for tunables the environment's bakes depend on. */
export const markSkyRetune = (): void => {

	skyRetune.dirty = true;

};
