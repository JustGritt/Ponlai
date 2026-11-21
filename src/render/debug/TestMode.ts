// The test mode's one switch, read once at load, which history-dependent effects obey.
//
// The test mode's one switch (docs/reference/test-mode.md), read once at load: on with `paused` in the
// query, which every capture, parity and verify tool loads its pages with, unless `testMode=0`
// leaves it out (the clips do: they step the clock and show what moves). Every history-dependent
// rendering effect that does not settle on its own on a still scene reads it, and is frozen or off
// under it: the falling leaves are not made (world/FallingLeaves.ts).

import { settings } from '../Settings.ts';

const query = settings;

export const TEST_MODE = query.has( 'paused' ) && query.get( 'testMode' ) !== '0';
