import { defineConfig } from 'vite';

// The render pipeline starts its workers as ES modules. The tools write their captures and browser
// profiles under artifacts/, which the dev server must not watch (Chrome keeps its files locked).
export default defineConfig( {
  worker: { format: 'es' },
  server: { watch: { ignored: [ '**/artifacts/**' ] } },
} );
