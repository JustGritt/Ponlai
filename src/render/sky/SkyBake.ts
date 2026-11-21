/*!
 * Portions adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { uv, texture, vec4, float, mix, pmremTexture } from 'three/tsl';
import { SCENE_RADIANCE_SCALE } from '../lighting/Radiometry.ts';
import { storePress } from '../post/PreExposure.ts';
import type { WeatherState } from './Weather.ts';
import { environmentSteps } from '../three/Environment.ts';
import { luma } from '../look/SkyLook.ts';
import type { Sky, SkyEndpoint } from './Sky.ts';

// The sky's bakes: reflection panorama, environment and irradiance, spread over frames.
//
// The sky's bakes (Sky.ts, whose header says what they hold), each a function of the sky; a name in
// backticks with no file is this file's or Sky.ts's.
//
// Only the reflection panorama, environment and irradiance cross-fade between bakes. A periodic
// bake on a running clock spreads over frames: its panoramas and irradiance in the frame that asks
// for it, then the environment's PMREM conversion and GGX filters at most four render passes a
// frame, published (with the same products as a whole bake) a few frames later. Explicit bakes
// (presets, seeks, weather jumps, the first bake) and any bake while the scene clock is paused run
// whole, so paused frames stay exact. A paused page keeps its last explicit bake's products, and
// every explicit bake (the load's, a scene jump's, a seek's, a retune's) starts from the state
// before a fresh load's first bake, the clouds set to the scene clock (`reset`), so each makes the ones
// a fresh load of its scene makes. A bake waits for the warm-up's
// pipelines (`warm`) before it draws: three r186 skips a draw whose pipeline an asynchronous
// compile is still creating, so a bake drawn meanwhile left its products empty (the irradiance
// probe at 0: no sky light, and the exposure about five stops off at noon), and a paused page,
// which bakes no more, kept them. The spread drives three's `PMREMGenerator` private steps
// (three/Environment.ts, which throws on a three without them).

// what a panorama that lends the land no light takes for its tint (`bakeLandTint`)
const NEUTRAL_TINT = new THREE.Vector4( 1, 1, 1, 0 );

// One bake: its endpoint, whether it is explicit (reads its irradiance back and shows at once),
// and the environment atlas it refreshes (null: it keeps the one shown).
interface Bake {
	version: number;
	readback: boolean;
	transitionSeconds: number;
	next: number;
	state: SkyEndpoint;
	environment: number | null;
}

// A step of a periodic bake spread over frames; `cost` counts its render passes.
export interface Slice {
	cost: number;
	run: () => void;
}

// A periodic bake whose frames stop coming (a stopped loop, a hidden page) completes after this
// long, so it never holds the sky busy.
const SLICE_TIMEOUT_MS = 250;

export async function bake( sky: Sky, transitionSeconds: number ): Promise<void> {

	// eslint-disable-next-line ponlai/scene-clock -- the bake's time, a readout
	const started = performance.now();
	const version = ++ sky.bakeVersion;
	const readback = transitionSeconds === 0 || ! sky.initialized;
	const weather = sky.weatherState ? { ...sky.weatherState } : null;
	await sky.renderer.init();
	// the warm-up's pipelines first: r186 skips a draw whose pipeline is still being created
	// (see the header)
	await sky.warmed;
	if ( ! sky.clouds.ready ) await sky.clouds.init( sky.renderer );
	await sky.night.load();
	if ( ! sky.night.ready ) await sky.night.init();
	// An explicit jump (preset, seek, weather change) refreshes the whole cloud map now.
	if ( readback ) {

		sky.clouds.invalidate();
		sky.clouds.update( sky.renderer, sky.viewCamera, sky.viewWidth, sky.viewHeight, sky.cloudTime );

	}
	const bake = beginBake( sky, version, readback, transitionSeconds, weather );
	renderPanoramas( sky, bake );
	const r = sky.renderer;
	if ( readback ) {

		const W = sky.small.width, H = sky.small.height;
		const px = await r.readRenderTargetPixelsAsync( sky.small, 0, 0, W, H ) as Float32Array;
		const E = [ 0, 0, 0 ];
		const dOmegaLon = 2 * Math.PI / W, dTh = Math.PI / H;
		for ( let k = 0; k < W * H; k ++ ) {

			const sinTh = px[ k * 4 + 3 ];
			if ( sinTh <= 0 ) continue;
			const dOmega = dOmegaLon * dTh * Math.sqrt( Math.max( 0, 1 - sinTh * sinTh ) );
			for ( let c = 0; c < 3; c ++ ) E[ c ] += px[ k * 4 + c ] * sinTh * dOmega;

		}
		bake.state.irradiance.setRGB( E[ 0 ] / Math.PI, E[ 1 ] / Math.PI, E[ 2 ] / Math.PI );
		sky.landSea( bake.state, version, await r.readRenderTargetPixelsAsync( bake.state.deckIrradianceMap, 0, 0, 1, 1 ) as Float32Array );

	} else sky.readIrradiance( bake.state, version );
	if ( bake.environment !== null ) filterEnvironment( sky, bake.environment );
	if ( ! sky.initialized ) {

		// The first bake fills both atlases from the same source.
		sky.front = bake.next;
		filterEnvironment( sky, 1 );

	}
	publishBake( sky, bake );
	// eslint-disable-next-line ponlai/scene-clock -- the bake's time, a readout
	sky.lastBakeMs = performance.now() - started;

}

// Pick the endpoint a bake writes and capture the state it bakes: the live state from
// track() (the atmosphere LUT, key light, clouds and stars already describe this frame's sun
// and moon) and the weather it was requested with.
export function beginBake( sky: Sky, version: number, readback: boolean, transitionSeconds: number, weather: WeatherState | null ): Bake {

	const illuminance = sky.sunIlluminance * SCENE_RADIANCE_SCALE;
	const next = sky.initialized ? 1 - sky.front : 0;
	const state = sky.endpoints[ next ];
	state.version = version;
	// the weather's sky light; its day sky (`daySky`, from track) is on the sun's light alone: the
	// deck's irradiance takes it through the sky and the clouds, the sun's direct light on the
	// level water here (`sea.sun`), while the body's colour, its knee and the night's share keep
	// the weather's gain alone, so below the knee the body follows the day sky and at night it is
	// as it was
	const ambientGain = weather?.ambientGain ?? 1;
	sky.ambientGain.value = ambientGain;
	// the distant sea's body (`look.seaKnee`): the body is written as it stood before it followed the
	// light, the colour times the illuminance times the gain, so a body at the knee is that value
	const sea = state.sea;
	sea.body.copy( sky.seaColor ).multiplyScalar( illuminance ).multiplyScalar( ambientGain );
	sea.night.copy( sky.displayNightAmbient.value ).multiplyScalar( ambientGain );
	sea.sun = luma( sky.sunColor ) * Math.max( sky.sunDir.y, 0 ) / Math.PI * ambientGain * sky.daySky;
	sea.knee = sky.look.seaKnee() * illuminance * ambientGain;
	sky.seaBody.value.copy( sea.body );
	sky.seaNight.value.copy( sea.night );
	sky.seaSun.value = sea.sun;
	sky.seaKnee.value = sea.knee;
	// The environment refreshes on explicit bakes and on every environmentEvery-th periodic one,
	// into the atlas not shown.
	let environment: number | null = 0;
	if ( sky.initialized ) {

		const refresh = readback || ++ sky.bakesSinceEnvironment >= sky.environmentEvery;
		environment = refresh ? 1 - sky.environmentFront : null;

	}
	if ( environment !== null ) sky.bakesSinceEnvironment = 0;
	return { version, readback, transitionSeconds, next, state, environment };

}

// The cloud layer, the panoramas (seen, the environment's source, irradiance probe) and the
// irradiance integral.
export function renderPanoramas( sky: Sky, bake: Bake ): void {

	const { state } = bake;
	// The bake's powers of two, the layer's and the panoramas' at this moment (trackClouds, see
	// the stores above): the layer is written and read within the bake, the panoramas keep
	// theirs with them.
	const layer = sky.layerStops, stops = sky.panoramaStops;
	sky.layerStore.value = 2 ** layer * storePress.store.value;
	sky.layerRead.value = 2 ** - layer * storePress.read.value;
	state.stops = stops;
	if ( ! sky.initialized ) sky.endpoints[ 1 ].stops = stops;
	sky.sourceStops = stops;
	for ( const atlas of sky.initialized ? bake.environment !== null ? [ bake.environment ] : [] : [ 0, 1 ] ) sky.environmentStops[ atlas ] = stops;
	const r = sky.renderer, prev = r.getRenderTarget();
	try {

		r.setRenderTarget( state.clouds );
		sky.upperQuad.render( r );
		sky.cloudLayer.value = state.clouds.texture;
		// the sky's light without its haze, which the panoramas' haze takes under a deck
		// (VolumetricHaze.ts `openSky`)
		for ( const deck of sky.initialized ? [ state.deckIrradianceMap ] : [ state.deckIrradianceMap, sky.endpoints[ 1 ].deckIrradianceMap ] ) {

			r.setRenderTarget( deck );
			sky.deckIrradianceQuad.render( r );

		}
		sky.bakeDeckIrradiance.value = state.deckIrradianceMap.texture;
		// Both samplers must have distinct textures at the first material compilation:
		// Three deduplicates texture uniforms by UUID, even for separate TextureNodes.
		const panoramas = sky.initialized ? [ state.equirect, sky.reflectionSource, sky.small ] :
			[ state.equirect, sky.endpoints[ 1 ].equirect, sky.reflectionSource, sky.small ];
		for ( const rt of panoramas ) {

			sky.irradiancePass.value = rt === sky.small ? 1 : 0;
			sky.bakeStore.value = rt === sky.small ? 1 : 2 ** stops * storePress.store.value;
			// Only the environment's source takes the island's light after sunset (`look.landSky`).
			const lends = rt === sky.reflectionSource;
			sky.bakeLandTint.value.copy( lends ? sky.landTint.value : NEUTRAL_TINT );
			sky.bakeLandGain.value = lends ? sky.landGain.value : 1;
			sky.bakeHaze.value = lends || rt === sky.small ? 1 : 0;
			sky.bakeGlow.value = lends || rt === sky.small ? 0 : 1;
			sky.bakeKeyFloor.value = lends || rt === sky.small ? 0 : 1;
			r.setRenderTarget( rt );
			( lends ? sky.landQuad : sky.quad ).render( r );

		}
		for ( const irradiance of sky.initialized ? [ state.irradianceMap ] : [ state.irradianceMap, sky.endpoints[ 1 ].irradianceMap ] ) {

			r.setRenderTarget( irradiance );
			sky.irradianceQuad.render( r );

		}
		// (after the probe's panorama, so the bake's uniforms are the probe's)
		for ( const irradiance of sky.initialized ? [ state.clearIrradianceMap ] : [ state.clearIrradianceMap, sky.endpoints[ 1 ].clearIrradianceMap ] ) {

			r.setRenderTarget( irradiance );
			sky.clearIrradianceQuad.render( r );

		}

	} finally {

		r.setRenderTarget( prev );

	}
	sky.onEnvironmentSource?.( sky.reflectionSource.texture, sky.initialized ? bake.environment !== null ? [ bake.environment ] : [] : [ 0, 1 ], 2 ** - stops * storePress.read.value );

}

// The environment of the source in atlas `index`, whole.
export function filterEnvironment( sky: Sky, index: number ): void {

	sky.environments[ index ] = sky.pmrem.fromEquirectangular( sky.reflectionSource.texture, sky.environments[ index ] );

}

// The same as filterEnvironment, as steps of one or two render passes: the conversion to the
// cube atlas, then each level's GGX filter from the one above (PMREMGenerator's own steps:
// three/Environment.ts). Each step restores what PMREMGenerator restores after a whole conversion.
export function environmentSlices( sky: Sky, index: number ): Slice[] {

	const target = sky.environments[ index ];
	if ( ! target ) return [ { cost: 20, run: () => filterEnvironment( sky, index ) } ];
	const pmrem = environmentSteps( sky.pmrem );
	const r = sky.renderer, source = sky.reflectionSource.texture;
	const step = ( work: () => void ) => () => {

		const previous = r.getRenderTarget(), face = r.getActiveCubeFace(), level = r.getActiveMipmapLevel(), autoClear = r.autoClear;
		try {

			work();

		} finally {

			r.autoClear = autoClear;
			r.setRenderTarget( previous, face, level );
			target.scissorTest = false;
			pmrem.resetViewport( target );

		}

	};
	const slices: Slice[] = [ { cost: 1, run: step( () => {

		pmrem.convert( source, target );

	} ) } ];
	// the generator keeps the levels of the source size the first bake set up, so their count is known
	const levels = pmrem.levels();
	for ( let i = 1; i < levels; i ++ ) slices.push( { cost: 2, run: step( () => {

		r.autoClear = false;
		pmrem.filter( target, i - 1, i );

	} ) } );
	return slices;

}

// Start a periodic bake over frames: the panoramas now, the rest from advance().
export function startSlices( sky: Sky, transitionSeconds: number ): void {

	// eslint-disable-next-line ponlai/scene-clock -- the bake's time, a readout
	const started = performance.now();
	const weather = sky.weatherState ? { ...sky.weatherState } : null;
	const bake = beginBake( sky, ++ sky.bakeVersion, false, transitionSeconds, weather );
	renderPanoramas( sky, bake );
	sky.readIrradiance( bake.state, bake.version );
	sky.slices = bake.environment === null ? [] : environmentSlices( sky, bake.environment );
	sky.slices.push( { cost: 0, run: () => publishBake( sky, bake ) } );
	// eslint-disable-next-line ponlai/scene-clock -- the bake's time, a readout
	sky.sliceMs = performance.now() - started;
	// A paused scene does it whole, so a paused frame never depends on how many frames ran.
	if ( sky.clockStill ) finishSlices( sky );
	else armSliceTimer( sky );

}

// Run whole steps of the bake under way, up to `passes` render passes (at least one step); its
// publication ends it. An error ends the bake and rejects its promise, as a whole bake's would,
// so the sky is free for the next one.
export function runSlices( sky: Sky, passes: number ): void {

	// eslint-disable-next-line ponlai/scene-clock -- the bake's time, a readout
	const started = performance.now();
	let spent = 0;
	try {

		while ( sky.slices.length > 0 && ( spent === 0 || spent + sky.slices[ 0 ].cost <= passes ) ) {

			const slice = sky.slices.shift()!;
			spent += slice.cost;
			slice.run();

		}

	} catch ( error ) {

		abortSlices( sky, error );
		return;

	}
	// eslint-disable-next-line ponlai/scene-clock -- the bake's time, a readout
	sky.sliceMs += performance.now() - started;
	if ( sky.slices.length > 0 ) {

		armSliceTimer( sky );
		return;

	}
	if ( sky.sliceTimer !== null ) clearTimeout( sky.sliceTimer );
	sky.sliceTimer = null;
	sky.lastBakeMs = sky.sliceMs;
	sky.busy = false;
	sky.pendingUpdate = null;
	const done = sky.sliceDone;
	sky.sliceDone = null;
	done?.resolve();

}

// End the bake under way unfinished: `error` rejects its promise, none resolves it (dispose).
export function abortSlices( sky: Sky, error?: unknown ): void {

	if ( sky.sliceTimer !== null ) clearTimeout( sky.sliceTimer );
	sky.sliceTimer = null;
	sky.slices = [];
	sky.busy = false;
	sky.pendingUpdate = null;
	const done = sky.sliceDone;
	sky.sliceDone = null;
	if ( error === undefined ) done?.resolve();
	else done?.reject( error );

}

export function finishSlices( sky: Sky ): void {

	runSlices( sky, Infinity );

}

export function armSliceTimer( sky: Sky ): void {

	if ( sky.sliceTimer !== null ) clearTimeout( sky.sliceTimer );
	sky.sliceTimer = setTimeout( () => {

		sky.sliceTimer = null;
		if ( sky.slices.length > 0 ) finishSlices( sky );

	}, SLICE_TIMEOUT_MS );

}

// Show a finished bake: an explicit one at once, a periodic one by cross-fading to it.
export function publishBake( sky: Sky, bake: Bake ): void {

	const { state, transitionSeconds } = bake;
	const r = sky.renderer;
	const previous = sky.endpoints[ sky.front ];
	const displayedNext = sky.initialized ? state : sky.endpoints[ 1 ];
	if ( ! sky.initialized ) {

		// Both endpoints start from this bake's cloud layer.
		const prev = r.getRenderTarget();
		r.setRenderTarget( sky.endpoints[ 1 ].clouds );
		sky.upperQuad.render( r );
		r.setRenderTarget( prev );

	}
	sky.panoramaA.value = previous.equirect.texture;
	sky.panoramaB.value = displayedNext.equirect.texture;
	sky.panoramaReadA.value = 2 ** - previous.stops * storePress.read.value;
	sky.panoramaReadB.value = 2 ** - displayedNext.stops * storePress.read.value;
	sky.irradianceA.value = previous.irradianceMap.texture;
	sky.irradianceB.value = displayedNext.irradianceMap.texture;
	sky.clearIrradianceA.value = previous.clearIrradianceMap.texture;
	sky.clearIrradianceB.value = displayedNext.clearIrradianceMap.texture;
	sky.deckIrradianceA.value = previous.deckIrradianceMap.texture;
	sky.deckIrradianceB.value = displayedNext.deckIrradianceMap.texture;
	// The environment cross-fades from the atlas shown to the one this bake filtered (the
	// first bake filled both); a bake that kept it fades it to itself.
	const shown = sky.environments[ sky.environmentFront ]!;
	const nextEnvironment = ! sky.initialized ? sky.environments[ 1 ]! : bake.environment !== null ? sky.environments[ bake.environment ]! : shown;
	sky.environmentPair = [ sky.environmentFront, ! sky.initialized ? 1 : bake.environment ?? sky.environmentFront ];
	// the displayed environment at the second atlas's power of two (measured at most two stops
	// over the first, in a front between the kept skies through the twilight, and never apart
	// over a jump of the clock from noon to midnight: so a store's brightest, under 2^13, stays
	// in half float through the fade)
	const [ stopsA, stopsB ] = sky.environmentPair.map( ( atlas ) => sky.environmentStops[ atlas ] );
	sky.environmentGainA.value = 2 ** ( stopsB - stopsA );
	sky.environmentReadValue = 2 ** - stopsB * storePress.read.value;
	if ( sky.initialized && bake.environment !== null ) sky.environmentFront = bake.environment;
	if ( ! sky.environmentA || ! sky.environmentB ) {

		sky.environmentA = texture( shown.texture );
		sky.environmentB = texture( nextEnvironment.texture );
		sky.envMap = shown.clone();
		sky.envMap.texture.mapping = THREE.CubeUVReflectionMapping;
		( sky.envMap.texture as THREE.Texture & { isPMREMTexture: boolean } ).isPMREMTexture = true;
		sky.envMap.texture.name = 'skyDisplayedPMREM';
		sky.envMap.scissorTest = false;
		const blendMat = new THREE.NodeMaterial();
		blendMat.depthTest = false;
		blendMat.depthWrite = false;
		blendMat.toneMapped = false;
		// Convolution is linear. Blending the already filtered atlases once keeps terrain
		// within WebGPU's portable 16-texture limit instead of doubling every IBL sample. (The
		// first atlas taken to the second's power of two first: both stored as powers of two,
		// the gain loses no bits.)
		const a = sky.environmentA.sample( uv() ).level( float( 0 ) ), b = sky.environmentB.sample( uv() ).level( float( 0 ) );
		blendMat.fragmentNode = vec4( mix( a.rgb.mul( sky.environmentGainA ), b.rgb, sky.blend ), mix( a.a, b.a, sky.blend ) );
		sky.environmentBlendQuad = new THREE.QuadMesh( blendMat );
		sky.environment = pmremTexture( sky.envMap.texture );

	} else {

		sky.environmentA.value = shown.texture;
		sky.environmentB.value = nextEnvironment.texture;

	}
	sky.destination = bake.next;
	sky.transitionElapsed = 0;
	sky.transitionDuration = sky.initialized ? transitionSeconds : 0;
	sky.blend.value = ! sky.initialized || sky.transitionDuration > 0 ? 0 : 1;
	sky.equirect = state.equirect;
	sky.initialized = true;
	sky.publish( sky.blend.value );
	if ( sky.transitionDuration === 0 ) sky.front = sky.destination;
	sky.onEnvironmentStore?.();

}
