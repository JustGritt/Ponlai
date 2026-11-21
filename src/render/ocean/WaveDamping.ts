import * as THREE from 'three/webgpu';
import { texture, float, vec4, max, mix, smoothstep } from 'three/tsl';
import type { OceanFFT } from './OceanFFT.ts';
import type { ShoreWaves } from './ShoreWaves.ts';
import type { Shared } from '../shared.ts';

// How much of each FFT cascade a point of the sea keeps: depth, the lee and the rock.
//
// How much of each FFT cascade a point of the sea keeps: tidewater's shallow-water attenuation by
// the floor's depth under it (OceanFFT.attenuation), the island's lee (the shelter field: every
// band, see LEE) and, for the two short cascades, the last 7 m before the rock.
// The two long cascades (waves over 5.5 m) keep LONG_KEEP of their height over a deeper floor on
// the shelf, where tidewater's attenuation would leave them a tenth: tidewater hands its shallows
// to beach waves of their own, while ours keep the sea's own waves, which reach the rock and break
// on it (ShoreWaves.ts). On the shelf (about 2 m deep) the sea's long waves, 5 to 25 m, slow by a
// quarter at most and bend little, so they keep their open-sea course.
// The sea's surface (Water.ts), the long waves the light under it follows (WaveMaps.ts) and the
// waves that break on the rock (ShoreWaves.ts) all take these weights, so all see one sea.

type V2 = THREE.Node<'vec2'>;

// What the island's lee leaves of each band of waves at its deepest (the shelter field's R at 0).
// The sea's waves are one field over the whole sea that knows nothing of the island: whatever the
// lee leaves of them runs on behind it in their own direction, as if through the rock, and the
// rock's foam follows them. An island casts a wave shadow: the short wind waves (1 to 5 m)
// and the ripples are cut off, as waves much shorter than the island are, and a cove or the water
// behind the island lies nearly smooth, with only a low swell bending in. So the lee leaves the
// short waves a tenth, the ripples under a third and the long waves an eighth. The slopes no pixel
// resolves are damped with them in the glints, which narrow there (Water.ts `surface`); the sky's
// reflection keeps the open sea's spread, so the lee stands level with the sea round it.
export const LEE = { long: 0.12, medium: 0.08, fine: 0.3 };

// the share of a long wave's height over LONG_KEEP_DEPTH that it keeps on the shelf, at least
const LONG_KEEP = 0.75;
// below the open sea's floor anywhere (Bathymetry.ts openSeaHeight: 9.35 m, give or take 1.2), so
// the open sea's waves stay as tidewater's attenuation leaves them
const LONG_KEEP_DEPTH = 8;

export interface WaveDampingInputs {
	fft: OceanFFT;
	shared: Shared;
	shore: THREE.Texture | THREE.TextureNode; // the shore's sea-level distance (world fields)
	seabedHeight: THREE.Texture | THREE.TextureNode; // the rendered floor's height (world fields)
	shoreWaves: ShoreWaves;
}

/** The cascades' weights at a world xz (explicit levels: any stage, any branch), and the floor's
 * depth there under the mean sea. */
export function waveDamping( { fft, shared, shore, seabedHeight, shoreWaves }: WaveDampingInputs, xz: V2 ): { weights: THREE.Node<'vec4'>; seaDepth: THREE.Node<'float'> } {

	const fieldUV = ( p: V2 ) => p.sub( shared.shoreXform.xy ).mul( shared.shoreXform.zw );
	const sd = texture( shore, fieldUV( xz ) ).level( float( 0 ) ).r;
	// Shelter changes the physical wave amplitude; the finest wind detail stays alive.
	const shelter = shoreWaves.shelter.sample( fieldUV( xz ) ).level( float( 0 ) );
	const coastDamp = mix( float( 0.45 ), float( 1 ), smoothstep( 0, 7, sd ) );
	const longDamp = mix( float( LEE.long ), float( 1 ), shelter.r );
	const mediumDamp = mix( float( LEE.medium ), float( 1 ), shelter.r );
	const fineDamp = mix( float( LEE.fine ), float( 1 ), shelter.r );
	const seaDepth = shared.waterLevel.sub( texture( seabedHeight, fieldUV( xz ) ).level( float( 0 ) ).r );
	const depthDamp = ( c: number ) => c <= 1 ? max( fft.attenuation( c, seaDepth ), fft.attenuation( c, float( LONG_KEEP_DEPTH ) ).mul( LONG_KEEP ) ) : fft.attenuation( c, seaDepth );
	const damping = [ longDamp, longDamp, mediumDamp, fineDamp ].map( ( lee, c ) => depthDamp( c ).mul( lee ).mul( c <= 1 ? float( 1 ) : coastDamp ) );
	return { weights: vec4( damping[ 0 ], damping[ 1 ], damping[ 2 ], damping[ 3 ] ), seaDepth };

}
