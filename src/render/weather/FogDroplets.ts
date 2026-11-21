// The fog layer's droplets: their phase toward the sun and the parameters every keyframe shares.
//
// The haze's fog layer (VolumetricHaze.ts, whose header describes it, The fog layer) lights its
// droplets through `dropletPhase`, two Henyey-Greenstein lobes summed over Wrenninge's octaves of
// multiple scattering (Wrenninge, "Oz: The Great and Volumetric", SIGGRAPH 2013 talk). `DROPLET` holds the lobes and Wrenninge's b and c, the same in every key and
// tunable on debug pages (`haze.fog.*`; a change rebakes the sky).

import * as THREE from 'three/webgpu';
import { Fn, exp, float, max, mix } from 'three/tsl';
import { tune } from '../debug/Tune.ts';
import { markSkyRetune } from '../sky/SkyRetune.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

// The fog layer's droplets' phase toward the sun with their multiple scattering (see the header,
// The fog layer): `nu` the cosine of the scattering angle, `tau` the fog layer's optical depth
// toward the sun, `lobes` the two Henyey-Greenstein lobes ( forward g, backward g, backward
// share ), `octaves` Wrenninge's ( a, b, c ). DROPLET_OCTAVES terms, the first the single scatter.
const DROPLET_OCTAVES = 4;
export const dropletPhase = Fn( ( [ nu, tau, lobes, octaves ]: [ F, F, V3, V3 ] ) => {

	const hg = ( g: F ) => {

		const g2 = g.mul( g );
		return float( 1 ).sub( g2 ).div( max( float( 1 ).add( g2 ).sub( g.mul( nu ).mul( 2 ) ), 1e-4 ).pow( 1.5 ).mul( 4 * Math.PI ) );

	};
	let sum: F = float( 0 ), a: F = float( 1 ), b: F = float( 1 ), c: F = float( 1 );
	for ( let k = 0; k < DROPLET_OCTAVES; k ++ ) {

		const phase = mix( hg( lobes.x.mul( c ) ), hg( lobes.y.mul( c ) ), lobes.z );
		sum = sum.add( a.mul( phase ).mul( exp( tau.mul( b ).negate() ) ) );
		a = a.mul( octaves.x );
		b = b.mul( octaves.y );
		c = c.mul( octaves.z );

	}
	return sum;

} ).setLayout( { name: 'hazeDropletPhase', type: 'float', inputs: [
	{ name: 'nu', type: 'float' }, { name: 'tau', type: 'float' }, { name: 'lobes', type: 'vec3' }, { name: 'octaves', type: 'vec3' },
] } );

// The fog layer's parameters shared by every keyframe (see the header, The fog layer): the droplets' lobes and
// Wrenninge's b and c.
export const DROPLET = {
	forward: tune( 'haze.fog.forward', 0.8, { min: 0.6, max: 0.9, onChange: markSkyRetune } ),
	backward: tune( 'haze.fog.backward', - 0.3, { min: - 0.5, max: - 0.2, onChange: markSkyRetune } ),
	backShare: tune( 'haze.fog.backShare', 0.3, { min: 0.2, max: 0.5, onChange: markSkyRetune } ),
	octaveExtinction: tune( 'haze.fog.octaveExtinction', 0.5, { min: 0.3, max: 0.8, onChange: markSkyRetune } ),
	octaveAnisotropy: tune( 'haze.fog.octaveAnisotropy', 0.5, { min: 0.3, max: 0.8, onChange: markSkyRetune } ),
};
