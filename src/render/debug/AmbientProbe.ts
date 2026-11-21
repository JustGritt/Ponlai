import * as THREE from 'three/webgpu';
import { Fn, If, instanceIndex, instancedArray, uint, vec3, vec4 } from 'three/tsl';
import { ambientField, ambientSkyVisibility } from '../lighting/AmbientSample.ts';
import type { Bounds } from '../RenderTypes.ts';
import { releaseAttribute } from '../three/Attributes.ts';

// `__PONLAI__.ambientProbe`: the ambient light's irradiance on open level ground, from the probes.
//
// `__PONLAI__.ambientProbe( options )`: the ambient light's irradiance on open level ground, read
// from the probes' volume itself rather than from screen cells (E_up for a script that reads cells: the island's level
// ground is mostly meadow, so a frame holds few bare open cells and their mean was noise). Points on
// a world-anchored lattice over the land's box, on terrain within `slope` degrees of level, `lift` m
// along the terrain's normal (0.5, the normal offset the island's materials read the volume at:
// AmbientSample.ts `ambientPoint`, whose quarter metre toward the eye follows the view and is left
// out), so the white card takes the light our ground takes; at each, the volume's irradiance facing
// up (AmbientSample.ts `ambientField`, the front face's, in the scene's units as three's
// `iblIrradiance`) and its sky visibility. The open ones are
// those whose visibility reaches the `open` quantile of the points'; `eUp` is their mean
// irradiance's luminance. Evaluated in a compute kernel built on the call (none exists in an app that
// never calls it, so the app's pipelines are unchanged). The volume is relit every frame, so on a
// paused page the read is the frame's.

const DEFAULTS = { step: 2, lift: 0.5, slope: 15, open: 0.9 };

interface AmbientProbeResult {
	/** the open points' mean irradiance facing up, its luminance and channels */
	eUp: number;
	rgb: [ number, number, number ];
	/** the sky visibility the open points reach at least, and the points' counts */
	visibility: number;
	open: number;
	points: number;
}

export async function ambientProbe( context: { renderer: THREE.WebGPURenderer; heightAt: ( x: number, z: number ) => number; land: Bounds }, options: Partial<typeof DEFAULTS> = {} ): Promise<AmbientProbeResult | null> {

	const { step, lift, slope, open } = { ...DEFAULTS, ...options };
	const { renderer, heightAt, land } = context;
	const cosMax = Math.cos( slope * Math.PI / 180 ), d = 0.5;
	const points: number[] = [];
	for ( let z = Math.ceil( land.z0 / step ) * step; z <= land.z1; z += step ) for ( let x = Math.ceil( land.x0 / step ) * step; x <= land.x1; x += step ) {

		const y = heightAt( x, z );
		if ( ! ( y > 0 ) ) continue;
		const n = new THREE.Vector3( heightAt( x - d, z ) - heightAt( x + d, z ), 2 * d, heightAt( x, z - d ) - heightAt( x, z + d ) ).normalize();
		if ( n.y >= cosMax ) points.push( x + n.x * lift, y + n.y * lift, z + n.z * lift, 0 );

	}
	const count = points.length / 4;
	if ( ! count ) return null;
	const input = instancedArray( new Float32Array( points ), 'vec4' ), out = instancedArray( count, 'vec4' );
	const kernel = Fn( () => {

		If( instanceIndex.lessThan( uint( count ) ), () => {

			const P = input.element( instanceIndex ).xyz.toVar();
			const E = ambientField( P, 2 ).front( vec3( 0, 1, 0 ) ).toVar();
			out.element( instanceIndex ).assign( vec4( E, ambientSkyVisibility( P ) ) );

		} );

	} )().compute( count, [ 64 ] );
	await renderer.computeAsync( kernel );
	const read = new Float32Array( ( await renderer.getArrayBufferAsync( out.value as THREE.StorageBufferAttribute ) ).slice( 0 ) );
	kernel.dispose();
	// the two storage buffers' GPU memory: r186's BufferAttribute.dispose() reaches no listener for
	// a storage attribute, so they are deleted from the renderer's attribute records, which destroys
	// the buffers (three/Attributes.ts)
	for ( const a of [ input.value, out.value ] as THREE.BufferAttribute[] ) releaseAttribute( renderer, a );
	const vis = Array.from( { length: count }, ( _, i ) => read[ i * 4 + 3 ] ).sort( ( a, b ) => a - b );
	const threshold = vis[ Math.min( count - 1, Math.floor( open * ( count - 1 ) ) ) ];
	const sum = [ 0, 0, 0 ];
	let n = 0;
	for ( let i = 0; i < count; i ++ ) {

		if ( read[ i * 4 + 3 ] < threshold ) continue;
		for ( let c = 0; c < 3; c ++ ) sum[ c ] += read[ i * 4 + c ];
		n ++;

	}
	const rgb = sum.map( ( s ) => s / n ) as [ number, number, number ];
	return { eUp: 0.2126 * rgb[ 0 ] + 0.7152 * rgb[ 1 ] + 0.0722 * rgb[ 2 ], rgb, visibility: threshold, open: n, points: count };

}
