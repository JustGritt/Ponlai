// The sea's surface height and the water's velocity at given points, read back from the GPU.
//
// The ocean's waves exist only on the GPU (render/ocean/OceanFFT.ts: four FFT cascades evolved each
// frame), so a floating body cannot ask the CPU where the surface is. This compute kernel samples
// the same fields the water's vertex stage displaces its grid by (render/ocean/Water.ts): the four
// cascades' displacement, each damped as the water damps it (WaveDamping.ts: the floor's depth, the
// lee and the rock), plus the shore's breaking waves. The FFT's displacement is choppy (it moves the
// surface sideways as well as up), so the undisplaced point whose displaced position lies over the
// query is found first by a short fixed-point iteration, then the height is read there.
//
// One batch is in flight at a time: `request` writes the points and dispatches the kernel, the
// result comes back a frame or two later (`onResult`). At wave periods of seconds that lag is a
// few centimetres of surface motion.
import * as THREE from 'three/webgpu';
import { Fn, If, float, instanceIndex, instancedArray, int, texture, uniform, vec2, vec4 } from 'three/tsl';
import { shared, type RenderParts } from './render/index.ts';
import { waveDamping } from './render/ocean/WaveDamping.ts';

type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

export interface WaveSample {
  /** the surface's height (m) */
  height: number;
  /** the water's orbital velocity (m/s) */
  vx: number; vy: number; vz: number;
}

export class WaveProbe {

  readonly capacity: number;
  /** the batch number of the latest result, and how many batches were sent */
  received = 0;
  sent = 0;
  private readonly input: THREE.StorageBufferNode<'vec4'>;
  private readonly output: THREE.StorageBufferNode<'vec4'>;
  private readonly count = uniform( 0, 'uint' );
  private readonly kernel: THREE.ComputeNode;
  private readonly renderer: THREE.WebGPURenderer;
  private inFlight = false;
  /** the latest result: ( height, vx, vy, vz ) per point */
  readonly result: Float32Array;

  constructor( render: RenderParts, fields: { shore: THREE.TextureNode; floor: THREE.TextureNode }, capacity = 1024 ) {

    this.capacity = capacity;
    this.renderer = render.renderer;
    this.result = new Float32Array( capacity * 4 );
    this.input = instancedArray( new Float32Array( capacity * 4 ), 'vec4' );
    this.output = instancedArray( new Float32Array( capacity * 4 ), 'vec4' );
    const { fft, shoreWaves } = render;
    const inputs = { fft, shared, shore: fields.shore, seabedHeight: fields.floor, shoreWaves };
    const cascade = ( field: THREE.Texture, c: number, at: V2 ) => texture( field, at.div( fft.sizes[ c ] ) ).depth( int( c ) ).level( float( 0 ) ).xyz as V3;
    // the damped displacement ( Dx, Dy, Dz ) of the undisplaced point `at`
    const displacement = ( at: V2 ) => {

      const w = waveDamping( inputs, at ).weights;
      return cascade( fft.displacement, 0, at ).mul( w.x ).add( cascade( fft.displacement, 1, at ).mul( w.y ) )
        .add( cascade( fft.displacement, 2, at ).mul( w.z ) ).add( cascade( fft.displacement, 3, at ).mul( w.w ) );

    };
    const input = this.input, output = this.output, count = this.count;
    this.kernel = Fn( () => {

      const i = instanceIndex;
      If( i.lessThan( count ), () => {

        const target = ( input.element( i ) as unknown as THREE.Node<'vec4'> ).xy.toVar();
        const q = vec2( target ).toVar();
        for ( let k = 0; k < 4; k ++ ) q.assign( target.sub( displacement( q ).xz ) );
        const d = displacement( q ).toVar();
        const height = shared.waterLevel.add( d.y ).add( shoreWaves.surfaceAt( q.add( d.xz ), shared.time ).x );
        const w = waveDamping( inputs, q ).weights;
        // the velocity field holds ( Ux, Uz, Vy )
        const v = cascade( fft.velocity, 0, q ).mul( w.x ).add( cascade( fft.velocity, 1, q ).mul( w.y ) )
          .add( cascade( fft.velocity, 2, q ).mul( w.z ) ).add( cascade( fft.velocity, 3, q ).mul( w.w ) );
        ( output.element( i ) as unknown as THREE.Node<'vec4'> ).assign( vec4( height, v.x, v.z, v.y ) );

      } );

    } )().compute( capacity, [ 64 ] );
    this.kernel.name = 'wave probe';

  }

  /** Whether a batch can be sent now. */
  get ready(): boolean {

    return ! this.inFlight;

  }

  /** Sends `n` points ( x, z ) from `xz` (x0, z0, x1, z1, ...); `onResult` gets the batch's number
   * once the samples are in `result`. Returns false while a batch is still in flight. */
  request( xz: Float32Array, n: number, onResult: ( batch: number ) => void ): boolean {

    if ( this.inFlight || n === 0 ) return false;
    n = Math.min( n, this.capacity );
    const attribute = this.input.value as THREE.StorageInstancedBufferAttribute;
    const array = attribute.array as Float32Array;
    for ( let i = 0; i < n; i ++ ) {

      array[ i * 4 ] = xz[ i * 2 ];
      array[ i * 4 + 1 ] = xz[ i * 2 + 1 ];

    }
    attribute.needsUpdate = true;
    this.count.value = n;
    this.inFlight = true;
    const batch = ++ this.sent;
    this.renderer.compute( this.kernel );
    this.renderer.getArrayBufferAsync( this.output.value as THREE.StorageBufferAttribute ).then( ( buffer ) => {

      this.result.set( new Float32Array( buffer, 0, n * 4 ) );
      this.received = batch;
      this.inFlight = false;
      onResult( batch );

    }, ( error ) => {

      this.inFlight = false;
      console.warn( '[wave probe] readback failed', error );

    } );
    return true;

  }

}
