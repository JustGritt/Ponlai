import * as THREE from 'three/webgpu';
import { AttributeType, attributeUploaded, releaseAttribute, uploadAttribute } from '../three/Attributes.ts';
import { gpuBuffer, gpuDevice, gpuTexture, setGpuBuffer, wrapAttributeWrites } from '../three/Backend.ts';

// Uploads in steps: a buffer's or a data texture's GPU data written a slice at a time.
//
// Uploads in steps, for the growth's idle-time preparation (Growth.ts): a buffer attribute's GPU
// buffer, or a data texture's texels, written a slice at a time through a buffer mapped at
// creation, then unmapped (and for a texture copied on the GPU). The data the GPU ends up with is
// what three's own upload gives it; only the work is spread, so no step takes more than a few
// tenths of a millisecond, where a whole upload took several (a mapped buffer of megabytes filled
// at once).
//
// Never queue.writeBuffer or queue.writeTexture: a write sends its data through the GPU process's
// command stream, and with the GPU busy a few megabytes stalled the main thread for hundreds of
// milliseconds (WorldFields.ts).
//
// The buffer of an attribute is made as three r186's WebGPUAttributeUtils.createAttribute makes it
// (the same usage for its type, the size rounded to 4 bytes) and set where three keeps it (the
// backend's data of the attribute's buffer: three/Backend.ts `setGpuBuffer`), so three's own update
// then finds it made and only records its version: three owns it from then on (frees it on
// dispose, writes it on a change). Data three converts on upload (16-bit or 8-bit integers, items
// padded to 4 bytes, vec3 storage) goes through three's own upload, in one step. Re-check against
// three's createAttribute on a three upgrade.
//
// Writes in place (`stagedWriteSteps`; kept for the plant kinds to come, no kind uses it today: it
// was written for the removed rim bushes' instances): an attribute rewritten at the swap keeps its
// buffer, so its new data is staged ahead in a buffer mapped at creation, and when three next writes that range of the attribute's array (its
// first draw after the swap) the backend copies the staged data on the GPU instead of a queue write
// (`updateAttribute` wrapped once per backend: three/Backend.ts `wrapAttributeWrites`). The writes were under a megabyte, yet each 256 KB
// instance-matrix write blocked the main thread for 15-35 ms on a running page with the GPU busy,
// and the first after-swap step (the rain's map, whose proxies share those buffers) took 50-200 ms.
// The key is the typed array: three uploads an instanced mesh's matrices and colours through
// buffers of its own (r186 nodes/accessors/Instance.js) that wrap the attribute's array, so the
// array is what the write and the stage share. There three also repeats the write: it keeps the
// uploaded version per node attribute, and every material built on the mesh (its shadow, prepass
// and wind variants, the rain map's proxies) has four on the one buffer, so the first write takes
// the update range and each later one wrote the whole buffer again (256 KB a species). A write of
// the staged version is therefore the copy once per GPU buffer and skipped after it; a write of
// an older version (before the swap) or of other data goes through three.

export { AttributeType };

// WebGPU's GPUBufferUsage
const COPY_SRC = 0x0004, COPY_DST = 0x0008, INDEX = 0x0010, VERTEX = 0x0020, STORAGE = 0x0080;
// bytes written a step
const SLICE = 1 << 18;
// a texture copy's rows start on 256-byte boundaries
const ROW_ALIGN = 256;

type GpuBuffer = { getMappedRange(): ArrayBuffer; unmap(): void; destroy(): void };
type GpuDevice = {
	createBuffer( descriptor: { label?: string; size: number; usage: number; mappedAtCreation: boolean } ): GpuBuffer;
	createCommandEncoder(): {
		copyBufferToTexture( source: { buffer: GpuBuffer; offset: number; bytesPerRow: number; rowsPerImage: number }, destination: { texture: object; origin: { x: number; y: number } }, size: { width: number; height: number } ): void;
		finish(): unknown;
	};
	queue: { submit( buffers: unknown[] ): void };
};
type Attribute = THREE.BufferAttribute | THREE.InterleavedBufferAttribute;

/** `attribute`'s GPU buffer, made and filled in steps (see the header). */
export function* uploadSteps( renderer: THREE.WebGPURenderer, attribute: Attribute | null | undefined, type: number ): Generator<void, void> {

	if ( ! attribute || attributeUploaded( renderer, attribute ) ) return;
	const source = ( attribute as THREE.InterleavedBufferAttribute ).isInterleavedBufferAttribute ? ( attribute as THREE.InterleavedBufferAttribute ).data : attribute as THREE.BufferAttribute;
	let array = source.array as ArrayBufferView;
	const storage = ( source as unknown as { isStorageBufferAttribute?: boolean; isStorageInstancedBufferAttribute?: boolean } );
	const isStorage = !! ( storage.isStorageBufferAttribute || storage.isStorageInstancedBufferAttribute );
	const narrow = array instanceof Uint16Array || array instanceof Uint8Array || array instanceof Int16Array || array instanceof Int8Array;
	const plain = array instanceof Float32Array || array instanceof Uint32Array || array instanceof Int32Array;
	const itemSize = ( source as THREE.BufferAttribute ).itemSize;
	// (a narrow item three pads, or one it keeps narrow when normalised, goes through three)
	const widen = narrow && attribute.normalized === false && ! ( attribute as THREE.InterleavedBufferAttribute ).isInterleavedBufferAttribute && itemSize === 1;
	if ( ! ( plain || widen ) || ( isStorage && itemSize === 3 ) || gpuBuffer( renderer, source ) ) {

		uploadAttribute( renderer, attribute, type );
		yield;
		return;

	}
	if ( widen ) {

		// three's patch for 16- and 8-bit integers: widened to 32 bits (an index's 0xffff to the
		// 32-bit primitive restart), and the attribute keeps the wide array
		const from = array as Uint16Array | Uint8Array | Int16Array | Int8Array, signed = from instanceof Int16Array || from instanceof Int8Array;
		const wide = signed ? new Int32Array( from.length ) : new Uint32Array( from.length );
		const restart = ! signed && from instanceof Uint16Array && type === AttributeType.INDEX;
		for ( let k = 0; k < from.length; k += SLICE / 4 ) {

			for ( let i = k, end = Math.min( k + SLICE / 4, from.length ); i < end; i ++ ) wide[ i ] = restart && from[ i ] === 0xffff ? 0xffffffff : from[ i ];
			yield;

		}
		( source as { array: ArrayBufferView } ).array = array = wide;

	}
	const usage = type === AttributeType.INDEX ? INDEX | COPY_SRC | COPY_DST | ( isStorage ? STORAGE : 0 )
		: type === AttributeType.STORAGE ? STORAGE | VERTEX | COPY_SRC | COPY_DST : VERTEX | COPY_SRC | COPY_DST;
	const bytes = new Uint8Array( array.buffer, array.byteOffset, array.byteLength );
	const buffer = gpuDevice<GpuDevice>( renderer ).createBuffer( { label: ( source as { name?: string } ).name ?? '', size: Math.ceil( bytes.length / 4 ) * 4, usage, mappedAtCreation: true } );
	const mapped = new Uint8Array( buffer.getMappedRange() );
	for ( let k = 0; k < bytes.length; k += SLICE ) {

		yield;
		mapped.set( bytes.subarray( k, k + SLICE ), k );

	}
	yield;
	buffer.unmap();
	setGpuBuffer( renderer, source, buffer );
	uploadAttribute( renderer, attribute, type );

}

/** Frees `attribute`'s GPU buffer, as three frees a geometry's on dispose (three/Attributes.ts
 * `releaseAttribute`). For a buffer no geometry owns, such as a storage node's that a swap replaced,
 * once nothing draws with it. */
export function freeAttribute( renderer: THREE.WebGPURenderer, attribute: Attribute ): void {

	releaseAttribute( renderer, attribute );

}

// (`staged`: the array's version when it was staged; `version`: the write's, once the copy is made)
type Staged = { buffer: GpuBuffer | null; count: number; staged: number; version: number; copied: WeakSet<GpuBuffer> };
const stagedByRenderer = new WeakMap<object, WeakMap<ArrayBufferView, Staged>>();

// The renderer's staged writes, its backend's attribute writes routed through them on first use
// (see the header).
function stagedWrites( renderer: THREE.WebGPURenderer ): WeakMap<ArrayBufferView, Staged> {

	let staged = stagedByRenderer.get( renderer );
	if ( staged ) return staged;
	const writes = staged = new WeakMap();
	stagedByRenderer.set( renderer, writes );
	const device = gpuDevice<GpuDevice>( renderer );
	// (`target`: the attribute's GPU buffer, undefined for data three pads, whose writes are three's own)
	wrapAttributeWrites( renderer, ( source, target, update ) => {

		const { array, updateRanges, version } = source as { array: ArrayBufferView; updateRanges: { start: number; count: number }[]; version: number };
		const write = writes.get( array );
		const buffer = target as GpuBuffer | undefined;
		// (a write of the data before it was staged is three's own)
		if ( ! write || ! buffer || version <= write.staged ) return update();
		// the staged range, or the whole array (three's write once another of its attributes has
		// taken the ranges): beyond the range the array and the buffer hold the same old data
		const ranged = updateRanges.length === 0 || ( updateRanges.length === 1 && updateRanges[ 0 ].start === 0 && updateRanges[ 0 ].count === write.count );
		if ( write.version < 0 && ranged ) write.version = version;
		if ( write.version !== version ) {

			// a write of other data: three's own
			writes.delete( array );
			return update();

		}
		( source as { clearUpdateRanges(): void } ).clearUpdateRanges();
		if ( write.copied.has( buffer ) ) return;
		if ( ! write.buffer ) return update();
		const encoder = device.createCommandEncoder() as unknown as { copyBufferToBuffer( source: GpuBuffer, sourceOffset: number, destination: GpuBuffer, destinationOffset: number, size: number ): void; finish(): unknown };
		encoder.copyBufferToBuffer( write.buffer, 0, buffer, 0, write.count * 4 );
		device.queue.submit( [ encoder.finish() ] );
		write.copied.add( buffer );

	} );
	return staged;

}

/** Stages `values` for a later in-place write of `attribute`'s first `values.length` elements (an
 * update range of that length from 0), so that three's write becomes a copy on the GPU (see the
 * header). Returns what frees the stage once the write has been drawn, or null where nothing was
 * staged (three's own write then). */
export function* stagedWriteSteps( renderer: THREE.WebGPURenderer, attribute: THREE.BufferAttribute, values: Float32Array ): Generator<void, ( () => void ) | null> {

	const array = attribute.array;
	if ( ! ( array instanceof Float32Array ) || values.length === 0 ) return null;
	const writes = stagedWrites( renderer );
	const buffer = gpuDevice<GpuDevice>( renderer ).createBuffer( { label: 'stagedWrite', size: values.byteLength, usage: COPY_SRC, mappedAtCreation: true } );
	const mapped = new Float32Array( buffer.getMappedRange() );
	for ( let k = 0; k < values.length; k += SLICE / 4 ) {

		yield;
		mapped.set( values.subarray( k, k + SLICE / 4 ), k );

	}
	yield;
	buffer.unmap();
	const write: Staged = { buffer, count: values.length, staged: attribute.version, version: - 1, copied: new WeakSet() };
	writes.set( array, write );
	return () => {

		// (the entry stays: a later write of the same version is one three repeats for another of
		// the buffer's attributes, and is skipped where the copy went)
		write.buffer = null;
		buffer.destroy();

	};

}

/** Every buffer of `geometry` uploaded in steps. */
export function* geometrySteps( renderer: THREE.WebGPURenderer, geometry: THREE.BufferGeometry ): Generator<void, void> {

	for ( const attribute of Object.values( geometry.attributes ) ) yield* uploadSteps( renderer, attribute as Attribute, AttributeType.VERTEX );
	yield* uploadSteps( renderer, geometry.getIndex(), AttributeType.INDEX );

}

/** Makes `texture`'s GPU texture and fills it from its data in steps: a staging buffer mapped at
 * creation and written a slice at a time, then a copy on the GPU (see the header). three's own
 * upload where three made no GPU texture. */
export function* textureSteps( renderer: THREE.WebGPURenderer, texture: THREE.DataTexture ): Generator<void, void> {

	const data = texture.image.data as ArrayBufferView, { width, height } = texture.image;
	texture.source.dataReady = false;
	renderer.initTexture( texture );
	texture.source.dataReady = true;
	const gpu = gpuTexture( renderer, texture ), device = gpuDevice<GpuDevice>( renderer );
	if ( ! gpu ) {

		texture.needsUpdate = true;
		renderer.initTexture( texture );
		return;

	}
	yield;
	const row = data.byteLength / height, pitch = Math.ceil( row / ROW_ALIGN ) * ROW_ALIGN;
	const source = new Uint8Array( data.buffer, data.byteOffset, data.byteLength );
	const buffer = device.createBuffer( { size: Math.ceil( pitch * height / 4 ) * 4, usage: COPY_SRC, mappedAtCreation: true } );
	const mapped = new Uint8Array( buffer.getMappedRange() );
	const rows = Math.max( 1, Math.floor( SLICE / pitch ) );
	for ( let j0 = 0; j0 < height; j0 += rows ) {

		yield;
		const j1 = Math.min( height, j0 + rows );
		if ( pitch === row ) mapped.set( source.subarray( j0 * row, j1 * row ), j0 * row );
		else for ( let j = j0; j < j1; j ++ ) mapped.set( source.subarray( j * row, ( j + 1 ) * row ), j * pitch );

	}
	yield;
	buffer.unmap();
	const encoder = device.createCommandEncoder();
	encoder.copyBufferToTexture( { buffer, offset: 0, bytesPerRow: pitch, rowsPerImage: height }, { texture: gpu, origin: { x: 0, y: 0 } }, { width, height } );
	device.queue.submit( [ encoder.finish() ] );
	// (freed once the copy has run)
	buffer.destroy();

}
