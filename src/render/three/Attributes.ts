import type { BufferAttribute, InterleavedBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { missing } from './Internals.ts';

// The renderer's attribute records in three: whether uploaded, the upload, the buffer's release.
//
// The renderer's attribute records (three r186's `Renderer._attributes`): whether three has
// uploaded an attribute, its own upload, and the release of an attribute's GPU buffer. The stepped
// uploads (util/Upload.ts) make a buffer as three would and hand it to three through these.

/** three's AttributeType (common/Constants.js): which GPU buffer an attribute becomes. */
export const AttributeType = { VERTEX: 1, INDEX: 2, STORAGE: 3 } as const;

type Attribute = BufferAttribute | InterleavedBufferAttribute;
type Records = { get( attribute: unknown ): { version?: number }; update( attribute: unknown, type: number ): void; delete( attribute: unknown ): unknown };

function records( renderer: WebGPURenderer, user: string ): Records {

	const attributes = ( renderer as unknown as { _attributes?: Records | null } )._attributes;
	if ( ! attributes ) throw missing( 'Renderer._attributes', user );
	return attributes;

}

/** Whether three has uploaded `attribute` (it keeps the version it uploaded). */
export function attributeUploaded( renderer: WebGPURenderer, attribute: Attribute ): boolean {

	return records( renderer, 'attributeUploaded (packages/render/three/Attributes.ts)' ).get( attribute ).version !== undefined;

}

/** three's own upload of `attribute` as a buffer of `type` (AttributeType), as its first draw would
 * make it; it only records the version where the GPU buffer is already made. */
export function uploadAttribute( renderer: WebGPURenderer, attribute: Attribute, type: number ): void {

	records( renderer, 'uploadAttribute (packages/render/three/Attributes.ts)' ).update( attribute, type );

}

/** Frees `attribute`'s GPU buffer, as three frees a geometry's on dispose (r186 Attributes.delete:
 * the buffer destroyed, the renderer's memory info updated). */
export function releaseAttribute( renderer: WebGPURenderer, attribute: Attribute ): void {

	records( renderer, 'releaseAttribute (packages/render/three/Attributes.ts)' ).delete( attribute );

}
