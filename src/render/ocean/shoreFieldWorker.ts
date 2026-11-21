// The rocky shore's wave fields of a grown island, computed off the main thread.
//
// The rocky shore's wave fields of a grown island (ShoreWaveField.ts), off the main thread: a job is
// the shore's distance field and the swell's direction, the reply the two fields' texels
// (transferred). ShoreWaves.ts `fieldsFor` runs one worker per growth step.

import { shoreBoundaryPixels, shoreShelterPixels, type FieldPixels } from './ShoreWaveField.ts';
import type { ShoreField } from '../RenderTypes.ts';

export interface ShoreFieldJob { shore: ShoreField; direction: { x: number; y: number } }
export interface ShoreFieldReply { boundary: FieldPixels; shelter: FieldPixels & { info: Record<string, unknown> } }

self.onmessage = ( e: MessageEvent<ShoreFieldJob> ) => {

	const { shore, direction } = e.data;
	const reply: ShoreFieldReply = { boundary: shoreBoundaryPixels( shore ), shelter: shoreShelterPixels( shore, direction ) };
	self.postMessage( reply, { transfer: [ reply.boundary.pixels.buffer, reply.shelter.pixels.buffer ] } );

};
