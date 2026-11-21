import * as THREE from 'three/webgpu';

// The instance store of an instanced vegetation kind: its per-instance buffers and capacity.
//
// The instance store of an instanced vegetation kind: one `InstancePool` per kind, which names the kind's per-instance buffers beside the matrices and
// keeps its capacity, and every write into those buffers goes through it. The kind keeps its
// geometry, material and placement; the pool keeps the buffers.
// - Capacity: an instanced mesh's buffers hold a capacity, not the count, so a grown island's
//   instances are written into the same mesh (`set`) and draw with the pipeline it drew with. Up to
//   1024 instances three keeps the matrices in a uniform array of exactly the capacity, which the
//   shader names, so a pool holds `capacity` instances (at most 1024) whatever the land and draws
//   only the first `capacity` of a larger land (warned).
// - Writes (`writes`, `set`): each buffer's used part, from its start; the update range covers the
//   values written (at least one item), the count follows, and the mesh's box and sphere are cleared
//   or set to the box found ahead (`box`: InstancedMesh.computeBoundingBox's, found in the
//   preparation's steps; the sun's caster box reads it). A grown island's writes are staged ahead
//   (Growth.ts, Upload.ts stagedWriteSteps), so the swap's write is a copy on the GPU.
// - Layout: the attributes go on the geometry in the order the pool names them (`instanceColor` is
//   the mesh's own, set after the mesh is made, as three's setColorAt would), and the writes go in
//   that order after the matrices.
// Kept for the plant kinds to come; no kind uses it today (the rim bushes it was written for are
// removed; the meadow instances itself from the cull's storage buffers, VegetationCull.ts).

/** One per-instance buffer beside the matrices: `key` names its values in the instances, `name` the
 * geometry's attribute (or `instanceColor`, the mesh's colour), `itemSize` its floats per instance. */
export interface InstanceAttribute<K extends string> { key: K; name: string; itemSize: number }

/** A kind's instances as its mesh's buffers take them: `count` instances, 16 floats of matrix each
 * and each attribute's `itemSize` floats under its key. */
export type Instances<K extends string> = { count: number; matrices: Float32Array } & { [ key in K ]: Float32Array };

type Write = [ THREE.BufferAttribute, Float32Array ];

export class InstancePool<K extends string> {

	readonly attributes: readonly InstanceAttribute<K>[];
	readonly capacity: number;

	constructor( attributes: readonly InstanceAttribute<K>[], capacity: number ) {

		if ( capacity > 1024 ) throw new RangeError( `InstancePool: a pool of ${ capacity } would cross 1024 instances` );
		this.attributes = attributes;
		this.capacity = capacity;

	}

	/** How many of `count` instances the pool draws; `what` names the kind in the warning. */
	fit( count: number, what: string ): number {

		if ( count <= this.capacity ) return count;
		console.warn( `${ count } ${ what }: only the first ${ this.capacity } are drawn` );
		return this.capacity;

	}

	/** Empty instances of `count`, to fill. */
	empty( count: number ): Instances<K> {

		const out = { count, matrices: new Float32Array( count * 16 ) } as Instances<K>;
		for ( const a of this.attributes ) ( out as Record<string, Float32Array> )[ a.key ] = new Float32Array( count * a.itemSize );
		return out;

	}

	/** Puts the pool's per-instance attributes of `capacity` instances on `geometry`. */
	addAttributes( geometry: THREE.BufferGeometry, capacity: number ): THREE.BufferGeometry {

		for ( const a of this.attributes ) {

			if ( a.name !== 'instanceColor' ) geometry.setAttribute( a.name, new THREE.InstancedBufferAttribute( new Float32Array( capacity * a.itemSize ), a.itemSize ) );

		}
		return geometry;

	}

	/** An instanced mesh of `capacity` instances on `geometry` (its attributes added: addAttributes),
	 * empty: `set` fills it. */
	mesh( geometry: THREE.BufferGeometry, material: THREE.Material, capacity: number ): THREE.InstancedMesh {

		const mesh = new THREE.InstancedMesh( geometry, material, capacity );
		// (a colour per instance whatever the count, so the shader is the same on every land)
		const color = this.attributes.find( ( a ) => a.name === 'instanceColor' );
		if ( color ) mesh.instanceColor = new THREE.InstancedBufferAttribute( new Float32Array( capacity * color.itemSize ), color.itemSize );
		return mesh;

	}

	/** The mesh's buffers `instances` are written into, each with its values. */
	writes( mesh: THREE.InstancedMesh, instances: Instances<K> ): Write[] {

		const out: Write[] = [ [ mesh.instanceMatrix, instances.matrices ] ];
		for ( const a of this.attributes ) {

			const attribute = a.name === 'instanceColor' ? mesh.instanceColor! : mesh.geometry.getAttribute( a.name ) as THREE.BufferAttribute;
			out.push( [ attribute, instances[ a.key ] ] );

		}
		return out;

	}

	/** Writes `instances` into the mesh's buffers (their used part re-uploaded), sets the count and
	 * the box (`box`, or cleared, with the sphere, for three to find). */
	set( mesh: THREE.InstancedMesh, instances: Instances<K>, box: THREE.Box3 | null = null ): void {

		for ( const [ attribute, values ] of this.writes( mesh, instances ) ) {

			( attribute.array as Float32Array ).set( values );
			attribute.clearUpdateRanges();
			attribute.addUpdateRange( 0, Math.max( values.length, attribute.itemSize ) );
			attribute.needsUpdate = true;

		}
		mesh.count = instances.count;
		mesh.boundingBox = null;
		mesh.boundingSphere = null;
		if ( box ) mesh.boundingBox = box;

	}

	/** The box of `instances` on `mesh`'s geometry, as InstancedMesh.computeBoundingBox finds it. */
	box( mesh: THREE.InstancedMesh, instances: Instances<K> ): THREE.Box3 {

		const geometry = mesh.geometry;
		if ( geometry.boundingBox === null ) geometry.computeBoundingBox();
		const box = new THREE.Box3(), part = new THREE.Box3(), m = new THREE.Matrix4();
		for ( let i = 0; i < instances.count; i ++ ) {

			m.fromArray( instances.matrices, i * 16 );
			part.copy( geometry.boundingBox! ).applyMatrix4( m );
			box.union( part );

		}
		return box;

	}

}
