import * as THREE from 'three/webgpu';
import { Fn, float, instanceIndex, int, ivec3, normalGeometry, positionGeometry, select, storage, storageTexture, texture3D, textureLoad, uint, uniform, varyingProperty, vec3 } from 'three/tsl';
import { META_VEC4, PROBE_STRIDE, type AmbientLattice } from './ProbeFormat.ts';
import { shEvaluate, toMat3 } from './Harmonics.ts';
import { AttributeType, freeAttribute, uploadSteps } from '../util/Upload.ts';

// The ambient light's probes on the GPU: the lattice's buffer and the irradiance 3D texture.
//
// The ambient light's probes on the GPU: the merged lattice (world/AmbientProbes.ts, merged in
// ChunkMerge.ts, its invalid probes filled from their neighbours there) as one read-only storage
// buffer of vec4, described by uniforms, for the relight (AmbientRelight.ts) to read; and the
// relight's output, the probes' irradiance, as a 3D texture (`irradiance`; its layout and how a
// material reads it: AmbientRelight.ts).
//
// - The buffer: PROBE_STRIDE / 4 = 20 vec4 per probe. For the coefficient i (0 .. 8, three.js's
//   SphericalHarmonics3 order, y up) the vec4 i holds ( sky visibility, bounced r, g, b ): the
//   projections, not yet convolved with the cosine lobe ( pi, 2 pi / 3, pi / 4 ); the sky
//   visibility counts only the rays that leave upward. The vec4 9 to 15 hold the groves' in-scatter
//   among the bounced light, r g b per coefficient in turn (ProbeFormat.ts SCATTER_VEC4).
//   The vec4 16 to 18 (SEA_VEC4) hold the sea's visibility, the nine coefficients of the rays that
//   leave downward and reach the sea's surface unblocked, and three spare. The vec4 19 (META_VEC4)
//   holds
//   ( the probe's offset from its lattice point, valid ): a probe moved out of a solid was baked at
//   the offset point; an invalid one (valid 0, no offset) lay deeper in a solid, and holds the mean
//   of its neighbours the merge filled it with (world/AmbientProbes.ts `mergeProbes`).
// - The probes: `dims` ( nx, ny, nz ) probes in rows ( z, then y, then x ), the probe ( i, j, k ) at
//   `origin` + ( i, j, k ) `spacing` (the index ( k ny + j ) nx + i); then ny more, the open sea's
//   probe per level (the index nx ny nz + j), for every point outside the box. Inside the box a
//   square without land near holds copies of it too.
// - Growth: a new version's lattice gets a new buffer, uploaded in the preparation's idle steps
//   (`prepare`, Upload.ts), and a new output texture of its size, made there too; both are swapped
//   in by pointing their nodes at them (`apply`): new bind groups and no pipeline (the shaders
//   declare runtime-sized arrays, and a texture's size is not in its type), the uniforms set in the
//   same frame, before that frame's relight writes the new texture. The old ones are freed once the
//   frames after the swap are drawn. The data is the merge's, a function of the land's chunks, so
//   a grown page holds a fresh load's buffer bit for bit (`hash`).
// - The debug view (`ambientProbes=sky|bounce|irradiance` on a debug page, the GUI's Ground
//   folder): a small sphere at every probe of the box, at the point it was baked at, coloured at
//   its normal by: `sky` the sky visibility convolved with the cosine lobe, over pi (white where a
//   side sees the whole sky); `bounce` the bounced sky visibility likewise (eight times brighter);
//   `irradiance` the relight's output (times its storage scale), E( n ) / pi, the light a white Lambertian surface would
//   reflect lit by the probes alone, in the scene's units. Red where invalid in the first two (the
//   third shows the fill's value). Built on first show, so its pipeline exists only in an app that
//   shows it.

/** The debug view's modes. */
export type AmbientView = 'off' | 'sky' | 'bounce' | 'irradiance';
const MODES: Record<AmbientView, number> = { off: 0, sky: 0, bounce: 1, irradiance: 2 };

/** A lattice prepared for the swap: its buffer, uploaded, and its output texture, made. */
export interface AmbientPrepared { lattice: AmbientLattice; attribute: THREE.StorageBufferAttribute; output: THREE.Storage3DTexture }

/** The output's texels per probe along y: seven slabs of ny levels (AmbientRelight.ts). */
export const IRRADIANCE_SLABS = 7;

// the debug view's spheres (m), and its gains: the colours are drawn unlit into the scene's HDR
// target, where a fair noon's exposure puts about 0.04 at the top of the tone curve, so a side that
// sees the whole sky shows just under white; the bounce's values are a tenth of the sky's
const RADIUS = 0.18, SKY_GAIN = 0.03, BOUNCE_GAIN = 0.24;
const VEC4S = PROBE_STRIDE / 4;

export class AmbientVolume {

	/** the probe ( 0, 0, 0 )'s point (m) */
	readonly origin = uniform( new THREE.Vector3() );
	/** ( nx, ny, nz ): the box's probes per axis (whole numbers, as floats) */
	readonly dims = uniform( new THREE.Vector3() );
	/** the lattice's spacing (m), set from the lattice's own */
	readonly spacing = uniform( 0 );
	/** the probes: VEC4S vec4 each (see the header) */
	readonly probes;
	/** the relight's output, for its kernel to write (AmbientRelight.ts) */
	readonly irradianceStore;
	/** the relight's output, for materials to sample (filtered) or load (AmbientRelight.ts) */
	readonly irradiance;
	/** the output's storage scale at a shader (AmbientRelight.ts `storageGain`, which sets it): its
	 * readers multiply what they read by it */
	storageGain: () => THREE.Node<'float'> = () => float( 1 );
	lattice: AmbientLattice;
	private attribute: THREE.StorageBufferAttribute;
	private output: THREE.Storage3DTexture;
	private readonly renderer: THREE.WebGPURenderer;
	private readonly mode = uniform( 0 );
	private readonly gain = uniform( SKY_GAIN );
	private view: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.MeshBasicNodeMaterial> | null = null;

	constructor( renderer: THREE.WebGPURenderer, lattice: AmbientLattice ) {

		this.renderer = renderer;
		this.lattice = lattice;
		this.attribute = probeBuffer( lattice );
		this.probes = storage( this.attribute, 'vec4', this.attribute.count ).toReadOnly();
		this.output = outputTexture( lattice );
		this.irradianceStore = storageTexture( this.output );
		this.irradiance = texture3D( this.output );
		this.setUniforms( lattice );

	}

	/** A version's lattice for the swap, its buffer uploaded in idle-time steps (Growth.ts). */
	*prepare( lattice: AmbientLattice ): Generator<void, AmbientPrepared> {

		const attribute = probeBuffer( lattice );
		yield* uploadSteps( this.renderer, attribute, AttributeType.STORAGE );
		const output = outputTexture( lattice );
		this.renderer.initTexture( output );
		yield;
		return { lattice, attribute, output };

	}

	/** Swaps a prepared lattice in (the swap's frame); returns what frees the old buffer once
	 * nothing draws with it. */
	apply( prepared: AmbientPrepared ): () => void {

		const old = this.attribute, oldOutput = this.output;
		this.attribute = prepared.attribute;
		this.lattice = prepared.lattice;
		this.output = prepared.output;
		( this.probes as unknown as { value: THREE.StorageBufferAttribute } ).value = prepared.attribute;
		this.irradianceStore.value = prepared.output;
		this.irradiance.value = prepared.output;
		this.setUniforms( prepared.lattice );
		if ( this.view ) this.view.geometry.instanceCount = boxProbes( prepared.lattice );
		return () => {

			freeAttribute( this.renderer, old );
			oldOutput.dispose();

		};

	}

	/** Shows the debug view in `mode` under `parent` (built on first show). */
	show( mode: AmbientView, parent: THREE.Object3D ): void {

		if ( mode !== 'off' && ! this.view ) parent.add( this.view = this.buildView() );
		if ( ! this.view ) return;
		this.view.visible = mode !== 'off';
		this.mode.value = MODES[ mode ];
		this.gain.value = mode === 'bounce' ? BOUNCE_GAIN : mode === 'irradiance' ? 1 : SKY_GAIN;

	}

	/** The lattice's description, for the diagnostics. */
	info(): { origin: number[]; dims: number[]; spacing: number; probes: number; bytes: number } {

		const { x0, y0, z0, nx, ny, nz, spacing, probes } = this.lattice;
		return { origin: [ x0, y0, z0 ], dims: [ nx, ny, nz ], spacing, probes: probes.length / PROBE_STRIDE, bytes: probes.byteLength };

	}

	/** SHA-256 of the lattice (its description and every probe's bits), for comparing a grown page
	 * with a fresh load. */
	async hash(): Promise<string> {

		const { x0, y0, z0, nx, ny, nz, spacing, probes } = this.lattice;
		const head = new Float64Array( [ x0, y0, z0, spacing, nx, ny, nz ] );
		const bytes = new Uint8Array( head.byteLength + probes.byteLength );
		bytes.set( new Uint8Array( head.buffer ), 0 );
		bytes.set( new Uint8Array( probes.buffer, probes.byteOffset, probes.byteLength ), head.byteLength );
		const digest = await crypto.subtle.digest( 'SHA-256', bytes );
		return Array.from( new Uint8Array( digest ), ( b ) => b.toString( 16 ).padStart( 2, '0' ) ).join( '' );

	}

	private setUniforms( lattice: AmbientLattice ): void {

		this.origin.value.set( lattice.x0, lattice.y0, lattice.z0 );
		this.dims.value.set( lattice.nx, lattice.ny, lattice.nz );
		this.spacing.value = lattice.spacing;

	}

	// The debug view (see the header): one instanced mesh of spheres over the box's probes.
	private buildView(): THREE.Mesh<THREE.InstancedBufferGeometry, THREE.MeshBasicNodeMaterial> {

		const geometry = new THREE.InstancedBufferGeometry().copy( new THREE.SphereGeometry( RADIUS, 10, 6 ) as unknown as THREE.InstancedBufferGeometry );
		geometry.instanceCount = boxProbes( this.lattice );
		const material = new THREE.MeshBasicNodeMaterial();
		material.fog = false;
		const colour = varyingProperty( 'vec3', 'vAmbientProbe' );
		material.positionNode = Fn( () => {

			const nx = uint( this.dims.x ), ny = uint( this.dims.y );
			const index = instanceIndex;
			const i = index.mod( nx ), j = index.div( nx ).mod( ny ), k = index.div( nx.mul( ny ) );
			const base = index.mul( uint( VEC4S ) ).toVar();
			const offset = this.probes.element( base.add( uint( META_VEC4 ) ) ).toVar();
			const point = vec3( float( i ), float( j ), float( k ) ).mul( this.spacing ).add( this.origin ).add( offset.xyz );
			// the harmonics at the sphere's normal, convolved with the cosine lobe, over pi
			const n = normalGeometry;
			const c = ( m: number ) => this.probes.element( base.add( uint( m ) ) );
			const band1 = c( 1 ).mul( n.y ).add( c( 2 ).mul( n.z ) ).add( c( 3 ).mul( n.x ) ).mul( 0.488603 * 2 / 3 );
			const band2 = c( 4 ).mul( n.x.mul( n.y ).mul( 1.092548 ) )
				.add( c( 5 ).mul( n.y.mul( n.z ).mul( 1.092548 ) ) )
				.add( c( 6 ).mul( n.z.mul( n.z ).mul( 3 ).sub( 1 ).mul( 0.315392 ) ) )
				.add( c( 7 ).mul( n.x.mul( n.z ).mul( 1.092548 ) ) )
				.add( c( 8 ).mul( n.x.mul( n.x ).sub( n.y.mul( n.y ) ).mul( 0.546274 ) ) ).mul( 1 / 4 );
			const e = c( 0 ).mul( 0.282095 ).add( band1 ).add( band2 ).max( 0 ).toVar();
			// the relight's irradiance over pi (AmbientRelight.ts's packing)
			const t = ( s: number ) => textureLoad( this.irradiance, ivec3( int( i ), int( j ).add( int( ny ).mul( s ) ), int( k ) ) );
			const t0 = t( 0 ).toVar(), t1 = t( 1 ).toVar(), t2 = t( 2 ).toVar(), t3 = t( 3 ).toVar(), t4 = t( 4 ).toVar(), t5 = t( 5 ).toVar(), t6 = t( 6 ).toVar();
			const channel = ( low: typeof t0, high: typeof t0, last: THREE.Node<'float'> ) => shEvaluate( toMat3( [ low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w, last ] ), n );
			const irradiance = vec3( channel( t0, t3, t6.x ), channel( t1, t4, t6.y ), channel( t2, t5, t6.z ) ).max( 0 ).mul( this.storageGain() ).div( Math.PI ).toVar();
			// (the selects pick between values built before them: TSL may branch a select)
			const sky = vec3( e.x ).toVar(), bounce = e.yzw.toVar();
			const baked = select( this.mode.greaterThan( 0.5 ), bounce, sky ).toVar();
			const shown = select( this.mode.greaterThan( 1.5 ), irradiance, baked ).mul( this.gain ).toVar();
			const flagged = select( this.mode.greaterThan( 1.5 ), shown, vec3( SKY_GAIN, 0, 0 ) ).toVar();
			colour.assign( select( offset.w.greaterThan( 0.5 ), shown, flagged ) );
			return point.add( positionGeometry );

		} )();
		material.colorNode = colour;
		const mesh = new THREE.Mesh( geometry, material );
		mesh.name = 'ambient-probes';
		mesh.frustumCulled = false;
		mesh.castShadow = mesh.receiveShadow = false;
		return mesh;

	}

}

// the lattice's buffer: its probes as vec4
function probeBuffer( lattice: AmbientLattice ): THREE.StorageBufferAttribute {

	const attribute = new THREE.StorageBufferAttribute( lattice.probes, 4 );
	attribute.name = 'ambientProbes';
	return attribute;

}

// the relight's output for a lattice: ( nx + 1 ) x IRRADIANCE_SLABS ny x nz half-float texels,
// filtered (AmbientRelight.ts)
function outputTexture( lattice: AmbientLattice ): THREE.Storage3DTexture {

	const output = new THREE.Storage3DTexture( lattice.nx + 1, IRRADIANCE_SLABS * lattice.ny, Math.max( lattice.nz, 1 ) );
	output.type = THREE.HalfFloatType;
	output.format = THREE.RGBAFormat;
	output.minFilter = output.magFilter = THREE.LinearFilter;
	output.generateMipmaps = false;
	output.name = 'ambientIrradiance';
	return output;

}

// the probes in the box (the debug view's instances; the open sea's per level follow them)
function boxProbes( lattice: AmbientLattice ): number {

	return lattice.nx * lattice.ny * lattice.nz;

}
