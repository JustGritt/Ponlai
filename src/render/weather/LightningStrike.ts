/*!
 * Adapted from LightningStrike.js by yomboprime, from the three.js examples (removed in r153),
 * as kept in three-stdlib (https://github.com/pmndrs/three-stdlib,
 * src/geometries/LightningStrike.js).
 * SPDX-FileCopyrightText: Copyright 2010-2023 three.js authors
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 Poimandres
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { SimplexNoise } from 'three/addons/math/SimplexNoise.js';

// The branching channel of one lightning strike as triangular prisms.
//
// The branching channel of one lightning strike as triangular prisms, after three.js's former
// LightningStrike: a subray from the source to the destination is split recursively, each
// midpoint displaced by simplex noise in proportion to the segment's length (`roughness`
// scales the displacement down each level, `straightness` damps the first), and leaf segments
// spawn child subrays ("ramifications") that leave in a cone around their parent.
//
// Changes from the original: the shape is fixed for a strike, so the tree is generated once
// instead of every frame. Its life (the leader propagating down from the source, the steady
// channel, the vanishing) moves to the GPU: every vertex carries the times at which the
// propagation front reaches it and leaves it, and the material compares them with the strike's
// age, so the front advances smoothly between frames and a paused frame is bit-identical.
// Each vertex also keeps the point on the channel's axis it was offset from, so the material
// can widen a channel kilometres away to a pixel without moving it. All randomness comes from
// one seed. Branches are not recycled over time (the original's subray period and duty cycle),
// because a strike lasts a second.

export interface LightningStrikeParameters {
	// The ends of the main channel (metres).
	source: THREE.Vector3;
	destination: THREE.Vector3;
	// A direction across the channel at the source; it only orients the prisms.
	up: THREE.Vector3;
	// Radii of the main channel at its ends, and of each generation of branches relative to its
	// parent (at the branching point and at the tip), never below minRadius.
	radius0: number;
	radius1: number;
	radius0Factor: number;
	radius1Factor: number;
	minRadius: number;
	// 0 - 1: how wrinkled the channel is, and how straight each subray runs overall.
	roughness: number;
	straightness: number;
	// Seconds after the strike begins: the leader reaches the destination, the channel begins to
	// vanish, and it is gone.
	propagation: number;
	vanishing: number;
	death: number;
	// The main channel has 2^maxIterations leaf segments, each generation of branches half as many.
	maxIterations: number;
	// Mean number of branches a subray grows, the generations of branches, and how much less
	// likely each generation is to branch again.
	ramification: number;
	maxSubrayRecursion: number;
	recursionProbability: number;
	// Branches drop up to heightFactor of the parent's remaining length and lean out up to
	// sideWidthFactor of their drop; they never reach below `floor` (metres).
	heightFactor: number;
	sideWidthFactor: number;
	floor: number;
	seed: number;
}

interface Subray {
	seed: number;
	maxIterations: number;
	recursion: number;
	pos0: THREE.Vector3;
	pos1: THREE.Vector3;
	linPos0: THREE.Vector3;
	linPos1: THREE.Vector3;
	up0: THREE.Vector3;
	up1: THREE.Vector3;
	radius0: number;
	radius1: number;
	birthTime: number;
	endPropagationTime: number;
	beginVanishingTime: number;
	deathTime: number;
}

interface Segment {
	iteration: number;
	pos0: THREE.Vector3;
	pos1: THREE.Vector3;
	linPos0: THREE.Vector3;
	linPos1: THREE.Vector3;
	up0: THREE.Vector3;
	up1: THREE.Vector3;
	radius0: number;
	radius1: number;
	fraction0: number;
	fraction1: number;
	positionVariationFactor: number;
}

const COS30DEG = Math.cos( 30 * Math.PI / 180 );
const SIN30DEG = Math.sin( 30 * Math.PI / 180 );

// The original's deterministic generator: a fixed table of random numbers read from a seekable
// position, so a branch can restart the sequence from its own seed. The table is filled from
// one integer seed (mulberry32) instead of Math.random.
class SeededGenerator {

	private readonly seeds: Float64Array;
	private current = 0;

	constructor( seed: number ) {

		const n = 2053;
		this.seeds = new Float64Array( n );
		let s = seed | 0;
		for ( let i = 0; i < n; i ++ ) {

			let t = s = ( s + 0x6D2B79F5 ) | 0;
			t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
			t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
			this.seeds[ i ] = ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

		}

	}

	random = (): number => {

		const value = this.seeds[ this.current ];
		this.current = ( this.current + 1 ) % this.seeds.length;
		return value;

	};

	getSeed(): number { return this.current / this.seeds.length; }

	setSeed( seed: number ): void { this.current = Math.floor( seed * this.seeds.length ) % this.seeds.length; }

}

export class LightningStrike extends THREE.BufferGeometry {

	readonly maxIterations: number;
	readonly maxSubrays: number;
	// Vertex count of the last build, and its bounds (for culling).
	vertexCount = 0;

	private parameters: LightningStrikeParameters;
	private random!: SeededGenerator;
	private simplexX!: SimplexNoise;
	private simplexY!: SimplexNoise;
	private simplexZ!: SimplexNoise;
	private readonly subrays: Subray[] = [];
	private readonly raySegments: Segment[] = [];
	private numSubrays = 0;
	private currentSubray: Subray | null = null;
	private currentSegmentIndex = 0;
	private isInitialSegment = false;
	private subrayProbability = 0;
	private currentVertex = 0;
	private currentIndex = 0;
	private readonly vertices: Float32Array;
	private readonly axes: Float32Array;
	private readonly timings: Float32Array;
	private readonly indices: Uint32Array;
	private readonly positionAttribute: THREE.BufferAttribute;
	private readonly axisAttribute: THREE.BufferAttribute;
	private readonly timingAttribute: THREE.BufferAttribute;
	// Temporaries.
	private readonly forwards = new THREE.Vector3();
	private readonly forwardsFill = new THREE.Vector3();
	private readonly side = new THREE.Vector3();
	private readonly down = new THREE.Vector3();
	private readonly middlePos = new THREE.Vector3();
	private readonly middleLinPos = new THREE.Vector3();
	private readonly newPos = new THREE.Vector3();
	private readonly vPos = new THREE.Vector3();
	private readonly cross1 = new THREE.Vector3();
	private readonly vec1Pos = new THREE.Vector3();
	private readonly vec2Forward = new THREE.Vector3();
	private readonly vec3Side = new THREE.Vector3();
	private readonly vec4Up = new THREE.Vector3();

	/** Allocates buffers for trees of at most these sizes; `build` fills them. */
	constructor( maxIterations: number, ramification: number, maxSubrayRecursion: number ) {

		super();
		this.maxIterations = maxIterations;
		this.maxSubrays = Math.ceil( 1 + Math.pow( ramification, Math.max( 0, maxSubrayRecursion - 1 ) ) );
		for ( let i = 0; i < this.maxSubrays; i ++ ) this.subrays.push( this.createSubray() );
		for ( let i = 0; i < 2 * ( 1 << maxIterations ); i ++ ) this.raySegments.push( this.createSegment() );
		const maxDrawableSegmentsPerSubRay = 1 << maxIterations;
		const maxVerts = 3 * ( maxDrawableSegmentsPerSubRay + 1 ) * this.maxSubrays;
		const maxIndices = 18 * maxDrawableSegmentsPerSubRay * this.maxSubrays;
		this.vertices = new Float32Array( maxVerts * 3 );
		this.axes = new Float32Array( maxVerts * 3 );
		// (time the front arrives, time it vanishes, fraction along its subray, branch generation)
		this.timings = new Float32Array( maxVerts * 4 );
		// 32-bit, and kept so: three r186 copies a 16-bit index into a new 32-bit array when it
		// creates the GPU buffer and uploads that copy from then on, so `build` must write the
		// array the attribute holds.
		this.indices = new Uint32Array( maxIndices );
		// Static usage: a tree is built once per strike and lives on the GPU, so `build` uploads
		// only the vertices it wrote, once, through the version bump and update ranges. Marked
		// dynamic, three re-uploads the whole buffers on every draw (some 340 KB a frame, and a
		// stall of hundreds of milliseconds at a bolt's first frame).
		this.positionAttribute = new THREE.BufferAttribute( this.vertices, 3 );
		this.axisAttribute = new THREE.BufferAttribute( this.axes, 3 );
		this.timingAttribute = new THREE.BufferAttribute( this.timings, 4 );
		this.setIndex( new THREE.BufferAttribute( this.indices, 1 ) );
		this.setAttribute( 'position', this.positionAttribute );
		this.setAttribute( 'axis', this.axisAttribute );
		this.setAttribute( 'timing', this.timingAttribute );
		this.setDrawRange( 0, 0 );
		this.parameters = null!;

	}

	/** Generate the whole tree of a strike. */
	build( parameters: LightningStrikeParameters ): void {

		this.parameters = parameters;
		this.random = new SeededGenerator( parameters.seed );
		this.simplexX = new SimplexNoise( this.random );
		this.simplexY = new SimplexNoise( this.random );
		this.simplexZ = new SimplexNoise( this.random );
		this.currentVertex = 0;
		this.currentIndex = 0;
		this.fractalRay();
		this.vertexCount = this.currentVertex;
		this.setDrawRange( 0, this.currentIndex );
		for ( const attribute of [ this.positionAttribute, this.axisAttribute, this.timingAttribute ] ) {

			attribute.clearUpdateRanges();
			attribute.addUpdateRange( 0, this.currentVertex * attribute.itemSize );
			attribute.needsUpdate = true;

		}
		const index = this.index!;
		index.clearUpdateRanges();
		index.addUpdateRange( 0, this.currentIndex );
		index.needsUpdate = true;
		// The bounds of the channel, widened for the branches.
		const box = this.boundingBox ??= new THREE.Box3();
		box.makeEmpty().expandByPoint( parameters.source ).expandByPoint( parameters.destination );
		const spread = parameters.source.distanceTo( parameters.destination ) * 0.6;
		box.expandByScalar( spread );
		this.boundingSphere ??= new THREE.Sphere();
		box.getBoundingSphere( this.boundingSphere );

	}

	private initSubray( subray: Subray ): void {

		const p = this.parameters;
		subray.pos0.copy( p.source );
		subray.pos1.copy( p.destination );
		subray.up0.copy( p.up );
		subray.up1.copy( p.up );
		subray.radius0 = p.radius0;
		subray.radius1 = p.radius1;
		subray.birthTime = 0;
		subray.endPropagationTime = p.propagation;
		subray.beginVanishingTime = p.vanishing;
		subray.deathTime = p.death;
		subray.maxIterations = this.maxIterations;
		subray.seed = 0;
		subray.recursion = 0;

	}

	private fractalRay(): void {

		this.numSubrays = 0;
		// Add the top level subray
		this.initSubray( this.addNewSubray() );
		// Process all subrays that are being generated until consuming all of them
		for ( let subrayIndex = 0; subrayIndex < this.numSubrays; subrayIndex ++ ) {

			const subray = this.subrays[ subrayIndex ];
			this.currentSubray = subray;
			this.random.setSeed( subray.seed );
			const random1 = this.random.random;
			subray.linPos0.set( random1(), random1(), random1() ).multiplyScalar( 1000 );
			subray.linPos1.set( random1(), random1(), random1() ).multiplyScalar( 1000 );
			this.currentSegmentIndex = 0;
			this.isInitialSegment = true;
			const segment = this.getNewSegment();
			segment.iteration = 0;
			segment.pos0.copy( subray.pos0 );
			segment.pos1.copy( subray.pos1 );
			segment.linPos0.copy( subray.linPos0 );
			segment.linPos1.copy( subray.linPos1 );
			segment.up0.copy( subray.up0 );
			segment.up1.copy( subray.up1 );
			segment.radius0 = subray.radius0;
			segment.radius1 = subray.radius1;
			segment.fraction0 = 0;
			segment.fraction1 = 1;
			segment.positionVariationFactor = 1 - this.parameters.straightness;
			this.subrayProbability = this.parameters.ramification * Math.pow( this.parameters.recursionProbability, subray.recursion ) / ( 1 << subray.maxIterations );
			this.fractalRayRecursive( segment );

		}
		this.currentSubray = null;

	}

	private fractalRayRecursive( segment: Segment ): void {

		// Leave recursion condition
		if ( segment.iteration >= this.currentSubray!.maxIterations ) {

			this.createPrism( segment );
			this.decideSubrayCreation( segment );
			return;

		}
		// Interpolation
		this.forwards.subVectors( segment.pos1, segment.pos0 );
		let lForwards = this.forwards.length();
		if ( lForwards < 0.000001 ) {

			this.forwards.set( 0, 0, 0.01 );
			lForwards = this.forwards.length();

		}
		const middleRadius = ( segment.radius0 + segment.radius1 ) * 0.5;
		const middleFraction = ( segment.fraction0 + segment.fraction1 ) * 0.5;
		this.middlePos.lerpVectors( segment.pos0, segment.pos1, 0.5 );
		this.middleLinPos.lerpVectors( segment.linPos0, segment.linPos1, 0.5 );
		const p = this.middleLinPos;
		// Noise (the original's fourth, time dimension stays still: the shape is fixed)
		this.newPos.set( this.simplexX.noise3d( p.x, p.y, p.z ), this.simplexY.noise3d( p.x, p.y, p.z ), this.simplexZ.noise3d( p.x, p.y, p.z ) );
		this.newPos.multiplyScalar( segment.positionVariationFactor * lForwards );
		this.newPos.add( this.middlePos );
		// Recursion
		const roughness = this.parameters.roughness;
		const newSegment1 = this.getNewSegment();
		newSegment1.pos0.copy( segment.pos0 );
		newSegment1.pos1.copy( this.newPos );
		newSegment1.linPos0.copy( segment.linPos0 );
		newSegment1.linPos1.copy( this.middleLinPos );
		newSegment1.up0.copy( segment.up0 );
		newSegment1.up1.copy( segment.up1 );
		newSegment1.radius0 = segment.radius0;
		newSegment1.radius1 = middleRadius;
		newSegment1.fraction0 = segment.fraction0;
		newSegment1.fraction1 = middleFraction;
		newSegment1.positionVariationFactor = segment.positionVariationFactor * roughness;
		newSegment1.iteration = segment.iteration + 1;
		const newSegment2 = this.getNewSegment();
		newSegment2.pos0.copy( this.newPos );
		newSegment2.pos1.copy( segment.pos1 );
		newSegment2.linPos0.copy( this.middleLinPos );
		newSegment2.linPos1.copy( segment.linPos1 );
		this.cross1.crossVectors( segment.up0, this.forwards.normalize() );
		newSegment2.up0.crossVectors( this.forwards, this.cross1 ).normalize();
		newSegment2.up1.copy( segment.up1 );
		newSegment2.radius0 = middleRadius;
		newSegment2.radius1 = segment.radius1;
		newSegment2.fraction0 = middleFraction;
		newSegment2.fraction1 = segment.fraction1;
		newSegment2.positionVariationFactor = segment.positionVariationFactor * roughness;
		newSegment2.iteration = segment.iteration + 1;
		this.fractalRayRecursive( newSegment1 );
		this.fractalRayRecursive( newSegment2 );

	}

	private createPrism( segment: Segment ): void {

		// Creates one triangular prism and its vertices at the segment
		this.forwardsFill.subVectors( segment.pos1, segment.pos0 ).normalize();
		if ( this.isInitialSegment ) {

			this.createTriangleVertices( segment.pos0, segment.up0, this.forwardsFill, segment.radius0, segment.fraction0 );
			this.isInitialSegment = false;

		}
		this.createTriangleVertices( segment.pos1, segment.up0, this.forwardsFill, segment.radius1, segment.fraction1 );
		this.createPrismFaces();

	}

	// An equilateral triangle around the axis point, with its timing: the leader's front reaches
	// a fraction f of the subray at lerp( birth, endPropagation, f ), and the channel vanishes
	// from its start, at lerp( beginVanishing, death, f ).
	private createTriangleVertices( pos: THREE.Vector3, up: THREE.Vector3, forwards: THREE.Vector3, radius: number, fraction: number ): void {

		const subray = this.currentSubray!;
		this.side.crossVectors( up, forwards ).multiplyScalar( radius * COS30DEG );
		this.down.copy( up ).multiplyScalar( - radius * SIN30DEG );
		const arrive = THREE.MathUtils.lerp( subray.birthTime, subray.endPropagationTime, fraction );
		const leave = THREE.MathUtils.lerp( subray.beginVanishingTime, subray.deathTime, fraction );
		const p = this.vPos;
		for ( let k = 0; k < 3; k ++ ) {

			if ( k === 0 ) p.copy( pos ).sub( this.side ).add( this.down );
			else if ( k === 1 ) p.copy( pos ).add( this.side ).add( this.down );
			else p.copy( up ).multiplyScalar( radius ).add( pos );
			const v = this.currentVertex + k;
			this.vertices[ v * 3 ] = p.x;
			this.vertices[ v * 3 + 1 ] = p.y;
			this.vertices[ v * 3 + 2 ] = p.z;
			this.axes[ v * 3 ] = pos.x;
			this.axes[ v * 3 + 1 ] = pos.y;
			this.axes[ v * 3 + 2 ] = pos.z;
			this.timings[ v * 4 ] = arrive;
			this.timings[ v * 4 + 1 ] = leave;
			this.timings[ v * 4 + 2 ] = fraction;
			this.timings[ v * 4 + 3 ] = subray.recursion;

		}
		this.currentVertex += 3;

	}

	private createPrismFaces(): void {

		const indices = this.indices;
		const vertex = this.currentVertex - 6;
		const faces = [ 1, 2, 5, 1, 5, 4, 0, 1, 4, 0, 4, 3, 2, 0, 3, 2, 3, 5 ];
		for ( const f of faces ) indices[ this.currentIndex ++ ] = vertex + f;

	}

	// Decide subray creation at a parent (sub)ray's leaf segment: the original's default
	// callback without the periodic recycling. A branch starts when the parent's front reaches
	// its root and grows at a similar pace.
	private decideSubrayCreation( segment: Segment ): void {

		const subray = this.currentSubray!;
		const p = this.parameters;
		const random1 = this.random.random;
		const childSubraySeed = random1();
		if ( subray.recursion < p.maxSubrayRecursion && this.numSubrays < this.maxSubrays && random1() < this.subrayProbability ) {

			const childSubray = this.addNewSubray();
			const parentSeed = this.random.getSeed();
			childSubray.seed = childSubraySeed;
			this.random.setSeed( childSubraySeed );
			childSubray.recursion = subray.recursion + 1;
			childSubray.maxIterations = Math.max( 1, subray.maxIterations - 1 );
			childSubray.linPos0.set( random1(), random1(), random1() ).multiplyScalar( 1000 );
			childSubray.linPos1.set( random1(), random1(), random1() ).multiplyScalar( 1000 );
			childSubray.up0.copy( subray.up0 );
			childSubray.up1.copy( subray.up1 );
			childSubray.radius0 = Math.max( p.minRadius, segment.radius0 * p.radius0Factor );
			childSubray.radius1 = Math.max( p.minRadius * 0.5, Math.min( p.minRadius, segment.radius1 * p.radius1Factor ) );
			childSubray.birthTime = THREE.MathUtils.lerp( subray.birthTime, subray.endPropagationTime, segment.fraction0 );
			const growth = ( subray.endPropagationTime - subray.birthTime ) * ( 0.4 + 0.5 * random1() );
			childSubray.endPropagationTime = childSubray.birthTime + growth;
			childSubray.beginVanishingTime = subray.beginVanishingTime;
			childSubray.deathTime = subray.deathTime;
			this.subrayConePosition( segment, subray, childSubray );
			this.random.setSeed( parentSeed );

		}

	}

	// Sets the child's pos0 and pos1 in a cone around its parent (after the original's
	// subrayConePosition): from its root on the parent, a branch drops part of the way the
	// parent still has to go and leans out 20 to 50 degrees from it, never below the floor.
	private subrayConePosition( segment: Segment, parentSubray: Subray, childSubray: Subray ): void {

		const p = this.parameters;
		const random1 = this.random.random;
		const { vec1Pos, vec2Forward, vec3Side, vec4Up } = this;
		childSubray.pos0.copy( segment.pos0 );
		vec1Pos.subVectors( parentSubray.pos1, parentSubray.pos0 );
		vec2Forward.copy( vec1Pos ).normalize();
		const drop = vec1Pos.length() * ( 1 - segment.fraction0 ) * p.heightFactor * ( 0.25 + 0.75 * random1() );
		vec3Side.crossVectors( parentSubray.up0, vec2Forward );
		const angle = 2 * Math.PI * random1();
		vec3Side.multiplyScalar( Math.cos( angle ) );
		vec4Up.copy( parentSubray.up0 ).multiplyScalar( Math.sin( angle ) );
		childSubray.pos1.copy( vec3Side ).add( vec4Up ).multiplyScalar( drop * p.sideWidthFactor * ( 0.35 + random1() * 0.65 ) )
			.addScaledVector( vec2Forward, drop ).add( segment.pos0 );
		childSubray.pos1.y = Math.max( childSubray.pos1.y, p.floor );

	}

	private addNewSubray(): Subray { return this.subrays[ this.numSubrays ++ ]; }

	private getNewSegment(): Segment { return this.raySegments[ this.currentSegmentIndex ++ ]; }

	private createSubray(): Subray {

		return {
			seed: 0, maxIterations: 0, recursion: 0,
			pos0: new THREE.Vector3(), pos1: new THREE.Vector3(), linPos0: new THREE.Vector3(), linPos1: new THREE.Vector3(),
			up0: new THREE.Vector3(), up1: new THREE.Vector3(), radius0: 0, radius1: 0,
			birthTime: 0, endPropagationTime: 0, beginVanishingTime: 0, deathTime: 0,
		};

	}

	private createSegment(): Segment {

		return {
			iteration: 0, pos0: new THREE.Vector3(), pos1: new THREE.Vector3(), linPos0: new THREE.Vector3(), linPos1: new THREE.Vector3(),
			up0: new THREE.Vector3(), up1: new THREE.Vector3(), radius0: 0, radius1: 0, fraction0: 0, fraction1: 0, positionVariationFactor: 0,
		};

	}

}
