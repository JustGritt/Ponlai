/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/include/gpu/
 * fsr3upscaler/ffx_fsr3upscaler_callbacks_hlsl.h).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import {
	uniform, texture, storageTexture, textureStore, textureSize, storage, float, int, uint, vec2, vec4, ivec2, uvec2,
	select, max, mix, If, floor, round, saturate, atomicMin, atomicMax,
} from './Tsl.ts';
import { ffxAsFloat, ffxAsUInt32, ffxFract, type N } from './Core.ts';
import {
	FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE, getResourceAliases,
	type Fsr3ShaderOptions, type Fsr3FramePlan, type Fsr3UpscalerGenerateReactiveConstants,
} from './Host.ts';
import * as R from './Resources.ts';

// FSR 3's bindings: the constant buffers and resource callbacks by their upstream names.
//
// ffx_fsr3upscaler_callbacks_hlsl.h as one object: the three constant buffers (cbFSR3Upscaler,
// cbSPD, cbRCAS, and cbGenerateReactive) as uniforms with upstream's getters, and every resource
// callback (`Load*`, `Store*`, `Sample*`, `Get*ResourceDimensions`) by its upstream name. The
// shader modules call only these, so a later diff against a new SDK release maps line to line.
//
// The interface the integration fills in (`Fsr3BindingsDescription`):
// - `textures`: the internal resources by their concrete identifier (Resources.ts:
//   ACCUMULATION_1 / _2, INTERNAL_UPSCALED_COLOR_1 / _2, LUMA_HISTORY_1 / _2, LUMA_1 / _2,
//   INTERMEDIATE_FP16x1, SHADING_CHANGE, NEW_LOCKS, SPD_MIPS, FARTHEST_DEPTH_MIP1,
//   DILATED_REACTIVE_MASKS, DILATED_DEPTH, DILATED_MOTION_VECTORS, LANCZOS_LUT), plus
//   UPSCALED_OUTPUT (the upscaler's output, a StorageTexture of the upscale-max size) and, for the
//   reactive-mask generator, AUTOREACTIVE (render-max size). Each written one is a StorageTexture
//   in `FSR3_INTERNAL_RESOURCES`'s format; the linearly sampled ones have LinearFilter and
//   ClampToEdgeWrapping (SPD_MIPS a mip filter: `LinearMipmapNearestFilter`).
// - `buffers`: the reconstructed previous nearest depth (StorageBufferAttribute of render-max
//   width x height u32) and FRAME_INFO (StorageBufferAttribute of one vec4).
// - `inputs`: the frame's inputs as accessors the integration writes, so it can adapt its own
//   formats (view z to device depth, NDC velocity to FSR's motion) without touching the port. Each
//   returns what upstream's resource read returns, before upstream's callback adds anything
//   (the motion-vector scale and jitter cancellation, the reactiveness scale, the exposure's
//   fallback stay here, as upstream has them):
//     LoadInputColor( p: ivec2 ) -> vec3           r_input_color_jittered[ p ].rgb
//     LoadInputDepth( p: ivec2 ) -> float          r_input_depth[ p ]
//     LoadInputMotionVector( p: ivec2 ) -> vec2    r_input_motion_vectors[ p ].xy
//     LoadInputExposure() -> float                 r_input_exposure[ 0, 0 ].x (optional: upstream's
//                                                  default exposure resource, 0, meaning exposure 1)
//     LoadReactiveMask( p ), SampleReactiveMask( uv ), GetReactiveMaskResourceDimensions()
//                                                  (optional: the default reactivity, 0, 1 x 1)
//     LoadTransparencyAndCompositionMask( p ), SampleTransparencyAndCompositionMask( uv ),
//     GetTransparencyAndCompositionMaskResourceDimensions()   (optional, as the reactive mask)
//     LoadOpaqueOnly( p ) -> vec3                  (the reactive-mask generator only)
//     SampleInputColor( uv ), SampleInputDepth( uv )   (no pass calls them; optional)
//   Out-of-bounds coordinates reach LoadInputMotionVector only with display-resolution motion
//   vectors; an accessor returns 0 there, as a D3D load does.
//   Every accessor builds and returns a new node on each call (or returns a uniform node), never
//   one node it keeps: three r186 assigns a texture read (and any temporary used twice) to a
//   variable where it is first built and reads that variable at every later use, and the kernels
//   first call some accessors inside a branch (`Exposure()` inside the accumulate kernel's
//   reprojection and LumaInstability's `If`), so a kept node would read 0 on every pixel that
//   skips the branch. A float uniform is emitted as its uniform at each use. The app's exposure
//   is a texture load (the exposure state's texel, `Pipeline.ts`), so its accessor builds the load
//   anew at each call too.
// - `setFrame( plan )` each frame before the dispatches: copies the plan's constants into the
//   uniforms and points every ping-pong and aliased binding at this frame's side (a new bind
//   group, no new pipeline).
//
// Differences from upstream (the WebGPU mapping):
// - Each UAV (a resource's storage binding at one mip level and access) has a uniform hash of its
//   own: three r186 merges texture nodes of one texture into one binding, which sent every SPD
//   level's store through level 0's view.
// - Upstream's texture formats map to WebGPU's as Resources.ts says; NEW_LOCKS is r32float
//   (read-write), so `StoreNewLocks` rounds to 8 bits as upstream's R8_UNORM stores would.
// - D3D discards a store outside a resource and reads 0 outside one; WGSL may clamp either to an
//   edge texel. Every `Store*` here skips texels outside its resource (render-max, half of it,
//   upscale-max, a mip's size), and the loads whose coordinates can leave the resource
//   (`LoadDilatedDepth`, `LoadDilatedMotionVector`, `LoadReconstructedPrevDepth`, reached from
//   ComputeMotionDivergence's reprojected position, and `LoadRCAS_Input`, whose cross reads one
//   texel past the image's edges) return 0 outside it. The other loads clamp
//   their coordinates to the live size first (`ClampLoad`, `ClampCoord`, `IsOnScreen`), except
//   `LoadRwNewLocks`: it reads past NEW_LOCKS only in the last partial workgroup of an upscale
//   size that is not a multiple of 8, where the store it feeds (`storeIf`) drops the texel.
// - The reconstructed previous nearest depth is a u32 storage buffer: `StoreReconstructedDepth`
//   is an `atomicMin` (`atomicMax` when inverted) on its element; FRAME_INFO is a vec4 storage
//   buffer (`LoadFrameInfo` / `StoreFrameInfo` read and write element 0).
// - The SPD atomic counter callbacks (`SPD_IncreaseAtomicCounter`, `SPD_ResetAtomicCounter`) are
//   gone with the counter (Spd.ts); `RWLoadPyramid( p, 5 )`, the tail's only read, loads SPD_MIPS's
//   level 5 through a sampled binding in the tail's own dispatch.
// - r32float does not filter in WebGPU core: `SampleLanczos2Weight` and `SampleDilatedDepth`
//   filter by hand, the same weights as a bilinear clamp-to-edge sample.
// - `GetSPDMipDimensions` and the resource dimension getters read `textureDimensions` (level 0's:
//   three r186 drops the level for a storage texture, so a mip's size is level 0's halved).
// - The deprecated auto-composition and pre/post-alpha callbacks (`StoreAutoReactive`,
//   `LoadPrevPreAlpha`, ...) belong to the removed TCR pass and are not ported; the reactive-mask
//   generator stores straight into `rw_output_autoreactive` as its pass file does.

/** The frame's inputs (see the header). */
export interface Fsr3Inputs {
	LoadInputColor: ( p: N ) => N;
	LoadInputDepth: ( p: N ) => N;
	LoadInputMotionVector: ( p: N ) => N;
	LoadInputExposure?: () => N;
	LoadReactiveMask?: ( p: N ) => N;
	SampleReactiveMask?: ( uv: N ) => N;
	GetReactiveMaskResourceDimensions?: () => N;
	LoadTransparencyAndCompositionMask?: ( p: N ) => N;
	SampleTransparencyAndCompositionMask?: ( uv: N ) => N;
	GetTransparencyAndCompositionMaskResourceDimensions?: () => N;
	LoadOpaqueOnly?: ( p: N ) => N;
	SampleInputColor?: ( uv: N ) => N;
	SampleInputDepth?: ( uv: N ) => N;
}

export interface Fsr3BindingsDescription {
	/** the context's flags (Host.ts FFX_FSR3UPSCALER_ENABLE_*) */
	flags: number;
	textures: Map<number, THREE.Texture>;
	buffers: {
		reconstructedPrevNearestDepth: THREE.StorageBufferAttribute;
		frameInfo: THREE.StorageBufferAttribute;
	};
	inputs: Fsr3Inputs;
}

export class Fsr3Bindings {

	readonly flags: number;
	private readonly textures: Map<number, THREE.Texture>;
	private readonly inputs: Fsr3Inputs;
	private readonly srvNodes = new Map<number, N>();
	private readonly uavNodes = new Map<string, N>();
	private srvAliases: Map<number, number>;
	private uavAliases: Map<number, number>;
	private reconstructedDepthBuffer: THREE.StorageBufferAttribute;
	private readonly frameInfoBuffer: THREE.StorageBufferAttribute;
	private reconstructedDepthRead: N = null;
	private reconstructedDepthAtomic: N = null;
	private reconstructedDepthWrite: N = null;
	private frameInfoRead: N = null;
	private frameInfoReadWrite: N = null;

	// cbFSR3Upscaler
	readonly iRenderSize = uniform( 'ivec2' ).setName( 'iRenderSize' ) as N;
	readonly iPreviousFrameRenderSize = uniform( 'ivec2' ).setName( 'iPreviousFrameRenderSize' ) as N;
	readonly iUpscaleSize = uniform( 'ivec2' ).setName( 'iUpscaleSize' ) as N;
	readonly iPreviousFrameUpscaleSize = uniform( 'ivec2' ).setName( 'iPreviousFrameUpscaleSize' ) as N;
	readonly iMaxRenderSize = uniform( 'ivec2' ).setName( 'iMaxRenderSize' ) as N;
	readonly iMaxUpscaleSize = uniform( 'ivec2' ).setName( 'iMaxUpscaleSize' ) as N;
	readonly fDeviceToViewDepth = uniform( 'vec4' ).setName( 'fDeviceToViewDepth' ) as N;
	readonly fJitter = uniform( 'vec2' ).setName( 'fJitter' ) as N;
	readonly fPreviousFrameJitter = uniform( 'vec2' ).setName( 'fPreviousFrameJitter' ) as N;
	readonly fMotionVectorScale = uniform( 'vec2' ).setName( 'fMotionVectorScale' ) as N;
	readonly fDownscaleFactor = uniform( 'vec2' ).setName( 'fDownscaleFactor' ) as N;
	readonly fMotionVectorJitterCancellation = uniform( 'vec2' ).setName( 'fMotionVectorJitterCancellation' ) as N;
	readonly fTanHalfFOV = uniform( 'float' ).setName( 'fTanHalfFOV' ) as N;
	readonly fJitterSequenceLength = uniform( 'float' ).setName( 'fJitterSequenceLength' ) as N;
	readonly fDeltaTime = uniform( 'float' ).setName( 'fDeltaTime' ) as N;
	readonly fDeltaPreExposure = uniform( 'float' ).setName( 'fDeltaPreExposure' ) as N;
	readonly fViewSpaceToMetersFactor = uniform( 'float' ).setName( 'fViewSpaceToMetersFactor' ) as N;
	readonly fFrameIndex = uniform( 'float' ).setName( 'fFrameIndex' ) as N;
	readonly fVelocityFactor = uniform( 'float' ).setName( 'fVelocityFactor' ) as N;
	readonly fReactivenessScale = uniform( 'float' ).setName( 'fReactivenessScale' ) as N;
	readonly fShadingChangeScale = uniform( 'float' ).setName( 'fShadingChangeScale' ) as N;
	readonly fAccumulationAddedPerFrame = uniform( 'float' ).setName( 'fAccumulationAddedPerFrame' ) as N;
	readonly fMinDisocclusionAccumulation = uniform( 'float' ).setName( 'fMinDisocclusionAccumulation' ) as N;

	// cbSPD
	readonly mips = uniform( 'uint' ).setName( 'mips' ) as N;
	readonly numWorkGroups = uniform( 'uint' ).setName( 'numWorkGroups' ) as N;
	readonly workGroupOffset = uniform( 'uvec2' ).setName( 'workGroupOffset' ) as N;
	readonly renderSize = uniform( 'uvec2' ).setName( 'renderSize' ) as N;

	// cbRCAS
	readonly rcasConfig = uniform( 'uvec4' ).setName( 'rcasConfig' ) as N;

	// cbGenerateReactive (the reactive-mask generator)
	readonly gen_reactive_scale = uniform( 'float' ).setName( 'gen_reactive_scale' ) as N;
	readonly gen_reactive_threshold = uniform( 'float' ).setName( 'gen_reactive_threshold' ) as N;
	readonly gen_reactive_binaryValue = uniform( 'float' ).setName( 'gen_reactive_binaryValue' ) as N;
	readonly gen_reactive_flags = uniform( 'uint' ).setName( 'gen_reactive_flags' ) as N;

	constructor( desc: Fsr3BindingsDescription ) {

		this.flags = desc.flags;
		this.textures = desc.textures;
		this.inputs = desc.inputs;
		this.reconstructedDepthBuffer = desc.buffers.reconstructedPrevNearestDepth;
		this.frameInfoBuffer = desc.buffers.frameInfo;
		const aliases = getResourceAliases( 0 );
		this.srvAliases = aliases.srv;
		this.uavAliases = aliases.uav;

	}

	/** Copies a frame plan's constants into the uniforms and binds this frame's ping-pong sides. */
	setFrame( plan: Fsr3FramePlan ): void {

		const c = plan.constants;
		this.iRenderSize.value.set( c.renderSize[ 0 ], c.renderSize[ 1 ] );
		this.iPreviousFrameRenderSize.value.set( c.previousFrameRenderSize[ 0 ], c.previousFrameRenderSize[ 1 ] );
		this.iUpscaleSize.value.set( c.upscaleSize[ 0 ], c.upscaleSize[ 1 ] );
		this.iPreviousFrameUpscaleSize.value.set( c.previousFrameUpscaleSize[ 0 ], c.previousFrameUpscaleSize[ 1 ] );
		this.iMaxRenderSize.value.set( c.maxRenderSize[ 0 ], c.maxRenderSize[ 1 ] );
		this.iMaxUpscaleSize.value.set( c.maxUpscaleSize[ 0 ], c.maxUpscaleSize[ 1 ] );
		this.fDeviceToViewDepth.value.set( c.deviceToViewDepth[ 0 ], c.deviceToViewDepth[ 1 ], c.deviceToViewDepth[ 2 ], c.deviceToViewDepth[ 3 ] );
		this.fJitter.value.set( c.jitterOffset[ 0 ], c.jitterOffset[ 1 ] );
		this.fPreviousFrameJitter.value.set( c.previousFrameJitterOffset[ 0 ], c.previousFrameJitterOffset[ 1 ] );
		this.fMotionVectorScale.value.set( c.motionVectorScale[ 0 ], c.motionVectorScale[ 1 ] );
		this.fDownscaleFactor.value.set( c.downscaleFactor[ 0 ], c.downscaleFactor[ 1 ] );
		this.fMotionVectorJitterCancellation.value.set( c.motionVectorJitterCancellation[ 0 ], c.motionVectorJitterCancellation[ 1 ] );
		this.fTanHalfFOV.value = c.tanHalfFOV;
		this.fJitterSequenceLength.value = c.jitterPhaseCount;
		this.fDeltaTime.value = c.deltaTime;
		this.fDeltaPreExposure.value = c.deltaPreExposure;
		this.fViewSpaceToMetersFactor.value = c.viewSpaceToMetersFactor;
		this.fFrameIndex.value = c.frameIndex;
		this.fVelocityFactor.value = c.velocityFactor;
		this.fReactivenessScale.value = c.reactivenessScale;
		this.fShadingChangeScale.value = c.shadingChangeScale;
		this.fAccumulationAddedPerFrame.value = c.accumulationAddedPerFrame;
		this.fMinDisocclusionAccumulation.value = c.minDisocclusionAccumulation;

		this.mips.value = plan.spd.mips;
		this.numWorkGroups.value = plan.spd.numWorkGroups;
		this.workGroupOffset.value.set( plan.spd.workGroupOffset[ 0 ], plan.spd.workGroupOffset[ 1 ] );
		this.renderSize.value.set( plan.spd.renderSize[ 0 ], plan.spd.renderSize[ 1 ] );

		const r = plan.rcas.rcasConfig;
		this.rcasConfig.value.set( r[ 0 ], r[ 1 ], r[ 2 ], r[ 3 ] );

		this.srvAliases = plan.srvAliases;
		this.uavAliases = plan.uavAliases;
		for ( const [ id, node ] of this.srvNodes ) node.value = this.texture( this.srvAliases.get( id ) ?? id );
		for ( const [ key, node ] of this.uavNodes ) {

			const id = Number( key.split( ':' )[ 0 ] );
			node.value = this.texture( this.uavAliases.get( id ) ?? id );

		}

	}

	/** A new reconstructed previous nearest depth buffer (the render-max size changed: a storage
	 * buffer cannot grow in place), bound by the nodes that read and write it (new bind groups, no
	 * new pipeline: the shaders declare runtime-sized arrays). */
	setReconstructedDepthBuffer( buffer: THREE.StorageBufferAttribute ): void {

		this.reconstructedDepthBuffer = buffer;
		for ( const node of [ this.reconstructedDepthRead, this.reconstructedDepthAtomic, this.reconstructedDepthWrite ] ) if ( node !== null ) node.value = buffer;

	}

	/** The reactive-mask generator's constants (ffxFsr3UpscalerContextGenerateReactiveMask). */
	setGenerateReactive( c: Fsr3UpscalerGenerateReactiveConstants ): void {

		this.gen_reactive_scale.value = c.scale;
		this.gen_reactive_threshold.value = c.threshold;
		this.gen_reactive_binaryValue.value = c.binaryValue;
		this.gen_reactive_flags.value = c.flags;

	}

	// -----------------------------------------------------------------------------------------
	// resources

	private texture( id: number ): THREE.Texture {

		const t = this.textures.get( id );
		if ( t === undefined ) throw new Error( `FSR3: no texture bound for resource ${ id }` );
		return t;

	}

	/** The SRV binding of an identifier (an alias resolves to this frame's side). */
	srv( id: number ): N {

		let node = this.srvNodes.get( id );
		if ( node === undefined ) {

			node = texture( this.texture( this.srvAliases.get( id ) ?? id ) );
			this.srvNodes.set( id, node );

		}
		return node;

	}

	/** The UAV binding of an identifier at one mip level. */
	uav( id: number, mip = 0, readWrite = false ): N {

		const key = `${ id }:${ mip }:${ readWrite ? 'rw' : 'w' }`;
		let node = this.uavNodes.get( key );
		if ( node === undefined ) {

			node = storageTexture( this.texture( this.uavAliases.get( id ) ?? id ) as THREE.StorageTexture ) as N;
			if ( mip !== 0 ) node.setMipLevel( mip );
			if ( readWrite ) node.toReadWrite();
			// three r186 shares one binding between texture nodes of the same texture (their uniform
			// hash is the texture's uuid): every mip level's store then went through the first level's
			// view (SPD's levels 1..5 landed in level 0's corner), and a kernel that samples a texture
			// and stores into it would bind it once. Each UAV of a level and access is a binding of
			// its own. (A store goes through a clone of the node, `textureStore`: clones keep the hash.)
			const hash = `:fsr3-uav:${ mip }:${ readWrite ? 'rw' : 'w' }`;
			const own = ( n: N ): void => {

				n.getUniformHash = function ( this: { value: THREE.Texture } ) {

					return this.value.uuid + hash;

				};
				const clone = n.clone;
				n.clone = function ( this: N ) {

					const c = clone.call( this );
					own( c );
					return c;

				};

			};
			own( node );
			this.uavNodes.set( key, node );

		}
		return node;

	}

	/** The size of a resource class (upstream's allocation sizes, from the constants). */
	resourceSize( sizeClass: R.Fsr3SizeClass ): N {

		switch ( sizeClass ) {

			case 'renderMax': return this.iMaxRenderSize;
			case 'renderMaxHalf': return this.iMaxRenderSize.div( 2 );
			case 'upscaleMax': return this.iMaxUpscaleSize;
			case 'fixed': return ivec2( 1, 1 );

		}

	}

	private inside( p: N, size: N ): N {

		const q = ivec2( p );
		return q.x.greaterThanEqual( int( 0 ) ).and( q.y.greaterThanEqual( int( 0 ) ) ).and( q.x.lessThan( size.x ) ).and( q.y.lessThan( size.y ) );

	}

	/** A store that skips texels outside the resource (see the header). */
	private storeIf( node: N, p: N, value: N, size: N ): void {

		const q = ivec2( p ).toVar();
		If( this.inside( q, size ), () => {

			textureStore( node, uvec2( q ), value );

		} );

	}

	/** A load through an SRV: three r186 gives a clone made by `load()` the texture's uv matrix (a
	 * mat3 uniform per load site, updated per dispatch), the identity on FSR's textures; a clone
	 * made by `sample().level()` carries none. */
	private loadAt( node: N, p: N ): N {

		return node.load( p ).setUpdateMatrix( false );

	}

	private srvLoad( id: number, p: N ): N {

		return this.loadAt( this.srv( id ), ivec2( p ) );

	}

	private srvLoadZeroOutside( id: number, p: N ): N {

		const node = this.srv( id );
		const q = ivec2( p ).toVar();
		const size = ivec2( textureSize( node, int( 0 ) ) );
		return select( this.inside( q, size ), this.loadAt( node, q ), vec4( 0 ) );

	}

	private srvSample( id: number, uv: N ): N {

		return this.srv( id ).sample( uv ).level( 0 );

	}

	/** A bilinear clamp-to-edge sample of a non-filterable (r32float) texture, by hand. */
	private srvSampleBilinear( id: number, uv: N ): N {

		const node = this.srv( id );
		const size = ivec2( textureSize( node, int( 0 ) ) ).toVar();
		const fPx = uv.mul( vec2( size ) ).sub( 0.5 ).toVar();
		const fFrac = ffxFract( fPx ).toVar();
		const base = ivec2( floor( fPx ) ).toVar();
		const last = size.sub( 1 );
		const at = ( dx: number, dy: number ) => this.loadAt( node, ivec2( max( ivec2( 0 ), base.add( ivec2( dx, dy ) ).min( last ) ) ) ).x;
		const top = mix( at( 0, 0 ), at( 1, 0 ), fFrac.x );
		const bottom = mix( at( 0, 1 ), at( 1, 1 ), fFrac.x );
		return mix( top, bottom, fFrac.y );

	}

	// -----------------------------------------------------------------------------------------
	// cbFSR3Upscaler getters, in the order they are defined in the CB

	RenderSize(): N { return this.iRenderSize; }
	PreviousFrameRenderSize(): N { return this.iPreviousFrameRenderSize; }
	MaxRenderSize(): N { return this.iMaxRenderSize; }
	UpscaleSize(): N { return this.iUpscaleSize; }
	PreviousFrameUpscaleSize(): N { return this.iPreviousFrameUpscaleSize; }
	MaxUpscaleSize(): N { return this.iMaxUpscaleSize; }
	Jitter(): N { return this.fJitter; }
	PreviousFrameJitter(): N { return this.fPreviousFrameJitter; }
	DeviceToViewSpaceTransformFactors(): N { return this.fDeviceToViewDepth; }
	MotionVectorScale(): N { return this.fMotionVectorScale; }
	DownscaleFactor(): N { return this.fDownscaleFactor; }
	MotionVectorJitterCancellation(): N { return this.fMotionVectorJitterCancellation; }
	TanHalfFoV(): N { return this.fTanHalfFOV; }
	JitterSequenceLength(): N { return this.fJitterSequenceLength; }
	DeltaTime(): N { return this.fDeltaTime; }
	DeltaPreExposure(): N { return this.fDeltaPreExposure; }
	ViewSpaceToMetersFactor(): N { return this.fViewSpaceToMetersFactor; }
	FrameIndex(): N { return this.fFrameIndex; }
	VelocityFactor(): N { return this.fVelocityFactor; }
	AccumulationAddedPerFrame(): N { return this.fAccumulationAddedPerFrame; }
	MinDisocclusionAccumulation(): N { return this.fMinDisocclusionAccumulation; }

	// cbRCAS
	RCASConfig(): N { return this.rcasConfig; }

	// cbGenerateReactive
	GenReactiveScale(): N { return this.gen_reactive_scale; }
	GenReactiveThreshold(): N { return this.gen_reactive_threshold; }
	GenReactiveBinaryValue(): N { return this.gen_reactive_binaryValue; }
	GenReactiveFlags(): N { return this.gen_reactive_flags; }

	// cbSPD
	MipCount(): N { return this.mips; }
	NumWorkGroups(): N { return this.numWorkGroups; }
	WorkGroupOffset(): N { return this.workGroupOffset; }
	SPD_RenderSize(): N { return this.renderSize; }

	// -----------------------------------------------------------------------------------------
	// resource callbacks, in upstream's order

	GetSPDMipDimensions( uMipLevel: number ): N {

		// three r186 drops the level of a storage texture's `textureSize`: level 0's size, halved
		const size0 = ivec2( textureSize( this.srv( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS ), int( 0 ) ) );
		return uMipLevel === 0 ? size0 : max( ivec2( 1 ), size0.shiftRight( ivec2( uMipLevel ) ) );

	}

	SampleSPDMipLevel( fUV: N, mipLevel: number ): N {

		return this.srv( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS ).sample( fUV ).level( float( mipLevel ) ).xy;

	}

	LoadInputDepth( iPxPos: N ): N { return this.inputs.LoadInputDepth( iPxPos ); }

	SampleInputDepth( fUV: N ): N {

		if ( this.inputs.SampleInputDepth === undefined ) throw new Error( 'FSR3: SampleInputDepth not provided' );
		return this.inputs.SampleInputDepth( fUV );

	}

	LoadReactiveMask( iPxPos: N ): N {

		const v = this.inputs.LoadReactiveMask ? this.inputs.LoadReactiveMask( iPxPos ) : float( 0 );
		return v.mul( this.fReactivenessScale );

	}

	GetReactiveMaskResourceDimensions(): N {

		return this.inputs.GetReactiveMaskResourceDimensions ? this.inputs.GetReactiveMaskResourceDimensions() : ivec2( 1, 1 );

	}

	SampleReactiveMask( fUV: N ): N {

		const v = this.inputs.SampleReactiveMask ? this.inputs.SampleReactiveMask( fUV ) : float( 0 );
		return v.mul( this.fReactivenessScale );

	}

	LoadTransparencyAndCompositionMask( iPxPos: N ): N {

		return this.inputs.LoadTransparencyAndCompositionMask ? this.inputs.LoadTransparencyAndCompositionMask( iPxPos ) : float( 0 );

	}

	GetTransparencyAndCompositionMaskResourceDimensions(): N {

		return this.inputs.GetTransparencyAndCompositionMaskResourceDimensions ? this.inputs.GetTransparencyAndCompositionMaskResourceDimensions() : ivec2( 1, 1 );

	}

	SampleTransparencyAndCompositionMask( fUV: N ): N {

		return this.inputs.SampleTransparencyAndCompositionMask ? this.inputs.SampleTransparencyAndCompositionMask( fUV ) : float( 0 );

	}

	LoadInputColor( iPxPos: N ): N { return this.inputs.LoadInputColor( iPxPos ); }

	SampleInputColor( fUV: N ): N {

		if ( this.inputs.SampleInputColor === undefined ) throw new Error( 'FSR3: SampleInputColor not provided' );
		return this.inputs.SampleInputColor( fUV );

	}

	/** The input motion vector in UV space (scaled, jitter cancelled when the option is on). */
	LoadInputMotionVector( o: Fsr3ShaderOptions, iPxDilatedMotionVectorPos: N ): N {

		const fSrcMotionVector = this.inputs.LoadInputMotionVector( iPxDilatedMotionVectorPos );
		let fUvMotionVector = vec2( fSrcMotionVector ).mul( this.MotionVectorScale() );
		if ( o.FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS ) {

			fUvMotionVector = fUvMotionVector.sub( this.MotionVectorJitterCancellation() );

		}
		return fUvMotionVector;

	}

	LoadHistory( iPxHistory: N ): N { return this.srvLoad( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR, iPxHistory ); }

	SampleHistory( fUV: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR, fUV ); }

	StoreLumaHistory( iPxPos: N, fLumaHistory: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY ), iPxPos, fLumaHistory, this.resourceSize( 'renderMax' ) );

	}

	LoadLumaHistory( iPxPos: N ): N { return this.srvLoad( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY, iPxPos ); }

	SampleLumaHistory( fUV: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_HISTORY, fUV ); }

	/** ( RCAS reads one texel past the image's edges: 0 outside the resource, as a D3D load ) */
	LoadRCAS_Input( iPxPos: N ): N { return this.srvLoadZeroOutside( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RCAS_INPUT, iPxPos ); }

	SampleRCAS_Input( fUV: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RCAS_INPUT, fUV ).rgb; }

	StoreReprojectedHistory( iPxHistory: N, fHistory: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR ), iPxHistory, fHistory, this.resourceSize( 'upscaleMax' ) );

	}

	StoreInternalColorAndWeight( iPxPos: N, fColorAndWeight: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR ), iPxPos, fColorAndWeight, this.resourceSize( 'upscaleMax' ) );

	}

	StoreUpscaledOutput( iPxPos: N, fColor: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT ), iPxPos, vec4( fColor, 1.0 ), this.resourceSize( 'upscaleMax' ) );

	}

	SampleAccumulation( fUV: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION, fUV ).x; }

	StoreAccumulation( iPxPos: N, fAccumulation: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_ACCUMULATION ), iPxPos, vec4( fAccumulation, 0, 0, 0 ), this.resourceSize( 'renderMax' ) );

	}

	LoadShadingChange( iPxPos: N ): N { return this.srvLoad( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SHADING_CHANGE, iPxPos ).x.mul( this.fShadingChangeScale ); }

	SampleShadingChange( fUV: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SHADING_CHANGE, fUV ).x.mul( this.fShadingChangeScale ); }

	StoreShadingChange( iPxPos: N, fShadingChange: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SHADING_CHANGE ), iPxPos, vec4( fShadingChange, 0, 0, 0 ), this.resourceSize( 'renderMaxHalf' ) );

	}

	GetFarthestDepthResourceDimensions(): N { return ivec2( textureSize( this.srv( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH ), int( 0 ) ) ); }

	LoadFarthestDepth( iPxPos: N ): N { return this.srvLoad( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH, iPxPos ).x; }

	SampleFarthestDepth( fUV: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH, fUV ).x; }

	StoreFarthestDepth( iPxPos: N, fDepth: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH ), iPxPos, vec4( fDepth, 0, 0, 0 ), this.resourceSize( 'renderMax' ) );

	}

	GetFarthestDepthMip1ResourceDimensions(): N { return ivec2( textureSize( this.srv( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH_MIP1 ), int( 0 ) ) ); }

	LoadFarthestDepthMip1( iPxPos: N ): N { return this.srvLoad( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH_MIP1, iPxPos ).x; }

	SampleFarthestDepthMip1( fUV: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH_MIP1, fUV ).x; }

	StoreFarthestDepthMip1( iPxPos: N, fDepth: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FARTHEST_DEPTH_MIP1 ), iPxPos, vec4( fDepth, 0, 0, 0 ), this.resourceSize( 'renderMaxHalf' ) );

	}

	LoadCurrentLuma( iPxPos: N ): N { return this.srvLoad( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_CURRENT_LUMA, iPxPos ).x; }

	SampleCurrentLuma( uv: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_CURRENT_LUMA, uv ).x; }

	StoreCurrentLuma( iPxPos: N, fLuma: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_CURRENT_LUMA ), iPxPos, vec4( fLuma, 0, 0, 0 ), this.resourceSize( 'renderMax' ) );

	}

	SampleLumaInstability( uv: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_INSTABILITY, uv ).x; }

	StoreLumaInstability( iPxPos: N, fLumaInstability: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LUMA_INSTABILITY ), iPxPos, vec4( fLumaInstability, 0, 0, 0 ), this.resourceSize( 'renderMax' ) );

	}

	LoadPreviousLuma( iPxPos: N ): N { return this.srvLoad( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_PREVIOUS_LUMA, iPxPos ).x; }

	SamplePreviousLuma( uv: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_PREVIOUS_LUMA, uv ).x; }

	LoadNewLocks( iPxPos: N ): N { return this.srvLoad( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NEW_LOCKS, iPxPos ).x; }

	LoadRwNewLocks( iPxPos: N ): N {

		return this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NEW_LOCKS, 0, true ).load( uvec2( ivec2( iPxPos ) ) ).x;

	}

	/** Rounds to 8 bits as upstream's R8_UNORM store would (the texture is r32float, see the header). */
	StoreNewLocks( iPxPos: N, newLock: N ): void {

		const unorm8 = round( saturate( float( newLock ) ).mul( 255.0 ) ).div( 255.0 );
		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NEW_LOCKS, 0, true ), iPxPos, vec4( unorm8, 0, 0, 0 ), this.resourceSize( 'upscaleMax' ) );

	}

	private reconstructedDepthIndex( p: N ): N {

		const q = ivec2( p );
		return uint( q.y ).mul( uint( this.iMaxRenderSize.x ) ).add( uint( q.x ) );

	}

	LoadReconstructedPrevDepth( iPxPos: N ): N {

		if ( this.reconstructedDepthRead === null ) {

			this.reconstructedDepthRead = storage( this.reconstructedDepthBuffer, 'uint', this.reconstructedDepthBuffer.count ).toReadOnly();

		}
		const q = ivec2( iPxPos ).toVar();
		const v = this.reconstructedDepthRead.element( this.reconstructedDepthIndex( q ) );
		return ffxAsFloat( select( this.inside( q, this.iMaxRenderSize ), v, uint( 0 ) ) );

	}

	StoreReconstructedDepth( o: Fsr3ShaderOptions, iPxSample: N, fDepth: N ): void {

		if ( this.reconstructedDepthAtomic === null ) {

			this.reconstructedDepthAtomic = storage( this.reconstructedDepthBuffer, 'uint', this.reconstructedDepthBuffer.count ).toAtomic();

		}
		const uDepth = ffxAsUInt32( fDepth ).toVar();
		const q = ivec2( iPxSample ).toVar();
		If( this.inside( q, this.iMaxRenderSize ), () => {

			const e = this.reconstructedDepthAtomic.element( this.reconstructedDepthIndex( q ) );
			if ( o.FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH ) {

				atomicMax( e, uDepth ).toStack();

			} else {

				atomicMin( e, uDepth ).toStack(); // min for standard, max for inverted depth

			}

		} );

	}

	SetReconstructedDepth( iPxSample: N, uValue: N ): void {

		if ( this.reconstructedDepthWrite === null ) {

			this.reconstructedDepthWrite = storage( this.reconstructedDepthBuffer, 'uint', this.reconstructedDepthBuffer.count );

		}
		const q = ivec2( iPxSample ).toVar();
		If( this.inside( q, this.iMaxRenderSize ), () => {

			this.reconstructedDepthWrite.element( this.reconstructedDepthIndex( q ) ).assign( uValue );

		} );

	}

	StoreDilatedDepth( iPxPos: N, fDepth: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_DEPTH ), iPxPos, vec4( fDepth, 0, 0, 0 ), this.resourceSize( 'renderMax' ) );

	}

	StoreDilatedMotionVector( iPxPos: N, fMotionVector: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS ), iPxPos, vec4( fMotionVector, 0, 0 ), this.resourceSize( 'renderMax' ) );

	}

	LoadDilatedMotionVector( iPxInput: N ): N { return this.srvLoadZeroOutside( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS, iPxInput ).xy; }

	SampleDilatedMotionVector( fUV: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS, fUV ).xy; }

	LoadDilatedDepth( iPxInput: N ): N { return this.srvLoadZeroOutside( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_DEPTH, iPxInput ).x; }

	SampleDilatedDepth( fUV: N ): N { return this.srvSampleBilinear( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_DEPTH, fUV ); }

	/** r_input_exposure[ 0, 0 ].x with upstream's fallback (0 -> 1); with auto exposure the input is FRAME_INFO. */
	Exposure(): N {

		let exposureIn: N;
		if ( this.flags & FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE ) {

			exposureIn = this.FrameInfo().x;

		} else {

			exposureIn = this.inputs.LoadInputExposure ? this.inputs.LoadInputExposure() : float( 0 );

		}
		// upstream's PC build maps only 0 to 1 (its `< 2^-15` test is the Xbox build's)
		const e0 = float( exposureIn ).toVar();
		return select( e0.equal( 0.0 ), float( 1.0 ), e0 );

	}

	SampleLanczos2Weight( x: N ): N {

		return this.srvSampleBilinear( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LANCZOS_LUT, vec2( x.div( 2 ), 0.5 ) );

	}

	SampleDilatedReactiveMasks( fUV: N ): N { return this.srvSample( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS, fUV ); }

	StoreDilatedReactiveMasks( iPxPos: N, fDilatedReactiveMasks: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS ), iPxPos, fDilatedReactiveMasks, this.resourceSize( 'renderMax' ) );

	}

	LoadOpaqueOnly( iPxPos: N ): N {

		if ( this.inputs.LoadOpaqueOnly === undefined ) throw new Error( 'FSR3: LoadOpaqueOnly not provided' );
		return this.inputs.LoadOpaqueOnly( iPxPos );

	}

	/** The reactive-mask generator's output (its pass file stores straight into rw_output_autoreactive). */
	StoreAutoReactiveValue( iPxPos: N, fReactive: N ): void {

		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_AUTOREACTIVE ), iPxPos, vec4( fReactive, 0, 0, 0 ), this.resourceSize( 'renderMax' ) );

	}

	LoadFrameInfo(): N {

		return this.frameInfoRW().element( 0 );

	}

	StoreFrameInfo( fInfo: N ): void {

		this.frameInfoRW().element( 0 ).assign( fInfo );

	}

	FrameInfo(): N {

		if ( this.frameInfoRead === null ) this.frameInfoRead = storage( this.frameInfoBuffer, 'vec4', 1 ).toReadOnly();
		return this.frameInfoRead.element( 0 );

	}

	/** FRAME_INFO for the clear kernel and the luminance pyramid. */
	frameInfoRW(): N {

		if ( this.frameInfoReadWrite === null ) this.frameInfoReadWrite = storage( this.frameInfoBuffer, 'vec4', 1 );
		return this.frameInfoReadWrite;

	}

	/** SPD_MIPS level `index` (0..5) for the pyramids' stores; levels above 5 store nothing, as
	 * upstream's StorePyramid has no UAV for them. */
	StorePyramid( iPxPos: N, outValue: N, index: number ): void {

		if ( index > 5 ) return;
		const size = max( ivec2( 1 ), this.resourceSize( 'renderMaxHalf' ).shiftRight( ivec2( index ) ) );
		this.storeIf( this.uav( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS, index ), iPxPos, vec4( outValue, 0, 0 ), size );

	}

	/** RWLoadPyramid( p, 5 ): the tail's read of level 5, through the tail dispatch's sampled binding. */
	RWLoadPyramid( iPxPos: N, index: number ): N {

		const node = this.srv( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS );
		const q = ivec2( iPxPos ).toVar();
		// level `index`'s own size (a storage texture's `textureSize` is level 0's in r186), so
		// the tail reads 0 past it as upstream's UAV load does
		const size = this.spdMipSize( index );
		return select( this.inside( q, size ), node.load( q ).level( int( index ) ), vec4( 0 ) ).xy;

	}

	/** The size of SPD_MIPS level `index`, for the clear kernel. */
	spdMipSize( index: number ): N {

		return max( ivec2( 1 ), this.resourceSize( 'renderMaxHalf' ).shiftRight( ivec2( index ) ) );

	}

	/** A plain guarded store into a resource by identifier (the clear kernels). */
	storeResource( id: number, mip: number, iPxPos: N, value: N, size: N ): void {

		this.storeIf( this.uav( id, mip ), iPxPos, value, size );

	}

}

