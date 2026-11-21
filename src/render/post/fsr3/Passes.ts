/*!
 * Adapted from the AMD FidelityFX SDK 2.3.0, FSR 3 upscaler 3.1.5 (upscalers/fsr3/internal/shaders/
 * ffx_fsr3upscaler_*_pass.hlsl, and the clear jobs of internal/ffx_fsr3upscaler.cpp).
 * SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

import type * as THREE from 'three/webgpu';
import { Fn, If, Return, uint, vec2, vec4, ivec2, uvec2, workgroupId, localId, globalId } from './Tsl.ts';
import { ffxAsUInt32Cpu, type N } from './Core.ts';
import {
	FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS, FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID, FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID,
	FFX_FSR3UPSCALER_PASS_SHADING_CHANGE, FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY, FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY,
	FFX_FSR3UPSCALER_PASS_ACCUMULATE, FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN, FFX_FSR3UPSCALER_PASS_RCAS,
	FFX_FSR3UPSCALER_PASS_DEBUG_VIEW, FFX_FSR3UPSCALER_PASS_GENERATE_REACTIVE,
	getPassOptions, type Fsr3ClearJob,
} from './Host.ts';
import * as R from './Resources.ts';
import { ffxSpdSetup } from './Spd.ts';
import type { Fsr3Bindings } from './Bindings.ts';
import { PrepareInputs } from './PrepareInputs.ts';
import { ComputeAutoExposure, ComputeAutoExposureTail } from './LumaPyramid.ts';
import { ComputeShadingChangePyramid, ComputeShadingChangePyramidTail } from './ShadingChangePyramid.ts';
import { ShadingChange, s_MipLevelsToUse } from './ShadingChange.ts';
import { iShadingChangeMipStart } from './Common.ts';
import { PrepareReactivity } from './PrepareReactivity.ts';
import { LumaInstability } from './LumaInstability.ts';
import { AccumulatePixel } from './Accumulate.ts';
import { RCAS } from './Rcas.ts';
import { DebugView } from './DebugView.ts';
import { GenerateReactive } from './AutogenReactive.ts';

// FSR 3's pass entry points: one compute kernel per upstream pass, with its permutation.
//
// The pass entry points: one compute kernel per `*_pass.hlsl` with upstream's workgroup size
// (8 x 8, SPD's 256 x 1, RCAS's 64 x 1) and the permutation upstream builds it with
// (`getPipelinePermutationFlags`: HDR colour, low-resolution and jittered motion vectors, inverted
// depth, sharpening for ACCUMULATE_SHARPEN, the reprojection's Lanczos type), the SPD tails as
// kernels of their own, and kernels for upstream's clear jobs.
//
// How the integration dispatches them (three r186): `renderer.compute( kernel, [ x, y, 1 ] )` with
// the workgroup counts of `getFixedDispatchSize` (from the context's max render and upscale sizes,
// the canvas's: fixed per canvas, so no pipeline changes with the render size); three passes an
// array dispatch size straight to `dispatchWorkgroups` and bakes no count into the shader (only
// a numeric `count` adds a bounds check). The kernels read the live sizes from the uniforms and
// skip whole workgroups past upstream's own workgroup counts for this frame (`Host.ts`'s plan
// lists them), so they write exactly the texels upstream's dispatches write; threads of a
// partial edge group run, as upstream's do, and stores outside a resource are dropped
// (Bindings.ts). Workgroup ids, the local index (SV_GroupIndex = local_invocation_id.x for the
// 1D groups) and the global id are WGSL's builtins. The skip is an early return on the workgroup
// id and uniform values, so the SPD kernels' barriers stay in uniform control flow.
//
// Per frame, in order: `Host.ts` `Fsr3UpscalerContext.dispatch` -> `Bindings.setFrame( plan )` ->
// the plan's clears (`clear( job )`) -> each job's kernel (`pass( id )`), the SPD tail
// (`lumaPyramidTail`) right after the luminance pyramid when `plan.spdTail` (the shading-change
// tail stores nothing, ShadingChangePyramid.ts: the integration skips it, Fsr3Upscaler.ts) -> RCAS when sharpening.
//
// Differences from upstream: the early-outs above (upstream sizes its dispatches instead); the SPD
// tails; the SPD_MIPS clear clears only the levels some pass reads (0..2 and 5); the clears are kernels (a WebGPU clear exists only for render attachments and buffers),
// one per target, writing the values Host.ts lists; the Xbox variants (paired 16-bit
// accumulation, wave32, scheduler hints) and the root signatures are not ported.

// SPD_MIPS's levels some pass reads: the shading change's ( iShadingChangeMipStart ..
// s_MipLevelsToUse - 1 ) and the luminance pyramid tail's 5
const SPD_LEVELS_READ = [ ...Array.from( { length: s_MipLevelsToUse - iShadingChangeMipStart }, ( _, i ) => iShadingChangeMipStart + i ), 5 ];

export interface Fsr3PassKernels {
	/** the kernel of a pass ( FFX_FSR3UPSCALER_PASS_* ) */
	pass( passId: number ): THREE.ComputeNode;
	lumaPyramidTail(): THREE.ComputeNode;
	shadingChangePyramidTail(): THREE.ComputeNode;
	/** the kernel of a clear job ( its target and values ) */
	clear( job: Fsr3ClearJob ): THREE.ComputeNode;
	/** the workgroup counts to dispatch a kernel with, fixed per context ( see the header ) */
	getFixedDispatchSize( kernel: THREE.ComputeNode ): [ number, number, number ];
}

type Size = { width: number, height: number };

/** Builds the pass and clear kernels of a context on its bindings. `useLut` picks the Lanczos LUT
 * for the history reprojection (upstream's choice on AMD hardware; Host.ts). */
export function createPassKernels( b: Fsr3Bindings, maxRenderSize: Size, maxUpscaleSize: Size, useLut = false ): Fsr3PassKernels {

	const flags = b.flags;
	const kernels = new Map<number, THREE.ComputeNode>();
	const clears = new Map<string, THREE.ComputeNode>();
	const dispatchSizes = new Map<THREE.ComputeNode, [ number, number, number ]>();
	let lumaTail: THREE.ComputeNode | null = null;
	let shadingTail: THREE.ComputeNode | null = null;

	const groups = ( n: number, d: number ) => Math.trunc( ( n + d - 1 ) / d );
	const src: [ number, number, number ] = [ groups( maxRenderSize.width, 8 ), groups( maxRenderSize.height, 8 ), 1 ];
	const dst: [ number, number, number ] = [ groups( maxUpscaleSize.width, 8 ), groups( maxUpscaleSize.height, 8 ), 1 ];
	const half: [ number, number, number ] = [ groups( Math.trunc( Math.fround( maxRenderSize.width * 0.5 ) ), 8 ), groups( Math.trunc( Math.fround( maxRenderSize.height * 0.5 ) ), 8 ), 1 ];
	const spdSetup = ffxSpdSetup( [ 0, 0, maxRenderSize.width, maxRenderSize.height ] );
	const spd: [ number, number, number ] = [ spdSetup.dispatchThreadGroupCountXY[ 0 ], spdSetup.dispatchThreadGroupCountXY[ 1 ], 1 ];
	const rcas: [ number, number, number ] = [ groups( maxUpscaleSize.width, 16 ), groups( maxUpscaleSize.height, 16 ), 1 ];

	// upstream's workgroup counts for this frame, from the live sizes
	const srcGroups = () => ivec2( b.RenderSize() ).add( 7 ).div( 8 );
	const dstGroups = () => ivec2( b.UpscaleSize() ).add( 7 ).div( 8 );
	const shadingChangeGroups = () => ivec2( vec2( b.RenderSize() ).mul( 0.5 ) ).add( 7 ).div( 8 );
	const spdGroups = () => ivec2( uvec2( b.SPD_RenderSize() ).sub( 1 ).div( 64 ).add( 1 ).sub( b.WorkGroupOffset() ) );
	const rcasGroups = () => ivec2( b.UpscaleSize() ).add( 15 ).div( 16 );

	const skipWorkgroupsOutside = ( count: N ) => {

		const g = ivec2( workgroupId.xy );
		If( g.x.greaterThanEqual( count.x ).or( g.y.greaterThanEqual( count.y ) ), () => {

			Return();

		} );

	};

	const kernel = ( name: string, size: number[], dispatch: [ number, number, number ], body: () => void ): THREE.ComputeNode => {

		const k = Fn( body )().computeKernel( size ).setName( `FSR3.${ name }` ) as THREE.ComputeNode;
		dispatchSizes.set( k, dispatch );
		return k;

	};

	const build = ( passId: number ): THREE.ComputeNode => {

		const o = getPassOptions( flags, passId, useLut );
		switch ( passId ) {

			case FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS: return kernel( 'PrepareInputs', [ 8, 8 ], src, () => {

				skipWorkgroupsOutside( srcGroups() );
				PrepareInputs( b, o, ivec2( globalId.xy ) );

			} );
			case FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID: return kernel( 'LumaPyramid', [ 256 ], spd, () => {

				skipWorkgroupsOutside( spdGroups() );
				ComputeAutoExposure( b, workgroupId, localId.x );

			} );
			case FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID: return kernel( 'ShadingChangePyramid', [ 256 ], spd, () => {

				skipWorkgroupsOutside( spdGroups() );
				ComputeShadingChangePyramid( b, workgroupId, localId.x );

			} );
			case FFX_FSR3UPSCALER_PASS_SHADING_CHANGE: return kernel( 'ShadingChange', [ 8, 8 ], half, () => {

				skipWorkgroupsOutside( shadingChangeGroups() );
				ShadingChange( b, ivec2( globalId.xy ) );

			} );
			case FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY: return kernel( 'PrepareReactivity', [ 8, 8 ], src, () => {

				skipWorkgroupsOutside( srcGroups() );
				PrepareReactivity( b, ivec2( globalId.xy ) );

			} );
			case FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY: return kernel( 'LumaInstability', [ 8, 8 ], src, () => {

				skipWorkgroupsOutside( srcGroups() );
				LumaInstability( b, ivec2( globalId.xy ) );

			} );
			case FFX_FSR3UPSCALER_PASS_ACCUMULATE:
			case FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN: return kernel( passId === FFX_FSR3UPSCALER_PASS_ACCUMULATE ? 'Accumulate' : 'AccumulateSharpen', [ 8, 8 ], dst, () => {

				skipWorkgroupsOutside( dstGroups() );
				AccumulatePixel( b, o, ivec2( globalId.xy ) );

			} );
			case FFX_FSR3UPSCALER_PASS_RCAS: return kernel( 'RCAS', [ 64 ], rcas, () => {

				skipWorkgroupsOutside( rcasGroups() );
				RCAS( b, localId, workgroupId );

			} );
			case FFX_FSR3UPSCALER_PASS_DEBUG_VIEW: return kernel( 'DebugView', [ 8, 8 ], dst, () => {

				skipWorkgroupsOutside( dstGroups() );
				DebugView( b, ivec2( globalId.xy ) );

			} );
			case FFX_FSR3UPSCALER_PASS_GENERATE_REACTIVE: return kernel( 'GenerateReactive', [ 8, 8 ], src, () => {

				// ( the generator's own render size is the description's; the integration passes the
				// upscaler's, so the live count reads RenderSize() )
				skipWorkgroupsOutside( srcGroups() );
				GenerateReactive( b, ivec2( workgroupId.xy.mul( uvec2( 8, 8 ) ).add( localId.xy ) ) );

			} );
			default: throw new Error( `FSR3: no kernel for pass ${ passId }` );

		}

	};

	// a clear kernel writes `color` into every texel of a target (each mip level of SPD_MIPS)
	const buildClear = ( job: Fsr3ClearJob ): THREE.ComputeNode => {

		const [ r, g, bl, a ] = job.color;
		switch ( job.target ) {

			case R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_FRAME_INFO: return kernel( 'ClearFrameInfo', [ 1 ], [ 1, 1, 1 ], () => {

				b.frameInfoRW().element( 0 ).assign( vec4( r, g, bl, a ) );

			} );
			case R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH: {

				// the bits of the float value (Host.ts: what the backends clear this R32_UINT to)
				const bits = ffxAsUInt32Cpu( r ); // 1.0 -> 0x3f800000, or 0 when inverted
				return kernel( 'ClearReconstructedPrevNearestDepth', [ 8, 8 ], src, () => {

					b.SetReconstructedDepth( ivec2( globalId.xy ), uint( bits ) );

				} );

			}
			case R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS: {

				const size: [ number, number, number ] = [ groups( Math.trunc( maxRenderSize.width / 2 ), 8 ), groups( Math.trunc( maxRenderSize.height / 2 ), 8 ), 1 ];
				return kernel( 'ClearSpdMips', [ 8, 8 ], size, () => {

					const p = ivec2( globalId.xy );
					// (the levels anything reads: 0..2, the shading change's, and 5, the luminance tail's;
					// 3 and 4 are neither stored nor read, ShadingChangePyramid.ts, and a kernel binds at
					// most four storage textures by default)
					for ( const mip of SPD_LEVELS_READ ) b.storeResource( R.FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_SPD_MIPS, mip, p, vec4( r, g, bl, a ), b.spdMipSize( mip ) );

				} );

			}
			default: {

				const desc = R.FSR3_INTERNAL_RESOURCES.find( ( d ) => d.id === job.target );
				if ( desc === undefined ) throw new Error( `FSR3: no clear for resource ${ job.target }` );
				const s = R.getResourceSize( desc, maxRenderSize, maxUpscaleSize );
				return kernel( `Clear.${ desc.name }`, [ 8, 8 ], [ groups( s.width, 8 ), groups( s.height, 8 ), 1 ], () => {

					b.storeResource( job.target, 0, ivec2( globalId.xy ), vec4( r, g, bl, a ), b.resourceSize( desc.sizeClass ) );

				} );

			}

		}

	};

	return {

		pass( passId: number ): THREE.ComputeNode {

			let k = kernels.get( passId );
			if ( k === undefined ) {

				k = build( passId );
				kernels.set( passId, k );

			}
			return k;

		},

		lumaPyramidTail(): THREE.ComputeNode {

			if ( lumaTail === null ) lumaTail = kernel( 'LumaPyramidTail', [ 256 ], [ 1, 1, 1 ], () => ComputeAutoExposureTail( b, localId.x ) );
			return lumaTail;

		},

		shadingChangePyramidTail(): THREE.ComputeNode {

			if ( shadingTail === null ) shadingTail = kernel( 'ShadingChangePyramidTail', [ 256 ], [ 1, 1, 1 ], () => ComputeShadingChangePyramidTail( b, localId.x ) );
			return shadingTail;

		},

		clear( job: Fsr3ClearJob ): THREE.ComputeNode {

			const key = `${ job.target }:${ job.color.join( ',' ) }`;
			let k = clears.get( key );
			if ( k === undefined ) {

				k = buildClear( job );
				clears.set( key, k );

			}
			return k;

		},

		getFixedDispatchSize( k: THREE.ComputeNode ): [ number, number, number ] {

			const d = dispatchSizes.get( k );
			if ( d === undefined ) throw new Error( 'FSR3: not a kernel of this context' );
			return d;

		},

	};

}
