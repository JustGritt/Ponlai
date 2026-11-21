import { BackSide, DoubleSide, FrontSide, NodeMaterial, NoBlending } from 'three/webgpu';
import type { Camera, LightShadow, Material, Node, Renderer, RenderTarget, Scene } from 'three/webgpu';
import { vec4 } from 'three/tsl';
import { missing } from './Internals.ts';

// A light's shadow as three draws it: the shadow node, its map and the caster's material.
//
// A light's shadow as three r186 draws it (lighting/SunShadow.ts): the shadow node that draws its
// map, the map it made, and the material a caster is drawn with into it. three draws a caster with
// the shadow node's material in its own's place (scene.overrideMaterial), taking the caster's
// colour, depth and position nodes and side into it per draw and restoring them after
// (Renderer.renderObject); `shadowPassMaterial` makes that material, as the draw leaves it, so the
// pipeline can be precompiled before the draw. It mirrors three's shadow material (ShadowBaseNode) and its non-VSM
// side rule (`SHADOW_SIDE`).

/** A light's shadow node as three draws its map with it. */
export type ShadowRenderer = { updateShadow( frame: { renderer: Renderer; scene: Scene; camera: Camera } ): void; shadowMap: RenderTarget | null };

/** The node drawing `shadow`'s map, once three has made it (with the first receiver's build). */
export function shadowRenderer( shadow: LightShadow ): ShadowRenderer | undefined {

	return shadow.shadowNode as unknown as ShadowRenderer | undefined;

}

/** The map `node` draws into, once three has made it. */
export function builtShadowMap( node: ShadowRenderer ): RenderTarget | null {

	return node.shadowMap;

}

/** Redraws the map of `node` as three's shadow pass does. */
export function redrawShadow( node: ShadowRenderer, renderer: Renderer, scene: Scene, camera: Camera ): void {

	node.updateShadow( { renderer, scene, camera } );

}

/** The side three draws a caster's faces into a (non-VSM) map with, by the material's side
 * (Renderer's `_shadowSide`). */
export const SHADOW_SIDE = { [ FrontSide ]: BackSide, [ BackSide ]: FrontSide, [ DoubleSide ]: DoubleSide };

type ShadowNodes = { colorNode: Node<'vec4'> | null; depthNode: Node<'float'> | null; positionNode: Node<'vec3'> | null };

/** three's shadow material as a draw of `material`'s caster into a map leaves it: its colour, depth
 * and position nodes (the renderer's `_getShadowNodes`) and side taken into it. */
export function shadowPassMaterial( renderer: Renderer, material: NodeMaterial ): NodeMaterial {

	const shadowNodes = ( renderer as unknown as { _getShadowNodes?( material: Material ): ShadowNodes } )._getShadowNodes;
	if ( typeof shadowNodes !== 'function' ) throw missing( 'Renderer._getShadowNodes', 'shadowPassMaterial (packages/render/three/ShadowPass.ts)' );
	const pass = new NodeMaterial();
	( pass as unknown as { isShadowPassMaterial: boolean } ).isShadowPassMaterial = true;
	pass.name = 'ShadowMaterial';
	pass.blending = NoBlending;
	pass.fog = false;
	pass.colorNode = vec4( 0, 0, 0, 1 );
	const nodes = shadowNodes.call( renderer, material );
	if ( material.positionNode ) pass.positionNode = material.positionNode;
	if ( nodes.colorNode ) pass.colorNode = nodes.colorNode;
	if ( nodes.depthNode ) pass.depthNode = nodes.depthNode;
	if ( nodes.positionNode ) pass.positionNode = nodes.positionNode;
	pass.side = material.shadowSide ?? SHADOW_SIDE[ material.side ];
	pass.alphaTest = material.alphaTest;
	( pass as unknown as { alphaMap: unknown } ).alphaMap = ( material as unknown as { alphaMap: unknown } ).alphaMap;
	pass.transparent = material.transparent;
	return pass;

}
