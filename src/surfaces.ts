// Surface materials for the scene's own objects, lit the way the render pipeline lights everything.
//
// Every opaque surface here uses MeshStandardNodeMaterial with the pipeline's underwater lighting
// model, which lights it like any other surface above the water and by the light carried down the
// water column below it. It also writes the metering illuminance, so the auto exposure sees the
// object. The same setup is used by the pipeline's open-sea test scene (app/scenes/sea.ts upstream).
import * as THREE from 'three/webgpu';
import { color } from 'three/tsl';
import { shared, underwaterLight, UnderwaterLightingModel, writeMeter, type RenderParts } from './render/index.ts';

export interface SurfaceOptions {
  name: string;
  /** linear albedo, or a node that computes it */
  color: THREE.Color | THREE.Node<'vec3'>;
  roughness?: number | THREE.Node<'float'>;
  metalness?: number;
  /** scene-linear radiance the surface emits (lamps, lit windows) */
  emissive?: THREE.Node<'vec3'>;
  side?: THREE.Side;
}

/** A lit surface material that reads correctly above and below the sea. */
export function surface( render: RenderParts, o: SurfaceOptions ): THREE.MeshStandardNodeMaterial {

  const material = new THREE.MeshStandardNodeMaterial( { metalness: o.metalness ?? 0, side: o.side ?? THREE.FrontSide } );
  material.name = o.name;
  if ( o.color instanceof THREE.Color ) material.color.copy( o.color );
  else material.colorNode = o.color;
  if ( typeof o.roughness === 'number' || o.roughness === undefined ) material.roughness = o.roughness ?? 0.8;
  else material.roughnessNode = o.roughness;
  if ( o.emissive ) material.emissiveNode = o.emissive;
  const light = underwaterLight( shared, render.waveMaps );
  ( material as unknown as { setupLightingModel: () => THREE.LightingModel } ).setupLightingModel = () => new UnderwaterLightingModel( light );
  writeMeter( material );
  return material;

}

/** An sRGB hex colour as linear albedo. */
export function albedo( hex: number ): THREE.Color {

  return new THREE.Color().setHex( hex, THREE.SRGBColorSpace );

}

/** An sRGB hex colour as a linear albedo node, for colours computed in a shader. */
export function rgb( hex: number ): THREE.Node<'vec3'> {

  return color( albedo( hex ) ) as unknown as THREE.Node<'vec3'>;

}
