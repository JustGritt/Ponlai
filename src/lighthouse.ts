// The lighthouse on the stack: a banded tower, its gallery and lantern, a keeper's cottage, and the
// light itself, which turns two beams through the air by night.
//
// By night (`night`, 0 by day to 1 by night, from the sky's own night-glow share) the lamp glows, the
// windows light up, a local light warms the stone round the lantern and the beams sweep. The beams
// are cones drawn additively: a fake volumetric, brightest along its axis as seen (the view
// direction against the cone's normal), fading with distance from the lamp. Facing a beam, the lamp
// flares, which reads as the lighthouse's flash.
import * as THREE from 'three/webgpu';
import {
  abs, dot, float, max, mix, normalize, normalView, positionGeometry, positionLocal, positionView, positionWorld, cameraPosition,
  pow, smoothstep, step, uniform, vec3, vec4,
} from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { RenderParts } from './render/index.ts';
import { albedo, rgb, surface } from './surfaces.ts';

// The tower's parts, in metres above its foot.
const TOWER = { height: 14, baseRadius: 2.3, topRadius: 1.75 };
const GALLERY_Y = TOWER.height;
const LAMP_Y = GALLERY_Y + 1.9;
/** the lamp's height above the tower's foot */
export const LAMP_HEIGHT = LAMP_Y;
// the beams: length and the radius they open to (m), turns per second
const BEAM = { length: 150, radius: 11, turn: 1 / 12 };

/** The night's glows, in the scene's radiance units (render/lighting/Radiometry.ts), at full
 * night. A clear night meters at about EV 19 (2^19 from scene radiance to the exposed image), so
 * a soft glow is a few millionths: the island's glowing plants emit about 4e-6. A page can
 * override each one for tuning, as `?glow.beam=5e-6`. */
export const GLOW = {
  /** the lamp's own radiance; the flash multiplies it as a beam faces the eye */
  lamp: uniform( 3e-4 ),
  /** the lens round it */
  lens: uniform( 2e-5 ),
  /** the air lit along a beam's axis, at the lamp */
  beam: uniform( 4e-6 ),
  /** the lit windows */
  windows: uniform( 6e-6 ),
};
for ( const [ key, node ] of Object.entries( GLOW ) ) {

  const value = Number( new URLSearchParams( location.search ).get( `glow.${ key }` ) );
  if ( value > 0 ) node.value = value;

}

export interface Lighthouse {
  group: THREE.Group;
  /** the lamp's world position */
  lamp: THREE.Vector3;
  /** 0 by day, 1 by night */
  night: THREE.UniformNode<'float', number>;
  /** turns the beams to the scene time (s) */
  update( time: number ): void;
  /** what casts into the sun's shadow map */
  bounds(): THREE.Box3;
}

/** Builds the lighthouse with its foot at `foot` (the stack's top). */
export function createLighthouse( render: RenderParts, foot: THREE.Vector3 ): Lighthouse {

  const night = uniform( 0 );
  const group = new THREE.Group();
  group.name = 'lighthouse';
  group.position.copy( foot );

  const iron = albedo( 0x23272a );
  // windows: dark glass by day, warm lamplight by night
  const windowGlow = vec3( 1.0, 0.62, 0.3 ).mul( night.mul( GLOW.windows ) );

  // ---- the tower: white with two red bands, a slight taper
  const towerProfile = [
    new THREE.Vector2( 0, 0 ), new THREE.Vector2( TOWER.baseRadius + 0.35, 0 ), new THREE.Vector2( TOWER.baseRadius + 0.35, 0.6 ),
    new THREE.Vector2( TOWER.baseRadius, 0.75 ), new THREE.Vector2( TOWER.topRadius, TOWER.height - 0.4 ),
    new THREE.Vector2( TOWER.topRadius + 0.15, TOWER.height - 0.2 ), new THREE.Vector2( TOWER.topRadius + 0.15, TOWER.height ),
    new THREE.Vector2( 0, TOWER.height ),
  ];
  const y = positionLocal.y;
  const band = step( 4.4, y ).mul( step( y, 6.4 ) ).add( step( 9.0, y ).mul( step( y, 11.0 ) ) );
  const towerColour = mix( rgb( 0xe8e4da ), rgb( 0x9b2a22 ), band );
  const tower = new THREE.Mesh( new THREE.LatheGeometry( towerProfile, 64 ), surface( render, { name: 'tower', color: towerColour, roughness: 0.55 } ) );
  tower.name = 'tower';
  group.add( tower );

  // door and windows: thin boxes set into the tower's face
  const glass = surface( render, { name: 'window', color: albedo( 0x1a1d20 ), roughness: 0.15, emissive: windowGlow } );
  const door = surface( render, { name: 'door', color: albedo( 0x3a2a1e ), roughness: 0.7 } );
  const setInto = ( mesh: THREE.Mesh, height: number, angle: number ) => {

    const r = TOWER.baseRadius + ( TOWER.topRadius - TOWER.baseRadius ) * ( height - 0.75 ) / ( TOWER.height - 1.15 );
    mesh.position.set( Math.sin( angle ) * ( r - 0.02 ), height, Math.cos( angle ) * ( r - 0.02 ) );
    mesh.rotation.y = angle;
    group.add( mesh );

  };
  setInto( new THREE.Mesh( new THREE.BoxGeometry( 0.95, 1.9, 0.2 ), door ), 1.6, 0.4 );
  for ( const [ h, a ] of [ [ 3.6, 2.2 ], [ 7.8, 0.9 ], [ 12.0, 3.9 ] ] ) setInto( new THREE.Mesh( new THREE.BoxGeometry( 0.5, 0.8, 0.12 ), glass ), h, a );

  // ---- the gallery: a deck that overhangs the tower, with a railing
  const metal = surface( render, { name: 'iron', color: iron, roughness: 0.45, metalness: 0.6 } );
  const deck = new THREE.Mesh( new THREE.CylinderGeometry( 2.75, 2.45, 0.3, 48 ), metal );
  deck.position.y = GALLERY_Y + 0.15;
  group.add( deck );
  const rail: THREE.BufferGeometry[] = [];
  for ( const h of [ 0.55, 1.05 ] ) {

    const ring = new THREE.TorusGeometry( 2.65, 0.035, 6, 64 );
    ring.rotateX( Math.PI / 2 );
    ring.translate( 0, GALLERY_Y + 0.3 + h, 0 );
    rail.push( ring );

  }
  for ( let i = 0; i < 28; i ++ ) {

    const a = i / 28 * Math.PI * 2;
    const post = new THREE.CylinderGeometry( 0.03, 0.03, 1.05, 5 );
    post.translate( Math.cos( a ) * 2.65, GALLERY_Y + 0.3 + 0.525, Math.sin( a ) * 2.65 );
    rail.push( post );

  }
  group.add( new THREE.Mesh( mergeGeometries( rail ), metal ) );

  // ---- the lantern: an iron base, glazing bars round the lamp, a red roof and its finial
  const roofRed = surface( render, { name: 'lantern roof', color: albedo( 0x7d221c ), roughness: 0.5, metalness: 0.3 } );
  const base = new THREE.Mesh( new THREE.CylinderGeometry( 1.55, 1.6, 0.8, 32 ), roofRed );
  base.position.y = GALLERY_Y + 0.7;
  group.add( base );
  const bars: THREE.BufferGeometry[] = [];
  for ( let i = 0; i < 10; i ++ ) {

    const a = ( i + 0.5 ) / 10 * Math.PI * 2;
    const bar = new THREE.BoxGeometry( 0.07, 2.0, 0.07 );
    bar.translate( Math.cos( a ) * 1.45, GALLERY_Y + 2.1, Math.sin( a ) * 1.45 );
    bars.push( bar );

  }
  for ( const h of [ GALLERY_Y + 1.1, GALLERY_Y + 3.1 ] ) {

    const ring = new THREE.TorusGeometry( 1.47, 0.05, 6, 40 );
    ring.rotateX( Math.PI / 2 );
    ring.translate( 0, h, 0 );
    bars.push( ring );

  }
  group.add( new THREE.Mesh( mergeGeometries( bars ), metal ) );
  const roof = new THREE.Mesh( new THREE.ConeGeometry( 1.85, 1.3, 32 ), roofRed );
  roof.position.y = GALLERY_Y + 3.75;
  group.add( roof );
  const finial = new THREE.Mesh( new THREE.SphereGeometry( 0.2, 16, 12 ), metal );
  finial.position.y = GALLERY_Y + 4.55;
  group.add( finial );
  const rod = new THREE.Mesh( new THREE.CylinderGeometry( 0.02, 0.02, 1.2, 4 ), metal );
  rod.position.y = GALLERY_Y + 5.1;
  group.add( rod );

  // ---- the lamp and its lens. It flares as a beam swings toward the eye (`flash`).
  const beams = new THREE.Group();
  beams.name = 'beams';
  beams.position.y = LAMP_Y;
  group.add( beams );
  const beamAxis = uniform( new THREE.Vector3( 1, 0, 0 ) );
  const lampWorld = uniform( new THREE.Vector3() );
  const toEye = normalize( cameraPosition.sub( lampWorld ) );
  const facing = max( abs( dot( beamAxis, toEye ) ), 0 );
  const flash = pow( facing, float( 40 ) ).mul( 12 ).add( 1 );
  const lampRadiance = vec3( 1.0, 0.86, 0.62 ).mul( night.mul( GLOW.lamp ).mul( flash ) );
  const lamp = new THREE.Mesh( new THREE.SphereGeometry( 0.45, 24, 16 ), surface( render, { name: 'lamp', color: albedo( 0xfff4dd ), roughness: 0.2, emissive: lampRadiance } ) );
  lamp.position.y = LAMP_Y;
  group.add( lamp );
  const lens = new THREE.Mesh( new THREE.CylinderGeometry( 0.75, 0.75, 1.3, 24, 1, true ), surface( render, {
    name: 'lens', color: albedo( 0x9fb6b4 ), roughness: 0.08, metalness: 0.2, emissive: vec3( 1.0, 0.86, 0.62 ).mul( night.mul( GLOW.lens ).mul( flash ) ), side: THREE.DoubleSide,
  } ) );
  lens.position.y = LAMP_Y;
  group.add( lens );

  // ---- the beams: two opposite cones, their apex at the lamp, along +x and -x of `beams`
  const cone = new THREE.CylinderGeometry( BEAM.radius, 0.45, BEAM.length, 48, 1, true );
  cone.translate( 0, BEAM.length / 2, 0 );
  cone.rotateZ( - Math.PI / 2 ); // +y -> +x
  const beamMaterial = new THREE.MeshBasicNodeMaterial( { transparent: true, depthWrite: false, depthTest: true, side: THREE.DoubleSide } );
  beamMaterial.name = 'beam';
  beamMaterial.forceSinglePass = true;
  // premultiplied: additive radiance, alpha 0 (the spray's blending, ocean/Spray.ts)
  beamMaterial.blending = THREE.CustomBlending;
  beamMaterial.blendEquation = THREE.AddEquation;
  beamMaterial.blendSrc = THREE.OneFactor;
  beamMaterial.blendDst = THREE.OneMinusSrcAlphaFactor;
  beamMaterial.blendSrcAlpha = THREE.OneFactor;
  beamMaterial.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
  const along = abs( positionGeometry.x ).div( BEAM.length );
  // brightest along the axis as seen, falling off smoothly to the cone's edges
  const core = pow( abs( dot( normalize( normalView ), normalize( positionView ) ) ), float( 3 ) );
  // the flux spreads as the cone widens, and the air takes some of it with distance
  const fade = pow( float( 1 ).sub( along ), float( 1.5 ) ).div( along.mul( 5 ).add( 1 ) ).mul( smoothstep( 0, 0.015, along ) );
  // no beam cutting through the eye when the camera is inside it
  const near = smoothstep( 2, 14, positionWorld.sub( cameraPosition ).length() );
  const beamRadiance = vec3( 1.0, 0.9, 0.7 ).mul( night.mul( GLOW.beam ).mul( core ).mul( fade ).mul( near ) );
  beamMaterial.colorNode = vec4( beamRadiance, 0 );
  beamMaterial.opacityNode = float( 0 );
  for ( const turn of [ 0, Math.PI ] ) {

    const beam = new THREE.Mesh( cone, beamMaterial );
    beam.rotation.y = turn;
    beam.name = 'beam';
    beam.frustumCulled = false;
    beam.castShadow = beam.receiveShadow = false;
    beam.renderOrder = 10;
    beams.add( beam );

  }

  // ---- the keeper's cottage: white walls, a red gable roof, a chimney, lit windows
  const cottage = new THREE.Group();
  cottage.name = 'cottage';
  const W = 3.4, D = 2.8, H = 2.5;
  const walls = new THREE.Mesh( new THREE.BoxGeometry( W, H, D ), surface( render, { name: 'cottage walls', color: albedo( 0xdcd6c8 ), roughness: 0.8 } ) );
  walls.position.y = H / 2;
  cottage.add( walls );
  const gable = new THREE.Shape();
  gable.moveTo( - D / 2 - 0.3, 0 ).lineTo( D / 2 + 0.3, 0 ).lineTo( 0, 1.5 ).closePath();
  const roofGeometry = new THREE.ExtrudeGeometry( gable, { depth: W + 0.5, bevelEnabled: false } );
  roofGeometry.translate( 0, 0, - ( W + 0.5 ) / 2 );
  roofGeometry.rotateY( Math.PI / 2 );
  const cottageRoof = new THREE.Mesh( roofGeometry, surface( render, { name: 'cottage roof', color: albedo( 0x6e2a22 ), roughness: 0.75 } ) );
  cottageRoof.position.y = H;
  cottage.add( cottageRoof );
  const chimney = new THREE.Mesh( new THREE.BoxGeometry( 0.45, 1.3, 0.45 ), surface( render, { name: 'chimney', color: albedo( 0x7a7468 ), roughness: 0.9 } ) );
  chimney.position.set( W / 2 - 0.6, H + 1.1, 0.5 );
  cottage.add( chimney );
  for ( const x of [ - 0.8, 0.8 ] ) {

    const win = new THREE.Mesh( new THREE.BoxGeometry( 0.6, 0.7, 0.08 ), glass );
    win.position.set( x, 1.4, D / 2 );
    cottage.add( win );

  }
  const cottageDoor = new THREE.Mesh( new THREE.BoxGeometry( 0.08, 1.8, 0.85 ), door );
  cottageDoor.position.set( W / 2, 0.9, 0 );
  cottage.add( cottageDoor );
  cottage.position.set( 1.2, - 0.15, 3.6 );
  cottage.rotation.y = - 0.5;
  group.add( cottage );

  for ( const mesh of group.children ) {

    mesh.traverse( ( o ) => {

      if ( ( o as THREE.Mesh ).isMesh && o.parent !== beams ) o.castShadow = o.receiveShadow = true;

    } );

  }
  group.updateMatrixWorld( true );
  const lampPosition = lamp.getWorldPosition( new THREE.Vector3() );

  const axis = new THREE.Vector3();
  return {
    group,
    lamp: lampPosition,
    night,
    update( time: number ) {

      beams.rotation.y = ( time * BEAM.turn % 1 ) * Math.PI * 2;
      beams.updateMatrixWorld( true );
      axis.set( 1, 0, 0 ).applyQuaternion( beams.getWorldQuaternion( new THREE.Quaternion() ) );
      beamAxis.value.copy( axis );
      lampWorld.value.copy( lampPosition );
      beams.visible = night.value > 0.01;

    },
    bounds() {

      const box = new THREE.Box3();
      for ( const child of group.children ) if ( child !== beams ) box.expandByObject( child );
      return box;

    },
  };

}
