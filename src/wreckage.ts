// The wreck: what is left of a ship the storm drove onto the stack.
//
// Everything here is built from code, seeded, so it is the same on every load:
// - hull pieces from one parametric hull (`hullGeometry`): a section turned along the length, its
//   half-beam, depth and sheer varying from end to end, ending at a bow (a stem), at a second bow
//   (a boat) or at a jagged break, with stove-in planks near a break, ribs inside and gunwale rails;
// - spars, torn canvas, rope, the wheel, gratings, splintered planks and the cargo's barrels.
// Surfaces read uv in metres: u along the grain or the length, v round the hull from the gunwale,
// so the planking's seams, its paint and its copper sheathing follow the hull.
import * as THREE from 'three/webgpu';
import {
  abs, float, fract, floor, max, mix, mx_fractal_noise_float, mx_noise_float, positionLocal, positionWorld, smoothstep, step, uv, vec2, vec3,
} from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { ALPHA_TO_COVERAGE, type RenderParts } from './render/index.ts';
import { rgb, surface } from './surfaces.ts';

export function mulberry32( seed: number ): () => number {

  let s = seed >>> 0;
  return () => {

    s = ( s + 0x6d2b79f5 ) >>> 0;
    let t = Math.imul( s ^ ( s >>> 15 ), 1 | s );
    t = ( t + Math.imul( t ^ ( t >>> 7 ), 61 | t ) ) ^ t;
    return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

  };

}

// ------------------------------------------------------------------------------ the materials

export interface WreckMaterials {
  /** the hull's planking: seams, a faded painted wale, copper sheathing low down */
  hull: THREE.Material;
  /** bare timber: ribs, spars, planks, gratings */
  timber: THREE.Material;
  canvas: THREE.Material;
  rope: THREE.Material;
  brass: THREE.Material;
  /** the barrels' staves and hoops */
  barrel: THREE.Material;
}

export function wreckMaterials( render: RenderParts, sea: number ): WreckMaterials {

  // darker and glossier where the sea wets it
  const wet = float( 1 ).sub( smoothstep( sea + 0.05, sea + 0.6, positionWorld.y ) );
  const wetColour = ( c: THREE.Node<'vec3'> ) => mix( c, c.mul( 0.55 ), wet );
  const wetRough = ( r: number ) => mix( float( r ), float( 0.35 ), wet );

  // ---- the hull: planks along the length (`v` across them, from the gunwale), butt joints staggered
  const u = uv().x, v = uv().y;
  const PLANK = 0.19;
  const row = floor( v.div( PLANK ) );
  const across = fract( v.div( PLANK ) );
  const butt = fract( u.div( 3.3 ).add( fract( row.mul( 0.618 ) ) ) );
  const seam = max( float( 1 ).sub( smoothstep( 0, 0.07, across ) ), float( 1 ).sub( smoothstep( 0, 0.012, butt ) ) );
  const grain = mx_fractal_noise_float( vec3( u.mul( 0.8 ), v.mul( 14 ), row ), 3, 2, 0.5, 1 );
  const plankTone = mx_noise_float( vec2( row.mul( 3.1 ), floor( u.div( 3.3 ).add( fract( row.mul( 0.618 ) ) ) ) ) );
  const oak = mix( rgb( 0x3e3022 ), rgb( 0x6d5b45 ), smoothstep( - 0.6, 0.6, grain.add( plankTone.mul( 0.6 ) ) ) );
  // the wale: a band of faded, chipped paint below the gunwale
  const chips = mx_fractal_noise_float( vec3( u.mul( 2.5 ), v.mul( 2.5 ), 7 ), 3, 2, 0.5, 1 );
  const wale = step( 0.22, v ).mul( step( v, 0.6 ) ).mul( smoothstep( 0, 0.3, chips ) ).mul( 0.8 );
  // copper sheathing low on the hull, gone green
  const copperBand = smoothstep( 1.25, 1.35, v ).mul( smoothstep( - 0.35, - 0.1, chips ) .mul( 0.25 ).add( 0.75 ) );
  // dull dark copper, verdigris only in streaks and where plates meet
  const verdigris = smoothstep( 0.25, 0.55, mx_fractal_noise_float( vec3( u.mul( 0.6 ), v.mul( 4 ), 3 ), 3, 2, 0.5, 1 ) ).mul( 0.6 );
  const patina = mix( mix( rgb( 0x4a3222 ), rgb( 0x5e4330 ), grain.mul( 0.5 ).add( 0.5 ) ), rgb( 0x3e5548 ), verdigris );
  let hullColour = mix( oak, rgb( 0x23302e ), wale );
  hullColour = mix( hullColour.mul( float( 1 ).sub( seam.mul( 0.6 ) ) ), patina, copperBand );
  const hull = surface( render, {
    name: 'hull', color: wetColour( hullColour ), side: THREE.DoubleSide,
    roughness: mix( wetRough( 0.85 ), float( 0.55 ), copperBand ), metalness: 0,
  } );

  // ---- bare timber: grain along local x
  const g = mx_fractal_noise_float( positionLocal.mul( vec3( 0.9, 9, 9 ) ), 3, 2, 0.5, 1 );
  const timberColour = mix( rgb( 0x3a2c1f ), rgb( 0x75634c ), smoothstep( - 0.6, 0.6, g ) );
  const timber = surface( render, { name: 'timber', color: wetColour( timberColour ), roughness: wetRough( 0.88 ), side: THREE.DoubleSide } );

  // ---- canvas: stained, sodden sailcloth, torn: a ragged edge, holes and a long rip, cut by alpha
  // (alpha to coverage under MSAA, which also keeps it out of the depth prepass; else alpha test)
  const stain = mx_fractal_noise_float( positionLocal.mul( 0.9 ), 4, 2, 0.5, 1 );
  const canvas = surface( render, {
    name: 'sailcloth', color: wetColour( mix( rgb( 0xa39a82 ), rgb( 0x6a6250 ), smoothstep( - 0.3, 0.6, stain ) ) ),
    roughness: wetRough( 0.92 ), side: THREE.DoubleSide,
  } );
  const cloth = positionLocal.xz;
  const fray = mx_fractal_noise_float( vec3( cloth.mul( 1.8 ), 3 ), 3, 2, 0.5, 1 );
  const edgeKept = smoothstep( - 0.02, 0.02, uv().x.sub( 0.14 ).add( fray.mul( 0.16 ) ) );
  const holes = smoothstep( 0.02, - 0.02, mx_fractal_noise_float( vec3( cloth.mul( 0.7 ), 9 ), 3, 2, 0.5, 1 ).sub( 0.3 ) );
  const rip = smoothstep( 0.05, 0.09, abs( cloth.y.mul( 0.8 ).add( cloth.x.mul( 1.4 ).sin().mul( 0.3 ) ).sub( 0.5 ).add( fray.mul( 0.08 ) ) ) );
  canvas.opacityNode = edgeKept.mul( holes ).mul( rip );
  if ( ALPHA_TO_COVERAGE ) canvas.alphaToCoverage = true;
  else canvas.alphaTest = 0.5;
  const rope = surface( render, { name: 'rope', color: wetColour( rgb( 0x7a6646 ) ), roughness: 0.95 } );
  const brass = surface( render, { name: 'brass', color: rgb( 0x8a6a2e ), roughness: 0.42, metalness: 0.85 } );

  // ---- barrels: oak staves, iron hoops near each end
  const y = abs( positionLocal.y );
  const hoop = step( 0.25, y ).mul( step( y, 0.31 ) ).add( step( 0.4, y ).mul( step( y, 0.44 ) ) );
  const staves = mix( rgb( 0x4a3220 ), rgb( 0x6e4c30 ), smoothstep( - 0.5, 0.5, g ) );
  const barrel = surface( render, { name: 'barrel', color: wetColour( mix( staves, rgb( 0x2a2724 ), hoop ) ), roughness: wetRough( 0.75 ) } );

  return { hull, timber, canvas, rope, brass, barrel };

}

// ------------------------------------------------------------------------------ the hull

export interface HullShape {
  length: number;
  /** the widest beam (m) and the depth from gunwale to keel (m) */
  beam: number;
  depth: number;
  /** how far the gunwale rises toward a bow (m) */
  sheer: number;
  /** 'bow': a break at u = 0 and a stem at u = 1; 'boat': stems at both ends; 'broken': breaks at both */
  ends: 'bow' | 'boat' | 'broken';
  /** how far a break's jagged edge reaches into the length (a share of it) */
  jag: number;
  /** the share of planks stove in near a break */
  holes: number;
  /** the share of the section's sides kept: 1 the whole section, less for a piece torn from one side */
  span?: [ number, number ];
  seed: number;
}

function frameAt( s: HullShape, u: number ) {

  const e = s.ends === 'boat' ? Math.abs( 2 * u - 1 ) : s.ends === 'bow' ? u : 0;
  const taper = s.ends === 'broken' ? 1 : Math.sqrt( Math.max( 0, 1 - e ** 2.4 ) );
  return { half: s.beam / 2 * Math.max( taper, 0.015 ), depth: s.depth * ( 1 - 0.2 * e * e ), top: s.sheer * e * e };

}

/** A point of the hull at u (0 to 1 along it) and t (-1 port gunwale, 0 keel, 1 starboard gunwale). */
function hullPoint( s: HullShape, u: number, t: number, out = new THREE.Vector3() ): THREE.Vector3 {

  const { half, depth, top } = frameAt( s, u );
  const a = t * Math.PI / 2;
  return out.set( u * s.length, top - depth * Math.pow( Math.cos( a ), 0.55 ), half * Math.sin( a ) );

}

/** The hull's planking (`shell`) and its ribs and rails (`frames`), in the hull's frame: x along it
 * from u = 0, y up from the gunwale's height at the middle, z across. */
export function hullGeometry( s: HullShape ): { shell: THREE.BufferGeometry; frames: THREE.BufferGeometry } {

  const random = mulberry32( s.seed );
  const I = Math.max( 12, Math.round( s.length * 6 ) ), J = 26;
  const [ t0, t1 ] = s.span ?? [ - 1, 1 ];
  // each column's ends: jagged at a break, smoothed over neighbours so the teeth are a few planks wide
  const jag = ( ) => {

    const raw = Array.from( { length: J + 1 }, () => random() );
    return raw.map( ( _, j ) => ( raw[ j ] * 0.6 + ( raw[ j - 1 ] ?? raw[ j ] ) * 0.2 + ( raw[ j + 1 ] ?? raw[ j ] ) * 0.2 ) );

  };
  const startJag = s.ends === 'boat' ? null : jag();
  const endJag = s.ends === 'broken' ? jag() : null;
  const u0 = ( j: number ) => startJag ? startJag[ j ] * s.jag : 0;
  const u1 = ( j: number ) => endJag ? 1 - endJag[ j ] * s.jag : 1;

  const positions: number[] = [], uvs: number[] = [], index: number[] = [];
  const p = new THREE.Vector3(), q = new THREE.Vector3();
  for ( let i = 0; i <= I; i ++ ) {

    // the arc length from each gunwale, for the planking's v
    const ts = Array.from( { length: J + 1 }, ( _, j ) => t0 + ( t1 - t0 ) * j / J );
    const us = ts.map( ( _, j ) => u0( j ) + ( u1( j ) - u0( j ) ) * i / I );
    const arc = [ 0 ];
    for ( let j = 1; j <= J; j ++ ) arc.push( arc[ j - 1 ] + hullPoint( s, us[ j ], ts[ j ], p ).distanceTo( hullPoint( s, us[ j - 1 ], ts[ j - 1 ], q ) ) );
    const total = arc[ J ];
    for ( let j = 0; j <= J; j ++ ) {

      hullPoint( s, us[ j ], ts[ j ], p );
      positions.push( p.x, p.y, p.z );
      // from the nearest gunwale that the piece still has
      const fromPort = t0 <= - 0.999 ? arc[ j ] : Infinity, fromStarboard = t1 >= 0.999 ? total - arc[ j ] : Infinity;
      const fromGunwale = Math.min( fromPort, fromStarboard );
      uvs.push( us[ j ] * s.length, Number.isFinite( fromGunwale ) ? fromGunwale : arc[ j ] + 0.8 );

    }

  }
  const holeRandom = mulberry32( s.seed + 101 );
  for ( let i = 0; i < I; i ++ ) for ( let j = 0; j < J; j ++ ) {

    const u = ( i + 0.5 ) / I;
    const nearBreak = s.ends === 'boat' ? 0 : Math.max( 1 - u / 0.35, s.ends === 'broken' ? 1 - ( 1 - u ) / 0.35 : 0 );
    // stove-in planks: runs of a few quads along a plank, near a break
    if ( holeRandom() < s.holes * nearBreak && j > 2 && j < J - 2 ) continue;
    const a = i * ( J + 1 ) + j, b = a + J + 1;
    index.push( a, b, a + 1, a + 1, b, b + 1 );

  }
  const shell = new THREE.BufferGeometry();
  shell.setAttribute( 'position', new THREE.Float32BufferAttribute( positions, 3 ) );
  shell.setAttribute( 'uv', new THREE.Float32BufferAttribute( uvs, 2 ) );
  shell.setIndex( index );
  shell.computeVertexNormals();

  // ribs inside the planking, some broken off short, and those at a break standing proud of it
  const frames: THREE.BufferGeometry[] = [];
  const ribs = Math.max( 3, Math.round( s.length / 0.55 ) );
  for ( let k = 0; k <= ribs; k ++ ) {

    const u = k / ribs;
    if ( s.ends !== 'broken' && u > 0.93 ) continue;
    const broken = random() < 0.35;
    const ta = t0 + ( broken && random() < 0.5 ? random() * 0.6 : 0 ), tb = t1 - ( broken ? random() * 0.6 : 0 );
    const points: THREE.Vector3[] = [];
    for ( let n = 0; n <= 10; n ++ ) {

      const pt = hullPoint( s, u, ta + ( tb - ta ) * n / 10 );
      // inside the planking
      pt.y += 0.05;
      pt.z *= 0.93;
      points.push( pt );

    }
    frames.push( new THREE.TubeGeometry( new THREE.CatmullRomCurve3( points ), 12, Math.min( 0.07, s.depth * 0.05 ), 4, false ) );

  }
  // the gunwale rails along whichever sides remain
  for ( const t of [ - 1, 1 ] ) {

    if ( ( t < 0 && t0 > - 0.999 ) || ( t > 0 && t1 < 0.999 ) ) continue;
    const j = t < 0 ? 0 : J;
    const a = u0( j ) + 0.02, b = u1( j ) - 0.02;
    const points = Array.from( { length: 12 }, ( _, n ) => hullPoint( s, a + ( b - a ) * n / 11, t ) );
    frames.push( new THREE.TubeGeometry( new THREE.CatmullRomCurve3( points ), 24, Math.min( 0.08, s.depth * 0.06 ), 5, false ) );

  }
  return { shell, frames: mergeGeometries( frames.map( stripForMerge ) ) };

}

// Every geometry merged here keeps position, normal and uv only.
function stripForMerge( g: THREE.BufferGeometry ): THREE.BufferGeometry {

  const out = g.index ? g.toNonIndexed() : g;
  for ( const name of Object.keys( out.attributes ) ) if ( ! [ 'position', 'normal', 'uv' ].includes( name ) ) out.deleteAttribute( name );
  if ( ! out.attributes.uv ) out.setAttribute( 'uv', new THREE.Float32BufferAttribute( new Float32Array( out.attributes.position.count * 2 ), 2 ) );
  if ( ! out.attributes.normal ) out.computeVertexNormals();
  return out;

}

export function merge( parts: THREE.BufferGeometry[] ): THREE.BufferGeometry {

  return mergeGeometries( parts.map( stripForMerge ) );

}

// ------------------------------------------------------------------------------ the loose parts

/** A splintered plank along x: its ends torn into teeth. */
export function plankGeometry( length: number, width: number, thick: number, seed: number ): THREE.BufferGeometry {

  const random = mulberry32( seed );
  const teeth = 6;
  const geometry = new THREE.BoxGeometry( length, thick, width, 1, 1, teeth );
  const cut = { [ - 1 ]: Array.from( { length: teeth + 1 }, () => random() * 0.45 ), [ 1 ]: Array.from( { length: teeth + 1 }, () => random() * 0.45 ) };
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const uvs = new Float32Array( position.count * 2 );
  for ( let i = 0; i < position.count; i ++ ) {

    const x = position.getX( i ), z = position.getZ( i );
    const side = Math.sign( x ) as - 1 | 1;
    const k = Math.round( ( z / width + 0.5 ) * teeth );
    position.setX( i, x - side * cut[ side ][ k ] );
    uvs[ i * 2 ] = position.getX( i );
    uvs[ i * 2 + 1 ] = z;

  }
  geometry.setAttribute( 'uv', new THREE.BufferAttribute( uvs, 2 ) );
  geometry.computeVertexNormals();
  return geometry;

}

/** A spar along x of `radius`, its far end snapped off jagged. */
export function sparGeometry( length: number, radius: number, seed: number, snapped = true ): THREE.BufferGeometry {

  const random = mulberry32( seed );
  const geometry = new THREE.CylinderGeometry( radius * 0.85, radius, length, 14, 6 );
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const top = length / 2;
  for ( let i = 0; i < position.count; i ++ ) {

    if ( snapped && position.getY( i ) > top - 1e-4 ) {

      // splinters: the end's rim pulled out to points, the centre torn short
      const r = Math.hypot( position.getX( i ), position.getZ( i ) );
      position.setY( i, top - ( r < radius * 0.1 ? 0.25 : random() * 0.5 ) );

    }

  }
  geometry.rotateZ( - Math.PI / 2 );
  geometry.computeVertexNormals();
  return geometry;

}

/** Sailcloth: a sheet of `w` by `h` (in x and z), slightly rumpled. Its holes and torn edge are cut
 * by its material (`WreckMaterials.canvas`), which reads how far in from the edge a point is in uv.x. */
export function canvasGeometry( w: number, h: number, seed: number ): THREE.BufferGeometry {

  const random = mulberry32( seed );
  const phase = random() * 10;
  const geometry = new THREE.PlaneGeometry( w, h, 30, 40 );
  geometry.rotateX( - Math.PI / 2 );
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const uvs = geometry.attributes.uv as THREE.BufferAttribute;
  for ( let i = 0; i < position.count; i ++ ) {

    const x = position.getX( i ), z = position.getZ( i );
    position.setY( i, Math.sin( x * 2.1 + z * 1.3 + phase ) * 0.06 + Math.sin( z * 3.7 ) * 0.04 );
    // u: how far in from the nearest edge, as a share of the half size (0 at the edge, 1 in the middle)
    uvs.setXY( i, Math.min( 1 - Math.abs( x ) / ( w / 2 ), 1 - Math.abs( z ) / ( h / 2 ) ), 0 );

  }
  geometry.computeVertexNormals();
  return geometry;

}

/** A rope through `points`. */
export function ropeGeometry( points: THREE.Vector3[], radius = 0.025 ): THREE.BufferGeometry {

  return new THREE.TubeGeometry( new THREE.CatmullRomCurve3( points ), points.length * 8, radius, 5, false );

}

/** The ship's wheel, flat in x and z: its rim cracked through, eight spokes with handles. */
export function wheelGeometry(): { timber: THREE.BufferGeometry; brass: THREE.BufferGeometry } {

  const R = 0.55;
  const parts: THREE.BufferGeometry[] = [];
  // the rim, a sixth of it gone
  const rim = new THREE.TorusGeometry( R, 0.045, 6, 40, Math.PI * 2 * 0.83 );
  rim.rotateX( Math.PI / 2 );
  parts.push( rim );
  for ( let k = 0; k < 8; k ++ ) {

    const a = k / 8 * Math.PI * 2 + 0.2;
    const missing = k === 7;
    const len = missing ? R * 0.6 : R + 0.22;
    const spoke = new THREE.CylinderGeometry( 0.025, 0.03, len, 6 );
    spoke.translate( 0, len / 2, 0 );
    spoke.rotateZ( - Math.PI / 2 );
    spoke.rotateY( - a );
    parts.push( spoke );
    if ( ! missing ) {

      const handle = new THREE.CylinderGeometry( 0.035, 0.03, 0.14, 6 );
      handle.rotateZ( - Math.PI / 2 );
      handle.translate( R + 0.28, 0, 0 );
      handle.rotateY( - a );
      parts.push( handle );

    }

  }
  const hub = new THREE.CylinderGeometry( 0.12, 0.12, 0.16, 16 );
  return { timber: merge( parts ), brass: hub };

}

/** A hatch grating: a frame round a lattice, `w` by `d`, flat. */
export function gratingGeometry( w: number, d: number ): THREE.BufferGeometry {

  const parts: THREE.BufferGeometry[] = [];
  const add = ( sx: number, sz: number, x: number, z: number, h = 0.07 ) => {

    const g = new THREE.BoxGeometry( sx, h, sz );
    g.translate( x, 0, z );
    parts.push( g );

  };
  add( w, 0.1, 0, - d / 2 + 0.05, 0.1 );
  add( w, 0.1, 0, d / 2 - 0.05, 0.1 );
  add( 0.1, d, - w / 2 + 0.05, 0, 0.1 );
  add( 0.1, d, w / 2 - 0.05, 0, 0.1 );
  for ( let k = 1; k < 7; k ++ ) add( w - 0.1, 0.035, 0, - d / 2 + k * d / 7 );
  for ( let k = 1; k < 9; k ++ ) add( 0.035, d - 0.1, - w / 2 + k * w / 9, 0, 0.05 );
  return merge( parts );

}

/** A barrel along y, its staves bulging. */
export function barrelGeometry(): THREE.BufferGeometry {

  const R = 0.3, H = 0.9;
  const points: THREE.Vector2[] = [ new THREE.Vector2( 0, - H / 2 ) ];
  for ( let i = 0; i <= 12; i ++ ) {

    const t = i / 12;
    points.push( new THREE.Vector2( R * ( 0.88 + 0.12 * Math.sin( Math.PI * t ) ), - H / 2 + t * H ) );

  }
  points.push( new THREE.Vector2( 0, H / 2 ) );
  return new THREE.LatheGeometry( points, 24 );

}

/** The hull pieces' and loose parts' meshes, as one group per piece; `bake` turns a group's meshes
 * into world-space points for a collider. */
export function group( ...meshes: [ THREE.BufferGeometry, THREE.Material ][] ): THREE.Group {

  const out = new THREE.Group();
  for ( const [ geometry, material ] of meshes ) {

    const mesh = new THREE.Mesh( geometry, material );
    mesh.castShadow = mesh.receiveShadow = true;
    out.add( mesh );

  }
  return out;

}

/** Every vertex of `geometries` as one flat array, for a convex hull collider. */
export function hullPoints( ...geometries: THREE.BufferGeometry[] ): Float32Array {

  const all: number[] = [];
  for ( const g of geometries ) {

    const p = g.attributes.position;
    // every few vertices are enough for the hull's outline
    for ( let i = 0; i < p.count; i += 3 ) all.push( p.getX( i ), p.getY( i ), p.getZ( i ) );

  }
  return new Float32Array( all );

}

// ------------------------------------------------------------------------------ the wreck itself

/** The ship's bow, driven onto the stack: its stem rammed into the rock just above the waterline at
 * `stem` (world), the hull coming in along `heading` (radians about y, the direction it was
 * sailing), pitched up and listing, the rest of the ship broken off and sunk. Returns the group (in
 * world space) and its planking in world space, for the physics. */
export function bowWreck( m: WreckMaterials, stem: THREE.Vector3, heading: number ): { group: THREE.Group; collider: THREE.BufferGeometry } {

  const shape: HullShape = { length: 13, beam: 5.2, depth: 3.4, sheer: 1.2, ends: 'bow', jag: 0.2, holes: 0.12, seed: 5 };
  const L = shape.length;
  const { shell, frames } = hullGeometry( shape );
  // the deck over the bow's last third, its aft edge torn
  const random = mulberry32( 6 );
  const deckParts: THREE.BufferGeometry[] = [];
  const N = 10;
  for ( let k = 0; k < N; k ++ ) {

    // a deck plank across the hull at u, cut where the hull's sides are
    const u = 0.66 + 0.3 * k / N + random() * 0.01;
    const e = u, half = shape.beam / 2 * Math.sqrt( Math.max( 0, 1 - e ** 2.4 ) ) * 0.96;
    if ( half < 0.1 ) continue;
    const plank = new THREE.BoxGeometry( 0.3 * shape.length / N, 0.07, 2 * half );
    plank.translate( u * shape.length, shape.sheer * e * e - 0.12, 0 );
    deckParts.push( plank );

  }
  // the foremast's stump, snapped off over the deck, and the bowsprit's broken heel
  const mastAt = 0.7 * L, deckAt = shape.sheer * 0.49;
  const stump = sparGeometry( 4.2, 0.3, 7 );
  stump.rotateZ( Math.PI / 2 );
  stump.translate( mastAt, deckAt + 1.6, 0 );
  const sprit = sparGeometry( 2.2, 0.22, 8 );
  sprit.rotateZ( 0.35 );
  sprit.translate( L + 0.6, shape.sheer + 0.6, 0 );
  // rigging hanging from the stump over the side into the sea
  const B = shape.beam / 2;
  const rigging = merge( [
    ropeGeometry( [ new THREE.Vector3( mastAt, deckAt + 3.4, 0 ), new THREE.Vector3( mastAt - 0.6, deckAt + 1.8, B * 0.7 ), new THREE.Vector3( mastAt - 1.8, deckAt - 0.2, B * 1.05 ), new THREE.Vector3( mastAt - 3.2, deckAt - 2.6, B * 1.15 ) ], 0.045 ),
    ropeGeometry( [ new THREE.Vector3( mastAt, deckAt + 3.1, 0 ), new THREE.Vector3( mastAt - 1.2, deckAt + 1.5, - B * 0.7 ), new THREE.Vector3( mastAt - 2.2, deckAt + 0.1, - B * 1.05 ), new THREE.Vector3( mastAt - 2.8, deckAt - 1.6, - B * 1.1 ) ], 0.04 ),
    ropeGeometry( [ new THREE.Vector3( mastAt + 0.2, deckAt + 2.6, 0.2 ), new THREE.Vector3( mastAt + 1.6, deckAt + 1.2, B * 0.5 ), new THREE.Vector3( mastAt + 2.4, deckAt + 0.3, B * 0.85 ) ], 0.035 ),
  ] );
  const timber = merge( [ frames, ...deckParts, stump, sprit ] );

  const wreck = group( [ shell, m.hull ], [ timber, m.timber ], [ rigging, m.rope ] );
  wreck.name = 'wreck';
  // driven bow-first up the rock: pitched up, listing to one side, its stem where it struck
  const pitch = THREE.MathUtils.degToRad( 22 ), list = THREE.MathUtils.degToRad( - 16 );
  wreck.rotation.set( list, heading, pitch, 'YZX' );
  wreck.updateMatrixWorld( true );
  const stemLocal = new THREE.Vector3( L, shape.sheer - shape.depth * 0.25, 0 ).applyQuaternion( wreck.quaternion );
  wreck.position.copy( stem ).sub( stemLocal );
  wreck.updateMatrixWorld( true );
  const collider = shell.clone().applyMatrix4( wreck.matrixWorld );
  return { group: wreck, collider };

}
