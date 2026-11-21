// The sea stack: a column of layered rock rising out of deep water, and the fields that tell the
// sea where its shore is.
//
// The rock is a lathe (a profile turned about the y axis) pushed in and out by seeded noise and by
// sedimentary layers, so its cliff steps in ledges. The same displacement gives the waterline's
// radius at every angle, from which the scene's fields are written: the shore's sea-level distance
// (negative over the rock) and the floor's height, as the render pipeline's world fields
// (render/NeutralInputs.ts has the format). With those the sea damps its short waves near the
// rock, casts a lee behind it and breaks shore waves against it.
import * as THREE from 'three/webgpu';
import { float, mix, mx_fractal_noise_float, normalWorld, positionWorld, smoothstep, vec3 } from 'three/tsl';
import { SimplexNoise } from 'three/addons/math/SimplexNoise.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import type { RenderParts } from './render/index.ts';
import { rgb, surface } from './surfaces.ts';

// Heights are metres above the mean sea level.
export const STACK = {
  /** the flat top the lighthouse stands on */
  top: 18,
  /** the deep floor round the stack */
  floor: - 30,
};

// The profile from the floor to the top's centre: ( radius, height ).
const PROFILE: [ number, number ][] = [
  [ 19, STACK.floor - 1 ], [ 16, - 20 ], [ 12.8, - 9 ], [ 10.6, - 3.5 ], [ 9.7, - 0.5 ], [ 9.3, 3 ],
  [ 9.0, 7 ], [ 8.6, 11 ], [ 8.1, 14.5 ], [ 7.4, 16.6 ], [ 6.5, 17.6 ], [ 4.5, STACK.top - 0.05 ], [ 0, STACK.top + 0.1 ],
];

// a seeded generator, so the rock is the same on every load
function mulberry32( seed: number ) {

  let s = seed >>> 0;
  return () => {

    s = ( s + 0x6d2b79f5 ) >>> 0;
    let t = Math.imul( s ^ ( s >>> 15 ), 1 | s );
    t = ( t + Math.imul( t ^ ( t >>> 7 ), 61 | t ) ) ^ t;
    return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

  };

}

const noise = new SimplexNoise( { random: mulberry32( 7 ) } );

/** How far the rock's surface moves out along its radius at a point of the undisplaced lathe. */
function radialOffset( x: number, y: number, z: number, radius: number ): number {

  // only the cliff moves outward; the top's flat centre stays put
  const cliff = THREE.MathUtils.smoothstep( radius, 3.5, 6.5 );
  const large = noise.noise3d( x * 0.06, y * 0.04, z * 0.06 ) * 1.9;
  const medium = noise.noise3d( x * 0.18, y * 0.2, z * 0.18 ) * 0.6;
  const fine = noise.noise3d( x * 0.7, y * 0.9, z * 0.7 ) * 0.14;
  // vertical joints: grooves and buttresses stretched up the cliff
  const joints = Math.abs( noise.noise3d( x * 0.32, y * 0.035, z * 0.32 ) ) * - 1.1;
  // layers: each bed juts out at its base and recedes upward (a sawtooth), the beds of uneven
  // thickness and the ledges stronger in some places than others
  const bed = ( y + noise.noise3d( x * 0.08, y * 0.05, z * 0.08 ) * 1.4 ) / 3.1;
  const strength = THREE.MathUtils.clamp( 0.5 + noise.noise3d( x * 0.05, y * 0.03 + 9, z * 0.05 ), 0, 1 );
  const layer = Math.pow( 1 - ( bed - Math.floor( bed ) ), 2 ) * 0.5 * strength;
  return ( large + medium + fine + joints + layer ) * cliff;

}

/** The waterline's radius at angle `theta` (radians about y, from +x toward +z). */
export function waterlineRadius( theta: number ): number {

  const r = profileRadiusAt( 0 );
  const x = Math.cos( theta ) * r, z = Math.sin( theta ) * r;
  return r + radialOffset( x, 0, z, r );

}

function profileRadiusAt( y: number ): number {

  for ( let i = 1; i < PROFILE.length; i ++ ) {

    const [ r0, y0 ] = PROFILE[ i - 1 ], [ r1, y1 ] = PROFILE[ i ];
    if ( y <= y1 ) return r0 + ( r1 - r0 ) * ( y - y0 ) / ( y1 - y0 );

  }
  return 0;

}

/** The stack's geometry, centred on the origin with the sea level at y = `sea`. */
export function stackGeometry( sea: number ): THREE.BufferGeometry {

  const curve = new THREE.SplineCurve( PROFILE.map( ( [ r, y ] ) => new THREE.Vector2( r, y ) ) );
  const points = curve.getSpacedPoints( 140 );
  // close the top exactly on the axis
  points[ points.length - 1 ].set( 0, STACK.top + 0.1 );
  let geometry: THREE.BufferGeometry = new THREE.LatheGeometry( points, 180 );
  // weld the lathe's seam so the normals run smoothly across it
  geometry.deleteAttribute( 'uv' );
  geometry.deleteAttribute( 'normal' );
  geometry = mergeVertices( geometry, 1e-4 );
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const v = new THREE.Vector3();
  for ( let i = 0; i < position.count; i ++ ) {

    v.fromBufferAttribute( position, i );
    const radius = Math.hypot( v.x, v.z );
    if ( radius > 1e-6 ) {

      const out = radialOffset( v.x, v.y, v.z, radius );
      v.x += v.x / radius * out;
      v.z += v.z / radius * out;

    }
    // the top: a gentle uneven surface
    if ( v.y > STACK.top - 1 ) v.y += noise.noise3d( v.x * 0.25, 0, v.z * 0.25 ) * 0.12 * THREE.MathUtils.smoothstep( 6.5 - radius, 0, 2 );
    v.y += sea;
    position.setXYZ( i, v.x, v.y, v.z );

  }
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;

}

/** The rock's material: layered stone, dark and glossy where the sea wets it, weed at the waterline,
 * grass on the flat top. */
export function stackMaterial( render: RenderParts, sea: number ): THREE.MeshStandardNodeMaterial {

  const p = positionWorld;
  const h = p.y.sub( sea );
  const big = mx_fractal_noise_float( p.mul( 0.15 ), 4, 2, 0.5, 1 );
  const grain = mx_fractal_noise_float( p.mul( 1.7 ), 3, 2, 0.5, 1 );
  // beds of slightly different stone
  const beds = mx_fractal_noise_float( vec3( p.x.mul( 0.05 ), h.mul( 0.9 ), p.z.mul( 0.05 ) ), 2, 2, 0.5, 1 );
  // weathering streaks: rain and salt run down the cliff from the ledges
  const streaks = mx_fractal_noise_float( vec3( p.x.mul( 0.9 ), h.mul( 0.07 ), p.z.mul( 0.9 ) ), 3, 2, 0.5, 1 );
  const dark = rgb( 0x34312d ), light = rgb( 0x5f594f ), rust = rgb( 0x5e4636 );
  let rock = mix( dark, light, smoothstep( - 0.45, 0.55, big.add( grain.mul( 0.3 ) ) ) );
  rock = mix( rock, rust, smoothstep( 0.15, 0.6, beds ).mul( 0.35 ) );
  rock = rock.mul( float( 1 ).sub( smoothstep( 0.05, 0.45, streaks ).mul( 0.35 ) ) ).mul( grain.mul( 0.12 ).add( 1 ) );
  // pale lichen and salt crust on the upper cliff, where it faces up
  const lichen = smoothstep( 0.25, 0.65, normalWorld.y.add( grain.mul( 0.25 ) ) ).mul( smoothstep( 4, 9, h ) ).mul( smoothstep( 0.1, 0.5, big.add( 0.2 ) ) );
  rock = mix( rock, rgb( 0x8f8b78 ), lichen.mul( 0.55 ) );
  // the splash zone: wet stone up to a ragged line a metre or two above the mean sea
  const splash = float( 1.6 ).add( big.mul( 0.9 ) );
  const wet = float( 1 ).sub( smoothstep( splash.sub( 0.6 ), splash, h ) );
  // weed and barnacles round the mean waterline
  const weedBand = smoothstep( - 1.2, - 0.2, h ).mul( float( 1 ).sub( smoothstep( 0.4, 1.1, h.add( grain.mul( 0.4 ) ) ) ) );
  const weed = mix( rgb( 0x3b3a22 ), rgb( 0x5d5a3a ), grain.mul( 0.5 ).add( 0.5 ) );
  // grass on the flat top
  const up = smoothstep( 0.72, 0.9, normalWorld.y ).mul( smoothstep( STACK.top - 1.2, STACK.top - 0.4, h ) );
  const grass = mix( rgb( 0x4d5a28 ), rgb( 0x7a7a3c ), smoothstep( - 0.3, 0.5, grain.add( big ) ) );
  let colour = mix( rock, rock.mul( 0.45 ), wet );
  colour = mix( colour, weed, weedBand );
  colour = mix( colour, grass, up.mul( smoothstep( - 0.2, 0.1, big.add( 0.25 ) ) ) );
  const roughness = mix( float( 0.88 ).sub( grain.mul( 0.06 ) ), float( 0.38 ), wet.mul( float( 1 ).sub( up ) ) );
  return surface( render, { name: 'sea stack', color: colour, roughness } );

}

/** The scene's fields round the stack, in the formats NeutralInputs.ts `neutralFields` returns. */
export function stackFields( { half, cells, sea, xform }: { half: number; cells: number; sea: number; xform: THREE.Vector4 } ) {

  const shore = new Float32Array( cells * cells ), floor = new Float32Array( cells * cells );
  const cell = 2 * half / cells;
  for ( let j = 0; j < cells; j ++ ) for ( let i = 0; i < cells; i ++ ) {

    const x = - half + ( i + 0.5 ) * cell, z = - half + ( j + 0.5 ) * cell;
    const rho = Math.hypot( x, z );
    const distance = rho - waterlineRadius( Math.atan2( z, x ) );
    shore[ j * cells + i ] = distance;
    // a rocky apron round the foot, dropping to the deep floor within about 25 m
    const out = Math.max( distance, 0 );
    floor[ j * cells + i ] = distance < 0 ? sea + 6 : sea + STACK.floor + ( - 1.5 - STACK.floor ) * Math.exp( - out / 7 );

  }
  const texture = ( data: Float32Array, name: string ) => {

    const tex = new THREE.DataTexture( data, cells, cells, THREE.RedFormat, THREE.FloatType );
    tex.minFilter = tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    tex.name = name;
    return tex;

  };
  xform.set( - half, - half, 1 / ( 2 * half ), 1 / ( 2 * half ) );
  return {
    textures: { shore: texture( shore, 'shore-distance' ), floor: texture( floor, 'seabed-height' ) },
    shoreData: { data: shore, nx: cells, nz: cells, x0: - half, z0: - half, cell },
    bounds: { x0: - half, z0: - half, x1: half, z1: half },
  };

}
