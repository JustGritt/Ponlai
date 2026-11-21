// The rocks: the sea stack the lighthouse stands on and the rocks round it (two lesser stacks, the
// stump of a third, a whaleback, skerries at the stack's foot and a reef awash), and the fields
// that tell the sea where their shores are.
//
// Each rock is a lathe (a profile turned about the y axis), stretched in plan and pushed in and out
// by seeded noise and by sedimentary layers, so its cliffs step in ledges. The same displacement
// gives its waterline's radius at every angle, from which the scene's fields are written: the
// shore's sea-level distance (negative over rock) and the floor's height, as the render pipeline's
// world fields (render/NeutralInputs.ts has the format). With those the sea damps its short waves
// near the rocks, casts a lee behind them and breaks shore waves against them.
import * as THREE from 'three/webgpu';
import { float, length, mix, mx_fractal_noise_float, normalWorld, positionWorld, smoothstep, vec3 } from 'three/tsl';
import { SimplexNoise } from 'three/addons/math/SimplexNoise.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import type { RenderParts } from './render/index.ts';
import { rgb, surface } from './surfaces.ts';

// Heights are metres above the mean sea level.
/** the deep floor round the rocks */
export const FLOOR = - 30;
/** the main stack's flat top, the lighthouse's ground */
export const STACK_TOP = 15.5;

export interface Rock {
  name: string;
  /** where its axis meets the mean sea */
  x: number; z: number;
  /** its profile from the floor up to the top's centre: ( radius, height ) */
  profile: [ number, number ][];
  /** the size of its relief: 1 for the main stack; the noise's heights and lengths scale with it */
  relief: number;
  /** its plan's stretch (the long axis's radius over the short one's) and the long axis's angle */
  stretch: number; yaw: number;
  /** the radius within which its top stays flat (the main stack's, under the lighthouse); 0 for a crag */
  flat: number;
  /** how far (radians) it leans over about its long axis: tilted beds, a lopsided crown */
  lean: number;
  seed: number;
  /** the lathe's rings up the profile and segments round it */
  rings: number; segments: number;
  /** the rubble round its foot: the floor's depth (m) at its waterline and the distance (m) over
   * which it falls away to the deep floor */
  apron: { depth: number; length: number };
  /** how much surf it is heard to take, 0 to 1 (0: none of its own; the sound groups the small ones) */
  surf: number;
}

/** The profile of a rock `r` metres in radius at the waterline and `top` high: flaring out under the
 * sea to the floor, a wall up to its crown (narrowing by `taper`), the crown rounded over. */
function crag( r: number, top: number, taper = 0.85 ): [ number, number ][] {

  const crown = Math.min( top, r * 0.7 );
  const shoulder = top - crown;
  const profile: [ number, number ][] = [ [ r * 2.2 + 3, FLOOR - 1 ], [ r * 1.7 + 1.5, - 14 ], [ r * 1.3, - 5 ], [ r * 1.08, - 1.5 ], [ r, 0 ] ];
  let r1 = r;
  if ( shoulder > 0.5 ) {

    for ( const t of [ 0.35, 0.7, 1 ] ) profile.push( [ r * ( 1 - ( 1 - taper ) * t ), shoulder * t ] );
    r1 = r * taper;

  }
  for ( let k = 1; k <= 6; k ++ ) {

    const a = k / 6 * Math.PI / 2;
    profile.push( [ r1 * Math.pow( Math.cos( a ), 0.5 ), shoulder + crown * Math.pow( Math.sin( a ), 0.7 ) ] );

  }
  return profile;

}

const around = ( angle: number, distance: number ) => ( { x: Math.cos( angle ) * distance, z: Math.sin( angle ) * distance } );

// The rocks. Angles are about the main stack from +x toward +z; the first view looks at it from
// about 0.75, so the lesser stacks stand off to either side and behind it, and the reef lies to the
// right in the middle distance.
export const ROCKS: Rock[] = [
  {
    name: 'sea stack', x: 0, z: 0,
    profile: [
      [ 16.5, FLOOR - 1 ], [ 13.6, - 20 ], [ 10.9, - 9 ], [ 9.0, - 3.5 ], [ 8.2, - 0.5 ], [ 7.9, 2.6 ], [ 7.6, 6 ],
      [ 7.3, 9.5 ], [ 6.9, 12.4 ], [ 6.6, 14.2 ], [ 6.2, 15.1 ], [ 4.5, STACK_TOP - 0.05 ], [ 0, STACK_TOP + 0.1 ],
    ],
    relief: 1, stretch: 1, yaw: 0, flat: 6.5, lean: 0, seed: 7, rings: 150, segments: 180, apron: { depth: 1.5, length: 7 }, surf: 1,
  },
  // a lesser stack off the lighthouse's west side, another far beyond it, and the stump of a third
  { name: 'sister stack', ...around( 2.75, 40 ), profile: crag( 4.3, 10.5, 0.72 ), relief: 0.8, stretch: 1.35, yaw: 0.5, flat: 0, lean: 0.1, seed: 11, rings: 90, segments: 120, apron: { depth: 1.2, length: 5 }, surf: 0.8 },
  { name: 'far stack', ...around( - 2.0, 52 ), profile: crag( 3.6, 7.5, 0.7 ), relief: 0.75, stretch: 1.2, yaw: 1.9, flat: 0, lean: - 0.14, seed: 13, rings: 70, segments: 96, apron: { depth: 1.2, length: 5 }, surf: 0.6 },
  { name: 'stump', ...around( - 1.05, 34 ), profile: crag( 3.3, 4.2, 0.85 ), relief: 0.7, stretch: 1.5, yaw: 1.2, flat: 0, lean: 0.17, seed: 17, rings: 60, segments: 96, apron: { depth: 1, length: 4.5 }, surf: 0.7 },
  // a low whaleback of rock in the debris's path
  { name: 'whaleback', ...around( 1.75, 30 ), profile: crag( 3.6, 2.4 ), relief: 0.6, stretch: 2, yaw: 0.2, flat: 0, lean: 0.08, seed: 19, rings: 44, segments: 96, apron: { depth: 1, length: 4 }, surf: 0.7 },
  // skerries round the stack's foot: ( angle, distance, radius, top )
  ...( [ [ - 0.35, 15, 2.0, 1.5 ], [ - 0.05, 18.5, 1.3, 0.8 ], [ 2.3, 15, 2.4, 2.0 ], [ 3.5, 14.5, 2.2, 1.7 ], [ 4.4, 16, 1.6, 1.0 ] ] as const ).map( ( [ angle, distance, radius, top ], i ): Rock => ( {
    name: `skerry ${ i + 1 }`, ...around( angle, distance ), profile: crag( radius, top ), relief: 0.5, stretch: 1.3 + ( i % 3 ) * 0.2, yaw: angle * 2.1,
    flat: 0, lean: ( i % 2 ? - 1 : 1 ) * ( 0.12 + i * 0.03 ), seed: 23 + i, rings: 32, segments: 56, apron: { depth: 0.8, length: 3 }, surf: i === 0 || i === 2 ? 0.6 : 0,
  } ) ),
  // a reef awash, the sea breaking over it
  ...( [ [ - 0.45, 44, 1.6, 0.7 ], [ - 0.55, 48.5, 1.1, 0.4 ], [ - 0.63, 52, 1.8, 0.9 ], [ - 0.71, 55.5, 1.0, 0.35 ] ] as const ).map( ( [ angle, distance, radius, top ], i ): Rock => ( {
    name: `reef ${ i + 1 }`, ...around( angle, distance ), profile: crag( radius, top ), relief: 0.4, stretch: 1.4 + ( i % 2 ) * 0.4, yaw: 0.9 + i,
    flat: 0, lean: ( i % 2 ? - 0.15 : 0.15 ), seed: 31 + i, rings: 24, segments: 40, apron: { depth: 0.6, length: 3 }, surf: i === 1 ? 0.8 : 0,
  } ) ),
];

// a seeded generator, so the rocks are the same on every load
function mulberry32( seed: number ) {

  let s = seed >>> 0;
  return () => {

    s = ( s + 0x6d2b79f5 ) >>> 0;
    let t = Math.imul( s ^ ( s >>> 15 ), 1 | s );
    t = ( t + Math.imul( t ^ ( t >>> 7 ), 61 | t ) ) ^ t;
    return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

  };

}

// the waterline's radius is tabled at this many angles round the axis
const WATERLINE_STEPS = 720;

/** A rock's shape: its mesh, the waterline measured on it, and its distance field. */
export class RockShape {

  readonly top: number;
  private readonly noise: SimplexNoise;
  private readonly waterline = new Float32Array( WATERLINE_STEPS );
  /** the mesh about its own axis, the mean sea at y = 0 */
  private readonly local: THREE.BufferGeometry;

  constructor( readonly rock: Rock ) {

    this.top = rock.profile[ rock.profile.length - 1 ][ 1 ];
    this.noise = new SimplexNoise( { random: mulberry32( rock.seed ) } );
    this.local = this.build();
    this.measureWaterline();

  }

  /** The waterline's radius at angle `theta` about the rock's axis (radians from +x toward +z). */
  waterlineRadius( theta: number ): number {

    const u = ( theta + Math.PI ) / ( 2 * Math.PI ) * WATERLINE_STEPS;
    const i = Math.floor( u ), f = u - i;
    const a = this.waterline[ ( ( i % WATERLINE_STEPS ) + WATERLINE_STEPS ) % WATERLINE_STEPS ];
    const b = this.waterline[ ( ( ( i + 1 ) % WATERLINE_STEPS ) + WATERLINE_STEPS ) % WATERLINE_STEPS ];
    return a + ( b - a ) * f;

  }

  /** How far (m) the point x, z at sea level lies outside the waterline, along the ray from the
   * axis; negative within it. */
  distance( x: number, z: number ): number {

    const dx = x - this.rock.x, dz = z - this.rock.z;
    return Math.hypot( dx, dz ) - this.waterlineRadius( Math.atan2( dz, dx ) );

  }

  /** The rock's geometry in world space, with the mean sea at y = `sea`. */
  geometry( sea: number ): THREE.BufferGeometry {

    const geometry = this.local.clone().translate( this.rock.x, sea, this.rock.z );
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    return geometry;

  }

  /** The mesh: the lathe, stretched, pushed out by the relief, a crag's crown broken and the whole
   * crag jostled by a 3D noise (so it reads as broken stone, not a turned one) and leaned over. */
  private build(): THREE.BufferGeometry {

    const { rock } = this;
    let geometry: THREE.BufferGeometry = new THREE.LatheGeometry( lathePoints( rock.profile, rock.rings ), rock.segments );
    // weld the lathe's seam so the normals run smoothly across it
    geometry.deleteAttribute( 'uv' );
    geometry.deleteAttribute( 'normal' );
    geometry = mergeVertices( geometry, 1e-4 );
    const position = geometry.attributes.position as THREE.BufferAttribute;
    const v = new THREE.Vector3();
    const s = rock.relief;
    for ( let i = 0; i < position.count; i ++ ) {

      v.fromBufferAttribute( position, i );
      const radius = Math.hypot( v.x, v.z );
      if ( radius > 1e-6 ) {

        const theta = Math.atan2( v.z, v.x );
        const r = radius * this.stretchAt( theta ) + this.offset( v.x, v.y, v.z, radius );
        v.x = Math.cos( theta ) * r;
        v.z = Math.sin( theta ) * r;

      }
      if ( rock.flat > 0 ) {

        // the flat top: a gentle uneven surface
        if ( v.y > this.top - 1 ) v.y += this.noise.noise3d( v.x * 0.25, 0, v.z * 0.25 ) * 0.12 * THREE.MathUtils.smoothstep( rock.flat - radius, 0, 2 );

      } else {

        // a crag's crown: broken, not turned
        v.y += this.noise.noise3d( v.x * 0.5 / s, 7, v.z * 0.5 / s ) * 0.5 * s * THREE.MathUtils.smoothstep( v.y, this.top - 2.5 * s, this.top );
        // and the whole of it jostled in three dimensions, which breaks the lathe's symmetry
        const f = 0.3 / s, a = 0.8 * s;
        v.x += this.noise.noise3d( v.x * f + 11.3, v.y * f, v.z * f ) * a;
        v.y += this.noise.noise3d( v.x * f, v.y * f + 5.7, v.z * f + 3.1 ) * a * 0.6;
        v.z += this.noise.noise3d( v.x * f + 7.9, v.y * f + 2.2, v.z * f ) * a;

      }
      position.setXYZ( i, v.x, v.y, v.z );

    }
    if ( rock.lean !== 0 ) geometry.applyMatrix4( new THREE.Matrix4().makeRotationAxis( new THREE.Vector3( Math.cos( rock.yaw ), 0, Math.sin( rock.yaw ) ), rock.lean ) );
    geometry.computeVertexNormals();
    return geometry;

  }

  /** The waterline's radius round the axis, where the mesh's edges cross the mean sea: the farthest
   * crossing at each angle, the angles no edge crosses filled between their neighbours. */
  private measureWaterline(): void {

    const position = this.local.attributes.position as THREE.BufferAttribute;
    const index = this.local.index!.array;
    const table = this.waterline.fill( - 1 );
    const cross = ( a: number, b: number ) => {

      const ya = position.getY( a ), yb = position.getY( b );
      if ( ( ya > 0 ) === ( yb > 0 ) ) return;
      const t = ya / ( ya - yb );
      const x = position.getX( a ) + ( position.getX( b ) - position.getX( a ) ) * t;
      const z = position.getZ( a ) + ( position.getZ( b ) - position.getZ( a ) ) * t;
      const bin = Math.round( ( Math.atan2( z, x ) + Math.PI ) / ( 2 * Math.PI ) * WATERLINE_STEPS ) % WATERLINE_STEPS;
      table[ bin ] = Math.max( table[ bin ], Math.hypot( x, z ) );

    };
    for ( let k = 0; k < index.length; k += 3 ) {

      cross( index[ k ], index[ k + 1 ] );
      cross( index[ k + 1 ], index[ k + 2 ] );
      cross( index[ k + 2 ], index[ k ] );

    }
    const filled: number[] = [];
    table.forEach( ( r, i ) => {

      if ( r >= 0 ) filled.push( i );

    } );
    for ( let n = 0; n < filled.length; n ++ ) {

      const i0 = filled[ n ], i1 = filled[ ( n + 1 ) % filled.length ];
      const span = ( ( i1 - i0 ) + WATERLINE_STEPS ) % WATERLINE_STEPS || WATERLINE_STEPS;
      for ( let k = 1; k < span; k ++ ) table[ ( i0 + k ) % WATERLINE_STEPS ] = table[ i0 ] + ( table[ i1 ] - table[ i0 ] ) * k / span;

    }

  }

  /** How far the rock's surface moves out along its radius at a point of the undisplaced lathe
   * (in the rock's own frame). */
  private offset( x: number, y: number, z: number, radius: number ): number {

    const { rock } = this;
    const s = rock.relief, n = this.noise;
    // only the cliff moves outward; a flat top's centre stays put, and a crag's apex
    const cliff = rock.flat > 0 ? THREE.MathUtils.smoothstep( radius, rock.flat - 3, rock.flat ) : THREE.MathUtils.smoothstep( radius, 0, 0.8 * s );
    const large = n.noise3d( x * 0.06 / s, y * 0.04 / s, z * 0.06 / s ) * 1.9 * s;
    const medium = n.noise3d( x * 0.18 / s, y * 0.2 / s, z * 0.18 / s ) * 0.6 * s;
    const fine = n.noise3d( x * 0.7 / s, y * 0.9 / s, z * 0.7 / s ) * 0.14 * s;
    // vertical joints: grooves and buttresses stretched up the cliff
    const crown = rock.flat > 0 ? 0 : THREE.MathUtils.smoothstep( y, this.top - 1.5 * s, this.top );
    const joints = Math.abs( n.noise3d( x * 0.32 / s, y * 0.035 / s, z * 0.32 / s ) ) * - 1.1 * s * ( 1 - 0.8 * crown );
    // layers: each bed juts out at its base and recedes upward (a sawtooth), the beds of uneven
    // thickness and the ledges stronger in some places than others
    const bed = ( y + n.noise3d( x * 0.08 / s, y * 0.05 / s, z * 0.08 / s ) * 1.4 * s ) / ( 3.1 * s );
    const strength = THREE.MathUtils.clamp( 0.5 + n.noise3d( x * 0.05 / s, y * 0.03 / s + 9, z * 0.05 / s ), 0, 1 );
    const layer = Math.pow( 1 - ( bed - Math.floor( bed ) ), 2 ) * 0.5 * s * strength;
    // on a crag's crown it fades: grown from the apex outward it would rib the crown like petals
    return ( large + medium + fine + joints + layer ) * cliff * ( 1 - 0.7 * crown );

  }

  /** The plan's stretch at angle `theta`: how far an ellipse of the rock's stretch (and the area of
   * a unit circle) reaches along it. */
  private stretchAt( theta: number ): number {

    const { stretch, yaw } = this.rock;
    if ( stretch === 1 ) return 1;
    const a = Math.sqrt( stretch ), b = 1 / a;
    const c = Math.cos( theta - yaw ), s = Math.sin( theta - yaw );
    return 1 / Math.sqrt( ( c / a ) ** 2 + ( s / b ) ** 2 );

  }

}

/** The lathe's points along a smooth curve through `profile`: `rings` of them, spaced along it, but
 * five times as far apart on the deep flanks, which only the refraction sees, as above them. */
function lathePoints( profile: [ number, number ][], rings: number ): THREE.Vector2[] {

  const curve = new THREE.SplineCurve( profile.map( ( [ r, y ] ) => new THREE.Vector2( r, y ) ) );
  const fine = curve.getSpacedPoints( 1200 );
  const along = [ 0 ];
  for ( let i = 1; i < fine.length; i ++ ) {

    const y = ( fine[ i ].y + fine[ i - 1 ].y ) / 2;
    along.push( along[ i - 1 ] + fine[ i ].distanceTo( fine[ i - 1 ] ) * ( y < - 4 ? 0.2 : 1 ) );

  }
  const total = along[ along.length - 1 ];
  const points: THREE.Vector2[] = [];
  let j = 0;
  for ( let k = 0; k <= rings; k ++ ) {

    const target = total * k / rings;
    while ( j < along.length - 2 && along[ j + 1 ] < target ) j ++;
    const f = THREE.MathUtils.clamp( ( target - along[ j ] ) / Math.max( along[ j + 1 ] - along[ j ], 1e-9 ), 0, 1 );
    const point = fine[ j ].clone().lerp( fine[ j + 1 ], f );
    point.x = Math.max( point.x, 0 );
    points.push( point );

  }
  // close the top exactly on the axis
  points[ points.length - 1 ].set( 0, profile[ profile.length - 1 ][ 1 ] );
  return points;

}

/** How far (m) the point x, z at sea level lies from the nearest rock's waterline; negative within one. */
export function shoreDistance( rocks: readonly RockShape[], x: number, z: number ): number {

  let d = Infinity;
  for ( const rock of rocks ) d = Math.min( d, rock.distance( x, z ) );
  return d;

}

/** The rocks' material: layered stone, dark and glossy where the sea wets it, weed at the waterline,
 * grass on the main stack's flat top, whitewash where the seabirds roost on the outer stacks. */
export function rockMaterial( render: RenderParts, sea: number ): THREE.MeshStandardNodeMaterial {

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
  // whitewash: the gulls roost on the outer stacks' tops and ledges
  const roost = smoothstep( 16, 22, length( p.xz ) ).mul( smoothstep( 0.5, 0.85, normalWorld.y.add( grain.mul( 0.2 ) ) ) )
    .mul( smoothstep( 2.5, 4.5, h ) ).mul( smoothstep( - 0.25, 0.25, big.add( streaks.mul( 0.5 ) ) ) );
  rock = mix( rock, rgb( 0xb9b4a6 ), roost.mul( 0.45 ) );
  // the splash zone: wet stone up to a ragged line a metre or two above the mean sea
  const splash = float( 1.6 ).add( big.mul( 0.9 ) );
  const wet = float( 1 ).sub( smoothstep( splash.sub( 0.6 ), splash, h ) );
  // weed and barnacles round the mean waterline
  const weedBand = smoothstep( - 1.2, - 0.2, h ).mul( float( 1 ).sub( smoothstep( 0.4, 1.1, h.add( grain.mul( 0.4 ) ) ) ) );
  const weed = mix( rgb( 0x2c2a1a ), rgb( 0x45422b ), grain.mul( 0.5 ).add( 0.5 ) );
  // grass on the main stack's flat top
  const up = smoothstep( 0.72, 0.9, normalWorld.y ).mul( smoothstep( STACK_TOP - 1.2, STACK_TOP - 0.4, h ) );
  const grass = mix( rgb( 0x4d5a28 ), rgb( 0x7a7a3c ), smoothstep( - 0.3, 0.5, grain.add( big ) ) );
  let colour = mix( rock, rock.mul( 0.45 ), wet );
  colour = mix( colour, weed, weedBand );
  colour = mix( colour, grass, up.mul( smoothstep( - 0.2, 0.1, big.add( 0.25 ) ) ) );
  const roughness = mix( float( 0.88 ).sub( grain.mul( 0.06 ) ), float( 0.6 ), wet.mul( float( 1 ).sub( up ) ) );
  return surface( render, { name: 'rock', color: colour, roughness } );

}

/** The scene's fields round the rocks, in the formats NeutralInputs.ts `neutralFields` returns. */
export function rockFields( { rocks, half, cells, sea, xform }: { rocks: readonly RockShape[]; half: number; cells: number; sea: number; xform: THREE.Vector4 } ) {

  const shore = new Float32Array( cells * cells ), floor = new Float32Array( cells * cells );
  const cell = 2 * half / cells;
  for ( let j = 0; j < cells; j ++ ) for ( let i = 0; i < cells; i ++ ) {

    const x = - half + ( i + 0.5 ) * cell, z = - half + ( j + 0.5 ) * cell;
    let distance = Infinity, height = sea + FLOOR, inside = - Infinity;
    for ( const rock of rocks ) {

      const d = rock.distance( x, z );
      distance = Math.min( distance, d );
      if ( d < 0 ) {

        inside = Math.max( inside, Math.min( rock.top, 6 ) );

      } else {

        // a rocky apron round each foot, dropping to the deep floor
        const { depth, length } = rock.rock.apron;
        height = Math.max( height, sea + FLOOR + ( - depth - FLOOR ) * Math.exp( - d / length ) );

      }

    }
    shore[ j * cells + i ] = distance;
    floor[ j * cells + i ] = inside > - Infinity ? sea + inside : height;

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
