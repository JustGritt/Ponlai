// The water the render pipeline does not draw: the falls' sheets, the river above the main falls,
// the foam on the pool, and the mist, droplets and splashes.
//
// The render pipeline's water is one level, the pool's (the sea's level). The rest of the river
// stands higher, so it is drawn here, in the water's pass (render.waterScene), after the opaque
// image is resolved: it reads that image and its depth to show the bed through the water, as the
// pipeline's water does. Nothing in that pass is lit by the scene's lights, so each shader lights
// itself as the pipeline's water and spray do (ocean/Water.ts, ocean/Spray.ts): the key (the sun or
// the moon) cut by the clouds and the sun's shadow map at the point, and the sky's irradiance.
// - The river (`streamSurface`): an opaque surface. It mirrors the sky above the gorge's horizon
//   and the gorge below it (each vertex carries its horizon in eight directions, scanned over the
//   heights), shows the bed and anything in the water through it, absorbed along the path (the
//   pool's water: shared.sigA), and carries ripples and foam with its current (two phases of a
//   flow map, each reset while the other shows).
// - The falls (`fallsSheets`): two sheets per falls, the water's core and an outer veil, each a
//   strip under the crest along the water's ballistic path. Their streaks are noise read at the
//   time the water left the crest, so they travel with it and stretch as it speeds up. Glassy at the
//   crest, whiter and more broken as they fall. Blended over the scene, premultiplied.
// - The pool's foam (`poolFoam`): lace carried on a flow out from the falls' feet, blended over
//   the pool a few centimetres above its mean level.
// - The mist and the droplets at the falls' feet (`fallsMist`), each particle a function of the
//   scene's clock, and the splashes (`splashes`) the fish throw up, spawned from the CPU.
// They follow the scene's clock (shared.time), so a paused page holds them.
import * as THREE from 'three/webgpu';
import {
  Discard, Fn, If, abs, atan, attribute, cameraPosition, cameraViewMatrix, clamp, cross, dot, exp, float, fract, ivec2, length, max, mix,
  mx_fractal_noise_float, mx_noise_float, mx_noise_vec3, mx_worley_noise_vec2, normalWorld, normalize, positionGeometry, positionView, positionWorld,
  pow, reflect, screenUV, select, smoothstep, texture, textureLoad, textureSize, varying, vec2, vec3, vec4,
} from 'three/tsl';
import { shared, type RenderParts } from '../render/index.ts';
import { sceneUnits } from '../render/post/PreExposure.ts';
import { mulberry32 } from '../wreckage.ts';
import { HORIZON_DIRECTIONS, type HeightGrid } from './valley.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

const G = 9.81;
// the water's own light, as an irradiance reflectance (0.33 bb / ( a + bb ) for the pool's water)
const BODY = [ 0.0012, 0.006, 0.004 ] as const;
// what the gorge reflects where the river's mirror meets it: an albedo for the sky's and the key's light
const GORGE = [ 0.06, 0.07, 0.045 ] as const;
const FOAM_ALBEDO = 0.82;

// ---- the light and the opaque image

interface WaterInputs {
  /** the key's visibility at p: the clouds' and the sun's shadow map's */
  key( p: V3 ): F;
  /** the opaque image at a screen uv, in scene units */
  scene( q: V2 ): V3;
  /** its view-space z there */
  sceneZ( q: V2 ): F;
}

function waterInputs( render: RenderParts ): WaterInputs {

  const { sky, sunShadow, pipeline } = render;
  const tDepth = texture( pipeline.depthRT.texture );
  const depthSize = vec2( textureSize( tDepth ) as unknown as THREE.Node<'ivec2'> );
  const pixel = ( q: V2 ) => ivec2( clamp( q.mul( depthSize ), vec2( 0 ), depthSize.sub( 1 ) ) );
  return {
    key: ( p ) => sky.seaKeyVisibility( p ).mul( sunShadow.volumeVisibility( sunShadow.coordinate.mul( vec4( p, 1 ) ).xyz ) ),
    scene: ( q ) => sceneUnits( texture( pipeline.opaqueRT.texture, q ).level( float( 0 ) ).rgb ),
    sceneZ: ( q ) => textureLoad( tDepth, pixel( q ) ).x as unknown as F,
  };

}

/** Henyey-Greenstein's phase function. */
const phaseHG = ( cosT: F, g: number ) => float( ( 1 - g * g ) / ( 4 * Math.PI ) ).div( pow( float( 1 + g * g ).sub( cosT.mul( 2 * g ) ), 1.5 ) );

/** The sine of the horizon's elevation toward `dir`'s azimuth, from the eight a vertex carries. */
function horizonToward( a: V4, b: V4, dir: V3 ): F {

  const az = atan( dir.z, dir.x );
  const sines = [ a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w ];
  let sum: F = float( 0 );
  for ( let k = 0; k < HORIZON_DIRECTIONS; k ++ ) {

    const ak = k / HORIZON_DIRECTIONS * Math.PI * 2;
    const diff = abs( fract( az.sub( ak ).div( Math.PI * 2 ).add( 0.5 ) ).sub( 0.5 ) ).mul( HORIZON_DIRECTIONS );
    sum = sum.add( max( float( 1 ).sub( diff ), 0 ).mul( sines[ k ] ) );

  }
  return sum;

}

/** What the water mirrors toward R: the sky over the horizon, the gorge (lit by the sky and the key)
 * under it. */
function mirrored( render: RenderParts, R: V3, a: V4, b: V4, key: F, mip: number ): V3 {

  const horizon = horizonToward( a, b, R );
  const open = smoothstep( horizon.sub( 0.05 ), horizon.add( 0.05 ), R.y );
  const sky = render.sky.sample( normalize( vec3( R.x, max( R.y, 0.004 ), R.z ) ), float( mip ) );
  const gorge = vec3( ...GORGE ).div( Math.PI ).mul( shared.skyIrradianceSource.mul( 0.6 ).add( shared.sunColor.mul( key ).mul( max( shared.sunDir.y, 0 ).mul( 0.35 ) ) ) );
  return mix( gorge, sky, open );

}

/** The sun's glint on the water (a normalized Blinn-Phong lobe, Fresnel weighted). */
function glint( N: V3, V: V3, key: F, shininess: number ): V3 {

  const L = shared.sunDir;
  const H = normalize( V.add( L ) );
  const f = float( 0.02 ).add( pow( float( 1 ).sub( max( dot( V, H ), 0 ) ), 5 ).mul( 0.98 ) );
  const d = pow( max( dot( N, H ), 0 ), shininess ).mul( ( shininess + 8 ) / ( 8 * Math.PI ) );
  return shared.sunColor.mul( key ).mul( d.mul( f ).mul( max( dot( N, L ), 0 ) ) ).min( vec3( 2e5 ) );

}

/** White water's light: a thick scatterer, lit round its edges ( wrap ), and through its thin parts
 * toward the key ( `thin`, 0 to 1 ). */
function foamLight( N: V3, V: V3, key: F, sky: F, thin: F ): V3 {

  const L = shared.sunDir;
  const wrap = dot( N, L ).mul( 0.5 ).add( 0.5 );
  const through = phaseHG( dot( V.negate(), L ), 0.55 ).mul( thin ).mul( 0.55 );
  const sun = shared.sunColor.mul( key ).mul( wrap.mul( FOAM_ALBEDO / Math.PI ).add( through ) );
  return sun.add( shared.skyIrradianceSource.mul( sky ).mul( FOAM_ALBEDO / Math.PI ) );

}

const fresnel = ( N: V3, V: V3 ) => float( 0.02 ).add( pow( float( 1 ).sub( max( dot( N, V ), 0 ) ), 5 ).mul( 0.98 ) );

/** A flow map's two phases (each reset while the other shows) of `sample` at p carried by v. */
function carried<T extends THREE.Node>( p: V2, v: V2, period: number, sample: ( q: V2 ) => T ): T {

  const t = shared.time.div( period );
  const ph0 = fract( t ), ph1 = fract( t.add( 0.5 ) );
  const w0 = float( 1 ).sub( abs( ph0.mul( 2 ).sub( 1 ) ) );
  const a = sample( p.sub( v.mul( ph0.mul( period ) ) ) );
  const b = sample( p.sub( v.mul( ph1.mul( period ) ) ).add( vec2( 7.31, 3.17 ) ) );
  return mix( b as unknown as V3, a as unknown as V3, w0 ) as unknown as T;

}

/** Foam's pattern, 0 to 1: drifting rafts (smooth noise), laced with bubbles' edges (Worley cells'). */
function lace( q: V2, scale: number ): F {

  const rafts = mx_fractal_noise_float( vec3( q.mul( scale * 0.55 ), 1.7 ), 3, 2.1, 0.5, 1 ).mul( 0.5 ).add( 0.5 );
  // the cells warped, so their edges wander rather than tile
  const warp = mx_noise_vec3( vec3( q.mul( scale * 0.9 ), 4.2 ) ).xy.mul( 0.45 );
  const c = mx_worley_noise_vec2( q.mul( scale * 2.2 ).add( warp ) );
  return rafts.mul( 0.74 ).add( smoothstep( 0.08, 0.0, c.y.sub( c.x ) ).mul( 0.12 ) ).add( smoothstep( 0.5, 0.1, c.x ).mul( 0.14 ) );

}

/** How much of a pattern 0 to 1 foam covers at a density 0 to 1: the pattern's peaks at a low
 * density, nearly all of it at 1. Far off, where its cells shrink under a pixel, the pattern gives
 * way to its mean. */
const coverage = ( pattern: F, density: F, p: V3 ) => {

  const far = smoothstep( 16, 45, length( cameraPosition.sub( p ) ) );
  const threshold = float( 1.2 ).sub( density.mul( 1.4 ) );
  return smoothstep( threshold.sub( 0.15 ), threshold.add( 0.15 ), mix( pattern, float( 0.4 ), far ) );

};

/** The horizon (two vec4s) and the sky's share at each vertex of `position`. */
function horizonAttributes( geometry: THREE.BufferGeometry, grid: HeightGrid, sea: number ): void {

  const p = geometry.getAttribute( 'position' ) as THREE.BufferAttribute;
  const a = new Float32Array( p.count * 4 ), b = new Float32Array( p.count * 4 ), sky = new Float32Array( p.count );
  const h = new Float32Array( HORIZON_DIRECTIONS );
  for ( let i = 0; i < p.count; i ++ ) {

    const x = p.getX( i ), y = p.getY( i ) - sea + 0.1, z = p.getZ( i );
    grid.horizon( x, y, z, h );
    a.set( h.subarray( 0, 4 ), i * 4 );
    b.set( h.subarray( 4, 8 ), i * 4 );
    let s = 0;
    for ( let k = 0; k < HORIZON_DIRECTIONS; k ++ ) s += 1 - Math.max( h[ k ], 0 ) ** 2;
    sky[ i ] = s / HORIZON_DIRECTIONS;

  }
  geometry.setAttribute( 'aHorizonA', new THREE.BufferAttribute( a, 4 ) );
  geometry.setAttribute( 'aHorizonB', new THREE.BufferAttribute( b, 4 ) );
  geometry.setAttribute( 'aSky', new THREE.BufferAttribute( sky, 1 ) );

}

// ---- the river

/** A reach of the river drawn here: its surface on a grid of `across` by `along` points, each placed
 * by `at` ( u across, s along, both 0 to 1 ) with its current ( vx, vz ) and its foam's density. */
export interface Reach {
  across: number; along: number;
  at( u: number, s: number ): { x: number; y: number; z: number; vx: number; vz: number; foam: number };
}

/** The river's surface over the given reaches (world positions, `sea` added). */
export function streamSurface( render: RenderParts, grid: HeightGrid, sea: number, reaches: readonly Reach[] ): THREE.Mesh {

  const parts = reaches.map( ( reach ) => {

    const { across: nu, along: ns } = reach;
    const position = new Float32Array( ( nu + 1 ) * ( ns + 1 ) * 3 ), flow = new Float32Array( ( nu + 1 ) * ( ns + 1 ) * 4 );
    for ( let j = 0; j <= ns; j ++ ) for ( let i = 0; i <= nu; i ++ ) {

      const k = j * ( nu + 1 ) + i, w = reach.at( i / nu, j / ns );
      position.set( [ w.x, sea + w.y, w.z ], k * 3 );
      flow.set( [ w.vx, w.vz, w.foam, 0 ], k * 4 );

    }
    const index: number[] = [];
    for ( let j = 0; j < ns; j ++ ) for ( let i = 0; i < nu; i ++ ) {

      const a = j * ( nu + 1 ) + i, b = a + 1, c = a + nu + 1, d = c + 1;
      // (u runs along +x and s upstream, toward -z: this winding faces up)
      index.push( a, b, c, b, d, c );

    }
    const g = new THREE.BufferGeometry();
    g.setAttribute( 'position', new THREE.BufferAttribute( position, 3 ) );
    g.setAttribute( 'aFlow', new THREE.BufferAttribute( flow, 4 ) );
    g.setIndex( index );
    return g;

  } );
  const geometry = parts.length === 1 ? parts[ 0 ] : mergeAll( parts );
  horizonAttributes( geometry, grid, sea );
  geometry.computeBoundingSphere();

  const inputs = waterInputs( render );
  const flow = attribute( 'aFlow', 'vec4' );
  const hA = attribute( 'aHorizonA', 'vec4' ), hB = attribute( 'aHorizonB', 'vec4' ), sky = attribute( 'aSky', 'float' );
  const material = new THREE.MeshBasicNodeMaterial();
  material.name = 'river';
  material.colorNode = Fn( () => {

    const p = positionWorld;
    const V = normalize( cameraPosition.sub( p ) ).toVar();
    const v = flow.xy, speed = length( v );
    // ripples carried by the current, rougher where it runs fast
    const ripple = ( q: V2 ) => mx_noise_vec3( vec3( q.mul( 1.2 ), shared.time.mul( 0.4 ) ) ).xy.mul( 0.6 )
      .add( mx_noise_vec3( vec3( q.mul( 3.7 ), shared.time.mul( 1.1 ).add( 5 ) ) ).xy.mul( 0.35 ) );
    const slope = carried( p.xz, v, 1.6, ripple ).mul( speed.mul( 0.22 ).add( 0.07 ) ).toVar();
    const N = normalize( vec3( slope.x.negate(), 1, slope.y.negate() ) ).toVar();
    const foam = coverage( carried( p.xz, v, 1.6, ( q ) => lace( q, 1.6 ) ), flow.z, p ).toVar();
    const key = inputs.key( p ).toVar();
    // what it mirrors
    const R = reflect( V.negate(), N ).toVar();
    const F = fresnel( N, V ).toVar();
    const mirror = mirrored( render, R, hA, hB, key, 2 );
    // what lies under it, seen through the water: refracted on the screen, absorbed along its path
    const nView = cameraViewMatrix.mul( vec4( N, 0 ) ).xyz;
    const surfaceZ = positionView.z;
    const uvR = screenUV.add( vec2( nView.x, nView.y.negate() ).mul( 0.035 ) ).toVar();
    const uvB = select( inputs.sceneZ( uvR ).lessThan( surfaceZ.sub( 0.02 ) ), uvR, screenUV ).toVar();
    const path = max( surfaceZ.sub( inputs.sceneZ( uvB ) ), 0 ).mul( length( positionView ).div( max( surfaceZ.negate(), 0.01 ) ) ).min( 60 );
    const through = exp( shared.sigA.add( shared.scatter.mul( 0.3 ) ).mul( path ).negate() ).toVar();
    const body = shared.skyIrradianceSource.mul( sky ).mul( vec3( ...BODY ).div( Math.PI ) ).mul( float( 1 ).sub( through ) );
    let colour: V3 = mirror.mul( F ).add( inputs.scene( uvB ).mul( through ).add( body ).mul( float( 1 ).sub( F ) ) ).add( glint( N, V, key, 300 ) );
    colour = mix( colour, foamLight( N, V, key, sky, float( 0.2 ) ), foam );
    return vec4( colour, 1 );

  } )();
  const mesh = new THREE.Mesh( geometry, material );
  mesh.name = 'river';
  mesh.renderOrder = 2;
  return mesh;

}

function mergeAll( parts: THREE.BufferGeometry[] ): THREE.BufferGeometry {

  const out = new THREE.BufferGeometry();
  let vertices = 0;
  const names = Object.keys( parts[ 0 ].attributes );
  const arrays: Record<string, number[]> = {};
  for ( const name of names ) arrays[ name ] = [];
  const index: number[] = [];
  for ( const g of parts ) {

    for ( const name of names ) arrays[ name ].push( ...( g.getAttribute( name ).array as Float32Array ) );
    for ( const i of g.index!.array ) index.push( i + vertices );
    vertices += g.getAttribute( 'position' ).count;

  }
  for ( const name of names ) out.setAttribute( name, new THREE.BufferAttribute( new Float32Array( arrays[ name ] ), parts[ 0 ].getAttribute( name ).itemSize ) );
  out.setIndex( index );
  return out;

}

// ---- the falls

/** A falls: the crest's half width and line, the water's height at the lip and where it lands (over
 * the sea), its speed off the lip and how much goes over (0 to 1), along the crest, and how fast its
 * veil spreads outward as it falls (m/s). */
export interface FallsSpec {
  name: string;
  half: number;
  crest: ( x: number ) => number;
  lip: number;
  plunge: number;
  speed: ( x: number ) => number;
  flow: ( x: number ) => number;
  spread: number;
}

/** Where the water off the crest at x is t seconds later, and its outward heading: the crest's
 * normal in the ground plane, toward +z. */
export function fallsPoint( spec: FallsSpec, x: number, t: number, out = new THREE.Vector3() ): THREE.Vector3 {

  const z0 = spec.crest( x ), slope = ( spec.crest( x + 0.01 ) - spec.crest( x - 0.01 ) ) / 0.02;
  const l = Math.hypot( slope, 1 ), dx = - slope / l, dz = 1 / l;
  const v0 = spec.speed( x ), tt = Math.max( t, 0 );
  return out.set( x + dx * v0 * t, spec.lip - 0.5 * G * tt * tt, z0 + dz * v0 * t );

}

/** The time the water off the crest takes to reach its plunge. */
export const fallTime = ( spec: FallsSpec ) => Math.sqrt( 2 * ( spec.lip - spec.plunge ) / G );

/** The falls' two sheets (the core and the veil), drawn over the scene. */
export function fallsSheets( render: RenderParts, grid: HeightGrid, sea: number, spec: FallsSpec ): THREE.Mesh {

  const nu = Math.ceil( spec.half * 2 / 0.12 ), nt = 30;
  const tEnd = fallTime( spec ) + 0.05, tStart = - 0.14;
  const positions: number[] = [], normals: number[] = [], sheet: number[] = [], index: number[] = [];
  const p = new THREE.Vector3(), q = new THREE.Vector3(), r = new THREE.Vector3();
  for ( const layer of [ 0, 1 ] ) {

    const base = positions.length / 3;
    for ( let i = 0; i <= nu; i ++ ) {

      const x = - spec.half + 2 * spec.half * i / nu;
      const slope = ( spec.crest( x + 0.01 ) - spec.crest( x - 0.01 ) ) / 0.02;
      const l = Math.hypot( slope, 1 ), dx = - slope / l, dz = 1 / l;
      const v0 = spec.speed( x );
      for ( let j = 0; j <= nt; j ++ ) {

        const t = tStart + ( tEnd - tStart ) * ( j / nt ) ** 1.2;
        fallsPoint( spec, x, t, p );
        // the veil stands off the core and spreads as it falls
        const off = layer === 0 ? 0 : 0.05 + spec.spread * Math.max( t, 0 );
        p.x += dx * off;
        p.z += dz * off;
        positions.push( p.x, sea + p.y, p.z );
        // the normal: across the crest x down the path, outward
        q.set( 1, 0, slope ).normalize();
        r.set( dx * v0, - G * Math.max( t, 0 ), dz * v0 ).normalize();
        const n = new THREE.Vector3().crossVectors( q, r ).normalize();
        if ( n.x * dx + n.z * dz < 0 ) n.negate();
        normals.push( n.x, n.y, n.z );
        sheet.push( x + spec.half, t, spec.flow( x ), layer );

      }

    }
    for ( let i = 0; i < nu; i ++ ) for ( let j = 0; j < nt; j ++ ) {

      const a = base + i * ( nt + 1 ) + j, b = a + 1, c = a + nt + 1, d = c + 1;
      index.push( a, b, c, b, d, c );

    }

  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( positions, 3 ) );
  geometry.setAttribute( 'normal', new THREE.Float32BufferAttribute( normals, 3 ) );
  geometry.setAttribute( 'aSheet', new THREE.Float32BufferAttribute( sheet, 4 ) );
  geometry.setIndex( index );
  horizonAttributes( geometry, grid, sea );
  geometry.computeBoundingSphere();

  const inputs = waterInputs( render );
  const s = attribute( 'aSheet', 'vec4' );
  const hA = attribute( 'aHorizonA', 'vec4' ), hB = attribute( 'aHorizonB', 'vec4' ), sky = attribute( 'aSky', 'float' );
  const end = fallTime( spec );
  const material = new THREE.MeshBasicNodeMaterial( { side: THREE.DoubleSide, transparent: true, depthWrite: false } );
  material.name = `${ spec.name } sheets`;
  material.forceSinglePass = true;
  material.blending = THREE.CustomBlending;
  material.blendEquation = THREE.AddEquation;
  material.blendSrc = material.blendSrcAlpha = THREE.OneFactor;
  material.blendDst = material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
  const shaded = Fn( () => {

    const along = s.x, t = s.y, flowAt = s.z, veil = s.w;
    const age = max( t, 0 ), fallen = age.div( end ).toVar();
    // the streaks, read where the water was when it left the crest: they travel with it
    const left = shared.time.sub( t );
    const broad = mx_fractal_noise_float( vec3( along.mul( 3.0 ), left.mul( 2.0 ), veil.mul( 5.3 ) ), 3, 2.1, 0.55, 1 ).toVar();
    const fine = mx_fractal_noise_float( vec3( along.mul( 13 ), left.mul( 7 ), veil.mul( 3.1 ).add( 13 ) ), 2, 2.2, 0.5, 1 ).toVar();
    // the aerated water's grain, finer still
    const grain = mx_noise_float( vec3( along.mul( 38 ), left.mul( 21 ), veil.add( 7 ) ) ).toVar();
    const thick = flowAt.mul( broad.mul( 0.4 ).add( 0.8 ) ).add( fine.mul( 0.18 ) ).toVar();
    // white as it fills with air; the veil is white from the start
    const air = max( smoothstep( 0.02, 0.32, age ), veil.mul( 0.7 ) );
    const foam = clamp( air.mul( thick.mul( 0.7 ).add( 0.45 ) ).add( fine.mul( 0.25 ) ).add( grain.mul( 0.12 ) ), 0, 1 ).toVar();
    // the core breaks up as it falls, the veil more, and the veil only forms below the crest
    const breakup = mix( float( 0.12 ).add( fallen.mul( 0.4 ) ), float( 0.42 ).add( fallen.mul( 0.45 ) ), veil );
    const cover = smoothstep( breakup, breakup.add( 0.18 ), thick.add( fine.mul( 0.15 ) ).add( grain.mul( 0.06 ) ) )
      .mul( mix( float( 0.95 ), smoothstep( 0.04, 0.3, age ).mul( 0.5 ), veil ) ).mul( smoothstep( - 0.14, - 0.06, t ) ).toVar();
    If( cover.lessThan( 0.003 ), () => {

      Discard();

    } );
    const p = positionWorld;
    const V = normalize( cameraPosition.sub( p ) ).toVar();
    const Ng = normalize( normalWorld );
    const facing = select( dot( Ng, V ).lessThan( 0 ), Ng.negate(), Ng );
    const across = normalize( cross( facing, vec3( 0, 1, 0 ) ).add( vec3( 1e-4, 0, 0 ) ) );
    const N = normalize( facing.add( across.mul( fine.mul( 0.35 ) ) ) ).toVar();
    const key = inputs.key( p ).toVar();
    // glassy where it leaves the crest: the rock behind through it, the sky and the gorge on it
    const R = reflect( V.negate(), N );
    const F = fresnel( N, V );
    const nView = cameraViewMatrix.mul( vec4( N, 0 ) ).xyz;
    // (a sheet some centimetres thick: the rock through it darkened and green, the water's own light)
    const behind = inputs.scene( screenUV.add( vec2( nView.x, nView.y.negate() ).mul( 0.02 ) ) ).mul( vec3( 0.42, 0.6, 0.52 ) )
      .add( shared.skyIrradianceSource.mul( sky ).mul( vec3( 0.006, 0.02, 0.016 ).div( Math.PI ) ) );
    let colour: V3 = mirrored( render, R, hA, hB, key, 3 ).mul( F ).add( behind.mul( float( 1 ).sub( F ) ) ).add( glint( N, V, key, 120 ) );
    const white = foamLight( N, V, key, sky, float( 1 ).sub( thick ).clamp( 0.15, 1 ) ).mul( thick.mul( 0.25 ).add( 0.75 ).min( 1 ).mul( grain.mul( 0.08 ).add( 0.96 ) ) );
    colour = mix( colour, white, foam );
    return vec4( colour.mul( cover ), cover );

  } )();
  material.colorNode = shaded;
  material.opacityNode = ( shaded as V4 ).w;
  const mesh = new THREE.Mesh( geometry, material );
  mesh.name = spec.name;
  mesh.frustumCulled = false;
  mesh.renderOrder = 12;
  return mesh;

}

// ---- the foam on the pool

/** The pool's foam: over the cells of `area` ( x0, z0, x1, z1 at `step` m ) where the pool is, each
 * point with the current that carries the foam ( vx, vz ) and its density, from `field`. */
export function poolFoam( render: RenderParts, grid: HeightGrid, sea: number, o: {
  area: { x0: number; z0: number; x1: number; z1: number }; step: number;
  wet: ( x: number, z: number ) => boolean;
  field: ( x: number, z: number ) => { vx: number; vz: number; foam: number; boil: number };
} ): THREE.Mesh {

  const { x0, z0, x1, z1 } = o.area;
  const nx = Math.ceil( ( x1 - x0 ) / o.step ), nz = Math.ceil( ( z1 - z0 ) / o.step );
  const position = new Float32Array( ( nx + 1 ) * ( nz + 1 ) * 3 ), flow = new Float32Array( ( nx + 1 ) * ( nz + 1 ) * 4 );
  const wet = new Uint8Array( ( nx + 1 ) * ( nz + 1 ) );
  for ( let j = 0; j <= nz; j ++ ) for ( let i = 0; i <= nx; i ++ ) {

    const k = j * ( nx + 1 ) + i, x = x0 + i * o.step, z = z0 + j * o.step;
    const f = o.field( x, z );
    position.set( [ x, sea + 0.045, z ], k * 3 );
    flow.set( [ f.vx, f.vz, f.foam, f.boil ], k * 4 );
    wet[ k ] = o.wet( x, z ) && f.foam > 0.01 ? 1 : 0;

  }
  const index: number[] = [];
  for ( let j = 0; j < nz; j ++ ) for ( let i = 0; i < nx; i ++ ) {

    const a = j * ( nx + 1 ) + i, b = a + 1, c = a + nx + 1, d = c + 1;
    if ( wet[ a ] + wet[ b ] + wet[ c ] + wet[ d ] === 0 ) continue;
    index.push( a, c, b, b, c, d );

  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute( 'position', new THREE.BufferAttribute( position, 3 ) );
  geometry.setAttribute( 'aFlow', new THREE.BufferAttribute( flow, 4 ) );
  geometry.setIndex( index );
  horizonAttributes( geometry, grid, sea );
  geometry.computeBoundingSphere();

  const inputs = waterInputs( render );
  const flowAt = attribute( 'aFlow', 'vec4' ), sky = attribute( 'aSky', 'float' );
  const material = new THREE.MeshBasicNodeMaterial( { transparent: true, depthWrite: false } );
  material.name = 'pool foam';
  material.forceSinglePass = true;
  material.blending = THREE.CustomBlending;
  material.blendEquation = THREE.AddEquation;
  material.blendSrc = material.blendSrcAlpha = THREE.OneFactor;
  material.blendDst = material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
  const shaded = Fn( () => {

    const p = positionWorld;
    const v = flowAt.xy;
    const pattern = carried( p.xz, v, 2.4, ( q ) => lace( q, 1.2 ) );
    // the trails gather in rafts and lines rather than evenly; the boil at the falls' feet is solid
    const patches = mx_noise_float( vec3( p.xz.mul( 0.22 ), 3.7 ) ).mul( 0.5 ).add( 0.75 );
    const foam = coverage( pattern, max( flowAt.w, flowAt.z.mul( patches ) ).min( 1 ), p ).mul( 0.94 ).toVar();
    If( foam.lessThan( 0.003 ), () => {

      Discard();

    } );
    const V = normalize( cameraPosition.sub( p ) );
    const colour = foamLight( vec3( 0, 1, 0 ), V, inputs.key( p ), sky, float( 0.1 ) );
    return vec4( colour.mul( foam ), foam );

  } )();
  material.colorNode = shaded;
  material.opacityNode = ( shaded as V4 ).w;
  const mesh = new THREE.Mesh( geometry, material );
  mesh.name = 'pool foam';
  mesh.renderOrder = 8;
  return mesh;

}

// ---- the mist, the droplets and the splashes

const MIST = 0, DROP = 1, PUFF = 2;

/** The particles' material: camera-facing sprites, mist and puffs soft and lit as a thin cloud,
 * droplets as small bright beads; `place` gives each one's centre, size, opacity and kind. */
function spriteMaterial( render: RenderParts, name: string, place: () => { centre: V3; size: F; alpha: F; kind: F } ): THREE.MeshBasicNodeMaterial {

  const inputs = waterInputs( render );
  const material = new THREE.MeshBasicNodeMaterial( { transparent: true, depthWrite: false } );
  material.name = name;
  material.forceSinglePass = true;
  material.blending = THREE.CustomBlending;
  material.blendEquation = THREE.AddEquation;
  material.blendSrc = material.blendSrcAlpha = THREE.OneFactor;
  material.blendDst = material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
  const vCentre = varying( vec3( 0 ) ), vInfo = varying( vec4( 0 ) ), vCorner = varying( vec2( 0 ) );
  material.positionNode = Fn( () => {

    const { centre, size, alpha, kind } = place();
    const toCam = normalize( cameraPosition.sub( centre ) );
    const right = normalize( cross( vec3( 0, 1, 0 ), toCam ).add( vec3( 1e-5, 0, 0 ) ) );
    const up = cross( toCam, right );
    const corner = positionGeometry.xy;
    vCentre.assign( centre );
    vInfo.assign( vec4( alpha, kind, size, 0 ) );
    vCorner.assign( corner );
    return select( alpha.greaterThan( 1e-4 ), centre.add( right.mul( corner.x.mul( size ) ) ).add( up.mul( corner.y.mul( size ) ) ), vec3( 0, - 1e5, 0 ) );

  } )();
  const shaded = Fn( () => {

    const alpha = vInfo.x, kind = vInfo.y, size = vInfo.z;
    const r2 = dot( vCorner, vCorner );
    const soft = exp( r2.mul( - 3.2 ) ).sub( 0.04 ).max( 0 ).mul( mx_noise_float( vec3( vCorner.mul( 1.7 ), vCentre.x.add( vCentre.z ) ) ).mul( 0.45 ).add( 0.55 ) );
    const bead = exp( r2.mul( - 5 ) );
    const shape = select( kind.equal( DROP ), bead, soft );
    // soft where it meets the opaque scene
    const fade = clamp( positionView.z.sub( inputs.sceneZ( screenUV ) ).div( size.mul( 0.6 ).add( 0.05 ) ), 0, 1 );
    const a = alpha.mul( shape ).mul( fade ).min( 1 ).toVar();
    If( a.lessThan( 0.002 ), () => {

      Discard();

    } );
    const V = normalize( cameraPosition.sub( vCentre ) );
    const key = inputs.key( vCentre );
    const sun = shared.sunColor.mul( key );
    const cosT = dot( V.negate(), shared.sunDir );
    const cloud = sun.mul( phaseHG( cosT, 0.6 ).mul( 1.1 ).add( 0.7 / Math.PI ) ).add( shared.skyIrradianceSource.mul( 0.9 / Math.PI ) );
    const drop = render.sky.sample( vec3( 0, 1, 0 ), float( 4 ) ).mul( 0.9 ).add( sun.mul( phaseHG( cosT, 0.85 ).mul( 0.8 ).add( 0.04 ) ) );
    const colour = select( kind.equal( DROP ), drop, cloud );
    return vec4( colour.mul( a ), a );

  } )();
  material.colorNode = shaded;
  material.opacityNode = ( shaded as V4 ).w;
  return material;

}

/** The mist rising from a falls' foot and the droplets thrown up there: `count` particles along its
 * plunge line, each looping through its life on the scene's clock. */
export function fallsMist( render: RenderParts, sea: number, spec: FallsSpec, count: number, drops: number ): THREE.Mesh {

  const rng = mulberry32( Math.round( spec.lip * 1000 ) );
  const quad = new THREE.PlaneGeometry( 2, 2 );
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = quad.index;
  geometry.setAttribute( 'position', quad.getAttribute( 'position' ) );
  const total = count + drops;
  const baseAttr = new Float32Array( total * 4 ), dirAttr = new Float32Array( total * 4 ), seedAttr = new Float32Array( total * 4 );
  const end = fallTime( spec ), p = new THREE.Vector3();
  for ( let i = 0; i < total; i ++ ) {

    let x = - spec.half + rng() * 2 * spec.half;
    // more where more water falls
    for ( let k = 0; k < 4 && rng() > spec.flow( x ); k ++ ) x = - spec.half + rng() * 2 * spec.half;
    fallsPoint( spec, x, end, p );
    const slope = ( spec.crest( x + 0.01 ) - spec.crest( x - 0.01 ) ) / 0.02, l = Math.hypot( slope, 1 );
    baseAttr.set( [ p.x, sea + spec.plunge, p.z, i < count ? MIST : DROP ], i * 4 );
    dirAttr.set( [ - slope / l, 1 / l, spec.flow( x ), 0 ], i * 4 );
    seedAttr.set( [ rng(), rng(), rng(), rng() ], i * 4 );

  }
  geometry.setAttribute( 'aBase', new THREE.InstancedBufferAttribute( baseAttr, 4 ) );
  geometry.setAttribute( 'aDir', new THREE.InstancedBufferAttribute( dirAttr, 4 ) );
  geometry.setAttribute( 'aSeed', new THREE.InstancedBufferAttribute( seedAttr, 4 ) );
  geometry.instanceCount = total;
  const base = attribute( 'aBase', 'vec4' ), dir = attribute( 'aDir', 'vec4' ), seed = attribute( 'aSeed', 'vec4' );
  const height = spec.lip - spec.plunge;
  const material = spriteMaterial( render, `${ spec.name } mist`, () => {

    const drop = base.w.equal( DROP );
    const life = select( drop, seed.z.mul( 0.5 ).add( 0.55 ), seed.z.mul( 3 ).add( 3.5 ) );
    const age = fract( shared.time.add( seed.y.mul( 17.3 ) ).div( life ) ).mul( life ).toVar();
    const out = vec3( dir.x, 0, dir.y );
    const side = vec3( dir.y, 0, dir.x.negate() );
    // mist: rises slowly, drifts out from the foot and spreads
    const drift = float( 1 ).sub( exp( age.mul( - 0.7 ) ) ).div( 0.7 );
    const mist = base.xyz.add( out.mul( drift.mul( seed.w.mul( 0.8 ).add( 0.5 ) ) ) )
      .add( side.mul( seed.x.sub( 0.5 ).mul( age ).mul( 0.6 ) ) )
      .add( vec3( 0, age.mul( seed.w.mul( 0.3 ).add( 0.25 ) ).sub( age.mul( age ).mul( 0.015 ) ).add( 0.1 ), 0 ) );
    // droplets: thrown up and out, falling back
    const up = seed.w.mul( 2.2 ).add( 1.6 ).mul( Math.min( 1, Math.sqrt( height / 6 ) + 0.35 ) ), outward = seed.x.mul( 1.6 ).add( 0.3 );
    const dropAt = base.xyz.add( out.mul( outward.mul( age ) ) ).add( side.mul( seed.y.sub( 0.5 ).mul( age ).mul( 1.5 ) ) )
      .add( vec3( 0, up.mul( age ).sub( age.mul( age ).mul( G / 2 ) ), 0 ) );
    const centre = select( drop, dropAt, mist );
    const fadeIn = smoothstep( 0, select( drop, float( 0.02 ), float( 0.8 ) ), age );
    const fadeOut = float( 1 ).sub( smoothstep( life.mul( 0.55 ), life, age ) );
    const above = smoothstep( sea + spec.plunge - 0.05, sea + spec.plunge + 0.15, centre.y );
    const alpha = select( drop, float( 0.45 ).mul( above ), float( 0.032 ).mul( dir.z.mul( 0.6 ).add( 0.4 ) ) ).mul( fadeIn ).mul( fadeOut );
    const size = select( drop, float( 0.024 ), age.mul( 0.22 ).add( 0.35 ).mul( Math.min( 1.3, 0.7 + height / 14 ) ) );
    return { centre, size, alpha, kind: select( drop, float( DROP ), float( MIST ) ) };

  } );
  const mesh = new THREE.Mesh( geometry, material );
  mesh.name = `${ spec.name } mist`;
  mesh.frustumCulled = false;
  mesh.renderOrder = 30;
  return mesh;

}

/** The splashes something breaking the water throws up: droplets and a puff, spawned from the CPU
 * into a ring of particles that fly on the GPU. */
export interface Splashes {
  mesh: THREE.Mesh;
  /** a splash at x, y, z (world) on water at `level`, of `size` (1 for a salmon's) */
  splash( x: number, y: number, z: number, level: number, size?: number ): void;
}

export function splashes( render: RenderParts, capacity = 1536 ): Splashes {

  const rng = mulberry32( 515 );
  const quad = new THREE.PlaneGeometry( 2, 2 );
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = quad.index;
  geometry.setAttribute( 'position', quad.getAttribute( 'position' ) );
  const start = new THREE.InstancedBufferAttribute( new Float32Array( capacity * 4 ).fill( - 1e4 ), 4 );
  const velocity = new THREE.InstancedBufferAttribute( new Float32Array( capacity * 4 ), 4 );
  start.setUsage( THREE.DynamicDrawUsage );
  velocity.setUsage( THREE.DynamicDrawUsage );
  geometry.setAttribute( 'aStart', start );
  geometry.setAttribute( 'aVelocity', velocity );
  geometry.instanceCount = capacity;
  const a = attribute( 'aStart', 'vec4' ), v = attribute( 'aVelocity', 'vec4' );
  // aStart: x, y, z, the time it was thrown; aVelocity: vx, vy, vz, the kind + its water level's
  // offset under its start (fraction: the level is y - 4 x fract)
  const material = spriteMaterial( render, 'splashes', () => {

    const age = shared.time.sub( a.w ).toVar();
    const kind = v.w.floor();
    const below = fract( v.w ).mul( 4 );
    const level = a.y.sub( below );
    const puff = kind.equal( PUFF );
    const centre = a.xyz.add( v.xyz.mul( age ) ).add( vec3( 0, age.mul( age ).mul( select( puff, float( - 1.5 ), float( - G / 2 ) ) ), 0 ) );
    const alive = age.greaterThan( 0 ).and( age.lessThan( select( puff, float( 0.7 ), float( 1.6 ) ) ) ).and( centre.y.greaterThan( level.sub( 0.03 ) ) );
    const alpha = select( alive, select( puff, float( 0.5 ).mul( float( 1 ).sub( smoothstep( 0.1, 0.7, age ) ) ), float( 0.8 ) ), float( 0 ) );
    const size = select( puff, age.mul( 0.9 ).add( 0.12 ), float( 0.028 ) );
    return { centre, size, alpha, kind: select( puff, float( MIST ), float( DROP ) ) };

  } );
  const mesh = new THREE.Mesh( geometry, material );
  mesh.name = 'splashes';
  mesh.frustumCulled = false;
  mesh.renderOrder = 31;
  let next = 0;
  return {
    mesh,
    splash( x, y, z, level, size = 1 ) {

      const now = shared.time.value;
      const drops = Math.round( 26 * size ), puffs = Math.max( 1, Math.round( 3 * size ) );
      const below = Math.min( Math.max( y - level, 0 ), 3.99 ) / 4;
      const sa = start.array as Float32Array, va = velocity.array as Float32Array;
      for ( let k = 0; k < drops + puffs; k ++ ) {

        const i = next ++ % capacity, puff = k >= drops;
        const ang = rng() * Math.PI * 2, out = ( puff ? 0.3 : 0.4 + rng() * 1.4 ) * Math.sqrt( size );
        const up = puff ? 0.4 + rng() * 0.5 : ( 1.4 + rng() * 2.8 ) * Math.sqrt( size );
        sa.set( [ x + Math.cos( ang ) * 0.1, y, z + Math.sin( ang ) * 0.1, now - rng() * 0.03 ], i * 4 );
        va.set( [ Math.cos( ang ) * out, up, Math.sin( ang ) * out, ( puff ? PUFF : DROP ) + Math.min( below + 0.01, 0.999 ) ], i * 4 );

      }
      start.needsUpdate = true;
      velocity.needsUpdate = true;

    },
  };

}
