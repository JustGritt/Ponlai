import * as THREE from 'three/webgpu';
import { Fn, If, abs, clamp, float, floor, int, ivec2, max, min, mix, positionView, screenCoordinate, select, smoothstep, texture, textureLoad, uniform, vec2, vec4 } from 'three/tsl';
import { SCENE_SAMPLES, TEMPORAL } from './Antialiasing.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { tuneUniform } from '../debug/Tune.ts';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type M4 = THREE.UniformNode<'mat4', THREE.Matrix4>;

// The meadow's screen-space shadows: the shadows the blades and tussocks cast on each other.
//
// The meadow's screen-space shadows: the key light's shadows the blades and tussocks cast on each
// other, at the scale the sun's map cannot hold (the blades do not cast into it, MeadowCanopy.ts).
// Screen-space shadows over the depth (Graham Aldridge, Bend Studio, Days Gone, SIGGRAPH 2023) as a
// per-pixel march, for the grass's fine shadows beside the coarse shadow of its heights (Eric
// Wohllaib, "Procedural Grass in 'Ghost of Tsushima'", GDC 2021). The march runs from the
// point a pixel sees, toward the light, over the depth the vegetation's prepass drew, each step
// occluded by how far the depth there lies in front of the ray, relative to the view depth and
// squared, so a surface just in front of it barely counts and one well in front counts whole; the
// steps' mean occlusion goes through a narrow threshold band to the shadow, which is solid at a
// contact and leaves light through the narrow gaps between the blades. A tussock's side turned from the sun and the sward behind it lie in
// its shadow, the tips over it stay lit, and the meadow shows where the light comes from. It stands
// in for the canopy's mean shadowing of the blades (Grass.ts): the mean gave every blade at one depth
// the same share of the key, so no blade was lit and none shaded. The ground under the sward runs
// a march of its own from its own point (`meadowGroundShadow`, TerrainMaterial.ts), as it is not in
// the vegetation's depth: a close look at a sunlit sward shows the blades' shadows on the soil
// between them, which the crowns' cast (MeadowCanopy.ts), a field of a quarter-metre texels, cannot.
// The ground's march is the same response as the blades', run apart only because under MSAA in a
// forward pass the ground is not the depth's nearest sample; beyond some 20 m it fades into the
// pixel's shadow (GROUND_FADE), which changes only where the screen-space shadow comes from: the
// meadow layer's shadowing (Grass.ts `meadowLayerLight`) and the ground's occlusion stay as they are at
// every distance. No Beer-Lambert accumulation along the march, on the ground or the blades.
//
// The frame (Pipeline.ts): the vegetation's depth copies draw first, in a scene and a pass of their
// own (VegetationPrepass.ts `prepassScene`), into the scene's multisampled depth; that depth's
// nearest sample per pixel is resolved (`meadowDepthMaterial`), marched (`meadowShadowMaterial`),
// and the opaque pass then loads the depth, its shaded vegetation drawing to the copies' samples
// as before, and the blades take the shadow into their sun visibility (Grass.ts, `meadowShadowAt`)
// while the ground marches over the resolved depth itself (`meadowDepthMap`; over the whole scene
// the shadow texture's own second and third channels, half-float copies of that view z and its surface's
// thickness, so the ground binds one texture for all: the terrain's fragment stage holds sixteen,
// WebGPU's limit). The opaque scene's depth copies (post/OpaquePrepass.ts) draw before the
// resolve, at half resolution into a target of their own, which the ambient occlusion reads too (GTAO.ts); without `SS_SHADOW` the resolve
// leaves them out and the shadow sees the vegetation's depth alone.
// It is a function of the frame's depth alone: no noise, so a paused frame is bit-identical.
// At 4x outside the temporal modes (the prepass draws the coverage there; the temporal modes keep
// it in the opaque pass); off with `flags=meadowShadows:false` at load.
//
// Over the whole scene (`SS_SHADOW`): the resolve takes the nearer of the vegetation's depth and
// the opaque scene's (post/OpaquePrepass.ts, its half-resolution copies' distance along the view,
// drawn before the resolve), so the march also finds the bamboo's culms, the stones and the
// island's own relief, and the shadow falls at a culm's foot and in a rock's seams, at the
// centimetres the sun's map cannot hold. A surface runs THICK behind its front unless its
// material says otherwise (`screenThickness`, written beside the copies' distance and resolved
// with it): the culms, a few centimetres across, run a grown culm's foot diameter. With THICK
// they were walls: the sunlit ground up to 0.3 m behind a culm took its shadow, which the steps cut
// into a stepped band of teeth beside the culm's foot (bare ground at noon, 1.6 levels darker over
// its box, 60 at most). More steps only made the teeth finer, a per-pixel jitter turned them into
// noise, and a thinner THICK everywhere would also thin a tussock's shadow; the culm's own depth
// leaves the shadow it truly casts, a strip a diameter deep running from its foot away from the
// sun for the march's reach. Thinner culms still count up to that diameter behind them, and the
// vegetation, the bamboo's leaves included, keeps THICK. Every receiver in that depth takes it into
// its key's visibility: the blades as before (Grass.ts), the bamboo and the yunpeng through the
// sun's filter (`receiveScreenShadow`, lighting/SunShadow.ts), the island's ground and stones in
// their own light (materials/TerrainMaterial.ts: the ground's march under the sward near the eye,
// fading into the pixel's shadow beyond it, and the pixel's shadow everywhere else). Not the
// islanders and the game's props, which are not in the depth (skinned, or outside the prepass's
// roots: their pixels would take the shadow of what lies behind them), nor the sea floor, lit by the
// refracted sun. The opaque copies' half resolution puts the surface beside an edge into some of its
// pixels; the receivers take the shadow of the neighbour nearest their own depth there
// (`screenShadowAt`), while the shadow's own outline keeps the copies' two-pixel steps. The
// receivers read it in the opaque pass alone (`screenShadowView.on`): the refraction's draws of the
// same materials read 1. Built with the meadow's screen-space shadows unless
// `flags=ssShadow:false` at load (then nothing here differs from the meadow's screen-space shadows
// alone); at run time
// `flags.ssShadow = false` keeps the opaque depth out of the resolve and the receivers read 1, so
// tools/render/profile-frame.mjs --toggle "s.flags.ssShadow = on" times it.
export const MEADOW_SHADOWS = SCENE_SAMPLES === 4 && ! TEMPORAL && debugFlags.meadowShadows !== false;
export const SS_SHADOW = MEADOW_SHADOWS && debugFlags.ssShadow !== false;

// The march: STEPS lookups over REACH (m) toward the light. Step j occludes by
// c_j = saturate( ( D_j K / z )^2 ) / STEPS, D_j how far the depth there lies in front of the ray
// (m; none behind it, and none past THICK, a tussock's depth: a surface farther in front is no
// wall behind which the ray runs; over the whole scene the surface's own depth where its material
// marks one, `screenThickness`), z the receiver's view depth; the shadow is
// 1 - saturate( ( sum c_j - O0 ) / DELTA ). K = 500 counts a step whole once the depth lies 0.2 % of the view depth in front, twice
// the former bias's share of the distance, so the square's soft start takes over that bias's work
// against a blade shading itself. O0 and DELTA centre a band of 1.6 steps on the mean occlusion
// where the former falloff, exp( -6 m^-1 x the length run behind blades ), halved the light (0.14);
// the steps weigh alike (no data for another profile). A march twice as long and a wall three
// times as thick shaded the game's view by about a quarter more, still in specks: the clumps'
// broad shadow is the crowns' (MeadowCanopy.ts `meadowCastLight`) and the clumps' dome (Grass.ts).
const STEPS = 16, REACH = 0.8, THICK = 0.3, K = 500, O0 = 0.1, DELTA = 0.1;
/** The lookups of a march: `steps` over `reach` m, a surface in front of a lookup by more than
 * `thick` m no wall. */
type MarchTerms = { readonly steps: number; readonly reach: number; readonly thick: number };
const MARCH: MarchTerms = { steps: STEPS, reach: REACH, thick: THICK };
// The shadow's floor on a translucent receiver (`meadowShadowFloor`): F t, t the receiver's
// translucency; an opaque receiver (t 0, the ground) takes its shadow whole. A blade in the shadow of
// the blades between it and the sun still takes the light they let through, t of their diffuse
// (world/Grass.ts TRANSLUCENCY), their t taken as the receiver's own (one sward): F is the blades'
// albedo, 0.08 (their mean over the meadow in our views, `flags=albedoOnly`). That light
// comes along the sun whatever the view; what a sun ahead of the view adds is the receiver's own
// light through (vegetation/Material.ts `transmission`), not its shadow's. Tunable
// (`post.meadowShadows.floor`), a uniform.
const F0 = /*@__PURE__*/ tuneUniform( 'post.meadowShadows.floor', 0.08, { min: 0, max: 0.5 } );

/** The base node the blades load the shadow through, at their pixel: the frame graph's texture
 * (Pipeline.ts sets it); over the whole scene its second and third channels hold the resolved
 * view z and thickness the ground marches over. */
export const meadowShadows = texture( new THREE.DataTexture( new Uint8Array( [ 255, 255, 255, 255 ] ), 1, 1 ) );

/** The shadow at the fragment's pixel (1 lit). */
export function meadowShadowAt(): F {

	return ( textureLoad( meadowShadows, ivec2( screenCoordinate.xy ) ) as unknown as THREE.Node<'vec4'> ).x;

}

/** The resolved view z the ground marches over (Pipeline.ts sets it to the frame graph's), its
 * size in pixels, the camera's projection and the key's shading direction in view space (the
 * lifted key, lighting/KeyLift.ts, as the march's; Pipeline.ts shares its own values with them). */
export const meadowDepthMap = texture( new THREE.DataTexture( new Float32Array( [ - 1e5, 0, 0, 0 ] ), 1, 1, THREE.RGBAFormat, THREE.FloatType ) );
export const meadowGroundView = { size: uniform( new THREE.Vector2( 1, 1 ) ), proj: uniform( new THREE.Matrix4() ), key: uniform( new THREE.Vector3( 0, 1, 0 ) ) };

/** The whole scene's screen-space shadows (`SS_SHADOW`): `on` 1 while the opaque pass draws with it on (else
 * the receivers read 1), `opaque` 1 while the resolve takes the opaque scene's depth (Pipeline.ts
 * sets both every frame from `flags.ssShadow`). */
export const screenShadowView = { on: uniform( 0 ), opaque: uniform( 0 ) };

/** Marks a material whose key light takes the whole scene's screen-space shadows through the sun's filter
 * (lighting/SunShadow.ts): one drawn into the vegetation's or the opaque scene's depth. */
export function receiveScreenShadow( material: THREE.Material ): void {

	if ( SS_SHADOW ) material.userData.screenShadow = true;

}

/** Marks a material's surface as `thickness` m deep behind its front for the march over the whole
 * scene (written beside its distance in the opaque copies, post/OpaquePrepass.ts), where it is
 * thinner than THICK: the bamboo's culms. */
export function screenThickness( material: THREE.Material, thickness: number ): void {

	if ( SS_SHADOW ) material.userData.screenThickness = thickness;

}

/** The whole scene's screen-space shadow at the fragment's pixel for a receiver in the opaque pass
 * (1 lit, and 1 in any other pass or with `flags.ssShadow = false`). The opaque copies are at half
 * resolution, so along an edge a pixel can hold the depth, and the shadow, of the surface beside it
 * (a culm's rim the ground's behind it, the ground beside a culm the culm's): where the view z
 * beside the shadow lies farther than THICK from the fragment's own, the nearest in depth of the
 * pixels two away, in the next 2 x 2 blocks, gives the shadow instead (nearest-depth upsampling). */
export function screenShadowAt(): F {

	const own = positionView.z;
	const coord = screenCoordinate.xy;
	const load = ( dx: number, dy: number ) => textureLoad( meadowShadows, ivec2( dx === 0 && dy === 0 ? coord : max( min( coord.add( vec2( dx, dy ) ), meadowGroundView.size.sub( 1 ) ), vec2( 0 ) ) ) ) as unknown as THREE.Node<'vec4'>;
	const best = load( 0, 0 ).toVar();
	If( abs( best.y.sub( own ) ).greaterThan( THICK ), () => {

		for ( const [ dx, dy ] of [ [ - 2, 0 ], [ 2, 0 ], [ 0, - 2 ], [ 0, 2 ] ] ) {

			const next = load( dx, dy ).toVar();
			If( abs( next.y.sub( own ) ).lessThan( abs( best.y.sub( own ) ) ), () => {

				best.assign( next );

			} );

		}

	} );
	return mix( float( 1 ), best.x, screenShadowView.on );

}

/** The least shadow of a receiver of translucency `t` (F0 above). */
export function meadowShadowFloor( t: F ): F {

	return F0.mul( t );

}

/** The ground's march runs out between GROUND_FADE[ 0 ] and [ 1 ] m from the eye: beyond them a
 * blade spans about a render pixel at the high tier's resolution and its shadow only darkens the ground
 * slightly, while the march over every turf pixel of an island view cost about 6 % of its frame
 * (with the fade, within the noise). */
const GROUND_FADE = [ 10, 20 ] as const;
/** Within GROUND_FADE the ground's own march gives its screen-space shadow: the soil the sun reaches
 * between the blades lies in their light and the soil behind them in their shadow, as a close look at
 * a sunlit sward shows; the meadow layer's shadowing multiplies it as it multiplies the blades' (Grass.ts
 * `meadowLayerLight`, TerrainMaterial.ts). The ground's march takes
 * lookups of its own: `steps` over `reach` m (past the blades' tops at a high sun), a blade a sheet
 * no more than `thick` m deep, through the same occlusion and band as the blades'. With the blades'
 * sixteen lookups over 0.8 m and their 0.3 m walls each blade cast a comb of dashes on the soil,
 * its lookups stepping over a centimetre-wide leaf, and a broad smear behind it; the thin sheet
 * casts the leaf's own shape. Measured with an exponential falloff of the length run behind the
 * blades in place of the band: in the close views the darkest tenth of the sunlit lawn went from
 * half its median to a third, the soil's p90 over p10 from 3.5 to 7, and the sunlit soil came to
 * the sunlit blades' value (tools/island/meadow-light.mjs `spread`, `soil`). The fine lookups cost on the
 * turf within GROUND_FADE only: little of a game view's. */
const GROUND: MarchTerms = { steps: 64, reach: 0.6, thick: 0.06 };

/** How much of the ground's screen-space shadow at view z `z` its own march gives (1 within
 * GROUND_FADE[ 0 ] m of the eye, 0 beyond [ 1 ], where the pixel's shadow gives it). */
function meadowGroundNear( z: F ): F {

	return float( 1 ).sub( smoothstep( GROUND_FADE[ 0 ], GROUND_FADE[ 1 ], z.negate() ) );

}

/** The blades' shadow on the ground at the view-space point `p` (1 lit): the same march from the
 * ground's own point, as the ground under the blades is not the depth's nearest, on its own terms
 * (GROUND); beyond GROUND_FADE none, or over the whole scene (`SS_SHADOW`) the pixel's shadow. */
export function meadowGroundShadow( p: V3 ): F {

	const far = SS_SHADOW ? screenShadowAt() : float( 1 );
	const shade = far.toVar();
	If( p.z.greaterThan( - GROUND_FADE[ 1 ] ), () => {

		const lit = SS_SHADOW
			? march( meadowShadows, p, p.z, meadowGroundView.size, meadowGroundView.proj, meadowGroundView.key, 'y', 'z', GROUND )
			: march( meadowDepthMap, p, p.z, meadowGroundView.size, meadowGroundView.proj, meadowGroundView.key, 'x', null, GROUND );
		shade.assign( mix( far, lit, meadowGroundNear( p.z ) ) );

	} );
	return shade;

}

/** The resolve: the nearest sample's view z of the multisampled depth `depth` (-1e5 where nothing
 * was drawn); over the whole scene the nearer of it and the opaque copies' distance along the view
 * `opaque` (0 where none drew), a texel of `halfSize` per 2 x 2 pixels, while
 * `screenShadowView.opaque` holds. */
export function meadowDepthMaterial( depth: THREE.Texture, size: V2, projInv: M4, opaque?: { map: THREE.Texture; halfSize: V2 } ): THREE.NodeMaterial {

	const ms = texture( depth );
	const copies = opaque ? texture( opaque.map ) : null;
	const m = new THREE.NodeMaterial();
	m.fragmentNode = Fn( () => {

		const coord = screenCoordinate.xy;
		const at = ( i: number ) => textureLoad( ms, ivec2( coord ), int( i ) ) as unknown as F;
		const d = min( min( at( 0 ), at( 1 ) ), min( at( 2 ), at( 3 ) ) ).toVar();
		const z = float( - 1e5 ).toVar();
		const thick = float( THICK ).toVar();
		If( d.lessThan( 0.999999 ), () => {

			const q = coord.div( size );
			const v = projInv.mul( vec4( q.x.mul( 2 ).sub( 1 ), float( 1 ).sub( q.y ).mul( 2 ).sub( 1 ), d, 1 ) ).toVar();
			z.assign( v.z.div( v.w ) );

		} );
		if ( copies && opaque ) {

			const texel = ivec2( min( floor( coord.mul( 0.5 ) ), opaque.halfSize.sub( 1 ) ) );
			const copy = ( textureLoad( copies, texel ) as unknown as THREE.Node<'vec4'> ).toVar();
			If( screenShadowView.opaque.greaterThan( 0 ).and( copy.x.greaterThan( 0 ) ).and( copy.x.negate().greaterThan( z ) ), () => {

				z.assign( copy.x.negate() );
				thick.assign( select( copy.y.greaterThan( 0 ), copy.y, float( THICK ) ) );

			} );

		}
		return vec4( z, thick, 0, 0 );

	} )();
	m.name = 'meadowDepth';
	return m;

}

/** The light left after the march from `p` (view space, its view z `z`) toward `key` (view space,
 * unit) over the resolved view z in channel `channel` of `zMap` of `size` pixels, projected by
 * `proj` (1 lit); each surface's thickness in channel `thickness` (THICK without one), held to
 * `terms.thick`; `terms.steps` lookups over `terms.reach` m. */
function march( zMap: THREE.TextureNode, p: V3, z: F, size: V2, proj: THREE.Node<'mat4'>, key: V3, channel: 'x' | 'y' = 'x', thickness: 'y' | 'z' | null = null, terms: MarchTerms = MARCH ): F {

	const { steps, reach, thick } = terms;
	const scale = float( K ).div( max( z.negate(), 1e-3 ) ).toVar();
	const occlusion = float( 0 ).toVar();
	for ( let i = 1; i <= steps; i ++ ) {

		const at = p.add( key.mul( reach * i / steps ) ).toVar();
		const clip = proj.mul( vec4( at, 1 ) ).toVar();
		const uv = clip.xy.div( clip.w ).mul( vec2( 0.5, - 0.5 ) ).add( 0.5 );
		const pixel = uv.mul( size ).toVar();
		const inside = pixel.x.greaterThanEqual( 0 ).and( pixel.y.greaterThanEqual( 0 ) ).and( pixel.x.lessThan( size.x ) ).and( pixel.y.lessThan( size.y ) );
		const sample = ( textureLoad( zMap, ivec2( max( min( pixel, size.sub( 1 ) ), vec2( 0 ) ) ) ) as unknown as THREE.Node<'vec4'> ).toVar();
		const front = sample[ channel ].sub( at.z );
		const wall = thickness ? ( terms === MARCH ? sample[ thickness ] : min( sample[ thickness ], thick ) ) : float( thick );
		const blocked = select( inside.and( front.greaterThan( 0 ) ).and( front.lessThan( wall ) ), front, float( 0 ) ).mul( scale );
		occlusion.addAssign( clamp( blocked.mul( blocked ), 0, 1 ).div( steps ) );

	}
	return float( 1 ).sub( clamp( occlusion.sub( O0 ).div( DELTA ), 0, 1 ) );

}

/** The march over the resolved view z `depth` toward `key` (view space, unit); over the whole
 * scene the view z is written beside the shade, held to the half float's range. */
export function meadowShadowMaterial( depth: THREE.Texture, size: V2, proj: M4, projInv: M4, key: V3 ): THREE.NodeMaterial {

	const zMap = texture( depth );
	const m = new THREE.NodeMaterial();
	m.fragmentNode = Fn( () => {

		const coord = screenCoordinate.xy;
		const resolved = ( textureLoad( zMap, ivec2( coord ) ) as unknown as THREE.Node<'vec4'> ).toVar();
		const z = resolved.x;
		const shade = float( 1 ).toVar();
		If( z.greaterThan( - 1e4 ), () => {

			// the point the pixel sees, from its ray through the far plane
			const q = coord.div( size );
			const ray = projInv.mul( vec4( q.x.mul( 2 ).sub( 1 ), float( 1 ).sub( q.y ).mul( 2 ).sub( 1 ), 1, 1 ) ).toVar();
			const p = ray.xyz.div( ray.w ).mul( z.div( ray.z.div( ray.w ) ) ).toVar();
			shade.assign( SS_SHADOW ? march( zMap, p, z, size, proj, key, 'x', 'y' ) : march( zMap, p, z, size, proj, key ) );

		} );
		return SS_SHADOW ? vec4( shade, max( z, - 6e4 ), resolved.y, 1 ) : vec4( shade, 0, 0, 1 );

	} )();
	m.name = 'meadowShadows';
	return m;

}
