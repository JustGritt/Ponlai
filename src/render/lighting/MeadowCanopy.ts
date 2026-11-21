import { Fn, clamp, exp, float, length, max, min, mix } from 'three/tsl';
import type * as THREE from 'three/webgpu';
import { shared } from '../shared.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

// The meadow's canopy shadowing: the key light that survives the blades above a point.
//
// Grass blades are narrower than a sun-shadow texel, so the meadow does not cast into the sun's
// shadow map: as the sun turns, re-rasterising sub-texel blades reshuffles their whole shadow
// pattern at every map update. Larger casters (rock, figures) keep the map.
// Shading inside the meadow is instead the light that survives the canopy above a point on its way
// to the key light, the same at every point of the meadow at one depth: a lawn of even blades
// darkens evenly toward the ground. A field of gaps once let light through in patches, which from
// the overview read as fine dark blotches and, under a low sun, as dark streaks along its azimuth;
// the lawn's own root-to-tip colour (Grass.ts) carries the darkening near the ground instead. Of
// the key light, `open` passes as if through the gaps between blades, and the rest falls off with
// the path through the canopy, so a low sun deepens the shadow without turning the meadow olive.
// That canopy is level, as deep over every point, and gives every blade at one depth the same share
// of the key, so none is lit and none shaded: with the meadow's screen-space shadows (post/
// MeadowShadows.ts) the blades take its march in its place and keep this only under the temporal
// modes. What it leaves out is the sward's own relief: the taller tussocks stand over their
// neighbours and shade them on the side away from the light (`meadowCastLight`, once per tuft), and
// the crowns shade the ground under and beside them (`meadowGroundLight`, the turf's).
// Its medium's two numbers are the look's, uniforms the shared object makes from the look's tables
// where it is created (shared.ts `meadowCanopy`: `density`, the extinction per metre of canopy
// along the ray, and `open`, the share of the key light that reaches any depth between the blades).

// Uniforms reach a layout function only as arguments.
const canopyLight = Fn( ( [ depth, key, density, open ]: [ F, V3, F, F ] ) => {

	// Grazing light crosses at most this much canopy; beyond it the key is dark anyway.
	const rise = max( key.y, 0.12 );
	const reach = min( max( depth, 0 ).div( rise ), 2.5 );
	return mix( exp( density.negate().mul( reach ) ), float( 1 ), open );

} ).setLayout( { name: 'ponlaiMeadowCanopy', type: 'float', inputs: [
	{ name: 'depth', type: 'float' }, { name: 'key', type: 'vec3' }, { name: 'density', type: 'float' }, { name: 'open', type: 'float' },
] } );

// Light transmitted to a point `depth` metres below the canopy top. `key` is the unit direction
// toward the light. Uses no textures or derivatives, so it is safe inside branches and in the
// vertex stage.
export function meadowCanopyLight( depth: F, key: V3 ): F {

	return canopyLight( depth, key, shared.meadowCanopy.density, shared.meadowCanopy.open );

}

// A tuft's top stands CANOPY_TOP times its height scale above its root (plantGrass). The canopy's
// depth above the turf is one value for the whole meadow, about the lawn's tuft top: the turf's
// shade then varies with the sun's height alone. A blade takes this depth scaled by its share of
// the way down its own tuft (Grass.ts), so its root meets the turf's shade; a plant that
// stands in the sward takes its depth below its top (the meadow's flowers did; it draws none now).
export const CANOPY_TOP = 0.8;
export const TURF_CANOPY_DEPTH = 0.2;

// The taller tussocks' shade on their neighbours: the key's ray from a tuft's top crosses the
// crowns of the sward's top (SwardField.ts `canopyTop`) that stand over it, and the light falls off
// with the length it runs through them, as through the canopy above. Over a level sward the ray
// rises clear of its neighbours' crowns, so this adds nothing to the canopy's own shade; a tuft
// lower than those toward the light lies in their shade, a long one under a low sun. The ray is
// looked up at CAST_STEPS (m along the ground) up to 3.5 m, where a tussock a metre over its
// neighbour still shades it with the sun 16 degrees up; CAST_SOFT (m) softens a crown's edge.
// Its medium's two numbers are the look's, as the canopy's (shared.ts `meadowCast`: the extinction
// per metre of crowns along the ray, and the share of the key that passes between the crowns'
// blades).
const CAST_STEPS = [ 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5 ];
const CAST_SOFT = 0.05;

/** The share of the key light that reaches `top` (a tuft's top, world) past the crowns of `crowns`
 * (the sward's top, over a square of `extent` (m) centred on the world's origin) toward `key`
 * (unit, toward the light). For a compute kernel: it samples at level 0, and unrolls its lookups. */
export function meadowCastLight( crowns: THREE.TextureNode, extent: number, top: V3, key: V3 ): F {

	const flat = max( length( key.xz ), 1e-3 ).toVar();
	const along = key.xz.div( flat ).toVar(), rise = key.y.div( flat ).toVar();
	let path: F = float( 0 ), last = 0;
	for ( const t of CAST_STEPS ) {

		const q = top.xz.add( along.mul( t ) );
		const crown = crowns.sample( q.div( extent ).add( 0.5 ) ).level( float( 0 ) ).r;
		path = path.add( clamp( crown.sub( top.y.add( rise.mul( t ) ) ).div( CAST_SOFT ), 0, 1 ).mul( t - last ) );
		last = t;

	}
	// (along the ray: its run over the ground over the cosine of the key's elevation)
	return mix( exp( shared.meadowCast.density.negate().mul( path ).div( flat ) ), float( 1 ), shared.meadowCast.open );

}

// The ground under the meadow lit through its crowns: the key's ray from a point of the ground
// runs through the crowns over it and round it (their height over the ground, `over`, the meadow
// map's B: SwardField.ts), the light falling off with the length it runs through them, as the
// tufts' own cast (`meadowCastLight`). Under a tuft the ray starts inside its crown and the ground
// lies in its shade; in a gap the ground is lit unless a taller clump stands toward the light, and
// a clump's shade falls on the ground beside it. Its lookups at GROUND_STEPS (m along the ground);
// the ground beside a clump is level with the clump's root, near enough over the plateau.
const GROUND_STEPS = [ 0.08, 0.16, 0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 2.8 ];

/** The share of the key light that reaches the ground at `at` (world) past the crowns, `over( xz )`
 * their height over the ground there (m), toward `key` (unit). Its lookups take no derivatives. */
export function meadowGroundLight( over: ( xz: THREE.Node<'vec2'> ) => F, at: V3, key: V3 ): F {

	const flat = max( length( key.xz ), 1e-3 ).toVar();
	const along = key.xz.div( flat ).toVar(), rise = key.y.div( flat ).toVar();
	let path: F = float( 0 ), last = 0;
	for ( const t of GROUND_STEPS ) {

		path = path.add( clamp( over( at.xz.add( along.mul( t ) ) ).sub( rise.mul( t ) ).div( CAST_SOFT ), 0, 1 ).mul( t - last ) );
		last = t;

	}
	return mix( exp( shared.meadowCast.density.negate().mul( path ).div( flat ) ), float( 1 ), shared.meadowCast.open );

}
