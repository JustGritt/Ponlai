import * as THREE from 'three/webgpu';
import { Fn, uniform, uniformArray, texture, vec4, int, float, clamp, floor, dot, max, select, mix, positionLocal, positionView, normalWorld, modelWorldMatrixInverse } from 'three/tsl';
import { compileByRendering, compileLatentInto } from '../post/graph/Compile.ts';
import { solarSteppedShadowVisibility, solarVolumeVisibility, solarPCFVisibility, solarDiscVisibility } from './SolarShadow.ts';
import { screenShadowAt } from '../post/MeadowShadows.ts';
import type { CasterColumns } from './CasterFormat.ts';
import { shadowWindOn } from '../vegetation/VegetationShadow.ts';
import { leafThrough } from '../vegetation/LeafLight.ts';
import { debugFlags } from '../debug/DebugFlags.ts';
import { tuneUniform } from '../debug/Tune.ts';
import { builtShadowMap, redrawShadow, shadowPassMaterial, shadowRenderer, type ShadowRenderer } from '../three/ShadowPass.ts';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type FilterInput = { shadowCoord: V3 };

// The key light's shadow map, redrawn in steps of the light's direction and cross-faded.
//
// The map is drawn along the direction `update` is given, not the light's own: main.ts passes the
// key's lifted direction (lighting/KeyLift.ts `liftedKeyDirection`), so under a low key the shadows
// fall along the direction the surfaces' whole direct term takes (the moon's fixed direction by
// night, which is then the key's own too), while the light keeps its true direction for what does
// not lift (the water's glint, the refracted sun). The light stands along the given direction only while three's pass draws the map
// (`render`), which builds the map's view from the light.
//
// Stems, leaf stalks and leaves seen edge-on are thinner than a shadow texel. Whether such a
// caster covers a texel centre flips as the projection turns, so a map redrawn every frame
// made their shadows boil while the clock ran, even though the light moves a shadow edge by
// less than a tenth of a pixel per frame. Instead the map is redrawn only after the key light
// has turned by STEP, the previous map is kept, and every receiver blends the two by how far
// the light has turned since the last redraw. A texel that changes between two redraws then
// fades over a step instead of popping, and the blended shadow still moves continuously: it
// trails the light by about one step, 3.5 mm per metre of throw, a third of the sun's own
// penumbra there. Each redraw re-rolls every sub-texel caster (the projection's turn moves a
// point 10 m from its centre by more than a texel), so a longer step also means fewer of those
// fades. The blend is a function of the key direction alone, so a paused clock holds it exactly.
//
// The map is fitted to the island's casters; the sea floor's stones cast only into the refracted
// map (RefractedSun.ts). Grass blades, stems, leaf stalks and twigs stay out of it (VegetationShadow.ts);
// the meadow's layer shades the blades and the ground under and beside it where they read the map
// (Grass.ts `meadowLayerLight`, through a material's `userData.sunOcclusion`). Both stepped
// maps are filtered alike, each on its own, so the cross-fade stays continuous across a redraw.
// A receiver's screen-space shadows (post/MeadowShadows.ts) multiplies the filter's visibility: its own
// (`userData.sunOcclusion`, the meadow's blades) or the whole scene's at its pixel (`receiveScreenShadow`).
// A tree's leaves, marked `userData.keyThrough`, take the two's product on a floor (`leafThrough`,
// vegetation/LeafLight.ts): the map draws every leaf as black, where a crown's leaves pass on
// a share of the sun to the leaves under them. The floor is a uniform stand-in for that light, on the
// receivers alone; a caster that let part of the sun through would change the shadow on the grove's
// ground too and cost a map of its own. The clouds' shadow multiplies it as it does the map.
// Receivers read the maps through SolarShadow's disk filter (a fixed radius of POISSON_RADIUS
// texels, so its penumbra does not widen with the caster's distance; see its rules on sampler types
// and derivatives). The caster-depth grid (`nearestCasterDepth`) lets a march through the water
// skip the 2x2 volume lookups where nothing casts, never the surface filter's disk. It projects
// world-space columns of the casters, walked from their vertices, or for meshes made in the world's
// workers folded in from the columns they came with (`setCasters`, ChunkDerived.ts), so the main
// thread does not walk those.
//
// Closed solids (the island's terrain, `castFrontFaces`) cast their front faces into this map,
// not the back faces three casts for a FrontSide material. At the cliff's creases, scar-notch
// corners and sharp arrises a wall turned away from the sun has, as its only back-face occluder,
// the crease's other side a few millimetres off, inside the bias: such pixels read lit and showed
// as bright specks in the shadow. Back faces also read every point inside the rock as lit, and
// submerged receivers read this map at their refracted entry point on the surface (RefractedSun),
// which for the sea floor and walls along the cliff's foot lies inside the rock: thin lines and
// patches of sun showed there inside the island's shadow. The front faces put the occluder at the
// solid's sunlit skin, a body's thickness away. Lit ground then receives its own depth: the
// receiver-plane filter and the normal bias keep the turf itself clean, but grass blades and stems
// a few centimetres over it caught the turf in their filter's taps and darkened. So the solids'
// cast depth is pushed FRONT_PUSH away from the light (the map's own bias stays small, for the
// vegetation's contact shadows); rock thinner than that along the light leaks again, so it stays
// a few centimetres. Only this map changes: three's `shadowSide` is set for the draw and restored,
// and the refracted map draws its casters with a material of its own.
//
// A receiver of the filter looks its shadow up a normal offset away (`offsetCoordinate`), in the
// current map's texels and growing with view depth (NORMAL_OFFSET), the normal turned toward the
// light on vegetation; three's own normal bias is 0. The sea reads the map apart from this path, in
// its vertex stage over a disc metres wide (`discVisibility`, Water.ts), with a fixed offset of its own.
//
// The map's first redraw builds every caster's pass into it, and three builds the map itself only
// with the first receiver, the throwaway one built just before the loop (three/ThreePatches.ts
// `primeSharedNodes`), so no warm-up before the loop could compile those passes into it. Drawn as
// usual, that redraw (the loop's first frame, before the receivers build) created their pipelines
// synchronously and stalled. It is made first with its pipelines compiled asynchronously
// (graph/Compile.ts `compileByRendering`), drawing nothing, and drawn for real once they exist; until
// then the receivers stay lit (`ready`) and `pending` holds the settled signal. The compile is the
// redraw itself, its node builds made in its order, as three would make them: precompiled in
// another order, the receivers' and casters' shaders came out with other temporaries and moved a
// few pixels by one level (three r186 emits from shared node state).
//
// The plants cast in their swayed pose (VegetationShadow.ts), so the map is redrawn every frame, as
// a cut is, with no previous map to cross-fade: the sub-texel casters boil again, the price of
// shadows that move with the wind. `flags.shadowWind = false` (read every frame) casts the rest
// pose and brings back the stepped redraws above. Those redraws leave `draws` alone, as the map changes only with the
// clock then: a paused page holds the wind, so every redraw draws the same map and it still settles.
// What follows the light's steps rather than every map (the ground bounce's bake, GroundBounce.ts)
// keys on `steps` instead: every counted redraw, and under the swaying redraws each turn of `step`
// since the last, so a running clock still steps it and a paused page (no turn) still settles.
// A change of the flag redraws from scratch on the next update.
const STEP = THREE.MathUtils.degToRad( 0.2 );
// how far closed solids' cast depth is pushed away from the light (m), see above
const FRONT_PUSH = 0.03;
// A larger turn in one frame is a cut (a new hour, preset or manual sun): redraw without history.
const CUT = THREE.MathUtils.degToRad( 1 );
const PREVIOUS = 0, CURRENT = 1;
// The receivers' normal offset (`offsetCoordinate`), in the current map's texels: `near` at the
// eye, `far` from `depth` metres of view depth on (`depth` the island camera's middle distance).
// It is what keeps a lit plane out of its own shadow under the filter's hardware comparisons
// (SolarShadow.ts): a bilinear comparison tests its footprint's four texels at one depth, and a
// texel up to one texel away along each axis sits higher on a plane lit at elevation e by up to
// cot e * ( |cos a| + |sin a| ) texels (a the plane's direction across the map), while an offset of
// d texels along the normal clears d / sin e. Both meet at d = cos e * ( |cos a| + |sin a| ), at
// most cos e * 1.41: 1.4 texels, about 3 cm in a clear noon's map (45 m over 2048 texels). Measured
// on the ramp probe (verifySolarShadowRamp, sun 32 and 10 degrees up): a clear plane kept 1 at every
// phase with 1.4, but fell to 0.93 at 32 degrees and 0.80 at 10 with the former 0.55 (1.2 cm).
// A caster nearer its receiver than this clears is not resolved by the map: the screen-space
// shadow draws those contacts. `far` keeps the same floor, since the footprint's rise is the same
// in texels at any distance. Vegetation turns the offset toward the light by
// clamp( VEGETATION_FLIP * N.L, -1, 1 ): a leaf lit from behind looks up its shadow from its
// sunlit side, not from inside its own depth, which left lit specks on backlit foliage (tunable,
// `lighting.sunShadow.vegetationFlip`, a uniform). By the bound above, a face at incidence cosine c
// needs at least sqrt( 1 - c^2 ) of the offset, which clamp( k c ) gives for c >= 1 / sqrt( 1 + k^2 ):
// a band of grazing leaves about 1 / k wide still looks its shadow up too near itself. Measured on
// paused captures of a bamboo crown, lit from in front and from behind, under a clear noon and a
// clear golden hour, against the sign's limit (k = 10^4): the share of foliage pixels more than 8
// levels off falls about as 1 / k (golden hour from behind, the worst: 0.585 % at 4, 0.046 % at 24,
// 0.018 % at 48; the noon views settle by 3). At 48 the isolated specks (at most 4 pixels, more than
// 8 levels off) come to about 1 in 10^4 foliage pixels at worst: 1.07 from behind in the golden
// hour, against 24 at 4, with 0.82 at 64 gaining little; 0.12 from in front; none at noon. It pushes
// no lit leaf into shadow: from 4 to 48 the mean of the brightest tenth of the backlit crown's
// foliage moves under 0.05 % (the specks it loses), the front-lit crowns' under 0.01 %.
export const NORMAL_OFFSET = { near: 1.4, far: 1.4, depth: 100 };
const VEGETATION_FLIP = /*@__PURE__*/ tuneUniform( 'lighting.sunShadow.vegetationFlip', 48, { min: 1, max: 128, scale: 'log' } );
// The caster-depth grid (nearestCasterDepth): light-space cells per side of each map, world cells
// along the casters' longer side, and the padding for what the rest pose leaves out (wind sway,
// the rim's lean): metres up, and world cells sideways.
const GRID = 16, WORLD_CELLS = 64, GRID_HEIGHT_PAD = 0.35, GRID_SIDE_PAD = 1;
// a cell without casters: farther than any receiver's depth
const NO_CASTER = 2;
// a cell not projected yet: nearer than any receiver, so nothing is skipped on its word
const UNKNOWN = 0;

// three's shadow coordinate is ( s.x, 1 - s.y, s.z + bias ) for s = shadow.matrix * world
const FLIP_Y = new THREE.Matrix4().set(
	1, 0, 0, 0,
	0, - 1, 0, 1,
	0, 0, 1, 0,
	0, 0, 0, 1,
);

function shadowCoordinateMatrix( shadow: THREE.LightShadow, target: THREE.Matrix4 ): THREE.Matrix4 {

	return target.makeTranslation( 0, 0, shadow.bias ).multiply( FLIP_Y ).multiply( shadow.matrix );

}

export interface SunShadowOptions {
	// Bias along the light ray, in metres; converted to each fitted map's depth range.
	bias?: number;
	// Lowest surface that reads this map: the sea's wave troughs. Submerged receivers read it at
	// their refracted entry point on the surface (see RefractedSun), never below it.
	floor?: number;
	// light-space margin around the casters (m): the filter's footprint and the normal bias
	margin?: number;
}

export class SunShadow {

	// share of the current map; 1 while there is no previous one
	readonly weight = uniform( 1 );
	// the current map's shadow coordinate -> the previous map's
	readonly toPrevious = uniform( new THREE.Matrix4() );
	readonly mapSize = uniform( new THREE.Vector2( 1, 1 ) );
	readonly span = uniform( new THREE.Vector2( 1, 1 ) );
	// false until the first map has been drawn: receivers are lit rather than reading an empty layer
	readonly ready = uniform( 0 );
	// world -> the current map's shadow coordinate (three's, bias included), for lookups made away
	// from three's receiver path (GroundBounce's bake, the haze)
	readonly coordinate = uniform( new THREE.Matrix4() );
	// the direction the map is drawn along (world, toward the light), for the normal offset
	readonly direction = uniform( new THREE.Vector3( 0, 1, 0 ) );
	private readonly sun: THREE.DirectionalLight;
	// Both maps share one array texture (previous, current), so a receiver binds one texture.
	// three's own map is only the raster each step is drawn into. A render target owns the array,
	// though nothing draws into it: a depth texture without one is taken for a multisampled
	// attachment of the main pass.
	private readonly maps: THREE.RenderTarget;
	private readonly depth: THREE.TextureNode;
	private readonly casters = new THREE.Box3();
	// materials of closed solids, cast by their front faces (see the header)
	private readonly frontCasters = new Set<THREE.Material>();
	// their cast depth, pushed FRONT_PUSH away from the light (world space, set at each draw)
	private readonly frontPush = uniform( new THREE.Vector3() );
	private readonly drawn = new THREE.Vector3();
	private readonly bias: number;
	private readonly floor: number;
	private readonly margin: number;
	// the light's turn between redraws (radians)
	step = STEP;
	private hasPrevious = false;
	private cut = true;
	// Whether the casters' passes into the map have their pipelines (see the header).
	private casterPipelines: 'none' | 'compiling' | 'ready' = 'none';
	/** Maps drawn since the page loaded (the settled signal's signature, post/Settled.ts; the dynamic
	 * resolution leaves frames with periodic work out of its quantiles). */
	draws = 0;
	/** The light's steps since the page loaded: the counted redraws, and under the swaying casters'
	 * per-frame redraws each turn of `step` since the last (see the header). */
	steps = 0;
	private readonly stepFrom = new THREE.Vector3();
	private readonly previousCoordinate = new THREE.Matrix4();
	private readonly currentCoordinate = new THREE.Matrix4();
	private readonly corner = new THREE.Vector3();
	private readonly target = new THREE.Vector3();
	private readonly lightPosition = new THREE.Vector3();
	private readonly lightBounds = new THREE.Box3();
	private readonly currentLayer = new THREE.Box3( new THREE.Vector3( 0, 0, CURRENT ), new THREE.Vector3( 1, 1, CURRENT + 1 ) );
	private readonly toCurrentLayer = new THREE.Vector3( 0, 0, CURRENT );
	private readonly toPreviousLayer = new THREE.Vector3( 0, 0, PREVIOUS );
	// Per map (previous, current), the nearest caster's depth in each GRID x GRID light-space cell,
	// four cells per vec4. Built only once a receiver asks for it (nearestCasterDepth); until then
	// every cell reads UNKNOWN, which never lets a march skip a lookup.
	private readonly casterDepths = uniformArray( Array.from( { length: 2 * GRID * GRID / 4 }, () => new THREE.Vector4().setScalar( UNKNOWN ) ), 'vec4' );
	private gridWanted = false;
	// A receiver asked for the grid since the last projection: project both halves on the next
	// update, whether or not the map is redrawn then (a paused clock may never redraw it).
	private gridDirty = false;
	// world-space caster columns: per WORLD_CELLS cell, the lowest and highest caster point
	private columns: { x0: number; z0: number; cell: number; nx: number; nz: number; low: Float32Array; high: Float32Array } | null = null;
	// meshes whose columns came with them (setCasters): folded in rather than walked
	private given: ReadonlyMap<THREE.Object3D, CasterColumns> = new Map();

	constructor( sun: THREE.DirectionalLight, { bias = 0.01, floor = - 3, margin = 0.6 }: SunShadowOptions = {} ) {

		this.sun = sun;
		this.bias = bias;
		this.floor = floor;
		this.margin = margin;
		// the format three gives the sun's own map (ShadowNode.setupRenderTarget, PCF filtering)
		const { width, height } = sun.shadow.mapSize;
		const depth = new THREE.DepthTexture( width, height, THREE.UnsignedIntType, undefined, undefined, undefined,
			THREE.LinearFilter, THREE.LinearFilter, undefined, THREE.DepthFormat, 2 );
		depth.compareFunction = THREE.LessEqualCompare;
		depth.name = 'sunShadowSteps';
		this.maps = new THREE.RenderTarget( width, height, { depth: 2, depthTexture: depth, depthBuffer: true } );
		this.depth = texture( depth );
		sun.shadow.autoUpdate = false;

	}

	/** Install the blended filter on the sun's shadow; receivers built afterwards use it. */
	install(): void {

		const filter = Fn( ( { shadowCoord: unbiased }: FilterInput, builder: THREE.NodeBuilder ) => {

			const material = builder.material as THREE.Material | null;
			// A receiver that brings its own visibility takes it in place of the filter.
			const own = material?.userData.sunVisibility as F | undefined;
			if ( own ) return own;
			const shadowCoord = this.offsetCoordinate( unbiased, material?.userData.vegetation === true );
			// Vegetation is told by its tag, not by alphaToCoverage, which is off without multisampling.
			const visibility = this.surfaceVisibility( shadowCoord );
			// A receiver's own shadow on top of the map's (the meadow's blades: their screen-space
			// shadows, post/MeadowShadows.ts; they and the ground under and beside them: the meadow's
			// layer, Grass.ts `meadowLayerLight`), or the whole scene's screen-space shadow at its
			// pixel for one marked for it (`receiveScreenShadow`: the bamboo, the yunpeng)
			const shade = ( material?.userData.sunOcclusion as F | undefined ) ?? ( material?.userData.screenShadow === true ? screenShadowAt() : undefined );
			const seen = shade ? visibility.mul( shade ) : visibility;
			// the light a shaded leaf takes through the leaves over it (`userData.keyThrough`)
			return material?.userData.keyThrough === true ? seen.add( leafThrough.mul( float( 1 ).sub( seen ) ) ) : seen;

		} );
		// The runtime exposes filterNode; @types/three has not declared this WebGPU hook yet.
		( this.sun.shadow as unknown as { filterNode: typeof filter } ).filterNode = filter;

	}

	/** A receiver's coordinate in the current map moved along its normal by the normal offset
	 * (NORMAL_OFFSET, in texels, growing with view depth), the normal turned toward the light for
	 * vegetation. three's own normal bias stays 0 (main.ts): this one replaces it. */
	offsetCoordinate( coordinate: V3, vegetation: boolean ): V3 {

		const texel = max( this.span.x.div( this.mapSize.x ), this.span.y.div( this.mapSize.y ) );
		const depth = clamp( positionView.z.negate().div( NORMAL_OFFSET.depth ), 0, 1 );
		const metres = mix( float( NORMAL_OFFSET.near ), float( NORMAL_OFFSET.far ), depth ).mul( texel );
		const along = vegetation ? normalWorld.mul( clamp( dot( normalWorld, this.direction ).mul( VEGETATION_FLIP ), - 1, 1 ) ) : normalWorld;
		return coordinate.add( this.coordinate.mul( vec4( along.mul( metres ), 0 ) ).xyz ).toVar();

	}

	/** The previous map's coordinate for a coordinate in the current map (both orthographic). */
	previousCoordinateNode( coordinate: V3 ): V3 {

		return this.toPrevious.mul( vec4( coordinate, 1 ) ).xyz;

	}

	/** A surface's visibility from its coordinate in the current map (three's shadow coordinate).
	 * Both maps keep the same filter, so the cross-fade between them stays continuous. */
	surfaceVisibility( coordinate: V3 ): F {

		return solarSteppedShadowVisibility( this.depth,
			{ coordinate: this.previousCoordinateNode( coordinate ), active: this.weight.lessThan( 1 ), layer: PREVIOUS },
			{ coordinate, active: this.ready.greaterThan( 0 ), layer: CURRENT },
			this.weight, { mapSize: this.mapSize } );

	}

	/** The same for a point in a volume (no receiver plane, no derivatives). */
	volumeVisibility( coordinate: V3 ): F {

		return mix(
			solarVolumeVisibility( this.depth, this.previousCoordinateNode( coordinate ), this.mapSize, this.weight.lessThan( 1 ), PREVIOUS ),
			solarVolumeVisibility( this.depth, coordinate, this.mapSize, this.ready.greaterThan( 0 ), CURRENT ),
			this.weight );

	}

	/** A soft visibility over a disc of a level surface, in the vertex stage (the sea's,
	 * solarDiscVisibility): `coordinate` is the disc's centre in the current map, `across` and `along`
	 * its radii there. It reads the current map alone: the swaying redraw leaves no previous map, so
	 * `weight` is 1 on every frame and that layer would be gathers for nothing. Under
	 * `flags.shadowWind = false`, a debug flag, the sea's shadow steps with the redraws. */
	discVisibility( coordinate: V3, across: V3, along: V3 ): F {

		return solarDiscVisibility( this.depth, coordinate, across, along, this.ready, CURRENT );

	}

	/** Three's five-tap PCF (solarPCFVisibility) from the current map alone, for a bake drawn at the
	 * map's steps (GroundBounce.ts). */
	currentPCFVisibility( coordinate: V3 ): F {

		return solarPCFVisibility( this.depth, coordinate, this.mapSize, this.ready.greaterThan( 0 ), CURRENT );

	}

	/** The nearest caster's depth (in that map's shadow coordinate z, bias included) over the
	 * light-space cell holding a coordinate of the current map, or of the previous one with
	 * `previous`; 2 where nothing casts, 0 before the grid's first projection. A point of that
	 * map no deeper than it is lit by the 2x2 bilinear lookups (volumeVisibility, the water
	 * column), whose footprint the cells already include, so a march may skip those lookups. It
	 * does not bound the surface filter's disk, which reaches SolarShadow's
	 * POISSON_RADIUS texels further and compares along
	 * the receiver's plane: never skip surfaceVisibility on its word. The grid comes from the casters' geometry at rest,
	 * padded for wind and the rim's lean, and is re-projected at each step. */
	nearestCasterDepth( coordinate: V3, previous = false ): F {

		if ( ! this.gridWanted ) this.gridWanted = this.gridDirty = true;
		const cell = clamp( floor( coordinate.xy.mul( GRID ) ), 0, GRID - 1 );
		const index = int( cell.y.mul( GRID ).add( cell.x ) ).add( previous ? 0 : GRID * GRID );
		const packed = this.casterDepths.element( index.div( 4 ) ) as unknown as THREE.Node<'vec4'>;
		const lane = index.mod( 4 );
		const pick = ( k: number ) => select( lane.equal( k ), float( 1 ), float( 0 ) );
		return dot( packed, vec4( pick( 0 ), pick( 1 ), pick( 2 ), pick( 3 ) ) );

	}

	/** World-space bounds of everything that casts into this map (static). Only their part
	 * above the floor can shade a receiver of this map. `columns` gives meshes' caster columns
	 * made off the main thread (the chunks', ChunkDerived.ts: world space, for meshes drawn with an
	 * identity world matrix); the caster-depth grid folds them in instead of walking those meshes'
	 * vertices, and walks every other caster as before. Without it the columns given last stay
	 * (a prop that grows the box keeps them). */
	setCasters( box: THREE.Box3, columns?: ReadonlyMap<THREE.Object3D, CasterColumns> ): void {

		if ( columns ) this.given = columns;
		this.casters.copy( box );
		this.casters.min.y = Math.max( this.casters.min.y, this.floor );
		this.columns = null;
		this.cut = true;

	}

	/**
	 * Compile the pipelines `casters` draw with into this map, off the frame (graph/Compile.ts),
	 * so that a caster first added mid-game (a game's prop) builds none in the frame that redraws
	 * the map. Null until three has built the map and its node (with the first receiver compiled
	 * or drawn). The casters need not be in the scene.
	 *
	 * three draws a caster into the map with the shadow node's material in its own's place
	 * (scene.overrideMaterial), taking the caster's colour, depth and position nodes and side into
	 * it per draw and restoring them after (Renderer.renderObject, r186). Each caster here compiles
	 * with a pass material of its own that already holds what the draw takes into it, so the build
	 * is the draw's whether it runs before that restore (the bound compile) or after it. This
	 * mirrors three's shadow material and its non-VSM side rule (three/ShadowPass.ts).
	 */
	compileCasters( renderer: THREE.WebGPURenderer, casters: THREE.Object3D[] ): Promise<void> | null {

		const node = shadowRenderer( this.sun.shadow );
		const map = node ? builtShadowMap( node ) : null;
		if ( ! node || ! map ) return null;
		const meshes: THREE.Mesh[] = [];
		for ( const root of casters ) root.traverse( ( object ) => {

			if ( ( object as THREE.Mesh ).isMesh && object.castShadow ) meshes.push( object as THREE.Mesh );

		} );
		// one caster at a time, as a frame draws them (see main.ts on the game's props)
		const compile = ( mesh: THREE.Mesh ): Promise<void> => {

			// three's shadow material (ShadowBaseNode), as the draw leaves it for this caster
			const pass = shadowPassMaterial( renderer, mesh.material as THREE.NodeMaterial );
			const scene = new THREE.Scene();
			scene.overrideMaterial = pass;
			return compileLatentInto( renderer, [ mesh ], scene, this.sun.shadow.camera, map );

		};
		return meshes.reduce( ( done, mesh ) => done.then( () => compile( mesh ) ), Promise.resolve() );

	}

	/** Cast the front faces of every shadow-casting mesh under `root` into this map: for closed
	 * solids only (see the header), never for open sheets or cards. Their materials' cast
	 * position becomes the push away from this light, so no other light's shadow may draw them. */
	castFrontFaces( root: THREE.Object3D ): void {

		root.traverse( ( object ) => {

			const mesh = object as THREE.Mesh;
			if ( ! mesh.isMesh || ! mesh.castShadow ) return;
			for ( const material of Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ] ) {

				if ( this.frontCasters.has( material ) ) continue;
				this.frontCasters.add( material );
				const node = material as THREE.NodeMaterial;
				node.castShadowPositionNode = positionLocal.add( modelWorldMatrixInverse.mul( vec4( this.frontPush, 0 ) ).xyz );
				// (a bias that follows the sun: the rain's collision map draws the surface itself, RainCollision.ts)
				node.userData.shadowPush = true;
				node.needsUpdate = true;

			}

		} );
		this.cut = true;

	}

	/** Redraw from scratch on the next update: casters changed or the light jumped. */
	invalidate(): void {

		this.cut = true;

	}

	/** Whether the next update redraws the map or projects the caster grid without the light
	 * having moved (an invalidation, a new receiver): the settled signal waits for it. */
	get pending(): boolean {

		return this.cut || this.gridDirty;

	}

	/** Once per frame, after the light has moved and before anything renders with the map, which
	 * is drawn along `keyDir` (a world direction toward the light: see the header). */
	update( renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera, keyDir: THREE.Vector3 ): void {

		const shadow = this.sun.shadow;
		this.direction.value.copy( keyDir ).normalize();
		// three builds its map and the node drawing it with the first receiver's build (primeSharedNodes)
		const node = shadowRenderer( shadow );
		const map = node ? builtShadowMap( node ) : null;
		if ( ! node || ! map ) {

			shadow.needsUpdate = false;
			return;

		}
		if ( this.casterPipelines !== 'ready' ) {

			// the map's first redraw, its pipelines compiled first (see the header)
			if ( this.casterPipelines === 'none' ) {

				this.casterPipelines = 'compiling';
				const ready = () => { this.casterPipelines = 'ready'; };
				compileByRendering( renderer, () => this.render( renderer, scene, camera, node, keyDir ) ).then( ready, ready );

			}
			shadow.needsUpdate = false;
			return;

		}
		const turned = this.drawn.lengthSq() > 0 ? this.drawn.angleTo( keyDir ) : Infinity;
		const resized = this.maps.width !== shadow.mapSize.width || this.maps.height !== shadow.mapSize.height;
		const swaying = debugFlags.shadowWind !== false;
		if ( swaying !== ( shadowWindOn.value > 0.5 ) ) {

			shadowWindOn.value = swaying ? 1 : 0;
			this.cut = true;

		}
		const cut = this.cut || turned > CUT || resized;
		if ( cut || swaying ) {

			if ( ! cut && this.stepFrom.angleTo( keyDir ) >= this.step ) {

				this.steps ++;
				this.stepFrom.copy( keyDir );

			}
			this.draw( renderer, scene, camera, node, map, keyDir, cut );
			this.hasPrevious = false;
			this.cut = false;
			if ( this.gridWanted ) {

				this.projectGrid( scene, CURRENT, shadowCoordinateMatrix( shadow, new THREE.Matrix4() ) );
				this.clearGrid( PREVIOUS );

			}

		} else if ( turned >= this.step ) {

			// the current map becomes the previous one, with its projection
			// eslint-disable-next-line ponlai/history-on-graph -- the sun's map array is three's shadow, drawn outside the frame graph (three/ShadowPass.ts): the current step becomes the previous one the cross-fade reads
			renderer.copyTextureToTexture( this.maps.depthTexture!, this.maps.depthTexture!, this.currentLayer, this.toPreviousLayer );
			shadowCoordinateMatrix( shadow, this.previousCoordinate );
			this.draw( renderer, scene, camera, node, map, keyDir );
			this.hasPrevious = true;
			if ( this.gridWanted ) {

				this.shiftGrid();
				this.projectGrid( scene, CURRENT, shadowCoordinateMatrix( shadow, new THREE.Matrix4() ) );

			}

		}
		if ( this.gridDirty ) {

			// a receiver arrived after the last redraw: project what is drawn now
			this.projectGrid( scene, CURRENT, shadowCoordinateMatrix( shadow, new THREE.Matrix4() ) );
			if ( this.hasPrevious ) this.projectGrid( scene, PREVIOUS, this.previousCoordinate );
			else this.clearGrid( PREVIOUS );
			this.gridDirty = false;

		}
		if ( this.hasPrevious ) {

			shadowCoordinateMatrix( shadow, this.currentCoordinate ).invert();
			this.toPrevious.value.multiplyMatrices( this.previousCoordinate, this.currentCoordinate );

		}
		this.weight.value = this.hasPrevious ? Math.min( this.drawn.angleTo( keyDir ) / this.step, 1 ) : 1;

	}

	// `counted: false` for the swaying casters' redraws, which change the map only as the clock does
	// (the settled signal and the frame pacer read `draws`)
	private draw( renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera, node: ShadowRenderer, map: THREE.RenderTarget, keyDir: THREE.Vector3, counted = true ): void {

		const shadow = this.sun.shadow, shadowCamera = shadow.camera;
		if ( counted ) {

			this.draws ++;
			this.steps ++;
			this.stepFrom.copy( keyDir );

		}
		this.drawn.copy( keyDir );
		// Draw now rather than inside the next scene render, so the copy below lands before any
		// receiver reads the array. This is what ShadowNode.updateBefore would do with the flag.
		this.render( renderer, scene, camera, node, keyDir );
		if ( this.maps.width !== map.width || this.maps.height !== map.height ) this.maps.setSize( map.width, map.height, 2 );
		// eslint-disable-next-line ponlai/history-on-graph -- the sun's map array is three's shadow, drawn outside the frame graph (three/ShadowPass.ts): the map just drawn goes into the current step's layer
		renderer.copyTextureToTexture( map.depthTexture!, this.maps.depthTexture!, null, this.toCurrentLayer );
		this.currentLayer.max.set( map.width, map.height, CURRENT + 1 );
		this.mapSize.value.set( map.width, map.height );
		this.span.value.set( shadowCamera.right - shadowCamera.left, shadowCamera.top - shadowCamera.bottom );
		shadowCoordinateMatrix( shadow, this.coordinate.value );
		this.ready.value = 1;

	}

	// three's map, fitted to the casters for `keyDir` and drawn along it (the light stands there
	// for the draw: see the header)
	private render( renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera, node: ShadowRenderer, keyDir: THREE.Vector3 ): void {

		// (a light already along it stays as it is, so an unlifted key's map is the light's own bit for bit)
		const sun = this.sun, along = this.target.subVectors( sun.position, sun.target.position );
		const moved = along.angleTo( keyDir ) > 1e-6;
		this.lightPosition.copy( sun.position );
		if ( moved ) sun.position.copy( keyDir ).normalize().multiplyScalar( along.length() ).add( sun.target.position );
		this.fit( renderer, keyDir );
		// closed solids cast their front faces into this map alone (see the header)
		const sides = new Map( [ ...this.frontCasters ].map( ( material ) => [ material, material.shadowSide ] ) );
		for ( const material of this.frontCasters ) material.shadowSide = THREE.FrontSide;
		this.frontPush.value.copy( keyDir ).normalize().multiplyScalar( - FRONT_PUSH );
		redrawShadow( node, renderer, scene, camera );
		for ( const [ material, side ] of sides ) material.shadowSide = side;
		sun.shadow.needsUpdate = false;
		if ( moved ) {

			sun.position.copy( this.lightPosition );
			sun.updateMatrixWorld();

		}

	}

	// The casters' columns on a world-space grid over their bounds: the lowest and highest point
	// of every shadow-casting mesh in each cell (instances by their bounding boxes), padded for
	// what the rest pose leaves out. Generic: it follows whatever casts, on any layout. It walks
	// every casting vertex once, so each is binned into its own cell and the sideways padding is
	// a separate pass over the cells. A mesh with given columns (setCasters) folds each of its
	// columns into every cell the column meets, so the grid holds all its walk would, and at most a
	// column's width more.
	private buildColumns( scene: THREE.Scene ): void {

		const box = this.casters;
		const cell = Math.max( box.max.x - box.min.x, box.max.z - box.min.z, 1e-3 ) / WORLD_CELLS;
		const nx = Math.ceil( ( box.max.x - box.min.x ) / cell ) + 2 * GRID_SIDE_PAD + 1, nz = Math.ceil( ( box.max.z - box.min.z ) / cell ) + 2 * GRID_SIDE_PAD + 1;
		const x0 = box.min.x - GRID_SIDE_PAD * cell, z0 = box.min.z - GRID_SIDE_PAD * cell;
		const binLow = new Float64Array( nx * nz ).fill( Infinity ), binHigh = new Float64Array( nx * nz ).fill( - Infinity );
		// a caster outside the bounds lands in the nearest border cell, which only adds to it
		const column = ( x: number ) => Math.min( Math.max( Math.floor( ( x - x0 ) / cell ), 0 ), nx - 1 );
		const row = ( z: number ) => Math.min( Math.max( Math.floor( ( z - z0 ) / cell ), 0 ), nz - 1 );
		const mark = ( xa: number, xb: number, za: number, zb: number, ya: number, yb: number ) => {

			const ib = column( xb ), jb = row( zb );
			for ( let j = row( za ); j <= jb; j ++ ) for ( let i = column( xa ); i <= ib; i ++ ) {

				const k = j * nx + i;
				if ( ya < binLow[ k ] ) binLow[ k ] = ya;
				if ( yb > binHigh[ k ] ) binHigh[ k ] = yb;

			}

		};
		const bounds = new THREE.Box3(), instance = new THREE.Matrix4();
		const shadowLayers = this.sun.shadow.camera.layers;
		scene.updateMatrixWorld();
		scene.traverseVisible( ( object ) => {

			const mesh = object as THREE.Mesh;
			if ( ! mesh.isMesh || ! mesh.castShadow || ! mesh.layers.test( shadowLayers ) ) return;
			const given = this.given.get( mesh );
			if ( given ) {

				for ( let j = 0; j < given.nz; j ++ ) {

					const za = given.z0 + j * given.cell, zb = za + given.cell;
					const ja = row( za ), jb = Math.min( Math.max( Math.ceil( ( zb - z0 ) / cell ) - 1, ja ), nz - 1 );
					for ( let i = 0; i < given.nx; i ++ ) {

						const k = j * given.nx + i;
						if ( ! ( given.high[ k ] >= given.low[ k ] ) ) continue;
						const xa = given.x0 + i * given.cell, ia = column( xa ), ib = Math.min( Math.max( Math.ceil( ( xa + given.cell - x0 ) / cell ) - 1, ia ), nx - 1 );
						for ( let b = ja; b <= jb; b ++ ) for ( let a = ia; a <= ib; a ++ ) {

							const c = b * nx + a;
							if ( given.low[ k ] < binLow[ c ] ) binLow[ c ] = given.low[ k ];
							if ( given.high[ k ] > binHigh[ c ] ) binHigh[ c ] = given.high[ k ];

						}

					}

				}
				return;

			}
			const geometry = mesh.geometry;
			if ( ! geometry.boundingBox ) geometry.computeBoundingBox();
			const instanced = mesh as THREE.InstancedMesh;
			const position = geometry.getAttribute( 'position' );
			if ( instanced.isInstancedMesh ) {

				for ( let n = 0; n < instanced.count; n ++ ) {

					instanced.getMatrixAt( n, instance );
					bounds.copy( geometry.boundingBox! ).applyMatrix4( instance.premultiply( mesh.matrixWorld ) );
					mark( bounds.min.x, bounds.max.x, bounds.min.z, bounds.max.z, bounds.min.y, bounds.max.y );

				}

			} else if ( position && ! position.normalized && ! ( mesh as THREE.SkinnedMesh ).isSkinnedMesh ) {

				// a mesh's world matrix is affine: apply it by hand, without Vector3's divide
				const e = mesh.matrixWorld.elements;
				const interleaved = position as THREE.InterleavedBufferAttribute;
				const values = interleaved.isInterleavedBufferAttribute ? interleaved.data.array : position.array;
				const stride = interleaved.isInterleavedBufferAttribute ? interleaved.data.stride : position.itemSize;
				const offset = interleaved.isInterleavedBufferAttribute ? interleaved.offset : 0;
				for ( let v = offset, end = offset + position.count * stride; v < end; v += stride ) {

					const px = values[ v ], py = values[ v + 1 ], pz = values[ v + 2 ];
					const x = e[ 0 ] * px + e[ 4 ] * py + e[ 8 ] * pz + e[ 12 ];
					const y = e[ 1 ] * px + e[ 5 ] * py + e[ 9 ] * pz + e[ 13 ];
					const z = e[ 2 ] * px + e[ 6 ] * py + e[ 10 ] * pz + e[ 14 ];
					const k = row( z ) * nx + column( x );
					if ( y < binLow[ k ] ) binLow[ k ] = y;
					if ( y > binHigh[ k ] ) binHigh[ k ] = y;

				}

			} else {

				bounds.copy( geometry.boundingBox! ).applyMatrix4( mesh.matrixWorld );
				mark( bounds.min.x, bounds.max.x, bounds.min.z, bounds.max.z, bounds.min.y, bounds.max.y );

			}

		} );
		// pad each column by its neighbours within GRID_SIDE_PAD cells, and upward
		const low = new Float32Array( nx * nz ).fill( Infinity ), high = new Float32Array( nx * nz ).fill( - Infinity );
		for ( let j = 0; j < nz; j ++ ) for ( let i = 0; i < nx; i ++ ) {

			const k = j * nx + i;
			if ( ! ( binHigh[ k ] >= binLow[ k ] ) ) continue;
			const top = binHigh[ k ] + GRID_HEIGHT_PAD;
			const jb = Math.min( j + GRID_SIDE_PAD, nz - 1 ), ib = Math.min( i + GRID_SIDE_PAD, nx - 1 );
			for ( let b = Math.max( j - GRID_SIDE_PAD, 0 ); b <= jb; b ++ ) for ( let a = Math.max( i - GRID_SIDE_PAD, 0 ); a <= ib; a ++ ) {

				const c = b * nx + a;
				if ( binLow[ k ] < low[ c ] ) low[ c ] = binLow[ k ];
				if ( top > high[ c ] ) high[ c ] = top;

			}

		}
		this.columns = { x0, z0, cell, nx, nz, low, high };

	}

	// The previous map's half becomes the current one's on a step.
	private shiftGrid(): void {

		const values = this.casterDepths.array as THREE.Vector4[];
		const half = GRID * GRID / 4;
		for ( let k = 0; k < half; k ++ ) values[ PREVIOUS * half + k ].copy( values[ CURRENT * half + k ] );

	}

	private clearGrid( layer: number ): void {

		const values = this.casterDepths.array as THREE.Vector4[];
		const half = GRID * GRID / 4;
		for ( let k = 0; k < half; k ++ ) values[ layer * half + k ].setScalar( UNKNOWN );

	}

	// Project the columns into one map's light-space grid, through that map's shadow coordinate.
	private projectGrid( scene: THREE.Scene, layer: number, matrix: THREE.Matrix4 ): void {

		if ( ! this.columns ) this.buildColumns( scene );
		const { x0, z0, cell, nx, nz, low, high } = this.columns!;
		const values = this.casterDepths.array as THREE.Vector4[];
		const half = GRID * GRID / 4;
		const depth = new Float32Array( GRID * GRID ).fill( NO_CASTER );
		const p = new THREE.Vector3();
		// the bilinear footprint beyond a cell's edge, in map units
		const reach = 2 / Math.max( this.mapSize.value.x, 1 );
		for ( let j = 0; j < nz; j ++ ) for ( let i = 0; i < nx; i ++ ) {

			const k = j * nx + i;
			if ( ! ( high[ k ] >= low[ k ] ) ) continue;
			let minX = Infinity, maxX = - Infinity, minY = Infinity, maxY = - Infinity, nearest = Infinity;
			for ( let c = 0; c < 8; c ++ ) {

				p.set( x0 + ( i + ( c & 1 ) ) * cell, c & 4 ? high[ k ] : low[ k ], z0 + ( j + ( ( c >> 1 ) & 1 ) ) * cell ).applyMatrix4( matrix );
				minX = Math.min( minX, p.x ); maxX = Math.max( maxX, p.x );
				minY = Math.min( minY, p.y ); maxY = Math.max( maxY, p.y );
				nearest = Math.min( nearest, p.z );

			}
			const ia = Math.max( Math.floor( ( minX - reach ) * GRID ), 0 ), ib = Math.min( Math.floor( ( maxX + reach ) * GRID ), GRID - 1 );
			const ja = Math.max( Math.floor( ( minY - reach ) * GRID ), 0 ), jb = Math.min( Math.floor( ( maxY + reach ) * GRID ), GRID - 1 );
			for ( let b = ja; b <= jb; b ++ ) for ( let a = ia; a <= ib; a ++ ) depth[ b * GRID + a ] = Math.min( depth[ b * GRID + a ], nearest );

		}
		for ( let k = 0; k < half; k ++ ) values[ layer * half + k ].fromArray( depth, k * 4 );

	}

	// Fit the orthographic frustum to the casters and the shadows they throw down to the floor.
	// Light-space x and y of a shadow equal its caster's, so the casters' footprint bounds every
	// shadowed receiver; only the depth range reaches down to the floor. The island's footprint
	// is far smaller than the sea floor's stones' (which cast into the refracted map, not this
	// one), so its texels are a few times finer than a fixed frame around both.
	private fit( renderer: THREE.WebGPURenderer, keyDir: THREE.Vector3 ): void {

		const sun = this.sun, shadow = sun.shadow, camera = shadow.camera;
		sun.updateMatrixWorld();
		sun.target.updateMatrixWorld();
		if ( ! this.casters.isEmpty() ) {

			// the view LightShadow.updateMatrices builds, before its projection is known
			camera.position.setFromMatrixPosition( sun.matrixWorld );
			camera.lookAt( this.target.setFromMatrixPosition( sun.target.matrixWorld ) );
			camera.updateMatrixWorld();
			const bounds = this.lightBounds.makeEmpty(), box = this.casters;
			// Toward the horizon the depth reach is bounded; the key light has faded out by then.
			const rise = Math.max( keyDir.y, 0.02 );
			for ( let i = 0; i < 8; i ++ ) {

				const corner = this.corner.set( i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z );
				const drop = Math.max( corner.y - this.floor, 0 ) / rise;
				bounds.expandByPoint( this.target.copy( corner ).applyMatrix4( camera.matrixWorldInverse ) );
				bounds.expandByPoint( corner.addScaledVector( keyDir, - drop ).applyMatrix4( camera.matrixWorldInverse ) );

			}
			const margin = this.margin;
			camera.left = bounds.min.x - margin;
			camera.right = bounds.max.x + margin;
			camera.bottom = bounds.min.y - margin;
			camera.top = bounds.max.y + margin;
			camera.near = Math.max( 0.1, - bounds.max.z - margin );
			camera.far = - bounds.min.z + margin;

		}
		camera.coordinateSystem = renderer.coordinateSystem;
		camera.updateProjectionMatrix();
		shadow.bias = - this.bias / ( camera.far - camera.near );
		shadow.updateMatrices( sun );

	}

}
