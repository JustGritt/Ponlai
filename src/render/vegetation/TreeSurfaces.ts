import * as THREE from 'three/webgpu';
import {
	Fn, Loop, attribute, dot, float, floor, fract, int, length, materialColor, max, mix, mod, normalGeometry, normalMap, normalize,
	positionGeometry, select, sqrt, texture, uv, vec2, vec3, vec4, If,
} from 'three/tsl';
import { GroundBounceLightingModel } from '../lighting/GroundBounce.ts';
import { pnoise, hash22 } from '../materials/ProceduralTextures.ts';
import type { Shared } from '../shared.ts';
import { branchBend, leafFlutter, mainBend, sprayBend } from './Deform.ts';
import { rootHash } from '../weather/Wind.ts';
import { coverageAlphaTexture, textureCanvas } from './LeafTexture.ts';
import { vegetationMaterial } from './Material.ts';
import type { PlacedVertex } from './PlantInstances.ts';
import { writeMeter } from '../lighting/MeterLight.ts';

// What every tree's surfaces are made from, with no image files: baked bark and its material.
//
// What every tree's surfaces are made from (BambooMaterials.ts; the bark for the trees to
// come), with no image files:
// - Bark: a tree's bake (a TSL function of the tile's uv: R bark height, G deadwood height, B
//   tone, A fissure depth) rendered once at load into a tileable pair of maps (`bakeBarkMaps`:
//   heights and the normals from them), on the tubes' uv (vegetation/TreeMesh.ts: around the
//   branch, along it); `barkMaterial` mixes bark, deadwood and a hollow's dark by the wood's
//   `wood` attribute.
// - Leaves: a 2x2 atlas of sprays drawn once on canvases (`LeafPainter`): the colour map holds
//   the leaves' own colours; the alpha map their silhouette, its mips keeping the coverage
//   (vegetation/LeafTexture.ts); the id map a random per leaf (RG its facet's tilt, B its tone, A 1
//   on a leaf), point-sampled, its mips point-decimated so a leaf keeps one value at every
//   distance. `leafMaterial` puts the vegetation's material on it (vegetation/Material.ts: leaf
//   family, alpha to coverage, the depth prepass, light through the leaves, a cuticle's low specular
//   reflectance), each leaf toned by its id, the crown bending in the one wind and each card
//   fluttering.
// - Each plant's own tint (`plantTint`): its hash looks up a gradient of two or three colours about
//   overlay's neutral grey, laid over the albedo where the plant has leaves, less the share it
//   keeps, then a colour matrix of the material's (near the identity), so plants of one kind differ
//   a little in colour.

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// ------------------------------------------------------------------------------------------- bark

/** A periodic fractal sum with a period per axis (integers, so the tile repeats). */
export const fbm2 = ( q: V2, px: number, py: number, octaves: number ) => {

	let s: F = float( 0 ), a = 0.5, n = 0;
	for ( let o = 0; o < octaves; o ++ ) {

		const k = 2 ** o;
		s = s.add( pnoise( q.mul( vec2( px * k, py * k ) ), vec2( px * k, py * k ) ).mul( a ) );
		n += a;
		a *= 0.5;

	}
	return s.div( n );

};

/** Periodic cellular noise with a period per axis: F1, F2. */
export const cells = Fn( ( [ q, per ]: [ V2, V2 ] ) => {

	const p = q.mul( per );
	const i = floor( p ).toVar();
	const f = fract( p ).toVar();
	const f1 = float( 9 ).toVar(), f2 = float( 9 ).toVar();
	Loop( { start: int( - 1 ), end: int( 1 ), condition: '<=' }, { start: int( - 1 ), end: int( 1 ), condition: '<=' }, ( { i: y, j: x } ) => {

		const o = vec2( float( x ), float( y ) );
		const d = length( o.add( 0.5 ).add( hash22( mod( i.add( o ), per ) ).mul( 0.42 ) ).sub( f ) ).toVar();
		If( d.lessThan( f1 ), () => {

			f2.assign( f1 );
			f1.assign( d );

		} ).ElseIf( d.lessThan( f2 ), () => {

			f2.assign( d );

		} );

	} );
	return vec2( f1, f2 );

} ).setLayout( { name: 'barkCells', type: 'vec2', inputs: [ { name: 'q', type: 'vec2' }, { name: 'per', type: 'vec2' } ] } );

export interface BarkMaps { heights: THREE.Texture; normals: THREE.Texture }

/** A tree's bark: its tile (m around the branch, along it), its bake (R bark height, G deadwood
 * height, B tone, A fissure depth; u around, v along) and the relief its heights span (m, bark and
 * deadwood). */
export interface BarkBake {
	name: string;
	tile: { readonly around: number; readonly along: number };
	bake: ( q: V2 ) => V4;
	relief: { bark: number; dead: number };
}

/** The bark's maps, baked on the GPU (ProceduralTextures.ts's way): heights and normals. */
export function bakeBarkMaps( renderer: THREE.WebGPURenderer, bark: BarkBake, size = 512 ): BarkMaps {

	const make = ( name: string ) => {

		const rt = new THREE.RenderTarget( size, size, {
			type: THREE.UnsignedByteType, generateMipmaps: true,
			minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter,
			wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping, depthBuffer: false, anisotropy: 8,
		} );
		rt.texture.name = name;
		return rt;

	};
	const heights = make( `${ bark.name } heights` ), normals = make( `${ bark.name } normals` );
	const quad = new THREE.QuadMesh( new THREE.NodeMaterial() );
	const render = ( rt: THREE.RenderTarget, node: THREE.Node ) => {

		const material = new THREE.NodeMaterial();
		material.fragmentNode = node;
		quad.material = material;
		renderer.setRenderTarget( rt );
		quad.render( renderer );
		material.dispose();

	};
	const previous = renderer.getRenderTarget();
	render( heights, bark.bake( uv() ) );
	// normals from the heights: relief over texel size, per axis (the tile is longer than wide)
	const h = heights.texture, e = 1 / size;
	const l = texture( h, uv().sub( vec2( e, 0 ) ) ), r = texture( h, uv().add( vec2( e, 0 ) ) );
	const d = texture( h, uv().sub( vec2( 0, e ) ) ), u = texture( h, uv().add( vec2( 0, e ) ) );
	const { relief } = bark;
	const sx = size / bark.tile.around, sy = size / bark.tile.along;
	const nb = normalize( vec3( r.r.sub( l.r ).mul( 0.5 * relief.bark * sx ).negate(), u.r.sub( d.r ).mul( 0.5 * relief.bark * sy ).negate(), 1 ) );
	const nd = normalize( vec3( r.g.sub( l.g ).mul( 0.5 * relief.dead * sx ).negate(), u.g.sub( d.g ).mul( 0.5 * relief.dead * sy ).negate(), 1 ) );
	render( normals, vec4( nb.xy.mul( 0.5 ).add( 0.5 ), nd.xy.mul( 0.5 ).add( 0.5 ) ) );
	renderer.setRenderTarget( previous );
	( quad.material as THREE.Material ).dispose();
	return { heights: heights.texture, normals: normals.texture };

}

/** A bark's linear albedos: plate tops, fissures, deadwood, a hollow's dark; how far the bake's
 * tone varies the bark; how far the fissures darken its ambient light. */
export interface BarkColours {
	plate: readonly number[]; fissure: readonly number[]; dead: readonly number[]; hollow: readonly number[];
	tone: number; fissureAo: number;
}

/** The wood: bark and deadwood from the maps, mixed by the `wood` attribute (deadwood, hollow). */
export function barkMaterial( maps: BarkMaps, colours: BarkColours, name: string ): THREE.MeshStandardNodeMaterial {

	const material = new THREE.MeshStandardNodeMaterial( { roughness: 0.9, metalness: 0 } );
	const wood = attribute<'vec2'>( 'wood', 'vec2' );
	const dead = wood.x, hollow = wood.y;
	const h = texture( maps.heights, uv() ).toVar();
	const n = texture( maps.normals, uv() ).toVar();
	const bark = mix( vec3( ...colours.fissure ), vec3( ...colours.plate ), h.r ).mul( h.b.mul( colours.tone ).add( 1 - colours.tone / 2 ) );
	const deadwood = vec3( ...colours.dead ).mul( h.g.mul( 0.7 ).add( 0.45 ) );
	material.colorNode = mix( mix( bark, deadwood, dead ), vec3( ...colours.hollow ), hollow );
	// the two normals mixed in tangent space, flattened in the hollow's dark
	const xy = mix( n.rg, n.ba, dead ).mul( 2 ).sub( 1 ).mul( float( 1 ).sub( hollow.mul( 0.8 ) ) );
	const z = sqrt( max( float( 1 ).sub( dot( xy, xy ) ), 0 ) );
	material.normalNode = normalMap( vec3( xy, z ).mul( 0.5 ).add( 0.5 ) );
	material.roughnessNode = mix( 0.92, 0.7, dead );
	material.aoNode = mix( float( 1 ), mix( colours.fissureAo, 0.8, dead ), h.a.mul( float( 1 ).sub( dead ) ) );
	material.name = name;
	( material as unknown as { setupLightingModel: () => THREE.LightingModel } ).setupLightingModel = () => new GroundBounceLightingModel();
	writeMeter( material );
	return material;

}

// ----------------------------------------------------------------------------------------- leaves

export interface LeafAtlas { map: THREE.Texture; alpha: THREE.Texture; ids: THREE.Texture }

/** The three canvases of a leaf atlas (colour, silhouette, ids) and what draws on them; `finish`
 * makes the atlas's textures. `curve`: how far a leaf's facet tilt turns across its width (of the
 * id map's 255), so up close only a band of a leaf catches the sun, not the whole leaf. */
export class LeafPainter {

	readonly colour: CanvasRenderingContext2D;
	readonly shape: CanvasRenderingContext2D;
	readonly id: CanvasRenderingContext2D;

	constructor( readonly size: number, readonly rng: () => number, readonly curve: number, under: string ) {

		const canvas = () => {

			const c = textureCanvas( size );
			return c.getContext( '2d' )!;

		};
		this.colour = canvas();
		this.shape = canvas();
		this.id = canvas();
		// under the leaves the colour map holds their mean, so its mips do not darken their edges
		this.colour.fillStyle = under;
		this.colour.fillRect( 0, 0, size, size );

	}

	/** A leaf's blade (`path`, its stalk's end at the origin and its tip along +x) at ( x, y ),
	 * turned by `angle`, in `rgb`: colour and silhouette, then `detail` on the colour (veins,
	 * shading), then its id (`width` across). */
	leaf( path: Path2D, x: number, y: number, angle: number, width: number, rgb: readonly number[], detail?: ( g: CanvasRenderingContext2D ) => void ): void {

		const { rng, curve } = this;
		for ( const [ g, style ] of [ [ this.colour, `rgb(${ rgb[ 0 ] },${ rgb[ 1 ] },${ rgb[ 2 ] })` ], [ this.shape, '#fff' ] ] as const ) {

			g.save();
			g.translate( x, y );
			g.rotate( angle );
			g.fillStyle = style;
			g.fill( path );
			g.restore();

		}
		if ( detail ) {

			this.colour.save();
			this.colour.translate( x, y );
			this.colour.rotate( angle );
			detail( this.colour );
			this.colour.restore();

		}
		// its id: the facet's tilt (RG) turning across its width, its tone (B), a leaf (A)
		const gi = this.id;
		gi.save();
		gi.translate( x, y );
		gi.rotate( angle );
		const tr = curve + Math.floor( rng() * ( 256 - 2 * curve ) ), tg = curve + Math.floor( rng() * ( 256 - 2 * curve ) ), tb = Math.floor( rng() * 256 );
		const across = gi.createLinearGradient( 0, - width / 2, 0, width / 2 );
		across.addColorStop( 0, `rgb(${ tr - curve },${ tg - curve },${ tb })` );
		across.addColorStop( 1, `rgb(${ tr + curve },${ tg + curve },${ tb })` );
		gi.fillStyle = across;
		gi.fill( path );
		gi.restore();

	}

	/** A stroke on the colour (in `style`) and the silhouette. */
	twig( x0: number, y0: number, x1: number, y1: number, width: number, style: string ): void {

		for ( const g of [ this.colour, this.shape ] ) {

			g.strokeStyle = g === this.colour ? style : '#fff';
			g.lineWidth = width;
			g.lineCap = 'round';
			g.beginPath();
			g.moveTo( x0, y0 );
			g.lineTo( x1, y1 );
			g.stroke();

		}

	}

	/** Runs `draw` clipped to atlas tile `t` (of four, 2 px in from its edges), with its corner. */
	tile( t: number, draw: ( ox: number, oy: number, tile: number ) => void ): void {

		const tile = this.size / 2, ox = ( t % 2 ) * tile, oy = Math.floor( t / 2 ) * tile;
		for ( const g of [ this.colour, this.shape, this.id ] ) {

			g.save();
			g.beginPath();
			g.rect( ox + 2, oy + 2, tile - 4, tile - 4 );
			g.clip();

		}
		draw( ox, oy, tile );
		for ( const g of [ this.colour, this.shape, this.id ] ) g.restore();

	}

	finish( name: string ): LeafAtlas {

		const { size } = this;
		const map = new THREE.CanvasTexture( this.colour.canvas );
		map.colorSpace = THREE.SRGBColorSpace;
		map.minFilter = THREE.LinearMipmapLinearFilter;
		map.anisotropy = 4;
		map.name = name;
		const coverage = this.shape.getImageData( 0, 0, size, size ).data;
		const alpha = new Float32Array( size * size );
		for ( let i = 0; i < alpha.length; i ++ ) alpha[ i ] = coverage[ i * 4 + 3 ] / 255;
		const alphaMap = coverageAlphaTexture( alpha, size );
		alphaMap.name = `${ name } alpha`;
		const ids = pointMips( this.id.getImageData( 0, 0, size, size ), size );
		ids.name = `${ name } ids`;
		return { map, alpha: alphaMap, ids };

	}

}

/** A data texture point-sampled at every level: each mip keeps one texel of each 2x2 of the level
 * above (no average), so a leaf's id stays a leaf's at a distance. */
function pointMips( image: ImageData, size: number ): THREE.Texture {

	const levels: HTMLCanvasElement[] = [];
	let data = image.data, w = size;
	for ( ; ; ) {

		const c = textureCanvas( w );
		const level = new ImageData( new Uint8ClampedArray( data ), w, w );
		c.getContext( '2d' )!.putImageData( level, 0, 0 );
		levels.push( c );
		if ( w === 1 ) break;
		const half = w >> 1, next = new Uint8ClampedArray( half * half * 4 );
		for ( let y = 0; y < half; y ++ ) for ( let x = 0; x < half; x ++ ) {

			// the texel of each 2x2 that holds a leaf, if any, so a leaf wins over the gaps
			let pick = ( 2 * y * w + 2 * x ) * 4;
			for ( const o of [ 0, 1, w, w + 1 ] ) {

				const k = ( 2 * y * w + 2 * x + o ) * 4;
				if ( data[ k + 3 ] > 0 ) { pick = k; break; }

			}
			next.set( data.subarray( pick, pick + 4 ), ( y * half + x ) * 4 );

		}
		data = next;
		w = half;

	}
	const tex = new THREE.CanvasTexture( levels[ 0 ] );
	tex.mipmaps = levels as unknown as THREE.Texture[ 'mipmaps' ];
	tex.generateMipmaps = false;
	tex.minFilter = THREE.NearestMipmapNearestFilter;
	tex.magFilter = THREE.NearestFilter;
	tex.colorSpace = THREE.NoColorSpace;
	tex.premultiplyAlpha = false;
	return tex;

}

/** A material's per-plant tint (`plantTint`): the gradient's three keys (linear colours about 0.5,
 * overlay's neutral, at the plant's hash 0, 0.5 and 1), the share of the albedo kept from it, and
 * the material's colour matrix (rows, row-major). */
export interface PlantTint { keys: readonly ( readonly [ number, number, number ] )[]; keep: number; matrix: readonly number[] }

/** The identity colour matrix. */
export const IDENTITY_COLOUR = [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ] as const;

/** `albedo` tinted for the plant of hash `xi` (0..1) by `t`: the gradient's colour at xi blended
 * toward overlay's neutral 0.5 outside the `mask` (1 on leaves) and by the share kept, overlaid on
 * the albedo ( 2 a b below a half, 1 - 2 ( 1 - a )( 1 - b ) above ), then the material's matrix. */
export function plantTint( albedo: V3, xi: F, mask: F, t: PlantTint ): V3 {

	const [ k0, k1, k2 ] = t.keys.map( ( k ) => vec3( ...k ) );
	const key = mix( mix( k0, k1, xi.mul( 2 ).clamp( 0, 1 ) ), k2, xi.mul( 2 ).sub( 1 ).clamp( 0, 1 ) );
	const blend = mix( vec3( 0.5 ), key, mask.mul( 1 - t.keep ) ).toVar();
	const a = albedo.toVar();
	const low = a.mul( blend ).mul( 2 ).toVar(), high = float( 1 ).sub( float( 1 ).sub( a ).mul( float( 1 ).sub( blend ) ).mul( 2 ) ).toVar();
	const over = vec3( select( a.x.lessThan( 0.5 ), low.x, high.x ), select( a.y.lessThan( 0.5 ), low.y, high.y ), select( a.z.lessThan( 0.5 ), low.z, high.z ) );
	const m = t.matrix;
	return vec3( dot( vec3( m[ 0 ], m[ 1 ], m[ 2 ] ), over ), dot( vec3( m[ 3 ], m[ 4 ], m[ 5 ] ), over ), dot( vec3( m[ 6 ], m[ 7 ], m[ 8 ] ), over ) );

}

/** A crown's wind: it bends from the trunk's foot by the main bend (Deform.ts `mainBend`, at its
 * sway `rate`, swings a metre of the plants' travel, its drag growing as the wind to `drag`) by the
 * `leafSway` attribute's x (the yield ( 1 - stiffness ) eta^2, eta, the point's share of the
 * plant's height, in w) under the plant's wind load (an instance's; 1 without one); each card
 * flutters on its own phase (z) by y (A_l w^2, m: w 0 where it holds to its twig, 1 at its tip), at
 * `flutter` Hz (Deform.ts `leafFlutter`: in the gusts, across the wind). */
export interface CrownWind { rate: number; drag: number; flutter: number }

/** A crown's leaves: the vegetation's leaf family on the atlas, each leaf toned by its id (by
 * `tone` about 1), swaying from `foot` (a tree's, or a node: several trees' in one mesh, per
 * vertex), letting light through by `translucency` (VegetationLighting). With `branches` the
 * cards move with their branch and their spray too (Deform.ts `branchBend`, `sprayBend`, the mesh's
 * TreeMesh.ts BRANCH_ATTRIBUTES). With `tint` each plant (its foot's hash, or `tintAt` where
 * given) takes its own tint over the leaves (`plantTint`). With `place` the mesh is a prototype drawn as instances
 * (PlantInstances.ts): every point, normal and limb placed by the instance, the plant's foot its
 * root. */
export function leafMaterial( shared: Shared, atlas: LeafAtlas, foot: THREE.Vector3 | V3, o: {
	name: string; roughness: number; wind: CrownWind; branches?: boolean; tint?: PlantTint; tone: number; translucency: F | number;
	/** an instanced plant's vertex (PlantInstances.ts `placeVertex`; `foot` is then ignored) and its
	 * leaves' shade (fragment) */
	place?: PlacedVertex; shade?: F;
	/** the plant's place on its tint's gradient (0..1, fragment), in place of its foot's hash */
	tintAt?: F;
} ): THREE.MeshStandardNodeMaterial {

	const place = o.place;
	const sway = place ? place.sway : attribute<'vec4'>( 'leafSway', 'vec4' );
	const root = place ? place.root : foot instanceof THREE.Vector3 ? vec3( foot.x, foot.y, foot.z ) : foot;
	// (the per-plant tint in the fragment: an instance's foot through a varying)
	const tintRoot = place ? place.rootVarying : root;
	const position = place ? place.position : positionGeometry;
	const normal = place ? place.normal : normalGeometry;
	const limb = ( name: string ) => place ? place.limb( name ) : attribute<'vec4'>( name, 'vec4' );
	// the leaf under the fragment: sampled once, in the colour's (uniform) flow
	const id = texture( atlas.ids, uv() ).toVar( 'leafId' );
	let crown: V2 | undefined;
	return vegetationMaterial( shared, {
		family: 'leaf', name: o.name, roughness: o.roughness,
		leaf: { map: atlas.map, alphaMap: atlas.alpha },
		wet: { tint: 0.75, roughness: 0.25 },
		// each leaf's own tone (the atlas's tones vary by colour already: this varies them by leaf)
		// (and the look's gain over the atlas: shared.ts leafGain)
		color: ( tint ) => {

			const leaf = vec3( id.b.mul( o.tone ).add( 1 - o.tone / 2 ).mul( tint ).mul( shared.leafGain ) );
			const base = materialColor.mul( vec4( o.shade ? leaf.mul( o.shade ) : leaf, 1 ) );
			if ( ! o.tint ) return base;
			const c = base.toVar();
			return vec4( plantTint( c.rgb, o.tintAt ?? rootHash( tintRoot.xz ), float( 1 ), o.tint ), c.a );

		},
		rest: position,
		displacement: ( wind ) => {

			const bent = mainBend( wind, position, root, sway.x, sway.w, place ? place.load : float( 1 ), o.wind );
			if ( wind === shared.wind ) crown = bent.sway;
			const flutter = leafFlutter( wind, bent.field, normal, sway.y, sway.z, o.wind.flutter, position );
			if ( ! o.branches ) return bent.position.add( flutter );
			const limbs = branchBend( wind, position, root, limb( 'branchFrom' ), limb( 'branchTo' ), bent.field )
				.add( sprayBend( wind, position, root, limb( 'sprayFrom' ), limb( 'sprayTo' ), bent.field ) );
			return bent.position.add( limbs ).add( flutter );

		},
		sway: () => crown,
		beforePosition: place?.before,
		light: { scale: float( 1 ), translucency: o.translucency },
	} );

}
