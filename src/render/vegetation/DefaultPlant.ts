import * as THREE from 'three/webgpu';
import { float, positionGeometry } from 'three/tsl';
import { vegetationMaterial } from './Material.ts';
import { leafAlphaTexture, leafTexture } from './LeafTexture.ts';
import type { PlantSite } from './Registry.ts';
import type { Shared } from '../shared.ts';

// The default plant: two crossed leaf cards for a plant kind without art of its own.
//
// The default plant: what a plant kind without art of its own is drawn with (Registry.ts), so a
// new kind can be placed, lit, shadowed and measured on the day it is registered. One mesh for
// all its sites: at each site two upright cards crossed at a right angle, SIZE wide and tall,
// their foot on the site, with the folded leaf's texture (LeafTexture.ts) on the default
// vegetation material (Material.ts: the 'leaf' family, so it takes the vegetation's shadow filter,
// the depth prepass's tag and the leaf's light through). It does not sway: its swayed pose is its
// rest pose. Every card's normal is up, as a leaf card's shading wants (lit by the sky and the key
// from above, whichever way the card faces).
//
// Nothing is made as the module loads: a page with no such kind builds no material here.

/** A card's width and height (m). */
const SIZE = 1;

/** The default plants at `sites` as one mesh, named `name`; `leaf` is the cards' colour and
 * cut-out (the folded leaf's, drawn on canvases, unless given: a tool under node, which has no
 * canvas). */
export function defaultPlant( shared: Shared, sites: readonly PlantSite[], name = 'default plant', leaf: { map: THREE.Texture; alphaMap: THREE.Texture } = { map: leafTexture(), alphaMap: leafAlphaTexture() } ): THREE.Mesh {

	const positions: number[] = [], normals: number[] = [], uvs: number[] = [], colours: number[] = [], index: number[] = [];
	const h = SIZE / 2;
	for ( const site of sites ) {

		// two cards: along x, along z
		for ( const [ ax, az ] of [ [ 1, 0 ], [ 0, 1 ] ] ) {

			const base = positions.length / 3;
			for ( const [ u, v ] of [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ] ) {

				positions.push( site.x + ( u * 2 - 1 ) * h * ax, site.y + v * SIZE, site.z + ( u * 2 - 1 ) * h * az );
				normals.push( 0, 1, 0 );
				uvs.push( u, v );
				colours.push( 1, 1, 1 );

			}
			index.push( base, base + 1, base + 2, base, base + 2, base + 3 );

		}

	}
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( positions, 3 ) );
	geometry.setAttribute( 'normal', new THREE.Float32BufferAttribute( normals, 3 ) );
	geometry.setAttribute( 'uv', new THREE.Float32BufferAttribute( uvs, 2 ) );
	geometry.setAttribute( 'color', new THREE.Float32BufferAttribute( colours, 3 ) );
	geometry.setIndex( new THREE.Uint32BufferAttribute( index, 1 ) );
	const material = vegetationMaterial( shared, {
		family: 'leaf', name, roughness: 0.75,
		leaf,
		wet: { tint: 0.75, roughness: 0.25 },
		rest: positionGeometry,
		displacement: () => positionGeometry,
		light: { scale: float( 1 ), translucency: 0.5 },
	} );
	const mesh = new THREE.Mesh( geometry, material );
	mesh.name = name;
	mesh.castShadow = true;
	mesh.receiveShadow = true;
	return mesh;

}
