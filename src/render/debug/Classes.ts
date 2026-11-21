import * as THREE from 'three/webgpu';
import { min, normalWorldGeometry, select, vec3 } from 'three/tsl';
import { debugFlags } from './DebugFlags.ts';
import { onSceneDispose } from '../SceneDispose.ts';
import { STORE_CEILING, preExposure, sceneUnits } from '../post/PreExposure.ts';

// The kinds' view: every lit surface drawn unlit in a flat colour that names its kind.
//
// The kinds' view (debug builds, `flags=classes`, read at load): every lit surface drawn unlit in a
// flat colour that names its kind, so a script reads what each pixel shows (a script that reads cells:
// `__PONLAI__.radiance( { aov: 'class' } )`, debug/RadianceAov.ts).
//
// The kinds (FIT_CLASSES, by index k): sky, sea, ground (the terrain's turf and the sea floor),
// rock (the cliffs and the stones), grass (the meadow's blades and flowers), leaves (the bamboo's
// and the yunpeng's), wood (the bamboo's culms), figure (the islanders) and other (anything else
// lit). The sky and the sea are not lit by these models: a reader finds them by depth (the sky's
// view z beyond the far sentinel, the sea's surface in front of the opaque pass). A kind's colour
// is the linear ( 1, t, t^2 ) with t = ( k + 1 ) / 10, decoded from the ratios G / R and B / R, so
// the decoding holds whatever scale the target stores at (the pre-exposure) and a pixel whose
// samples mix two kinds decodes to none. Under MSAA that is every pixel the cut-out leaves cover in
// part (half the island's pixels seen from above), and two kinds' blends of four samples can equal
// two others', so a page read for its kinds loads with `aa=off` (the classes page of a script that reads cells).
// The code is stored at the pre-exposure's scale up to STORE_CEILING (2^15), not past it: a code at
// 2^P overflowed half float from P = 16 on (a deep overcast night's), its red clamped where its
// green and blue were not, and every lit cell decoded to none. Capped, the ratios stay exact (a power
// of two). Inside the cap (P up to 15) the haze and emission added after the code are as small
// beside it as by day; outside it (P over 15) they keep the pre-exposure's scale while the code does
// not, so beside the code they are 2^( P - 15 ) times larger than by day, and a cell where they
// were near the code's size by day can read as no kind there.
//
// The kind comes from the material and the mesh it is built for (their names and userData, set
// where they are made), never from colours. This module holds the mechanism and knows no name of
// the scene's: whoever makes the scene registers rules (`classifyBy`), each of which names a
// build's kind or passes (null), asked in the order registered; the first answer stands. A rule
// may answer 'slope' for a surface that is ground where the geometry's normal is within 40
// degrees of up and rock elsewhere. What no rule names is leaves when its material carries the
// vegetation's tag (`userData.vegetation`, vegetation/Coverage.ts) and other otherwise. The
// island's rules: world/IslandClasses.ts. The rules are asked as each material builds, so a scene
// registers them before its materials build: a material built earlier never sees a rule
// registered after it, and nothing reports that.
//
// Run from debug/LightLayer.ts's hook (the lighting model roots' constructors), it overrides the
// model's light like `albedoOnly` and leaves what a material adds after its lighting: a material's
// emission (the yunpeng's glow at night) lands on the code, and those pixels decode to none.

/** The kinds' names, by index. */
export const FIT_CLASSES = [ 'sky', 'sea', 'ground', 'rock', 'grass', 'leaves', 'wood', 'figure', 'other' ] as const;
export type FitClass = typeof FIT_CLASSES[ number ];

/** Whether the app draws the kinds' view. */
export const CLASSES = debugFlags.classes === true || debugFlags.classes === 1;

// the 40 degrees from up that split a 'slope' surface's ground from its rock
const GROUND_COS = Math.cos( 40 * Math.PI / 180 );

const code = ( kind: FitClass ) => {

	const t = ( FIT_CLASSES.indexOf( kind ) + 1 ) / 10;
	return vec3( 1, t, t * t );

};

type Reflected = { directDiffuse: THREE.Node<'vec3'>; directSpecular: THREE.Node<'vec3'>; indirectDiffuse: THREE.Node<'vec3'>; indirectSpecular: THREE.Node<'vec3'> };

/** A rule's answer for the mesh and material a build is for: its kind, 'slope' (ground on the
 * flat, rock on the steep) or null when the rule does not know it. */
export type ClassRule = ( object: THREE.Object3D | null, material: THREE.Material | null ) => FitClass | 'slope' | null;

const rules: ClassRule[] = [];
// an app registers its rules as its scene builds (island/world/IslandClasses.ts)
onSceneDispose( 'classes', () => { rules.length = 0; } );

/** Registers a rule that names builds' kinds; rules are asked in the order registered. */
export function classifyBy( rule: ClassRule ): void {

	rules.push( rule );

}

// the kind's colour for the mesh and material a build is for
function kindColour( object: THREE.Object3D | null, material: THREE.Material | null ): THREE.Node<'vec3'> {

	for ( const rule of rules ) {

		const kind = rule( object, material );
		if ( kind === 'slope' ) return select( normalWorldGeometry.y.greaterThan( GROUND_COS ), code( 'ground' ), code( 'rock' ) );
		if ( kind !== null ) return code( kind );

	}
	return code( material?.userData.vegetation ? 'leaves' : 'other' );

}

/** Overrides a lighting model so it draws its kind's colour (see the header). */
export function useClasses( model: THREE.LightingModel ): void {

	const m = model as unknown as { direct(): void; indirect( builder: THREE.NodeBuilder ): void };
	m.direct = () => {};
	m.indirect = ( builder ) => {

		const reflected = ( builder.context as unknown as { reflectedLight: Reflected } ).reflectedLight;
		const kind = kindColour( builder.object as THREE.Object3D | null, builder.material as THREE.Material | null );
		reflected.directDiffuse.assign( sceneUnits( kind.mul( min( preExposure.scale, STORE_CEILING ) ) ) );
		reflected.directSpecular.assign( vec3( 0 ) );
		reflected.indirectDiffuse.assign( vec3( 0 ) );
		reflected.indirectSpecular.assign( vec3( 0 ) );

	};

}
