import type * as THREE from 'three/webgpu';
import { checkDeclarations, type Declarations } from '../Extensions.ts';

// The vegetation's registry: every plant kind the app draws, once, with what it takes part in.
// The lists of vegetation roots read it (RenderStages.ts `placeVegetation`: the scene, the depth
// prepasses; `vegetationCasters`: the sun's caster box; `bakeRainCollision`: the rain's collision
// bake; Growth.ts: the kinds' growth and the rain's bake after a swap) instead of naming the kinds
// at each place.
// - Closed once placed: `placeVegetation` reads the kinds once and then closes the registry
//   (`close`), and an `add` after it throws, naming the kind and what had already read the kinds: a
//   kind registered late would be in no scene, prepass, shadow box or rain bake, and nothing else
//   would say so. Every kind is registered before that call. The growth in place adds none: it
//   reads the registered kinds and their roots.
// - Order: the kinds are registered in the order the scene draws them (today the grass, then the
//   yunpeng, then the bamboo's culms and leaves), and
//   every list keeps it: draws of equal render order and depth keep their order.
// - The depth prepass (`prepass`, VegetationPrepass.ts) and the shadow filter (`shadowFilter`,
//   SunShadow.ts) are separate choices. The shadow filter is still read from the materials'
//   vegetation tag (`userData.vegetation`, vegetation/Coverage.ts markVegetation), which the depth
//   prepass under the temporal modes, the rain's bake and the wind probe read too; `add` checks that
//   a kind's materials carry the tag its `shadowFilter` names.
// - Growth (`growth`): a kind that changes with the land prepares its next version in the growth's
//   idle-time steps and swaps it in, in the swap's frame, without building a pipeline (Growth.ts).
//   The kinds prepare and swap in registry order. A kind that writes its buffers in place at the
//   swap stages the write ahead (`stage`, Upload.ts stagedWriteSteps): kept for the plant kinds to
//   come; no kind uses it today (the grass takes new buffers). The next version's land data is
//   whatever the app's world is (`L`: the island's world/types.ts WorldData); the registry only
//   passes it on.
// - The three declarations (Extensions.ts): a kind states the quality setting that governs its cost,
//   its test-mode behaviour and how it follows the land's growth, or `add` refuses it. A
//   kind with a `growth` of its own names it as its follower; one that another kind's growth
//   carries (the bamboo's leaves, swapped with its culms) names that kind.
// - The default: a kind without art of its own gives no `root` and says where it stands (`sites`);
//   it is drawn with the registry's default plant (vegetation/DefaultPlant.ts: crossed leaf cards
//   on the default vegetation material, which the registry's maker hands in), so it can be placed,
//   lit, shadowed and measured before its art exists.
// The registry reads its kinds in the order registered and never by name
// (tools/render/verify-extensions.mjs registers a kind of its own in node and searches this file, the
// registry's owner, for the kinds' names; it does not search the files that read the registry).

/** A preparation step yields nothing (a slice ends) or a promise (wait for it: a worker, a compile). */
export type GrowthSteps<T> = Generator<void | Promise<unknown>, T>;

/** What a kind's growth preparation reads (Growth.ts); `L` is the world's data. */
export interface GrowthInput<L> {
	renderer: THREE.WebGPURenderer;
	/** the next version's world */
	data: L;
	/** metres from a point to the scene's path on the next land (for an island, its walking loop) */
	pathDistance( x: number, z: number ): number;
	/** keeps a staged in-place write (Upload.ts stagedWriteSteps) until the swap's frames are done */
	stage( free: ( () => void ) | null ): void;
}

/** A kind's part in the island's growth (Growth.ts). */
export interface KindGrowth<P, L> {
	/** the kind's next version, in steps of a few tenths of a millisecond, its buffers uploaded */
	prepare( input: GrowthInput<L> ): GrowthSteps<P>;
	/** swaps `prepared` in, in the swap's frame; returns what to dispose of once nothing draws the
	 * old version (a frame or more later), or null */
	swap( prepared: P ): ( () => void ) | null;
}

/** Where a plant stands (world space, m). */
export interface PlantSite { x: number; y: number; z: number }

export interface VegetationKind<P = unknown, L = unknown> extends Declarations {
	/** 'grass', 'yunpeng', 'bamboo', ... */
	name: string;
	/** what the app adds to the scene; without it the kind is drawn with the default plant at its `sites` */
	root?: THREE.Object3D;
	/** where the default plant stands, for a kind without a `root` */
	sites?: readonly PlantSite[];
	/** drawn after a depth-only copy (VegetationPrepass.ts) */
	prepass: boolean;
	/** the sun's shadow filter its materials take (SunShadow.ts) */
	shadowFilter: 'vegetation' | 'ordinary';
	/** inside the sun's caster box (RenderStages.ts `vegetationCasters`) */
	sunCaster: boolean;
	/** baked into the rain's collision map */
	rainBake: boolean;
	growth?: KindGrowth<P, L>;
}

/** A kind as the registry holds it: its root made, its own or the default. */
export type RegisteredKind<P = unknown, L = unknown> = VegetationKind<P, L> & { root: THREE.Object3D };

export class Vegetation<L = unknown> {

	readonly kinds: RegisteredKind<unknown, L>[] = [];
	/** why the registry takes no more kinds, once it is closed */
	private closed: string | null = null;

	/** `defaultRoot` draws a kind without art of its own (DefaultPlant.ts `defaultPlant`, with the
	 * scene's uniforms, which the registry's maker has). */
	constructor( private readonly defaultRoot?: ( sites: readonly PlantSite[], name: string ) => THREE.Object3D ) {}

	add<P>( definition: VegetationKind<P, L> ): void {

		if ( this.closed !== null ) throw new Error( `Vegetation: '${ definition.name }' is registered too late: ${ this.closed }, so its root would be in no scene, prepass, shadow box or rain bake. Register every kind before that call` );
		checkDeclarations( 'Vegetation', definition.name, definition );
		if ( definition.growth && definition.land === 'independent' ) throw new Error( `Vegetation: '${ definition.name }' has a growth and declares that it does not depend on the land` );
		let root = definition.root;
		if ( ! root ) {

			if ( ! this.defaultRoot ) throw new Error( `Vegetation: '${ definition.name }' has no root, and this registry was given no default plant` );
			root = this.defaultRoot( definition.sites ?? [], definition.name );

		}
		const kind = { ...definition, root } as RegisteredKind<P, L>;
		const tagged = kind.shadowFilter === 'vegetation';
		kind.root.traverse( ( object ) => {

			const mesh = object as THREE.Mesh;
			if ( ! mesh.isMesh || mesh.userData.prepassCopy ) return;
			for ( const material of Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ] ) {

				if ( ( material.userData.vegetation === true ) !== tagged ) throw new Error( `Vegetation: ${ kind.name }'s material ${ material.name || material.type } is ${ tagged ? 'not ' : '' }tagged as vegetation, against its shadow filter '${ kind.shadowFilter }'` );

			}

		} );
		this.kinds.push( kind as RegisteredKind<unknown, L> );

	}

	/** Closes the registry: `add` throws from here on, giving `reason` (what has read the kinds and
	 * will not read them again). The stage that places the kinds calls it (RenderStages.ts
	 * `placeVegetation`). */
	close( reason: string ): void {

		this.closed ??= reason;

	}

	/** The kinds' roots in registry order, those `filter` keeps. */
	roots( filter: ( kind: RegisteredKind<unknown, L> ) => boolean = () => true ): THREE.Object3D[] {

		return this.kinds.filter( filter ).map( ( kind ) => kind.root );

	}

}
