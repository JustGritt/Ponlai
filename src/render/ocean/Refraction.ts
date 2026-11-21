/*!
 * Adapted from tidewater by Daniel Greenheck (https://github.com/dgreenheck/tidewater).
 * SPDX-FileCopyrightText: Copyright (c) 2026 DRG Software Solutions LLC
 * SPDX-License-Identifier: MIT
 */

import * as THREE from 'three/webgpu';
import { float, uniform } from 'three/tsl';

// The water's refraction source: the scene below the water only, drawn at half density.
//
// The water's refraction source (tidewater's RefractionPass, 4811ba4): the scene below the water
// only, drawn by the frame graph's `refraction` pass (Pipeline.ts) right after the opaque pass, with
// the same (jittered) camera through a wider view, at half the render's pixel density. The water
// samples it at the end point of its refracted view ray (Water.ts).
//
// The opaque copy cannot serve for this alone: wherever something above the water (the cliffs above
// the waterline, a figure on the rim) stands in front of the refracted end point, the floor there
// was never drawn, and the water fell back to what lies straight behind it. Here fragments higher
// than the sea level + CLIP_MARGIN are clipped, so those surfaces never enter the image, and the
// submerged ones (the floor, the stones, the cliffs' feet) are shaded with their full materials and
// lighting.
//
// What it draws: the world's group (World.ts: the island, the stones, the floor) is a ClippingGroup
// whose plane keeps what lies below that height, and it lies in REFRACTED_LAYER alone; its meshes
// and the key light lie in REFRACTED_LAYER as well as the default layer. The pass's camera sees
// REFRACTED_LAYER only: the group is visible to it, so its clipping applies, and of the scene it
// draws those meshes lit by the key light (the plants, the islanders, the sky never enter). Every
// other camera sees the default layer, to which the group itself is invisible: three applies a
// ClippingGroup's planes only where the group is visible, and it draws the children of an invisible
// group as usual, so the opaque pass and the shadow maps draw the world unclipped. A mesh that can
// reach under the sea and should show through it joins with `showUnderwater`: under the world's
// group if it also rises above the clip height, anywhere if it stays below it. A light it is lit
// by joins too (the key light does: main.ts).
//
// The target: rgb the lit colour, a its coverage (cleared to 0: where nothing below the water was
// drawn, the water falls back to the opaque copy), and its own depth. Most of the screen's floor the
// opaque image holds as well, and the water reads it from there (the underwater pyramid), so a mask
// drawn first into the depth at the nearest depth (Pipeline.ts) leaves those texels unshaded: the
// pass shades what stands behind the surfaces above the water, beside them, and the part of the
// guard band the water's lookups reach in this frame's view.
//
// The view is wider than the screen (a guard band): light bends down into the water, so a refracted
// ray lands on floor lower on the screen than the pixel it is seen through, and near the bottom edge
// below the screen. The band holds that floor, so the water never has to invent it. The extra view
// is in NDC units (1 = half the screen): GUARD_SIDE on the left and the right (tidewater widens them
// by 0.15; here 0.3, so the target's width is a render-relative size class), none above, and below
// as far as this frame's refracted ends reach (`setGuard`). How far that is depends on the camera
// alone: from the island camera's height a few per cent of the screen, but seen low over the sea,
// and above all looking up a little, a refracted ray from the bottom edge runs down to the floor
// several screen heights below it (four and more NDC units below the edge with the camera half a
// metre over the sea, pitched up 20 degrees). A fixed band either misses those ends, which then
// read the band's last row stretched down the screen as stripes with a hard line where the band
// ends, or spends its texels in every view on a reach only a few need. So the band below is
// measured every frame on the CPU, from the camera: the bottom half of the screen's rays to the
// level sea, refracted through the flat surface and tilted by REACH_SLOPE (as the reach pass and the
// water bound them: kept heading down, at most MAX_PATH metres), their ends projected; the farthest
// below the edge, with a margin, between GUARD_BELOW and GUARD_BELOW_MAX. The target keeps its size,
// so a deeper band lowers the image's density down the screen (by a fifth at a band of one, a half
// at three): only the views that look past the edge pay for it. The mask (Pipeline.ts) still leaves
// unshaded what lies past the lookups' measured reach, so a deeper band does not shade more.
// `guard` is the clip-space scale ( sx, sy ) and centre ( cx, cy ) as a uniform:
// ndc_refraction = ( ndc_screen - c ) / s, as the water and the mask map their lookups.

/** Fragments above this height over the sea level are left out (m): the troughs and crests move
 * the real surface around it. */
export const CLIP_MARGIN = 0.4;

/** The layer of what the refraction's camera draws. */
export const REFRACTED_LAYER = 7;

/** The band beside the screen, left and right (NDC units). */
const GUARD_SIDE = 0.3;
/** The band below the screen at its least, and at its most (NDC units): the most covers a camera half a
 * metre over the sea, pitched up 30 degrees, under a vertical view of 65 degrees. */
export const GUARD_BELOW = 0.6, GUARD_BELOW_MAX = 6;
/** The band's horizontal scale, fixed: the target's width follows it. */
const SCALE_X = 1 + GUARD_SIDE;
/** The bound of the band on each side (NDC units), for what must cover all of it (the clusters of the
 * local lights, LocalLightClusters.ts). */
export const REFRACTION_GUARD_BOUND = { left: GUARD_SIDE, right: GUARD_SIDE, bottom: GUARD_BELOW_MAX, top: 0 } as const;

/** The target's size: half the render's pixel density across the wider view (and down it at the
 * least band). */
export const REFRACTION_SIZE = { of: 'render', scale: 0.5 * SCALE_X, min: 1 } as const;

// The band's measure (setGuard): the surface's tilt besides the flat sea, as the reach pass follows it
// (Pipeline.ts REACH_SLOPE), the water's bounds on a refracted ray (Water.ts: kept heading down, at
// most MAX_PATH metres to its lookup), the margin past the farthest end, and the screen's points
// sampled (a grid over its bottom half: the ends that leave it lie under it).
const TILT = 0.3, DOWN = - 0.08, MAX_PATH = 80, MARGIN = 0.05, IOR = 1.333;
const SAMPLES: [ number, number ][] = [];
for ( let j = 0; j <= 4; j ++ ) for ( let i = 0; i <= 8; i ++ ) SAMPLES.push( [ i / 4 - 1, - 1 + j / 8 ] );

const _dir = new THREE.Vector3(), _eye = new THREE.Vector3(), _surface = new THREE.Vector3(), _n = new THREE.Vector3();
const _along = new THREE.Vector3(), _across = new THREE.Vector3(), _t = new THREE.Vector3(), _end = new THREE.Vector3();
const _clip = new THREE.Vector4();

/** The refraction pass's camera: the scene camera, as it draws this frame, through the guard band. */
export class RefractionView {

	readonly camera = new THREE.PerspectiveCamera();
	/** The band's clip-space ( sx, sy, cx, cy ) this frame (setGuard). */
	readonly guard = uniform( new THREE.Vector4( SCALE_X, 1 + GUARD_BELOW / 2, 0, - GUARD_BELOW / 2 ) );
	private readonly guardMatrix = new THREE.Matrix4();

	constructor() {

		this.camera.layers.set( REFRACTED_LAYER );
		this.camera.matrixAutoUpdate = false;
		this.camera.matrixWorldAutoUpdate = false;
		this.setBelow( GUARD_BELOW );

	}

	/** At the frame's start, before anything reads the band: the band below the screen that this
	 * frame's refracted ends reach (see the header), from the camera without its jitter. */
	setGuard( camera: THREE.Camera, seaLevel: number ): void {

		_eye.setFromMatrixPosition( camera.matrixWorld );
		let below = GUARD_BELOW;
		if ( _eye.y > seaLevel ) {

			const projection = camera.projectionMatrix, view = camera.matrixWorldInverse;
			for ( const [ x, y ] of SAMPLES ) {

				_dir.set( x, y, 0.5 ).applyMatrix4( camera.projectionMatrixInverse ).applyMatrix4( camera.matrixWorld ).sub( _eye ).normalize();
				if ( _dir.y > - 1e-3 ) continue;
				_surface.copy( _eye ).addScaledVector( _dir, ( _eye.y - seaLevel ) / - _dir.y );
				_along.set( _dir.x + 1e-5, 0, _dir.z ).normalize();
				_across.set( - _along.z, 0, _along.x );
				for ( const [ a, b ] of [ [ 0, 0 ], [ 1, 0 ], [ - 1, 0 ], [ 0, 1 ], [ 0, - 1 ] ] ) {

					_n.set( 0, 1, 0 ).addScaledVector( _across, a * TILT ).addScaledVector( _along, b * TILT ).normalize();
					if ( ! refract( _dir, _n, 1 / IOR, _t ) ) continue;
					_t.y = Math.min( _t.y, DOWN );
					_t.normalize();
					_end.copy( _surface ).addScaledVector( _t, MAX_PATH );
					_clip.set( _end.x, _end.y, _end.z, 1 ).applyMatrix4( view ).applyMatrix4( projection );
					// (an end behind the eye cannot be on the image: the band then takes its most)
					below = Math.max( below, _clip.w > 1e-3 ? - 1 - _clip.y / _clip.w + MARGIN : GUARD_BELOW_MAX );

				}

			}

		}
		this.setBelow( Math.min( below, GUARD_BELOW_MAX ) );

	}

	private setBelow( below: number ): void {

		this.guard.value.set( SCALE_X, 1 + below / 2, 0, - below / 2 );
		const g = this.guard.value;
		this.guardMatrix.set(
			1 / g.x, 0, 0, - g.z / g.x,
			0, 1 / g.y, 0, - g.w / g.y,
			0, 0, 1, 0,
			0, 0, 0, 1,
		);

	}

	/** Before the pass: the scene camera's place and projection (with this frame's jitter, so the
	 * temporal resolve sees the floor as in the scene). */
	update( camera: THREE.Camera ): void {

		const c = this.camera, scene = camera as THREE.PerspectiveCamera;
		c.matrixWorld.copy( camera.matrixWorld );
		c.matrixWorldInverse.copy( camera.matrixWorldInverse );
		c.projectionMatrix.multiplyMatrices( this.guardMatrix, camera.projectionMatrix );
		c.projectionMatrixInverse.copy( c.projectionMatrix ).invert();
		c.near = scene.near;
		c.far = scene.far;

	}

}

// Snell's refraction of the unit direction `d` through the unit normal `n` at the ratio `eta`, into
// `out`; false under total internal reflection (GLSL's refract).
function refract( d: THREE.Vector3, n: THREE.Vector3, eta: number, out: THREE.Vector3 ): boolean {

	const c = d.dot( n ), k = 1 - eta * eta * ( 1 - c * c );
	if ( k < 0 ) return false;
	out.copy( d ).multiplyScalar( eta ).addScaledVector( n, - ( eta * c + Math.sqrt( k ) ) );
	return true;

}

/** A ClippingGroup that keeps what lies below the sea level + CLIP_MARGIN, for the refraction's
 * camera only (see the header). */
export function refractedGroup( seaLevel: number ): THREE.ClippingGroup {

	const group = new THREE.ClippingGroup();
	group.clippingPlanes = [ new THREE.Plane( new THREE.Vector3( 0, - 1, 0 ), seaLevel + CLIP_MARGIN ) ];
	group.layers.set( REFRACTED_LAYER );
	return group;

}

/** Marks `material` as something that swims in the water (the wenyao), which the refraction
 * image leaves out: it writes an alpha of 0 into the opaque image (three assigns opaque materials
 * an alpha of 1 under normal blending, so the mark takes none), the underwater pyramid leaves it
 * out (Pipeline.ts), and the water finds it along its refracted rays (Water.ts). A swimmer is not
 * shown to the refraction's camera: there it would stand in front of the floor that the rays
 * passing beside it reach, which the water then could not read anywhere. */
export function markSwimmer( material: THREE.NodeMaterial ): void {

	material.blending = THREE.NoBlending;
	material.opacityNode = float( 0 );

}

/** Shows `object` (a mesh that reaches under the sea, or a light) to the refraction's camera. */
export function showUnderwater( object: THREE.Object3D ): void {

	object.layers.enable( REFRACTED_LAYER );

}
