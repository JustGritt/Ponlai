import * as THREE from 'three/webgpu';
import { Fn, positionViewDirection, normalView, normalViewGeometry, negateOnBackSide, diffuseColor, diffuseContribution, specularColor, specularF90, F_Schlick, abs, clamp, dot, exp, float, luminance, max, mix, smoothstep, uniform, vec2, vec3 } from 'three/tsl';
import { GroundBounceLightingModel, diffuseFresnel } from '../lighting/GroundBounce.ts';
import { translucentWrap, type AmbientReceiver } from '../lighting/AmbientSample.ts';
import { meterAmbient } from '../lighting/MeterLight.ts';
import { tuneApply } from '../debug/Tune.ts';
import { OPAQUE_VEGETATION } from '../debug/LightLayer.ts';
import { leafBack, leafTint, leafWarm } from './LeafLight.ts';
import { shared } from '../shared.ts';

type V3 = THREE.Node<'vec3'>;
type F = THREE.Node<'float'>;

// The vegetation's lighting: physical lighting plus light through thin leaves and blades.
//
// Physical lighting plus light through thin leaves and blades: `extra( L, V, N )` (view space)
// scales an unshadowed-by-orientation diffuse term added on top of the usual one (a vec3 tints it,
// as light through a blade turns yellower than the light off it), and `scale` is the occlusion
// from root to tip (deep inside a bush, down at the roots of a tuft): it dims the ambient terms
// alone (the ambient lights and the ground's bounce in the irradiance, the environment's diffuse
// and specular), not the key, whose darkening down there is its shadows', nor the local lights.
// `translucency` t (0..1, a number or a node per pixel: the meadow's blades from 0.6 at the root
// to 0.9 at the tip, world/Grass.ts; the bamboo's leaves 0.5) is how much the surface lets light
// through. A blade (`wrap`) also wraps the key's diffuse round to its far side: three's Lambert
// term, the clamped cosine, moves by smoothstep( 0, 0.55, t ) (lighting/AmbientSample.ts
// `translucentWrap`, which the ambient's band weights share) toward the squared half-Lambert
// ( ( N.L + 1 ) / 2 )^2 scaled by 3/4, under the same Fresnel weight the Lambert takes
// (1 - lum( F ): lighting/GroundBounce.ts), so every blade, down to its least translucent root,
// wraps whole, and a less translucent kind would ease into the wrap rather than turn at one t.
// A leaf does not wrap: its t goes only into the light through it. Over the sphere of normals the
// clamped cosine gathers pi,
// the squared half-Lambert 4 pi / 3, so the 3/4 keeps the light a surface of every orientation
// gathers on average: a lit tuft reads as one soft surface with little contrast between its
// blades, and gains no light for it (a horizontal blade under a low sun even takes a little less).
// The wrap and the light through to the far side (`extra`) take the key's shading
// direction (GroundBounceLightingModel `shadeDirection`: lifted under a low key, lighting/KeyLift.ts),
// as the physical direct term does, so a plant's lit side, back light and shadows face one way.
// Light through to the far side, and for the most translucent forward into a view toward the
// sun, is the kind's `extra` (vegetation/Material.ts `transmission`). The whole diffuse of one
// light (the wrapped or plain Lambert under its Fresnel weight, the light through, the thin layer)
// is `diffuseResponse`, laid-out shader functions the key's direct term and every local light
// (lighting/LocalLights.ts, along its own direction, without a shadow) call alike, so a clump's
// light wraps round the blades and glows through the leaves on its far side as the key does.
// A leaf (the leaf family, translucent) is also a thin layer (`THIN_LAYER`): single scattering in
// a layer (Hanrahan and Krueger, *Reflection from Layered Surfaces due to Subsurface Scattering*,
// SIGGRAPH 1993), on the face toward the viewer, by the light's wrapped cosine and the view's
// against the geometric normal, round the half vector in a lobe (flat by default), times a gain:
// it takes the Lambert's place by smoothstep( 0, 0.5, t ) (`THIN_SHARE`), not added to it (a leaf
// of t 0.5, the bamboo's, or more is all layer), so a sunlit leaf sends back about two thirds of what a
// Lambertian surface of its albedo would (`THIN_LAYER`, its gain). `extra` is a blade's: a
// leaf's light through it is one term on the same share, the light on its far face by its cosine
// times a share keyed by the sun and the weather (vegetation/LeafLight.ts `back`, 0 by night and
// under a closed sky), so a leaf lit from behind glows on the same albedo and a crown has no black
// side toward the sun; and under the ambient light its ambient diffuse is
// mix( E( n ), E_HK, t ): the plain irradiance at its normal, unweighted, blended toward the
// layer's, E_HK = E01( 3/4 n ) G k_sky F( mu_v ) (lighting/AmbientSample.ts `thin`: the constant
// and the linear band at three quarters of the geometric normal; F the layer's response under a
// uniform sky), so the more a leaf lets through, the less its ambient follows its facing; under
// the occlusion and `scale` as before. (The half-Lambert's shape: Mitchell, *Shading in Valve's
// Source Engine*, SIGGRAPH 2006, slide 25. A translucency kept energy-conserving beside the
// Lambertian diffuse: Jasmin Patry, *Samurai Shading in Ghost of Tsushima*, SIGGRAPH 2020 Physically
// Based Shading course, p. 9.)
// The ground's bounce joins the sky's light, and the lighting layer's debug views apply, as for
// every model built on GroundBounceLightingModel (lighting/GroundBounce.ts); the ambient diffuse
// is the ambient volume's, at degree 2 by default (`ambient`, lighting/AmbientSample.ts, as the
// island's other receivers). A blade reads it weighted by the same t (`translucency` there),
// unless its receiver names a t of its own: its ambient flattens and brightens as it lets light
// through. A leaf's ambient is not weighted: its translucency goes into the thin layer's blend
// alone. The metering illuminance (lighting/MeterLight.ts) takes each light by the cosine the
// diffuse takes (the wrapped Lambert and the light through, or a leaf's blend of the Lambert and
// its thin layer) and the ambient by the
// thin layer's blend and `scale`, as the colour does. A blade's ambient diffuse takes the sward's
// tint under a closed sky (materials/SwardTint.ts) and a leaf's the leaves' (vegetation/LeafLight.ts
// `leafTint`), both of which the metering leaves out.

/** The squared half-Lambert's scale that gives it the clamped cosine's integral over the sphere
 * (pi against 4 pi / 3). */
const WRAP_NORMALISATION = 3 / 4;

/** The translucency at which a leaf is all thin layer (`diffuseResponse`): the least a leaf kind
 * has (the bamboo's and the falling leaves' 0.5), so every leaf drawn is wholly the layer and the
 * Lambert only eases out over a kind less translucent than those. */
const THIN_SHARE = 0.5;

/** A blade's diffuse cosine and its Fresnel weight, ( mix( saturate( N.L ), 3/4 ( ( N.L + 1 ) / 2 )^2,
 * s( t ) ), 1 - lum( F( V.H ) ) ), s the wrap share (`translucentWrap`) (view space; lighting/GroundBounce.ts `diffuseFresnel`
 * with the wrap): one shader function for the key and every local light. */
const wrappedDiffuseFresnel = /*@__PURE__*/ Fn( ( [ N, V, L, t, f0, f90 ]: [ V3, V3, V3, F, V3, F ] ) => {

	const NL = dot( N, L ).toVar();
	const half = NL.mul( 0.5 ).add( 0.5 );
	const H = L.add( V ).normalize();
	const F = luminance( F_Schlick( { f0, f90, dotVH: clamp( dot( V, H ), 0, 1 ) } ) as unknown as V3 );
	return vec2( mix( clamp( NL, 0, 1 ), half.mul( half ).mul( WRAP_NORMALISATION ), translucentWrap( t ) ), float( 1 ).sub( F ) );

} ).setLayout( { name: 'wrappedDiffuseFresnel', type: 'vec2', inputs: [
	{ name: 'N', type: 'vec3' }, { name: 'V', type: 'vec3' }, { name: 'L', type: 'vec3' }, { name: 't', type: 'float' },
	{ name: 'f0', type: 'vec3' }, { name: 'f90', type: 'float' },
] } );

/** A leaf's thin layer (Hanrahan and Krueger's single scattering in a layer): its optical depth
 * `tau`, its gain, the wrap of its light's cosine (`wrap`: a leaf turned edge-on to the key does
 * not cut off at zero), its lobe's roughness (`lobe`, a below), the ambient's own gain (`sky`), the
 * floor of the view's cosine (`floor`: no blow-up at grazing views). The lobe round the half vector
 * is a very rough microfacet distribution, ( 1 - a ( 1 - c^2 ) )^-2 with c = |H.n|: 1 along the
 * normal, 1 / ( 1 - a )^2 with the half vector in the leaf's plane. All tunable
 * (`lighting.thinLayer.*`).
 * Leaf BRDF models fitted to measured leaves take what is left once the specular sheen off the
 * cuticle is removed as Lambertian: independent of the light's and the view's directions (Bousquet,
 * Lacherade, Jacquemoud and Moya, *Leaf BRDF measurements and model for specular and diffuse
 * components differentiation*, Remote Sensing of Environment, 2005; Wang, Wang, Dorsey, Yang, Guo
 * and Shum, *Real-time rendering of plant leaves*, SIGGRAPH 2005), measured on seven species,
 * fescue grass and corn among them, at incidences of 5, 25, 45 and 65 degrees (Bousquet, Jacquemoud
 * and Moya, *Leaf BRDF and BTDF measurements and model*, ISPMSRS 2005). So `tau`, `gain` and `lobe`
 * are the least-squares fit of gain S lobe to the cosine of incidence, on the relative error, over
 * that geometry (those incidences, views to 75 degrees every 7.5, every 30 degrees round: 484
 * directions). The fit runs to an optically thick layer (its error falls from 0.231 at
 * tau 2 to 0.229 in the thick limit, the gain stays 1.52 to 1.53, rounded here to 1.5, and the lobe
 * goes to 0, its bound, at every tau); tau 2.3 is where the layer is thick, its direct light through it exp( -2 tau ) at
 * most 1 % along the shortest path. The fit's rms relative error is 0.23, and a sunlit leaf's
 * directional-hemispherical reflectance is 0.92 to 0.99 of a Lambertian surface's at incidences of
 * 0 to 65 degrees (0.88 at 80). The gain stands at 1, not the fit's 1.5: a sunlit crown at 1.5
 * stood about a third brighter than the meadow beside it and read white on screen, its leaves in a
 * dense crown lit on both faces and overlapping, which a single leaf's fit does not see. */
const THIN_LAYER = { tau: 2.3, gain: 1, wrap: 0.1, lobe: 0, sky: 1, floor: 0.05 };

/** The layer's response under a uniform sky, F( mu_v ), for this tau, lobe and wrap: the direct
 * term's S times the lobe integrated numerically over the sphere of light directions (a 64 x 128
 * grid), over pi (the light E01( 3/4 n ) a uniform sky of radiance 1 gives, the constant once:
 * AmbientSample.ts), at 24 views from 0.05 to 1, fitted by least squares on the relative error with
 * F = c0 + c1 s + c2 mu_v + c3 mu_v s + c4 mu_v^2, s = sqrt( mu_v ): F falls steeply toward grazing
 * views (about 1.5 at 0.05 to 0.47 at 1 for tau 2.3 and a flat lobe), which a quadratic in mu_v
 * misses by a sixth; this form stays within 0.02 % of the integral there, 0.2 % at tau 0.5 and
 * 1.4 % over tau 0.25 to 1. So
 * E01( 3/4 n ) G F( mu_v ) under a uniform sky is G times the direct term over the sky. */
function thinLayerFit( tau: number, a: number, wrap: number ): number[] {

	const NT = 64, NP = 128, NV = 24, N = 5;
	const A = Array.from( { length: N }, () => new Array( N ).fill( 0 ) ), b = new Array( N ).fill( 0 );
	for ( let k = 0; k < NV; k ++ ) {

		const m = 0.05 + 0.95 * k / ( NV - 1 ), vx = Math.sqrt( 1 - m * m ), muV = Math.max( m, THIN_LAYER.floor );
		let y = 0;
		for ( let i = 0; i < NT; i ++ ) {

			const th = ( i + 0.5 ) / NT * Math.PI, st = Math.sin( th ), lz = Math.cos( th );
			const c = Math.min( Math.max( ( lz + wrap ) / ( 1 + wrap ), 0 ), 1 ), muL = c * c, sum = muL + muV;
			if ( muL === 0 ) continue;
			const S = muL * ( 1 - Math.exp( - tau * sum / Math.max( muL * muV, 1e-6 ) ) ) / sum;
			for ( let j = 0; j < NP; j ++ ) {

				const ph = ( j + 0.5 ) / NP * 2 * Math.PI;
				const hx = st * Math.cos( ph ) + vx, hy = st * Math.sin( ph ), hz = lz + m, h = Math.hypot( hx, hy, hz );
				const hn = h > 0 ? Math.min( Math.abs( hz / h ), 1 ) : 0;
				const l = 1 - a * ( 1 - hn * hn );
				y += S / ( l * l ) * st;

			}

		}

		y *= ( Math.PI / NT ) * ( 2 * Math.PI / NP ) / Math.PI;
		const sq = Math.sqrt( m ), p = [ 1, sq, m, m * sq, m * m ], w = 1 / ( y * y );
		for ( let r = 0; r < N; r ++ ) {

			b[ r ] += w * p[ r ] * y;
			for ( let c = 0; c < N; c ++ ) A[ r ][ c ] += w * p[ r ] * p[ c ];

		}

	}

	// Gauss-Jordan with partial pivoting
	const M = A.map( ( row, r ) => [ ...row, b[ r ] ] );
	for ( let i = 0; i < N; i ++ ) {

		let p = i;
		for ( let r = i + 1; r < N; r ++ ) if ( Math.abs( M[ r ][ i ] ) > Math.abs( M[ p ][ i ] ) ) p = r;
		[ M[ i ], M[ p ] ] = [ M[ p ], M[ i ] ];
		for ( let r = 0; r < N; r ++ ) {

			if ( r === i ) continue;
			const f = M[ r ][ i ] / M[ i ][ i ];
			for ( let c = i; c <= N; c ++ ) M[ r ][ c ] -= f * M[ i ][ c ];

		}

	}

	return M.map( ( row, i ) => row[ N ] / row[ i ] );

}

// (passed to the shader functions as arguments: a laid-out function that read them itself would be
// emitted once, with the first material's uniform names, for every material that calls it; the
// wrap's scale held as its reciprocal)
const thin = {
	negTau: uniform( - THIN_LAYER.tau ), gain: uniform( THIN_LAYER.gain ), wrap: uniform( THIN_LAYER.wrap ), perWrapScale: uniform( 1 / ( 1 + THIN_LAYER.wrap ) ),
	lobe: uniform( THIN_LAYER.lobe ), sky: uniform( THIN_LAYER.sky ),
	fit: thinLayerFit( THIN_LAYER.tau, THIN_LAYER.lobe, THIN_LAYER.wrap ).map( ( c ) => uniform( c ) ),
};
const shape = { tau: THIN_LAYER.tau, lobe: THIN_LAYER.lobe, wrap: THIN_LAYER.wrap };
const refitThinLayer = () => thinLayerFit( shape.tau, shape.lobe, shape.wrap ).forEach( ( c, i ) => { thin.fit[ i ].value = c; } );
tuneApply( 'lighting.thinLayer.tau', THIN_LAYER.tau, { min: 0.1, max: 8, scale: 'log' }, ( tau ) => { if ( tau === shape.tau ) return; shape.tau = tau; thin.negTau.value = - tau; refitThinLayer(); } );
tuneApply( 'lighting.thinLayer.gain', THIN_LAYER.gain, { min: 0.25, max: 5, scale: 'log' }, ( g ) => { thin.gain.value = g; } );
tuneApply( 'lighting.thinLayer.wrap', THIN_LAYER.wrap, { min: 0, max: 0.5 }, ( w ) => { if ( w === shape.wrap ) return; shape.wrap = w; thin.wrap.value = w; thin.perWrapScale.value = 1 / ( 1 + w ); refitThinLayer(); } );
tuneApply( 'lighting.thinLayer.lobe', THIN_LAYER.lobe, { min: 0, max: 0.5 }, ( a ) => { if ( a === shape.lobe ) return; shape.lobe = a; thin.lobe.value = a; refitThinLayer(); } );
tuneApply( 'lighting.thinLayer.sky', THIN_LAYER.sky, { min: 0.25, max: 4, scale: 'log' }, ( k ) => { thin.sky.value = k; } );
// the share of t the ambient's blend takes (1; 0 reads the leaf's ambient without its thin layer)
const thinAmbient = uniform( 1 );
tuneApply( 'lighting.thinLayer.ambient', 1, { min: 0, max: 1 }, ( a ) => { thinAmbient.value = a; } );

/** A thin layer's single scattering toward the viewer for the cosines N.L (signed), |V.n| and
 * |H.n| (n the geometric normal of the face toward the viewer): the light's wrapped cosine squared
 * mu_l, S = mu_l ( 1 - exp( -tau ( mu_l + mu_v ) / ( mu_l mu_v ) ) ) / ( mu_l + mu_v ), times the
 * lobe ( 1 - a ( 1 - |H.n|^2 ) )^-2, times the gain. */
const thinLayerScatter = /*@__PURE__*/ Fn( ( [ NL, VN, HN, wrap, perWrapScale, negTau, lobe, gain ]: [ F, F, F, F, F, F, F, F ] ) => {

	const c = clamp( NL.add( wrap ).mul( perWrapScale ), 0, 1 );
	const muL = c.mul( c ).toVar();
	const muV = max( VN, THIN_LAYER.floor ).toVar();
	const sum = muL.add( muV ).toVar();
	const S = muL.mul( float( 1 ).sub( exp( sum.mul( negTau ).div( max( muL.mul( muV ), 1e-6 ) ) ) ) ).div( sum );
	const h = clamp( HN, 0, 1 );
	const l = float( 1 ).sub( lobe.mul( float( 1 ).sub( h.mul( h ) ) ) ).toVar();
	return S.mul( gain ).div( l.mul( l ) );

} ).setLayout( { name: 'thinLayerScatter', type: 'float', inputs: [
	{ name: 'NL', type: 'float' }, { name: 'VN', type: 'float' }, { name: 'HN', type: 'float' },
	{ name: 'wrap', type: 'float' }, { name: 'perWrapScale', type: 'float' }, { name: 'negTau', type: 'float' },
	{ name: 'lobe', type: 'float' }, { name: 'gain', type: 'float' },
] } );

/** A thin layer's response under a uniform sky, times its gain and the ambient's own (`sky`), at
 * the view's cosine |V.n|. */
const thinLayerSky = ( VN: F ): F => {

	const m = max( VN, THIN_LAYER.floor ).toVar(), s = m.sqrt();
	const [ c0, c1, c2, c3, c4 ] = thin.fit;
	return m.mul( c4 ).add( s.mul( c3 ) ).add( c2 ).mul( m ).add( s.mul( c1 ) ).add( c0 ).mul( thin.gain ).mul( thin.sky );

};

/** The geometric normal of the face toward the viewer (view space). */
const facingGeometry = (): V3 => negateOnBackSide( normalViewGeometry ) as unknown as V3;

export class VegetationLighting extends GroundBounceLightingModel {

	scale: F;
	translucency: F | number;
	wrap: boolean;
	thin: boolean;
	extra: ( L: V3, V: V3, N: V3 ) => F | V3;

	constructor( scale: F, translucency: F | number, family: 'leaf' | 'blade' | 'sod', extra: ( L: V3, V: V3, N: V3 ) => F | V3, ambient: AmbientReceiver | null = { bands: 2, kind: 'leaves' } ) {

		// the debug view of the vegetation as opaque surfaces (debug/LightLayer.ts): no light
		// through, no wrap, no translucent ambient
		if ( OPAQUE_VEGETATION ) {

			translucency = 0;
			extra = () => float( 0 );

		}
		// a blade's ambient weighted by its translucency, or the receiver's own; a leaf's thin
		// layer's light, its ambient unweighted (lighting/AmbientSample.ts)
		const translucent = typeof translucency !== 'number' || translucency > 0;
		const thin = family === 'leaf' && translucent;
		const weighted = family === 'blade' && translucent;
		super( ambient && ( weighted || thin ) ? { ...( weighted ? { translucency } : { thin } ), ...ambient } : ambient );
		this.scale = scale;
		this.translucency = translucency;
		this.wrap = family === 'blade';
		this.thin = thin;
		this.extra = extra;
		// the environment's diffuse and specular (the ambient grade's, lighting/AmbientGrade.ts) by
		// `scale`
		const graded = this.indirectSpecular.bind( this );
		this.indirectSpecular = ( builder ) => {

			const r = ( builder.context as unknown as { reflectedLight: { indirectDiffuse: V3; indirectSpecular: V3 } } ).reflectedLight;
			const diffuse = r.indirectDiffuse, specular = r.indirectSpecular;
			r.indirectDiffuse = vec3( 0 ).toVar();
			r.indirectSpecular = vec3( 0 ).toVar();
			// the metering illuminance's ambient (lighting/MeterLight.ts) takes the same scale and
			// thin layer as the diffuse
			const metered = meterAmbient.toVar();
			graded( builder );
			// a leaf's thin layer under the ambient light: E_HK = E01( 3/4 n ) times its response
			// under a uniform sky, on the albedo as the Lambert term (lighting/AmbientSample.ts
			// `thin`), the ambient diffuse blended toward it by t, and the metering's share alike
			const thinLight = ( builder.context as unknown as { thinIrradiance?: V3 } ).thinIrradiance;
			if ( this.thin && thinLight ) {

				const VN = abs( dot( positionViewDirection, facingGeometry() ) );
				const thinE = thinLight.mul( thinLayerSky( VN ) ).toVar();
				const t = this.t().mul( thinAmbient ).toVar();
				r.indirectDiffuse.assign( mix( r.indirectDiffuse, thinE.mul( diffuseColor.rgb ).mul( 1 / Math.PI ), t ) );
				meterAmbient.assign( metered.add( mix( meterAmbient.sub( metered ), thinE, t ) ) );

			}
			meterAmbient.assign( metered.add( meterAmbient.sub( metered ).mul( this.scale ) ) );
			if ( family === 'blade' ) r.indirectDiffuse.mulAssign( shared.swardTint );
			else if ( family === 'leaf' ) r.indirectDiffuse.mulAssign( leafTint );
			diffuse.addAssign( r.indirectDiffuse.mul( this.scale ) );
			specular.addAssign( r.indirectSpecular.mul( this.scale ) );
			r.indirectDiffuse = diffuse;
			r.indirectSpecular = specular;

		};

	}

	/** The translucency as a node. */
	t(): F {

		return typeof this.translucency === 'number' ? float( this.translucency ) : this.translucency;

	}

	/** The ambient lights and the ground's bounce in the irradiance by `scale`, before the local
	 * lights join it (lighting/GroundBounce.ts). */
	ambientScale(): F {

		return this.scale;

	}

	/** One light's diffuse (lighting/GroundBounce.ts): a blade's wrapped Lambert (or an opaque
	 * leaf's plain one) under the Fresnel weight plus the light through (`extra`), or a translucent
	 * leaf's plain Lambert blended toward its thin layer by smoothstep( 0, THIN_SHARE, t ), lit on the face toward
	 * the viewer, plus the light on its far face by `leafBack` on that same share, on the albedo as the Lambert term (1 / pi),
	 * coloured on that share by what passing through the leaf adds (vegetation/LeafLight.ts `leafWarm`). The key's `direct` (along the lifted key, lighting/KeyLift.ts) and every
	 * local light take it. */
	diffuseResponse( L: V3, light: V3 ): { reflected: V3; meter: V3 } {

		const V = positionViewDirection, N = normalView;
		const f0 = specularColor as unknown as V3, f90 = specularF90 as unknown as F;
		const r = ( this.wrap ? wrappedDiffuseFresnel( N, V, L, this.t(), f0, f90 ) : diffuseFresnel( N, V, L, f0, f90 ) ).toVar();
		if ( this.thin ) {

			// a leaf: the thin layer in the Lambert's place by smoothstep( 0, THIN_SHARE, t ), and
			// the light on its far face through it on the same share
			const n = facingGeometry();
			const H = L.add( V ).normalize();
			const NL = dot( N, L ).toVar();
			const hk = thinLayerScatter( NL, abs( dot( V, n ) ), abs( dot( H, n ) ), thin.wrap, thin.perWrapScale, thin.negTau, thin.lobe, thin.gain ).toVar();
			const s = smoothstep( 0, THIN_SHARE, this.t() ).toVar();
			const back = clamp( NL.negate(), 0, 1 ).mul( leafBack ).mul( s ).toVar();
			return {
				reflected: light.mul( diffuseContribution.mul( mix( r.x.mul( r.y ), hk, s ).add( back ) ) ).mul( mix( vec3( 1 ), leafWarm, s ) ),
				meter: light.mul( mix( r.x, hk, s ).add( back ) ),
			};

		}
		const through = this.extra( L, V, N ).toVar();
		return {
			reflected: light.mul( diffuseContribution.mul( r.x.mul( r.y ) ).add( diffuseColor.rgb.mul( through ) ) ),
			meter: light.mul( through.add( r.x ) ),
		};

	}

}

export function useLighting( mat: THREE.MeshStandardNodeMaterial, model: () => THREE.LightingModel ): void {

	( mat as unknown as { setupLightingModel: () => THREE.LightingModel } ).setupLightingModel = model;

}
