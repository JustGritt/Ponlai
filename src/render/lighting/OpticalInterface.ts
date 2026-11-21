import * as THREE from 'three/webgpu';
import { Fn, array, float, int, vec3, vec4, uv, refract } from 'three/tsl';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// Snell direction and radiance transport follow PBRT 4 sections 9.3.4 and 9.5.2.
// https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission#SpecularTransmission
// https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF#Non-SymmetricScatteringandRefraction
// No upstream code is used. Normal and viewDirection are unit vectors pointing into air.
export const airToWaterDirectionNode = Fn( ( [ viewDirection, normal, waterIOR ]: [ V3, V3, F ] ) => {

	return refract( viewDirection.negate(), normal, float( 1 ).div( waterIOR ) );

} ).setLayout( { name: 'opticalAirToWaterDirection', type: 'vec3', inputs: [
	{ name: 'viewDirection', type: 'vec3' }, { name: 'normal', type: 'vec3' }, { name: 'waterIOR', type: 'float' },
] } );

// Forward light transport preserves L / n² across a lossless refractive interface.
// Fresnel transmission is separate: Water already applies 1 - F at its final mix.
export const radianceAcrossInterfaceNode = Fn( ( [ radiance, fromIOR, toIOR ]: [ V3, F, F ] ) => {

	const ratio = toIOR.div( fromIOR );
	return radiance.mul( ratio.mul( ratio ) );

} ).setLayout( { name: 'opticalInterfaceRadiance', type: 'vec3', inputs: [
	{ name: 'radiance', type: 'vec3' }, { name: 'fromIOR', type: 'float' }, { name: 'toIOR', type: 'float' },
] } );

interface OpticalInterfaceProbeResult {
	passed: boolean;
	maximumDirectionError: number;
	maximumSnellResidual: number;
	maximumUnitLengthError: number;
	maximumRadianceScaleError: number;
	cases: { airAngleDegrees: number; normalTiltDegrees: number; ior: number; waterAngleDegrees: number; expectedWaterAngleDegrees: number; waterToAirRadianceScale: number }[];
}

// Independent GPU check for the functions used by Water: full-angle Snell bending, tilted
// wave normals, index matching, unit direction length and the water-to-air radiance Jacobian.
export async function probeOpticalInterface( renderer: THREE.WebGPURenderer ): Promise<OpticalInterfaceProbeResult> {

	const fixtures = [
		...[ 0, 15, 30, 45, 60, 75, 89 ].map( ( angle ) => ( { angle, tilt: 0, ior: 1.333 } ) ),
		{ angle: 35, tilt: 25, ior: 1.333 }, { angle: 60, tilt: 25, ior: 1.333 },
		{ angle: 45, tilt: 0, ior: 1 }, { angle: 35, tilt: 25, ior: 1 },
	].map( ( fixture ) => {

		const theta = fixture.angle * Math.PI / 180, tilt = fixture.tilt * Math.PI / 180;
		const normal = new THREE.Vector3( Math.sin( tilt ), Math.cos( tilt ), 0 );
		const tangent = new THREE.Vector3( Math.cos( tilt ), - Math.sin( tilt ), 0 );
		const view = normal.clone().multiplyScalar( Math.cos( theta ) ).addScaledVector( tangent, Math.sin( theta ) );
		const cosWater = Math.sqrt( 1 - ( Math.sin( theta ) / fixture.ior ) ** 2 );
		const expected = view.clone().multiplyScalar( - 1 / fixture.ior ).addScaledVector( normal, Math.cos( theta ) / fixture.ior - cosWater );
		return { ...fixture, normal, view, expected };

	} );
	const target = new THREE.RenderTarget( fixtures.length, 1, { type: THREE.FloatType, depthBuffer: false } );
	const material = new THREE.NodeMaterial();
	material.depthTest = material.depthWrite = false;
	material.blending = THREE.NoBlending;
	material.toneMapped = false;
	material.fragmentNode = Fn( () => {

		const index = int( uv().x.mul( fixtures.length ) );
		const sample = array( fixtures.map( ( fixture ) => vec4( ...fixture.view.toArray(), fixture.ior ) ) ).element( index ) as V4;
		const normal = array( fixtures.map( ( fixture ) => vec3( ...fixture.normal.toArray() ) ) ).element( index ) as V3;
		const direction = airToWaterDirectionNode( sample.xyz, normal, sample.w ).toVar();
		const scale = radianceAcrossInterfaceNode( vec3( 1, 1, 1 ), sample.w, float( 1 ) ).x;
		return vec4( direction.x, direction.y, direction.z, scale );

	} )();
	const quad = new THREE.QuadMesh( material );
	const animation = renderer.getAnimationLoop();
	await renderer.setAnimationLoop( null );
	const state = THREE.RendererUtils.resetRendererState( renderer, {} as THREE.RendererUtils.RendererState );
	const before = quad.onBeforeRender, originalVertex = material.vertexNode;
	let fullscreenVertex = originalVertex;
	quad.onBeforeRender = ( ...args ) => {

		fullscreenVertex = material.vertexNode;
		before.apply( quad, args );

	};
	let pixels: THREE.TypedArray;
	try {

		try {

			renderer.setScissorTest( false );
			renderer.setRenderTarget( target );
			quad.render( renderer );
			material.vertexNode = fullscreenVertex;
			await renderer.compileAsync( quad, quad.camera );
			material.vertexNode = originalVertex;
			quad.render( renderer );

		} finally {

			quad.onBeforeRender = before;
			material.vertexNode = originalVertex;
			THREE.RendererUtils.restoreRendererState( renderer, state );

		}
		pixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, fixtures.length, 1 );

	} finally {

		target.dispose();
		material.dispose();
		await renderer.setAnimationLoop( animation );

	}
	let maximumDirectionError = 0, maximumSnellResidual = 0, maximumUnitLengthError = 0, maximumRadianceScaleError = 0;
	const cases = fixtures.map( ( fixture, i ) => {

		const direction = new THREE.Vector3( pixels[ i * 4 ], pixels[ i * 4 + 1 ], pixels[ i * 4 + 2 ] );
		const cosWater = Math.min( 1, Math.max( - 1, - fixture.normal.dot( direction ) ) );
		const sinWater = Math.sqrt( Math.max( 1 - cosWater * cosWater, 0 ) );
		const expectedSin = Math.sin( fixture.angle * Math.PI / 180 ) / fixture.ior;
		maximumDirectionError = Math.max( maximumDirectionError, direction.distanceTo( fixture.expected ) );
		maximumSnellResidual = Math.max( maximumSnellResidual, Math.abs( sinWater - expectedSin ) );
		maximumUnitLengthError = Math.max( maximumUnitLengthError, Math.abs( direction.length() - 1 ) );
		maximumRadianceScaleError = Math.max( maximumRadianceScaleError, Math.abs( pixels[ i * 4 + 3 ] - 1 / ( fixture.ior * fixture.ior ) ) );
		return { airAngleDegrees: fixture.angle, normalTiltDegrees: fixture.tilt, ior: fixture.ior,
			waterAngleDegrees: Math.acos( cosWater ) * 180 / Math.PI,
			expectedWaterAngleDegrees: Math.asin( expectedSin ) * 180 / Math.PI,
			waterToAirRadianceScale: pixels[ i * 4 + 3 ] };

	} );
	return { passed: maximumDirectionError < 2e-6 && maximumSnellResidual < 2e-6
		&& maximumUnitLengthError < 2e-6 && maximumRadianceScaleError < 2e-6,
		maximumDirectionError, maximumSnellResidual, maximumUnitLengthError, maximumRadianceScaleError, cases };

}
