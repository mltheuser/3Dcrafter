<script lang="ts">
	import { onMount } from 'svelte';
	import * as BABYLON from '@babylonjs/core';
	import '@babylonjs/loaders/glTF';
	import '@babylonjs/loaders/OBJ';

	let canvas: HTMLCanvasElement;
	let scene: BABYLON.Scene;
	let engine: BABYLON.Engine;

	let camera: BABYLON.ArcRotateCamera;

	let voxelRoot: BABYLON.Mesh;

	let voxelGrid3D: Array<any>;

	const subdivisions = 50;

	function createSurfacePoints(mesh: BABYLON.Mesh, pointDensity: number) {
		let vertexData = BABYLON.VertexData.ExtractFromMesh(mesh);
		mesh.computeWorldMatrix(true);
		vertexData.transform(mesh.getWorldMatrix());

		const positions = vertexData.positions;
		const indices = vertexData.indices;

		if (positions == null || indices == null) {
			return [];
		}

		const points = [];

		for (let index = 0; index < indices.length; index += 3) {
			const id0 = indices[index];
			const id1 = indices[index + 1];
			const id2 = indices[index + 2];

			const v0X = positions[3 * id0];
			const v0Y = positions[3 * id0 + 1];
			const v0Z = positions[3 * id0 + 2];
			const v1X = positions[3 * id1];
			const v1Y = positions[3 * id1 + 1];
			const v1Z = positions[3 * id1 + 2];
			const v2X = positions[3 * id2];
			const v2Y = positions[3 * id2 + 1];
			const v2Z = positions[3 * id2 + 2];

			const vertex0 = new BABYLON.Vector3(v0X, v0Y, v0Z);
			const vertex1 = new BABYLON.Vector3(v1X, v1Y, v1Z);
			const vertex2 = new BABYLON.Vector3(v2X, v2Y, v2Z);

			// Generate uniformly distributed points on the surface of the triangle
			for (let i = 0; i < pointDensity; i++) {
				const r1 = Math.random();
				const r2 = Math.random();
				const u = 1 - Math.sqrt(r1);
				const v = Math.sqrt(r1) * (1 - r2);
				const w = Math.sqrt(r1) * r2;

				const point = vertex0.scale(u).add(vertex1.scale(v)).add(vertex2.scale(w));
				points.push(point);
			}
		}

		return points;
	}

	onMount(() => {
		engine = new BABYLON.Engine(canvas, true);
		scene = new BABYLON.Scene(engine);

		// Set the background color to black
		scene.clearColor = new BABYLON.Color4(0, 0, 0, 1);

		camera = new BABYLON.ArcRotateCamera(
			'camera1',
			Math.PI / 2,
			Math.PI / 2,
			10,
			BABYLON.Vector3.Zero(),
			scene
		);
		camera.attachControl(canvas, true);
		camera.lowerRadiusLimit = 0.1;
		camera.upperRadiusLimit = 1000;

		new BABYLON.HemisphericLight('light1', new BABYLON.Vector3(0, 1, 0), scene);

		createVoxelizedVersion(scene, subdivisions)

		engine.runRenderLoop(() => {
			scene.render();
		});

		window.addEventListener('resize', () => {
			engine.resize();
		});
	});

	const handleFileUpload = (event: Event) => {
		const fileObj = (event.target as HTMLInputElement).files[0];
		const url = URL.createObjectURL(fileObj);

		BABYLON.SceneLoader.ImportMeshAsync('', url, '', scene, null, '.glb').then((result) => {
			// This requires that mesh 0 is the root of the model
			const rootNode = result.meshes[0] as BABYLON.Mesh;

			// scale model to unit cube (not necessarily centered on 0,0,0)
			rootNode.normalizeToUnitCube(true);

			// now move the center of the unit cube the root of the meshes was scaled to to the coordinate center
			const { min, max } = rootNode.getHierarchyBoundingVectors(true);
			const center = new BABYLON.Vector3(
				(min.x + max.x) / 2,
				(min.y + max.y) / 2,
				(min.z + max.z) / 2
			);
			rootNode.position = center.negate();

			createVoxelizedVersion(scene, subdivisions);
		});
	};

	function createVoxelizedVersion(scene: BABYLON.Scene, subdivisions: number) {
		// Create a 3D voxel grid with the specified subdivisions
		voxelGrid3D = new Array(subdivisions)
			.fill(null)
			.map(() => new Array(subdivisions).fill(null).map(() => new Array(subdivisions).fill(false)));

		// Iterate over all meshes in the scene
		for (const mesh of scene.meshes) {
			// Get vertices in world space

			const surfacePoints = createSurfacePoints(mesh as BABYLON.Mesh, subdivisions);

			// Iterate over all vertices
			for (const point of surfacePoints) {
				// check vertex world coord to voxel grid
				const voxelGridPosition = new BABYLON.Vector3(
					Math.floor((point.x + 0.5) / (1 / subdivisions)),
					Math.floor((point.y + 0.5) / (1 / subdivisions)),
					Math.floor((point.z + 0.5) / (1 / subdivisions))
				);

				if (voxelGridPosition._x >= 0 && voxelGridPosition._x < subdivisions) {
					if (voxelGridPosition._y >= 0 && voxelGridPosition._y < subdivisions) {
						if (voxelGridPosition._z >= 0 && voxelGridPosition._z < subdivisions) {
							voxelGrid3D[voxelGridPosition._x][voxelGridPosition._y][voxelGridPosition._z] = true;
						}
					}
				}
			}
		}

		voxelRoot = new BABYLON.Mesh('voxelRoot', scene);

		// Iterate over all voxels in the grid
		for (let x = 0; x < subdivisions; x++) {
			for (let y = 0; y < subdivisions; y++) {
				for (let z = 0; z < subdivisions; z++) {
					// If the voxel is occupied, create a cube at its position
					if (voxelGrid3D[x][y][z]) {
						const cube = BABYLON.MeshBuilder.CreateBox('cube', { size: 1 / subdivisions }, scene);
						cube.position = new BABYLON.Vector3(
							x / subdivisions - 0.5 + 1 / subdivisions / 2,
							y / subdivisions - 0.5 + 1 / subdivisions / 2,
							z / subdivisions - 0.5 + 1 / subdivisions / 2
						);

						// Generate a random color value
						const randomColor = new BABYLON.Color3(Math.random(), Math.random(), Math.random());

						// Create a new standard material with the random color
						const material = new BABYLON.StandardMaterial('material', scene);
						material.diffuseColor = randomColor;

						// Assign the material to the cube
						cube.material = material;

						cube.parent = voxelRoot;
					}
				}
			}
		}
	}

	function matrix4x4ToArray(matrix: BABYLON.Matrix) {
		var array: any[] = [];
		for (var i = 0; i < 4; i++) {
			array[i] = [];
			for (var j = 0; j < 4; j++) {
				array[i][j] = matrix.m[i * 4 + j];
			}
		}
		return array;
	}

	let screenshotData: any[] = [];

	function makeScreenshot() {
		if (voxelRoot) {
			voxelRoot.visibility = 0.0;
			voxelRoot.getChildren().forEach((m) => ((m as BABYLON.Mesh).visibility = 0.0));
		}

		BABYLON.Tools.CreateScreenshot(engine, camera, { width: 512, height: 512 }, (data: string) => {
			
			camera.computeWorldMatrix()

			screenshotData = [
				...screenshotData,
				{
					image: data,
					camera: {
						viewMatrix: [
							[...camera.getDirection(BABYLON.Vector3.Right()).asArray(), 0],
							[...camera.getDirection(BABYLON.Vector3.Up()).asArray(), 0],
							[...camera.getDirection(BABYLON.Vector3.Forward()).asArray(), 0],
							[...camera.globalPosition.asArray(), 1],
						]
					}
				}
			];

			if (voxelRoot) {
				voxelRoot.visibility = 1.0;
				voxelRoot.getChildren().forEach((m) => ((m as BABYLON.Mesh).visibility = 1.0));
			}
		});
	}

	async function sendData() {
		const data = {
			subdivisions: subdivisions,
			voxelGrid: voxelGrid3D,
			views: screenshotData
		};

		try {
			const response = await fetch('http://localhost:5000/builder', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(data)
			});

			if (response.ok) {
				console.log('Data sent successfully');
			} else {
				console.error('Error sending data:', response.statusText);
			}
		} catch (error) {
			console.error('Error sending data:', error);
		}
	}
</script>

<canvas bind:this={canvas} />

<div>
	<input type="file" accept=".glb" on:change={handleFileUpload} />
</div>
<div>
	<button on:click={makeScreenshot}>Screenshot</button>
	<button on:click={sendData}>Send</button>
</div>
<div>
	<ul>
		{#each screenshotData as screenshot}
			<li>
				<img src={screenshot.image} alt="Screenshot" />
			</li>
		{/each}
	</ul>
</div>

<style>
	canvas {
		width: 100%;
		height: 100%;
	}
</style>


"""
precompute(view_matrix) -> Array of color contributions along the ray (Will be padded to max amounts of voxel passes possible) it then has the shape C (image_width, image_height, VOXEl_INTERSECTIONS_PADDING, PROB_DIST_DIM) + We also need to relate the probabilty dists in the trainable voxel grid VG (n, n, n, PROB_DIST_DIM) to these, so we also return a Matrix T (image_width, image_height, VOXEl_INTERSECTIONS_PADDING, 1) that will resort them accordingle so that we can then do VG ** T (image_width, image_height, VOXEl_INTERSECTIONS_PADDING, PROB_DIST_DIM) * (image_width, image_height, VOXEl_INTERSECTIONS_PADDING, PROB_DIST_DIM)
render(voxel_grid, selection_mode) -> takes the pre computed stuff above and either mixes the values in C or picks the most likely
backprop() -> weil oben nur matrix operationnen genuzt werden sollte das leich mit grad machbar sein.
"""