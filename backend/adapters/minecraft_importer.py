import json
from pathlib import Path

import jax.numpy as jnp
from PIL import Image
from matplotlib import pyplot as plt

debug_combined_texture = True


def load_texture(file_path):
    image = Image.open(file_path)
    image = image.convert('RGBA')
    texture = jnp.array(image, dtype=jnp.float32) / 255
    return texture


def resolve_namespace_paths(path_to_asset_dir, namespace_path, namespace):
    namespace_path = namespace_path.replace('minecraft:', '')
    return f"{path_to_asset_dir}/{namespace}/{namespace_path}"


def get_base_path(full_path):
    base_path = full_path.split('minecraft/', 1)[0] + 'minecraft'
    return base_path


def load_state_file(file_path: str):
    variants = []
    with open(file_path) as f:
        blockstates = json.load(f)

        path_to_asset_dir = get_base_path(f.name)
        block_name = Path(f.name).stem

        # flatten variants
        for variant_name, variant_data in blockstates['variants'].items():
            if isinstance(variant_data, list):
                blockstates['variants'][variant_name] = variant_data[0]

        for variant_name, variant_data in blockstates['variants'].items():
            model = load_model(resolve_namespace_paths(path_to_asset_dir, variant_data['model'], "models") + '.json',
                               {})

            if 'x' in variant_data:
                model = rotate(model, variant_data['x'], 0)
            if 'y' in variant_data:
                model = rotate(model, 0, variant_data['y'])

            variants.append({
                "name": block_name + variant_name,
                "model": model,
            })
    return variants


def is_variable(x):
    return x.startswith("#")


def parse_texture_value(val: str, parameters: dict):
    if is_variable(val):
        val = parameters[val[1:]]
    return val


def rotate(model, x, y):
    model_tuple, texture = model
    R = rotation_matrix(x, y)
    vertices, tex_coords, faces = model_tuple
    vertices = jnp.dot(vertices, R)

    return (vertices, tex_coords, faces), texture


def rotation_matrix(x, y):
    """
    Returns a rotation matrix that rotates a model's vertices by x degrees around the x-axis
    and y degrees around the y-axis.

    Args:
        x (float): Rotation angle around the x-axis in degrees.
        y (float): Rotation angle around the y-axis in degrees.

    Returns:
        jnp.ndarray: A 3x3 rotation matrix.
    """
    x_rad = jnp.deg2rad(x)
    y_rad = jnp.deg2rad(y)

    # Rotation around x-axis
    Rx = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(x_rad), -jnp.sin(x_rad)],
        [0, jnp.sin(x_rad), jnp.cos(x_rad)]
    ])

    # Rotation around y-axis
    Ry = jnp.array([
        [jnp.cos(y_rad), 0, jnp.sin(y_rad)],
        [0, 1, 0],
        [-jnp.sin(y_rad), 0, jnp.cos(y_rad)]
    ])

    # Combine rotations
    R = jnp.dot(Ry, Rx)

    return R


def rotate_vertices(vertices, angle_deg, axis, origin, rescale: bool):
    # Convert angle from degrees to radians and invert the direction
    angle_rad = -jnp.deg2rad(angle_deg)

    # Translate vertices to the origin
    vertices_translated = vertices - origin

    # Create rotation matrix based on the axis
    if axis == 'x':
        rotation_matrix = jnp.array([
            [1, 0, 0],
            [0, jnp.cos(angle_rad), -jnp.sin(angle_rad)],
            [0, jnp.sin(angle_rad), jnp.cos(angle_rad)]
        ])
        scale_axes = (1, 2)  # Scale y and z axes
    elif axis == 'y':
        rotation_matrix = jnp.array([
            [jnp.cos(angle_rad), 0, jnp.sin(angle_rad)],
            [0, 1, 0],
            [-jnp.sin(angle_rad), 0, jnp.cos(angle_rad)]
        ])
        scale_axes = (0, 2)  # Scale x and z axes
    elif axis == 'z':
        rotation_matrix = jnp.array([
            [jnp.cos(angle_rad), -jnp.sin(angle_rad), 0],
            [jnp.sin(angle_rad), jnp.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        scale_axes = (0, 1)  # Scale x and y axes
    else:
        raise ValueError("Invalid rotation axis. Must be 'x', 'y', or 'z'.")

    # Perform rotation
    rotated_vertices = jnp.dot(vertices_translated, rotation_matrix)

    if rescale:
        # Calculate the scaling factor
        scale_factor = 1 + 1 / (jnp.cos(angle_deg) - 1)

        # Create scaling matrix
        scaling_matrix = jnp.eye(3)
        scaling_matrix = scaling_matrix.at[scale_axes].set(scale_factor)

        # Scale the rotated vertices
        rotated_vertices = jnp.dot(rotated_vertices, scaling_matrix)

    # Translate vertices back to the original position
    rotated_vertices += origin

    return rotated_vertices


def load_model(file_path: str, parameters: dict):
    with open(file_path) as f:
        model = json.load(f)

        if 'textures' in model:
            for texture_var, texture_path in model['textures'].items():
                parameters[texture_var] = parse_texture_value(texture_path, parameters)

        if 'elements' in model:
            for element_id, element in enumerate(model['elements']):
                # make elements into obj format
                from_pos = jnp.array(element['from'])
                to_pos = jnp.array(element['to'])

                if jnp.any(from_pos > 16) or jnp.any(from_pos < 0):
                    raise Exception("model out of bounds.")
                if jnp.any(to_pos > 16) or jnp.any(to_pos < 0):
                    raise Exception("model out of bounds.")

                faces = element['faces']

                for face_orientation, face_data in faces.items():
                    # face_orientation can be down, up, north, south, west, east.
                    # from, to are [x, y, z] positions that form a cuboid.
                    # Now create two triangles that form the face for the current face_orientation

                    # Extract vertex positions based on face_orientation
                    if face_orientation == 'down':
                        v1 = jnp.array([from_pos[0], from_pos[1], from_pos[2]])
                        v2 = jnp.array([to_pos[0], from_pos[1], from_pos[2]])
                        v3 = jnp.array([to_pos[0], from_pos[1], to_pos[2]])
                        v4 = jnp.array([from_pos[0], from_pos[1], to_pos[2]])
                    elif face_orientation == 'up':
                        v1 = jnp.array([from_pos[0], to_pos[1], from_pos[2]])
                        v2 = jnp.array([to_pos[0], to_pos[1], from_pos[2]])
                        v3 = jnp.array([to_pos[0], to_pos[1], to_pos[2]])
                        v4 = jnp.array([from_pos[0], to_pos[1], to_pos[2]])
                    elif face_orientation == 'north':
                        v1 = jnp.array([from_pos[0], from_pos[1], from_pos[2]])
                        v2 = jnp.array([to_pos[0], from_pos[1], from_pos[2]])
                        v3 = jnp.array([to_pos[0], to_pos[1], from_pos[2]])
                        v4 = jnp.array([from_pos[0], to_pos[1], from_pos[2]])
                    elif face_orientation == 'south':
                        v1 = jnp.array([from_pos[0], from_pos[1], to_pos[2]])
                        v2 = jnp.array([to_pos[0], from_pos[1], to_pos[2]])
                        v3 = jnp.array([to_pos[0], to_pos[1], to_pos[2]])
                        v4 = jnp.array([from_pos[0], to_pos[1], to_pos[2]])
                    elif face_orientation == 'west':
                        v1 = jnp.array([from_pos[0], from_pos[1], from_pos[2]])
                        v2 = jnp.array([from_pos[0], from_pos[1], to_pos[2]])
                        v3 = jnp.array([from_pos[0], to_pos[1], to_pos[2]])
                        v4 = jnp.array([from_pos[0], to_pos[1], from_pos[2]])
                    elif face_orientation == 'east':
                        v1 = jnp.array([to_pos[0], from_pos[1], from_pos[2]])
                        v2 = jnp.array([to_pos[0], from_pos[1], to_pos[2]])
                        v3 = jnp.array([to_pos[0], to_pos[1], to_pos[2]])
                        v4 = jnp.array([to_pos[0], to_pos[1], from_pos[2]])
                    else:
                        continue
                        raise Exception("Invalid face orientation.")

                    vertices = jnp.stack([v1, v2, v3, v4]) / 16

                    if 'rotation' in element:
                        rotation = element['rotation']
                        # angle in degrees
                        # axis is x, y or z
                        # origin is [x, y, z] pos
                        vertices = rotate_vertices(vertices, rotation['angle'], rotation['axis'],
                                                   jnp.array(rotation['origin']) / 16,
                                                   rescale='rescale' in rotation and rotation['rescale'])

                    face_data['vertices'] = vertices
                    face_data['faces'] = jnp.array([
                        [(1, 1), (2, 2), (3, 3)],
                        [(1, 1), (3, 3), (4, 4)]
                    ])

                    if 'uv' not in face_data:
                        # Compute the UV coordinates depending on the face orientation
                        if face_orientation == 'down' or face_orientation == 'up':
                            uv = jnp.array([
                                from_pos[0], 16 - to_pos[2],
                                to_pos[0], 16 - from_pos[2],
                            ])
                        elif face_orientation == 'north' or face_orientation == 'south':
                            # 1, 8,
                            # 14, 15
                            uv = jnp.array([
                                from_pos[0], 16 - to_pos[1],
                                to_pos[0], 16 - from_pos[1],
                            ])
                        elif face_orientation == 'west' or face_orientation == 'east':
                            uv = jnp.array([
                                from_pos[2], 16 - to_pos[1],
                                to_pos[2], 16 - from_pos[1],
                            ])
                        else:
                            raise Exception("Invalid face orientation.")

                        face_data['uv'] = uv

                    assert all([0 <= coord <= 16 for coord in face_data['uv']])
                    uv_from_x, uv_from_y, uv_to_x, uv_to_y = face_data['uv']

                    uv_to_x -= 1
                    uv_to_y -= 1

                    face_data['texture_coords'] = jnp.array([
                        (uv_to_y, uv_to_x), (uv_to_y, uv_from_x), (uv_from_y, uv_from_x), (uv_from_y, uv_to_x)
                    ])

                    face_data['texture'] = parse_texture_value(face_data['texture'], parameters)

                    parameter_elements = parameters.setdefault('elements', [])
                    if element_id >= len(parameter_elements):
                        parameter_elements.append({})

                    parameter_elements[element_id].setdefault('faces', {})[face_orientation] = face_data

        if 'parent' in model:
            return load_model(resolve_namespace_paths(get_base_path(f.name), model['parent'], "models") + '.json',
                              parameters)
        else:
            # ...

            used_textures = set()
            for element in parameters.get('elements', []):
                for face in element.get('faces', {}).values():
                    used_textures.add(face['texture'])

            loaded_textures = {}
            for texture_path in used_textures:
                loaded_textures[texture_path] = load_texture(
                    resolve_namespace_paths(get_base_path(f.name), texture_path, "textures") + '.png')

            # Tinting
            tinted = []
            for element in parameters.get('elements', []):
                for face in element.get('faces', {}).values():
                    if 'tintindex' in face and face['texture'] not in tinted:
                        tinted.append(face['texture'])
                        loaded_textures[face['texture']] *= (jnp.array([124, 189, 107, 255])[None, None, :] / 255)

            max_texture_height = max((img.shape[0] for img in loaded_textures.values()), default=0)

            combined_texture = jnp.zeros((max_texture_height, 0, 4))
            for texture_path in used_textures:
                texture_img = loaded_textures[texture_path]
                # store texture offset
                loaded_textures[texture_path] = combined_texture.shape[1]

                texture_img = jnp.pad(texture_img, [(0, max_texture_height - texture_img.shape[0]), (0, 0), (0, 0)])
                combined_texture = jnp.concatenate([combined_texture, texture_img], axis=1)

            # combined_texture = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0, 1.0]), (1, 1, 4))

            combined_texture_size = jnp.array(combined_texture.shape)[None, :2]

            if debug_combined_texture and all([dim > 0 for dim in combined_texture.shape]):
                plt.imshow(combined_texture)
                plt.show()

            texture_coords = jnp.zeros((0, 2))
            vertices = jnp.zeros((0, 3))
            for element in parameters.get('elements', []):
                for face in element.get('faces', {}).values():
                    vertices = jnp.unique(jnp.concatenate([vertices, face['vertices']], axis=0), axis=0)

                    texture_offset = jnp.array([0, loaded_textures[face['texture']]])[None, :]
                    combined_texture_coords = (texture_offset + face['texture_coords'])

                    texture_coords = jnp.unique(jnp.concatenate([texture_coords, combined_texture_coords], axis=0),
                                                axis=0)
                    face['texture_coords'] = combined_texture_coords

            faces = jnp.zeros((0, 3, 2), dtype=jnp.int32)
            for element in parameters.get('elements', []):
                for face in element.get('faces', {}).values():
                    for triangle in face['faces']:  # shape (3, 2)
                        new_triangle = jnp.zeros((3, 2), dtype=jnp.int32)
                        for i in range(triangle.shape[0]):
                            vertex_id, uv_id = triangle[i]
                            target_vertex = face['vertices'][vertex_id - 1]
                            target_uv_coord = face['texture_coords'][uv_id - 1]

                            # Find the indices of the target vertex and UV coordinates in the respective arrays
                            vertex_index = jnp.where((vertices == target_vertex).all(axis=1))[0][0]
                            uv_index = jnp.where((texture_coords == target_uv_coord).all(axis=1))[0][0]

                            # Assign the new indices to the triangle
                            new_triangle = new_triangle.at[i].set(jnp.array([vertex_index, uv_index]))

                        # Append the new triangle to the faces array
                        faces = jnp.concatenate([faces, new_triangle[None, ...]], axis=0)

            # Important postprocessing
            vertices = vertices - 0.5
            texture_coords = texture_coords / (combined_texture_size - 1)

            # just for debugging:
            debugging_face_uvs = texture_coords[faces[..., 1]]
            tex_w, tex_h = combined_texture.shape[:2]
            tex_x = jnp.round(debugging_face_uvs[..., 0] * (tex_w - 1)).astype(int)
            tex_y = jnp.round(debugging_face_uvs[..., 1] * (tex_h - 1)).astype(int)

            return (vertices, texture_coords, faces), combined_texture
            pass
    pass


air = (jnp.zeros((0, 3)), jnp.zeros((0, 2)), jnp.zeros((0, 3, 2), dtype=jnp.int32))

voxel_assets = [
    *load_state_file("data/minecraft/blockstates/brewing_stand.json"),
    *load_state_file("data/minecraft/blockstates/air.json"),
]
