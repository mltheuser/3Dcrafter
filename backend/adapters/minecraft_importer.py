import json
from pathlib import Path

import jax.numpy as jnp
from PIL import Image
from matplotlib import pyplot as plt

debug_combined_texture = False


def load_texture(file_path):
    image = Image.open(file_path)
    image = image.convert('RGBA')
    texture = jnp.array(image, dtype=jnp.uint8)
    return texture


def resolve_namespace_paths(path_to_asset_dir, namespace_path, namespace):
    namespace_path = namespace_path.replace('minecraft:', '')
    return f"{path_to_asset_dir}/{namespace}/{namespace_path}"


def get_base_path(full_path):
    base_path = full_path.split('minecraft/', 1)[0] + 'minecraft'
    return base_path


def all_multipart_subsets(lst):
    """
    Returns all possible subsets of the given list.

    Example:
        >>> all_multipart_subsets([1, 2, 3])
        [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
    """
    if not lst:
        return [[]]  # base case: empty list has only one subset, the empty set

    subsets = []
    first_elem = lst[0]
    rest_list = lst[1:]
    for subset in all_multipart_subsets(rest_list):
        subsets.append(subset)  # add the subset without the first element
        # only add the first element if no existing condition contradicts it
        has_contradiction = False
        for key in first_elem['when']:
            for e in subset:
                if key in e['when'] and e['when'][key] != first_elem['when'][key]:
                    has_contradiction = True
                    break
        # else you can add it
        if not has_contradiction:
            subsets.append([first_elem] + subset)  # add the subset with the first element

    return subsets


def load_model_apply(path_to_asset_dir, parameters):
    textured_model = load_model(
        resolve_namespace_paths(path_to_asset_dir, parameters['model'], "models") + '.json',
        {})

    if 'x' in parameters:
        textured_model['model']['vertices'] = rotate_vertices(textured_model['model']['vertices'],
                                                              parameters['x'], axis='x',
                                                              origin=jnp.zeros((1, 3)), rescale=False)

    if 'y' in parameters:
        textured_model['model']['vertices'] = rotate_vertices(textured_model['model']['vertices'],
                                                              parameters['y'], axis='y',
                                                              origin=jnp.zeros((1, 3)), rescale=False)
    return textured_model


def load_multipart_block(parts, path_to_asset_dir, block_name):
    fixed_parts = []
    optional_parts = []
    for part in parts:
        if 'when' in part:
            optional_parts.append(part)
        else:
            fixed_parts.append(part)

    extracted_variants = [subset + fixed_parts for subset in all_multipart_subsets(optional_parts)]

    variants = []
    for extracted_variant in extracted_variants:
        if len(extracted_variant) == 0:
            continue

        models = []
        for apply_parameters in extracted_variant:
            if isinstance(apply_parameters['apply'], list):
                print("Random model selection not supported. Will take first option for now.")
                apply_parameters['apply'] = apply_parameters['apply'][0]
            textured_model = load_model_apply(path_to_asset_dir, apply_parameters['apply'])
            models.append(textured_model)

        # combine models
        combined_vertices = jnp.concatenate([model['model']['vertices'] for model in models], axis=0)

        # combine textures
        tex_x = [
            jnp.round(model['model']['texture_coords'][..., 0] * (model['texture'].shape[0] - 1)).astype(int)
            for
            model_id, model in enumerate(models)]
        tex_y = [
            jnp.round(model['model']['texture_coords'][..., 1] * (model['texture'].shape[1] - 1)).astype(int)
            for
            model_id, model in enumerate(models)]

        textures = [model['texture'] for model in models]
        max_tex_width = max([tex.shape[1] for tex in textures])
        combined_textures = jnp.zeros((0, max_tex_width, 4), dtype=jnp.uint8)
        for texture_id, texture in enumerate(textures):
            tex_x[texture_id] += combined_textures.shape[0]
            combined_textures = jnp.concatenate([combined_textures, texture], axis=0)

        # plt.imshow(combined_textures)
        # plt.show()

        offset_tex_coords = [jnp.concatenate([t_x[:, None], t_y[:, None]], axis=-1) for t_x, t_y in zip(tex_x, tex_y)]
        offset_tex_coords = [coords / (jnp.array(combined_textures.shape[:2])[None, :] - 1) for coords in
                             offset_tex_coords]
        combined_tex_coords = jnp.concatenate(offset_tex_coords, axis=0)

        # Now iterate over the faces and find the original things again.
        combined_faces = jnp.zeros((0, 3, 2), dtype=jnp.int32)
        for model_id, model in enumerate(models):
            for triangle in model['model']['faces']:
                new_triangle = jnp.zeros((3, 2), dtype=jnp.int32)
                for i in range(triangle.shape[0]):
                    vertex_id, uv_id = triangle[i]

                    target_vertex = model['model']['vertices'][vertex_id]
                    target_uv_coord = offset_tex_coords[model_id][uv_id]

                    # Find the indices of the target vertex and UV coordinates in the respective arrays
                    vertex_index = jnp.where((combined_vertices == target_vertex).all(axis=1))[0][0]
                    uv_index = jnp.where((combined_tex_coords == target_uv_coord).all(axis=1))[0][0]

                    # Assign the new indices to the triangle
                    new_triangle = new_triangle.at[i].set(jnp.array([vertex_index, uv_index]))

                # Append the new triangle to the faces array
                combined_faces = jnp.concatenate([combined_faces, new_triangle[None, ...]], axis=0)

        variants.append({
            "name": block_name,
            "model": {
                'vertices': combined_vertices,
                'texture_coords': combined_tex_coords,
                'faces': combined_faces,
            },
            "texture": combined_textures,
        })
        pass

    return variants


def load_state_file(file_path: str):
    variants = []
    with open(file_path) as f:
        blockstates = json.load(f)

        path_to_asset_dir = get_base_path(f.name)
        block_name = Path(f.name).stem

        # extract variants from multiparts
        if 'multipart' in blockstates:
            return load_multipart_block(blockstates['multipart'], path_to_asset_dir, block_name)

        # flatten variants
        for variant_name, variant_data in blockstates['variants'].items():
            if isinstance(variant_data, list):
                print("Random model selection not supported. Will take first option for now.")
                blockstates['variants'][variant_name] = variant_data[0]

        for variant_name, variant_data in blockstates['variants'].items():
            textured_model = load_model_apply(path_to_asset_dir, variant_data)

            variants.append({
                "name": block_name + variant_name,
                "model": textured_model['model'],
                "texture": textured_model['texture'],
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


def clean_array_list(array_list):
    if array_list.shape[0] == 0:
        return array_list

    old_shape = array_list.shape
    vector_list = jnp.reshape(array_list, (old_shape[0], -1))

    vector_list = jnp.unique(vector_list, axis=0)
    vector_list = vector_list[jnp.lexsort(jnp.rot90(vector_list))]

    target_shape = [-1, *old_shape[1:]]
    return jnp.reshape(vector_list, target_shape)


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

                    vertices = jnp.stack([v1, v2, v3, v4])

                    if 'rotation' in element:
                        rotation = element['rotation']
                        vertices = rotate_vertices(vertices, rotation['angle'], rotation['axis'],
                                                   jnp.array(rotation['origin']),
                                                   rescale='rescale' in rotation and rotation['rescale'])

                    face_data['vertices'] = vertices / 16
                    face_data['faces'] = jnp.array([
                        [(1, 1), (2, 2), (3, 3)],
                        [(1, 1), (3, 3), (4, 4)]
                    ])

                    if 'uv' not in face_data:
                        # Compute the UV coordinates depending on the face orientation
                        if face_orientation == 'down' or face_orientation == 'up':
                            uv = [
                                from_pos[0], 16 - to_pos[2],
                                to_pos[0], 16 - from_pos[2],
                            ]
                        elif face_orientation == 'north' or face_orientation == 'south':
                            uv = [
                                from_pos[0], 16 - to_pos[1],
                                to_pos[0], 16 - from_pos[1],
                            ]
                        elif face_orientation == 'west' or face_orientation == 'east':
                            uv = [
                                from_pos[2], 16 - to_pos[1],
                                to_pos[2], 16 - from_pos[1],
                            ]
                        else:
                            raise Exception("Invalid face orientation.")

                        face_data['uv'] = uv

                    uv = jnp.array(face_data['uv'])

                    if uv[0] > uv[2]:
                        raise Exception(
                            'Tried to load model with flipped texture. See docs. This is not supported currently.')

                    uv = uv - jnp.array([0, 0, 1, 1])

                    assert jnp.all(uv >= 0)

                    uv_from_x, uv_from_y, uv_to_x, uv_to_y = uv

                    face_data['texture_coords'] = jnp.array([
                        (uv_to_y, uv_to_x), (uv_to_y, uv_from_x), (uv_from_y, uv_from_x), (uv_from_y, uv_to_x)
                    ])

                    parameter_elements = parameters.setdefault('elements', [])
                    if element_id >= len(parameter_elements):
                        parameter_elements.append({})

                    parameter_elements[element_id].setdefault('faces', {})[face_orientation] = face_data

        if 'parent' in model:
            return load_model(resolve_namespace_paths(get_base_path(f.name), model['parent'], "models") + '.json',
                              parameters)
        else:
            used_textures = set()
            for element in parameters.get('elements', []):
                for face in element.get('faces', {}).values():
                    used_textures.add(face['texture'])

            used_textures = sorted(used_textures)

            loaded_textures = []
            for texture_var in used_textures:
                texture_path = parse_texture_value(texture_var, parameters)
                texture = load_texture(
                    resolve_namespace_paths(get_base_path(f.name), texture_path, "textures") + '.png') / 255
                loaded_textures.append(texture)

            # Tinting
            tinted = []
            for element in parameters.get('elements', []):
                for face in element.get('faces', {}).values():
                    if 'tintindex' in face and face['texture'] not in tinted:
                        tinted.append(face['texture'])
                        loaded_texture_index = used_textures.index(face['texture'])
                        loaded_textures[loaded_texture_index] *= (jnp.array([124, 189, 107, 255])[None, None, :] / 255)

            max_tex_height = max([tex.shape[1] for tex in loaded_textures], default=0)

            combined_texture = jnp.zeros((0, max_tex_height, 4), dtype=jnp.uint8)
            tex_offsets = [0] * len(loaded_textures)
            for tex_id, loaded_texture in enumerate(loaded_textures):
                tex_offsets[tex_id] = combined_texture.shape[0]

                loaded_texture = jnp.clip(jnp.round(loaded_texture * 255), 0, 255).astype(jnp.uint8)

                combined_texture = jnp.concatenate([combined_texture, loaded_texture], axis=0)

            combined_texture_size = jnp.array(combined_texture.shape)[None, :2]

            if debug_combined_texture and all([dim > 0 for dim in combined_texture.shape]):
                plt.imshow(combined_texture)
                plt.show()

            texture_coords = jnp.zeros((0, 2))
            vertices = jnp.zeros((0, 3))
            for element in parameters.get('elements', []):
                for face in element.get('faces', {}).values():
                    vertices = jnp.unique(jnp.concatenate([vertices, face['vertices']], axis=0), axis=0)

                    tex_id = used_textures.index(face['texture'])
                    texture_offset = jnp.array([tex_offsets[tex_id], 0])[None, :]
                    combined_texture_coords = (texture_offset + face['texture_coords'])

                    texture_coords = jnp.concatenate([texture_coords, combined_texture_coords], axis=0)
                    face['texture_coords'] = combined_texture_coords

            # sort texture_coords and vertices in a unique way
            texture_coords = clean_array_list(texture_coords)
            vertices = clean_array_list(vertices)

            faces = jnp.zeros((0, 3, 2), dtype=jnp.int32)
            for element in parameters.get('elements', []):
                sorted_element_faces = [value for key, value in sorted(element.get('faces', {}).items(), reverse=True)]
                for face in sorted_element_faces:
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

            faces = clean_array_list(faces)

            # Important postprocessing
            vertices = vertices - 0.5
            texture_coords = texture_coords / (combined_texture_size - 1)

            return {
                "model": {
                    "vertices": vertices,
                    "texture_coords": texture_coords,
                    "faces": faces,
                },
                "texture": combined_texture,
            }
            pass
    pass


def load_slabs():
    return [
        *load_state_file("data/minecraft/blockstates/oak_slab.json"),
        *load_state_file("data/minecraft/blockstates/spruce_slab.json"),
        *load_state_file("data/minecraft/blockstates/birch_slab.json"),
        *load_state_file("data/minecraft/blockstates/jungle_slab.json"),
        *load_state_file("data/minecraft/blockstates/acacia_slab.json"),
        *load_state_file("data/minecraft/blockstates/dark_oak_slab.json"),
        *load_state_file("data/minecraft/blockstates/mangrove_slab.json"),
        *load_state_file("data/minecraft/blockstates/cherry_slab.json"),
        *load_state_file("data/minecraft/blockstates/bamboo_slab.json"),
        *load_state_file("data/minecraft/blockstates/bamboo_mosaic_slab.json"),
        *load_state_file("data/minecraft/blockstates/crimson_slab.json"),
        *load_state_file("data/minecraft/blockstates/warped_slab.json"),
        *load_state_file("data/minecraft/blockstates/stone_slab.json"),
        *load_state_file("data/minecraft/blockstates/cobblestone_slab.json"),
        *load_state_file("data/minecraft/blockstates/mossy_cobblestone_slab.json"),
        *load_state_file("data/minecraft/blockstates/smooth_stone_slab.json"),
        *load_state_file("data/minecraft/blockstates/stone_brick_slab.json"),
        *load_state_file("data/minecraft/blockstates/mossy_stone_brick_slab.json"),
        *load_state_file("data/minecraft/blockstates/granite_slab.json"),
        *load_state_file("data/minecraft/blockstates/polished_granite_slab.json"),
        *load_state_file("data/minecraft/blockstates/diorite_slab.json"),
        *load_state_file("data/minecraft/blockstates/polished_diorite_slab.json"),
        *load_state_file("data/minecraft/blockstates/andesite_slab.json"),
        *load_state_file("data/minecraft/blockstates/polished_andesite_slab.json"),
        *load_state_file("data/minecraft/blockstates/cobbled_deepslate_slab.json"),
        *load_state_file("data/minecraft/blockstates/polished_deepslate_slab.json"),
        *load_state_file("data/minecraft/blockstates/deepslate_brick_slab.json"),
        *load_state_file("data/minecraft/blockstates/deepslate_tile_slab.json"),
        *load_state_file("data/minecraft/blockstates/brick_slab.json"),
        *load_state_file("data/minecraft/blockstates/mud_brick_slab.json"),
        *load_state_file("data/minecraft/blockstates/sandstone_slab.json"),
        *load_state_file("data/minecraft/blockstates/smooth_sandstone_slab.json"),
        *load_state_file("data/minecraft/blockstates/cut_sandstone_slab.json"),
        *load_state_file("data/minecraft/blockstates/red_sandstone_slab.json"),
        *load_state_file("data/minecraft/blockstates/smooth_red_sandstone_slab.json"),
        *load_state_file("data/minecraft/blockstates/cut_red_sandstone_slab.json"),
        *load_state_file("data/minecraft/blockstates/prismarine_slab.json"),
        *load_state_file("data/minecraft/blockstates/prismarine_brick_slab.json"),
        *load_state_file("data/minecraft/blockstates/dark_prismarine_slab.json"),
        *load_state_file("data/minecraft/blockstates/nether_brick_slab.json"),
        *load_state_file("data/minecraft/blockstates/red_nether_brick_slab.json"),
        *load_state_file("data/minecraft/blockstates/blackstone_slab.json"),
        *load_state_file("data/minecraft/blockstates/polished_blackstone_slab.json"),
        *load_state_file("data/minecraft/blockstates/polished_blackstone_brick_slab.json"),
        *load_state_file("data/minecraft/blockstates/end_stone_brick_slab.json"),
        *load_state_file("data/minecraft/blockstates/purpur_slab.json"),
        *load_state_file("data/minecraft/blockstates/quartz_slab.json"),
        *load_state_file("data/minecraft/blockstates/smooth_quartz_slab.json"),
        *load_state_file("data/minecraft/blockstates/cut_copper_slab.json"),
        *load_state_file("data/minecraft/blockstates/exposed_cut_copper_slab.json"),
        *load_state_file("data/minecraft/blockstates/weathered_cut_copper_slab.json"),
        *load_state_file("data/minecraft/blockstates/oxidized_cut_copper_slab.json"),
    ]


def load_colored_concrete():
    return [
        *load_state_file("data/minecraft/blockstates/white_concrete.json"),
        *load_state_file("data/minecraft/blockstates/light_gray_concrete.json"),
        *load_state_file("data/minecraft/blockstates/gray_concrete.json"),
        *load_state_file("data/minecraft/blockstates/black_concrete.json"),
        *load_state_file("data/minecraft/blockstates/brown_concrete.json"),
        *load_state_file("data/minecraft/blockstates/red_concrete.json"),
        *load_state_file("data/minecraft/blockstates/orange_concrete.json"),
        *load_state_file("data/minecraft/blockstates/yellow_concrete.json"),
        *load_state_file("data/minecraft/blockstates/lime_concrete.json"),
        *load_state_file("data/minecraft/blockstates/green_concrete.json"),
        *load_state_file("data/minecraft/blockstates/cyan_concrete.json"),
        *load_state_file("data/minecraft/blockstates/light_blue_concrete.json"),
        *load_state_file("data/minecraft/blockstates/blue_concrete.json"),
        *load_state_file("data/minecraft/blockstates/purple_concrete.json"),
        *load_state_file("data/minecraft/blockstates/magenta_concrete.json"),
        *load_state_file("data/minecraft/blockstates/pink_concrete.json"),
    ]


def load_leaves():
    return [
        *load_state_file("data/minecraft/blockstates/oak_leaves.json"),
        *load_state_file("data/minecraft/blockstates/spruce_leaves.json"),
        *load_state_file("data/minecraft/blockstates/birch_leaves.json"),
        *load_state_file("data/minecraft/blockstates/jungle_leaves.json"),
        *load_state_file("data/minecraft/blockstates/acacia_leaves.json"),
        *load_state_file("data/minecraft/blockstates/dark_oak_leaves.json"),
        *load_state_file("data/minecraft/blockstates/mangrove_leaves.json"),
        *load_state_file("data/minecraft/blockstates/cherry_leaves.json"),
        *load_state_file("data/minecraft/blockstates/azalea_leaves.json"),
        *load_state_file("data/minecraft/blockstates/flowering_azalea_leaves.json"),
    ]


def load_carpets():
    return [
        *load_state_file("data/minecraft/blockstates/white_carpet.json"),
        *load_state_file("data/minecraft/blockstates/light_gray_carpet.json"),
        *load_state_file("data/minecraft/blockstates/gray_carpet.json"),
        *load_state_file("data/minecraft/blockstates/black_carpet.json"),
        *load_state_file("data/minecraft/blockstates/brown_carpet.json"),
        *load_state_file("data/minecraft/blockstates/red_carpet.json"),
        *load_state_file("data/minecraft/blockstates/orange_carpet.json"),
        *load_state_file("data/minecraft/blockstates/yellow_carpet.json"),
        *load_state_file("data/minecraft/blockstates/lime_carpet.json"),
        *load_state_file("data/minecraft/blockstates/green_carpet.json"),
        *load_state_file("data/minecraft/blockstates/cyan_carpet.json"),
        *load_state_file("data/minecraft/blockstates/light_blue_carpet.json"),
        *load_state_file("data/minecraft/blockstates/blue_carpet.json"),
        *load_state_file("data/minecraft/blockstates/purple_carpet.json"),
        *load_state_file("data/minecraft/blockstates/magenta_carpet.json"),
        *load_state_file("data/minecraft/blockstates/pink_carpet.json"),
    ]


def load_glass_blocks():
    return [
        *load_state_file("data/minecraft/blockstates/glass.json"),
        *load_state_file("data/minecraft/blockstates/white_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/light_gray_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/gray_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/black_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/brown_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/red_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/orange_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/yellow_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/lime_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/green_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/cyan_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/light_blue_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/blue_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/purple_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/magenta_stained_glass.json"),
        *load_state_file("data/minecraft/blockstates/pink_stained_glass.json"),
    ]

def load_test_set():
    return [
        *load_state_file("data/minecraft/blockstates/azalea_leaves.json"),
        *load_state_file("data/minecraft/blockstates/oak_leaves.json"),
        *load_state_file("data/minecraft/blockstates/dirt.json"),
        *load_state_file("data/minecraft/blockstates/cobblestone_slab.json"),
        *load_state_file("data/minecraft/blockstates/dark_prismarine_slab.json"),
        *load_state_file("data/minecraft/blockstates/deepslate.json"),
        *load_state_file("data/minecraft/blockstates/stripped_birch_log.json"),
        *load_state_file("data/minecraft/blockstates/oak_leaves.json"),
        *load_state_file("data/minecraft/blockstates/oak_leaves.json"),
    ]


voxel_assets = [
    #*load_leaves(),

    # *load_colored_concrete(),

    #*load_slabs(),

    # *load_carpets(),

    # *load_glass_blocks(),

    *load_test_set(),

    #*load_state_file("data/minecraft/blockstates/dirt.json"),

    *load_state_file("data/minecraft/blockstates/air.json"),
]