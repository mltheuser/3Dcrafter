import jax.numpy as jnp
from PIL import Image


def load_obj(file_path):
    vertices = []
    texture_coords = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = [float(x) for x in line.split()[1:]]
                vertices.append(vertex)
            elif line.startswith('vt '):
                tex_coord = [float(x) for x in line.split()[1:]]
                texture_coords.append(tex_coord)
            elif line.startswith('f '):
                face = []
                for vertex_data in line.split()[1:]:
                    vertex_indices = vertex_data.split('/')
                    vertex_index = int(vertex_indices[0]) - 1
                    texture_index = int(vertex_indices[1]) - 1
                    face.append((vertex_index, texture_index))
                faces.append(face)

    vertices = jnp.array(vertices, dtype=jnp.float32)
    texture_coords = jnp.array(texture_coords, dtype=jnp.float32)
    faces = jnp.array(faces, dtype=jnp.int32)

    return vertices, texture_coords, faces


def load_texture(file_path):
    image = Image.open(file_path)
    image = image.convert('RGBA')
    texture = jnp.array(image, dtype=jnp.float32) / 255
    return texture


def texture_variant(asset_tuple, texture):
    model_tuple, _ = asset_tuple
    return (model_tuple, texture)


def translation_variant(asset_tuple, translation_vec=jnp.zeros((3,)), rotation_mat=jnp.eye(3)):
    model_tuple, texture = asset_tuple
    vertices, texture_coords, faces = model_tuple

    # apply transformation matrix to (n, 3) shaped vertecies array
    vertices = jnp.matmul(vertices, rotation_mat) + translation_vec[None, :]

    return ((vertices, texture_coords, faces), texture)


air = (jnp.zeros((0, 3)), jnp.zeros((0, 2)), jnp.zeros((0, 3, 2), dtype=jnp.int32))

cube = (load_obj("models/cube.obj"), jnp.ones((1, 1, 4)))
slab = (load_obj("models/slab.obj"), jnp.ones((1, 1, 4)))
stairs = (load_obj("models/stairs.obj"), jnp.ones((1, 1, 4)))


voxel_assets = [
    texture_variant(cube, load_texture("models/azalea_leaves.png")),
    texture_variant(cube, load_texture("models/dirt.png")),
    texture_variant(cube, load_texture("models/cobblestone.png")),
    texture_variant(cube, load_texture("models/dark_prismarine.png")),
    texture_variant(cube, load_texture("models/deepslate.png")),
    texture_variant(cube, load_texture("models/stripped_birch_log.png")),

    texture_variant(slab, load_texture("models/dark_prismarine.png")),
    translation_variant(texture_variant(slab, load_texture("models/dark_prismarine.png")), jnp.array([0.0, 0.5, 0.0])),

    #texture_variant(stairs, load_texture("models/cobblestone.png")),

    (air, jnp.zeros((1, 1, 4))),
]
