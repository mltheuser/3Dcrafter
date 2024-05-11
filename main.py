import base64
import io
import math
import random
import time
from functools import partial

import jax
import jax.numpy as jnp
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from jax import lax, vmap, jit
from jax.example_libraries import optimizers
from matplotlib import pyplot as plt, image as mpimg

print(jax.version)


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


epsilon = 1e-6


def find_intersections(ray_origin, ray_direction, vertices, texture_coords, faces, texture):
    intersection_colors_with_distance = jnp.zeros((faces.shape[0], 4 + 1), dtype=jnp.float32)

    def append_intersection(i, t, u, v, face):
        # Get texture coordinates of the intersection point
        uv = texture_coords[face[:, 1]]
        barycentric_coords = jnp.array([1 - u - v, u, v])
        interpolated_uv = jnp.sum(uv * barycentric_coords[:, None], axis=0)

        # Get color from the texture using nearest neighbor interpolation
        tex_h, tex_w = texture.shape[:2]
        tex_x = jnp.clip(jnp.round(interpolated_uv[0] * (tex_w - 1)), 0, tex_w - 1).astype(int)
        tex_y = jnp.clip(jnp.round(interpolated_uv[1] * (tex_h - 1)), 0, tex_h - 1).astype(int)
        color = texture[tex_y, tex_x]

        color_with_distance = jnp.concatenate([color, jnp.expand_dims(t, axis=0)], axis=0)

        return intersection_colors_with_distance.at[i].set(color_with_distance)

    def do_nothing(i, t, u, v, face):
        return intersection_colors_with_distance

    for i in range(faces.shape[0]):
        face = faces[i]

        v0, v1, v2 = vertices[face[:, 0]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        ray_cross_e2 = jnp.cross(ray_direction, edge2)
        det = jnp.dot(edge1, ray_cross_e2)

        invalid_det = jnp.logical_and(det > -epsilon, det < epsilon)

        inv_det = 1.0 / det
        s = ray_origin - v0
        u = inv_det * jnp.dot(s, ray_cross_e2)

        invalid_u = jnp.logical_or(u < 0, u > 1)

        s_cross_e1 = jnp.cross(s, edge1)
        v = inv_det * jnp.dot(ray_direction, s_cross_e1)

        invalid_uv = jnp.logical_or(v < 0, u + v > 1)

        t = inv_det * jnp.dot(edge2, s_cross_e1)

        invalid_t = t < epsilon

        is_invalid_intersection = jnp.logical_or(invalid_det, invalid_u)
        is_invalid_intersection = jnp.logical_or(is_invalid_intersection, invalid_uv)
        is_invalid_intersection = jnp.logical_or(is_invalid_intersection, invalid_t)

        intersection_colors_with_distance = lax.cond(is_invalid_intersection,
                                                     lambda _: do_nothing(i, t, u, v, face),
                                                     lambda _: append_intersection(i, t, u, v, face),
                                                     operand=None)

    return intersection_colors_with_distance


@jit
def get_ray_colour(ray_origin, ray_direction, vertices, texture_coords, faces, texture):
    # Find intersection points with triangles
    intersection_colors_with_distance = find_intersections(
        ray_origin, ray_direction, vertices, texture_coords, faces, texture
    )

    # Sort colors by distance
    distances = intersection_colors_with_distance[:, -1]
    indices = jnp.argsort(distances)
    sorted_colors = intersection_colors_with_distance[indices]

    # Blend colors using alpha channel
    blended_color = jnp.zeros(4)

    for color in sorted_colors:
        blended_color = blend(blended_color, color[:-1])  # exclude distance from blending

    return blended_color


image_width = 512
image_height = 512

jax_random_key = jax.random.key(0)

verts, texs, faces = load_obj("models/cube.obj")
voxel_assets = [
    (verts, texs, faces, load_texture("models/azalea_leaves.png")),
    (jnp.zeros((0, 3)), jnp.zeros((0, 2)), jnp.zeros((0, 3, 2), dtype=jnp.int32), load_texture("models/dirt.png")),
    # (verts, texs, faces, load_texture("models/glass.png")),
    # (verts, texs, faces, load_texture("models/light_gray_stained_glass.png")),
    (verts, texs, faces, load_texture("models/dirt.png")),
    (verts, texs, faces, load_texture("models/cobblestone.png")),
]

num_voxel_assets = voxel_assets.__len__()


@jit
def spawn_ray(pixel_coords_x, pixel_coords_y, camera_view_matrix):
    camera_fov = 0.8

    # Extract the missing pieces from the view matrix (BABYLON JS uses the vM convention)
    camera_right = jnp.array(camera_view_matrix[0, :3])
    camera_up = jnp.array(camera_view_matrix[1, :3])
    camera_forward = jnp.array(camera_view_matrix[2, :3])
    camera_position = jnp.array(camera_view_matrix[3, :3])

    aspect_ratio = image_width / image_height
    px = (2.0 * ((pixel_coords_x + 0.5) / image_width) - 1.0) * jnp.tan(camera_fov / 2.0) * aspect_ratio
    py = (1.0 - 2.0 * ((pixel_coords_y + 0.5) / image_height)) * jnp.tan(camera_fov / 2.0)
    ray_direction = jnp.array([px, py, 1.0])

    # Rotate the ray direction based on the camera's orientation
    ray_direction = (
            ray_direction[0] * camera_right +
            ray_direction[1] * camera_up +
            ray_direction[2] * camera_forward
    )

    ray_direction = ray_direction / jnp.linalg.norm(ray_direction)

    return camera_position, ray_direction


def is_inside_unit_cube(point):
    return jnp.logical_and(jnp.all(point <= 0.5), jnp.all(point >= -0.5))


def convert_position_to_voxel_indices(position, subdivisions):
    voxel_indices = jnp.floor(
        (position + 0.5) / (1 / subdivisions)
    ).astype(int)

    return voxel_indices


def get_ray_origin(ray_origin, ray_direction):
    return ray_origin


def compute_intersection_point_from_outside(ray_origin, ray_direction):
    # Compute intersection points with each face
    t1 = (-0.5 - ray_origin) / ray_direction
    t2 = (0.5 - ray_origin) / ray_direction

    t = jnp.stack([t1, t2], axis=0)
    t_min = jnp.max(jnp.min(t, axis=0))
    t_max = jnp.max(jnp.max(t, axis=0))

    # Check for intersection
    no_intersection = t_max <= t_min

    # Compute intersection point
    intersection_point = ray_origin + t_min * ray_direction

    # Check if intersection point is within cube boundaries
    within_bounds = (jnp.abs(intersection_point) <= 0.5 + epsilon).all(axis=-1)

    # If not within bounds, set intersection point to sentinel value
    intersection_point = jnp.where(within_bounds & ~no_intersection,
                                   jnp.clip(intersection_point, -0.5 + epsilon, 0.5 - epsilon),
                                   jnp.array([-1.0, -1.0, -1.0]))

    return intersection_point


def get_first_intersection_with_voxel_grid(ray_origin, ray_direction):
    return lax.cond(is_inside_unit_cube(ray_origin), get_ray_origin,
                    compute_intersection_point_from_outside, ray_origin, ray_direction)


def getBackgroundColor(intersection_point, ray_direction, subdivisions):
    return jnp.zeros((num_voxel_assets, 4))


def get_subdivisions(voxel_grid):
    return jnp.int32(voxel_grid.shape[0])


def get_color_per_asset(voxel_indices, voxel_size, intersection_point, ray_direction):
    # Return the color contribution for the asset at the given voxel indices
    # using the intersection point and ray direction

    # transform ray_origin to unit cube
    # Currently the ray intersects a subvoxel of a voxelgrid that is centeres around 0,0,0 with a site length of 1 in all dimensions
    # The asset models are also scaled to this unit cube so in order to compute the intersection correctly we need to transform the subvoxel local origin to the unit cube.
    # Would we also need to do this for the direction I wonder?

    epsilon = 1e-5

    intersection_point = intersection_point - ray_direction * epsilon
    voxel_local_position = jnp.clip(intersection_point - (voxel_size * voxel_indices - 0.5), 0.0, voxel_size)

    #jax.debug.print("isVlaid: {c}, voxel_indices: {x}, voxel_calc: {y}, intersection_point: {z}, 🤯 {r} 🤯",
    #                c=valid_grid_indices(voxel_indices, 1 / voxel_size), x=voxel_indices,
    #                y=(voxel_size * voxel_indices - 0.5), z=intersection_point, r=voxel_local_position)

    unit_cube_origin = (voxel_local_position / voxel_size) - 0.5

    # ray_direction = jnp.zeros((3,)) - unit_cube_origin
    # ray_direction = ray_direction / jnp.linalg.norm(ray_direction)

    # Move the origin slightly along the negative direction to avoid self-intersection
    unit_cube_origin = unit_cube_origin - ray_direction * epsilon

    asset_colors = jnp.zeros((num_voxel_assets, 4))

    for i in range(num_voxel_assets):
        vertices, texture_coords, faces, texture_image = voxel_assets[i]

        colour = get_ray_colour(unit_cube_origin, ray_direction, vertices, texture_coords, faces, texture_image)

        asset_colors = asset_colors.at[i].set(colour)

    return asset_colors


def valid_grid_indices(voxel_indices, subdivisions):
    return jnp.all(jnp.logical_and(voxel_indices >= 0, voxel_indices < subdivisions))


def cond_fun(carry):
    voxel_indices, _, _, color_contributions_in_visited_voxels, _ = carry
    subdivisions = color_contributions_in_visited_voxels.shape[0]
    return valid_grid_indices(voxel_indices, subdivisions)


def ray_voxel_traversal(intersection_point, ray_direction, subdivisions, empty_contributions, empty_contribution_map):
    # Convert the intersection point to the index of the hit voxel
    voxel_indices = convert_position_to_voxel_indices(intersection_point, subdivisions)

    voxel_size = 1 / subdivisions

    # Implement DDA algorithm for ray voxel traversal here:
    step = jnp.sign(ray_direction)

    step_between_0_and_1 = jnp.clip(step, 0, 1)

    t_max = jnp.where(
        step == 0,
        jnp.inf,
        ((voxel_indices + step_between_0_and_1) * voxel_size - 0.5 - intersection_point) / ray_direction
    )

    t_delta = jnp.where(
        step == 0,
        jnp.inf,
        voxel_size / jnp.abs(ray_direction)
    )

    def write_voxel_hit(color_contributions_in_visited_voxels, contributing_voxel_matrix, step_counter, voxel_indices,
                        local_intersection_point):
        color_contributions_in_visited_voxels = color_contributions_in_visited_voxels.at[step_counter].set(
            lax.cond(valid_grid_indices(voxel_indices, subdivisions),
                     lambda: get_color_per_asset(voxel_indices, voxel_size, local_intersection_point, ray_direction),
                     lambda: jnp.zeros([num_voxel_assets, 4]),
                     )
        )
        contributing_voxel_matrix = lax.cond(valid_grid_indices(voxel_indices, subdivisions),
                                             lambda voxel_indices, step_counter: contributing_voxel_matrix.at[
                                                 step_counter].set(voxel_indices),
                                             lambda voxel_indices, step_counter: contributing_voxel_matrix,
                                             voxel_indices, step_counter
                                             )
        return color_contributions_in_visited_voxels, contributing_voxel_matrix

    def body_fun(carry):
        voxel_indices, t_max, step_counter, color_contributions_in_visited_voxels, contributing_voxel_matrix = carry

        # Update voxel indices and t_max based on the minimum t_max component
        axis = jnp.argmin(t_max)
        voxel_indices = voxel_indices.at[axis].add(step[axis])

        local_intersection_point = intersection_point + ray_direction * t_max[axis]

        t_max = t_max.at[axis].add(t_delta[axis])

        color_contributions_in_visited_voxels, contributing_voxel_matrix = write_voxel_hit(
            color_contributions_in_visited_voxels, contributing_voxel_matrix, step_counter, voxel_indices,
            local_intersection_point
        )

        step_counter += 1

        return voxel_indices, t_max, step_counter, color_contributions_in_visited_voxels, contributing_voxel_matrix

    # Initialize the carry variables
    color_contributions_in_visited_voxels = empty_contributions
    contributing_voxel_matrix = empty_contribution_map

    # process initial intersection
    color_contributions_in_visited_voxels, contributing_voxel_matrix = write_voxel_hit(
        color_contributions_in_visited_voxels, contributing_voxel_matrix, 0, voxel_indices, intersection_point)

    step_counter = 1
    carry = (voxel_indices, t_max, step_counter, color_contributions_in_visited_voxels, contributing_voxel_matrix)

    # Run the while loop using lax.while_loop
    voxel_indices, t_max, step_counter, color_contributions_in_visited_voxels, contributing_voxel_matrix = lax.while_loop(
        cond_fun, body_fun,
        carry)

    return color_contributions_in_visited_voxels, contributing_voxel_matrix


def get_empty_contr(intersection_point, ray_direction, subdivisions, empty_contributions, empty_contribution_map):
    return empty_contributions, empty_contribution_map


def compute_num_max_intersections_for(subdivisions):
    return int(math.ceil(math.sqrt(subdivisions ** 2 + subdivisions ** 2)))


def trace_ray(ray_origin, ray_direction, subdivisions):
    intersection_point = get_first_intersection_with_voxel_grid(ray_origin, ray_direction)

    no_intersection = jnp.all(jnp.equal(intersection_point, jnp.array([-1.0, -1.0, -1.0])))

    num_max_intersection = compute_num_max_intersections_for(subdivisions)

    empty_contributions = jnp.zeros((num_max_intersection, num_voxel_assets, 4), dtype=jnp.float16)
    empty_contribution_map = jnp.full((num_max_intersection, 3), -1, dtype=jnp.int16)

    return lax.cond(no_intersection,
                    get_empty_contr,
                    ray_voxel_traversal,
                    intersection_point, ray_direction, subdivisions, empty_contributions, empty_contribution_map)

def render_pixel(x, y, camera_view_matrix, subdivisions):
    origin, direction = spawn_ray(x, y, camera_view_matrix)
    color = trace_ray(origin, direction, subdivisions)
    return color


def get_num_voxels(subdivisions):
    return subdivisions ** 3

@partial(jax.jit, static_argnums=1)
def render(camera_view_matrix, subdivisions):
    pixel_coords = jnp.meshgrid(jnp.arange(image_width), jnp.arange(image_height))
    pixel_coords = jnp.stack(pixel_coords, axis=-1).reshape(-1, 2)

    def render_fn(coords):
        return render_pixel(coords[0], coords[1], camera_view_matrix, subdivisions)

    num_max_intersection = compute_num_max_intersections_for(subdivisions)
    image = jnp.zeros((image_height * image_width, num_max_intersection, num_voxel_assets, 4), dtype=jnp.float32)
    index_map = jnp.zeros((image_height * image_width, num_max_intersection, 3), dtype=jnp.int32)

    def body_fn(i, val):
        coords = pixel_coords[i]
        image_part, index_map_part = render_fn(coords)
        image = val[0].at[i].set(image_part)
        index_map = val[1].at[i].set(index_map_part)
        return image, index_map

    image, index_map = lax.fori_loop(0, image_height * image_width, body_fn, (image, index_map))

    image = image.reshape(image_height, image_width, num_max_intersection, num_voxel_assets, 4)
    index_map = index_map.reshape(image_height, image_width, num_max_intersection, 3)
    return image, index_map


def loss_fn(rendered_image, original_image):
    # Compute and return the loss value between the rendered and original images
    # Compute the squared difference between the images
    squared_diff = jnp.square(rendered_image - original_image)

    # Calculate the mean of the squared differences
    mse = jnp.mean(squared_diff)

    return mse


def optimization_step(caches, index_maps, original_images, opt_state, get_params, opt_update, step):
    print("starting gradient compiling")
    grad_fn = jax.grad(compute_visual_loss_for, argnums=0)
    print("starting gradient computation")
    start_time = time.time()
    voxel_grid_grad = grad_fn(get_params(opt_state), caches, index_maps, original_images)
    end_time = time.time()
    print(f"make_cache_to_image took {end_time - start_time:.2f} seconds")
    print("Got Grad")
    opt_state = opt_update(step, voxel_grid_grad, opt_state)

    return opt_state


def compute_visual_loss_for(voxel_grid, caches, index_maps, original_images):
    rendered_images = make_cache_to_image(caches, index_maps, voxel_grid)
    loss = loss_fn(rendered_images, original_images)
    return loss


def get_train_batch(caches, index_maps, target_images, size: int):
    num_views = caches.shape[0]

    # Generate random indices to select views
    indices = jnp.array(random.sample(range(num_views), min(size, num_views)))

    # Select the camera matrices and target images for the batch
    batch_caches = caches[indices, ...]
    batch_index_maps = index_maps[indices, ...]
    batch_target_images = target_images[indices, ...]

    return batch_caches, batch_index_maps, batch_target_images


def train(camera_view_matrices, original_images, voxel_grid, num_iterations, learning_rate):
    caches = []
    index_maps = []

    num_views = camera_view_matrices.shape[0]

    for i in range(num_views):
        cache, index_map = render(camera_view_matrices[i], get_subdivisions(voxel_grid).item())
        caches.append(cache)
        index_maps.append(index_map)

    caches = jnp.stack(caches, axis=0)
    index_maps = jnp.stack(index_maps, axis=0)

    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(voxel_grid)

    for step in range(num_iterations):
        batch_caches, batch_index_maps, batch_target_images = get_train_batch(caches, index_maps, original_images, 4)

        opt_state = optimization_step(batch_caches, batch_index_maps, batch_target_images, opt_state, get_params,
                                      opt_update, step)

        if step % 10 == 0:
            rendered_images = make_cache_to_image(batch_caches[:1], batch_index_maps[:1], get_params(opt_state))
            show_visual_comparison(batch_target_images[0], rendered_images[0])

    return voxel_grid


app = Flask(__name__)
CORS(app)


def transform_initial_voxel_grid(voxel_grid, sigma=10.0):
    # Check assumptions
    assert voxel_grid.ndim == 3, "Input voxel grid must be a 3D array"
    assert voxel_grid.dtype == bool, "Input voxel grid must be a bool array"
    n = voxel_grid.shape[0]
    assert voxel_grid.shape[1] == n and voxel_grid.shape[2] == n, "Input voxel grid must be a cube"

    # Generate random Gaussian matrix B
    B = jax.random.uniform(jax_random_key, (n, n, n, num_voxel_assets), dtype=jnp.float32)

    return B


def show_visual_comparison(original_image, rendered_image):
    # Display original image with rendered image overlay, and rendered image separately
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Adjust the figsize as needed

    # Plot the original image
    axs[0].imshow(original_image)
    axs[0].axis('off')
    axs[0].set_title('Original Image with Rendered Overlay')

    # Plot the rendered image without transparency
    axs[1].imshow(rendered_image)
    axs[1].axis('off')
    axs[1].set_title('Rendered Image')

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()


def blend(current_color, local_color):
    # Extract the color components
    r, g, b, a = local_color

    # Calculate the remaining alpha
    remaining_alpha = 1.0 - current_color[3]

    # Apply the modified "over" operator
    result_color = jnp.array([
        r * a * remaining_alpha + current_color[0],
        g * a * remaining_alpha + current_color[1],
        b * a * remaining_alpha + current_color[2],
        a * remaining_alpha + current_color[3]
    ])

    return result_color


def compute_final_color(colors):
    final_color = jnp.array([0.0, 0.0, 0.0, 0.0])
    return lax.fori_loop(0, colors.shape[0], lambda i, final_color: blend(final_color, colors[i]), final_color)[:3]


def make_cache_to_image(caches, index_maps, voxel_grid):
    batch_size, image_width, image_height, max_num_intersections, num_options, _ = caches.shape

    # Create a mask for valid voxel indices
    valid_mask = jnp.all(index_maps >= 0, axis=-1)

    # Gather the voxel probabilities based on the index map
    voxel_indices = jnp.where(valid_mask[..., None], index_maps, 0)
    # apply softmax
    voxel_logits = voxel_grid[voxel_indices[..., 0], voxel_indices[..., 1], voxel_indices[..., 2]]
    voxel_probs = jax.nn.softmax(voxel_logits, axis=-1)

    # Compute the weighted sum of asset colors
    weighted_sum = jnp.sum(caches * voxel_probs[..., None], axis=-2)
    weighted_sum = weighted_sum.reshape(batch_size * image_width * image_height, -1, 4)

    # Apply compute_final_color to each batch element
    images = jax.vmap(compute_final_color)(weighted_sum)

    images = images.reshape(batch_size, image_width, image_height, 3)

    return images


@app.route('/builder', methods=['POST'])
def handle_builder_request():
    data = request.get_json()
    voxel_grid = transform_initial_voxel_grid(jnp.array(data['voxelGrid']))
    views = data['views']

    camera_matrices = []
    target_images = []

    for view in views:
        # Get camera matrix
        cam_data = view['camera']
        view_matrix = jnp.array(cam_data['viewMatrix'])

        # Decode base64 encoded image data
        image_data = view['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image_buf = io.BytesIO(image_bytes)
        target_image = mpimg.imread(image_buf, format='png')
        target_image = target_image[:, :, :3]

        camera_matrices.append(view_matrix)
        target_images.append(target_image)

    camera_matrices = jnp.stack(camera_matrices, axis=0)
    target_images = jnp.stack(target_images, axis=0)

    train(camera_matrices, target_images, voxel_grid, 200, 0.5)

    return jsonify({'message': 'Data received successfully'})


def render_block(camera_view_matrix):
    pixel_coords = jnp.meshgrid(jnp.arange(image_width), jnp.arange(image_height))
    pixel_coords = jnp.stack(pixel_coords, axis=-1).reshape(-1, 2)

    def render_fn(coords):
        origin, direction = spawn_ray(coords[0], coords[1], camera_view_matrix)
        colour = get_ray_colour(origin, direction, voxel_assets[0][0], voxel_assets[0][1], voxel_assets[0][2],
                                voxel_assets[0][3])
        return colour

    image = vmap(render_fn)(pixel_coords)
    image = image.reshape(image_height, image_width, 4)

    return image


def render_a_block():
    camera_view_matrix = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -0.1, 1.0, 0.0],
        [0.0, 1.0, -4.0, 1.0]
    ])

    camera_view_matrix = jnp.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, -0.1, 0.0, 0.0],
        [-4.0, 1.0, 0.0, 1.0]
    ])

    image = render_block(camera_view_matrix)

    show_visual_comparison(image, image)
    pass


if __name__ == '__main__':
    # render_a_block()
    app.run(port=5000)
