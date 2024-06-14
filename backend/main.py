import base64
import io
import math
import random
import time
from functools import partial

import jax
import jax.numpy as jnp
from flask import Flask, request, jsonify
from flask_cors import CORS
from flaxmodels import VGG16
from jax import lax, vmap, jit
from jax.example_libraries import optimizers
from matplotlib import pyplot as plt, image as mpimg

from adapters.minecraft_importer import voxel_assets

print(jax.version)

jax.config.update('jax_backend_target', 'cpu')

epsilon = 1e-6


def find_intersections(ray_origin, ray_direction, vertices, texture_coords, faces, possible_textures):
    intersection_colors = jnp.zeros((faces.shape[0], possible_textures.shape[0] * 4), dtype=jnp.uint8)
    intersection_distances = jnp.zeros((faces.shape[0], 1), dtype=jnp.float32)

    def append_intersection(i, t, u, v, face):
        # Get texture coordinates of the intersection point
        uv = texture_coords[face[:, 1]]
        barycentric_coords = jnp.array([1 - u - v, u, v])
        interpolated_uv = jnp.sum(uv * barycentric_coords[:, None], axis=0)

        # Get color from the texture using nearest neighbor interpolation
        tex_w, tex_h = possible_textures.shape[1:3]
        tex_x = jnp.clip(jnp.round(interpolated_uv[0] * (tex_w - 1)), 0, tex_w - 1).astype(int)
        tex_y = jnp.clip(jnp.round(interpolated_uv[1] * (tex_h - 1)), 0, tex_h - 1).astype(int)

        # Parallel texture lookup using JAX's vectorized operations
        colors = possible_textures[:, tex_x, tex_y, :]

        return intersection_colors.at[i].set(jnp.reshape(colors, (-1,))), intersection_distances.at[i].set(t)

    def do_nothing(i, t, u, v, face):
        return intersection_colors, intersection_distances

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

        intersection_colors, intersection_distances = lax.cond(is_invalid_intersection,
                                                               lambda _: do_nothing(i, t, u, v, face),
                                                               lambda _: append_intersection(i, t, u, v, face),
                                                               operand=None)

    return intersection_colors, intersection_distances


@jit
def get_ray_colour(ray_origin, ray_direction, vertices, texture_coords, faces, possible_textures):
    # Find intersection points with triangles
    intersection_colors, intersection_distances = find_intersections(
        ray_origin, ray_direction, vertices, texture_coords, faces, possible_textures
    )

    # Sort colors by distance
    intersection_colors_per_texture = intersection_colors.reshape(
        (faces.shape[0], possible_textures.shape[0], 4))
    indices = jnp.argsort(intersection_distances[:, -1])
    sorted_colors_per_texture = intersection_colors_per_texture[indices]

    sorted_alphas = sorted_colors_per_texture[..., 3:] / 255

    # Apply compute_final_color to each batch element
    transparency_factors = jnp.cumprod(1.0 - (sorted_alphas[:-1, ...]), axis=0)
    transparency_factors = jnp.concatenate(
        [jnp.ones((1, possible_textures.shape[0], 1)), transparency_factors], axis=0)

    opacity_factors = sorted_alphas * transparency_factors

    # Compute the weighted sum of colors and alpha values
    final_colors = jnp.sum(sorted_colors_per_texture * opacity_factors, axis=0)

    final_colors = jnp.clip(jnp.round(final_colors), 0, 255).astype(jnp.uint8)

    final_colors = final_colors.reshape(possible_textures.shape[0], 4)

    return final_colors


image_width = 200
image_height = 200

jax_random_key = jax.random.key(0)

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
    ).astype(jnp.int32)

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


def is_close_match(x, y):
    return x.shape == y.shape and jnp.all(jnp.abs(x - y) < 1e-6)


def get_model_groups():
    model_groups = []

    for asset in voxel_assets:
        i = 0
        while i < len(model_groups):
            if (is_close_match(model_groups[i][0][0], asset['model']['vertices'])
                    and is_close_match(model_groups[i][0][1], asset['model']['texture_coords'])
                    and is_close_match(model_groups[i][0][2], asset['model']['faces'])):
                break
            i += 1

        model_tuple = (asset['model']['vertices'], asset['model']['texture_coords'], asset['model']['faces'])
        texture = asset['texture']

        if i >= len(model_groups):
            model_groups.append([model_tuple, [texture]])
        else:
            model_groups[i][1].append(texture)

    for model_group in model_groups:
        model_group[1] = jnp.stack(model_group[1])

    return model_groups


model_groups = get_model_groups()


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

    unit_cube_origin = (voxel_local_position / voxel_size) - 0.5

    # Move the origin slightly along the negative direction to avoid self-intersection
    unit_cube_origin = unit_cube_origin - ray_direction * epsilon

    asset_colors = jnp.zeros((0, 4), dtype=jnp.uint8)

    for model_group in model_groups:
        model_tuple, possible_textures = model_group

        vertices, texture_coords, faces = model_tuple

        colours = get_ray_colour(unit_cube_origin, ray_direction, vertices, texture_coords, faces, possible_textures)

        asset_colors = jnp.concatenate([asset_colors, colours], axis=0)

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
    step = jnp.sign(ray_direction).astype(voxel_indices.dtype)

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
                     lambda: jnp.zeros([num_voxel_assets, 4], dtype=jnp.uint8),
                     )
        )
        contributing_voxel_matrix = lax.cond(valid_grid_indices(voxel_indices, subdivisions),
                                             lambda voxel_indices, step_counter: contributing_voxel_matrix.at[
                                                 step_counter].set(voxel_indices.astype(contributing_voxel_matrix.dtype)),
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

    empty_contributions = jnp.zeros((num_max_intersection, num_voxel_assets, 4), dtype=jnp.uint8)
    empty_contribution_map = jnp.full((num_max_intersection, 3), 0, dtype=jnp.uint8)

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

    image = jnp.zeros((image_height * image_width, num_max_intersection, num_voxel_assets, 4), dtype=jnp.uint8)
    index_map = jnp.zeros((image_height * image_width, num_max_intersection, 3), dtype=jnp.uint8)

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


# Load pre-trained VGG16 model
vgg16 = VGG16(output='logits', pretrained='imagenet', include_head=False)
params = vgg16.init(jax_random_key, jnp.zeros((1, image_width, image_height, 3), dtype=jnp.float32))


# Extract features from specific layers
@jit
def extract_features(image, jax_random_key):
    layers = ['conv1_2', 'conv2_2']

    alphas = image[..., 3:]
    rgb = image[..., :3]  # shape (batch_size, image_width, image_height, rgb)

    random_background_rgb = jax.random.uniform(jax_random_key, (rgb.shape[0], 1, 1, 3))

    perceptual_image = rgb * alphas + (1 - alphas) * random_background_rgb

    features = jnp.zeros((0,))
    _, utils = vgg16.apply(params, perceptual_image, capture_intermediates=True)

    intermediates = utils['intermediates']

    for layer_key in layers:
        layer_features = intermediates[layer_key]['__call__'][0]
        layer_features = layer_features.reshape((-1,))
        features = jnp.concatenate([features, layer_features], axis=-1)

    return features


# Perceptual loss function
def perceptual_loss(image_a, image_b, jax_random_key):
    loss = jnp.mean(jnp.square(extract_features(image_a, jax_random_key) - extract_features(image_b, jax_random_key)),
                    axis=-1)
    return loss


def squared_diff(image_a, image_b, jax_random_key):
    return jnp.square(image_a - image_b)


def loss_fn(rendered_image, original_image, jax_random_key):
    perception_loss = perceptual_loss(rendered_image, original_image, jax_random_key)

    return jnp.mean(perception_loss)


def temperature_annealing(epoch, warmup_epochs, initial_temperature, final_temperature, num_epochs):
    if epoch < warmup_epochs:
        # Warmup phase:
        current_temperature = initial_temperature
    else:
        # Exponential decay phase: decay temperature exponentially from peak to final value
        current_temperature = initial_temperature * (final_temperature / initial_temperature) ** (
                (epoch - warmup_epochs) / (num_epochs - warmup_epochs))

    return current_temperature


def optimization_step(grad_fn, caches, index_maps, original_images, opt_state, get_params, opt_update, step,
                      temperature):
    global jax_random_key
    jax_random_key, sub_key = jax.random.split(jax_random_key)

    voxel_grid_grad = grad_fn(get_params(opt_state), caches, index_maps, original_images, temperature, sub_key)

    opt_state = opt_update(step, voxel_grid_grad, opt_state)

    return opt_state


def compute_visual_loss_for(voxel_grid, caches, index_maps, original_images, temperature, jax_random_key):
    rendered_images = make_cache_to_image(caches, index_maps, voxel_grid, temperature, jax_random_key)
    loss = loss_fn(rendered_images, original_images, jax_random_key)
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
        print(f'{i + 1}/{num_views}')

    caches = jnp.stack(caches, axis=0)
    index_maps = jnp.stack(index_maps, axis=0)

    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(voxel_grid)

    grad_fn = jax.jit(jax.grad(compute_visual_loss_for, argnums=0))

    for step in range(num_iterations):

        step = jnp.array(step, dtype=jnp.uint16)

        batch_caches, batch_index_maps, batch_target_images = get_train_batch(caches, index_maps, original_images, 4)

        temperature = temperature_annealing(step, 0, 1.0, 0.1, num_iterations)

        start = time.time()
        opt_state = optimization_step(grad_fn, batch_caches, batch_index_maps, batch_target_images, opt_state,
                                      get_params,
                                      opt_update, step, temperature)
        end = time.time()
        print(f'step took: {end - start}')

        if step % 100 == 0:
            print(f'step {step}/{num_iterations}: temperature={temperature};')
            rendered_images = make_cache_to_image(batch_caches[:1], batch_index_maps[:1], get_params(opt_state),
                                                  temperature, jax_random_key)
            rendered_images_greedy = make_cache_to_image(batch_caches[:1], batch_index_maps[:1], get_params(opt_state),
                                                         0.0, jax_random_key)

            show_visual_comparison(
                [("Original", batch_target_images[0]), (f'Learned temperature={temperature}', rendered_images[0].astype(jnp.float32)),
                 (f'Learned temperature=0', rendered_images_greedy[0].astype(jnp.float32))])

    return voxel_grid


app = Flask(__name__)
CORS(app)


def transform_initial_voxel_grid(voxel_grid, last_asset_bias=8.0):
    # Check assumptions
    assert voxel_grid.ndim == 3, "Input voxel grid must be a 3D array"
    assert voxel_grid.dtype == bool, "Input voxel grid must be a bool array"
    n = voxel_grid.shape[0]
    assert voxel_grid.shape[1] == n and voxel_grid.shape[2] == n, "Input voxel grid must be a cube"

    # Generate random Gaussian matrix B
    B = jnp.zeros((n, n, n, num_voxel_assets), dtype=jnp.float32)

    # make the last voxel asset have prob logits that are much higher than the rest while not so high that backpropagation through a softmax function is hard.
    B = B.at[..., -1].add(last_asset_bias)

    test = jax.nn.softmax(B, axis=-1)

    return B


def show_visual_comparison(image_tuples, max_images_per_row=5):
    num_images = len(image_tuples)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row
    num_cols = min(num_images, max_images_per_row)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))  # Adjust the figsize as needed

    for i in range(num_images):
        image_label, image = image_tuples[i]
        row = i // max_images_per_row
        col = i % max_images_per_row

        if num_rows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]

        ax.imshow(image)
        ax.axis('off')
        ax.set_title(image_label, fontsize=12)  # Adjust the fontsize as needed

    # Remove any unused subplots
    for i in range(num_images, num_rows * num_cols):
        row = i // max_images_per_row
        col = i % max_images_per_row

        if num_rows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]

        ax.axis('off')

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()


def gumble_softmax(logits, temperature, jax_random_key):
    dtype = jnp.float16

    # Sample Gumbel noise
    gumbel_noise = jax.random.gumbel(jax_random_key, logits.shape, dtype=dtype)

    # Perturb logits with Gumbel noise and scale by temperature
    gumbels = (logits + gumbel_noise) / temperature

    # Softmax to get probabilities
    y_soft = jax.nn.softmax(gumbels.astype(dtype), axis=-1)

    return lax.cond(temperature < epsilon, lambda: jax.nn.one_hot(jnp.argmax(logits, axis=-1), logits.shape[-1], dtype=y_soft.dtype),
                    lambda: y_soft)


@jit
def make_cache_to_image(caches, index_maps, voxel_grid, temperature, jax_random_key):
    batch_size, image_width, image_height, _, _, _ = caches.shape

    # Apply Gumbel-Softmax
    voxel_logits = voxel_grid[index_maps[..., 0], index_maps[..., 1], index_maps[..., 2]]
    voxel_probs = gumble_softmax(voxel_logits, temperature, jax_random_key)

    # Compute the weighted sum of asset colors
    rgba = jnp.sum(caches * voxel_probs[..., None], axis=-2) / 255

    # Split rgba and alphas without reshaping
    alphas = rgba[..., 3:]

    # Apply compute_final_color to each batch element
    padded_alphas = jnp.concatenate([
        jnp.zeros((batch_size, image_width, image_height, 1, 1), dtype=alphas.dtype),
        alphas[..., :-1, :]
    ], axis=-2)
    transparency_factors = jnp.cumprod(1 - padded_alphas, axis=-2, dtype=alphas.dtype)

    opacity_factors = alphas * transparency_factors

    final_rgb = jnp.sum(rgba * opacity_factors, axis=-2)

    return final_rgb


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
        target_image = target_image[:, :, :4]

        camera_matrices.append(view_matrix)
        target_images.append(target_image)

    camera_matrices = jnp.stack(camera_matrices, axis=0)
    target_images = jnp.stack(target_images, axis=0)

    train(camera_matrices, target_images, voxel_grid, 20000, 0.05)

    return jsonify({'message': 'Data received successfully'})


def render_block(camera_view_matrix, model, texture):
    pixel_coords = jnp.meshgrid(jnp.arange(image_width), jnp.arange(image_height))
    pixel_coords = jnp.stack(pixel_coords, axis=-1).reshape(-1, 2)

    def render_fn(coords):
        origin, direction = spawn_ray(coords[0], coords[1], camera_view_matrix)
        colours = get_ray_colour(origin, direction, model['vertices'], model['texture_coords'], model['faces'],
                                 jnp.expand_dims(texture, axis=0))
        return colours[0]

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

    assets = voxel_assets
    rendered_assets = []
    for asset in assets:
        image = render_block(camera_view_matrix, asset['model'], asset['texture'])
        rendered_assets.append((asset['name'], image))

    show_visual_comparison(rendered_assets)
    pass


if __name__ == '__main__':
    render_a_block()
    app.run(port=5000)
