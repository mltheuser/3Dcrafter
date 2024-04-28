import base64
import io

import jax
import jax.numpy as jnp
from flask import Flask, request, jsonify
from flask_cors import CORS
from jax import lax, vmap, jit
from matplotlib import pyplot as plt, image as mpimg

epsilon = 1e-6

image_width = 250
image_height = 250


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
    jax.debug.print("Should not happen")
    jax.debug.print("{x}", x=ray_origin)
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


def getBackgroundColor(intersection_point, ray_direction, voxel_grid):
    return jnp.zeros(4)


def compute_voxel_color(voxel_indices, voxel_grid):
    return voxel_grid[voxel_indices[0], voxel_indices[1], voxel_indices[2]]


def ray_stopped_by_occlusion(current_color):
    return current_color[3] > 0.99


def blend(current_color, local_color):
    alpha_blend = current_color[3] * (1.0 - local_color[3])
    new_r = current_color[0] * alpha_blend + local_color[0] * local_color[3]
    new_g = current_color[1] * alpha_blend + local_color[1] * local_color[3]
    new_b = current_color[2] * alpha_blend + local_color[2] * local_color[3]
    new_a = current_color[3] * (1.0 - local_color[3]) + local_color[3]
    return jnp.array([new_r, new_g, new_b, new_a])


def get_subdivisions(voxel_grid):
    return voxel_grid.shape[0]


def ray_voxel_traversal(intersection_point, ray_direction, voxel_grid):
    subdivisions = get_subdivisions(voxel_grid)

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

    def valid_grid_indices(voxel_indices):
        return jnp.all(jnp.logical_and(voxel_indices >= 0, voxel_indices < subdivisions))

    def cond_fun(carry):
        voxel_indices, t_max, color = carry
        return jnp.logical_and(jnp.logical_not(ray_stopped_by_occlusion(color)),
                               valid_grid_indices(voxel_indices))

    def body_fun(carry):
        voxel_indices, t_max, color = carry

        # Update voxel indices and t_max based on the minimum t_max component
        axis = jnp.argmin(t_max)
        voxel_indices = voxel_indices.at[axis].add(step[axis])
        t_max = t_max.at[axis].add(t_delta[axis])

        def get_updated_ray_color(current_color):
            local_color = compute_voxel_color(voxel_indices, voxel_grid)

            return blend(current_color, local_color)

        color = lax.cond(valid_grid_indices(voxel_indices), get_updated_ray_color, lambda current_color: current_color,
                         color)

        return voxel_indices, t_max, color

    # Initialize the carry variables
    color = compute_voxel_color(voxel_indices, voxel_grid)
    carry = (voxel_indices, t_max, color)

    # Run the while loop using lax.while_loop
    voxel_indices, t_max, color = lax.while_loop(cond_fun, body_fun, carry)

    return color


def trace_ray(ray_origin, ray_direction, voxel_grid):
    intersection_point = get_first_intersection_with_voxel_grid(ray_origin, ray_direction)

    no_intersection = jnp.all(jnp.equal(intersection_point, jnp.array([-1.0, -1.0, -1.0])))

    return lax.cond(no_intersection, getBackgroundColor, ray_voxel_traversal,
                    intersection_point, ray_direction, voxel_grid)


def render_pixel(x, y, camera_view_matrix, voxel_grid):
    origin, direction = spawn_ray(x, y, camera_view_matrix)
    color = trace_ray(origin, direction, voxel_grid)
    return color

@jit
def render(camera_view_matrix, voxel_grid):
    pixel_coords = jnp.meshgrid(jnp.arange(image_width), jnp.arange(image_height))
    pixel_coords = jnp.stack(pixel_coords, axis=-1).reshape(-1, 2)

    def render_fn(coords):
        return render_pixel(coords[0], coords[1], camera_view_matrix, voxel_grid)

    image = vmap(render_fn)(pixel_coords)[:, :3].reshape(image_height, image_width, 3)
    return image


def loss_fn(rendered_image, original_image):
    # Compute and return the loss value between the rendered and original images
    # Compute the squared difference between the images
    squared_diff = jnp.square(rendered_image - original_image)

    # Calculate the mean of the squared differences
    mse = jnp.mean(squared_diff)

    return mse


def optimization_step(camera_view_matrix, voxel_grid, original_image, learning_rate):
    print("starting gradient compiling")
    grad_fn = jax.grad(compute_visual_loss_for, argnums=1)
    print("starting gradient computation")
    voxel_grid_grad = grad_fn(camera_view_matrix, voxel_grid, original_image)
    print("Got Grad")
    voxel_grid -= learning_rate * voxel_grid_grad
    return voxel_grid

@jit
def compute_visual_loss_for(camera_view_matrix, voxel_grid, original_image):
    rendered_image = render(camera_view_matrix, voxel_grid)
    loss = loss_fn(rendered_image, original_image)
    return loss

def train(camera_view_matrix, voxel_grid, original_image, num_iterations, learning_rate):
    rendered_image = render(camera_view_matrix, voxel_grid)
    print(loss_fn(rendered_image, original_image))
    show_visual_comparison(original_image, rendered_image)

    for _ in range(num_iterations):
        voxel_grid = optimization_step(camera_view_matrix, voxel_grid, original_image, learning_rate)

        rendered_image = render(camera_view_matrix, voxel_grid)
        show_visual_comparison(original_image, rendered_image)

    return voxel_grid


def main():
    camera_view_matrix = jnp.array([
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0, 1.0]
    ])

    image = render(camera_view_matrix)

    # Display the image using Matplotlib
    # Save the image to a file
    plt.imshow(image)
    plt.imsave('white_image.png', image)
    plt.axis('off')
    plt.show(block=True)


app = Flask(__name__)
CORS(app)


def transform_initial_voxel_grid(voxel_grid):
    """
    Transform a 3D bool array of shape (n, n, n) into a 4D float array of shape (n, n, n, 4)
    where each voxel is assigned a random color with alpha 1.0 if the corresponding bool value is True.
    """
    # Check assumptions
    assert voxel_grid.ndim == 3, "Input voxel grid must be a 3D array"
    assert voxel_grid.dtype == bool, "Input voxel grid must be a bool array"
    n = voxel_grid.shape[0]
    assert voxel_grid.shape[1] == n and voxel_grid.shape[2] == n, "Input voxel grid must be a cube"

    # Create a random color array with shape (n, n, n, 3)
    random_colors = jax.random.uniform(jax.random.PRNGKey(0), shape=(n, n, n, 3), minval=0, maxval=1)

    # Create an alpha channel with shape (n, n, n, 1) filled with 1.0
    alpha_channel = jnp.ones((n, n, n, 1))

    # Combine the random colors and alpha channel to get the final output
    output = jnp.where(voxel_grid[..., None], jnp.concatenate((random_colors, alpha_channel), axis=-1), 0)

    return output


def show_visual_comparison(original_image, rendered_image):
    # Display original image with rendered image overlay, and rendered image separately
    fig, axs = plt.subplots(1, 2)

    # Plot the original image
    axs[0].imshow(original_image)

    axs[0].axis('off')
    axs[0].set_title('Original Image with Rendered Overlay')

    # Plot the rendered image without transparency
    axs[1].imshow(rendered_image)
    axs[1].axis('off')
    axs[1].set_title('Rendered Image')

    plt.show()


@app.route('/builder', methods=['POST'])
def handle_builder_request():
    data = request.get_json()
    voxel_grid = transform_initial_voxel_grid(jnp.array(data['voxelGrid']))
    views = data['views']

    for view in views:
        cam_data = view['camera']

        # Decode base64 encoded image data
        image_data = view['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image_buf = io.BytesIO(image_bytes)
        original_image = mpimg.imread(image_buf, format='png')
        original_image = original_image[:, :, :3]

        view_matrix = jnp.array(cam_data['viewMatrix'])

        rendered_image = render(view_matrix, voxel_grid)

        show_visual_comparison(original_image, rendered_image)

    return jsonify({'message': 'Data received successfully'})


if __name__ == '__main__':
    # main()
    app.run(port=5000)
