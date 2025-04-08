import numpy as np
from dlutils.data import DataHandler

# ------ Triangle Mocks ------ #

def random_unit_vector_2d(num_vectors):
    vec = np.random.randn(num_vectors, 2)
    vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
    return vec


def get_random_orthog_vecs_2d(num_vectors):
    i = random_unit_vector_2d(num_vectors)
    j = np.stack([-i[:, 1], i[:, 0]], axis=1)
    return i, j

def add_triangle_to_grid(size, a, b, num_triangles):
    grid = np.zeros((size, size), dtype=int)

    # Generate floating point coordinates for the first point of the triangles
    x1, y1 = np.random.uniform(0, size, (2, num_triangles))  # just make it num_triangles, 2 in the first place?
    point_1 = np.stack([x1, y1], axis=1)

    direction_2, direction_3 = get_random_orthog_vecs_2d(num_triangles)

    point_2 = point_1 + a * direction_2
    point_3 = point_1 + b * direction_3

    for p in [point_1, point_2, point_3]:
        # Map the points to the nearest grid points
        p_grid = np.round(p).astype(int) % size
        np.add.at(grid, tuple(p_grid.T), 1)

    return grid


def make_triangle_mocks(num_mocks, size, a, b, num_triangles, min_scale=1.0, max_scale=1.0):
    # Validate scale inputs
    try:
        min_scale = float(min_scale)
        max_scale = float(max_scale)
        if min_scale > max_scale or min_scale <= 0 or max_scale <= 0:
            raise ValueError
    except (ValueError, TypeError):
        min_scale = max_scale = 1.0  # fallback to default

    all_mocks = np.zeros((num_mocks, size, size))  # 2D mocks

    for i in range(num_mocks):
        scale = np.random.uniform(min_scale, max_scale)
        scaled_a = a * scale
        scaled_b = b * scale
        resulting_grid = add_triangle_to_grid(size, scaled_a, scaled_b, num_triangles)
        all_mocks[i] = resulting_grid

    return all_mocks


def create_triangle_mock_set_2d(num_mocks, field_size, total_num_triangles, ratio_left, length_side1, length_side2,
                                 min_scale=1.0, max_scale=1.0):

    num_left = round(total_num_triangles * ratio_left)
    num_right = round(total_num_triangles * (1 - ratio_left))

    fields_left = make_triangle_mocks(num_mocks, field_size, length_side1, length_side2, num_left, min_scale, max_scale)
    fields_right = make_triangle_mocks(num_mocks, field_size, length_side1, -length_side2, num_right, min_scale, max_scale)

    return fields_left + fields_right

# -------- SPIRAL MOCKS -------- #

def add_spiral_binary_once(size, scale, angle_range, num_points, handedness):
    """
    Draw a single spiral using binary (0/1), returned as a separate mask.
    """
    mask = np.zeros((size, size), dtype=int)

    x0, y0 = np.random.uniform(0, size, 2)
    theta0 = np.random.uniform(0, 2 * np.pi)
    theta = np.linspace(0, angle_range, num_points) * handedness + theta0
    r = scale * (theta - theta0)

    x_offset = r * np.cos(theta)
    y_offset = r * np.sin(theta)
    spiral_points = np.stack([x0 + x_offset, y0 + y_offset], axis=1)

    spiral_points_grid = np.round(spiral_points).astype(int) % size
    mask[spiral_points_grid[:, 0], spiral_points_grid[:, 1]] = 1

    return mask


def add_multiple_spirals_stack_binary_masks(size, scale, angle_range, num_points, num_spirals, handedness):
    """
    Sum multiple binary spiral masks to form a combined grid.
    """
    grid = np.zeros((size, size), dtype=int)
    for _ in range(num_spirals):
        mask = add_spiral_binary_once(size, scale, angle_range, num_points, handedness)
        grid += mask
    return grid


def create_spiral_mock_set_2d(
            num_mocks, field_size, total_num_spirals, ratio_left,
            scale, angle_range, num_points
    ):
        """
        Generate multiple 2D fields with a parity-violating mix of left- and right-handed spirals.

        Parameters:
            num_mocks         : int, number of mock fields to generate
            field_size        : int, size of each square grid
            total_num_spirals : int, total number of spirals per grid
            ratio_left        : float, fraction of spirals that are left-handed
            scale             : float, spiral scale factor
            angle_range       : float, angular extent of the spiral (in radians)
            num_points        : int, number of points along each spiral

        Returns:
            3D numpy array (num_mocks x field_size x field_size) with stacked spiral masks
        """
        num_left = round(total_num_spirals * ratio_left)
        num_right = total_num_spirals - num_left

        all_mocks = np.zeros((num_mocks, field_size, field_size), dtype=int)

        for i in range(num_mocks):
            grid_left = add_multiple_spirals_stack_binary_masks(field_size, scale, angle_range, num_points, num_left,
                                                                handedness=1)
            grid_right = add_multiple_spirals_stack_binary_masks(field_size, scale, angle_range, num_points, num_right,
                                                                 handedness=-1)
            all_mocks[i] = grid_left + grid_right

        return all_mocks

# --------- Main Functions --------- #

def create_parity_violating_mocks_2d(num_mocks, field_size, total_num, ratio_left, length_side1=None, length_side2=None,
                                     min_scale=1.0, max_scale=1.0, spirals=False, spiral_angle_range=None,
                                     spiral_scale=None,
                                     spiral_num_points=None, poisson_noise_level=None):
    if spirals:
        if spiral_scale is None:
            spiral_scale = 1.0
        if spiral_angle_range is None:
            spiral_angle_range = 2 * np.pi
        if spiral_num_points is None:
            spiral_num_points = 10
        mocks = create_spiral_mock_set_2d(num_mocks, field_size, total_num, ratio_left, spiral_scale,
                                          spiral_angle_range,
                                          spiral_num_points)
    else:

        mocks = create_triangle_mock_set_2d(num_mocks, field_size, total_num, ratio_left, length_side1, length_side2,
                                            min_scale, max_scale)

    if poisson_noise_level is not None:
        noisy_mocks = np.random.poisson(lam=mocks * poisson_noise_level).astype(int)
        return noisy_mocks
    return mocks


def make_mock_dataloaders_2d(num_mocks, val_fraction, mock_kwargs, batch_size=64):
    mocks = create_parity_violating_mocks_2d(num_mocks, **mock_kwargs)
    data_handler = DataHandler(mocks)
    train_loader, val_loader = data_handler.make_dataloaders(batch_size=batch_size, val_fraction=val_fraction)
    return train_loader, val_loader


### 3D Mocks ###


def random_unit_vector_3d(num_vectors):
    vec = np.random.randn(num_vectors, 3)
    vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
    return vec


def get_random_orthog_vecs_3d(num):
    # Get the first random vectors
    vector_1 = random_unit_vector_3d(num)

    # Get another set of random vectors that we will use to get an orthogonal second vector
    intermediate_vecs = random_unit_vector_3d(num)

    # Compute the cross products to get vectors orthogonal to vector_1
    vector_2 = np.cross(vector_1, intermediate_vecs)
    vector_2 /= np.linalg.norm(vector_2, axis=1)[:, np.newaxis]  # Normalize vector_2

    # Compute the cross product of v1 and v2 to get the third set of vectors
    vector_3 = np.cross(vector_1, vector_2)
    vector_3 /= np.linalg.norm(vector_3, axis=1)[:, np.newaxis]  # Normalize vector_3

    return vector_1, vector_2, vector_3


def add_tetrahedron_to_grid(size, a, b, c, num_tetrahedra):
    grid = np.zeros((size, size, size), dtype=int)
    point_1 = np.random.random((3, num_tetrahedra)) * size

    direction_2, direction_3, direction_4 = get_random_orthog_vecs_3d(num_tetrahedra)

    point_2 = point_1 + (a * direction_2)
    point_3 = point_1 + (b * direction_3)
    point_4 = point_1 + (c * direction_4)


    for p in [point_1, point_2, point_3, point_4]:
        p = p.astype(int)
        p = p % size

        np.add.at(grid, tuple(p.T), 1)

    return grid


def make_tetrahedron_mocks(num_mocks, size, a, b, c, num_tetrahedra):
    all_mocks = np.zeros((num_mocks, size, size, size))

    for i in range(num_mocks):
        resulting_grid = add_tetrahedron_to_grid(size, a, b, c, num_tetrahedra)
        all_mocks[i] = resulting_grid

    return all_mocks


def create_parity_violating_mocks_3d(num_mocks, field_size, total_num_tetrahedra, ratio_left, length_side1, length_side2, length_side3):

    num_left = round(total_num_tetrahedra * ratio_left)
    num_right = round(total_num_tetrahedra * (1 - ratio_left))

    fields_left = make_tetrahedron_mocks(num_mocks, field_size, length_side1, length_side2, length_side3, num_left)
    fields_right = make_tetrahedron_mocks(num_mocks, field_size, length_side1, length_side2, -length_side3, num_right)

    return fields_left + fields_right



### Point mocks ###

def make_tetrahedron_points(num_tetrahedra, size, a, b, c):
    x1, y1, z1 = np.random.random((3, num_tetrahedra)) * size
    point_1 = np.stack([x1, y1, z1], axis=1)

    direction_2, direction_3, direction_4 = get_random_orthog_vecs_3d(num_tetrahedra)

    point_2 = point_1 + (a * direction_2)
    point_3 = point_1 + (b * direction_3)
    point_4 = point_1 + (c * direction_4)

    tetrahedra_points = np.concatenate([point_1, point_2, point_3, point_4], axis=0) % size

    return tetrahedra_points


def create_parity_violating_point_mocks_3d(num_mocks, field_size, total_num_tetrahedra, ratio_left, length_side1,
                                     length_side2, length_side3):
    num_left = round(total_num_tetrahedra * ratio_left)
    num_right = round(total_num_tetrahedra * (1 - ratio_left))

    all_mocks = []

    for mock in range(num_mocks):
        points_left = make_tetrahedron_points(num_left, field_size, length_side1, length_side2, length_side3)
        points_right = make_tetrahedron_points(num_right, field_size, length_side1, length_side2, -length_side3)

        points_both = np.concatenate([points_left, points_right], axis=0)

        all_mocks.append(points_both)

    return np.stack(all_mocks, axis=0)
