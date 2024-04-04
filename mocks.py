import numpy as np
from dlutils.data import DataHandler

### 2D Mocks ###

def random_unit_vector_2d(num_vectors):
    vec = np.random.randn(num_vectors, 2)
    vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
    return vec


def get_random_orthog_vecs_2d(num_vectors):
    i = random_unit_vector_2d(num_vectors)
    j = np.stack([-i[:, 1], i[:, 0]], axis=1)
    return i, j



# def add_triangle_to_grid(size, a, b, num_triangles):
#     grid = np.zeros((size, size), dtype=int)
#     x1, y1 = np.random.randint(0, size, (2, num_triangles))
#     point_1 = np.stack([x1, y1], axis=1)
#
#     direction_2, direction_3 = get_random_orthog_vecs_2d(num_triangles)
#
#     point_2 = point_1 + (a * direction_2).astype(int)
#     point_3 = point_1 + (b * direction_3).astype(int)
#
#     for p in [point_1, point_2, point_3]:
#         p = p % size
#         np.add.at(grid, tuple(p.T), 1)
#
#     return grid


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


def make_2d_mocks(num_mocks, size, a, b, num_triangles):
    all_mocks = np.zeros((num_mocks, size, size))

    for i in range(num_mocks):
        resulting_grid = add_triangle_to_grid(size, a, b, num_triangles)
        all_mocks[i] = resulting_grid

    return all_mocks


def make_triangle_mocks(num_mocks, size, a, b, num_triangles):
    all_mocks = np.zeros((num_mocks, size, size))  # Change to 2D mocks

    for i in range(num_mocks):
        resulting_grid = add_triangle_to_grid(size, a, b, num_triangles)
        all_mocks[i] = resulting_grid

    return all_mocks


def create_parity_violating_mocks_2d(num_mocks, field_size, total_num_triangles, ratio_left, length_side1, length_side2):
    num_left = round(total_num_triangles * ratio_left)
    num_right = round(total_num_triangles * (1 - ratio_left))

    fields_left = make_triangle_mocks(num_mocks, field_size, length_side1, length_side2, num_left)
    fields_right = make_triangle_mocks(num_mocks, field_size, length_side1, -length_side2, num_right)

    return fields_left + fields_right


def make_mock_dataloaders_2d(num_mocks, val_fraction, field_size, total_num_triangles, ratio_left, length_side1=4, length_side2=8, batch_size=64):
    mocks = create_parity_violating_mocks_2d(num_mocks, field_size, total_num_triangles, ratio_left, length_side1, length_side2)
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
