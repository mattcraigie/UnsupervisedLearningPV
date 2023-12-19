import numpy as np
from datautils import DataHandler

def random_unit_vector_2d():
    vec = np.random.randn(2)  # Change 3 to 2 for 2D
    vec /= np.linalg.norm(vec)
    return vec


def get_random_orthog_vecs_2d():
    # Align i with the random unit vector
    i = random_unit_vector_2d()

    # Calculate j based on i
    j = np.array([-i[1], i[0]])  # Perpendicular vector in 2D

    return i, j


def add_triangle_to_grid(size, a, b, num_triangles):
    # a and b are the sizes of the triangle legs
    grid = np.zeros((size, size), dtype=int)  # Change to 2D grid
    for i in range(num_triangles):
        # Choose a random position for the first point
        x1, y1 = np.random.randint(0, size), np.random.randint(0, size)
        point_1 = np.array([x1, y1])

        direction_2, direction_3 = get_random_orthog_vecs_2d()

        # Calculate the position of the points
        point_2 = point_1 + (a * direction_2).astype(int)
        point_3 = point_1 + (b * direction_3).astype(int)

        # Add the points to the grid, wrapping if they go over
        for p in [point_1, point_2, point_3]:
            p = p % size
            grid[p[0], p[1]] += 1

    return grid


def make_triangle_mocks(num_mocks, size, a, b, num_triangles):
    all_mocks = np.zeros((num_mocks, size, size))  # Change to 2D mocks

    for i in range(num_mocks):
        resulting_grid = add_triangle_to_grid(size, a, b, num_triangles)
        all_mocks[i] = resulting_grid

    return all_mocks


def create_parity_violating_mocks(num_mocks, field_size, total_num_triangles, ratio_left, length_side1, length_side2):
    num_left = round(total_num_triangles * ratio_left)
    num_right = round(total_num_triangles * (1 - ratio_left))

    fields_left = make_triangle_mocks(num_mocks, field_size, length_side1, length_side2, num_left)
    fields_right = make_triangle_mocks(num_mocks, field_size, length_side1, -length_side2, num_right)

    return fields_left + fields_right


def make_mock_dataloaders(num_mocks, val_fraction, field_size, total_num_triangles, ratio_left, length_side1=4, length_side2=8, batch_size=64):
    mocks = create_parity_violating_mocks(num_mocks, field_size, total_num_triangles, ratio_left, length_side1, length_side2)
    data_handler = DataHandler(mocks)
    train_loader, val_loader = data_handler.make_dataloaders(batch_size=64, val_fraction=val_fraction)
    return train_loader, val_loader
