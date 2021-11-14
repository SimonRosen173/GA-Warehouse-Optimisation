import random
import os

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from Visualisation.vis import VisGrid

CONFIG = {
    # "opt_grid_start_x": 6,
    "no_storage_locs": 560,
    # "opt_grid_shape": (22, 44),
}

EMPTY = 0
PICKUP = 1
DROPOFF = 2
OBSTACLE = 3
NON_TASK_ENDPOINT = 4

# Init config
# CONFIG["no_opt_locs"] = CONFIG["opt_grid_shape"][0]*CONFIG["opt_grid_shape"][1]
# CONFIG["no_static_locs"] = CONFIG["opt_grid_shape"][0]*CONFIG["opt_grid_start_x"]


def print_grid(grid: list):
    for row in grid:
        row_str = " ".join([str(el) for el in row])
        print(row_str)


def txt_to_grid(file_name, simple_layout=False, use_curr_workspace=False):
    if use_curr_workspace:
        workspace_path = "\\".join(os.getcwd().split("\\")[:-1])
        file_name = workspace_path + "/Benchmark/maps/" + file_name

    grid = None
    with open(file_name) as f:
        curr_line = f.readline()
        width = len(curr_line) - 3  # Note that '\n' is included
        # print(width)
        grid = []
        while curr_line:
            curr_line = f.readline()
            if curr_line[1] == "#":
                break
            curr_row = []
            for i in range(1, len(curr_line)-2):
                if curr_line[i] == " ":
                    curr_row.append(0)
                else:
                    if simple_layout:
                        curr_row.append(1)
                    else:
                        curr_row.append(int(curr_line[i]))
            grid.append(curr_row)
    return grid


class RealWarehouse:
    def __init__(self):
        self.dropoff_locs = []
        self.non_task_endpoints = []

        self._static_grid_shape = (22, 14)
        self._opt_grid_shape = (22, 44)

        CONFIG["no_storage_locs"] = 560
        CONFIG["no_static_locs"] = self._static_grid_shape[0] * self._static_grid_shape[1]
        CONFIG["no_opt_locs"] = self._opt_grid_shape[0] * self._opt_grid_shape[1]
        CONFIG["opt_grid_start_x"] = self._static_grid_shape[1]+1

        self._static_grid = self.create_static_grid()

    def create_static_grid(self):
        static_grid = np.zeros(self._static_grid_shape, dtype=int)
        for i in range(7):
            static_grid[i*3 + 1: i*3 + 3, 1:4] = NON_TASK_ENDPOINT
            static_grid[i*3 + 1: i*3 + 3, 5:8] = NON_TASK_ENDPOINT
            static_grid[i*3 + 1: i*3 + 3, 10:12] = DROPOFF

        self.non_task_endpoints = []
        self.dropoff_locs = []
        for iy,ix in np.ndindex(self._static_grid_shape):
            if static_grid[iy, ix] == DROPOFF:
                self.dropoff_locs.append((iy, ix))
            elif static_grid[iy, ix] == NON_TASK_ENDPOINT:
                self.non_task_endpoints.append((iy, ix))

        return static_grid

    def create_grid(self):
        arr = np.zeros((22, 44), dtype=int)
        for y in range(7):
            for x in range(4):
                arr[y*3+1: y*3+3, x*11: x*11+10] = PICKUP

        out_grid = np.concatenate([self._static_grid, arr], axis=1)
        return out_grid


# Assuming fixed dims
class UniformRandomGrid:
    def __init__(self):
        self.dropoff_locs = []
        self.non_task_endpoints = []

        self._static_grid_shape = (22, 14)
        self._opt_grid_shape = (22, 44)

        CONFIG["no_storage_locs"] = 560
        CONFIG["no_static_locs"] = self._static_grid_shape[0] * self._static_grid_shape[1]
        CONFIG["no_opt_locs"] = self._opt_grid_shape[0] * self._opt_grid_shape[1]
        CONFIG["opt_grid_start_x"] = self._static_grid_shape[1]+1

        self._static_grid = self.create_static_grid()

    def create_static_grid(self):
        static_grid = np.zeros(self._static_grid_shape, dtype=int)
        for i in range(7):
            static_grid[i*3 + 1: i*3 + 3, 1:4] = NON_TASK_ENDPOINT
            static_grid[i*3 + 1: i*3 + 3, 5:8] = NON_TASK_ENDPOINT
            static_grid[i*3 + 1: i*3 + 3, 10:12] = DROPOFF

        self.non_task_endpoints = []
        self.dropoff_locs = []
        for iy,ix in np.ndindex(self._static_grid_shape):
            if static_grid[iy, ix] == DROPOFF:
                self.dropoff_locs.append((iy, ix))
            elif static_grid[iy, ix] == NON_TASK_ENDPOINT:
                self.non_task_endpoints.append((iy, ix))

        return static_grid

    def get_uniform_random_grid(self):
        # workspace_path = "\\".join(os.getcwd().split("\\")[:-1])`
        static_grid = self._static_grid
        shape = self._opt_grid_shape
        num_pickup_locs = CONFIG["no_storage_locs"]

        assert num_pickup_locs < shape[0] * shape[1], "num_pickup_locs must be less than number of elements in created grid"
        rand_grid = np.zeros(shape, dtype=int)
        y_len, x_len = shape[0], shape[1]
        curr_no_locs = 0
        while curr_no_locs < num_pickup_locs:
            y = np.random.randint(0, y_len)
            x = np.random.randint(0, x_len)
            if rand_grid[y][x] == 0:
                rand_grid[y][x] = 1
                curr_no_locs += 1
        out_grid = np.concatenate([static_grid, rand_grid], axis=1)
        return out_grid


# def get_uniform_random_grid(shape: Tuple[int, int], num_pickup_locs):
#     # workspace_path = "\\".join(os.getcwd().split("\\")[:-1])
#     abs_path = os.path.dirname(os.path.abspath(__file__))
#     file_name = abs_path + "/maps/droppoff_grid.txt"
#     static_grid = np.array(txt_to_grid(file_name))
#     assert num_pickup_locs < shape[0] * shape[1], "num_pickup_locs must be less than number of elements in created grid"
#     rand_grid = np.zeros(shape, dtype=int)
#     y_len, x_len = shape[0], shape[1]
#     curr_no_locs = 0
#     while curr_no_locs < num_pickup_locs:
#         y = np.random.randint(0, y_len)
#         x = np.random.randint(0, x_len)
#         if rand_grid[y][x] == 0:
#             rand_grid[y][x] = 1
#             curr_no_locs += 1
#     grid = np.concatenate([static_grid, rand_grid], axis=1)
#     # print(np.count_nonzero(grid))
#     return grid.tolist()


# def get_pickup_points(grid):
#     for y in range(grid):
#         for x in range(grid[0]):
#             pass
#
#
# def get_dropoff_points(grid):
#     pass


def get_rand_valid_point(grid):
    x, y = -1, -1
    valid_coord_found = False
    while not valid_coord_found:
        x = random.randint(0, len(grid[0])-1)
        y = random.randint(0, len(grid)-1)
        if grid[y][x] == 0:
            valid_coord_found = True
    return x, y


def test_urg():
    urg = UniformRandomGrid()
    grid_arr = urg.get_uniform_random_grid()
    vis_grid = VisGrid(grid_arr, (950, 400))
    vis_grid.save_to_png("imgs/test_urg")


def test_real():
    urg = RealWarehouse()
    grid_arr = urg.create_grid()
    vis_grid = VisGrid(grid_arr, (950, 400))
    vis_grid.save_to_png("imgs/test_real")


def main():
    pass


if __name__ == "__main__":
    # main()
    test_real()
