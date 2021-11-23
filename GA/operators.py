import numpy as np
from typing import List, Tuple, Optional, Set, Dict, Callable
from numba import njit

# BAD way to do this but I need to get this done :(
# from Warehouse.grid import CONFIG
opt_grid_start_x = 15
NO_STORAGE_LOCS = 560  # CONFIG["no_storage_locs"]


# def set_from_config(CONFIG):
#     global opt_grid_start_x
#     global NO_STORAGE_LOCS
#     opt_grid_start_x = None   # CONFIG["opt_grid_start_x"]
#     NO_STORAGE_LOCS = None  # CONFIG["no_storage_locs"]


@njit
def mutate_flip_bits(a: np.ndarray, p: float = 0.05) -> np.ndarray:
    return a ^ (np.random.random(a.shape) < p)


@njit
def one_point_crossover_2d(arr_1: np.ndarray, arr_2: np.ndarray,
                           crossover_point: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    c_pt = crossover_point
    # print(f"arr_1 no of storage locs: {np.count_nonzero(arr_1)}")
    # print(f"arr_2 no of storage locs: {np.count_nonzero(arr_2)}")
    c_arr_1 = arr_1.copy()
    c_arr_2 = arr_2.copy()

    tl_slices = (slice(None, c_pt[0]), slice(None, c_pt[0]))
    tr_slices = (slice(None, c_pt[0]), slice(c_pt[0], None))
    bl_slices = (slice(c_pt[0], None), slice(None, c_pt[0]))
    br_slices = (slice(c_pt[0], None), slice(c_pt[0], None))

    slices = None
    rand_quad = np.random.randint(0, 4)  # Randomly choose which quadrant to do crossover off
    if rand_quad == 0:
        slices = tl_slices
    elif rand_quad == 1:
        slices = tr_slices
    elif rand_quad == 2:
        slices = bl_slices
    else:
        slices = br_slices

    c_arr_1[slices] = arr_2[slices].copy()
    c_arr_2[slices] = arr_1[slices].copy()

    # print(f"c_arr_1 no of storage locs: {np.count_nonzero(c_arr_1)}")
    # print(f"c_arr_2 no of storage locs: {np.count_nonzero(c_arr_2)}")

    return c_arr_1, c_arr_2


@njit
def tiled_crossover(arr_1, arr_2, tile_size, no_tiles=1):
    assert arr_1.shape == arr_2.shape, "Arrays must be of same shape"

    max_y = arr_1.shape[0] - tile_size
    max_x = arr_1.shape[1] - tile_size
    new_arr_1 = arr_1.copy()
    new_arr_2 = arr_2.copy()

    for i in range(no_tiles):
        x = np.random.randint(0, max_x + 1)
        y = np.random.randint(0, max_y + 1)
        slices = (slice(y, y+tile_size), slice(x, x+tile_size))

        new_arr_1[slices] = arr_2[slices].copy()
        new_arr_2[slices] = arr_1[slices].copy()

    return new_arr_1, new_arr_2


@njit
def regain_no_storage_locs(arr: np.ndarray) -> np.ndarray:
    no_storage_locs = np.count_nonzero(arr)

    if no_storage_locs < NO_STORAGE_LOCS:
        while no_storage_locs < NO_STORAGE_LOCS:
            y = np.random.randint(0, arr.shape[0])
            x = np.random.randint(0, arr.shape[1])
            if arr[y, x] == 0:
                arr[y, x] = 1
                no_storage_locs += 1
    elif no_storage_locs > NO_STORAGE_LOCS:
        while no_storage_locs > NO_STORAGE_LOCS:
            y = np.random.randint(0, arr.shape[0])
            x = np.random.randint(0, arr.shape[1])
            if arr[y, x] == 1:
                arr[y, x] = 0
                no_storage_locs -= 1

    return arr


# Workaround since RUCK doesn't support njit func
def mate(arr_1: np.ndarray, arr_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return mate_njit(arr_1, arr_2)


@njit
def mate_njit(arr_1: np.ndarray, arr_2: np.ndarray, tile_size, tile_no=1) -> Tuple[np.ndarray, np.ndarray]:
    static_arr_1 = arr_1[:, :opt_grid_start_x]
    static_arr_2 = arr_2[:, :opt_grid_start_x]
    opt_arr_1 = arr_1[:, opt_grid_start_x:]
    opt_arr_2 = arr_2[:, opt_grid_start_x:]

    # c_pt = (opt_arr_1.shape[0]//2, opt_arr_1.shape[1]//2)
    # opt_c_arr_1, opt_c_arr_2 = one_point_crossover_2d(opt_arr_1, opt_arr_2, c_pt)

    opt_c_arr_1, opt_c_arr_2 = tiled_crossover(opt_arr_1, opt_arr_2, tile_size, tile_no)

    # Can't use concatenate cause of njit
    new_shape = (static_arr_1.shape[0], static_arr_1.shape[1] + opt_c_arr_1.shape[1])
    c_arr_1 = np.zeros(new_shape)
    c_arr_2 = np.zeros(new_shape)
    c_arr_1[:, :static_arr_1.shape[1]] = static_arr_1
    c_arr_1[:, static_arr_1.shape[1]:] = opt_c_arr_1
    c_arr_2[:, :static_arr_1.shape[1]] = static_arr_2
    c_arr_2[:, static_arr_1.shape[1]:] = opt_c_arr_2
    # c_arr_1 = np.concatenate([static_arr_1, opt_c_arr_1], axis=1)
    # c_arr_2 = np.concatenate([static_arr_2, opt_c_arr_2], axis=1)
    # return c_arr_1, c_arr_2
    return arr_1, arr_2


# # Workaround since RUCK doesn't support njit func
# def mutate(arr: np.ndarray) -> np.ndarray:
#     return mutate_njit(arr)


def mutate(arr: np.ndarray, p: float = 0.05) -> np.ndarray:
    static_arr = arr[:, :opt_grid_start_x]
    opt_arr = arr[:, opt_grid_start_x:]

    opt_arr = opt_arr.astype(bool)
    opt_arr = mutate_flip_bits(opt_arr, p=p)
    opt_arr = opt_arr.astype(int)

    # Need to retain number of storage locs
    opt_arr = regain_no_storage_locs(opt_arr)

    # Can't use concatenate cause of njit
    new_shape = (static_arr.shape[0], static_arr.shape[1] + opt_arr.shape[1])
    arr = np.zeros(new_shape)
    arr[:, :static_arr.shape[1]] = static_arr
    arr[:, static_arr.shape[1]:] = opt_arr

    return arr
