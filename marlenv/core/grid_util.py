from typing import List

import numpy as np
import copy
from PIL import Image

DOWN = (0, 1)
RIGHT = (1, 0,)
UP = (0, -1)
LEFT = (-1, 0)
SHIFTS = [DOWN, RIGHT, UP, LEFT]


def make_grid(height, width, empty_value=0, wall_value=1) -> np.ndarray:
    grid = np.full((height, width), fill_value=empty_value)
    # construct wall
    grid[[0, -1]] = wall_value
    grid[:, [0, -1]] = wall_value

    return grid


def make_grid_from_txt(map_path: str, mapper: dict) -> np.ndarray:
    with open(map_path, 'r') as fp:
        lines = fp.read().split('\n')

    ls = []
    for line in lines:
        # split string into list of char
        ls.append([mapper[c] for c in list(line)])
    grid = np.asarray(ls)

    return grid


def find_k_consec(grid: np.ndarray, k: int):
    # Returns list of possible k consecutive cells from given grid
    empty_mask = (grid == 0)
    answers = []
    for r in range(grid.shape[0]):
        start = None
        end = None
        for rc in range(grid.shape[1]):
            if empty_mask[r, rc]:
                if start is None:
                    start = rc
                else:
                    end = rc
                if end - start == k + 1:
                    answers.append([(r, start), (r, end)])
                    start += 1
            else:
                start = None
                end = None
    for c in range(grid.shape[1]):
        start = None
        end = None
        for cr in range(grid.shape[0]):
            if empty_mask[r, rc]:
                if start is None:
                    start = cr
                else:
                    end = cr
                if end - start == k + 1:
                    answers.append([(start, c), (end, c)])
                    start += 1
            else:
                start = None
                end = None
    return answers


def dfs_sweep_empty(grid: np.ndarray, k: int):
    answers = []
    empty_mask = grid == 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if empty_mask[(r, c)]:
                answers.extend(_dfs_helper(empty_mask, (r, c), [], k))
    return answers


def _dfs_helper(grid: np.ndarray, node: tuple, history: List[tuple], k: int):
    history.append(node)
    answers = []
    if len(history) == k:
        answers.append(history)
    else:
        for shift in SHIFTS:
            # Get shifted rows and colums
            rs, cs = node[0] + shift[0], node[1] + shift[1]
            candidate = (rs, cs)
            if (_inbound(candidate, grid)
               and candidate not in history and grid[candidate]):
                if not _head_blocked(grid, history, candidate):
                    answers.extend(
                        _dfs_helper(grid, candidate, copy.deepcopy(history), k)
                    )
    return answers


def _head_blocked(mask: np.ndarray, history: List[tuple], extra_node: tuple):
    check_status = 0
    first_node = history[0]
    for shift in SHIFTS:
        node = first_node[0] + shift[0], first_node[1] + shift[1]
        if (mask[node] == 0 or node in history or node == extra_node
           or not _inbound(node, mask)):
            check_status += 1
    return check_status == len(SHIFTS)


def _inbound(node, grid):
    r, c = node
    return r >= 0 and c >= 0 and r < grid.shape[0] and c < grid.shape[1]


def random_empty_coord(grid: np.ndarray):
    # has to sweep entire grid O(H*W)
    xs, ys = np.where(grid == 0)
    idx = np.random.randint(0, len(xs) - 1)

    return xs[idx], ys[idx]


def random_empty_coords(grid, num_coords: int):
    xs, ys = np.where(grid == 0)
    if len(xs) == 0:
        return None, None
    idxes = np.random.randint(0, len(xs), size=num_coords)
    # coords = np.stack(xs[idxes], ys[idxes]).T

    return xs[idxes], ys[idxes]


def poll_empty_coord(grid):
    h, w = grid.shape
    # max index - wall * 2
    x, y = 0, 0
    while grid[x, y] != 0:
        x = np.random.randint(0, h - 1)
        y = np.random.randint(0, w - 1)

    # +1 to omit wall
    return x, y


def draw(grid, coords: List[tuple], value: int):
    h, w = grid.shape
    xs, ys = [], []
    for p in coords:
        xs.append(p[0])
        ys.append(p[1])

    # wall
    if (0 in xs) or ((h - 1) in xs) or (0 in ys) or ((w - 1) in ys):
        return False

    grid[xs, ys] = value

    return True


def rgb_from_grid(grid, enum, color_dict):
    rgb_array = np.zeros((*grid.shape, 3), dtype=np.uint8)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            cell_value = grid[r, c] % 10
            cell_id = grid[r, c] // 10
            color_list = color_dict[enum(cell_value).value]
            cell_color = np.array(color_list[cell_id % len(color_list)])
            cycle = cell_id // len(color_list)
            rgb_array[r, c] = (cell_color * 0.7**cycle).astype(np.uint8)

    return rgb_array


def image_from_grid(grid, enum, color_dict, max_size=300):
    bigger = max(list(grid.shape))
    scale = max(max_size // bigger, 1)
    rgb_array = rgb_from_grid(grid, enum, color_dict)
    rgb_array = np.repeat(np.repeat(rgb_array, scale, axis=0), scale, axis=1)
    image = Image.fromarray(rgb_array, 'RGB')

    return image
