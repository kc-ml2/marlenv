from typing import List

import numpy as np


def make_grid(height, width, empty_value=0, wall_value=1):
    grid = np.full((height, width), fill_value=empty_value)
    # construct wall
    grid[[0, -1]] = wall_value
    grid[:, [0, -1]] = wall_value

    return grid


def make_grid_from_txt(map_path: str, mapper: dict):
    with open(map_path, 'r') as fp:
        lines = fp.read().split('\n')

    ls = []
    for line in lines:
        # split string into list of char
        l = list(line)
        ls.append([mapper[c] for c in l])
    data = np.asarray(ls)

    return data


def random_empty_point(grid):
    # has to sweep entire grid O(H*W)
    xs, ys = np.where(grid == 0)
    idx = np.random.randint(0, len(xs) - 1)

    return xs[idx], ys[idx]


def random_empty_points(grid, num_points: int):
    xs, ys = np.where(grid == 0)
    idxes = np.random.randint(0, len(xs) - 1, size=num_points)
    # points = np.stack(xs[idxes], ys[idxes]).T

    return xs[idxes], ys[idxes]


def poll_empty_point(grid):
    h, w = grid.shape
    # max index - wall * 2
    x, y = 0, 0
    while grid[x, y] != 0:
        x = np.random.randint(0, h - 1)
        y = np.random.randint(0, w - 1)

    # +1 to omit wall
    return x, y


def draw(grid, points: List[tuple], value: int):
    h, w = grid.shape
    xs, ys = [], []
    for p in points:
        xs.append(p[0])
        ys.append(p[1])

    # wall
    if (0 in xs) or ((h - 1) in xs) or (0 in ys) or ((w - 1) in ys):
        return False

    grid[xs, ys] = value

    return True
