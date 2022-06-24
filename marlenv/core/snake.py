from collections import deque
from enum import Enum


class Cell(Enum):
    EMPTY = 0
    WALL = 1
    FRUIT = 2
    HEAD = 3
    BODY = 4
    TAIL = 5


_color_wheel = [(104, 255, 0), (255, 191, 0), (255, 0, 92), (0, 111, 255)]


CellColors = {
    Cell.EMPTY.value: [(0, 0, 0)],
    Cell.WALL.value: [(32, 32, 32)],
    Cell.FRUIT.value: [(223, 7, 22)],
    Cell.HEAD.value: _color_wheel,
    Cell.BODY.value: _color_wheel,
    Cell.TAIL.value: _color_wheel
}


class Direction(Enum):
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)

    """
    little magic for convenience, coord + direction
    """

    def __radd__(self, other):
        dr, dc = self.value
        return other[0] + dr, other[1] + dc

    def __rsub__(self, other):
        dr, dc = self.value
        return other[0] - dr, other[1] - dc


class Snake:
    def __init__(self, idx, coords):
        assert len(coords) > 1
        self.idx: int = idx
        self.head_coord: tuple = coords[0]
        self.tail_coord: tuple = coords[-1]
        self.direction: Direction = Direction(
            (coords[0][0] - coords[1][0],
            coords[0][1] - coords[1][1])
        )
        prev_coord = self.head_coord
        direction_list = []
        for next_coord in coords[1:]:
            direction = Direction(
                (prev_coord[0] - next_coord[0], prev_coord[1] - next_coord[1])
            )
            direction_list.append(direction)
            prev_coord = next_coord

        self.directions = deque(direction_list)

        self.alive = True
        self._reset_reward_state()

    def __len__(self):
        return len(self.directions + 1)

    def _reset_reward_state(self):
        self.fruit = False
        self.death = False
        self.kills = 0
        self.win = False
        self.reward = 0.

    @property
    def coords(self):
        coord = self.head_coord
        coords = [coord]
        for direction in self.directions:
            coord -= direction
            coords.append(coord)

        return coords

    def move(self):
        self.head_coord += self.direction
        self.directions.appendleft(self.direction)

        prev_tail_coord = None
        if not self.fruit:
            prev_tail_coord = self.tail_coord
            tail_direction = self.directions.pop()
            self.tail_coord += tail_direction
        self._reset_reward_state()

        return prev_tail_coord
