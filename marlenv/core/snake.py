from collections import deque
from enum import Enum


class Cell(Enum):
    EMPTY = 0
    FRUIT = 1
    WALL = 2
    HEAD = 3
    BODY = 4
    TAIL = 5


class Direction(Enum):
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)

    """
    little magic for convenience, coord + direction
    """
    def __radd__(self, other):
        dx, dy = self.value
        return other[0] + dx, other[1] + dy

    def __rsub__(self, other):
        dx, dy = self.value
        return other[0] - dx, other[1] - dy


# class Segment:
#     def __init__(self, coord, direction):
#         self.coord = coord
#         self.direction = direction

class Snake:
    def __init__(self, idx, head_coord, direction: Direction = Direction.RIGHT):
        self.idx: int = idx
        self.head_coord: tuple = head_coord
        # left coord of head for now
        self.tail_coord: tuple = (head_coord[0], head_coord[1] - 1)
        self.direction: Direction = direction
        self.directions = deque([direction, direction])

        self.alive = True
        self.reward = 0.

    def __len__(self):
        return len(self.directions)

    @property
    def coords(self):
        coord = self.head_coord
        coords = []
        for direction in self.directions:
            coords.append(coord)
            coord -= direction

        return coords

    def move(self):
        self.head_coord += self.direction
        self.directions.appendleft(self.direction)

        prev_tail_coord = None
        if self.reward != Cell.FRUIT.value:
            prev_tail_coord = self.tail_coord
            tail_direction = self.directions.pop()
            self.tail_coord += tail_direction

        return prev_tail_coord
