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

    def __radd__(self, other):
        dx, dy = self.value
        return other[0] + dx, other[1] + dy

    def __rsub__(self, other):
        dx, dy = self.value
        return other[0] - dx, other[1] - dy


# class Segment:
#     def __init__(self, point, direction):
#         self.point = point
#         self.direction = direction

class Snake:
    def __init__(self, idx, head_point, direction: Direction = Direction.RIGHT):
        self.idx: int = idx
        self.head_point: tuple = head_point
        # left point of head for now
        self.tail_point: tuple = (head_point[0], head_point[1] - 1)
        self.direction: Direction = direction
        self.directions = deque([direction, direction])

        self.alive = True
        self.reward = 0.

    def __len__(self):
        return len(self.directions)

    @property
    def points(self):
        point = self.head_point
        points = []
        for direction in self.directions:
            points.append(point)
            point -= direction

        return points

    def move(self):
        self.head_point += self.direction
        self.directions.appendleft(self.direction)

        prev_tail_point = None
        if self.reward != Cell.FRUIT.value:
            prev_tail_point = self.tail_point
            tail_direction = self.directions.pop()
            self.tail_point += tail_direction

        return prev_tail_point
