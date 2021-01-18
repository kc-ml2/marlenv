"""modified from YuriyGuts/snake-ai-reinforcement"""
from collections import deque

from marlenv.envs.ml2_python.common import Direction


class Python:
    def __init__(self, position, direction, length=3):
        self.direction = direction
        self.directions = [
            Direction.NORTH,
            Direction.EAST,
            Direction.SOUTH,
            Direction.WEST,
        ]
        self.alive = True
        self.on_field = True
        self.body = deque([position])
        for _ in range(length - 1):
            self.grow()

    @property
    def head(self):
        return self.body[0]

    @property
    def tail(self):
        return self.body[-1]

    @property
    def length(self):
        return len(self.body)

    @property
    def next(self):
        return self.head + self.direction

    def turn_left(self):
        idx = self.directions.index(self.direction)
        self.direction = self.directions[idx - 1]

    def turn_right(self):
        idx = self.directions.index(self.direction)
        self.direction = self.directions[(idx + 1)%len(self.directions)]

    def grow(self):
        self.body.appendleft(self.next)

    def move(self):
        self.body.appendleft(self.next)
        self.body.pop()
