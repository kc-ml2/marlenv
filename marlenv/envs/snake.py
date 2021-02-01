from collections import defaultdict
from enum import Enum
from typing import List

import numpy as np
import gym

from marlenv.core.grid_util import random_empty_points, random_empty_point, draw, make_grid
from marlenv.core.snake import Direction, Snake, Cell


class Action(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2


class Reward(Enum):
    FRUIT = 1.0
    KILL = 0.
    LOSE = -1.9
    WIN = 0.
    TIME = 0.


class SnakeEnv(gym.Env):
    def __init__(self, num_snakes=4, height=20, width=20, *args, **kwargs):
        self.num_snakes = num_snakes
        self.num_fruits = int(round(num_snakes * 0.8))
        self.grid_shape = (height, width)
        self.snakes: List[Snake]
        self.grid: np.ndarray

        low = 0
        high = 255
        self.action_space = gym.spaces.Discrete(len(list(Action)))
        self.observation_space = gym.spaces.Box(low, high, shape=self.grid_shape, dtype=np.uint8)

    def reset(self):
        self.grid = make_grid(*self.grid_shape, empty_value=Cell.EMPTY.value, wall_value=Cell.WALL.value)
        self.snakes = self._generate_snakes()
        for snake in self.snakes:
            self.grid[snake.head_point] = Cell.HEAD.value
            self.grid[snake.tail_point] = Cell.TAIL.value

        xs, ys = self._generate_fruits(self.num_fruits)
        self.grid[xs, ys] = Cell.FRUIT.value

    def render(self, mode='human'):
        # just for debugging
        SYM2CHR = {
            Cell.EMPTY.value: '.',
            Cell.WALL.value: '#',
            Cell.FRUIT.value: 'o',
            Cell.BODY.value: 'b',
            Cell.HEAD.value: 'H',
            Cell.TAIL.value: 't'
        }

        CHR2SYM = {v: k for k, v in SYM2CHR.items()}

        data = np.vectorize(SYM2CHR.get)(self.grid)
        data = [''.join(i) for i in data]
        data = '\n'.join(data)
        print(data)

    def step(self, actions):
        # O(3*num_snakes) + O(H*W) when generate fruits(maybe better approach, like random polling)
        if isinstance(actions, int):
            actions = [actions]
        # preprocess
        # head bang O(num_snakes), count duplicates by dictionary else it requires O(num_snakes^2)
        next_head_points = defaultdict(list)
        for snake, action in zip(self.snakes, actions):
            snake.direction = self._next_direction(snake.direction, action)
            next_head_points[snake.head_point + snake.direction].append(snake.idx)
            snake.alive, snake.reward = self._look_ahead(snake)

        dead_idxes = self._check_headbang(next_head_points)
        for idx in dead_idxes:
            self.snakes[idx].alive = False

        rews = []
        dones = []
        # postprocess
        for snake in self.snakes:
            self._update_grid(snake)

            rews.append(snake.reward)
            dones.append(not snake.alive)

        obs = self._encode(self.grid)

        return obs, rews, dones, None

    def _encode(self, obs):
        return obs

    def _look_ahead(self, snake):
        next_head_point = snake.head_point + snake.direction
        cell_value = self.grid[next_head_point]

        if cell_value == Cell.FRUIT.value:
            alive = True
            reward = Reward.FRUIT.value
        elif cell_value == Cell.EMPTY.value:
            alive = True
            reward = Reward.TIME.value
        else:
            alive = False
            reward = Reward.LOSE.value

        return alive, reward

    def _update_grid(self, snake):
        if snake.alive:
            self.grid[snake.head_point] = Cell.BODY.value
            # could be current or prev if ate fruit
            prev_tail_point = snake.move()
            self.grid[snake.head_point] = Cell.HEAD.value
            if prev_tail_point:
                self.grid[prev_tail_point] = Cell.EMPTY.value

            self.grid[snake.tail_point] = Cell.TAIL.value
        else:
            if draw(self.grid, snake.points, Cell.EMPTY.value) is False:
                print('draw faileds')

    def _check_headbang(self, next_head_points):
        dead_idxes = []
        for point, idxes in next_head_points.items():
            if len(idxes) > 1:
                dead_idxes.extend(idxes)

        return dead_idxes

    def _generate_snakes(self):
        snakes = []
        for idx in range(self.num_snakes):
            point = random_empty_point(self.grid)
            snakes.append(Snake(idx, point, Direction.RIGHT))

        return snakes

    def _generate_fruits(self, num_fruits=1):
        xs, ys = random_empty_points(self.grid, num_points=num_fruits)

        return xs, ys

    def _next_direction(self, direction, action):
        """
        0 == forward
        1 == left
        2 == right
        """
        if direction == Direction.UP:
            if action == 0:
                return Direction.UP
            elif action == 1:
                return Direction.LEFT
            else:
                return Direction.RIGHT
        elif direction == Direction.RIGHT:
            if action == 0:
                return Direction.RIGHT
            elif action == 1:
                return Direction.UP
            else:
                return Direction.DOWN
        elif direction == Direction.DOWN:
            if action == 0:
                return Direction.DOWN
            elif action == 1:
                return Direction.RIGHT
            else:
                return Direction.LEFT
        elif direction == Direction.LEFT:
            if action == 0:
                return Direction.LEFT
            elif action == 1:
                return Direction.DOWN
            else:
                return Direction.UP


env = SnakeEnv(height=10, width=10)
env.reset()
dones = [False] * 4
for _ in range(30):
    if not all(dones):
        env.render()
        _, _, dones, _ = env.step([0, 0, 0, 0])
