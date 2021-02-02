from collections import defaultdict
from typing import List, Tuple

import gym
import math
import numpy as np
from gym.utils import seeding

from marlenv.core.grid_util import random_empty_coords, random_empty_coord, draw, make_grid
from marlenv.core.snake import Direction, Snake, Cell


class SnakeEnv(gym.Env):
    default_action_dict = {
        'noop': 0,
        'left': 1,
        'right': 2,
    }
    action_keys = default_action_dict.keys()

    action_angle_dict = {
        0: 0.0,
        1: -math.pi/2.0,
        2: math.pi/2.0
    }

    default_reward_dict = {
        'fruit': 1.0,
        'kill': 0,
        'lose': -1.0,
        'win': 0.,
        'time': 0.,
    }
    reward_keys = default_reward_dict.keys()

    def __init__(
            self,
            num_snakes=4,
            height=20,
            width=20,
            *args,
            **kwargs
    ):
        """
        kwargs
        'reward_dict', 'num_fruits'
        """
        self.action_dict = SnakeEnv.default_action_dict

        reward_dict = kwargs.pop('reward_dict', SnakeEnv.default_reward_dict)
        if reward_dict.keys() != SnakeEnv.reward_keys:
            raise KeyError(f'reward dict keys must correspond to {SnakeEnv.reward_keys}')
        else:
            self.reward_dict = reward_dict

        self.num_snakes = num_snakes
        self.num_fruits = kwargs.pop('num_fruits', int(round(num_snakes * 0.8)))
        self.grid_shape: Tuple = (height, width)
        self.grid: np.ndarray
        self.snakes: List[Snake]

        low = 0
        high = 255
        self.action_space = gym.spaces.Discrete(len(self.action_dict))
        self.observation_space = gym.spaces.Box(
            low, high, shape=self.grid_shape, dtype=np.uint8)

    def reset(self):
        self.grid = make_grid(*self.grid_shape,
                              empty_value=Cell.EMPTY.value,
                              wall_value=Cell.WALL.value)
        self.snakes = self._generate_snakes()
        for snake in self.snakes:
            self.grid[snake.head_coord] = Cell.HEAD.value
            self.grid[snake.tail_coord] = Cell.TAIL.value

        xs, ys = self._generate_fruits(self.num_fruits)
        self.grid[xs, ys] = Cell.FRUIT.value

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

    def close(self):
        pass

    def step(self, actions):
        """
        Time Complexity:
            O(3*num_snakes) + (O(H*W) when generate fruits(maybe better approach, like random polling))

        updates snake status and grid
        all snake statuses are updated only in step method
        """

        if isinstance(actions, int):
            actions = [actions]
        # preprocess
        # head bang O(num_snakes), count duplicates by dictionary else it requires O(num_snakes^2)
        next_head_coords = defaultdict(list)
        for snake, action in zip(self.snakes, actions):
            snake.direction = self._next_direction(snake.direction, action)
            next_head_coords[snake.head_coord + snake.direction].append(snake.idx)
            # snake.alive, snake.reward = self._look_ahead(snake)
        dead_idxes, fruit_idxes = self._check_collision(next_head_coords)

        # dead_idxes = self._check_headbang(next_head_coords)
        for idx in dead_idxes:
            self.snakes[idx].death = True
            self.snakes[idx].alive = False
        for idx in fruit_idxes:
            tail_coord = self.snakes[idx].tail_coord
            if tail_coord in next_head_coords.keys():
                for s in self.snakes[next_head_coords[tail_coord]]:
                    s.death = True
                    s.alive = False
            self.snake[idx].fruit = True

        rews = []
        dones = []
        # postprocess
        for snake in self.snakes:
            snake.reward = self.reward_dict['time'] * snake.alive
            snake.reward += self.reward_dict['fruit'] * snake.fruit
            snake.reward += self.reward_dict['lose'] * snake.death
            # TODO
            # snake.reward += self.reward_dict['kill']
            # snake.reward += self.reward_dict['win']
            self._update_grid(snake)

            rews.append(snake.reward)
            dones.append(not snake.alive)

        obs = self._encode(self.grid)

        return obs, rews, dones, None

    def _encode(self, obs):

        return obs

    def _check_collision(self, next_head_coords):
        dead_idxes = []
        fruit_idxes = []
        for coord, idxes in next_head_coords.items():
            cell_value = self.grid[coord]
            if len(idxes) > 1 or cell_value in (Cell.WALL.value, Cell.BODY.value):
                dead_idxes.extend(idxes)
            elif len(idxes) == 1 and cell_value == Cell.FRUIT.value:
                fruit_idxes.append(idxes)
        dead_idxes = list(set(dead_idxes))
        fruit_idxes = fruit_idxes
        return dead_idxes, fruit_idxes

    def _look_ahead(self, snake):
        next_head_coord = snake.head_coord + snake.direction
        cell_value = self.grid[next_head_coord]

        reward = self.reward_dict['time']
        if cell_value == Cell.FRUIT.value:
            alive = True
            reward += self.reward_dict['fruit']
        else:
            alive = False
            reward += self.reward_dict['lose']

        return alive, reward

    def _update_grid(self, snake):
        if snake.alive:
            self.grid[snake.head_coord] = Cell.BODY.value
            # could be current or prev if ate fruit
            prev_tail_coord = snake.move()
            self.grid[snake.head_coord] = Cell.HEAD.value
            if prev_tail_coord:
                self.grid[prev_tail_coord] = Cell.EMPTY.value

            self.grid[snake.tail_coord] = Cell.TAIL.value
        else:
            if draw(self.grid, snake.coords, Cell.EMPTY.value) is False:
                print('draw failed')

    def _check_headbang(self, next_head_coords):
        dead_idxes = []
        for coord, idxes in next_head_coords.items():
            if len(idxes) > 1:
                dead_idxes.extend(idxes)

        return dead_idxes

    def _generate_snakes(self):
        snakes = []
        for idx in range(self.num_snakes):
            coord = random_empty_coord(self.grid)
            snakes.append(Snake(idx, coord, Direction.RIGHT))

        return snakes

    def _generate_fruits(self, num_fruits=1):
        xs, ys = random_empty_coords(self.grid, num_coords=num_fruits)

        return xs, ys

    def _next_direction(self, direction, action):
        """
        0 == noop
        1 == left
        2 == right
        if direction == Direction.UP:
            if action == self.action_dict['noop']:
                return Direction.UP
            elif action == self.action_dict['left']:
                return Direction.LEFT
            else:
                return Direction.RIGHT
        elif direction == Direction.RIGHT:
            if action == self.action_dict['noop']:
                return Direction.RIGHT
            elif action == self.action_dict['left']:
                return Direction.UP
            else:
                return Direction.DOWN
        elif direction == Direction.DOWN:
            if action == self.action_dict['noop']:
                return Direction.DOWN
            elif action == self.action_dict['left']:
                return Direction.RIGHT
            else:
                return Direction.LEFT
        elif direction == Direction.LEFT:
            if action == self.action_dict['noop']:
                return Direction.LEFT
            elif action == self.action_dict['left']:
                return Direction.DOWN
            else:
                return Direction.UP
        """
        angle = math.atan2(direction.value[1], direction.value[0])
        new_coord = (int(math.cos(angle + self.action_angle_dict[action])),
                     int(math.sin(angle + self.action_angle_dict[action])))
        return Direction(new_coord)



# TODO:
# encoding ->
# reward engineering -> wrapper
# fruits -> cell list
# coord vs pos(position) vs coord(coordinate)
# pygame
# 1player survive win
# body hit other head kill

# custom reward structure
# obs -> cell states
# win terminal condition setting

