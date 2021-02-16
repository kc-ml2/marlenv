from collections import defaultdict
from typing import List, Tuple

import gym
import math
import numpy as np
from gym.utils import seeding

from marlenv.core.grid_util import (
    random_empty_coords, random_empty_coord, draw, make_grid, dfs_sweep_empty)
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
        1: math.pi/2.0,
        2: -math.pi/2.0
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
            height=20,
            width=20,
            num_snakes=4,
            snake_length=3,
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
            raise KeyError(
                f'reward dict keys must correspond to {SnakeEnv.reward_keys}'
            )
        else:
            self.reward_dict = reward_dict

        self.num_snakes = num_snakes
        self.num_fruits = kwargs.pop('num_fruits',
                                     int(round(num_snakes * 0.8)))
        self.grid_shape: Tuple = (height, width)
        self.grid: np.ndarray
        self.snakes: List[Snake]
        self.snake_length = snake_length

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
            coords = snake.coords
            snake_id = 10 * snake.idx
            for coord in coords:
                self.grid[coord] = Cell.BODY.value + snake_id
            self.grid[snake.head_coord] = Cell.HEAD.value + snake_id
            self.grid[snake.tail_coord] = Cell.TAIL.value + snake_id

        xs, ys = self._generate_fruits(self.num_fruits)
        self.grid[xs, ys] = Cell.FRUIT.value
        self.alive_snakes = self.num_snakes

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

        data = np.vectorize(SYM2CHR.get)(self.grid % 10)
        data = [''.join(i) for i in data]
        data = '\n'.join(data)
        print(data)

    def close(self):
        pass

    def step(self, actions):
        """
        Time Complexity:
            O(3*num_snakes)
            + (O(H*W) w/ fruits(maybe better approach, like random polling))

        updates snake status and grid
        all snake statuses are updated only in step method
        """

        if isinstance(actions, int):
            actions = [actions]
        assert len(actions) == self.num_snakes
        # preprocess
        next_head_coords = defaultdict(list)
        alive_snakes = []
        for snake, action in zip(self.snakes, actions):
            if snake.alive:
                snake.direction = self._next_direction(snake.direction, action)
                new_head_coord = snake.head_coord + snake.direction
                next_head_coords[new_head_coord].append(snake.idx)
                alive_snakes.append(snake.idx)
        dead_idxes, fruit_idxes = self._check_collision(next_head_coords)
        print(dead_idxes)

        self.alive_snakes -= len(dead_idxes)
        for idx in dead_idxes:
            self.snakes[idx].death = True
            self.snakes[idx].alive = False
        for idx in fruit_idxes:
            tail_coord = self.snakes[idx].tail_coord
            if tail_coord in next_head_coords.keys():
                for s in [self.snakes[di]
                          for di in next_head_coords[tail_coord]]:
                    s.death = True
                    s.alive = False
                    self.alive_snakes -= 1
                    self.snakes[idx].kills += 1
            self.snakes[idx].fruit = True
        if self.alive_snakes == 1:
            for snake in self.snakes:
                if snake.alive:
                    print('player {} wins!'.format(snake.idx))
                    snake.win = True
                    break

        rews = []
        dones = []
        # postprocess
        for snake in self.snakes:
            if not snake.death and not snake.alive:
                snake.reward = 0.
                rews.append(snake.reward)
            else:
                snake.reward = self.reward_dict['time'] * snake.alive
                snake.reward += self.reward_dict['fruit'] * snake.fruit
                snake.reward += self.reward_dict['lose'] * snake.death
                snake.reward += self.reward_dict['kill'] * snake.kills
                snake.reward += self.reward_dict['win'] * snake.win
                rews.append(snake.reward)
                self._update_grid(snake)

            dones.append(not snake.alive)

        obs = self._encode(self.grid)

        return obs, rews, dones, None

    def _encode(self, obs):
        # Encode the observation. obs is self.grid
        # Returns the obs in HxWxC
        # May be overriden for customized observation
        env_objs = np.zeros([*obs.shape, 2], dtype=np.float32)
        snake_objs = [np.zeros([*obs.shape, 3], dtype=np.float32)
                      for _ in range(self.num_snakes)]
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                cell_value = obs[r, c]
                if cell_value in (Cell.WALL.value, Cell.FRUIT.value):
                    env_objs[r, c, cell_value - 1] = 1
                elif cell_value != Cell.EMPTY.value:
                    snake_id = cell_value // 10
                    obj_id = cell_value % 10
                    snake_objs[snake_id][r, c, obj_id - 3] = 1
        encoded_obs = [np.concatenate([env_objs, snake_obj], axis=-1)
                       for snake_obj in snake_objs]

        return encoded_obs

    def _check_collision(self, next_head_coords):
        # Check for head, body, wall, fruit collision and assign new status
        dead_idxes = []
        fruit_idxes = []
        for coord, idxes in next_head_coords.items():
            cell_value = self.grid[coord] % 10
            # Head collision or clear death
            if len(idxes) > 1 or cell_value in (Cell.WALL.value,
                                                Cell.BODY.value,
                                                Cell.HEAD.value):
                if len(idxes) > 1:
                    print("Death by collision", idxes)
                dead_idxes.extend(idxes)
                if cell_value in (Cell.BODY.value, Cell.HEAD.value):
                    self.snakes[self.grid[coord] // 10].kills += 1
            elif len(idxes) == 1 and cell_value == Cell.FRUIT.value:
                fruit_idxes.extend(idxes)
        dead_idxes = list(set(dead_idxes))
        fruit_idxes = fruit_idxes
        return dead_idxes, fruit_idxes

    def _update_grid(self, snake):
        # Update the grid according to a snake's status
        if snake.alive:
            snake_id = 10 * snake.idx  # For distinguishing snakes in grid
            self.grid[snake.head_coord] = Cell.BODY.value + snake_id
            # could be current or prev if ate fruit
            prev_tail_coord = snake.move()
            if prev_tail_coord:
                # Didn't eat the fruit
                if self.grid[prev_tail_coord] == Cell.TAIL.value + snake_id:
                    self.grid[prev_tail_coord] = Cell.EMPTY.value
            self.grid[snake.head_coord] = Cell.HEAD.value + snake_id

            self.grid[snake.tail_coord] = Cell.TAIL.value + snake_id
        else:
            coord_list = snake.coords
            if self.grid[snake.coords[-1]] // 10 != snake.idx:
                coord_list = coord_list[:-1]
            if draw(self.grid, coord_list, Cell.EMPTY.value) is False:
                print('draw failed')
            snake.move()

    def _clear_overlap(self, list_of_coords):
        # Check for overlap in given snake coordinates
        flat_list = []
        for element in list_of_coords:
            flat_list.extend(element)
        unique_list = list(set(flat_list))
        return len(unique_list) == len(flat_list)

    def _generate_snakes(self):
        # Depth-first-search through the grid for possible snake positions
        candidates = dfs_sweep_empty(self.grid, self.snake_length)
        while True:
            # Randomly select init snake poses untill no overlap
            sample_idx = np.random.permutation(
                len(candidates)
            )[:self.num_snakes]
            samples = [candidates[si] for si in sample_idx]
            if self._clear_overlap(samples):
                break
        snakes = [Snake(idx, coords) for idx, coords in enumerate(samples)]

        return snakes

    def _generate_fruits(self, num_fruits=1):
        xs, ys = random_empty_coords(self.grid, num_coords=num_fruits)

        return xs, ys

    def _next_direction(self, direction, action):
        """
        0 == noop
        1 == left
        2 == right
        Change direction by +-90 degrees
        """
        angle = math.atan2(direction.value[1], direction.value[0])
        new_coord = (int(math.cos(angle + self.action_angle_dict[action])),
                     int(math.sin(angle + self.action_angle_dict[action])))
        return Direction(new_coord)



# TODO:
# encoding ->
# reward engineering -> wrapper
# custom reward structure
# fruits -> cell list
# coord vs pos(position) vs coord(coordinate)
# pygame
# obs -> cell states
