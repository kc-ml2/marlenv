import numpy as np
import gym
import random
import math
from PIL import Image
from gym.spaces import Discrete, Box

from marlenv.envs.ml2_python.common import Point, Cell, Direction
from marlenv.envs.ml2_python.field import Field
from marlenv.envs.ml2_python.python import Python

import collections.abc as cabc


class Action:
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class Reward:
    '''
    Default reward class
    '''
    FRUIT = 1.0
    KILL = 0
    LOSE = -1.0
    WIN = 0
    TIME = 0

    # count-based bonus
    alpha = 0
    beta = 0
    NEAR = 0.2


class ML2Python(gym.Env):
    # metadata = {'render.modes': ['rgb_array']}
    def __init__(self, init_map, players=None, full_observation=False,
                 vision_range=20, num_fruits=4, rewards={}):
        self.init_map = init_map
        self.players = players
        self.observability = full_observation

        # Set the global reward function of agents
        self.reward_keys = ['FRUIT', 'KILL', 'LOSE', 'WIN', 'TIME']
        self.reward_func = {}
        for rk in self.reward_keys:
            self.reward_func[rk] = getattr(Reward, rk)
        for rk, kv in rewards:
            self.reward_func[rk] = kv

        self.field = Field(init_map, players)
        self.num_players = len(self.field.players)
        self.playerpos = players
        self.visits = np.zeros((self.num_players, np.prod(self.field.size)))

        self.num_fruits = num_fruits
        self.fruits = []

        if self.observability:
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(4, self.field.size[0], self.field.size[1]),
                dtype=np.uint8
            )
        else:
            # Set vision range to vision_range(nxn)
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(4, vision_range, vision_range),
                dtype=np.uint8
            )
        self.action_space = Discrete(5)

        self.reset()

    def reset(self):
        if len(self.players) == 1:
            # Randomize snake's initial position
            players = [
                Python(Point(np.random.randint(3, 4, 1)[0],
                             np.random.randint(3, 4, 1)[0]),
                       random.choice(Direction.DIRECTIONLIST), 3)
            ]
        else:
            players = self.playerpos
        self.field = Field(self.init_map, players)

        # Initialize dones and infos
        self.dones = np.zeros(self.num_players, dtype=bool)
        self.epinfos = {
            'step': 0,
            'scores': np.zeros(self.num_players),
            'fruits': np.zeros(self.num_players),
            'kills': np.zeros(self.num_players),
            'terminal_observation': 0
        }

        for i in range(self.num_fruits):
            self.fruit = self.field.get_empty_cell()
            self.field[self.fruit] = Cell.FRUIT
            self.fruits.append(self.fruit)

        if self.observability:
            return self.full_observation()
        return self.encode()

    def _choose_action(self, python, action):
        if python.direction == Direction.NORTH:
            if action == Action.LEFT:
                python.turn_left()
            elif action == Action.RIGHT:
                python.turn_right()
        elif python.direction == Direction.EAST:
            if action == Action.DOWN:
                python.turn_right()
            elif action == Action.UP:
                python.turn_left()
        elif python.direction == Direction.SOUTH:
            if action == Action.RIGHT:
                python.turn_left()
            elif action == Action.LEFT:
                python.turn_right()
        elif python.direction == Direction.WEST:
            if action == Action.UP:
                python.turn_right()
            elif action == Action.DOWN:
                python.turn_left()

    def step(self, actions):
        if not isinstance(actions, cabc.Iterable):
            actions = [actions]
        assert len(actions) == self.num_players
        self.epinfos['step'] += 1
        rewards = np.zeros(self.num_players, dtype=float)

        for idx, action in enumerate(actions):
            python = self.field.players[idx]
            if not python.alive:
                continue

            # Choose action
            self._choose_action(python, action)

            # Eat fruit
            if self.field[python.next] == Cell.FRUIT:
                python.grow()
                rewards[idx] += self.reward_func['FRUIT']
                self.epinfos['fruits'][idx] += 1
                if python.head in self.fruits:
                    self.fruits.remove(python.head)
                    self.fruit = self.field.get_empty_cell()
                    self.field[self.fruit] = Cell.FRUIT
                    self.fruits.append(self.fruit)

            # Or just starve
            else:
                self.field[python.tail] = Cell.EMPTY
                python.move()
                rewards[idx] += self.reward_func['TIME']

            self.field.players[idx] = python

            # Add count-based bonus
            '''
            cell = int(python.head.x + python.head.y*self.field.size[0])
            self.visits[idx][cell] += 1
            bonus = Reward.beta*(self.visits[idx][cell] + Reward.alpha)**(-0.5)
            rewards[idx] += bonus
            '''

        # Resolve conflicts
        conflicts = self.field.update_cells()
        for conflict in conflicts:
            idx = conflict[0]
            python = self.field.players[idx]
            python.alive = False
            rewards[idx] += self.reward_func['LOSE']
            self.dones[idx] = True

            # When colliding with another player
            if len(conflict) > 1:
                idx = conflict[1]
                if idx != conflict[0]:
                    other = self.field.players[idx]
                    # Head to head
                    if self.field[python.head] in Cell.HEAD:
                        other.alive = False
                        rewards[idx] += self.reward_func['LOSE']
                        self.dones[idx] = True
                    # Head to body
                    else:
                        rewards[idx] += self.reward_func['KILL']
                        self.epinfos['kills'][idx] += 1

        self.epinfos['scores'] += rewards

        if self.observability:
            return (self.full_observation(), rewards,
                    self.dones, self.epinfos)
        return self.encode(), rewards, self.dones, self.epinfos

    def get_custom_state(self):
        python = self.field.players[0]
        direction = python.direction
        position = python.head.position()
        actionv = direction.position()
        nextpos = python.next.position()
        x, y = np.where(self.field._cells == 1)
        fruitpos = (x[0], y[0])
        state = np.array([position, actionv, nextpos, fruitpos]).flatten()
        return state

    def full_observation(self):
        self.field.clear()
        body = np.zeros((self.num_players, *self.field.size))
        for idx in range(self.num_players):
            head_cell = np.isin(
                self.field._cells,
                Cell.HEAD[idx]
            ).astype(np.float32)
            body_cell = np.isin(
                self.field._cells,
                Cell.BODY[idx]
            ).astype(np.float32)
            body[idx] = head_cell + body_cell

        fruit = np.isin(self.field._cells, Cell.FRUIT).astype(np.float32)
        wall = np.isin(self.field._cells, Cell.WALL).astype(np.float32)
        fruit = fruit * 3
        wall = wall * 2

        state = np.zeros((self.num_players, *self.observation_space.shape))

        for idx in range(self.num_players):
            if self.field.players[idx].alive:
                state[idx][0] = body[idx] + wall
                state[idx][1] = np.sum(body, axis=0) - body[idx] + wall
                state[idx][2] = fruit + wall
                state[idx][3] = wall
        return state

    def encode(self):
        self.field.clear()
        body = np.zeros((self.num_players, *self.field.size))
        for idx in range(self.num_players):
            head_cell = np.isin(
                self.field._cells,
                Cell.HEAD[idx]
            ).astype(np.float32)
            body_cell = np.isin(
                self.field._cells,
                Cell.BODY[idx]
            ).astype(np.float32)
            body[idx] = head_cell + body_cell

        fruit = np.isin(self.field._cells, Cell.FRUIT).astype(np.float32)
        wall = np.isin(self.field._cells, Cell.WALL).astype(np.float32)

        # give different numbers to each array
        fruit = fruit * 3
        wall = wall * 2

        state = np.zeros((self.num_players, *self.observation_space.shape))

        for idx in range(self.num_players):
            if self.field.players[idx].alive:
                myself = body[idx]
                others = np.sum(body, axis=0) - myself
                state[idx][0] = self.get_vision(idx, body[idx])
                state[idx][1] = self.get_vision(idx, others) * 2
                state[idx][1] += self.get_vision(idx, wall)
                state[idx][2] = self.get_vision(idx, fruit)
                state[idx][3] = self.get_vision(idx, wall)

                # When Wall not included in body
                # state[idx][0] = self.get_vision(idx, myself)
                # state[idx][1] = self.get_vision(idx, np.sum(body, axis=0)
                # - myself)
                # state[idx][2] = self.get_vision(idx, fruit)
                # state[idx][3] = self.get_vision(idx, wall)

        if self.num_players == 1:
            state = state[0]
        return state

    def get_vision(self, idx, arr):
        head = np.where(np.isin(self.field._cells, Cell.HEAD[idx]))
        h, w = self.observation_space.shape[1:]
        x1 = int(head[1] - w // 2)
        x2 = int(head[1] + w // 2)
        y1 = int(head[0] - h // 2)
        y2 = int(head[0] + h // 2)

        arr = arr[max(y1, 0):y2, max(0, x1):x2]

        if y1 < 0:
            arr = np.pad(arr, ((-y1, 0), (0, 0),), mode="constant")
        elif y2 > self.field.size[0]:
            arr = np.pad(arr, ((0, y2 - self.field.size[0]), (0, 0),),
                         mode="constant")
        if x1 < 0:
            arr = np.pad(arr, ((0, 0), (-x1, 0)), mode="constant")
        elif x2 > self.field.size[1]:
            arr = np.pad(arr, ((0, 0), (0, x2 - self.field.size[1])),
                         mode="constant")
        return arr

    def render(self, mode='rgb_array'):
        img = np.uint8(self.field._cells * 255)
        stacked_img = np.stack((img,) * 3, axis=-1)
        image = Image.fromarray(stacked_img)

        image = image.resize((400, 400))
        image = np.asarray(image, dtype="uint8")
        print(self.field)
        return image

    def distance(self, pos1, pos2):
        return math.sqrt((pos2[1] - pos1[1])**2 + (pos2[0] - pos1[0])**2)
