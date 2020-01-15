import numpy as np
import gym
from gym.spaces import Discrete, Box
from ml2_python.common import Point, Cell, Direction
from ml2_python.field import Field
from ml2_python.python import Python
import random
import math
import torch
from PIL import Image

class Action:
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class Reward:
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
    def __init__(self, init_map, players=None, full_observation=True, vision_range=20, num_fruits=4):
        self.init_map = init_map
        self.players = players
        self.observability = full_observation

        self.field = Field(init_map, players)
        self.num_players = len(self.field.players)
        self.playerpos = players
        self.visits = np.zeros((self.num_players, np.prod(self.field.size)))
        # self.num_envs= 32
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
                shape=(self.num_players, 4, vision_range, vision_range),
                dtype=np.uint8
            )
        
        self.action_space = Discrete(5)

        self.reset()

    def reset(self):
        
        if len(self.players)==1:

            #randomize snake's initial position 
            players = [
                Python(Point(np.random.randint(3, 4, 1)[0],np.random.randint(3, 4, 1)[0]), random.choice(Direction.DIRECTIONLIST), 3)
            ]
        else:
            players = self.playerpos


        self.field = Field(self.init_map, players)

        #initialize dones and infos 
        self.dones = np.zeros(self.num_players, dtype=bool)
        self.epinfos = {
            'step': 0,
            'scores': np.zeros(self.num_players),
            'fruits': np.zeros(self.num_players),
            'kills': np.zeros(self.num_players),
            'terminal_observation' : 0
        }

        for i in range(self.num_fruits):   
            self.fruit = self.field.get_empty_cell()
            self.field[self.fruit] = Cell.FRUIT
            self.fruits.append(self.fruit)
        
        if self.observability:
            return self.full_observation()

        return self.encode()

    def step(self, actions):
        assert len(actions) == self.num_players
        # actions = [actions]
        self.epinfos['step'] += 1
        rewards = np.zeros(self.num_players, dtype=float)
        
        for idx, action in enumerate(actions):
            python = self.field.players[idx]
            if not python.alive:
                continue

            # Choose action
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

            # Eat fruit
            if self.field[python.next] == Cell.FRUIT:
                python.grow()
                rewards[idx] += Reward.FRUIT
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
                rewards[idx] += Reward.TIME

            self.field.players[idx] = python

            # Add count-based bonus
            # cell = int(python.head.x + python.head.y*self.field.size[0])
            # self.visits[idx][cell] += 1
            # bonus = Reward.beta*(self.visits[idx][cell] + Reward.alpha)**(-0.5)
            # rewards[idx] += bonus

        # Resolve conflicts
        conflicts = self.field.update_cells()
        for conflict in conflicts:
            idx = conflict[0]
            python = self.field.players[idx]
            python.alive = False
            rewards[idx] += Reward.LOSE
            self.dones[idx] = True

            # If collided with another player
            if len(conflict) > 1:
                idx = conflict[1]
                if idx != conflict[0]:
                    other = self.field.players[idx]
                    # Head to head
                    if self.field[python.head] in Cell.HEAD:
                        other.alive = False
                        rewards[idx] += Reward.LOSE
                        self.dones[idx] = True
                    # Head to body
                    else:
                        rewards[idx] += Reward.KILL
                        self.epinfos['kills'][idx] += 1
        
        # python = self.field.players[0]
        # headpos = python.head.position()
        # fruit = np.where(np.isin(self.field._cells, Cell.FRUIT))
        # print(self.field._cells)
        # print(self.field)
        # print(headpos, fruit)

        # rewards[0] += 1 / self.distance(headpos, fruit) 

        # Check if done and calculate scores
        # if self.num_players > 1 and np.sum(~self.dones) == 1:
        #     idx = list(self.dones).index(False)
        #     self.dones[idx] = True
        #     rewards[idx] += Reward.WIN

        self.epinfos['scores'] += rewards
        done = False
        if self.dones[0] == True:
            done =True
        
        if self.observability:
            return self.full_observation(), rewards[0], self.dones[0], self.epinfos

        return self.encode(), rewards, self.dones, self.epinfos


    def get_custom_state(self):
        python = self.field.players[0]
        direction = python.direction
        position = python.head.position()
        actionv = direction.position()
        nextpos = python.next.position()
        (x, y) = np.where(self.field._cells == 1)
        fruitpos = (x[0],y[0])
        # print(position, actionv, nextpos, fruitpos)
        # print(self.field)
        state = np.array([position, actionv, nextpos, fruitpos]).flatten()
        # print(state)
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

        state = np.zeros(self.observation_space.shape)

        for idx in range(self.num_players):
            if self.field.players[idx].alive:
                state[0] = body[idx] + wall
                state[1] = body[idx] + wall
                state[2] = fruit + wall
                state[3] = wall
        # for i in range(4):
        #     state[0][i]=self.field._cells
        # # print(self.field._cells)
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
        fruit = fruit * 3
        wall = wall * 2

        state = np.zeros(self.observation_space.shape)

        for idx in range(self.num_players):
            if self.field.players[idx].alive:
                state[idx][0] = self.get_vision(idx, body[idx]) + self.get_vision(idx, wall)
                state[idx][1] = self.get_vision(idx, np.sum(body, axis=0) - body[idx])*2 + self.get_vision(idx, wall)

                # state[idx][1] = self.get_vision(idx, np.sum(body, axis=0) - body[idx])
                # state[idx][1] = fruit 
                state[idx][2] = self.get_vision(idx, fruit)
                state[idx][3] = self.get_vision(idx, wall) 
                
                # state[0] = self.get_vision(idx, body[idx])
                # state[1] = self.get_vision(idx, np.sum(body, axis=0) - body[idx])
                # state[2] = self.get_vision(idx, fruit)
                # state[3] = self.get_vision(idx, wall) 
        # print(state[0])
        return state
    
    def get_vision(self, idx, arr):
        head = np.where(np.isin(self.field._cells, Cell.HEAD[idx]))
        # print(self.observation_space.shape)
        h, w = self.observation_space.shape[2:]
        x1 = int(head[1] - w//2)
        x2 = int(head[1] + w//2)
        y1 = int(head[0] - h//2)
        y2 = int(head[0] + h//2)
        
        arr = arr[max(y1,0):y2, max(0, x1):x2]

        if y1 < 0:
            arr = np.pad(arr, ((-y1, 0), (0, 0),), mode="constant")
        elif y2 > self.field.size[0]:
            arr = np.pad(arr, ((0, y2 - self.field.size[0]), (0, 0),),mode="constant")
        if x1 < 0:
            arr = np.pad(arr, ((0, 0), (-x1, 0)),mode="constant")
        elif x2 > self.field.size[1]:
            arr = np.pad(arr, ((0, 0), (0,x2 - self.field.size[1])),mode="constant")

        return arr
 
    def render(self, mode='rgb_array'):
        # field = str(self.field)
        # print(np.uint8(self.field._cells * 255))

        img = np.uint8(self.field._cells * 255)
        stacked_img = np.stack((img,)*3, axis=-1)
        image = Image.fromarray(stacked_img)

        image = image.resize((400, 400))  
        image = np.asarray(image, dtype="uint8")
        print(self.field)
        return image


    # def render(self):

    #     return print(self.field)


    def distance(self, pos1, pos2): 
        return math.sqrt( (pos2[1] - pos1[1])**2 + (pos2[0]-pos1[0])**2 ) 
