from envs.grid_explore.gridworld import GridWorld 
import random
from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

import numpy as np
import math
from gym import spaces
from PIL import ImageColor
import copy

CELL_SIZE = 30

PRE_IDS = {
    'agent': ['3','4','5','6'],
    'wall': '2',
    'empty': '0',
    'prey':'1'
}

WALL_COLOR = 'black'
PREY_COLOR = 'red'
AGENT_COLOR = 'blue'

class Cell:
    UNVISITED = 0
    PREY = 1
    WALL = 2
    AGENTS = [3, 4, 5, 6]

class Move:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 3 
    STAY = 4 

class Rewards:
    TIMEPENALTY = -1
    DISTANCEBONUS = 1
    WIN = 50 

class Agent:

        def __init__(self, posx, posy, idx, sight=2):
                
                self.idx = int(idx)
                self.x = posx
                self.y = posy 
                self.position = (self.x, self.y)
                
                self.sight = sight

        def __str__(self):
                return "x: " + str(self.x) +  " y: "+ str(self.y)+ " agent idx: " + str(self.idx)

        def position_yx(self):
                return self.y, self.x

        def move(self, MOVE, grid):
                new_x, new_y = self.position[0], self.position[1]
                org_x, org_y = self.position[0], self.position[1]

                if MOVE==Move.UP and self.position[1] != 1:
                        new_y -= 1 
                elif MOVE == Move.DOWN and self.position[1] != len(grid)-1:
                        new_y += 1      
                if MOVE==Move.LEFT and self.position[0] != 1:
                        new_x -= 1 
                elif MOVE == Move.RIGHT and self.position[0] != len(grid)-1:
                        new_x += 1      

                #check if wall exist there
                if grid[new_y][new_x] != 2 and grid[new_y][new_x] not in Cell.AGENTS:
                        self.makeMove(new_x, new_y)
                        grid[org_y][org_x] = 1
                        return new_x, new_y
                else:
                        # print(self.idx, end=":        ")
                        # print(grid[new_y][new_x], " ", grid[org_y][org_x], " ", org_x, org_y)
                        # print(grid)
                        return self.position[0], self.position[1]

        def makeMove(self, new_x, new_y):
                self.x = new_x
                self.y = new_y
                self.position = (self.x, self.y)
