from gym.envs.registration import registry, register, make, spec
import gym
import numpy as np
import random 
import sys
import os
from settings import PROJECT_ROOT, INF
sys.path.append("../")
sys.path.append(os.path.join(PROJECT_ROOT, 'envs'))

# ML2 Python environments
from envs.ml2_python.common import Point, Direction
from envs.ml2_python.python import Python
from envs.ml2_python.graphics import ML2PythonGUI


init_length = 3
init_map = os.path.join(PROJECT_ROOT, 'assets', '10x10.txt')
players = [
    Python(Point(np.random.randint(2, 4, 1)[0],np.random.randint(2, 4, 1)[0]), Direction.EAST, 3)
]

register(
    id='python_1p-v0',
    entry_point='ml2_python:ML2Python',
    max_episode_steps=INF,
    kwargs={
        'init_map': init_map,
        'players': players,
        'num_fruits': 1,
        'full_observation' : True
    }
)

init_length = 3
init_map = os.path.join(PROJECT_ROOT, 'assets', '20x20.txt')
players = [
    Python(Point(np.random.randint(2, 4, 1)[0],np.random.randint(2, 4, 1)[0]), Direction.EAST, 3)
]

register(
    id='python_1p-v1',
    entry_point='ml2_python:ML2Python',
    max_episode_steps=INF,
    kwargs={
        'init_map': init_map,
        'players': players
    }
)


init_length = 3
init_map = os.path.join(PROJECT_ROOT, 'assets', '20x20.txt')
players = [
    Python(Point(3, 3), Direction.EAST, init_length),
    Python(Point(16, 3), Direction.SOUTH, init_length),
    Python(Point(16, 16), Direction.WEST, init_length),
    Python(Point(3, 16), Direction.NORTH, init_length)
]

register(
    id='python_4p-v1',
    entry_point='ml2_python:ML2Python',
    max_episode_steps=INF,
    kwargs={
        'init_map': init_map,
        'players': players
    }
)

init_map = os.path.join(PROJECT_ROOT, 'assets', 'ml2.txt')
players = [
    Python(Point(3, 3), Direction.EAST, init_length),
    Python(Point(36, 3), Direction.SOUTH, init_length),
    Python(Point(36, 36), Direction.WEST, init_length),
    Python(Point(3, 36), Direction.NORTH, init_length)
]
register(
    id='python_ml2-v0',
    entry_point='ml2_python:ML2Python',
    max_episode_steps=INF,
    kwargs={
        'init_map': init_map,
        'players': players

    }
)



register(
	id='GridExplore-v0',
	entry_point='envs.gridworld:GridExplore',
	kwargs={
		'full_observable' : False,
		'size' : 10
	}
)

register(
	id='GridExplore-v1',
	entry_point='envs.gridworld:GridExplore',
	kwargs={
		'full_observable' : False,
		'size' : 15,
		'n_agents' : 4
	}
)

register(
	id='GridPath-v0',
	entry_point='envs.gridworld:GridPath',
	kwargs={
		'full_observable' : False,
		'size' : 11
	}
)