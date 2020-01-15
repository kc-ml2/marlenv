import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace


class MultiAgentEnv(gym.Env):

    def __init__(self, n_agents, full_observable):

        self.n_agents = n_agents
        self.full_observable = full_observable

        self._colaboration_reward = None
        self._step_count = 0 


    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError
        
