import gym
import numpy as np


class SingleAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.num_snakes == 1, "Number of player must be one"
        self.action_space = gym.spaces.Discrete(len(self.env.action_dict))
        self.observation_space = gym.spaces.Box(
            self.low, self.high,
            shape=(*self.env.grid_shape, 6), dtype=np.uint8)  # 8

    def reset(self, **kwargs):
        wrapped_obs = self.env.reset(**kwargs)
        return wrapped_obs[0]

    def step(self, action, **kwargs):
        obs, rews, dones, infos = self.env.step(action[0], **kwargs)
        return obs[0], rews[0], dones[0], {}

# import random
# import numpy as np
# import gym
#
# from stable_baselines import PPO2
# from stable_baselines.common.vec_env import DummyVecEnv
#
#
# class SAhandler(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.npcs = []
#         self.npc_list = []
#         self.env = env
#         self.player_idx = [i for i in range(env.num_players)]
#         self.obs = None
#
#     def set_npcs(self, npc_list):
#         '''
#         This function is needed so that the player class
#         can be set as the npcs as well.
#         '''
#         assert len(npc_list) == self.env.num_players - 1
#         self.npc_list = npc_list
#
#     def reset(self, **kwargs):
#         '''
#         Reset the MARL env.
#         Make sure there are enough nps in the game and shuffle their order.
#         Update the saved observatino for the npc's to predict action.
#         '''
#         self.npcs = []
#         for npc_f in self.npc_list:
#             # TODO: change this to load directly from a pickle file
#             # Later, maybe change to load from single model in shmem?
#             self.npcs.append(PPO2.load(npc_f))
#         assert len(self.npcs) == self.env.num_players - 1
#         random.shuffle(self.player_idx)
#         self.obs = self.env.reset(**kwargs)
#         return self.obs[self.player_idx[-1]]
#
#     def step(self, action):
#         '''
#         Combine the player's action with that of the NPC's.
#         Make sure enough npcs are provided and
#         the player's index is always the last in the list.
#         '''
#         actions = [-1] * self.env.num_players
#         actions[self.player_idx[-1]] = action
#         for i, agent in zip(self.player_idx[:-1], self.npcs):
#             actions[i] = agent.predict(self.obs[i])
#         if -1 in actions:
#             raise ValueError("Not enough npcs defined")
#         self.obs, rews, dones, infos = self.env.step(actions)
#         curr_idx = self.player_idx[-1]
#         return self.obs[curr_idx], rews[curr_idx], dones[curr_idx], infos
#
#
# class Vechandler(gym.Wrapper):
#     def __init__(self, env):
#         env.reward_range = env.get_attr('reward_range')[0]
#         super().__init__(env)
#         self.env = env
#
#     def reset(self, **kwargs):
#         obs = self.env.reset(**kwargs)
#         obs = obs.reshape((-1, *obs.shape[2:]))
#         return obs
#
#     def step(self, actions):
#         actions = np.asarray(actions).reshape(
#             (self.env.num_envs, self.env.get_attr('num_players')[0])
#         )
#         obs, rews, dones, infos = self.env.step(actions)
#         obs = obs.reshape(-1, *obs.shape[2:])
#         rews = rews.flatten()
#         dones = dones.flatten()
#         infos = {i: info for i, info in enumerate(infos)}
#         # import pdb; pdb.set_trace()
#         return obs, rews, dones, infos
#
#
# class FalseVecEnv(DummyVecEnv):
#     def step_wait(self):
#         obs, rews, dones, infos = self.envs[0].step(self.actions)
#         infos = [infos]
#         return obs, rews, dones, infos
#
#     def reset(self):
#         return self.envs[0].reset()
