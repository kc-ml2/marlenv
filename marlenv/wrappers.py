import sys
from enum import Enum
import multiprocessing as mp
import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.error import AlreadyPendingCallError, NoAsyncCallError
from gym.vector.utils import write_to_shared_memory
import numpy as np


class AsyncState(Enum):
    DEFAULT = 'default'
    WAITING_RESET = 'reset'
    WAITING_STEP = 'step'
    WAITING_RENDER = 'render'


class SingleAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.num_snakes == 1, "Number of player must be one"
        self.action_space = gym.spaces.Discrete(len(self.env.action_dict))
        if self.vision_range:
            h = w = self.vision_range * 2 + 1
            self.observation_space = gym.spaces.Box(
                self.low, self.high,
                shape=(h, w,  self.obs_ch), dtype=np.uint8)  # 8
        else:
            self.observation_space = gym.spaces.Box(
                self.low, self.high,
                shape=(*self.grid_shape, self.obs_ch), dtype=np.uint8)  # 8

    def reset(self, **kwargs):
        wrapped_obs = self.env.reset(**kwargs)
        return wrapped_obs[0]

    def step(self, action, **kwargs):
        obs, rews, dones, infos = self.env.step([action], **kwargs)
        return obs[0], rews[0], dones[0], {}


class SingleMultiAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.env.action_dict))
        if self.vision_range:
            h = w = self.vision_range * 2 + 1
            self.observation_space = gym.spaces.Box(
                self.low, self.high,
                shape=(self.num_snakes, h, w,  self.obs_ch), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(
                self.low, self.high,
                shape=(self.num_snakes, *self.grid_shape, self.obs_ch),
                dtype=np.uint8)

    # def reset(self, **kwargs):
    #     wrapped_obs = self.env.reset(**kwargs)
    #     return np.concatenate(wrapped_obs, axis=-1)

    # def step(self, action, **kwargs):
    #     action = [ac for ac in action]
    #     obs, rews, dones, infos = self.env.step(action, **kwargs)
    #     obs = np.concatenate(obs, axis=-1)
    #     rews = np.concatenate(rews, axis=-1)
    #     dones = np.concatenate(dones, axis=-1)
    #     return obs, rews, dones, {}


class AsyncVectorMultiEnv(AsyncVectorEnv):
    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, worker=_worker_shared_memory, **kwargs)
        self.default_state = None

    def render_async(self):
        self._assert_is_running()
        if self._state.value != AsyncState.DEFAULT.value:
            raise AlreadyPendingCallError('Calling `render_async` while waiting '
                            'for a pending call to `{0}` to complete.'.format(
                            self._state.value), self._state.value)
        else:
            self.default_state = self._state
        self.parent_pipes[0].send(('render', None))
        self._state = AsyncState.WAITING_RENDER

    def render_wait(self, timeout=None):
        self._assert_is_running()
        if self._state.value != AsyncState.WAITING_RENDER.value:
            raise NoAsyncCallError('Calling `render_wait` without any prior '
                    'call to `render_async`.', AsyncState.WAITING_RESET.value)

        if not self._poll(timeout):
            self._state = self.default_state
            raise mp.TimeoutError('The call to `render_wait` has timed out after '
                '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        result, success = self.parent_pipes[0].recv()
        self._raise_if_errors([success])
        self._state = self.default_state

        return result

    def render(self, *args):
        self.render_async()
        return self.render_wait()



# def AsyncVectorMultiEnv(env_fns):
#     return AsyncVectorEnv(env_fns, worker=_worker_shared_memory)


def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                pipe.send((observation, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                if all(done):
                    observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_check_observation_space':
                pipe.send((data == env.observation_space, True))
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                                    'be one of {`reset`, `step`, `seed`, `close`, '
                                    '`_check_observation_space`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send((None, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                if all(done):
                    observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send(((None, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == 'render':
                img = env.render('rgb_array')
                pipe.send((img, True))
            elif command == '_check_observation_space':
                pipe.send((data == observation_space, True))
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                                   'be one of {`reset`, `step`, `seed`, `close`, '
                                   '`_check_observation_space`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


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
