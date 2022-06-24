import sys
from enum import Enum
import multiprocessing as mp
import numpy as np

import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.error import AlreadyPendingCallError, NoAsyncCallError
from gym.vector.utils import write_to_shared_memory


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
                shape=(h, w, self.obs_ch), dtype=np.uint8)  # 8
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
                shape=(self.num_snakes, h, w, self.obs_ch), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(
                self.low, self.high,
                shape=(self.num_snakes, *self.grid_shape, self.obs_ch),
                dtype=np.uint8)


class AsyncVectorMultiEnv(AsyncVectorEnv):
    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, worker=_worker_shared_memory, **kwargs)
        self.default_state = None

    def render_async(self):
        self._assert_is_running()
        if self._state.value != AsyncState.DEFAULT.value:
            raise AlreadyPendingCallError(
                'Calling `render_async` while waiting '
                'for a pending call to `{0}` to complete.'.format(
                    self._state.value), self._state.value)
        else:
            self.default_state = self._state
        self.parent_pipes[0].send(('render', None))
        self._state = AsyncState.WAITING_RENDER

    def render_wait(self, timeout=None):
        self._assert_is_running()
        if self._state.value != AsyncState.WAITING_RENDER.value:
            raise NoAsyncCallError(
                'Calling `render_wait` without any prior '
                'call to `render_async`.', AsyncState.WAITING_RESET.value)

        if not self._poll(timeout):
            self._state = self.default_state
            raise mp.TimeoutError(
                'The call to `render_wait` has timed out after '
                '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        result, success = self.parent_pipes[0].recv()
        self._raise_if_errors([success])
        self._state = self.default_state

        return result

    def render(self, *args):
        self.render_async()
        return self.render_wait()


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
                if type(done) == bool:
                    cond = done
                else:
                    cond = all(done)
                if cond:
                    observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
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
                pipe.send((data == env.observation_space, True))
            else:
                raise RuntimeError(
                    'Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, '
                    '`_check_observation_space`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory,
                          error_queue):
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
                if type(done) == bool:
                    cond = done
                else:
                    cond = all(done)
                if cond:
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
                raise RuntimeError(
                    'Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, '
                    '`_check_observation_space`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def make_snake(num_envs=1, num_snakes=4, env_id="Snake-v1", **kwargs):
    """A function just for me.

    :param my_arg: The first of my arguments.
    :param my_other_arg: The second of my arguments.

    :returns: A message (just for me, of course).
    """
    if num_snakes > 1:
        env_wrapper = SingleMultiAgent
    else:
        env_wrapper = SingleAgent
    if num_envs > 1:
        vec_wrapper = AsyncVectorMultiEnv

    def _make():
        env = gym.make(env_id, num_snakes=num_snakes, **kwargs)
        env = env_wrapper(env)
        return env

    dummyenv = _make()
    observation_shape = dummyenv.observation_space.shape
    if num_snakes > 1:
        observation_shape = observation_shape[1:]
    action_shape = (dummyenv.action_space.n,)
    high = dummyenv.observation_space.high
    low = dummyenv.observation_space.low

    if 'Discrete' in str(type(dummyenv.action_space)):
        action_info = {'action_n': dummyenv.action_space.n}
        discrete = True

    if 'Box' in str(type(dummyenv.action_space)):
        action_info = {
            'action_high': dummyenv.action_space.high,
            'action_low': dummyenv.action_space.low
        }
        discrete = False

    del dummyenv

    if num_envs > 1:
        env = vec_wrapper([_make for _ in range(num_envs)])
    else:
        env = _make()

    properties = {
        'high': high,
        'low': low,
        'num_envs': num_envs,
        'num_snakes': num_snakes,
        'discrete': discrete,
        'action_info': action_info,
    }

    return env, observation_shape, action_shape, properties
