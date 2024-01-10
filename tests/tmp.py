from marlenv.wrappers import make_snake
import gym
from gym.vector.async_vector_env import AsyncVectorEnv
import sys
from gym.vector.utils import write_to_shared_memory
# env, observation_shape, action_shape, properties = make_snake(env_id='SnakeCoop-v1', num_snakes=4, num_envs=2)
# env.reset()
# env.render()

env, observation_shape, action_shape, properties = make_snake(num_envs=10, num_snakes=4, env_id='SnakeCoop-v1')
print(env.reset().shape)
print(env.render().shape)