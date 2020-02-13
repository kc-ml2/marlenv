import gym
import envs
import pytest 
import time 

def test_gridPath():
		
	env = gym.make('GridPath-v0')

	env.reset()
	done = [False for i in range(env.n_agents)]
	while not all(done):

		obs, r, done, info = env.step([ env.action_space[i].sample() for i in range(env.n_agents)])

		env.render_graphic()
		# time.sleep(0.01)
	env.render_graphic()

	env.close()

def test_gridExplore():
	env = gym.make('GridExplore-v1')

	env.reset()
	done = [False for i in range(env.n_agents)]
	while not all(done):

		obs, r, done, info = env.step([ env.action_space[i].sample() for i in range(env.n_agents)])

		env.render_graphic()
		time.sleep(0.01)

	env.render_graphic()

	env.close()