# import argparse
# import time
#
# import envs
# import gym
# import torch
# from stable_baselines import PPO2
#
#
# def test_gridPath():
#
# 	env = gym.make('GridPath-v0')
#
# 	env.reset()
# 	done = [False for i in range(env.n_agents)]
# 	while not all(done):
#
# 		obs, r, done, info = env.step([ env.action_space[i].sample() for i in range(env.n_agents)])
#
# 		env.render_graphic()
#
# 	env.render_graphic()
#
# 	env.close()
#
# def test_gridExplore():
# 	env = gym.make('GridExplore-v1')
#
# 	env.reset()
# 	done = [False for i in range(env.n_agents)]
# 	while not all(done):
#
# 		obs, r, done, info = env.step([ env.action_space[i].sample() for i in range(env.n_agents)])
#
# 		env.render_graphic()
#
# 	env.render_graphic()
#
# 	env.close()
#
#
# def test_python_1p():
# 	env = gym.make('python_1p-v0')
#
# 	env.reset()
# 	done = False
# 	while not done:
#
# 		obs, r, done, info = env.step([env.action_space.sample()])
# 		env.render()
# 		time.sleep(0.01)
#
# 	env.render()
#
# 	env.close()
#
# def test_python_4p():
# 	env = gym.make('python_4p-v1')
#
# 	env.reset()
# 	done = [False for i in range(env.num_players)]
# 	while not all(done):
#
# 		obs, r, done, info = env.step([env.action_space[i].sample() for i in range(env.num_players)])
# 		env.render()
#
# 	env.render()
#
# 	env.close()
#
#
# def test_graphics_python():
# 	env = gym.make('python_1p-v0')
# 	net = PPO2.load('assets/ppo2_GridExplore_10x10.pth')
#
# 	parser = argparse.ArgumentParser(description="ml2_snake")
#
# 	parser.add_argument("--mode", type=str, default='runGUI')
# 	parser.add_argument("--human", action='store_true')
# 	parser.add_argument("--cell_size", type=int, default=20)
#
# 	args = parser.parse_args()
# 	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# 	gui = envs.ML2PythonGUI(env,args)
# 	gui.baselines_run(net)
#
#
