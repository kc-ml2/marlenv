import random
import gym
import argparse
import time
import tensorflow as tf
import numpy as np
from stable_baselines import DQN, PPO2
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.deepq.policies import FeedForwardPolicy as DqnFFPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc

import marlenv.envs
from marlenv.utils.subproc_vec_env import SubprocVecEnv
from marlenv.utils.wrappers import SAhandler, Vechandler, FalseVecEnv


def custom_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=2,
                         stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=2,
                         stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_2)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64],
                                                          vf=[64])],
                                           feature_extraction="cnn",
                                           cnn_extractor=custom_cnn)


class DqnCnnPolicy(DqnFFPolicy):
    def __init__(self, *args, **kwargs):
        super(DqnCnnPolicy, self).__init__(*args, **kwargs,
                                           feature_extraction="cnn",
                                           cnn_extractor=custom_cnn)


# Register the policy, it will check that the name is not already taken
register_policy('CustomPolicy', CustomPolicy)
register_policy('DqnCnnPolicy', DqnCnnPolicy)


def traindqn(args):
    '''
    An example with an agent which requires a single agent setup
    '''
    with tf.device('/gpu:0'):
        env = gym.make('python_1p-v0')
        env = SAhandler(env)

        model = DQN(DqnCnnPolicy, env, verbose=1, learning_rate=5e-4,
                    exploration_fraction=0.1,
                    exploration_final_eps=0.01,
                    buffer_size=50000,
                    train_freq=1,
                    prioritized_replay=True,
                    target_network_update_freq=1000)
        model.learn(int(1e6))
        model.save("dqnwithcnn", cloudpickle=True)


def trainppo(args):
    '''
    An example with an agent which will play against itself (while learning)
    Each agent will be controlled by a copy of itself and all data will be used
    to train the agent.
    The env can be vectorized as below.
    '''
    # TODO: package the sequence into a function
    with tf.device('/gpu:0'):
        env = gym.make('python_4p-v1')
        num_players = env.num_players
        num_envs = 2

        env = make_vec_env(env, num_envs)
        env = Vechandler(env)
        env = FalseVecEnv([lambda: env])
        setattr(env, 'num_envs', num_players * num_envs)

        model = PPO2(CustomPolicy, env, noptepochs=10, nminibatches=8)
        model.learn(int(1e7))
        model.save("ppo_4p", cloudpickle=True)


def make_env(env_id, mpi_rank=0, subrank=0, seed=0, initializer=None, npcs=[]):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)
    env = gym.make(env_id)
    env = SAhandler(env)
    env.seed(seed + mpi_rank)
    env.set_npcs([npc for npc in npcs])
    return env


def make_vec_env(env, num_env):
    def make_thunk(rank):
        return lambda: env
    return SubprocVecEnv([make_thunk(i) for i in range(num_env)])


def trainmulti(args):
    '''
    An example of training a single agent in which the env holds different
    models to control other agents.
    Needs to provide num_players - 1 agents as "npcs".
    '''
    with tf.device('/gpu:0'):
        env = gym.make('python_4p-v1', full_observation=False, vision_range=10)
        env = SAhandler(env)
        env.set_npcs(["ppo_4p.pth" for _ in range(3)])
        env = make_vec_env(env, 10)

        model = PPO2(CustomPolicy, env, noptepochs=4, nminibatches=8)
        model.learn(int(1e7))
        model.save("ppo_4p", cloudpickle=True)


def runGUI(args):
    env = gym.make('python_4p-v1', full_observation=False, vision_range=10)
    net = PPO2.load("4p_compete.pth", policy=CustomPolicy)
    gui = envs.ML2PythonGUI(env, args)

    # gui.baselines_run(net)
    gui.run_model(net)


def runGUImulti(args):
    env = gym.make('python_4p-v1', full_observation=False, vision_range=10)
    net = PPO2.load("4p_compete.pth", policy=CustomPolicy)
    # net = PPO2.load("4p_map_32b_2kstep.pth")

    obs = env.reset()

    done_n = [False for _ in range(4)]

    while not all(done_n):
        actions = []
        for i in range(4):
            action, _ = net.predict(obs[i])
            actions.append(action)

        obs, reward, done_n, info = env.step(actions)

        env.render()
        time.sleep(0.05)

    env.close()


def saveImage(args):
    import imageio
    images = []

    env = gym.make('python_4p-v1', full_observation=False, vision_range=10)
    # net = PPO2.load("4p_compete.pth", policy=CustomPolicy)
    net = PPO2.load("ppo_4p.pkl", policy=CustomPolicy)
    # net = PPO2.load("ppo_4p.pth", policy=CustomPolicy)

    obs = env.reset()

    done_n = [False for _ in range(4)]

    while not all(done_n):
        actions = []
        for i in range(4):
            action, _ = net.predict(obs[i])
            actions.append(action)

        obs, reward, done_n, info = env.step(actions)
        img = env.render()

        images.append(img)

    env.close()

    imageio.mimsave('result.gif', images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ml2_snake")

    parser.add_argument("--mode", type=str, default='runGUI')
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument_group("interface options")
    parser.add_argument("--human", action='store_true')
    parser.add_argument("--cell_size", type=int, default=20)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    '''
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    '''
    globals()[args.mode](args)
