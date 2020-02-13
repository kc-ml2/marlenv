from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines.common.policies import FeedForwardPolicy, register_policy, ActorCriticPolicy, CnnPolicy, MlpPolicy

import random
import envs
import gym 
import argparse
import time 
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter, make_vec_env
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm

import tensorflow as tf 
import numpy as np 
import torch
from stable_baselines import DQN
from stable_baselines.deepq.policies import FeedForwardPolicy as DqnFFPolicy
from stable_baselines.common.env_checker import check_env
from utils.common import ArgumentParser, save_model, load_model

def custom_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=32, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_2)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64],
                                                          vf=[64])],
                                           feature_extraction="cnn", cnn_extractor =custom_cnn )

class DqnCnnPolicy(DqnFFPolicy):
    def __init__(self, *args, **kwargs):
        super(DqnCnnPolicy, self).__init__(*args, **kwargs,
                                           feature_extraction="cnn", cnn_extractor =custom_cnn )

# Register the policy, it will check that the name is not already taken
register_policy('CustomPolicy', CustomPolicy)
register_policy('DqnCnnPolicy', DqnCnnPolicy)


def ppo2train(args):

    with tf.device('/device:CUDA:1'):
        # gpus = tf.config.experimental.list_physical_devices('CUDA')

        # tf.config.experimental.set_visible_devices(gpus[1], 'CUDA')
        env = make_vec_env('python_1p-v0', n_envs=4)

        # env = gym.make('python_1p-v0')
        # env = Monitor(env, filename=None, allow_early_resets=True)
        # env = DummyVecEnv([lambda: env])

        model = PPO2(CustomPolicy, env, verbose=1, n_steps=500, nminibatches=16)
        model.learn(5000000)
        model.save("ppo2withcnnxr.pth")

        mean = 0
        finished = 0
        rewards=0
        for i in range(10):
            obs = env.reset()
            for step in range(500):
                action, _ = model.predict(obs)

                obs, reward, done, info = env.step(action)
                rewards += reward 

                env.env_method("render")
                if all(done):
                    break
                time.sleep(0.05)
            print(rewards , " points " )
            mean += rewards
            
            finished +=1 

        print("mean score : " , mean/10)
        # model.load("mlppolicy.pth")
        # #mlp0103.pth

def traindqn(args):
    

    # with tf.device('/device:CUDA:1'):
    with tf.device('/gpu:0'):

        env = gym.make('python_1p-v0')
        # env = Monitor(env, filename=None, allow_early_resets=True)
        env = DummyVecEnv([lambda: env])

        model = DQN(DqnCnnPolicy, env, verbose=1,learning_rate=0.0001, exploration_fraction=0.4, train_freq=10)
        model.learn(5000000)
        model.save("dqnwithcnn.pth")

        # model.load("mlppolicy.pth")
        # #mlp0103.pth





def test(args):

    env = make_vec_env('python_1p-v0', n_envs=4)

    # env = gym.make('python_1p-v0')
    # env = Monitor(env, filename=None, allow_early_resets=True)
    # env = DummyVecEnv([lambda: env])

    # model = PPO2(CustomPolicy, env, verbose=1, n_steps=500, nminibatches=16)

    model = PPO2.load("ppo2initial.pth")


    mean = 0
    finished = 0
    while finished != 100:
            
        obs= env.reset()
        # print(state)
        rewards = 0 
        for step in range(500):
            action, _ = model.predict(obs)

            obs, reward, done, info = env.step(action)
            rewards += reward 

            env.env_method("render")
            if all(done):
                break
            time.sleep(0.05)
        print(rewards , " points " )
        mean += rewards
        
        finished +=1 

    print("mean score : " , mean/100)



def runGUI(args):

    # env = gym.make('python_1p-v0', full_observation = True)
    env = gym.make('python_4p-v1', full_observation=False, vision_range=10)
    
    # net = PPO2.load("64batch_wall.pth")
    net = PPO2.load("4p_compete.pth")
    # net = PPO2.load("assets/ppo2_GridExplore_10x10.pth")

    gui = envs.ML2PythonGUI(env, args)

    # gui.baselines_run(net)
    gui.run_model(net)

def runGUImulti(args):

    env = gym.make('python_4p-v1', full_observation=False, vision_range=10)
    net = PPO2.load("4p_compete.pth")
    # net = PPO2.load("4p_map_32b_2kstep.pth")

    obs = env.reset()

    done_n = [False for _ in range(4)]

    while not all(done_n):
        actions = []
        for i in range(4):
            action , _ = net.predict(obs[i])
            # print(action)
            actions.append(action)

        obs, reward, done_n, info = env.step(actions)

        env.render()
        time.sleep(0.05)

    env.close()


def saveVideo(args):
    import gym
    from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
    from gym.wrappers import Monitor
    import imageio
    from PIL import Image

    env_id = 'python_1p-v1'

    env = gym.make('python_1p-v0')
    env = Monitor(env, directory='video',video_callable=lambda episode_id: True, force=True)

    # ppo multiagent
    # shape addition to relative position
    # batch snake action 
    
    video_folder = 'logs/videos/'
    video_length = 100
    # env = DummyVecEnv([lambda: gym.make(env_id)])
    # env = Monitor(env, './video', force=store_true)
    net = PPO2.load("ppo2_10x10.pth")
    images=[]
    obs= env.reset()
    img = env.render()
    while True:
        image = Image.fromarray(img)

        image = image.resize((400, 400))  
        image = np.asarray(image, dtype="uint8" )
        images.append(image)
        img = env.render()
        action , _ = net.predict(obs)
        obs, r, done, info = env.step(action)
        if done: 
            break
    imageio.mimsave('ppo2.gif', np.array(images), fps=15 )


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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    globals()[args.mode](args)
    
