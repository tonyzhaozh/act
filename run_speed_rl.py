import time

from policy_speed_env import create_speed_env
#from rl.agents.DQN import DQNAgent
from rl.rainbowDQN.dqnAgent import DQNAgent
from utils import set_seed

from ee_sim_env import DISABLE_RENDER


import IPython
e = IPython.embed

def run(seed):
    ######################################################

    # training
    num_frames = 1000000
    batch_size = 128

    # RL params
    frame_skip = 10
    gamma = 0.99
    memory_size = 1000000
    target_update = 50
    num_frames = num_frames // frame_skip


    # no speed
    # def reward_fn(speed, done, success):
    #     reward = 100 if done and success else 0
    #     return reward

    # pow 1
    # def reward_fn(speed, done, success):
    #     reward = 100 if done and success else 0
    #     reward += (speed ** 1.0) / 100
    #     return reward

    # pow 2
    def reward_fn(speed, done, success):
        reward = 100 if done and success else 0
        reward += (speed ** 2.0) / 100
        return reward

    # pow 2 higher weight
    # def reward_fn(speed, done, success):
    #     reward = 80 if done and success else 0
    #     reward += (speed ** 2.0) / 50
    #     return reward

    # saving
    ckpt_save_freq = 50000

    # name
    name = f"sweep4_SpeedRewardPow2" + f"_seed{seed}"
    model_path = f"/scr2/tonyzhao/dynamic_train_logs/{name}"

    # others
    disable_render = True
    train_model = True
    load_model = False
    test_model = True
    num_tests = 1

    ######################################################

    print(f'\n\n===============Starting experiment: {name}===============\n\n')
    time.sleep(1)

    set_seed(seed)
    DISABLE_RENDER[0] = disable_render
    env = create_speed_env(reward_fn=reward_fn, use_parallel=False)

    # train
    agent = DQNAgent(env,
                     memory_size,
                     batch_size,
                     target_update,
                     seed,
                     frame_skip=frame_skip,
                     gamma=gamma,
                     name=name,
                     ckpt_save_freq=ckpt_save_freq,
                     log_dir="logs",
                     )
    if load_model:
        print("Loading model from: ", model_path)
        agent.load(model_path)
    if train_model:
        print("Training model...")
        agent.train(num_frames)
        agent.save(model_path)

    # # test
    # if test_model:
    #     env = create_speed_env(onscreen_render=True, save_video=True)

    #     agent = DQNAgent(env, memory_size, batch_size, target_update, seed)
    #     print("Loading model from: ", model_path)
    #     agent.load(model_path)

    #     print("Testing model...")
    #     agent.test(num_tests)

if __name__=='__main__':
    for seed in [0, 1, 2, 3, 4]:
        run(seed)
