from policy_speed_env import create_speed_env
#from rl.agents.DQN import DQNAgent
from rl.rainbowDQN.dqnAgent import DQNAgent
from utils import set_seed

from ee_sim_env import DISABLE_RENDER


import IPython
e = IPython.embed

def main():
    ######################################################
    seed = 0

    # training
    num_frames = 1000000
    batch_size = 128

    # RL params
    frame_skip = 10
    gamma = 0.99
    memory_size = 1000000
    target_update = 50

    def reward_fn(speed, done, success):
        reward = 100 if done and success else 0
        reward += (speed ** 2.0) / 200
        return reward

    # saving
    ckpt_save_freq = 50000

    # name
    name = f"try3_base_update_seed{seed}"
    model_path = f"/scr2/tonyzhao/train_logs/dynamic_act_speed_rainbow/{name}"

    # others
    disable_render = True
    train_model = True
    load_model = False
    test_model = True
    num_tests = 1

    ######################################################

    set_seed(seed)
    DISABLE_RENDER[0] = disable_render
    env = create_speed_env(reward_fn=reward_fn, use_parallel=True)

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

    # test
    if test_model:
        env = create_speed_env(onscreen_render=True, save_video=True)

        agent = DQNAgent(env, memory_size, batch_size, target_update, seed)
        print("Loading model from: ", model_path)
        agent.load(model_path)

        print("Testing model...")
        agent.test(num_tests)

if __name__=='__main__':
    main()
