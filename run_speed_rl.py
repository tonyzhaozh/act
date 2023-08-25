from policy_speed_env import create_speed_env
#from rl.agents.DQN import DQNAgent
from rl.rainbowDQN.dqnAgent import DQNAgent
from utils import set_seed

def main():
    seed = 3407
    set_seed(seed)
    env = create_speed_env(use_parallel=True)

    model_path = "dynamic_act_speed_rainbow/rainbow_state"

    # parameters
    train_model = True
    load_model = True
    test_model = True

    num_frames = 1000000
    memory_size = 50000
    batch_size = 128
    target_update = 50
    num_tests = 1
    ckpt_save_freq = 50000

    gamma = 0.99
    name = "state_gamma" + repr(gamma) + "_" + "state_parallel"

    # train
    agent = DQNAgent(env, memory_size, batch_size, target_update, seed, gamma = gamma, name = name, ckpt_save_freq=ckpt_save_freq)
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
