from isaac_sim_env import IsaacSimEnv


def make_isaac_sim_env() -> IsaacSimEnv:
    return IsaacSimEnv()


def move_grippers():
    print('moving grippers should probaly implement this')