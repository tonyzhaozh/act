import numpy as np
import json
from policy_speed_env import test_speed_env
import matplotlib.pyplot as plt
from tqdm import tqdm

magnitude_history = []
PROFILE_LENGTH = 25
MAX_SPEED = 10.0

def speed_profile_search(max_training_steps = 10000):
    episode_length = 500
    #profile = [1., 1., 1., 1., 1.]
    #profile = [2.270912801635182, 1.6739333666496263, 3.6791611252443315, 2.9596551195573024, 2.39143839342337]
    #profile = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    profile = [1 for i in range(PROFILE_LENGTH)]
    segments = (episode_length // len(profile)) + 1
    idx = 0
    step = 0
    for _ in tqdm(range(max_training_steps)):
        updated_profile = profile.copy()
        if updated_profile[idx] < MAX_SPEED:
            updated_profile[idx] += np.random.uniform(0.0, 0.6)
            updated_profile[idx] = min(updated_profile[idx], MAX_SPEED)
        #print("Testing:", updated_profile)

        res = test_speed_env(speed_func=lambda obs, t: updated_profile[int(t) // segments])
        success = res == 100

        if success:
            profile = updated_profile


        idx = (idx + 1) % len(profile)

        step += 1
        magnitude = np.linalg.norm(profile) / np.sqrt(PROFILE_LENGTH)
        print("Step:",step, " Magnitude:", magnitude)

        magnitude_history.append(magnitude)

        if step % 10 == 0:
            with open(f'./tmp/speed_profile_{PROFILE_LENGTH}.json', 'w') as f:
                f.write(json.dumps(profile))

            save_history()

    with open(f'./tmp/speed_profile_{PROFILE_LENGTH}.json', 'w') as f:
        f.write(json.dumps(profile))

    save_history()

def test_speed_profile():
    episode_length = 500
    with open(f'./tmp/speed_profile_{PROFILE_LENGTH}.json', 'r') as f:
        profile = json.load(f)
    segments = (episode_length // len(profile)) + 1
    speed_func = lambda obs, t: profile[int(t) // segments]

    timesteps = np.arange(501)
    speed_profile = [speed_func(None, t) for t in timesteps]

    # Plot the speed profile
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, speed_profile, label="Speed Profile")
    plt.xlabel("Timesteps")
    plt.ylabel("Speed")
    plt.title("Speed Profile over 500 Timesteps")
    plt.legend()
    plt.grid()
    plt.savefig(f'tmp/speed_profile_{PROFILE_LENGTH}.png')
    plt.close()

    test_speed_env(speed_func=speed_func, onscreen_render=True, use_parallel=True)

def test_speed_profile_constant():
    speed_func = lambda obs, t: 1.0

    test_speed_env(speed_func=speed_func, onscreen_render=True, use_parallel=True)


def save_history():
    with open(f'./tmp/speed_history_{PROFILE_LENGTH}.json', 'w') as f:
        f.write(json.dumps(magnitude_history))

if __name__=='__main__':
    #speed_profile_search(3000)
    #save_history()
    test_speed_profile()
