import os
import time
import numpy as np
import sounddevice as sd
import wavio

print(sd.query_devices('USB PnP Audio Device'))

max_timesteps = 200
dt = 0.02
dataset_dir = "."
dataset_name = "test"

# Start recording audio
sd.default.device = 'USB PnP Audio Device'
audio_duration = max_timesteps * dt  # Total duration of the episode
audio_sampling_rate = 48000  # Standard sampling rate for this device
audio_recording = sd.rec(int(audio_duration * audio_sampling_rate), samplerate=audio_sampling_rate, channels=1)

# Stop and save audio recording
t1 = time.time()
sd.wait()  # Wait until recording is complete
audio_recording_int16 = (audio_recording * (2**15 - 1)).astype(np.int16)
wavio.write(os.path.join(dataset_dir, dataset_name + ".wav"), audio_recording_int16, audio_sampling_rate, sampwidth=2)
print(f'Saving audio: {time.time() - t1:.1f} secs')