import mmap
import struct
import numpy as np
from typing import Tuple
import socketio
# import logging
# import threading

MMAP_JOINT_POSE_COMMAND_PATH = "/tmp/ark_joint_commands"
MMAP_OBSERVATIONS_PATH = "/tmp/ark_observations"
JOINT_POSE_FILE_SIZE = 128

OBSERVATION_IMAGE_SIZE = 640*480*3
OBSERVATION_BUFFER_ENTRY_SIZE = OBSERVATION_IMAGE_SIZE*4+128
OBSERVATION_BUFFER_ENTRY_COUNT = 2
OBSERVATIONS_FILE_SIZE = OBSERVATION_BUFFER_ENTRY_SIZE * OBSERVATION_BUFFER_ENTRY_COUNT

# IMPORTANT YOU MUST HAVE LOTS OF RAM OR ENABLE OVERCOMMIT MEMORY ON THE RUNNING MACHINE
# the following command should return 1 if it doesn't you need to enable overcommit
# cat /proc/sys/vm/overcommit_memory


class MLBridge:
    def __init__(self):
        self.sio = socketio.SimpleClient()
        self.sim_state = [0.0]
    
    def open(self):
        self.sio.connect('http://0.0.0.0:5555')
        with open(MMAP_JOINT_POSE_COMMAND_PATH, "r+b") as f:
            self.joint_pos_commands_mmap = mmap.mmap(f.fileno(), JOINT_POSE_FILE_SIZE)
        with open(MMAP_OBSERVATIONS_PATH, "r") as f:
            self.observations_mmap = mmap.mmap(f.fileno(), OBSERVATIONS_FILE_SIZE, access=mmap.ACCESS_READ)
    
    def _setup_socketio(self):
        self.sio = socketio.Server(cors_allowed_origins="*", async_mode='threading')
        # self.sio.on('reset', self._handle_reset)
        # self.app = Flask("vr_server")
        # self.app.wsgi_app = socketio.WSGIApp(self.sio, self.app.wsgi_app)

    def read_joint_command(self, joint_index):
        # Check if the joint index is valid
        if not (0 <= joint_index < 14):
            raise ValueError("Joint index out of range")

        # Seek to the position of the joint command in the file
        self.joint_pos_commands_mmap.seek(joint_index * 8)

        # Read 8 bytes from the file
        data = self.joint_pos_commands_mmap.read(8)

        # Unpack the data as a 64-bit floating point number
        joint_command = struct.unpack('d', data)[0]

        return joint_command

    def read_all_joint_commands(self) -> np.array:
        # Seek to the start of the file
        self.joint_pos_commands_mmap.seek(0)

        # Read all joint commands at once
        data = self.joint_pos_commands_mmap.read(14 * 8)

        # Unpack the data as 14 64-bit floating point numbers
        joint_commands = struct.unpack('14d', data)

        return np.array(joint_commands)
    
    def write_observation(self, qpos: np.array, camera_array: Tuple[np.array, np.array, np.array, np.array, np.array]):
        entry_index = self.get_current_entry_index()  # Get the current index

        data = struct.pack('d'*len(qpos), *qpos)

        for item in camera_array:
            data += self._rgba_to_uint_rgb(item).tobytes()  # Convert each item in the camera array to bytes

        # Write the data to the memory-mapped file
        start = entry_index * OBSERVATION_BUFFER_ENTRY_SIZE
        self.observations_mmap[start:start+len(data)] = data
        self.observations_mmap[-1] = entry_index  # Update the current index
    
    def reset_sim(self) -> float:
        self.sio.emit('reset')
        reset_completed = self.sio.receive(timeout=5)

        if reset_completed is not None:
            message, data = reset_completed

            if message == 'reset_completed':
                return data['reward']
    

    def read_observations(self) -> dict:
        observations = dict()
        images = dict()

        entry_index = self.get_current_entry_index()
        start = entry_index * OBSERVATION_BUFFER_ENTRY_SIZE
        end = start + OBSERVATION_BUFFER_ENTRY_SIZE
        data = self.observations_mmap[start:end]
        qpos = struct.unpack('d'*14, data[0:14*8])
        sim_state = struct.unpack('d'*1, data[14*8:15*8])
        camera_array = []
        for i in range(4):
            start = 9*14 + i*OBSERVATION_IMAGE_SIZE
            end = start + OBSERVATION_IMAGE_SIZE
            camera_array.append(np.frombuffer(data[start:end], dtype=np.uint8).reshape((480, 640, 3)))

        images['cam_high'] = camera_array[0]
        images['cam_low'] = camera_array[1]
        images['cam_left_wrist'] = camera_array[2]
        images['cam_right_wrist'] = camera_array[3]
        
        observations['qpos'] = np.array(qpos)
        observations['images'] = images
            
        return observations, sim_state

    
    def write_joint_commands(self, joint_commands: np.array):
        #  Convert the joint_commands to bytes using struct.pack
        joint_commands_bytes = struct.pack('14d', *joint_commands)
        joint_commands[6] = -joint_commands[6] + 1
        joint_commands[13] = -joint_commands[13] + 1
        self.joint_pos_commands_mmap[0:14*8] = joint_commands_bytes

    def get_current_entry_index(self):
        return (self.observations_mmap[-1] + 1) % 2  # Get the current index

    def close(self):
        self.joint_pos_commands_mmap.close()
        self.observations_mmap.close()
    
    def _rgba_to_uint_rgb(self, rgba: np.array) -> np.array:
        rgb = rgba[:,:,:3]
        return np.transpose(rgb, (1, 0, 2)).astype(np.uint8)