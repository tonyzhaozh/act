import h5py
import matplotlib.pyplot as plt

path = "./aloha_data/episode_0.hdf5"
h5 = h5py.File(path, 'r')
h5.keys()  # Outputs Trials per h5, Trial 0, 1, 2, ...
print("keys: ", h5.keys())
print("actions: ", h5["action"])
# print("intents: ", h5["intent"])

# print(h5["task"][0].decode('utf-8'))
print("observation: ", h5["observations"].keys())
print("observation: ", h5["observations"]["images"].keys())
print(h5["observations"]["images"]["angle"].shape)

tensor = h5["observations"]["images"]["angle"][0]
print(tensor.shape)

# Plotting the tensor as an image
plt.imshow(tensor)
plt.title('Random Image from Tensor')
plt.axis('off')  # Turn off the axis numbers and ticks
plt.show()
# h5['Trial0'].keys()  # outputs data, derived, and config
# # to extract the data
# h5["Trial0"]['data'][data_key]  # where data_key is one of the cells from the data tab