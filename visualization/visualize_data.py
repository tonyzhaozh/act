import json
import numpy as np
import matplotlib.pyplot as plt

data_path = "/scr2/davidyuan/train_logs/sim_cube_transfer_baseline/speed_result.json"

# Read the JSON data from the file
with open(data_path, 'r') as file:
    data = json.load(file)

# Extract speed and success values from the data
speed_values = [float(entry['speed']) for entry in data]
success_values = [entry['success'] for entry in data]

# Calculate success rate in each 0.1 bin for the speed
bins = np.arange(0, 10, 0.25)

# Calculate success rate in each 0.1 bin for the speed
bin_counts, _ = np.histogram(speed_values, bins=bins)
success_counts, _ = np.histogram(speed_values, bins=bins, weights=success_values)

# Calculate success rates
success_rates = success_counts / bin_counts

# Plot the success rates
plt.figure(figsize=(10, 6))
plt.bar(bins[:-1], success_rates, width=0.2)
plt.xlabel('Speed')
plt.ylabel('Success Rate')
plt.title('Success Rate in Each 0.25 Bin for Speed')

plt.show()