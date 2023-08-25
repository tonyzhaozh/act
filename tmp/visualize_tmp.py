import matplotlib.pyplot as plt
import json

with open('speed_history_25.json', 'r') as f:
    res = json.load(f)

plt.plot(res)
plt.xlabel("Training steps")
plt.ylabel("Relative speed")
plt.title("training curve for searching")
plt.savefig("training_curve_25.png")