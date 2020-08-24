import matplotlib.pyplot as plt
import numpy as np

num = np.arange(0, 2.0, 0.1)

print(type(num))

advantage = -5

epsilon = 0.2

print(num)

def formula(x):
    return np.minimum(x * advantage, np.clip(x, 1 - epsilon, 1 + epsilon) * advantage)


results = formula(num)

print(results)

plt.plot(num, results)

plt.show()

