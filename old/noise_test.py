import numpy as np
noise = np.random.normal(scale=1, size=100)
for x in range(1000):
    noise = noise + np.random.normal(scale=1, size=100)

print(np.std(noise))