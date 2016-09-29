import pml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


obs = pml.loadTxt('/tmp/obs.txt')
states = pml.loadTxt('/tmp/states.txt')

mean = pml.loadTxt('/tmp/mean.txt')
cpp = pml.loadTxt('/tmp/cpp.txt')

mean2 = pml.loadTxt('/tmp/mean2.txt')
cpp2 = pml.loadTxt('/tmp/cpp2.txt')

mean3 = pml.loadTxt('/tmp/mean3.txt')
cpp3 = pml.loadTxt('/tmp/cpp3.txt')

obs = obs[0, :]
states = states[0, :]
mean = mean[0, :]
mean2 = mean2[0, :]
mean3 = mean3[0, :]

fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(7, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1])

ax = plt.subplot(gs[0])
ax.plot(range(len(obs)), obs, 'b-')
ax.plot(range(len(obs)), states, 'r-')

ax = plt.subplot(gs[1])
ax.plot(range(len(mean)), mean, 'g-')

ax = plt.subplot(gs[2])
ax.vlines(np.arange(0, len(cpp)), 0, cpp,  colors='r', linewidth=2)

ax = plt.subplot(gs[3])
ax.plot(range(len(mean2)), mean2, 'c-')

ax = plt.subplot(gs[4])
ax.vlines(np.arange(0, len(cpp2)), 0, cpp2,  colors='r', linewidth=2)

ax = plt.subplot(gs[5])
ax.plot(range(len(mean3)), mean3, 'c-')

ax = plt.subplot(gs[6])
ax.vlines(np.arange(0, len(cpp3)), 0, cpp3,  colors='r', linewidth=2)

plt.show()