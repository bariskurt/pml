import pml
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def get_cpp(states):
    T = states.shape[1]
    cps = np.zeros(T)
    for i in range(1, T):
        if not np.array_equal(states[:, i], states[:, i-1]):
            cps[i] = 1
    return cps

def plot_multinomial_reset():
    states = pml.loadTxt('/tmp/states.txt')
    obs = pml.loadTxt('/tmp/obs.txt')

    mean1 = pml.loadTxt('/tmp/mean_filtering.txt')
    cpp1 = pml.loadTxt('/tmp/cpp_filtering.txt')
    mean2 = pml.loadTxt('/tmp/mean_smoothing.txt')
    cpp2 = pml.loadTxt('/tmp/cpp_smoothing.txt')
    mean3 = pml.loadTxt('/tmp/mean_fixed_lag.txt')
    cpp3 = pml.loadTxt('/tmp/cpp_fixed_lag.txt')
    cpp_real = get_cpp(states)

    K = obs.shape[0]
    T = obs.shape[1]

    f, ax = plt.subplots(4, sharex=True,figsize=(18,10))

    y, x = np.mgrid[slice(0, 1, 1/K),slice(0, T+1, 1)]
    # Plot Data Histogram
    ax[0].pcolormesh(x, y, obs, cmap=plt.cm.Greys, vmin=0, vmax=np.max(np.max(obs)))
    ax[0].vlines(np.arange(0,T), 0, cpp_real,  colors='b', linestyles='-',
                    linewidth=2, label='change point prob.')
    ax[0].set_title('Observations')

    ax[1].pcolormesh(x, y, mean1, cmap=plt.cm.Greys, vmin=0, vmax=1)
    ax[1].vlines(np.arange(0, T), 0, cpp1,  colors='r', linestyles='-',
                    linewidth=2, label='change point prob.')
    ax[1].set_title('Filtered Density')

    ax[2].pcolormesh(x, y, mean2, cmap=plt.cm.Greys, vmin=0, vmax=1)
    ax[2].vlines(np.arange(0, T), 0, cpp2,  colors='r', linestyles='-',
                    linewidth=2, label='change point prob.')
    ax[2].set_title('Smoothed Density')

    ax[3].pcolormesh(x, y, mean3, cmap=plt.cm.Greys, vmin=0, vmax=1)
    ax[3].vlines(np.arange(0, T), 0, cpp2,  colors='r', linestyles='-',
                 linewidth=2, label='change point prob.')
    ax[3].set_title('Fixed Lag Smoothing')

    plt.show()


if __name__ == '__main__':
    plot_multinomial_reset()