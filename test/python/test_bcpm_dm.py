import pml
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import numpy as np


def get_cps(states):
    cps = np.zeros((states.shape[1], 1))
    for i in range(1,len(cps)):
        if any(states[:, i] != states[:, i-1]):
            cps[i] = 1
    return cps


def plot(ax, alpha, cpp, color='r'):
    K = alpha.shape[0]
    T = alpha.shape[1]
    ax.pcolormesh(alpha, cmap=plt.cm.Greys, vmin=0, vmax=np.max(np.max(alpha)))
    ax.vlines(np.arange(0, T), 0, cpp*K,  colors=color, linestyles='-',
             linewidth=2, label='change point prob.')


def plot_dm(em_result=False):
    states = pml.loadTxt('/tmp/states.txt')
    obs = pml.loadTxt('/tmp/obs.txt')

    if em_result == 'True':
        mean = pml.loadTxt('/tmp/mean.txt')
        cpp = pml.loadTxt('/tmp/cpp.txt')

        mean2 = pml.loadTxt('/tmp/initial/mean.txt')
        cpp2 = pml.loadTxt('/tmp/initial/cpp.txt')

        mean3 = pml.loadTxt('/tmp/final/mean.txt')
        cpp3 = pml.loadTxt('/tmp/final/cpp.txt')
    else:
        mean = pml.loadTxt('/tmp/filtering/mean.txt')
        cpp = pml.loadTxt('/tmp/filtering/cpp.txt')

        mean2 = pml.loadTxt('/tmp/smoothing/mean.txt')
        cpp2 = pml.loadTxt('/tmp/smoothing/cpp.txt')

        mean3 = pml.loadTxt('/tmp/online_smoothing/mean.txt')
        cpp3 = pml.loadTxt('/tmp/online_smoothing/cpp.txt')

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

    ax0 = plt.subplot(gs[0])
    ax0.set_title("Observations")
    plot(ax0, obs, get_cps(states), 'b')
    ax0.set_xticks([])

    ax1 = plt.subplot(gs[1])
    plot(ax1, mean, cpp)
    ax1.set_xticks([])

    ax2 = plt.subplot(gs[2])
    plot(ax2, mean2, cpp2)
    ax2.set_xticks([])

    ax3 = plt.subplot(gs[3])
    plot(ax3, mean3, cpp3)

    if em_result == 'True' or em_result == '1' or em_result == True:
        ax0.legend(['Change points'])
        ax1.set_title("Smoothing - True Parameters")
        ax2.set_title("Smoothing - Initial EM")
        ax3.set_title("Smoothing - After EM")
    else:
        ax0.legend(['Change points'])
        ax1.set_title("Filtering")
        ax2.set_title("Smoothing")
        ax3.set_title("Online Smoothing")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_dm(sys.argv[1])