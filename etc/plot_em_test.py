import pml
import numpy as np
import matplotlib.pyplot as plt


def plot_data(ax):
    obs = pml.loadTxt('/tmp/obs.txt')
    temp = pml.loadTxt('/tmp/cps.txt')
    cps = [i for i, x in enumerate(temp) if x == 1]

    # Plot histogram
    num_headers = obs.shape[0]
    obs_length = obs.shape[1]
    y, x = np.mgrid[slice(0, num_headers+1, 1), slice(0, obs_length+1, 1)]

    ax.pcolormesh(x, y, obs, cmap=plt.cm.Greys, vmin=0, vmax=np.max(np.max(obs)))
    ax.vlines(cps, 0, num_headers,  colors='r', linewidth=2)

    # Decorate
    ax.set_title('DM Data', fontsize=16)
    # Y-Axis
    ax.get_yaxis().set_tick_params(length=0)
    ax.set_yticks(np.arange(num_headers) + 0.2)
    # X-Axis
    ax.get_xaxis().set_tick_params(direction='out', top='off')
    ax.set_xlabel("Time", fontsize=16)
    ax.set_xticks(np.arange(0, obs_length, 20))
    ax.set_aspect('auto')


def plot_result(ax, mean, cpp):

    # Plot histogram
    num_headers = mean.shape[0]
    obs_length = mean.shape[1]
    y, x = np.mgrid[slice(0, num_headers+1, 1), slice(0, obs_length+1, 1)]

    ax.pcolormesh(x, y, mean, cmap=plt.cm.Greys, vmin=0, vmax=np.max(np.max(mean)))
    ax.vlines(np.arange(0, obs_length), 0, num_headers*cpp,  colors='b',
              linewidth=2,
              label='change point prob.')

    # Decorate
    ax.set_title('DM Data', fontsize=16)
    # Y-Axis
    ax.get_yaxis().set_tick_params(length=0)
    ax.set_yticks(np.arange(num_headers) + 0.2)
    # X-Axis
    ax.get_xaxis().set_tick_params(direction='out', top='off')
    ax.set_xlabel("Time", fontsize=16)
    ax.set_xticks(np.arange(0, obs_length, 20))
    ax.set_aspect('auto')


if __name__ == '__main__':

    fig, ax= plt.subplots(3, sharex=True,figsize=(18,10))

    plot_data(ax[0])

    mean = pml.loadTxt('/tmp/mean.txt')
    cpp = pml.loadTxt('/tmp/cpp.txt')
    plot_result(ax[1], mean, cpp)
    ax[1].set_title('Result with true parameters')

    mean2 = pml.loadTxt('/tmp/mean2.txt')
    cpp2 = pml.loadTxt('/tmp/cpp2.txt')
    plot_result(ax[2], mean2, cpp2)
    ax[2].set_title('Result with learned parameters')

    plt.tight_layout()
    plt.show()