import pml
import numpy as np
import matplotlib.pyplot as plt

def get_cpp(states):
    cps = np.zeros(len(states))
    for i in range(1, len(states)):
        if(states[i] != states[i-1]):
            cps[i] = 1
    return cps


def plot_coupled_poisson_reset():
    states = pml.loadTxt('/tmp/states.txt')
    obs = pml.loadTxt('/tmp/obs.txt')
    mean = pml.loadTxt('/tmp/mean.txt')
    cpp = pml.loadTxt('/tmp/cpp.txt')
    real_cps = get_cpp(states[0,:])

    T = states.shape[1]

    fig, ax= plt.subplots(3, sharex=True, figsize=(18, 10))

    ax[0].plot(range(T), states[0, :], 'r-')
    ax[0].plot(range(T), obs[0, :], 'b-')
    ax[0].plot(range(T), mean[0, :], 'g-')
    ax[0].set_title("Sequence")
    ax[0].legend(['Hidden States', 'Observations'])

    ax[1].plot(range(T), states[1, :], 'r-')
    ax[1].plot(range(T), obs[1, :], 'b-')
    ax[1].plot(range(T), mean[1, :], 'g-')
    ax[1].set_title("Sequence")
    ax[1].legend(['Hidden States', 'Observations'])

    ax[2].vlines(np.arange(0,T), 0, cpp,  colors='r', linestyles='-', linewidth=2, label='change point prob.')
    ax[2].vlines(np.arange(0,T), 0, real_cps,  colors='b', linestyles='--', linewidth=2, label='real change points')
    ax[2].set_ylim([0, 1])

    fig.show()


if __name__ == '__main__':
    plot_coupled_poisson_reset()
    input("enter a character to continue: ")