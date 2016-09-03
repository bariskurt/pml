import pml
import matplotlib.pyplot as plt


def plot_coupled_poisson_reset():
    states = pml.loadTxt('/tmp/states.txt')
    obs = pml.loadTxt('/tmp/obs.txt')
    T = states.shape[1]

    fig, ax= plt.subplots(2, sharex=True, figsize=(18, 8))

    ax[0].plot(range(T), states[0, :], 'r-')
    ax[0].plot(range(T), obs[0, :], 'b-')
    ax[0].set_title("Sequence")
    ax[0].legend(['Hidden States', 'Observations'])

    ax[1].plot(range(T), states[1, :], 'r-')
    ax[1].plot(range(T), obs[1, :], 'b-')
    ax[1].set_title("Sequence")
    ax[1].legend(['Hidden States', 'Observations'])

    fig.show()


if __name__ == '__main__':
    plot_coupled_poisson_reset()
    input("enter a character to continue: ")