import pml
import matplotlib.pyplot as plt


def plot_poisson_reset():
    states = pml.loadTxt('/tmp/states.txt')
    obs = pml.loadTxt('/tmp/obs.txt')
    mean = pml.loadTxt('/tmp/mean.txt')
    cpp = pml.loadTxt('/tmp/cpp.txt')

    states = states[0, :]
    obs = obs[0, :]
    mean = mean[0, :]
    T = len(states)

    fig, ax= plt.subplots(2, sharex=True, figsize=(18, 8))

    ax[0].plot(range(T), states, 'r-')
    ax[0].plot(range(T), obs, 'b-')
    ax[0].plot(range(T), mean, 'g-')
    ax[0].set_title("Sequence")
    ax[0].legend(['Hidden States', 'Observations', 'Mean'])

    ax[1].bar(range(T), cpp)
    ax[1].set_ylim([0, 1])


    fig.show()


if __name__ == '__main__':
    plot_poisson_reset()
    input("enter a character to continue: ")