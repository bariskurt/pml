import pml
import matplotlib.pyplot as plt

def plot_poisson_reset():
    seq = pml.loadTxt('/tmp/prm.txt')
    states = seq[:, 0]
    obs = seq[:, 1]
    mean = pml.loadTxt('/tmp/mean.txt')
    cpp = pml.loadTxt('/tmp/cpp.txt')

    fig, ax= plt.subplots(2, sharex=True, figsize=(18, 8))

    ax[0].plot(range(len(states)), states, 'r-')
    ax[0].plot(range(len(obs)), obs, 'b-')
    ax[0].plot(range(len(mean)), mean, 'g-')
    ax[0].set_title("Sequence")
    ax[0].legend(['Hidden States', 'Observations', 'Mean'])

    ax[1].bar(range(len(cpp)), cpp)
    ax[1].set_ylim([0, 1])


    fig.show()


if __name__ == '__main__':
    plot_poisson_reset()
    input("enter a character to continue: ")