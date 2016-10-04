import pml
import matplotlib.pyplot as plt


def plot_dhmm():

    states = pml.loadTxt("/tmp/states.txt")
    obs = pml.loadTxt("/tmp/obs.txt")

    alpha = pml.loadTxt("/tmp/alpha.txt")
    beta = pml.loadTxt("/tmp/beta.txt")
    gamma = pml.loadTxt("/tmp/gamma.txt")

    fig, ax = plt.subplots(3, sharex=True, figsize=(18,10))

    ax[0].pcolormesh(alpha, cmap=plt.cm.Greys, vmin=0, vmax=1)
    ax[0].plot(range(len(states)), states + 0.5, 'r*')
    ax[0].plot(range(len(obs)), obs + 0.4, 'b*')
    ax[0].set_title("Forward Recursion")

    ax[1].pcolormesh(beta, cmap=plt.cm.Greys, vmin=0, vmax=1)
    ax[1].plot(range(len(states)), states + 0.5, 'r*')
    ax[1].plot(range(len(obs)), obs + 0.4, 'b*')
    ax[1].set_title("Backward Recursion")

    ax[2].pcolormesh(gamma, cmap=plt.cm.Greys, vmin=0, vmax=1)
    ax[2].plot(range(len(states)), states + 0.5, 'r*')
    ax[2].plot(range(len(obs)), obs + 0.4, 'b*')
    ax[2].set_title("Smoothed")
    ax[2].set_xlabel("Time")

    plt.show()

if __name__ == '__main__':
    plot_dhmm()