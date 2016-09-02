import numpy as np
import matplotlib.pyplot as plt


def load_array(filename):
    X = np.loadtxt(filename)
    dim = int(X[0])
    size = []
    for i in range(dim):
        size.append(int(X[i+1]))
    X = np.reshape(X[dim+1:], size, order='F')
    return X


def plot_dhmm():
    seq = load_array("/tmp/seq.txt")
    states = seq[:, 0]
    obs = seq[:, 1]
    alpha = load_array("/tmp/alpha.txt")
    beta = load_array("/tmp/beta.txt")
    gamma = load_array("/tmp/gamma.txt")

    fig, ax= plt.subplots(3, sharex=True, figsize=(18,10))
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

    fig.show()


if __name__=='__main__':
    plot_dhmm()
    input("enter a character to continue: ")