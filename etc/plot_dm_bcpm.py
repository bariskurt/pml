import numpy as np
import matplotlib.pyplot as plt

def load_array(filename):
    X = np.loadtxt(filename)
    dim = int(X[0]);
    size = []
    for i in range(dim):
        size.append(int(X[i+1]));
    X = np.reshape(X[dim+1:], size, order='F')
    return X;

def plot_dm_bcpm(input_dir='/tmp/'):

    mean = load_array(input_dir + "mean.txt")
    cpp = load_array(input_dir + "cpp.txt")
    real_cps = load_array(input_dir + "real_cps.txt")
    data = load_array(input_dir + "obs.txt")

    K = data.shape[0]
    T = data.shape[1]

    f, axarr = plt.subplots(2, sharex=True,figsize=(18,10))

    y,x = np.mgrid[slice(0, 1, 1/K),slice(0, T+1, 1)]
    # Plot Data Histogram
    axarr[0].pcolormesh(x, y, data, cmap=plt.cm.Greys, vmin=0, vmax=np.max(np.max(data)))
    axarr[0].hold(True)

    # Plot true change points:
    plt11 = axarr[0].vlines(np.arange(0,T), 0, real_cps,  colors='b',linestyles='--', linewidth=2, label='real change points')

    axarr[0].set_yticks(np.arange(K+1)/K, minor=False)
    axarr[0].set_title("Change Point Probabilities on the Generated Data")
    axarr[0].set_xlabel("Time")
    axarr[0].set_ylabel("Change Point Probability")
    axarr[0].legend(handles=[plt11])

    # Plot Mean Histogram
    axarr[1].pcolormesh(x, y, mean, cmap=plt.cm.Greys, vmin=0, vmax=1)
    axarr[1].hold(True)

    # Plot change point probabilities
    plt21 = axarr[1].vlines(np.arange(0,T), 0, cpp,  colors='r', linestyles='-', linewidth=2, label='change point prob.')
    plt22 = axarr[1].vlines(np.arange(0,T), 0, real_cps,  colors='b', linestyles='--', linewidth=2, label='real change points')

    axarr[1].set_yticks(np.arange(K+1)/K, minor=False)
    axarr[1].set_title("Change Point Probabilities and Smoothed Density")
    axarr[1].set_xlabel("Time")
    axarr[1].set_ylabel("Change Point Probability")
    axarr[1].legend(handles=[plt21,plt22])

    f.show()

if __name__=='__main__':
    plot_dm_bcpm()
    input("enter a character to continue: ")