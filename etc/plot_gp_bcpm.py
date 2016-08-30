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

def plot_gp_bcpm(input_dir='/tmp/'):

    mean = load_array(input_dir + "mean.txt")
    cpp = load_array(input_dir + "cpp.txt")
    real_cps = load_array(input_dir + "real_cps.txt")
    data = load_array(input_dir + "obs.txt")

    # GP data is 1-dimensional:
    data = data[0,:]
    mean = mean[0,:]
    T = len(data)

    f, axarr = plt.subplots(2, sharex=True,figsize=(18,10))

    # Plot 1:  True and Filtered Data
    axarr[0].plot(range(T), data)
    axarr[0].hold(True)
    axarr[0].plot(range(T), mean)
    axarr[0].set_xlabel("Time")
    axarr[0].legend(['Observations', 'Mean'])
    axarr[0].set_title("Observations and Smoothed Density")

    # Plot 2: True and estimated change points
    plt21 = axarr[1].vlines(np.arange(0,T), 0, cpp,  colors='r', linestyles='-', linewidth=2, label='change point prob.')
    plt22 = axarr[1].vlines(np.arange(0,T), 0, real_cps,  colors='b', linestyles='--', linewidth=2, label='real change points')
    axarr[1].set_title("Change Point Probabilities")
    axarr[1].set_xlabel("Time")
    axarr[1].set_ylabel("Change Point Probability")
    axarr[1].legend(handles=[plt21, plt22])

    f.show()

if __name__=='__main__':
    plot_gp_bcpm()
    input("enter a character to continue: ")