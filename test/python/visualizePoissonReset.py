import pml
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_poisson_reset():
    states = pml.loadTxt('/tmp/states.txt')
    obs = pml.loadTxt('/tmp/obs.txt')
    mean1 = pml.loadTxt('/tmp/mean_filtering.txt')
    cpp1 = pml.loadTxt('/tmp/cpp_filtering.txt')
    mean2 = pml.loadTxt('/tmp/mean_smoothing.txt')
    cpp2 = pml.loadTxt('/tmp/cpp_smoothing.txt')
    mean3 = pml.loadTxt('/tmp/mean_fixed_lag.txt')
    cpp3 = pml.loadTxt('/tmp/cpp_fixed_lag.txt')

    states = states[0, :]
    obs = obs[0, :]
    mean1 = mean1[0, :]
    mean2 = mean2[0, :]
    mean3 = mean3[0, :]
    T = len(states)

    #fig, ax = plt.subplots(3, sharex=True, figsize=(18, 8))

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])

    ax = plt.subplot(gs[0])
    ax.plot(range(T), states, 'r-')
    ax.plot(range(T), obs, 'b-')
    ax.plot(range(T), mean1, 'g-')
    ax.plot(range(T), mean2, 'c-')
    ax.plot(range(T), mean3, 'm-')
    ax.set_title("Sequence")
    ax.legend(['Hidden States', 'Observations', 'Mean Filtering',
                  'Mean Smoothing', 'Mean Fixed-Lag'])

    ax = plt.subplot(gs[1])
    ax.bar(range(T), cpp1)
    ax.set_ylim([0, 1])
    ax.set_title("CPP Filtering")

    ax = plt.subplot(gs[2])
    ax.bar(range(T), cpp2)
    ax.set_ylim([0, 1])
    ax.set_title("CPP Smoothing")

    ax = plt.subplot(gs[3])
    ax.bar(range(T), cpp3)
    ax.set_ylim([0, 1])
    ax.set_title("CPP Fixed Lag")


    fig.show()


if __name__ == '__main__':
    plot_poisson_reset()
    input("enter a character to continue: ")