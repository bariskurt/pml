import pml
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_pg_em():
    states = pml.loadTxt('/tmp/states.txt')
    obs = pml.loadTxt('/tmp/obs.txt')

    mean = pml.loadTxt('/tmp/mean.txt')
    cpp = pml.loadTxt('/tmp/cpp.txt')

    mean2 = pml.loadTxt('/tmp/mean2.txt')
    cpp2 = pml.loadTxt('/tmp/cpp2.txt')

    mean3 = pml.loadTxt('/tmp/mean3.txt')
    cpp3 = pml.loadTxt('/tmp/cpp3.txt')


    states = states[0, :]
    obs = obs[0, :]
    mean = mean[0, :]
    mean2 = mean2[0, :]
    mean3 = mean3[0, :]
    T = len(states)

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[5, 1, 1, 1])

    ax = plt.subplot(gs[0])
    ax.plot(range(T), states, 'r-')
    ax.plot(range(T), obs, 'b-')
    ax.plot(range(T), mean, 'g-')
    ax.plot(range(T), mean2, 'm-')
    ax.plot(range(T), mean3, 'c-')
    ax.set_title("Sequence")
    ax.legend(['Hidden States', 'Observations', 'Mean Filtering',
               'Mean Smoothing(EM)', 'Mean Smootring dummy'])

    ax = plt.subplot(gs[1])
    ax.bar(range(T), cpp)
    ax.set_ylim([0, 1])
    ax.set_title("CPP Filtering")

    ax = plt.subplot(gs[2])
    ax.bar(range(T), cpp2)
    ax.set_ylim([0, 1])
    ax.set_title("CPP Smoothine after EM")

    ax = plt.subplot(gs[3])
    ax.bar(range(T), cpp3)
    ax.set_ylim([0, 1])
    ax.set_title("CPP Smoothine dummy")

    plt.show()


if __name__ == '__main__':
    plot_pg_em()