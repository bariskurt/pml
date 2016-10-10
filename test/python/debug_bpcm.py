import pml
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_pg():
    states = pml.loadTxt('/tmp/states.txt')
    obs = pml.loadTxt('/tmp/obs.txt')

    mean = pml.loadTxt('/tmp/mean.txt')
    cpp = pml.loadTxt('/tmp/cpp.txt')

    states = states[0, :]
    obs = obs[0, :]
    mean = mean[0, :]
    T = len(states)

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    ax0.plot(range(T), states, 'r-')
    ax0.plot(range(T), obs, 'b-')
    ax0.plot(range(T), mean, 'g-')
    ax0.set_title("Sequence")
    ax0.legend(['Hidden States', 'Observations', 'Mean Smoothing'])
    ax0.set_xticks([])

    ax1 = plt.subplot(gs[1])
    ax1.bar(range(T), cpp)
    ax1.set_ylim([0, 1])
    ax1.set_xticks([])
    ax1.set_title("CPP Smoothing")

    plt.show()

if __name__ == '__main__':
    plot_pg()