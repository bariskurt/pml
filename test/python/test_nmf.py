import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pml


def plot(x, t, v, estimation=False):
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[x.shape[1], t.shape[1], v.shape[1]])

    matrices = (x,t,v)
    if estimation:
        titles = ('x_est', 't_est', 'v_est')
    else:
        titles = ('x', 't', 'v')

    for i in range(len(matrices)):
        ax = plt.subplot(gs[i])
        ax.imshow(matrices[i], interpolation='nearest', cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[i], fontsize = 16)


def plot_nmf():
    x = pml.loadTxt("/tmp/x.txt")
    t = pml.loadTxt("/tmp/t.txt")
    v = pml.loadTxt("/tmp/v.txt")

    t_est = pml.loadTxt("/tmp/sol/t.txt")
    v_est = pml.loadTxt("/tmp/sol/v.txt")
    kl = pml.loadTxt("/tmp/sol/kl.txt")
    x_est = np.dot(t_est,v_est)

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[x.shape[1], t.shape[1], v.shape[1]])

    matrices = (x, t, v, x_est, t_est, v_est)
    titles = ('x', 't', 'v', 'x_est', 't_est', 'v_est')

    for i in range(len(matrices)):
        ax = plt.subplot(gs[i])
        ax.imshow(matrices[i], interpolation='nearest', cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[i], fontsize = 16)

    fig.tight_layout()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.plot(kl)
    ax.set_title('KL Divergence')
    ax.set_xlim([0, len(kl)])
    ax.set_xlabel('epoch')

    plt.show()

if __name__ == '__main__':
    plot_nmf()