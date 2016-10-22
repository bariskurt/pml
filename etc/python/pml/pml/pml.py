import numpy as np


def loadTxt(filename):
    X = np.loadtxt(filename)
    dim = int(X[0])
    size = []
    for i in range(dim):
        size.append(int(X[i+1]))
    X = np.reshape(X[dim+1:], size, order='F')
    return X


def saveTxt(filename, X, format = '%.6f'):
    with open(filename, 'w') as f:
        dim = len(X.shape)
        f.write('%d\n' % dim)
        for i in range(dim):
            f.write('%d\n' % X.shape[i])
        temp = X.reshape(np.product(X.shape), order='F')
        np.savetxt(f, temp, fmt = format)

