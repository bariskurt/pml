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
    with open(filename, 'wb') as f:
        header = np.asarray(len(X.shape))
        header = np.append(header, X.shape)
        np.savetxt(f, header, fmt='%d')
    with open(filename, 'ab') as f:
        temp = X.reshape(np.product(X.shape), order='F')
        np.savetxt(f, temp, fmt = format)

