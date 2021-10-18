import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    k_mat = np.power(np.dot(X, Y.T) + c, p)
    return k_mat
    raise NotImplementedError



def rbf_kernel(X, Y, gamma):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - Y[j]) ** 2)
    return K
    raise NotImplementedError
