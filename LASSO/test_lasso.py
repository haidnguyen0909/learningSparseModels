import numpy as np
from LASSO import lasso_regression as header
import matplotlib.pyplot as pl

def test_lasso():
    dim = 10
    n = 5

    nTasks = 30

    rng = np.random.RandomState(42)
    W = rng.randn(dim, nTasks)

    X = []
    Y = []

    for i in range(nTasks):
        # task i-th
        W[:, i] = rng.rand(dim)
        W[:, i][W[:, i] < 0.9] = 0
        print(W[:,i])
        Xi = rng.randn(n, dim)
        X.append(Xi)
        Yi = np.dot(Xi, W[:, i])
        Y.append(Yi)
    opts = {'maxIters':1000, 'tol':0.00001, 'rho_L1':0.5, 'rho_L2':0.5}
    Wzp, funcVal = header.lasso_regression(X, Y, 0.01, opts)

    pl.figure()
    niters = len(funcVal)
    niters = range(niters)

    pl.plot(niters, funcVal)
    pl.xlabel("iter")
    pl.ylabel("loss")
    pl.show()


test_lasso()



