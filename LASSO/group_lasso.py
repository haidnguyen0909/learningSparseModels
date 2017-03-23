
import numpy as np
from numpy import linalg


def group_lasso(X, Y, opts):

    nTasks = len(X)
    dim = X[0].shape[1]
    n = X[0].shape[0]

    if opts.has_key('tFlag'):
        tFlag = opts['tFlag']
    else:
        tFlag = 0

    if opts.has_key('tol'):
        tol = opts['tol']
    else:
        tol = 0.00001

    if opts.has_key('rho_L1'):
        rho_L1 = opts['rho_L1']
    else:
        rho_L1 = 0.5

    if opts.has_key('rho_L2'):
        rho_L2 = opts['rho_L2']
    else:
        rho_L2 = 0.5

    if opts.has_key('W0'):
        W0 = opts['W0']
    else:
        W0 = np.random.randn(dim, nTasks)

    if opts.has_key('maxIters'):
        maxIters = opts['maxIters']
    else:
        maxIters = 1000

    Wz = W0
    Wz_old = W0

    t = 1
    t_old = 0

    iter = 0
    gamma = 1
    gamma_inc = 2
    funcs = []

    while iter < maxIters:
        alpha = (t_old - 1)/t
        Ws = (1 + alpha) *Wz - alpha * Wz_old
        grad_Ws = calc_grad(X, Y, Ws, rho_L2)
        F_Ws = calc_smooth_func(X, Y, Ws, rho_L2)

        while True:
            Wzp = prox(Ws - grad_Ws / gamma, rho_L1)
            F_Wzp = calc_smooth_func(X, Y, Wzp, rho_L2)

            delta_W = Wzp - Ws
            r_sum = linalg.norm(delta_W)**2

            if r_sum < 0.00000001:
                break
            else:
                gamma = gamma * gamma_inc
        Wz_old = Wz
        Wz = Wzp
        funcs.append(F_Wzp + calc_nonsmooth(X, Y, Wz, rho_L1))

        if tFlag == 0:
            if np.abs(funcs[iter] - funcs[iter - 1]) < tol and iter > 1:
                break

        iter = iter + 1
        t_old = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t * t))


    return Wz, funcs

def prox(Z, t):
    X = np.zeros_like(Z)
    nTasks = Z.shape[1]
    dim = Z.shape[0]

    for i in range(dim):
        nm = linalg.norm(Z[i, :], 2)
        if nm == 0:
            x = np.zeros_like(Z[i,:])
        else:
            x = np.maximum(nm - t, 0) / nm * Z[i,:]
        X[i, :] = x
    print(X)
    return X


def calc_grad(X, Y, W, rho_L2):
    grad_W = np.zeros_like(W)
    nTasks = len(X)

    for i in range(nTasks):
        grad_W[:, i] = X[i].T.dot(X[i].dot(W[:, i]) - Y[i])

    grad_W = grad_W + 2 * rho_L2 * W
    return grad_W

def calc_smooth_func(X, Y, W, rho_L2):
    nTasks = len(X)
    val = 0

    for i in range(nTasks):
        val = val + 0.5 * linalg.norm(Y[i] - X[i].dot(W[:, i]))**2
    val = val + rho_L2 * linalg.norm(W)**2
    return val

def calc_nonsmooth(X, Y, W, rho_L1):
    nTasks = len(X)
    value = 0
    dim = W.shape[0]
    for i in range(dim):
        value = value + rho_L1 * linalg.norm(W[i,:], 2)
    return value


