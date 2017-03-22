import numpy as np

from numpy import linalg


def lasso_regression(X, Y, rho1, opts):

    funcVal = []
    nTasks = len(X)


    dim = X[0].shape[1]
    print(dim)

    if opts.has_key('tol'):
         tol = opts['tol']
    else:
         tol = 0.0001
    #
    if opts.has_key('rho_L2'):
         rho_L2 = opts['rho_L2']
    else:
         rho_L2 = 0.0
    #
    if opts.has_key('maxIters'):
         maxIters = opts['maxIters']
    else:
         maxIters = 1000
    #
    if opts.has_key('W0'):
         W0 = opts['W0']
    else:
         W0 = np.random.randn(dim, nTasks)
    if opts.has_key('tFlag'):
         tFlag = opts['tFlag']
    else:
         tFlag = 1
    #
    Wz = W0
    Wz_old = W0

    t = 1
    t_old = 0
    #
    gamma = 0.1
    gamma_inc = 2
    bFlag = 0
    #
    iter = 0

    while iter < maxIters:
         print ("iter %d ..." % iter)
         alpha = (t_old - 1)/t
         Ws = (1 + alpha) * Wz - alpha * Wz_old
    #
    #
         gWs = gradVal_eval(X, Y, Ws, rho_L2)
         Fs = funcVal_eval(X, Y, Ws, rho_L2)
    #
         while True:
             [Wzp, l1_wzp] = soft_thresh(Ws - gWs/gamma, 2 * rho1 /gamma)
             Fzp = funcVal_eval(X, Y, Wzp, rho_L2)
    #
             delta_Wzp = Wzp - Ws
    #
             r_sum = linalg.norm(delta_Wzp) ** 2
    #
             if r_sum <= 0.000001:
                 bFlag = 1
                 break
             else:
                 gamma = gamma * gamma_inc
    #
         Wz_old = Wz
         Wz = Wzp
    #
         fval = Fzp + rho1 * l1_wzp
         funcVal.append(fval)
    #
         #if bFlag:
         #    print("\n terminated due to little change in weights")
             #break
         if tFlag == 0:
             if np.abs(funcVal[iter - 1] - funcVal[iter - 2]) < tol:
                 break
         if tFlag == 1:
             if iter >= maxIters:
                 break
    #
         iter = iter + 1
         t_old = tt = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
    #
    return Wzp, funcVal

def soft_thresh(v, beta):

    z = np.zeros(v.size)
    z = np.sign(v) * np.maximum(np.abs(v) - 1/2 * beta, 0)
    L1_val = np.sum(np.abs(z))
    return z, L1_val


def gradVal_eval(X, Y, W, rho_L2):
    grad_W = np.zeros_like(W)
    nTasks = len(X)

    for i in range(nTasks):
        grad_Wi = X[i].T.dot(X[i].dot(W[:, i]) - Y[i])
        grad_W[:, i] = grad_Wi
    grad_W = grad_W + 2 * rho_L2 * W
    return grad_W

def funcVal_eval(X, Y, W, rho_L2):

    funcVal = 0.0
    nTasks = len(X)

    for i in range(nTasks):
        funcVal = funcVal + 0.5 * linalg.norm(Y[i] - X[i].dot(W[:, i]))**2 + rho_L2 * linalg.norm(W[:, i], 2)**2
    return funcVal



