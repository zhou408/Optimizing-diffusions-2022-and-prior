import math
import numpy as np
from scipy.stats import norm
import random
from SPSA import mc2, mcsd, mcgx
from GRAD import gradvec, gradvecgx
import pandas as pd
# with fixed step size


def opt4(step, rep, dim, h):
    random.seed(np.random.uniform(0, 10000, 1)[0])
    # mu = np.random.uniform(-5, 1, int(dim/h))
    mu = np.full(int(dim/h), 1)
    # mu = rep(10,5)
    old_int = 1e100
    new_int = mc2(mu, rep, h)
    count = 0
    results = [new_int]
    results = np.asarray(results)
    # grad = [1]*(dim/h)
    # while abs(old_int-new_int) >= 0.0005:
    while count <= 1000:
        # deltamu = np.random.uniform(0, 10000, dim/h)
        # scale = np.random.uniform(0, 2, 1)[0]
        # for now, 30 is the sample size for gradient estimation
        grad = gradvec(mu, h, rep)
        mu = mu-step*grad
        # setting threshold for mu
        for n in range(len(mu)):
            # print("length(mu)",length(mu),mu[[n]])
            if mu[n] >= 10:
                mu[n] = 10
            if mu[n] <= -10000:
                mu[n] = -10000
        # print('mu is', mu, 'grad is', grad, '\n')
        # print('mu', mu, '\n')
        old_int = new_int
        new_int = mc2(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        if count % 5 == 0:
            print("the", count, "iteration:", new_int, "\n")
        stdev = mcsd(mu, rep, h)
        print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "std:", stdev, "mu", mu, "\n")
    return [count, mu, new_int, stdev, results]


# with fixed step size and terminal cost
def opt5(step, rep, dim, h):
    random.seed(np.random.uniform(0, 10000, 1)[0])
    mu = np.full(int(dim / h), 1)
    # mu = np.random.uniform(-5, 0.5, int(dim/h))
    # mu = rep(10,5)
    old_int = 1e100
    new_int = mcgx(mu, rep, h)
    count = 0
    results = [new_int]
    results = np.asarray(results)
    grad = np.full(int((dim/h)), 5)
    # while abs(old_int-new_int) >= 0.0005:
    while count <= 1000:
    # while sum(abs(grad)) >= 0.005:
        # deltamu = np.random.uniform(0, 10000, dim/h)
        # scale = np.random.uniform(0, 2, 1)[0]
        # for now, 30 is the sample size for gradient estimation
        grad = gradvecgx(mu, h, rep, -3)
        mu = mu-step*grad
        # setting threshold for mu
        for n in range(len(mu)):
            # print("length(mu)",length(mu),mu[[n]])
            if mu[n] >= 10:
                mu[n] = 10
            if mu[n] <= -10000:
                mu[n] = -10000
        print('mu is', mu, 'grad is', grad, '\n')
        # print('mu', mu, '\n')
        old_int = new_int
        new_int = mcgx(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        if count % 5 == 0:
            print("the", count, "iteration:", new_int, "\n")
        stdev = mcsd(mu, rep, h)
        print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "std:", stdev, "mu", mu, "\n")
    return [count, mu, new_int, stdev, results]

res5 = opt4(1, 50, 5, 1)
data = pd.DataFrame(res5).reset_index(drop=True)
data.to_csv("opt4_30.csv", index=None)
data = pd.DataFrame(res5[4]).reset_index(drop=True)
data.columns = ['MCGD']
data.to_csv("opt4(value)_30.csv", index=None)

