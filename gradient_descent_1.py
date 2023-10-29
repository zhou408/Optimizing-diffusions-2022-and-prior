from scipy.stats import norm
import numpy as np
import math
import matplotlib.pyplot as plt


def rbm2(driftvec, h):
    """

    :param driftvec:
    :param h:
    :return:
    """
    time = len(driftvec)
    T1 = np.zeros(time)
    T2 = np.zeros(time)
    M = np.zeros(time+1)
    X = np.zeros(time+1)
    B = np.zeros(time+1)
    BB = np.zeros(time + 1)
    for t in range(time):
        # scale down the drifts and dc according to h
        # randomnormal = np.random.normal(0, 1, 1)[0]*math.sqrt(h)
        # T1[t] = (-driftvec[t])*h + randomnormal
        T1[t] = np.random.normal((-driftvec[t])*h, math.sqrt(h), 1)[0]
        B[t + 1] = B[t] + T1[t]
        T2[t] = T1[t] / 2 + math.sqrt(T1[t] ** 2 - 2 * h * math.log(np.random.uniform(0, 1, 1)[0])) / 2
        M[t+1] = max(M[t], B[t]+T2[t])
        BB[t + 1] = BB[t] - T1[t]
        X[t+1] = M[t+1]-B[t+1]
    return [BB, X]

# trial = rbm2(np.zeros(1000), 0.1)
# print(trial[0])
# print(trial[1])
#
# plt.plot(trial[0], label="BM")
# plt.plot(trial[1], label="RBM")print("hello")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.xlabel('t')
# plt.ylabel('value')
# plt.show


def mc2(driftvec, rep, h):
    """

    :param driftvec:
    :param rep:
    :param h:
    :return:
    """
    totalsum = 0
    for i in range(rep):
        rbmpath = rbm2(driftvec, h)[1]*h
        totalsum = totalsum+sum(rbmpath)
    expect = totalsum/rep
    return expect


def grad_descent(step, rep, tau, h):
    """

    :param step: step size
    :param rep: Monte Carlo sample size
    :param tau: time horizon
    :param h: time discretization step size
    :return: [count, mu, new_int, results]:[iteration, optimal drift, optimal value, stdev in monte car, array of values
    in all iterations]
    """
    random.seed(np.random.uniform(0, 10000, 1)[0])
    # mu = np.random.uniform(0, 10, int(tau/h))
    mu = np.full(int(tau / h), 1)
    # print(mu, "\n")
    old_int = 1e1000000
    new_int = mc2(mu, rep, h)
    count = 0
    results = [0]
    results = np.asarray(results)
    while count <= 1000:
        deltamu = np.random.uniform(0, 1, int(tau/h))
        scale = np.random.uniform(0, 2, 1)[0]
        # grad = (mc2(mu+scale*deltamu, rep, h)-mc2(mu-scale*deltamu, rep, h))/(2*scale*deltamu)
        mu = mu-step*grad
        old_int = new_int
        new_int = mc2(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        if count % 10 == 0:
            print("the", count, "iteration:", new_int, "mu", mu, "\n")
        print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "std:", stdev, "mu", mu, "\n")
    return [count, mu, new_int, stdev, results]


