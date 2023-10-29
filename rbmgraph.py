#!  /usr/bin/python
# #testing theorem
import time
import numpy as np
import math
import random
import sys
from scipy.stats import norm
from scipy.special import legendre
from scipy.integrate import quad
import matplotlib
from datetime import datetime
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def Brownian(tau, h, sigma):
    """
    This function produces a single Brownian path.
    :param tau: time horizon
    :param h: time disretization step size
    :return: Brownian path
    """

    time = int(tau / h)
    Bpath = np.zeros(time)
    for t in range(time):
        Bpath[t] = np.random.normal(0, math.sqrt(sigma **2 * h), 1)[0]
        # Bpath[t] = 0
    return Bpath


def RBM(driftvec, sample, h):
    """
    This function generates the RBM sample path corresponding to a drift vector and a Brownian path.
    :param driftvec: drift vector
    :param h: time disretization step size
    :return: drifted Brownian path and the corresponding RBM path
    """

    time = len(driftvec)
    T1 = np.zeros(time)
    T2 = np.zeros(time)
    M = np.zeros(time)
    X = np.zeros(time)
    B = np.zeros(time)
    BB = np.zeros(time)
    M[0] = 0#driftvec[0]
    BB[0] = driftvec[0]
    # B[0] = -driftvec[0]
    X[0] = max(driftvec[0], 0)
    driftvec = driftvec[1:]
    for t in range(time-1):
        T1[t] = -driftvec[t] * h + sample[t]
        B[t + 1] = B[t] + T1[t]
        T2[t] = T1[t] / 2 + math.sqrt(T1[t] ** 2 - 2 * h * math.log(np.random.uniform(0, 1, 1)[0])) / 2
        M[t + 1] = max(M[t], B[t] + T2[t])
        # X[t+1] = T1[t] + max(X[t],-T2[t])#M[t+1]
        # M[t + 1] = max(0, -X[t] + T2[t])
        # BB[t + 1] = BB[t] - T1[t]
        X[t + 1] = M[t + 1] - B[t + 1]
    plt.figure(1)
    plt.plot(M, 'bo', label="Reflection")
    plt.plot(-B, 'ro', label="BM")
    # print(X)
    plt.plot(X, 'go', label="RBM")
    plt.title('Sample path')
    plt.legend()
    # plt.show()
    return [-B, X]


def dir_fun(axis, x_value):
    return legendre(axis)(x_value)


def Bpathgrab(pa, power1, power2, rep):
    paa = []
    size = int(2**(power2 - power1))
    # print(power2, power1, size)
    # print(rep, len(pa))
    for kk in range(rep):
        # print('org_pathlen:', pa[kk].shape)
        paa.append(np.reshape(pa[kk], (-1, size)).sum(axis=-1))
    return paa


def mc2(driftvec, rep, h, samples):
    """
    This function generates the obj function value.
    :param driftvec: drift vector
    :param rep: Monte Carlo sample size
    :param h: time disretization step size
    :return: obj function value
    """
    totalsum = 0
    for i in range(rep):
        rbmpath = RBM(driftvec, samples[i], h)[1][0:-1] * h
        totalsum = totalsum + sum(rbmpath)
    expect = totalsum / rep
    return expect


def driver(se, sigma):
    random.seed(se)
    sigma = sigma
    N2_list = [1000]
    F_list = [-1, -100, - 1000, -10000]
    tau = 0.1
    hh = 0.01
    res_list = []
    N2s = []
    Fs = []
    sigmas = []
    taus = []
    hs = []
    for i in range(len(N2_list)):
        for j in range(len(F_list)):
            samples = []
            for k in range(N2_list[i]):
                sample = Brownian(tau, hh, sigma)
                samples.append(sample)
            res = mc2(np.repeat(F_list[j], int(tau/hh)), N2_list[i], hh, samples)
            now1 = datetime.now()
            current_time = now1.strftime("%H:%M:%S")
            print('time is', current_time)
            print('N2 is ', N2_list[i], 'F is ', F_list[j], 'res is ', res)
            res_list.append(res)
            N2s.append(N2_list[i])
            Fs.append(F_list[j])
            taus.append(tau)
            hs.append(hh)
            sigmas.append(sigma)
    d = {'N2': N2s, 'F': Fs, 'sigma': sigmas, 'tau': taus, 'h': hs, 'J_tilda': res_list}
    s = pd.DataFrame(data=d)
    data = s.reset_index(drop=True)
    data.to_csv('rbmgraph_sigma_' + str(sigma) + '_seed_' + str(se))


driver(1000, 10000)
driver(1000, 1000)
driver(1000, 100)

# samples = []
# for i in range(1000):
#     sample = Brownian(0.1, 0.001, 10000)
#     samples.append(sample)
# res = mc2(np.repeat(-1000, 100), 1000, 0.001, samples)
# print(res)

# RBM(np.repeat(-10, 100), sample, 0.001)

# F = -1t
# sigma = 0.1: 0.003507925329388863
# sigma = 1:   0.014990981940644082
# sigma = 10:  0.15753973390436
# sigma = 100: 1.510884331509896

# F = -10t
# sigma = 0.1: 0.0017838038824083433
# sigma = 1:   0.004869357462787201
# sigma = 10:  0.13738995998151998
# sigma = 100: 1.3698152465318734

# F = -100t
# sigma = 0.1: 0.00045201763436074967
# sigma = 1:   0.0004864184486249391
# sigma = 10:  0.034090787775412974
# sigma = 100: 1.3548465740471323

# F = -1000t
# sigma = 0.1: 4.926277083300452e-05
# sigma = 1:   4.935431567214336e-05
# sigma = 10:  6.505899783984474e-05
# sigma = 100: 0.312900788585785


### N = 5
#sigma = 100
# F = -1000t: 0.34332685628961057
# F = -100t: 1.387523382684037
# F = -10t: 1.516310694497254
# F = -1t: 1.7918166978123409

#sigma = 1000
# F = -1000t: 6.915631380284414
# F = -100t: 17.514287870499544
# F = -10t: 18.652129919706294
# F = -1t: 11.066315414207514

#sigma = 10000
# F = -1000t: 83.19086157437712
# F = -100t: 172.13749714733655
# F = -10t: 160.73076662195842
# F = -1t: 142.68490133540905

### N = 100
#sigma = 10000
# F = -1000t: 132.87673561087138
# F = -100t: 131.70188389823778
# F = -10t: 157.78511915463628
# F = -1t: 155.26644765136666


### N = 1000
#sigma = 10000
# F = -1000t: 132.87673561087138
# F = -100t: 131.70188389823778
# F = -10t: 157.78511915463628
# F = -1t: 155.26644765136666