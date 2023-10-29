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
    # ress = [82.12952705950624, 64.21857808976907, 77.15314301787224, 45.23445303038091, 87.83011686148284, 73.2483761206428, 58.80549317433073, 58.771870088875836, 96.66909776841348, 93.12438652189907, 80.36018784467122, 79.25314404070247, 79.4416541342629, 80.9868938322988, 90.20606788275192, 82.8924821800646]
    N2_list = [5, 10, 100, 200]
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
    data.to_csv('rbmgraphless_sigma_' + str(sigma) + '_seed_' + str(se))


# driver(1000, 10000)
# driver(1000, 100)
driver(1000, 2500)
# driver(1000, 1)
# driver(1000, 100)