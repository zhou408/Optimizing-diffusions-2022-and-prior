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
# plt.rcParams['text.usetex'] = True

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
    # plt.figure(1)
    # plt.plot(M, 'bo', label="Reflection")
    # plt.plot(-B, 'ro', label="BM")
    # # print(X)
    # plt.plot(X, 'go', label="RBM")
    # plt.title('Sample path')
    # plt.legend()
    # plt.show()
    return [-B, X]

sigma = 10
F_list = [-10, -5, 5, 10]
tau = 1
hh = 0.01
sample = Brownian(tau, hh, sigma)
# print(sample)
x = np.arange(hh, tau+hh, hh)

driftvec = np.repeat(F_list[0], int(tau/hh))
rbmpath = RBM(driftvec, sample, hh)[1]
plt.plot(x, rbmpath, label=r'$X^{F_1}\sim\pi^{\Gamma}_{F_1}$')

driftvec = np.repeat(F_list[1], int(tau/hh))
rbmpath = RBM(driftvec, sample, hh)[1]
plt.plot(x, rbmpath, label=r'$X^{F_2}\sim\pi^{\Gamma}_{F_2}$')

driftvec = np.repeat(F_list[2], int(tau/hh))
rbmpath = RBM(driftvec, sample, hh)[1]
plt.plot(x, rbmpath, label=r'$X^{F_3}\sim\pi^{\Gamma}_{F_3}$')

plt.plot(x, np.cumsum(sample), label=r'$Z\sim\pi_0}$')
plt.title('Simulation of ' + r'X_F' + ' Based on Z Path', fontsize = 20)
plt.legend(fontsize=12)
plt.show()


# x = np.linspace(-1, 1, 100)
# y = 2*x
# y1 = x + 0.3
# y2 = 1.5*x - 0.1
# y3 = -0.5*x + 0.4
# y4 = -3*x - 0.2
# plt.plot(x, y)
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
# plt.plot(x, y4)
# plt.title('Linear Functions')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.ylim(-1,1)
# plt.show()
#
# x = np.linspace(-1, 1, 100)
# def correct_legs(x, c):
#    if c == 0: return(np.ones(len(x)))
#    if c == 1: return(x)
#    if c == 2: return(0.5 * (3 * x ** 2 - 1))
#    if c == 3: return(0.5 * (5 * x ** 3 - 3 * x))
#    if c == 4: return((1. / 8.) * (35 * x ** 4 - 30 * x ** 2 + 3))
#    if c > 4 or c < 0 or type(c)!=int: return(np.nan)
# for order in range(5):
#     x, y = np.polynomial.legendre.Legendre.basis(order, [-1, 1]).linspace(100)
#     plt.plot(x, y, label=order)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot()
# plt.title('Legendre Polynomials')
# plt.show()
