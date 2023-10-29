#!  /usr/bin/python
#testing theorem
import time
import numpy as np
import math
import random
import sys
from scipy.stats import norm
from numpy import linalg as LA
from numpy.polynomial.legendre import Legendre
from datetime import datetime
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def Brownian(tau, h, sigma=1):
    """
    This function produces a single Brownian path.
    :param tau: time horizon
    :param h: time disretization step size
    :return: Brownian path
    """

    time = int(tau / h)
    Bpath = np.zeros(time)
    for t in range(time):
        Bpath[t] = np.random.normal(0, math.sqrt(sigma ** 2 * h), 1)[0]
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
    # plt.plot(X, 'go', label="RBM")
    # plt.title('Sample path')
    # plt.legend()
    # plt.show()
    return [-B, X]


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


def mc3(driftvec, rep, h, samples, lambda1, driftcoe, tau, M):
    """
    This function generates the exponential obj function value
    """
    totalsum = 0
    for i in range(rep):
        rbmpath = np.exp(-1 * RBM(driftvec, samples[i], h)[1][0:-1]) * h
        totalsum = totalsum + sum(rbmpath)
    integrand = Legendre(driftcoe)**2
    L2norm = quad(integrand, 0, tau)[0]
    expect = totalsum / rep + lambda1 * max(L2norm - M ** 2, 0)
    return expect


def gprime3(x):
    comp1 = -1 * np.exp(-x)
    return comp1


def penaltyderiv3(lambda1, driftcoe, tau, M):
    integrand = Legendre(driftcoe) ** 2
    L2norm = quad(integrand, 0, tau)[0]
    comp2 = np.zeros(driftcoe.shape[0])
    if L2norm > M ** 2:
        print('yes')
        for axis in range(driftcoe.shape[0]):
            integrand2 = 2 * Legendre(driftcoe).deriv() * Legendre.basis(axis).deriv()
            comp2[axis] = lambda1 * quad(integrand2, 0, tau)[0]

    else:
        for axis in range(driftcoe.shape[0]):
            comp2[axis] = 0
    return comp2


def mc4(driftvec, rep, h, samples, t, b):
    """
    This function generates the exponential obj function value
    """
    totalsum = 0
    for i in range(rep):
        rbmpath = t * (RBM(driftvec, samples[i], h)[1][0:-1]) * h + b * (RBM(driftvec, samples[i], h)[1][0:-1]) ** 2 * h
        totalsum = totalsum + sum(rbmpath)
    expect = totalsum / rep
    return expect


def gprime4(x, t, b):
    comp1 = 2 * b * x + t
    return comp1


def penaltyderiv4():
    return 0


def dir_fun(axis, x_value):
    return Legendre.basis(axis)(x_value)


def Bpathgrab(pa, power1, power2, rep):
    paa = []
    size = int(2**(power2 - power1))
    # print(power2, power1, size)
    # print(rep, len(pa))
    for kk in range(rep):
        # print('org_pathlen:', pa[kk].shape)
        paa.append(np.reshape(pa[kk], (-1, size)).sum(axis=-1))
    return paa


def backtracking_line_search3(Bpaths_true, true_int, mu_in, gradient, tau, h, lambda1, M, beta, sim_scale=1):
    cost_val = true_int
    step = 1
    mu = mu_in - step * gradient
    mu_presim = np.zeros(int(tau / h) + 1)
    index_array = np.arange(0, len(mu_presim) * h, h)[0: len(mu_presim)]
    for axis in range(mu_in.shape[0]):
        mu_presim = mu_presim + Legendre.basis(axis)(index_array) * mu[axis]
    mumu = (mu_presim[1:] - mu_presim[:-1]) / h
    mumu = np.insert(mumu, 0, mu_presim[0] / h)
    mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
    LHS = mc3(mu_sim, 400, 0.5 ** 10 / sim_scale, Bpaths_true, lambda1, mu, tau, M)
    while LHS > cost_val - step/2*LA.norm(gradient)**2:
        # print(LHS, cost_val, LA.norm(gradient)**2)
        step = beta * step
        mu = mu_in - step * gradient
        mu_presim = np.zeros(int(tau / h) + 1)
        index_array = np.arange(0, len(mu_presim) * h, h)[0: len(mu_presim)]
        for axis in range(mu_in.shape[0]):
            mu_presim = mu_presim + Legendre.basis(axis)(index_array) * mu[axis]
        mumu = (mu_presim[1:] - mu_presim[:-1]) / h
        mumu = np.insert(mumu, 0, mu_presim[0] / h)
        mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
        LHS = mc3(mu_sim, 400, 0.5 ** 10 / sim_scale, Bpaths_true, lambda1, mu, tau, M)
        # print('check step:', step)
    return step


def backtracking_line_search4(Bpaths_true, true_int, mu_in, gradient, tau, h, t, b, beta, sim_scale=1):
    small_step = True
    cost_val = true_int
    step = 1
    mu = mu_in - step * gradient
    mu_presim = np.zeros(int(tau / h) + 1)
    index_array = np.arange(0, len(mu_presim) * h, h)[0: len(mu_presim)]
    for axis in range(mu_in.shape[0]):
        mu_presim = mu_presim + Legendre.basis(axis)(index_array) * mu[axis]
    mumu = (mu_presim[1:] - mu_presim[:-1]) / h
    mumu = np.insert(mumu, 0, mu_presim[0] / h)
    mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
    LHS = mc4(mu_sim, 400, 0.5 ** 10 / sim_scale, Bpaths_true, t, b)
    while (LHS > cost_val - step/2*LA.norm(gradient)**2) and small_step:
        # print(LHS, cost_val, LA.norm(gradient)**2)
        step = beta * step
        if step < 1e-10:
            small_step = False
            step = 0
        mu = mu_in - step * gradient
        mu_presim = np.zeros(int(tau / h) + 1)
        index_array = np.arange(0, len(mu_presim) * h, h)[0: len(mu_presim)]
        for axis in range(mu_in.shape[0]):
            mu_presim = mu_presim + Legendre.basis(axis)(index_array) * mu[axis]
        mumu = (mu_presim[1:] - mu_presim[:-1]) / h
        mumu = np.insert(mumu, 0, mu_presim[0] / h)
        mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
        LHS = mc4(mu_sim, 400, 0.5 ** 10 / sim_scale, Bpaths_true, t, b)
        # print('yes', t)
    return step


def gd3(tau, h, dim, step, replication, iteration, sim_scale, Bpa, Bpa_power, runTrue, lambda1, M, beta=0.5):
    """
    This function carries out thr gradient descent algorithm on the obj function.
    :param tau: time horizon
    :param h: time disretization step size
    :param step: step size in gd
    :param rep: Monte Carlo sample size
    :param iteration: max iteration numbers
    :return:
    """
    ite = 0
    dim = int(dim)
    mu = np.array([-10, -10, -10])
    h_power = round(math.log(h, 0.5))
    h = 0.5 ** h_power
    # if replication >= 100:
    #     rep = 100
    # else:
    #     rep = replication
    rep = replication
    # print('h power is:', h_power)
    # mu = np.array([-100, -49.696, -49.75448, 12.31851, 0])
    # mu = np.array([1,1,1,1,1,10,10,10,10,10])
    # mu = np.random.uniform(0, 100, 10)
    tau = round(tau/h)*h
    mu_presim = np.zeros(int(tau / h)+1)
    index_array = np.arange(0, len(mu_presim)*h, h)[0: len(mu_presim)]
    # index_array = np.arange(0, tau+h, h)
    # plt.figure(0)
    for axis in range(dim):
        mu_presim = mu_presim + Legendre.basis(axis)(index_array)*mu[axis]
    # plt.show()
    mumu = (mu_presim[1:] - mu_presim[:-1])/h
    mumu = np.insert(mumu, 0, mu_presim[0])
    mu_sim = np.repeat(mumu, sim_scale, axis=0)/sim_scale
    mu_arr = np.array([mu])
    presim_arr = np.array([mu_presim])
    # generate Brownian paths
    # print(Bpa)
    Bpaths = Bpathgrab(Bpa, h_power, Bpa_power, rep)
    # mc3(driftvec, rep, h, samples, lambda1, driftcoe, tau, M)
    new_int = mc3(mu_sim, rep, h/sim_scale, Bpaths, lambda1, mu, tau, M)
    results = np.array(new_int)
    if runTrue:
        Bpaths_true = Bpathgrab(Bpa, 10, Bpa_power, 400)
        mumu = (mu_presim[1:] - mu_presim[:-1]) / (0.5 ** 10)
        mumu = np.insert(mumu, 0, mu_presim[0])
        mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
        true_int = mc3(mu_sim, 400, 0.5 ** 10 / sim_scale, Bpaths_true,lambda1, mu, tau, M)
    else:
        true_int = float('nan')
    if iteration >= 50:
        #storestep = int(iteration / 50)
        storestep = 1
        storelist = np.arange(storestep, iteration + storestep, storestep).tolist()
    else:
        storelist = np.arange(1, iteration + 1, 1).tolist()
    trueres_list = [true_int]
    res_list = [new_int]
    n_list = [dim]
    N_list = [rep]
    h_list = [0.5**h_power]
    h_orglist = [h]
    work_list = [0]
    # i is the index of gd iterations
    for i in range(iteration):
        # gradsum = np.empty((0, int(tau / h)))
        gradsum = np.zeros(dim)
        # k is the index of omega
        # RBMpath = RBM(mu_sim, sample, h)
        for k in range(rep):
            RBMpath = RBM(mu_sim, Bpaths[k], h / sim_scale)
            RBM_arr = np.array(RBM(mu_sim, Bpaths[k], h / sim_scale)[1])
            # plt.figure(4)
            # plt.plot(RBMpath[1])
            # print(RBMpath[1])
            gradrep = np.zeros(dim)
            # axis is the index of gradient entries
            Phi_record = []
            infimum = 0
            Phi = [0]
            for time in range(int(tau / (h / sim_scale))):
                if RBMpath[0][time + 1] < infimum:
                    # print(RBMpath[0][time+1])
                    Phi = [time + 1]
                    Phi_record.append(Phi)
                    infimum = RBMpath[0][time + 1]
                elif RBMpath[0][time + 1] == infimum:
                    # print(RBMpath[0][time+1])
                    Phi.append(Phi[-1])
                    Phi_record.append(Phi)
                else:
                    Phi_record.append(Phi)
            # Phi_arr = np.unique(Phi_record)
            for axis in range(dim):
                DuJ2 = 0
                # integrand = Legendre.basis(axis)
                # DuJ1 = quad(integrand, 0, tau)[0]
                time_arr = np.arange(h, RBM_arr.shape[0] * h + h, h)
                # print(RBMpath, gprime3(RBMpath))
                integrand = Legendre.basis(axis)(time_arr) * gprime3(RBM_arr)
                DuJ1 = sum(integrand) * h
                # test = DuJ1
                for time in range(int(tau / (h / sim_scale))):
                    Phi = Phi_record[time]
                    addon = dir_fun(axis, Phi[0] * h / sim_scale)
                    for item in Phi:
                        if dir_fun(axis, item * h / sim_scale) < addon:
                            addon = dir_fun(axis, item * h / sim_scale)
                    DuJ2 = DuJ2 - addon * gprime3(RBM_arr[time]) * h / sim_scale
                if axis == 0:
                    gradrep[axis] = DuJ1  # + DuJ2
                else:
                    gradrep[axis] = DuJ1 + DuJ2
            gradsum = gradsum + gradrep
        # plt.show()
        directional = gradsum/rep + penaltyderiv3(lambda1, mu, tau, M)
        weights = np.zeros(dim)
        for axis in range(dim):
            weights[axis] = directional[axis]
        step = backtracking_line_search3(Bpaths_true, true_int, mu, directional, tau, h, lambda1, M, beta, sim_scale=1)
        mu = mu - step * directional
        mu_presim = np.zeros(int(tau/h) + 1)
        mu_arr = np.append(mu_arr, [mu], axis=0)
        presim_arr = np.append(presim_arr, [mu_presim], axis=0)
        index_array = np.arange(0, len(mu_presim)*h, h)[0: len(mu_presim)]
        # print(mu_presim, index_array)
        # index_array = np.arange(0, tau + h, h)
        for axis in range(dim):
            mu_presim = mu_presim + Legendre.basis(axis)(index_array) * mu[axis]
        mumu = (mu_presim[1:] - mu_presim[:-1])/h
        mumu = np.insert(mumu, 0, mu_presim[0]/h)
        mu_sim = np.repeat(mumu, sim_scale, axis=0)/sim_scale
        new_int = mc3(mu_sim, rep, h/sim_scale, Bpaths, lambda1, mu, tau, M)
        if runTrue:
            mumu = (mu_presim[1:] - mu_presim[:-1]) / (0.5 ** 10)
            mumu = np.insert(mumu, 0, mu_presim[0])
            mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
            true_int = mc3(mu_sim, 400, 0.5**10 / sim_scale, Bpaths_true, lambda1, mu, tau, M)
        else:
            true_int = float('nan')
        ite = ite + 1
        results = np.append(results, new_int)
        norm = math.sqrt(np.dot(weights, weights))
        # print("iteration:", ite, norm, weights, mu_presim, mu_sim, new_int)
        if ite % 1 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('time is', current_time)
            print("iteration:", ite, 'step:', step, 'gradient:', directional, 'drift:', mu, 'val', true_int)
        if ite in storelist:
            trueres_list.append(true_int)
            res_list.append(new_int)
            n_list.append(dim)
            N_list.append(replication)
            h_orglist.append(h)
            h_list.append(0.5**h_power)
            work_list.append(rep*ite*dim/h)
            #work_list.append(ite*dim/h)
    d = {'n': n_list, 'h': h_list, 'org_h': h_orglist, 'W': work_list, 'value': res_list, 'trueval': trueres_list}
    s = pd.DataFrame(data=d)
    print("iteration:", ite, "value:", new_int, "drift:", mu_presim, "\n")
    return [ite, mu_sim, new_int, results, dim, h, mu_arr, presim_arr, tau, s]


def gd4(tau, h, dim, step, replication, iteration, sim_scale, Bpa, Bpa_power, runTrue, t, b, beta = 0.5):
    """
    This function carries out thr gradient descent algorithm on the obj function.
    :param tau: time horizon
    :param h: time disretization step size
    :param step: step size in gd
    :param rep: Monte Carlo sample size
    :param iteration: max iteration numbers
    :return:
    """
    ite = 0
    step = 1
    dim = int(dim)
    mu = np.full(dim, 0)
    h_power = round(math.log(h, 0.5))
    # if replication >= 100:
    #     rep = 100
    # else:
    #     rep = replication
    rep = replication
    # print('h power is:', h_power)
    # mu = np.array([-100, -49.696, -49.75448, 12.31851, 0])
    # mu = np.array([1,1,1,1,1,10,10,10,10,10])
    # mu = np.random.uniform(0, 100, 10)
    h = 0.5 ** h_power
    tau = round(tau/h)*h
    mu_presim = np.zeros(int(tau / h)+1)
    index_array = np.arange(0, len(mu_presim)*h, h)[0: len(mu_presim)]
    # index_array = np.arange(0, tau+h, h)
    # plt.figure(0)
    for axis in range(dim):
        mu_presim = mu_presim + Legendre.basis(axis)(index_array)*mu[axis]
    # plt.show()
    mumu = (mu_presim[1:] - mu_presim[:-1])/h
    mumu = np.insert(mumu, 0, mu_presim[0])
    mu_sim = np.repeat(mumu, sim_scale, axis=0)/sim_scale
    mu_arr = np.array([mu])
    presim_arr = np.array([mu_presim])
    # generate Brownian paths
    # print(Bpa)
    Bpaths = Bpathgrab(Bpa, h_power, Bpa_power, rep)
    new_int = mc4(mu_sim, rep, h/sim_scale, Bpaths, t, b)
    results = np.array(new_int)
    if runTrue:
        Bpaths_true = Bpathgrab(Bpa, 10, Bpa_power, 400)
        mumu = (mu_presim[1:] - mu_presim[:-1]) / (0.5 ** 10)
        mumu = np.insert(mumu, 0, mu_presim[0])
        mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
        true_int = mc4(mu_sim, 400, 0.5 ** 10 / sim_scale, Bpaths_true, t, b)
    else:
        true_int = float('nan')
    if iteration >= 50:
        #storestep = int(iteration / 50)
        storestep = 1
        storelist = np.arange(storestep, iteration + storestep, storestep).tolist()
    else:
        storelist = np.arange(1, iteration + 1, 1).tolist()
    print("iteration:", 0, 'drift:', mu, 'val', true_int)
    trueres_list = [true_int]
    res_list = [new_int]
    n_list = [dim]
    N_list = [rep]
    h_list = [0.5**h_power]
    h_orglist = [h]
    work_list = [0]
    # i is the index of gd iterations
    for i in range(iteration):
        # gradsum = np.empty((0, int(tau / h)))
        gradsum = np.zeros(dim)
        # k is the index of omega
        # RBMpath = RBM(mu_sim, sample, h)
        for k in range(rep):
            RBMpath = RBM(mu_sim, Bpaths[k], h / sim_scale)
            RBM_arr = np.array(RBM(mu_sim, Bpaths[k], h / sim_scale)[1])
            # plt.figure(4)
            # plt.plot(RBMpath[1])
            # print(RBMpath[1])
            gradrep = np.zeros(dim)
            # axis is the index of gradient entries
            Phi_record = []
            infimum = 0
            Phi = [0]
            for time in range(int(tau / (h / sim_scale))):
                if RBMpath[0][time + 1] < infimum:
                    # print(RBMpath[0][time+1])
                    Phi = [time + 1]
                    Phi_record.append(Phi)
                    infimum = RBMpath[0][time + 1]
                elif RBMpath[0][time + 1] == infimum:
                    # print(RBMpath[0][time+1])
                    Phi.append(Phi[-1])
                    Phi_record.append(Phi)
                else:
                    Phi_record.append(Phi)
            # Phi_arr = np.unique(Phi_record)
            for axis in range(dim):
                DuJ2 = 0
                # integrand = Legendre.basis(axis)
                # DuJ1 = quad(integrand, 0, tau)[0]
                time_arr = np.arange(h, RBM_arr.shape[0] * h + h, h)
                # print(RBMpath, gprime3(RBMpath))
                integrand = Legendre.basis(axis)(time_arr) * gprime4(RBM_arr, t, b)
                DuJ1 = sum(integrand) * h
                # test = DuJ1
                for time in range(int(tau / (h / sim_scale))):
                    Phi = Phi_record[time]
                    addon = dir_fun(axis, Phi[0] * h / sim_scale)
                    for item in Phi:
                        if dir_fun(axis, item * h / sim_scale) < addon:
                            addon = dir_fun(axis, item * h / sim_scale)
                    DuJ2 = DuJ2 - addon * gprime4(RBM_arr[time], t, b) * h / sim_scale
                if axis == 0:
                    gradrep[axis] = DuJ1  # + DuJ2
                else:
                    gradrep[axis] = DuJ1 + DuJ2
            gradsum = gradsum + gradrep
        # plt.show()
        directional = gradsum/rep + penaltyderiv4()
        weights = np.zeros(dim)
        for axis in range(dim):
            weights[axis] = directional[axis]
        step = backtracking_line_search4(Bpaths_true, true_int, mu, directional, tau, h, t, b, beta, sim_scale=1)
        # step = 1e-6
        mu = mu - step * directional
        mu_presim = np.zeros(int(tau/h) + 1)
        mu_arr = np.append(mu_arr, [mu], axis=0)
        presim_arr = np.append(presim_arr, [mu_presim], axis=0)
        index_array = np.arange(0, len(mu_presim)*h, h)[0: len(mu_presim)]
        # print(mu_presim, index_array)
        # index_array = np.arange(0, tau + h, h)
        for axis in range(dim):
            mu_presim = mu_presim + Legendre.basis(axis)(index_array) * mu[axis]
        mumu = (mu_presim[1:] - mu_presim[:-1])/h
        mumu = np.insert(mumu, 0, mu_presim[0]/h)
        mu_sim = np.repeat(mumu, sim_scale, axis=0)/sim_scale
        new_int = mc4(mu_sim, rep, h / sim_scale, Bpaths, t, b)
        if runTrue:
            mumu = (mu_presim[1:] - mu_presim[:-1]) / (0.5 ** 10)
            mumu = np.insert(mumu, 0, mu_presim[0])
            mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
            true_int = mc4(mu_sim, 400, 0.5**10 / sim_scale, Bpaths_true, t, b)
        else:
            true_int = float('nan')
        ite = ite + 1
        results = np.append(results, new_int)
        norm = math.sqrt(np.dot(weights, weights))
        # print("iteration:", ite, norm, weights, mu_presim, mu_sim, new_int)
        if ite % 1 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('time is', current_time)
            print("iteration:", ite, 'step:', step, 'gradient:', directional, 'drift:', mu, 'val', true_int)
        if ite in storelist:
            trueres_list.append(true_int)
            res_list.append(new_int)
            n_list.append(dim)
            N_list.append(replication)
            h_orglist.append(h)
            h_list.append(0.5**h_power)
            work_list.append(rep*ite*dim/h)
            #work_list.append(ite*dim/h)
    d = {'n': n_list, 'h': h_list, 'org_h': h_orglist, 'W': work_list, 'value': res_list, 'trueval': trueres_list}
    s = pd.DataFrame(data=d)
    print("iteration:", ite, "value:", new_int, "drift:", mu_presim, "\n")
    return [ite, mu_sim, new_int, results, dim, h, mu_arr, presim_arr, tau, s]


def driver(nnn, seedd, sigma):
    '''experiment 7 '''
    seednow = seedd
    random.seed(seednow)
    # b_list = [10, 100, 1000, 10000]
    # plan_power = [2/3, 1/2, 1/3]
    b_list = [10000]
    plan_power = [1/2]
    Pathsource = []
    for j in range(470):
        Pathsource.append(Brownian(1, 0.5**10, sigma))
    for id in range(nnn):
        rep_list = [round(item**(plan_power[id])) for item in b_list]
        h_list = [item**(plan_power[id]-1) for item in b_list]
        for k in range(len(b_list)):
            n_val = float(3)
            h_now = float(h_list[k])
            h_power = round(math.log(h_now, 0.5))
            h_now = float(0.5 ** h_power)
            replication = int(rep_list[k])
            itenum = int(100)
            lambda1 = 1000.0
            M = 10.0
            beta = 0.1
            print('current setting is:', n_val, h_now, itenum)
            # full_res = gd(1, h_now, n_val, 0.1, replication, itenum, 1, Pathsource, 10)
            full_res = gd3(1, h_now, n_val, 1e-8, replication, itenum, 1, Pathsource, 10, True, lambda1, M, beta)
            df = full_res[-1]
            df['plan_power'] = np.full(df.shape[0], plan_power[id])
            df['plan_cost'] = np.full(df.shape[0], b_list[k])
            print("current comb for n = ", n_val, "h = ", h_now, "plan = ", b_list[k], plan_power[id], " is:", df)
            if id == int(0) and k == int(0):
                data = df
            else:
                data = data.append(df)
    data = data.reset_index(drop=True)
    data.to_csv('7.1plans_init-10-10-10_lambda=1000_M=10_ite100_seed_' + str(seednow))
    return data


# best = driver(1, 1000)
# best = driver(3, 2000)
# best = driver(3, 3000)
# best = driver(3, 4000)
# best = driver(3, 5000)


def driver2(nnn, seedd, t, b, sigma):
    '''experiment 7.2 '''
    seednow = seedd
    random.seed(seednow)
    # b_list = [10, 100, 1000, 10000]
    b_list = [1e4]
    # plan_power = [3/5]
    plan_power = [1/2, 3/5, 2/3]
    Pathsource = []
    for j in range(470):
        Pathsource.append(Brownian(1, 0.5**10, sigma))
    for id in range(nnn):
        h_list = [item**(-plan_power[id]) for item in b_list]
        rep_list = [round(item**(1-plan_power[id])) for item in b_list]
        for k in range(len(b_list)):
            n_val = float(3)
            h_now = float(h_list[k])
            h_power = round(math.log(h_now, 0.5))
            h_now = float(0.5 ** h_power)
            replication = int(rep_list[k])
            itenum = int(100)
            beta = 0.1
            print('current setting is:', n_val, h_now, replication, itenum)
            # full_res = gd(1, h_now, n_val, 0.1, replication, itenum, 1, Pathsource, 10)
            full_res = gd4(1, h_now, n_val, 1e-8, replication, itenum, 1, Pathsource, 10, True, t, b, beta)
            df = full_res[-1]
            df['plan_power'] = np.full(df.shape[0], plan_power[id])
            df['plan_cost'] = np.full(df.shape[0], b_list[k])
            print("current comb for n = ", n_val, "h = ", h_now, "plan = ", b_list[k], plan_power[id], " is:", df)
            if id == int(0) and k == int(0):
                data = df
            else:
                data = data.append(df)
    data = data.reset_index(drop=True)
    data.to_csv('7.2plans_init+0+0+0_coeb='+str(b)+'t='+str(t)+'sigma='+str(sigma)+'_ite100_b1e4_3etas_seed_' + str(seednow))
    return data


best = driver2(3, 1000, -200, 1, 100)
# best = driver2(1, 1000, -2, 1, 30)
# best = driver2(1, 1000, -2, 1, 50)
# best = driver2(1, 1000, -200, 1, 10)
# best = driver2(1, 1000, -200, 1, 30)
# best = driver2(1, 1000, -200, 1, 50)
# best = driver2(1, 1000, -20000, 1, 0)
# best = driver2(1, 1000, -20000, 1, 30)
# best = driver2(1, 1000, -20000, 1, 50)



