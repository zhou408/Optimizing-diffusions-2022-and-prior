#!  /usr/bin/python
#testing theorem
import time
import numpy as np
import math
import random
import sys
from scipy.stats import norm
from scipy.special import legendre
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def Brownian(tau, h):
    """
    This function produces a single Brownian path.
    :param tau: time horizon
    :param h: time disretization step size
    :return: Brownian path
    """

    time = int(tau / h)
    Bpath = np.zeros(time)
    for t in range(time):
        Bpath[t] = np.random.normal(0, math.sqrt(h), 1)[0]
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
    B[0] = -driftvec[0]
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


def gd5(tau, h, dim, step, replication, iteration, sim_scale, Bpa, Bpa_power, runTrue):
    """
    experiment 4, only change N and h for graident, compute true, trading n and h only for gradient
    """
    ite = 0
    dim = int(dim)
    mu = np.full(dim, 0)
    h2 = 0.5**4
    h_power = round(math.log(h, 0.5))
    h_power2 = round(math.log(h2, 0.5))
    rep = replication
    rep2 = 50
    # print('h power is:', h_power)
    # mu = np.array([-100, -49.696, -49.75448, 12.31851, 0])
    # mu = np.array([1,1,1,1,1,10,10,10,10,10])
    # mu = np.random.uniform(0, 100, 10)
    tau = round(tau/h2)*h2
    mu_presim = np.zeros(int(tau / h2)+1)
    index_array = np.arange(0, len(mu_presim)*h2, h2)[0: len(mu_presim)]
    # index_array = np.arange(0, tau+h, h)
    # plt.figure(0)
    for axis in range(dim):
        mu_presim = mu_presim + legendre(axis)(index_array)*mu[axis]
    # plt.show()
    mumu = (mu_presim[1:] - mu_presim[:-1])/h2
    mumu = np.insert(mumu, 0, mu_presim[0])
    mu_sim = np.repeat(mumu, sim_scale, axis=0)/sim_scale
    mu_arr = np.array([mu])
    presim_arr = np.array([mu_presim])
    # generate Brownian paths
    # print(Bpa)
    Bpaths2 = Bpathgrab(Bpa, h_power2, Bpa_power, rep2)
    new_int = mc2(mu_sim, rep2, h/sim_scale, Bpaths2)
    results = np.array(new_int)
    if runTrue:
        Bpaths_true = Bpathgrab(Bpa, 10, Bpa_power, 400)
        mumu = (mu_presim[1:] - mu_presim[:-1]) / (0.5 ** 10)
        mumu = np.insert(mumu, 0, mu_presim[0])
        mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
        true_int = mc2(mu_sim, 400, 0.5 ** 10 / sim_scale, Bpaths_true)
    else:
        true_int = float('nan')
    if iteration >= 50:
        #storestep = int(iteration / 50)
        storestep = 1
        storelist = np.arange(storestep, iteration + storestep, storestep).tolist()
    else:
        storelist = np.arange(1, iteration + 1, 1).tolist()
    res_list = [new_int]
    trueres_list = [true_int]
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
            mu_presim = np.zeros(int(tau / h) + 1)
            index_array = np.arange(0, len(mu_presim) * h, h)[0: len(mu_presim)]
            for axis in range(dim):
                mu_presim = mu_presim + legendre(axis)(index_array) * mu[axis]
            mumu = (mu_presim[1:] - mu_presim[:-1]) / h
            mumu = np.insert(mumu, 0, mu_presim[0])
            mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
            Bpaths = Bpathgrab(Bpa, h_power, Bpa_power, rep)
            RBMpath = RBM(mu_sim, Bpaths[k], h / sim_scale)
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
            Phi_arr = np.unique(Phi_record)
            for axis in range(dim):
                DuJ2 = 0
                integrand = legendre(axis)
                DuJ1 = quad(integrand, 0, tau)[0]
                test = DuJ1
                for time in range(int(tau / (h / sim_scale))):
                    Phi = Phi_record[time]
                    addon = dir_fun(axis, Phi[0] * h / sim_scale)
                    for item in Phi:
                        if dir_fun(axis, item * h / sim_scale) < addon:
                            addon = dir_fun(axis, item * h / sim_scale)
                    DuJ2 = DuJ2 - addon * h / sim_scale
                if axis == 0:
                    gradrep[axis] = DuJ1  # + DuJ2
                else:
                    gradrep[axis] = DuJ1 + DuJ2
            gradsum = gradsum + gradrep
        # plt.show()
        directional = gradsum/rep
        weights = np.zeros(dim)
        for axis in range(dim):
            weights[axis] = directional[axis]
        mu = mu - step * directional
        mu_presim = np.zeros(int(tau/h2) + 1)
        mu_arr = np.append(mu_arr, [mu], axis=0)
        presim_arr = np.append(presim_arr, [mu_presim], axis=0)
        index_array = np.arange(0, len(mu_presim)*h2, h2)[0: len(mu_presim)]
        # print(mu_presim, index_array)
        # index_array = np.arange(0, tau + h, h)
        for axis in range(dim):
            mu_presim = mu_presim + legendre(axis)(index_array) * mu[axis]
        mumu = (mu_presim[1:] - mu_presim[:-1])/h2
        mumu = np.insert(mumu, 0, mu_presim[0]/h2)
        mu_sim = np.repeat(mumu, sim_scale, axis=0)/sim_scale
        new_int = mc2(mu_sim, rep2, h2 / sim_scale, Bpaths2)
        if runTrue:
            mumu = (mu_presim[1:] - mu_presim[:-1]) / (0.5 ** 10)
            mumu = np.insert(mumu, 0, mu_presim[0])
            mu_sim = np.repeat(mumu, sim_scale, axis=0) / sim_scale
            true_int = mc2(mu_sim, 400, 0.5**10 / sim_scale, Bpaths_true)
        else:
            true_int = float('nan')
        ite = ite + 1
        results = np.append(results, new_int)
        norm = math.sqrt(np.dot(weights, weights))
        # print("iteration:", ite, norm, weights, mu_presim, mu_sim, new_int)
        if ite%20 == 0:
            print("iteration:", ite, norm, directional, mu, mu_presim, new_int)
        if ite in storelist:
            res_list.append(new_int)
            trueres_list.append(true_int)
            n_list.append(dim)
            N_list.append(replication)
            h_orglist.append(h)
            h_list.append(0.5**h_power)
            work_list.append(rep*ite/h)
    d = {'n': n_list, 'h': h_list, 'org_h': h_orglist, 'W': work_list, 'value': res_list, 'trueval': trueres_list}
    s = pd.DataFrame(data=d)
    print("iteration:", ite, "value:", new_int, "drift:", mu_presim, "\n")
    return [ite, mu_sim, new_int, results, dim, h, mu_arr, presim_arr, tau, s]


def driver6(seedd):
    """
    experiment 4, only change N and h for graident, compute true, trading n and h only for gradient
    """
    seednow = seedd
    random.seed(seednow)
    res_list = []
    # b_list = [10, 100, 1000, 10000]
    # plan_power = [2/3, 1/2, 1/3]
    b_list = [50, 100, 200, 400]
    n_list = [3, 4, 5]
    Pathsource = []
    for j in range(401):
        Pathsource.append(Brownian(1, 0.5**10))
    for id in range(len(n_list)):
        # rep_list = [round(item**(plan_power[id])) for item in b_list]
        rep_list = [50 for item in b_list]
        h_list = [n_list[id]/item for item in b_list]
        for k in range(len(b_list)):
            n_val = int(n_list[id])
            h_now = float(h_list[k])
            h_power = round(math.log(h_now, 0.5))
            h_now = float(0.5 ** h_power)
            replication = rep_list[k]
            itenum = int(100)
            if id == int(0) and k == int(0):
                record_true = True
            else:
                # record_true = False
                record_true = True
            print('current setting is:', n_val, h_now)
            full_res = gd5(1, h_now, n_val, 0.1, replication, itenum, 1, Pathsource, 10, record_true)
            df = full_res[-1]
            # df['plan_power'] = np.full(df.shape[0], plan_power[id])
            df['plan_cost'] = np.full(df.shape[0], b_list[k])
            print("current comb for n = ", n_val, "h = ", h_now, "plan = ", b_list[k], " is:", df)
            if id == int(0) and k == int(0):
                data = df
            else:
                data = data.append(df)
    data = data.reset_index(drop=True)
    data.to_csv('6plans_withTrue_trial_ite100_seed_' + str(seednow))
    # data.to_csv('RBMexperiments2021/6plans_withTrue_trial_ite100_seed_' + str(seednow))
    return data


best = driver6(1000)
best = driver6(2000)
best = driver6(3000)
best = driver6(4000)
best = driver6(5000)
