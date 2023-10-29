from scipy.stats import norm
import numpy as np
import math


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
    for t in range(time):
        # scale down the drifts and dc according to h
        # T1[t] = (-driftvec[t])*h + np.random.normal(0, math.sqrt(h), 1)[0]
        T1[t] = np.random.normal((-driftvec[t]) * h, math.sqrt(h), 1)[0]
        T2[t] = T1[t]/2+math.sqrt(T1[t]**2-2*h*math.log(np.random.uniform(0, 1, 1)[0]))/2
        M[t+1] = max(M[t], B[t]+T2[t])
        B[t+1] = B[t]+T1[t]
        X[t+1] = M[t+1]-B[t+1]
    return [T1, X]


# trial = rbm2(np.zeros(100), 1)
# print(trial[1])
# plt.plot(trial[1])
# plt.show

def expectation(driftvec, rep, h):
    """

    :param driftvec:
    :param rep:
    :param h:
    :return:
    """
    totalsum = 0
    for i in range(rep):
        rbmvalue = rbm2(driftvec, h)[1][-1]
        totalsum = totalsum+rbmvalue
    expect = totalsum/rep
    return expect

# expectation([-1,-1,-1,-1,-1], 20, 1)

# expectation(np.full(10, -1), 10, 0.5)

# T = 5

def real_exp(T):
    exp = 0.5 + math.sqrt(T)*norm.pdf(math.sqrt(T))-T*(1-norm.cdf(math.sqrt(T)))
    return exp


def table(drift, ite):

    T = 10
    h = 0.5
    k = 100
    content = np.zeros(ite+1)
    diff = np.zeros(ite)
    content[-1] = real_exp(T)
    for i in range(ite):
        vec = np.full(k, drift)
        content[i] = expectation(vec, k, h)
        diff[i] = abs(content[i]-content[-1])*math.sqrt(k * T / h)
        h = h / 2
        k = k * 2
    mse = (content-content[-1])**2
    sum_reduc = 0
    sum_reduc_raw = 0
    for j in range(ite-1):
        # sum_reduc = sum_reduc + math.sqrt(mse[j]/mse[j+1])
        sum_reduc = sum_reduc + (diff[j] / diff[j + 1])
        sum_reduc_raw = sum_reduc_raw + (math.sqrt(mse[j]/mse[j+1]))
    avg_reduc_raw = sum_reduc_raw / (ite - 1)
    avg_reduc = sum_reduc/(ite-1)
    return [content, mse, diff, avg_reduc_raw, avg_reduc]



def table2(drift, ite):

    T = 10
    h = 0.5
    k = 100
    content = np.zeros(ite+1)
    diff = np.zeros(ite)
    content[-1] = real_exp(T)
    for i in range(ite):
        vec = np.full(k, drift)
        content[i] = expectation(vec, k, h)
        diff[i] = abs(content[i] - content[-1]) * math.sqrt(k * T / h)
        h = h / 2
        k = k * 3
    mse = (content-content[-1])**2
    sum_reduc = 0
    sum_reduc_raw = 0
    for j in range(ite-1):
        # sum_reduc = sum_reduc + math.sqrt(mse[j]/mse[j+1])
        sum_reduc = sum_reduc + (diff[j] / diff[j + 1])
        sum_reduc_raw = sum_reduc_raw + (math.sqrt(mse[j] / mse[j + 1]))
    avg_reduc_raw = sum_reduc_raw / (ite - 1)
    avg_reduc = sum_reduc/(ite-1)
    return [content, mse, diff, avg_reduc_raw, avg_reduc]


def table3(drift, ite):

    T = 10
    h = 0.5
    k = 100
    content = np.zeros(ite+1)
    diff = np.zeros(ite)
    content[-1] = real_exp(T)
    for i in range(ite):
        vec = np.full(k, drift)
        content[i] = expectation(vec, k, h)
        diff[i] = abs(content[i] - content[-1]) * math.sqrt(k * T / h)
        h = h / 2
        k = k * 1
    mse = (content-content[-1])**2
    sum_reduc = 0
    sum_reduc_raw = 0
    for j in range(ite-1):
        # sum_reduc = sum_reduc + math.sqrt(mse[j]/mse[j+1])
        sum_reduc = sum_reduc + (diff[j] / diff[j + 1])
        sum_reduc_raw = sum_reduc_raw + (math.sqrt(mse[j] / mse[j + 1]))
    avg_reduc_raw = sum_reduc_raw / (ite - 1)
    avg_reduc = sum_reduc/(ite-1)
    return [content, mse, diff, avg_reduc_raw, avg_reduc]

def table4(drift, ite):

    T = 10
    h = 0.5
    k = 100
    content = np.zeros(ite+1)
    diff = np.zeros(ite)
    content[-1] = real_exp(T)
    for i in range(ite):
        vec = np.full(k, drift)
        content[i] = expectation(vec, k, h)
        diff[i] = abs(content[i] - content[-1]) * math.sqrt(k * T / h)
        h = h / 1
        k = k * 2
    mse = (content-content[-1])**2
    sum_reduc = 0
    sum_reduc_raw = 0
    for j in range(ite-1):
        # sum_reduc = sum_reduc + math.sqrt(mse[j]/mse[j+1])
        sum_reduc = sum_reduc + (diff[j] / diff[j + 1])
        sum_reduc_raw = sum_reduc_raw + (math.sqrt(mse[j] / mse[j + 1]))
    avg_reduc_raw = sum_reduc_raw / (ite - 1)
    avg_reduc = sum_reduc/(ite-1)
    return [content, mse, diff, avg_reduc_raw, avg_reduc]


# table(-1,5)
# table2(-1,5)
# rbm2(np.full(10, -1), 1)[1][-1]
# rbm2(np.full(20, -1), 0.5)[1][-1]
# expectation(np.full(10, -1), 2000, 1)

