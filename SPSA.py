import math
import numpy as np
import random
import matplotlib.pyplot as plt
# from step_size_debug import wolfe6, wolfe7
import pandas as pd

# sample path using Asmussen,Glynn,Pitman
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
        # T1[t] = (-driftvec[t])*h + np.random.normal(0, 1, 1)[0]*math.sqrt(h)
        T1[t] = np.random.normal((-driftvec[t]) * h, math.sqrt(h), 1)[0]
        T2[t] = T1[t] / 2 + math.sqrt(T1[t] ** 2 - 2 * h * math.log(np.random.uniform(0, 1, 1)[0])) / 2
        M[t+1] = max(M[t], B[t]+T2[t])
        B[t+1] = B[t]+T1[t]
        X[t+1] = M[t+1]-B[t+1]
    return [T1, X]


# trial = rbm2(np.zeros(100), 1)
# print(trial[1])
# plt.plot(trial[1])
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


def mcsd(driftvec, rep, h):
    """

    :param driftvec:
    :param rep:
    :param h:
    :return:
    """
    vector = np.zeros(rep)
    for i in range(rep):
        vector[i] = sum(rbm2(driftvec, h)[1])*h
    # expect = sum(vector)/rep
    std = np.std(vector)
    return std


def mcgx(driftvec, rep, h):
    """

    :param driftvec:
    :param rep:
    :param h:
    :return:
    """
    totalsum = 0
    for i in range(rep):
        rbmpath = rbm2(driftvec, h)[1]
        totalsum = totalsum+sum(rbmpath)*h-3*rbmpath[-1]*h
    expect = totalsum/rep
    return expect


# with fixed step size and mc2
def opt(step, rep, dim, h):
    """

    :param step:
    :param rep:
    :param dim:
    :param h:
    :return:
    """
    random.seed(np.random.uniform(0, 10000, 1)[0])
    # mu = np.random.uniform(0, 10, int(dim/h))
    mu = np.full(int(dim / h), 1)
    # print(mu, "\n")
    old_int = 1e1000000
    new_int = mc2(mu, rep, h)
    count = 0
    results = [0]
    results = np.asarray(results)
    # grad = rep(1, int(dim/h) )
    # while abs(old_int-new_int) >= 0.000005:
    while count <= 1000:
    # while(max(abs(grad))>=0.0001){
    # while(max(abs(grad))>=1e-3){
        # if(abs(old_int-new_int)<=0.05){step = step/5}
        # deltamu = runif(int(dim/h) , -1, -1)
        # when deltamu is set to runif(int(dim/h) , -1, -1), obejective function value barely changes. cancel out effect?
        deltamu = np.random.uniform(0, 1, int(dim/h) )
        scale = np.random.uniform(0, 2, 1)[0]
        grad = (mc2(mu+scale*deltamu, rep, h)-mc2(mu-scale*deltamu, rep, h))/(2*scale*deltamu)
        # print(grad, "\n")
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


# with fixed stepsize and mcgx
def opt1(step, rep, dim, h):
    """

    :param step:
    :param rep:
    :param dim:
    :param h:
    :return:
    """
    random.seed(np.random.uniform(0, 10000, 1)[0])
    # mu = np.random.uniform(0, 10, int(dim/h))
    mu = np.full(int(dim / h), 1)
    old_int = 1e1000000
    new_int = mcgx(mu, rep, h)
    count = 0
    results = [0]
    results = np.asarray(results)
    # grad = rep(1, int(dim/h) )
    while count <= 1000:
    # while abs(old_int-new_int) >= 0.0005:
    # while(max(abs(grad))>=0.0001){
    # while(max(abs(grad))>=1e-3){
        # if(abs(old_int-new_int)<=0.05){step = step/5}
        # deltamu = runif(int(dim/h) , -1, -1)
        # when deltamu is set to runif(int(dim/h) , -1, -1), obejective function value barely changes. cancel out effect?
        deltamu = np.random.uniform(0, 1, int(dim/h))
        scale = np.random.uniform(0, 2, 1)[0]
        grad = (mcgx(mu+scale*deltamu, rep, h)-mcgx(mu-scale*deltamu, rep, h))/(2*scale*deltamu)
        mu = mu-step*grad
        old_int = new_int
        new_int = mcgx(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        if count % 5 == 0:
            stdev = mcsd(mu, rep, h)
            print("the", count, "iteration:", new_int, "stdev", stdev, "\n")
        print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "std:", stdev, "\n")
    return [count, mu, new_int, stdev, results]


# with decreasing step size and mc2
def opt2(init, rep, dim, h):
    """

    :param init:
    :param rep:
    :param dim:
    :param h:
    :return:
    """
    step = init
    random.seed(np.random.uniform(0, 10000, 1)[0])
    # mu = np.random.uniform(0, 10, int(dim/h))
    mu = np.full(int(dim / h), 1)
    old_int = 1e1000000
    new_int = mc2(mu, rep, h)
    count = 0
    results = [0]
    results = np.asarray(results)
    # grad = rep(1, int(dim/h) )
    while count <= 1000:
    # while abs(old_int-new_int) >= 0.005:
    # while(max(abs(grad))>=0.0001){
    # while(max(abs(grad))>=1e-3){
        # if(abs(old_int-new_int)<=0.05){step = step/5}
        # deltamu = runif(int(dim/h) , -1, -1)
        # when deltamu is set to runif(int(dim/h) , -1, -1), obejective function value barely changes. cancel out effect?
        deltamu = np.random.uniform(0, 1, int(dim/h) )
        scale = np.random.uniform(0, 2, 1)[0]
        grad = (mc2(mu+scale*deltamu, rep, h)-mc2(mu-scale*deltamu, rep, h))/(2*scale*deltamu)
        mu = mu-step*grad
        old_int = new_int
        new_int = mc2(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        step = 5 / count
        if count % 5 == 0:
            stdev = mcsd(mu, rep, h)
            print("the", count, "iteration:", new_int, "stdev", stdev, "\n")
        print("the", count, 'mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "std:", stdev, "\n")
    return [count, mu, new_int, stdev, results]

# with decreasing step size and mcgx
def opt3(init, rep, dim, h):
    """

    :param init:
    :param rep:
    :param dim:
    :param h:
    :return:
    """
    step = init
    random.seed(np.random.uniform(0, 10000, 1)[0])
    # mu = np.random.uniform(0, 10, int(dim/h))
    mu = np.full(int(dim / h), 1)
    old_int = 1e1000000
    new_int = mcgx(mu, rep, h)
    count = 0
    results = [0]
    results = np.asarray(results)
    # grad = rep(1, int(dim/h) )
    # while abs(old_int-new_int) >= 0.000005:
    while count <= 1000:
    # while(max(abs(grad))>=0.0001){
    # while(max(abs(grad))>=1e-3){
        # if(abs(old_int-new_int)<=0.05){step = step/5}
        # deltamu = runif(int(dim/h) , -1, -1)
        # when deltamu is set to runif(int(dim/h) , -1, -1), obejective function value barely changes. cancel out effect?
        deltamu = np.random.uniform(0, 1, int(dim/h) )
        scale = np.random.uniform(0, 2, 1)[0]
        grad = (mcgx(mu+scale*deltamu, rep, h)-mcgx(mu-scale*deltamu, rep, h))/(2*scale*deltamu)
        mu = mu-step*grad
        old_int = new_int
        new_int = mcgx(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        step = 5 / count
        if count % 5 == 0:
            stdev = mcsd(mu, rep, h)
            print("the", count, "iteration:", new_int, "stdev", stdev, "\n")
        print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "std:", stdev, "\n")
    return [count, mu, new_int, stdev, results]

'''
# res1 = opt(1, 50, 5, 1)
# pd.DataFrame(res1).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt.csv",index=None)
# pd.DataFrame(res1[4]).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt(value).csv", index=None)
data = pd.DataFrame(res1).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt.csv", index=None)

data = pd.DataFrame(res1[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt(value).csv", index=None)

# res2 = opt1(1, 50, 5, 1)
# pd.DataFrame(res2).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt1.csv", index=None)
# pd.DataFrame(res2[4]).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt1(value).csv", index=None)
data = pd.DataFrame(res2).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt1.csv", index=None)

data = pd.DataFrame(res2[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt1(value).csv", index=None)

###################
res1 = opt(1, 50, 5, 1)
data = pd.DataFrame(res1).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt.csv", index=None)

data = pd.DataFrame(res1[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt(value).csv", index=None)

res2 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res2).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt1.csv", index=None)

data = pd.DataFrame(res2[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt1(value).csv", index=None)

###################

res1 = opt(1, 50, 5, 1)
data = pd.DataFrame(res1).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt.csv", index=None)

data = pd.DataFrame(res1[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt(value).csv", index=None)

res2 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res2).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt1.csv", index=None)

data = pd.DataFrame(res2[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt1(value).csv", index=None)

###################

res1 = opt(1, 50, 5, 1)
data = pd.DataFrame(res1).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt.csv", index=None)

data = pd.DataFrame(res1[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt(value).csv", index=None)

res2 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res2).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt1.csv", index=None)

data = pd.DataFrame(res2[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt1(value).csv", index=None)

###########################################

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data11/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data11/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data12/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data12/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data13/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data13/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data14/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data14/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data15/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data15/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data16/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data16/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data17/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data17/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data18/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data18/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data19/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data19/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data20/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data20/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data21/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data21/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data22/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data22/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data23/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data23/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data24/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data24/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data25/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data25/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data26/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data26/opt(value).csv", index=None)


res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data27/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data27/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data28/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data28/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data29/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data29/opt(value).csv", index=None)

res8 = opt(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data30/opt.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA']
data.to_csv("D:/purdue/RBM/Sim3/Python/data30/opt(value).csv", index=None)

#################################################################
res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data11/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data11/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data12/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data12/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data13/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data13/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data14/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data14/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data15/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data15/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data16/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data16/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data17/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data17/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data18/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data18/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data19/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data19/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data20/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data20/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data21/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data21/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data22/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data22/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data23/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data23/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data24/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data24/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data25/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data25/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data26/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data26/opt1(value).csv", index=None)


res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data27/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data27/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data28/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data28/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data29/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data29/opt1(value).csv", index=None)

res8 = opt1(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data30/opt1.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)']
data.to_csv("D:/purdue/RBM/Sim3/Python/data30/opt1(value).csv", index=None)
'''