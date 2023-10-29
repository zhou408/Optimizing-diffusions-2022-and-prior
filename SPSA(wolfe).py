import math
import numpy as np
import random
import matplotlib.pyplot as plt
from step_size_debug import wolfe6, wolfe7
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
        T1[t] = (-driftvec[t])*h + np.random.normal(0, 1, 1)[0]*h
        T2[t] = T1[t]/2+(T1[t]**2-2*math.log(np.random.uniform(0, 1, 1)[0]))**0.5/2
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


# with wolfe and mc2
def opt6(init, rep, dim, h):
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
        if count % 5 == 0:
            print("step is")
            step = wolfe6(mu, grad, rep, h, dim)
            print(step, "\n")
        mu = mu-step*grad
        old_int = new_int
        new_int = mc2(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        if count % 5 == 0:
            stdev = mcsd(mu, rep, h)
            print("the", count, "iteration:", new_int, "stdev", stdev, "\n")
        print('mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "std:", stdev, "\n")
    return [count, mu, new_int, stdev, results]


# with wolfe and mcgx
def opt7(init, rep, dim, h):
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
        deltamu = np.random.uniform(0, 1, int(dim/h))
        scale = np.random.uniform(0, 2, 1)[0]
        grad = (mcgx(mu+scale*deltamu, rep, h)-mcgx(mu-scale*deltamu, rep, h))/(2*scale*deltamu)
        if count % 5 == 0:
            print("step is")
            step = wolfe7(mu, grad, rep, h, dim)
            print(step, "\n")
        mu = mu-step*grad
        old_int = new_int
        new_int = mcgx(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        if count % 5 == 0:
            stdev = mcsd(mu, rep, h)
            print("the", count, "iteration:", new_int, "stdev", stdev, "\n")
        print('mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "std:", stdev, "\n")
    return [count, mu, new_int, stdev, results]

'''
# opt6(1, 30, 5, 1)

# opt7(1, 30, 5, 1)

# res3 = opt6(1, 50, 5, 1)
# pd.DataFrame(res3).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt6.csv", header=None, index=None)
# pd.DataFrame(res3[4]).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt6(value).csv", header=None, index=None)

res3 = opt6(1, 50, 5, 1)
data = pd.DataFrame(res3).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt6.csv", index=None)

data = pd.DataFrame(res3[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt6(value).csv", index=None)

# res4 = opt7(1, 50, 5, 1)
# pd.DataFrame(res4).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt7.csv", header=None, index=None)
# pd.DataFrame(res4[4]).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt7(value).csv", header=None, index=None)
res4 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res4).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt7.csv", index=None)

data = pd.DataFrame(res4[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt7(value).csv", index=None)

###############################
res3 = opt6(1, 50, 5, 1)
data = pd.DataFrame(res3).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt6.csv", index=None)

data = pd.DataFrame(res3[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt6(value).csv", index=None)

res4 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res4).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt7.csv", index=None)

data = pd.DataFrame(res4[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt7(value).csv", index=None)
################################
res3 = opt6(1, 50, 5, 1)
data = pd.DataFrame(res3).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt6.csv", index=None)

data = pd.DataFrame(res3[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt6(value).csv", index=None)

res4 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res4).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt7.csv", index=None)

data = pd.DataFrame(res4[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt7(value).csv", index=None)
################################

res3 = opt6(1, 50, 5, 1)
data = pd.DataFrame(res3).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt6.csv", index=None)

data = pd.DataFrame(res3[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt6(value).csv", index=None)

res4 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res4).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt7.csv", index=None)

data = pd.DataFrame(res4[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt7(value).csv", index=None)
###########################################

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data11/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data11/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data12/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data12/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data13/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data13/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data14/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data14/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data15/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data15/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data16/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data16/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data17/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data17/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data18/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data18/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data19/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data19/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data20/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data20/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data21/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data21/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data22/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data22/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data23/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data23/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data24/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data24/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data25/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data25/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data26/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data26/opt6(value).csv", index=None)


res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data27/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data27/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data28/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data28/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data29/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data29/opt6(value).csv", index=None)

res8 = opt6(1,50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data30/opt6.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data30/opt6(value).csv", index=None)

#################################################################

###########################################

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data11/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data11/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data12/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data12/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data13/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data13/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data14/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data14/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data15/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data15/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data16/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data16/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data17/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data17/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data18/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data18/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data19/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data19/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data20/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data20/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data21/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data21/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data22/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data22/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data23/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data23/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data24/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data24/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data25/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data25/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data26/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data26/opt7(value).csv", index=None)


res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data27/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data27/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data28/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data28/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data29/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data29/opt7(value).csv", index=None)

res8 = opt7(1, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data30/opt7.csv", index=None)
data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['SPSA_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data30/opt7(value).csv", index=None)

#################################################################
'''