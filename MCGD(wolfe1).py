import math
import numpy as np
from scipy.stats import norm
import random
from SPSA import mc2, mcsd, mcgx
from GRAD import gradvec, gradvecgx
from step_size_debug import wolfe4, wolfe5
import pandas as pd

# with fixed step size
def opt4(rep, dim, h):
    random.seed(np.random.uniform(0, 10000, 1)[0])
    # mu = np.random.uniform(-5, 1, int(dim/h))
    mu = np.full(int(dim/h), 1)
    # mu = rep(10,5)
    old_int = 1e100
    new_int = mc2(mu, rep, h)
    count = 0
    results = [0]
    results = np.asarray(results)
    # grad = [1]*(dim/h)
    # while abs(old_int-new_int) >= 0.0005:
    step = 1
    while count <= 1000:
        # deltamu = np.random.uniform(0, 10000, dim/h)
        # scale = np.random.uniform(0, 2, 1)[0]
        # for now, 30 is the sample size for gradient estimation
        grad = gradvec(mu, h, rep)
        if count%5 == 0:
            print("step is")
            step = wolfe4(mu, grad, rep, h)
            print(step, "\n")
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
            stdev = mcsd(mu, rep, h)
            print("the", count, "iteration:", new_int, "\n", "std:", stdev)
        print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "std:", stdev, "mu", mu, "\n")
    return [count, mu, new_int, stdev, results]


# with fixed step size and terminal cost
def opt5(rep, dim, h):
    random.seed(np.random.uniform(0, 10000, 1)[0])
    # mu = np.random.uniform(-5, 0.5, int(dim/h))
    mu = np.full(int(dim / h), 1)
    old_int = 1e100
    new_int = mcgx(mu, rep, h)
    count = 0
    results = [0]
    results = np.asarray(results)
    grad = np.full(int((dim/h)), 1)
    step = 1
    # while abs(old_int-new_int) >= 0.0005:
    # while sum(abs(grad)) >= 0.005:
    while count <= 1000:
        # deltamu = np.random.uniform(0, 10000, dim/h)
        # scale = np.random.uniform(0, 2, 1)[0]
        # for now, 30 is the sample size for gradient estimation
        grad = gradvecgx(mu, h, rep, -3)
        if count%5 == 0:
            print("step is")
            step = wolfe5(mu, grad, rep, h)
            print(step, "\n")
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
            stdev = mcsd(mu, rep, h)
            print("the", count, "iteration:", new_int, "\n", "std:", stdev)
        print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "std:", stdev, "mu", mu, "\n")
    return [count, mu, new_int, stdev, results]


# opt4(50, 5, 1)
# opt5(50, 5, 1)

# res7 = opt4(50, 5, 1)
# pd.DataFrame(res7).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt4(wolfe).csv", header=None, index=None)
# pd.DataFrame(res7[4]).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt4(value)(wolfe).csv", header=None, index=None)
res7 = opt4(50, 5, 1)
data = pd.DataFrame(res7).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt4_Wolfe.csv", index=None)

data = pd.DataFrame(res7[4]).reset_index(drop=True)
data.columns = ['MCGD_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt4(value)_Wolfe.csv", index=None)

# res8 = opt5(50, 5, 1)
# pd.DataFrame(res8).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt5(wolfe).csv", header=None, index=None)
# pd.DataFrame(res8[4]).to_csv("D:/purdue/RBM/Sim3/Python/data1/opt5(value)(wolfe).csv", header=None, index=None)
res8 = opt5(50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt5_Wolfe.csv", index=None)

data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['MCGD_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt5(value)_Wolfe.csv", index=None)
###############################
res7 = opt4(50, 5, 1)
data = pd.DataFrame(res7).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt4_Wolfe.csv", index=None)

data = pd.DataFrame(res7[4]).reset_index(drop=True)
data.columns = ['MCGD_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt4(value)_Wolfe.csv", index=None)

res8 = opt5(50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt5_Wolfe.csv", index=None)

data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['MCGD_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt5(value)_Wolfe.csv", index=None)
###############################

res7 = opt4(50, 5, 1)
data = pd.DataFrame(res7).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt4_Wolfe.csv", index=None)

data = pd.DataFrame(res7[4]).reset_index(drop=True)
data.columns = ['MCGD_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt4(value)_Wolfe.csv", index=None)

res8 = opt5(50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt5_Wolfe.csv", index=None)

data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['MCGD_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt5(value)_Wolfe.csv", index=None)
###############################

res7 = opt4(50, 5, 1)
data = pd.DataFrame(res7).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt4_Wolfe.csv", index=None)

data = pd.DataFrame(res7[4]).reset_index(drop=True)
data.columns = ['MCGD_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt4(value)_Wolfe.csv", index=None)

res8 = opt5(50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt5_Wolfe.csv", index=None)

data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['MCGD_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt5(value)_Wolfe.csv", index=None)
###############################

res8 = opt5(50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt5_Wolfe.csv", index=None)

data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['MCGD_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt5(value)_Wolfe.csv", index=None)

res8 = opt5(50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt5_Wolfe.csv", index=None)

data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['MCGD_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt5(value)_Wolfe.csv", index=None)

res8 = opt5(50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt5_Wolfe.csv", index=None)

data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['MCGD_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt5(value)_Wolfe.csv", index=None)

res8 = opt5(50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt5_Wolfe.csv", index=None)

data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['MCGD_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt5(value)_Wolfe.csv", index=None)

res8 = opt5(50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt5_Wolfe.csv", index=None)

data = pd.DataFrame(res8[4]).reset_index(drop=True)
data.columns = ['MCGD_G(x)_Wolfe']
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt5(value)_Wolfe.csv", index=None)

###############################
