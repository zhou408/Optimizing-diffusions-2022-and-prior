import math
import numpy as np
from scipy.stats import norm
import random
import pandas as pd
from SPSA import rbm2, mcsd


def mctc(driftvec, rep, h):
    totalsum = 0
    for i in range(rep):
        rbmpath = rbm2(driftvec, h)[1]
        totalsum = totalsum+sum(rbmpath)*h+sum((rbmpath-1)**2)
    expect = totalsum/rep
    return expect


summ = 0
for i in range(10):
    summ = summ + mctc([-7.361362529812827127e-01, -0.991546, -0.964050, -0.957955, -0.937242], 50, 1)
summ/10

# Wolfe step size for pde_comparison_mcgd
def wolfe8(mu, gradient, rep, h, dim):
    c1 = 0.3
    c2 = 0.7
    alpha = 0
    size = 1
    beta = math.inf
    stop_con = 0
    mu = np.asarray(mu)
    gradient = np.asarray(gradient)
    old_f = mctc(mu, rep, h)
    old_grad = gradient
    direction = old_grad/sum(np.square(old_grad))
    old_direc = np.dot(old_grad, direction)
    ite = 0
    while stop_con == 0:
        new_loc = mu-size*old_grad
        for n in range(len(new_loc)):
            if new_loc[n] >= 10:
                 new_loc[n] = 10
        new_f = mctc(new_loc, rep, h)
        deltamu = np.random.uniform(0, 1, int(dim / h))
        scale = np.random.uniform(0, 2, 1)[0]
        new_grad = (mctc(new_loc + scale * deltamu, rep, h) - mctc(new_loc - scale * deltamu, rep, h)) / (2 * scale * deltamu)
        new_direc = np.dot(new_grad, direction)
        if new_f > (old_f + c1*size*old_direc):
            beta = size
            size = (alpha+beta)/2
            ite = ite + 1
        elif new_direc < c2*old_direc:
            alpha = size
            if beta == math.inf:
                size = 2*alpha
            else:
                size = (alpha+beta)/2
            ite = ite + 1
        else:
            stop_con = 1
        if size > 10:
            size = 1
            stop_con = 1
        if ite > 30:
            stop_con = 1
            size = 1
    return size


def opt8(step, rep, dim, h):
    random.seed(np.random.uniform(0, 10000, 1)[0])
    mu = np.full(int(dim / h), 1)
    old_int = 1e100000
    new_int = mctc(mu, rep, h)
    count = 0
    results = [new_int]
    results = np.asarray(results)
    while count <= 100:
        deltamu = np.random.uniform(0, 1, int(dim/h) )
        scale = np.random.uniform(0, 2, 1)[0]
        grad = (mctc(mu+scale*deltamu, rep, h)-mctc(mu-scale*deltamu, rep, h))/(2*scale*deltamu)
        mu = mu-step*grad
        old_int = new_int
        new_int = mctc(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        if count % 10 == 0:
            print("the", count, "iteration:", new_int, "mu", mu, "\n")
        #print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    stdev = mcsd(mu, rep, h)
    print("the", count, "iteration:", new_int, "mu", mu, "\n")
    return [count, mu, new_int, results, stdev]


def opt88(rep, dim, h):
    random.seed(np.random.uniform(0, 10000, 1)[0])
    mu = np.full(int(dim / h), 1)
    old_int = 1e100000
    new_int = mctc(mu, rep, h)
    count = 0
    results = [0]
    step = 1
    results = np.asarray(results)
    while count <= 1000:
        deltamu = np.random.uniform(0, 1, int(dim/h))
        scale = np.random.uniform(0, 2, 1)[0]
        grad = (mctc(mu+scale*deltamu, rep, h)-mctc(mu-scale*deltamu, rep, h))/(2*scale*deltamu)
        if count%5 == 0:
            print("step is")
            step = wolfe8(mu, grad, rep, h, dim)
            print(step, "\n")
        mu = mu-step*grad
        old_int = new_int
        new_int = mctc(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        if count % 10 == 0:
            print("the", count, "iteration:", new_int, "mu", mu, "\n")
        print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    print("the", count, "iteration:", new_int, "mu", mu, "\n")
    return [count, mu, new_int, results]


'''
res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data1/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data1/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data2/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data3/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data4/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data5/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data6/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data7/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data8/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data9/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data10/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data11/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data11/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data12/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data12/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data13/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data13/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data14/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data14/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data15/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data15/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data16/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data16/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data17/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data17/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data18/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data18/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data19/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data19/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data20/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data20/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data21/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data21/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data22/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data22/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data23/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data23/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data24/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data24/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data25/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data25/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data26/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data26/opt8_value.csv", index=None)


res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data27/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data27/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data28/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data28/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data29/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data29/opt8_value.csv", index=None)

res8 = opt8(0.001, 50, 5, 1)
data = pd.DataFrame(res8).reset_index(drop=True)
data.to_csv("D:/purdue/RBM/Sim3/Python/data30/opt8.csv", index=None)
data = pd.DataFrame(res8[3]).reset_index(drop=True)
data.columns = ['SPSA_pde']
data.to_csv("D:/purdue/RBM/Sim3/Python/data30/opt8_value.csv", index=None)
'''
