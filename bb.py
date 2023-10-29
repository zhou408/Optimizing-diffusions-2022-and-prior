import math
import numpy as np
from scipy.stats import norm
import random
import pandas as pd


def rbm2(driftvec, h):
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


def mctc(driftvec, rep, h):
    totalsum = 0
    for i in range(rep):
        rbmpath = rbm2(driftvec, h)[1]
        totalsum = totalsum+sum(rbmpath)*h+sum((rbmpath-1)**2)
    expect = totalsum/rep
    return expect



def rbm3(driftvec, j, t, h, N):
    Time = len(driftvec)
    T1_list = []
    T2_list = []
    M_list = [np.zeros(1)]
    B_list = [np.zeros(1)]
    X_list = [np.zeros(1)]
    samplen = N
    for i in range(Time):
        old_samplen = samplen
        if (i+1) == t:
            samplen = samplen*N
        if ((i+1) == 1) & ((i+1) == t):
            samplen = int(samplen/N)
        T1 = np.zeros(samplen)
        T2 = np.zeros(samplen)
        M = np.zeros(samplen+1)
        B = np.zeros(samplen+1)
        X = np.zeros(samplen+1)
        # print(M_list)
        last_M = M_list[-1]
        last_B = B_list[-1]
        last_X = X_list[-1]
        # print("last_X is",last_X)
        for n in range(samplen):
            # label = int(samplen/N - 1)
            if i == 0:
                label = 0
            elif old_samplen == samplen:
                label = int(n)
            else:
                label = math.floor(n/old_samplen)
            # cale down the drifts and dc according to h
            # T1[n] = (-driftvec[t-1])*h+np.random.normal(0, 1, 1)[0]*h
            T1[n] = (-driftvec[i]) * h + np.random.normal(0, 1, 1)[0] * h
            T2[n] = T1[n]/2+math.sqrt(T1[n]**2-2*math.log(np.random.uniform(0, 1, 1)[0]))/2
            # print(label, last_B, last_M)
            M[n+1] = max(last_M[label], last_B[label]+T2[n])
            B[n+1] = last_B[label]+T1[n]
            X[n+1] = M[n+1]-B[n+1]
        # print(T1, X)
        T1_list.append(T1)
        T2_list.append(T2)
        M_list.append(M)
        B_list.append(B)
        X_list.append(X)
    return X_list


def ex(time, driftvec, rep, h):
    totalsum = 0
    for i in range(rep):
        rbmpath = rbm2(driftvec, h)[1]
        totalsum = totalsum + rbmpath[time-1]
    expect = totalsum / rep
    return expect


# if(1<i<t)
def grad1(sample, i, t, driftvec, h, N):
    total1 = 0
    tracker = 1
    for n in range(N):
        total3 = 0
        for m in range(N):
            now3 = sample[t][tracker]
            total3 = total3+now3
            tracker = tracker+1
            #print(tracker)
        # print(length(sample[[i]]),length(sample[[i+1]]))
        y = sample[i-1][n+1]
        x = sample[i][n+1]
        mu = driftvec[i-1]
        term1 = (-(-x+y+mu*h))/ math.sqrt(h)*norm.pdf((-x+y+mu*h)/ math.sqrt(h))-(2* math.exp(2*mu*x)+4*mu*x* math.exp(2*mu*x))*norm.cdf((-x-y-mu*h)/ math.sqrt(h))- math.exp(2*mu*x)*(-(-x-y-mu*h)/ math.sqrt(h)*norm.pdf((-x-y-mu*h)/ math.sqrt(h),0,1))
        term2 = (2*mu* math.exp(2*mu*x)* math.sqrt(h)+2*x* math.exp(2*mu*x)/ math.sqrt(h))*norm.pdf((-x-y-mu*h)/ math.sqrt(h))
        term3 = norm.pdf((-x+y+mu*h)/ math.sqrt(h))/ math.sqrt(h)-2*mu* math.exp(2*mu*x)*norm.cdf((-x-y-mu*h)/ math.sqrt(h))+ math.exp(2*mu*x)*norm.pdf((-x-y-mu*h)/ math.sqrt(h))/ math.sqrt(h)
        dlogpdmu = (term1+term2)/term3
        # print(y, x, term1, term2, term3, dlogpdmu, "\n")
        if math.isnan(dlogpdmu):
            dlogpdmu = 1
        now1 = total3/N*dlogpdmu
        total1 = total1+now1
    gradest = total1/N
    return gradest



#if(1=i<t)
def grad2(sample, i, t, driftvec, h, N):
    total1 = 0
    tracker = 1
    for j in range(N):
        total2 = 0
        for n in range(N):
            now2 = sample[t][tracker]
            tracker = tracker+1
            total2 = total2+now2
            #print(tracker)
        y = 0
        x = sample[i][j+1]
        #print(y,x)
        mu = driftvec[i-1]
        term1 = (-(-x+y+mu*h))/ math.sqrt(h)*norm.pdf((-x+y+mu*h)/ math.sqrt(h))-(2* math.exp(2*mu*x)+4*mu*x* math.exp(2*mu*x))*norm.cdf((-x-y-mu*h)/ math.sqrt(h))- math.exp(2*mu*x)*(-(-x-y-mu*h)/ math.sqrt(h)*norm.pdf((-x-y-mu*h)/ math.sqrt(h)))
        term2 = (2*mu* math.exp(2*mu*x)* math.sqrt(h)+2*x* math.exp(2*mu*x)/ math.sqrt(h))*norm.pdf((-x-y-mu*h)/ math.sqrt(h))
        term3 = norm.pdf((-x+y+mu*h)/ math.sqrt(h))/ math.sqrt(h)-2*mu* math.exp(2*mu*x)*norm.cdf((-x-y-mu*h)/ math.sqrt(h))+ math.exp(2*mu*x)*norm.pdf((-x-y-mu*h)/ math.sqrt(h))/ math.sqrt(h)
        dlogpdmu = (term1+term2)/term3
        if math.isnan(dlogpdmu):
            dlogpdmu = 1
        # print(y, x, term1, term2, term3, dlogpdmu)
        now1 = total2/N*dlogpdmu
        total1 = total1+now1
    gradest = total1/N
    return gradest


# if(1=i=t)
def grad3(sample, i, t, driftvec, h, N):
    total1 = 0
    for j in range(N):
        y = 0
        x = sample[i][j+1]
        # print(y,x)
        mu = driftvec[i-1]
        term1 = (-(-x+y+mu*h))/ math.sqrt(h)*norm.pdf((-x+y+mu*h)/ math.sqrt(h))-(2* math.exp(2*mu*x)+4*mu*x* math.exp(2*mu*x))*norm.cdf((-x-y-mu*h)/ math.sqrt(h))- math.exp(2*mu*x)*(-(-x-y-mu*h)/ math.sqrt(h)*norm.pdf((-x-y-mu*h)/ math.sqrt(h)))
        term2 = (2*mu* math.exp(2*mu*x)* math.sqrt(h)+2*x* math.exp(2*mu*x)/ math.sqrt(h))*norm.pdf((-x-y-mu*h)/ math.sqrt(h))
        term3 = norm.pdf((-x+y+mu*h)/ math.sqrt(h))/ math.sqrt(h)-2*mu* math.exp(2*mu*x)*norm.cdf((-x-y-mu*h)/ math.sqrt(h))+ math.exp(2*mu*x)*norm.pdf((-x-y-mu*h)/ math.sqrt(h))/ math.sqrt(h)
        dlogpdmu = (term1+term2)/term3
        if math.isnan(dlogpdmu):
            dlogpdmu = 1
        now1 = dlogpdmu*x
        total1 = total1+now1
        # print(total1)
    gradest = total1/N
    return gradest


# if(1<i=t)
def grad4(sample, i, t, driftvec, h, N):
    total1 = 0
    for j in range(N):
        y = sample[i-1][j+1]
        x = sample[i][j+1]
        # print(y,x)
        mu = driftvec[i-1]
        # print(y, x, mu)
        term1 = (-(-x+y+mu*h))/ math.sqrt(h)*norm.pdf((-x+y+mu*h)/ math.sqrt(h))-(2* math.exp(2*mu*x)+4*mu*x* math.exp(2*mu*x))*norm.cdf((-x-y-mu*h)/ math.sqrt(h))- math.exp(2*mu*x)*(-(-x-y-mu*h)/ math.sqrt(h)*norm.pdf((-x-y-mu*h)/ math.sqrt(h)))
        term2 = (2*mu* math.exp(2*mu*x)* math.sqrt(h)+2*x* math.exp(2*mu*x)/ math.sqrt(h))*norm.pdf((-x-y-mu*h)/ math.sqrt(h))
        term3 = norm.pdf((-x+y+mu*h)/ math.sqrt(h))/ math.sqrt(h)-2*mu* math.exp(2*mu*x)*norm.cdf((-x-y-mu*h)/ math.sqrt(h))+ math.exp(2*mu*x)*norm.pdf((-x-y-mu*h)/ math.sqrt(h))/ math.sqrt(h)
        dlogpdmu = (term1+term2)/term3
        if math.isnan(dlogpdmu):
            dlogpdmu = 1
        now1 = dlogpdmu*x
        # print(dlogpdmu,x)
        total1 = total1+now1
        # print(total1)
    gradest = total1/(N**2)
    return gradest


def gradvec1(driftvec, h, N):
    Time = len(driftvec)
    vec = []
    for i in range(1, Time+1):
        gradi = 0
        for t in range(1, Time+1):
            sample = rbm3(driftvec, i, t, h, N)
            if (i > 1) & (i < t)&(i != t):
                incre = grad1(sample, i, t, driftvec, h, N)
            elif (i == 1) & (i < t):
                incre = grad2(sample, i, t, driftvec, h, N)
            elif(i == 1) & (i == t):
                incre = grad3(sample, i, t, driftvec, h, N)
            elif (i > 1) & (i == t):
                incre = grad4(sample, i, t, driftvec, h, N)
            elif i > t:
                incre = 0
            else:
                incre = 0
            gradi = gradi + incre
            # print(i, t, incre, "\n")
        expect = ex(i, driftvec, 50, h)
        gradi = gradi * (h - 2 + 2 * expect)
        vec.append(gradi)
    vec = np.asarray(vec)
    return vec


# Wolfe step size for pde_comparison_mcgd
def wolfe9(mu, gradient, rep, h):
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
        new_grad = gradvec1(new_loc, h, rep)
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


# with fixed step size
def opt9(step, rep, dim, h):
    random.seed(np.random.uniform(0, 10000, 1)[0])
    mu = np.full(int(dim/h), 1)
    old_int = 1e100
    new_int = mctc(mu, rep, h)
    count = 0
    results = [0]
    results = np.asarray(results)
    while count <= 3000:
        grad = gradvec1(mu, h, rep)
        mu = mu-step*grad
        for n in range(len(mu)):
            if mu[n] >= 10:
                mu[n] = 10
            if mu[n] <= -10000:
                mu[n] = -10000
        old_int = new_int
        new_int = mctc(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        if count % 5 == 0:
            print("the", count, "iteration:", new_int, "\n")
        print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    print("the", count, "iteration:", new_int, "mu", mu, "\n")
    return [count, mu, new_int, results]



# with fixed wolfe
def opt99(rep, dim, h):
    random.seed(np.random.uniform(0, 10000, 1)[0])
    mu = np.full(int(dim/h), 1)
    old_int = 1e100
    new_int = mctc(mu, rep, h)
    count = 0
    results = [0]
    results = np.asarray(results)
    step = 1
    while count <= 5000:
        grad = gradvec1(mu, h, rep)
        if count % 5 == 0:
            print("step is")
            step = wolfe9(mu, grad, rep, h)
            print(step, "\n")
        mu = mu-step*grad
        for n in range(len(mu)):
            if mu[n] >= 10:
                mu[n] = 10
            if mu[n] <= -10000:
                mu[n] = -10000
        old_int = new_int
        new_int = mctc(mu, rep, h)
        count = count+1
        results = np.append(results, new_int)
        if count % 5 == 0:
            print("the", count, "iteration:", new_int, "\n")
        print("the", count, "iteration:", 'mu is', mu, 'grad is', grad, '\n')
    print("the", count, "iteration:", new_int, "mu", mu, "\n")
    return [count, mu, new_int, results]


res99 = opt99(50, 5, 1)
data = pd.DataFrame(res99).reset_index(drop=True)
data.to_csv("/home/zihe/opt9_5000_Wolfe.csv", index=None)

data = pd.DataFrame(res99[3]).reset_index(drop=True)
data.columns = ['MCGD_pde_Wolfe']
data.to_csv("/home/zihe/opt9_5000(value)_Wolfe.csv", index=None)