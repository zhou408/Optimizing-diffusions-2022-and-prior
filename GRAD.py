import math
import numpy as np
from scipy.stats import norm
import random


# sample path using Asmussen,Glynn,Pitman
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


def gradvec(driftvec, h, N):
    Time = len(driftvec)
    gradvec = []
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
            gradi = gradi+incre
            # print(i, t, incre, "\n")
        gradvec.append(gradi)
    gradvec = np.asarray(gradvec)
    return gradvec


def gradvecgx(driftvec, h, N, dgdx):
    Time = len(driftvec)
    gradvec = []
    for i in range(1, Time+1):
        gradi = 0
        for t in range(1, Time+1):
            sample = rbm3(driftvec, i, t, h, N)
            if (i > 1) & (i < t) & (i != t):
                incre = grad1(sample, i, t, driftvec, h, N)
                if t == Time:
                    incre = incre + dgdx * incre
            elif (i == 1) & (i < t):
                incre = grad2(sample, i, t, driftvec, h, N)
                if t == Time:
                    incre = incre + dgdx * incre
            elif(i == 1) & (i == t):
                incre = grad3(sample, i, t, driftvec, h, N)
                if t == Time:
                    incre = incre + dgdx * incre
            elif (i > 1) & (i == t):
                incre = grad4(sample, i, t, driftvec, h, N)
                if t == Time:
                    incre = incre + dgdx * incre
            elif i > t:
                incre = 0
            else:
                incre = 0
            gradi = gradi+incre
            # print(i, t, incre, "\n")
        gradvec.append(gradi)
    vec = np.asarray(gradvec)
    return vec


# gradvec([-71.06204245, 3.39644043,  -4.71827213,  -5.48200689,  -3.20969803], 1, 30)

'''
sample = rbm3([-71.06204245, 3.39644043,  -4.71827213,  -5.48200689,  -3.20969803], 1, 2, 1, 30)
grad2(sample, 1, 2, [-71.06204245, 3.39644043,  -4.71827213,  -5.48200689,  -3.20969803], 1, 30)

[-3.08300550e+03, -3.06331901e+03,  1.00000000e+01, -6.44277059e+00, -2.99929979e+00]
gradvec([-3.08300550e+03, -3.06331901e+03,  1.00000000e+01, -6.44277059e+00, -2.99929979e+00], 1, 30)

sample = rbm3([-3.08300550e+03, -3.06331901e+03,  1.00000000e+01, -6.44277059e+00, -2.99929979e+00], 1, 2, 1, 30)
grad2(sample, 1, 2, [-3.08300550e+03, -3.06331901e+03,  1.00000000e+01, -6.44277059e+00, -2.99929979e+00], 1, 30)

mu = np.asarray([-9.85973925, -3.67029055, -3.45530343, -4.36389416, -4.12744128])-2*np.asarray([ 0.00237413,  0.00387692,  0.00274985, -0.00419346,  0.00183779])

# math range error when: y x mu = 5.266674270387739e-05, 2.1876712708035484e-06, -12320.668164297576

'''
