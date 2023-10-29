from scipy.stats import norm
import numpy as np
import math
import matplotlib.pyplot as plt


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
    BB = np.zeros(time + 1)
    for t in range(time):
        # scale down the drifts and dc according to h
        # randomnormal = np.random.normal(0, 1, 1)[0]*math.sqrt(h)
        # T1[t] = (-driftvec[t])*h + randomnormal
        T1[t] = np.random.normal((-driftvec[t])*h, math.sqrt(h), 1)[0]
        B[t + 1] = B[t] + T1[t]
        T2[t] = T1[t] / 2 + math.sqrt(T1[t] ** 2 - 2 * h * math.log(np.random.uniform(0, 1, 1)[0])) / 2
        M[t+1] = max(M[t], B[t]+T2[t])
        BB[t + 1] = BB[t] - T1[t]
        X[t+1] = M[t+1]-B[t+1]
    return [BB, X]


trial = rbm2(np.zeros(1000), 0.1)
print(trial[0])
print(trial[1])

plt.plot(trial[0], label="BM")
plt.plot(trial[1], label="RBM")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('t')
plt.ylabel('value')
plt.show

#
# def rbm2(driftvec, h):
#     """
#
#     :param driftvec:
#     :param h:
#     :return:
#     """
#     time = len(driftvec)
#     T1 = np.zeros(time)
#     T2 = np.zeros(time)
#     M = np.zeros(time+1)
#     X = np.zeros(time+1)
#     B = np.zeros(time+1)
#     BB = np.zeros(time + 1)
#     XX = np.zeros(time + 1)
#     for t in range(time):
#         # scale down the drifts and dc according to h
#         rando = np.random.normal(0, 1, 1)[0]
#         for i in range(100):
#             randoo = np.random.normal(0, 1, 1)[0]
#             if randoo < rando:
#                 rando = randoo
#         if rando > 0:
#             rando = 0
#         randomnormal = np.random.normal(0, 1, 1)[0]*h
#         T1[t] = (-driftvec[t])*h + randomnormal
#         B[t + 1] = B[t] + T1[t]
#         T2[t] = T1[t]/2+(T1[t]**2-2*math.log(np.random.uniform(0, 1, 1)[0]))**0.5/2
#         M[t+1] = max(M[t], B[t]+T2[t])
#         BB[t + 1] = BB[t] - T1[t]
#         X[t+1] = M[t+1]-B[t+1]
#         XX[t+1] = BB[t] - rando
#     return [BB, X, XX]
#
#
# trial = rbm2(np.zeros(100), 1)
#
# plt.plot(trial[0], label="BM")
# plt.plot(trial[1], label="RBM")
# plt.plot(trial[2], label="RBM_Euler")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.xlabel('t')
# plt.ylabel('value')
# plt.show
#
#
