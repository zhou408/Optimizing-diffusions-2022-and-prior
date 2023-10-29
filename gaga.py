import scipy.integrate as integrate
from scipy.stats import norm
import numpy as np
import math
import matplotlib.pyplot as plt

#result = integrate.quad(lambda x: norm.cdf((-x+5*(-10))/math.sqrt(5))-math.exp(2*(-10)*x)*norm.cdf((-x-5*(-10))/math.sqrt(5)), 0, np.inf)

#norm.cdf((-x+5*mu)/math.sqrt(5))-math.exp(2*mu*x)*norm.cdf((-x-5*mu)/math.sqrt(5))

#result = integrate.quad(lambda x: norm.cdf((-x+5*(-20))/math.sqrt(5))-math.exp(2*(-20)*x)*norm.cdf((-x-5*(-20))/math.sqrt(5)), 0, np.inf)

#result = integrate.quad(lambda x: norm.cdf((-x+5*(-15))/math.sqrt(5))-math.exp(2*(-15)*x)*norm.cdf((-x-5*(-15))/math.sqrt(5)), 0, np.inf)


def F(X):
    res = np.zeros(len(X))
    for i in range(len(X)):
        t = X[i]
        y1 = integrate.quad(lambda x: norm.cdf((-x+t*(-10))/math.sqrt(t))-math.exp(2*(-10)*x)*norm.cdf((-x-t*(-10))/math.sqrt(t)), 0, np.inf)[0]
        y2 = integrate.quad(lambda x: norm.cdf((-x+t*(-20))/math.sqrt(t))-math.exp(2*(-20)*x)*norm.cdf((-x-t*(-20))/math.sqrt(t)), 0, np.inf)[0]
        y3 = integrate.quad(lambda x:norm.cdf((-x+t*(-15))/math.sqrt(t))-math.exp(2*(-15)*x)*norm.cdf((-x-t*(-15))/math.sqrt(t)), 0, np.inf)[0]
        res[i] = 0.5*(y1+y2)-y3
    return res


X = np.arange(0.01, 1, 0.01)
plt.plot(X, F(X))
##########################################################################################################################################

def F(X):
    res = np.zeros(len(X))
    y4 = integrate.quad(lambda x: norm.cdf((-x + 1 * (-100)) / math.sqrt(1)) - math.exp(2 * (-100) * x) * norm.cdf(
        (-x - 1 * (-100)) / math.sqrt(1)), 0, np.inf)[0]
    y5 = integrate.quad(lambda x: norm.cdf((-x + 1 * (-200)) / math.sqrt(1)) - math.exp(2 * (-200) * x) * norm.cdf(
        (-x - 1 * (-200)) / math.sqrt(1)), 0, np.inf)[0]
    y6 = integrate.quad(lambda x: norm.cdf((-x + 1 * (-150)) / math.sqrt(1)) - math.exp(2 * (-150) * x) * norm.cdf(
        (-x - 1 * (-150)) / math.sqrt(1)), 0, np.inf)[0]
    yy = 3*(0.5*(y4+y5)-y6)
    for i in range(len(X)):
        t = X[i]
        y1 = integrate.quad(lambda x: norm.cdf((-x+t*(-100))/math.sqrt(t))-math.exp(2*(-100)*x)*norm.cdf((-x-t*(-100))/math.sqrt(t)), 0, np.inf)[0]
        y2 = integrate.quad(lambda x: norm.cdf((-x+t*(-200))/math.sqrt(t))-math.exp(2*(-200)*x)*norm.cdf((-x-t*(-200))/math.sqrt(t)), 0, np.inf)[0]
        y3 = integrate.quad(lambda x: norm.cdf((-x+t*(-150))/math.sqrt(t))-math.exp(2*(-150)*x)*norm.cdf((-x-t*(-150))/math.sqrt(t)), 0, np.inf)[0]
        res[i] = 0.5*(y1+y2)-yy
    return res

plt.rc('font', size=14)
X = np.arange(0.01, 1, 0.01)
plt.plot(X, F(X))
plt.xlabel('t')
plt.ylabel('integrand')
plt.title('integrand value vs. t')
##########################################################################################################################################

def F(X):
    res = np.zeros(len(X))
    for i in range(len(X)):
        t = X[i]
        y = integrate.quad(lambda x: 0.5*(norm.cdf((-x+t*(-1))/math.sqrt(t))-math.exp(2*(-1)*x)*norm.cdf((-x-t*(-1))/math.sqrt(t))+norm.cdf((-x+t*(-2))/math.sqrt(t))-math.exp(2*(-2)*x)*norm.cdf((-x-t*(-2))/math.sqrt(t))) - norm.cdf((-x+t*(-1.5))/math.sqrt(t))-math.exp(2*(-1.5)*x)*norm.cdf((-x-t*(-1.5))/math.sqrt(t)), 0, np.inf)[0]
        res[i] = y
    return res


X = np.arange(9.01, 10, 0.01)
plt.plot(X, F(X))


def F(X):
    res = np.zeros(len(X))
    for i in range(len(X)):
        t = X[i]
        y = integrate.quad(lambda x: 0.05*(norm.cdf((-x+t*(0.01))/math.sqrt(t))-math.exp(2*(0.01)*x)*norm.cdf((-x-t*(0.01))/math.sqrt(t))+norm.cdf((-x+t*(0.02))/math.sqrt(t))-math.exp(2*(0.02)*x)*norm.cdf((-x-t*(0.02))/math.sqrt(t))) - norm.cdf((-x+t*(0.015))/math.sqrt(t))-math.exp(2*(0.015)*x)*norm.cdf((-x-t*(0.015))/math.sqrt(t)), 0, np.inf)[0]
        res[i] = y
    return res

X = np.arange(0.01, 1, 0.01)
plt.plot(X, F(X))
