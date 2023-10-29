import math
import numpy as np
from scipy.stats import norm
import random
from SPSA import mc2, mcsd, mcgx
from GRAD import gradvec, gradvecgx

def wolfe(mu, gradient, rep, h):
    c1 = 0.3
    c2 = 0.7
    alpha = 0
    size = 1
    beta = math.inf
    stop_con = 0
    mu = np.asarray(mu)
    gradient = np.asarray(gradient)
    old_f = mc2(mu, rep, h)
    old_grad = gradient
    direction = old_grad/sum(np.square(old_grad))
    old_direc = np.dot(old_grad, direction)
    ite = 0
    while stop_con == 0:
      new_loc = mu-size*old_grad
      for n in range(len(new_loc)):
        if new_loc[n] >= 10:
          new_loc[n] = 10
      new_f = mc2(new_loc, rep, h)
      new_grad = gradvec(new_loc, h, rep)
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
      if ite > 100:
        stop_con = 1
        size = 1
    return size

def wolfe1():
    c1 = 0.3
    c2 = 0.7
    alpha = 0
    size = 1
    beta = math.inf
    stop_con = 0
    mu = np.asarray(mu)
    gradient = np.asarray(gradient)
    old_f = mcgx(mu, rep, h)
    old_grad = gradient
    direction = old_grad / sum(np.square(old_grad))
    old_direc = np.dot(old_grad, direction)
    ite = 0
    while stop_con == 0:
      new_loc = mu - size * old_grad
      for n in range(len(new_loc)):
        if new_loc[n] >= 10:
          new_loc[n] = 10
      new_f = mcgx(new_loc, rep, h)
      new_grad = gradvecgx(new_loc, h, rep, -3)
      new_direc = np.dot(new_grad, direction)
      if new_f > (old_f + c1 * size * old_direc):
        beta = size
        size = (alpha + beta) / 2
        ite = ite + 1
      elif new_direc < c2 * old_direc:
        alpha = size
        if beta == math.inf:
          size = 2 * alpha
        else:
          size = (alpha + beta) / 2
        ite = ite + 1
      else:
        stop_con = 1
      if ite > 100:
        stop_con = 1
        size = 1
    return size
