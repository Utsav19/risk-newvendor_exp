# write comments to the code below
# #
import csv
import torch
import torch.nn as nn
import torch.distributions as Dist
torch.set_default_tensor_type("torch.DoubleTensor")
import numpy as np
import pickle
import pandas as pd
import cvxpy as cp
import math
from multiprocessing import Pool
import multiprocessing as mp
def median_argmin(x):
    """
    Returns the median and the index of the median of a vector
    """
    x = np.array(x)
    idx = np.argsort(x)
    x = x[idx]
    med = np.median(x)
    if len(x) % 2 == 0:
        idx_med = idx[len(x) // 2]
    else:
        idx_med = idx[(len(x) + 1) // 2]
    return med, idx_med

def log_entropic_risk(y, z, b, h, c, alpha, prob): 
    loss = b*torch.maximum(y-z,torch.zeros_like(y)) + h*(torch.maximum(z-y,torch.zeros_like(y)))+c*z
    if alpha>0:
      return torch.exp(prob).T @ torch.exp(alpha*loss)# log(p'*exp(alpha*loss))
    else:
      return torch.exp(prob).T @ loss #p'*loss

def entropic_risk(y, z, b, h, c, alpha, prob): 
    loss = b*torch.maximum(y-z,torch.zeros_like(y)) + h*(torch.maximum(z-y,torch.zeros_like(y)))+c*z
    if alpha>0:
      return (torch.exp(prob).T @ torch.exp(alpha*loss))
    else:
      return (torch.exp(prob).T @ loss)

def cvx_erm(b, h, c,  alpha, prob, y, reg, opt=False):
  if opt:
    p=prob
  else:
    p = torch.ones(len(y))/len(y)   
  zb = cp.Variable((len(y), ))
  zh = cp.Variable((len(y), ))
  z = cp.Variable()
  t=cp.Variable()
  constraints = [z >= 0]
  constraints += [zh>=0, zb>=0]
  constraints += [zh>= z - y,  zb>=y-z]
  constraints+=[t>=p.T @ cp.exp(alpha*(b*zb+h*zh+c*z))]
  objective = cp.Minimize(t +reg*(cp.norm(z,2)**2))

  problem = cp.Problem(objective, constraints)
  assert problem.is_dpp()
  problem.solve(verbose=False)
  return torch.tensor(problem.value), torch.tensor(z.value)

def wass_infty(b, h, c,  alpha, prob, y, eps_wass):
  N=len(y)
  y_min=torch.maximum(y-eps_wass,torch.zeros_like(y))
  y_plus= y+eps_wass
  p = torch.ones(N)/N  
  zb_min = cp.Variable((N, ))
  zh_min = cp.Variable((N, ))
  zb_plus = cp.Variable((N, ))
  zh_plus = cp.Variable((N, ))
  z = cp.Variable()
  s=cp.Variable((N,))
  constraints = [z >= 0]
  constraints += [zh_plus>=0, zb_plus>=0]
  constraints += [zh_min>=0, zb_min>=0]
  constraints += [zh_min>= z - y_min,  zb_min>=y_min-z]
  constraints += [zh_plus>= z - y_plus,  zb_plus>=y_plus-z]
  constraints+=[s>=(1/N)*cp.exp(alpha*(b*zb_min+h*zh_min+c*z))]
  constraints+=[s>=(1/N)*cp.exp(alpha*(b*zb_plus+h*zh_plus+c*z))]
  objective = cp.Minimize(cp.sum(s))

  problem = cp.Problem(objective, constraints)
  assert problem.is_dpp()
  problem.solve(solver='MOSEK',verbose=False)#, abstol=1e-4, feastol=1e-6)
  return torch.tensor(problem.value), torch.tensor(z.value)




def cvx_robust_RU_micq(b, h, c,  alpha,  y, Gamma,num_part):
  le = int(len(y)/num_part)
  p= np.ones((le,))/le
  M=50
  phi = cp.Variable(num_part, integer=True)
  b=b.numpy()
  h=h.numpy()
  c=c.numpy()
  alpha = alpha.numpy()
  zb = cp.Variable((num_part,le))
  zh = cp.Variable((num_part,le))
  z = cp.Variable()
  t = cp.Variable()
  a = cp.Variable()
  za = cp.Variable((num_part,le))
  constraints = [z>=0, zb>=0, zh>=0]
  for i in  range(num_part):
      constraints += [t+M*phi[i]>=p.T @ ((1/Gamma)*cp.exp(alpha*(b*zb[i,:] + h*zh[i,:]+c*z))+(1-1/Gamma)*a)+\
            cp.sum(za[i,:])]
      constraints+=[za>=0, za[i,:]>=(Gamma-(1/Gamma))*(cp.multiply(p, cp.exp(alpha*(b*zb[i,:] + h*zh[i,:]+c*z)))-a)]
      constraints += [zb[i,:]>=y[i*le:(i+1)*le]-z]
      constraints += [zh[i,:]>=z-y[i*le:(i+1)*le] ]
  constraints += [phi>=0, phi<=1, cp.sum(phi)==math.ceil(num_part/2)-1]
  objective = cp.Minimize(t)
  problem = cp.Problem(objective, constraints)
  # assert problem.is_dpp()
  problem.solve(verbose=False)
  return torch.tensor(problem.value),torch.tensor(z.value)


def cvx_robust_erm_micq(b, h, c,  alpha,  y,num_part):
  le = int(len(y)/num_part)
  M=50
  phi = cp.Variable(num_part, integer=True)
  b=b.numpy()
  h=h.numpy()
  c=c.numpy()
  alpha = alpha.numpy()
  w = cp.Variable((num_part,le))
  zb = cp.Variable((num_part,le))
  zh = cp.Variable((num_part,le))
  z = cp.Variable()
  t = cp.Variable()
  constraints = [z>=0, zb>=0, zh>=0]
  for i in  range(num_part):
      constraints += [t+M*phi[i]>=cp.sum(w[i,:])/le]
      constraints += [zb[i,:]>=y[i*le:(i+1)*le]-z]
      constraints += [zh[i,:]>=z-y[i*le:(i+1)*le] ]
      constraints += [w[i,:]>= cp.exp(alpha*(c*z+b*zb[i,:]+h*zh[i,:])) ]
  constraints += [phi>=0, phi<=1, cp.sum(phi)==math.ceil(num_part/2)-1]
  objective = cp.Minimize(t)
  problem = cp.Problem(objective, constraints)
  # assert problem.is_dpp()
  problem.solve(verbose=False)
  return torch.tensor(problem.value),torch.tensor(z.value)

def cvx_robust_erm(b, h, c,  alpha,  y, num_part):
  zs = np.linspace(0,3,50)
  a = np.zeros(len(zs),)
  for kk, z1 in enumerate(zs):
    z=z1
    le = int(len(y)/num_part)
    v =  cp.Variable((num_part,le))
    t = cp.Variable()
    phi = cp.Variable(num_part, integer=True)
    constraints= [phi>=0, phi<=1, cp.sum(phi)==int(num_part/2)+1]
    w=np.zeros((num_part,le))
    for i in range(num_part):
      w[i,:]=np.maximum(alpha*(c*z+b*y[i*le:(i+1)*le] -b*z),\
      alpha*(c*z-h*y[i*le:(i+1)*le] +h*z))
      constraints += [t>=cp.sum(v[i,:])/le]
      for j in range(le):
        constraints+= [phi[i]*np.exp(w[i,j])<= v[i,j]]
    objective = cp.Minimize(t)
    problem = cp.Problem(objective, constraints)
    # assert problem.is_dpp()
    problem.solve(solver='MOSEK',verbose=False)
    a[kk]=problem.value
  z_opt = zs[np.argmin(a)]
  return torch.tensor(problem.value), torch.tensor(z_opt)
def cvx_erm_cvarlb(b, h, c,  alpha, prob, y, Gamma, RU=1):
    p = torch.exp(prob)
    zb = cp.Variable((len(y), ))
    zh = cp.Variable((len(y), ))
    za = cp.Variable((len(y), ))
    z = cp.Variable()
    a = cp.Variable()
    constraints = [z >= 0]
    constraints += [zh>=0, zb>=0]
    #CVaR with lowerbound wager
    if RU==1:
        if alpha>0:
            constraints += [zh>= z - y,  zb>=y-z, za>=0, za>=(Gamma-(1/Gamma))*(cp.multiply(p, cp.exp(alpha*(b*zb + h*zh+c*z)))-a)]
            objective = cp.Minimize(p.T @ ((1/Gamma)*cp.exp(alpha*(b*zb + h*zh+c*z))+(1-1/Gamma)*a)+\
            cp.sum(za))
        else:
            constraints += [zh>= z - y,  zb>=y-z, za>=0, za>=(Gamma-(1/Gamma))*(cp.multiply(p, (b*zb + h*zh+c*z))-a)]
            objective = cp.Minimize(p.T @ ((1/Gamma)*(b*zb + h*zh+c*z)+(1-1/Gamma)*a)+\
            cp.sum(za))

    #CVAR a + Gamma * sum_i p_i(exp(alpha*loss - a)_+) = a +  * sum_i (p_i*Gamma*(exp(alpha*loss - a)_+))
    else:
        if alpha>0:
            constraints += [zh>= z - y,  zb>=y-z, za>=0, za>=Gamma*(cp.multiply(p, cp.exp(alpha*(b*zb + h*zh+c*z)))-a)]
            objective = cp.Minimize(a+cp.sum(za))
        else:
            constraints += [zh>= z - y,  zb>=y-z, za>=0, za>=Gamma*(cp.multiply(p, (b*zb + h*zh+c*z))-a)]
            objective = cp.Minimize(a+cp.sum(za))

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()
    problem.solve(solver ='MOSEK', verbose=False)
    # print("Gamma", Gamma)
    return torch.tensor(problem.value), torch.tensor(z.value)
def bisection_loss_count(b, h, c,  alpha, epsilon, prob, y, ys, reg=1, prob_mle=1., pred=False, rudin=False):
    lb = torch.tensor(0)  
    ub = torch.tensor(100)
    grad1=torch.tensor(1)
    it = 5
    if pred:
      f = lambda z: torch.log((1-reg)*entropic_risk(y, z, b,h, c, alpha, prob)\
         + reg*entropic_risk(ys, z, b, h, c, alpha, prob_mle))
    elif rudin:
      f = lambda z: log_entropic_risk(y, z, b,h, c, alpha, prob) + reg*(torch.norm(z,2)**2)
    else:
      f = lambda z: log_entropic_risk(y, z, b,h, c, alpha, prob) + 1e-4*(z**2)
    while (torch.abs(grad1) > epsilon and (it<1e4 and torch.abs(ub-lb)>1e-4*epsilon)):
      z0 = (lb+ub)/2.0
      grad1 = (f(z0+epsilon)-f(z0))/epsilon
      if grad1 < 0: 
        lb = z0
      else:
        ub = z0
      it+=1
    return  f(z0), z0


def task_loss(ys, zopt,  b, h, c, alpha,  lam_true):
    dist = Dist.Poisson(lam_true)
    m = nn.LogSoftmax(dim=0)
    prob = m(dist.log_prob(ys))
    c_out = log_entropic_risk(ys, zopt, b, h, c, alpha, prob)
    return c_out
   
def task_loss_emp(zopt,  b, h, c, alpha, lam_true):
    samp = Dist.Exponential(lam_true)
    ys = torch.rand(int(1e8))
    # samp.sample(sample_shape=torch.Size([int(1e6)]))
    prob = torch.log(torch.ones_like(ys)/len(ys))
    c_out = log_entropic_risk(ys, zopt, b, h, c, alpha, prob)
    return c_out


def e2e_reg_rudin(reg,  b, h, c, alpha, zs, y):
  loss_e2e = torch.zeros(len(zs),)
  prob = torch.log(torch.ones_like(y)/len(y))
  for j, z in enumerate(zs):   
    loss_e2e[j,]=torch.log(entropic_risk(y, z, b, h, c, alpha, prob)\
        + reg*torch.norm(z,1))
  idx = loss_e2e.argmin()
  return zs[idx], loss_e2e

def mle(trainy, means, ys):
    mle_loss = torch.zeros(len(means), )
    for j, lam in enumerate(means):
          dist= Dist.Poisson(lam)
    logprob = dist.log_prob(trainy) - torch.logsumexp(dist.log_prob(ys),dim=0) # sum_{i} log(Poi(y)/[sum_{i<=UB} Poi(ys_i)]
    # logprob = dist.log_prob(trainy)
    mle_loss[j,] = -logprob.mean() 
    # f(x) = p*(1_x==0)+(1-p)*Poi(x)
    js = mle_loss.argmin()
    return means[js]

