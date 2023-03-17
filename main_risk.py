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
from functions_outsamrisk import *
if __name__ == '__main__':
    b, h, c = 5, 1, 2
    b = torch.tensor(b)
    h = torch.tensor(h)
    c = torch.tensor(c)
    alphas = torch.tensor([1])
    epsilon = 1e-4
    UB = torch.tensor(5.)
    trains= 10
    num_ins =1
    threshold=0.5
    N=int(trains/math.sqrt(trains))
    # int(trains/5)
    num_parts = np.linspace(N,N,1, dtype=int)
    #   num_parts = np.linspace(1,1,1)
    num_gammas=6#20
    Gammas = torch.logspace(0, 3, num_gammas)
    val =True
    # regs = torch.tensor([0.0])
    regs = torch.cat([torch.tensor(0.0).unsqueeze(0),torch.logspace(-3, 2., 10)])
    global data_all
    data_all=[]
    torch.tensor(0.5)
    lam_true = 0.9
    samp = Dist.Poisson(lam_true)
    train_size = int(0.7*trains)
    val_size = int(0.3*trains)
    size = train_size+val_size
    alpha=alphas[0]
    samp = Dist.Exponential(lam_true)
    ys = torch.rand(int(1e6))
    # ys = samp.sample(sample_shape=torch.Size([int(1e4)]))
    prob = torch.log(torch.ones_like(ys)/len(ys))
    loss_opt, z_opt = bisection_loss_count(b, h, c,  alpha, epsilon, prob, ys,ys, 0)  
    # cvx_erm(b, h, c,  alpha, prob, ys,0, True)
    loss_opt = task_loss_emp( z_opt, b, h, c, alpha, lam_true)
    for it1 in range(trains):
        dataP= torch.rand(size)
        #samp.sample(sample_shape=torch.Size([size]))
        data_all.append(dataP)
    output(b, h,c, alpha, data_all, num_ins, train_size, epsilon, lam_true, UB, loss_opt,z_opt, val)
  
    data = list_result
    data = np.array(data[:], dtype=object)
    with open('coverage200.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['i_g', 'erm_val', 'erm', 'RU_val', 'RU', 'Robust_val', 'Robust_mean', 'opt', 'z_erm', 'z_RU','z_opt','z_robust',\
                         'l_wass_in', 'l_wass_out', 'z_wass'])
        for row in data:
            writer.writerow(row)
