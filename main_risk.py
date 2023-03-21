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
global list_result
global opt_result
list_result =[]
opt_result =[]
def log_optresult(result):
    opt_result.append(result)
def log_result(result):
    list_result.append(result)
def to_numpy(x):
    return np.array([np.round(x[i].item(),2) for i in range(len(x))])
def compute_meansx(x):
    return [torch.mean(x[i]).item() for i in range(len(x))]

def compute_means(x):
    return [np.mean(x[i]) for i in range(len(x))]


def models(data_all, i_glob, train_size, b, h, c, alpha, epsilon, lam_true, UB,loss_opt,z_opt, val=False):
    dataP = data_all[i_glob]
    trainy = dataP[:train_size]
    valy = dataP[train_size:]
    trainy_mle = torch.cat((trainy, valy))
    prob = torch.log(torch.ones_like(trainy_mle)/len(trainy_mle))
    # l_erm, z_erm = bisection_loss_count(b, h, c,  alpha, epsilon, prob, trainy_mle, trainy_mle, 0) 
    # l_erm, z_erm = cvx_erm(b, h, c,  alpha, prob, trainy_mle,0)
    # loss_erm_val = task_loss_emp( z_erm, b, h, c, alpha, lam_true)
    # Gamma=1.5
    # lw, zw = cvx_erm_cvarlb(b, h, c,  alpha, prob, trainy_mle, Gamma)
    # loss_RU_insamp = task_loss_emp( zw, b, h, c, alpha, lam_true)
    # ro_mean, z_robust_mean = cvx_robust_erm_micq(b, h, c,  alpha, trainy, int(1))
    # val_ro_mean = task_loss_emp(z_robust_mean, b, h, c, alpha,  lam_true)
    all_eps = torch.cat([torch.tensor(0.0).unsqueeze(0),torch.logspace(-3, 0.6, 10)])
    l_wass_in = np.zeros(len(all_eps),)
    l_wass_out = np.zeros(len(all_eps),)
    for j, eps_wass in enumerate(all_eps):
      print(j, eps_wass)
      l_wass, z_wass = wass_infty(b, h, c,  alpha, prob, trainy_mle, eps_wass)
      l_wass_in[j]=l_wass
      l_wass_out[j,] =task_loss_emp(z_wass.item(), b, h, c, alpha,  lam_true)


    # return i_glob,   loss_erm_val.item(), l_erm.item(),loss_RU_insamp.item(),lw.item(),val_ro_mean.item(), ro_mean.item(), loss_opt.item(),\
    #  z_erm.item(), zw.item(), z_opt.item(), z_robust_mean.item(), l_wass_in.item(), l_wass_out.item(), z_wass.item()
    return l_wass_in.item(), l_wass_out.item(), z_wass.item()
def output(b, h,c, alpha, data_all, num_ins, train_size, epsilon, lam_true, UB,loss_opt,z_opt, val):
    pool = mp.Pool()
    for i_g in range(num_ins):
        if i_g%20==0:
          print(i_g)
        pool.apply_async(models, args=(data_all, i_g,\
         train_size, b, h, c, alpha, epsilon, lam_true, UB,loss_opt,z_opt, val), callback=log_result) 
    pool.close()
    pool.join()
# def output(b, h,c, alpha, data_all, num_ins, train_size, epsilon, lam_true, UB, loss_opt,z_opt,  val):
#     for i_g in range(num_ins):
#         print(i_g)
#         result = models(data_all, i_g, train_size, b, h, c, alpha, epsilon, lam_true, UB,  loss_opt,z_opt, val)
#         log_result(result)
def find_rowcolumn(array, threshold):
    min_row_index = np.min(np.argwhere(np.max(array>threshold,1)))
    max_col_index = np.max(np.argwhere(array[min_row_index]>threshold))
    return min_row_index, max_col_index
if __name__ == '__main__':
    b, h, c = 3, 1, 2
    b = torch.tensor(b)
    h = torch.tensor(h)
    c = torch.tensor(c)
    alphas = torch.tensor([1])
    epsilon = 1e-4
    UB = torch.tensor(5.)
    trains= 100
    num_ins =10
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
