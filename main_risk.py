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



def models(dataP, i_glob, train_size, b, h, c, alpha, epsilon, lam_true, UB,loss_opt,z_opt,all_eps,samp,ys, val=False):
    trainy = dataP[:train_size]
    valy = dataP[train_size:]
    trainy_mle = torch.cat((trainy, valy))
    prob = torch.log(torch.ones_like(trainy_mle)/len(trainy_mle))
    l_erm, z_erm = bisection_loss_count(b, h, c,  alpha, epsilon, prob, trainy_mle, trainy_mle, 0) 
    # l_erm, z_erm = cvx_erm(b, h, c,  alpha, prob, trainy_mle,0)
    # loss_erm_val = task_loss_emp( z_erm, b, h, c, alpha, lam_true)
    # Gamma=1.5
    # lw, zw = cvx_erm_cvarlb(b, h, c,  alpha, prob, trainy_mle, Gamma)
    # loss_RU_insamp = task_loss_emp( zw, b, h, c, alpha, lam_true,samp,ys)
    # ro_mean, z_robust_mean = cvx_robust_erm_micq(b, h, c,  alpha, trainy, int(1))
    # val_ro_mean = task_loss_emp(z_robust_mean, b, h, c, alpha,  lam_true,samp,ys)
    N_parts = np.linspace(1 , 200, 10, dtype=int)
    l_mom_in = np.zeros(len(N_parts),)
    z_mom_all = np.zeros(len(N_parts),)
    l_mom_out = np.zeros(len(N_parts),)
    
    for j, num_part in enumerate(N_parts):
        l_mom, z_mom = cvx_robust_erm_micq(b, h, c,  alpha,  trainy_mle, num_part)
        l_mom = task_loss_emp_insamp(z_mom, trainy_mle,  b, h, c, alpha, lam_true)
        l_mom_in[j,] = l_mom.item()
        z_mom_all[j,] = z_mom.item()
        l_mom_out[j,] = task_loss_emp(z_mom.item(), b, h, c, alpha,  lam_true,samp,ys).item()

    # l_wass_in = np.zeros(len(all_eps),)
    # l_wass_out = np.zeros(len(all_eps),)
    # z_wass_all = np.zeros(len(all_eps),)
    # for j, eps_wass in enumerate(all_eps):
    #     print(j, eps_wass) 
    #     l_wass, z_wass = wass_infty(b, h, c,  alpha, prob, trainy_mle, eps_wass)
    #     l_wass = task_loss_emp_insamp(z_wass, trainy_mle,  b, h, c, alpha, lam_true)
    #     z_wass_all[j] = z_wass.item()
    #     l_wass_in[j]=l_wass.item()
    #     l_wass_out[j,] =task_loss_emp(z_wass.item(), b, h, c, alpha,  lam_true,samp,ys).item()
    # l_wass_mom_in = np.zeros((len(all_eps),len(N_parts)))
    # l_wass_mom_out = np.zeros((len(all_eps),len(N_parts)))
    # z_wass_mom_all = np.zeros((len(all_eps),len(N_parts)))
    # for j, num_part in enumerate(N_parts):
    #     for i, eps_wass in enumerate(all_eps):
    #         l_wass_mom, z_wass_mom = robust_wass_infty(b, h, c,  alpha, prob, trainy_mle, eps_wass, num_part)
    #         l_wass_mom= task_loss_emp_insamp(z_wass_mom, trainy_mle,  b, h, c, alpha, lam_true)
    #         z_wass_mom_all[i,j] = z_wass_mom.item()
    #         l_wass_mom_in[i,j]=l_wass_mom
    #         l_wass_mom_out[i,j] =task_loss_emp(z_wass_mom.item(), b, h, c, alpha,  lam_true,samp,ys).item()
    # return i_glob,  l_erm.item(),loss_RU_insamp.item(),lw.item(),val_ro_mean.item(), ro_mean.item(), loss_opt.item(),\
    #  z_erm.item(), zw.item(), z_opt.item(), z_robust_mean.item(), l_wass_in, l_wass_out, z_wass_all, \
# l_mom_in, l_mom_out, z_mom_all, l_wass_mom_in, l_wass_mom_out, z_wass_mom_all,
    print(i_glob,l_erm.item(), l_mom_in, l_mom_out)
    return i_glob,l_erm.item(), l_mom_in, l_mom_out

def output(b, h,c, alpha, data_all, num_ins, train_size, epsilon, lam_true, UB,loss_opt,z_opt,all_eps,samp, ys, val):
    pool = mp.Pool()
    for i_g in range(num_ins):
        print(i_g)
        np.random.seed(42+i_g)  # Set the seed to a fixed value for reproducibility
        idx = np.random.randint(low=0, high=10000000, size=train_size)
        dataP= ys[idx]
        pool.apply_async(models, args=(dataP, i_g,\
         train_size, b, h, c, alpha, epsilon, lam_true, UB,loss_opt,z_opt,all_eps, samp,ys, val), callback=log_result) 
    pool.close()
    pool.join()
# def output(b, h,c, alpha, data_all, num_ins, train_size, epsilon, lam_true, UB, loss_opt,z_opt, all_eps,samp,ys, val):
#     for i_g in range(num_ins):
#         print(i_g)
#         np.random.seed(42+i_g)  # Set the seed to a fixed value for reproducibility
#         idx = np.random.randint(low=0, high=10000000, size=train_size)
#         dataP= ys[idx]
#         result = models(dataP, i_g, train_size, b, h, c, alpha, epsilon, lam_true, UB,  loss_opt,z_opt, all_eps,samp,ys,val)
#         log_result(result)
if __name__ == '__main__':
    b, h, c = 0.8, 0.3, 0.5
    b = torch.tensor(b)
    h = torch.tensor(h)
    c = torch.tensor(c)
    alpha = torch.tensor(1.)
    epsilon = 1e-4
    UB = torch.tensor(5.)
    trains = 2000
    size = trains
    train_size = int(0.7*trains)
    val_size = int(0.3*trains)
    num_ins = 10
    N_eps=4
    threshold=0.5
    N=int(trains/math.sqrt(trains))
    all_eps = torch.cat([torch.tensor(0.0).unsqueeze(0),torch.logspace(0.1,1, N_eps)])
    # int(trains/5)
    num_parts = np.linspace(N,N,1, dtype=int)
    #   num_parts = np.linspace(1,1,1)
    num_gammas=6#20
    Gammas = torch.logspace(0, 3, num_gammas)
    val =True
    # regs = torch.tensor([0.0])
    regs = torch.cat([torch.tensor(0.0).unsqueeze(0),torch.linspace(0.05, 1., 10)])
    global data_all
    data_all=[]
    lam_true = 0.9
    torch.manual_seed(42)
    samp = Dist.Exponential(lam_true)
#     ys = torch.rand(10000000)
    torch.manual_seed(42)
    ys = samp.sample(sample_shape=torch.Size([10000000]))
    prob = torch.ones_like(ys)/len(ys)
    # loss_opt, z_opt =    bisection_loss_count(b, h, c,  alpha, epsilon, torch.log(prob), ys,ys, 0)
    # cvx_erm(b, h, c,  alpha, prob, ys,0, True)
    # loss_opt = task_loss_emp( z_opt, b, h, c, alpha, lam_true,samp,ys)
    loss_opt=0
    z_opt=0
    output(b, h,c, alpha, data_all, num_ins, train_size, epsilon, lam_true, UB, loss_opt,z_opt,all_eps, samp,ys,val)
  
    data = list_result
    data = np.array(data[:], dtype=object)
    print(data)
    with open('coverage2000cp.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        # writer.writerow(['i_g', 'erm', 'RU_val', 'RU', 'Robust_val', 'Robust_mean', 'opt', 'z_erm', 'z_RU','z_opt','z_robust',\
        #                  'l_wass_in', 'l_wass_out', 'z_wass', 'l_mom_in', 'l_mom_out', 'z_mom',\
        #                 'l_wass_mom_in', 'l_wass_mom_out', 'z_was_mom'])
        writer.writerow(['i_g','erm', 'l_mom_in','l_mom_out'])
        for row in data:
            writer.writerow(row)
    with open('data2000cp.pkl', 'wb') as outfile: 
        pickle.dump(list_result, outfile, pickle.HIGHEST_PROTOCOL)

