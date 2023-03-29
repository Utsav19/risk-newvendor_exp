# import csv

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
# %autoreload 2
# %load_ext autoreload


with open('data2000c.pkl', 'rb') as infile:
    list_result = pickle.load(infile)

data = list_result
data = np.array(data, dtype=object)
erm = np.stack(data[:,1],axis=0)
mom_in = np.stack(data[:,2],axis=0)
mom_out = np.stack(data[:,3],axis=0)
# print(data[:,1])
# move all files 
print(pd.DataFrame(mom_in>mom_out).mean(0))
# wass_in = np.stack(data[:,-9],axis=0)
# wass_out = np.stack(data[:,-8],axis=0)
# pd.DataFrame(wass_in>wass_out).mean(0)
# wmom_i = np.stack(data[:,-3],axis=0)
# wmom_o = np.stack(data[:,-2],axis=0)
# print(((mom_in>mom_out).mean(0)).T)
# print(wass_in-wass_out)
# print(pd.DataFrame(wass_in>wass_out).mean(0).T)
# a =0
# for i in range(wmom_i.shape[2]):
#     mat = pd.DataFrame(wmom_i[:,:,i]>wmom_o[:,:,i]).mean(0).values
#     x = mat>0.55
#     a=a+np.array([int(x1) for x1 in x])
# # print(a)
# idx = np.argwhere(a==wmom_i.shape[2])[0][0]
# print("a",a)
# print(wmom_i[:,:,idx].mean(0))
# print(pd.DataFrame(mom_in>mom_out).mean(0))
