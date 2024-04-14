# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:12:38 2024

@author: hp
"""
import numpy as np
import pandas as pd
import os
import scipy.io as scio
import statsmodels.api as sm
feature = np.load("DELFI_short.npy") #Same with DELFI_long
region = pd.read_excel("/GRC37_DELFI_region_100kb.xlsx",header=None)
region = np.array(region)
feature_gc = np.zeros([feature.shape[0],feature.shape[1]])
lowess = sm.nonparametric.lowess

for i in range(feature.shape[1]):
    print(i)
    pos = 0
    
    for j in range(1,23):
        mat = ''.join(['chr',str(j),'.mat'])
        mat2 = os.path.join("/GC/",mat)
        mat3 = scio.loadmat(mat2)   #处理.mat文件
        gc = mat3['loc']
        
        reg = region[region[:,0]==j]
        n = reg.shape[0]
        x = np.zeros([n,1])
        y = np.zeros([n,1])
        res = np.zeros([n,1])
        
        for k in range(n):
            st = reg[k,1]
            end = reg[k,2]
            gc_mean = np.average(gc[(st-1):end,:])
            
            x[k,0] = gc_mean
            y[k,0] = feature[pos+k,i]
           
        x = x.flatten()
        y = y.flatten()
        yout = lowess(y, x, frac=0.75, return_sorted=False)
        res = y-yout
        mean_ave = np.mean(y)
        val = res+mean_ave
        val[val<0] = 0
        val2 = val.reshape(val.shape[0],1)
        feature_gc[pos:pos+n,i:i+1] = val2
        pos += n

np.save("GC_short.npy",feature_gc)
