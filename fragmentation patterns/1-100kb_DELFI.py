# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:05:30 2024

@author: hp
"""
import numpy as np
import pandas as pd
import os
import h5py
region = pd.read_excel("/GRC37_DELFI_region_100kb.xlsx",header=None)
region = np.array(region)
sample = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])
short = np.zeros([region.shape[0],len(sample)])
Long = np.zeros([region.shape[0],len(sample)])

for i in range(len(sample)):
    print(i)
    name = sample.iloc[i,0]
    path = os.path.join("dataset_path",name,'data_n')
    pos = 0
    
    for j in range(1,23):
        mat = ''.join(['chr',str(j),'.mat'])
        mat2 = os.path.join(path,mat)
        mat3 = h5py.File(mat2,'r')
        
        read = np.transpose(mat3['read'])
        region_len = mat3['region_len']
        chroms = np.zeros([int(region_len[0,0]),1])
        chroml = np.zeros([int(region_len[0,0]),1])
        
        reg = region[region[:,0]==j]
        
        for m in range(read.shape[0]):
            mid = int(read[m,0]+read[m,1]/2)
            length = int(read[m,1])
            
            if length >= 100 and length <= 150:
                chroms[mid-1,0] += 1
            elif length >= 151 and length <= 220:
                chroml[mid-1,0] += 1
            else:
                continue

        for k in range(reg.shape[0]):
            st = reg[k,1]
            end = reg[k,2]
            short[pos+k,i] = sum(chroms[(st-1):end,:])
            Long[pos+k,i] = sum(chroml[(st-1):end,:])
        
        pos += reg.shape[0]

np.save("DELFI_short.npy",short)
np.save("DELFI_long.npy",Long)

