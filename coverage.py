# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:44:52 2022

@author: yyh
"""
import numpy as np
import pandas as pd
import os
import h5py
region = np.load('open_region.npy')  #Import candidate region files (open chromatin regions or TSS regions)
sample = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
coverage = np.zeros([region.shape[0],len(sample)])  #Create a matrix to store the results

for i in range(len(sample)):
    print(i)
    name = sample.iloc[i,0]
    path = os.path.join("dataset_path",name,'data_n')   #Full path to the splice sample file
    pos = 0
    
    for j in range(1,23):   #Treat each chromosome sequentially
        mat = ''.join(['chr',str(j),'.mat'])
        mat2 = os.path.join(path,mat)
        mat3 = h5py.File(mat2,'r')
        #print(mat3.keys())
        read = np.transpose(mat3['read'])  #Extract the reads of the sample
        
        region_len = mat3['region_len']  #Total length of chromosome extracted
        loc = np.zeros([int(region_len[0,0]),1])
        loc2 = np.zeros([int(region_len[0,0]),1])
        
        reg = region[region[:,0]==j]
        
        for k in range(reg.shape[0]):
            st = reg[k,1]
            end = reg[k,2]
            loc[(st-1):end,0] += 1
            
        for m in range(read.shape[0]):
            mid = int(read[m,0]+read[m,1]/2)
            
            if loc[mid-1,0] == 1:   #Determine whether the mid-point of the read is in the region
                loc2[mid-1,0] += 1   #if so add 1 to the coverage
        
        for s in range(reg.shape[0]):  #Count the number of all fragments in each region and the total length of the fragments
            coverage[pos+s,i] = sum(loc2[(reg[s,1]-1):reg[s,2],0])
        
        pos += reg.shape[0]
        
np.save('coverage.npy',coverage)   #Storage results