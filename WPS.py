# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:30:12 2022

@author: yyh
"""
import numpy as np
import pandas as pd
import os
import h5py
region = np.load('open_region.npy')  #Import candidate region files (open chromatin regions or TSS regions)
sample = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
wps = np.zeros([region.shape[0],len(sample)])  #Create a matrix to store the results

for i in range(len(sample)):
    print(i)
    name = sample.iloc[i,0]
    path = os.path.join("dataset_path",name,'data_n')   #Full path to the splice sample file
    pos = 0
    
    for j in range(1,23):  #Treat each chromosome sequentially
        mat = ''.join(['chr',str(j),'.mat'])  
        mat2 = os.path.join(path,mat)
        mat3 = h5py.File(mat2,'r')
        #print(mat3.keys())
        read = np.transpose(mat3['read'])  #Extract the reads of the sample
        
        region_len = mat3['region_len']  #Total length of chromosome extracted 
        loc = np.zeros([int(region_len[0,0]),1])
        
        reg = region[region[:,0]==j] 
        
        for k in range(read.shape[0]):
            st = int(read[k,0])
            end = int(read[k,0]+read[k,1]-1)
            length = int(read[k,1])
            
            if length >= 120:
                loc[(st-59-1):(st+59),0] += -1   #Possible location of endpoints (-1)
                loc[(st+60-1):(end-60),0] += 1     #for fully covered positions  (+1)
                loc[(end-59-1):(end+59),0] += -1
            else:
                loc[(st-59-1):(end+59),0] += -1
                
        for m in range(region.shape[0]):  
            st = int(reg[m,1])
            end = int(reg[m,2])
            wps[pos+m,i] = np.mean(loc[st-1:end,0])  #Calculate the mean value of wps in each region
            
        pos += region.shape[0]

np.save('WPS.npy',wps)