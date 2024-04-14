# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 19:43:55 2022

@author: yyh
"""
import numpy as np
import pandas as pd
import os
import h5py
region = np.load('open_region.npy')  #Import candidate region files (open chromatin regions or TSS regions)
sample = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
ifs = np.zeros([region.shape[0],len(sample)])

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
        loc_number = np.zeros([int(region_len[0,0]),1])
        loc_length = np.zeros([int(region_len[0,0]),1])
        
        reg = region[region[:,0]==j]
        
        for m in range(read.shape[0]):
            mid = int(read[m,0]+read[m,1]/2)
            length = int(read[m,1])
            
            loc_number[mid-1,0] += 1
            loc_length[mid-1,0] += length
        
        total = sum(loc_length) / sum(loc_number)
        
        for k in range(reg.shape[0]):
            st = reg[k,1]
            end = reg[k,2]
            
            n = sum(loc_number[(st-1):end,0])
            l = sum(loc_length[(st-1):end,0])
            
            if n != 0:
                ifs[pos+k,i] = int(n * (1 + (l / n) / total))
        
        pos += reg.shape[0]
        
np.save('IFS.npy',ifs)
