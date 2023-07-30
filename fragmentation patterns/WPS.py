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
name = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
name_array = np.array(name)
name_list = name_array.tolist()
n = len(name_list)
wps = np.zeros([561414,n])  #Create a matrix to store the results

for i in range(1,n+1):
    print(i)
    nam = name_list[i-1]
    nam2 = "".join(nam)
    path = os.path.join("dataset_path",nam2,'data_n')   #Full path to the splice sample file
    pos = 0
    
    for j in range(1,23):  #Treat each chromosome sequentially
        j_str = str(j)
        mat = ''.join(['chr',j_str,'.mat'])  
        mat2 = os.path.join(path,mat)
        mat3 = h5py.File(mat2,'r')
        #print(mat3.keys())
        read = np.transpose(mat3['read'])  #Extract the reads of the sample
        num = read.shape[0]
        
        region_len = mat3['region_len']  #Total length of chromosome extracted
        chrom = int(region_len[0,0])
        loc = np.zeros([chrom,1])
        
        reg = region[region[:,0]==j]
        num2 = region.shape[0]
        
        for k in range(1,num+1):
            st = int(read[k-1,0])
            end = int(read[k-1,0]+read[k-1,1]-1)
            length = int(read[k-1,1])
            
            if length >= 120:
                loc[(st-59-1):(st+59),0] += -1   #Possible location of endpoints (-1)
                loc[(st+60-1):(end-60),0] += 1     #for fully covered positions  (+1)
                loc[(end-59-1):(end+59),0] += -1
            else:
                loc[(st-59-1):(end+59),0] += -1
                
        for m in range (1,num2+1):  
            st = int(reg[m-1,1])
            end = int(reg[m-1,2])
            wps[pos+m-1,i-1] = np.mean(loc[st-1:end,0])  #Calculate the mean value of wps in each region
            
        pos += num2

np.save('WPS.npy',wps)