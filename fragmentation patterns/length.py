# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:39:50 2022

@author: yyh
"""
import numpy as np
import pandas as pd
import os
import h5py
region = np.load('open_region.npy')  #Import candidate region files (open chromatin regions or TSS regions)
sample = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
length = np.zeros([31*22,len(sample)])    #Create a matrix to store the results

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
        chrom = np.zeros([int(region_len[0,0]),1])
        data = np.zeros([1000,1])
        reg = region[region[:,0]==j]
        
        for k in range(reg.shape[0]):
            st = reg[k,1]
            end = reg[k,2]
            chrom[(st-1):end,0] += 1
            
        for m in range(read.shape[0]):
            mid = int(read[m,0]+read[m,1]/2)
            size = int(read[m,1])
            if chrom[mid-1,0] == 1:   #Counting the number of fragments per bp in all regions
                data[(size-1),0] += 1  
                
        total = sum(data[:,0])
        bp = 0
        
        for s in range(1,31):  #Divided into different categories
            number = sum(data[bp:(bp+10),0])
            length[pos+s-1,i] = number/total
            bp += 10
            
        number2 = sum(data[(301-1):1000,0])
        length[pos+30,i] = number2/total
        pos += 31
        
np.save('length.npy',length)