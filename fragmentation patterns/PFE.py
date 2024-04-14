# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:18:57 2022

@author: yyh
"""
import numpy as np
import pandas as pd
import os
import h5py
import math
region = np.load('open_region.npy')  #Import candidate region files (open chromatin regions or TSS regions)
sample  = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
pfe = np.zeros([region.shape[0],len(sample)])  #Create a matrix to store the results

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
        chrom = np.zeros([int(region_len[0,0]),17],dtype='float16')
        
        reg = region[region[:,0]==j]
        
        for m in range(read.shape[0]):  #Counting the number of fragments in different categories
            mid = int(read[m,0]+read[m,1]/2)
            length = int(read[m,1])
            
            if length < 100:
                chrom[mid-1,0] += 1
            elif length >= 250:
                chrom[mid-1,16] += 1
            else:
                res = length//10
                chrom[mid-1,res-9] += 1
                
        for k in range(reg.shape[0]):
            st = reg[k,1]
            end = reg[k,2]
            data = chrom[(st-1):end,:]
            
            total = np.sum(data)
            density = np.zeros([1,17])
            p = 0
            
            for s in range(1,18):  #Calculation ratio
                ni = np.sum(data[:,s-1])
                if ni != 0 :
                    density[0,s-1] = ni/total
                else:
                    density[0,s-1] = 1
            
            for s in range(1,18):
                p -= density[0,s-1]*math.log2(density[0,s-1]) #Shannon's Entropy
                
            pfe[k-1+pos,i] = p
            
        pos += reg.shape[0]
        
np.save('PFE.npy',pfe)