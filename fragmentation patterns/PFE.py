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
name = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
name_array = np.array(name)
name_list = name_array.tolist()
n = len(name_list)
pfe = np.zeros([561414,n])  #Create a matrix to store the results

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
        size = int(region_len[0,0])
        chrom = np.zeros([size,17],dtype='float16')
        
        reg = region[region[:,0]==j]
        num2 = reg.shape[0]
        
        for m in range(1,num+1):  #Counting the number of fragments in different categories
            mid = int(read[m-1,0]+read[m-1,1]/2)
            length = int(read[m-1,1])
            
            if length < 100:
                chrom[mid-1,0] += 1
            elif length >= 250:
                chrom[mid-1,16] += 1
            else:
                res = length//10
                chrom[mid-1,res-9] += 1
                
        for k in range(1,num2+1):
            st = reg[k-1,1]
            end = reg[k-1,2]
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
                
            pfe[k-1+pos,i-1] = p
            
        pos += num2
        
np.save('PFE.npy',pfe)