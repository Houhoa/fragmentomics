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
name = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
name_array = np.array(name)
name_list = name_array.tolist()
n = len(name_list)
coverage = np.zeros([561414,n])  #Create a matrix to store the results
size = np.zeros([561414,n])

for i in range(1,n+1):
    print(i)
    nam = name_list[i-1]
    nam2 = "".join(nam)
    path = os.path.join("dataset_path",nam2,'data_n')   #Full path to the splice sample file
    pos = 0
    
    for j in range(1,23):   #Treat each chromosome sequentially
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
        loc2 = np.zeros([chrom,1])
        loc3 = np.zeros([chrom,1])
        
        reg = region[region[:,0]==j]
        num2 = reg.shape[0]
        
        for k in range(1,num2+1):
            st = reg[k-1,1]
            end = reg[k-1,2]
            loc[(st-1):end,0] += 1
            
        for m in range(1,num+1):
            st = int(read[m-1,0])
            end = int(read[m-1,0]+read[m-1,1]-1)
            mid = int(read[m-1,0]+read[m-1,1]/2)
            length = int(read[m-1,1])
            if loc[mid-1,0] == 1:   #Determine whether the mid-point of the read is in the region
                loc2[mid-1,0] += 1   #if so add 1 to the coverage
                loc3[mid-1,0] += length  #the total length of the fragments in the region plus the length of the read
        
        for s in range(1,num2+1):  #Count the number of all fragments in each region and the total length of the fragments
            coverage[pos+s-1,i-1] = sum(loc2[(reg[s-1,1]-1):reg[s-1,2],0])
            size[pos+s-1,i-1] = sum(loc3[(reg[s-1,1]-1):reg[s-1,2],0])
        
        pos += num2
        
np.save('coverage.npy',coverage)   #Storage results
np.save('size.npy',size)