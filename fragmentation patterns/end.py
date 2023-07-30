# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:30:05 2022

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
End = np.zeros([561414,n])  #Create a matrix to store the results

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
        read = np.transpose(mat3['read'])   #Extract the reads of the sample
        num = read.shape[0]
        
        region_len = mat3['region_len']  #Total length of chromosome extracted
        size = int(region_len[0,0])
        chrom = np.zeros([size,1])

        reg = region[region[:,0]==j]
        num2 = reg.shape[0]
        
        for k in range(1,num+1):
            end = int(read[k-1,0]+read[k-1,1]-1) #Counting the position and number of fragment ends
            chrom[end-1,0] += 1
            
        for m in range (1,num2+1):  #Counting the total number of fragment ends in each region
            End[pos+m-1,i-1] = sum(chrom[(reg[m-1,1]-1):reg[m-1,2],0])
            
        pos += num2

np.save('end.npy',End)
