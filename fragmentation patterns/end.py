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
sample   = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
End = np.zeros([region.shape[0],len(sample)])  #Create a matrix to store the results

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
        read = np.transpose(mat3['read'])   #Extract the reads of the sample
        
        region_len = mat3['region_len']  #Total length of chromosome extracted
        chrom = np.zeros([int(region_len[0,0]),1])

        reg = region[region[:,0]==j]
        
        for k in range(read.shape[0]):
            end = int(read[k,0]+read[k,1]-1) #Counting the position and number of fragment ends
            chrom[end-1,0] += 1
            
        for m in range(reg.shape[0]):  #Counting the total number of fragment ends in each region
            End[pos+m,i] = sum(chrom[(reg[m,1]-1):reg[m,2],0])
            
        pos += reg.shape[0]

np.save('end.npy',End)
