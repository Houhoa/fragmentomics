# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:05:43 2022

@author: yyh
"""
import numpy as np
import pandas as pd
import os
import h5py
region = np.load('open_region.npy')  #Import candidate region files (open chromatin regions or TSS regions)
sample = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
fsd = np.zeros([67*22,len(sample)])  #Create a matrix to store the results

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
        chrom = np.zeros([int(region_len[0,0]),1])
        
        reg = region[region[:,0]==j]
        
        number = np.zeros([400,1])
        number2 = np.zeros([67,1])
        
        for k in range(reg.shape[0]):
            st = reg[k,1]
            end = reg[k,2]
            chrom[(st-1):end,0] += 1
        
        for m in range(read.shape[0]):
            mid = int(read[m,0]+read[m,1]/2)
            length = int(read[m,1])
            
            if chrom[mid-1,0] == 1 and (length >= 65 and length < 400):  #Extraction of fragments of 65-400 bp in length
                number[length-1,0] += 1
        
        total = sum(number[64:400],0)
        bp = 0      
        
        for s in range(1,68): #Divided into categories at 5bp intervals
            number2[s-1,0] = sum(number[(64+bp):(69+bp)])
            fsd[s-1+pos,i] = number2[s-1,0]/total
            bp += 5
       
        pos += 67
        
np.save('FSD.npy',fsd)
