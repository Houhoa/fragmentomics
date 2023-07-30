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
name = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
name_array = np.array(name)
name_list = name_array.tolist()
n = len(name_list)
fsd = np.zeros([67*22,n])  #Create a matrix to store the results

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
        size = int(region_len[0,0])
        chrom = np.zeros([size,1])
        
        reg = region[region[:,0]==j]
        num2 = reg.shape[0]
        
        number = np.zeros([400,1])
        number2 = np.zeros([67,1])
        
        for k in range(1,num2+1):
            st = reg[k-1,1]
            end = reg[k-1,2]
            chrom[(st-1):end,0] += 1
        
        for m in range(1,num+1):
            st = int(read[m-1,0])
            end = int(read[m-1,0]+read[m-1,1]-1)
            mid = int(read[m-1,0]+read[m-1,1]/2)
            length = int(read[m-1,1])
            if chrom[mid-1,0] == 1 and (length >= 65 and length < 400):  #Extraction of fragments of 65-400 bp in length
                number[length-1,0] += 1
        
        total = sum(number[64:400],0)
        bp = 0      
        
        for s in range(1,68): #Divided into categories at 5bp intervals
            number2[s-1,0] = sum(number[(64+bp):(69+bp)])
            fsd[s-1+pos,i-1] = number2[s-1,0]/total
            bp += 5
       
        pos += 67
        
np.save('FSD.npy',fsd)
