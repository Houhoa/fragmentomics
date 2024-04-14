# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:53:57 2022

@author: yyh
"""
import numpy as np
import pandas as pd
import os
import h5py
region = np.load('open_region.npy')  #Import candidate region files (open chromatin regions or TSS regions)
sample   = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
fsr = np.zeros([region.shape[0]*3,len(sample)])  #Create a matrix to store the results

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
        
        reg = region[region[:,0]==j]
        
        number = np.zeros([int(region_len[0,0]),3],dtype='float16')
        number2 = np.zeros([reg.shape[0],3])
        types = 0
        
        for m in range(read.shape[0]):  #Counting the location and number of short, medium and long fragments
            mid = int(read[m,0]+read[m,1]/2)
            length = int(read[m,1])
            
            if length >= 65 and length <= 150:
                number[mid-1,0] += 1
            elif length >= 151 and length <= 220:
                number[mid-1,1] += 1
            elif length >= 221 and length <= 400:
                number[mid-1,2] += 1

        for k in range(reg.shape[0]):
            st = reg[k,1]
            end = reg[k,2]
            number2[k,:] = np.sum(number[(st-1):end,:],axis=0) #Counting the total number of each fragment in each region

        total = np.sum(number2[:,:])
        
        for s in range(reg.shape[0]):  #Calculation ratio
            fsr[s+types+pos,i] = number2[s,0]/total
            fsr[s+1+types+pos,i] = number2[s,1]/total
            fsr[s+2+types+pos,i] = number2[s,2]/total
            
            types += 2
            
        pos += reg.shape[0]*3
        
np.save('FSR.npy',fsr)
