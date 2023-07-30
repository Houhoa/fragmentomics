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
name = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
name_array = np.array(name)
name_list = name_array.tolist()
n = len(name_list)
fsr = np.zeros([561414*3,n])  #Create a matrix to store the results

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
        
        reg = region[region[:,0]==j]
        num2 = reg.shape[0]
        
        number = np.zeros([size,3],dtype='float16')
        number2 = np.zeros([num2,3])
        types = 0
        
        for m in range(1,num+1):  #Counting the location and number of short, medium and long fragments
            st = int(read[m-1,0])
            end = int(read[m-1,0]+read[m-1,1]-1)
            mid = int(read[m-1,0]+read[m-1,1]/2)
            length = int(read[m-1,1])
            if length >= 65 and length <= 150:
                number[mid-1,0] += 1
            elif length >= 151 and length <= 220:
                number[mid-1,1] += 1
            elif length >= 221 and length <= 400:
                number[mid-1,2] += 1

        for k in range(1,num2+1):
            st = reg[k-1,1]
            end = reg[k-1,2]
            number2[k-1,:] = np.sum(number[(st-1):end,:],axis=0) #Counting the total number of each fragment in each region

        total = np.sum(number2[:,:])
        
        for s in range(1,num2+1):  #Calculation ratio
            fsr[s-1+types+pos,i-1] = number2[s-1,0]/total
            fsr[s+types+pos,i-1] = number2[s-1,1]/total
            fsr[s+1+types+pos,i-1] = number2[s-1,2]/total
            
            types += 2
        pos += num2*3
        
np.save('FSR.npy',fsr)
