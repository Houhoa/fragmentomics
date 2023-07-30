# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 19:43:55 2022

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
size = np.zeros(n)
number = np.zeros([22,n])

for i in range(1,n+1):
    print(i)
    nam = name_list[i-1]
    nam2 = "".join(nam)
    path = os.path.join("dataset_path",nam2,'data_n')   #Full path to the splice sample file
    slen = 0
    scov = 0
    
    for j in range(1,23):
        j_str = str(j)
        mat = ''.join(['chr',j_str,'.mat'])
        mat2 = os.path.join(path,mat)
        mat3 = h5py.File(mat2,'r')
        #print(mat3.keys()
        read = np.transpose(mat3['read'])
        slen += np.sum(read[:,1])  #Total read length
        scov = read.shape[0]
        number[j-1,i-1] = scov  #Total number of reads
        
    size[i-1] = slen


number = np.load("coverage.npy")
size = np.load("size.npy")

ifs = np.zeros([region.shape[0],number.shape[1]])  #Create a matrix to store the results
m = number.shape[1]
pos = 0

for i in range(1,23):
    print(i)
    reg = region[region[:,0]==i]
    n = reg.shape[0]
    ifs[pos:n+pos,0:m] = number[pos:n+pos,0:m]+size[pos:n+pos,0:m]*(scov[i-1,:]/slen)
    pos += n

np.save('IFS.npy',ifs)
