# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:12:03 2022

@author: yyh
"""
from functools import reduce
import numpy as np
import pandas as pd
import os
import h5py
import scipy.io as scio
region = np.load('open_region.npy')  #Import candidate region files (open chromatin regions or TSS regions)
name = pd.read_excel("sample_ID.xlsx",sheet_name='Sheet1',usecols=[0])  #Sample ID of the dataset
name_array = np.array(name)
name_list = name_array.tolist()
n = len(name_list)

mer = np.zeros([256*22,n+1]) #Create a matrix to store the results
a,b = ['1','2','3','4'],4   #1, 2, 3, 4 represents 4 bases
c = 0
lst = reduce(lambda x, y:[z0 + z1 for z0 in x for z1 in y], [a] * b)   #4 base permutations, 256 in total
for d in range(1,23):
    for me in range(1,257):
        mer[c+me-1,0] = int(lst[me-1])
    c += 256

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
        read = np.transpose(mat3['read'])  #Extract the reads of the sample
        num = read.shape[0]
        
        region_len = mat3['region_len']  #Total length of chromosome extracted
        size = int(region_len[0,0])
        chrom = np.zeros([size,1])
        
        genome = os.path.join("reference_genome_path",mat)  #File path of the reference genome
        genome2 = scio.loadmat(genome)
        loc = genome2['loc']
        reg = region[region[:,0]==j]
        num2 = reg.shape[0]
        
        types = np.zeros([256,2])        #Deposit of each motif and its quantity
        for t in range(1,257):
            types[t-1,0] = int(lst[t-1])
        
        for k in range(1,num2+1):
            st = reg[k-1,1]
            end = reg[k-1,2]
            chrom[(st-1):end,0] += 1
            
        for m in range(1,num+1):
            st = int(read[m-1,0])
            end = int(read[m-1,0]+read[m-1,1]-1)
            mid = int(read[m-1,0]+read[m-1,1]/2)
            
            if chrom[mid-1,0] == 1:    #Determine if the mid-point of the fragment is within the open region
                #Mapping reference genome, extracted base composition (4 bases at the start of the 5' end)
                val = loc[st-1,0]*1000+loc[st,0]*100+loc[st+1,0]*10+loc[st+2,0]
                types[(types[:,0]==val),1] += 1
                
                loci = loc[(end-4):end]    #Extracted 4 bases at the end of the 3' end
                loci[loci==1] = 20  #Convert to complementary bases
                loci[loci==2] = 10
                loci[loci==3] = 40
                loci[loci==4] = 30
                #Mapping reference genomes, extraction of base composition (4 bases complementary at the end of the 3' end terminus)
                val2 = int(loci[0,0]*100+loci[1,0]*10+loci[2,0]+loci[3,0]/10)
                types[(types[:,0]==val2),1] += 1
                
        total = sum(types[:,1])
        
        for s in range(1,257):
            mer[pos+s-1,i] = types[s-1,1]/total  #Proportion of each motif
            
        pos += 256

mer = mer[:,0:n]
np.save('EDM.npy',mer)