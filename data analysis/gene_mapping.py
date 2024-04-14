# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:48:58 2024

@author: hp
"""
import numpy as np
import pandas as pd
region = pd.read_csv("region_15k.csv", header=None, index_col=None)
region = np.array(region)
data = pd.read_csv('TSS_all_genes.csv', header=None, index_col=None)
data2 = np.array(data)
gene = np.empty(shape=(100000,6),dtype=object)
gene[:,0:5] = region
gene_id = []
pos = 0

for i in range(1,23):
    print(i)
    reg = region[region[:,1]==i]
    
    tss = data2[data2[:,2]==i]
    loc = np.empty(shape=(249250621,2),dtype=object)
    loc[:,1] = 0
    
    for k in range(tss.shape[0]):
        st = int(tss[k,4]-75000)
        end = int(tss[k,4]+75000)
        loc[(st-1):end,0] = tss[k,0]
        loc[(st-1):end,1] = 1
    
    for j in range(reg.shape[0]):
        st2 = int(reg[j,2])
        end2 = int(reg[j,3])
        
        if sum(loc[(st2-1):end2,1]) >= 1:
            gene[j+pos,5] = loc[st2-1,0]
            gene_id.append(loc[st2-1,0])
            
    pos += reg.shape[0]
     
gene_id2 = np.array(gene_id,dtype=object)
print(gene_id2.shape)
np.savetxt('region_ensg.csv',gene,delimiter=',',fmt = '%s')
np.savetxt('ensg_id.csv',gene_id2,delimiter=',',fmt = '%s')

