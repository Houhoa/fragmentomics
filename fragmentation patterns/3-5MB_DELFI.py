# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:32:09 2024

@author: hp
"""
import numpy as np
import pandas as pd
large = pd.read_excel("DELFI_5MB.xlsx",header=None)
large = np.array(large)
small = pd.read_excel("/GRC37_DELFI_region_100kb.xlsx",header=None)
small = np.array(small)
feature = np.load("GC_short.npy")  #Same with DELFI_long
feature = np.concatenate((small[:,0].reshape(-1,1),feature),axis=1)
result = np.zeros([large.shape[0],feature.shape[1]-1])
pos = 0
    
for j in range(1,23):
    print(j)
    large2 = large[large[:,0]==j]
    n = large2.shape[0]
    small2 = small[small[:,0]==j]
    m = small2.shape[0]
    feature2 = feature[feature[:,0]==j]
    
    for s in range(n):
        st = large2[s,1]
        end = large2[s,2]
        idx = []
        
        for k in range(m):
            st2 = small2[k,1]
            end2 = small2[k,2]
            
            if st2 >= st and end2 <= end:
                idx.append(k)
            else:
                pass
        
        selected_rows = feature2[idx]
        result[pos+s,:] = np.sum(selected_rows[:,1:feature.shape[1]],axis=0)
    
    pos += n
    
np.save("5MB_short.npy",result)


