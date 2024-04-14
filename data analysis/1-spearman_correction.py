# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:41:55 2024

@author: hp
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import preprocessing as prep
X = np.load("feature.npy")
X = prep.scale(X)
X = np.transpose(X)
label = np.load("label.npy",allow_pickle=True)
y = label.reshape(-1,1)
No = np.zeros([423,1])
for i in range(1,424):
    No[i-1,0] = i
y = np.concatenate((y,No),axis=1)

scores = np.zeros([1,2])
KF = KFold(n_splits=10, shuffle=True)
classifier = SVC(C=1, kernel='linear', probability=True)
i = 1

for train, test in KF.split(X, y):
    print(i)
    probas = classifier.fit(X[train], y[:,0][train]).predict_proba(X[test])
    score = np.concatenate((y[:,1][test].reshape(-1,1), probas[:,1].reshape(-1,1)), axis=1)
    scores = np.concatenate((scores, score), axis=0)
    i += 1
    
index = np.argsort(scores[:, 0])
sorted_scores = scores[index]
sorted_scores = sorted_scores[1:424,1]
np.savetxt("feature.txt",sorted_scores)

#%%
import numpy as np
from scipy.stats import spearmanr
result_array = np.zeros([423,10])
lst = []

with open("feature.txt", 'r') as file:
    for line in file:
        cleaned_line = line.strip()
        lst.append(cleaned_line)
        
result_array[:,9] = np.array(lst)

correlation_matrix = np.zeros((result_array.shape[1], result_array.shape[1]))

for i in range(result_array.shape[1]):
    for j in range(result_array.shape[1]):
        correlation, _ = spearmanr(result_array[:,i], result_array[:,j])
        correlation_matrix[i, j] = correlation