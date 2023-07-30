# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 08:52:27 2022

@author: yyh
"""
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn.model_selection import KFold
from sklearn import preprocessing as prep

X = np.load('fragmentation_pattern_path')   #Import fragmentation patterns as features, e.g. IFS, WPS
X = prep.scale(X)   #Features are z-score normalized
X = np.transpose(X)
label = np.load("dataset_samples_label_path",allow_pickle=True)   #Class labeling of samples
y = np.ravel(label)
y = y.astype('int')

tprs = []
aucs = []
specificity = []
specificity2 = []
mean_fpr = np.linspace(0, 1, 100)

KF = KFold(n_splits=10, shuffle=True)  #10-fold cross validation to break up the data before dividing it
classifier = SVC(C=1, kernel='linear', probability=True)  #SVM parameters

for i in range(1,11): #10 times 10-fold cross validation
    print(i)
    for train, test in KF.split(X, y):
        probas = classifier.fit(X[train], y[train]).predict_proba(X[test])  #Training the model and predicting the probability of the test set
        fpr, tpr, thresholds = roc_curve(y[test], probas[:,1], drop_intermediate=False)
        tprs.append(interp(mean_fpr, fpr, tpr))
        
        specificity.append(tprs[-1][5])  #Sensitivity at 95% specificity
        specificity2.append(tprs[-1][15])  #Sensitivity at 85% specificity
        
        roc_auc = auc(fpr, tpr)   #AUC
        aucs.append(roc_auc)

#Range of values within 95% confidence intervals
mean_auc = np.mean(aucs)
n = len(aucs)
auc_SEM = np.std(aucs)/math.sqrt(n)
auc_l = mean_auc - 1.96*auc_SEM
auc_r = mean_auc + 1.96*auc_SEM
print('AUC: %.4f (95 CI: %.4f - %.4f)' % (mean_auc, auc_l, auc_r))

mean_spec = np.mean(specificity)
num = len(specificity)
spec_SEM = np.std(specificity)/math.sqrt(num)
spec_l = mean_spec - 1.96*spec_SEM
spec_r = mean_spec + 1.96*spec_SEM
print('95 specificity: %.4f (95 CI: %.4f - %.4f)' %(mean_spec, spec_l, spec_r))

mean_spec2 = np.mean(specificity2)
num2 = len(specificity2)
spec2_SEM = np.std(specificity2)/math.sqrt(num2)
spec2_l = mean_spec2 - 1.96*spec2_SEM
spec2_r = mean_spec2 + 1.96*spec2_SEM
print('85 specificity: %.4f (95 CI: %.4f - %.4f)' %(mean_spec2, spec2_l, spec2_r))

tprs2 = np.array(tprs)
aucs2 = np.array(aucs)
np.save("tprs.npy",tprs2)
np.save("aucs.npy",aucs2)
