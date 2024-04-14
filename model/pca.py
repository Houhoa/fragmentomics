# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 08:52:27 2022

@author: yyhou
"""
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn.model_selection import KFold
from sklearn import preprocessing as prep
from sklearn.decomposition import PCA
X = np.load("feature.npy")
X = prep.scale(X)
X = np.transpose(X)
label = np.load("label.npy",allow_pickle=True)
y = np.ravel(label)
y = y.astype('int')

tprs = []
aucs = []
specificity = []
specificity2 = []
mean_fpr = np.linspace(0, 1, 100)

KF = KFold(n_splits=10, shuffle=True)
classifier = SVC(C=1, kernel='linear', probability=True)

for i in range(1,11):
    print(i)
    for train, test in KF.split(X, y):
        pca = PCA(n_components=X[train].shape[0]-1)
        x_train_pca = pca.fit_transform(X[train])
        x_test_pca = pca.transform(X[test])
        print(x_train_pca.shape)
        print(x_test_pca.shape)
        
        probas = classifier.fit(x_train_pca, y[train]).predict_proba(x_test_pca)
        fpr, tpr, thresholds = roc_curve(y[test], probas[:,1], drop_intermediate=False)
        tprs.append(interp(mean_fpr, fpr, tpr))
        
        specificity.append(tprs[-1][5])
        specificity2.append(tprs[-1][15])
        
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

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
np.save("feature_tprs.npy",tprs2)
np.save("feature_aucs.npy",aucs2)
