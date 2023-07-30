# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:17:20 2022

@author: yyh
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn import preprocessing as prep

X_train = np.load('fragmentation_pattern_path')   #Features of the training data
X_train = prep.scale(X_train)   #Features are z-score normalized
X_train = np.transpose(X_train)
label = np.load("dataset_samples_label_path",allow_pickle=True)  #label of the training data
y_train = np.ravel(label)
y_train = y_train.astype('int')

X_validation = np.load('fragmentation_pattern_path') #Features of the validation data
X_validation = prep.scale(X_validation) #Features are z-score normalized
X_validation = np.transpose(X_validation)
label2 = np.load("dataset_samples_label_path",allow_pickle=True)  #label of the validation data
y_validation = np.ravel(label2)
y_validation = y_validation.astype('int')

tprs = []
mean_fpr = np.linspace(0, 1, 100)

classifier = SVC(C=1, kernel='linear', probability=True)  #SVM parameters
probas = classifier.fit(X_train, y_train).predict_proba(X_validation)
fpr, tpr, thresholds = roc_curve(y_validation, probas[:,1], drop_intermediate=False)
tprs.append(interp(mean_fpr, fpr, tpr))
specificity = tprs[-1][5]  #Sensitivity at 95% specificity
specificity2 = tprs[-1][15]  #Sensitivity at 85% specificity
auc = auc(fpr, tpr)  #AUC

print('AUC: %.4f' % (auc))
print('95 specificity: %.4f' %(specificity))
print('85 specificity: %.4f' %(specificity2))

tprs2 = np.array(tprs)
np.save("tprs.npy",tprs2)
