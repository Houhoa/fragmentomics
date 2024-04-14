# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 21:28:39 2022

@author: yyh
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn import preprocessing as prep

y = np.load("dataset_samples_label_path",allow_pickle=True)   #label of the training data
y = y.astype('int')
y = np.ravel(y)

X1 = np.load('length_path')  #Feature the fragmentation pattern of the training set
X1 = prep.scale(X1)
X1 = np.transpose(X1)
X2 = np.load('PFE_path')
X2 = prep.scale(X2)
X2 = np.transpose(X2)
X3 = np.load('FSR_path')
X3 = prep.scale(X3)
X3 = np.transpose(X3)
X4 = np.load('FSD_path')
X4 = prep.scale(X4)
X4 = np.transpose(X4)
X5 = np.load('coverage_path')
X5 = prep.scale(X5)
X5 = np.transpose(X5)
X6 = np.load('end_path')
X6 = prep.scale(X6)
X6 = np.transpose(X6)
X7 = np.load('OCF_path')
X7 = prep.scale(X7)
X7 = np.transpose(X7)
X8 = np.load('IFS_path')
X8 = prep.scale(X8)
X8 = np.transpose(X8)
X9 = np.load('WPS_path')
X9 = prep.scale(X9)
X9 = np.transpose(X9)
X10 = np.load('EDM_path')
X10 = prep.scale(X10)
X10 = np.transpose(X10)

y_val = np.load("dataset_samples_label_path",allow_pickle=True)  #label of the validation data
y_val = y_val.astype('int')
y_val = np.ravel(y_val)

X_val1 = np.load('length_path') #Feature the fragmentation pattern of the validation set
X_val1 = prep.scale(X_val1)
X_val1 = np.transpose(X_val1)
X_val2 = np.load('PFE_path')
X_val2 = prep.scale(X_val2)
X_val2 = np.transpose(X_val2)
X_val3 = np.load('FSR_path')
X_val3 = prep.scale(X_val3)
X_val3 = np.transpose(X_val3)
X_val4 = np.load('FSD_path')
X_val4 = prep.scale(X_val4)
X_val4 = np.transpose(X_val4)
X_val5 = np.load('coverage_path')
X_val5 = prep.scale(X_val5)
X_val5 = np.transpose(X_val5)
X_val6 = np.load('end_path')
X_val6 = prep.scale(X_val6)
X_val6 = np.transpose(X_val6)
X_val7 = np.load('OCF_path')
X_val7 = prep.scale(X_val7)
X_val7 = np.transpose(X_val7)
X_val8 = np.load('IFS_path')
X_val8 = prep.scale(X_val8)
X_val8 = np.transpose(X_val8)
X_val9 = np.load('WPS_path')
X_val9 = prep.scale(X_val9)
X_val9 = np.transpose(X_val9)
X_val10 = np.load('EDM_path')
X_val10 = prep.scale(X_val10)
X_val10 = np.transpose(X_val10)

classifier = SVC(C=1, kernel='linear', probability=True)
X_train = np.zeros([X1.shape[0],10])
X_validation = np.zeros([X_val1.shape[0],10])

#Layer 1: Get the predicted probabilities of the training and validation sets
model = classifier.fit(X1, y)
train_score = model.predict_proba(X1)
validation_score = model.predict_proba(X_val1)
X_train[:,0] = train_score[:,1]
X_validation[:,0] = validation_score[:,1]

model2 = classifier.fit(X2, y)
train_score2 = model2.predict_proba(X2)
validation_score2 = model2.predict_proba(X_val2)
X_train[:,1] = train_score2[:,1]
X_validation[:,1] = validation_score2[:,1]

model3 = classifier.fit(X3, y)
train_score3 = model3.predict_proba(X3)
validation_score3 = model3.predict_proba(X_val3)
X_train[:,2] = train_score3[:,1]
X_validation[:,2] = validation_score3[:,1]

model4 = classifier.fit(X4, y)
train_score4 = model4.predict_proba(X4)
validation_score4 = model4.predict_proba(X_val4)
X_train[:,3] = train_score4[:,1]
X_validation[:,3] = validation_score4[:,1]

model5 = classifier.fit(X5, y)
train_score5 = model5.predict_proba(X5)
validation_score5 = model5.predict_proba(X_val5)
X_train[:,4] = train_score5[:,1]
X_validation[:,4] = validation_score5[:,1]

model6 = classifier.fit(X6, y)
train_score6 = model6.predict_proba(X6)
validation_score6 = model6.predict_proba(X_val6)
X_train[:,5] = train_score6[:,1]
X_validation[:,5] = validation_score6[:,1]

model7 = classifier.fit(X7, y)
train_score7 = model7.predict_proba(X7)
validation_score7 = model7.predict_proba(X_val7)
X_train[:,6] = train_score7[:,1]
X_validation[:,6] = validation_score7[:,1]

model8 = classifier.fit(X8, y)
train_score8 = model8.predict_proba(X8)
validation_score8 = model8.predict_proba(X_val8)
X_train[:,7] = train_score8[:,1]
X_validation[:,7] = validation_score8[:,1]

model9 = classifier.fit(X9, y)
train_score9 = model9.predict_proba(X9)
validation_score9 = model9.predict_proba(X_val9)
X_train[:,8] = train_score9[:,1]
X_validation[:,8] = validation_score9[:,1]

model10 = classifier.fit(X10, y)
train_score10 = model10.predict_proba(X10)
validation_score10 = model10.predict_proba(X_val10)
X_train[:,9] = train_score10[:,1]
X_validation[:,9] = validation_score10[:,1]

tprs = []
mean_fpr = np.linspace(0, 1, 100)

#Layer 2: Using the predicted probability matrix as a new feature
probas = classifier.fit(X_train, y).predict_proba(X_validation)
fpr, tpr, thresholds = roc_curve(y_val, probas[:,1], drop_intermediate=False)
tprs.append(interp(mean_fpr, fpr, tpr))
specificity_95 = tprs[-1][5]  
specificity_85 = tprs[-1][15]
auc = auc(fpr, tpr)

print('AUC: %.4f' % (auc))
print('95 specificity: %.4f' %(specificity_95))
print('85 specificity: %.4f' %(specificity_85))

tprs2 = np.array(tprs)
np.save("IFP_tprs.npy",tprs2)

