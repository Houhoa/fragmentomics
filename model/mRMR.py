# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:57:53 2024

@author: hp
"""
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import preprocessing as prep
y = np.load("label.npy")
y = y.astype('int')
y = np.ravel(y)

X1 = np.load("coverage.npy")
X1 = prep.scale(X1)
X1 = np.transpose(X1)
X2 = np.load("end.npy")
X2 = prep.scale(X2)
X2 = np.transpose(X2)
X3 = np.load("FSR.npy")
X3 = prep.scale(X3)
X3 = np.transpose(X3)
X4 = np.load("IFS.npy")
X4 = prep.scale(X4)
X4 = np.transpose(X4)
X5 = np.load("EDM.npy")
X5 = prep.scale(X5)
X5 = np.transpose(X5)
X6 = np.load("PFE.npy")
X6 = prep.scale(X6)
X6 = np.transpose(X6)
X7 = np.load("WPS.npy")
X7 = prep.scale(X7)
X7 = np.transpose(X7)
X8 = np.load("OCF.npy")
X8 = prep.scale(X8)
X8 = np.transpose(X8)
X9 = np.load("FSD.npy")
X9 = prep.scale(X9)
X9 = np.transpose(X9)
X10 = np.load("length.npy")
X10 = prep.scale(X10)
X10 = np.transpose(X10)

classifier = SVC(C=1, kernel='linear', probability=True)
lst = [5,10,15,20,25,30,35,40,45,50]
train_pred = []
test_pred = []
train_y = []
test_y = []

for i in range(1,11):
    KF = KFold(n_splits=10, shuffle=True, random_state=lst[i-1])
    print(i,lst[i-1])
    
    IFP = np.zeros([380,10])
    IFP2 = np.zeros([380,10])
    IFP3 = np.zeros([380,10])
    IFP4 = np.zeros([381,10])
    IFP5 = np.zeros([381,10])
    IFP6 = np.zeros([381,10])
    IFP7 = np.zeros([381,10])
    IFP8 = np.zeros([381,10])
    IFP9 = np.zeros([381,10])
    IFP10 = np.zeros([381,10])
    
    sco = np.zeros([43,10])
    sco2 = np.zeros([43,10])
    sco3 = np.zeros([43,10])
    sco4 = np.zeros([42,10])
    sco5 = np.zeros([42,10])
    sco6 = np.zeros([42,10])
    sco7 = np.zeros([42,10])
    sco8 = np.zeros([42,10])
    sco9 = np.zeros([42,10])
    sco10 = np.zeros([42,10])
    
    result = []
    val = []
    data = []
    data2 = []
    
    for train, test in KF.split(X1, y):
        model = classifier.fit(X1[train], y[train])
        y_train_pred = model.predict_proba(X1[train])
        y_test_pred = model.predict_proba(X1[test])
        result.append(y_train_pred[:,1])
        val.append(y_test_pred[:,1])
        data.append(y[train])
        data2.append(y[test])
    
    for train, test in KF.split(X2, y):
        model2 = classifier.fit(X2[train], y[train])
        y_train_pred2 = model2.predict_proba(X2[train])
        y_test_pred2 = model2.predict_proba(X2[test])
        result.append(y_train_pred2[:,1])
        val.append(y_test_pred2[:,1])
    
    for train, test in KF.split(X3, y):
        model3 = classifier.fit(X3[train], y[train])
        y_train_pred3 = model3.predict_proba(X3[train])
        y_test_pred3 = model3.predict_proba(X3[test])
        result.append(y_train_pred3[:,1])
        val.append(y_test_pred3[:,1])
    
    for train, test in KF.split(X4, y):
        model4 = classifier.fit(X4[train], y[train])
        y_train_pred4 = model4.predict_proba(X4[train])
        y_test_pred4 = model4.predict_proba(X4[test])
        result.append(y_train_pred4[:,1])
        val.append(y_test_pred4[:,1])
    
    for train, test in KF.split(X5, y):
        model5 = classifier.fit(X5[train], y[train])
        y_train_pred5 = model5.predict_proba(X5[train])
        y_test_pred5 = model5.predict_proba(X5[test])
        result.append(y_train_pred5[:,1])
        val.append(y_test_pred5[:,1])
    
    for train, test in KF.split(X6, y):
        model6 = classifier.fit(X6[train], y[train])
        y_train_pred6 = model6.predict_proba(X6[train])
        y_test_pred6 = model6.predict_proba(X6[test])
        result.append(y_train_pred6[:,1])
        val.append(y_test_pred6[:,1])
    
    for train, test in KF.split(X7, y):
        model7 = classifier.fit(X7[train], y[train])
        y_train_pred7 = model7.predict_proba(X7[train])
        y_test_pred7 = model7.predict_proba(X7[test])
        result.append(y_train_pred7[:,1])
        val.append(y_test_pred7[:,1])
        
    for train, test in KF.split(X8, y):
        model8 = classifier.fit(X8[train], y[train])
        y_train_pred8 = model8.predict_proba(X8[train])
        y_test_pred8 = model8.predict_proba(X8[test])
        result.append(y_train_pred8[:,1])
        val.append(y_test_pred8[:,1])
    
    for train, test in KF.split(X9, y):
        model9 = classifier.fit(X9[train], y[train])
        y_train_pred9 = model9.predict_proba(X9[train])
        y_test_pred9 = model9.predict_proba(X9[test])
        result.append(y_train_pred9[:,1])
        val.append(y_test_pred9[:,1])
        
    for train, test in KF.split(X10, y):
        model10 = classifier.fit(X10[train], y[train])
        y_train_pred10 = model10.predict_proba(X10[train])
        y_test_pred10 = model10.predict_proba(X10[test])
        result.append(y_train_pred10[:,1])
        val.append(y_test_pred10[:,1])

    pos = 0
    for k in range(1,11):
        IFP[:,k-1] = result[pos]
        sco[:,k-1] = val[pos]
        pos += 10

    pos = 1
    for k in range(1,11):
        IFP2[:,k-1] = result[pos]
        sco2[:,k-1] = val[pos]
        pos += 10

    pos = 2
    for k in range(1,11):
        IFP3[:,k-1] = result[pos]
        sco3[:,k-1] = val[pos]
        pos += 10
    
    pos = 3
    for k in range(1,11):
        IFP4[:,k-1] = result[pos]
        sco4[:,k-1] = val[pos]
        pos += 10

    pos = 4
    for k in range(1,11):
        IFP5[:,k-1] = result[pos]
        sco5[:,k-1] = val[pos]
        pos += 10

    pos = 5
    for k in range(1,11):
        IFP6[:,k-1] = result[pos]
        sco6[:,k-1] = val[pos]
        pos += 10
    
    pos = 6
    for k in range(1,11):
        IFP7[:,k-1] = result[pos]
        sco7[:,k-1] = val[pos]
        pos += 10
    
    pos = 7
    for k in range(1,11):
        IFP8[:,k-1] = result[pos]
        sco8[:,k-1] = val[pos]
        pos += 10
    
    pos = 8
    for k in range(1,11):
        IFP9[:,k-1] = result[pos]
        sco9[:,k-1] = val[pos]
        pos += 10

    pos = 9
    for k in range(1,11):
        IFP10[:,k-1] = result[pos]
        sco10[:,k-1] = val[pos]
        pos += 10
    
    train_pred.append(IFP)
    train_pred.append(IFP2)
    train_pred.append(IFP3)
    train_pred.append(IFP4)
    train_pred.append(IFP5)
    train_pred.append(IFP6)
    train_pred.append(IFP7)
    train_pred.append(IFP8)
    train_pred.append(IFP9)
    train_pred.append(IFP10)
    test_pred.append(sco)
    test_pred.append(sco2)
    test_pred.append(sco3)
    test_pred.append(sco4)
    test_pred.append(sco5)
    test_pred.append(sco6)
    test_pred.append(sco7)
    test_pred.append(sco8)
    test_pred.append(sco9)
    test_pred.append(sco10)
    train_y.append(data[0])
    train_y.append(data[1])
    train_y.append(data[2])
    train_y.append(data[3])
    train_y.append(data[4])
    train_y.append(data[5])
    train_y.append(data[6])
    train_y.append(data[7])
    train_y.append(data[8])
    train_y.append(data[9])
    test_y.append(data2[0])
    test_y.append(data2[1])
    test_y.append(data2[2])
    test_y.append(data2[3])
    test_y.append(data2[4])
    test_y.append(data2[5])
    test_y.append(data2[6])
    test_y.append(data2[7])
    test_y.append(data2[8])
    test_y.append(data2[9])

with open("train_pred.pkl", 'wb') as file:
    pickle.dump(train_pred, file)
    
with open("test_pred.pkl", 'wb') as file:
    pickle.dump(test_pred, file)
    
with open("train_y.pkl", 'wb') as file:
    pickle.dump(train_y, file)
    
with open("test_y.pkl", 'wb') as file:
    pickle.dump(test_y, file)
    
#%%
import pickle
import math
import pymrmr
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from numpy import interp
with open("train_pred.pkl", 'rb') as file:
    train_pred = pickle.load(file)

with open("test_pred.pkl", 'rb') as file:
    test_pred = pickle.load(file)
    
with open("train_y.pkl", 'rb') as file:
    train_y = pickle.load(file)

with open("test_y.pkl", 'rb') as file:
    test_y = pickle.load(file)

classifier = SVC(C=1, kernel='linear', probability=True)
tprs = []
aucs = []
specificity_95 = []
specificity_85 = []
mean_fpr = np.linspace(0, 1, 100)

for i in range(100):
    print(i)
    X_train = train_pred[i]
    y_train = train_y[i]
    X_test = test_pred[i]
    y_test = test_y[i]
    
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    X_train_df.columns = [f"a{i}" for i in range(1, 11)]
    X_test_df.columns = [f"a{i}" for i in range(1, 11)]

    selected_features = pymrmr.mRMR(X_train_df, 'MIQ', 3)
    print(selected_features)
    X_train_df2 = X_train_df[selected_features]
    X_test_df2 = X_test_df[selected_features]

    X_train2 = np.array(X_train_df2)
    X_test2 = np.array(X_test_df2)

    probas = classifier.fit(X_train2, y_train).predict_proba(X_test2)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity_95.append(tprs[-1][5])
    specificity_85.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

mean_auc = np.mean(aucs)
n = len(aucs)
auc_SEM = np.std(aucs)/math.sqrt(n)
auc_l = mean_auc - 1.96*auc_SEM
auc_r = mean_auc + 1.96*auc_SEM
print('AUC: %.4f (95 CI: %.4f - %.4f)' % (mean_auc, auc_l, auc_r))

mean_spec95 = np.mean(specificity_95)
num95 = len(specificity_95)
spec95_SEM = np.std(specificity_95)/math.sqrt(num95)
spec95_l = mean_spec95 - 1.96*spec95_SEM
spec95_r = mean_spec95 + 1.96*spec95_SEM
print('95 specificity: %.4f (95 CI: %.4f - %.4f)' %(mean_spec95, spec95_l, spec95_r))

mean_spec85 = np.mean(specificity_85)
num85 = len(specificity_85)
spec85_SEM = np.std(specificity_85)/math.sqrt(num85)
spec85_l = mean_spec85 - 1.96*spec85_SEM
spec85_r = mean_spec85 + 1.96*spec85_SEM
print('85 specificity: %.4f (95 CI: %.4f - %.4f)' %(mean_spec85, spec85_l, spec85_r))
