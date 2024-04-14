# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 10:28:21 2022

@author: yyh
"""
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn.model_selection import KFold
from sklearn import preprocessing as prep

y = np.load("dataset_samples_label_path")   #Class labeling of samples
y = y.reshape(-1,1)
No = np.zeros([y.shape[0],1])
for i in range(1,y.shape[0]+1):
    No[i-1,0] = i
y = np.concatenate((y,No),axis=1)     #Put a number on each sample

X1 = np.load('length_path')
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

scores = []    #Number and predicted probability of each sample
tprs = []
aucs = []
specificity = []
specificity2 = []
mean_fpr = np.linspace(0, 1, 100)
classifier = SVC(C=1, kernel='linear', probability=True)  #SVM parameters
lst = [5,10,15,20,25,30,35,40,45,50]  #Assigning a random seed ensures that each feature division is the same

for i in range(1,11):
    random = lst[i-1]
    KF = KFold(n_splits=10, shuffle=True, random_state=random)  #Taking a different random seed each time can be divided to get a different dataset
    print(i,random)
    
    IFP = np.zeros([380,10])  #Predicted probability of the training set
    IFP2 = np.zeros([380,10])
    IFP3 = np.zeros([380,10])
    IFP4 = np.zeros([381,10])
    IFP5 = np.zeros([381,10])
    IFP6 = np.zeros([381,10])
    IFP7 = np.zeros([381,10])
    IFP8 = np.zeros([381,10])
    IFP9 = np.zeros([381,10])
    IFP10 = np.zeros([381,10])
    
    sco = np.zeros([43,10])  #Predicted probability of the test set
    sco2 = np.zeros([43,10])
    sco3 = np.zeros([43,10])
    sco4 = np.zeros([42,10])
    sco5 = np.zeros([42,10])
    sco6 = np.zeros([42,10])
    sco7 = np.zeros([42,10])
    sco8 = np.zeros([42,10])
    sco9 = np.zeros([42,10])
    sco10 = np.zeros([42,10])
    
    result = []   #Prediction probability for all training sets
    val = []      #Prediction probability for all test sets
    data = []     
    data2 = []    
    
    #Layer 1: Sequentially train 10 fragmentation patterns to get the prediction probabilities for the training and test sets
    for train, test in KF.split(X1, y):
        model = classifier.fit(X1[train], y[:,0][train])
        y_train_pred = model.predict_proba(X1[train])  
        y_test_pred = model.predict_proba(X1[test])   
        result.append(y_train_pred[:,1])  #Predicted probability of the training set
        val.append(y_test_pred[:,1])    #Predicted probability of the test set
        data.append(y[train])  #the number of the training set
        data2.append(y[test])  #the number of the test set
    
    for train, test in KF.split(X2, y):
        model2 = classifier.fit(X2[train], y[:,0][train])
        y_train_pred2 = model2.predict_proba(X2[train])
        y_test_pred2 = model2.predict_proba(X2[test])
        result.append(y_train_pred2[:,1])
        val.append(y_test_pred2[:,1])
    
    for train, test in KF.split(X3, y):
        model3 = classifier.fit(X3[train], y[:,0][train])
        y_train_pred3 = model3.predict_proba(X3[train])
        y_test_pred3 = model3.predict_proba(X3[test])
        result.append(y_train_pred3[:,1])
        val.append(y_test_pred3[:,1])
    
    for train, test in KF.split(X4, y):
        model4 = classifier.fit(X4[train], y[:,0][train])
        y_train_pred4 = model4.predict_proba(X4[train])
        y_test_pred4 = model4.predict_proba(X4[test])
        result.append(y_train_pred4[:,1])
        val.append(y_test_pred4[:,1])
    
    for train, test in KF.split(X5, y):
        model5 = classifier.fit(X5[train], y[:,0][train])
        y_train_pred5 = model5.predict_proba(X5[train])
        y_test_pred5 = model5.predict_proba(X5[test])
        result.append(y_train_pred5[:,1])
        val.append(y_test_pred5[:,1])
    
    for train, test in KF.split(X6, y):
        model6 = classifier.fit(X6[train], y[:,0][train])
        y_train_pred6 = model6.predict_proba(X6[train])
        y_test_pred6 = model6.predict_proba(X6[test])
        result.append(y_train_pred6[:,1])
        val.append(y_test_pred6[:,1])
    
    for train, test in KF.split(X7, y):
        model7 = classifier.fit(X7[train], y[:,0][train])
        y_train_pred7 = model7.predict_proba(X7[train])
        y_test_pred7 = model7.predict_proba(X7[test])
        result.append(y_train_pred7[:,1])
        val.append(y_test_pred7[:,1])
        
    for train, test in KF.split(X8, y):
        model8 = classifier.fit(X8[train], y[:,0][train])
        y_train_pred8 = model8.predict_proba(X8[train])
        y_test_pred8 = model8.predict_proba(X8[test])
        result.append(y_train_pred8[:,1])
        val.append(y_test_pred8[:,1])
    
    for train, test in KF.split(X9, y):
        model9 = classifier.fit(X9[train], y[:,0][train])
        y_train_pred9 = model9.predict_proba(X9[train])
        y_test_pred9 = model9.predict_proba(X9[test])
        result.append(y_train_pred9[:,1])
        val.append(y_test_pred9[:,1])
        
    for train, test in KF.split(X10, y):
        model10 = classifier.fit(X10[train], y[:,0][train])
        y_train_pred10 = model10.predict_proba(X10[train])
        y_test_pred10 = model10.predict_proba(X10[test])
        result.append(y_train_pred10[:,1])
        val.append(y_test_pred10[:,1])
    
    '''
        10 cross-validations were performed, each with the same dataset divided by different fragmentation patterns, 
        so the predicted probabilities of the Nth dataset of the 10 fragmentation patterns were sequentially combined 
        to form a new predicted probability matrix with [sample size × 10] as the input to the Layer 2
    '''
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
    
    #Layer 2: Predictive probability matrices for the new training and test sets are obtained, and the model is again constructed using SVMs
    y_score = classifier.fit(IFP,data[0][:,0]).predict_proba(sco)
    fpr, tpr, thresholds = roc_curve(data2[0][:,0], y_score[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity.append(tprs[-1][5])
    specificity2.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    scores.append(data2[0][:,1])  #Store the sample number
    scores.append(y_score[:,1])  #Store the predicted probability of the second layer of the model for the sample
    
    y_score2 = classifier.fit(IFP2,data[1][:,0]).predict_proba(sco2)
    fpr, tpr, thresholds = roc_curve(data2[1][:,0], y_score2[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity.append(tprs[-1][5])
    specificity2.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    scores.append(data2[1][:,1])
    scores.append(y_score2[:,1])

    y_score3 = classifier.fit(IFP3,data[2][:,0]).predict_proba(sco3)
    fpr, tpr, thresholds = roc_curve(data2[2][:,0], y_score3[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity.append(tprs[-1][5])
    specificity2.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    scores.append(data2[2][:,1])
    scores.append(y_score3[:,1])

    y_score4 = classifier.fit(IFP4,data[3][:,0]).predict_proba(sco4)
    fpr, tpr, thresholds = roc_curve(data2[3][:,0], y_score4[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity.append(tprs[-1][5])
    specificity2.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    scores.append(data2[3][:,1])
    scores.append(y_score4[:,1])

    y_score5 = classifier.fit(IFP5,data[4][:,0]).predict_proba(sco5)
    fpr, tpr, thresholds = roc_curve(data2[4][:,0], y_score5[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity.append(tprs[-1][5])
    specificity2.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    scores.append(data2[4][:,1])
    scores.append(y_score5[:,1])

    y_score6 = classifier.fit(IFP6,data[5][:,0]).predict_proba(sco6)
    fpr, tpr, thresholds = roc_curve(data2[5][:,0], y_score6[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity.append(tprs[-1][5])
    specificity2.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    scores.append(data2[5][:,1])
    scores.append(y_score6[:,1])

    y_score7 = classifier.fit(IFP7,data[6][:,0]).predict_proba(sco7)
    fpr, tpr, thresholds = roc_curve(data2[6][:,0], y_score7[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity.append(tprs[-1][5])
    specificity2.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    scores.append(data2[6][:,1])
    scores.append(y_score7[:,1])

    y_score8 = classifier.fit(IFP8,data[7][:,0]).predict_proba(sco8)
    fpr, tpr, thresholds = roc_curve(data2[7][:,0], y_score8[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity.append(tprs[-1][5])
    specificity2.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    scores.append(data2[7][:,1])
    scores.append(y_score8[:,1])

    y_score9 = classifier.fit(IFP9,data[8][:,0]).predict_proba(sco9)
    fpr, tpr, thresholds = roc_curve(data2[8][:,0], y_score9[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity.append(tprs[-1][5])
    specificity2.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    scores.append(data2[8][:,1])
    scores.append(y_score9[:,1])

    y_score10 = classifier.fit(IFP10,data[9][:,0]).predict_proba(sco10)
    fpr, tpr, thresholds = roc_curve(data2[9][:,0], y_score10[:,1], drop_intermediate=False)
    tprs.append(interp(mean_fpr, fpr, tpr))
    specificity.append(tprs[-1][5])
    specificity2.append(tprs[-1][15])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    scores.append(data2[9][:,1])
    scores.append(y_score10[:,1])
    
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

#10 times 10-fold cross-validations were conducted, with 10 predicted probabilities for each sample, arranged by sample number, taking the mean of the 10 probabilities
test = np.zeros([y.shape[0],12])
a = 0
for i in range(1,11):
    test2 = np.zeros([y.shape[0],2])
    pos = 0
    for j in range(1,11):
        n = scores[a].shape[0]
        number = scores[a]
        score = scores[a+1]
        test2[pos:pos+n,0] = number
        test2[pos:pos+n,1] = score
        pos += n
        a += 2
    test2 = test2[test2[:,0].argsort()]
    test[:,i] = test2[:,1]
test[:,0] = test2[:,0]
test[:,11] = np.average(test[:,1:11],axis=1)
score2 = np.zeros([y.shape[0],2])
score2[:,0] = test[:,0]
score2[:,1] = test[:,-1]
np.savetxt("IFP_scores.txt",score2)



