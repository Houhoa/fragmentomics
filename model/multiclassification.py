# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 21:48:33 2022

@author: yyh
"""
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing as prep

y = np.load("dataset_samples_label_path")   #Class labeling of samples
y = y.reshape(-1,1)
y = np.concatenate((y[0:133,:],y[359:421,:],y[422:423,:]),axis=0)  #Only cancer samples were used
y = y.astype('int')
y = label_binarize(y, classes=[1,2,3,4,6,7])

X1 = np.load('length_path')
X1 = np.concatenate((X1[:,0:133],X1[:,359:421],X1[:,422:423]),axis=1)
X1 = prep.scale(X1)
X1 = np.transpose(X1)
X2 = np.load('PFE_path')
X2 = np.concatenate((X2[:,0:133],X2[:,359:421],X2[:,422:423]),axis=1)
X2 = prep.scale(X2)
X2 = np.transpose(X2)
X3 = np.load('FSR_path')
X3 = np.concatenate((X3[:,0:133],X3[:,359:421],X3[:,422:423]),axis=1)
X3 = prep.scale(X3)
X3 = np.transpose(X3)
X4 = np.load('FSD_path')
X4 = np.concatenate((X4[:,0:133],X4[:,359:421],X4[:,422:423]),axis=1)
X4 = prep.scale(X4)
X4 = np.transpose(X4)
X5 = np.load('coverage_path')
X5 = np.concatenate((X5[:,0:133],X5[:,359:421],X5[:,422:423]),axis=1)
X5 = prep.scale(X5)
X5 = np.transpose(X5)
X6 = np.load('end_path')
X6 = np.concatenate((X6[:,0:133],X6[:,359:421],X6[:,422:423]),axis=1)
X6 = prep.scale(X6)
X6 = np.transpose(X6)
X7 = np.load('OCF_path')
X7 = np.concatenate((X7[:,0:133],X7[:,359:421],X7[:,422:423]),axis=1)
X7 = prep.scale(X7)
X7 = np.transpose(X7)
X8 = np.load('IFS_path')
X8 = np.concatenate((X8[:,0:133],X8[:,359:421],X8[:,422:423]),axis=1)
X8 = prep.scale(X8)
X8 = np.transpose(X8)
X9 = np.load('WPS_path')
X9 = np.concatenate((X9[:,0:133],X9[:,359:421],X9[:,422:423]),axis=1)
X9 = prep.scale(X9)
X9 = np.transpose(X9)
X10 = np.load('EDM_path')
X10 = np.concatenate((X10[:,0:133],X10[:,359:421],X10[:,422:423]),axis=1)
X10 = prep.scale(X10)
X10 = np.transpose(X10)

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True)) #SVM parameters
lst = [5,10,15,20,25,30,35,40,45,50] #Assigning a random seed ensures that each feature division is the same

acc1 = []  
acc2 = []
top1_acc = np.zeros([196,10])
top2_acc = np.zeros([196,10])
label_true = np.zeros([196,10])

for i in range(1,11):
    random = lst[i-1]
    KF = KFold(n_splits=10, shuffle=True, random_state=random)  #Taking a different random seed each time can be divided to get a different dataset
    print(i,random)
    
    IFP = np.zeros([176,60])  #Predicted probability of the training set
    IFP2 = np.zeros([176,60])
    IFP3 = np.zeros([176,60])
    IFP4 = np.zeros([176,60])
    IFP5 = np.zeros([176,60])
    IFP6 = np.zeros([176,60])
    IFP7 = np.zeros([177,60])
    IFP8 = np.zeros([177,60])
    IFP9 = np.zeros([177,60])
    IFP10 = np.zeros([177,60])
    
    sco = np.zeros([20,60])  #Predicted probability of the test set
    sco2 = np.zeros([20,60])
    sco3 = np.zeros([20,60])
    sco4 = np.zeros([20,60])
    sco5 = np.zeros([20,60])
    sco6 = np.zeros([20,60])
    sco7 = np.zeros([19,60])
    sco8 = np.zeros([19,60])
    sco9 = np.zeros([19,60])
    sco10 = np.zeros([19,60])
    
    result = []   #Prediction probability for all training sets
    val = []      #Prediction probability for all test sets
    data = []
    data2 = []
    
    #Layer 1: Sequentially train 10 fragmentation patterns to get the prediction probabilities for the training and test sets
    for train, test in KF.split(X1, y):
        model = classifier.fit(X1[train], y[train])
        y_train_pred = model.predict_proba(X1[train])
        y_test_pred = model.predict_proba(X1[test])
        result.append(y_train_pred[:,1])  #Predicted probability of the training set
        val.append(y_test_pred[:,1])    #Predicted probability of the test set
        data.append(y[train])  #the number of the training set
        data2.append(y[test])  #the number of the test set
        
    for train, test in KF.split(X2, y):
        model2 = classifier.fit(X2[train], y[train])
        y_train_pred2 = model2.predict_proba(X2[train])
        y_test_pred2 = model2.predict_proba(X2[test])
        result.append(y_train_pred2)
        val.append(y_test_pred2)
    
    for train, test in KF.split(X3, y):
        model3 = classifier.fit(X3[train], y[train])
        y_train_pred3 = model3.predict_proba(X3[train])
        y_test_pred3 = model3.predict_proba(X3[test])
        result.append(y_train_pred3)
        val.append(y_test_pred3)
    
    for train, test in KF.split(X4, y):
        model4 = classifier.fit(X4[train], y[train])
        y_train_pred4 = model4.predict_proba(X4[train])
        y_test_pred4 = model4.predict_proba(X4[test])
        result.append(y_train_pred4)
        val.append(y_test_pred4)
    
    for train, test in KF.split(X5, y):
        model5 = classifier.fit(X5[train], y[train])
        y_train_pred5 = model5.predict_proba(X5[train])
        y_test_pred5 = model5.predict_proba(X5[test])
        result.append(y_train_pred5)
        val.append(y_test_pred5)
    
    for train, test in KF.split(X6, y):
        model6 = classifier.fit(X6[train], y[train])
        y_train_pred6 = model6.predict_proba(X6[train])
        y_test_pred6 = model6.predict_proba(X6[test])
        result.append(y_train_pred6)
        val.append(y_test_pred6)
        
    for train, test in KF.split(X7, y):
        model7 = classifier.fit(X7[train], y[train])
        y_train_pred7 = model7.predict_proba(X7[train])
        y_test_pred7 = model7.predict_proba(X7[test])
        result.append(y_train_pred7)
        val.append(y_test_pred7)
    
    for train, test in KF.split(X8, y):
        model8 = classifier.fit(X8[train], y[train])
        y_train_pred8 = model8.predict_proba(X8[train])
        y_test_pred8 = model8.predict_proba(X8[test])
        result.append(y_train_pred8)
        val.append(y_test_pred8)
    
    for train, test in KF.split(X9, y):
        model9 = classifier.fit(X9[train], y[train])
        y_train_pred9 = model9.predict_proba(X9[train])
        y_test_pred9 = model9.predict_proba(X9[test])
        result.append(y_train_pred9)
        val.append(y_test_pred9)
    
    for train, test in KF.split(X10, y):
        model10 = classifier.fit(X10[train], y[train])
        y_train_pred10 = model10.predict_proba(X10[train])
        y_test_pred10 = model10.predict_proba(X10[test])
        result.append(y_train_pred10)
        val.append(y_test_pred10)

    '''
        10 cross-validations were performed, each with the same dataset divided by different fragmentation patterns, 
        so the predicted probabilities of the Nth dataset of the 10 fragmentation patterns were sequentially combined 
        to form a new predicted probability matrix with [sample size × (10*6)] as the input to the Layer 2
    '''
    pos = 0
    pos2 = 0
    for k in range(1,11):
        IFP[:,pos2:pos2+6] = result[pos]
        sco[:,pos2:pos2+6] = val[pos]
        pos += 10
        pos2 += 6

    pos = 1
    pos2 = 0
    for k in range(1,11):
        IFP2[:,pos2:pos2+6] = result[pos]
        sco2[:,pos2:pos2+6] = val[pos]
        pos += 10
        pos2 += 6

    pos = 2
    pos2 = 0
    for k in range(1,11):
        IFP3[:,pos2:pos2+6] = result[pos]
        sco3[:,pos2:pos2+6] = val[pos]
        pos += 10
        pos2 += 6
    
    pos = 3
    pos2 = 0
    for k in range(1,11):
        IFP4[:,pos2:pos2+6] = result[pos]
        sco4[:,pos2:pos2+6] = val[pos]
        pos += 10
        pos2 += 6

    pos = 4
    pos2 = 0
    for k in range(1,11):
        IFP5[:,pos2:pos2+6] = result[pos]
        sco5[:,pos2:pos2+6] = val[pos]
        pos += 10
        pos2 += 6
        
    pos = 5
    pos2 = 0
    for k in range(1,11):
        IFP6[:,pos2:pos2+6] = result[pos]
        sco6[:,pos2:pos2+6] = val[pos]
        pos += 10
        pos2 += 6
        
    pos = 6
    pos2 = 0
    for k in range(1,11):
        IFP7[:,pos2:pos2+6] = result[pos]
        sco7[:,pos2:pos2+6] = val[pos]
        pos += 10
        pos2 += 6
        
    pos = 7
    pos2 = 0
    for k in range(1,11):
        IFP8[:,pos2:pos2+6] = result[pos]
        sco8[:,pos2:pos2+6] = val[pos]
        pos += 10
        pos2 += 6
    
    pos = 8
    pos2 = 0
    for k in range(1,11):
        IFP9[:,pos2:pos2+6] = result[pos]
        sco9[:,pos2:pos2+6] = val[pos]
        pos += 10
        pos2 += 6

    pos = 9
    pos2 = 0
    for k in range(1,11):
        IFP10[:,pos2:pos2+6] = result[pos]
        sco10[:,pos2:pos2+6] = val[pos]
        pos += 10
        pos2 += 6
    
    #Layer 2: Predictive probability matrices for the new training and test sets are obtained, and the model is again constructed using SVMs
    y_score1 = classifier.fit(IFP,data[0]).predict_proba(sco)
    y_score2 = classifier.fit(IFP2,data[1]).predict_proba(sco2)
    y_score3 = classifier.fit(IFP3,data[2]).predict_proba(sco3)
    y_score4 = classifier.fit(IFP4,data[3]).predict_proba(sco4)
    y_score5 = classifier.fit(IFP5,data[4]).predict_proba(sco5)
    y_score6 = classifier.fit(IFP6,data[5]).predict_proba(sco6)
    y_score7 = classifier.fit(IFP7,data[6]).predict_proba(sco7)
    y_score8 = classifier.fit(IFP8,data[7]).predict_proba(sco8)
    y_score9 = classifier.fit(IFP9,data[8]).predict_proba(sco9)
    y_score10 = classifier.fit(IFP10,data[9]).predict_proba(sco10)
    
    y_score = np.concatenate((y_score1,y_score2,y_score3,y_score4,y_score5,y_score6,
                              y_score7,y_score8,y_score9,y_score10),axis=0)
    y_true = np.concatenate((data2[0],data2[1],data2[2],data2[3],data2[4],data2[5], #actual label
                             data2[6],data2[7],data2[8],data2[9]),axis=0)
    y_pred = (y_score == y_score.max(axis=1, keepdims=1)).astype(float)  #First Prediction Label
    
    nth_element = -2  #Second Predictive Labeling
    index = np.arange(y_score.shape[0]), np.argsort(y_score, axis=1)[:, nth_element]
    answer = y_score[index]
    y_pred2 = np.zeros(y_score.shape)
    y_pred2[index] = y_score[index]
    y_pred2[y_pred2 != 0] = 1
    
    accuracy = accuracy_score(y_true, y_pred)  #First accuracy
    accuracy2 = accuracy_score(y_true, y_pred2)  #Second accuracy
    
    acc1.append(accuracy)
    acc2.append(accuracy2)
    
    '''
        Storing the actual labels and the predicted labels for each division of the subdata 
        makes it possible to see the types of cancer that have been misclassified 
        for a certain type of cancer
    '''
    
    number = np.zeros([196,1])
    number2 = np.zeros([196,1])
    number3 = np.zeros([196,1])
    
    n = y_pred.shape[0]
    for j in range(1,n+1):   
        a = y_pred[j-1,:]
        b = np.argwhere(a == 1)
        number[j-1,0] = b[0][0]
        
        c = y_pred2[j-1,:]
        d = np.argwhere(c == 1)
        number2[j-1,0] = d[0][0]
        
        e = y_true[j-1,:]
        f = np.argwhere(e == 1)
        number3[j-1,0] = f[0][0]
    
    top1_acc[:,(i-1):i] = number
    top2_acc[:,(i-1):i] = number2
    label_true[:,(i-1):i] = number3
    
print('acc1_mean:',np.mean(acc1))
print('acc2_mean:',np.mean(acc2))

np.savetxt("top1_acc.csv",top1_acc,delimiter=',')
np.savetxt("top2_acc.csv",top2_acc,delimiter=',')
np.savetxt("label_true.csv",label_true,delimiter=',')

