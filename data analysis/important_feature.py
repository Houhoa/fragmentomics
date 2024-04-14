# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:45:10 2024

@author: hp
"""
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing as prep
y = np.load("label.npy")
y = np.concatenate((y[24:78,:],y[133:347,:],y[421:422,:]),axis=0)
y = np.ravel(y)

region = np.load("open_region.npy",allow_pickle=True)

X1 = np.load("coverage.npy")
X1 = np.concatenate((X1[:,24:78],X1[:,133:347],X1[:,421:422]),axis=1)
X1 = prep.scale(X1)
X1 = np.transpose(X1)
X2 = np.load("end.npy")
X2 = np.concatenate((X2[:,24:78],X2[:,133:347],X2[:,421:422]),axis=1)
X2 = prep.scale(X2)
X2 = np.transpose(X2)
X3 = np.load("PFE.npy")
X3 = np.concatenate((X3[:,24:78],X3[:,133:347],X3[:,421:422]),axis=1)
X3 = prep.scale(X3)
X3 = np.transpose(X3)
X4 = np.load("IFS.npy")
X4 = np.concatenate((X4[:,24:78],X4[:,133:347],X4[:,421:422]),axis=1)
X4 = prep.scale(X4)
X4 = np.transpose(X4)
X5 = np.load("WPS.npy")
X5 = np.concatenate((X5[:,24:78],X5[:,133:347],X5[:,421:422]),axis=1)
X5 = prep.scale(X5)
X5 = np.transpose(X5)
X6 = np.load("OCF.npy")
X6 = np.concatenate((X6[:,24:78],X6[:,133:347],X6[:,421:422]),axis=1)
X6 = prep.scale(X6)
X6 = np.transpose(X6)
X7 = np.load("FSR.npy")
X7 = np.concatenate((X7[:,24:78],X7[:,133:347],X7[:,421:422]),axis=1)
X7 = prep.scale(X7)
X7 = np.transpose(X7)

classifier = SVC(C=1, kernel='linear', probability=True)
coefficient = np.zeros([561414,12])
coefficient[:,8:11] = region

y_pred = classifier.fit(X1, y)
coefficient[:,0] = abs(classifier.coef_)

y_pred2 = classifier.fit(X2, y)
coefficient[:,1] = abs(classifier.coef_)

y_pred3 = classifier.fit(X3, y)
coefficient[:,2] = abs(classifier.coef_)

y_pred4 = classifier.fit(X4, y)
coefficient[:,3] = abs(classifier.coef_)

y_pred5 = classifier.fit(X5, y)
coefficient[:,4] = abs(classifier.coef_)

y_pred6 = classifier.fit(X6, y)
coefficient[:,5] = abs(classifier.coef_)

y_pred7 = classifier.fit(X7, y)
a = abs(classifier.coef_)
pos = 0
for i in range(561414):
    coe = sum(a[:,pos:pos+3])
    coefficient[i,6] = coe
    pos += 3

for i in range(region.shape[0]):
    coefficient[i,-1] = i

coefficient[:,7] = sum(coefficient[:,0:7],axis=1)
coeffi = coefficient[np.argsort(-coefficient[:,7])]
differ_region = coeffi[0:15000,7:12]
print(differ_region.shape)
differ_region = differ_region[np.lexsort([differ_region[:,2], differ_region[:,1]])]
print(differ_region)
np.savetxt("region_15k.csv",differ_region,delimiter=',')



