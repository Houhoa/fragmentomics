# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:11:18 2024

@author: hp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.pyplot import MultipleLocator
data = pd.read_excel("data.xlsx")
healthy = data[data['stage.1']==0].iloc[:,-1]
desired_fpr = 1 - 0.95

I = data[data['stage.1']==1].iloc[:,-1]
ture_label_I = np.array([0]*len(healthy) + [1]*len(I))
pred_label_I = np.concatenate([healthy.values.ravel(), I.values.ravel()])
fpr1, tpr1, thresholds1 = roc_curve(ture_label_I, pred_label_I, drop_intermediate=False)
roc_auc1 = auc(fpr1, tpr1)
index1 = np.argmin(np.abs(fpr1 - desired_fpr))
sensitivity1 = tpr1[index1]
print("I期的灵敏度为")
print(sensitivity1)

II = data[data['stage.1']==2].iloc[:,-1]
ture_label_II = np.array([0]*len(healthy) + [1]*len(II))
pred_label_II = np.concatenate([healthy.values.ravel(), II.values.ravel()])
fpr2, tpr2, thresholds2 = roc_curve(ture_label_II, pred_label_II, drop_intermediate=False)
roc_auc2 = auc(fpr2, tpr2)
index2 = np.argmin(np.abs(fpr2 - desired_fpr))
sensitivity2 = tpr2[index2]
print("II期的灵敏度为")
print(sensitivity2)

III = data[data['stage.1']==3].iloc[:,-1]
ture_label_III = np.array([0]*len(healthy) + [1]*len(III))
pred_label_III = np.concatenate([healthy.values.ravel(), III.values.ravel()])
fpr3, tpr3, thresholds3 = roc_curve(ture_label_III, pred_label_III, drop_intermediate=False)
roc_auc3 = auc(fpr3, tpr3)
index3 = np.argmin(np.abs(fpr3 - desired_fpr))
sensitivity3 = tpr3[index3]
print("III期的灵敏度为")
print(sensitivity3)

IV = data[data['stage.1']==4].iloc[:,-1]
ture_label_IV = np.array([0]*len(healthy) + [1]*len(IV))
pred_label_IV = np.concatenate([healthy.values.ravel(), IV.values.ravel()])
fpr4, tpr4, thresholds4 = roc_curve(ture_label_IV, pred_label_IV, drop_intermediate=False)
roc_auc4 = auc(fpr4, tpr4)
index4 = np.argmin(np.abs(fpr4 - desired_fpr))
sensitivity4 = tpr4[index4]
print("IV期的灵敏度为")
print(sensitivity4)

plt.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Arial', 'weight' : 'normal', 'size' : 8}
fig, ax = plt.subplots(figsize=(2.2,2.5),dpi=300)

plt.plot(fpr1, tpr1, color='green', alpha=1, linewidth=0.5, label='I : AUC : {:.4f}'.format(roc_auc1))
plt.plot(fpr2, tpr2, color='blue', alpha=1, linewidth=0.5, label='II : AUC : {:.4f}'.format(roc_auc2))
plt.plot(fpr3, tpr3, color='orange', alpha=1, linewidth=0.5, label='III : AUC : {:.4f}'.format(roc_auc3))
plt.plot(fpr4, tpr4, color='red', alpha=1, linewidth=0.5, label='IV : AUC : {:.4f}'.format(roc_auc4))

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
major_locator = MultipleLocator(0.1)
ax = plt.gca()
ax.xaxis.set_major_locator(major_locator)
ax.yaxis.set_major_locator(major_locator)

plt.tick_params(direction='in', width = 0.3, length=1.5)
plt.tick_params(top=False,bottom=True,left=True,right=False)

TK = plt.gca()
width = 0.5
TK.spines['bottom'].set_linewidth(width)
TK.spines['left'].set_linewidth(width)
TK.spines['top'].set_linewidth(width)
TK.spines['right'].set_linewidth(width)

plt.yticks(fontproperties = 'Arial', size = 8) 
plt.xticks(fontproperties = 'Arial', size = 8, rotation=90)
plt.xlabel('1-Specificity', font=font)
plt.ylabel('Sensitivity', font=font)
plt.title('Cristiano et al. dataset', font=font, y=1)
plt.legend(loc="best", frameon=False, prop=font)
plt.show()