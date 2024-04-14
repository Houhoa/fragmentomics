# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:42:58 2024

@author: hp
"""
import numpy as np
from scipy.stats import spearmanr
length = np.load("length.npy")
length = np.concatenate((length[:,133:347],length[:,421:422]),axis=1)
length_medians = np.median(length, axis=1)

PFE = np.load("PFE.npy")
PFE = np.concatenate((PFE[:,133:347],PFE[:,421:422]),axis=1)
PFE_medians = np.median(PFE, axis=1)

FSR = np.load("FSR.npy")
FSR = np.concatenate((FSR[:,133:347],FSR[:,421:422]),axis=1)
FSR_medians = np.median(FSR, axis=1)

FSD = np.load("FSD.npy")
FSD = np.concatenate((FSD[:,133:347],FSD[:,421:422]),axis=1)
FSD_medians = np.median(FSD, axis=1)

coverage = np.load("coverage.npy")
coverage = np.concatenate((coverage[:,133:347],coverage[:,421:422]),axis=1)
coverage_medians = np.median(coverage, axis=1)

end = np.load("end.npy")
end = np.concatenate((end[:,133:347],end[:,421:422]),axis=1)
end_medians = np.median(end, axis=1)

OCF = np.load("OCF.npy")
OCF = np.concatenate((OCF[:,133:347],OCF[:,421:422]),axis=1)
OCF_medians = np.median(OCF, axis=1)

IFS = np.load("IFS.npy")
IFS = np.concatenate((IFS[:,133:347],IFS[:,421:422]),axis=1)
IFS_medians = np.median(IFS, axis=1)

WPS = np.load("WPS.npy")
WPS = np.concatenate((WPS[:,133:347],WPS[:,421:422]),axis=1)
WPS_medians = np.median(WPS, axis=1)

EDM = np.load("EDM.npy")
EDM = np.concatenate((EDM[:,133:347],EDM[:,421:422]),axis=1)
EDM_medians = np.median(EDM, axis=1)

arrays = [PFE_medians,coverage_medians,end_medians,OCF_medians,IFS_medians,WPS_medians]

correlation_matrix = np.zeros((len(arrays), len(arrays)))

for i in range(len(arrays)):
    for j in range(len(arrays)):
        correlation, _ = spearmanr(arrays[i], arrays[j])
        correlation_matrix[i, j] = correlation



