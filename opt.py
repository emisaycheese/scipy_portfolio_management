# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np

import os

from sklearn.metrics import roc_auc_score
from scipy import optimize


lsts = os.listdir()[:-1]
all_csv = pd.DataFrame()


for l in lsts:
    
    all_csv[l] = pd.read_csv(l).iloc[:,1]

all_csv = all_csv.to_numpy()



def auc(x, matrix,label):
    
    cols = matrix.shape[1]
    
    f=0
    
    for c in range(cols):
        f+=roc_auc_score(label, matrix[:,c])
    
    return -f


def optimal_auc_max(matrix,label):
    x = np.array([1/matrix.shape[1]]*matrix.shape[1])
    cons = ({'type': 'eq',
             'fun': lambda x: np.sum(x) - 1})
    
    bounds = [(0, 1) for i in range(len(x))]

        
    minimize = optimize.minimize(auc, x, args=(matrix,label), bounds=bounds,
                                 constraints=cons)
    return minimize


label = np.array([1]*506691)
label[0]=0

opt = optimal_auc_max(all_csv, label)

print(opt.x)


