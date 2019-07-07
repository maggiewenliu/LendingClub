#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 20:58:39 2019

@author: beckswu
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import tqdm

from sklearn.model_selection import cross_val_score
import multiprocessing
import matplotlib.pyplot as plt
import sklearn.linear_model

import data_process
import logistic_regression

    
from sklearn import svm
from sklearn import preprocessing



# ---------------Read & Process Data-------------------------


if os.path.isfile("x.pkl") and os.path.isfile("y.pkl") and os.path.isfile("x_has_category.pkl"):
    x = pd.read_pickle("x.pkl")
    x_has_category = pd.read_pickle("x_has_category.pkl")
    y  = pd.read_pickle("y.pkl")
else:
    if os.path.isfile("zipdata.pkl"):
        zipdata = pd.read_pickle("zipdata.pkl")
    else:
        zipdata = data_process.process_zip()
        zipdata.to_pickle("zipdata.pkl")
    
    if os.path.isfile("loan.pkl"):
        df = pd.read_pickle("loan.pkl")
    else:
        df =  data_process.readcsv()
        df.to_pickle("loan.pkl")
    
    x , y = data_process.cleaning(df,zipdata,keep_desc=False,categorical_to_binary=True)
    x.to_pickle("x.pkl")
    y.to_pickle("y.pkl")
    
    x_has_category , y = data_process.cleaning(df,zipdata,keep_desc=False,categorical_to_binary=False)
    x_has_category.to_pickle("x_has_category.pkl")
    

# Split, split 方法跟logistic regression 一样的
x_train, x_test, y_train, y_test = logistic_regression.split(x,y,rand=None)



# ---------------Normalization--------------------
not_bi = logistic_regression.generate_not_bi(x) #不是dummy variable columns
#Normalize
scaler = sklearn.preprocessing.StandardScaler()  #初始化Normalize library
scaler.fit(x_train[not_bi]) #fit 表示求所有x_train[not_bi] mean variance

x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()

x_train_scaled[not_bi] = scaler.transform(x_train[not_bi]) #Normalize 不是dummy variable column
x_test_scaled[not_bi]  = scaler.transform(x_test[not_bi]) #Normalize 不是dummy variable column





#SVM model needs normalization

model = sklearn.svm.SVC() #初始化
model.fit(x_train_scaled, y_train)



# get score
y_predicted = logistic_regression.y_pred(x_test_scaled, model,threshold=0.5)
logistic_regression.GetScores(y_test,y_predicted)
logistic_regression.confusion(y_test,y_predicted,'Default Confusion Matrix')



"""
# ---------------Fit Model--------------------
model =  logistic_regression.reg(x_train_scaled,y_train)

#画ROC Curve的
logistic_regression.ModelValuation(x_test_scaled,y_test,model)

#根据model 计算predict的值
y_predicted = logistic_regression.y_pred(x_test_scaled,model,threshold=0.5)

#算model sensitivity & specificity, G score
spec , G = logistic_regression.GetScores(y_test,y_predicted)

#画confusion matrix
logistic_regression.confusion(y_test,y_predicted,'Default Confusion Matrix')
"""
#F1-score
print("F1 score is {}".format(sklearn.metrics.f1_score(y_test,y_predicted)))    
    
    


""" 
# Process data
        
zipdata = data_proc.process_zip()
df = data_proc.readcsv()
x , y = data_proc.cleaning(df,zipdata,keep_desc=False,categorical_to_binary=True)
#x_=x.copy()
#y_=y.copy()

# Split
x_train, x_test, y_train, y_test = log_reg.split(x,y,rand=None)
# Normalize
not_bi = log_reg.not_bi(x)
scaler = StandardScaler()
scaler.fit(x_train[not_bi]) 

x_train_scaled=x_train
x_test_scaled=x_test

x_train_scaled[not_bi] = scaler.transform(x_train[not_bi])
x_test_scaled[not_bi]  = scaler.transform(x_test[not_bi])





#SVM model needs normalization

model = svm.SVC()
model.fit(x_train_scaled, y_train)

# get score
y_predicted = log_reg.y_pred(x_test_scaled,threshold=0.5)
log_reg.GetScores(y_test,y_predicted)
log_reg.confusion(y_test,y_predicted,'Default Confusion Matrix')
#%%



k=['linear','polynomial','rbf','sigmoid ']

def f1(i):
    model = svm.SVC(kernel=k[i])
    model.fit(x_train_scaled, y_train)
    GetScores(x_test_scaled,y_test,model)
def f2(i):
    model = svm.SVC(kernel=k[i])
    model.fit(x_train_scaled, y_train)
    GetScores(x_test_scaled,y_test,model)
def f3(i):
    model = svm.SVC(kernel=k[i])
    model.fit(x_train_scaled, y_train)
    GetScores(x_test_scaled,y_test,model)
def f4(i):
    model = svm.SVC(kernel=k[i])
    model.fit(x_train_scaled, y_train)
    GetScores(x_test_scaled,y_test,model)    

threads=[]
t1=multiprocessing.Process(target=f1(0),daemon=True)
threads.append(t1)
t2=multiprocessing.Process(target=f2(1),daemon=True)
threads.append(t2)
t3=multiprocessing.Process(target=f3(2),daemon=True)
threads.append(t3)
t4=multiprocessing.Process(target=f4(3),daemon=True)
threads.append(t4)


if __name__ == '__main__':
    for i in threads:
        i.start()
    for i in threads:
        i.join()



#for i in kernel:
#    model = svm.SVC(kernel=i)
#    model.fit(x_train, y_train)
#    GetScores(x_test,y_test,model)

"""        
            
    
    
    