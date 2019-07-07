#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 14:40:58 2019

@author: Maggie
"""

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import sklearn.linear_model

# Evaluate the model by splitting into train and test sets
def split(x,y,rand=0):
    
    #Y 是 pands. Series, 有index 的, np.ravel 就会把Series值抽取然后排开生成一个np.array
    """
        ind     Y
        0       1    
        1       1
        2       1
        
        np.ravel(y)  = [1,1,1]
    
    """
    
    y = np.ravel(y) 
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.25,random_state=rand)
    
    return x_train, x_test, y_train, y_test 


 # method two
def generate_not_bi(x): #找到所有,不是dummy variable的columns
    not_bi=[]
    for i in list(x): #list(x) 是所有columns 的名字，i是现在这个column的名字
        u=x[i].unique()
        if not (0 in u and 1 in u and len(u)==2): #if not binary
            not_bi.append(i)
    return not_bi


def reg(x_train, y_train):
           
        model = sklearn.linear_model.LogisticRegression(penalty='l2',class_weight='balanced',solver='sag',n_jobs=-1)
        #penalty ='l2' 是regularization 就是视频里看的regularization
        
        
        #Why we need standardize?
        
       # Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features 
       ## with approximately the same scale. You can preprocess the data with 
       # a scaler from sklearn.preprocessing.
      
        model = model.fit(x_train, y_train)
        
        return model
    
    
def ModelValuation(x_test,y_test,model):
    
    probs = model.predict_proba(x_test)
    #predict_proba 是返回estimates for all classes are ordered by the label of classes.
    #因为分类只有0 和 1，所以probs 的shape 是 m * 2, m是x_test数量，每行2个表示预测0和1的概率，加一起等于1
    #比如[0.237, 0.763]， 表示对于这个test_case, 预测0的概率是0.237， 预测1的概率是0.763

    
    #ROC 是true_positive_rate (sensitivity) 在y轴 vs false negative rate (= 1 - specificity) 在x轴
    
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, probs[:, 1])
    # y_test 真是test cast label, probs[:, 1] #我们model预测1的概率probs[:, 1]
    #fpr false positive rate, tpr true positive rate, 两个都是np.array()
    # thresholds: threshold 用来judge true & false positive rate
    
    plt.figure(1)
    plt.plot(fpr, tpr, label='LogisticRegression')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    
    print("Area Under the Curve (AUC) from prediction score is %f" % sklearn.metrics.roc_auc_score(y_test, probs[:, 1]))

    return None      
    

def y_pred(x_test,model,threshold=0.5):
    #用来计算y_predict的值，
    
    if threshold == 0.5:
        y_predicted = model.predict(x_test) #generate output class/label
        #model.predict default 的threshold 是0
        
    else:
        probs = model.predict_proba(x_test) #generate output probability
        #The first index refers to the probability that the data belong to class 0, and the second refers to the probability that the data belong to class 1.
        y_predicted = np.array(probs[:,1] >= threshold).astype(int)
    
    return y_predicted  





    
def GetScores(y_test,y_predicted):
    #y_test 是真实的label
    #y_predicted 是model 预测出来的值
    
    
    #G means score 是paper 中提到的
    CM =  sklearn.metrics.confusion_matrix(y_test, y_predicted)
    TN = CM[0,0]
    FN = CM[1,0]
    TP = CM[1,1]
    FP = CM[0,1]
    
   #confusion_matrix,  the count of true negatives is [0,0] ,  the count of false negatives is [0,1] 
   #  the count of true positives is [1,1]  and  the count of false positives is [0,1].


    
    sensitivity = float(TP)/float(TP+FN) #true positive rate, recall, 预测为1的 占所有应该预测为1 的概率
    specificity = float(TN)/float(TN+FP) #true negative rate, 预测为0的 占所有应该预测为1 的概率
    
    G = np.sqrt(sensitivity*specificity)
    print("G score is %f" % G)
    print("Specificity is %f" % specificity)
    
    # Generate and display different evaluation metrics
    # accuracy_score 是表示预测对的 占所有test的比例， 预测对的表示 model 预测0， 真实也是0，model 预测1， 真实也是1
    print("Mean accuracy score is %f" %  sklearn.metrics.accuracy_score(y_test, y_predicted))
      
    print("Confusion Marix")
    print(CM)
    
    return specificity , G


def confusion(y_test,y_predicted,title):
    
    # Define names for the three Iris types
    names = ['Default', 'Not Default']

    # Make a 2D histogram from the test and result arrays
    pts, xe, ye = np.histogram2d(y_test, y_predicted, bins=2)
    #xe 是The bin edges along the first dimension.
    #xe 是The bin edges along the second dimension.
    #pts 是与confusion_matrix产生结果是相同的, 是two dimension array
    
    #pts: The bi-dimensional histogram of samples x and y. Values in x are histogrammed along the first dimension and values in y are histogrammed along the second dimension.

    # For simplicity we create a new DataFrame,转化成 dataframe
    pd_pts = pd.DataFrame(pts.astype(int), index=names, columns=names )
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot()
    
    
    # Display heatmap and add decorations
    hm = sns.heatmap(pd_pts, annot=True, fmt="d",ax=ax1)
    #annot=True 表示把数字标进每个cell作为注释
    #fmt="d", fmt时表示注释 进每个cell的format， d - double number 数字
    hm.axes.set_title(title)
    plt.show()





def find_threshold(x_test,y_test, model):
    
    #找到最后的threshold

    probs = model.predict_proba(x_test)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, probs[:, 1])
    
    
    sensitivity = tpr #sensitivity 是np.array, size 与fpr, tpr,thresholds 一样
    specificity = 1 - fpr #specificity 是np.array, size 与fpr, tpr,thresholds 一样
    G = np.sqrt(sensitivity*specificity)#GG 是Np.array，size 与fpr, tpr,thresholds 一样 
    
    plt.figure(2)
    plt.plot(thresholds,G)
    plt.xlabel('Thresholds')
    plt.ylabel('G-Scores')
    plt.title('G-Scores with different thresholds')
    plt.show()
    
    
    print("The highest G score is %f with threshold at %f" % (np.amax(G),thresholds[np.argmax(G)]) )
    
    #np.argmax(G), 找到最大数的index
    #比如G = np.array([5,6,1,2]), np.argmax(G) = 1 (因为6最大)
    return thresholds[np.argmax(G)]