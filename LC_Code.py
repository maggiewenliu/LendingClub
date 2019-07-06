# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:44:03 2017
@author: Maggie
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import cross_val_score
import multiprocessing
import matplotlib.pyplot as plt


import data_process


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
    

def y_pred(x_test,threshold=0.5):
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
    
    
    # Display heatmap and add decorations
    hm = sns.heatmap(pd_pts, annot=True, fmt="d")
    #annot=True 表示把数字标进每个cell作为注释
    #fmt="d", fmt时表示注释 进每个cell的format， d - double number 数字
    hm.axes.set_title(title)


#%% logistic regression
    
"""
class log_reg():

    #we need to add validation dataset here
    
    # Find binary column method one
    def bool_cols(df,isbool=True):
        bool_cols=[]
        for col in df:
            if isbool==True:
                if df[col].dropna().value_counts().index.isin([0,1]).all():
                    bool_cols.append(col)
            else:
                if not df[col].dropna().value_counts().index.isin([0,1]).all():
                    bool_cols.append(col)
        return bool_cols
    # this above step is to facilitate normalization later
   

        
    # Convenience function to plot confusion matrix

            
    def find_threshold(x_test,y_test):
    
        probs = model.predict_proba(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:, 1])
        
        sensitivity = tpr
        specificity = 1 - fpr
        G = np.sqrt(sensitivity*specificity)
        
        plt.figure(2)
        plt.plot(thresholds,G)
        plt.xlabel('Thresholds')
        plt.ylabel('G-Scores')
        plt.title('G-Scores with different thresholds')
        plt.show()
        
        
        print("The highest G score is %f with threshold at %f" % (np.amax(G),thresholds[np.argmax(G)]) )
        
        return thresholds[np.argmax(G)]
    # this is just testing, we add weight so we don't need to adjust threshold
#%% Run logistic regression
# Process data 
"""


# ---------------Read & Process Data-------------------------


if os.path.isfile("x.pkl") and os.path.isfile("y.pkl"):
    x = pd.read_pickle("x.pkl")
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


# -----------------------------------Machine Learning-------------------------


# Split
x_train, x_test, y_train, y_test = split(x,y,rand=None)



# ---------------Normalization--------------------

not_bi = generate_not_bi(x) #不是dummy variable columns
#Normalize
scaler = sklearn.preprocessing.StandardScaler()  #初始化Normalize library
scaler.fit(x_train[not_bi]) #fit 表示求所有x_train[not_bi] mean variance

x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()

x_train_scaled[not_bi] = scaler.transform(x_train[not_bi]) #Normalize 不是dummy variable column
x_test_scaled[not_bi]  = scaler.transform(x_test[not_bi]) #Normalize 不是dummy variable column



# ---------------Fit Model--------------------
model =  reg(x_train_scaled,y_train)

#画ROC Curve的
ModelValuation(x_test_scaled,y_test,model)

#根据model 计算predict的值
y_predicted = y_pred(x_test_scaled,threshold=0.5)

#算model sensitivity & specificity, G score
spec , G = GetScores(y_test,y_predicted)

#画confusion matrix
confusion(y_test,y_predicted,'Default Confusion Matrix')



"""
#%% Pre selection
init=np.zeros([10,2])

# Process data
zipdata = data_proc.process_zip()
df = data_proc.readcsv()
x , y = data_proc.cleaning(df,zipdata,keep_desc=False,categorical_to_binary=True)

for i in range(10):
       
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
    
    # Fit model
    model = log_reg.reg(x_train_scaled,y_train)
    # Evaluate model
    log_reg.ModelValuation(x_test_scaled,y_test,model)
    y_predicted = log_reg.y_pred(x_test_scaled,threshold=0.5)
    init_spec , init_G = log_reg.GetScores(y_test,y_predicted)
    init[i,0] = init_spec
    init[i,1] = init_G
    log_reg.confusion(y_test,y_predicted,'Default Confusion Matrix')

init_spec = np.mean(init[:,0])
init_G = np.mean(init[:,1])

#%% Feature selection with ablative method
x_stor , y = data_proc.cleaning(df,zipdata,keep_desc=False,categorical_to_binary=False)

categorical_features=['addr_state','application_type','emp_length','grade','home_ownership','initial_list_status','pymnt_plan','sub_grade','term','verification_status']
features=list(x_stor)

times_each_i=10
ablative=np.zeros([len(features)*times_each_i,5])

row=0
for i in range(len(features)):
    
    print(i)
    
    x = x_stor.drop(features[i],inplace=False,axis=1)

    # get_dummies
    if features[i] in categorical_features:
        new_cat = [x for x in categorical_features if not x==features[i]]
    else:
        new_cat = categorical_features
            
    for j in new_cat:
        if j in list(x):
            x[j]=x[j].astype('category')
            x = pd.get_dummies(x,columns={j},drop_first=False)
    for j in range(times_each_i):

        print(j)
        
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
        
        # Fit model
        model = log_reg.reg(x_train_scaled,y_train)
        # Evaluate model
        
        y_predicted = log_reg.y_pred(x_test_scaled,threshold=0.5)
        spec , G = log_reg.GetScores(y_test,y_predicted)
                
        ablative[row,0] = i
        ablative[row,1] = j
        ablative[row,2] = spec
        ablative[row,3] = G
        ablative[row,4] = init_spec - spec
        
        row=row+1
#%%
ab=ablative.copy()        
ab=pd.DataFrame(ab)        
mean=ab.groupby(0)[4].mean()
mean=mean.rename('marginal_contribution')      
#%%
df1=pd.DataFrame(features,columns=['features'])
df2=mean
result=pd.concat([df1,df2],axis=1)
result=result.set_index('features')
result=result.sort_values('marginal_contribution')
result['deprecated']=(result['marginal_contribution']<=0)
useful_features=result.index[result['deprecated']==False].tolist()


#%% use the filter features to fit model
x=x_stor[useful_features]

categorical_features=['addr_state','application_type','emp_length','grade','home_ownership','initial_list_status','pymnt_plan','sub_grade','term','verification_status']
categorical_features=list(set(categorical_features) & set(list(x)) )

for i in categorical_features:
    x[i]=x[i].astype('category')
    x = pd.get_dummies(x,columns=[i],drop_first=False)

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

# Fit model
model = log_reg.reg(x_train_scaled,y_train)

# get score
y_predicted = log_reg.y_pred(x_test_scaled,threshold=0.5)
log_reg.GetScores(y_test,y_predicted)
log_reg.confusion(y_test,y_predicted,'Default Confusion Matrix')

# Get coef
df1 = pd.DataFrame(list(x),columns=['features'])
df2 = pd.DataFrame(np.transpose(model.coef_),columns=['coef'])
df3 = pd.DataFrame(abs(np.transpose(model.coef_)),columns=['coef_abs'])
coefficients = pd.concat([df1,df2,df3], axis = 1)
coefficients = coefficients.sort_values(by='coef_abs',ascending=False)
#%% graph

index = np.arange(len(coefficients))
bar_width = 0.35

plt.bar(index, coefficients['coef_abs'])
plt.xticks(index + bar_width / 2, coefficients['features'],rotation=-90)

#%% horizontal graph

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
y_pos = np.arange(len(coefficients['coef_abs']))
bar_width = 0.35

ax.barh(y_pos + bar_width / 2, coefficients['coef_abs'], align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(coefficients['features'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('coef_abs')
ax.set_title('Absolute coefficient for each feature')

plt.show()

#%% Find the best threshold, should be run after previous section
    
best_threshold = log_reg.find_threshold(x_test_scaled,y_test)
# Evaluate model
y_predicted = log_reg.y_pred(x_test_scaled,threshold=best_threshold)
log_reg.GetScores(y_test,y_predicted)
log_reg.confusion(y_test,y_predicted,'Default Confusion Matrix')

#%% cross validate

x=x_stor[useful_features]
x=x_stor

categorical_features=['addr_state','application_type','emp_length','grade','home_ownership','initial_list_status','pymnt_plan','sub_grade','term','verification_status']
categorical_features=list(set(categorical_features) & set(list(x)) )

for i in categorical_features:
    x[i]=x[i].astype('category')
    x = pd.get_dummies(x,columns={i},drop_first=False)

def scr(y_test, y_predicted): 
    
    CM = metrics.confusion_matrix(y_test, y_predicted)
    TN = CM[0,0]
    FN = CM[1,0]
    TP = CM[1,1]
    FP = CM[0,1]
        
    sensitivity = float(TP)/float(TP+FN)
    specificity = float(TN)/float(TN+FP)
    G = np.sqrt(sensitivity*specificity)
    return G

score = metrics.make_scorer(scr, greater_is_better=True)
model = LogisticRegression(penalty='l2',class_weight='balanced',solver='sag',n_jobs=-1)
scores = cross_val_score(model, x, y, cv=3, n_jobs=-1 ,scoring = 'accuracy')
scores

#%% SVM
from sklearn import svm
from sklearn import preprocessing

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
        
        
        
        
        
        
        
