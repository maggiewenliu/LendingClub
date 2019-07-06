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
import tqdm

from sklearn.model_selection import cross_val_score
import multiprocessing
import matplotlib.pyplot as plt
import sklearn.linear_model

import data_process
import logistic_regression


    
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
    
    
#%% Process data 
"""


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
#%%  Logistic Regression方法一： 最普通的Logistic Regression


"""



# -----------------------------------Machine Learning-------------------------


# Split
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

#F1-score
print("F1 score is {}".format(sklearn.metrics.f1_score(y_test,y_predicted)))


"""








#%% Pre selection

#Logistic Regression方法二： run 10 logistic regression, 然后看specificity & sensitivity 的平均值



"""
init=np.zeros([10,3]) #产生一个two dimensional 的np.array, size = 10


for i in range(10):
       
    # Split
    x_train, x_test, y_train, y_test = logistic_regression.split(x,y,rand=None)
    # Normalize
    
    # ---------------Normalization--------------------
    not_bi = logistic_regression.generate_not_bi(x)
    scaler =  sklearn.preprocessing.StandardScaler()
    scaler.fit(x_train[not_bi]) 
    
    x_train_scaled=x_train.copy()
    x_test_scaled=x_test.copy()
    
    x_train_scaled[not_bi] = scaler.transform(x_train[not_bi])
    x_test_scaled[not_bi]  = scaler.transform(x_test[not_bi])
    
    
    
    # Fit model
    model = logistic_regression.reg(x_train_scaled,y_train)
    # Evaluate model, 画ROC Curve的
    logistic_regression.ModelValuation(x_test_scaled,y_test,model)
    
    #根据model 计算predict的值
    y_predicted = logistic_regression.y_pred(x_test_scaled,model,threshold=0.5)
    
    #算model sensitivity & specificity, G score
    init_spec , init_G = logistic_regression.GetScores(y_test,y_predicted)
    init[i,0] = init_spec
    init[i,1] = init_G
    init[i,2] = sklearn.metrics.f1_score(y_test,y_predicted)
    
    #画confusion matrix
    logistic_regression.confusion(y_test,y_predicted,'Default Confusion Matrix')

init_spec_average = np.mean(init[:,0])
init_G_average  = np.mean(init[:,1])
f1_average = np.mean(init[:,2])

print("average specificity {0}, average G_Score {1}, average F1-score {2}".\
      format(init_spec_average, init_G_average,f1_average))


f = open("average_specificity.txt", "w")
f.write(str(init_spec_average))
f.close()


"""








#%% Feature selection with ablative method



##Logistic Regression方法三：这种方法是： 
    #每次drop 一个feature， 然后如果这个feature 是category的

# 方法三需要方法二中的 average specificity: 方法二的specificity 是所有variable的

if os.path.isfile("average_specificity.txt"):
    f = open("average_specificity.txt", "r")
    number = f.read()
    avrg_speci = float(number)
    f.close()
    
else:
    print("please run Logisitic Regression 方法二 first ")
    exit



x_original = x_has_category.copy()#因为后面会重命名 x， 所以给原始的x 一个新的名字

#这是所有category的数据

features=list(x_original.columns) #列出数据所有的column

times_each_i=1 #代表每个选择的feature集合 run 10次
ablative=np.zeros([len(features)*times_each_i,5]) #ablative 用来记录每次选择feature的performane

row=0 #用来记录ablative 现在在哪行

for i, feature in tqdm.tqdm(enumerate(features)):
    #每次drop 一个variable，
    
    x = x_original.drop(feature,inplace=False,axis=1)

    categorical_features=set(['addr_state','application_type','emp_length','grade','home_ownership','initial_list_status','pymnt_plan','sub_grade','term','verification_status'])

    if feature in categorical_features:
        categorical_features.discard(feature)

    #把category column 变成Category的格式
    #因为categorical_features 是set 格式的，我们要把它变成list 的形式，加list ()
    x[list(categorical_features)] = x[list(categorical_features)].apply(pd.Categorical)
    
    #把category column 生成dummy variable
    x=pd.get_dummies(x,columns=list(categorical_features),drop_first=True)
     #产生dummy variable 的同时，会drop 掉原来的column
    
    for j in range(times_each_i):

        print(j)
        
        # Split
        x_train, x_test, y_train, y_test = logistic_regression.split(x,y,rand=None)
        
        # Normalize
        not_bi = logistic_regression.generate_not_bi(x)
        scaler =  sklearn.preprocessing.StandardScaler()
        scaler.fit(x_train[not_bi]) 
        
        x_train_scaled=x_train
        x_test_scaled=x_test
        
        x_train_scaled[not_bi] = scaler.transform(x_train[not_bi])
        x_test_scaled[not_bi]  = scaler.transform(x_test[not_bi])
        
        # Fit model
        model = logistic_regression.reg(x_train_scaled,y_train)
        # Evaluate model
        
        y_predicted = logistic_regression.y_pred(x_test_scaled,model, threshold=0.5)
        spec , G = logistic_regression.GetScores(y_test,y_predicted)
                
        ablative[row,0] = i
        ablative[row,1] = j
        ablative[row,2] = spec
        ablative[row,3] = G
        ablative[row,4] = avrg_speci - spec
        
        row=row+1


"""

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
            #drop first表示比如 有a, b, c 三个category, 但dummy drop a， 只生成 b, c 两个 dummy variable
                
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
        
        
        
        
        
        
        
