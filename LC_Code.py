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
from sklearn import metrics
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import multiprocessing
import matplotlib.pyplot as plt

os.chdir(r"/Users/beckswu/Desktop/Lending Club/For Maggie")


#This is a sample change

def process_zip():
    zip_data=pd.read_csv('ZIP.csv')
    #zip_data['Zip']=zip_data['Zip'].astype(str)
    
    #转化string, 只存zip的前三位fsa
    zip_data['Zip'] = zip_data['Zip'].apply(lambda x: (('0' if x<10000 else '')+ str(x))[:3]) 
    
    #去掉'Median','Mean','Pop' 数据中的逗号, 比如'52,234' 变成 ‘52234’
    zip_data[['Median','Mean','Pop']]=zip_data[['Median','Mean','Pop']].apply(lambda x: x.str.replace(',',''))
    
    #把 'Median','Mean','Pop' 从string 变成 double value
    for i in ['Median','Mean','Pop']:
        zip_data.loc[:,i]=pd.to_numeric(zip_data.loc[:,i],errors='coerce')
        
    #zip_data.groupby('Zip') 是把同一个zip 的group 到一起变成，tuple, tuple[0]是三位数的zip， tuple[1] 是属于zip的dataframe, 
    #  比如原来一行 原zip 是 8990, 另一行 原zip 是 8991, 那么这两行同时放进group 后同一个tuple, tuple[0] = 089, tuple[1] 包含这两行的数据
    #  把每个tuple[1]的population 相加得到sum, 用tansform的作用是让它格式get back to original dataframe, 
    # 比如原来原zip 是 8990, zip = 089, zip_data.groupby('Zip')['Pop'] 会生成一个sum, 在这行，表示所有089的sum
    
    
    zip_data['weight']=zip_data['Pop']/zip_data.groupby('Zip')['Pop'].transform(sum)

    #乘以weight
    zip_data['new_mean']=zip_data['Mean']*zip_data['weight']
    
    #乘以new_median 
    zip_data['new_median']=zip_data['Median']*zip_data['weight']
    
    zip_new=pd.DataFrame()
    
    """
    现在zip 格式是
    
     Zip  Median     Mean    Pop    weight      new_mean    new_median
0    010   56663  66688.0  16445  0.036549   2437.373812   2437.373812  
1    010   49853  75063.0  28069  0.062383   4682.668653   3109.988681
    
    
    """
    
    
    
    #根据zip groupby，把同一个zip的文件的new_mean, new_median 都sum 起来
    zip_new=zip_data.groupby('Zip')['new_mean','new_median'].sum()
    
    """
    现在zip_new 格式是， 注意现在index 是zip, 而不是数字了
    
        new_mean     new_median
Zip                              
010   68997.226192   57290.974026
011   54979.626027   42377.127888
012   65706.935838   50845.603610
    """
    #zip key is zip
    return zip_new





def readcsv():

    csv_list=['LoanStats3a_securev1.csv', 'LoanStats3b_securev1.csv', 
              'LoanStats3c_securev1.csv','LoanStats3d_securev1.csv']
   
    df=pd.DataFrame()           
    for i in csv_list:
       cur = pd.read_csv(i, encoding = "ISO-8859-1", low_memory=False) #不用管encoding
       df = pd.concat([df,cur])
    return df    












        
def cleaning(df,zip_new,keep_desc=True,categorical_to_binary=True):
    #drop the observation that was missing for ALL field
    #python 中axis = 0表示列， axis = 1 表示行
    #how = ‘any’ : If any NA values are present, drop that row or column.
    #how = ‘all’ : If all values are NA, drop that row or column.
    df=df.dropna(axis=0,how='all') #当一行都是na，drop
    
    
   
    #drop the meaningless features
    #inplace=True 表示在df中操作，如果不在df中操作, inplace=False is passed 表示在df copy中操作，不影响df
    df.drop(['id', 'last_pymnt_d','url','earliest_cr_line', 'emp_title','issue_d','last_credit_pull_d','purpose','title','hardship_flag','policy_code'],inplace=True,axis=1,errors='ignore')
    #drop features that have confilicting meaning for y
    df.drop(['total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt'],inplace=True,axis=1,errors='ignore')
    #drop the features that have nothing
    df.dropna(inplace=True,axis=1,how='all')
    
    #pick up description
    desc=df['desc']
    #drop the features for which greater than 10% of the loans were missing data for
    #如果缺失的数据多于整个dataframe的row 的10%， 丢掉
    num_rows=df.count(axis=0)
    df=df.iloc[:,(num_rows>=0.9*len(df)).tolist()]
    #merge back desc
    if keep_desc==True:
        df=pd.concat([df,desc],axis=1)

    #drop the observation that was missing for any field
    df=df.dropna(axis=0,how='any')
    
    #deal with percentage mark
    df['int_rate']=df['int_rate'].replace('%','',regex=True).astype('float')/100
    df['revol_util']=df['revol_util'].replace('%','',regex=True).astype('float')/100
    
    #dealing with categorical features
    if categorical_to_binary==True:
        categorical_features=['addr_state','application_type','emp_length','grade','home_ownership','initial_list_status','pymnt_plan','sub_grade','term','verification_status']
        for i in categorical_features:
            if i in list(df): #list(df) 就是 df.columns
                df[i]=df[i].astype('category')
                df=pd.get_dummies(df,columns={i},drop_first=True)
    #get dummies drop first 是只弄k-1个variable 把第1个variable 去掉
    
    
    #merge zipcode with census data
    df['zip_code']=df['zip_code'].apply(lambda x: x[:3])
    df=df.join(zip_new,on='zip_code')
    df.drop('zip_code',inplace=True,axis=1)
    #drop the observation that was missing for any field
    df=df.dropna(axis=0,how='any')
        
    #label the dataset to create y, 建立0，1 variables
    y=df['loan_status'].replace(['Charged Off','Does not meet the credit policy. Status:Charged Off','Late (31-120 days)','In Grace Period','Late (16-30 days)','Default'],0)
    y=y.replace(['Fully Paid','Does not meet the credit policy. Status:Fully Paid','Current'],1)
    df=df.drop(['loan_status'],axis=1)    
    
    return df,y


"""
class data_proc():
    


#%% logistic regression
class log_reg():
    # Evaluate the model by splitting into train and test sets
    def split(x,y,rand=0):
        
        y = np.ravel(y)
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,random_state=rand)
        
        return x_train, x_test, y_train, y_test 
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
    # method two
    def not_bi(x): #找到不是dummy variable的columns
        not_bi=[]
        for i in list(x):
            u=x[i].unique()
            if not (0 in u and 1 in u and len(u)==2): #if not binary
                not_bi.append(i)
        return not_bi
    
    def reg(x_train, y_train):
           
        model = LogisticRegression(penalty='l2',class_weight='balanced',solver='sag',n_jobs=-1)
        
        
        #Why we need standardize?
        
       # Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features 
       ## with approximately the same scale. You can preprocess the data with 
       # a scaler from sklearn.preprocessing.
       
        
        model = model.fit(x_train, y_train)
        
        return model
    
    def ModelValuation(x_test,y_test,model):
        
        probs = model.predict_proba(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:, 1])
        
        plt.figure(1)
        plt.plot(fpr, tpr, label='LogisticRegression')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
        
        print("Area Under the Curve (AUC) from prediction score is %f" % metrics.roc_auc_score(y_test, probs[:, 1]))
    
        return None  
    
    def y_pred(x_test,threshold=0.5):
        
        if threshold == 0.5:
            y_predicted = model.predict(x_test) #generate output class/label
        else:
            probs = model.predict_proba(x_test) #generate output probability
            #The first index refers to the probability that the data belong to class 0, and the second refers to the probability that the data belong to class 1.
            y_predicted = np.array(probs[:,1] >= threshold).astype(int)
        
        return y_predicted    
    
    def GetScores(y_test,y_predicted):
        #G means score 
        CM = metrics.confusion_matrix(y_test, y_predicted)
        TN = CM[0,0]
        FN = CM[1,0]
        TP = CM[1,1]
        FP = CM[0,1]
        
        sensitivity = float(TP)/float(TP+FN)
        specificity = float(TN)/float(TN+FP)
        G = np.sqrt(sensitivity*specificity)
        print("G score is %f" % G)
        print("Specificity is %f" % specificity)
        
        # Generate and display different evaluation metrics
        print("Mean accuracy score is %f" % metrics.accuracy_score(y_test, y_predicted))
          
        print("Confusion Marix")
        print(CM)
        
        return specificity , G
        
    # Convenience function to plot confusion matrix
    def confusion(y_test,y_predicted,title):
        
        # Define names for the three Iris types
        names = ['Default', 'Not Default']
    
        # Make a 2D histogram from the test and result arrays
        pts, xe, ye = np.histogram2d(y_test, y_predicted, bins=2)
    
        # For simplicity we create a new DataFrame
        pd_pts = pd.DataFrame(pts.astype(int), index=names, columns=names )
        
        # Display heatmap and add decorations
        hm = sns.heatmap(pd_pts, annot=True, fmt="d")
        hm.axes.set_title(title)
        
        return None
            
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

if os.path.isfile("zipdata.pkl"):
    zipdate = pd.read_pickle("zipdata.pkl")
else:
    zipdata = process_zip()
    zipdata.to_pickle("zipdata.pkl")

if os.path.isfile("loan.pkl"):
    df = pd.read_pickle("loan.pkl")
else:
    df =  readcsv()
    df.to_pickle("loan.pkl")

x , y = cleaning(df,zipdata,keep_desc=False,categorical_to_binary=True)


"""

#x_=x.copy()
#y_=y.copy()

# Split
x_train, x_test, y_train, y_test = log_reg.split(x,y,rand=None)
# Normalize
not_bi = log_reg.not_bi(x) #不是dummy variable columns
scaler = StandardScaler() #normalization
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
spec , G = log_reg.GetScores(y_test,y_predicted)
log_reg.confusion(y_test,y_predicted,'Default Confusion Matrix')

"""

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
        
        
        
        
        
        
        
