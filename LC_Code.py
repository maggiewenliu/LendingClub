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


#zipdata = data_proc.process_zip()
#df = data_proc.readcsv()
#x , y = data_proc.cleaning(df,zipdata,keep_desc=False,categorical_to_binary=True)

#%%  Logistic Regression方法一： 最普通的Logistic Regression






# -----------------------------------Machine Learning-------------------------

"""
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

# -----------------------------------方法 二  ------------------------------


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

# -----------------------------------方法 三  ------------------------------


need_run_selection = True #是不是需要run feature selection
#因为很费时间，如果已经run好，直接读数据，不用重新run

##Logistic Regression方法三：这种方法是： 
    #每次drop 一个feature， 然后如果这个feature 是category的, 也不生成dummy variable
        #， run logistic regression， 得到drop variable的一个specificity的difference 的值
    #然后根据这个值选出我们想要的variable（有16个），然后再run 一个logistics regression

# 方法三需要方法二中的 average specificity: 方法二的specificity 是所有variable的

if os.path.isfile("average_specificity.txt"):
    f = open("average_specificity.txt", "r")
    number = f.read()
    avrg_speci = float(number)
    f.close()
    
else:
    print("please run Logisitic Regression 方法二 first ")
    exit


if need_run_selection or os.path.isfile("useful_feature.txt"): 
    
    x_original = x_has_category.copy()#因为后面会重命名 x， 所以给原始的x 一个新的名字
    
    #这是所有category的数据
    
    features=list(x_original.columns) #列出数据所有的column
    
    times_each_i=10 #代表每个选择的feature集合 run 10次
    ablative=np.zeros([len(features)*times_each_i,5]) #ablative 用来记录每次选择feature的performane
    
    row=0 #用来记录ablative 现在在哪行
    
    for i, feature in tqdm.tqdm(enumerate(features)):
        #每次drop 一个variable，
        print("current Feature",feature)
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
    
            print("case",j)
            
            # Split
            x_train, x_test, y_train, y_test = logistic_regression.split(x,y,rand=None)
            
            # Normalize
            not_bi = logistic_regression.generate_not_bi(x)
            scaler =  sklearn.preprocessing.StandardScaler()
            scaler.fit(x_train[not_bi]) 
            
            x_train_scaled=x_train.copy()
            x_test_scaled=x_test.copy()
            
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
      
    ab=pd.DataFrame(ablative.copy()) #需要 ablative.copy(）, 否则 更改 ab 也会更改 ablative    
    mean=ab.groupby(0)[4].mean() #生成一个dataframe 以去掉每个variable 的集合 groupby 求avrg_speci-specificity的平均值
    mean=mean.rename('marginal_contribution') #重新命名column的名字
    
    
    df1=pd.DataFrame(features,columns=['features'])
    result=pd.concat([df1,mean],axis=1)
    
    #result 是dataframe 是取掉的variable 和 avearge specificity 和 specifity 的mean,
    #e,g
    #   features         means
    #  acc_now_delinq -0.00380852683686983
    #  addr_state    0.0027713851171323745
    #  annual_inc    0.0030456075697351537
    
    
    result=result.set_index('features')
    result=result.sort_values('marginal_contribution') #根据marginal_contribution 来sortmarginal_contribution
    
    
    result['deprecated']=(result['marginal_contribution']<=0)
    #生成一个column 看 marginal_contribution(average difference) 是不是小于0
    #e,g
    #   features         marginal_contribution  deprecated
    #  acc_now_delinq -0.00380852683686983     True
    #  addr_state    0.0027713851171323745    False
    #  annual_inc    0.0030456075697351537   False
    
    useful_features=result.index[result['deprecated']==False].tolist()
    #把有用的feature 跳出来，
    
    #把useful_feature output出去
    outfile =  open("useful_feature.txt","w")
    outfile.write(",".join(useful_features)) 
    #",".join(useful_features) 表示把list中每个string 用"," 连在一起 变成一个大string
    outfile.close()

else:
    read_file =  open("useful_feature.txt","r")
    useful_feature = read_file.read().split(',') 
    #read_file.read() 生成一个大string，把attribute 用逗号隔开, e.g. 'out_prncp,revol_bal,.....'
    #.split(',') 把string 用逗号分开变成list 
    read_file.close()
    


#------------------------根据选出的feature，再run logistic regression---------------------------------

x=x_original[useful_features]

categorical_features=set(['addr_state','application_type','emp_length','grade','home_ownership','initial_list_status','pymnt_plan','sub_grade','term','verification_status'])
#选取只有useful_features中出现有用的feature

categorical_features=list( categorical_features & set(list(x)) )
#list(x) 表示 x.columns 
# categorical_features & set(list(x)) 表示两个set 取交集

#建立dummy variable
x = pd.get_dummies(x,columns=categorical_features,drop_first=False)

# Split
x_train, x_test, y_train, y_test = logistic_regression.split(x,y,rand=None)
# Normalize
not_bi = logistic_regression.generate_not_bi(x)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train[not_bi]) 

x_train_scaled=x_train
x_test_scaled=x_test

x_train_scaled[not_bi] = scaler.transform(x_train[not_bi])
x_test_scaled[not_bi]  = scaler.transform(x_test[not_bi])

# Fit model
model = logistic_regression.reg(x_train_scaled,y_train)

# get score
y_predicted = logistic_regression.y_pred(x_test_scaled,model,threshold=0.5)
logistic_regression.ModelValuation(x_test_scaled,y_test,model)
logistic_regression.GetScores(y_test,y_predicted)
logistic_regression.confusion(y_test,y_predicted,'Default Confusion Matrix')

#------------------------找到最理想的threshold---------------------------------


best_threshold = logistic_regression.find_threshold(x_test_scaled,y_test,model)
# Evaluate model, 用刚选取的threshold 来test model
y_predicted = logistic_regression.y_pred(x_test_scaled,model, threshold=best_threshold)
logistic_regression.GetScores(y_test,y_predicted)
logistic_regression.confusion(y_test,y_predicted,'Default Confusion Matrix')


print("F1 score is {}".format(sklearn.metrics.f1_score(y_test,y_predicted)))




# ---------------------------coef 没有什么用处--------------------------------------------

# Get coef
df1 = pd.DataFrame(list(x),columns=['features'])
df2 = pd.DataFrame(np.transpose(model.coef_),columns=['coef'])
#np.transpose 是因为 model.coef_是1*74, 我们想让它变成74*1 dataframe

df3 = pd.DataFrame(abs(np.transpose(model.coef_)),columns=['coef_abs'])

coefficients = pd.concat([df1,df2,df3], axis = 1)
# coffeicient 是
#   features         coffeicient    coffeicient绝对值
#  acc_now_delinq       -5               5
#  addr_state          10.2             10.2
#  annual_inc           -3.78            3.78

#根据coef_absolute value 来sort dataframe
coefficients = coefficients.sort_values(by='coef_abs',ascending=False)



# ---------------------------——画图, 画出coefficient 没有什么用处----------------------
index = np.arange(len(coefficients))#生成一个np.array([1,2,3....,74])
bar_width = 0.35


plt.bar(index, coefficients['coef_abs'])
plt.xticks(index + bar_width / 2, coefficients['features'],rotation=-90)


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





"""



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
"""
        
        
        
        
        
        
        
