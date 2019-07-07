#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 00:14:03 2019

@author: beckswu
"""

import os
import pandas as pd


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












        
def cleaning(df,zipdata,keep_desc=True,categorical_to_binary=True):
    #drop the observation that was missing for ALL field
    # drop 一整行都丢失的数据
    #python 中axis = 0 along the row， axis = 1 along the column
    #how = ‘any’ : If any NA values are present, drop that row or column.
    #how = ‘all’ : If all values are NA, drop that row or column.
    df=df.dropna(axis=0,how='all').copy() #当一行都是na，drop
    #这么做的目的是因为有的df， read from csv, csv中间有很多空行
    
    
    
    #---------------------------去掉列------------------------------
   
    #drop the meaningless features
    #inplace=True 表示在df中操作，如果不在df中操作, inplace=False is passed 表示在df copy中操作，不影响df
    df.drop(['id', 'last_pymnt_d','url','earliest_cr_line', 'emp_title','issue_d','last_credit_pull_d','purpose','title','hardship_flag','policy_code'],inplace=True,axis=1,errors='ignore')
    #drop features that have confilicting meaning for y
    df.drop(['total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt'],inplace=True,axis=1,errors='ignore')
    #drop the features that have nothing
    df.dropna(inplace=True,axis=1,how='all')
    
    #pick up description
    desc=df['desc']
    
    #数每个attribute(column) 不是nan的个数
    num_rows=df.count(axis=0) # 是个list  1*128
    
    #drop the features for which greater than 10% of the loans were missing data for
    #如果缺失的数据多于整个dataframe的row 的10%丢掉， 只保留attribute whose 有90% 不为NA。
    df=df.iloc[:,(num_rows>=0.9*len(df)).tolist()]
    
    
    #merge back desc, 如果需要description, 就把description放进dataframe
    if keep_desc==True:
        df=pd.concat([df,desc],axis=1)


    #---------------------------去掉行------------------------------
   
    #drop the observation that was missing for any field
    df=df.dropna(axis=0,how='any')
    
    
    
    
    #---------------------------数据处理------------------------------
    
    #deal with percentage mark
    #e.g. 把5% 变成0.05
    # 首先把string 的百分号去掉, 然后换成float 再除以100
    df['int_rate']=df['int_rate'].replace('%','',regex=True).astype('float')/100
    df['revol_util']=df['revol_util'].replace('%','',regex=True).astype('float')/100
    
    #dealing with categorical features
    if categorical_to_binary==True:
        categorical_features=['addr_state','application_type','emp_length','grade','home_ownership','initial_list_status','pymnt_plan','sub_grade','term','verification_status']
        for i in categorical_features:
            if i in list(df): #list(df) 就是 df.columns
                #list(df) 等于 df.columns
                df[i]=df[i].astype('category')
                
                df=pd.get_dummies(df,columns={i},drop_first=True)
                #产生dummy variable 的同时，会drop 掉原来的column
                #drop first表示比如 有a, b, c 三个category, 但dummy drop a， 只生成 b, c 两个 dummy variable
                #生成dummy variable 
        
        """
        另一种写法, 不用for loop      
        categorical_features=['addr_state','application_type','emp_length','grade','home_ownership','initial_list_status','pymnt_plan','sub_grade','term','verification_status']
        
        df1[categorical_features] = df1[categorical_features].apply(pd.Categorical)
        
        df1=pd.get_dummies(df1,columns=categorical_features,drop_first=True)
        
        """
                
    #get dummies drop first 是只弄k-1个variable 把第1个variable 去掉
    
    
    #merge zipcode with census data
    #join zip_data的dataframe(有mean, median income的)，然后再去掉zip column
    df['zip_code']=df['zip_code'].astype(str)
    df['zip_code']=df['zip_code'].apply(lambda x: x[:3])
    df=df.join(zipdata,on='zip_code')
    df.drop('zip_code',inplace=True,axis=1)
    #drop the observation that was missing for any field
    
    #如果新加的mean, median income 有丢失的数据，一种可能是这个zip 的mean, median income 没有这个zip code 
    df=df.dropna(axis=0,how='any')
        
    #label the dataset to create y, 建立0，1 variables
    
    
    
    
    #---------------------------处理y------------------------------
    #把没有还款的标上0
    y=df['loan_status'].replace(['Charged Off','Does not meet the credit policy. Status:Charged Off','Late (31-120 days)','In Grace Period','Late (16-30 days)','Default'],0)
     #把还款的标上1
    y=y.replace(['Fully Paid','Does not meet the credit policy. Status:Fully Paid','Current'],1)
    df=df.drop(['loan_status'],axis=1)    
    
    return df,y

