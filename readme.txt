1. Read Data Science Process
2. Read data

需要懂的点

1. 什么是default / Non default 定义, y = 0的定义
    - 比如什么是 'Charged Off',  'Does not meet the credit policy. Status:Charged Off','Late (31-120 days)','In Grace Period','Late (16-30 days)','Default'
2. 什么是不default 定义, y = 1的定义
    - 'Fully Paid','Does not meet the credit policy. Status:Fully Paid','Current'


Python Code 逻辑

1. 读ZIP csv
    - 处理ZIP, 只保留前3位，比如 8999 -> 保留089, 10001 -> 保留100
    - 得到一个地区的 population 总和，比如
    -  最终zip file格式是：
     现在zip_new 格式是， 注意现在index 是zip, 而不是数字了
    
        new_mean     new_median
Zip                              
010   68997.226192   57290.974026
011   54979.626027   42377.127888
012   65706.935838   50845.603610



2. 读Load.csv
    if 有保存的pickle file，节省时间
        读取pickle file 
    else 
        读取excel file,总共4个file
        每读完一个file， concat到df的下面

3. cleanData:
    (1). drop 掉所有missing 数据的列
    (2). drop 掉一些没有意义的列