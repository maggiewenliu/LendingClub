1. Read Data Science Process
2. Read data

sample change

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