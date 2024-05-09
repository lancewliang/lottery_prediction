# 你扮演一位python工程师
# 读取一个份cvs文件，使用pands库，
# cvs有8列，其中7列分别为数字，第8列为日期
# 需要计算出7个数字的如下值:和数/中位数/连续的个数/单数个数，
# 将这些计算出来的数值变成新的列，保存成为新的文件



import numpy as np  
import pandas as pd
from number_utils import *
from yijing_utils import *
root = "/home/lanceliang/cdpwork/ai/ddd/lottery_prediction/qlc/"


def prepare():
    df = pd.read_csv(root+"data/row_data.csv")
    # 假设数字列的名称分别是'col1', 'col2', ..., 'col7'  
    # 使用str.split()方法按空格分割字符串，并扩展结果到新的列  
    # 注意：expand=True会将分割后的列表扩展为DataFrame的列  
    df[['n1', 'n2', 'n3','n4', 'n5', 'n6', 'n7', 'n8']] = df['n'].str.split(expand=True)  
    
    # 因为分割后的数据是字符串类型，你可能需要将它们转换为整数或浮点数  
    df[['n1', 'n2', 'n3','n4', 'n5', 'n6', 'n7', 'n8']] = df[['n1', 'n2', 'n3','n4', 'n5', 'n6', 'n7', 'n8']].astype(int)  # 或者 float  
    df.drop('n', axis=1, inplace=True) 
    # 显示结果  
    

    df['date'] = df.apply(convert_date, axis=1) 
    
    # 计算和  
    df['sum'] = df[numeric_cols].sum(axis=1)  
    
    # 计算中位数（假设所有列都是数值型）  
    df['median'] = df[numeric_cols].median(axis=1)  
    
    # 定义计算连续序列数量的函数
    df['consecutive'] = df.apply(count_consecutive_sequences, axis=1) 
    
    # 计算单数（奇数）个数  
    df['odd'] = df[numeric_cols].apply(lambda row: (row % 2 != 0).sum(), axis=1)  
    
    # AC值（即所有两个号码相减的绝对差值的去重个数）     
    df['ac'] = df.apply(calculate_ac_value, axis=1)  
    
    # 五行相克性    
    df['wuxing_ke'] = df.apply(calculate_ke_degree, axis=1)  
    
    # 应用函数计算每行质数的个数  
    df['prime'] = df.apply( prime , axis=1)  
    
    # 分区
    df['partition'] = df.apply( partition , axis=1)  
 
    # 最幸运的匹配人，将生日变成天干地之->五行比较数字的亲和性
    df['luckman'] = df.apply( luckman , axis=1)  
    #热温冷：热号、温号、冷号均非固定号码,注数随走势情况变化
    # 而变化。①热号：最近7期内中出2次以上的号码称为热号；②温号：最近7期内中出2次的号码称为温号；③冷号：最近7期内中出次数在2次以下的号码称为冷号。热温冷可分为10种组选形态：1热1温1冷、2热1温、2热1冷、2温1热、2温1冷、2冷1热、2冷1温、3热、3温、3冷。 
    
    # 计算前20%和剩余80%的索引位置  

    fraction_to_split = 3  
    split_index = int(len(df) -3) 
    # 将前20%的行存储到新的DataFrame中  
  
    # 将剩余的行（即后80%）存储到另一个新的DataFrame中 
    df_traning = df  
    
    df_traning.sort_values(by='r', inplace=True, ascending=True)  
    df_test = df_traning.iloc[split_index:]  
    df_test.sort_values(by='r', inplace=True, ascending=True)  
    df_traning.to_csv(root+"data/prepared_traning_data.csv", index=False)
    df_test.to_csv(root+"data/prepared_test_data.csv", index=False)
    print("处理完成，新文件已保存为'data/prepared_data.csv'")
    # print(df)  


prepare()