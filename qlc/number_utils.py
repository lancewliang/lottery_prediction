import numpy as np  
import pandas as pd

numeric_cols = ['n1', 'n2', 'n3','n4', 'n5', 'n6', 'n7']







#AC值（即所有两个号码相减的绝对差值的去重个数）  
def calculate_ac_value(row):  
    # 创建一个集合来存储不同的差值 
    row_list = row[numeric_cols].astype(int).tolist()  
    # 初始化连续序列计数和当前序列长度  
    row_list.sort()
    differences = set()     
    # 遍历所有可能的号码对  
    for i in range(len(row_list)):  
        for j in range(i+1, len(row_list)):  # 从i+1开始以避免重复计算  
            # 计算两个号码的差的绝对值，并添加到集合中  
            diff = abs(row_list[i] - row_list[j])  
            differences.add(diff)  
    # 返回集合中元素的个数，即不同的差值个数  
    return len(differences)  

# 辅助函数：判断一个数是否为质数  

def prime(row):  
    row_list = row[numeric_cols].astype(int).tolist()  
    # 初始化连续序列计数和当前序列长度  
    row_list.sort()
    return  sum(is_prime(x) for x in row_list)
def is_prime(n):  
    if n < 2:  
        return False  
    for i in range(2, int(np.sqrt(n)) + 1):  
        if n % i == 0:  
            return False  

    return True  
# 定义计算连续序列数量的函数
def count_consecutive_sequences(row):  
    # 将行转换为列表  
    row_list = row[numeric_cols].astype(int).tolist()  
    # 初始化连续序列计数和当前序列长度  
    row_list.sort()
    count = 0  
    current_sequence_length = 1  
    # 遍历除了最后一个元素外的所有元素  
    for i in range(len(row_list) - 1):  
        # 如果当前元素和下一个元素的差为1，则当前序列长度加1  
        if abs(row_list[i] - row_list[i+1]) == 1:  
            current_sequence_length += 1  
        # 否则，如果当前序列长度大于1，则增加计数并重置序列长度  
        else:  
            if current_sequence_length > 1:  
                count += 1  
            current_sequence_length = 1  
    # 检查最后一个序列（如果有的话）  
    if current_sequence_length > 1:  
        count += 1  
    # 返回连续序列的数量  
    return count  

#数字分 
from itertools import combinations   
# 枚举所有可能的分区形态（包括0的情况）  
configurations =  {
    1:(2, 2, 3),
    2:(3, 2, 2),
    3:(2, 3, 2),  
    4:(3, 3, 1),
    5:(3, 1, 3), 
    6:(1, 3, 3),  
    7:(4, 2, 1),
    8:(4, 1, 2), 
    9:(2, 4, 1),  
    10:(1, 4, 2),
    11:(2, 1, 4),
    12:(1, 2, 4),  
    13:(5, 1, 1), 
    14:(1, 5, 1), 
    15:(1, 1, 5),  
    16:(6, 1, 0),
    17:(1, 6, 0), 
    18:(0, 6, 1),  
    19:(6, 0, 1),
    20:(0, 1, 6),
    21:(1, 0, 6),  
    22:(5, 2, 0), 
    23:(5, 0, 2), 
    24:(2, 5, 0),  
    25:(0, 2, 5), 
    26:(2, 0, 5), 
    27:(0, 5, 2),  
    28:(7, 0, 0), 
    29:(0, 7, 0), 
    30:(0, 0, 7),
    31:(0, 3, 4),
    32:(0, 4, 3),
    33:(3, 4, 0),
    34:(4, 0, 3),
    35:(3, 0, 4),
    36:(4, 3, 0) # 所有数字都在一个分区内的情况  
} 
  # 辅助函数：检查数字列表是否符合给定的分区形态  
def check_configuration(nums, config):  
    zone_ranges = [(1, 11), (12, 22), (23, 30)]  
    zone_counts = [0, 0, 0]  
    for num in nums:  
        for i, (start, end) in enumerate(zone_ranges):  
            if start <= num <= end:  
                zone_counts[i] += 1  
                break  
    return zone_counts == list(config)  
#数字分区
def partition(row):  
    # 将行转换为列表  
    row_list = row[numeric_cols].astype(int).tolist()  
    # 初始化连续序列计数和当前序列长度  
    row_list.sort()
    # 遍历所有分区形态和数字组合  
    for idx, (key, value) in enumerate(configurations.items()):  
        zone1_count, zone2_count, zone3_count = value  
        if check_configuration(row_list, value): 
            return key
    # print('')
  
#日期转格式
def convert_date(row):  
    date_str = row['date']
    return convert_date_to_yyyymmdd(date_str)
    
    
from datetime import datetime  
  
def convert_date_to_yyyymmdd(date_str, date_format="%m/%d/%Y"):  

    date = datetime.strptime(date_str, date_format)  
    return int(date.strftime("%Y%m%d"))