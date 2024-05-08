
import numpy as np  
import pandas as pd

bagua_dict = { 
        0: "坤",  # 坤为地，五行属土  
        1: "震",  # 震为雷，五行属木  
        2: "坎",  # 坎为水，五行属水  
        3: "巽",  # 巽为风，五行属木  
        4: "乾",  # 乾为天，五行属金（有时也视为属阳土）  
        5: "兑",  # 兑为泽，五行属金  
        6: "艮",  # 艮为山，五行属土  
        7: "离",  # 离为火，五行属火  
    }  
wuxing_dict = {  
        "坤": "土",  
        "震": "木",  
        "坎": "水",  
        "巽": "木",  
        "乾": "金",  
        "兑": "金",  
        "艮": "土",  
        "离": "火",  
    }  
# 定义五行相克的规则（一个五行克另一个五行则记为True）  
ke_rules = {  
    '木': ['土'],  
    '土': ['水'],  
    '水': ['火'],  
    '火': ['金'],  
    '金': ['木']  
}     

numeric_cols = ['n1', 'n2', 'n3','n4', 'n5', 'n6', 'n7']

# 函数：检查两个五行是否相克  
def is_wuxing_ke(wuxing1, wuxing2):  
    return wuxing2 in ke_rules[wuxing1]  

# 函数：计算一组数字（通过八卦换算成五行后）的相克程度  
def calculate_ke_degree(row):  
    row_list = row[numeric_cols].astype(int).tolist()  
    # 初始化连续序列计数和当前序列长度  
    row_list.sort()
    wuxings = [number_to_wuxing(num) for num in row_list]  
    ke_degree = 0  
    for i in range(len(wuxings) - 1):  # 检查相邻的五行  
        if is_wuxing_ke(wuxings[i], wuxings[i+1]):  
            ke_degree += 1  
    return ke_degree  
def number_to_bagua(number):  
    #index = number - 1  # 因为数字从1开始，但索引从0开始  
    bagua_index = number % 8  
    if not 0 <= bagua_index <= 7:  
        raise ValueError("Number must be between 0 and 7.")  

    return bagua_dict[bagua_index]  

     
def number_to_wuxing(number): 
    return wuxing_dict[number_to_bagua(number)]  # 取余确保在0-7范围内  
      
# 定义天干地支与五行的对应关系  
tian_gan = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']  
di_zhi = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']  
wu_xing = ['木', '火', '土', '金', '水']  
  
# 根据年份计算天干地支  
def calculate_tian_gan_di_zhi(year):  
    gan_index = (year - 4) % 10  # 天干的索引  
    zhi_index = (year - 4) % 12  # 地支的索引  
    return tian_gan[gan_index], di_zhi[zhi_index]  
  
# 根据天干地支计算五行属性（简化版，仅根据天干计算）  
def calculate_wu_xing(gan):  
    if gan in ['甲', '乙']:  
        return '木'  
    elif gan in ['丙', '丁']:  
        return '火'  
    elif gan in ['戊', '己']:  
        return '土'  
    elif gan in ['庚', '辛']:  
        return '金'  
    elif gan in ['壬', '癸']:  
        return '水'  
    else:  
        return '未知'  
  
# 输入公历生日，计算五行属性  
def get_wu_xing_by_birthday(year, month, day):  
    gan, zhi = calculate_tian_gan_di_zhi(year)  
    wu_xing_of_gan = calculate_wu_xing(gan)  
    print(f"出生年份的天干是：{gan}，地支是：{zhi}")  
    print(f"根据天干计算出的五行属性是：{wu_xing_of_gan}")  
    return wu_xing_of_gan  
        
       
      
luckman_dict = {  
        1: {"name":"冯柏通","day":19931205},  
        2: {"name":"戴泽明","day":19950226},  
        3: {"name":"陈华祥","day":19950131},  
        4: {"name":"朱文涛","day":19900904},  
        5: {"name":"梁伟","day":19830817},  
        6: {"name":"陈新宇","day":19900904},          
        7: {"name":"李思杨","day":19931125},   
        8: {"name":"刘翊然","day":20240425},    
        # "9": {"name":"丁聪","day":"19931125"},             
    }  

for idx, (key, value) in enumerate(luckman_dict.items()):  
    luckman = luckman_dict[key]
    name = luckman["name"]            
    date_str = str(luckman["day"])
    # 切片得到年、月、日  
    year = int(date_str[:4])  # 取前4位作为年份  
    month = int(date_str[4:6])  # 取第5、6位作为月份  
    day = int(date_str[6:])  # 取剩余部分作为日期 
    wuxing = get_wu_xing_by_birthday(year, month, day)
    luckman["wuxing"] = wuxing

luckman_wuxing_dict = {}
                    

def luckman(row):
     # 将行转换为列表  
    row_list = row[numeric_cols].astype(int).tolist()  
    # 初始化连续序列计数和当前序列长度  
    row_list.sort()
    # 遍历所有分区形态和数字组合  
    num_wuxings = [number_to_wuxing(num) for num in row_list]  
 
    luckman_dict_ke_degree = {}
    for i in range(len(num_wuxings) - 1):  # 检查相邻的五行  
        for key, value in luckman_dict.items():  
            luckman = luckman_dict[key]
            ke_degree = luckman_dict_ke_degree.get(key,0)
            if not is_wuxing_ke(luckman["wuxing"],num_wuxings[i] ):  
                ke_degree += 1  
            luckman_dict_ke_degree[key]=ke_degree
                # 初始化一个变量来保存当前找到的最大value和对应的key  

    max_value = float('-inf')  # 初始化为负无穷大  
    luckman_num = None  
    # 遍历字典项  
    for key, value in luckman_dict_ke_degree.items():  
        if value > max_value:  # 如果当前value大于已知的最大value  
            max_value = value  # 更新最大value  
            luckman_num = key      # 更新最大value对应的key  
    return luckman_num 
