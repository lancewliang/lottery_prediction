import numpy as np  
import pandas as pd

# 假设的状态数量  
output_size_class={
    'sum':210,
    'median':30,
    'consecutive':7,
    'odd':7,
    'ac':100,
    'wuxing_ke':10,
    'prime':7,
    'partition':40,
    'luckman':10
}
# 生成一些随机的历史数据作为示例（实际中你会从数据源获取这些数据）  
# np.random.seed(0)  # 为了结果的可复现性  
root = "/home/lanceliang/cdpwork/ai/ddd/lottery_prediction/qlc/"
history_data = pd.read_csv(root+"data/prepared_traning_data.csv")   
  
# 为每个特征构建转移概率矩阵  
def estimate_transition_matrix(feature_data):  
    num_states = max(np.unique(feature_data))+1  
    transition_matrix = np.zeros((num_states, num_states))  
    for i in range(1, len(feature_data)):  
        transition_matrix[feature_data[i-1], feature_data[i]] += 1  
    # 归一化转移矩阵  
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)  
    return transition_matrix  
  
transition_matrix_feature1 = estimate_transition_matrix(history_data["consecutive"].values)  
transition_matrix_feature2 = estimate_transition_matrix(history_data["ac"].values)  
transition_matrix_feature3 = estimate_transition_matrix(history_data["partition"].values)  
  
# 预测后一天的特征状态（假设我们已知前一天的状态）  
def predict_next_state(transition_matrix, current_state):  
    # 选择下一个状态，这里使用最简单的方法：按照转移概率随机选择  
    next_state_probs = transition_matrix[int(current_state)]  
    next_state = np.random.choice(range(len(next_state_probs)), p=next_state_probs)  
    return next_state  

last_state_feature1=0
last_state_feature2=0
last_state_feature3=0 
suc = []
# 假设我们知道前一天的特征状态  
root = "/home/lanceliang/cdpwork/ai/ddd/lottery_prediction/qlc/"
test_data = pd.read_csv(root+"data/prepared_traning_data.csv")   
for index, row in test_data.iterrows():  
    previous_state_feature1 =row["consecutive"]
    previous_state_feature2 =row["ac"]
    previous_state_feature3 =row["partition"]
    r = row["r"]
    if last_state_feature1==previous_state_feature1 and last_state_feature2==previous_state_feature2 and last_state_feature3==previous_state_feature3:
        suc.append(r)    
    
    
    print(f"----------------------------预测-{r}-的下一组---------------------------------------------")
    print(f"sum: {previous_state_feature1} ac:{previous_state_feature2} partition:{previous_state_feature3}")      
    # 预测后一天的特征状态  
    next_state_feature1 = predict_next_state(transition_matrix_feature1, previous_state_feature1)  
    next_state_feature2 = predict_next_state(transition_matrix_feature2, previous_state_feature2)  
    next_state_feature3 = predict_next_state(transition_matrix_feature3, previous_state_feature3)  
    
    last_state_feature1=next_state_feature1
    last_state_feature2=next_state_feature2
    last_state_feature3=next_state_feature3
    
  
    print(f"Predicted next state for feature")
    print(f"sum: {next_state_feature1} ac:{next_state_feature2} partition:{next_state_feature3}")  
print("----------------------------------------------------------------------------------------")
#看是否有预测成功的。几乎没有
print (r)

print("----------------------------------------------------------------------------------------")
