import numpy as np  
from collections import defaultdict  
  
# 假设历史数据是一个二维列表，其中每个子列表包含7个数字（1-30）  
history_data = [...]  # 填充你的2000组历史数据  
  
# 将每组数字转换为一个唯一的标识符（这里简单地使用字符串表示）  
def encode_state(state):  
    return ''.join(map(str, state))  
  
# 构建状态转移字典  
transitions = defaultdict(list)  
for i in range(len(history_data) - 1):  
    current_state = encode_state(history_data[i])  
    next_state = encode_state(history_data[i+1])  
    transitions[current_state].append(next_state)  
  
# 计算状态转移概率（这里使用频率作为概率的近似）  
transition_probs = {}  
for state, next_states in transitions.items():  
    total_count = len(next_states)  
    probs = {next_state: count / total_count for next_state, count in zip(*np.unique(next_states, return_counts=True))}  
    transition_probs[state] = probs  
  
# 预测下一个状态（这里只演示如何选择一个状态，而不是完整的7个数字序列）  
def predict_next_state(current_state):  
    if current_state not in transition_probs:  
        return None  # 或选择一个随机状态作为默认  
    next_states = list(transition_probs[current_state].keys())  
    probabilities = list(transition_probs[current_state].values())  
    return np.random.choice(next_states, p=probabilities)  
  
# 示例：从一个初始状态开始预测下一个状态  
initial_state = encode_state([np.random.randint(1, 31) for _ in range(7)])  
predicted_next_state = predict_next_state(initial_state)  
print(f"Initial state: {initial_state}")  
print(f"Predicted next state: {predicted_next_state}")