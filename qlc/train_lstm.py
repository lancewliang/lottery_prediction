# 你扮演一个算法工程师.
# 目的是生成一段python代码,主要使用的技术是pytorch和LSTM神经网络.
# 我的测试数据有因子X有5个特征,标签Y有2个特征。标签Y的第一个特征是分类有4种，标签y的第二特征是数值。
# 标签Y的2种特征应该使用不同的损失函数来计算损失。
# 训练和测试数据可以通过随机生成。
# 使用LSTM神经网络
# 代码我只是作为学习使用
# 我希望通过因子X来预测标签Y
# 请使用matplotlib库画可观测的映像，让开发者了解训练的是否收敛
# 并且提供通过模型来预测一组数据的代码


import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
X_Featrues = [ 'r','date','n1','n2','n3','n4','n5','n6','n7','n8','sum','median','consecutive','odd','ac','wuxing_ke','prime','partition','luckman']
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
idx_output_size_class={
    1:'sum',
    2:'median',
    3:'consecutive',
    4:'odd',
    5:'ac',
    6:'wuxing_ke',
    7:'prime',
    8:'partition',
    9:'luckman'
}
# 定义超参数  
input_size = 19  # 因子X的特征数量  
hidden_size = 64  # LSTM隐藏层大小  
num_layers = 1  # LSTM层数  


# output_size_reg = 1  # 标签Y的回归特征数量 
num_epochs = 2000  # 训练轮数  
batch_size = 64  # 批处理大小  
learning_rate = 0.001  # 学习率  
  
# 生成训练和测试数据  
# num_samples = 2  # 生成1000个样本  
sequence_length = 3  # 每个样本有10个时间步长  

root = "/home/lanceliang/cdpwork/ai/lottery/lottery_prediction/qlc/"

    
def load_traning_data():  
    file = root+"data/prepared_traning_data.csv"
    x_df = pd.read_csv(file)   
    #删除最后一行
    df_len= len(x_df)
    x_df = x_df.iloc[:-1]     
    X = transfer2dTo3dWithSequenceLen(sequence_length,x_df)
    X_length = X.size(0) 
    
    #X和Y的行数应该一样多。一些行需要放弃
    y_df = pd.read_csv(file)   
    #删除第一行  
    y_rm_len = df_len - X_length
    y_df = y_df.iloc[y_rm_len:] 
    
    Y = {}     
    for key, value in output_size_class.items():  
        Y_class = y_df[key].astype('int32').values  
        Y[key] = torch.from_numpy(Y_class).to(device,dtype=torch.long)
 
    print(Y)
    return X.to(device) , Y

def load_test_data():  
    file = root+"data/prepared_test_data.csv" 
    x_df = pd.read_csv(file)   
    #删除最后一行
    df_len= len(x_df)    
    X = transfer2dTo3dWithSequenceLen(sequence_length,x_df)
    return X.to(device) , None

def transfer2dTo3dWithSequenceLen(sequence_length,df):
    # 3. 初始化一个空列表来存储三维数据块（每个数据块是一个时间步长）  
    data_cubes = []  
    
    # 4. 逆序遍历数据（从最后一行开始），除了前7行之外（因为我们需要至少7行来形成一个时间步长）  
    for i in range(len(df) , sequence_length - 1, -1):  
        if i < sequence_length:  # 如果i小于7，则没有足够的行来形成一个时间步长，跳过  
            break  
        # 选择当前行及其之前的7行（不包括当前行）  
        start_idx = i - sequence_length  
        end_idx = i  
        current_slice = df.iloc[start_idx:end_idx]  
        
        # 将数据转换为NumPy数组  
        numpy_array = current_slice[X_Featrues].astype('int32').values  
        
        # 如果你需要，可以在这里对数据进行进一步处理（例如，标准化、归一化等）  
        
        # 将NumPy数组添加到列表中  
        data_cubes.append(numpy_array)  
    
    # 因为我们是逆序遍历的，所以需要将列表反转回来以保持原始的顺序（如果需要的话）  
    data_cubes.reverse()  
    
    # 5. 将列表中的NumPy数组转换为PyTorch的三维张量  
    # 首先确定张量的形状（时间步长数量, 时间步长内的行数, 特征数量）  
    num_cubes = len(data_cubes)  
    num_rows_per_cube = sequence_length  # 因为每个时间步长有7行  
    num_features = data_cubes[0].shape[1]  # 假设所有时间步长有相同的特征数量  
    tensor_3d = torch.zeros((num_cubes, num_rows_per_cube, num_features), dtype=torch.int32)  

    # 将NumPy数组填充到三维张量中  
    for i, numpy_cube in enumerate(data_cubes): 
        tensor_3d[i] = torch.from_numpy(numpy_cube)
    return tensor_3d

# LSTM模型定义  
class LSTMClassifier(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers, num_classes):  
        super(LSTMClassifier, self).__init__()  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  
        
        fc_class_list = []
        
        for key, value in num_classes.items():  
            fc_class_list .append(nn.Linear(hidden_size, value) ) 
        self.fc_layers = nn.ModuleList(fc_class_list)
        # self.fc_reg = nn.Linear(hidden_size, output_size_reg)   
        
  
    def forward(self, x):  
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  
        out, _ = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]
        
        class_output = {}
        for i,layer in enumerate(self.fc_layers):  
          class_output[idx_output_size_class[i+1]] =layer(out)
        
        # reg_out = self.fc_reg(out) 
        # , reg_out  
        return class_output 

def predict(model, X):
    model.eval()
    with torch.no_grad():
        outputs_class = model(X.float())
        print("Class Predictions:", outputs_class)
        print("Predicted labels for test data:")
        for key, value in class_output.items():  
            predicted_labels = value.argmax(dim=1)                        
            print("Label ",key,":", predicted_labels[0].item()) 
        
        # class_preds = torch.argmax(outputs_class, dim=1)        
        return outputs_class
      
# 初始化模型、损失函数和优化器  
model = LSTMClassifier(input_size, hidden_size, num_layers, output_size_class)  
 
criterion_class = nn.CrossEntropyLoss()  # 二损失函数，用于多标签分类  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
  
# 将模型和数据移动到GPU（如果有的话），否则使用CPU  

model = model.to(device)  
  

X_train, Y_train_class = load_traning_data()  
X_test, Y_test_class = load_test_data()   # 使用一半数量的样本作为测试集  
print(X_test.size())
print(X_test)

# 训练模型  
train_losses_class = []

 
  
model.train()  
for epoch in range(num_epochs):  
    optimizer.zero_grad()  
    class_output = model(X_train.float())  
    loss_class = None
    for key, value in class_output.items():  
        _loss_class = criterion_class(value, Y_train_class[key])  
        if loss_class is None:
            loss_class = _loss_class
        else:
            loss_class+= _loss_class
    
    loss_class.backward()  
    optimizer.step()  
    train_losses_class.append(loss_class.item())  
    
    # 计算平均损失  

    if (epoch+1) % 10 == 0:  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss Class: {loss_class.item():.4f}')  
  
# # 绘制损失曲线  

# plt.figure(figsize=(10, 5))  
# plt.subplot(1, 2, 1)  
# plt.plot(train_losses_class, label='Training Loss Class')  
# plt.title('Training Losses')  
# plt.xlabel('Epoch')  
# plt.ylabel('Loss')  
# plt.legend()  


# plt.tight_layout()
# plt.show()



print(X_test.size())

class_preds = predict(model, X_test.float())

