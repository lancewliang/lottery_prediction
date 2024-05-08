### 第1步：
到项目根目录运行 poetry install
### 第2步：
prepare_data.py 文件会生成训练数据和测试数据到data目录中
* row_data.csv 从网站抓的原始数据
* prepared_traning_data.csv 增加了各种特征的训练数据
* prepared_test_data.csv 测试数据，将训练数据的最后3行变成测试数据，预测下一次的特征。

### 第3步：
train_lstm.py  训练模型和根据当前组的数据预测下一组，预测的特征在prepared_test_data.csv中<br>
测试结果是离谱的，特征预测因为随机的原因，特征非常不正确。符合随机的情况。
