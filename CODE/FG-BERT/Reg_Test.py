import tensorflow as tf
import numpy as np
import pandas as pd
from dataset_scaffold_random import Graph_Regression_Dataset
from sklearn.metrics import r2_score


independent_test_data_path = 'Testdata.csv'  # 独立测试集的路径
smiles_field = 'smiles'  # SMILES 字段名
label_field = ['pIC50']  # 标签字段
batch_size = 512  # 测试集批量大小

# 创建测试数据集
graph_dataset = Graph_Regression_Dataset(independent_test_data_path, smiles_field=smiles_field, label_field=label_field, normalize=True, seed=42, batch_size=batch_size, a=len(label_field), max_len=500, addH=True)
_, independent_test_dataset, _ = graph_dataset.get_data()  # 只需要测试集

# 加载最佳模型
num_layers = 6
d_model = 256
dff = d_model * 2
num_heads = 8
vocab_size = 18
dense_dropout = 0.05
value_range = graph_dataset.value_range  # 获取数据集的取值范围

best_model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size, a=len(label_field), dense_dropout=dense_dropout)
best_model.load_weights('regression_weights/BestModel111.h5'.format('raw_data_QSAR'))  # 加载保存的最佳模型权重

# 测试模型
y_true = []
y_preds = []
smiles_list = []

for x, adjoin_matrix, y in independent_test_dataset:
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    preds = best_model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
    y_true.append(y.numpy())
    y_preds.append(preds.numpy())
    smiles_list.append(x.numpy())  # 添加 SMILES 到列表

# 将真实值和预测值拼接成数组
y_true = np.concatenate(y_true, axis=0).reshape(-1, len(label_field))
y_preds = np.concatenate(y_preds, axis=0).reshape(-1, len(label_field))

# 计算 RMSE、R² 和 MAE
test_mse = tf.keras.metrics.MeanSquaredError()(y_true, y_preds).numpy() * (value_range ** 2)
test_rmse = np.sqrt(test_mse)

mae = tf.keras.metrics.MeanAbsoluteError()(y_true, y_preds).numpy() * (value_range)

r2 = r2_score(y_true.reshape(-1), y_preds.reshape(-1))

print('Independent test set RMSE: {:.4f}'.format(test_rmse))
print('Independent test set MAE: {:.4f}'.format(mae))
print('Independent test set R²: {:.4f}'.format(r2))

# 将预测值和真实值输出到 CSV 文件
df_output = pd.DataFrame({
    'SMILES': np.concatenate(smiles_list).tolist(),  # 将 SMILES 添加到 CSV
    'True Values': y_true.flatten(),
    'Predicted Values': y_preds.flatten()
})

# 保存到 CSV
df_output.to_csv('independent_test_predictions.csv', index=False)
print('Predictions saved to independent_test_predictions.csv')
