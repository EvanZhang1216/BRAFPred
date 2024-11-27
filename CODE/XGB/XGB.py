import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from scipy.stats import pearsonr

# 定义文件路径和结果保存路径
data_dir = os.path.join("..", "data", "Train")
results_dir = os.path.join("..", "results")
os.makedirs(results_dir, exist_ok=True)

# 定义特征文件名
fp = [
    'AtomPairs2DCount',
    'AtomPairs2D',
    'EState',
    'CDKextended',
    'CDK',
    'CDKgraphonly',
    'KlekotaRothCount',
    'KlekotaRoth',
    'MACCS',
    'PubChem',
    'SubstructureCount',
    'Substructure'
]

# 定义XGBoost参数
param = {
    'gamma': 0,
    'reg_lambda': 1,
    'reg_alpha': 0,
    'max_depth': 6,
    'n_estimators': 100,
    'learning_rate': 0.3,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# 结果存储
results = []

for wenjian in fp:
    print("__________________________________________________________________________")
    print(f"{wenjian}\n")
    data_name = os.path.join(data_dir, wenjian + "_BRAF.csv")

    # 加载数据
    df = pd.read_csv(data_name)

    # 分离特征和目标值
    X = df.iloc[:, 2:]  # 假设第一列是目标值，所以取剩下的列作为特征
    y = df.iloc[:, 1]  # 第一列作为目标值

    # 设置十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 初始化结果列表
    r2_scores = []
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    pcc_scores = []

    # 执行十折交叉验证
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 转换为DMatrix格式
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        # 训练模型
        bst = xgb.train(param, dtrain, num_boost_round=100)

        # 预测
        preds = bst.predict(dtest)

        # 计算R²、MAE、MSE、RMSE和PCC
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        pcc, _ = pearsonr(y_test, preds)

        # 保存结果
        r2_scores.append(r2)
        mae_scores.append(mae)
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        pcc_scores.append(pcc)

    # 计算平均R²、MAE、MSE、RMSE和PCC
    mean_r2 = np.mean(r2_scores)
    mean_mae = np.mean(mae_scores)
    mean_mse = np.mean(mse_scores)
    mean_rmse = np.mean(rmse_scores)
    mean_pcc = np.mean(pcc_scores)

    # 打印结果
    print(f"Mean R2: {mean_r2}")
    print(f"Mean MAE: {mean_mae}")
    print(f"Mean MSE: {mean_mse}")
    print(f"Mean RMSE: {mean_rmse}")
    print(f"Mean PCC: {mean_pcc}")

    # 保存结果到列表
    results.append([wenjian, mean_r2, mean_mae, mean_mse, mean_rmse, mean_pcc])

# 将结果保存到CSV文件
results_df = pd.DataFrame(results, columns=['Feature', 'Mean R2', 'Mean MAE', 'Mean MSE', 'Mean RMSE', 'Mean PCC'])
results_df.to_csv(os.path.join(results_dir, 'XGB_10CV_results.csv'), index=False)

print(f"Results saved to {os.path.join(results_dir, 'XGB_10CV_results.csv')}")
