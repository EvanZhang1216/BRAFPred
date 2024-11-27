import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import pearsonr
# 原始数组保存
c_values = {
    'AtomPairs2D': 10,
    'AtomPairs2DCount': 10,
    'CDK': 6,
    'CDKextended': 5,
    'CDKgraphonly': 6,
    'EState': 6,
    'KlekotaRoth': 10,
    'KlekotaRothCount': 10,
    'MACCS': 8,
    'PubChem': 10,
    'Substructure': 10,
    'SubstructureCount': 10
}

fp_suofang = list(c_values.keys())
fp_nosuofang = list(c_values.keys())

results = []

for wenjian in fp_nosuofang:
    print(f"____________________________________{wenjian}________________________________________")
    data_name = os.path.join('..', 'data', 'Train', wenjian + '_BRAF.csv')
    # 加载数据
    df = pd.read_csv(data_name)
    # 分离特征和目标值
    X = df.iloc[:, 2:]
    y = df.iloc[:, 1]

    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 初始化SVR模型
    svr_model = SVR(C=c_values[wenjian])

    # 10折交叉验证
    kf = KFold(n_splits=10)
    y_pred = cross_val_predict(svr_model, X, y, cv=kf)

    # 计算指标
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    pcc, _ = pearsonr(y, y_pred)

    print(f"R^2 : {r2}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"PCC: {pcc}")

    # 保存结果
    results.append([wenjian, r2, mae, mse, rmse , pcc])

# 保存结果到CSV文件
results_df = pd.DataFrame(results, columns=['Dataset', 'R2', 'MAE', 'MSE', 'RMSE' , 'PCC'])
results_path = os.path.join('..', 'results', 'SVR_10CV_evaluation_metrics.csv')
results_df.to_csv(results_path, index=False)

print(f"结果已保存到 {results_path}")
