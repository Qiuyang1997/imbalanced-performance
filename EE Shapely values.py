import itertools
import numpy as np
import pandas as pd
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import train_test_split


# 1. 读取数据集
data = pd.read_excel(r'originaldata.xls')
X = data.drop(columns=['NO3'])  # 特征列
y = data['NO3']  # 标签列
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=123)

# 训练示例模型
model = EasyEnsembleClassifier(random_state=123)
model.fit(X_train, y_train)

# 定义特征集合和目标函数
features = X_train
y_pred = model.predict_proba(features)[:, 1]


def calculate_shapley_values(model, X, instance):
    feature_indices = range(X.shape[1])
    shapley_values = np.zeros(X.shape[1])

    for i in feature_indices:
        # 构建所有可能的特征子集
        subsets = list(itertools.combinations(feature_indices, i))
        for subset in subsets:
            # 加入特征 i 的子集
            subset_with_i = list(subset) + [i]
            # 不加入特征 i 的子集
            subset_without_i = list(subset)

            # 计算模型预测
            pred_with_i = model.predict_proba(np.expand_dims(instance[subset_with_i], axis=0))[:, 1]
            pred_without_i = model.predict_proba(np.expand_dims(instance[subset_without_i], axis=0))[:, 1]

            # 计算边际贡献
            marginal_contribution = pred_with_i - pred_without_i
            shapley_values[i] += marginal_contribution / len(subsets)

    return shapley_values


# 计算某个实例的 Shapley 值
instance = X[0]
shap_values = calculate_shapley_values(model, X, instance)
print("Shapley Values:", shap_values)
