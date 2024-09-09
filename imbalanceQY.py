import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler, NearMiss,TomekLinks
from imblearn.ensemble import EasyEnsembleClassifier,BalancedRandomForestClassifier
#######RFC on original dataset########
# 1. 读取数据集
data = pd.read_excel(r'originaldata.xls')

# 2. 数据预处理
X = data.drop(columns=['NO3'])  # 特征列
y = data['NO3']  # 标签列

# 3. 划分数据集为训练集和测试集
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=123)

# 4. 定义超参数搜索空间
param_space = {
    'n_estimators': Integer(100, 10000),
    'max_depth': Integer(1, 100),
    'max_features': Categorical([ 'sqrt', 'log2']),
    'min_samples_split': Integer(2, 100),
    'min_samples_leaf': Integer(1, 10),
    'criterion': Categorical(['entropy', 'gini'])
}

# 5. 使用Bayesian Optimization进行参数搜索和交叉验证
rfc = RandomForestClassifier()
bayes_search = BayesSearchCV(rfc, param_space, cv=5, scoring='f1', n_jobs=-1, n_iter=50)
bayes_search.fit(X_train, y_train)

print("best param:", bayes_search.best_params_)

# 6. Validate on the test set using the best parameters
best_model = bayes_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_holdout)
y_pred_proba = best_model.predict_proba(X_holdout)[:, 1]

# 7. Evaluate model performance
fpr, tpr, threshold = roc_curve(y_holdout, y_pred_proba)

auc_score = auc(fpr, tpr)
recall = recall_score(y_holdout, y_pred)
precision = precision_score(y_holdout, y_pred)
f1 = f1_score(y_holdout, y_pred)
accuracy = accuracy_score(y_holdout, y_pred)

print("AUC:", auc_score)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)
print("Overall Accuracy:", accuracy)

cv_results = pd.DataFrame(bayes_search.cv_results_)
cv_results.to_excel(r'performance.xlsx', index=False)

#######RO, SMOTE, ADASYN, RU, TLU, and NMU combined RFC on original dataset########

# 8. sampling processing
sampler = TomekLinks() ###
# also
# sampler = RandomOverSampler(random_state=123)
# sampler = RandomUnderSampler(random_state=123)
# sampler = RandomUnderSampler(random_state=123)
# sampler = NearMiss(version=3)
# sampler = SMOTE(random_state=123)
# sampler = ADASYN(random_state=123)

X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

# 9.  Validation set
X_train_r, X_validation, y_train_r, y_validation = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=123)

# 10. Validate on the test set using the best parameters
rfc = RandomForestClassifier()
bayes_search = BayesSearchCV(rfc, param_space, cv=5, scoring='f1', n_jobs=-1, n_iter=50)
bayes_search.fit(X_train, y_train)

print("best param:", bayes_search.best_params_)
best_model = bayes_search.best_estimator_
best_model.fit(X_train_r, y_train_r)

y_pred_validation = best_model.predict(X_validation)
y_pred_proba_validation = best_model.predict_proba(X_validation)[:, 1]

# 11. Evaluate model performance on Validation set
fpr_validation, tpr_validation, threshold_validation = roc_curve(y_validation, y_pred_proba_validation)

auc_score_validation = auc(fpr_validation, tpr_validation)
recall_validation = recall_score(y_validation, y_pred_validation)
precision_validation = precision_score(y_validation, y_pred_validation)
f1_validation = f1_score(y_validation, y_pred_validation)
accuracy_validation = accuracy_score(y_validation, y_pred_validation)


# 12. Validate on the holdout set using the best parameters
best_model.fit(X_train_resampled, y_train_resampled)
y_pred_holdout = best_model.predict(X_holdout)
y_pred_proba_holdout = best_model.predict_proba(X_holdout)[:, 1]


fpr, tpr, threshold = roc_curve(y_holdout, y_pred_proba)

auc_score = auc(fpr, tpr)
recall = recall_score(y_holdout, y_pred)
precision = precision_score(y_holdout, y_pred)
f1 = f1_score(y_holdout, y_pred)
accuracy = accuracy_score(y_holdout, y_pred)

print("AUC:", auc_score)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)
print("Overall Accuracy:", accuracy)

cv_results = pd.DataFrame(bayes_search.cv_results_)
cv_results.to_excel(r'sampling Performance.xlsx', index=False)


#######EE, BRFC, and UOB########

###EE
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

# Define an rfc base classifier
base_estimator = rfc()

ee_classifier = EasyEnsembleClassifier(random_state=123)

# Define an rfc base classifier
ee_classifier.base_estimator = base_estimator

param_space = {
    'n_estimators': (100, 10000)
}

# 使用贝叶斯调参进行参数搜索
bayes_search = BayesSearchCV(estimator=ee_classifier, search_spaces=param_space, cv=5, scoring='f1', n_jobs=-1, n_iter=50)
bayes_search.fit(X_train, y_train)

print("Best Parameters:", bayes_search.best_params_)

# 使用最佳参数的模型进行预测
best_ee_classifier = bayes_search.best_estimator_
best_ee_classifier.fit(X_train, y_train)
y_pred = best_ee_classifier.predict(X_val)
y_pred_proba = best_ee_classifier.predict_proba(X_val)[:, 1]

# 评估模型性能
fpr, tpr, threshold = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)


print("AUC:", roc_auc)
print("Overall Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)


cv_results = pd.DataFrame(bayes_search.cv_results_)
cv_results.to_excel(r'EE.xlsx', index=False)


####BRFC

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

# 4. 定义参数搜索范围
param_space = {
    'n_estimators': (100, 10000),
    'max_depth': (1, 100),
    'max_features': ([ 'sqrt', 'log2']),
    'min_samples_split': (2, 100),
    'min_samples_leaf': (1, 10),
    'criterion': (['entropy', 'gini'])
}

# 5. 创建 BalancedRandomForestClassifier 实例
brf_classifier = BalancedRandomForestClassifier(random_state=123)

# 6. 使用贝叶斯调参进行参数搜索
bayes_search = BayesSearchCV(estimator=brf_classifier, search_spaces=param_space, cv=5, scoring='f1', n_jobs=-1, n_iter=50)
bayes_search.fit(X_train, y_train)

best_brf_classifier = bayes_search.best_estimator_
y_pred = best_brf_classifier.predict(X_val)
y_pred_proba = best_brf_classifier.predict_proba(X_val)[:, 1]


fpr, tpr, threshold = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)


print("Best Parameters:", bayes_search.best_params_)
print("AUC:", roc_auc)
print("Overall Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)


cv_results.to_excel(r'BRFC.xlsx', index=False)

###UOB
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

base_estimator = rfc()

ee_classifier = BalancedBaggingClassifier(random_state=123)

ee_classifier.base_estimator = base_estimator

param_space = {
    'n_estimators': (100, 10000)
}


bayes_search = BayesSearchCV(estimator=ee_classifier, search_spaces=param_space, cv=5, scoring='f1', n_jobs=-1, n_iter=50)
bayes_search.fit(X_train, y_train)


print("Best Parameters:", bayes_search.best_params_)

# 使用最佳参数的模型进行预测
best_ee_classifier = bayes_search.best_estimator_
best_ee_classifier.fit(X_train, y_train)
y_pred = best_ee_classifier.predict(X_val)
y_pred_proba = best_ee_classifier.predict_proba(X_val)[:, 1]

fpr, tpr, threshold = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print("AUC:", roc_auc)
print("Overall Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)


cv_results = pd.DataFrame(bayes_search.cv_results_)
cv_results.to_excel(r'UOB.xlsx', index=False)