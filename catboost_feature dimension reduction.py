import catboost as cb
import numpy as np
import pandas as pd

# 读取数据集
data = pd.read_csv('Processed_data.csv', encoding='utf-8')

n = 20

# 分割特征属性和目标属性
X = data.drop(['Label', 'Flow Duration'], axis=1)  # 特征属性
y = data['Label']  # 目标属性
model = cb.CatBoostClassifier(iterations=100, depth=10, learning_rate=0.1)
model.fit(X, y)
feature_importance = model.feature_importances_
# 创建特征重要性字典
feature_importance_dict = dict(zip(X.columns, feature_importance))

# 根据特征重要性进行排序
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# 选择前n个重要特征
top_n_features = [feature[0] for feature in sorted_feature_importance[:n]]

print(top_n_features)

# data_selected = data[top_n_features]
# data_selected = np.array(data_selected)
# data_selected = np.insert(data_selected, data_selected.shape[1], y, axis=1)
# data_selected = pd.DataFrame(data_selected)
# data_selected.columns = [('feature' + str(i + 1)) for i in range(data_selected.shape[1])]
# data_selected.rename(columns={'feature'+str(data_selected.shape[1]): 'target'}, inplace=True)
# data_selected.to_csv("./数据降维/catboost方法提取重要特征.csv")
