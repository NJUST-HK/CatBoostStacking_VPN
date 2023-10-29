from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm
# from sklearn.pegasos import SVC
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import StackingClassifier
import warnings


class Pegasos:

    def __init__(self, lamb=0.01, max_epochs=1000):
        self.w = None
        self.lamb = lamb
        self.max_epochs = max_epochs
        self._estimator_type = "classifier"
        self.positive_ratio = 0.8
        self.negative_ratio = 0.2

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)
        t = 0
        for epoch in range(1, self.max_epochs + 1):
            for i in range(n_samples):
                t += 1
                if y[i] == 1:
                    eta = 1 / (self.positive_ratio * self.lamb * t)
                else:
                    eta = 1 / (self.negative_ratio * self.lamb * t)
                if y[i] * np.dot(x[i], self.w) < 1:
                    self.w = (1 - eta * self.lamb) * self.w + eta * y[i] * x[i]
                else:
                    self.w = (1 - eta * self.lamb) * self.w
        return self

    def predict(self, x):
        return np.sign(np.dot(x, self.w))

    def get_params(self, deep=False):
        return {'lamb': self.lamb, 'max_epochs': self.max_epochs}


warnings.filterwarnings("ignore")
test_size = 0.3

df = pd.read_csv("CatBoostFeature.csv", index_col=0)
# 计算每列的均值
mean_values = df.mean()

# 用均值替换空值
df = df.fillna(mean_values)

df = np.array(df)



X = df[:, 0:(df.shape[1]-1)]
Y = df[:, df.shape[1]-1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=True)

'''
模型训练
'''
lgb = lightgbm.sklearn.LGBMClassifier(max_depth=15, n_estimators=250, learning_rate=0.1,
                                      reg_alpha=0.5, reg_lambda=0.5, min_child_weight=0,
                                      colsample_bytree=0.8, subsample=0.8)
lgb.fit(X_train, y_train)


mlp = MLPClassifier(hidden_layer_sizes=(40, 80, 20), learning_rate_init=0.001, activation='relu',
                    solver='adam', max_iter=10000)
mlp.fit(X_train, y_train)

pegasos = Pegasos()
pegasos.fit(X_train, y_train)

cat = CatBoostClassifier()
cat.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=250, max_depth=20)
rf.fit(X_train, y_train)

et = ExtraTreesClassifier(n_estimators=250, max_depth=20)
et.fit(X_train, y_train)


'''
进行预测获取误差
'''
y_test_pre_lgb = lgb.predict(X_test)
precision_lgb = precision_score(y_test, y_test_pre_lgb, average="weighted")

y_test_pre_mlp = mlp.predict(X_test)
precision_mlp = precision_score(y_test, y_test_pre_mlp, average="weighted")

y_test_pre_pegasos = pegasos.predict(X_test)
precision_pegasos = precision_score(y_test, y_test_pre_pegasos, average="weighted")

y_test_pre_et = et.predict(X_test)
precision_et = precision_score(y_test, y_test_pre_et, average="weighted")

y_test_pre_cat = cat.predict(X_test)
precision_cat = precision_score(y_test, y_test_pre_cat, average="weighted")

y_test_pre_rf = rf.predict(X_test)
precision_rf = precision_score(y_test, y_test_pre_rf, average="weighted")

pre_w_max = np.array([precision_lgb, precision_mlp, precision_pegasos, precision_et, precision_cat,
                      precision_rf]).max(initial=None)
pre_w_min = np.array([precision_lgb, precision_mlp, precision_pegasos, precision_et, precision_cat,
                      precision_rf]).min(initial=None)
precision_lgb_ = (precision_lgb - pre_w_min) / (pre_w_max - pre_w_min)
precision_mlp_ = (precision_mlp - pre_w_min) / (pre_w_max - pre_w_min)
precision_pegasos_ = (precision_pegasos - pre_w_min) / (pre_w_max - pre_w_min)
precision_et_ = (precision_et - pre_w_min) / (pre_w_max - pre_w_min)
precision_cat_ = (precision_cat - pre_w_min) / (pre_w_max - pre_w_min)
precision_rf_ = (precision_rf - pre_w_min) / (pre_w_max - pre_w_min)

print("!!!")
'''
Enhancing VPN Traffic Recognition through CatBoost Feature Engineering and Stacking Ensemble Learning
'''
SuSuModel = StackingClassifier(estimators=[('lgb', lgb), ('mlp', mlp), ('et', et), ('cat', cat),
                                           ('rf', rf)],
                               final_estimator=MLPClassifier(hidden_layer_sizes=(50, 50), learning_rate_init=0.001,
                                                             activation='relu', solver='adam', max_iter=10000))
SuSuModel.fit(X_train, y_train)
y_pred = SuSuModel.predict(X_test)


# 计算四个评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: ", "{:.2%}".format(accuracy))
print("Precision: ", "{:.2%}".format(precision))
print("Recall: ", "{:.2%}".format(recall))
print("F1-Score: ", "{:.2%}".format(f1))

scanf("%d%d%d",&a, &b, &c)