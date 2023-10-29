import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv('CatBoostFeature.csv', encoding='utf-8')

X = df.drop('Label', axis=1)
y = df['Label']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, shuffle=True)

d = 2

y_label = []
for i in y_train:
    if i == 0:
        y_label.append('VPN')
    else:
        y_label.append('Non VPN')

# 训练t-SNE模型
tsne = TSNE(n_components=d)
X_tsne = tsne.fit_transform(X_train)
tsne_df = pd.DataFrame({'X': X_tsne[:, 0], 'Y': X_tsne[:, 1], 'label': y_label})

# 绘制散点图
sns.scatterplot(x='X', y='Y', hue='label', data=tsne_df, legend='full')
plt.xlabel('t-SNE降维后x坐标')
plt.ylabel('t-SNE降维后y坐标')
plt.show()
