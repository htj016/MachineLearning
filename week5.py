# -*- coding: utf-8 -*-
# Split samples 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

diabete=pd.read_csv(r"D:\学校相关\DM&ML\MLPython_Data\MLPython_Data\PimaIndiansDiabetes.csv")
diabete.head()
diabete.describe()


d = {'neg': '0', 'pos': '1'}
diabete['diabetes'] = diabete['diabetes'].map(d)


pd.options.display.max_columns = 40 
diabete.head(2)

diabete.iloc[:,:3].describe()



sns.boxplot(x='diabetes', y='age', data=diabete)
sns.boxplot(x='diabetes', y='mass', data=diabete)
sns.boxplot(x='diabetes', y='pregnant', data=diabete)
X=diabete.drop(columns=['diabetes'])
y=diabete['diabetes']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

np.mean(X_train_s, axis=0)
np.std(X_train_s, axis=0)
np.mean(X_test_s, axis=0)
np.std(X_test_s, axis=0)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
pred = model.predict(X_test_s)
pred

pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])


# Choose optimal K via test set

scores = []
ks = range(1, 80)
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s, y_train)
    score = model.score(X_test_s, y_test)
    scores.append(score)
max(scores)
index_max = np.argmax(scores)
print(f'Optimal K: {ks[index_max]}')

# Graph accuracy versus K 

plt.plot(ks, scores, 'o-')
plt.xlabel('K')
plt.axvline(ks[index_max], linewidth=1, linestyle='--', color='k')
plt.ylabel('Accuracy')
plt.title('KNN')
plt.tight_layout()

# Graph error rate versus K

errors = 1 - np.array(scores)
plt.plot(ks, errors, 'o-')
plt.xlabel('K')
plt.axvline(ks[index_max], linewidth=1, linestyle='--', color='k')
plt.ylabel('Error Rate')
plt.title('KNN')
plt.tight_layout()

# Graph error rate versus 1/K

errors = 1 - np.array(scores)
ks_inverse = 1 / np.array(ks)
plt.plot(ks_inverse, errors, 'o-')
plt.xlabel('1/K')
plt.ylabel('Error Rate')
plt.title('KNN')
plt.tight_layout()

# Choose optimal K via CV

param_grid = {'n_neighbors': range(1, 51)}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kfold)
model.fit(X_train_s, y_train)

model.best_params_
model.score(X_test_s, y_test)

## Decision boundary for KNN on iris data

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


X2 = diabete[['age','triceps']].to_numpy()
y=diabete['diabetes'].to_numpy().astype(int)

# The plot below is time-consuming

fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.1, wspace=0.1)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X2, y)
plot_decision_regions(X2, y, model)
plt.xlabel('age')
plt.ylabel('triceps')
plt.tight_layout()




#RF Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
rf = RandomForestClassifier(
n_estimators=100, # 树的数量
max_depth=5, # 最大深度
oob_score=True, # 启用袋外评估
random_state=42
)

# 训练模型
rf.fit(X_train_s, y_train)

# 预测与评估
y_pred = rf.predict(X_test_s)
print("准确率:", accuracy_score(y_test, y_pred))
print("OOB分数:", rf.oob_score_)
print("分类报告:\n", classification_report(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))