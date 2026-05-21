

#PCA reg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits import mplot3d

### PCA with the audiometric data


abalone_0 = pd.read_csv(r"D:\学校相关\DM&ML\MLPython_Data\MLPython_Data\abalone.csv")
type(abalone_0)
abalone_0.shape
abalone_0.head()
abalone=abalone_0.drop(columns=["Type","Rings"])
pd.options.display.max_columns = 10
round(abalone.corr(), 2)

np.mean(abalone, axis=0)
np.std(abalone, axis=0)

scaler = StandardScaler()
scaler.fit(abalone)
X = scaler.transform(abalone)

np.mean(X, axis=0)
np.std(X, axis=0)

model = PCA()
model.fit(X)

model.explained_variance_

plt.plot(model.explained_variance_, 'o-')
plt.axhline(model.explained_variance_[2], color='k', linestyle='--', linewidth=1)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')

model.explained_variance_ratio_

plt.plot(model.explained_variance_ratio_, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('PVE')
plt.figure()
plt.plot(model.explained_variance_ratio_.cumsum(), 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.axhline(0.9, color='k', linestyle='--', linewidth=1)
plt.title('Cumulative PVE')
plt.figure()
model.components_
columns = ['PC' + str(i) for i in range(1, model.components_.shape[0] + 1)]

pca_loadings = pd.DataFrame(
    model.components_.T,
    index=abalone.columns,
    columns=columns
)

round(pca_loadings, 2)

# Visualize pca loadings

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.8, wspace=0.4)

for i in range(4):
    ax = axes[i // 2, i % 2]

    ax.plot(pca_loadings['PC' + str(i + 1)], 'o-')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)

    ax.set_xticks(range(len(abalone.columns)))
    ax.set_xticklabels(abalone.columns, rotation=90)

    ax.set_title('PCA Loadings for PC' + str(i + 1))

# PCA Scores

pca_scores = model.transform(X)
pca_scores = pd.DataFrame(pca_scores, columns=columns)
pca_scores.shape
pca_scores.head()

# visualize pca scores via biplot

sns.scatterplot(x='PC1', y='PC2',data=pca_scores)
plt.title('Biplot')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
X=abalone
y=abalone_0["Rings"]
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

np.mean(X_train, axis=0)
np.std(X_train, axis=0)

np.mean(X_test, axis=0)
np.std(X_test, axis=0)

scores_mse = []
for k in range(1, 8):
    model = PCA(n_components=k)
    model.fit(X_train)
    X_train_pca = model.transform(X_train)
    loo = LeaveOneOut()
    mse = -cross_val_score(LinearRegression(), X_train_pca, y_train, 
                           cv=loo, scoring='neg_mean_squared_error')
    scores_mse.append(np.mean(mse))
min(scores_mse)
index = np.argmin(scores_mse)
print(index)

plt.plot(range(1, 8), scores_mse)
plt.axvline(index + 1, color='k', linestyle='--', linewidth=1)
plt.xlabel('Number of Components')
plt.ylabel('Mean Squared Error')
plt.title('Leave-one-out Cross-validation Error')
plt.tight_layout()

model = PCA(n_components = 6)
model.fit(X_train)

X_train_pca = model.transform(X_train)
X_test_pca = model.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_pca, y_train)

X_pca = np.vstack((X_train_pca, X_test_pca))
X_pca.shape

pred = reg.predict(X_pca)

reg.score(X_test_pca, y_test)

reg_1=LinearRegression()
reg_1.fit(X_train,y_train)
pred_1=reg_1.predict(X_test)
reg_1.score(X_test,y_test)