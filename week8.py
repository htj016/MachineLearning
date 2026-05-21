# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:42:43 2026

@author: Ted Huang
"""


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


spam = pd.read_csv(r"D:\学校相关\DM&ML\MLPython_Data\MLPython_Data\spam.csv")
type(spam)
spam.shape
spam.head()
spam=spam.drop(columns=["spam"])
pd.options.display.max_columns = 10
round(spam.corr(), 2)

np.mean(spam, axis=0)
np.std(spam, axis=0)

scaler = StandardScaler()
scaler.fit(spam)
X = scaler.transform(spam)

np.mean(X, axis=0)
np.std(X, axis=0)

model = PCA()
model.fit(X)

model.explained_variance_

plt.plot(model.explained_variance_, 'o-')
plt.axhline(model.explained_variance_[3], color='k', linestyle='--', linewidth=1)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')

model.explained_variance_ratio_

plt.plot(model.explained_variance_ratio_, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('PVE')

plt.plot(model.explained_variance_ratio_.cumsum(), 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.axhline(0.9, color='k', linestyle='--', linewidth=1)
plt.title('Cumulative PVE')

model.components_
columns = ['PC' + str(i) for i in range(1, model.components_.shape[0] + 1)]

pca_loadings = pd.DataFrame(
    model.components_.T,
    index=spam.columns,
    columns=columns
)

round(pca_loadings, 2)

# Visualize pca loadings

fig, axes = plt.subplots(2, 2, figsize=(50, 8))
plt.subplots_adjust(hspace=0.8, wspace=0.4)

for i in range(4):
    ax = axes[i // 2, i % 2]

    ax.plot(pca_loadings['PC' + str(i + 1)], 'o-')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)

    ax.set_xticks(range(len(spam.columns)))
    ax.set_xticklabels(spam.columns, rotation=90)

    ax.set_title('PCA Loadings for PC' + str(i + 1))

# PCA Scores

pca_scores = model.transform(X)
pca_scores = pd.DataFrame(pca_scores, columns=columns)
pca_scores.shape
pca_scores.head()

# visualize pca scores via biplot

sns.scatterplot(x='PC1', y='PC2',data=pca_scores)
plt.title('Biplot')

# Visualize pca scores via triplot
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_scores['PC1'], pca_scores['PC2'], pca_scores['PC3'], c='b')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=1, n_init=20)
model.fit(X)
model.labels_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_scores['PC1'], pca_scores['PC2'], pca_scores['PC3'],
           c=model.labels_, cmap='rainbow')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
