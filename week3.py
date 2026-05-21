#!/usr/bin/env python
# coding: utf-8

# In[2]:


#"D:\学校相关\DM&ML\MLPython_Data\MLPython_Data\wine.csv"
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

Wine = pd.read_csv(r'D:\学校相关\DM&ML\MLPython_Data\MLPython_Data\wine.csv')
Wine.shape

Wine.head()

Wine.Type.value_counts()
X = Wine.iloc[:,1:]
y = Wine.iloc[:,0]
print(X.columns)
print(y)
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, stratify=y, random_state=0)

# model =  LogisticRegression(multi_class='multinomial',solver = 'newton-cg', C=1e10, max_iter=1000)

model =  LogisticRegression(solver = 'newton-cg', C=1e10, max_iter=1000)  # 不用再写multinomal
model.fit(X_train, y_train)

model.n_iter_
model.intercept_
model.coef_
model.score(X_test, y_test)    # Accuracy

prob = model.predict_proba(X_test)
prob[:3]

pred = model.predict(X_test)
pred[:5]

table = confusion_matrix(y_test, pred)
table

# For better display,use pandas 
table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
table
print(table.dtypes)
print(table.values.dtype)

sns.heatmap(table,cmap='Blues', annot=True)
plt.tight_layout()

print(classification_report(y_test, pred))

cohen_kappa_score(y_test, pred)


# In[20]:


### Chapter 7 Discriminant Analysis

# Decriptive Statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
seed_df=pd.read_csv(r'D:\学校相关\DM&ML\MLPython_Data\MLPython_Data\seeds_dataset.csv')

seed_df.head()
seed_df.shape


#seed_df.Type.value_counts()
X = seed_df.iloc[:,:-1]
y = seed_df.iloc[:,-1]
print(X.columns)
print(y)

X.corr()
sns.heatmap(X.corr(), cmap='Blues', annot=True)




# In[21]:


# LDA for full sample

model = LinearDiscriminantAnalysis()
model.fit(X, y)
print(model.score(X, y))

print(model.priors_)
print(model.means_)

print(model.explained_variance_ratio_)

print(model.scalings_)





# In[22]:


lda_loadings = pd.DataFrame(model.scalings_, index=X.columns, columns=['LD1', 'LD2'])
lda_loadings

lda_scores = model.fit(X, y).transform(X)
lda_scores.shape
lda_scores[:5, :]

lda_scores = model.fit_transform(X, y)

LDA_scores = pd.DataFrame(lda_scores, columns=['LD1', 'LD2'])
LDA_scores['Type'] = y.values   # 直接用你的分类变量
LDA_scores.head()

d = {1: '1', 2: '2', 3: '3'}
LDA_scores['Type'] = LDA_scores['Type'].map(d) 
LDA_scores.head()

sns.scatterplot(x='LD1', y='LD2', data=LDA_scores, hue='Type')


# In[24]:


# Plot decision boundary for LDA with two features

X2 = X.iloc[:, 2:4]

model = LinearDiscriminantAnalysis()
model.fit(X2, y)
model.score(X2, y)
model.explained_variance_ratio_



# pip install mlxtend (machine learning extension library), or to avoid timeout
# pip --default-timeout=100 install mlxtend 

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(np.array(X2), np.array(y), model)   # 因为画图时用的是数组，和前面训练时用的数据库结构
# 上有不同，会提示警告，不用理会。。解决的方案之一是训练时也用数值model.fit(X2.values, y)

plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for LDA')


# In[27]:


# LDA for split sample

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
model.score(X_test, y_test)    # Accuracy

prob = model.predict_proba(X_test)
prob[:3]

pred = model.predict(X_test)
pred[:5]

confusion_matrix(y_test, pred)

print(classification_report(y_test, pred))

cohen_kappa_score(y_test, pred)

# QDA for split sample

model = QuadraticDiscriminantAnalysis(reg_param=0.1)
model.fit(X_train, y_train)
model.score(X_test, y_test)    # Accuracy

prob = model.predict_proba(X_test)
prob[:3]

pred = model.predict(X_test)
pred[:5]

confusion_matrix(y_test, pred)

print(classification_report(y_test, pred))

cohen_kappa_score(y_test, pred)

# Plot decision boundary for QDA with two features

X2 = X.iloc[:, 2:4]
model = QuadraticDiscriminantAnalysis()
model.fit(X2, y)
model.score(X2, y)

plot_decision_regions(np.array(X2), np.array(y), model)  # 和前面问题类似，该函数更加支持array，而非dataframe
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for QDA')


# In[ ]:





# In[ ]:




