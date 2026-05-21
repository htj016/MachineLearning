# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:47:46 2026

@author: Ted Huang
"""

import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
data=sm.datasets.engel.load_pandas().data
data.shape
data.head()
model = smf.ols('foodexp ~ income',data=data)
results = model.fit()
results.params
print(results.summary())
sns.regplot(x='income',y='foodexp',data=data)
#柯布道格拉斯方程
import pandas as pd
df=pd.read_csv("D:/学校相关/DM&ML/MLPython_Data/MLPython_Data/cobb_douglas.csv")
print(df.columns)
model=smf.ols()

#波士顿房价回归
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]
df = pd.DataFrame(data, columns=feature_names)
df['MEDV']=target
print(df.columns)
import statsmodels.api as sm
X=df.drop(columns='MEDV',axis=1)
y=df['MEDV']
model=sm.OLS(y,X).fit()
print(model.summary())
#划分训练集测试集进行回归
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)
model_ml = LinearRegression().fit(X_train, y_train)
pred = model_ml.predict(X_test)
MSE=mean_squared_error(y_test, pred)
R2=r2_score(y_test, pred)
print(MSE)
print(R2)
#week1小作业
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import seaborn as sns
al12=pd.read_excel("D:/学校相关/DM&ML/al12-1.xlsx")
print(al12.columns)
X=al12.drop(columns='profit')
X = pd.get_dummies(X, columns=["dq","year"], drop_first=True)
y=al12['profit']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=0)
model1 = LinearRegression().fit(X_train, y_train)
pred = model1.predict(X_test)
MSE=mean_squared_error(y_test, pred)
R2=r2_score(y_test, pred)
print(MSE)
print(R2)
X_1=al12.drop(columns=['dq','year','profit'])
y_1=al12['profit']
X_train, X_test, y_train, y_test = train_test_split(X_1,
                                                    y_1,
                                                    test_size=0.1,
                                                    random_state=0)
model2 = LinearRegression().fit(X_train, y_train)
pred = model2.predict(X_test)
MSE_without_dummy=mean_squared_error(y_test, pred)
R2_without_dummy=r2_score(y_test, pred)
print(MSE_without_dummy)
print(R2_without_dummy)
# Cross Validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10,shuffle=True, random_state=1)
scores = cross_val_score(model1, X, y, cv=kfold,scoring="r2")
scores
print(scores.mean())
print(scores.std())