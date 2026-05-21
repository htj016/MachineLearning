# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:48:50 2026

@author: Ted Huang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor,export_text
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import cohen_kappa_score
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
df.describe()
print(df[:10])
Boston = pd.DataFrame(data, columns=feature_names)

Boston['MEDV'] = target

X = Boston.drop(columns=['MEDV'])
y = Boston['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Regression Tree 

model = DecisionTreeRegressor(max_depth=2, random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)


feature_names_new = list(Boston.drop(columns=['MEDV']).columns)
print(export_text(model, feature_names=feature_names_new))


plot_tree(model, feature_names=feature_names_new, node_ids=True, rounded=True, precision=2)

# Graph total impurities versus ccp_alphas 

model = DecisionTreeRegressor(random_state=123)
path = model.cost_complexity_pruning_path(X_train, y_train)

plt.plot(path.ccp_alphas, path.impurities, marker='o', drawstyle='steps-post')
plt.xlabel('alpha (cost-complexity parameter)')
plt.ylabel('Total Leaf MSE')
plt.title('Total Leaf MSE vs alpha for Training Set')

max(path.ccp_alphas),  max(path.impurities)

# Choose optimal ccp_alpha via CV

param_grid = {'ccp_alpha': path.ccp_alphas} 
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeRegressor(random_state=123), param_grid, cv=kfold)
model.fit(X_train, y_train)

model.best_params_
model = model.best_estimator_
model.score(X_test,y_test)

plot_tree(model, feature_names=feature_names_new, node_ids=True, rounded=True, precision=2)

model.get_depth()
model.get_n_leaves()
model.get_params()

# Visualize Feature Importance
          
model.feature_importances_

sorted_index = model.feature_importances_.argsort()
sorted_index 

X = pd.DataFrame(X, columns=feature_names_new)

plt.barh(range(X.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]), X.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree')
plt.tight_layout()

# Visualize prediction fit

pred = model.predict(X_test)

plt.scatter(pred, y_test, alpha=0.6)
w = np.linspace(min(pred), max(pred), 100)
plt.plot(w, w)
plt.xlabel('pred')
plt.ylabel('y_test')
plt.title('Tree Prediction')

from sklearn.linear_model import LinearRegression
model_ols = LinearRegression().fit(X_train, y_train)
model_ols.score(X_test, y_test)
pred_ols=model_ols.predict(X_test)
plt.scatter(pred_ols,y_test)
m = np.linspace(min(pred_ols), max(pred_ols), 100)
plt.plot(m, m)
plt.xlabel('pred')
plt.ylabel('y_test')
plt.title('linear Prediction')

## Classification Tree with bank dataset

bank = pd.read_csv(r"D:\学校相关\DM&ML\MLPython_Data\MLPython_Data\bank-additional.csv", sep=';')

bank.shape

pd.options.display.max_columns = 70 
bank.head()

# Drop 'duration' variable
bank = bank.drop('duration', axis=1)

bank.y.value_counts()
bank.y.value_counts(normalize=True)

X_raw = bank.iloc[:, :-1]
X = pd.get_dummies(X_raw)
X.head(2)

y = bank.iloc[:, -1]
y = y.map({'no': 0, 'yes': 1})
print(y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=1000, random_state=1)

# Classification Tree 

model = DecisionTreeClassifier(max_depth=2, random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)
plot_tree(model, feature_names=X.columns, node_ids=True, rounded=True, precision=2)

# Graph total impurities versus ccp_alphas 

model = DecisionTreeClassifier(random_state=123)
path = model.cost_complexity_pruning_path(X_train, y_train)

plt.plot(path.ccp_alphas, path.impurities, marker='o', drawstyle='steps-post')
plt.xlabel('alpha (cost-complexity parameter)')
plt.ylabel('Total Leaf Impurities')
plt.title('Total Leaf Impurities vs alpha for Training Set')

max(path.ccp_alphas),  max(path.impurities)

# Choose optimal ccp_alpha via CV

param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, cv=kfold)
model.fit(X_train, y_train)     

model.best_params_

model = model.best_estimator_
model.score(X_test, y_test)

plot_tree(model, feature_names=X.columns, node_ids=True, impurity=True, proportion=True, rounded=True, precision=2)

# Feature importance

model.feature_importances_
plt.figure(figsize=(8, 10))
sorted_index = model.feature_importances_.argsort()
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index],fontsize=6)

plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree')
plt.tight_layout()

# Prediction Performance 
     
pred = model.predict(X_test)
table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
table
import seaborn as sns
plt.figure(figsize=(5,4))
sns.heatmap(table, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

table = np.array(table)
Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Sensitivity  = table[1, 1] / (table[1, 0] + table[1, 1])
Sensitivity

from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_test, pred)

# Use a different threshold for prediction

prob = model.predict_proba(X_test)
prob
model.classes_

prob_yes = prob[:, 1]
pred_new = (prob_yes >= 0.1)

table = pd.crosstab(y_test, pred_new, rownames=['Actual'], colnames=['Predicted'])
table

table = np.array(table)
Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Sensitivity  = table[1, 1] / (table[1, 0] + table[1, 1])
Sensitivity

## Entropy criterion

param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=123), param_grid, cv=kfold)

model.fit(X_train, y_train)     
model.score(X_test, y_test)

pred = model.predict(X_test)
pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])

## Decision boundary for iris data

from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions

X,y = load_iris(return_X_y=True)
X2 = X[:, 2:4]

model = DecisionTreeClassifier(random_state=123)
path = model.cost_complexity_pruning_path(X2, y)
param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, cv=kfold)
model.fit(X2, y)
model.score(X2, y)

plot_decision_regions(X2, y, model)
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision Boundary for Decision Tree')

mushroom=pd.read_csv("D:\学校相关\DM&ML\MLPython_Data\MLPython_Data\Mushroom.csv")
mushroom.head()
y=mushroom["Class"]
y[:5]
y = y.map({'poisonous': 0, 'edible': 1})
X=mushroom.iloc[:,1:]
X.head()
X=pd.get_dummies(X)
X.head(2)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

# Classification Tree 

model = DecisionTreeClassifier(max_depth=2, random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)
plt.figure(figsize=(20,30))
plot_tree(model, feature_names=X.columns, node_ids=True, rounded=True, precision=2)
# Graph total impurities versus ccp_alphas 

model = DecisionTreeClassifier(random_state=123)
path = model.cost_complexity_pruning_path(X_train, y_train)

plt.plot(path.ccp_alphas, path.impurities, marker='o', drawstyle='steps-post')
plt.xlabel('alpha (cost-complexity parameter)')
plt.ylabel('Total Leaf Impurities')
plt.title('Total Leaf Impurities vs alpha for Training Set')

max(path.ccp_alphas),  max(path.impurities)

# Choose optimal ccp_alpha via CV

param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, cv=kfold)
model.fit(X_train, y_train)     

model.best_params_

model = model.best_estimator_
model.score(X_test, y_test)

plot_tree(model, feature_names=X.columns, node_ids=True, impurity=True, proportion=True, rounded=True, precision=2)

# Feature importance

model.feature_importances_
plt.figure(figsize=(8, 10))
sorted_index = model.feature_importances_.argsort()
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index],fontsize=6)

plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree')
plt.tight_layout()

# Prediction Performance 
     
pred = model.predict(X_test)
table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
table
import seaborn as sns
plt.figure(figsize=(5,4))
sns.heatmap(table, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

table = np.array(table)
Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Sensitivity  = table[1, 1] / (table[1, 0] + table[1, 1])
Sensitivity

from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_test, pred)

# Use a different threshold for prediction

prob = model.predict_proba(X_test)
prob
model.classes_

prob_yes = prob[:, 1]
pred_new = (prob_yes >= 0.1)

table = pd.crosstab(y_test, pred_new, rownames=['Actual'], colnames=['Predicted'])
table

table = np.array(table)
Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Sensitivity  = table[1, 1] / (table[1, 0] + table[1, 1])
Sensitivity

## Entropy criterion

param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=123), param_grid, cv=kfold)

model.fit(X_train, y_train)     
model.score(X_test, y_test)

pred = model.predict(X_test)
pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
