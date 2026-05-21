#!/usr/bin/env python
# coding: utf-8

# In[1]:


#第一部分：lazypredict选模型


# In[1]:


import tqdm
import tqdm.notebook

# 关键：让 tqdm.notebook.tqdm 指向普通 tqdm.tqdm
tqdm.notebook.tqdm = tqdm.tqdm
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[2]:


file_path = r"E:/Research/数据/Desktop Data/最终表格3.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet3')
#标准化
cols = ["hour","to_station_duration_s"]
df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
df["fairness"] = -(df["hour"] + df["to_station_duration_s"]) / 2
y=df["fairness"]
#y=df["Equity"]
#y=df["Price"]
floor_map = {
    "low": 1,
    "medium": 2,
    "high": 3
}
df["楼层"] = df["楼层"].str.replace(r"共\d+层", "", regex=True)
df["楼层"] = df["楼层"].str.replace(r"[()]", "", regex=True)
df["楼层"] = df["楼层"].str.strip()
df["floor_ord"] = df["楼层"].map(floor_map)
df["direction"]=df["direction"].notna().astype(int)
print(df["floor_ord"])
print(df["direction"])
#df = pd.get_dummies(df, columns=["朝向"], drop_first=False)
X=df.drop(columns=["closeness_centrality","degree",'business','residential','entropy', "rooms","floor_ord","low_high_floor","Equity","entropy_norm","hour均值", "hour标准差", "duration均值","duration方差","station_X","station_Y","NEAR_DIST","户型","楼层","fairness","FID","community","business_area","挂牌价","成交价","成交周","建筑类","竣工时","NEAR_FID","station_name","CBD_X","CBD_Y","hour","XY","station_XY","to_station_duration_s"])
X = pd.get_dummies(X, drop_first=False)
print(X.columns)
# 划分数据集为训练集和测试集


# In[25]:


file_path = r"E:/Research/数据/Desktop Data/最终表格3.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet3')
#标准化
cols = ["hour","to_station_duration_s"]
df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
df["fairness"] = -(df["hour"] + df["to_station_duration_s"]) / 2
y=df["Price"]
#y = np.log(df["Price"])
#y=df["Equity"]
#y=df["Price"]

df["楼层"] = df["楼层"].str.replace(r"共\d+层", "", regex=True)
df["楼层"] = df["楼层"].str.replace(r"[()]", "", regex=True)
df["楼层"] = df["楼层"].str.strip()

df["direction"]=df["direction"].notna().astype(int)
print(df["direction"])

#df = pd.get_dummies(df, columns=["朝向"], drop_first=False)
X=df.drop(columns=["楼层","Price","closeness_centrality","degree",'residential','business', 'entropy',"Equity","entropy_norm","hour均值", "hour标准差", "duration均值","duration方差","station_X","station_Y","NEAR_DIST","户型","楼层","FID","community","business_area","挂牌价","成交价","成交周","建筑类","竣工时","NEAR_FID","station_name","CBD_X","CBD_Y","hour","XY","station_XY","to_station_duration_s"])
X = pd.get_dummies(X, drop_first=False,dtype=int)
print(X.columns)
print(X["low_high_floor_high"])
# 划分数据集为训练集和测试集


# In[22]:


print(X.dtypes)
print(y.dtype)


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# 初始化LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

# 训练模型并比较
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
# 打印性能报告
print(models)


# In[3]:


import shap


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators":[500,800],
    "max_depth":[10,15,20],
    "min_samples_leaf":[2,5],
    "max_features":["sqrt","log2"]
}

rf = RandomForestRegressor(random_state=42)

regressor = GridSearchCV(rf,param_grid,cv=5)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn import metrics
print("R2 Score:", metrics.r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[5]:


print(regressor.best_params_)


# In[6]:


# Tree SHAP
best_model = regressor.best_estimator_
explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X)

# summary plot
shap.summary_plot(shap_values, X, max_display=50)

# dependence plot
#shap.dependence_plot("Equity", shap_values, X)


# In[7]:


shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])


# In[14]:


feature_name="Price"
shap.dependence_plot(
    feature_name,
    shap_values.values,
    X_test,
    interaction_index="auto"
)



# In[8]:


df_0 = pd.DataFrame({
    "feature": X_test.columns,
    "shap_value": shap_values[0]
})
df_0


# In[9]:


importance = best_model.feature_importances_

# 变成表格
feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
}).sort_values("importance", ascending=False)

print(feat_imp)


# In[11]:


#交叉验证
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
est = RandomForestRegressor(
    n_estimators=800,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)


# In[18]:


kfold = KFold(n_splits=5,shuffle=True,random_state=1)
scores = cross_val_score(est, X, y, cv=kfold, scoring="r2")
print(scores.mean(), scores.std())


# In[26]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score
groups = df["business_area"]
est = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)
cv = GroupKFold(n_splits=5)
scores = cross_val_score(est, X, y, cv=cv,groups=groups,scoring="r2")
print(scores.mean(), scores.std())


# In[27]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score
groups = df["community"]
est = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)
cv = GroupKFold(n_splits=5)
scores = cross_val_score(est, X, y, cv=cv,groups=groups,scoring="r2")
print(scores.mean(), scores.std())


# In[10]:


#线性回归检验


# In[28]:


import statsmodels.api as sm
# 添加常数项（截距）
X = sm.add_constant(X)
#X = X.drop(columns="low_high_floor_high")
# OLS回归
model = sm.OLS(y, X).fit()

# 输出结果
print(model.summary())


# In[29]:


# 计算VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) 
              for i in range(X.shape[1])]

print(vif)


# In[11]:


#K-means聚类
#画散点图
import matplotlib.pyplot as plt
df["Price_std"]=(df["Price"]-df["Price"].mean())/df["Price"].std()
df["Fairness_std"]=(df["fairness"]-df["fairness"].mean())/df["fairness"].std()
x=df["Price_std"]
y=df["Fairness_std"]
fig, ax = plt.subplots(figsize=(8, 6))

# 四个象限分别画
ax.scatter(x[(x >= 0) & (y >= 0)], y[(x >= 0) & (y >= 0)],
           color=(213/255, 94/255, 0/255),
           alpha=0.7, s=20, label='High Price - High Equity')

ax.scatter(x[(x < 0) & (y >= 0)], y[(x < 0) & (y >= 0)],
           color=(0/255, 158/255, 115/255),
           alpha=0.7, s=20, label='Low Price - High Equity')

ax.scatter(x[(x < 0) & (y < 0)], y[(x < 0) & (y < 0)],
           color=(86/255, 180/255, 233/255),
           alpha=0.7, s=20, label='Low Price - Low Equity')

ax.scatter(x[(x >= 0) & (y < 0)], y[(x >= 0) & (y < 0)],
           color=(204/255, 121/255, 167/255),
           alpha=0.7, s=20, label='High Price - Low Equity')

# 参考线
ax.axvline(0, color='black', linewidth=1)
ax.axhline(0, color='black', linewidth=1)

ax.set_xlabel("Price", fontsize=12)
ax.set_ylabel("Equity", fontsize=12)
ax.set_title("Price-Equity Scatter", fontsize=14)

ax.grid(alpha=0.2)
ax.legend(frameon=False)

total = len(x)

q1 = ((x >= 0) & (y >= 0)).sum() / total  # 右上
q2 = ((x < 0) & (y >= 0)).sum() / total   # 左上
q3 = ((x < 0) & (y < 0)).sum() / total    # 左下
q4 = ((x >= 0) & (y < 0)).sum() / total   # 右下
ax.text(3, 1,
        f"{q1*100:.1f}%",
        ha='center', va='center',
        fontsize=9,
        bbox=dict(boxstyle="circle,pad=0.6",
                  fc=(213/255,94/255,0/255),  # 和点一样的颜色
                  ec="none",
                  alpha=0.1))
ax.text(-1.8, 1,
        f"{q2*100:.1f}%",
        ha='center', va='center',
        fontsize=9,
        bbox=dict(boxstyle="circle,pad=0.6",
                  fc=(0/255, 158/255, 115/255),  # 和点一样的颜色
                  ec="none",
                  alpha=0.1))
ax.text(-1.8, -2.5,
        f"{q3*100:.1f}%",
        ha='center', va='center',
        fontsize=9,
        bbox=dict(boxstyle="circle,pad=0.6",
                  fc=(86/255, 180/255, 233/255),  # 和点一样的颜色
                  ec="none",
                  alpha=0.1))
ax.text(3, -2.5,
        f"{q4*100:.1f}%",
        ha='center', va='center',
        fontsize=9,
        bbox=dict(boxstyle="circle,pad=0.6",
                  fc=(204/255, 121/255, 167/255),  # 和点一样的颜色
                  ec="none",
                  alpha=0.1))



plt.show()


# In[ ]:




