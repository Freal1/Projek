#!/usr/bin/env python
# coding: utf-8

# ### IMPORT LIBRARIES

# In[2]:


pip install streamlit


# In[3]:


import streamlit as st


# In[72]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import numpy as np


# ### DATA UNDERSTANDING

# #### Data Exploration

# In[2]:


data = pd.read_csv('employee-attrition.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# ### DATA PREPARATION

# In[20]:


data.replace("", float("nan"), inplace=True)
data.replace("NA", float("nan"), inplace=True)
data.replace("NaN", float("nan"), inplace=True)

data = data.dropna(axis=0, how='any').dropna(axis=1, how='any')


# In[21]:


data.isnull().sum()


# In[22]:


categorical_cols = data.select_dtypes(include='object').columns
df = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

df.head()


# ### DATA VISUALIZATION

# In[47]:


data.hist(bins=10,figsize=(20,15))


# In[25]:


department_performance = data.groupby(['Department', 'PerformanceRating']).size().unstack()

for department in department_performance.index:
    ratings = department_performance.loc[department]
    labels = ratings.index
    sizes = ratings.values

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'Performance Ratings Distribution in {department} Department')
    plt.show()


# ## MACHINE LEARNING PREDICT

# ### MACHINE LEARNING - RANDOM FOREST 

# In[26]:


X = df.drop(columns=["Attrition_Yes"])
y = df["Attrition_Yes"]


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# #### Model

# In[28]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# In[29]:


y_pred = rf_model.predict(X_test)


# In[30]:


report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(report)


# In[31]:


print(accuracy)


# In[32]:


print(confusion)


# ### MACHINE LEARNING - SIMPLE LINEAR REGRESSION

# In[33]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[35]:


y_pred = lr.predict(X_test)
y_pred_class = [1 if x >= 0.5 else 0 for x in y_pred]


# In[45]:


meanSquared = mean_squared_error(y_test, y_pred)
Rsquared =  r2_score(y_test, y_pred)
confusionlr = confusion_matrix(y_test, y_pred_class)
reportlr = classification_report(y_test, y_pred_class)
accuracylr = accuracy_score(y_test, y_pred_class)


# In[40]:


print(meanSquared)


# In[41]:


print(Rsquared)


# In[42]:


print(confusionlr)


# In[43]:


print(reportlr)


# In[46]:


print(accuracylr)


# ## MACHINE LEARNING CLASSIFICATION

# ### MACHINE LEARNING - DECISION TREE

# In[51]:


dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)


# In[69]:


accuracy = accuracy_score(y_test, y_pred)
confusiondt = confusion_matrix(y_test, y_pred)
reportdt = classification_report(y_test, y_pred)
print("Akurasi: ", accuracy)


# In[70]:


print(confusiondt)


# In[71]:


print(reportdt)


# In[68]:


plt.figure(figsize=(12, 8))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['Tidak Churn', 'Churn'], rounded=True)
plt.show()


# In[ ]:




