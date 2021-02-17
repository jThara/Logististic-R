#!/usr/bin/env python
# coding: utf-8

# # Data Collection

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv(r'C:\Users\thara\Downloads\archive (3).zip')


# In[3]:


df.head()


# In[4]:


df.shape


# # Data Analysis

# In[5]:


import seaborn as sns


# In[6]:


sns.countplot(x="Purchased", data=df)


# In[7]:


df["Purchased"].value_counts()


# In[8]:


sns.countplot(x="Purchased", hue="Gender", data=df)


# In[9]:


sns.boxplot(x='Purchased', y='EstimatedSalary', data= df)


# In[10]:


sns.boxplot(x="Purchased", y="Age", data=df)


# In[11]:


sns.scatterplot(df["Age"], df["EstimatedSalary"])


# In[12]:



sns.distplot(df['EstimatedSalary'],bins=50,kde=False)


# # Data Wrangling

# In[13]:


df.isnull().sum()


# In[14]:


df["Gender"]=pd.get_dummies(df["Gender"], drop_first=True)


# # Train and Test

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


Feature=df.iloc[:,[1,2,3]].values
Feature


# In[17]:


X=Feature
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[18]:


y=df.iloc[:,[4]].values
y


# In[19]:


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3,random_state=0)


# In[20]:


X_test.shape


# In[21]:


y_test.shape


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


LR=LogisticRegression()


# In[24]:


LR.fit(X_train, y_train)


# In[25]:


prediction=LR.predict(X_test)


# In[26]:


from sklearn.metrics import classification_report


# In[27]:


classification_report(y_test, prediction)


# In[28]:


from sklearn.metrics import confusion_matrix


# In[29]:


confusion_matrix(y_test, prediction)


# In[30]:


from sklearn.metrics import accuracy_score


# In[31]:


accuracy_score(y_test, prediction)*100


# In[ ]:




