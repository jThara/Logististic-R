#!/usr/bin/env python
# coding: utf-8

# # Data Collection

# In[1]:


import pandas as pd
import numpy as np
df=pd.read_csv(r'C:\Users\thara\Downloads\archive (2).zip')
df.head()
df["Survived"].value_counts()


# In[2]:


len(df["PassengerId"])
df.info()


# In[3]:


df['Cabin']


# 
# ## Analyzing Data

# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


sns.countplot(x="Survived", data=df)


# In[6]:


sns.countplot(x="Survived", hue="Sex", data=df)
df["Survived"].value_counts()


# In[7]:


sns.countplot(x="Survived", hue="Pclass", data=df)


# In[8]:


df['Age'].plot.hist()


# In[9]:


df.info()


# In[10]:


sns.countplot(x="Parch", data=df)


# # Data Wrangling

# In[11]:


df.isnull()


# In[12]:


df.isnull().sum()


# In[13]:


sns.heatmap(df.isnull(), yticklabels=False)
df["Survived"].value_counts()


# In[14]:


sns.boxplot(x='Class', y='Age', data=df)


# In[15]:


df.head(5)
df["Survived"].value_counts()


# In[16]:


df.drop("Cabin", axis=1, inplace=True)


# In[17]:


df["Survived"].value_counts()


# In[18]:


df.isnull().sum()


# In[19]:


df.info()


# In[20]:


df.drop("Body", axis=1, inplace=True)


# In[21]:


df["Survived"].value_counts()


# In[22]:


df.isnull().sum()


# In[23]:


df.drop("Lifeboat", axis=1, inplace=True)


# In[24]:


df["Survived"].value_counts()


# In[25]:


df.dropna(subset=["Survived"], axis=0, inplace=True)


# In[26]:


df["Survived"].value_counts()


# In[27]:


df.isnull().sum()


# In[28]:


df.dropna(subset=["Age"], axis=0, inplace=True)


# In[29]:


df.isnull().sum()


# In[30]:


df["Survived"].value_counts()


# In[31]:


df.head()


# In[32]:


df.isnull().sum()


# In[33]:


#df.drop("Lifeboat", axis=1, inplace=True)


# In[34]:


df.isnull().sum()


# In[35]:


df.dropna(subset=["Age"], axis=0, inplace=True)


# In[36]:


df.info()


# In[37]:


sns.heatmap(df.isnull(), yticklabels=False)


# In[38]:


df["Sex"]=pd.get_dummies(df["Sex"], drop_first=True)


# In[39]:


df["Pclass"].value_counts()


# In[40]:


df["Pclass"]=pd.get_dummies(df["Pclass"], drop_first=True)


# In[41]:


df["Embarked"]=pd.get_dummies(df["Embarked"],drop_first=True)


# In[42]:


df.head()


# In[43]:


df.drop(["PassengerId", "Name", "Ticket","WikiId", "Name_wiki", "Hometown","Boarded", "Destination", "Class"], axis=1, inplace=True)


# In[44]:


df.head()
df.reset_index(drop=True, inplace=True)
df.head()
df["Survived"].value_counts()


# # Train and Test

# In[45]:


Feature=df[["Pclass","Sex","Age","SibSp","Parch","Embarked"]]
X=Feature
X.shape


# In[46]:


y=df["Survived"]


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


# In[49]:



X_train.shape


# In[50]:


y_train.shape


# In[51]:


from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression()


# In[52]:


LR_model.fit(X_train, y_train)


# In[53]:


predict=LR_model.predict(X_test)


# In[54]:


from sklearn.metrics import classification_report


# In[55]:


classification_report(y_test, predict)


# In[56]:


from sklearn.metrics import confusion_matrix


# In[57]:


confusion_matrix(y_test, predict)


# In[ ]:





# # Accuracy Check

# In[58]:


from sklearn.metrics import accuracy_score


# In[59]:


accuracy_score(y_test, predict)


# In[ ]:




