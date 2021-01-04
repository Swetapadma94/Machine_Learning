#!/usr/bin/env python
# coding: utf-8

# In[14]:


import statsmodels.api as sm


# In[22]:


df=pd.read_csv(r'E:\Krish naik\python code\Machine Learning\multi colinearity\Salary_Data.csv',encoding='latin1')


# In[23]:


df


# In[25]:


df.columns


# In[26]:


X=df[['YearsExperience']]
y=df['Salary']


# In[27]:


###33Fit a OLS model with intercept 
X=sm.add_constant(X)
X


# In[29]:


model = sm.OLS(y,X).fit()


# In[30]:


model.summary()


# In[31]:


X.iloc[:,1:].corr()


# In[ ]:




