#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston


# In[2]:


df=load_boston()


# In[3]:


df


# In[4]:


data=pd.DataFrame(df.data)


# In[5]:


data.head()


# In[6]:


data.columns=df.feature_names


# In[7]:


data.head()


# In[8]:


df.target.shape


# In[9]:


data["Price"]=df.target


# In[10]:


data.Price


# In[11]:


data.head()


# In[12]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[13]:


x


# Linear Regression

# In[14]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
linear=LinearRegression()
mse=cross_val_score(linear,x,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)


# In[15]:


mean_mse


# #Ridge Regression
# 

# In[20]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x,y)


# In[21]:



print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# Lasso

# In[23]:



from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(x,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[25]:


prediction_lasso=lasso_regressor.predict(X_test)
prediction_ridge=ridge_regressor.predict(X_test)


# In[26]:


import seaborn as sns

sns.distplot(y_test-prediction_lasso)


# In[27]:



import seaborn as sns

sns.distplot(y_test-prediction_ridge)


# In[ ]:




