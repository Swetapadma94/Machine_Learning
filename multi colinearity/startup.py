# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 21:22:57 2020

@author: NEW
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('E:\Assignment\multidataset\Startups.csv')
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

states=pd.get_dummies(X['State'],drop_first=True)
X=X.drop('State',axis=1)
X=pd.concat([X,states],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)


