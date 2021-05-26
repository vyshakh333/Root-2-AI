#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd         # importing the nessary libaries
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[91]:


from pandas import ExcelWriter
from pandas import ExcelFile


# In[92]:


data=pd.read_excel("root2ai.xlsx")  # To read the dataset as we used to work


# In[93]:


data


# In[94]:


type(data)


# In[95]:


data.head()


# In[96]:


data.tail()


# In[97]:


data.describe()   # To describe the dataset overall


# In[98]:


data.isnull().sum()     # Finding the null values


# In[106]:


from sklearn.model_selection import train_test_split  # By importing the train test split form sklearn


# In[109]:


x_train,x_test,y_train,y_test=train_test_split(data.Text,data.Target,train_size=0.9)


# In[110]:


x_train


# In[111]:


len(x_test)


# In[112]:


len(y_test)


# In[113]:


len(y_train)


# In[114]:


from sklearn.linear_model import LogisticRegression #To import the logistic regression used to finding the probability of a target variables


# In[115]:


model=LogisticRegression()


# In[119]:


model.fit(x_train,y_train) #The given dataset are in text format so e cant fit the variables,ValueError: could not convert string to float.
                          #we should have to convert numeric or should use nlp by this dataset, here we can solve the problem.


# In[ ]:


model.score(x_test,y_test)*100  # The way of finding score.

