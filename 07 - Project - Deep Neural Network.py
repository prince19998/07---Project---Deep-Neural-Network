#!/usr/bin/env python
# coding: utf-8

# # Project: Deep Neural Network
# - Identify false banknotes

# ### Step 1: Import libraries

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# ### Step 2: Read the data
# - Use Pandas [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) method to read **files/banknotes.csv**

# In[4]:


data = pd.read_csv('files/banknotes.csv')
data.head()


# In[ ]:





# ### Step 3: Investitigate the data
# - Check how many classes (class)
#     - HINT: use [unique()](https://pandas.pydata.org/docs/reference/api/pandas.unique.html)
# - Check for missing data
#     - HINT: use [isna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html)[.sum()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html)

# In[5]:


data['class'].unique()


# In[6]:


data.isna().sum()


# ### Step 4: Divite data into feature vectors and labels
# - Assign the feature vectors to $X$
#     - HINT: that is all but the last column of the data
# - Assign the labels to $y$
#     - HINT: that is the last column (**class**)

# In[7]:


X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X.head()


# In[8]:


y.head()


# ### Step 5: Create training and test datasets
# - Split $X$ and $y$ into train and test sets using **train_test_split** with **test_size=.4**

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)


# In[ ]:





# ### Step 6: Create and compile the model
# - Create a **Sequential** model
#     - **Dense** with 8 nodes with **input_dim=4, activaition='relu'**
#     - **Dense** with 1 (the output node) with **activaition='sigmoid'**
# - Complie the model with **optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']**

# In[10]:


model = Sequential()
model.add(Dense(8, input_dim = 4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:





# ### Step 7: Fit and test the accuracy
# - Fit the model on training data with **epochs=20**
# - Evaluate the model with test data with **verbose=2**

# In[11]:


model.fit(X_train, y_train, epochs=20)
model.evaluate(X_test, y_test, verbose=2)


# In[ ]:





# ### Step 8 (Optional): Add another hidden layer
# - Add another hidden layer in the model
# - Test performance

# In[12]:


model = Sequential()
model.add(Dense(8, input_dim = 4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[13]:


model.fit(X_train, y_train, epochs=20)
model.evaluate(X_test, y_test, verbose=2)


# In[ ]:




