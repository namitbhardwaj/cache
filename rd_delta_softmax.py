#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import defaultdict 
import sys
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

import keras 
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM, Flatten
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import time


# In[2]:


import multiprocessing
print(multiprocessing.cpu_count())


# In[ ]:





# In[54]:


ADDR = '/nfs_home/nbhardwaj/data/rds_final/'
w_ADDR = '/nfs_home/nbhardwaj/results/'


# In[55]:


files = [510
          , 511, 526, 600, 602, 620, 623, 625, 631, 641, 648, 657
         ]
sets = [
    x for x in range(64)
]


# In[25]:


df_m = pd.DataFrame(columns = ['fname', 'set', 'train_acc', 'test_acc', 'len'])

for fname in files:
    begin = time.time()
    for cset in sets:
        df = pd.read_csv(ADDR+str(fname)+'_'+str(cset)+'.csv', index_col = [0], usecols = [0, 2, 7, 8])
        X = df[['delta']].values[1:]
        y = df[['label']].values[1:].reshape((-1,))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
#         print(X_train[:5])
        clf = LogisticRegression(random_state = 42, n_jobs = -1, solver = 'sag')
        clf.fit(X_train, y_train)
        tr_ac =  clf.score(X_train, y_train)
        t_ac = clf.score(X_test, y_test)
        df_m.loc[len(df_m)] = [fname, cset, tr_ac, t_ac, len(df)]
    print("finished in ", time.time()-begin, "s")
    print("DONE FOR FILE", fname)
df_m.to_csv(w_ADDR+'softmax_delta.csv')
print("-----------ITS DONEXX----------")


# In[43]:


# a = df.delta.values[1:1000]
# plt.plot([x for x in range(len(a))], a)


# In[51]:


# d[::-1]


# In[52]:


# perc = []
# tot = np.sum(d)
# s = 0
# for x in d[::-1]:
#     s+=x
#     perc.append(s/tot)


# In[53]:


# plt.plot(perc)


# In[28]:


# acc = np.dot(df_m['test_acc'], df_m['len'])


# In[30]:


# print(acc/np.sum(df_m['len']))


# In[ ]:






# begin = time.time()
# X = df[['Instruction', 'delta', 'Mode']].values[1:]
# y = df[['label']].values[1:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# print(X_train[:5])
# clf = LogisticRegression(random_state = 42, n_jobs = -1, solver = 'sag').fit(X_train, y_train)
# print("training acc->", clf.score(X_train, y_train))
# print("testing acc->", clf.score(X_test, y_test))
# print("finished in ", time.time()-begin, "s")
# print(clf.coef_)
# print(clf.intercept_)



# begin = time.time()
# X = df[['Instruction']].values[1:]
# y = df[['label']].values[1:].reshape((-1,))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# print(X_train[:5])
# clf4 = LogisticRegression(random_state = 42, n_jobs = -1, solver = 'sag').fit(X_train, y_train)
# clf4.fit(X_train, y_train)
# print("training acc->", clf4.score(X_train, y_train))
# print("testing acc->", clf4.score(X_test, y_test))
# print("finished in ", time.time()-begin, "s")
# print(clf4.coef_)
# print(clf4.intercept_)

# begin = time.time()
# X = df[['delta']].values[1:]
# y = df[['label']].values[1:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# clf5 = LogisticRegression(random_state = 42, n_jobs = -1, solver = 'sag')
# clf5.fit(X_train, y_train)
# print("training acc->", clf5.score(X_train, y_train))
# print("testing acc->", clf5.score(X_test, y_test))
# print("finished in ", time.time()-begin, "s")
# print(clf5.coef_)
# print(clf5.intercept_)

# begin = time.time()
# X = df[['Mode']].values[1:]
# y = df[['label']].values[1:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# print(X_train[:5])
# clf6 = LogisticRegression(random_state = 42, n_jobs = -1, solver = 'sag')
# clf6.fit(X_train, y_train)
# print("training acc->", clf6.score(X_train, y_train))
# print("testing acc->", clf6.score(X_test, y_test))
# print("finished in ", time.time()-begin, "s")
# print(clf6.coef_)
# print(clf6.intercept_)

# begin = time.time()
# X = df[['Data']].values[1:]
# y = df[['label']].values[1:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)
# # scaler = MinMaxScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)
# print(X_train[:5])
# clf7 = LogisticRegression(random_state = 42, n_jobs = -1, solver = 'sag')
# clf7.fit(X_train, y_train)
# print("training acc->", clf7.score(X_train, y_train))
# print("testing acc->", clf7.score(X_test, y_test))
# print("finished in ", time.time()-begin, "s")
# print(clf7.coef_)
# print(clf7.intercept_)













# #### Log transformations



# X = df[['Instruction', 'delta', 'Mode']].values[1:]
# # X[:, 1] = np.log(X[:, 1])
# y = df[['label']].values[1:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# sign = np.where(X[:, 1]>0, 1, -1)
# X[:, 1] = np.multiply(np.log(np.abs(X[:, 1])), sign)

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(random_state = 0).fit(X_train, y_train)

# clf.score(X_train, y_train)

# clf.score(X_test, y_test)

# No effect of Log transformations or random state change

