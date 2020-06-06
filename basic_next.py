#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import defaultdict 
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tabulate import tabulate


# In[2]:


ADDR = '/nfs_home/nbhardwaj/data/rds_final/'
RES_ADDR = '/nfs_home/nbhardwaj/results/'


# - Basic idea is to capture all x->y relations in sequences of instr/addresses. 
# - Will capture same repetition, may not work for multiple such sub-sequences.
# - Won't be able to capture the drifting trends only the repetitions

# In[4]:


files = [510
         , 511, 526, 600, 602, 620, 623, 625, 631, 641, 648, 657  ]

sets = [x for x in range(64)]


# In[ ]:


# too slow 22m for 300M file

# acc = []
# len_ = []
# for fname in files:
#     for cset in sets:
#         df = pd.read_csv(ADDR+str(fname)+'_'+str(cset)+'.csv', index_col = [0], usecols = [0,2,3])
#         m = {}
#         f, nf = 0, 0
#         for i in range(len(df)):
#             inst = df.iloc[i, 0]
#             data = df.iloc[i, 1] 
#             if(inst in m.keys() and m[inst]==data):
#                 f+=1
#             else:
#                 nf+=1
#                 m[inst] = data
#         print("___>", cset, fname, f/(f+nf))
#         acc.append(f*100/(f+nf))
#         len_.append(len(df))
        
#     print("done for file->", fname)
#     df = pd.DataFrame(list(zip(acc, len_)), columns = ['accuracy', 'length'])
#     df.to_csv(RES_ADDR+'base_next.csv')


# In[34]:


m = {}
for fname in files:
    acc = []
    len_ = []
    for cset in sets:
        df = pd.read_csv(ADDR+str(fname)+'_'+str(cset)+'.csv', index_col = [0], usecols = [0,2,3])
        df = df.sort_values(by = 'Instruction', kind = 'mergesort')
        tots = len(df)
        pos = len(df[df.Data == df.Data.shift(periods =1)])
        print("___>", cset, fname, pos/tots)
        acc.append(pos*100/tots)
        len_.append(tots)
    m[fname] = np.dot(acc, len_)/np.sum(len_)
    print("done for file->", fname)
np.save(RES_ADDR+'basic_next.npy', m)


# In[36]:


m = np.load(RES_ADDR+'basic_next.npy', allow_pickle = True).item()
print(m)


# In[ ]:




