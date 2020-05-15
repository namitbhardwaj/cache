#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from collections import defaultdict 
import sys
import os
import matplotlib.pyplot as plt
from tabulate import tabulate


# In[48]:


ADDR = '/nfs_home/nbhardwaj/data/set_data/400_'
ORIG_ADDR = '/nfs_home/SPEC_2006/400.perlbench/400.perlbench.pinatrace.csv'
final_ADDR = '/nfs_home/nbhardwaj/data/rds_data/400/'


# ### Testing integrity of split files

# In[25]:


# sets = [x for x in range(64)]
# f = open(ADDR+'test.csv', "a+")
# for curr_set in sets:
#     df = pd.read_csv(ADDR + str(curr_set)+'.csv')
#     for x in df.ICount.values:
#         f.write(str(x)+'\n')
# f.close()

# print(len(df.ind.unique()))

# df = pd.read_csv(ORIG_ADDR, skiprows = [0], usecols = ['ICount'], dtype = {'ICount':'str'})

# df.head()

# df.tail()

# print(len(df.ICount.unique()))


# ### Calculating Reuse Distance

# In[73]:


def create_label(x):
    if(x<=1):
        return 1
    elif(x>1 and x<=2):
        return 2
    elif(x>2 and x<=4):
        return 3
    elif(x>4 and x<=8):
        return 4
    elif(x>8 and x<=16):
        return 5
    elif(x>16 and x<=32):
        return 6
    elif(x>32 and x<=64):
        return 7
    elif(x>64):
        return 8


# In[65]:


sets = [x for x in range(64)]
for cset in sets:
    df = pd.read_csv(ADDR + str(cset)+'.csv')
    m = defaultdict(list)
        # Map : <Data val> : <indexes where it appears>
    for ind, x in enumerate(df.Data.values):
        m[x].append(ind)
    rd_map = {}
    for k in m.keys():
        rds = np.diff(np.asarray(m[k]), append = sys.maxsize)
        rds = np.where(rds>64, 65, rds)
        rd_map.update(zip(m[k], rds))
    df['rd'] = df.index.to_series().map(rd_map)
    df.Data = df.Data.astype('int64')
    df['delta'] = np.diff(df.Data.values, prepend = np.nan)
    df['label'] = df.rd.map(create_label)
    df.to_csv(final_ADDR+str(cset)+'.csv')
    print("------Done processing set->", cset)


# In[68]:





# In[ ]:




