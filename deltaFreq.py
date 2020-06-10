#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import defaultdict 
import sys
import os
import matplotlib.pyplot as plt
import _thread as thread
from matplotlib.pyplot import figure
from tabulate import tabulate
import threading
import time


# In[14]:


addr_l = [510
          , 511, 526, 600, 602, 620, 623, 625, 631, 641, 648, 657
         ] 


sets = [x for x in range(64)]
# ADDR = '/nfs_home/nbhardwaj/data/rds_data/SPEC2017/'
ADDR = '/nfs_home/nbhardwaj/data/rds_final/'
w_ADDR = '/nfs_home/nbhardwaj/results/'


# In[34]:


# instr maps
# m = defaultdict(list)
# fm = defaultdict(list)

# #data maps
# m2 = defaultdict(list)
# fm2 = defaultdict(list)
m3 = {}
# s_inst = defaultdict(set) # unique instr across sets 0-63
# u_inst = defaultdict(list) # unique instr in a file and set
# f_inst = defaultdict(list) # unique instr in a file
for fname in addr_l:
    begin = time.time()
    for cset in sets:
        cADDR = ADDR+str(fname)+'_'+str(cset)+'.csv'
        df = pd.read_csv(cADDR,usecols = ['delta'])
        udelta, counts = np.unique(df.delta.values, return_counts = True)
        for idx in range(len(udelta)):
            val = udelta[idx]
            cnt = counts[idx]
            if(m3.get(val) != None):
                m3[val][fname]+=cnt
            else:
                m3[val] = {}
                m3[val].update(zip(addr_l, np.zeros((len(addr_l)))))
                m3[val][fname] = cnt
    print("finished-->", fname, "|| time passed->",time.time()-begin, "seconds")
np.save(w_ADDR+'m_delta.npy', m3)
# np.save(w_ADDR+'m2.npy', m2)
# np.save(RES_ADDR+'fm.npy', fm)
# np.save(RES_ADDR+'fm2.npy', fm2)
print("XX || Its DONE || XX")


# In[16]:


# m = np.load(w_ADDR+'m.npy', allow_pickle = True).item()
# m2 = np.load(w_ADDR+'m2.npy', allow_pickle = True).item()
# fm = np.load(w_ADDR+'fm.npy', allow_pickle = True).item()
# fm2 = np.load(w_ADDR+'fm2.npy', allow_pickle = True).item()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[101]:


# mdelta = defaultdict(list) # global unique delta
# s_delta = defaultdict(set) # unique delta across sets 0-63
# fs_delta = defaultdict(list) # unique delta in a file and set
# f_delta = defaultdict(list) # unique delta in a file
# freq_delta = {}
# for fname in addr_l:
#     p = set()
#     start_time = time.time()
#     for cset in sets:
#         cADDR = ADDR+str(fname)+'_'+str(cset)+'.csv'
#         df = pd.read_csv(cADDR, usecols = ['delta'])
#         delta_unique = df.delta.unique()
#         fs_delta[str(fname)+'_'+str(cset)] = len(delta_unique)
#         for d in delta_unique:
#             s_delta[cset].add(d)
#             p.add(d)
#             mdelta[d].append(str(fname)+'_'+str(cset)) 
#         for d in df.delta:
#             if(d in freq_delta.keys()):
#                 freq_delta[d]+=1
#             else:
#                 freq_delta[d] = 1
# #         print("done ->", cset)
#     f_delta[fname] = len(p)
#     end_time = time.time()
#     print("finished-->", fname, "||",end_time - start_time, "secs")


# In[110]:


# a = sorted(freq_delta.items(), key = lambda x:x[1], reverse = True)


# In[140]:


# plt.plot([x for x in range(len(b))], np.log10(b))
# plt.xlabel('# unique addresses')
# plt.ylabel('log10 frequency')
# plt.show()
# plt.savefig('graphs/freqVSAddr')


# In[151]:


# fname = '510'
# cset = '10'
# cADDR = ADDR+str(fname)+'_'+str(cset)+'.csv'
# df = pd.read_csv(cADDR)

# df['log_Data'] = np.log10(df.Data)


# ld = df.log_Data
# figure(figsize = (15, 1))
# plt.plot([x for x in range(len(ld))], ld)


# In[ ]:




