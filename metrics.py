#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[18]:


addr_l = [510
          , 511, 526, 600, 602, 620, 623, 625, 631, 641, 648, 657
         ]
sets = [x for x in range(64)]
# ADDR = '/nfs_home/nbhardwaj/data/rds_data/SPEC2017/'
ADDR = '/nfs_home/nbhardwaj/data/rds_final/'
w_ADDR = '/nfs_home/nbhardwaj/results/'


# In[ ]:


# instr maps
m = defaultdict(list)
fm = defaultdict(list)

#data maps
m2 = defaultdict(list)
fm2 = defaultdict(list)
# s_inst = defaultdict(set) # unique instr across sets 0-63
# u_inst = defaultdict(list) # unique instr in a file and set
# f_inst = defaultdict(list) # unique instr in a file
begin = time.time()
for fname in addr_l:
    for cset in sets:
        cADDR = ADDR+str(fname)+'_'+str(cset)+'.csv'
        df = pd.read_csv(cADDR, usecols = ['Instruction', 'Data'])
        uinst = df.Instruction.unique()
        udata = df.Data.unique()
        for i in uinst:
            m[i].append(str(fname)+'_'+str(cset))
        for d in udata:
            m2[d].append(str(fname)+'_'+str(cset))
        for x, y in df.values:
            if(x in fm.keys()):
                fm[x] += 1
            else:
                fm[x] = 1
            if(y in fm2.keys()):
                fm2[y] += 1
            else:
                fm2[y] = 1
    print("finished-->", fname, "|| time passed->",time.time()-begin, "seconds")
np.save(w_ADDR+'m.npy', m)
np.save(w_ADDR+'m2.npy', m2)
np.save(RES_ADDR+'fm.npy', fm)
np.save(RES_ADDR+'fm2.npy', fm2)
print("XX || Its DONE || XX")


# In[ ]:





# In[25]:


# a = sorted(fm2.items(), key = lambda x:x[1])


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




