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


# In[ ]:


# ADDR = './401part.csv'

# file name below also in os delete command

ADDR = '/nfs_home/SPEC_2006/400.perlbench/400.perlbench.pinatrace.csv'
final_ADDR = '/nfs_home/nbhardwaj/data/set_data/400'

# df = pd.read_csv(ADDR, skiprows = [0], nrows = 10000)


# In[ ]:





# In[ ]:


iter = 0
for df in pd.read_csv(ADDR, skiprows = [0], chunksize = 100000000):
    iter += 1
    print("---------iters-------------->", iter)
    
    df.rename(columns = {'Data':'hex_Data', 'Instruction':'hex_Instruction'}, inplace = True) 

#     df.drop('ICount', axis = 1, inplace = True)

    print("No of lines before->",len(df))
    df = df.dropna()
    print("No of lines after dropping nan",len(df))

    #Pre-processing
    # HEX to INT
    df['Instruction'] = df.hex_Instruction.apply(lambda x:int(x,16))


    # Change to line No resolution, remove the offset data
    df['Data'] = df.hex_Data.apply(lambda x:int(x, 16)//64)
    print("done hex to int")
    # df['new_Data'] = df.Data.apply(lambda x: x//64)


    # HEX to Binary with pre-padding till 64 spaces
    df['set'] = df.hex_Instruction.apply(lambda x: int((bin(int(x, 16))[2:].zfill(64))[-12:-6], 2))
    # df.reset_index(inplace = True, level = 'orig_idx')

    # Least significant 7 to 12 bits represent the set
    # df['set'] = df.bin_Data.apply(lambda x: int(x[-12:-6], 2))
    print("done preprocessing")

    # Grouping values with the same set no
    df.sort_values(by = 'set', inplace = True, kind = 'mergesort')
    print("done sorting")


    for curr_set in df['set'].unique():
        print("Processing for set->", curr_set)
        df_sub = df[df.set.eq(curr_set)][['ICount','Instruction','Data', 'Mode','set']]

        # to allow chunk processing
        if not os.path.isfile(final_ADDR+'_'+str(curr_set)+'.csv'):
            df_sub.to_csv(final_ADDR+'_'+str(curr_set)+'.csv', index = False)
        else:
            df_sub.to_csv(final_ADDR+'_'+str(curr_set)+'.csv', mode = 'a', header = False, index = False)
    #     break
#     break


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




