#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
from collections import defaultdict 
import sys
import os
import matplotlib.pyplot as plt
import _thread as thread
from tabulate import tabulate
import threading
import time


# In[54]:


# ADDR = './401part.csv'

# file name below also in os delete command
ADDR_LIST = [
#           '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/510.parest_r/510.parest_r.pinatrace.csv',
 #           '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/511.povray_r/511.povray_r.pinatrace.csv',
  #           '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/526.blender_r/526.blender_r.pinatrace.csv'
       #      '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/600.perlbench_s/600.perlbench_s.pinatrace.csv',
        #     '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/602.gcc_s/602.gcc_s.pinatrace.csv',
         #  '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/605.mcf_s/605.mcf_s.pinatrace.csv',
         #  '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/620.omnetpp_s/620.omnetpp_s.pinatrace.csv'
            '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/623.xalancbmk_s/623.xalancbmk_s.pinatrace.csv',
            '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/625.x264_s/625.x264_s.pinatrace.csv',
            '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/631.deepsjeng_s/631.deepsjeng_s.pinatrace.csv',
    #        '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/641.leela_s/641.leela_s.pinatrace.csv',
   #         '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/648.exchange2_s/648.exchange2_s.pinatrace.csv',
  #          '/nfs_home/matkade/pinplay/Pintool/SPEC_2017_ST_First_run/657.xz_s/657.xz_s.pinatrace.csv'
            ]
# ADDR = '/nfs_home/SPEC_2006/400.perlbench/400.perlbench.pinatrace.csv'
final_ADDR = '/nfs_home/nbhardwaj/data/SPEC_final/'
addr_l = [
   # 510, 511, 526
     #      600, 602, 605,620 
    623, 625, 631
    # , 641, 648, 657
         ]
# df = pd.read_csv(ADDR, skiprows = [0], nrows = 10000)


# In[55]:


# verify addresses 
for ADDR in ADDR_LIST:
    if not os.path.isfile(ADDR):
        print("xxxx|| file not found ||"+str(ADDR))
for ind, ADDR in enumerate(ADDR_LIST):
    print(addr_l[ind], ADDR)
print("file location verified")


# In[52]:


def file_split(iaddr, ADDR, iters):
    print("---------processing list----->", ADDR)
    for df in pd.read_csv(ADDR, skiprows = [0], usecols = ['ICount', 'Instruction', 'Mode', 'Data'],chunksize = 50000000):
        iters += 1
        print("--iters--------->", iters, ADDR)

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
        df['set'] = df.hex_Data.apply(lambda x: int((bin(int(x, 16))[2:].zfill(64))[-12:-6], 2))
        # df.reset_index(inplace = True, level = 'orig_idx')

        # Least significant 7 to 12 bits represent the set
        # df['set'] = df.bin_Data.apply(lambda x: int(x[-12:-6], 2))
        print("done preprocessing")

        # Grouping values with the same set no
        df.sort_values(by = 'set', inplace = True, kind = 'mergesort')
        print("done sorting")

        for curr_set in df['set'].unique():
            faddr = final_ADDR+str(addr_l[iaddr])+'_'+str(curr_set)+'.csv'
            print("Processing for set->", curr_set, ADDR)
            df_sub = df[df.set.eq(curr_set)][['ICount','Instruction','Data', 'Mode','set']]
    
            # to allow chunk processing
            if not os.path.isfile(faddr):
                df_sub.to_csv(faddr, index = False)
            else:
                df_sub.to_csv(faddr, mode = 'a', header = False, index = False)


# In[48]:


threads = []
iters = 0
for iaddr, ADDR in enumerate(ADDR_LIST):
    t = threading.Thread(target = file_split, args = (iaddr, ADDR, iters))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
    
print("XXXXXXXXXXXX || DONE FILE SPLIT || XXXXXXXXXXXXXXXXXX")
print(ADDR_LIST)


# In[39]:





# In[ ]:




