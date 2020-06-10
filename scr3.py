#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from collections import defaultdict 
import sys
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import time
import keras 
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM, Flatten, Lambda
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from keras.utils.vis_utils import plot_model


# In[7]:


ADDR = '/nfs_home/nbhardwaj/data/rds_final/'
w_ADDR = '/nfs_home/nbhardwaj/model_weights/finalwts/'


# In[8]:


from sklearn.preprocessing import LabelEncoder

class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        m = {}
        for x in self.label_encoder.classes_:
            m[x] = True
        for ind, y in enumerate(new_data_list):
            if(m.get(y) is None):
                new_data_list[ind] = 'Unknown'
#         for unique_item in np.unique(data_list):
#             if unique_item not in self.label_encoder.classes_:
#                 new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]
        return self.label_encoder.transform(new_data_list)
    


# In[9]:


def create_model(embed_size = 10):
    inp1 = Input(shape = (1,))
    inp2 = Input(shape = (1,))
    # inp3 = Input(shape = (1,))

    embed1 = Embedding(len(le_inst.classes_), embed_size, input_length = 1)(inp1)
    embed2 = Embedding(len(le_delta.classes_), embed_size, input_length = 1)(inp2)

    merged_inp = keras.layers.concatenate([embed1, embed2], axis = 1)
    merged_inp = Flatten()(merged_inp)
    # merged_inp = keras.layers.concatenate([merged_inp, inp3])
    
#     out = LSTM(64)(merged_inp)
    out = Dense(32, activation = 'relu')(merged_inp)
    out = Dense(8, activation = 'softmax')(out)

    model = Model([inp1, inp2], out)
    return model


# In[5]:


files = [
#510, 511, 526, 600
#, 602, 620, 623, 625, 
#631, 
641, 648, 657
         ]
sets = [x for x in range(64)]
df_m = pd.DataFrame(columns = ['fname', 'set', 'train_acc', 'test_acc', 'len', 'inst_v', 'delt_v'])
for fname in files:
    cw_ADDR = w_ADDR+str(fname)+'/'
    for cset in sets:
        if(not os.path.isdir(cw_ADDR)):
            os.system("mkdir "+cw_ADDR)
            
        df = pd.read_csv(ADDR+str(fname)+'_'+str(cset)+'.csv', index_col = [0], usecols = [0,2,7,8])
        df.Instruction = df.Instruction.astype('str')
        df.delta = df.delta.astype('float')

        X = df[['Instruction', 'delta']].values[1:]
        y = df[['label']].values[1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
        print("--------------------split done---------------------")
        le_inst = LabelEncoderExt()
        le_inst.fit(X_train[:, 0])
        le_delta = LabelEncoderExt()
        le_delta.fit(X_train[:, 1])
        print("----------------labels done----------------------")
        X_train[:, 0] = le_inst.transform(X_train[:, 0])
        X_train[:, 1] = le_delta.transform(X_train[:, 1])
        print("--------")

        X_test[:, 0] = le_inst.transform(X_test[:, 0])
        X_test[:, 1] = le_delta.transform(X_test[:, 1])
        print("-------------------labels transformed---------------------")
        filepath = cw_ADDR+str(fname)+'_'+str(cset)+'.hdf5'

        if(os.path.isfile(filepath)):
            model = load_model(filepath)
            print("using loaded model-->", filepath)
        else:
            model = create_model()
            model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
     
        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 2)
        mc = ModelCheckpoint(filepath, monitor = 'val_accuracy', save_best_only = True, mode = 'max')
        history = model.fit([X_train[:, 0], X_train[:, 1]], to_categorical(y_train), epochs = 3, 
                  validation_split = 0.2, use_multiprocessing = True, verbose = 2, callbacks = [es, mc], batch_size = 8192)

        print("------------training done------------")
    #     model.save_weights(filepath)
        t_ac = model.evaluate([X_test[:, 0], X_test[:, 1]], to_categorical(y_test), verbose = 2)[1]
        tr_ac = model.evaluate([X_train[:, 0], X_train[:, 1]], to_categorical(y_train), verbose = 2)[1]
        
        # need to reduce len to test_size later
        df_m.loc[len(df_m)] = [fname, cset, tr_ac, t_ac, len(df), len(le_inst.classes_), len(le_delta.classes_)]

        print("--------------done processing for set---------->", cset)
        print( '|| accuracy||', tr_ac, t_ac)
        print("inst vocal", len(le_inst.classes_), "|| delta vocal->", len(le_delta.classes_))
        print("____________________________________________________________________________________________________")
    df_m.to_csv(cw_ADDR+'results.csv')
    print("$$$$$$$ done for file ||------------>", fname)
df_m.to_csv(w_ADDR+'results.csv')
print("---------ALL DONE___________")


# In[34]:


# history.history.keys()


# In[35]:


# plt.plot(history.history['loss'], label = 'train')
# plt.plot(history.history['val_loss'] , label = 'validation')
# plt.legend()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()


# In[ ]:


# how to load the stored model

# from keras.models import load_model
# saved_model = load_model(filepath)
# tr_ac = model.evaluate([X_train[:, 0], X_train[:, 1], X_train[:, 2]], to_categorical(y_train))[1]
# print("train acc--->", tr_ac)

