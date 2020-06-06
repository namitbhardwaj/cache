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
import time
import keras 
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM, Flatten, Lambda
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from keras.utils.vis_utils import plot_model


# In[5]:


ADDR = '/nfs_home/nbhardwaj/data/rds_final/'
w_ADDR = '/nfs_home/nbhardwaj/model_weights/finalwts/'


# In[6]:


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
    


# In[ ]:


def create_model(embed_size = 10):
    inp1 = Input(shape = (1,))
    inp2 = Input(shape = (1,))
    inp3 = Input(shape = (1,))

    embed1 = Embedding(len(le_inst.classes_), embed_size, input_length = 1)(inp1)
    embed2 = Embedding(len(le_delta.classes_), embed_size, input_length = 1)(inp2)

    merged_inp = keras.layers.concatenate([embed1, embed2], axis = 1)
    merged_inp = Flatten()(merged_inp)
    merged_inp = keras.layers.concatenate([merged_inp, inp3])
    
#     out = LSTM(64)(merged_inp)
    out = Dense(32, activation = 'relu')(merged_inp)
    out = Dense(8, activation = 'softmax')(out)

    model = Model([inp1, inp2, inp3], out)
    return model


# In[ ]:


# sets = [x for x in range(18, 64)]
sets = [51]
inst_vocab = []
delta_vocab = []
train_acc = []
test_acc = []
lens = []
# print("enter file name -_--------------------------->")
# fname = input()
# fname = str(fname)
fname = '648'
#make dir for saving model weights
w_ADDR = w_ADDR + str(fname)
if(not os.path.isdir(w_ADDR)):
    if(os.system('mkdir '+ w_ADDR) != 0):
        print("error creating dir "+fname)
        exit()
for cset in sets:
    start = time.time()
    df = pd.read_csv(ADDR+fname+'_'+str(cset)+'.csv', index_col = [0], usecols = [0, 2, 4, 7, 8], nrows = 100000000)
    df.Mode = np.where(df.Mode.values=='R', 1, -1)
    df.Mode = df['Mode'].astype('str')
    df.Instruction = df.Instruction.astype('str')
    df.delta = df.delta.astype('float')
    
    X = df[['Instruction', 'delta', 'Mode']].values[1:]
    y = df[['label']].values[1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)
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
    model = create_model()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 2)
    filepath = w_ADDR+'/'+str(cset)+'.hdf5'
    mc = ModelCheckpoint(filepath, monitor = 'val_accuracy', save_best_only = True, mode = 'max')
    history = model.fit([X_train[:, 0], X_train[:, 1], X_train[:, 2]], to_categorical(y_train), epochs = 50, 
              validation_split = 0.2, use_multiprocessing = True, verbose = 2, callbacks = [es, mc])
    print("------------training done------------")
    t_ac = model.evaluate([X_test[:, 0], X_test[:, 1], X_test[:, 2]], to_categorical(y_test))[1]
    test_acc.append(t_ac)
    tr_ac = model.evaluate([X_train[:, 0], X_train[:, 1], X_train[:, 2]], to_categorical(y_train))[1]
    train_acc.append(tr_ac)
    inst_vocab.append(len(le_inst.classes_))
    delta_vocab.append(len(le_delta.classes_))
    end = time.time()
    print("--------------done processing for set---------->", cset, "|| time->", end-start, "s")
    print("train acc-->", tr_ac)
    print("test acc--->", t_ac)
df2 = pd.DataFrame(list(zip(train_acc, test_acc, lens)), columns = ['train_accuracy', 'test_accuracy','length'])
df2.to_csv(w_ADDR+'/acc.csv')


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

