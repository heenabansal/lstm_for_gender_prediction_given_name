#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:16:13 2017

@author: fractaluser
"""

from __future__ import print_function

from sklearn.preprocessing import OneHotEncoder
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import pandas as pd
import numpy as np
import os
os.chdir("/home/fractaluser/Backup/Courses/Tensorflow/lstm")

#parameters
maxlen = 30
labels = 2

input = pd.read_csv("gender_data.csv",header=0,usecols=(0,1))
input.columns = ['name','m_or_f']
input['namelen']= [len(str(i)) for i in input['name']]
input1 = input[(input['namelen'] >= 2) ]

input1.groupby('m_or_f').count()

names = input['name']
gender = input['m_or_f']
vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)

print(vocab)
print("vocab length is ",len_vocab)
print ("length of input is ",len(input1))

char_index = dict((c, i) for i, c in enumerate(vocab))

#train test split
msk = np.random.rand(len(input1)) < 0.8
train = input1[msk]
test = input1[~msk]

#take input upto max and truncate rest
#encode to vector space(one hot encoding)
#padd 'END' to shorter sequences
train_X = []
trunc_train_name = [str(i)[0:30] for i in train.name]
for i in trunc_train_name:
    tmp = [char_index[j] for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(char_index["END"])
    train_X.append(tmp)

np.asarray(train_X).shape

def set_flag(i):
    tmp = np.zeros(92);
    tmp[i] = 1
    return(tmp)
    
set_flag(3)    

#take input upto max and truncate rest
#encode to vector space(one hot encoding)
#padd 'END' to shorter sequences
#also convert each index to one-hot encoding
train_X = []
train_Y = []
trunc_train_name = [str(i)[0:maxlen] for i in train.name]
for i in trunc_train_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    train_X.append(tmp)
for i in train.m_or_f:
    if i == 'm':
        train_Y.append([1,0])
    else:
        train_Y.append([0,1])

np.asarray(train_X).shape

np.asarray(train_Y).shape








    
    
    
    
    
    
    
    
    
    
    





































