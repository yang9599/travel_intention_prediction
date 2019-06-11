# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 00:44:37 2019

@author: dell
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten,TimeDistributed
from keras.layers.convolutional import MaxPooling2D
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os


def get_session(gpu_fraction):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """
 
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
 
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(get_session(0.8))

dataset = pd.read_csv(r"G:\track data and travel prediction\dataset_all_flag")
user = pd.read_csv(r"G:\track data and travel prediction\dataset\DataTech_Travel_Train_User", 
                   names=['USER_ID', 'FLAG', 'TYPE'], sep='|')
dataset['longitude'] = (dataset['LONGITUDE'] * 100) // 1 / 100
dataset['latitude'] = (dataset['LATITUDE'] * 100) // 1 / 100
dataset['HOUR'] = dataset['START_TIME'] // 100 % 100
dataset.sort_values(['USER_ID', 'P_MONTH'], inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset = dataset.drop_duplicates(['USER_ID', 'longitude', 'latitude', 'P_MONTH', 'HOUR'])
dataset = pd.merge(dataset, user, on='USER_ID')
dataset = dataset.loc[:,('USER_ID', 'longitude', 'latitude', 'P_MONTH', 'TYPE')]

holidays = ['20180602', '20180603', '20180609', '20180610', '20180616', '20180617', '20180618',
            '20180623', '20180624', '20180630']
dataset['P_MONTH'] = dataset['P_MONTH'].astype(np.str)
dataset = dataset[~dataset['P_MONTH'].isin(holidays)]

user = []
for userid in tqdm(dataset['USER_ID'].unique()):
    dt = dataset[dataset['USER_ID']==userid]
    if len(dt['P_MONTH'].unique()) == 20:
        user.append(userid)
dataset = dataset[dataset['USER_ID'].isin(user)]

dt = dataset.copy()
dt['longitude'] = (dt['longitude'] * 100) // 1 / 100
dt['latitude'] = (dt['latitude'] * 100) // 1 / 100

j = 0
l = len(dt['USER_ID'].unique())
X = np.zeros((l, 20, 128, 101), dtype=np.int)
for userid in tqdm(dt['USER_ID'].unique()):
    dt_date = dt[dt['USER_ID']==userid]
    t = 0
    for date in (dt['P_MONTH'].unique()):
        dt_month = dt_date[dt_date['P_MONTH']==date]
        dt_month.reset_index(drop=True, inplace=True)
        for i in range(len(dt_month)):
            row = round((float(dt_month.loc[i, 'longitude']) - 118.04) * 100 // 4)
            col = round((float(dt_month.loc[i, 'latitude']) - 27.17) * 100 // 4)
            X[j, t, row, col] = 1
        t += 1
    j += 1
X = X.reshape(l, 20, 128, 101, 1)
dt_flag = dt.loc[:,('USER_ID','TYPE')]
dt_flag = dt_flag.groupby(['USER_ID']).mean()
y = np.asarray(dt_flag['TYPE']).astype('float16')
y = y.reshape(-1, 1)
y = to_categorical(y)
shape = y.shape[1]
y = y.reshape(l, 1, shape)
#定义ConvLSTM结构
'''
def model_structure1():
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), 
                         input_shape=(None, X_train.shape[2], X_train.shape[3], X_train.shape[4]),
                         padding='same', activation='relu', return_sequences=True))
    model.add(BatchNormalization())    
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                       padding='same', activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Reshape((-1,64)))    
    model.add(Dropout(0.2))
    model.add(Dense(shape, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return(model)
def model_structure2():
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3,3), input_shape=(None, 512,401,1), 
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3,3),padding='same',return_sequences=True))
    model.add(BatchNormalization())
    model.add(AveragePooling3D((1, 512, 401)))
    model.add(Reshape((512,401,40)))
    model.add(Dense(units=4,activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return(model)
'''    
def model_structure3():
    model = Sequential()
    model.add(ConvLSTM2D(filters=16,kernel_size=(3,3),input_shape=(None,128,101,1),
                         return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), return_sequences=True))
    model.add(BatchNormalization()) 
    model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(shape, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return(model)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)
model = model_structure3()
model.fit(X_train, y_train, epochs=20, batch_size=8)
scores = model.evaluate(X_test, y_test)
print(scores) 



















       