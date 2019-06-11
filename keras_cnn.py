# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:44:48 2019

@author: dell
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split


dataset = pd.read_csv(r"F:\python and machine learning\track data and travel prediction\sample_cnn")
dataset['longitude'] = (dataset['LONGITUDE'] * 100) // 1 / 100
dataset['latitude'] = (dataset['LATITUDE'] * 100) // 1 / 100
dataset['HOUR'] = dataset['START_TIME'] // 100 % 100
dataset.sort_values(['USER_ID', 'P_MONTH'], inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset = dataset.drop_duplicates(['USER_ID', 'longitude', 'latitude', 'P_MONTH', 'HOUR'])
dataset = dataset.loc[:,('USER_ID', 'longitude', 'latitude', 'P_MONTH', 'FLAG')]
dt = dataset.copy()
dt['longitude'] = (dt['longitude'] * 100) // 1 / 100
dt['latitude'] = (dt['latitude'] * 100) // 1 / 100

#将每个用户的轨迹，填充到轨迹网格中，轨迹网格的规模（512*401*1）
j = 0
l = len(dt['USER_ID'].unique())
holidays = ['20180602', '20180603', '20180609', '20180610', '20180616', '20180617', '20180618',
            '20180623', '20180624', '20180630']
result = np.zeros((l, 205312), dtype=np.float16)
for userid in (dt['USER_ID'].unique()):
    dataset_userid = dt[dt['USER_ID'] == userid]
    dataset_userid['P_MONTH'] = dataset_userid['P_MONTH'].astype(np.str)
    dataset_userid = dataset_userid[~dataset_userid['P_MONTH'].isin(holidays)]
    dataset_userid = dataset_userid.loc[:,('longitude', 'latitude')]
    dataset_userid.reset_index(drop=True, inplace=True)
    for i in range(0, len(dataset_userid)):
        row = round((float(dataset_userid.loc[i, 'longitude']) - 118.04) * 100)
        col = round((float(dataset_userid.loc[i, 'latitude']) - 27.17) * 100)
        result[j, (row*401+col)] += 1.0
    #result[j] /= np.max(result[j])
    j += 1
X = result.reshape(l, 512, 401, 1).astype('float16')

dt_flag = dt.loc[:,('USER_ID', 'FLAG')]
dt_flag = dt_flag.groupby('USER_ID').mean()
y = np.asarray(dt_flag['FLAG']).astype('float16')
y = y.reshape(l, 1)
def baseline_model():
    model = Sequential()
    model.add(Convolution2D(16, (3, 3), input_shape=(512, 401, 1), strides=3, activation='relu'))
    model.add(Convolution2D(16, (3, 3), strides=3, activation='relu'))
    model.add(Convolution2D(16, (3, 3), activation='relu'))
    #model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model.fit(X_train, y_train, epochs=20, batch_size=512)
scores = model.evaluate(X_test, y_test)
print(scores)          
