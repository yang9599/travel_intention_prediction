# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:59:22 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, concatenate, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.utils.np_utils import to_categorical
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
import gc

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

dt = dataset.copy()
dt['longitude'] = (dt['longitude'] * 100) // 1 / 100
dt['latitude'] = (dt['latitude'] * 100) // 1 / 100

j = 0
l = len(dt['USER_ID'].unique())
holidays = ['20180602', '20180603', '20180609', '20180610', '20180616', '20180617', '20180618',
            '20180623', '20180624', '20180630']
result = np.zeros((l, 205312), dtype=np.float16)
for userid in tqdm(dt['USER_ID'].unique()):
    dataset_userid = dt[dt['USER_ID'] == userid]
    dataset_userid['P_MONTH'] = dataset_userid['P_MONTH'].astype(np.str)
    dataset_userid = dataset_userid[~dataset_userid['P_MONTH'].isin(holidays)]
    dataset_userid = dataset_userid.loc[:,('longitude', 'latitude')]
    dataset_userid.reset_index(drop=True, inplace=True)
    for i in range(0, len(dataset_userid)):
        row = round((float(dataset_userid.loc[i, 'longitude']) - 118.04) * 100)
        col = round((float(dataset_userid.loc[i, 'latitude']) - 27.17) * 100)
        result[j, (row*401+col)] = 1.0
    #result[j] /= np.max(result[j])
    j += 1
#X = result.reshape(l, 512, 401, 1).astype('float16')
dt_flag = dt.loc[:,('USER_ID', 'TYPE')]
dt_flag = dt_flag.groupby('USER_ID').mean()
y = np.asarray(dt_flag['TYPE']).astype('float16')
y = y.reshape(l, 1)
y = to_categorical(y)

main_input = Input((512,401,1), dtype='float32', name='main_input')
x1 = Convolution2D(16, (3, 3), input_shape=(512, 401, 1), strides=3, activation='relu')(main_input)
conv_out1 = Convolution2D(16, (3, 3), strides=3, activation='relu')(x1)
conv_out2 = Convolution2D(16, (3, 3), strides=3, activation='relu')(conv_out1)
max_out = MaxPooling2D(pool_size=(2, 2))(conv_out2)
drop_out = Dropout(0.2)(max_out)
flat_out = Flatten()(drop_out)  
aux_input = Input((6,), name='aux_input')
#x2 = concatenate([flat_out, aux_input])
x3 = Dense(512, activation='relu')(flat_out)
x3 = Dropout(0.5)(x3) 
x4 = Dense(128, activation='relu')(x3)
x4 = Dropout(0.5)(x4)
x5 = Dense(32, activation='relu')(x4)
x6 = concatenate([x5, aux_input])
main_output = Dense(4, activation='softmax', name='main_output')(x6)

feature = pd.read_csv(r"G:\track data and travel prediction\feature_all.csv")
feature.sort_values(['USER_ID'], inplace=True)
feature = feature.loc[:,('num_of_points', 'covering', 'turning_radius', 'weekdays_weekends', 'entropy', 'similarity')]
result = pd.DataFrame(result)
feature = result.join(feature)
feature = feature.as_matrix()
s = []
for i in range(5):
    #seed = random.randint(0,50)
    #np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.3, random_state=5)
    X_train_main = X_train[:,:205312]
    X_train_main = X_train_main.reshape(X_train.shape[0],512,401,1)
    X_train_feature = X_train[:,205312:]
    X_test_main = X_test[:,:205312]
    X_test_main = X_test_main.reshape(X_test.shape[0],512,401,1)
    X_test_feature = X_test[:,205312:]
    model = Model(inputs=[main_input, aux_input], outputs=[main_output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x={'main_input': X_train_main, 'aux_input':X_train_feature },
                y={'main_output': y_train}, batch_size=32, epochs=20,verbose=1, validation_split=0.1)
    scores = model.evaluate(x={'main_input': X_test_main, 'aux_input': X_test_feature},
                y={'main_output': y_test}, batch_size=10, verbose=1)
    s.append(scores[1])
    del X_train
    del X_test
    gc.collect()
print(s)

y_pred = model.predict(x={'main_input': X_test_main, 'aux_input':X_test_feature })
matrix = confusion_matrix(np.argmax(y_test,1), np.argmax(y_pred,1))
fig, ax= plt.subplots(figsize=(9,9))
sns.heatmap(matrix, annot=True, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
ax.set_xlabel('y_pred')
ax.set_ylabel('y_true')
plt.show() 