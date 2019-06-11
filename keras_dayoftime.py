# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 11:24:17 2019

@author: Administrator
"""


import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

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
dataset = dataset.loc[:,('USER_ID', 'longitude', 'latitude', 'P_MONTH', 'HOUR', 'TYPE')]
dataset.sort_values('USER_ID', inplace=True)

dataset['0006'] = 0
dataset['0612'] = dataset['HOUR'].apply(lambda x:1 if((x>=6)&(x<12)) else 0)
dataset['1218'] = dataset['HOUR'].apply(lambda x:2 if((x>=12)&(x<18)) else 0)
dataset['1824'] = dataset['HOUR'].apply(lambda x:3 if((x>=18)&(x<24)) else 0)
dataset['TIME'] = dataset['0006'] + dataset['0612'] + dataset['1218'] + dataset['1824']
dataset.drop(['0006', '0612', '1218', '1824'], axis=1, inplace=True)
user = []
for userid in tqdm(dataset['USER_ID'].unique()):
    dt = dataset[dataset['USER_ID']==userid]
    if len(dt['P_MONTH'].unique()) == 30:
        user.append(userid)
dataset = dataset[dataset['USER_ID'].isin(user)]

l = len(dataset['USER_ID'].unique())
result = np.zeros((l,512,401,4), dtype=np.float16)
i = 0
for userid in tqdm(dataset['USER_ID'].unique()):
    dt_userid = dataset[dataset['USER_ID']==userid]
    dt_userid.reset_index(drop=True, inplace=True)
    for j in range(len(dt_userid)):
        row = round((float(dt_userid.loc[j, 'longitude']) - 118.04) * 100 // 4)
        col = round((float(dt_userid.loc[j, 'latitude']) - 27.17) * 100 // 4)
        time = dt_userid.loc[j, 'TIME']
        if time==0:
            result[i, row, col, 0] = 1.0
        elif time==1:
            result[i, row, col, 1] = 1.0
        elif time==2:
            result[i, row, col, 2] = 1.0
        else:
            result[i, row, col, 3] = 1.0
    i += 1

dt_flag = dataset.loc[:,('USER_ID', 'TYPE')]
dt_flag = dt_flag.drop_duplicates(['USER_ID', 'TYPE'])
y = np.asarray(dt_flag['TYPE']).astype('float16')
y = y.reshape(l, 1)
y = to_categorical(y)

def baseline_model():
    model = Sequential()
    model.add(Convolution2D(16, (3, 3), input_shape=(512, 401, 4), strides=3, activation='relu'))
    model.add(Convolution2D(16, (3, 3), strides=3, activation='relu'))
    model.add(Convolution2D(16, (3, 3), activation='relu'))
    #model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model    
model = baseline_model()
seed = random.randint(0,50)
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(result, y, test_size=0.3, random_state=seed)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
scores = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
matrix = confusion_matrix(np.argmax(y_test,1), np.argmax(y_pred,1))
fig, ax= plt.subplots(figsize=(9,9))
sns.heatmap(matrix, annot=True, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
ax.set_xlabel('y_pred')
ax.set_ylabel('y_true')
plt.show() 
'''
model = baseline_model()
import random
import gc
size = [0.8,0.7,0.6,0.5,0.4,0.3,0.2]
j = 0
score = pd.DataFrame(columns=['test_size', 'accuracy'])
for test_size in size:
    s = 0
    for i in range(2):
        seed = random.randint(0,50)
        np.random.seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(result, y, test_size=test_size, random_state=seed)
        hist = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
        scores = model.evaluate(X_test, y_test)
        s += scores[1]
        del X_train
        del X_test
        gc.collect()
    s /= 2 
    score.loc[j,'test_size'] = test_size
    score.loc[j, 'accuracy'] = s
    j += 1

model = baseline_model()
for i in range(3):
    score = 0
    
    seed = random.randint(0,50)
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(result, y, test_size=0.3, random_state=seed)
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    scores = model.evaluate(X_test, y_test)
    score += scores[1]
score /= 3 
print(score)

y_pred = model.predict(X_test)
matrix = confusion_matrix(np.argmax(y_test,1), np.argmax(y_pred,1))
fig, ax= plt.subplots(figsize=(9,9))
sns.heatmap(matrix, annot=True, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
ax.set_xlabel('y_pred')
ax.set_ylabel('y_true')
plt.show() 
'''