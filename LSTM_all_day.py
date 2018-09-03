#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:04:47 2018

@author: peter
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model

from keras.layers  import Masking
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from MAPE import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

PATH = '/home/peter/Desktop/Diplomovka/'

features = 5

pad = 450

data = pd.read_csv(f'{PATH}'+'call.csv')
data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','std','Last Price']]
data = data.dropna(0)
scaler = MinMaxScaler(feature_range=(0, 1))
date = data[['Date','Instrument Symbol','Expiry Date']]
date = pd.DataFrame(date)
date.reset_index(inplace=True, drop=True)
dataset = data.drop(['Date','Instrument Symbol','Expiry Date'], axis = 1)
dataset = dataset.astype('float32')
dataset.shape
dataset = scaler.fit_transform(dataset)
dataset = pd.DataFrame(dataset)
dataset = pd.concat([date, dataset],axis=1, ignore_index=True)
dataset.columns = ['Date','Instrument Symbol','Expiry Date','Strike Price', 'CTB3','Close','t','std','Last Price']

train_size = int(len(dataset) * 0.90)
valid_size = int(len(dataset) * 0.05)
test_size = len(dataset) - train_size - valid_size
trainX, validX, testX = dataset.iloc[0:train_size,:], dataset.iloc[train_size:train_size+valid_size,:], dataset.iloc[train_size+valid_size:len(dataset),:]
#testX, testY = data_dist_max_pad_2_gen_for_test(dataset, testX)

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(pad, 5)))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

def generator_day(dataset, part, batch_size):
 # Create empty arrays to contain batch of features and labels#
 while True:
    for i in range(batch_size):
      # choose random index in features
      index= part.sample(1, replace=False, weights=None, random_state=None, axis=0)
      dataX, dataY = data_dist_day(index, dataset, part)
      dataX = np.reshape(dataX, (len(dataX),pad,5))
      dataY = np.reshape(dataY,(len(dataY),1))
       
    yield dataX , dataY

def generator_pred_day(dataset, part, batch_size):
 # Create empty arrays to contain batch of features and labels#
 while True:
    for i in range(batch_size):
      # choose random index in features
      index= part.sample(1, replace=False, weights=None, random_state=None, axis=0)
      dataX, dataY = data_dist_day(index, dataset, part)
      dataX = np.reshape(dataX, (len(dataX),pad,5))
      dataY = np.reshape(dataY,(len(dataY),1))
       
    yield dataX

def data_dist_day(row, dataset, part):
    dataX, dataY = [], []
    dataset = pd.DataFrame(dataset)
    part = pd.DataFrame(part)
    dat = dataset[dataset['Date'] == row.iloc[0]['Date']]
    dat = dat.sort_values(['Strike Price'])
    result = np.zeros((pad,5))
    X = np.array(dat.iloc[:1,3:dat.shape[1]-1])
    result[:X.shape[0],:X.shape[1]] = X
    dataX.append(result)
    dataY.append(np.array(row.iloc[0]['Last Price']))
    return dataX, dataY

neurons = [5, 15, 50, 100, 200]
scores = {}
scor = []
for i in neurons:
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(pad, 5)))
    model.add(LSTM(i))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit_generator(generator_day(dataset, trainX, 64), steps_per_epoch=int(trainX.shape[0]/64), verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0, epochs=1)
    yhat = model.predict_generator(generator_pred_day(dataset, testX, 1), steps=testX.shape[0], max_queue_size=10, workers=1, use_multiprocessing=False)
    yhat = pd.DataFrame(yhat)
    result = pd.DataFrame(testX.iloc[:,:-1])
    result = pd.DataFrame(np.zeros((testX.shape[0], 5)))
    inv_yhat = pd.DataFrame(np.concatenate((result, yhat), axis=1))
    inv_yhat = scaler.inverse_transform(inv_yhat)
    pred = pd.DataFrame(inv_yhat[:,-1])
    testY = pd.DataFrame(testX.iloc[:,-1])
    inv_y = np.concatenate((result, testY), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    Y = pd.DataFrame(inv_y[:,-1])
    # calculate metrics
    rmse = np.sqrt(mean_squared_error(Y, pred))
    print('Test RMSE: %.3f' % rmse)
    mae = mean_absolute_error(Y, pred)
    print('Test MAE: %.3f' % mae)
    scor.append([rmse, mae])

import csv

with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(scor)