#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:58:41 2018

@author: peter
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:04:18 2018

@author: peter
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 19:52:26 2018

@author: peter
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from sklearn import preprocessing
#from keras.models import Sequential
#from keras.layers import Dense

PATH = '/home/peter/Desktop/Diplomovka/'

data = pd.read_csv(f'{PATH}'+'call.csv')
#data = data[data['Instrument Symbol'].str.contains("SXO 151120")]
#data = data.sort_values(['Date','Strike Price'])
data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','std','Last Price']]
data = data.dropna(0)
#data = data.iloc[1:10000,:]
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = MinMaxScaler((0,1))

dataset = data.drop(['Date','Instrument Symbol','Expiry Date'], axis = 1)
#dataset = scaler.fit_transform(dataset)
#dataset = data.drop(['Date','Instrument Symbol','Expiry Date'], axis = 1)
dataset = dataset.astype('float32')
dataset.shape
dataset = scaler.fit_transform(dataset)
dataset = pd.DataFrame(dataset)

dataset.columns = ['Strike Price', 'CTB3','Close','t','std','Last Price']

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

train_size = int(len(dataset) * 0.95)
test_size = int(len(dataset) * 0.05)
trainX, testX = dataset.iloc[0:train_size,:-1].values, dataset.iloc[train_size:,:-1].values
trainY, testY = dataset.iloc[0:train_size,-1].values, dataset.iloc[train_size:,-1].values

neurons = [5, 15, 50, 100, 150, 300]
scores = {}
for i in neurons:
    scor_mae = []
    scor_rmse = []
    for train_indices, test_indices in kf.split(trainX):
        model = Sequential()
        model.add(Dense(i, input_dim=trainX.shape[1], activation="relu", kernel_initializer="uniform"))
        model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX[train_indices], trainY[train_indices], epochs=10, batch_size=32, verbose=1, shuffle=True)
        yhat = model.predict(trainX[test_indices])
        inv_yhat = np.concatenate((trainX[test_indices], yhat), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        pred = inv_yhat[:,-1]
        testY = pd.DataFrame(trainY[test_indices])
        inv_y = np.concatenate((trainX[test_indices], testY), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        Y = inv_y[:,-1]
        rmse = np.sqrt(mean_squared_error(Y, pred))
        print('Test RMSE: %.3f' % rmse)
        mae = mean_absolute_error(Y, pred)
        print('Test MAE: %.3f' % mae)
        scor_mae.append(mae)
        scor_rmse.append(rmse)
    scores[f'{i}'+':MAE'] = sum(scor_mae) / float(len(scor_mae))
    scores[f'{i}'+':RMSE'] = sum(scor_rmse) / float(len(scor_rmse))


model = Sequential()
model.add(Dense(50, input_dim=trainX.shape[1], activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=1, shuffle=True)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='valid')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[1]))
# invert scaling for forecast
inv_yhat = np.concatenate((testX, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
pred = inv_yhat[:,testX.shape[1]]
# invert scaling for actual
#testY = testY.reshape((len(testY), 1))

testY = pd.DataFrame(testY)
inv_y = np.concatenate((testX, testY), axis=1)
inv_y = scaler.inverse_transform(inv_y)
Y = inv_y[:,testX.shape[1]]

# calculate metrics
rmse = np.sqrt(mean_squared_error(Y, pred))
print('Test RMSE: %.3f' % rmse)
mae = mean_absolute_error(Y, pred)
print('Test MAE: %.3f' % mae)
