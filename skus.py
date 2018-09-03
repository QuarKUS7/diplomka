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
from sklearn.metrics import r2_score
from MAPE import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
#from sklearn import preprocessing
#from keras.models import Sequential
#from keras.layers import Dense

PATH = '/home/peter/Desktop/Diplomovka/'

data = pd.read_csv(f'{PATH}'+'call.csv')
data = data[data['Instrument Symbol'].str.contains("SXO 181221")]
#data = data[['Strike Price', 'DTB3','Close','std','t','Last Close Price']]
p = data.groupby(['Strike Price']).count()
data = data[['Strike Price', 'DTB3','Close','t','std','Last Close Price']]

features = data.shape[1]-1

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data.values)

train_size = int(len(dataset) * 0.70)
valid_size = int(len(dataset) * 0.20)
test_size = len(dataset) - train_size - valid_size
train, valid, test = dataset[0:train_size,:], dataset[train_size:train_size+valid_size,:], dataset[train_size+valid_size:len(dataset),:]

trainX = np.delete(train, features, 1)
trainY = train[:,features]
validX = np.delete(valid, features, 1)
validY = valid[:,features]
testX = np.delete(test, features, 1)
testY = test[:,features]

model = Sequential()
model.add(Dense(12, input_dim=features, activation="relu", kernel_initializer="uniform"))
model.add(Dense(50, activation="relu", kernel_initializer="uniform"))
model.add(Dense(100, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(trainX, trainY, epochs=50, batch_size=1, validation_data=(validX, validY), verbose=2, shuffle=True)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='valid')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(testX)
#testX = testX.reshape((testX.shape[0], testX.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((testX, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
pred = inv_yhat[:,features]
# invert scaling for actual
#testY = testY.reshape((len(testY), 1))

testY = pd.DataFrame(testY)
inv_y = np.concatenate((testX, testY), axis=1)
inv_y = scaler.inverse_transform(inv_y)
Y = inv_y[:,features]

# calculate metrics
rmse = np.sqrt(mean_squared_error(Y, pred))
print('Test RMSE: %.3f' % rmse)
R = r2_score(Y, pred)
print('Test R^2: %.3f' % R)
mape = mean_absolute_percentage_error(Y, pred)
print('Test MAPE: %.3f' % mape)
mse = mean_squared_error(Y, pred)
print('Test MSE: %.3f' % mse)
mae = mean_absolute_error(Y, pred)
print('Test MAE: %.3f' % mae)


