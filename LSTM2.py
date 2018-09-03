#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 23:17:56 2018

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
import math
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

data['Instrument Symbol'] = data['Instrument Symbol'].astype('str')

p = data.groupby(['Instrument Symbol']).agg(['count'])
p = p.sort_values(p.columns[0], ascending=False)
indexy = p.index.values
indexy = indexy[0:1]
look_back = 40
vys = []
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

   
for dex in indexy:
    print(dex)
    sample = data[data['Instrument Symbol'] == dex]
    #sample = sample[['Strike Price', 'DTB3','Close','t','std','Last Close Price']]
    sample = sample[['Last Close Price']]
    features = sample.shape[1]-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(sample)

    train_size = int(len(dataset) * 0.70)
    valid_size = int(len(dataset) * 0.20)
    test_size = len(dataset) - train_size - valid_size
    train, valid, test = dataset[0:train_size,:], dataset[train_size:train_size+valid_size,:], dataset[train_size+valid_size:len(dataset),:]

    trainX, trainY = create_dataset(train, look_back)
    validX, validY = create_dataset(valid, look_back)
    testX, testY = create_dataset(test, look_back)

#    trainX = np.delete(train, features, 1)
#    trainY = train[:, features]
#    validX = np.delete(valid, features, 1)
#    validY = valid[:, features]
#    testX = np.delete(test, features, 1)
#    testY = test[:, features]

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    validX = np.reshape(validX, (validX.shape[0],  validX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(trainX, trainY, epochs=100, batch_size=1, validation_data=(validX, validY), verbose=2, shuffle=False)
# plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='valid')
    pyplot.legend()
    pyplot.show()

# make a prediction
    yhat = model.predict(testX)
#    testX = testX.reshape((testX.shape[0], testX.shape[2]))
    # invert scaling for forecast
#    inv_yhat = np.concatenate((testX, yhat), axis=1)
    inv_yhat = scaler.inverse_transform(yhat)
    pred = inv_yhat
    # invert scaling for actual
    testY = testY.reshape((len(testY), 1))
#    inv_y = np.concatenate((testX, testY), axis=1)
    inv_y = scaler.inverse_transform(testY)
    Y = inv_y
    
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

