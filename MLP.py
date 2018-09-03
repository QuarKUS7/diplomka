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
from sklearn.metrics import mean_absolute_error
#from sklearn import preprocessing
#from keras.models import Sequential
#from keras.layers import Dense
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe

PATH = '/home/peter/Desktop/Diplomovka/'

data = pd.read_csv(f'{PATH}'+'put.csv')
#data = data[data['Instrument Symbol'].str.contains("SXO 151120")]
#data = data.sort_values(['Date','Strike Price'])
#data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','std','Last Price']]
data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','RV','Last Price']]
data['RV'] = np.sqrt(data['RV'])*np.sqrt(252)
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

#dataset.columns = ['Strike Price', 'CTB3','Close','t','std','Last Price']
dataset.columns = ['Strike Price', 'CTB3','Close','t','RV','Last Price']
train_size = int(len(dataset) * 0.90)
valid_size = int(len(dataset) * 0.05)
test_size = len(dataset) - train_size - valid_size
trainX, validX, testX = dataset.iloc[0:train_size,:-1], dataset.iloc[train_size:train_size+valid_size,:-1], dataset.iloc[train_size+valid_size:len(dataset),:-1]
trainY, validY, testY = dataset.iloc[0:train_size,-1], dataset.iloc[train_size:train_size+valid_size,-1], dataset.iloc[train_size+valid_size:len(dataset),-1]

model = Sequential()
model.add(Dense(64, input_dim=trainX.shape[1], activation="relu", kernel_initializer="uniform"))
model.add(Dense(128, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(trainX, trainY, epochs=5, batch_size=8, validation_data=(validX, validY), verbose=1, shuffle=True)
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

Y = pd.DataFrame(Y)
div = np.concatenate((inv_yhat,Y), axis = 1)
div = pd.DataFrame(div)
div.columns = ['Strike Price', 'CTB3','Close','t','std','pred','Y']

ITMlong = div[(div['Close'] /div['Strike Price'] > 1.03) & (div['t'] > 180*1/252)]
ATMlong = div[(div['Close'] /div['Strike Price'] < 1.03) & (div['Close'] /div['Strike Price'] >= 0.97) &(div['t'] > 180*1/252)]
OTMlong = div[(div['Close'] /div['Strike Price'] < 0.97) & (div['t'] > 180*1/252)]
ITMshort = div[(div['Close'] /div['Strike Price'] > 1.03) & (div['t'] < 60*1/252)]
ATMshort = div[(div['Close'] /div['Strike Price'] < 1.03) & (div['Close'] /div['Strike Price'] >= 0.97) &(div['t'] < 60*1/252)]
OTMshort = div[(div['Close'] /div['Strike Price'] < 0.97) & (div['t'] < 60*1/252)]
ITMmid = div[(div['Close'] /div['Strike Price'] > 1.03) & (div['t'] >= 60*1/252) & (div['t'] < 180*1/252)]
ATMmid = div[(div['Close'] /div['Strike Price'] < 1.03) & (div['Close'] /div['Strike Price'] >= 0.97) & (div['t'] >= 60*1/252) & (div['t'] < 180*1/252)]
OTMmid = div[(div['Close'] /div['Strike Price'] < 0.97) & (div['t'] >= 60*1/252) & (div['t'] < 180*1/252)]

ITMlongRMSE = np.sqrt(mean_squared_error(ITMlong['Y'], ITMlong['pred']))
ATMlongRMSE = np.sqrt(mean_squared_error(ATMlong['Y'], ATMlong['pred']))
OTMlongRMSE = np.sqrt(mean_squared_error(OTMlong['Y'], OTMlong['pred']))
ITMshortRMSE = np.sqrt(mean_squared_error(ITMshort['Y'], ITMshort['pred']))
ATMshortRMSE = np.sqrt(mean_squared_error(ATMshort['Y'], ATMshort['pred']))
OTMlshortRMSE = np.sqrt(mean_squared_error(OTMshort['Y'], OTMshort['pred']))
ITMmidRMSE = np.sqrt(mean_squared_error(ITMmid['Y'], ITMmid['pred']))
ATMmidRMSE = np.sqrt(mean_squared_error(ATMmid['Y'], ATMmid['pred']))
OTMlmidRMSE = np.sqrt(mean_squared_error(OTMmid['Y'], OTMmid['pred']))

ITMlongMAE = mean_absolute_error(ITMlong['Y'], ITMlong['pred'])
ATMlongMAE = mean_absolute_error(ATMlong['Y'], ATMlong['pred'])
OTMlongMAE = mean_absolute_error(OTMlong['Y'], OTMlong['pred'])
ITMshortMAE = mean_absolute_error(ITMshort['Y'], ITMshort['pred'])
ATMshortMAE = mean_absolute_error(ATMshort['Y'], ATMshort['pred'])
OTMlshortMAE = mean_absolute_error(OTMshort['Y'], OTMshort['pred'])
ITMmidMAE = mean_absolute_error(ITMmid['Y'], ITMmid['pred'])
ATMmidMAE = mean_absolute_error(ATMmid['Y'], ATMmid['pred'])
OTMlmidMAE = mean_absolute_error(OTMmid['Y'], OTMmid['pred'])

put = data[(data['Call/Put'] == -1) & (data['Last Price'] > (data['Strike Price'] - data['Close']))]

for i in range(9):
    print(i)
