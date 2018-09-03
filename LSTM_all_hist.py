#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:32:06 2018

@author: peter
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 13:43:15 2018

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
from keras.layers import Activation
from keras.layers  import Masking
import random
from keras import optimizers
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from MAPE import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

PATH = '/home/peter/Desktop/Diplomovka/'

features = 5

pad = 581

def new(row, dataset):
    dataX, dataY = [], []
    dataset = pd.DataFrame(dataset)
    dat = dataset[dataset['Instrument Symbol'] == row.iloc[0]['Instrument Symbol']]
    dat = dat[dat['Date'] <= row.iloc[0]['Date']]
    dat = dat.drop(['Instrument Symbol','Date','Expiry Date'], axis = 1)
    #dat = dat.sort_values(['Date'])
    result = np.zeros((pad,5))
    X = np.array(dat.iloc[-pad:,:-1])
    result[:X.shape[0],:X.shape[1]] = X
    dataX.append(result)
    dataY.append(np.array(dat.iloc[-1,-1]))
    return dataX, dataY

data = pd.read_csv(f'{PATH}'+'call.csv')
#data = data[data['Instrument Symbol'].str.contains("SXO 151120")]
#data = data.sort_values(['Date','Strike Price'])
#data = data[data['Open Interest'] > 0]
#data = data[0:10000]
#data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','std','Last Price']]
data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','RV','Last Price']]
data = data.dropna(0)
scaler = MinMaxScaler(feature_range=(0, 1))

date = data[['Date','Instrument Symbol','Expiry Date']]
date = pd.DataFrame(date)
date.reset_index(inplace=True, drop=True)
#data = data.astype('float32')
dataset = data.drop(['Date','Instrument Symbol','Expiry Date'], axis = 1)
dataset = scaler.fit_transform(dataset)
#dataset = data.drop(['Date','Instrument Symbol','Expiry Date'], axis = 1)
dataset = dataset.astype('float32')
dataset.shape
#dataset = scaler.fit_transform(dataset)
dataset = pd.DataFrame(dataset)
dataset = pd.concat([date, dataset],axis=1, ignore_index=True)
#dataset.columns = ['Date','Instrument Symbol','Expiry Date','Strike Price', 'CTB3','Close','t','std','Last Price']
dataset.columns = ['Date','Instrument Symbol','Expiry Date','Strike Price', 'CTB3','Close','t','RV','Last Price']
train_size = int(len(dataset) * 0.90)
valid_size = int(len(dataset) * 0.05)
test_size = len(dataset) - train_size - valid_size
trainX, validX, testX = dataset.iloc[0:train_size,:], dataset.iloc[train_size:train_size+valid_size,:], dataset.iloc[train_size+valid_size:len(dataset),:]
#trainY, validY, testY = dataset.iloc[0:train_size,-1], dataset.iloc[train_size:train_size+valid_size,-1], dataset.iloc[train_size+valid_size:len(dataset),-1]

adam = optimizers.Adam(lr=0.0005)

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(pad, 5)))
model.add(LSTM(128,return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(16))
model.add(Dense(1))
#model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer=adam)

def generator(dataset, part, batch_size):
 # Create empty arrays to contain batch of features and labels#
 while True:
   for i in range(batch_size):
     # choose random index in features
     index= part.sample(1, replace=False, weights=None, random_state=None, axis=0)
     dataX, dataY = new(index, dataset)
     dataX = np.reshape(dataX, (len(dataX),pad,5))
     dataY = np.reshape(dataY,(len(dataY),1))
     yield (dataX , dataY)
      
#history = model.fit_generator(generator(dataset, trainX, 64), samples_per_epoch=trainX.shape[0], nb_epoch=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
history = model.fit_generator(generator(dataset, trainX, 128), steps_per_epoch=trainX.shape[0]/128, verbose=1, 
                              callbacks=None, validation_data=None, validation_steps=None, class_weight=None, 
                              max_queue_size=256, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0, 
                              epochs=4)

data_pred_x = []
data_pred_y = []
model.summary()
for i in range(testX.shape[0]):
      index= testX.iloc[i]
      index = index.to_frame()
      index = index.transpose()
      dataX, dataY = new(index, dataset)
      dataX = np.reshape(dataX, (len(dataX),pad,5))
      dataY = np.reshape(dataY,(len(dataY),1))
      data_pred_x.append(dataX)
      data_pred_y.append(dataY)
data_pred_x = np.reshape(data_pred_x, (len(data_pred_x),pad,5))     
data_pred_y = np.reshape(data_pred_y, (len(data_pred_x),1))     


yhat = model.predict(data_pred_x)    
  
#yhat = model.predict_generator(new_pred(dataset, testX, 0), steps=1, max_queue_size=10, workers=1, use_multiprocessing=False)

pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='valid')
pyplot.legend()
pyplot.show()

yhat = pd.DataFrame(yhat)
result = pd.DataFrame(np.zeros((testX.shape[0], 5)))
inv_yhat = pd.DataFrame(np.concatenate((result, yhat), axis=1))
inv_yhat = scaler.inverse_transform(inv_yhat)
pred = pd.DataFrame(inv_yhat[:,-1])
testY = pd.DataFrame(data_pred_y)
inv_y = np.concatenate((result, testY), axis=1)
inv_y = scaler.inverse_transform(inv_y)
Y = pd.DataFrame(inv_y[:,-1])
# calculate metrics
rmse = np.sqrt(mean_squared_error(Y, pred))
print('Test RMSE: %.3f' % rmse)
mae = mean_absolute_error(Y, pred)
print('Test MAE: %.3f' % mae)
Y = pd.DataFrame(Y)
pred = pd.DataFrame(pred)
vid = np.concatenate((Y,pred), axis = 1)


yhat = pd.DataFrame(yhat)
result = pd.DataFrame(np.zeros((testX.shape[0], 5)))
inv_yhat = pd.DataFrame(np.concatenate((result, yhat), axis=1))
inv_yhat = scaler.inverse_transform(inv_yhat)
pred = np.maximum(inv_yhat[:,-1], 0)
pred = pd.DataFrame(pred)
testY = pd.DataFrame(data_pred_y)
inv_y = np.concatenate((result, testY), axis=1)
inv_y = scaler.inverse_transform(inv_y)
Y = pd.DataFrame(inv_y[:,-1])
# calculate metrics
rmse = np.sqrt(mean_squared_error(Y, pred))
print('Test RMSE: %.3f' % rmse)
mae = mean_absolute_error(Y, pred)
print('Test MAE: %.3f' % mae)
Y = pd.DataFrame(Y)
pred = pd.DataFrame(pred)
vid = np.concatenate((Y,pred), axis = 1)

Y = pd.DataFrame(Y)
inv_g = data.iloc[train_size+valid_size:len(data),:]
inv_g = inv_g[['Strike Price', 'CTB3','Close','t','std']]
inv_yhat = np.concatenate((inv_g, pred), axis = 1)
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