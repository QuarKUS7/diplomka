#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:30:57 2018

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

import math
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from MAPE import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

PATH = '/home/peter/Desktop/Diplomovka/'

#data = pd.read_csv(f'{PATH}'+'call.csv')
#data = data[data['Instrument Symbol'].str.contains("SXO 181221")]
#data = data.sort_values(['Date','Strike Price'])
#
##data = data[['Strike Price', 'DTB3','Close','std','t','Last Close Price']]
##data = data.values
p = data.groupby(['Instrument Symbol']).count()
pad = 341
#data = data[['Date','Strike Price', 'Expiry Date','CTB3','Instrument Symbol','Close','t','RV','Last Price']]
##data = data.iloc[:14,:]

def data_dist_max(dataset):
    dataX, dataY = [], []
    dataset = pd.DataFrame(dataset)
    for i in range(len(dataset)):
        row = dataset.iloc[i,:]
        dat = dataset[dataset['Instrument Symbol'] == row['Instrument Symbol']]
        dat = dat.drop(['Instrument Symbol'], axis = 1)
        #dat = dat.sort_values(['Date'])
        o = dat['Date'] == row['Date']
        o = o.reset_index()
        u = o.index[o.iloc[:,1] == True].tolist()
        if u[0] == 0:
            dataX.append(np.array(dat.iloc[:1,2:dat.shape[1]-1]))
            dataY.append(np.array(dat.iloc[0,-1]))
        else:
            dataX.append(np.array(dat.iloc[:u[0]+1,2:dat.shape[1]-1]))
            dataY.append(np.array(dat.iloc[u[0],-1]))

    return dataX, dataY
    
#dataX, dataY = data_dist_max(data)
#for i in range(14):
#    print(dataX[i])
#    print('Kon')
#dataX[55]
#def normalize_dataset(dataset):
#    d = []
#    dataset = pd.DataFrame(dataset)
#    dataset['Date'] = pd.to_datetime(dataset['Date'])
#    dataset['Date'] = dataset['Date'].dt.date
#    dates = pd.DataFrame(dataset.Date.unique()).sort_values([0])
#    dates.iloc[:,0] = pd.to_datetime(dates.iloc[:,0])
#    dates.iloc[:,0] = dates.iloc[:,0].dt.date
#    for i in range(len(dataset)):
#        row = dataset.iloc[i,:]   
#        o = dates == row['Date']
#        u = o.index[o[0] == True].tolist()
#        u = u[0]+1
#        if u == dates.shape[0]:
#            break
#        dat = dataset[dataset['Date'] == dates.iloc[u,0]]
#        if row['Strike Price'] not in dat.iloc[:,1].values:
#            row['Date'] = dates.iloc[u,0]
#            d.append(row)
#    d = pd.DataFrame(d)
#    data = pd.concat([dataset, d])
#    data = data.sort_values(['Date'])
#    return pd.DataFrame(data)
#
#data = data[['Date','Strike Price', 'CTB3','Close','t','std','Last Close Price']]
#data = data.dropna(0)
##data = data.astype('float32')
#da11=normalize_dataset(data)
#dataX[-1
#      ]
#
#def create_data(dataset, look_back):
#    dataset = pd.DataFrame(dataset)
#    dataX, dataY = [], []
#    o = []
#    u = 0
#    dat = []
#    for i in range(len(dataset)):
#        row = dataset.iloc[i,:]
#        dat = dataset[dataset['Strike Price'] == row[1]]
#        dat = dat.sort_values(['Date'])
#        o = dat['Date'] == row[0]
#        o = o.reset_index()
#        u = o.index[o.iloc[:,1] == True].tolist()
#        if u[0] > look_back:
#            dataX.append(np.array(dat.iloc[u[0]-look_back:u[0],1:dat.shape[1]]))
#            dataY.append(np.array(dat.iloc[u[0],-1]))
#
#            
#    return np.array(dataX), np.array(dataY)
#    
#datX, datY = create_data(da11, 10)

data = pd.read_csv(f'{PATH}'+'call.csv')
#data = data[data['Instrument Symbol'].str.contains("SXO 151120")]
#data = data.sort_values(['Date','Strike Price'])
data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','std','Last Price']]
data = data.dropna(0)
data = data.iloc[1:10000,:]
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = MinMaxScaler()
date = data[['Date','Instrument Symbol','Expiry Date']]
date = pd.DataFrame(date)
date.reset_index(inplace=True, drop=True)
#data = data.astype('float32')
dataset = data.drop(['Date','Instrument Symbol','Expiry Date'], axis = 1)
#dataset = scaler.fit_transform(dataset)
dataset = data.drop(['Date','Instrument Symbol','Expiry Date'], axis = 1)
dataset = dataset.astype('float32')
dataset.shape
dataset = scaler.fit_transform(dataset)
dataset = pd.DataFrame(dataset)
dataset = pd.concat([date, dataset],axis=1, ignore_index=True)
dataset.columns = ['Date','Instrument Symbol','Expiry Date','Strike Price', 'CTB3','Close','t','std','Last Price']

p = data
#look_back = 1
datasetX, datasetY = data_dist_max(dataset)
features = 5

train_size = int(len(datasetX) * 0.70)
valid_size = int(len(datasetX) * 0.20)
test_size = len(datasetX) - train_size - valid_size
trainX, validX, testX = datasetX[0:train_size], datasetX[train_size:train_size+valid_size], datasetX[train_size+valid_size:len(dataset)]
trainY, validY, testY = datasetY[0:train_size], datasetY[train_size:train_size+valid_size], datasetY[train_size+valid_size:len(dataset)]


trainY = np.array(trainY)
gg = []
for i in range(len(trainX)):
    gg.append(np.reshape(trainX[i], (1,-1,5)))

model = Sequential()
model.add(LSTM(100, input_shape=(None, 5)))
#model.add(LSTM(100, batch_input_shape=(757, trainX.shape[1], trainX.shape[2]), stateful=True))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#trainX, trainY = create_dataset(train, look_back)
#validX, validY = create_dataset(valid, look_back)  
#testX, testY = create_dataset(test, look_back)
yy = np.reshape(trainY[1],(1,1))

gg[0]

history = model.train_on_batch(model, gg, trainY)

for i in range(len(gg)):
    yy = np.reshape(trainY[i],(1,1))
    history = model.fit(gg[i], yy, batch_size=1, verbose=2, shuffle=False)

result = np.zeros(b.shape)
# plot history
pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='valid')
pyplot.legend()
pyplot.show()    
gg[1]
validX, validY = create_data(valid, look_back)  
trainX, trainY = create_data(train, look_back)

testX, testY = create_data(test, look_back)

def data_dist_max_pad(dataset):
    dataX, dataY = [], []
    dataset = pd.DataFrame(dataset)
    for i in range(len(dataset)):
        row = dataset.iloc[i,:]
        dat = dataset[dataset['Instrument Symbol'] == row['Instrument Symbol']]
        dat = dat.drop(['Instrument Symbol'], axis = 1)
        #dat = dat.sort_values(['Date'])
        o = dat['Date'] == row['Date']
        o = o.reset_index()
        u = o.index[o.iloc[:,1] == True].tolist()
        if u[0] == 0:
            result = np.zeros((pad,5))
            X = np.array(dat.iloc[:1,2:dat.shape[1]-1])
            result[:X.shape[0],:X.shape[1]] = X
            dataX.append(X)
            dataY.append(np.array(dat.iloc[0,-1]))
        else:
            result = np.zeros((pad,5))
            X = np.array(dat.iloc[:u[0]+1,2:dat.shape[1]-1])
            result[:X.shape[0],:X.shape[1]] = X
            dataX.append(X)
            dataY.append(np.array(dat.iloc[u[0],-1]))

    return dataX, dataY

dataXp, dataYp = data_dist_max_pad(dataset)
