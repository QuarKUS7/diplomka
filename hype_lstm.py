#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:27:03 2018

@author: peter
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 22:11:54 2018

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
from keras.layers  import Masking
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe
from sklearn.preprocessing import StandardScaler

def new(row, dataset):
    dataX, dataY = [], []
    dataset = pd.DataFrame(dataset)
    dat = dataset[dataset['Instrument Symbol'] == row.iloc[0]['Instrument Symbol']]
    dat = dat[dat['Date'] <= row.iloc[0]['Date']]
    dat = dat.drop(['Instrument Symbol','Date','Expiry Date'], axis = 1)
    #dat = dat.sort_values(['Date'])
    result = np.zeros((581,5))
    X = np.array(dat.iloc[-581:,:-1])
    result[:X.shape[0],:X.shape[1]] = X
    dataX.append(result)
    dataY.append(np.array(dat.iloc[-1,-1]))
    return dataX, dataY

def generator_pred(dataset, part, batch_size):
 # Create empty arrays to contain batch of features and labels#
 while True:
    for i in range(batch_size):
      # choose random index in features
      index= part.sample(1, replace=False, weights=None, random_state=None, axis=0)
      dataX, dataY = new(index, dataset)
      dataX = np.reshape(dataX, (len(dataX),581,5))
      dataY = np.reshape(dataY,(len(dataY),1))
       
      yield dataX

def new_pred(dataset, part, batch_size):
  # Create empty arrays to contain batch of features and labels#
  # choose random index in features
  while True:
      for i in range(batch_size):
          index= part.iloc[i]
          index = index.to_frame()
          index = index.transpose()
          dataX, dataY = new(index, dataset)
          dataX = np.reshape(dataX, (len(dataX),581,5))
          dataY = np.reshape(dataY,(len(dataY),1))
          yield dataX
          
def generator(dataset, part, batch_size):
    # Create empty arrays to contain batch of features and labels#
    while True:
        for i in range(batch_size):
            # choose random index in features
            index= part.sample(1, replace=False, weights=None, random_state=None, axis=0)
            dataX, dataY = new(index, dataset)
            dataX = np.reshape(dataX, (len(dataX),581,5))
            dataY = np.reshape(dataY,(len(dataY),1))
            yield dataX , dataY

def data():
    PATH = '/home/peter/Desktop/Diplomovka/'
    data = pd.read_csv(f'{PATH}'+'call.csv')
    #data = data[data['Instrument Symbol'].str.contains("SXO 151120")]
    #data = data.sort_values(['Date','Strike Price'])
    data = data[data['Open Interest'] > 0]
    #data = data[0:10000]
    data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','RV','Last Price']]
    data['RV'] = np.sqrt(data['RV'])*np.sqrt(252)
    #data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','std','Last Price']]
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
    x_train,  x_test = dataset.iloc[0:train_size,:], dataset.iloc[train_size:train_size+valid_size,:]
    
    return x_train,  x_test, dataset

def create_model(x_train, x_test, dataset):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    pad = 581
    def generator(dataset, part, batch_size):
        # Create empty arrays to contain batch of features and labels#
        while True:
            for i in range(batch_size):
                # choose random index in features
                index= part.sample(1, replace=False, weights=None, random_state=None, axis=0)
                dataX, dataY = new(index, dataset)
                dataX = np.reshape(dataX, (len(dataX),581,5))
                dataY = np.reshape(dataY,(len(dataY),1))
                yield dataX , dataY
                
    def new(row, dataset):
        dataX, dataY = [], []
        dataset = pd.DataFrame(dataset)
        dat = dataset[dataset['Instrument Symbol'] == row.iloc[0]['Instrument Symbol']]
        dat = dat[dat['Date'] <= row.iloc[0]['Date']]
        dat = dat.drop(['Instrument Symbol','Date','Expiry Date'], axis = 1)
        #dat = dat.sort_values(['Date'])
        result = np.zeros((581,5))
        X = np.array(dat.iloc[-581:,:-1])
        result[:X.shape[0],:X.shape[1]] = X
        dataX.append(result)
        dataY.append(np.array(dat.iloc[-1,-1]))
        return dataX, dataY
    
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(pad, 5)))
    model.add(LSTM({{choice([2, 5, 8, 16, 32, 64, 96, 128, 256])}} ,return_sequences=True))
    model.add(LSTM({{choice([2, 5, 8, 16, 32, 64, 96, 128, 256])}}))
    #model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.fit_generator(generator(dataset, x_train, 64), steps_per_epoch=x_train.shape[0]/64, verbose=1, callbacks=None,#validation_data=generator(dataset, validX, 128), validation_steps=validX.shape[0], 
                        class_weight=None, max_queue_size=128, workers=1, use_multiprocessing=False, initial_epoch=0, epochs=8)
    score, acc = model.evaluate_generator(generator(dataset, x_test, 64), steps=x_test.shape[0]/64, max_queue_size=128 )
    print('Test accuracy:', acc)
    return {'loss': acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    features = 5

    pad = 581
    x_train, x_test, dataset = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    print(best_model.evaluate_generator(generator(dataset, x_test, 64), steps=x_test.shape[0]/64, max_queue_size=128))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
