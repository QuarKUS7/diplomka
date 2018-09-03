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
#from sklearn import preprocessing
#from keras.models import Sequential
#from keras.layers import Dense
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    PATH = '/home/peter/Desktop/Diplomovka/'
    print('Penis')
    data = pd.read_csv(f'{PATH}'+'call.csv')
    #data = data[data['Instrument Symbol'].str.contains("SXO 151120")]
    #data = data.sort_values(['Date','Strike Price'])
    #data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','std','Last Price']]
    data = data[['Date','Expiry Date','Instrument Symbol','Strike Price', 'CTB3','Close','t','RV','Last Price']]
    data = data.dropna(0)
    #data = data.iloc[1:10000,:]
    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = MinMaxScaler()    
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
    x_train, x_test = dataset.iloc[0:train_size,:-1], dataset.iloc[train_size:train_size+valid_size,:-1]
    y_train, y_test = dataset.iloc[0:train_size,-1], dataset.iloc[train_size:train_size+valid_size,-1]
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense({{choice([2, 5, 8, 16, 32, 64, 96, 128, 256])}}, input_dim=x_train.shape[1], activation="relu", kernel_initializer="uniform"))
        # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['one', 'two'])}}) == 'two':
        model.add(Dense({{choice([2, 5, 8, 16, 32, 64, 96, 128, 256])}}, activation="relu", kernel_initializer="uniform"))
        
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    model.fit(x_train, y_train,
              batch_size=8,
              epochs=15,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=55,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)