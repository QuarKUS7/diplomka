# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from Black_Scholes import black_scholes
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PATH = '/home/peter/Desktop/Diplomovka/'

data = pd.read_csv(f'{PATH}'+'put.csv')
#data = data[data['Date'].str.contains("2017")]
#data = data[data['Open Interest'] > 0]

data['RV'] = np.sqrt(data['RV'])*np.sqrt(252)

train_size = int(len(data) * 0.90)
valid_size = int(len(data) * 0.05)
test_size = len(data) - train_size - valid_size
train, valid, test = data.iloc[0:train_size,:], data.iloc[train_size:train_size+valid_size,:], data.iloc[train_size+valid_size:len(data),:]

data = test

test['BS'] = 1
for i in data.index:
    data['BS'][i] = black_scholes(cp=data['Call/Put'][i], s=data['Close'][i], 
        k=data['Strike Price'][i], t=data['t'][i], v=data['RV'][i], 
        rf=data['CTB3'][i], div=0)

rmse = np.sqrt(mean_squared_error(data['Last Price'], data['BS']))
print('Test RMSE: %.3f' % rmse)
mae = mean_absolute_error(data['Last Price'], data['BS'])
print('Test MAE: %.3f' % mae)

test.plot.scatter(x = ['t'],y =['BS'], marker = '.', linewidths = 0.05)

y = [test['Last Price'],test['BS']]
x = test['t']
for xe, ye in zip(x, y):
    test.plot.scatter([xe] * len(ye), ye)

div = data
div['diff'] = div['Last Price'] - div['BS']
sub = test[test['Instrument Symbol'] == 'SXO 180119C825.00']
sub.plot.scatter(x = ['t'],y =['diff'], marker = '.', linewidths = 0.05)

threedee = plt.figure().gca(projection='3d')
threedee.scatter(div.t, div['Close'], div['Last Price'])
threedee.set_xlabel('Index')
threedee.set_ylabel('H-L')
threedee.set_zlabel('Close')
plt.show()

ITMlong = div[(div['Close'] /div['Strike Price'] > 1.03) & (div['t'] > 180*1/252)]
ATMlong = div[(div['Close'] /div['Strike Price'] < 1.03) & (div['Close'] /div['Strike Price'] >= 0.97) &(div['t'] > 180*1/252)]
OTMlong = div[(div['Close'] /div['Strike Price'] < 0.97) & (div['t'] > 180*1/252)]
ITMshort = div[(div['Close'] /div['Strike Price'] > 1.03) & (div['t'] < 60*1/252)]
ATMshort = div[(div['Close'] /div['Strike Price'] < 1.03) & (div['Close'] /div['Strike Price'] >= 0.97) &(div['t'] < 60*1/252)]
OTMshort = div[(div['Close'] /div['Strike Price'] < 0.97) & (div['t'] < 60*1/252)]
ITMmid = div[(div['Close'] /div['Strike Price'] > 1.03) & (div['t'] >= 60*1/252) & (div['t'] < 180*1/252)]
ATMmid = div[(div['Close'] /div['Strike Price'] < 1.03) & (div['Close'] /div['Strike Price'] >= 0.97) & (div['t'] >= 60*1/252) & (div['t'] < 180*1/252)]
OTMmid = div[(div['Close'] /div['Strike Price'] < 0.97) & (div['t'] >= 60*1/252) & (div['t'] < 180*1/252)]

ITMlongRMSE = np.sqrt(mean_squared_error(ITMlong['Last Price'], ITMlong['BS']))
ATMlongRMSE = np.sqrt(mean_squared_error(ATMlong['Last Price'], ATMlong['BS']))
OTMlongRMSE = np.sqrt(mean_squared_error(OTMlong['Last Price'], OTMlong['BS']))
ITMshortRMSE = np.sqrt(mean_squared_error(ITMshort['Last Price'], ITMshort['BS']))
ATMshortRMSE = np.sqrt(mean_squared_error(ATMshort['Last Price'], ATMshort['BS']))
OTMlshortRMSE = np.sqrt(mean_squared_error(OTMshort['Last Price'], OTMshort['BS']))
ITMmidRMSE = np.sqrt(mean_squared_error(ITMmid['Last Price'], ITMmid['BS']))
ATMmidRMSE = np.sqrt(mean_squared_error(ATMmid['Last Price'], ATMmid['BS']))
OTMlmidRMSE = np.sqrt(mean_squared_error(OTMmid['Last Price'], OTMmid['BS']))

ITMlongMAE = mean_absolute_error(ITMlong['Last Price'], ITMlong['BS'])
ATMlongMAE = mean_absolute_error(ATMlong['Last Price'], ATMlong['BS'])
OTMlongMAE = mean_absolute_error(OTMlong['Last Price'], OTMlong['BS'])
ITMshortMAE = mean_absolute_error(ITMshort['Last Price'], ITMshort['BS'])
ATMshortMAE = mean_absolute_error(ATMshort['Last Price'], ATMshort['BS'])
OTMlshortMAE = mean_absolute_error(OTMshort['Last Price'], OTMshort['BS'])
ITMmidMAE = mean_absolute_error(ITMmid['Last Price'], ITMmid['BS'])
ATMmidMAE = mean_absolute_error(ATMmid['Last Price'], ATMmid['BS'])
OTMlmidMAE = mean_absolute_error(OTMmid['Last Price'], OTMmid['BS'])