# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import math
from pandas.tseries.offsets import *
import matplotlib.pyplot as plt
#from Black_Scholes import black_scholes
from datetime import datetime, timedelta

PATH = '/home/peter/Desktop/Diplomovka/'

data = pd.read_csv(f'{PATH}'+'data.csv')

#data = data[1:1000]
data = data[['Date', 'Strike Price', 'Expiry Date', 'Call/Put', 
             'Instrument Symbol', 'Last Price', 'Open Interest', 'Implied Volatility']]
data['Date'] = pd.to_datetime(data['Date'])
data['Expiry Date'] = pd.to_datetime(data['Expiry Date'])


r = pd.read_csv(f'{PATH}'+'CTB3.csv')
r['DATE'] = pd.to_datetime(r['DATE'])
r['CTB3'] = r['CTB3']/100
data = data.merge(r, how='inner', left_on='Date', right_on='DATE')
data = data.drop(['DATE'], axis=1)

spot = pd.read_csv(f'{PATH}'+'PriceHistory.csv', header =0)
spot['Exchange Date'] = pd.to_datetime(spot['Exchange Date'])
spot = spot[['Exchange Date','Close']]
spot = spot.iloc[::-1]
spot['u'] = np.log(spot.Close) - np.log(spot.Close.shift(1))
spot['Volatility'] = spot['u'].rolling(window=21,center=False).std()*math.sqrt(252)*100

data = data.merge(spot, how='inner', left_on='Date', right_on='Exchange Date')
data = data.drop(['Exchange Date','u'], axis=1)

RV = pd.read_csv(f'{PATH}'+'oxfordmanrealizedvolatilityindices.csv', header =0)
RV = RV[RV['Symbol'] == '.GSPTSE']
RV = RV[['Unnamed: 0','rv5_ss']]
RV['Date'] = pd.DatetimeIndex(RV['Unnamed: 0']).normalize()
RV = RV.drop(['Unnamed: 0'], axis = 1)
RV['Date'] = pd.to_datetime(RV['Date'])
RV['Date'] = RV['Date'].shift(1)
RV.columns = ['RV', 'Date']
data = data.merge(RV, how='inner', left_on='Date', right_on='Date')


data['t'] = data['Expiry Date'] - data['Date']
data['t'] = data['t'] / timedelta(days=1)
data['t'] = data['t']*0.7
data['t'] = data['t'] / 252

data['Call/Put'] = data['Call/Put'].map({0: 1, 1: -1})

data['Last Price'] = data['Last Price'].shift(-1)

data = data.dropna(0)

data = data[data.t > 0.0137]
data = data[data['Last Price'] > 0.375]

call = data[(data['Call/Put'] == 1) & (data['Last Price'] > (data['Close'] - data['Strike Price']))]
put = data[(data['Call/Put'] == -1) & (data['Last Price'] > (data['Strike Price'] - data['Close']))]

call = call.sort_values(['Date'], axis = 0)
put = put.sort_values(['Date'], axis = 0)

call.to_csv(f'{PATH}'+'call.csv')
put.to_csv(f'{PATH}'+'put.csv')

with plt.style.context(('ggplot')):
    data.plot(x =['Date'], y = ['Volatility'],linewidth=0.85)
    plt.xlabel('Date')
    plt.ylabel('%')
    plt.yticks(np.arange(0, 30, step=5))
