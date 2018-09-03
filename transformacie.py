# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import eikon as ek
import pandas as pd
import pickle


ek.set_app_id('E9DE83B192F920DC8739C')

df = ek.get_timeseries(["SPXo161828000.U"], 
                       start_date="2018-02-12",  
                       interval = 'daily')
data = {}
for i in range(1100,3525,25):
    try:
        df= ek.get_timeseries(["SPXo1618{}0.U".format(i)],
#                               start_date="2018-02-12",
                               interval = 'daily')
    except:
        continue
    data[i] = df

dataset = pd.DataFrame()
for i in range(1100,3525,25):
    try:
        dataset = pd.concat([dataset, data[i]['CLOSE']], axis=1)
    except:
        continue

name = list(data.keys())
dataset.columns = name   

for i in name:
    data[i].to_csv('16FEB18_{}.csv'.format(i))

data = {}
for i in range(1100,3525,25):
    try:
        data[i] = pd.read_csv('options/16FEB18_{}.csv'.format(i))
    except:
        continue

ir = pd.DataFrame()
ir = pd.read_csv('DGS3MO.csv', na_values = '.')
ir.columns = ['Date','r']
ir.dtypes

mat = pd.DataFrame()

for i in data:
    data[i]['STRIKE'] = i
    
for i in data:
    mat = pd.concat([mat,data[i][['Date','CLOSE','STRIKE']]])

cat = pd.merge(mat,ir, how= 'outer', on="Date")
cat = cat.dropna(axis=0)
cat.dtypes

spot = pd.DataFrame()
spot = pd.read_csv('^GSPC.csv', na_values = '.')
spot = spot.drop(['Open','High','Low','Close','Volume'], axis = 1)
dat = pd.merge(cat,spot, how= 'outer', on="Date")
dat = dat.dropna(axis=0)

dat.to_csv("data.csv")

#tick = {}
#for i in range(1100,1800,25):
#    try:
#        df= ek.get_timeseries(["SPXo1618{}0.U".format(i)],
#                               start_date="2018-02-12",
#                               interval = 'tick')
#    except:
#        continue
#    tick[i] = df
#tick

