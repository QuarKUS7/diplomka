#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 01:00:23 2018

@author: peter
"""
import pandas as pd
import glob

PATH = '/home/peter/Desktop/Diplomovka/'

sheet_names = glob.glob("/home/peter/Desktop/Diplomovka/opcie/opcie/*.xlsx")

excels = [pd.ExcelFile(name) for name in sheet_names]

frames = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels]

frames[1:] = [df[1:] for df in frames[1:]]

combined = pd.concat(frames)
combined.dtypes
cols = combined.iloc[0]

combined = combined[1:]
combined.columns = cols

combined.to_csv(f'{PATH}'+'data.csv')