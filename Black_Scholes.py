#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 20:51:09 2018

@author: peter
"""

import pandas as pd
from scipy import stats
import numpy as np

def black_scholes (cp, s, k, t, v, rf, div):
        """ Price an option using the Black-Scholes model.
        s: initial stock price
        k: strike price
        t: expiration time
        v: volatility
        rf: risk-free rate
        div: dividend
        cp: +1/-1 for call/put
        """
        #d1 = (1/v*np.sqrt(t))*[np.log(s/k)+(rf+(np.power(v,2))*0.5)*t]
        #d2 = d1 - v*np.sqrt(t)
        d1 = (np.log(s/k)+(rf-div+0.5*np.power(v,2))*t)/(v*np.sqrt(t))
        d2 = d1 - v*np.sqrt(t)

        optprice = (cp*s*stats.norm.cdf(cp*d1)) - (cp*k*np.exp(-rf*t)*stats.norm.cdf(cp*d2))
        return optprice