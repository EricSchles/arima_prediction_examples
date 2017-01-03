#http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/ - example comes from
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('portland-oregon-average-monthly-.csv', index_col=0)
df.index.name=None
df.reset_index(inplace=True)
df.drop(df.index[114], inplace=True)

start = datetime.datetime.strptime("1973-01-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,114)]
df['index'] = date_list
df.set_index(['index'], inplace=True)
df.index.name=None

df.columns = ['riders']
df['riders'] = df.riders.apply(lambda x: int(x) * 100)
import code
code.interact(local=locals())
model = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0, 1, 0), seasonal_order=(0, 1, 1, 12))
results = model.fit()
df['forecast'] = results.predict(start = 102, end = 114, dynamic = True)

