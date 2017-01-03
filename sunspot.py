import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#data preparation step
dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700','2008'))
del dta["YEAR"]

arma_mod20 = sm.tsa.ARMA(dta, (2, 0)).fit()
prediction = arma_mod20.predict('1990', '2012', dynamic=True)

