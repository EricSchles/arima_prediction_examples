#http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/tsa_arma_0.html - example comes from
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
prediction = arma_mod20.predict('1990', '2112', dynamic=True)
#http://stackoverflow.com/questions/16824607/pandas-appending-a-row-to-a-dataframe-and-specify-its-index-label
#http://stackoverflow.com/questions/24036911/how-to-update-values-in-a-specific-row-in-a-python-pandas-dataframe
transform = []
df = pd.DataFrame()
for index in range(len(prediction)):
    if prediction.index[index] not in dta.index:
        s = pd.Series({"prediction": prediction.xs(prediction.index[index])})
        s.name = prediction.index[index]
        dta = dta.append(s)
    else:
        s = pd.Series({
            "prediction": prediction.xs(prediction.index[index]),
            "SUNACTIVITY": dta.ix[index]["SUNACTIVITY"]
        })
        s.name = prediction.index[index]
        df = df.append(s)
dta.update(df)
dta.plot()
plt.show()
