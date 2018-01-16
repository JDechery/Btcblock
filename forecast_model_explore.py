import pandas as pd
from Btcblock import etl_utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

chart_files = ['transaction-fees', 'mempool-size', 'hash-rate', 'market-price', 'n-transactions', 'blocks-size', 'n-transactions-per-block']
df = etl_utils.chartcsv2dataframe(chart_names=chart_files)
df.set_index('time', inplace=True)
#df = df.iloc[-1000:, :]
# %%
#df[chart_files].apply(lambda x: x/np.max(x)).iloc[1400:, :].plot()
df['transaction-fees'].iloc[-500:].plot()
# df['mempool-size'].plot()
plt.show()
# %% statsmodels arima
from statsmodels.tsa.arima_model import ARIMA
model_var = 'transaction-fees'
forecast_dur = 6
order = (4, 1, 2)
# test_data = df['mempool-size'].iloc[1500:-forecast_dur]
test_data = df[model_var].iloc[1500:-forecast_dur]
model = ARIMA(test_data, order)
fitted_model = model.fit(maxiter=100, transparams=True)

# % plot results/residuals
# forecast_dates = pd.date_range(start='2018-01-11', end='2018-01-19').to_datetime()
# yhat = fitted_model.predict()
yfuture = fitted_model.forecast(steps=forecast_dur)#start=len(test_data)-1, end=len(test_data)+10)
fig, ax = plt.subplots(figsize=(8, 4))
plt.plot_date(df.index[-forecast_dur:], yfuture[0], fmt='-k')
plt.plot_date(df.index[-forecast_dur:], yfuture[0]+yfuture[1], fmt=':k')
plt.plot_date(df.index[-forecast_dur:], yfuture[0]-yfuture[1], fmt=':k')
# test_data.plot()
df[model_var].iloc[1500:].plot()
# %%
from statsmodels.tsa.stattools import acf, pacf
dat = np.log(test_data)
dat = dat - dat.rolling(5).mean()
dat.fillna(method='bfill', inplace=True)
lag_acf = acf(dat, nlags=10)
lag_pacf = pacf(np.log(test_data), nlags=10, method='ols')
plt.plot(lag_pacf)

# %%
df['mempool-size'].iloc[1000:].plot()
# %% sklearn linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
nlag = 10;
cutoff = 1400;
model = LinearRegression()
feature_names = ['mempool-size', 'n-transactions-per-block', 'n-transactions', 'hash-rate']
target_name = ['transaction-fees']

features = []
for lag in range(nlag):
    features.append(df[feature_names].shift(lag))
    features[-1].columns = [name+str(lag) for name in feature_names]

features = pd.concat(features, axis=1)
#features.fillna(method='backfill', axis=0, inplace=True)
features = features.loc[cutoff:, :]
target = df.loc[1200:, target_name]
#target = target.loc[cutoff:, :]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
