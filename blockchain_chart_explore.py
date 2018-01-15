# %% imports
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from btcblock import load_data
# %% blockchain info chart plotting
# chart_dir = '~/code/insight/'
chart_files = ['transaction-fees', 'mempool-size', 'hash-rate', 'market-price', 'n-transactions', 'blocks-size', 'n-transactions-per-block']
blockdf = load_data.chartcsv2dataframe(chart_names=chart_files)
# pdseries = []
# for f in chart_files:
#     chart_series = pd.read_csv(chart_dir+f+'.csv', names=['time',f], parse_dates=['time'])#dtype={'time': 'datetime64[ns]', f: 'float64'})
#     pdseries.append(chart_series)
# blockdf = reduce(lambda x, y: pd.merge_asof(x, y, on='time'), pdseries)

blockdf_norm = blockdf.apply(lambda x: (x-np.min(x)) / (np.max(x)-np.min(x)))
blockdf_norm['time'] = blockdf['time']
# %% plot timeseries
matplotlib.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(12, 6))

plotcols = ['mempool-size', 'n-transactions-per-block', 'n-transactions', 'hash-rate', 'transaction-fees']
line_colors = ['k','r','b','c','m']
blockdf_norm[plotcols].plot(x=blockdf['time'], ax=ax, alpha=.4, color=line_colors)
df_smooth = blockdf_norm[plotcols].rolling(7, center=True).mean()
df_smooth['time'] = blockdf['time']
df_smooth.plot(x=blockdf['time'], ax=ax, color=line_colors)
ax.legend(plotcols)
ax.set_ylabel('percent max')
ax.set_xlabel('time')
ax.set_xlim((17000, 17540))
plt.show()
# %%
fig, ax = plt.subplots(figsize=(12, 6))
#blockdf_norm[plotcols].plot(x=blockdf['time'], ax=ax)

blockdf_norm[['transaction-fees']].plot(x=blockdf['time'], ax=ax)
ax.set_ylabel('percent max')
ax.set_xlabel('time')
#axs[1].set_xlabel('time (year)')
ax.set_xlim((17000, 17540))
#ax.set_xlim((1100, 1600))
plt.show()
