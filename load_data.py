import pandas as pd
from functools import reduce

def chartcsv2dataframe(chart_names):
    chart_dir = '~/code/insight/'
    if isinstance(chart_names, str):
        chart_names = [chart_names]

    pdseries = []
    for f in chart_names:
        chart_series = pd.read_csv(chart_dir+f+'.csv', names=['time',f], parse_dates=['time'])#dtype={'time': 'datetime64[ns]', f: 'float64'})
        pdseries.append(chart_series)
    blockdf = reduce(lambda x, y: pd.merge_asof(x, y, on='time'), pdseries)
    return blockdf
