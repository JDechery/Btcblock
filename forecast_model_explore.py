from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from Btcblock import load_data
import matplotlib
import matplotlib.pyplot as plt

chart_files = ['transaction-fees', 'mempool-size', 'hash-rate', 'market-price', 'n-transactions', 'blocks-size', 'n-transactions-per-block']
df = load_data.chartcsv2dataframe(chart_names=chart_files)
df = df.loc[-1000:, :]

# %%
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

# %%
target.plot()
plt.show()
