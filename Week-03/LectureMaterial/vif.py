"""
Variance Inflation Factor (VIF)

measures multicolinearity

VIF is greater than 5 means high multicolinearity
"""
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/train.csv', index_col=0, parse_dates=True)
del df['atemp']
del df['humidity']
df = df.iloc[:, :-3]

vifs = [VIF(df.values, i) for i, colname in enumerate(df)]
s = pd.Series(vifs, index=df.columns)
s.plot.bar()
plt.show()
sns.heatmap(df.corr())
plt.show()

