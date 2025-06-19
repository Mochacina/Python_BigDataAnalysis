import pandas as pd
from scipy import stats
df = pd.read_csv("./gichul\gichul7\student_assessment.csv")
df = df.dropna()

print(df.head(5))

id = df['id_assessment'].value_counts().idxmax()
conf = (df['id_assessment'] == id)
df = df[conf]

import pandas as pd
from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()
df['score'] = ssc.fit_transform(df[['score']])
print(round(df['score'].max(),3))

df = pd.read_csv("./gichul\gichul7\stock_market.csv")
print(df.head(5))

df_corr = df.corr()['close'].abs()
print(df_corr)

col = df_corr.loc['DE1':'DE77'].idxmax()
print(round(df[col].mean(),4))

import pandas as pd
df = pd.read_csv("./gichul\gichul7\\air_quality.csv")

print(df.head())

q1 = df['CO2'].quantile(0.25)
q3 = df['CO2'].quantile(0.75)
iqr = q3-q1
_iqr = iqr*1.5

upper = q3+_iqr
lower = q1-_iqr

df_1 = df.sort_values('CO2', ascending=True)
print("df_1:\n", df_1['CO2'])

outliers = df[(df['CO2'] < lower) | (df['CO2'] > upper)]
print(outliers)