# 8-1-1
import pandas as pd
df = pd.read_csv("./gichul\\gichul8\\drinks.csv")

print("shape=", df.shape)
print("head=\n", df.head)

print("---")
beer_max_id = df.groupby('continent')['beer_servings'].mean().idxmax()
print(beer_max_id)

cond = df['continent'] == beer_max_id
df = df[cond]
df = df.sort_values('beer_servings', ascending=False)
print(df.iloc[4,0:2])

print("--- 8-1-2 ---")

# 8-1-2
import pandas as pd
df = pd.read_csv("./gichul\gichul8\\tourist.csv")

print("shape=",df.shape)
print("head=\n",df.head())

df['방문객합계'] = df.iloc[:,1:].sum(axis=1)
df['관광객비율'] = df['관광']/df['방문객합계']
print("head=\n",df.head())
a = df.sort_values('관광객비율', ascending=False).iloc[1,3]
b = df.sort_values('관광', ascending=False).iloc[1,2]
print(a+b)

print("--- 8-1-3 ---")
import pandas as pd
df = pd.read_csv("./gichul\gichul8\chem.csv")

print("shape=",df.shape)
print("head=\n",df.head())

from sklearn.preprocessing import MinMaxScaler
minmaxsc = MinMaxScaler()
df['co_scaled'] = minmaxsc.fit_transform(df[['co']])
df['nmhc_scaled'] = minmaxsc.fit_transform(df[['nmhc']])

co_std = df['co_scaled'].std()
nmhc_std = df['nmhc_scaled'].std()
print(co_std, nmhc_std)

std_diff = round(co_std - nmhc_std, 3)
print(std_diff)