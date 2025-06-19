import pandas as pd
from scipy import stats

df = pd.read_csv("./gichul\gichul6\data6-1-1.csv")

### EDA
print("df.head(5): \n",df.head(5))
df.info()

df['출동시간'] = pd.to_datetime(df['출동시간']) 
df['도착시간'] = pd.to_datetime(df['도착시간'])
df.info()

df['이동시간(분)'] = (df['도착시간']-df['출동시간']).dt.total_seconds() / 60
print("df.head(5): \n",df.head(5))

print(df.groupby("소방서")['이동시간(분)'].mean())
maxmeantime = max(df.groupby("소방서")['이동시간(분)'].mean().iloc[:])

print("평균 가장 오래 걸린 소방서:", round(maxmeantime))

#avg_diff = df.groupby("소방서")

df = pd.read_csv("./gichul\gichul6\data6-1-2.csv")

# EDA
print("df.head(5): \n",df.head(5))
#print("df.sample(5): \n",df.sample(5))
print("df.info:");df.info()
df['총학생수'] = df.iloc[:,2:].sum(axis=1)
df['교사 당 학생 수'] = df['총학생수'] / df['교사수']
print("df.head(5): \n",df.head(5))

idx = df['교사 당 학생 수'].idxmax()

print("교사 수: ", df.loc[idx,'교사수'])

df = df.sort_values('교사 당 학생 수', ascending=False)
print("df.head(5): \n",df.head(5))

df = pd.read_csv("./gichul\gichul6\data6-1-3.csv")
print("df.head(5):\n", df.head(5))
print("df.info():");df.info()

# 1) 총 범죄 건수 계산
df['총 범죄 건수'] = df.iloc[:, 1:7].sum(axis=1)

# 2) 연도 슬라이싱
df['연도'] = df["날짜"].str[:4]
print("df.head(5):\n", df.head(5))

print(round(df.groupby(df['연도'])['총 범죄 건수'].sum().max()/12))

