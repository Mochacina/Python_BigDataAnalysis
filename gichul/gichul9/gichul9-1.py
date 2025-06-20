import pandas as pd
df = pd.read_csv("./gichul\gichul9\loan.csv")

print("shape:",df.shape)

df['총대출액'] = df['신용대출'] + df['담보대출']
print("head:\n",df.head)

grouped = df.groupby(['지역코드','성별'])['총대출액'].sum().unstack()
grouped['성별간 차이'] = abs(grouped[1] - grouped[2])
print(grouped)

print(grouped['성별간 차이'].idxmax())

# 문제 2

import pandas as pd
df = pd.read_csv("gichul\gichul9\crime.csv")

print("shape:",df.shape)
#df['합계'] = df.iloc[:,2:].sum(axis=1)
print("head:\n",df.head)

df_cause = df['구분'] == '발생건수'
df_arrest = df['구분'] == '검거건수'
df_1 = df[df_cause].iloc[:,2:]
df_2 = df[df_arrest].iloc[:,2:]

df_1 = df_1.reset_index(drop=True)
df_2 = df_2.reset_index(drop=True)
df_3 = df_2 / df_1
print("df_3:\n",df_3)

list = df_3.idxmax(axis=1)
print(list)

result = 0
for i, crime in enumerate(list):
    n = df_2.loc[i,crime]
    print(n)
    result += n
print("result:",result)

# 문제 3

import pandas as pd
df = pd.read_csv("./gichul\gichul9\hr.csv")

mean = df['만족도'].mean()
df['만족도'] = df['만족도'].fillna(mean)

print("shape:",df.shape)
print("head:\n",df.head())

grouped = df.groupby(['부서', '성과등급'])['근속연수'].transform('mean')
grouped = grouped.astype(int)
print(grouped)

df['근속연수'] = df['근속연수'].fillna(grouped)
df['연봉/근속연수'] = df['연봉']/df['근속연수']
print(df.sort_values('연봉/근속연수',ascending=False).iloc[2]['근속연수'])

df_y = df.nlargest(3, '연봉/근속연수')
A = df_y.iloc[-1]["근속연수"]
print(A)

df['연봉/만족도'] = df['연봉']/df['만족도']
B = df.sort_values('연봉/만족도',ascending=False).iloc[1]['교육참가횟수']
print(B)

print(int(A+B))