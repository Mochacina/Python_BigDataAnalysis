# 회귀 분석

import pandas as pd
from scipy import stats

df = pd.read_csv("./part3\ch4\study.csv")
print(df)

### 상관 계수
correlation = df.corr(method='pearson', numeric_only=True)
print(correlation)

data = {
    '키': [150, 160, 170, 175, 165],
    '몸무게': [42, 50, 70, 64, 56]
}

df = pd.DataFrame(data)
print(df)
print(df.corr().iloc[0,1])

print(df['키'].corr(df['몸무게']))
print(df['몸무게'].corr(df['키']))

print(stats.pearsonr(df['몸무게'], df['키']))
print(stats.spearmanr(df['몸무게'], df['키']))
print(stats.kendalltau(df['몸무게'], df['키']))

### 단순 선형 회귀 분석
data = {
    '키': [150, 160, 170, 175, 165, 155, 172, 168, 174, 158,
          162, 173, 156, 159, 167, 163, 171, 169, 176, 161],
    '몸무게': [42, 50, 70, 64, 56, 48, 68, 60, 65, 52,
            54, 67, 49, 51, 58, 55, 69, 61, 66, 53]
}
df = pd.DataFrame(data)