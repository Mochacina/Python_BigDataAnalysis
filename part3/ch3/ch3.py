import pandas as pd
from scipy import stats

### 적합도 검정
observed = [150, 120, 30]
expected = [0.5*300, 0.35*300, 0.15*300]
print("chisquare results: \n", stats.chisquare(observed, expected))

### 독립성 검정 - 교차표 데이터

df = pd.DataFrame({'좋아함': [80, 90],
                   '좋아하지 않음': [30, 10]},
                  index=['남자', '여자'])
print(df)

print(stats.chi2_contingency(df))

## 로우 데이터

data = {
    '성별': ['남자']*110 + ['여자']*100,
    '운동': ['좋아함']*80 + ['좋아하지 않음']*30 + ['좋아함']*90 + ['좋아하지 않음']*10
}
df = pd.DataFrame(data)
print(df)

df = pd.crosstab(df['성별'], df['운동'])

print(stats.chi2_contingency(df))

### 동질성 검정
df = pd.DataFrame([[50,50],[30,70]])
print(stats.chi2_contingency(df))

data = {
    '학과': ['통계학과']*100 + ['컴퓨터공학과']*100,
    '동아리가입여부': ['가입']*50 + ['미가입']*50 + ['가입']*30 + ['미가입']*70
}
df = pd.DataFrame(data)
print(df)
print(df.sample(5))