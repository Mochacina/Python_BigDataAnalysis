import pandas as pd
from scipy import stats
df = pd.DataFrame({
    "항암약":[4,4,3,4,1,4,1,4,1,4,4,2,1,4,2,3,2,4,4,4]
    })
print(df.head(3))

cnt_4 = sum(df['항암약']==4)
print("cnt_4:",cnt_4)

ratio = cnt_4 / len(df['항암약'])
print("ratio:",ratio)
print("ratio:",df['항암약'].value_counts(normalize=True))

# 1) 각 카테고리의 비율을 리스트로 만들기
prob = [0.1, 0.05, 0.15, 0.7]

# 2-1) 기대 빈도수 계산
print("데이터 수: ", len(df))
expected_counts = [len(df)*x for x in prob]
print(expected_counts)

observed_count = df['항암약'].value_counts().sort_index().to_list()
print(observed_count)

print(stats.chisquare(observed_count, expected_counts))

df = pd.read_csv("./gichul\gichul6\data6-3-2.csv")
print(df.head(3))

from statsmodels.formula.api import ols

formula = 'temperature ~ solar + wind + o3'
model = ols(formula, data=df).fit()
summary = model.summary()
print(summary)

print("o3 회귀 계수:", model.params['o3'])
print("temperature, wind간 p-value:", model.pvalues['wind'])

pre_data = pd.DataFrame({
    'solar':[100],
    'wind':[5],
    'o3':[30]
})

pred = model.predict(pre_data)
print("예측값:", pred)