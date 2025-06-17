import pandas as pd
import scipy.stats as sct

df = pd.DataFrame({
    'before':[85, 90, 92, 88, 86, 89, 83, 87],
    'after':[86, 92, 94, 89, 84, 90, 84, 88]
})

df['diff'] = df['after'] - df['before']

print(df)
print(df.info())

# 샤피로 검정
print(sct.shapiro(df['diff']))
print(sct.wilcoxon(df['after'], df['before'], alternative='greater'))
print(sct.wilcoxon(df['diff'], alternative='greater'))

