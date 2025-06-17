import pandas as pd
import scipy.stats as sct
from scipy import stats

# 단일 표본 검정
import pandas as pd
df = pd.DataFrame({
    'weights':[122, 121, 120, 119, 125, 115, 121, 118, 117, 127,
           123, 129, 119, 124, 114, 126, 122, 124, 121, 116,
           120, 123, 127, 118, 122, 117, 124, 125, 123, 121],
})

# 대응 표본 검정

df = pd.DataFrame({
    'before':[85, 90, 92, 88, 86, 89, 83, 87],
    'after':[86, 92, 94, 89, 84, 90, 84, 88]
})

df['diff'] = df['after'] - df['before']

print(df)
print(df.info())

### 샤피로 검정
print(sct.shapiro(df['diff']))
print(sct.wilcoxon(df['after'], df['before'], alternative='greater'))
print(sct.wilcoxon(df['diff'], alternative='greater'))

print("------")

# 독립 표본 검정
class1 = [85, 90, 92, 88, 86, 89, 83, 87]
class2 = [80, 82, 88, 85, 84]

print(sum(class1)/len(class1), sum(class2)/len(class2))
print(sct.ttest_ind(class1, class2))

print(sct.ttest_ind(class1, class2, equal_var=True, alternative='less'))
print(sct.ttest_ind(class1, class2, equal_var=True, alternative='greater'))

print("------")

class1 = [85, 90, 92, 88, 86, 89, 83, 87]
class2 = [80, 82, 88, 85, 84]

print(stats.shapiro(class1))
print(stats.shapiro(class2))

print(stats.levene(class1, class2))
print(stats.ttest_ind(class1, class2, alternative='less', equal_var=True))

print("------")

class1 = [85, 90, 92, 88, 86, 89, 83, 87]
class2 = [80, 82, 88, 85, 130]

from scipy import stats
print(stats.shapiro(class1))
print(stats.shapiro(class2))

stats.mannwhitneyu(class1, class2, alternative='less')

print("------")

# Ch 2 분산 분석

