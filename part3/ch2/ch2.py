import pandas as pd
from scipy import stats

df = pd.DataFrame({
    'A': [10.5, 11.3, 10.8, 9.6, 11.1, 10.2, 10.9, 11.4, 10.5, 10.3],
    'B': [11.9, 12.4, 12.1, 13.2, 12.5, 11.8, 12.2, 12.9, 12.4, 12.3],
    'C': [11.2, 11.7, 11.6, 10.9, 11.3, 11.1, 10.8, 11.5, 11.4, 11.0],
    'D': [9.8, 9.4, 9.1, 9.5, 9.6, 9.9, 9.2, 9.7, 9.3, 9.4]
})
print(df.head(2))

print(stats.shapiro(df['A']))
print(stats.shapiro(df['B']))
print(stats.shapiro(df['C']))
print(stats.shapiro(df['D']))

print("\n === 등분산 검정 ===")
print(stats.levene(df['A'], df['B'], df['C'], df['D']))

print("\n === 일원 분산 분석 ===")
print(stats.f_oneway(df['A'], df['B'], df['C'], df['D']))

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm

df = pd.read_csv("./part3/ch2/fertilizer.csv")
#print(df)

model = ols('성장 ~ C(비료)', df).fit()
print(anova_lm(model))

df = pd.read_csv("./part3/ch2/tree.csv")
print(df.sample(10))

model = ols('성장률 ~ C(나무) + C(비료) + C(나무):C(비료)', data=df).fit()
anova_table = sm.stats.anova_lm(model)
print(anova_table)


