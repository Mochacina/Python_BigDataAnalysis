import pandas as pd
from scipy import stats

df = pd.read_csv("./part3\ch6\\tomato2.csv")

print("sample: \n", df.sample(10))
print("info: \n"); df.info()

import statsmodels.api as sm
from statsmodels.formula.api import ols

# 1~9 이원 분산 분석
model = ols('수확량 ~ C(비료유형) * C(물주기)', data=df).fit()
anova_t = sm.stats.anova_lm(model)
print("anova_t: \n", anova_t)

