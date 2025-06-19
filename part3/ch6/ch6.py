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

# 6 적합도 검정

observed = [550, 250, 100, 70, 30]
s = sum(observed)
expected = [s*i for i in [0.6, 0.25, 0.08, 0.05, 0.02]]
print(expected)
print(stats.chisquare(observed, expected))

# 7 독립성 검정
observed = pd.DataFrame([[50,30],[60,40]])
print(observed)

data = {
    '캠프': ['빅분기']*80 + ['정처기']*100,
    '등록여부': ['등록함']*50 + ['등록안함']*30 + ['등록함']*60 + ['등록안함']*40
}
df = pd.DataFrame(data)
df2 = pd.crosstab(df['캠프'], df['등록여부'])

print("chisq: \n",stats.chi2_contingency(observed))
print(df2)

print("chisq(df2): \n",stats.chi2_contingency(df2))

# 8. 다중 선형 회귀
# 데이터
import pandas as pd
df = pd.DataFrame({
    '할인율': [28, 24, 13, 0, 27, 30, 10, 16, 6, 5, 7, 11, 11, 30, 25,
            4, 7, 24, 19, 21, 6, 10, 26, 13, 15, 6, 12, 6, 20, 2],
    '온도': [15, 34, 15, 22, 29, 30, 14, 17, 28, 29, 19, 19, 34, 10,
           29, 28, 12, 25, 32, 28, 22, 16, 30, 11, 16, 18, 16, 33, 12, 22],
    '광고비': [342, 666, 224, 764, 148, 499, 711, 596, 797, 484, 986, 347, 146, 362, 642,
            591, 846, 260, 560, 941, 469, 309, 730, 305, 892, 147, 887, 526, 525, 884],
    '주문량': [635, 958, 525, 25, 607, 872, 858, 732, 1082, 863, 904, 686, 699, 615, 893,
            830, 856, 679, 918, 951, 789, 583, 988, 631, 866, 549, 910, 946, 647, 943]
})
print(df.head(3))