import pandas as pd
df = pd.read_csv("./gichul\gichul8\piq.csv")

print("shape:",df.shape)
print("head:\n",df.head())

# 1번 문제
from statsmodels.formula.api import ols
formula = 'PIQ ~ Brain + Height + Weight'
model = ols(formula, data=df).fit()

print(model.summary())
print("회귀 계수:",round(model.params['Brain'],3))

# 2번 문제
print("rsquared:",round(model.rsquared,2))

nd = pd.DataFrame({
    'Brain': [90],
    'Height': [70],
    'Weight': [150]
})

pred = model.predict(nd)
print("pred:\n",round(pred[0]))