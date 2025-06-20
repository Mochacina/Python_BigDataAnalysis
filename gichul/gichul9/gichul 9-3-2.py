import pandas as pd
from statsmodels.formula.api import logit

df = pd.read_csv("./gichul\gichul9\\retention.csv")

print("head:", df.head())
print("shape:", df.shape)

formula = 'Churn ~ MonthlyCharges + CustomerTenure + HasPhoneService + HasTechInsurance'
model = logit(formula, data=df).fit()
#print(model.summary())
print("MonthlyCharges pvalue:",round(model.pvalues['MonthlyCharges'],3))

import numpy as np
odds_ratio = np.exp(model.params['HasPhoneService'])
print("Odds Ratio:",round(odds_ratio,3))

result = model.predict(df)

re_sum = 0
for i in result:
    if i > 0.3: re_sum += 1
print("sum:",re_sum)