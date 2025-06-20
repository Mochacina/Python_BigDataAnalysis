import pandas as pd
df = pd.read_csv("./gichul\gichul8\churn.csv")

print("shape=", df.shape)
print("head=\n", df.head())

from statsmodels.formula.api import logit

# 8-3-1
formula = 'Churn ~ AccountWeeks + ContractRenewal + DataPlan + DataUsage + CustServCalls + DayMins + DayCalls + MonthlyCharge + OverageFee + RoamMins'

model = logit(formula, data=df).fit()
print(model.summary())
print("sum=",sum(model.pvalues[1:]>0.05))

# 8-3-2
formula = 'Churn ~ DataUsage + DayMins'
model = logit(formula, data=df).fit()
print(model.summary())
print(round(sum(model.params),3))

# 8-3-3
import numpy as np
coef = model.params['DataUsage']
odds_ratio = round(np.exp(coef*5),3)
print(odds_ratio)