import pandas as pd
df = pd.read_csv("./gichul\gichul7\clam.csv")

print("shape:",df.shape)
train = df.iloc[:210]
test = df.iloc[210:]
print("shape:",train.shape,test.shape)
print("train:",train.head())
print("test:",test.head())

from statsmodels.formula.api import ols
from statsmodels.formula.api import logit
import numpy as np

formula = 'gender ~ weight'
model = logit(formula, data=train).fit()

odds_ratio = np.exp(model.params['weight'])
print("odds_ratio:",round(odds_ratio,4))

formula = 'gender ~ age + length + diameter + height + weight'
model = logit(formula, data=train).fit()
res_devian = -2 * model.llf
print(round(res_devian,2))

from sklearn.metrics import accuracy_score

model = logit("gender ~ weight", data=train).fit()
target = test.pop('gender')
pred = model.predict(test)
pred = (pred > 0.5).astype(int)

accuracy = accuracy_score(target, pred)

error_rate = 1-accuracy
print(error_rate)