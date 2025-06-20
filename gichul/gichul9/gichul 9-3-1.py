import pandas as pd

df = pd.read_csv("./gichul\gichul9\design.csv")

print("shape:",df.shape)
print("head:",df.head())

from statsmodels.formula.api import ols

train_cond = (df['id'] <= 140) & (df['id'] >= 1)
test_cond = (df['id'] > 140)

train = df[train_cond].copy()
test = df[test_cond].copy()

print("shape:",train.shape, test.shape)
print("head:",train.head(), test.head())

formula = 'design ~ c1 + c2 + c3 + c4 + c5'
model = ols(formula, data=train).fit()
print("model summray:\n", model.summary())

# p-values가 0.05보다 작은 독립변수의 수
print(sum(model.pvalues[1:] < 0.05))

formula = 'design ~ c1 + c2 + c4'
model = ols(formula, data=train).fit()
train['pred_design'] = model.predict(train)
print(train)

result = train['design'].corr(train['pred_design'], method='pearson')

# design과 예측값을 비교한 피어슨 상관계수
print(round(result,3))

test['pred_design'] = model.predict(test)

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(test['design'], test['pred_design'])
print("rmse:",round(rmse,3))