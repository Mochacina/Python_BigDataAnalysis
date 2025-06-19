import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

train = pd.read_csv("./gichul\gichul7\mart_train.csv")
test = pd.read_csv("./gichul\gichul7\mart_test.csv")

print("shape:",train.shape, test.shape)
print("train head:\n",train.head())
print("test head:\n",train.head())

train.info()
test.info()

popped = train.pop('total')
print("shape:",train.shape, test.shape)

train = pd.get_dummies(train)
test = pd.get_dummies(test)
print("shape:",train.shape, test.shape)

x_tr, x_val, y_tr, y_val = train_test_split(train, popped, test_size=0.2, random_state=0)

# RMSE는 0에 가까울수록 성능이 좋음
from sklearn.metrics import root_mean_squared_error
lr = LinearRegression()
lr.fit(x_tr,y_tr)
pred = lr.predict(x_val)
rmse = root_mean_squared_error(y_val,pred)
print("rmse(lr):",rmse) #rmse(lr): 374016.3661648522

rf = RandomForestRegressor(random_state=0)
rf.fit(x_tr,y_tr)
pred = rf.predict(x_val)
rmse = root_mean_squared_error(y_val,pred)
print("rmse(rf):",rmse) #rmse(rf): 385935.56337360526

xg = xgb.XGBRegressor(random_state=0)
xg.fit(x_tr,y_tr)
pred = xg.predict(x_val)
rmse = root_mean_squared_error(y_val, pred)
print("rmse(xg):",rmse) #rmse(xg): 442570.7070483747

lg = lgb.LGBMRegressor(random_state=0)
lg.fit(x_tr,y_tr)
pred = lg.predict(x_val)
rmse = root_mean_squared_error(y_val, pred)
print("rmse(lg):",rmse) #rmse(lg): 404369.88780394377

pred = lr.predict(test)
result = pd.DataFrame({
    'pred': pred
})
print("pred:\n",result)
result.to_csv("./gichul/gichul7/result.csv", index=False)

# 레이블 인코딩 적용
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv("./gichul\gichul7\mart_train.csv")
test = pd.read_csv("./gichul\gichul7\mart_test.csv")

pop = train.pop('total')

cols = train.select_dtypes(include=object).columns
print(cols)
for col in cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])

x_tr, x_val, y_tr, y_val = train_test_split(train, popped, test_size=0.2, random_state=0)

from sklearn.metrics import root_mean_squared_error
lr = LinearRegression()
lr.fit(x_tr,y_tr)
pred = lr.predict(x_val)
rmse = root_mean_squared_error(y_val,pred)
print("rmse(lr):",rmse) #rmse(lr): 374016.3661648522

pred = lr.predict(test)
result = pd.DataFrame({
    'pred':pred
})
print(result)
result.to_csv("./gichul/gichul7/result.csv", index=False)