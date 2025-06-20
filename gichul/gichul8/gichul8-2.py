import pandas as pd
test = pd.read_csv("./gichul\gichul8\churn_test.csv")
train = pd.read_csv(("./gichul\gichul8\churn_train.csv"))

print("train shape =", train.shape)
print("test shape =", test.shape)

print("train head =", train.head())
print("test head =", test.head())

print("train info:")
train.info()
print("test info:")
test.info()
print("target describe:", train['TotalCharges'].describe())

train = train.drop('customerID', axis=1)
test = test.drop('customerID', axis=1)
target = train.pop('TotalCharges')

print("=== LabelEncode ===")
from sklearn.preprocessing import LabelEncoder
cols = train.select_dtypes(include='O').columns
for col in cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])

print("train shape =", train.shape)
print("test shape =", test.shape)
print("train info:")
train.info()
print("test info:")
test.info()

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(train, target, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0)
rf.fit(x_tr,y_tr)
pred = rf.predict(x_val)

from sklearn.metrics import mean_absolute_error
print("mae=", mean_absolute_error(y_val, pred)) # mae= 951.0960435538027

pred =rf.predict(test)
result = pd.DataFrame({
    'pred': pred
})

result.to_csv("gichul\gichul8\\result.csv",index=False)
print("shape=", test.shape, pred.shape) # shape= (1764, 17) (1764,)