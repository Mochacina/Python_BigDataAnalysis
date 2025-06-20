import pandas as pd
train = pd.read_csv("./gichul\gichul9\\farm_train.csv")
test = pd.read_csv("./gichul\gichul9\\farm_test.csv")

print("shape:", train.shape, test.shape)
print("train head:\n", train.head())
print("train head:\n", test.head())

print("train info:"); train.info()
print("test info:"); test.info()

target = train.pop('농약검출여부')

train = pd.get_dummies(train)
test = pd.get_dummies(test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb

x_tr, x_val, y_tr, y_val = train_test_split(train, target, test_size=0.2, random_state=0)

rf = RandomForestClassifier(random_state=0)
rf.fit(x_tr,y_tr)
pred = rf.predict(x_val)

from sklearn.metrics import f1_score
print("RF f1 macro:", f1_score(y_val, pred, average='macro')) # RF f1 macro: 0.8532014300116062

lg = lgb.LGBMClassifier(random_state=0)
lg.fit(x_tr, y_tr)
pred = lg.predict(x_val)
print("lightgbm f1 macro:", f1_score(y_val, pred, average='macro')) # lightgbm f1 macro: 0.9100316620356779

pred = lg.predict(test)
result = pd.DataFrame({
    'pred':pred
})
result.to_csv("./gichul\gichul9\\result.csv", index=False)

print("shape:",test.shape, pred.shape)