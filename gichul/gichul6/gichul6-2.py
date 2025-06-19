import pandas as pd
from scipy import stats

train = pd.read_csv("./gichul\gichul6\energy_train.csv")
test = pd.read_csv("./gichul\gichul6\energy_test.csv")

# EDA
print("train.head():\n",train.head(1))
print("test.head():\n",test.head(1))

print("shape:",train.shape, test.shape)

print("info:\n")
train.info(); test.info()

print("describe: \n",train.describe(include='O'))

# 결측치 확인
print("결측치 확인:", train.isnull().sum().sum(), test.isnull().sum().sum())

print(train['Heat_Load'].value_counts())

trainHeatLoad = train.pop('Heat_Load')

train = pd.get_dummies(train)
test = pd.get_dummies(test)

# 원-핫 인코딩 후 확인
print("train.head():\n",train.head(1))
print("test.head():\n",test.head(1))

print("shape:",train.shape, test.shape)

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(train, trainHeatLoad, test_size=0.2, random_state=0)
print("splited data shape:", x_tr.shape, x_val.shape, y_tr.shape, y_val.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score

dt = DecisionTreeClassifier(random_state=0)
dt.fit(x_tr, y_tr)
pred = dt.predict(x_val)

print("f1_score(dt):", f1_score(y_val, pred, average='macro')) # f1_score(dt): 0.9167995817564094

rf = RandomForestClassifier(random_state=0)
rf.fit(x_tr, y_tr)
pred = rf.predict(x_val)
print("f1_score(rf):", f1_score(y_val, pred, average='macro')) # f1_score(rf): 0.9277616846430405

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_tr_adj = le.fit_transform(y_tr)

xg = xgb.XGBClassifier(random_state=0)
xg.fit(x_tr, y_tr_adj)
pred = xg.predict(x_val)
pred = le.inverse_transform(pred)

print("f1_score(xg):", f1_score(y_val, pred, average='macro')) # f1_score(xg): 0.9374839068652628

lg = lgb.LGBMClassifier(random_state=0)
lg.fit(x_tr,y_tr)
pred = lg.predict(x_val)

print("f1_score(lg):", f1_score(y_val, pred, average='macro')) # f1_score(lg): 0.9319703995747778

pred = lg.predict(test)
submit = pd.DataFrame({
    'pred':pred
})

print(submit)

submit.to_csv("result.csv", index=False)