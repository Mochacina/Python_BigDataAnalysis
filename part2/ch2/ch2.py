import pandas as pd

from sklearn.model_selection import train_test_split

train = pd.read_csv('https://raw.githubusercontent.com/lovedlim/bigdata_analyst_cert/main/part2/ch2/train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/lovedlim/bigdata_analyst_cert/main/part2/ch2/test.csv')

y_train = train.pop("income")

X_train, X_val, y_train, y_val = train_test_split(train,
                                                  y_train,
                                                  test_size=0.2,
                                                  random_state=0)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
pred=rf.predict_proba(X_val) # 각 레이블에 속할 확률 값 반환
pred[:10]