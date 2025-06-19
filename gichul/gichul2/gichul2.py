import pandas as pd
from scipy import stats

X_test = pd.read_csv(".\\gichul\\gichul2\X_test.csv")
X_train = pd.read_csv(".\\gichul\\gichul2\X_train.csv")
y_train = pd.read_csv(".\\gichul\\gichul2\y_train.csv")

print(X_test.shape, X_train.shape, y_train.shape)

# 데이터 샘플 확인
print(X_train.head())

# 데이터 샘플 확인
print(y_train.head())

# 데이터 타입 확인
X_train.info()

# 기초 통계 획인
print(X_train.describe())

# 기초 통계 획인 object
print(X_train.describe(include='O')) # 또는 include='object'

# 기초 통계 획인 object
print(X_test.describe(include='O'))

# 결측치 확인 test
X_test.isnull().sum().sum()

# 타겟(레이블)확인 0:정시 도착, 1:정시 도착하지 않음
y_train['Reached.on.Time_Y.N'].value_counts()