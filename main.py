import pandas as pd

# 간단한 데이터 생성
data = {'col1': [1, 2], 'col2': [3, 4]}

# DataFrame 생성
df = pd.DataFrame(data)

df['new'] = 0
df['new2'] = df['new'] + 1

# DataFrame 출력
print(df)