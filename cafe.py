import pandas as pd
import numpy as np

# CSV 파일 읽기
csv_filename = 'cafe_menu.csv'
try:
    # UTF-8-SIG 인코딩으로 CSV 파일 읽기 시도
    df_loaded = pd.read_csv(csv_filename, encoding='utf-8-sig')
    print(f"'{csv_filename}' 파일을 성공적으로 불러왔습니다.")
    print("\n--- 불러온 데이터프레임 ---")
    print(df_loaded)
except FileNotFoundError:
    print(f"오류: '{csv_filename}' 파일을 찾을 수 없습니다. 먼저 cafe_menu.py를 실행하여 파일을 생성하세요.")
except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")

print(df_loaded.head(3))

df_loaded['new'] = 0

df_loaded.drop(1, axis=0, inplace=True)
print(df_loaded)

df_loaded.drop('new', axis=1, inplace=True)
print(df_loaded)

print("loc[0:3]")
print(df_loaded.loc[0:3])

print("iloc[0:3]")
print(df_loaded.iloc[:,:])

df_loaded['원산지'] = np.nan
print(df_loaded)

df_loaded.loc['시즌'] = {'메뉴':'닌자버블티', '가격':10000, '칼로리':100}
print(df_loaded)

#print(df_loaded.sort_index(ascending=False))
df_loaded.sort_values(['가격','칼로리'], ascending=(True,True), inplace=True)
df_loaded.reset_index(drop=True,inplace=True)
print(df_loaded)
