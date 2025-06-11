import pandas as pd

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

df_loaded.drop(1, axis=0, inplace=True)
print(df_loaded)