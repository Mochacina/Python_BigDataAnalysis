import pandas as pd

# 카페 메뉴 데이터 생성
menu_data = {
    '메뉴': ['아메리카노', '카페라떼', '카푸치노', '딸기 스무디', '망고 주스'],
    '가격': [4000, 4500, 4500, 5500, 5000],
    '칼로리': [10, 180, 150, 300, 250],
    '카페인(mg)': [150, 75, 75, 0, 0],
    '당류(g)': [0, 15, 12, 50, 45]
}

# DataFrame 생성
df_menu = pd.DataFrame(menu_data)

# DataFrame을 CSV 파일로 저장 (인덱스 제외)
csv_filename = 'cafe_menu.csv'
df_menu.to_csv(csv_filename, index=False, encoding='utf-8-sig')

print(f"카페 메뉴 데이터프레임이 '{csv_filename}'으로 저장되었습니다.")
print("\n--- 생성된 데이터프레임 ---")
print(df_menu)