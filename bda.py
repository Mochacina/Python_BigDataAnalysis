import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. 임의의 데이터 프레임 2개 생성 (학습용, 테스트용)
# 학습용 데이터프레임 생성
train_data = {
    'A': np.random.randint(0, 100, 50),
    'B': np.random.rand(50) * 100,
    'C': np.random.choice(['P', 'Q', 'R', 'S'], 50),
    'D': np.random.randn(50) * 50
}
train_df = pd.DataFrame(train_data)

# 테스트용 데이터프레임 생성
test_data = {
    'A': np.random.randint(0, 100, 20),
    'B': np.random.rand(20) * 100,
    'C': np.random.choice(['P', 'Q', 'R', 'S'], 20),
    'D': np.random.randn(20) * 50
}
test_df = pd.DataFrame(test_data)

print("학습용 데이터프레임 샘플:")
print(train_df.head())
print("\n테스트용 데이터프레임 샘플:")
print(test_df.head())

# 2. 탐색적 데이터 분석 (EDA)
print("\n--- 학습용 데이터프레임 탐색적 데이터 분석 ---")

# 데이터프레임 정보
print("\n[학습용 DF 정보]")
train_df.info()

# 기술 통계량
print("\n[학습용 DF 기술 통계량]")
print(train_df.describe(include='all'))

# 결측치 확인
print("\n[학습용 DF 결측치 확인]")
print(train_df.isnull().sum())

# 각 범주형 데이터의 빈도수
print("\n[학습용 DF 'C' 컬럼 값 빈도수]")
print(train_df['C'].value_counts())


print("\n--- 테스트용 데이터프레임 탐색적 데이터 분석 ---")

# 데이터프레임 정보
print("\n[테스트용 DF 정보]")
test_df.info()

# 기술 통계량
print("\n[테스트용 DF 기술 통계량]")
print(test_df.describe(include='all'))

# 결측치 확인
print("\n[테스트용 DF 결측치 확인]")
print(test_df.isnull().sum())

# 각 범주형 데이터의 빈도수
print("\n[테스트용 DF 'C' 컬럼 값 빈도수]")
print(test_df['C'].value_counts())

# 3. 데이터 전처리
# 특성과 타겟 분리
X = train_df.drop('C', axis=1)
y = train_df['C']

# 수치형 특성 정의
numeric_features = ['A', 'B', 'D']

# 전처리 파이프라인 설정
numeric_transformer = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# 4. 머신러닝 모델 학습 및 평가
# 모델 정의
models = {
    '로지스틱 회귀': LogisticRegression(random_state=42),
    '랜덤 포레스트': RandomForestClassifier(random_state=42),
    '그래디언트 부스팅': GradientBoostingClassifier(random_state=42)
}

# 각 모델의 성능 평가
results = {}
print("\n--- 모델 성능 평가 ---")
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name} 정확도: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 5. 최적 모델 선정 및 결과 제출
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n최적 모델: {best_model_name} (정확도: {results[best_model_name]:.4f})")

# 전체 학습 데이터로 최적 모델 재학습
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', best_model)])
final_pipeline.fit(X, y)

# 테스트 데이터 예측
X_test = test_df.drop('C', axis=1)
predictions = final_pipeline.predict(X_test)

# 제출 파일 생성
submission = pd.DataFrame({'ID': X_test.index, 'C_predicted': predictions})
submission.to_csv('submission.csv', index=False)

print("\n제출 파일 'submission.csv'이(가) 성공적으로 생성되었습니다.")