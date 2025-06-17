import pandas as pd
from scipy import stats

### 카이제곱 검정
observed = [150, 120, 30]
expected = [0.5*300, 0.35*300, 0.15*300]
print("chisquare results: \n", stats.chisquare(observed, expected))

df = pd.DataFrame({'좋아함': [80, 90],
                   '좋아하지 않음': [30, 10]},
                  index=['남자', '여자'])
print(df)