import pandas as pd
df = pd.read_csv("gichul\gichul7\system_cpu.csv")
print("head:",df.head())
print("shape:",df.shape)

corr_matrix = df.corr()
print(corr_matrix)

erp_corr = corr_matrix['ERP'].sort_values(ascending=False)
print(erp_corr)

from statsmodels.formula.api import ols
df_2 = df[df['CPU'] < 100]
print(df_2)

formula = 'ERP ~ Feature1 + Feature2 + Feature3 + CPU'
model = ols(formula, data=df_2).fit()
print(model.summary())