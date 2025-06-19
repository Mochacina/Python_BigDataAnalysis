import pandas as pd
from scipy import stats

train = pd.read_csv("./gichul\gichul6\energy_test.csv")
test = pd.read_csv("./gichul\gichul6\energy_train.csv")

# EDA
print("train.head():\n",train.head(1))
print("test.head():\n",test.head(1))

print("shape:",train.shape, test.shape)