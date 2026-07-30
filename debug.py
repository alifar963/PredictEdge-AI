import pandas as pd
df = pd.read_csv("data2/features.csv", sep=";")
print(df.groupby("label").mean(numeric_only=True))

 