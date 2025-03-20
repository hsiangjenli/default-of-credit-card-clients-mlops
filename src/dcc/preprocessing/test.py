import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv("data/raw/UCI_Credit_Card.csv")
df_y = df["default.payment.next.month"]
df = df.drop(columns=["ID", "default.payment.next.month"])  # Drop target column

# Category columns
pay_status_cols = [
    col for col in df.columns if col.startswith("PAY_") and not col.startswith("PAY_AMT")
]
cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
cat_cols.extend(pay_status_cols)

print(f"Category columns: {cat_cols}")

# Numerical columns
num_cols = [col for col in df.columns if col not in cat_cols]

# Double check if all columns' types are correct
df[num_cols] = df[num_cols].astype("float64")
df[cat_cols] = df[cat_cols].astype("category")

# Describe numerical columns
num_describe = df[num_cols].describe()
num_describe.to_csv("outputs/numerical_describe.csv")

print(f"Numerical columns: {num_cols}")

# Visualize categorical columns
for col in cat_cols:
    df[col].value_counts().sort_index().to_frame().plot(kind="bar")
    plt.savefig(f"outputs/images/{col}_countplot.png")

# Visualize numerical columns
for col in num_cols:  # 選三個關鍵變數
    plt.figure(figsize=(12, 6))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(col)
    plt.savefig(f"outputs/images/{col}_histplot.png")
    plt.close()

# Standardize numerical columns
ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])

# Transform categorical into dummies columns
# For later use in the model
df_cat_dummies = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Combine numerical and dummies columns
df = pd.concat([df[num_cols], df_cat_dummies], axis=1)

# Combine preprocessed data with target column
df = pd.concat([df, df_y], axis=1)

# Save preprocessed data
df.to_csv("data/processed/UCI_Credit_Card.csv", index=False)
