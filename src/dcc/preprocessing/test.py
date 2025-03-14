import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/raw/UCI_Credit_Card.csv")
df = df.drop(columns=["default.payment.next.month"])  # Drop target column

# Category columns
pay_status_cols = [
    col for col in df.columns if col.startswith("PAY_") and not col.startswith("PAY_AMT")
]
cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]

cat_cols.extend(pay_status_cols)

# Numerical columns
num_cols = [col for col in df.columns if col not in cat_cols]

# Visualize categorical columns
for col in cat_cols:
    df[col].value_counts().sort_index().to_frame().plot(kind="bar")
    plt.savefig(f"outputs/images/{col}_countplot.png")

# Standardize numerical columns
ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])

# Transform categorical into dummies columns
# For later use in the model
df_cat_dummies = pd.get_dummies(df, columns=cat_cols)

# Combine numerical and dummies columns
df = pd.concat([df[num_cols], df_cat_dummies], axis=1)

# Save preprocessed data
df.to_csv("data/processed/UCI_Credit_Card.csv", index=False)
