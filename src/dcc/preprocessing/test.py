import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/raw/UCI_Credit_Card.csv')

# Category columns
cat_cols = ["SEX", "EDUCATION", "MARRIAGE", "default.payment.next.month"]

# Numerical columns
num_cols = [col for col in df.columns if col not in cat_cols]


for col in cat_cols:
    
    df[col].value_counts().to_frame().plot(kind="bar")
    plt.savefig(f"outputs/images/{col}_countplot.png")