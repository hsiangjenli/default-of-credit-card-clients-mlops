from sklearn.model_selection import train_test_split

def splitter(df, target_col):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(x, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/processed/UCI_Credit_Card.csv")

    train_x, test_x, train_y, test_y = splitter(df, "default.payment.next.month")
    
    train_x.to_csv("data/processed/train_x.csv", index=False)
    test_x.to_csv("data/processed/test_x.csv", index=False)
    train_y.to_csv("data/processed/train_y.csv", index=False)
    test_y.to_csv("data/processed/test_y.csv", index=False)
