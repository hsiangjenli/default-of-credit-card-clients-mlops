import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def load_data():
    train_x = pd.read_csv("data/processed/train_x.csv")
    train_y = pd.read_csv("data/processed/train_y.csv")
    test_x = pd.read_csv("data/processed/test_x.csv")
    test_y = pd.read_csv("data/processed/test_y.csv")
    return train_x, train_y, test_x, test_y


def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return accuracy, f1, recall, precision


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from dvclive import Live
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd

    train_x, train_y, test_x, test_y = load_data()

    y_col = "default.payment.next.month"
    x_cols = train_x.columns

    # Train model
    with Live(dir="dvc/train") as live:
        model = LogisticRegression()
        model.fit(train_x, train_y)

    # Test model
    with Live(dir="dvc/test") as live:
        y_pred = model.predict(test_x)
        accuracy, f1, recall, precision = evaluate(test_y, y_pred)

        live.log_metric("accuracy", accuracy)
        live.log_metric("f1", f1)
        live.log_metric("recall", recall)
        live.log_metric("precision", precision)

        # Generate confusion matrix
        cm = confusion_matrix(y_true=test_y, y_pred=y_pred)
        fig = plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False, square=True)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        live.log_image("confusion_matrix.png", fig)