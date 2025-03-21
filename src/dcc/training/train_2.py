from train_1 import load_data, evaluate

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

    dummy_cols = train_x.columns[train_x.columns.str.contains("DV")].tolist()

    # pay duly
    dummy_cols_pay_due = [col for col in dummy_cols if "-1" in col]

    # pay delay
    dummy_cols_pay_delay = [col for col in dummy_cols if ("PAY" in col) and ("-1" not in col)]

    # edu, marriage, sex
    dummy_cols_others = [col for col in dummy_cols if col in ["EDUCATION", "MARRIAGE", "SEX"]]

    x_num_cols = [col for col in x_cols if col not in dummy_cols]

    x_cols_train = dummy_cols_pay_delay

    # Train model with delay info
    with Live(dir="dvc/train-delay-only") as live:
        model = LogisticRegression()
        model.fit(train_x[dummy_cols_pay_delay], train_y)

    # Test model with delay info
    with Live(dir="dvc/test-delay-only") as live:
        y_pred = model.predict(test_x[dummy_cols_pay_delay])
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

    # Train model with dummy variables * num cols (create new dataset)
    dummy_train_x = pd.concat(
        [
            train_x[x_num_cols].mul(train_x[d_col], axis=0).add_prefix(f"{d_col}_")
            for d_col in dummy_cols_pay_delay
        ],
        axis=1,
    )

    dummy_test_x = pd.concat(
        [
            test_x[x_num_cols].mul(test_x[d_col], axis=0).add_prefix(f"{d_col}_")
            for d_col in dummy_cols_pay_delay
        ],
        axis=1,
    )

    # Concatenate with other dummy variables
    dummy_train_x = pd.concat([dummy_train_x, train_x[dummy_cols_others]], axis=1)
    dummy_test_x = pd.concat([dummy_test_x, test_x[dummy_cols_others]], axis=1)

    with Live(dir="dvc/train-delay-num") as live:
        model = LogisticRegression()
        model.fit(dummy_train_x, train_y)

    # Test model with dummy variables * num cols
    with Live(dir="dvc/test-delay-num") as live:
        y_pred = model.predict(dummy_test_x)
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
