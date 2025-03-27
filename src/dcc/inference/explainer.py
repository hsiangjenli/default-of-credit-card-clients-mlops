import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

x_train = pd.read_csv("data/processed/train_x.csv")
x_test = pd.read_csv("data/processed/test_x.csv")


def load_data(prefix):
    x_col = joblib.load(f"dvc/{prefix}/x_cols.pkl")
    model = joblib.load(f"dvc/{prefix}/model.pkl")
    return x_col, model


# Train 1 explainer
x_col, model = load_data("train")
explainer = shap.Explainer(model, x_train[x_col], feature_names=x_col)
shap_values = explainer(x_test[x_col])

shap.plots.beeswarm(shap_values)
plt.savefig("data/processed/shap_train.png", bbox_inches="tight")
plt.close()

# Train 2 explainer
x_col, model = load_data("train-delay-only")
explainer = shap.Explainer(model, x_train[x_col], feature_names=x_col)
shap_values = explainer(x_test[x_col])

shap.plots.beeswarm(shap_values)
plt.savefig("data/processed/shap_train-delay-only.png", bbox_inches="tight")

# Train 3 explainer
# x_col, model = load_data("train-delay-num")
# explainer = shap.Explainer(model, x_train[x_col], feature_names=x_col)
# shap_values = explainer(x_test[x_col])
# print(shap_values)
