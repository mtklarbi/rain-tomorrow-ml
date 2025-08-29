import pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay


X_test = pd.read_parquet("data/X_test.parquet").reset_index(drop=True)
y_test = pd.read_csv("data/y_test.csv").squeeze("columns").reset_index(drop=True)

# Drop rows with NaNs
mask = X_test.notna().all(axis=1)
X_test = X_test[mask]
y_test = y_test[mask]

def report(name, model):
    p = model.predict(X_test)
    proba = getattr(model, "predict_proba", lambda X: None)(X_test)
    auc = roc_auc_score(y_test, proba[:,1]) if proba is not None else float("nan")
    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_test, p))
    print("Precision:", precision_score(y_test, p, zero_division=0))
    print("Recall   :", recall_score(y_test, p))
    print("ROC AUC  :", auc)

    if proba is not None:
        RocCurveDisplay.from_predictions(y_test, proba[:,1])
        plt.title(f"ROC — {name}")
        plt.show()



for name, path in [("Logistic", "models/logit.joblib"), ("RandomForest", "models/rf.joblib")]:
    report(name, joblib.load(path))