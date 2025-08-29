import pandas as pd, joblib, pathlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X_train = pd.read_parquet("data/X_train.parquet")
y_train = pd.read_csv("data/y_train.csv").squeeze("columns")


# 1) Baseline: Logistic Regression
logit = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
logit.fit(X_train, y_train)
pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump(logit, "models/logit.joblib")

# 2) RandomForest (strong baseline without scaling)
rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=7, n_jobs=-1)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/rf.joblib")

print("Saved models: models/logit.joblib, models/rf.joblib")