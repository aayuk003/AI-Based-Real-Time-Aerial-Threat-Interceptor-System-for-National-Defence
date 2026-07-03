import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier
from xgboost import plot_importance


# =====================
# LOAD DATA
# =====================

df = pd.read_csv("data/dummy_radar_dataset.csv")

print("Dataset loaded")
print(df.shape)


# =====================
# CLEAN DATA
# =====================

drop_cols = [
    "object_id",
    "timestamp",
    "trajectory_type",
    "iff_response",
    "threat_level"
]

df = df.drop(columns=drop_cols)

# Optional noisy feature removal
df = df.drop(columns=["distance_from_border_km"])

print("\nRemaining features:")
print(df.columns)


# =====================
# ENCODE TARGET
# =====================

le = LabelEncoder()

df["true_class"] = le.fit_transform(
    df["true_class"]
)

X = df.drop("true_class", axis=1)
y = df["true_class"]


# =====================
# TRAIN TEST SPLIT
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =====================
# XGBOOST + TUNING
# =====================

xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=len(le.classes_),
    random_state=42
)

param_grid = {
    "max_depth":[3,4,5,6],
    "n_estimators":[100,150,200,300],
    "learning_rate":[0.01,0.05,0.1,0.2],
    "subsample":[0.8,1.0],
    "colsample_bytree":[0.8,1.0]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=15,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest CV Score:")
print(search.best_score_)


# =====================
# EVALUATION
# =====================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nTest Accuracy: {accuracy:.4f}")

report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
)

print(report)


# =====================
# SAVE REPORT
# =====================

with open(
    "results/classification_report.txt",
    "w"
) as f:
    f.write(report)


# =====================
# CONFUSION MATRIX
# =====================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(10,7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png"
)

plt.close()


# =====================
# FEATURE IMPORTANCE
# =====================

plt.figure(figsize=(10,6))

plot_importance(
    best_model,
    max_num_features=10
)

plt.title(
    "Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png"
)

plt.close()


# =====================
# SAVE MODEL
# =====================

with open(
    "models/xgb_model.pkl",
    "wb"
) as f:
    pickle.dump(
        best_model,
        f
    )

print("\nModel saved")
print("Results saved")
print("Training complete")