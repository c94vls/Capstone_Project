"""
Zepto Analytics Guild - Module 2: Part B (Predictive Modeling & Inference Pipeline)
Implements:
1. Stratified Split (justified by class imbalance)
2. Leakage-free ColumnTransformer and Pipelines (fit only on train)
3. Logistic Regression, Decision Tree (with plot_tree), and Random Forest
4. Class Imbalance Benchmark: Baseline vs Balanced vs SMOTE
5. Hyperparameter Tuning on RandomForest with OOB Score
6. Multivariate Linear Regression for Fare + Heteroscedasticity Analysis
7. End-to-end Pipeline serialization and verification via joblib
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE

# ----------------------------------------------------------------------
# 1. Load Cleaned Dataset & Verify Class Balance
# ----------------------------------------------------------------------
df = pd.read_csv("titanic.csv")

# Feature selection: isolate clean, un-engineered predictors
feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target_col = "survived"

X = df[feature_cols].copy()
y = df[target_col].copy()

balance = y.value_counts(normalize=True) * 100
print(f"Class Balance: Died (0) = {balance[0]:.2f}%, Survived (1) = {balance[1]:.2f}%")

# Task 7: Stratified Split
# Stratification ensures that the ~38.4% survival proportion is identically represented
# in both training and test partitions, preventing sample distribution skew.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Split sizes: Train = {X_train.shape[0]}, Test = {X_test.shape[0]}")

# ----------------------------------------------------------------------
# Task 8: Leakage-Free Preprocessing Pipeline
# ----------------------------------------------------------------------
numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ----------------------------------------------------------------------
# Task 9 & 10: Model Training & Comprehensive Evaluation
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

eval_results = {}
roc_data = {}

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    # Fit strictly on train split
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    eval_results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": auc,
        "Confusion Matrix": cm
    }
    roc_data[name] = (y_prob, roc_curve(y_test, y_prob))

# Decision Tree visualization
dt_fitted = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=3, random_state=42))
]).fit(X_train, y_train)

ohe_cols = dt_fitted.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_cols)

plt.figure(figsize=(16, 8))
plot_tree(
    dt_fitted.named_steps["classifier"],
    feature_names=all_feature_names,
    class_names=["Died", "Survived"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree Visualization (max_depth=3)")
plt.savefig("decision_tree_structure.png")
plt.close()

# ----------------------------------------------------------------------
# Task 11: Imbalance Handling Comparison
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 11: Imbalance Handling Comparison (Random Forest)")
print("=" * 80)

# (a) Baseline
rf_base = Pipeline([("preprocessor", preprocessor), ("clf", RandomForestClassifier(random_state=42))])
rf_base.fit(X_train, y_train)
p_base = precision_score(y_test, rf_base.predict(X_test))
r_base = recall_score(y_test, rf_base.predict(X_test))
f1_base = f1_score(y_test, rf_base.predict(X_test))

# (b) class_weight='balanced'
rf_bal = Pipeline([("preprocessor", preprocessor), ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))])
rf_bal.fit(X_train, y_train)
p_bal = precision_score(y_test, rf_bal.predict(X_test))
r_bal = recall_score(y_test, rf_bal.predict(X_test))
f1_bal = f1_score(y_test, rf_bal.predict(X_test))

# (c) SMOTE (Applied ONLY to training partition to avoid leakage)
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_trans, y_train)

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_res, y_train_res)
y_pred_smote = rf_smote.predict(X_test_trans)
p_smote = precision_score(y_test, y_pred_smote)
r_smote = recall_score(y_test, y_pred_smote)
f1_smote = f1_score(y_test, y_pred_smote)

imbalance_df = pd.DataFrame({
    "Strategy": ["Baseline", "class_weight='balanced'", "SMOTE (Train fold only)"],
    "Precision": [p_base, p_bal, p_smote],
    "Recall": [r_base, r_bal, r_smote],
    "F1 Score": [f1_base, f1_bal, f1_smote]
})
print(imbalance_df.round(4).to_string(index=False))

# ----------------------------------------------------------------------
# Task 12: Hyperparameter Tuning with OOB Score
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 12: Random Forest Hyperparameter Tuning & OOB Score")
print("=" * 80)

# Constructing base estimator with oob_score=True, bootstrap=True
rf_oob = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)
pipeline_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", rf_oob)
])

param_grid = {
    "rf__n_estimators": [50, 100, 200],
    "rf__max_depth": [3, 5, 8, None],
    "rf__max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(pipeline_rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)

best_rf_model = grid.best_estimator_
best_params = grid.best_params_
oob_score = best_rf_model.named_steps["rf"].oob_score_

print(f"Best Hyperparameters: {best_params}")
print(f"Random Forest Out-of-Bag (OOB) Score: {oob_score:.4f}")

# ----------------------------------------------------------------------
# Task 13: Regression Side-Task (Predict Fare)
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 13: Multivariate Linear Regression (Predicting Fare)")
print("=" * 80)

# Build regression dataset
reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked"]
target_reg = "fare"

X_reg = df[reg_features].copy()
y_reg = df[target_reg].copy()

# Fill fare NaN if any for baseline
y_reg = y_reg.fillna(y_reg.median())

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

reg_num_cols = ["pclass", "age", "sibsp", "parch"]
reg_cat_cols = ["sex", "embarked"]

reg_preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), reg_num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(drop="first"))]), reg_cat_cols)
    ]
)

reg_pipe = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("regressor", LinearRegression())
])

reg_pipe.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_pipe.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)
n = len(y_reg_test)
p = X_reg_train.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(7, 4))
plt.scatter(y_reg_pred, residuals, alpha=0.6, color="darkorange")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Fitted Values (Predicted Fare)")
plt.ylabel("Residuals")
plt.title("Residual Plot: Multivariate Linear Regression on Fare")
plt.savefig("regression_residuals.png")
plt.close()

print(f"Regression Metrics: MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.4f}, Adjusted R² = {adj_r2:.4f}")

# ----------------------------------------------------------------------
# Task 14 & 15: Comparative Evaluation & joblib Pipeline Export
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 14 & 15: Pipeline Export & Verification")
print("=" * 80)

# Save the complete tuned pipeline
export_pipeline_path = "best_pipeline.joblib"
joblib.dump(best_rf_model, export_pipeline_path)
print(f"Fitted pipeline exported successfully to '{export_pipeline_path}'.")

# Reload and test inference on raw unseen data
loaded_pipeline = joblib.load(export_pipeline_path)

raw_sample = pd.DataFrame([{
    "pclass": 1,
    "sex": "female",
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 211.33,
    "embarked": "S"
}])

pred_class = loaded_pipeline.predict(raw_sample)[0]
pred_prob = loaded_pipeline.predict_proba(raw_sample)[0, 1]
print(f"Verification Inference on Raw Input: Class = {pred_class} (Survival Probability = {pred_prob:.4f})")