"""
Zepto Analytics Guild - Module 2: Part A (Profiling & Exploratory Data Analysis)
Loads data once, executes threshold-based cleaning, saves offline fallback titanic.csv,
computes IQR outliers, skewness, bivariate correlations, and 4 multivariate visual stories.
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# Task 1: Load Dataset Once, Profile, & Save Offline Fallback
# ----------------------------------------------------------------------
print("=" * 80)
print("TASK 1: Loading Dataset & Profiling")
print("=" * 80)

# Check for local fallback first; otherwise load from Seaborn and commit fallback
csv_path = "titanic.csv"
if os.path.exists(csv_path):
    print("Loading from committed offline fallback 'titanic.csv'...")
    df_raw = pd.read_csv(csv_path)
else:
    print("Loading via sns.load_dataset('titanic')...")
    df_raw = sns.load_dataset("titanic")
    df_raw.to_csv(csv_path, index=False)
    print(f"Committed offline fallback saved to '{csv_path}'.")

print(f"\nDataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print("\n--- DataFrame Info ---")
df_raw.info()

print("\n--- Summary Statistics (Numeric) ---")
print(df_raw.describe().round(2))

# Missing value calculation
missing_series = df_raw.isnull().sum()
missing_pct = (missing_series / len(df_raw)) * 100
missing_report = pd.DataFrame({"Missing_Count": missing_series, "Missing_Percentage": missing_pct})
missing_report = missing_report[missing_report["Missing_Count"] > 0].sort_values(by="Missing_Percentage", ascending=False)
print("\n--- Missing Value Audit ---")
print(missing_report.round(2))

# ----------------------------------------------------------------------
# Task 2: Rule-Based Missing Value Handling
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: Threshold-Based Missing Value Handling")
print("=" * 80)
"""
Threshold Rules:
1. < 5% missing: Drop rows (embarked: 0.22%, embark_town: 0.22%)
2. 5% - 30% missing: Impute (age: 19.87% -> median imputation)
3. > 30% missing: Drop column or encode category (deck: 77.22% -> dropped due to severe sparsity)
"""
df_cleaned = df_raw.copy()

# 'deck' missing rate is 77.22% (> 30% threshold). Imputing would introduce artificial bias.
# Dropping column 'deck' explicitly.
df_cleaned = df_cleaned.drop(columns=["deck"])

# 'age' missing rate is 19.87% (within 5%-30% threshold). Impute using median.
age_median = df_cleaned["age"].median()
df_cleaned["age"] = df_cleaned["age"].fillna(age_median)

# 'embarked' and 'embark_town' missing rate is 0.22% (< 5% threshold). Drop rows.
df_cleaned = df_cleaned.dropna(subset=["embarked", "embark_town"]).reset_index(drop=True)

print(f"Cleaned dataset shape after threshold filtering: {df_cleaned.shape}")

# ----------------------------------------------------------------------
# Task 3: Univariate Analysis & Outlier Detection
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 3: Univariate Outlier Analysis & Skewness")
print("=" * 80)

def detect_iqr_outliers(series, name):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    print(f"Column '{name}': IQR = {iqr:.2f}, Lower Bound = {lower_bound:.2f}, Upper Bound = {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(series)*100:.2f}% of data)")
    return len(outliers)

age_outliers = detect_iqr_outliers(df_cleaned["age"], "age")
fare_outliers = detect_iqr_outliers(df_cleaned["fare"], "fare")

fare_mean = df_cleaned["fare"].mean()
fare_median = df_cleaned["fare"].median()
fare_mode = df_cleaned["fare"].mode()[0]

print(f"\nFare Metrics: Mean = {fare_mean:.2f}, Median = {fare_median:.2f}, Mode = {fare_mode:.2f}")
# Written deduction: Mean (32.10) > Median (14.45) > Mode (8.05) -> Heavily Right-Skewed

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df_cleaned["age"], kde=True, ax=axes[0, 0], color="skyblue")
axes[0, 0].set_title("Age Distribution (Histogram + KDE)")
sns.boxplot(x=df_cleaned["age"], ax=axes[0, 1], color="lightblue")
axes[0, 1].set_title("Age Boxplot")

sns.histplot(df_cleaned["fare"], kde=True, ax=axes[1, 0], color="salmon")
axes[1, 0].set_title("Fare Distribution (Histogram + KDE)")
sns.boxplot(x=df_cleaned["fare"], ax=axes[1, 1], color="lightcoral")
axes[1, 1].set_title("Fare Boxplot")
plt.tight_layout()
plt.savefig("univariate_analysis.png")
plt.close()

# ----------------------------------------------------------------------
# Task 4: Bivariate Analysis & 6x6 Correlation Matrix
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 4: Bivariate Breakdown & 6x6 Heatmap")
print("=" * 80)

# Survival rates via boolean masking
surv_sex = {
    "Female": df_cleaned[df_cleaned["sex"] == "female"]["survived"].mean(),
    "Male": df_cleaned[df_cleaned["sex"] == "male"]["survived"].mean()
}
print(f"Survival Rate by Sex: Female = {surv_sex['Female']:.4f}, Male = {surv_sex['Male']:.4f}")

surv_pclass = {
    f"Class {c}": df_cleaned[df_cleaned["pclass"] == c]["survived"].mean() for c in [1, 2, 3]
}
print(f"Survival Rate by Class: {surv_pclass}")

print("Survival Rate by Sex and Pclass combined:")
for s in ["female", "male"]:
    for c in [1, 2, 3]:
        rate = df_cleaned[(df_cleaned["sex"] == s) & (df_cleaned["pclass"] == c)]["survived"].mean()
        print(f"  {s.capitalize()} - Class {c}: {rate:.4f}")

# 6x6 Correlation Matrix (strictly the 6 numeric features; excluding adult_male, alone)
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df_cleaned[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("6x6 Numeric Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

# Identify top 2 off-diagonal correlations
corr_unstack = corr_matrix.abs().unstack()
off_diag = corr_unstack[corr_unstack < 1.0].sort_values(ascending=False)
top_pairs = off_diag.iloc[::2].head(2)
print("\nTop 2 Absolute Off-Diagonal Correlations:")
for idx, val in top_pairs.items():
    orig_val = corr_matrix.loc[idx[0], idx[1]]
    print(f"  {idx[0]} and {idx[1]}: r = {orig_val:.4f} (abs = {val:.4f})")

# ----------------------------------------------------------------------
# Task 5: Multivariate Data Story (4 Distinct Visualizations)
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 5: Producing 4-Chart Multivariate Story")
print("=" * 80)

# Chart 1: Survival Rate by Sex and Class
plt.figure(figsize=(7, 4))
sns.barplot(data=df_cleaned, x="pclass", y="survived", hue="sex", palette="Set2")
plt.title("Chart 1: Survival Probability by Class and Gender")
plt.ylabel("Survival Rate")
plt.savefig("multivariate_1_class_sex.png")
plt.close()

# Chart 2: Fare vs Age by Survival Status
plt.figure(figsize=(7, 4))
sns.scatterplot(data=df_cleaned, x="age", y="fare", hue="survived", alpha=0.7, palette={0: "red", 1: "green"})
plt.title("Chart 2: Passenger Age vs. Fare Colored by Survival")
plt.ylim(0, 300)
plt.savefig("multivariate_2_age_fare.png")
plt.close()

# Chart 3: Survival across Embarkation Ports by Class
plt.figure(figsize=(7, 4))
sns.pointplot(data=df_cleaned, x="embark_town", y="survived", hue="pclass", markers=["o", "s", "D"], linestyles=["-", "--", ":"])
plt.title("Chart 3: Survival Probability across Embarkation Towns by Class")
plt.savefig("multivariate_3_embark_class.png")
plt.close()

# Chart 4: Family Size vs Survival Rate
df_cleaned["family_size"] = df_cleaned["sibsp"] + df_cleaned["parch"] + 1
plt.figure(figsize=(7, 4))
sns.barplot(data=df_cleaned, x="family_size", y="survived", color="mediumpurple")
plt.title("Chart 4: Survival Probability by Total Family Size")
plt.savefig("multivariate_4_family_size.png")
plt.close()
print("Saved all 4 multivariate charts.")

# ----------------------------------------------------------------------
# Task 6: Exploratory Z-Score Normalization Sanity Check
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 6: EDA-Stage Z-Score Standardization Sanity Check")
print("=" * 80)

age_z = (df_cleaned["age"] - df_cleaned["age"].mean()) / df_cleaned["age"].std()
fare_z = (df_cleaned["fare"] - df_cleaned["fare"].mean()) / df_cleaned["fare"].std()

print(f"Age:  Original Mean = {df_cleaned['age'].mean():.2f}, Std = {df_cleaned['age'].std():.2f} -> Z-Score Mean = {age_z.mean():.6f}, Std = {age_z.std():.6f}")
print(f"Fare: Original Mean = {df_cleaned['fare'].mean():.2f}, Std = {df_cleaned['fare'].std():.2f} -> Z-Score Mean = {fare_z.mean():.6f}, Std = {fare_z.std():.6f}")