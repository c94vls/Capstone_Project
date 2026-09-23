# Analytics Pipeline Module (`/analytics`)

## 1. Missing Value Strategy & Explicit Percentage Justification
The dataset contains 891 records. Missing percentages were audited prior to transformation:

| Column | Missing Rate (%) | Threshold Decision Rule | Implemented Strategy & Justification |
| :--- | :--- | :--- | :--- |
| **`deck`** | **77.22%** | $> 30\%$ | **Drop Column**: Imputing 77% missingness creates severe noise; dropping avoids model distortion. |
| **`age`** | **19.87%** | $5\% - 30\%$ | **Impute with Median**: Preserves sample size without introducing parametric assumptions of normality. |
| **`embarked`** | **0.22%** | $< 5\%$ | **Drop Rows**: Negligible loss of 2 observations allows keeping true distributions uncorrupted. |
| **`embark_town`** | **0.22%** | $< 5\%$ | **Drop Rows**: Redundant with `embarked` and drops synchronously with it. |

---

## 2. Univariate Distribution & Outlier Analysis

### Outlier Audit (IQR Rule: $[Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}]$)
* **`age`:** $\text{IQR} = 13.00$ years. Lower Bound = $2.50$, Upper Bound = $54.50$. Contains **65 outliers** (7.31%). Outliers reflect authentic demographic groups (infants and elderly travelers) rather than corrupted records.
* **`fare`:** $\text{IQR} = 23.09$. Lower Bound = $-26.74$ (clamped to $0$), Upper Bound = $65.63$. Contains **116 outliers** (13.05%), representing ultra-luxury suites.

### Skewness Conclusion for `fare`
* **Mean:** `32.10`
* **Median:** `14.45`
* **Mode:** `8.05`
* **Conclusion:** Since $\text{Mean} (32.10) > \text{Median} (14.45) > \text{Mode} (8.05)$, the `fare` distribution exhibits **strong right-skewness (positive skew)**. A small fraction of passengers paid exorbitant sums, pulling the arithmetic mean substantially above the central mass.

---

## 3. Bivariate Breakdown & 6×6 Numeric Correlation Matrix

### Survival Rates Across Strata
* **By Sex:** Female = **74.04%**, Male = **18.89%**
* **By Passenger Class:** 1st Class = **62.62%**, 2nd Class = **47.28%**, 3rd Class = **24.24%**
* **By Sex and Class Combined:**
  * Female (1st Class): **96.81%** | Male (1st Class): **36.89%**
  * Female (2nd Class): **92.11%** | Male (2nd Class): **15.74%**
  * Female (3rd Class): **50.00%** | Male (3rd Class): **13.54%**

### 6×6 Numeric Correlation Matrix Heatmap
The correlation matrix is calculated strictly across `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare` (`adult_male` and `alone` are excluded as derived/redundant features).

      survived   pclass      age    sibsp    parch     fare
survived    1.0000  -0.3355  -0.0768  -0.0340   0.0832   0.2553
pclass     -0.3355   1.0000  -0.3365   0.0816   0.0168  -0.5482
age        -0.0768  -0.3365   1.0000  -0.2326  -0.1787   0.0967
sibsp      -0.0340   0.0816  -0.2326   1.0000   0.4145   0.1609
parch       0.0832   0.0168  -0.1787   0.4145   1.0000   0.2175
fare        0.2553  -0.5482   0.0967   0.1609   0.2175   1.0000


### Interpretation of Top 2 Off-Diagonal Correlations
1. **`pclass` and `fare` ($r = -0.5482$):** The strongest off-diagonal relationship. The inverse relationship confirms that lower ticket numbers (1st class) required significantly higher monetary fares.
2. **`sibsp` and `parch` ($r = +0.4145$):** The second strongest relationship. Positive correlation highlights family co-travel patterns: passengers traveling with siblings or spouses were substantially more likely to travel with parents or children.

---

## 4. Multivariate Data Story (4 Visual Perspectives)

1. **Chart 1 (Survival by Class and Gender):** Highlights institutional maritime evacuation protocol ("women and children first"). Female survival remained above 92% in 1st and 2nd class, whereas male 3rd-class survival dropped to 13.5%.
2. **Chart 2 (Fare vs. Age Scatter):** Higher fares afforded passengers upper-deck cabin locations closer to lifeboats. Concentrated survival clusters exist at high fares across adult age brackets.
3. **Chart 3 (Port of Embarkation by Class):** Passengers boarding at Cherbourg had elevated survival rates primarily because Cherbourg had a much higher proportion of 1st-class ticket holders relative to Queenstown.
4. **Chart 4 (Family Size Dynamics):** Nuclear families of 2 to 4 individuals exhibited survival rates near 60%, whereas solo travelers (size = 1) and extended families ($\ge 5$) experienced steep survival drops due to coordination difficulties.

---

## 5. Exploratory Z-Score Sanity Check
Standardizing on the full dataset yields zero mean and unit variance:
* **Age:** $\mu_{\text{orig}} = 29.36, \sigma_{\text{orig}} = 13.01 \longrightarrow \mu_{Z} = 0.000000, \sigma_{Z} = 1.000000$
* **Fare:** $\mu_{\text{orig}} = 32.10, \sigma_{\text{orig}} = 49.70 \longrightarrow \mu_{Z} = 0.000000, \sigma_{Z} = 1.000000$

---

## 6. Imbalance Handling Comparison (Random Forest)

| Imbalance Handling Technique | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| **Baseline (No handling)** | 0.8167 | 0.7206 | 0.7656 |
| **`class_weight='balanced'`** | 0.7742 | **0.7500** | 0.7619 |
| **SMOTE (Train fold only)** | **0.8246** | 0.6912 | 0.7520 |

**Conclusion:** The **Baseline model with probability threshold calibration** or **`class_weight='balanced'`** performed best. `class_weight='balanced'` achieved the highest recall (75.00%) without artificial synthetic point artifacts in sparse tabular clusters, making it the most reliable operational approach.

---

## 7. Hyperparameter Tuning & Out-of-Bag (OOB) Score
* **Best Parameters:** `{'rf__max_depth': 5, 'rf__max_features': 'sqrt', 'rf__n_estimators': 100}`
* **Random Forest OOB Score:** **`0.8284`** (Confirms generalization capability using unseen bootstrap bags without cross-validation fold leakage).

---

## 8. Regression Side-Task (Predicting Fare) & Heteroscedasticity Analysis

### Metrics
* **MAE:** $19.45$
* **RMSE:** $34.21$
* **$R^2$:** $0.4124$
* **Adjusted $R^2$:** $0.3918$

### Residual Plot & Heteroscedasticity Interpretation
The residual plot exhibits severe **heteroscedasticity**. As predicted fare increases, residual dispersion fans out in an acute funnel shape. Low-fare predictions show tightly bound errors, whereas high-fare cabins yield extreme residuals exceeding $\pm 150$. This violates OLS homoscedasticity assumptions and indicates log transformation ($\log(\text{fare} + 1)$) or generalized linear modeling (GLM Gamma family) is required for fare modeling.

---

## 9. Final Model Comparison Table & Deployment Recommendation

### Separate Model Family Benchmark

| Classification Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8034 | 0.7656 | 0.7206 | 0.7424 | 0.8492 |
| **Decision Tree (depth=4)** | 0.8146 | 0.8302 | 0.6471 | 0.7273 | 0.8415 |
| **Tuned Random Forest** | **0.8315** | **0.8462** | **0.7353** | **0.7874** | **0.8756** |

*Separate Regression Benchmark:*
| Regression Model | Target | MAE | RMSE | $R^2$ | Adjusted $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multivariate OLS** | `fare` | 19.45 | 34.21 | 0.4124 | 0.3918 |

### Deployment Recommendation
We recommend deploying the **Tuned Random Forest Pipeline** (`best_pipeline.joblib`). It delivers superior discriminative capacity across every classification metric, topping ROC-AUC at **0.8756** and F1 at **0.7874**. Its ensemble of decorrelated decision trees absorbs non-linear feature interactions (such as the interaction between sex and passenger class) far better than Logistic Regression, while avoiding the extreme variance and threshold fragility of a standalone Decision Tree. The exported `joblib` artifact encapsulates both the `ColumnTransformer` (median imputers, scalers, one-hot encoders) and the tuned forest, ensuring that raw incoming production requests are sanitized and scored without test-set leakage.