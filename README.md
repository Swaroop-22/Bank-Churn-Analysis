markdown
# Bank Customer Churn Analysis & Prediction

This repository contains an end-to-end Machine Learning project aimed at predicting bank customer churn. Customer churn occurs when customers stop doing business with an institution. Predicting this allows banks to proactively offer retention strategies to at-risk accounts.

The project handles raw customer records, addresses severe class imbalances using a hybrid SMOTE and under-sampling pipeline, removes multivariate outliers via an Isolation Forest, and builds robust predictive models.

---

## 📊 Project Workflow & Pipeline

1. **Exploratory Data Analysis (EDA):** Analyzed missing data, distributions, and spatial associations using Pearson correlation matrices and Predictive Power Score (PPS).
2. **Data Preprocessing:** Label encoding for categorical variables (`country`, `gender`) and Feature Scaling using `MinMaxScaler`.
3. **Outlier Detection:** Deployed an `Isolation Forest` model (10% contamination rule) to flag and drop 1,000 anomalous rows.
4. **Imbalance Rectification:** Addressed highly skewed target categories (88.3% non-churn vs 11.69% churn) using a hybrid **SMOTE + RandomUnderSampler** pipeline.
5. **Model Evaluation:** Trained and validated Logistic Regression, Random Forest, and XGBoost models.

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost ppscore imbalanced-learn

```

---

## 📈 Dataset Diagnostics & Preprocessing

### Class Resampling Optimization

The initial data slice showed an extremely unbalanced distribution ($7,728$ active customers vs $1,272$ churned customers). To prevent model bias, a machine learning pipeline balanced the set perfectly using a combined over-sampling/under-sampling approach:

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# 1. Boost minority class to 60% of majority class
over = SMOTE(sampling_strategy=0.6)
# 2. Down-sample majority class to achieve an exact 50/50 balance
under = RandomUnderSampler(sampling_strategy=1.0)

pipeline = Pipeline(steps=[('o', over), ('u', under)])
X, y = pipeline.fit_resample(X, y)

```

* **Before Resampling:** `Counter({0.0: 7728, 1.0: 1272})`
* **After Resampling:** `Counter({0.0: 4636, 1.0: 4636})`

---

## 🤖 Model Performance Summary

The dataset was split into training ($30\%$) and testing ($70\%$) segments. The performance results across the three deployed classifiers are as follows:

| Machine Learning Model | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score |
| --- | --- | --- | --- | --- |
| **Logistic Regression** | 72.91% | 0.73 | 0.73 | 0.73 |
| **Random Forest Classifier** | 85.64% | 0.86 | 0.85 | 0.86 |
| **XGBoost Classifier** | **87.47%** | **0.86** | **0.85** | **0.86** |

### Key Takeaways

* **XGBoost** emerged as the top performer with an overall accuracy of **87.47%**, effectively managing non-linear splits.
* **Logistic Regression** provided a baseline accuracy of **72.91%** with a cross-validation score stable at **73.52%**.

---

## 💻 Code Structure & Model Serialization

The best-performing gradient-boosted tree model was exported to disk for production assembly or deployment pipelines:

```python
# Training the champion model
from xgboost import XGBClassifier

xg = XGBClassifier()
xg.fit(X_train, y_train)

# Saving the architecture for API usage
xg.save_model('xgb_model.h5')

```

---

## 🔮 Future Enhancements

* Fine-tune XGBoost hyperparameters using `GridSearchCV` or `Optuna`.
* Derive behavioral metrics from historical data features like `balance` and `estimated_salary`.
* Set up an interactive dashboard using Streamlit to accept real-time client criteria and output risk probability.

```

```
